from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PATCH_SIZE = 14.0
DEFAULT_SIZE_BIN_EDGES = (0.0, 4.0, 8.0, 16.0, 32.0, 64.0, math.inf)
DEFAULT_VISIB_BIN_EDGES = (0.1, 0.3, 0.5, 0.7, 0.9, 1.000001)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Failure-mode tables require pandas. Install the LMGeo env from "
            "requirements_common.txt or run `python -m pip install pandas`."
        ) from exc
    return pd


def _require_parquet_pandas():
    pd = _require_pandas()
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow. Install it with "
            "`python -m pip install pyarrow` in the pi3-lmgeo environment."
        ) from exc
    return pd


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Failure-mode plots require matplotlib. Install it with "
            "`python -m pip install matplotlib` in the pi3-lmgeo environment."
        ) from exc
    return plt


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return None
    return value


def _scene_rgb_ext(scene_dir: Path) -> str:
    rgb_dir = scene_dir / "rgb"
    for ext in (".png", ".jpg", ".jpeg"):
        if any(rgb_dir.glob(f"*{ext}")):
            return ext
    raise FileNotFoundError(f"No RGB images found in {rgb_dir}")


def _image_size(scene_dir: Path, im_id: int, rgb_ext: str | None = None) -> tuple[int, int]:
    from PIL import Image

    if rgb_ext is not None:
        path = scene_dir / "rgb" / f"{int(im_id):06d}{rgb_ext}"
        if path.exists():
            with Image.open(path) as image:
                return image.size
    stem = scene_dir / "rgb" / f"{int(im_id):06d}"
    for ext in (".png", ".jpg", ".jpeg"):
        path = stem.with_suffix(ext)
        if path.exists():
            with Image.open(path) as image:
                return image.size
    raise FileNotFoundError(f"Cannot find RGB image for {scene_dir} im_id={im_id}")


def preprocessing_resize_scale(
    image_size: tuple[int, int],
    intrinsics: Iterable[float] | np.ndarray,
    model_resolution: tuple[int, int],
) -> float:
    """Return Pi3's deterministic pre-final-crop resize scale for one image.

    ``BaseDataset._crop_resize_if_necessary`` first crops the image around the
    principal point, then rescales that crop so it covers the requested model
    resolution, and finally center-crops to exactly the model resolution. The
    BOP amodal pixel count lives in original-image pixels, so this scale is the
    area conversion used for the diagnostic patch-count proxy.
    """

    width, height = (int(image_size[0]), int(image_size[1]))
    target_w, target_h = (int(model_resolution[0]), int(model_resolution[1]))
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    cx = int(round(float(K[0, 2])))
    cy = int(round(float(K[1, 2])))
    min_margin_x = min(cx, width - cx)
    min_margin_y = min(cy, height - cy)
    if min_margin_x <= 0 or min_margin_y <= 0:
        raise ValueError(f"Bad principal point for image_size={image_size}, K={K.tolist()}")
    cropped_w = 2 * min_margin_x
    cropped_h = 2 * min_margin_y
    return float(max(target_w / cropped_w, target_h / cropped_h) + 1e-8)


def _bin_value(value: float, edges: tuple[float, ...], prefix: str) -> tuple[int, str]:
    for index in range(len(edges) - 1):
        lo = float(edges[index])
        hi = float(edges[index + 1])
        if lo <= value < hi:
            return index, f"{prefix}{lo:g}-{hi:g}"
    index = len(edges) - 2
    return index, f"{prefix}{edges[index]:g}+"


def _symmetric_flags(data_root: Path, models_folder: str) -> dict[int, bool]:
    models_info = _load_json(data_root / models_folder / "models_info.json")
    flags: dict[int, bool] = {}
    for key, info in models_info.items():
        discrete = info.get("symmetries_discrete") or []
        continuous = info.get("symmetries_continuous") or []
        flags[int(key)] = bool(discrete or continuous)
    return flags


def build_covariate_rows(
    dataset_root: str | Path,
    split: str,
    *,
    model_resolution: tuple[int, int] = (518, 518),
    models_folder: str = "models_eval",
    depth_unit_scale: float = 0.001,
    size_bin_edges: tuple[float, ...] = DEFAULT_SIZE_BIN_EDGES,
) -> list[dict[str, Any]]:
    data_root = Path(dataset_root)
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    symmetric_by_obj = _symmetric_flags(data_root, models_folder)
    rows: list[dict[str, Any]] = []
    scene_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir() and path.name.isdigit())
    for scene_dir in scene_dirs:
        scene_id = int(scene_dir.name)
        scene_gt = _load_json(scene_dir / "scene_gt.json")
        scene_gt_info = _load_json(scene_dir / "scene_gt_info.json")
        scene_camera = _load_json(scene_dir / "scene_camera.json")
        rgb_ext = _scene_rgb_ext(scene_dir)
        scene_image_size: tuple[int, int] | None = None

        for key in sorted(scene_gt.keys(), key=lambda value: int(value)):
            im_id = int(key)
            cam = scene_camera[key]
            if scene_image_size is None:
                scene_image_size = _image_size(scene_dir, im_id, rgb_ext)
            scale = preprocessing_resize_scale(
                scene_image_size,
                np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3),
                model_resolution,
            )
            for gt_id, gt in enumerate(scene_gt[key]):
                info = scene_gt_info[key][gt_id]
                obj_id = int(gt["obj_id"])
                px_count_all = float(info.get("px_count_all", 0.0))
                patch_count = px_count_all * scale * scale / (PATCH_SIZE * PATCH_SIZE)
                size_bin_index, size_bin = _bin_value(patch_count, size_bin_edges, "p")
                visib_fract = float(info.get("visib_fract", 1.0))
                visib_bin_index, visib_bin = _bin_value(visib_fract, DEFAULT_VISIB_BIN_EDGES, "v")
                t = np.asarray(gt["cam_t_m2c"], dtype=np.float64).reshape(3)
                rows.append(
                    {
                        "split": split,
                        "scene_id": scene_id,
                        "im_id": im_id,
                        "gt_id": int(gt_id),
                        "obj_id": obj_id,
                        "px_count_all": int(info.get("px_count_all", 0)),
                        "px_count_visib": int(info.get("px_count_visib", 0)),
                        "visib_fract": visib_fract,
                        "bbox_obj": info.get("bbox_obj"),
                        "bbox_visib": info.get("bbox_visib"),
                        "depth_z": float(t[2] * depth_unit_scale),
                        "resize_scale": scale,
                        "patch_count_amodal": patch_count,
                        "size_bin_index": int(size_bin_index),
                        "size_bin": size_bin,
                        "visib_bin_index": int(visib_bin_index),
                        "visib_bin": visib_bin,
                        "symmetric": bool(symmetric_by_obj.get(obj_id, False)),
                    }
                )
    return rows


def write_covariate_table(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    pd = _require_parquet_pandas()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def append_prediction_rows(out_path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    pd = _require_parquet_pandas()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        old_df = pd.read_parquet(path)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _recall(series: Any, threshold: float = 0.1) -> float:
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values < threshold))


def _safe_median(series: Any) -> float:
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _hard_mask(df: Any, hard_object_ids: Iterable[int] | None = None) -> Any:
    hard_ids = {int(obj_id) for obj_id in (hard_object_ids or [])}
    manual = df["obj_id"].astype(int).isin(hard_ids) if hard_ids else False
    return df["symmetric"].astype(bool) | (df["size_bin_index"].astype(int) <= 1) | manual


def _join_predictions_covariates(predictions: Any, covariates: Any) -> tuple[Any, dict[str, int]]:
    keys = ["scene_id", "im_id", "gt_id", "obj_id"]
    pred = predictions.copy()
    cov = covariates.copy()
    for key in keys:
        pred[key] = pred[key].astype(int)
        cov[key] = cov[key].astype(int)
    joined = pred.merge(cov, on=keys, how="left", suffixes=("", "_cov"), indicator=True)
    orphan_count = int((joined["_merge"] != "both").sum())
    joined = joined[joined["_merge"] == "both"].drop(columns=["_merge"])
    before_visibility = len(joined)
    joined = joined[joined["visib_fract"].astype(float) >= 0.1].copy()
    report = {
        "prediction_rows": int(len(pred)),
        "joined_rows": int(before_visibility),
        "orphan_rows": orphan_count,
        "rows_after_visibility": int(len(joined)),
    }
    return joined, report


def _per_object_table(df: Any) -> Any:
    pd = _require_pandas()
    rows = []
    for obj_id, group in df.groupby("obj_id", sort=True):
        unocc = group[group["visib_fract"].astype(float) >= 0.9]
        rows.append(
            {
                "obj_id": int(obj_id),
                "n": int(len(group)),
                "recall@0.1d": _recall(group["add_err_norm"]),
                "recall@0.1d | visib_fract >= 0.9": _recall(unocc["add_err_norm"]),
                "median add_err_norm": _safe_median(group["add_err_norm"]),
            }
        )
    return pd.DataFrame(rows)


def _heatmap_table(df: Any) -> Any:
    pd = _require_pandas()
    rows = []
    grouped = df.groupby(["size_bin_index", "size_bin", "visib_bin_index", "visib_bin"], sort=True)
    for (size_idx, size_bin, vis_idx, visib_bin), group in grouped:
        rows.append(
            {
                "size_bin_index": int(size_idx),
                "size_bin": size_bin,
                "visib_bin_index": int(vis_idx),
                "visib_bin": visib_bin,
                "n": int(len(group)),
                "recall@0.1d": _recall(group["add_err_norm"]),
            }
        )
    return pd.DataFrame(rows)


def _margin_size_table(df: Any) -> Any:
    pd = _require_pandas()
    visible = df[df["visib_fract"].astype(float) >= 0.9]
    rows = []
    for (idx, label), group in visible.groupby(["size_bin_index", "size_bin"], sort=True):
        rows.append({"size_bin_index": int(idx), "size_bin": label, "n": int(len(group)), "recall@0.1d": _recall(group["add_err_norm"])})
    return pd.DataFrame(rows)


def _margin_visib_table(df: Any) -> Any:
    pd = _require_pandas()
    middle = df[df["size_bin_index"].astype(int).isin([2, 3])]
    rows = []
    for (idx, label), group in middle.groupby(["visib_bin_index", "visib_bin"], sort=True):
        rows.append({"visib_bin_index": int(idx), "visib_bin": label, "n": int(len(group)), "recall@0.1d": _recall(group["add_err_norm"])})
    return pd.DataFrame(rows)


def _n_views_table(df: Any, hard_object_ids: Iterable[int] | None = None) -> Any:
    pd = _require_pandas()
    columns = ["class", "n_queries", "n", "recall@0.1d"]
    rows = []
    hard = _hard_mask(df, hard_object_ids)
    for class_name, class_df in (("easy", df[~hard]), ("hard", df[hard])):
        for n_queries, group in class_df.groupby("n_queries", sort=True):
            if int(n_queries) not in {5, 10, 20}:
                continue
            rows.append(
                {
                    "class": class_name,
                    "n_queries": int(n_queries),
                    "n": int(len(group)),
                    "recall@0.1d": _recall(group["add_err_norm"]),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _depth_spearman_table(df: Any) -> Any:
    pd = _require_pandas()
    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None
    rows = []
    for (idx, label), group in df.groupby(["size_bin_index", "size_bin"], sort=True):
        if len(group) < 3:
            corr = float("nan")
            pvalue = float("nan")
        elif spearmanr is not None:
            result = spearmanr(group["depth_z"].astype(float), group["add_err_norm"].astype(float), nan_policy="omit")
            corr = float(result.statistic)
            pvalue = float(result.pvalue)
        else:
            depth_rank = pd.Series(group["depth_z"].astype(float)).rank().to_numpy()
            err_rank = pd.Series(group["add_err_norm"].astype(float)).rank().to_numpy()
            corr = float(np.corrcoef(depth_rank, err_rank)[0, 1])
            pvalue = float("nan")
        rows.append({"size_bin_index": int(idx), "size_bin": label, "n": int(len(group)), "spearman_r": corr, "pvalue": pvalue})
    return pd.DataFrame(rows)


def coarse_failure_mode_scalars(
    df: Any,
    hard_object_ids: Iterable[int] | None = None,
    object_ids: Iterable[int] | None = None,
) -> dict[str, float]:
    scalars: dict[str, float] = {}
    small = df["size_bin_index"].astype(int) <= 1
    visible = df["visib_fract"].astype(float) >= 0.7
    buckets = {
        "small_visible_recall@0.1d": df[small & visible],
        "small_occluded_recall@0.1d": df[small & ~visible],
        "big_visible_recall@0.1d": df[~small & visible],
        "big_occluded_recall@0.1d": df[~small & ~visible],
    }
    for name, group in buckets.items():
        scalars[name] = _recall(group["add_err_norm"])

    hard = _hard_mask(df, hard_object_ids)
    scalars["easy_class_recall@0.1d"] = _recall(df[~hard]["add_err_norm"])
    scalars["hard_class_recall@0.1d"] = _recall(df[hard]["add_err_norm"])
    if object_ids is None:
        object_ids = sorted(int(obj_id) for obj_id in df["obj_id"].dropna().unique())
    for obj_id in sorted(int(obj_id) for obj_id in object_ids):
        group = df[df["obj_id"].astype(int) == int(obj_id)]
        scalars[f"obj_{int(obj_id):06d}_recall@0.1d"] = _recall(group["add_err_norm"])
    return scalars


def _plot_per_object(table: Any, out_path: Path) -> None:
    plt = _require_matplotlib()
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [f"{int(obj):02d}" for obj in table["obj_id"]]
    x = np.arange(len(labels))
    width = 0.26
    ax.bar(x - width, table["recall@0.1d"], width, label="overall")
    ax.bar(x, table["recall@0.1d | visib_fract >= 0.9"], width, label="unoccluded")
    ax.bar(x + width, table["median add_err_norm"], width, label="median error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(1.0, float(np.nanmax(table[["recall@0.1d", "recall@0.1d | visib_fract >= 0.9", "median add_err_norm"]].to_numpy())) * 1.1))
    ax.set_xlabel("object id")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_heatmap(table: Any, out_path: Path) -> None:
    plt = _require_matplotlib()
    if table.empty:
        return
    size_labels = table.sort_values("size_bin_index").drop_duplicates("size_bin_index")["size_bin"].tolist()
    vis_labels = table.sort_values("visib_bin_index").drop_duplicates("visib_bin_index")["visib_bin"].tolist()
    size_indices = sorted(table["size_bin_index"].unique())
    vis_indices = sorted(table["visib_bin_index"].unique())
    recall = np.full((len(size_indices), len(vis_indices)), np.nan, dtype=np.float64)
    counts = np.zeros_like(recall, dtype=np.int64)
    idx_map = {value: idx for idx, value in enumerate(size_indices)}
    vis_map = {value: idx for idx, value in enumerate(vis_indices)}
    for _, row in table.iterrows():
        i = idx_map[int(row["size_bin_index"])]
        j = vis_map[int(row["visib_bin_index"])]
        counts[i, j] = int(row["n"])
        if counts[i, j] >= 30:
            recall[i, j] = float(row["recall@0.1d"])
    masked = np.ma.masked_invalid(recall)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="0.82")
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap, origin="lower", aspect="auto")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            text = f"n={counts[i, j]}"
            if np.isfinite(recall[i, j]):
                text = f"{recall[i, j]:.2f}\n{text}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if np.isfinite(recall[i, j]) and recall[i, j] < 0.45 else "black")
    ax.set_xticks(np.arange(len(vis_labels)))
    ax.set_xticklabels(vis_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(size_labels)))
    ax.set_yticklabels(size_labels)
    ax.set_xlabel("visible fraction bin")
    ax.set_ylabel("amodal patch-count bin")
    fig.colorbar(image, ax=ax, label="recall@0.1d")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_margin_curves(size_table: Any, visib_table: Any, out_path: Path) -> None:
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if not size_table.empty:
        size_table = size_table.sort_values("size_bin_index")
        axes[0].plot(size_table["size_bin"], size_table["recall@0.1d"], marker="o")
        for x, y, n in zip(size_table["size_bin"], size_table["recall@0.1d"], size_table["n"]):
            axes[0].annotate(f"n={int(n)}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    axes[0].set_title("size | visib >= 0.9")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(alpha=0.2)
    if not visib_table.empty:
        visib_table = visib_table.sort_values("visib_bin_index")
        axes[1].plot(visib_table["visib_bin"], visib_table["recall@0.1d"], marker="o")
        for x, y, n in zip(visib_table["visib_bin"], visib_table["recall@0.1d"], visib_table["n"]):
            axes[1].annotate(f"n={int(n)}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    axes[1].set_title("occlusion | middle size bins")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_n_views(table: Any, out_path: Path) -> None:
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    if not table.empty and "class" in table:
        for class_name, group in table.groupby("class", sort=True):
            group = group.sort_values("n_queries")
            ax.plot(group["n_queries"], group["recall@0.1d"], marker="o", label=class_name)
            for x, y, n in zip(group["n_queries"], group["recall@0.1d"], group["n"]):
                ax.annotate(f"n={int(n)}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks([5, 10, 20])
    ax.set_ylim(0, 1)
    ax.set_xlabel("n_queries")
    ax.set_ylabel("recall@0.1d")
    if not table.empty:
        ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def aggregate_failure_modes(
    predictions_path: str | Path,
    covariates_path: str | Path,
    out_dir: str | Path,
    *,
    coarse: bool = False,
    plot: bool = True,
    checkpoint_step: int | None = None,
    split: str | None = None,
    val_name: str | None = None,
    hard_object_ids: Iterable[int] | None = None,
) -> dict[str, float]:
    pd = _require_parquet_pandas()
    predictions = pd.read_parquet(predictions_path)
    covariates = pd.read_parquet(covariates_path)
    if split is None and "split" in covariates and len(covariates):
        unique_splits = sorted(str(value) for value in covariates["split"].dropna().unique())
        if len(unique_splits) == 1:
            split = unique_splits[0]
    if split is not None and "split" in predictions:
        predictions = predictions[predictions["split"].astype(str) == str(split)].copy()
    if val_name is not None and "val_name" in predictions:
        predictions = predictions[predictions["val_name"].astype(str) == str(val_name)].copy()
    if "checkpoint_step" in predictions:
        if checkpoint_step is None and len(predictions):
            checkpoint_step = int(np.nanmax(predictions["checkpoint_step"].astype(float)))
        if checkpoint_step is not None:
            predictions = predictions[predictions["checkpoint_step"].astype(float) == float(checkpoint_step)].copy()
    joined, report = _join_predictions_covariates(predictions, covariates)
    if split is not None:
        report["split_filter"] = str(split)
    if val_name is not None:
        report["val_name_filter"] = str(val_name)
    if checkpoint_step is not None:
        report["checkpoint_step_filter"] = int(checkpoint_step)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "join_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if coarse:
        object_ids = sorted(int(obj_id) for obj_id in covariates["obj_id"].dropna().unique())
        scalars = coarse_failure_mode_scalars(joined, hard_object_ids, object_ids)
        pd.DataFrame([scalars]).to_csv(out / "coarse_scalars.csv", index=False)
        (out / "coarse_scalars.json").write_text(json.dumps(_json_ready(scalars), indent=2, sort_keys=True), encoding="utf-8")
        return scalars

    per_object = _per_object_table(joined)
    heatmap = _heatmap_table(joined)
    margin_size = _margin_size_table(joined)
    margin_visib = _margin_visib_table(joined)
    n_views = _n_views_table(joined, hard_object_ids)
    depth = _depth_spearman_table(joined)

    per_object.to_csv(out / "per_object_metrics.csv", index=False)
    heatmap.to_csv(out / "size_visib_heatmap.csv", index=False)
    margin_size.to_csv(out / "margin_size_visible.csv", index=False)
    margin_visib.to_csv(out / "margin_visib_middle_size.csv", index=False)
    n_views.to_csv(out / "n_views_curve.csv", index=False)
    depth.to_csv(out / "depth_spearman_by_size_bin.csv", index=False)

    if plot:
        _plot_per_object(per_object, out / "per_object_metrics.png")
        _plot_heatmap(heatmap, out / "size_visib_heatmap.png")
        _plot_margin_curves(margin_size, margin_visib, out / "margin_curves.png")
        _plot_n_views(n_views, out / "n_views_curve.png")

    return {}
