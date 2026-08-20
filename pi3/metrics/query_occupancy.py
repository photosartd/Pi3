from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_OCCUPANCY_BIN_EDGES = (
    0.0,
    0.005,
    0.01,
    0.02,
    0.04,
    0.08,
    1.000001,
)


def validate_occupancy_bin_edges(edges: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in edges)
    if len(values) < 2:
        raise ValueError("query occupancy analysis requires at least two bin edges")
    if values[0] > 0.0 or values[-1] <= 1.0:
        raise ValueError(
            "query occupancy bins must include zero and cover fraction 1.0"
        )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("query occupancy bin edges must be finite")
    if any(right <= left for left, right in zip(values[:-1], values[1:])):
        raise ValueError("query occupancy bin edges must be strictly increasing")
    return values


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_values(rows, key)
    return float(np.mean(values)) if len(values) else float("nan")


def _median(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_values(rows, key)
    return float(np.median(values)) if len(values) else float("nan")


def _percentile(rows: list[dict[str, Any]], key: str, value: float) -> float:
    values = _finite_values(rows, key)
    return float(np.percentile(values, value)) if len(values) else float("nan")


def _recall(rows: list[dict[str, Any]], key: str, threshold: float) -> float:
    values = _finite_values(rows, key)
    return float(np.mean(values < float(threshold))) if len(values) else float("nan")


def _percent_token(fraction: float) -> str:
    return f"{100.0 * float(fraction):g}".replace(".", "p")


def occupancy_bin_label(index: int, lower: float, upper: float) -> str:
    lower_token = _percent_token(lower)
    if upper > 1.0:
        return f"bin_{index:02d}_ge{lower_token}pct"
    return f"bin_{index:02d}_{lower_token}to{_percent_token(upper)}pct"


def summarize_query_occupancy(
    rows: list[dict[str, Any]],
    bin_edges: Iterable[float] = DEFAULT_OCCUPANCY_BIN_EDGES,
) -> list[dict[str, Any]]:
    """Aggregate query pose/geometry errors by final visible mask occupancy."""

    edges = validate_occupancy_bin_edges(bin_edges)
    output: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        group = [
            row
            for row in rows
            if math.isfinite(float(row.get("query_visible_fraction", np.nan)))
            and lower <= float(row["query_visible_fraction"]) < upper
        ]
        output.append(
            {
                "bin_index": int(index),
                "bin_label": occupancy_bin_label(index, lower, upper),
                "lower_fraction": float(lower),
                "upper_fraction": float(min(upper, 1.0)),
                "query_count": int(len(group)),
                "query_visible_fraction_median": _median(
                    group, "query_visible_fraction"
                ),
                "query_visible_patches_median": _median(
                    group, "query_visible_patches"
                ),
                "query_visibility_fraction_median": _median(
                    group, "query_visibility_fraction"
                ),
                "query_used_mean_d": _mean(group, "query_used_d"),
                "query_used_median_d": _median(group, "query_used_d"),
                "query_used_p90_d": _percentile(group, "query_used_d", 90),
                "query_used_0_1d": _recall(group, "query_used_d", 0.1),
                "query_used_0_5d": _recall(group, "query_used_d", 0.5),
                "query_used_1d": _recall(group, "query_used_d", 1.0),
                "query_used_2d": _recall(group, "query_used_d", 2.0),
                "query_rot_mean_deg": _mean(group, "query_rot_deg"),
                "query_rot_median_deg": _median(group, "query_rot_deg"),
                "query_trans_mean_m": _mean(group, "query_trans_m"),
                "query_trans_median_m": _median(group, "query_trans_m"),
                "query_trans_lateral_median_m": _median(
                    group, "query_trans_lateral_m"
                ),
                "query_trans_depth_median_m": _median(
                    group, "query_trans_depth_m"
                ),
                "query_center_mean_m": _mean(group, "query_center_m"),
                "query_center_median_m": _median(group, "query_center_m"),
                "query_center_radial_median_m": _median(
                    group, "query_center_radial_m"
                ),
                "query_center_tangential_median_m": _median(
                    group, "query_center_tangential_m"
                ),
                "query_direction_median_deg": _median(
                    group, "query_direction_deg"
                ),
                "query_depth_abs_median_m": _median(
                    group, "query_depth_abs_m"
                ),
                "query_depth_relative_median": _median(
                    group, "query_depth_relative"
                ),
            }
        )
    return output


def occupancy_summary_scalars(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> dict[str, float]:
    """Return a deliberately compact TensorBoard view of the full table."""

    scalars = {
        "occupancy/query_count": float(len(rows)),
        "occupancy/query_visible_fraction_median": _median(
            rows, "query_visible_fraction"
        ),
        "occupancy/query_visible_patches_median": _median(
            rows, "query_visible_patches"
        ),
    }
    logged_columns = (
        "query_count",
        "query_visible_fraction_median",
        "query_used_median_d",
        "query_used_p90_d",
        "query_used_0_1d",
        "query_used_0_5d",
        "query_used_1d",
        "query_used_2d",
        "query_rot_median_deg",
        "query_trans_median_m",
        "query_center_median_m",
        "query_center_tangential_median_m",
        "query_depth_abs_median_m",
    )
    for record in summary:
        prefix = f"occupancy/{record['bin_label']}"
        for key in logged_columns:
            scalars[f"{prefix}/{key}"] = float(record[key])
    return scalars


def _spearman(rows: list[dict[str, Any]], error_key: str) -> dict[str, float]:
    occupancy = np.asarray(
        [row.get("query_visible_fraction", np.nan) for row in rows],
        dtype=np.float64,
    )
    error = np.asarray([row.get(error_key, np.nan) for row in rows], dtype=np.float64)
    valid = np.isfinite(occupancy) & np.isfinite(error) & (occupancy > 0.0)
    if int(valid.sum()) < 3:
        return {"n": int(valid.sum()), "spearman_r": float("nan"), "pvalue": float("nan")}
    try:
        from scipy.stats import spearmanr

        result = spearmanr(np.log(occupancy[valid]), error[valid])
        return {
            "n": int(valid.sum()),
            "spearman_r": float(result.statistic),
            "pvalue": float(result.pvalue),
        }
    except ImportError:
        order_x = np.argsort(np.argsort(np.log(occupancy[valid])))
        order_y = np.argsort(np.argsort(error[valid]))
        return {
            "n": int(valid.sum()),
            "spearman_r": float(np.corrcoef(order_x, order_y)[0, 1]),
            "pvalue": float("nan"),
        }


def query_occupancy_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    error_keys = (
        "query_used_d",
        "query_rot_deg",
        "query_trans_m",
        "query_center_m",
        "query_center_tangential_m",
        "query_depth_abs_m",
    )
    high_visibility = [
        row
        for row in rows
        if math.isfinite(float(row.get("query_visibility_fraction", np.nan)))
        and float(row["query_visibility_fraction"]) >= 0.7
    ]
    object_ids = sorted(
        {
            int(row["obj_id"])
            for row in rows
            if row.get("obj_id") is not None
        }
    )
    per_object = {}
    for obj_id in object_ids:
        object_rows = [row for row in rows if int(row["obj_id"]) == obj_id]
        per_object[str(obj_id)] = {
            "n": len(object_rows),
            **{key: _spearman(object_rows, key) for key in error_keys},
        }

    within_object_rank = {}
    try:
        from scipy.stats import rankdata
    except ImportError:
        rankdata = None
    for error_key in error_keys:
        occupancy_ranks = []
        error_ranks = []
        for obj_id in object_ids:
            object_rows = [row for row in rows if int(row["obj_id"]) == obj_id]
            occupancy = np.asarray(
                [row.get("query_visible_fraction", np.nan) for row in object_rows],
                dtype=np.float64,
            )
            error = np.asarray(
                [row.get(error_key, np.nan) for row in object_rows],
                dtype=np.float64,
            )
            valid = np.isfinite(occupancy) & np.isfinite(error)
            occupancy = occupancy[valid]
            error = error[valid]
            if len(occupancy) < 2:
                continue
            if rankdata is not None:
                occupancy_rank = rankdata(occupancy, method="average") / len(occupancy)
                error_rank = rankdata(error, method="average") / len(error)
            else:
                occupancy_rank = np.argsort(np.argsort(occupancy)) / len(occupancy)
                error_rank = np.argsort(np.argsort(error)) / len(error)
            occupancy_ranks.extend(occupancy_rank.tolist())
            error_ranks.extend(error_rank.tolist())
        correlation = float("nan")
        if len(occupancy_ranks) >= 3:
            correlation = float(np.corrcoef(occupancy_ranks, error_ranks)[0, 1])
        within_object_rank[error_key] = {
            "n": len(occupancy_ranks),
            "spearman_r": correlation,
        }

    return {
        "all_queries": {key: _spearman(rows, key) for key in error_keys},
        "visibility_ge_0_7": {
            key: _spearman(high_visibility, key) for key in error_keys
        },
        "within_object_rank": within_object_rank,
        "per_object": per_object,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return None
    return value


def _write_plot(summary: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    nonempty = [row for row in summary if int(row["query_count"]) > 0]
    if not nonempty:
        return
    x = np.arange(len(nonempty))
    labels = []
    for row in nonempty:
        lower_pct = 100.0 * float(row["lower_fraction"])
        upper_pct = 100.0 * float(row["upper_fraction"])
        if upper_pct >= 100.0:
            labels.append(f">={lower_pct:g}%")
        else:
            labels.append(f"{lower_pct:g}-{upper_pct:g}%")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        ("query_used_median_d", "Median pose error", "diameters"),
        ("query_rot_median_deg", "Median rotation error", "degrees"),
        ("query_trans_median_m", "Median translation error", "metres"),
        ("query_used_0_5d", "Recall below 0.5d", "recall"),
    )
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        values = [float(row[key]) for row in nonempty]
        ax.plot(x, values, marker="o")
        for index, (value, row) in enumerate(zip(values, nonempty)):
            if np.isfinite(value):
                ax.annotate(
                    f"n={int(row['query_count'])}",
                    (index, value),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_query_occupancy_artifacts(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    bin_edges: Iterable[float] = DEFAULT_OCCUPANCY_BIN_EDGES,
) -> dict[str, float]:
    """Write reusable per-query rows, summaries, correlations, and a plot."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("query occupancy artifacts require pandas") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_query_occupancy(rows, bin_edges)
    high_visibility_rows = [
        row
        for row in rows
        if math.isfinite(float(row.get("query_visibility_fraction", np.nan)))
        and float(row["query_visibility_fraction"]) >= 0.7
    ]
    high_visibility_summary = summarize_query_occupancy(
        high_visibility_rows, bin_edges
    )
    correlations = query_occupancy_correlations(rows)
    rows_frame = pd.DataFrame(rows)
    rows_frame.to_csv(output / "query_rows.csv", index=False)
    try:
        rows_frame.to_parquet(output / "query_rows.parquet", index=False)
    except (ImportError, ModuleNotFoundError):
        pass
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(high_visibility_summary).to_csv(
        output / "summary_visibility_ge_0_7.csv", index=False
    )
    if not rows_frame.empty and {"obj_id", "query_visible_fraction"}.issubset(
        rows_frame.columns
    ):
        labels = [record["bin_label"] for record in summary]
        rows_frame["occupancy_bin"] = pd.cut(
            rows_frame["query_visible_fraction"],
            validate_occupancy_bin_edges(bin_edges),
            right=False,
            labels=labels,
        )
        pd.crosstab(rows_frame["obj_id"], rows_frame["occupancy_bin"]).to_csv(
            output / "object_bin_counts.csv"
        )
    (output / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "summary_visibility_ge_0_7.json").write_text(
        json.dumps(_json_ready(high_visibility_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "correlations.json").write_text(
        json.dumps(_json_ready(correlations), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_plot(summary, output / "occupancy_metrics.png")
    return occupancy_summary_scalars(rows, summary)
