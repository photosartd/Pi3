from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import rerun as rr

from pi3.metrics.utils import BopModelCache, sample_points
from pi3.visualizations.utils import tensor_image_to_uint8


DEFAULT_TRAIN_CONFIG = "train_lmgeo_scratch_dinos_small_a40_40gb_anchor_mask_conditioning"
DEFAULT_DATA_CONFIG = "lmgeo_trainpbr45_anchor_scene_pairs_masked_depth_query_masks"
DEFAULT_OUTPUT = "outputs/rerun/lmgeo_anchor_pair.rrd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one LMGeo anchor scene-pair sample in Rerun. The point "
            "clouds are the dataset GT depth maps unprojected into the anchor "
            "object/CAD coordinate frame."
        )
    )
    parser.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument(
        "--split",
        default="train",
        help=(
            "Dataset split to instantiate. Use 'train' or a named validation "
            "loader such as 'real_anchor_pairs' or 'pbr_anchor_pairs'."
        ),
    )
    parser.add_argument("--dataset-name", default="LMGeoAnchorScenePair")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument(
        "--frame-num",
        type=int,
        default=12,
        help=(
            "Total views to request from the dataset when sampling train "
            "anchor pairs. For validation configs this still overrides the "
            "dataset's current frame_num through the tuple index path."
        ),
    )
    parser.add_argument("--resolution-index", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--models-folder", default=None)
    parser.add_argument(
        "--full-scene-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force lmgeo.depth_masking=false before sampling so full_depth "
            "entities contain the whole scene depth. Use --no-full-scene-depth "
            "to reproduce the training config's object-only valid depth."
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-full-points-per-view",
        type=int,
        default=25000,
        help="Cap full-depth points per view after filtering. Use 0 for all points.",
    )
    parser.add_argument(
        "--max-mask-points-per-view",
        type=int,
        default=30000,
        help="Cap object-mask points per view after filtering. Use 0 for all points.",
    )
    parser.add_argument(
        "--max-model-points",
        type=int,
        default=30000,
        help="Cap CAD model points. Use 0 for all vertices.",
    )
    parser.add_argument("--voxel-size", type=float, default=0.002)
    parser.add_argument("--point-radius", type=float, default=0.0015)
    parser.add_argument("--mask-point-radius", type=float, default=0.003)
    parser.add_argument("--frustum-depth", type=float, default=0.15)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--ws-port", type=int, default=9091)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the script alive after logging when --serve is used.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides, e.g. lmgeo.data_root=/path/to/lm-o",
    )
    return parser.parse_args()


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.reshape(-1)[0].item()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_entity_fragment(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _compose_cfg(args: argparse.Namespace) -> DictConfig:
    overrides = [
        f"train={args.train_config}",
        f"data={args.data_config}",
        *args.overrides,
    ]
    if args.full_scene_depth:
        overrides.append("lmgeo.depth_masking=false")
    if args.data_root is not None:
        overrides.append(f"lmgeo.data_root={args.data_root}")
    if args.models_folder is not None:
        overrides.append(f"lmgeo.models_folder={args.models_folder}")

    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "configs"),
    ):
        return compose(config_name="default", overrides=overrides)


def _instantiate_dataset(cfg: DictConfig, args: argparse.Namespace):
    if args.split == "train":
        train_dataset_cfg = cfg.train_dataset
        if "weights" in train_dataset_cfg:
            if args.dataset_name not in train_dataset_cfg:
                raise KeyError(
                    f"Dataset {args.dataset_name!r} not found in train_dataset. "
                    f"Available: {list(train_dataset_cfg.weights.keys())}"
                )
            dataset_cfg = train_dataset_cfg[args.dataset_name]
        else:
            dataset_cfg = train_dataset_cfg
    else:
        if "val_datasets" not in cfg or args.split not in cfg.val_datasets:
            available = list(cfg.val_datasets.keys()) if "val_datasets" in cfg else []
            raise KeyError(
                f"Validation split {args.split!r} not found. Available: {available}"
            )
        dataset_cfg = cfg.val_datasets[args.split].dataset

    instantiate_kwargs = {}
    if "resolution" not in dataset_cfg:
        instantiate_kwargs["resolution"] = cfg.train.resolution

    dataset = hydra.utils.instantiate(dataset_cfg, **instantiate_kwargs)
    dataset.convert_attributes()
    dataset.set_epoch(args.epoch, base_seed=int(cfg.train.base_seed))
    return dataset


def _view_rgb(view: dict[str, Any]) -> np.ndarray:
    return tensor_image_to_uint8(view["img"])


def _view_group(view: dict[str, Any]) -> str:
    if bool(_scalar(view.get("is_reference"), False)):
        return "scene_1_reference"
    if bool(_scalar(view.get("is_query"), False)):
        return "scene_2_query"
    return "scene_unknown"


def _view_name(view: dict[str, Any], index: int) -> str:
    role = str(_scalar(view.get("view_role"), "view"))
    scene_id = int(_scalar(view.get("source_scene_id"), -1))
    subscene_id = int(_scalar(view.get("source_subscene_id"), -1))
    im_id = int(_scalar(view.get("im_id"), -1))
    gt_id = int(_scalar(view.get("gt_id"), -1))
    return (
        f"view_{index:02d}_{_safe_entity_fragment(role)}_"
        f"s{scene_id:06d}_sub{subscene_id:04d}_im{im_id:06d}_gt{gt_id:02d}"
    )


def _colors_from_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    return colors[np.asarray(mask, dtype=bool).reshape(-1)]


def _group_color(group: str) -> np.ndarray:
    if group == "scene_1_reference":
        return np.asarray([255, 145, 45], dtype=np.uint8)
    if group == "scene_2_query":
        return np.asarray([45, 185, 255], dtype=np.uint8)
    return np.asarray([210, 210, 210], dtype=np.uint8)


def _subsample_points_and_colors(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
        points = points[indices]
        colors = colors[indices]
    return points, colors


def _frustum_lines(
    camera_pose: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    *,
    depth: float,
) -> list[np.ndarray]:
    K = np.asarray(intrinsics, dtype=np.float64)
    T_O_C = np.asarray(camera_pose, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pixels = np.asarray(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [float(width), float(height)],
            [0.0, float(height)],
        ],
        dtype=np.float64,
    )
    corners_cam = np.stack(
        [
            (pixels[:, 0] - cx) * depth / fx,
            (pixels[:, 1] - cy) * depth / fy,
            np.full(4, depth),
        ],
        axis=1,
    )
    center = T_O_C[:3, 3]
    corners_obj = corners_cam @ T_O_C[:3, :3].T + center
    return [
        np.stack([center, corners_obj[0], corners_obj[1], corners_obj[2], corners_obj[3], corners_obj[0]]),
        np.stack([center, corners_obj[0]]),
        np.stack([center, corners_obj[1]]),
        np.stack([center, corners_obj[2]]),
        np.stack([center, corners_obj[3]]),
    ]


def _axis_lines(length: float = 0.1) -> tuple[list[np.ndarray], np.ndarray]:
    origin = np.zeros(3, dtype=np.float32)
    strips = [
        np.stack([origin, np.asarray([length, 0.0, 0.0], dtype=np.float32)]),
        np.stack([origin, np.asarray([0.0, length, 0.0], dtype=np.float32)]),
        np.stack([origin, np.asarray([0.0, 0.0, length], dtype=np.float32)]),
    ]
    colors = np.asarray([[255, 60, 60], [60, 220, 80], [80, 130, 255]], dtype=np.uint8)
    return strips, colors


def _log_cad_model(cfg: DictConfig, object_id: int, args: argparse.Namespace) -> None:
    try:
        cache = BopModelCache(
            cfg.lmgeo.data_root,
            models_folder=cfg.lmgeo.models_folder,
            max_model_points=args.max_model_points,
        )
        model_points = sample_points(
            cache.points(object_id),
            voxel_size=args.voxel_size,
            max_points=args.max_model_points,
        ).astype(np.float32)
        if len(model_points):
            rr.log(
                "world/cad_model/object_vertices",
                rr.Points3D(
                    model_points,
                    colors=np.tile(np.asarray([[180, 180, 180]], dtype=np.uint8), (len(model_points), 1)),
                    radii=args.point_radius,
                ),
            )
        diameter = cache.diameter(object_id)
        rr.log(
            "world/cad_model/info",
            rr.TextDocument(
                f"Object {object_id:06d}\\nDiameter: {diameter:.6f} m\\n"
                "CAD vertices are in object/CAD coordinates.",
                media_type="text/plain",
            ),
        )
    except Exception as exc:
        rr.log(
            "world/cad_model/load_warning",
            rr.TextLog(f"Could not load CAD model for object {object_id}: {exc}", level="WARN"),
        )


def _log_view(
    view: dict[str, Any],
    *,
    view_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    group = _view_group(view)
    name = _view_name(view, view_index)
    base = f"world/{group}/{name}"
    rgb = _view_rgb(view)
    height, width = rgb.shape[:2]
    points = _as_numpy(view["pts3d"]).astype(np.float32)
    valid_mask = _as_numpy(view["valid_mask"]).astype(bool)
    object_mask = _as_numpy(view.get("object_visibility_mask", np.zeros(valid_mask.shape))).astype(bool)
    condition_mask = _as_numpy(view.get("visibility_mask_condition", np.zeros(valid_mask.shape))).astype(bool)
    known_mask = _as_numpy(view.get("visibility_mask_known", np.zeros(valid_mask.shape))).astype(bool)
    camera_pose = _as_numpy(view["camera_pose"]).astype(np.float32)
    intrinsics = _as_numpy(view["camera_intrinsics"]).astype(np.float32)

    full_points = points[valid_mask]
    full_colors = _colors_from_rgb(rgb, valid_mask)
    full_points, full_colors = _subsample_points_and_colors(
        full_points,
        full_colors,
        max_points=args.max_full_points_per_view,
    )
    if len(full_points):
        rr.log(
            f"world/{group}/full_depth/{name}",
            rr.Points3D(full_points, colors=full_colors, radii=args.point_radius),
        )

    masked = valid_mask & object_mask
    masked_points = points[masked]
    masked_colors = _colors_from_rgb(rgb, masked)
    masked_points, masked_colors = _subsample_points_and_colors(
        masked_points,
        masked_colors,
        max_points=args.max_mask_points_per_view,
    )
    if len(masked_points):
        rr.log(
            f"world/{group}/object_masked/{name}",
            rr.Points3D(masked_points, colors=masked_colors, radii=args.mask_point_radius),
        )

    conditioned = valid_mask & condition_mask & known_mask
    conditioned_points = points[conditioned]
    conditioned_colors = _colors_from_rgb(rgb, conditioned)
    conditioned_points, conditioned_colors = _subsample_points_and_colors(
        conditioned_points,
        conditioned_colors,
        max_points=args.max_mask_points_per_view,
    )
    if len(conditioned_points):
        rr.log(
            f"world/{group}/model_visibility_condition/{name}",
            rr.Points3D(
                conditioned_points,
                colors=conditioned_colors,
                radii=args.mask_point_radius * 1.2,
            ),
        )

    rr.log(
        f"{base}/camera/frustum",
        rr.LineStrips3D(
            _frustum_lines(
                camera_pose,
                intrinsics,
                width,
                height,
                depth=args.frustum_depth,
            ),
            colors=np.tile(_group_color(group)[None], (5, 1)),
            radii=args.point_radius,
        ),
    )
    rr.log(
        f"{base}/camera/image_plane",
        rr.Pinhole(image_from_camera=intrinsics, resolution=[width, height]),
    )
    rr.log(f"{base}/camera/image_plane/rgb", rr.Image(rgb))

    metadata = {
        "group": group,
        "role": str(_scalar(view.get("view_role"), "")),
        "object_id": int(_scalar(view.get("object_id"), -1)),
        "scene_id": int(_scalar(view.get("source_scene_id"), -1)),
        "subscene_id": int(_scalar(view.get("source_subscene_id"), -1)),
        "im_id": int(_scalar(view.get("im_id"), -1)),
        "gt_id": int(_scalar(view.get("gt_id"), -1)),
        "valid_points": int(valid_mask.sum()),
        "object_points": int(masked.sum()),
        "condition_points": int(conditioned.sum()),
        "visib_fract": float(_scalar(view.get("visib_fract"), np.nan)),
    }
    rr.log(
        f"{base}/metadata",
        rr.TextDocument(
            "\n".join(f"{key}: {value}" for key, value in metadata.items()),
            media_type="text/plain",
        ),
    )
    return metadata


def _log_recording_metadata(
    cfg: DictConfig,
    args: argparse.Namespace,
    dataset: Any,
    sample: list[dict[str, Any]],
) -> int:
    object_id = int(_scalar(sample[0].get("object_id"), -1))
    views_info = getattr(dataset, "this_views_info", {})
    rr.log(
        "world",
        rr.ViewCoordinates.RDF,
    )
    strips, colors = _axis_lines(length=0.12)
    rr.log("world/object_axes", rr.LineStrips3D(strips, colors=colors, radii=args.mask_point_radius))
    rr.log(
        "world/README",
        rr.TextDocument(
            "\n".join(
                [
                    "LMGeo anchor scene-pair diagnostic",
                    "",
                    "All point clouds are in the anchor object's CAD frame.",
                    "scene_1_reference: key/reference subsequence",
                    "scene_2_query: query subsequence",
                    "full_depth: all valid depth pixels",
                    "object_masked: RGB-colored valid pixels inside the GT object visibility mask",
                    "model_visibility_condition: RGB-colored pixels from the visibility mask supplied to the model",
                    (
                        "full_scene_depth: enabled; lmgeo.depth_masking=false was forced for this diagnostic"
                        if args.full_scene_depth
                        else "full_scene_depth: disabled; depth masking follows the selected config"
                    ),
                    "",
                    f"train_config: {args.train_config}",
                    f"data_config: {args.data_config}",
                    f"split: {args.split}",
                    f"dataset_index: {args.index}",
                    f"frame_num: {args.frame_num}",
                    f"object_id: {object_id}",
                    f"views_info: {OmegaConf.to_yaml(OmegaConf.create(views_info)) if isinstance(views_info, dict) else views_info}",
                    f"data_root: {cfg.lmgeo.data_root}",
                    f"models_folder: {cfg.lmgeo.models_folder}",
                ]
            ),
            media_type="text/markdown",
        ),
    )
    return object_id


def main() -> None:
    args = parse_args()
    cfg = _compose_cfg(args)
    dataset = _instantiate_dataset(cfg, args)

    sample = dataset[(args.index, args.resolution_index, args.frame_num)]
    if not isinstance(sample, list) or not sample:
        raise RuntimeError(f"Expected dataset item to be a nonempty list of views, got {type(sample).__name__}")

    rr.init("lmgeo_anchor_pair_diagnostic", spawn=False)
    output_path = Path(args.output)
    if not args.serve:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output_path)
    else:
        serve_web = getattr(rr, "serve_web", rr.serve)
        serve_web(
            open_browser=bool(args.open_browser),
            web_port=int(args.web_port),
            ws_port=int(args.ws_port),
        )

    object_id = _log_recording_metadata(cfg, args, dataset, sample)
    _log_cad_model(cfg, object_id, args)

    rows = []
    for view_index, view in enumerate(sample):
        rows.append(_log_view(view, view_index=view_index, args=args))

    counts = {}
    for row in rows:
        counts.setdefault(row["group"], 0)
        counts[row["group"]] += 1
    total_valid_points = sum(int(row["valid_points"]) for row in rows)
    total_object_points = sum(int(row["object_points"]) for row in rows)
    total_condition_points = sum(int(row["condition_points"]) for row in rows)
    print("Logged LMGeo anchor pair to Rerun")
    print(f"  split: {args.split}")
    print(f"  object_id: {object_id:06d}")
    print(f"  views: {len(sample)} ({counts})")
    print(
        "  point counts before Rerun subsampling: "
        f"full_valid={total_valid_points}, "
        f"object_masked={total_object_points}, "
        f"model_condition={total_condition_points}"
    )
    print(f"  views_info: {getattr(dataset, 'this_views_info', {})}")
    if args.serve:
        print(f"  Web viewer: http://127.0.0.1:{args.web_port}")
        print(f"  WebSocket port: {args.ws_port}")
        print("  In VSCode Remote, forward both ports if the viewer stays disconnected.")
        if args.keep_alive:
            print("  Keeping Rerun server alive. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                print("Stopped.")
    else:
        print(f"  Recording: {output_path}")
        print(
            "  To open in browser on this server, rerun with --serve "
            "or use: rerun "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()
