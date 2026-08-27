#!/usr/bin/env python3
"""Materialize and visualize exact geometry-constrained Pi3 model inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw
import torch

from pi3.models.ray_conditioning import intrinsics_to_ray_map


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-config",
        default="train_megapose_gso_geometry_small_scratch_ray_336x252",
    )
    parser.add_argument(
        "--data-config", default="megapose_gso_geometry_n5_k1_masked"
    )
    parser.add_argument(
        "--component",
        choices=(
            "GSOSceneGeometryN5K1",
            "GSORenderGeometryN5K1",
            "gso_geometry_val_scene_n5_k1",
            "gso_geometry_val_render_n5_k1",
        ),
        default="GSOSceneGeometryN5K1",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _to_rgb_tensor(value) -> torch.Tensor:
    tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB tensor, got {tuple(tensor.shape)}")
    return tensor.float().clamp(0, 1)


def _rgb_image(value) -> Image.Image:
    tensor = _to_rgb_tensor(value)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _depth_image(depth: np.ndarray) -> Image.Image:
    depth = np.asarray(depth, dtype=np.float32)
    valid = depth > 0
    value = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        low, high = np.percentile(depth[valid], (2, 98))
        value[valid] = np.clip((depth[valid] - low) / max(float(high - low), 1e-6), 0, 1)
    # A compact blue-to-yellow diagnostic palette without a plotting dependency.
    rgb = np.stack((value, np.sqrt(value), 1.0 - value), axis=-1)
    rgb[~valid] = 0
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _ray_image(intrinsics: np.ndarray, height: int, width: int) -> Image.Image:
    rays = intrinsics_to_ray_map(
        torch.as_tensor(intrinsics, dtype=torch.float32), height, width
    ).numpy()
    scale = max(float(np.abs(rays).max()), 1e-6)
    x = np.clip(0.5 + 0.5 * rays[..., 0] / scale, 0, 1)
    y = np.clip(0.5 + 0.5 * rays[..., 1] / scale, 0, 1)
    rgb = np.stack((x, y, np.full_like(x, 0.5)), axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _view_direction(T_C_O: np.ndarray) -> np.ndarray:
    pose = np.asarray(T_C_O, dtype=np.float64)
    direction = -(pose[:3, :3].T @ pose[:3, 3])
    return direction / np.linalg.norm(direction)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _make_sheet(views, sample_metadata, path: Path) -> dict:
    first_rgb = _rgb_image(views[0]["img"])
    width, height = first_rgb.size
    label_height = 44
    canvas = Image.new("RGB", (width * len(views), (height + label_height) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    records = []
    directions = []
    for column, view in enumerate(views):
        rgb = first_rgb if column == 0 else _rgb_image(view["img"])
        if rgb.size != (width, height):
            raise AssertionError(f"Unexpected input size {rgb.size}")
        mask = np.asarray(view["object_visibility_mask"]) > 0.5
        depth = np.asarray(view["depthmap"], dtype=np.float32)
        rgb_array = np.asarray(rgb)
        if np.any(rgb_array[~mask] != 0):
            raise AssertionError("Object-only RGB contains nonzero background pixels")
        if np.any(depth[~mask] != 0):
            raise AssertionError("Object-only depth contains nonzero background pixels")
        intrinsics = np.asarray(view["camera_intrinsics"], dtype=np.float32)
        norm_fx = float(intrinsics[0, 0] / width)
        norm_fy = float(intrinsics[1, 1] / height)
        directions.append(_view_direction(view["T_C_O"]))
        tiles = (rgb, _depth_image(depth), _ray_image(intrinsics, height, width))
        for row, tile in enumerate(tiles):
            top = row * (height + label_height)
            canvas.paste(tile, (column * width, top + label_height))
            role = "REF" if bool(view["is_reference"]) else "QUERY"
            source = str(view["source_name"])
            caption = (
                f"{role} {source}  scene={int(view['source_scene_id'])}\n"
                f"fx/W={norm_fx:.3f} fy/H={norm_fy:.3f}"
                if row == 0
                else ("metric depth" if row == 1 else "ray-map input (x/z,y/z)")
            )
            draw.multiline_text((column * width + 5, top + 4), caption, fill="black")
            border = "#00a050" if role == "REF" else "#e0a000"
            draw.rectangle(
                (column * width, top + label_height, (column + 1) * width - 1, top + label_height + height - 1),
                outline=border,
                width=3,
            )
        records.append(
            {
                "role": str(view["view_role"]),
                "source": str(view["source_name"]),
                "object_id": int(view["object_id"]),
                "scene_id": int(view["source_scene_id"]),
                "view_id": int(view["view_id"]),
                "visibility": float(view["visib_fract"]),
                "crop_bbox_xyxy": np.asarray(
                    view.get("object_crop_bbox_xyxy", [-1.0] * 4)
                ).tolist(),
                "crop_center_shift_xy": np.asarray(
                    view.get("object_crop_center_shift_xy", [0.0, 0.0])
                ).tolist(),
                "target_normalized_focal": float(
                    view.get("object_crop_target_normalized_focal", -1.0)
                ),
                "actual_normalized_focal": float(
                    view.get("object_crop_actual_normalized_focal", norm_fx)
                ),
                "fx_over_width": norm_fx,
                "fy_over_height": norm_fy,
                "mask_fraction": float(mask.mean()),
                "virtual_camera_applied": bool(
                    view.get("virtual_camera_applied", False)
                ),
                "virtual_camera_zoom_requested": float(
                    view.get("virtual_camera_zoom_requested", 1.0)
                ),
                "virtual_camera_zoom_safe_max": float(
                    view.get("virtual_camera_zoom_safe_max", 1.0)
                ),
                "virtual_camera_zoom": float(
                    view.get("virtual_camera_zoom", 1.0)
                ),
            }
        )
    query_direction = directions[-1]
    positive_angle = min(
        float(np.degrees(np.arccos(np.clip(np.dot(value, query_direction), -1, 1))))
        for value in directions[:-1]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {
        "image": str(path),
        "sample_metadata": dict(sample_metadata),
        "measured_positive_angle_degrees": positive_angle,
        "views": records,
    }


def main() -> None:
    args = _arguments()
    with hydra.initialize_config_dir(
        version_base="1.2", config_dir=str(REPOSITORY_ROOT / "configs")
    ):
        config = hydra.compose(
            config_name="default",
            overrides=[f"train={args.train_config}", f"data={args.data_config}"],
        )
    component = (
        config.val_datasets[args.component].dataset
        if args.component.startswith("gso_geometry_val_")
        else config.train_dataset[args.component]
    )
    with open_dict(component):
        component.resolution = config.train.resolution
    OmegaConf.resolve(component)
    dataset = hydra.utils.instantiate(component)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    try:
        for sample_index in range(args.samples):
            sample_seed = args.seed + sample_index
            views = dataset[(sample_index, 0, 6, sample_seed)]
            reports.append(
                _make_sheet(
                    views,
                    dataset.this_views_info,
                    output / f"{args.component}_{sample_index:02d}.png",
                )
            )
    finally:
        dataset.close()
    report_path = output / f"{args.component}.json"
    report_path.write_text(
        json.dumps(reports, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"samples": len(reports), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
