#!/usr/bin/env python3
"""Render and verify exact matched LM-O model inputs before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch

from pi3.models.ray_conditioning import intrinsics_to_ray_map


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-config",
        default="train_megapose_gso_geometry_pi3_lmgeo_eval_rtxpro6000_70gb_336x252",
    )
    parser.add_argument(
        "--data-config", default="lmgeo_new_val_geometry_render_n5_k1_masked"
    )
    parser.add_argument(
        "--component", default="lmo_new_val_geometry_render_n5_k1"
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rgb_image(value) -> Image.Image:
    tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.float().clamp(0, 1)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB, got {tuple(tensor.shape)}")
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _depth_image(depth) -> Image.Image:
    depth = np.asarray(depth, dtype=np.float32)
    valid = depth > 0
    value = np.zeros_like(depth)
    if valid.any():
        low, high = np.percentile(depth[valid], (2, 98))
        value[valid] = np.clip(
            (depth[valid] - low) / max(float(high - low), 1e-6), 0, 1
        )
    rgb = np.stack((value, np.sqrt(value), 1.0 - value), axis=-1)
    rgb[~valid] = 0
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _ray_image(K, height, width) -> Image.Image:
    rays = intrinsics_to_ray_map(
        torch.as_tensor(K, dtype=torch.float32), height, width
    ).numpy()
    scale = max(float(np.abs(rays).max()), 1e-6)
    rgb = np.stack(
        (
            np.clip(0.5 + 0.5 * rays[..., 0] / scale, 0, 1),
            np.clip(0.5 + 0.5 * rays[..., 1] / scale, 0, 1),
            np.full((height, width), 0.5, dtype=np.float32),
        ),
        axis=-1,
    )
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _view_direction(T_C_O):
    pose = np.asarray(T_C_O, dtype=np.float64)
    center = -(pose[:3, :3].T @ pose[:3, 3])
    return center / np.linalg.norm(center)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _sheet(views, metadata, path):
    width, height = 336, 252
    header = 52
    canvas = Image.new("RGB", (6 * width, 3 * (height + header)), "white")
    draw = ImageDraw.Draw(canvas)
    targets = []
    directions = []
    records = []
    for column, view in enumerate(views):
        role = "REF" if bool(view["is_reference"]) else "QUERY"
        image = _rgb_image(view["img"])
        if image.size != (width, height):
            raise AssertionError(f"Unexpected model input size: {image.size}")
        mask = np.asarray(view["object_visibility_mask"]) > 0.5
        depth = np.asarray(view["depthmap"], dtype=np.float32)
        rgb = np.asarray(image)
        if not mask.any() or not np.any(depth > 0):
            raise AssertionError("Matched crop produced an empty object observation")
        if np.any(rgb[~mask] != 0) or np.any(depth[~mask] != 0):
            raise AssertionError("Object-only RGB/depth contains nonzero background")
        K = np.asarray(view["camera_intrinsics"], dtype=np.float32)
        target = float(view["object_crop_target_normalized_focal"])
        actual = float(view["object_crop_actual_normalized_focal"])
        targets.append(target)
        directions.append(_view_direction(view["T_C_O"]))
        tiles = (image, _depth_image(depth), _ray_image(K, height, width))
        for row, tile in enumerate(tiles):
            top = row * (height + header)
            canvas.paste(tile, (column * width, top + header))
            caption = (
                f"{role} obj={int(view['object_id'])} im={int(view['im_id'])}\n"
                f"f*= {target:.4f} actual={actual:.4f} mask={mask.mean():.3f}"
                if row == 0
                else ("metric object depth" if row == 1 else "ray map (x/z, y/z)")
            )
            draw.multiline_text((column * width + 5, top + 5), caption, fill="black")
            draw.rectangle(
                (
                    column * width,
                    top + header,
                    (column + 1) * width - 1,
                    top + header + height - 1,
                ),
                outline="#009050" if role == "REF" else "#e09000",
                width=3,
            )
        records.append(
            {
                "role": str(view["view_role"]),
                "source": str(view["source"]),
                "visibility": float(view["visib_fract"]),
                "crop_bbox_xyxy": np.asarray(
                    view["object_crop_bbox_xyxy"]
                ).tolist(),
                "target_normalized_focal": target,
                "actual_normalized_focal": actual,
                "fx_over_width": float(K[0, 0] / width),
                "fy_over_height": float(K[1, 1] / height),
                "mask_fraction": float(mask.mean()),
                "mean_object_depth_m": float(depth[depth > 0].mean()),
            }
        )
    if len(views) != 6 or [bool(view["is_reference"]) for view in views] != [
        True,
        True,
        True,
        True,
        True,
        False,
    ]:
        raise AssertionError("Expected five references followed by one query")
    if max(targets) - min(targets) > 1e-7:
        raise AssertionError("Validation views do not share one focal target")
    query_direction = directions[-1]
    positive_angle = min(
        float(
            np.degrees(
                np.arccos(np.clip(np.dot(direction, query_direction), -1.0, 1.0))
            )
        )
        for direction in directions[:-1]
    )
    if positive_angle > 10.0001:
        raise AssertionError(f"Positive reference angle is {positive_angle:.4f} degrees")
    coverage = float(metadata["reference_union_coverage"])
    if coverage < 0.5:
        raise AssertionError(f"Reference union coverage is {coverage:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {
        "image": str(path),
        "reference_union_coverage": coverage,
        "reported_positive_angle_degrees": float(metadata["positive_angle_degrees"]),
        "measured_positive_angle_degrees": positive_angle,
        "target_normalized_focal": targets[0],
        "views": records,
    }


def main():
    args = _arguments()
    with hydra.initialize_config_dir(
        version_base="1.2", config_dir=str(REPOSITORY_ROOT / "configs")
    ):
        config = hydra.compose(
            config_name="default",
            overrides=[f"train={args.train_config}", f"data={args.data_config}"],
        )
    component = config.val_datasets[args.component].dataset
    OmegaConf.resolve(component)
    dataset = hydra.utils.instantiate(component)
    output = args.output.expanduser().resolve()
    reports = []
    try:
        for sample_index in range(min(args.samples, len(dataset))):
            views = dataset[(sample_index, 0, 6, args.seed + sample_index)]
            reports.append(
                _sheet(
                    views,
                    dataset.this_views_info,
                    output / f"lmgeo_matched_{sample_index:02d}.png",
                )
            )
        eligibility = dict(dataset.geometry_eligibility)
    finally:
        dataset.close()
    report_path = output / "lmgeo_matched_report.json"
    report_path.write_text(
        json.dumps(
            {"eligibility": eligibility, "samples": reports},
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "samples": len(reports)}, indent=2))


if __name__ == "__main__":
    main()

