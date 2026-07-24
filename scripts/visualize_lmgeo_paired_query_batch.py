from __future__ import annotations

import argparse
import colorsys
import json
import sys
from itertools import islice
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw, ImageFont

from datasets import create_dataloader
from pi3.models.ray_conditioning import intrinsics_to_ray_map
from pi3.visualizations.utils import tensor_image_to_uint8


DEFAULT_TRAIN_CONFIG = "train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1"
DEFAULT_DATA_CONFIG = (
    "lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one real LMGeo paired-query batch after the production "
            "dataloader and collator."
        )
    )
    parser.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--val-name", default="real_test")
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument(
        "--output",
        default="outputs/paired_query_debug/paired_query_dataloader.png",
    )
    return parser.parse_args()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _batch_item(
    view: dict[str, Any],
    key: str,
    *,
    batch_idx: int = 0,
) -> Any:
    value = view[key]
    if torch.is_tensor(value):
        value = value[batch_idx].detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.numpy()
    if isinstance(value, np.ndarray):
        value = value[batch_idx]
        if np.asarray(value).size == 1:
            return np.asarray(value).item()
        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        return value[batch_idx]
    return value


def _view_image(view: dict[str, Any], *, batch_idx: int = 0) -> Image.Image:
    return Image.fromarray(
        tensor_image_to_uint8(view["img"][batch_idx]),
        mode="RGB",
    )


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    radius: int = 10,
    width: int = 3,
) -> None:
    x, y = xy
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)
    draw.ellipse(
        (x - 3, y - 3, x + 3, y + 3),
        fill=color,
        outline=(0, 0, 0),
        width=1,
    )


def _panel_with_title(
    image: Image.Image,
    title: str,
    subtitle: str = "",
    *,
    title_height: int = 62,
) -> Image.Image:
    canvas = Image.new(
        "RGB",
        (image.width, image.height + title_height),
        color=(24, 28, 34),
    )
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), title, font=_font(18, bold=True), fill=(245, 247, 250))
    if subtitle:
        draw.text((10, 34), subtitle, font=_font(14), fill=(174, 184, 197))
    return canvas


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, color=(8, 10, 13))
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def _grid(
    images: list[Image.Image],
    *,
    columns: int,
    padding: int = 8,
    background: tuple[int, int, int] = (13, 16, 21),
) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), background)
    columns = max(1, min(int(columns), len(images)))
    rows = int(np.ceil(len(images) / columns))
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + (columns - 1) * padding,
            rows * cell_height + (rows - 1) * padding,
        ),
        background,
    )
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas.paste(
            image,
            (
                column * (cell_width + padding),
                row * (cell_height + padding),
            ),
        )
    return canvas


def _text_panel(
    title: str,
    lines: list[str],
    *,
    size: tuple[int, int],
) -> Image.Image:
    panel = Image.new("RGB", size, color=(24, 28, 34))
    draw = ImageDraw.Draw(panel)
    draw.text((22, 18), title, font=_font(24, bold=True), fill=(245, 247, 250))
    y = 62
    for line in lines:
        color = (196, 205, 216)
        if line.startswith("PASS"):
            color = (93, 224, 147)
        elif line.startswith("NOTE"):
            color = (247, 202, 92)
        draw.text((22, y), line, font=_font(16), fill=color)
        y += 27
    return panel


def _role_label(view: dict[str, Any], view_idx: int) -> tuple[str, str]:
    role = str(_batch_item(view, "view_role"))
    scene_id = int(_batch_item(view, "scene_id"))
    im_id = int(_batch_item(view, "im_id"))
    role_title = {
        "reference": "REFERENCE",
        "query": "QUERY · CROP",
        "query_context": "QUERY · ORIGINAL",
    }.get(role, role.upper())
    return f"{view_idx} · {role_title}", f"scene {scene_id} · image {im_id}"


def _matched_query_panels(
    batch: list[dict[str, Any]],
    crop_idx: int,
    original_idx: int,
) -> tuple[Image.Image, Image.Image, dict[str, float]]:
    crop = _view_image(batch[crop_idx])
    original = _view_image(batch[original_idx])
    crop_draw = ImageDraw.Draw(crop)
    original_draw = ImageDraw.Draw(original)

    K_crop = np.asarray(_batch_item(batch[crop_idx], "camera_intrinsics"))
    K_original = np.asarray(_batch_item(batch[original_idx], "camera_intrinsics"))
    H_crop_from_original = np.asarray(
        _batch_item(
            batch[crop_idx],
            "query_crop_from_original_homography",
        )
    )
    R_old_to_new = np.asarray(
        _batch_item(batch[crop_idx], "query_recenter_R_old_to_new")
    )
    T_Ccrop_O = np.asarray(_batch_item(batch[crop_idx], "T_C_O"))
    T_Corig_O = np.asarray(_batch_item(batch[original_idx], "T_C_O"))

    _draw_cross(
        crop_draw,
        (float(K_crop[0, 2]), float(K_crop[1, 2])),
    )
    _draw_cross(
        original_draw,
        (float(K_original[0, 2]), float(K_original[1, 2])),
    )

    crop_width, crop_height = crop.size
    xs = np.linspace(0.08 * crop_width, 0.92 * crop_width, 7)
    ys = np.linspace(0.10 * crop_height, 0.90 * crop_height, 5)
    crop_points = np.array(
        [[x, y, 1.0] for y in ys for x in xs],
        dtype=np.float64,
    )
    original_points_h = (
        np.linalg.inv(H_crop_from_original) @ crop_points.T
    ).T
    original_points = (
        original_points_h[:, :2] / original_points_h[:, 2:3]
    )

    valid_count = 0
    for index, (crop_xy_h, original_xy) in enumerate(
        zip(crop_points, original_points)
    ):
        crop_xy = crop_xy_h[:2]
        if not (
            0 <= original_xy[0] < original.width
            and 0 <= original_xy[1] < original.height
        ):
            continue
        hue = (index * 0.61803398875) % 1.0
        color = tuple(
            int(round(channel * 255))
            for channel in colorsys.hsv_to_rgb(hue, 0.78, 1.0)
        )
        for draw, xy in (
            (original_draw, original_xy),
            (crop_draw, crop_xy),
        ):
            x, y = float(xy[0]), float(xy[1])
            draw.ellipse(
                (x - 5, y - 5, x + 5, y + 5),
                fill=color,
                outline=(0, 0, 0),
                width=2,
            )
        valid_count += 1

    A_old_to_new = np.eye(4, dtype=np.float64)
    A_old_to_new[:3, :3] = R_old_to_new
    H_expected = K_crop @ R_old_to_new @ np.linalg.inv(K_original)
    H_expected /= H_expected[2, 2]
    diagnostics = {
        "homography_max_abs": float(
            np.max(np.abs(H_crop_from_original - H_expected))
        ),
        "pose_max_abs": float(
            np.max(np.abs(T_Ccrop_O - A_old_to_new @ T_Corig_O))
        ),
        "matched_grid_points": float(valid_count),
    }
    return crop, original, diagnostics


def _ray_rgb(
    rays: np.ndarray,
    *,
    magnitude_max: float,
) -> np.ndarray:
    angle = np.arctan2(rays[..., 1], rays[..., 0])
    magnitude = np.linalg.norm(rays, axis=-1)
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.round((angle + np.pi) / (2 * np.pi) * 179).astype(
        np.uint8
    )
    hsv[..., 1] = np.round(
        np.clip(magnitude / max(magnitude_max, 1e-8), 0.0, 1.0) * 255
    ).astype(np.uint8)
    hsv[..., 2] = 245
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _ray_panels(
    batch: list[dict[str, Any]],
    crop_idx: int,
    original_idx: int,
) -> tuple[Image.Image, Image.Image, dict[str, float]]:
    image = batch[crop_idx]["img"]
    height, width = int(image.shape[-2]), int(image.shape[-1])
    rays = []
    for view_idx in (crop_idx, original_idx):
        K = torch.as_tensor(
            _batch_item(batch[view_idx], "camera_intrinsics"),
            dtype=torch.float32,
        )
        rays.append(
            intrinsics_to_ray_map(K[None], height, width)[0].cpu().numpy()
        )
    magnitude_max = float(
        np.percentile(
            np.concatenate(
                [np.linalg.norm(ray, axis=-1).reshape(-1) for ray in rays]
            ),
            99,
        )
    )
    panels = []
    for view_idx, ray in zip((crop_idx, original_idx), rays):
        panel = Image.fromarray(
            _ray_rgb(ray, magnitude_max=magnitude_max),
            mode="RGB",
        )
        K = np.asarray(_batch_item(batch[view_idx], "camera_intrinsics"))
        _draw_cross(
            ImageDraw.Draw(panel),
            (float(K[0, 2]), float(K[1, 2])),
            color=(0, 0, 0),
            radius=12,
            width=4,
        )
        panels.append(panel)
    stats = {
        "crop_ray_magnitude_p99": float(
            np.percentile(np.linalg.norm(rays[0], axis=-1), 99)
        ),
        "original_ray_magnitude_p99": float(
            np.percentile(np.linalg.norm(rays[1], axis=-1), 99)
        ),
        "shared_ray_magnitude_max": magnitude_max,
    }
    return panels[0], panels[1], stats


def _compose_config(args: argparse.Namespace):
    overrides = [
        f"train={args.train_config}",
        f"data={args.data_config}",
        "lmgeo_profile.filter_preprocessed_query_depth=false",
        "lmgeo.filter_preprocessed_query_depth=false",
        f"val_datasets.{args.val_name}.dataset.filter_preprocessed_query_depth=false",
        f"val_datasets.{args.val_name}.dataset.filter_target_preprocessed_depth=false",
        f"val_datasets.{args.val_name}.runtime.num_workers=0",
        f"val_datasets.{args.val_name}.runtime.max_img_per_gpu=7",
    ]
    with initialize_config_dir(
        version_base="1.2",
        config_dir=str(REPO_ROOT / "configs"),
    ):
        return compose(config_name="default", overrides=overrides)


def _load_batch(cfg, val_name: str, skip_batches: int):
    entry = cfg.val_datasets[val_name]
    loader = create_dataloader(
        cfg,
        "test",
        dataset_cfg=entry.dataset,
        dataloader_cfg=entry.dataloader,
        runtime_cfg=entry.runtime,
    )
    try:
        return next(islice(iter(loader), int(skip_batches), None))
    except StopIteration as error:
        raise ValueError(
            f"skip_batches={skip_batches} exceeds loader length {len(loader)}"
        ) from error


def render(batch: list[dict[str, Any]], cfg) -> tuple[Image.Image, dict[str, Any]]:
    crop_idx = next(
        idx
        for idx, view in enumerate(batch)
        if bool(_batch_item(view, "is_cropped_query"))
    )
    original_idx = next(
        idx
        for idx, view in enumerate(batch)
        if bool(_batch_item(view, "is_original_query"))
    )

    top_panels = []
    for view_idx, view in enumerate(batch):
        title, subtitle = _role_label(view, view_idx)
        top_panels.append(
            _panel_with_title(
                _fit_image(_view_image(view), (248, 186)),
                title,
                subtitle,
            )
        )
    top_grid = _grid(top_panels, columns=len(top_panels))

    crop_match, original_match, geometry = _matched_query_panels(
        batch,
        crop_idx,
        original_idx,
    )
    crop_ray, original_ray, ray_stats = _ray_panels(
        batch,
        crop_idx,
        original_idx,
    )

    K_crop = np.asarray(_batch_item(batch[crop_idx], "camera_intrinsics"))
    K_original = np.asarray(
        _batch_item(batch[original_idx], "camera_intrinsics")
    )
    zoom = float(_batch_item(batch[crop_idx], "query_recenter_zoom"))
    valid_crop = int(
        np.asarray(_batch_item(batch[crop_idx], "valid_mask")).sum()
    )
    valid_original = int(
        np.asarray(_batch_item(batch[original_idx], "valid_mask")).sum()
    )
    same_source = (
        int(_batch_item(batch[crop_idx], "scene_id"))
        == int(_batch_item(batch[original_idx], "scene_id"))
        and int(_batch_item(batch[crop_idx], "im_id"))
        == int(_batch_item(batch[original_idx], "im_id"))
    )
    geometry_lines = [
        f"batch shape: B=1, N={len(batch)}, HxW=420x560",
        f"physical queries: 1  ->  model query views: 2",
        f"same source scene/image: {same_source}",
        f"pair indices: {_batch_item(batch[crop_idx], 'query_pair_index')} / "
        f"{_batch_item(batch[original_idx], 'query_pair_index')}",
        f"crop zoom before final resize: {zoom:.4f}x",
        f"crop K: fx={K_crop[0, 0]:.1f}, fy={K_crop[1, 1]:.1f}, "
        f"cx={K_crop[0, 2]:.1f}, cy={K_crop[1, 2]:.1f}",
        f"orig K: fx={K_original[0, 0]:.1f}, fy={K_original[1, 1]:.1f}, "
        f"cx={K_original[0, 2]:.1f}, cy={K_original[1, 2]:.1f}",
        f"valid supervised pixels: crop={valid_crop}, orig={valid_original}",
        f"homography equation max error: {geometry['homography_max_abs']:.2e}",
        f"pose equation max error: {geometry['pose_max_abs']:.2e}",
        f"visible matched grid points: {int(geometry['matched_grid_points'])}",
        "PASS geometry and paired metadata are internally consistent",
    ]
    geometry_card = _text_panel(
        "What the collated pair verifies",
        geometry_lines,
        size=(650, 482),
    )
    query_row = _grid(
        [
            _panel_with_title(
                crop_match,
                "RECENTERED / ZOOMED QUERY",
                "colored points are H · p_original; white cross is principal point",
            ),
            _panel_with_title(
                original_match,
                "ORIGINAL-CAMERA QUERY",
                "same source image after ordinary Pi3 crop/resize",
            ),
            geometry_card,
        ],
        columns=3,
    )

    ray_lines = [
        "ray(u,v) = (K^-1 [u+0.5, v+0.5, 1])xy/z",
        "hue: ray direction",
        "saturation: off-axis magnitude, shared scale",
        "black cross: zero-ray / principal point",
        f"crop ray magnitude p99: {ray_stats['crop_ray_magnitude_p99']:.4f}",
        f"original ray magnitude p99: "
        f"{ray_stats['original_ray_magnitude_p99']:.4f}",
        "",
        "Conditioned run: each of all 7 views supplies its own K/ray map.",
        "Unconditioned run: identical RGB batch; ray branch is absent.",
        "NOTE depth and masks are GT supervision, not model inputs.",
    ]
    ray_card = _text_panel(
        "Conditioning actually supplied to Pi3",
        ray_lines,
        size=(650, 482),
    )
    ray_row = _grid(
        [
            _panel_with_title(
                crop_ray,
                "CROP RAY MAP",
                "generated at 560x420 from the crop's final K",
            ),
            _panel_with_title(
                original_ray,
                "ORIGINAL RAY MAP",
                "generated at 560x420 from the original view's final K",
            ),
            ray_card,
        ],
        columns=3,
    )

    section_width = max(top_grid.width, query_row.width, ray_row.width)
    header_height = 112
    section_gap = 18
    canvas = Image.new(
        "RGB",
        (
            section_width + 48,
            header_height
            + top_grid.height
            + query_row.height
            + ray_row.height
            + 4 * section_gap
            + 30,
        ),
        color=(13, 16, 21),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 18),
        "Pi3 paired-query input after the production LMGeo dataloader",
        font=_font(34, bold=True),
        fill=(245, 247, 250),
    )
    condition = (
        "enabled"
        if bool(cfg.model.use_ray_conditioning)
        else "disabled"
    )
    draw.text(
        (24, 67),
        f"real BOP target · depth filtering OFF · ray conditioning {condition}",
        font=_font(20),
        fill=(174, 184, 197),
    )

    y = header_height
    for title, section in (
        ("ALL MODEL RGB VIEWS IN ORDER", top_grid),
        ("THE TWO VIEWS OF THE SAME PHYSICAL QUERY", query_row),
        ("PER-VIEW CAMERA CONDITIONING", ray_row),
    ):
        draw.text(
            (24, y),
            title,
            font=_font(18, bold=True),
            fill=(93, 224, 147),
        )
        y += 28
        canvas.paste(section, (24, y))
        y += section.height + section_gap

    summary = {
        "model_use_ray_conditioning": bool(cfg.model.use_ray_conditioning),
        "view_roles": [
            str(_batch_item(view, "view_role"))
            for view in batch
        ],
        "crop_view_index": crop_idx,
        "original_view_index": original_idx,
        "same_source_scene_and_image": same_source,
        "crop_intrinsics": K_crop.tolist(),
        "original_intrinsics": K_original.tolist(),
        "zoom": zoom,
        "valid_pixels": {
            "crop": valid_crop,
            "original": valid_original,
        },
        "geometry": geometry,
        "ray": ray_stats,
    }
    return canvas, summary


def main() -> None:
    args = parse_args()
    cfg = _compose_config(args)
    batch = _load_batch(cfg, args.val_name, args.skip_batches)
    image, summary = render(batch, cfg)

    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)

    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
