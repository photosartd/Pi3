from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from pi3.metrics.utils import (
    batch_object_ids,
    estimate_world_to_object_sim3,
    extract_prediction,
    stack_view_tensor,
    view_bool_mask,
)


def as_numpy(value) -> np.ndarray:
    """Detach tensors and convert values to NumPy arrays."""

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_image_to_uint8(image) -> np.ndarray:
    """Convert a Pi3 image tensor to uint8 RGB."""

    array = as_numpy(image).astype(np.float32)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.min() < -0.05:
        array = array * 0.5 + 0.5
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).round().astype(np.uint8)


def mask_from_depth(depth: np.ndarray) -> np.ndarray:
    """Return a finite positive-depth mask."""

    return np.isfinite(depth) & (depth > 0)


def colorize_values(values, *, mask=None, vmin=None, vmax=None, cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """Apply an OpenCV colormap to finite scalar values."""

    values = np.asarray(values, dtype=np.float32)
    if mask is None:
        mask = np.isfinite(values)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(values)

    if not np.any(mask):
        return np.zeros((*values.shape, 3), dtype=np.uint8)

    valid = values[mask]
    if vmin is None:
        vmin = float(np.percentile(valid, 2))
    if vmax is None:
        vmax = float(np.percentile(valid, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    norm = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cmap)[:, :, ::-1]
    colored[~mask] = 0
    return colored


def pil_from_array(array: np.ndarray) -> Image.Image:
    """Create an RGB PIL image from a uint8 array."""

    array = np.asarray(array)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def add_title(image: Image.Image, title: str, *, height: int = 18) -> Image.Image:
    """Add a small title strip above an image."""

    canvas = Image.new("RGB", (image.width, image.height + height), color=(25, 25, 25))
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), str(title), fill=(240, 240, 240))
    return canvas


def make_grid(images: Iterable[Image.Image], *, columns: int | None = None, padding: int = 4) -> Image.Image:
    """Tile images into a simple RGB grid."""

    images = list(images)
    if not images:
        return Image.new("RGB", (1, 1), color=(0, 0, 0))
    if columns is None:
        columns = len(images)
    columns = max(1, int(columns))
    rows = int(np.ceil(len(images) / columns))
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new(
        "RGB",
        (columns * width + (columns - 1) * padding, rows * height + (rows - 1) * padding),
        color=(10, 10, 10),
    )
    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        canvas.paste(image, (col * (width + padding), row * (height + padding)))
    return canvas


def choose_view_indices(indices, max_views: int, rng: Any | None = None) -> np.ndarray:
    """Choose a capped, random subset of view indices while keeping display order."""

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(indices) <= int(max_views):
        return indices
    if rng is None:
        sampled = np.linspace(0, len(indices) - 1, int(max_views)).round().astype(np.int64)
        return indices[sampled]
    return np.sort(rng.choice(indices, size=int(max_views), replace=False).astype(np.int64))


def alignment_from_batch(
    prediction,
    batch,
    *,
    batch_idx: int = 0,
    solve_scale: bool = True,
    scale_estimation: str = "camera_centers",
    min_depth_pixels_per_view: int = 64,
):
    """Estimate reference Sim(3) and return cached batch arrays."""

    pred = extract_prediction(prediction)
    pred_T_W_C = pred["camera_poses"].detach().float().cpu().numpy()
    gt_T_C_O = stack_view_tensor(batch, "T_C_O").astype(np.float64)
    ref_mask = view_bool_mask(batch, "is_reference")
    refs = ref_mask[batch_idx]
    if refs.sum() == 0:
        return None, pred_T_W_C, gt_T_C_O, ref_mask
    alignment = estimate_world_to_object_sim3(
        pred_T_W_C[batch_idx, refs],
        gt_T_C_O[batch_idx, refs],
        solve_scale=solve_scale,
        scale_estimation=scale_estimation,
        pred_local_points_refs=(
            pred["local_points"].detach().float().cpu().numpy()[batch_idx, refs]
            if scale_estimation == "reference_depth"
            else None
        ),
        gt_points_object_refs=(
            stack_view_tensor(batch, "pts3d").astype(np.float64)[batch_idx, refs]
            if scale_estimation == "reference_depth"
            else None
        ),
        valid_masks_refs=(
            stack_view_tensor(batch, "valid_mask").astype(bool)[batch_idx, refs]
            if scale_estimation == "reference_depth"
            else None
        ),
        min_depth_pixels_per_view=min_depth_pixels_per_view,
    )
    return alignment, pred_T_W_C, gt_T_C_O, ref_mask


def object_bbox_corners(points: np.ndarray) -> np.ndarray:
    """Return the eight axis-aligned object-space bbox corners."""

    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=np.float64,
    )


BBOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def project_points(points_object: np.ndarray, T_C_O: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project object-frame 3D points with BOP-style ``T_C_O``."""

    points_camera = points_object @ T_C_O[:3, :3].T + T_C_O[:3, 3]
    depth = points_camera[:, 2]
    valid = depth > 1e-8
    uv = np.full((len(points_object), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = K[0, 0] * points_camera[valid, 0] / depth[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * points_camera[valid, 1] / depth[valid] + K[1, 2]
    return uv, valid


def draw_line_if_visible(draw: ImageDraw.ImageDraw, uv: np.ndarray, valid: np.ndarray, edge, color, width: int) -> None:
    """Draw one projected edge when both endpoints are in front of the camera."""

    i, j = edge
    if not (valid[i] and valid[j]):
        return
    if not np.isfinite(uv[[i, j]]).all():
        return
    draw.line([tuple(uv[i]), tuple(uv[j])], fill=color, width=width)


def draw_pose_projection(
    image: Image.Image,
    *,
    T_C_O: np.ndarray,
    K: np.ndarray,
    bbox_corners: np.ndarray,
    axis_length: float,
    bbox_color: tuple[int, int, int],
    axis_alpha: bool = True,
    width: int = 2,
) -> None:
    """Draw projected bbox and object axes onto an image in-place."""

    draw = ImageDraw.Draw(image)
    uv, valid = project_points(bbox_corners, T_C_O, K)
    for edge in BBOX_EDGES:
        draw_line_if_visible(draw, uv, valid, edge, bbox_color, width)

    axis = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )
    axis_uv, axis_valid = project_points(axis, T_C_O, K)
    colors = [(255, 70, 70), (70, 255, 70), (80, 160, 255)]
    for idx, color in enumerate(colors, start=1):
        if axis_alpha:
            draw_line_if_visible(draw, axis_uv, axis_valid, (0, idx), color, width + 1)


def draw_mask_contour(image: Image.Image, mask: np.ndarray, *, color=(255, 230, 0), width: int = 1) -> None:
    """Draw a mask contour onto an image in-place."""

    mask = np.asarray(mask, dtype=np.uint8)
    if mask.max() == 0:
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = ImageDraw.Draw(image)
    for contour in contours:
        if len(contour) < 2:
            continue
        points = [tuple(map(int, point[0])) for point in contour]
        draw.line(points + [points[0]], fill=color, width=width)


def first_object_id(batch: list[dict]) -> int:
    """Return the first object id from a collated batch."""

    return object_id_at(batch, 0)


def object_id_at(batch: list[dict], batch_idx: int = 0) -> int:
    """Return the object id for one sample in a collated multi-view batch."""

    return int(batch_object_ids(batch)[int(batch_idx)])
