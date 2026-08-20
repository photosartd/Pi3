#!/usr/bin/env python3
"""Small geometry helpers for the MegaPose-GSO keyframe audit notebook.

The source SQLite index deliberately stores metadata and random-access byte
ranges, not derived mesh-coverage labels.  This module keeps the expensive
geometry operations testable while the notebook owns exploration, plotting,
and the optional sidecar cache.

Surface coverage is estimated with the 2,000 uniformly sampled GSO surface
points from ``models_pointcloud``.  A model sample is observed in a frame when
its GT-pose projection lands on the released visible mask and agrees with the
released metric depth.  This accounts for self-occlusion and scene occlusion
without requiring MegaPose/Panda3D in the training environment.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.optimize import minimize


def unpack_matrix(value: bytes, rows: int, columns: int) -> np.ndarray:
    """Decode a little-endian float32 matrix from the source SQLite index."""

    matrix = np.frombuffer(value, dtype="<f4")
    expected = int(rows) * int(columns)
    if matrix.size != expected:
        raise ValueError(f"Expected {expected} float32 values, got {matrix.size}")
    return matrix.reshape(int(rows), int(columns)).astype(np.float64, copy=True)


def object_view_direction(T_C_O: np.ndarray) -> np.ndarray:
    """Unit object-to-camera-center direction in object coordinates."""

    T_C_O = np.asarray(T_C_O, dtype=np.float64)
    if T_C_O.shape != (4, 4) or not np.isfinite(T_C_O).all():
        raise ValueError(f"Invalid object-to-camera pose with shape {T_C_O.shape}")
    center_O = -(T_C_O[:3, :3].T @ T_C_O[:3, 3])
    norm = float(np.linalg.norm(center_O))
    if norm <= 1e-12:
        raise ValueError("Camera center coincides with the object origin")
    return center_O / norm


def load_surface_pointcloud(path: Path, scale_m: float) -> np.ndarray:
    """Load the vertex-only GSO OBJ point cloud and convert it to metres."""

    points = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                if len(values) < 4:
                    raise ValueError(f"Malformed OBJ vertex in {path}: {line!r}")
                points.append(tuple(float(value) for value in values[1:4]))
    if not points:
        raise ValueError(f"No OBJ vertices found in {path}")
    output = np.asarray(points, dtype=np.float64) * float(scale_m)
    if not np.isfinite(output).all():
        raise ValueError(f"Non-finite surface point in {path}")
    return output


def read_indexed_payload(
    data_root: Path, record: Mapping[str, object], payload: str
) -> bytes:
    """Read one tar member directly using the byte range stored in SQLite."""

    relative_path = Path(str(record["relative_path"]))
    path = relative_path if relative_path.is_absolute() else Path(data_root) / relative_path
    size = int(record[f"{payload}_size"])
    descriptor = os.open(path, os.O_RDONLY)
    try:
        value = os.pread(descriptor, size, int(record[f"{payload}_offset"]))
    finally:
        os.close(descriptor)
    if len(value) != size:
        raise IOError(f"Short read for {path}:{payload}: {len(value)}/{size}")
    return value


def decode_uncompressed_rle(rle: Mapping[str, object]) -> np.ndarray:
    """Decode the uncompressed COCO RLE representation used by this dataset."""

    height, width = (int(value) for value in rle["size"])  # type: ignore[index]
    counts = rle.get("counts")
    if not isinstance(counts, list):
        raise ValueError("Compressed COCO RLE is not supported")
    flat = np.zeros(height * width, dtype=bool)
    position = 0
    for run_index, raw_count in enumerate(counts):
        count = int(raw_count)
        if count < 0 or position + count > flat.size:
            raise ValueError("Invalid RLE run")
        if run_index % 2:
            flat[position : position + count] = True
        position += count
    if position != flat.size:
        raise ValueError("RLE does not cover the full mask")
    return flat.reshape((height, width), order="F")


def decode_instance_mask(payload: bytes, gt_id: int) -> np.ndarray:
    """Decode one target instance from a frame-level mask payload."""

    value = json.loads(payload)
    if isinstance(value, dict):
        rle = value[str(int(gt_id))]
    elif isinstance(value, list):
        rle = value[int(gt_id)]
    else:
        raise ValueError(f"Unexpected mask payload type {type(value).__name__}")
    return decode_uncompressed_rle(rle)


def load_source_geometry(
    data_root: Path, record: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    """Load metric depth and the released visible-instance mask for one row."""

    depth_bytes = read_indexed_payload(data_root, record, "depth")
    with Image.open(io.BytesIO(depth_bytes)) as image:
        depth = np.asarray(image).astype(np.float32) * float(record["depth_unit_m"])
    visible = decode_instance_mask(
        read_indexed_payload(data_root, record, "mask_visib"), int(record["gt_id"])
    )
    expected = (int(record["height"]), int(record["width"]))
    if depth.shape != expected or visible.shape != expected:
        raise ValueError(
            f"Geometry shape mismatch: depth={depth.shape}, mask={visible.shape}, "
            f"expected={expected}"
        )
    return depth, visible


def visible_surface_sample_mask(
    surface_points_O: np.ndarray,
    T_C_O: np.ndarray,
    K: np.ndarray,
    depth_m: np.ndarray,
    visible_mask: np.ndarray,
    *,
    depth_tolerance_m: float = 0.002,
    depth_tolerance_relative: float = 0.002,
    pixel_radius: int = 1,
) -> np.ndarray:
    """Estimate which fixed CAD surface samples are observed in a source frame.

    Projection is checked against a small image neighbourhood.  This avoids
    marking back-facing/self-occluded samples as visible and is less brittle to
    nearest-pixel quantization on slanted surfaces.
    """

    points = np.asarray(surface_points_O, dtype=np.float64)
    T_C_O = np.asarray(T_C_O, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(visible_mask, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"surface_points_O must have shape (P, 3), got {points.shape}")
    if T_C_O.shape != (4, 4) or K.shape != (3, 3):
        raise ValueError("T_C_O and K must have shapes (4,4) and (3,3)")
    if depth.shape != mask.shape:
        raise ValueError("depth and visible_mask shapes differ")
    if depth_tolerance_m < 0 or depth_tolerance_relative < 0 or pixel_radius < 0:
        raise ValueError("Visibility tolerances must be non-negative")

    points_C = points @ T_C_O[:3, :3].T + T_C_O[:3, 3]
    z = points_C[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-8)
    projected = points_C @ K.T
    u = np.zeros(len(points), dtype=np.int64)
    v = np.zeros(len(points), dtype=np.int64)
    u[valid_z] = np.rint(
        projected[valid_z, 0] / projected[valid_z, 2]
    ).astype(np.int64)
    v[valid_z] = np.rint(
        projected[valid_z, 1] / projected[valid_z, 2]
    ).astype(np.int64)
    height, width = depth.shape
    observed = np.zeros(len(points), dtype=bool)
    tolerance = float(depth_tolerance_m) + float(depth_tolerance_relative) * z
    for dv in range(-int(pixel_radius), int(pixel_radius) + 1):
        for du in range(-int(pixel_radius), int(pixel_radius) + 1):
            x = u + du
            y = v + dv
            inside = valid_z & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            indices = np.flatnonzero(inside & ~observed)
            if not len(indices):
                continue
            sampled_depth = depth[y[indices], x[indices]]
            agrees = (
                mask[y[indices], x[indices]]
                & np.isfinite(sampled_depth)
                & (sampled_depth > 0)
                & (np.abs(sampled_depth - z[indices]) <= tolerance[indices])
            )
            observed[indices[agrees]] = True
    return observed


def max_pairwise_angle_degrees(directions: np.ndarray) -> float:
    """Maximum angular separation of unit directions."""

    values = _unit_directions(directions)
    if len(values) < 2:
        return 0.0
    minimum_dot = float(np.min(values @ values.T))
    return math.degrees(math.acos(float(np.clip(minimum_dot, -1.0, 1.0))))


def minimum_enclosing_cap(directions: np.ndarray) -> dict[str, object]:
    """Approximate the minimum enclosing spherical cap with deterministic starts."""

    values = _unit_directions(directions)
    if len(values) == 1:
        center = values[0]
        radius = 0.0
    else:
        mean = values.mean(axis=0)
        starts = [values[index] for index in range(len(values))]
        if float(np.linalg.norm(mean)) > 1e-12:
            starts.insert(0, mean / np.linalg.norm(mean))

        def objective(candidate: np.ndarray) -> float:
            norm = float(np.linalg.norm(candidate))
            if norm <= 1e-12:
                return 2.0
            return -float(np.min(values @ (candidate / norm)))

        best = None
        constraint = {"type": "eq", "fun": lambda candidate: np.dot(candidate, candidate) - 1.0}
        for start in starts:
            result = minimize(
                objective,
                start,
                method="SLSQP",
                constraints=constraint,
                options={"ftol": 1e-12, "maxiter": 300},
            )
            if result.success and (best is None or result.fun < best.fun):
                best = result
        center = (
            np.asarray(best.x, dtype=np.float64)
            if best is not None
            else values[int(np.argmax(np.min(values @ values.T, axis=0)))]
        )
        center /= np.linalg.norm(center)
        radius = math.acos(float(np.clip(np.min(values @ center), -1.0, 1.0)))
    return {
        "center": center,
        "radius_degrees": math.degrees(radius),
        "solid_angle_steradians": 2.0 * math.pi * (1.0 - math.cos(radius)),
        "solid_angle_fraction_of_sphere": (1.0 - math.cos(radius)) / 2.0,
    }


def greedy_coverage_curve(
    frame_surface_masks: np.ndarray,
    directions: np.ndarray,
    maximum_views: int = 8,
    *,
    minimum_pairwise_angle_degrees: float = 0.0,
) -> dict[str, object]:
    """Greedily maximize union coverage for every prefix up to ``maximum_views``.

    The default relies on marginal surface gain itself to reject redundant
    views.  A hard angular constraint is optional because it can reject nearby
    views that expose different concavities or external occlusions.
    """

    masks = np.asarray(frame_surface_masks, dtype=bool)
    values = _unit_directions(directions)
    if masks.ndim != 2 or masks.shape[0] != len(values):
        raise ValueError("frame_surface_masks must be (frames, surface samples)")
    if maximum_views <= 0 or minimum_pairwise_angle_degrees < 0:
        raise ValueError("View limits and angular threshold must be non-negative")
    cosine_limit = math.cos(math.radians(float(minimum_pairwise_angle_degrees)))
    selected: list[int] = []
    covered = np.zeros(masks.shape[1], dtype=bool)
    coverage = []
    gains = []
    for _ in range(min(int(maximum_views), len(masks))):
        best_index = None
        best_gain = -1
        for index in range(len(masks)):
            if index in selected:
                continue
            if selected and minimum_pairwise_angle_degrees > 0:
                if bool((values[selected] @ values[index] > cosine_limit).any()):
                    continue
            gain = int(np.count_nonzero(masks[index] & ~covered))
            if gain > best_gain:
                best_index, best_gain = index, gain
        if best_index is None:
            break
        selected.append(best_index)
        covered |= masks[best_index]
        gains.append(best_gain)
        coverage.append(float(np.mean(covered)))
    return {
        "selected_indices": selected,
        "marginal_sample_gains": gains,
        "coverage_fractions": coverage,
        "covered_mask": covered,
    }


def pack_surface_mask(mask: np.ndarray) -> bytes:
    """Pack one boolean surface mask for compact SQLite BLOB storage."""

    return np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little").tobytes()


def unpack_surface_mask(value: bytes, point_count: int) -> np.ndarray:
    """Inverse of :func:`pack_surface_mask`."""

    return np.unpackbits(
        np.frombuffer(value, dtype=np.uint8), bitorder="little", count=int(point_count)
    ).astype(bool)


def _unit_directions(directions: np.ndarray) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"directions must have shape (N, 3), got {values.shape}")
    norms = np.linalg.norm(values, axis=1)
    if bool((norms <= 1e-12).any()):
        raise ValueError("Zero-length viewing direction")
    return values / norms[:, None]
