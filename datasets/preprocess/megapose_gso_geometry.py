"""Derived geometry index for constraint-aware MegaPose-GSO sampling.

The primary :mod:`datasets.preprocess.megapose_gso` SQLite database remains the
immutable source of truth.  This module builds a resumable sidecar containing
only quantities that are expensive to derive repeatedly:

* object-to-camera viewing directions;
* object-preserving crop/focal feasibility intervals;
* visible CAD-surface bitsets estimated from GT pose, depth and visible masks.

The sidecar is deliberately policy-light.  Visibility, reference count,
coverage, focal, positive-view and query-count thresholds are applied by
``collect_capacity_statistics`` without rebuilding the surface observations.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image


FORMAT = "pi3_megapose_gso_geometry_v1"
ALGORITHM = "area_surface_samples_depth_visible_mask_v1"
PLAN_FORMAT = "pi3_object_geometry_plan_v1"
POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def _portable_metadata_path(owner_path: Path, target_path: Path) -> str:
    """Store a path relative to the SQLite that owns the metadata."""

    owner_path = Path(owner_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()
    return os.path.relpath(target_path, start=owner_path.parent)


def _resolve_metadata_path(owner_path: Path, stored_path: str | Path) -> Path:
    """Read portable metadata and relocate legacy absolute sibling paths."""

    owner_path = Path(owner_path).expanduser().resolve()
    raw_path = Path(stored_path).expanduser()
    if not raw_path.is_absolute():
        return (owner_path.parent / raw_path).resolve()
    resolved = raw_path.resolve()
    if resolved.is_file():
        return resolved
    relocated = (owner_path.parent / raw_path.name).resolve()
    return relocated if relocated.is_file() else resolved


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS frame_status (
    frame_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    eligible_instance_count INTEGER NOT NULL,
    elapsed_seconds REAL NOT NULL,
    error TEXT
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS frame_features (
    feature_id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL,
    gt_id INTEGER NOT NULL,
    object_id INTEGER NOT NULL,
    scene_id INTEGER NOT NULL,
    view_id INTEGER NOT NULL,
    visib_fract REAL NOT NULL,
    px_count_all INTEGER NOT NULL,
    px_count_valid INTEGER NOT NULL,
    px_count_visib INTEGER NOT NULL,
    depth_corrupt INTEGER,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    view_x REAL NOT NULL,
    view_y REAL NOT NULL,
    view_z REAL NOT NULL,
    fx REAL NOT NULL,
    fy REAL NOT NULL,
    cx REAL NOT NULL,
    cy REAL NOT NULL,
    bbox_obj_x REAL NOT NULL,
    bbox_obj_y REAL NOT NULL,
    bbox_obj_w REAL NOT NULL,
    bbox_obj_h REAL NOT NULL,
    bbox_visib_x REAL NOT NULL,
    bbox_visib_y REAL NOT NULL,
    bbox_visib_w REAL NOT NULL,
    bbox_visib_h REAL NOT NULL,
    crop_width_min REAL NOT NULL,
    crop_width_max REAL NOT NULL,
    crop_feasible INTEGER NOT NULL,
    bbox_obj_touches_border INTEGER NOT NULL,
    base_norm_fx REAL NOT NULL,
    base_norm_fy REAL NOT NULL,
    max_norm_fx REAL NOT NULL,
    max_norm_fy REAL NOT NULL,
    surface_point_count INTEGER NOT NULL,
    observed_point_count INTEGER NOT NULL,
    observed_fraction REAL NOT NULL,
    bit_offset INTEGER NOT NULL,
    UNIQUE(frame_id, gt_id)
);
"""


SECONDARY_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_geometry_object_track "
    "ON frame_features(object_id, scene_id, gt_id, view_id)",
    "CREATE INDEX IF NOT EXISTS idx_geometry_query "
    "ON frame_features(object_id, visib_fract, scene_id, gt_id)",
)


def _drop_secondary_indexes(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_geometry_object_track")
    connection.execute("DROP INDEX IF EXISTS idx_geometry_query")


def _create_secondary_indexes(connection: sqlite3.Connection) -> None:
    for statement in SECONDARY_INDEX_SQL:
        connection.execute(statement)


SOURCE_FRAME_QUERY = """
SELECT
    f.id AS frame_id, f.scene_id, f.view_id, f.width, f.height,
    f.depth_unit_m, f.K_f32, f.depth_offset, f.depth_size,
    f.mask_visib_offset, f.mask_visib_size, s.relative_path,
    i.gt_id, i.object_id, i.T_C_O_f32, i.visib_fract,
    i.px_count_all, i.px_count_valid, i.px_count_visib, i.depth_corrupt,
    i.bbox_obj_x, i.bbox_obj_y, i.bbox_obj_w, i.bbox_obj_h,
    i.bbox_visib_x, i.bbox_visib_y, i.bbox_visib_w, i.bbox_visib_h
FROM instances AS i
JOIN frames AS f ON f.id=i.frame_id
JOIN shards AS s ON s.id=f.shard_id
JOIN active_objects AS ao ON ao.object_id=i.object_id
JOIN active_scenes AS ac ON ac.scene_id=i.scene_id
WHERE i.visib_fract >= ? AND i.px_count_visib >= ?
  AND (
    ? = 'all'
    OR (? = 'clean_only' AND i.depth_corrupt = 0)
    OR (? = 'exclude_known_bad' AND i.depth_corrupt IS NOT 1)
  )
ORDER BY f.id, i.gt_id
"""


@dataclass(frozen=True)
class BuildSettings:
    split: str = "train"
    visibility_floor: float = 0.1
    min_visible_pixels: int = 64
    depth_corruption_policy: str = "clean_only"
    surface_point_count: int = 8192
    surface_seed: int = 20260814
    crop_aspect: float = 4.0 / 3.0
    crop_margin_fraction: float = 0.05
    depth_tolerance_m: float = 0.002
    depth_tolerance_relative: float = 0.002
    pixel_radius: int = 1

    def validate(self) -> None:
        if self.split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val, or all")
        if not 0.0 <= self.visibility_floor <= 1.0:
            raise ValueError("visibility_floor must be in [0,1]")
        if self.min_visible_pixels <= 0 or self.surface_point_count <= 0:
            raise ValueError("pixel and surface-point counts must be positive")
        if self.depth_corruption_policy not in {
            "clean_only", "exclude_known_bad", "all"
        }:
            raise ValueError("invalid depth_corruption_policy")
        if self.crop_aspect <= 0 or self.crop_margin_fraction < 0:
            raise ValueError("crop settings must be non-negative")
        if self.depth_tolerance_m < 0 or self.depth_tolerance_relative < 0:
            raise ValueError("depth tolerances must be non-negative")
        if self.pixel_radius < 0:
            raise ValueError("pixel_radius must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class FrameJob:
    frame_id: int
    scene_id: int
    view_id: int
    width: int
    height: int
    depth_unit_m: float
    K_f32: bytes
    depth_offset: int
    depth_size: int
    mask_visib_offset: int
    mask_visib_size: int
    relative_path: str
    instances: tuple[Mapping[str, Any], ...]


@dataclass
class DecodedFrame:
    job: FrameJob
    depth: np.ndarray
    masks: np.ndarray


@dataclass(frozen=True)
class FrameResult:
    frame_id: int
    elapsed_seconds: float
    features: tuple[dict[str, Any], ...]
    observed_bits: tuple[bytes, ...]
    error: str | None = None


def readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{Path(path).resolve()}?mode=ro&immutable=1", uri=True, timeout=60.0
    )
    connection.row_factory = sqlite3.Row
    return connection


def unpack_matrix(value: bytes, rows: int, columns: int) -> np.ndarray:
    output = np.frombuffer(value, dtype="<f4")
    if output.size != int(rows) * int(columns):
        raise ValueError(f"Invalid matrix blob with {output.size} values")
    return output.reshape(int(rows), int(columns)).astype(np.float32, copy=True)


def object_view_direction(T_C_O: np.ndarray) -> np.ndarray:
    T_C_O = np.asarray(T_C_O, dtype=np.float64)
    center = -(T_C_O[:3, :3].T @ T_C_O[:3, 3])
    norm = float(np.linalg.norm(center))
    if T_C_O.shape != (4, 4) or not np.isfinite(T_C_O).all() or norm <= 1e-12:
        raise ValueError("Invalid T_C_O for viewing direction")
    return (center / norm).astype(np.float32)


def load_split_membership(split_path: Path, split: str) -> tuple[set[int], set[int]]:
    if split == "all":
        return set(), set()
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if payload.get("format") != "pi3_entity_split_v1":
        raise ValueError(f"Unsupported split manifest {split_path}")
    values = payload.get("splits", {}).get(split)
    if not isinstance(values, dict):
        raise ValueError(f"Missing split {split!r} in {split_path}")
    objects = {int(value) for value in values.get("object_ids", [])}
    scenes = {int(value) for value in values.get("scene_ids", [])}
    if not objects or not scenes:
        raise ValueError(f"Empty split membership for {split!r}")
    return objects, scenes


def load_model_catalog(path: Path) -> tuple[Path, dict[int, Mapping[str, Any]]]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "pi3_megapose_gso_model_catalog_v1":
        raise ValueError(f"Unsupported GSO model catalogue {path}")
    root = Path(payload["assets_root"]).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    rows = payload.get("objects")
    if not isinstance(rows, list):
        raise ValueError("Model catalogue objects must be a list")
    output = {int(row["object_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("Duplicate object ID in model catalogue")
    return root, output


def load_obj_triangles(path: Path, scale_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Load OBJ vertices/faces and triangulate polygons without extra dependencies."""

    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    with Path(path).open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif line.startswith("f "):
                fields = line.split()[1:]
                indices = []
                for field in fields:
                    raw = int(field.split("/", 1)[0])
                    index = raw - 1 if raw > 0 else len(vertices) + raw
                    indices.append(index)
                for offset in range(1, len(indices) - 1):
                    triangles.append((indices[0], indices[offset], indices[offset + 1]))
    points = np.asarray(vertices, dtype=np.float64) * float(scale_m)
    faces = np.asarray(triangles, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(faces):
        raise ValueError(f"OBJ has no usable mesh: {path}")
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError(f"OBJ face index is outside vertex range: {path}")
    return points, faces


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Deterministically sample a triangle mesh proportional to surface area."""

    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = np.isfinite(areas) & (areas > 1e-18)
    triangles, areas = triangles[valid], areas[valid]
    if not len(triangles) or float(areas.sum()) <= 0:
        raise ValueError("Mesh has no non-degenerate triangles")
    probabilities = areas / areas.sum()
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(triangles), size=int(count), replace=True, p=probabilities)
    selected = triangles[chosen]
    first = np.sqrt(rng.random(int(count)))
    second = rng.random(int(count))
    weights = np.stack(
        (1.0 - first, first * (1.0 - second), first * second), axis=1
    )
    return np.einsum("ni,nij->nj", weights, selected).astype(np.float32)


def prepare_surface_atlas(
    model_catalog_path: Path,
    atlas_path: Path,
    settings: BuildSettings,
    *,
    progress: bool = True,
) -> tuple[np.memmap, dict[int, int]]:
    """Create or validate a dense float32 surface atlas."""

    root, models = load_model_catalog(model_catalog_path)
    object_ids = sorted(models)
    metadata_path = Path(str(atlas_path) + ".json")
    signature = {
        "format": FORMAT,
        "algorithm": "area_weighted_obj_surface_samples_v1",
        "model_catalog": str(Path(model_catalog_path).resolve()),
        "model_catalog_mtime_ns": Path(model_catalog_path).stat().st_mtime_ns,
        "surface_point_count": settings.surface_point_count,
        "surface_seed": settings.surface_seed,
        "object_ids": object_ids,
    }
    if Path(atlas_path).is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise ValueError(
                f"Surface atlas parameters changed; choose a new path: {atlas_path}"
            )
        expected = len(object_ids) * settings.surface_point_count * 3 * 4
        if Path(atlas_path).stat().st_size != expected:
            raise ValueError(f"Surface atlas has unexpected size: {atlas_path}")
        atlas = np.memmap(
            atlas_path,
            mode="r",
            dtype="<f4",
            shape=(len(object_ids), settings.surface_point_count, 3),
        )
        return atlas, {object_id: index for index, object_id in enumerate(object_ids)}

    Path(atlas_path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(atlas_path) + f".tmp.{os.getpid()}")
    atlas = np.memmap(
        temporary,
        mode="w+",
        dtype="<f4",
        shape=(len(object_ids), settings.surface_point_count, 3),
    )
    iterable: Iterable[int] = object_ids
    if progress:
        from tqdm.auto import tqdm

        iterable = tqdm(
            object_ids, desc="surface atlas", unit="object", mininterval=1.0
        )
    try:
        for atlas_index, object_id in enumerate(iterable):
            row = models[object_id]
            mesh_path = Path(str(row["normalized_obj"]))
            if not mesh_path.is_absolute():
                mesh_path = root / mesh_path
            vertices, faces = load_obj_triangles(
                mesh_path, float(row.get("normalized_mesh_scale_m", 0.1))
            )
            atlas[atlas_index] = sample_mesh_surface(
                vertices,
                faces,
                settings.surface_point_count,
                seed=settings.surface_seed + object_id * 1_000_003,
            )
        atlas.flush()
        del atlas
        os.replace(temporary, atlas_path)
        metadata_path.write_text(
            json.dumps(signature, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    atlas = np.memmap(
        atlas_path,
        mode="r",
        dtype="<f4",
        shape=(len(object_ids), settings.surface_point_count, 3),
    )
    return atlas, {object_id: index for index, object_id in enumerate(object_ids)}


def decode_uncompressed_rle(rle: Mapping[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in rle["size"])
    counts = rle.get("counts")
    if not isinstance(counts, list):
        raise ValueError("Compressed COCO RLE is unsupported")
    flat = np.zeros(height * width, dtype=np.uint8)
    position = 0
    for index, raw_count in enumerate(counts):
        count = int(raw_count)
        if count < 0 or position + count > flat.size:
            raise ValueError("Invalid RLE counts")
        if index % 2:
            flat[position : position + count] = 1
        position += count
    if position != flat.size:
        raise ValueError("RLE does not cover the image")
    return flat.reshape((height, width), order="F")


class _ThreadShardCache:
    def __init__(self, data_root: Path, maximum: int = 8):
        self.data_root = Path(data_root)
        self.maximum = int(maximum)
        self.local = threading.local()

    def _files(self) -> OrderedDict[str, int]:
        if not hasattr(self.local, "files"):
            self.local.files = OrderedDict()
        return self.local.files

    def read(self, relative_path: str, offset: int, size: int) -> bytes:
        files = self._files()
        descriptor = files.pop(relative_path, None)
        if descriptor is None:
            path = Path(relative_path)
            if not path.is_absolute():
                path = self.data_root / path
            descriptor = os.open(path, os.O_RDONLY)
        files[relative_path] = descriptor
        while len(files) > self.maximum:
            _, old = files.popitem(last=False)
            os.close(old)
        value = os.pread(descriptor, int(size), int(offset))
        if len(value) != int(size):
            raise IOError(f"Short indexed read for {relative_path}")
        return value


def decode_frame(job: FrameJob, reader: _ThreadShardCache) -> DecodedFrame:
    depth_payload = reader.read(job.relative_path, job.depth_offset, job.depth_size)
    with Image.open(io.BytesIO(depth_payload)) as image:
        depth = np.asarray(image).astype(np.float32) * float(job.depth_unit_m)
    mask_payload = reader.read(
        job.relative_path, job.mask_visib_offset, job.mask_visib_size
    )
    mask_values = json.loads(mask_payload)
    masks = []
    for row in job.instances:
        gt_id = int(row["gt_id"])
        rle = mask_values[str(gt_id)] if isinstance(mask_values, dict) else mask_values[gt_id]
        masks.append(decode_uncompressed_rle(rle))
    output = np.stack(masks, axis=0).astype(np.uint8, copy=False)
    expected = (job.height, job.width)
    if depth.shape != expected or output.shape[1:] != expected:
        raise ValueError(f"Decoded geometry shape mismatch for frame {job.frame_id}")
    return DecodedFrame(job=job, depth=depth, masks=output)


def crop_feasibility(
    row: Mapping[str, Any],
    K: np.ndarray,
    *,
    width: int,
    height: int,
    aspect: float,
    margin_fraction: float,
) -> dict[str, float | int]:
    """Compute the continuous object-preserving crop/zoom interval."""

    x = float(row["bbox_obj_x"])
    y = float(row["bbox_obj_y"])
    w = float(row["bbox_obj_w"])
    h = float(row["bbox_obj_h"])
    margin_x = w * float(margin_fraction)
    margin_y = h * float(margin_fraction)
    left, top = x - margin_x, y - margin_y
    right, bottom = x + w + margin_x, y + h + margin_y
    touches = int(x <= 0 or y <= 0 or x + w >= width or y + h >= height)
    expanded_inside = left >= 0 and top >= 0 and right <= width and bottom <= height
    crop_width_min = max(right - left, (bottom - top) * float(aspect))
    crop_width_max = min(float(width), float(height) * float(aspect))
    feasible = int(expanded_inside and crop_width_min <= crop_width_max + 1e-6)
    if not feasible:
        crop_width_min = crop_width_max
    crop_height_max = crop_width_max / float(aspect)
    crop_height_min = crop_width_min / float(aspect)
    return {
        "crop_width_min": crop_width_min,
        "crop_width_max": crop_width_max,
        "crop_feasible": feasible,
        "bbox_obj_touches_border": touches,
        "base_norm_fx": float(K[0, 0]) / crop_width_max,
        "base_norm_fy": float(K[1, 1]) / crop_height_max,
        "max_norm_fx": float(K[0, 0]) / crop_width_min,
        "max_norm_fy": float(K[1, 1]) / crop_height_min,
    }


def feature_metadata(
    job: FrameJob,
    row: Mapping[str, Any],
    observed_count: int,
    settings: BuildSettings,
) -> dict[str, Any]:
    K = unpack_matrix(job.K_f32, 3, 3)
    pose = unpack_matrix(row["T_C_O_f32"], 4, 4)
    direction = object_view_direction(pose)
    crop = crop_feasibility(
        row,
        K,
        width=job.width,
        height=job.height,
        aspect=settings.crop_aspect,
        margin_fraction=settings.crop_margin_fraction,
    )
    output = {
        "frame_id": job.frame_id,
        "gt_id": int(row["gt_id"]),
        "object_id": int(row["object_id"]),
        "scene_id": job.scene_id,
        "view_id": job.view_id,
        "visib_fract": float(row["visib_fract"]),
        "px_count_all": int(row["px_count_all"]),
        "px_count_valid": int(row["px_count_valid"]),
        "px_count_visib": int(row["px_count_visib"]),
        "depth_corrupt": row["depth_corrupt"],
        "width": job.width,
        "height": job.height,
        "view_x": float(direction[0]),
        "view_y": float(direction[1]),
        "view_z": float(direction[2]),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "surface_point_count": settings.surface_point_count,
        "observed_point_count": int(observed_count),
        "observed_fraction": float(observed_count) / settings.surface_point_count,
    }
    for name in (
        "bbox_obj_x", "bbox_obj_y", "bbox_obj_w", "bbox_obj_h",
        "bbox_visib_x", "bbox_visib_y", "bbox_visib_w", "bbox_visib_h",
    ):
        output[name] = float(row[name])
    output.update(crop)
    return output


def visible_surface_mask_cpu(
    points_O: np.ndarray,
    T_C_O: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    settings: BuildSettings,
) -> np.ndarray:
    points_C = np.asarray(points_O, dtype=np.float32) @ T_C_O[:3, :3].T + T_C_O[:3, 3]
    z = points_C[:, 2]
    valid = np.isfinite(z) & (z > 1e-8)
    projected = points_C @ K.T
    u = np.zeros(len(points_C), dtype=np.int64)
    v = np.zeros(len(points_C), dtype=np.int64)
    u[valid] = np.rint(projected[valid, 0] / projected[valid, 2]).astype(np.int64)
    v[valid] = np.rint(projected[valid, 1] / projected[valid, 2]).astype(np.int64)
    observed = np.zeros(len(points_C), dtype=bool)
    tolerance = settings.depth_tolerance_m + settings.depth_tolerance_relative * z
    for dy in range(-settings.pixel_radius, settings.pixel_radius + 1):
        for dx in range(-settings.pixel_radius, settings.pixel_radius + 1):
            x, y = u + dx, v + dy
            inside = valid & (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
            indices = np.flatnonzero(inside & ~observed)
            if not len(indices):
                continue
            sampled = depth[y[indices], x[indices]]
            accepted = (
                mask[y[indices], x[indices]].astype(bool)
                & np.isfinite(sampled)
                & (sampled > 0)
                & (np.abs(sampled - z[indices]) <= tolerance[indices])
            )
            observed[indices[accepted]] = True
    return observed


class TorchCoverageBackend:
    """Batch surface projection and depth/mask tests on one Torch device."""

    def __init__(
        self,
        atlas: np.ndarray,
        object_to_atlas: Mapping[int, int],
        settings: BuildSettings,
        device: str,
    ):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch backend requested but torch is unavailable") from exc
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but CUDA is unavailable")
        # ``memmap(mode='r')`` is intentionally non-writable.  Make the one
        # host copy explicit before the single device upload to avoid PyTorch's
        # undefined-behaviour warning for read-only NumPy buffers.
        self.atlas = torch.from_numpy(np.array(atlas, copy=True)).to(self.device)
        self.object_to_atlas = dict(object_to_atlas)
        self.settings = settings

    def process(self, frames: Sequence[DecodedFrame]) -> list[FrameResult]:
        torch = self.torch
        started = time.monotonic()
        if not frames:
            return []
        shapes = {(item.job.height, item.job.width) for item in frames}
        if len(shapes) != 1:
            output = []
            for frame in frames:
                output.extend(self.process([frame]))
            return output
        depth = torch.as_tensor(
            np.stack([item.depth for item in frames]), device=self.device
        )
        masks_np = np.concatenate([item.masks for item in frames], axis=0)
        masks = torch.as_tensor(masks_np, device=self.device, dtype=torch.bool)
        frame_indices = []
        rows: list[Mapping[str, Any]] = []
        jobs: list[FrameJob] = []
        for frame_index, frame in enumerate(frames):
            for row in frame.job.instances:
                frame_indices.append(frame_index)
                rows.append(row)
                jobs.append(frame.job)
        object_indices = torch.as_tensor(
            [self.object_to_atlas[int(row["object_id"])] for row in rows],
            device=self.device,
            dtype=torch.long,
        )
        frame_index_tensor = torch.as_tensor(
            frame_indices, device=self.device, dtype=torch.long
        )
        points = self.atlas[object_indices]
        poses = torch.as_tensor(
            np.stack([unpack_matrix(row["T_C_O_f32"], 4, 4) for row in rows]),
            device=self.device,
        )
        intrinsics = torch.as_tensor(
            np.stack([unpack_matrix(job.K_f32, 3, 3) for job in jobs]),
            device=self.device,
        )
        points_C = torch.bmm(points, poses[:, :3, :3].transpose(1, 2)) + poses[:, None, :3, 3]
        z = points_C[:, :, 2]
        valid = torch.isfinite(z) & (z > 1e-8)
        projected = torch.bmm(points_C, intrinsics.transpose(1, 2))
        u = torch.round(projected[:, :, 0] / projected[:, :, 2].clamp_min(1e-8)).to(torch.long)
        v = torch.round(projected[:, :, 1] / projected[:, :, 2].clamp_min(1e-8)).to(torch.long)
        observed = torch.zeros_like(valid)
        tolerance = self.settings.depth_tolerance_m + self.settings.depth_tolerance_relative * z
        instance_index = torch.arange(len(rows), device=self.device)[:, None]
        height, width = next(iter(shapes))
        for dy in range(-self.settings.pixel_radius, self.settings.pixel_radius + 1):
            for dx in range(-self.settings.pixel_radius, self.settings.pixel_radius + 1):
                x, y = u + dx, v + dy
                inside = valid & (x >= 0) & (x < width) & (y >= 0) & (y < height)
                safe_x = x.clamp(0, width - 1)
                safe_y = y.clamp(0, height - 1)
                sampled = depth[frame_index_tensor[:, None], safe_y, safe_x]
                visible = masks[instance_index, safe_y, safe_x]
                observed |= (
                    inside
                    & visible
                    & torch.isfinite(sampled)
                    & (sampled > 0)
                    & ((sampled - z).abs() <= tolerance)
                )
        observed_np = observed.cpu().numpy()
        del depth, masks, points, poses, intrinsics, points_C, projected, observed
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - started
        results = []
        offset = 0
        for frame in frames:
            count = len(frame.job.instances)
            frame_masks = observed_np[offset : offset + count]
            offset += count
            features = tuple(
                feature_metadata(
                    frame.job, row, int(mask.sum()), self.settings
                )
                for row, mask in zip(frame.job.instances, frame_masks)
            )
            bits = tuple(
                np.packbits(mask, bitorder="little").tobytes()
                for mask in frame_masks
            )
            results.append(
                FrameResult(
                    frame_id=frame.job.frame_id,
                    elapsed_seconds=elapsed / len(frames),
                    features=features,
                    observed_bits=bits,
                )
            )
        return results


_CPU_STATE: dict[str, Any] = {}


def _init_cpu_worker(
    data_root: str,
    atlas_path: str,
    atlas_shape: tuple[int, int, int],
    object_to_atlas: Mapping[int, int],
    settings: BuildSettings,
) -> None:
    _CPU_STATE.update(
        {
            "reader": _ThreadShardCache(Path(data_root)),
            "atlas": np.memmap(atlas_path, mode="r", dtype="<f4", shape=atlas_shape),
            "object_to_atlas": dict(object_to_atlas),
            "settings": settings,
        }
    )


def _process_frame_cpu(job: FrameJob) -> FrameResult:
    started = time.monotonic()
    try:
        decoded = decode_frame(job, _CPU_STATE["reader"])
        settings: BuildSettings = _CPU_STATE["settings"]
        atlas = _CPU_STATE["atlas"]
        mapping = _CPU_STATE["object_to_atlas"]
        features, bits = [], []
        K = unpack_matrix(job.K_f32, 3, 3)
        for row, mask in zip(job.instances, decoded.masks):
            points = atlas[mapping[int(row["object_id"])]]
            pose = unpack_matrix(row["T_C_O_f32"], 4, 4)
            observed = visible_surface_mask_cpu(
                points, pose, K, decoded.depth, mask, settings
            )
            features.append(feature_metadata(job, row, int(observed.sum()), settings))
            bits.append(np.packbits(observed, bitorder="little").tobytes())
        return FrameResult(
            frame_id=job.frame_id,
            elapsed_seconds=time.monotonic() - started,
            features=tuple(features),
            observed_bits=tuple(bits),
        )
    except Exception as exc:  # returned to the parent for durable error recording
        return FrameResult(
            frame_id=job.frame_id,
            elapsed_seconds=time.monotonic() - started,
            features=(),
            observed_bits=(),
            error=f"{type(exc).__name__}: {exc}",
        )


def _source_connection_with_membership(
    index_path: Path, split_path: Path, split: str
) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{Path(index_path).resolve()}?mode=ro", uri=True, timeout=60.0
    )
    connection.row_factory = sqlite3.Row
    object_ids, scene_ids = load_split_membership(split_path, split)
    if not object_ids:
        object_ids = {
            int(row[0]) for row in connection.execute("SELECT object_id FROM objects")
        }
    if not scene_ids:
        scene_ids = {
            int(row[0]) for row in connection.execute("SELECT scene_id FROM scenes")
        }
    connection.execute("CREATE TEMP TABLE active_objects(object_id INTEGER PRIMARY KEY)")
    connection.execute("CREATE TEMP TABLE active_scenes(scene_id INTEGER PRIMARY KEY)")
    connection.executemany(
        "INSERT INTO active_objects VALUES (?)", ((value,) for value in object_ids)
    )
    connection.executemany(
        "INSERT INTO active_scenes VALUES (?)", ((value,) for value in scene_ids)
    )
    return connection


def iter_frame_jobs(
    index_path: Path,
    split_path: Path,
    settings: BuildSettings,
    *,
    completed_frame_ids: set[int] | None = None,
    maximum_frames: int | None = None,
) -> Iterator[FrameJob]:
    connection = _source_connection_with_membership(index_path, split_path, settings.split)
    completed = completed_frame_ids or set()
    try:
        rows = connection.execute(
            SOURCE_FRAME_QUERY,
            (
                settings.visibility_floor,
                settings.min_visible_pixels,
                settings.depth_corruption_policy,
                settings.depth_corruption_policy,
                settings.depth_corruption_policy,
            ),
        )
        current_frame: int | None = None
        frame_values: dict[str, Any] | None = None
        instances: list[dict[str, Any]] = []
        yielded = 0

        def finish() -> FrameJob | None:
            if frame_values is None or current_frame in completed:
                return None
            return FrameJob(instances=tuple(instances), **frame_values)

        for raw in rows:
            row = dict(raw)
            frame_id = int(row["frame_id"])
            if current_frame is not None and frame_id != current_frame:
                job = finish()
                if job is not None:
                    yield job
                    yielded += 1
                    if maximum_frames is not None and yielded >= maximum_frames:
                        return
                instances = []
                frame_values = None
            if frame_values is None:
                current_frame = frame_id
                frame_values = {
                    name: row[name]
                    for name in (
                        "frame_id", "scene_id", "view_id", "width", "height",
                        "depth_unit_m", "K_f32", "depth_offset", "depth_size",
                        "mask_visib_offset", "mask_visib_size", "relative_path",
                    )
                }
            instances.append(
                {
                    name: row[name]
                    for name in (
                        "gt_id", "object_id", "T_C_O_f32", "visib_fract",
                        "px_count_all", "px_count_valid", "px_count_visib",
                        "depth_corrupt", "bbox_obj_x", "bbox_obj_y", "bbox_obj_w",
                        "bbox_obj_h", "bbox_visib_x", "bbox_visib_y",
                        "bbox_visib_w", "bbox_visib_h",
                    )
                }
            )
        job = finish()
        if job is not None and (maximum_frames is None or yielded < maximum_frames):
            yield job
    finally:
        connection.close()


def count_source_frames(
    index_path: Path, split_path: Path, settings: BuildSettings
) -> int:
    connection = _source_connection_with_membership(index_path, split_path, settings.split)
    try:
        row = connection.execute(
            """
            SELECT COUNT(DISTINCT i.frame_id)
            FROM instances AS i
            JOIN active_objects AS ao ON ao.object_id=i.object_id
            JOIN active_scenes AS ac ON ac.scene_id=i.scene_id
            WHERE i.visib_fract >= ? AND i.px_count_visib >= ?
              AND (
                ? = 'all'
                OR (? = 'clean_only' AND i.depth_corrupt = 0)
                OR (? = 'exclude_known_bad' AND i.depth_corrupt IS NOT 1)
              )
            """,
            (
                settings.visibility_floor,
                settings.min_visible_pixels,
                settings.depth_corruption_policy,
                settings.depth_corruption_policy,
                settings.depth_corruption_policy,
            ),
        ).fetchone()
        return int(row[0])
    finally:
        connection.close()


def _fingerprint(
    index_path: Path,
    split_path: Path,
    model_catalog_path: Path,
    settings: BuildSettings,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "algorithm": ALGORITHM,
        "source_index": str(Path(index_path).resolve()),
        "source_index_size": Path(index_path).stat().st_size,
        "source_index_mtime_ns": Path(index_path).stat().st_mtime_ns,
        "split_manifest": str(Path(split_path).resolve()),
        "split_manifest_mtime_ns": Path(split_path).stat().st_mtime_ns,
        "model_catalog": str(Path(model_catalog_path).resolve()),
        "model_catalog_mtime_ns": Path(model_catalog_path).stat().st_mtime_ns,
        "settings": settings.as_dict(),
    }


def initialize_output(
    output_path: Path,
    bits_path: Path,
    fingerprint: Mapping[str, Any],
    settings: BuildSettings,
) -> tuple[sqlite3.Connection, int, set[int]]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output_path, timeout=60.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(SCHEMA)
    encoded = json.dumps(fingerprint, sort_keys=True)
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key='fingerprint'"
    ).fetchone()
    if existing is not None and str(existing[0]) != encoded:
        raise ValueError(
            f"Geometry index parameters differ; choose a new output path: {output_path}"
        )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('fingerprint', ?)", (encoded,)
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('index_complete', '0')"
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('bits_path', ?)",
        (_portable_metadata_path(output_path, bits_path),),
    )
    # These indexes are useful for reports but expensive to maintain across
    # roughly ten million append-only feature inserts.  A completed index is
    # immutable, so build them once during finalization instead.
    _drop_secondary_indexes(connection)
    connection.commit()
    next_id = int(
        connection.execute("SELECT COALESCE(MAX(feature_id), -1) + 1 FROM frame_features").fetchone()[0]
    )
    byte_count = (settings.surface_point_count + 7) // 8
    expected_size = next_id * byte_count
    if Path(bits_path).exists():
        actual_size = Path(bits_path).stat().st_size
        if actual_size < expected_size:
            raise ValueError("Surface bitset sidecar is shorter than SQLite metadata")
        if actual_size > expected_size:
            with Path(bits_path).open("r+b") as stream:
                stream.truncate(expected_size)
    elif expected_size:
        raise ValueError("SQLite has features but surface bitset sidecar is missing")
    completed = {
        int(row[0])
        for row in connection.execute(
            "SELECT frame_id FROM frame_status WHERE status='complete'"
        )
    }
    return connection, next_id, completed


FEATURE_COLUMNS = (
    "feature_id", "frame_id", "gt_id", "object_id", "scene_id", "view_id",
    "visib_fract", "px_count_all", "px_count_valid", "px_count_visib",
    "depth_corrupt", "width", "height", "view_x", "view_y", "view_z",
    "fx", "fy", "cx", "cy", "bbox_obj_x", "bbox_obj_y", "bbox_obj_w",
    "bbox_obj_h", "bbox_visib_x", "bbox_visib_y", "bbox_visib_w",
    "bbox_visib_h", "crop_width_min", "crop_width_max", "crop_feasible",
    "bbox_obj_touches_border", "base_norm_fx", "base_norm_fy",
    "max_norm_fx", "max_norm_fy", "surface_point_count",
    "observed_point_count", "observed_fraction", "bit_offset",
)


class GeometryWriter:
    def __init__(
        self,
        connection: sqlite3.Connection,
        bits_path: Path,
        next_feature_id: int,
        settings: BuildSettings,
        *,
        commit_frames: int = 128,
    ):
        self.connection = connection
        self.bits = Path(bits_path).open("ab", buffering=8 * 1024 * 1024)
        self.next_feature_id = int(next_feature_id)
        self.byte_count = (settings.surface_point_count + 7) // 8
        self.commit_frames = max(1, int(commit_frames))
        self.pending_frames = 0
        self.feature_insert = (
            f"INSERT INTO frame_features ({','.join(FEATURE_COLUMNS)}) "
            f"VALUES ({','.join('?' for _ in FEATURE_COLUMNS)})"
        )

    def write(self, result: FrameResult) -> None:
        if result.error is not None:
            self.connection.execute(
                "INSERT OR REPLACE INTO frame_status VALUES (?,?,?,?,?)",
                (result.frame_id, "error", 0, result.elapsed_seconds, result.error),
            )
        else:
            rows = []
            for feature, bits in zip(result.features, result.observed_bits):
                if len(bits) != self.byte_count:
                    raise ValueError("Unexpected packed surface-mask size")
                feature_id = self.next_feature_id
                bit_offset = feature_id * self.byte_count
                self.next_feature_id += 1
                self.bits.write(bits)
                value = dict(feature)
                value.update(feature_id=feature_id, bit_offset=bit_offset)
                rows.append(tuple(value[name] for name in FEATURE_COLUMNS))
            self.connection.executemany(self.feature_insert, rows)
            self.connection.execute(
                "INSERT OR REPLACE INTO frame_status VALUES (?,?,?,?,NULL)",
                (result.frame_id, "complete", len(rows), result.elapsed_seconds),
            )
        self.pending_frames += 1
        if self.pending_frames >= self.commit_frames:
            self.flush()

    def flush(self) -> None:
        self.bits.flush()
        self.connection.commit()
        self.pending_frames = 0

    def close(self) -> None:
        self.flush()
        self.bits.close()


def _batched(values: Iterable[Any], count: int) -> Iterator[list[Any]]:
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) >= count:
            yield batch
            batch = []
    if batch:
        yield batch


def build_geometry_index(
    *,
    data_root: Path,
    index_path: Path,
    split_path: Path,
    model_catalog_path: Path,
    output_path: Path,
    bits_path: Path,
    atlas_path: Path,
    settings: BuildSettings,
    device: str = "cuda",
    workers: int = 8,
    batch_frames: int = 8,
    maximum_frames: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Build or resume the derived index and return measured throughput."""

    settings.validate()
    atlas, object_to_atlas = prepare_surface_atlas(
        model_catalog_path, atlas_path, settings, progress=progress
    )
    fingerprint = _fingerprint(index_path, split_path, model_catalog_path, settings)
    connection, next_id, completed = initialize_output(
        output_path, bits_path, fingerprint, settings
    )
    writer = GeometryWriter(connection, bits_path, next_id, settings)
    total = count_source_frames(index_path, split_path, settings) - len(completed)
    if maximum_frames is not None:
        total = min(total, int(maximum_frames))
    jobs = iter_frame_jobs(
        index_path,
        split_path,
        settings,
        completed_frame_ids=completed,
        maximum_frames=maximum_frames,
    )
    progress_bar = None
    if progress:
        from tqdm.auto import tqdm

        progress_bar = tqdm(
            total=total,
            desc="geometry index",
            unit="frame",
            smoothing=0.05,
            mininterval=5.0,
            maxinterval=10.0,
        )
    started = time.monotonic()
    processed = 0
    errors = 0
    device_metrics: dict[str, Any] = {}
    try:
        if device == "cpu":
            worker_count = max(1, int(workers))
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_cpu_worker,
                initargs=(
                    str(Path(data_root).resolve()),
                    str(Path(atlas_path).resolve()),
                    tuple(atlas.shape),
                    object_to_atlas,
                    settings,
                ),
            ) as executor:
                pending = set()
                iterator = iter(jobs)
                exhausted = False
                while pending or not exhausted:
                    while not exhausted and len(pending) < worker_count * 3:
                        try:
                            pending.add(executor.submit(_process_frame_cpu, next(iterator)))
                        except StopIteration:
                            exhausted = True
                    if not pending:
                        continue
                    ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        result = future.result()
                        writer.write(result)
                        processed += 1
                        errors += int(result.error is not None)
                        if progress_bar is not None:
                            progress_bar.update(1)
        else:
            backend = TorchCoverageBackend(atlas, object_to_atlas, settings, device)
            reader = _ThreadShardCache(Path(data_root))
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
                for batch in _batched(jobs, max(1, int(batch_frames))):
                    decoded = list(executor.map(lambda job: decode_frame(job, reader), batch))
                    for result in backend.process(decoded):
                        writer.write(result)
                        processed += 1
                        errors += int(result.error is not None)
                        if progress_bar is not None:
                            progress_bar.update(1)
            if backend.device.type == "cuda":
                device_metrics = {
                    "cuda_device": str(backend.device),
                    "cuda_peak_allocated_bytes": int(
                        backend.torch.cuda.max_memory_allocated(backend.device)
                    ),
                    "cuda_peak_reserved_bytes": int(
                        backend.torch.cuda.max_memory_reserved(backend.device)
                    ),
                }
        writer.flush()
        remaining = count_source_frames(index_path, split_path, settings) - len(completed) - processed
        if maximum_frames is None and remaining <= 0 and errors == 0:
            _create_secondary_indexes(connection)
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('index_complete', '1')"
            )
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('updated_at_unix', ?)",
            (str(time.time()),),
        )
        connection.commit()
    finally:
        writer.close()
        if progress_bar is not None:
            progress_bar.close()
        connection.close()
    elapsed = time.monotonic() - started
    total_source_frames = count_source_frames(index_path, split_path, settings)
    return {
        "processed_frames": processed,
        "error_frames": errors,
        "elapsed_seconds": elapsed,
        "frames_per_second": processed / elapsed if elapsed else math.inf,
        "total_source_frames": total_source_frames,
        "remaining_source_frames": max(0, total_source_frames - len(completed) - processed),
        "estimated_total_seconds": (
            total_source_frames / (processed / elapsed)
            if processed and elapsed
            else math.inf
        ),
        **device_metrics,
    }


def _render_fingerprint(
    reference_index_path: Path,
    bank_root: Path,
    model_catalog_path: Path,
    settings: BuildSettings,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "algorithm": ALGORITHM,
        "source_kind": "render",
        "source_index": str(Path(reference_index_path).resolve()),
        "source_index_size": Path(reference_index_path).stat().st_size,
        "source_index_mtime_ns": Path(reference_index_path).stat().st_mtime_ns,
        "bank_root": str(Path(bank_root).resolve()),
        "model_catalog": str(Path(model_catalog_path).resolve()),
        "model_catalog_mtime_ns": Path(model_catalog_path).stat().st_mtime_ns,
        "settings": settings.as_dict(),
    }


def iter_render_jobs(
    reference_index_path: Path,
    bank_root: Path,
    settings: BuildSettings,
    *,
    completed_frame_ids: set[int] | None = None,
    maximum_frames: int | None = None,
) -> Iterator[FrameJob]:
    """Yield one stable geometry job per indexed BOP render view.

    Reads packed tar-shard payloads (``pi3_object_reference_index_v2``): each
    ``FrameJob`` carries the shard-relative path plus byte offset/size for its
    depth and mask_visib payloads, decoded the same way as scene frames.
    """

    completed = completed_frame_ids or set()
    connection = readonly_connection(reference_index_path)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("format") != "pi3_object_reference_index_v2":
            raise ValueError(
                f"Unsupported reference index {reference_index_path}: expected "
                "pi3_object_reference_index_v2. Rebuild it with "
                "datasets/preprocess/render/object_reference_index.py"
            )
        width = int(metadata["resolution_width"])
        height = int(metadata["resolution_height"])
        yielded = 0
        rows = connection.execute(
            """
            SELECT v.object_id, v.view_id, v.T_C_O_f32, v.K_f32, v.depth_scale,
                   v.depth_offset, v.depth_size, v.mask_visib_offset,
                   v.mask_visib_size, v.bbox_obj_json, v.bbox_visib_json,
                   v.px_count_all, v.px_count_visib, s.relative_path
            FROM views AS v
            JOIN shards AS s ON s.id = v.shard_id
            ORDER BY v.object_id, v.view_id
            """
        )
        for row in rows:
            object_id, view_id = int(row["object_id"]), int(row["view_id"])
            frame_id = object_id * 1_000_000 + view_id
            if frame_id in completed:
                continue
            px_all, px_visib = int(row["px_count_all"]), int(row["px_count_visib"])
            if px_visib < settings.min_visible_pixels:
                continue
            visibility = float(px_visib) / max(1, px_all)
            if visibility < settings.visibility_floor:
                continue
            bbox_obj = json.loads(row["bbox_obj_json"])
            bbox_visib = json.loads(row["bbox_visib_json"])
            instance = {
                "gt_id": 0,
                "object_id": object_id,
                "T_C_O_f32": row["T_C_O_f32"],
                "visib_fract": visibility,
                "px_count_all": px_all,
                "px_count_valid": px_all,
                "px_count_visib": px_visib,
                "depth_corrupt": 0,
                "bbox_obj_x": float(bbox_obj[0]),
                "bbox_obj_y": float(bbox_obj[1]),
                "bbox_obj_w": float(bbox_obj[2]),
                "bbox_obj_h": float(bbox_obj[3]),
                "bbox_visib_x": float(bbox_visib[0]),
                "bbox_visib_y": float(bbox_visib[1]),
                "bbox_visib_w": float(bbox_visib[2]),
                "bbox_visib_h": float(bbox_visib[3]),
            }
            yield FrameJob(
                frame_id=frame_id,
                scene_id=-1,
                view_id=view_id,
                width=width,
                height=height,
                depth_unit_m=float(row["depth_scale"]) * 0.001,
                K_f32=row["K_f32"],
                depth_offset=int(row["depth_offset"]),
                depth_size=int(row["depth_size"]),
                mask_visib_offset=int(row["mask_visib_offset"]),
                mask_visib_size=int(row["mask_visib_size"]),
                relative_path=str(row["relative_path"]),
                instances=(instance,),
            )
            yielded += 1
            if maximum_frames is not None and yielded >= int(maximum_frames):
                return
    finally:
        connection.close()


def decode_render_frame(job: FrameJob, reader: _ThreadShardCache) -> DecodedFrame:
    """Decode a render-bank frame from its packed tar-shard payloads.

    Unlike ``decode_frame`` (scene frames, mask stored as one COCO-RLE JSON
    blob per frame with one entry per ``gt_id``), the render bank stores one
    plain mask PNG per view -- there is exactly one object per reference view.
    """

    depth_payload = reader.read(job.relative_path, job.depth_offset, job.depth_size)
    with Image.open(io.BytesIO(depth_payload)) as image:
        depth = np.asarray(image).astype(np.float32) * float(job.depth_unit_m)
    mask_payload = reader.read(
        job.relative_path, job.mask_visib_offset, job.mask_visib_size
    )
    with Image.open(io.BytesIO(mask_payload)) as image:
        mask = np.asarray(image) > 0
    expected = (job.height, job.width)
    if depth.shape != expected or mask.shape != expected:
        raise ValueError(
            f"Render payload shape mismatch: depth={depth.shape}, mask={mask.shape}, "
            f"expected={expected}"
        )
    return DecodedFrame(
        job=job,
        depth=depth,
        masks=mask[None].astype(np.uint8, copy=False),
    )


def build_render_geometry_index(
    *,
    bank_root: Path,
    reference_index_path: Path,
    model_catalog_path: Path,
    output_path: Path,
    bits_path: Path,
    atlas_path: Path,
    settings: BuildSettings,
    device: str = "cuda",
    workers: int = 8,
    batch_frames: int = 16,
    maximum_frames: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Build exact surface/crop geometry for the indexed render bank."""

    settings.validate()
    atlas, object_to_atlas = prepare_surface_atlas(
        model_catalog_path, atlas_path, settings, progress=progress
    )
    fingerprint = _render_fingerprint(
        reference_index_path, bank_root, model_catalog_path, settings
    )
    connection, next_id, completed = initialize_output(
        output_path, bits_path, fingerprint, settings
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('source_kind','render')"
    )
    writer = GeometryWriter(connection, bits_path, next_id, settings)
    source = readonly_connection(reference_index_path)
    try:
        total_source = int(source.execute("SELECT COUNT(*) FROM views").fetchone()[0])
    finally:
        source.close()
    remaining = max(0, total_source - len(completed))
    total = min(remaining, int(maximum_frames)) if maximum_frames is not None else remaining
    jobs = iter_render_jobs(
        reference_index_path,
        bank_root,
        settings,
        completed_frame_ids=completed,
        maximum_frames=maximum_frames,
    )
    progress_bar = None
    if progress:
        from tqdm.auto import tqdm

        progress_bar = tqdm(total=total, desc="render geometry", unit="view", mininterval=5.0)
    started = time.monotonic()
    processed = 0
    errors = 0
    device_metrics: dict[str, Any] = {}
    try:
        backend = TorchCoverageBackend(atlas, object_to_atlas, settings, device)
        reader = _ThreadShardCache(Path(bank_root))
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            for batch in _batched(jobs, max(1, int(batch_frames))):
                decoded = list(
                    executor.map(lambda job: decode_render_frame(job, reader), batch)
                )
                for result in backend.process(decoded):
                    writer.write(result)
                    processed += 1
                    errors += int(result.error is not None)
                    if progress_bar is not None:
                        progress_bar.update(1)
        writer.flush()
        remaining_after = total_source - len(completed) - processed
        if maximum_frames is None and remaining_after <= 0 and errors == 0:
            _create_secondary_indexes(connection)
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('index_complete','1')"
            )
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('updated_at_unix',?)",
            (str(time.time()),),
        )
        connection.commit()
        if backend.device.type == "cuda":
            device_metrics = {
                "cuda_device": str(backend.device),
                "cuda_peak_allocated_bytes": int(
                    backend.torch.cuda.max_memory_allocated(backend.device)
                ),
                "cuda_peak_reserved_bytes": int(
                    backend.torch.cuda.max_memory_reserved(backend.device)
                ),
            }
    finally:
        writer.close()
        if progress_bar is not None:
            progress_bar.close()
        connection.close()
    elapsed = time.monotonic() - started
    return {
        "processed_frames": processed,
        "error_frames": errors,
        "elapsed_seconds": elapsed,
        "frames_per_second": processed / elapsed if elapsed else math.inf,
        "total_source_frames": total_source,
        "remaining_source_frames": max(0, total_source - len(completed) - processed),
        "estimated_total_seconds": (
            total_source / (processed / elapsed) if processed and elapsed else math.inf
        ),
        **device_metrics,
    }


def packed_union_curve(
    masks: np.ndarray,
    maximum_views: int,
) -> tuple[list[int], list[float]]:
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim != 2:
        raise ValueError("Packed masks must have shape (views, bytes)")
    selected: list[int] = []
    coverage: list[float] = []
    covered = np.zeros(masks.shape[1], dtype=np.uint8)
    bit_count = masks.shape[1] * 8
    for _ in range(min(int(maximum_views), len(masks))):
        best_index, best_gain = None, -1
        inverse = np.bitwise_not(covered)
        for index in range(len(masks)):
            if index in selected:
                continue
            gain = int(POPCOUNT[np.bitwise_and(masks[index], inverse)].sum())
            if gain > best_gain:
                best_index, best_gain = index, gain
        if best_index is None:
            break
        selected.append(best_index)
        covered |= masks[best_index]
        coverage.append(float(POPCOUNT[covered].sum()) / bit_count)
    return selected, coverage


def focal_band_feasible(
    references: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
    *,
    relative_tolerance: float,
    common_target: bool,
    focal_shape_log_tolerance: float = 0.03,
) -> bool:
    """Check crop-only focal feasibility, optionally with one common target."""

    qx, qy = float(query["base_norm_fx"]), float(query["base_norm_fy"])
    qf = math.sqrt(qx * qy)
    qshape = math.log(qx / qy)
    low = qf * (1.0 - float(relative_tolerance))
    high = qf * (1.0 + float(relative_tolerance))
    intersections = [(low, high)]
    for row in references:
        if not int(row["crop_feasible"]):
            return False
        rx, ry = float(row["base_norm_fx"]), float(row["base_norm_fy"])
        mx, my = float(row["max_norm_fx"]), float(row["max_norm_fy"])
        if abs(math.log(rx / ry) - qshape) > focal_shape_log_tolerance:
            return False
        interval = (math.sqrt(rx * ry), math.sqrt(mx * my))
        if max(interval[0], low) > min(interval[1], high):
            return False
        intersections.append(interval)
    if common_target:
        return max(value[0] for value in intersections) <= min(
            value[1] for value in intersections
        )
    return True


def _track_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["scene_id"]), int(row["gt_id"])


def _rows_by_track(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    output: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        output.setdefault(_track_key(row), []).append(row)
    return output


def _load_object_rows(
    connection: sqlite3.Connection, object_id: int
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM frame_features WHERE object_id=? ORDER BY scene_id,gt_id,view_id",
            (int(object_id),),
        )
    ]


def collect_capacity_statistics(
    index_path: Path,
    *,
    n_values: Sequence[int] = (2, 5, 8),
    k_values: Sequence[int] = (1, 5, 10),
    query_visibility_min: float = 0.1,
    reference_visibility_min: float = 0.3,
    per_reference_surface_min: float = 0.0,
    union_surface_min: float = 0.5,
    positive_angle_degrees: float = 10.0,
    focal_relative_tolerance: float = 0.1,
    common_focal_target: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Count guaranteed constructible scene-pair units for N/K grids.

    References are one deterministic greedy maximum-coverage subset from a
    single physical track.  A counted pair is therefore an actual valid plan;
    alternative query-seeded subsets can only increase the capacity.
    """

    connection = sqlite3.connect(Path(index_path))
    connection.row_factory = sqlite3.Row
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    fingerprint = json.loads(metadata["fingerprint"])
    point_count = int(fingerprint["settings"]["surface_point_count"])
    byte_count = (point_count + 7) // 8
    bits_path = _resolve_metadata_path(index_path, metadata["bits_path"])
    bit_count = bits_path.stat().st_size // byte_count
    bitsets = np.memmap(bits_path, mode="r", dtype=np.uint8, shape=(bit_count, byte_count))
    object_ids = [
        int(row[0])
        for row in connection.execute(
            "SELECT DISTINCT object_id FROM frame_features ORDER BY object_id"
        )
    ]
    iterator: Iterable[int] = object_ids
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(object_ids, desc="capacity", unit="object")
    counts = {
        (int(n), int(k)): {
            "reference_tracks": 0,
            "query_tracks": 0,
            "ordered_scene_track_pairs": 0,
            "query_frames_supported": 0,
            "objects": set(),
        }
        for n in n_values
        for k in k_values
    }
    cosine = math.cos(math.radians(float(positive_angle_degrees)))
    per_object: dict[int, dict[str, int]] = {}
    try:
        for object_id in iterator:
            rows = _load_object_rows(connection, object_id)
            tracks = _rows_by_track(rows)
            query_tracks = {
                key: [row for row in values if float(row["visib_fract"]) >= query_visibility_min]
                for key, values in tracks.items()
            }
            query_tracks = {key: value for key, value in query_tracks.items() if value}
            query_keys = list(query_tracks)
            query_track_index = {
                key: index for index, key in enumerate(query_keys)
            }
            query_frame_rows = [
                row for key in query_keys for row in query_tracks[key]
            ]
            query_frame_track_indices = np.asarray(
                [
                    query_track_index[key]
                    for key in query_keys
                    for _ in query_tracks[key]
                ],
                dtype=np.int64,
            )
            query_directions = np.asarray(
                [
                    [row["view_x"], row["view_y"], row["view_z"]]
                    for row in query_frame_rows
                ],
                dtype=np.float32,
            )
            query_scene_ids = np.asarray(
                [key[0] for key in query_keys], dtype=np.int64
            )
            query_focal = np.asarray(
                [
                    math.sqrt(
                        float(query_tracks[key][0]["base_norm_fx"])
                        * float(query_tracks[key][0]["base_norm_fy"])
                    )
                    for key in query_keys
                ],
                dtype=np.float64,
            )
            query_focal_shape = np.asarray(
                [
                    math.log(
                        float(query_tracks[key][0]["base_norm_fx"])
                        / float(query_tracks[key][0]["base_norm_fy"])
                    )
                    for key in query_keys
                ],
                dtype=np.float64,
            )
            try:
                from scipy.spatial import cKDTree

                direction_tree = cKDTree(query_directions)
            except ImportError:
                direction_tree = None
            chord_radius = 2.0 * math.sin(
                math.radians(float(positive_angle_degrees)) / 2.0
            )
            object_pair_total = 0
            for n in n_values:
                reference_plans: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
                for key, values in tracks.items():
                    eligible = [
                        row
                        for row in values
                        if float(row["visib_fract"]) >= reference_visibility_min
                        and float(row["observed_fraction"]) >= per_reference_surface_min
                        and int(row["crop_feasible"])
                    ]
                    if len(eligible) < int(n):
                        continue
                    masks = bitsets[[int(row["feature_id"]) for row in eligible]]
                    selected, curve = packed_union_curve(masks, int(n))
                    if len(selected) < int(n) or curve[-1] < union_surface_min:
                        continue
                    reference_plans[key] = [eligible[index] for index in selected]
                for k in k_values:
                    entry = counts[(int(n), int(k))]
                    entry["reference_tracks"] += len(reference_plans)
                    entry["query_tracks"] += sum(len(value) >= int(k) for value in query_tracks.values())
                # One coverage plan is valid for all K values.  Query the
                # spherical index once per plan, then reuse the per-track
                # positive-frame counts across the complete K grid.
                for reference_key, references in reference_plans.items():
                    ref_dirs = np.asarray(
                        [
                            [row["view_x"], row["view_y"], row["view_z"]]
                            for row in references
                        ],
                        dtype=np.float32,
                    )
                    if direction_tree is not None:
                        neighbour_lists = direction_tree.query_ball_point(
                            ref_dirs, chord_radius
                        )
                        nonempty = [
                            np.asarray(value, dtype=np.int64)
                            for value in neighbour_lists
                            if len(value)
                        ]
                        if not nonempty:
                            continue
                        positive_frames = np.unique(np.concatenate(nonempty))
                    else:
                        positive_frames = np.flatnonzero(
                            np.max(query_directions @ ref_dirs.T, axis=1) >= cosine
                        )
                    ref_low = np.asarray(
                        [
                            math.sqrt(
                                float(row["base_norm_fx"])
                                * float(row["base_norm_fy"])
                            )
                            for row in references
                        ]
                    )
                    ref_high = np.asarray(
                        [
                            math.sqrt(
                                float(row["max_norm_fx"])
                                * float(row["max_norm_fy"])
                            )
                            for row in references
                        ]
                    )
                    ref_shape = np.asarray(
                        [
                            math.log(
                                float(row["base_norm_fx"])
                                / float(row["base_norm_fy"])
                            )
                            for row in references
                        ]
                    )
                    query_low = query_focal * (1.0 - focal_relative_tolerance)
                    query_high = query_focal * (1.0 + focal_relative_tolerance)
                    focal_valid = (
                        (float(ref_low.max()) <= query_high)
                        & (float(ref_high.min()) >= query_low)
                        & (
                            query_focal_shape
                            >= float((ref_shape - 0.03).max())
                        )
                        & (
                            query_focal_shape
                            <= float((ref_shape + 0.03).min())
                        )
                    )
                    if common_focal_target:
                        focal_valid &= (
                            np.maximum(query_low, float(ref_low.max()))
                            <= np.minimum(query_high, float(ref_high.min()))
                        )
                    focal_valid &= query_scene_ids != int(reference_key[0])
                    positive_tracks = query_frame_track_indices[positive_frames]
                    keep = focal_valid[positive_tracks]
                    supported_counts = np.bincount(
                        positive_tracks[keep], minlength=len(query_keys)
                    )
                    for k in k_values:
                        accepted = supported_counts >= int(k)
                        pair_count = int(accepted.sum())
                        if not pair_count:
                            continue
                        entry = counts[(int(n), int(k))]
                        entry["ordered_scene_track_pairs"] += pair_count
                        entry["query_frames_supported"] += int(
                            supported_counts[accepted].sum()
                        )
                        entry["objects"].add(object_id)
                        object_pair_total += pair_count
            per_object[object_id] = {
                "tracks": len(tracks),
                "query_frames": sum(len(value) for value in query_tracks.values()),
                "constructible_pair_grid_sum": object_pair_total,
            }
    finally:
        connection.close()
    serializable = {}
    for (n, k), values in counts.items():
        serializable[f"N{n}_K{k}"] = {
            key: (len(value) if key == "objects" else int(value))
            for key, value in values.items()
        }
    return {
        "format": "pi3_megapose_gso_capacity_report_v1",
        "index_path": str(Path(index_path).resolve()),
        "index_complete": metadata.get("index_complete") == "1",
        "constraints": {
            "query_visibility_min": query_visibility_min,
            "reference_visibility_min": reference_visibility_min,
            "per_reference_surface_min": per_reference_surface_min,
            "union_surface_min": union_surface_min,
            "positive_angle_degrees": positive_angle_degrees,
            "focal_relative_tolerance": focal_relative_tolerance,
            "common_focal_target": common_focal_target,
            "planner": "deterministic_greedy_coverage_lower_bound",
        },
        "grid": serializable,
        "per_object": per_object,
    }


PLAN_SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE reference_plans (
    plan_id INTEGER PRIMARY KEY,
    object_id INTEGER NOT NULL,
    scene_id INTEGER NOT NULL,
    gt_id INTEGER NOT NULL,
    reference_count INTEGER NOT NULL,
    union_coverage REAL NOT NULL,
    feature_ids_i64 BLOB NOT NULL,
    frame_ids_i64 BLOB NOT NULL,
    view_ids_i32 BLOB NOT NULL,
    directions_f32 BLOB NOT NULL,
    focal_low REAL NOT NULL,
    focal_high REAL NOT NULL,
    focal_shape_low REAL NOT NULL,
    focal_shape_high REAL NOT NULL
);
CREATE INDEX reference_plans_object
    ON reference_plans(object_id, scene_id, gt_id);
"""


def build_reference_plan_catalog(
    geometry_index_path: Path,
    output_path: Path,
    *,
    reference_count: int = 5,
    reference_visibility_min: float = 0.3,
    per_reference_surface_min: float = 0.0,
    union_surface_min: float = 0.5,
    focal_shape_log_tolerance: float = 0.03,
    reference_source_kind: str = "scene",
    variants_per_track: int = 1,
    progress: bool = True,
) -> dict[str, Any]:
    """Materialize compact, coverage-valid reference plans.

    Query/angle/focal compatibility remains a cheap runtime vector filter.  A
    catalogue therefore stays reusable across query splits and avoids storing
    tens of millions of duplicated query-to-plan edges.
    """

    geometry_index_path = Path(geometry_index_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    reference_count = int(reference_count)
    if reference_count <= 0:
        raise ValueError("reference_count must be positive")
    variants_per_track = int(variants_per_track)
    if variants_per_track <= 0:
        raise ValueError("variants_per_track must be positive")
    if reference_source_kind not in {"scene", "render"}:
        raise ValueError("reference_source_kind must be scene or render")
    source = readonly_connection(geometry_index_path)
    try:
        metadata = dict(source.execute("SELECT key,value FROM metadata"))
        if metadata.get("index_complete") != "1":
            raise ValueError(f"Geometry index is incomplete: {geometry_index_path}")
        fingerprint = json.loads(metadata["fingerprint"])
        point_count = int(fingerprint["settings"]["surface_point_count"])
        byte_count = (point_count + 7) // 8
        bits_path = _resolve_metadata_path(geometry_index_path, metadata["bits_path"])
        bitsets = np.memmap(
            bits_path,
            mode="r",
            dtype=np.uint8,
            shape=(bits_path.stat().st_size // byte_count, byte_count),
        )
        object_ids = [
            int(row[0])
            for row in source.execute(
                "SELECT DISTINCT object_id FROM frame_features ORDER BY object_id"
            )
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
        if temporary.exists():
            temporary.unlink()
        destination = sqlite3.connect(temporary)
        destination.executescript(PLAN_SCHEMA)
        constraints = {
            "reference_count": reference_count,
            "reference_visibility_min": float(reference_visibility_min),
            "per_reference_surface_min": float(per_reference_surface_min),
            "union_surface_min": float(union_surface_min),
            "focal_shape_log_tolerance": float(focal_shape_log_tolerance),
            "variants_per_track": variants_per_track,
            "planner": (
                "deterministic_greedy_maximum_marginal_surface_coverage"
                if variants_per_track == 1
                else "seeded_view_plus_greedy_support_pool"
            ),
        }
        destination.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            (
                ("format", PLAN_FORMAT),
                ("index_complete", "0"),
                (
                    "geometry_index_path",
                    _portable_metadata_path(output_path, geometry_index_path),
                ),
                ("geometry_fingerprint", json.dumps(fingerprint, sort_keys=True)),
                ("reference_source_kind", reference_source_kind),
                ("constraints", json.dumps(constraints, sort_keys=True)),
            ),
        )
        iterator: Iterable[int] = object_ids
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(object_ids, desc="reference plans", unit="object")
        insert = "INSERT INTO reference_plans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        plan_id = 0
        objects_with_plans = 0
        try:
            for object_id in iterator:
                rows = _load_object_rows(source, object_id)
                object_plan_count = 0
                for (scene_id, gt_id), values in _rows_by_track(rows).items():
                    eligible = [
                        row
                        for row in values
                        if float(row["visib_fract"]) >= reference_visibility_min
                        and float(row["observed_fraction"]) >= per_reference_surface_min
                        and int(row["crop_feasible"])
                    ]
                    if len(eligible) < reference_count:
                        continue
                    masks = bitsets[
                        [int(row["feature_id"]) for row in eligible]
                    ]
                    selected, curve = packed_union_curve(masks, reference_count)
                    if (
                        len(selected) < reference_count
                        or float(curve[-1]) < float(union_surface_min)
                    ):
                        continue
                    candidate_sets: list[list[int]] = [selected]
                    if variants_per_track > 1:
                        # Render banks need many possible positive directions.
                        # Re-running a full O(V^2 N) greedy search for every
                        # seed is wasteful: a slightly larger global support
                        # pool already contains complementary views.  Anchor
                        # each variant on a different physical view and fill
                        # the remaining N-1 slots from that pool, retaining
                        # only variants that pass the exact union test below.
                        support, _ = packed_union_curve(
                            masks,
                            min(len(eligible), max(reference_count * 2, reference_count + 12)),
                        )
                        seed_count = min(variants_per_track, len(eligible))
                        seeds = np.linspace(
                            0, len(eligible) - 1, seed_count, dtype=np.int64
                        ).tolist()
                        candidate_sets = []
                        seen: set[tuple[int, ...]] = set()
                        for seed in seeds:
                            variant = [int(seed)] + [
                                index for index in support if index != int(seed)
                            ][: reference_count - 1]
                            if len(variant) != reference_count:
                                continue
                            identity = tuple(sorted(variant))
                            if identity in seen:
                                continue
                            seen.add(identity)
                            candidate_sets.append(variant)

                    for variant in candidate_sets:
                        union_bits = np.bitwise_or.reduce(masks[variant], axis=0)
                        union_coverage = float(POPCOUNT[union_bits].sum()) / float(
                            point_count
                        )
                        if union_coverage < float(union_surface_min):
                            continue
                        references = [eligible[index] for index in variant]
                        focal_low = max(
                            math.sqrt(float(row["base_norm_fx"]) * float(row["base_norm_fy"]))
                            for row in references
                        )
                        focal_high = min(
                            math.sqrt(float(row["max_norm_fx"]) * float(row["max_norm_fy"]))
                            for row in references
                        )
                        shapes = [
                            math.log(float(row["base_norm_fx"]) / float(row["base_norm_fy"]))
                            for row in references
                        ]
                        shape_low = max(value - focal_shape_log_tolerance for value in shapes)
                        shape_high = min(value + focal_shape_log_tolerance for value in shapes)
                        if focal_low > focal_high or shape_low > shape_high:
                            continue
                        feature_ids = np.asarray(
                            [row["feature_id"] for row in references], dtype="<i8"
                        )
                        frame_ids = np.asarray(
                            [row["frame_id"] for row in references], dtype="<i8"
                        )
                        view_ids = np.asarray(
                            [row["view_id"] for row in references], dtype="<i4"
                        )
                        directions = np.asarray(
                            [
                                [row["view_x"], row["view_y"], row["view_z"]]
                                for row in references
                            ],
                            dtype="<f4",
                        )
                        destination.execute(
                            insert,
                            (
                                plan_id,
                                object_id,
                                scene_id,
                                gt_id,
                                reference_count,
                                union_coverage,
                                feature_ids.tobytes(),
                                frame_ids.tobytes(),
                                view_ids.tobytes(),
                                directions.tobytes(),
                                focal_low,
                                focal_high,
                                shape_low,
                                shape_high,
                            ),
                        )
                        plan_id += 1
                        object_plan_count += 1
                if object_plan_count:
                    objects_with_plans += 1
                destination.commit()
            destination.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('index_complete','1')"
            )
            destination.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('plan_count',?)",
                (str(plan_id),),
            )
            destination.commit()
        finally:
            destination.close()
        os.replace(temporary, output_path)
        return {
            "format": PLAN_FORMAT,
            "geometry_index_path": str(geometry_index_path),
            "output_path": str(output_path),
            "reference_count": reference_count,
            "plans": plan_id,
            "objects": objects_with_plans,
            "constraints": constraints,
        }
    finally:
        source.close()


def index_summary(index_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(Path(index_path))
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        status = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT status,COUNT(*) FROM frame_status GROUP BY status"
            )
        }
        row = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT object_id), COUNT(DISTINCT scene_id),
                   COUNT(DISTINCT printf('%d:%d',scene_id,gt_id)),
                   AVG(observed_fraction), AVG(visib_fract)
            FROM frame_features
            """
        ).fetchone()
        thresholds = (0.3, 0.4, 0.5, 0.6)
        surface_counts = {
            str(value): int(
                connection.execute(
                    "SELECT COUNT(*) FROM frame_features WHERE observed_fraction >= ?",
                    (value,),
                ).fetchone()[0]
            )
            for value in thresholds
        }
        visibility_counts = {
            str(value): int(
                connection.execute(
                    "SELECT COUNT(*) FROM frame_features WHERE visib_fract >= ?",
                    (value,),
                ).fetchone()[0]
            )
            for value in thresholds
        }
        return {
            "format": FORMAT,
            "index_complete": metadata.get("index_complete") == "1",
            "frame_status": status,
            "features": int(row[0]),
            "objects": int(row[1]),
            "scenes": int(row[2]),
            "tracks": int(row[3]),
            "mean_surface_coverage": float(row[4] or 0),
            "mean_visibility": float(row[5] or 0),
            "surface_coverage_at_least": surface_counts,
            "visibility_at_least": visibility_counts,
            "crop_feasible_features": int(
                connection.execute(
                    "SELECT COUNT(*) FROM frame_features WHERE crop_feasible=1"
                ).fetchone()[0]
            ),
            "source_border_truncated_features": int(
                connection.execute(
                    "SELECT COUNT(*) FROM frame_features WHERE bbox_obj_touches_border=1"
                ).fetchone()[0]
            ),
            "bits_path": metadata.get("bits_path"),
            "fingerprint": json.loads(metadata["fingerprint"]),
        }
    finally:
        connection.close()
