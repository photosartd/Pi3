"""Derived geometry sidecars for BOP-format LM-O reference/query splits.

This module adapts the reusable MegaPose geometry representation to ordinary
BOP directories without changing :mod:`datasets.lmgeo_dataset`'s historical
indexing or sampling behaviour.  The output schema is intentionally identical
to the GSO geometry sidecar, so crop construction and reference-plan readers
remain dataset independent.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image
from plyfile import PlyData

from datasets.preprocess.megapose_gso_geometry import (
    BuildSettings,
    DecodedFrame,
    FrameJob,
    FrameResult,
    GeometryWriter,
    TorchCoverageBackend,
    _create_secondary_indexes,
    feature_metadata,
    initialize_output,
    sample_mesh_surface,
)


FORMAT = "pi3_lmgeo_bop_geometry_v1"
ALGORITHM = "area_surface_samples_depth_visible_mask_v1"


@dataclass(frozen=True)
class BOPDecodeJob:
    job: FrameJob
    depth_path: str
    mask_paths: tuple[str, ...]


def bop_frame_id(scene_id: int, view_id: int) -> int:
    """Stable frame identity shared with optional LMGeo record metadata."""

    scene_id, view_id = int(scene_id), int(view_id)
    if scene_id < 0 or not 0 <= view_id < 1_000_000:
        raise ValueError(f"Unsupported BOP frame identity: {(scene_id, view_id)}")
    return scene_id * 1_000_000 + view_id


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _source_signature(data_root: Path, split: str, models_folder: str) -> str:
    files = [data_root / models_folder / "models_info.json"]
    for scene_dir in sorted((data_root / split).glob("[0-9]*")):
        files.extend(
            scene_dir / name
            for name in ("scene_camera.json", "scene_gt.json", "scene_gt_info.json")
        )
    rows = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        rows.append((str(path.relative_to(data_root)), stat.st_size, stat.st_mtime_ns))
    encoded = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _triangulated_bop_mesh(path: Path, unit_scale: float) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    vertices = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1)
    vertices = np.asarray(vertices, dtype=np.float64) * float(unit_scale)
    if "face" not in ply:
        raise ValueError(f"BOP mesh has no faces: {path}")
    triangles: list[tuple[int, int, int]] = []
    for raw in ply["face"].data["vertex_indices"]:
        indices = [int(value) for value in raw]
        for offset in range(1, len(indices) - 1):
            triangles.append((indices[0], indices[offset], indices[offset + 1]))
    faces = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(faces):
        raise ValueError(f"BOP mesh has no usable triangle geometry: {path}")
    return vertices, faces


def prepare_bop_surface_atlas(
    data_root: Path,
    atlas_path: Path,
    settings: BuildSettings,
    *,
    models_folder: str = "models_eval",
    unit_scale: float = 0.001,
    progress: bool = True,
) -> tuple[np.memmap, dict[int, int]]:
    """Create a deterministic, area-weighted LM-O mesh surface atlas."""

    data_root = Path(data_root).expanduser().resolve()
    atlas_path = Path(atlas_path).expanduser().resolve()
    models_root = data_root / str(models_folder)
    models_info_path = models_root / "models_info.json"
    models_info = _load_json(models_info_path)
    object_ids = sorted(int(value) for value in models_info)
    mesh_stats = []
    for object_id in object_ids:
        mesh_path = models_root / f"obj_{object_id:06d}.ply"
        stat = mesh_path.stat()
        # Keep the in-memory signature JSON-canonical so a freshly generated
        # list compares equal to the same payload read back from disk.
        mesh_stats.append([object_id, stat.st_size, stat.st_mtime_ns])
    signature = {
        "format": FORMAT,
        "algorithm": "area_weighted_bop_ply_surface_samples_v1",
        "models_info": str(models_info_path),
        "models_info_mtime_ns": models_info_path.stat().st_mtime_ns,
        "mesh_stats": mesh_stats,
        "unit_scale": float(unit_scale),
        "surface_point_count": int(settings.surface_point_count),
        "surface_seed": int(settings.surface_seed),
        "object_ids": object_ids,
    }
    metadata_path = Path(str(atlas_path) + ".json")
    expected_bytes = len(object_ids) * settings.surface_point_count * 3 * 4
    if atlas_path.is_file() and metadata_path.is_file():
        if _load_json(metadata_path) != signature:
            raise ValueError(f"BOP surface-atlas parameters changed: {atlas_path}")
        if atlas_path.stat().st_size != expected_bytes:
            raise ValueError(f"BOP surface atlas has unexpected size: {atlas_path}")
        atlas = np.memmap(
            atlas_path,
            mode="r",
            dtype="<f4",
            shape=(len(object_ids), settings.surface_point_count, 3),
        )
        return atlas, {object_id: index for index, object_id in enumerate(object_ids)}

    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = atlas_path.with_name(atlas_path.name + f".tmp.{os.getpid()}")
    atlas = np.memmap(
        temporary,
        mode="w+",
        dtype="<f4",
        shape=(len(object_ids), settings.surface_point_count, 3),
    )
    iterator: Iterable[int] = object_ids
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(object_ids, desc="LM-O surface atlas", unit="object")
    try:
        for atlas_index, object_id in enumerate(iterator):
            vertices, faces = _triangulated_bop_mesh(
                models_root / f"obj_{object_id:06d}.ply", unit_scale
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
            json.dumps(signature, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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


def _pose_blob(gt: Mapping[str, Any], unit_scale: float) -> bytes:
    T_C_O = np.eye(4, dtype="<f4")
    T_C_O[:3, :3] = np.asarray(gt["cam_R_m2c"], dtype="<f4").reshape(3, 3)
    T_C_O[:3, 3] = np.asarray(gt["cam_t_m2c"], dtype="<f4") * float(unit_scale)
    return T_C_O.tobytes()


def iter_bop_jobs(
    data_root: Path,
    split: str,
    settings: BuildSettings,
    *,
    models_folder: str = "models_eval",
    pose_unit_scale: float = 0.001,
    depth_unit_scale: float = 0.001,
    completed_frame_ids: set[int] | None = None,
    maximum_frames: int | None = None,
) -> Iterator[BOPDecodeJob]:
    """Yield one job per BOP frame containing at least one eligible instance."""

    data_root = Path(data_root).expanduser().resolve()
    completed = completed_frame_ids or set()
    model_ids = {int(value) for value in _load_json(data_root / models_folder / "models_info.json")}
    yielded = 0
    for scene_dir in sorted((data_root / split).glob("[0-9]*")):
        if not (scene_dir / "scene_gt.json").is_file():
            continue
        scene_id = int(scene_dir.name)
        cameras = _load_json(scene_dir / "scene_camera.json")
        ground_truth = _load_json(scene_dir / "scene_gt.json")
        ground_truth_info = _load_json(scene_dir / "scene_gt_info.json")
        first_key = min(ground_truth, key=lambda value: int(value))
        first_depth = scene_dir / "depth" / f"{int(first_key):06d}.png"
        with Image.open(first_depth) as image:
            width, height = image.size
        for key in sorted(ground_truth, key=lambda value: int(value)):
            view_id = int(key)
            frame_id = bop_frame_id(scene_id, view_id)
            if frame_id in completed:
                continue
            camera = cameras[key]
            K = np.asarray(camera["cam_K"], dtype="<f4").reshape(3, 3)
            instances = []
            mask_paths = []
            for gt_id, gt in enumerate(ground_truth[key]):
                object_id = int(gt.get("obj_id", -1))
                if object_id not in model_ids:
                    continue
                info = ground_truth_info[key][gt_id]
                visibility = float(info.get("visib_fract", 1.0))
                visible_pixels = int(info.get("px_count_visib", 0))
                if (
                    visibility < settings.visibility_floor
                    or visible_pixels < settings.min_visible_pixels
                ):
                    continue
                bbox_obj = [float(value) for value in info["bbox_obj"]]
                bbox_visib = [float(value) for value in info["bbox_visib"]]
                instances.append(
                    {
                        "gt_id": int(gt_id),
                        "object_id": object_id,
                        "T_C_O_f32": _pose_blob(gt, pose_unit_scale),
                        "visib_fract": visibility,
                        "px_count_all": int(info.get("px_count_all", visible_pixels)),
                        "px_count_valid": int(
                            info.get("px_count_valid", info.get("px_count_all", visible_pixels))
                        ),
                        "px_count_visib": visible_pixels,
                        "depth_corrupt": 0,
                        "bbox_obj_x": bbox_obj[0],
                        "bbox_obj_y": bbox_obj[1],
                        "bbox_obj_w": bbox_obj[2],
                        "bbox_obj_h": bbox_obj[3],
                        "bbox_visib_x": bbox_visib[0],
                        "bbox_visib_y": bbox_visib[1],
                        "bbox_visib_w": bbox_visib[2],
                        "bbox_visib_h": bbox_visib[3],
                    }
                )
                mask_paths.append(
                    str(scene_dir / "mask_visib" / f"{view_id:06d}_{gt_id:06d}.png")
                )
            if not instances:
                continue
            job = FrameJob(
                frame_id=frame_id,
                scene_id=scene_id,
                view_id=view_id,
                width=width,
                height=height,
                depth_unit_m=float(camera.get("depth_scale", 1.0))
                * float(depth_unit_scale),
                K_f32=K.tobytes(),
                depth_offset=0,
                depth_size=0,
                mask_visib_offset=0,
                mask_visib_size=0,
                relative_path="",
                instances=tuple(instances),
            )
            yield BOPDecodeJob(
                job=job,
                depth_path=str(scene_dir / "depth" / f"{view_id:06d}.png"),
                mask_paths=tuple(mask_paths),
            )
            yielded += 1
            if maximum_frames is not None and yielded >= int(maximum_frames):
                return


def decode_bop_job(value: BOPDecodeJob) -> DecodedFrame:
    with Image.open(value.depth_path) as image:
        depth = np.asarray(image).astype(np.float32) * float(value.job.depth_unit_m)
    masks = []
    for path in value.mask_paths:
        with Image.open(path) as image:
            masks.append(np.asarray(image) > 0)
    masks_array = np.stack(masks, axis=0).astype(np.uint8, copy=False)
    expected = (value.job.height, value.job.width)
    if depth.shape != expected or masks_array.shape[1:] != expected:
        raise ValueError(
            f"LM-O geometry shape mismatch: depth={depth.shape}, "
            f"masks={masks_array.shape}, expected={expected}"
        )
    return DecodedFrame(job=value.job, depth=depth, masks=masks_array)


def _metadata_only_result(value: BOPDecodeJob, settings: BuildSettings) -> FrameResult:
    byte_count = (settings.surface_point_count + 7) // 8
    features = tuple(
        feature_metadata(value.job, row, 0, settings)
        for row in value.job.instances
    )
    return FrameResult(
        frame_id=value.job.frame_id,
        elapsed_seconds=0.0,
        features=features,
        observed_bits=tuple(bytes(byte_count) for _ in features),
    )


def _batched(values: Iterable[Any], count: int) -> Iterator[list[Any]]:
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) >= int(count):
            yield batch
            batch = []
    if batch:
        yield batch


def build_bop_geometry_index(
    *,
    data_root: Path,
    split: str,
    output_path: Path,
    bits_path: Path,
    atlas_path: Path,
    settings: BuildSettings,
    models_folder: str = "models_eval",
    model_unit_scale: float = 0.001,
    pose_unit_scale: float = 0.001,
    depth_unit_scale: float = 0.001,
    compute_surface: bool = True,
    device: str = "cuda",
    workers: int = 8,
    batch_frames: int = 16,
    maximum_frames: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Build or resume an LM-O split sidecar compatible with GSO plan readers."""

    settings.validate()
    data_root = Path(data_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    bits_path = Path(bits_path).expanduser().resolve()
    atlas_path = Path(atlas_path).expanduser().resolve()
    fingerprint = {
        "format": FORMAT,
        "algorithm": ALGORITHM if compute_surface else "metadata_only_v1",
        "data_root": str(data_root),
        "split": str(split),
        "models_folder": str(models_folder),
        "source_signature": _source_signature(data_root, split, models_folder),
        "model_unit_scale": float(model_unit_scale),
        "pose_unit_scale": float(pose_unit_scale),
        "depth_unit_scale": float(depth_unit_scale),
        "settings": settings.as_dict(),
    }
    atlas = None
    object_to_atlas = None
    if compute_surface:
        atlas, object_to_atlas = prepare_bop_surface_atlas(
            data_root,
            atlas_path,
            settings,
            models_folder=models_folder,
            unit_scale=model_unit_scale,
            progress=progress,
        )
    connection, next_id, completed = initialize_output(
        output_path, bits_path, fingerprint, settings
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('source_kind', ?)",
        ("bop_surface" if compute_surface else "bop_metadata",),
    )
    writer = GeometryWriter(connection, bits_path, next_id, settings)
    all_jobs = list(
        iter_bop_jobs(
            data_root,
            split,
            settings,
            models_folder=models_folder,
            pose_unit_scale=pose_unit_scale,
            depth_unit_scale=depth_unit_scale,
        )
    )
    jobs = [value for value in all_jobs if value.job.frame_id not in completed]
    if maximum_frames is not None:
        jobs = jobs[: int(maximum_frames)]
    progress_bar = None
    if progress:
        from tqdm.auto import tqdm

        progress_bar = tqdm(total=len(jobs), desc=f"LM-O {split} geometry", unit="frame")
    started = time.monotonic()
    processed = 0
    device_metrics: dict[str, Any] = {}
    try:
        if compute_surface:
            backend = TorchCoverageBackend(
                atlas, object_to_atlas, settings, device
            )
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
                for batch in _batched(jobs, max(1, int(batch_frames))):
                    decoded = list(executor.map(decode_bop_job, batch))
                    for result in backend.process(decoded):
                        writer.write(result)
                        processed += 1
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
        else:
            for value in jobs:
                writer.write(_metadata_only_result(value, settings))
                processed += 1
                if progress_bar is not None:
                    progress_bar.update(1)
        writer.flush()
        remaining = len(all_jobs) - len(completed) - processed
        if maximum_frames is None and remaining <= 0:
            _create_secondary_indexes(connection)
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('index_complete','1')"
            )
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('updated_at_unix',?)",
            (str(time.time()),),
        )
        connection.commit()
    finally:
        writer.close()
        connection.close()
        if progress_bar is not None:
            progress_bar.close()
    elapsed = time.monotonic() - started
    return {
        "format": FORMAT,
        "split": str(split),
        "compute_surface": bool(compute_surface),
        "output_path": str(output_path),
        "bits_path": str(bits_path),
        "processed_frames": processed,
        "total_source_frames": len(all_jobs),
        "remaining_source_frames": max(0, len(all_jobs) - len(completed) - processed),
        "elapsed_seconds": elapsed,
        "frames_per_second": processed / elapsed if elapsed else math.inf,
        **device_metrics,
    }


def geometry_index_summary(index_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(Path(index_path).expanduser().resolve())
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        row = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT object_id), COUNT(DISTINCT frame_id),
                   SUM(crop_feasible), AVG(visib_fract), AVG(observed_fraction),
                   MIN(base_norm_fx), MAX(max_norm_fx)
            FROM frame_features
            """
        ).fetchone()
        return {
            "index_path": str(Path(index_path).expanduser().resolve()),
            "index_complete": metadata.get("index_complete") == "1",
            "features": int(row[0] or 0),
            "objects": int(row[1] or 0),
            "frames": int(row[2] or 0),
            "crop_feasible": int(row[3] or 0),
            "mean_visibility": float(row[4] or 0.0),
            "mean_observed_fraction": float(row[5] or 0.0),
            "minimum_base_norm_fx": float(row[6] or 0.0),
            "maximum_crop_norm_fx": float(row[7] or 0.0),
        }
    finally:
        connection.close()


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def bop_camera_distribution_summary(
    data_root: Path,
    split: str,
    index_path: Path,
    *,
    models_folder: str = "models_eval",
    pose_unit_scale: float = 0.001,
) -> dict[str, Any]:
    """Summarize indexed BOP camera/object geometry without decoding images.

    ``object_center_z_m`` is the object-origin depth, not a mean over rendered
    surface pixels.  It is the stable metadata proxy useful for catching a
    millimetre/metre error or a gross distance-domain shift before evaluation.
    """

    data_root = Path(data_root).expanduser().resolve()
    index_path = Path(index_path).expanduser().resolve()
    models_info = _load_json(data_root / models_folder / "models_info.json")
    diameters_m = {
        int(object_id): float(info["diameter"]) * 0.001
        for object_id, info in models_info.items()
    }
    connection = sqlite3.connect(index_path)
    try:
        rows = connection.execute(
            """
            SELECT scene_id,view_id,gt_id,object_id,width,height,fx,fy,cx,cy,
                   bbox_obj_w,bbox_obj_h,visib_fract,view_z
            FROM frame_features ORDER BY scene_id,view_id,gt_id
            """
        ).fetchall()
    finally:
        connection.close()

    scene_ground_truth: dict[int, Any] = {}
    center_z = []
    center_distance = []
    distance_over_diameter = []
    normalized_focal = []
    bbox_area_fraction = []
    visibility = []
    view_z = []
    intrinsics = set()
    for row in rows:
        (
            scene_id,
            view_id,
            gt_id,
            object_id,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            bbox_w,
            bbox_h,
            visib_fract,
            direction_z,
        ) = row
        scene_id, view_id, gt_id, object_id = map(
            int, (scene_id, view_id, gt_id, object_id)
        )
        ground_truth = scene_ground_truth.get(scene_id)
        if ground_truth is None:
            ground_truth = _load_json(
                data_root / split / f"{scene_id:06d}" / "scene_gt.json"
            )
            scene_ground_truth[scene_id] = ground_truth
        gt = ground_truth[str(view_id)][gt_id]
        translation = (
            np.asarray(gt["cam_t_m2c"], dtype=np.float64)
            * float(pose_unit_scale)
        )
        distance = float(np.linalg.norm(translation))
        center_z.append(float(translation[2]))
        center_distance.append(distance)
        distance_over_diameter.append(distance / diameters_m[object_id])
        normalized_focal.append(
            math.sqrt((float(fx) / int(width)) * (float(fy) / int(height)))
        )
        bbox_area_fraction.append(
            float(bbox_w) * float(bbox_h) / (int(width) * int(height))
        )
        visibility.append(float(visib_fract))
        view_z.append(float(direction_z))
        intrinsics.add(
            tuple(round(float(value), 6) for value in (fx, fy, cx, cy, width, height))
        )
    return {
        "split": str(split),
        "features": len(rows),
        "unique_intrinsics": len(intrinsics),
        "normalized_focal": _distribution(normalized_focal),
        "object_center_z_m": _distribution(center_z),
        "camera_center_distance_m": _distribution(center_distance),
        "camera_distance_over_diameter": _distribution(distance_over_diameter),
        "full_object_bbox_area_fraction": _distribution(bbox_area_fraction),
        "visibility_fraction": _distribution(visibility),
        "object_frame_view_direction_z": _distribution(view_z),
    }
