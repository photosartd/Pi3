#!/usr/bin/env python3
"""Audit MegaPose-GSO capacity under matched-intrinsics sampling.

The production GSO trainer draws stochastic samples rather than enumerating a
finite list of frame combinations.  This script therefore counts physical
sampling units:

* render -> scene: one clean-render bank and one eligible query track;
* scene -> scene: one ordered reference-track/query-track pair in different
  scenes.
* positive-reference scene -> scene: one reference-track/query-frame anchor.
  All N references still come from that single reference track and scene, but
  at least one selected reference must be within the configured object-view
  angle of the query.

Frame subsets within those units are deliberately not counted.  For example,
one 512-view render bank and one 40-view query track admit an enormous number
of distinct N/K frame combinations, but the current sampler chooses them
randomly at runtime.

Only SQLite indexes and the prepared split manifest are read.  RGB, depth,
masks, and tar payloads are never decoded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NormalizedCameraProfile:
    """Camera profile with explicit target-resolution normalization."""

    width: int
    height: int
    raw_K: np.ndarray
    final_K: np.ndarray
    raw_focal: tuple[float, float]
    raw_principal: tuple[float, float]
    final_focal: tuple[float, float]
    final_principal: tuple[float, float]


@dataclass(frozen=True)
class Track:
    object_id: int
    scene_id: int
    track_id: int
    query_view_count: int
    reference_view_count: int


@dataclass
class CapacityMatrix:
    current: np.ndarray
    matched: np.ndarray
    current_objects: np.ndarray
    matched_objects: np.ndarray
    current_scene_pairs: np.ndarray
    matched_scene_pairs: np.ndarray


@dataclass
class PositiveReferenceCapacity:
    """Query-anchored capacity for one-scene positive-reference sampling."""

    all_anchor_pairs: np.ndarray
    positive_anchor_pairs: np.ndarray
    positive_focal_anchor_pairs: np.ndarray
    all_query_frames: np.ndarray
    positive_query_frames: np.ndarray
    positive_focal_query_frames: np.ndarray
    all_objects: np.ndarray
    positive_objects: np.ndarray
    positive_focal_objects: np.ndarray
    positive_track_pairs: np.ndarray
    positive_focal_track_pairs: np.ndarray


@dataclass(frozen=True)
class PoseTrack:
    """Eligible query and reference viewpoints for one physical scene track."""

    object_id: int
    scene_id: int
    track_id: int
    query_directions: np.ndarray
    reference_directions: np.ndarray


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro&immutable=1", uri=True, timeout=60.0
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _matrix_from_blob(value: bytes) -> np.ndarray:
    matrix = np.frombuffer(value, dtype="<f4")
    if matrix.size != 9:
        raise ValueError(f"Expected nine float32 intrinsics values, got {matrix.size}")
    return matrix.reshape(3, 3).astype(np.float64, copy=True)


def _pose_matrix_from_blob(value: bytes) -> np.ndarray:
    matrix = np.frombuffer(value, dtype="<f4")
    if matrix.size != 16:
        raise ValueError(f"Expected sixteen float32 pose values, got {matrix.size}")
    return matrix.reshape(4, 4).astype(np.float64, copy=True)


def object_view_direction(T_C_O: np.ndarray) -> np.ndarray:
    """Return the unit object-to-camera-center direction in object coordinates."""

    T_C_O = np.asarray(T_C_O, dtype=np.float64)
    if T_C_O.shape != (4, 4) or not np.isfinite(T_C_O).all():
        raise ValueError(f"Invalid object-to-camera pose: {T_C_O}")
    camera_center_O = -(T_C_O[:3, :3].T @ T_C_O[:3, 3])
    norm = float(np.linalg.norm(camera_center_O))
    if norm <= 1e-12:
        raise ValueError("Camera center coincides with the object origin")
    return camera_center_O / norm


def _validate_intrinsics(K: np.ndarray, label: str) -> None:
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"Invalid intrinsics for {label}: {K}")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"Non-positive focal length for {label}: {K}")
    canonical_tail = np.asarray([[K[1, 0]], [K[2, 0]], [K[2, 1]], [K[2, 2] - 1]])
    if float(np.max(np.abs(canonical_tail))) > 1e-6:
        raise ValueError(f"Non-canonical pinhole intrinsics for {label}: {K}")
    if abs(float(K[0, 1])) > 1e-6:
        raise ValueError(f"Skewed intrinsics are unsupported for {label}: {K}")


def _camera_matrix_of_crop(
    K: np.ndarray,
    input_resolution: Sequence[int],
    output_resolution: Sequence[int],
    *,
    scaling: float = 1.0,
    offset_factor: float = 0.5,
) -> np.ndarray:
    """Pure-metadata equivalent of pi3.utils.cropping.camera_matrix_of_crop."""

    input_resolution = np.asarray(input_resolution, dtype=np.float64)
    output_resolution = np.asarray(output_resolution, dtype=np.float64)
    margins = input_resolution * float(scaling) - output_resolution
    if bool((margins < -1e-6).any()):
        raise ValueError(
            f"Output resolution {output_resolution} exceeds scaled input "
            f"{input_resolution * scaling}"
        )
    offset = float(offset_factor) * margins
    output = np.asarray(K, dtype=np.float64).copy()
    # Match the OpenCV -> COLMAP -> OpenCV half-pixel convention in the helper.
    output[0, 2] += 0.5
    output[1, 2] += 0.5
    output[:2, :] *= float(scaling)
    output[:2, 2] -= offset
    output[0, 2] -= 0.5
    output[1, 2] -= 0.5
    return output


def effective_pi3_intrinsics(
    K: np.ndarray,
    width: int,
    height: int,
    target_resolution: Sequence[int],
) -> np.ndarray:
    """Reproduce BaseDataset's deterministic principal crop and resize on K."""

    K = np.asarray(K, dtype=np.float64).copy()
    _validate_intrinsics(K, "effective-intrinsics input")
    width, height = int(width), int(height)
    target = np.asarray(target_resolution, dtype=np.int64)
    if width <= 0 or height <= 0 or target.shape != (2,) or bool((target <= 0).any()):
        raise ValueError("Image and target resolutions must be positive width/height pairs")

    cx, cy = np.rint(K[:2, 2]).astype(np.int64)
    margin_x = min(int(cx), width - int(cx))
    margin_y = min(int(cy), height - int(cy))
    if margin_x <= width / 5 or margin_y <= height / 5:
        raise ValueError(
            f"Principal point {(float(K[0, 2]), float(K[1, 2]))} is outside "
            f"Pi3's accepted central region for {(width, height)}"
        )

    left, top = int(cx) - margin_x, int(cy) - margin_y
    cropped_resolution = np.asarray((2 * margin_x, 2 * margin_y), dtype=np.int64)
    K[0, 2] -= left
    K[1, 2] -= top

    scale = float(np.max(target / cropped_resolution)) + 1e-8
    resized_resolution = np.floor(cropped_resolution * scale).astype(np.int64)
    K = _camera_matrix_of_crop(
        K,
        cropped_resolution,
        resized_resolution,
        scaling=scale,
    )
    return _camera_matrix_of_crop(K, resized_resolution, target)


def make_camera_profile(
    K: np.ndarray,
    width: int,
    height: int,
    target_resolution: Sequence[int],
    *,
    label: str,
) -> NormalizedCameraProfile:
    K = np.asarray(K, dtype=np.float64)
    _validate_intrinsics(K, label)
    target_width, target_height = (int(value) for value in target_resolution)
    final_K = effective_pi3_intrinsics(K, width, height, target_resolution)
    return NormalizedCameraProfile(
        width=int(width),
        height=int(height),
        raw_K=K.copy(),
        final_K=final_K,
        raw_focal=(float(K[0, 0]) / width, float(K[1, 1]) / height),
        raw_principal=(float(K[0, 2]) / width, float(K[1, 2]) / height),
        final_focal=(
            float(final_K[0, 0]) / target_width,
            float(final_K[1, 1]) / target_height,
        ),
        final_principal=(
            float(final_K[0, 2]) / target_width,
            float(final_K[1, 2]) / target_height,
        ),
    )


def focal_log_distance(
    first: NormalizedCameraProfile,
    second: NormalizedCameraProfile,
    *,
    comparison_space: str,
) -> float:
    first_focal = (
        first.final_focal if comparison_space == "postprocess" else first.raw_focal
    )
    second_focal = (
        second.final_focal if comparison_space == "postprocess" else second.raw_focal
    )
    return max(
        abs(math.log(first_focal[0] / second_focal[0])),
        abs(math.log(first_focal[1] / second_focal[1])),
    )


def principal_distance(
    first: NormalizedCameraProfile,
    second: NormalizedCameraProfile,
    *,
    comparison_space: str,
) -> float:
    first_point = (
        first.final_principal if comparison_space == "postprocess" else first.raw_principal
    )
    second_point = (
        second.final_principal if comparison_space == "postprocess" else second.raw_principal
    )
    return max(
        abs(first_point[0] - second_point[0]),
        abs(first_point[1] - second_point[1]),
    )


def cameras_match(
    first: NormalizedCameraProfile,
    second: NormalizedCameraProfile,
    *,
    focal_log_tolerance: float,
    principal_point_tolerance: float | None,
    comparison_space: str,
) -> bool:
    if focal_log_distance(first, second, comparison_space=comparison_space) > float(
        focal_log_tolerance
    ):
        return False
    return principal_point_tolerance is None or principal_distance(
        first, second, comparison_space=comparison_space
    ) <= float(principal_point_tolerance)


def _validate_index_metadata(connection: sqlite3.Connection, label: str) -> None:
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    if metadata.get("index_complete") != "1":
        raise ValueError(f"{label} index is not marked complete")


def load_split_ids(split_path: Path, split_name: str) -> tuple[set[int], set[int]]:
    if split_name == "all":
        return set(), set()
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "pi3_entity_split_v1":
        raise ValueError(f"Unsupported split manifest: {split_path}")
    values = manifest.get("splits", {}).get(split_name)
    if not isinstance(values, dict):
        raise ValueError(f"Missing split {split_name!r} in {split_path}")
    object_ids = {int(value) for value in values.get("object_ids", [])}
    scene_ids = {int(value) for value in values.get("scene_ids", [])}
    if not object_ids or not scene_ids:
        raise ValueError(f"Split {split_name!r} has empty object or scene membership")
    return object_ids, scene_ids


def load_scene_cameras(
    index_path: Path,
    target_resolution: Sequence[int],
) -> dict[int, NormalizedCameraProfile]:
    connection = _readonly_connection(index_path)
    try:
        _validate_index_metadata(connection, "MegaPose-GSO")
        rows = connection.execute(
            """
            SELECT scene_id,
                   MIN(width) AS min_width, MAX(width) AS max_width,
                   MIN(height) AS min_height, MAX(height) AS max_height,
                   COUNT(DISTINCT K_f32) AS intrinsics_count,
                   MIN(K_f32) AS K_f32
            FROM frames GROUP BY scene_id ORDER BY scene_id
            """
        )
        cameras = {}
        for row in rows:
            scene_id = int(row["scene_id"])
            if (
                int(row["min_width"]) != int(row["max_width"])
                or int(row["min_height"]) != int(row["max_height"])
                or int(row["intrinsics_count"]) != 1
            ):
                raise ValueError(
                    f"Scene {scene_id} does not have one fixed resolution/intrinsics profile"
                )
            cameras[scene_id] = make_camera_profile(
                _matrix_from_blob(row["K_f32"]),
                int(row["min_width"]),
                int(row["min_height"]),
                target_resolution,
                label=f"scene {scene_id}",
            )
        if not cameras:
            raise ValueError(f"No scene cameras found in {index_path}")
        return cameras
    finally:
        connection.close()


def _depth_predicate(policy: str) -> str:
    if policy == "clean_only":
        return "i.depth_corrupt = 0"
    if policy == "exclude_known_bad":
        return "i.depth_corrupt IS NOT 1"
    if policy == "all":
        return "1 = 1"
    raise ValueError(f"Unsupported depth corruption policy {policy!r}")


def _load_track_view_counts(
    connection: sqlite3.Connection,
    *,
    visibility_min: float,
    visibility_inclusive: bool,
    min_visible_pixels: int,
    depth_corruption_policy: str,
) -> dict[tuple[int, int, int], int]:
    """Load eligible-frame counts per physical object track."""

    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    profile = None
    if visibility_inclusive and metadata.get("sampling_profiles_complete") == "1":
        profile = connection.execute(
            """
            SELECT profile_id FROM sampling_profiles
            WHERE ABS(visibility_min - ?) < 1e-12
              AND min_visible_pixels = ?
              AND depth_corruption_policy = ?
            LIMIT 1
            """,
            (visibility_min, min_visible_pixels, depth_corruption_policy),
        ).fetchone()
    if profile is not None:
        rows: Iterable[sqlite3.Row] = connection.execute(
            """
            SELECT object_id, scene_id, gt_id, view_count
            FROM track_sampling_profiles
            WHERE profile_id=? ORDER BY object_id, scene_id, gt_id
            """,
            (str(profile[0]),),
        )
    else:
        operator = ">=" if visibility_inclusive else ">"
        rows = connection.execute(
            f"""
            SELECT i.object_id, i.scene_id, i.gt_id, COUNT(*) AS view_count
            FROM instances AS i
            WHERE {_depth_predicate(depth_corruption_policy)}
              AND i.visib_fract {operator} ?
              AND i.px_count_visib >= ?
            GROUP BY i.object_id, i.scene_id, i.gt_id
            ORDER BY i.object_id, i.scene_id, i.gt_id
            """,
            (visibility_min, min_visible_pixels),
        )
    return {
        (int(row["object_id"]), int(row["scene_id"]), int(row["gt_id"])): int(
            row["view_count"]
        )
        for row in rows
    }


def load_tracks(
    index_path: Path,
    *,
    object_ids: set[int],
    scene_ids: set[int],
    visibility_min: float,
    visibility_inclusive: bool,
    reference_visibility_min: float,
    reference_visibility_inclusive: bool,
    min_visible_pixels: int,
    depth_corruption_policy: str,
) -> list[Track]:
    connection = _readonly_connection(index_path)
    try:
        query_counts = _load_track_view_counts(
            connection,
            visibility_min=visibility_min,
            visibility_inclusive=visibility_inclusive,
            min_visible_pixels=min_visible_pixels,
            depth_corruption_policy=depth_corruption_policy,
        )
        reference_counts = _load_track_view_counts(
            connection,
            visibility_min=reference_visibility_min,
            visibility_inclusive=reference_visibility_inclusive,
            min_visible_pixels=min_visible_pixels,
            depth_corruption_policy=depth_corruption_policy,
        )
        tracks = []
        for (object_id, scene_id, track_id), query_view_count in query_counts.items():
            if object_ids and object_id not in object_ids:
                continue
            if scene_ids and scene_id not in scene_ids:
                continue
            tracks.append(
                Track(
                    object_id=object_id,
                    scene_id=scene_id,
                    track_id=track_id,
                    query_view_count=query_view_count,
                    reference_view_count=reference_counts.get(
                        (object_id, scene_id, track_id), 0
                    ),
                )
            )
        return tracks
    finally:
        connection.close()


def iter_object_pose_tracks(
    index_path: Path,
    *,
    object_ids: set[int],
    scene_ids: set[int],
    visibility_min: float,
    visibility_inclusive: bool,
    reference_visibility_min: float,
    reference_visibility_inclusive: bool,
    min_visible_pixels: int,
    depth_corruption_policy: str,
) -> Iterator[list[PoseTrack]]:
    """Stream per-object pose tracks without retaining the full index in RAM."""

    query_operator = ">=" if visibility_inclusive else ">"
    reference_operator = (
        (lambda value: value >= reference_visibility_min)
        if reference_visibility_inclusive
        else (lambda value: value > reference_visibility_min)
    )
    connection = _readonly_connection(index_path)
    try:
        rows = connection.execute(
            f"""
            SELECT i.object_id, i.scene_id, i.gt_id, i.view_id,
                   i.visib_fract, i.T_C_O_f32
            FROM instances AS i
            WHERE {_depth_predicate(depth_corruption_policy)}
              AND i.visib_fract {query_operator} ?
              AND i.px_count_visib >= ?
            ORDER BY i.object_id, i.scene_id, i.gt_id, i.view_id
            """,
            (visibility_min, min_visible_pixels),
        )
        current_track_key: tuple[int, int, int] | None = None
        pose_blobs: list[bytes] = []
        reference_flags: list[bool] = []
        object_tracks: list[PoseTrack] = []

        def finish_track() -> PoseTrack | None:
            if current_track_key is None or not pose_blobs:
                return None
            object_id, scene_id, track_id = current_track_key
            poses = np.frombuffer(b"".join(pose_blobs), dtype="<f4").reshape(-1, 4, 4)
            camera_centers = -np.einsum(
                "nji,nj->ni",
                poses[:, :3, :3].astype(np.float64),
                poses[:, :3, 3].astype(np.float64),
            )
            norms = np.linalg.norm(camera_centers, axis=1)
            if bool((norms <= 1e-12).any()) or not np.isfinite(norms).all():
                raise ValueError(
                    f"Invalid object-camera center in track {current_track_key}"
                )
            directions = camera_centers / norms[:, None]
            reference_mask = np.asarray(reference_flags, dtype=bool)
            empty = np.empty((0, 3), dtype=np.float64)
            return PoseTrack(
                object_id=object_id,
                scene_id=scene_id,
                track_id=track_id,
                query_directions=directions,
                reference_directions=(
                    directions[reference_mask]
                    if bool(reference_mask.any())
                    else empty
                ),
            )

        for row in rows:
            object_id, scene_id, track_id = (
                int(row["object_id"]),
                int(row["scene_id"]),
                int(row["gt_id"]),
            )
            if object_ids and object_id not in object_ids:
                continue
            if scene_ids and scene_id not in scene_ids:
                continue
            track_key = (object_id, scene_id, track_id)
            if current_track_key is not None and track_key != current_track_key:
                completed = finish_track()
                if completed is not None:
                    object_tracks.append(completed)
                    if object_id != completed.object_id:
                        yield object_tracks
                        object_tracks = []
                pose_blobs = []
                reference_flags = []
            current_track_key = track_key
            pose_blobs.append(bytes(row["T_C_O_f32"]))
            reference_flags.append(reference_operator(float(row["visib_fract"])))

        completed = finish_track()
        if completed is not None:
            object_tracks.append(completed)
        if object_tracks:
            yield object_tracks
    finally:
        connection.close()


def load_render_banks(
    index_path: Path,
    target_resolution: Sequence[int],
) -> tuple[dict[int, int], dict[int, NormalizedCameraProfile]]:
    connection = _readonly_connection(index_path)
    try:
        _validate_index_metadata(connection, "reference")
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        width = int(metadata["resolution_width"])
        height = int(metadata["resolution_height"])
        rows = connection.execute(
            """
            SELECT o.object_id, o.view_count,
                   COUNT(DISTINCT v.K_f32) AS intrinsics_count,
                   MIN(v.K_f32) AS K_f32
            FROM objects AS o JOIN views AS v ON v.object_id=o.object_id
            GROUP BY o.object_id, o.view_count ORDER BY o.object_id
            """
        )
        counts = {}
        cameras = {}
        for row in rows:
            object_id = int(row["object_id"])
            if int(row["intrinsics_count"]) != 1:
                raise ValueError(
                    f"Reference object {object_id} has multiple intrinsic profiles; "
                    "this audit requires one profile per reference bank"
                )
            counts[object_id] = int(row["view_count"])
            cameras[object_id] = make_camera_profile(
                _matrix_from_blob(row["K_f32"]),
                width,
                height,
                target_resolution,
                label=f"reference object {object_id}",
            )
        if not counts:
            raise ValueError(f"No reference banks found in {index_path}")
        return counts, cameras
    finally:
        connection.close()


def _empty_capacity(reference_counts: Sequence[int], query_counts: Sequence[int]) -> CapacityMatrix:
    shape = (len(reference_counts), len(query_counts))
    return CapacityMatrix(
        current=np.zeros(shape, dtype=np.int64),
        matched=np.zeros(shape, dtype=np.int64),
        current_objects=np.zeros(shape, dtype=np.int64),
        matched_objects=np.zeros(shape, dtype=np.int64),
        current_scene_pairs=np.zeros(shape, dtype=np.int64),
        matched_scene_pairs=np.zeros(shape, dtype=np.int64),
    )


def count_render_to_scene_capacity(
    tracks: Sequence[Track],
    scene_cameras: Mapping[int, NormalizedCameraProfile],
    render_counts: Mapping[int, int],
    render_cameras: Mapping[int, NormalizedCameraProfile],
    reference_counts: Sequence[int],
    query_counts: Sequence[int],
    *,
    focal_log_tolerance: float,
    principal_point_tolerance: float | None,
    comparison_space: str,
) -> CapacityMatrix:
    result = _empty_capacity(reference_counts, query_counts)
    by_object: dict[int, list[Track]] = {}
    for track in tracks:
        by_object.setdefault(track.object_id, []).append(track)

    for object_id, object_tracks in by_object.items():
        if object_id not in render_counts or object_id not in render_cameras:
            continue
        views = np.asarray(
            [track.query_view_count for track in object_tracks], dtype=np.int64
        )
        scenes = np.asarray([track.scene_id for track in object_tracks], dtype=np.int64)
        matched_track = np.asarray(
            [
                cameras_match(
                    render_cameras[object_id],
                    scene_cameras[int(scene_id)],
                    focal_log_tolerance=focal_log_tolerance,
                    principal_point_tolerance=principal_point_tolerance,
                    comparison_space=comparison_space,
                )
                for scene_id in scenes
            ],
            dtype=bool,
        )
        for ref_index, reference_count in enumerate(reference_counts):
            if render_counts[object_id] < int(reference_count):
                continue
            for query_index, query_count in enumerate(query_counts):
                eligible = views >= int(query_count)
                current = int(eligible.sum())
                matched = int((eligible & matched_track).sum())
                result.current[ref_index, query_index] += current
                result.matched[ref_index, query_index] += matched
                result.current_objects[ref_index, query_index] += int(current > 0)
                result.matched_objects[ref_index, query_index] += int(matched > 0)
                result.current_scene_pairs[ref_index, query_index] += len(
                    set(int(value) for value in scenes[eligible])
                )
                result.matched_scene_pairs[ref_index, query_index] += len(
                    set(int(value) for value in scenes[eligible & matched_track])
                )
    return result


def count_scene_to_scene_capacity(
    tracks: Sequence[Track],
    scene_cameras: Mapping[int, NormalizedCameraProfile],
    reference_counts: Sequence[int],
    query_counts: Sequence[int],
    *,
    focal_log_tolerance: float,
    principal_point_tolerance: float | None,
    comparison_space: str,
) -> CapacityMatrix:
    result = _empty_capacity(reference_counts, query_counts)
    by_object: dict[int, list[Track]] = {}
    for track in tracks:
        by_object.setdefault(track.object_id, []).append(track)

    for object_tracks in by_object.values():
        scene_values = sorted({track.scene_id for track in object_tracks})
        scene_to_index = {scene_id: index for index, scene_id in enumerate(scene_values)}
        current_reference_availability = np.zeros(
            (len(scene_values), len(reference_counts)), dtype=np.int64
        )
        proposed_reference_availability = np.zeros(
            (len(scene_values), len(reference_counts)), dtype=np.int64
        )
        query_availability = np.zeros(
            (len(scene_values), len(query_counts)), dtype=np.int64
        )
        for track in object_tracks:
            scene_index = scene_to_index[track.scene_id]
            current_reference_availability[scene_index] += np.asarray(
                [track.query_view_count >= value for value in reference_counts],
                dtype=np.int64,
            )
            proposed_reference_availability[scene_index] += np.asarray(
                [track.reference_view_count >= value for value in reference_counts],
                dtype=np.int64,
            )
            query_availability[scene_index] += np.asarray(
                [track.query_view_count >= value for value in query_counts], dtype=np.int64
            )

        cameras = [scene_cameras[scene_id] for scene_id in scene_values]
        focal = np.asarray(
            [
                camera.final_focal
                if comparison_space == "postprocess"
                else camera.raw_focal
                for camera in cameras
            ],
            dtype=np.float64,
        )
        principal = np.asarray(
            [
                camera.final_principal
                if comparison_space == "postprocess"
                else camera.raw_principal
                for camera in cameras
            ],
            dtype=np.float64,
        )
        focal_delta = np.max(
            np.abs(np.log(focal[:, None, :] / focal[None, :, :])), axis=-1
        )
        match = focal_delta <= float(focal_log_tolerance)
        if principal_point_tolerance is not None:
            principal_delta = np.max(
                np.abs(principal[:, None, :] - principal[None, :, :]), axis=-1
            )
            match &= principal_delta <= float(principal_point_tolerance)
        # Production ScenePairPolicy requires physically different scenes.
        np.fill_diagonal(match, False)

        current = (
            np.outer(
                current_reference_availability.sum(axis=0),
                query_availability.sum(axis=0),
            )
            - current_reference_availability.T @ query_availability
        )
        matched = (
            proposed_reference_availability.T
            @ match.astype(np.int64)
            @ query_availability
        )
        current_reference_present = (
            current_reference_availability > 0
        ).astype(np.int64)
        proposed_reference_present = (
            proposed_reference_availability > 0
        ).astype(np.int64)
        query_present = (query_availability > 0).astype(np.int64)
        current_scene_pairs = (
            np.outer(current_reference_present.sum(axis=0), query_present.sum(axis=0))
            - current_reference_present.T @ query_present
        )
        matched_scene_pairs = (
            proposed_reference_present.T
            @ match.astype(np.int64)
            @ query_present
        )

        result.current += current
        result.matched += matched
        result.current_scene_pairs += current_scene_pairs
        result.matched_scene_pairs += matched_scene_pairs
        result.current_objects += current > 0
        result.matched_objects += matched > 0
    return result


def count_positive_reference_capacity(
    object_pose_tracks: Iterable[Sequence[PoseTrack]],
    scene_cameras: Mapping[int, NormalizedCameraProfile],
    reference_counts: Sequence[int],
    *,
    positive_view_angle_degrees: float,
    minimum_positive_references: int,
    focal_log_tolerance: float,
    principal_point_tolerance: float | None,
    comparison_space: str,
) -> PositiveReferenceCapacity:
    """Count one-reference-scene samples with a nearby object viewpoint.

    One physical unit is an ordered ``(reference_track, query_frame)`` anchor.
    All N references are drawn from the one reference track.  A unit is
    positive-feasible when that track has at least N eligible frames and at
    least ``minimum_positive_references`` frames whose object-to-camera-center
    directions are within ``positive_view_angle_degrees`` of the query.
    """

    shape = (len(reference_counts),)
    result = PositiveReferenceCapacity(
        **{
            field: np.zeros(shape, dtype=np.int64)
            for field in PositiveReferenceCapacity.__dataclass_fields__
        }
    )
    radius = 2.0 * math.sin(math.radians(positive_view_angle_degrees) / 2.0)
    minimum_reference_count = min(int(value) for value in reference_counts)

    for tracks_value in object_pose_tracks:
        tracks = list(tracks_value)
        if not tracks:
            continue
        track_count = len(tracks)
        scene_values = np.asarray([track.scene_id for track in tracks], dtype=np.int64)
        query_view_counts = np.asarray(
            [len(track.query_directions) for track in tracks], dtype=np.int64
        )
        reference_view_counts = np.asarray(
            [len(track.reference_directions) for track in tracks], dtype=np.int64
        )
        reference_track_indices = np.flatnonzero(
            reference_view_counts >= minimum_reference_count
        )
        if reference_track_indices.size == 0:
            continue

        query_directions = np.concatenate(
            [track.query_directions for track in tracks]
        )
        query_owners = np.repeat(np.arange(track_count), query_view_counts)
        reference_directions = np.concatenate(
            [tracks[index].reference_directions for index in reference_track_indices]
        )
        reference_owners = np.concatenate(
            [
                np.full(
                    len(tracks[index].reference_directions), index, dtype=np.int64
                )
                for index in reference_track_indices
            ]
        )

        # Sparse frame-level neighbours avoid an all-query/all-reference dense
        # angular matrix. Rows are query frames and columns are reference frames.
        neighbours = cKDTree(query_directions).sparse_distance_matrix(
            cKDTree(reference_directions),
            radius,
            output_type="coo_matrix",
        )
        anchor_codes = (
            neighbours.row.astype(np.int64) * track_count
            + reference_owners[neighbours.col]
        )
        anchor_codes, positive_frame_counts = np.unique(
            anchor_codes, return_counts=True
        )
        anchor_codes = anchor_codes[
            positive_frame_counts >= int(minimum_positive_references)
        ]
        positive_query_indices = anchor_codes // track_count
        positive_reference_owners = anchor_codes % track_count
        positive_query_owners = query_owners[positive_query_indices]
        different_scene = (
            scene_values[positive_reference_owners]
            != scene_values[positive_query_owners]
        )
        positive_query_indices = positive_query_indices[different_scene]
        positive_reference_owners = positive_reference_owners[different_scene]
        positive_query_owners = positive_query_owners[different_scene]

        cameras = [scene_cameras[int(scene_id)] for scene_id in scene_values]
        focal = np.asarray(
            [
                camera.final_focal
                if comparison_space == "postprocess"
                else camera.raw_focal
                for camera in cameras
            ],
            dtype=np.float64,
        )
        principal = np.asarray(
            [
                camera.final_principal
                if comparison_space == "postprocess"
                else camera.raw_principal
                for camera in cameras
            ],
            dtype=np.float64,
        )
        focal_delta = np.max(
            np.abs(np.log(focal[:, None, :] / focal[None, :, :])), axis=-1
        )
        focal_match = focal_delta <= float(focal_log_tolerance)
        if principal_point_tolerance is not None:
            principal_delta = np.max(
                np.abs(principal[:, None, :] - principal[None, :, :]), axis=-1
            )
            focal_match &= principal_delta <= float(principal_point_tolerance)
        focal_match &= scene_values[:, None] != scene_values[None, :]
        anchor_focal_match = focal_match[
            positive_reference_owners, positive_query_owners
        ]

        local_all = np.zeros(shape, dtype=np.int64)
        local_positive = np.zeros(shape, dtype=np.int64)
        local_positive_focal = np.zeros(shape, dtype=np.int64)
        for count_index, reference_count in enumerate(reference_counts):
            eligible_reference = reference_view_counts >= int(reference_count)
            reference_per_scene: dict[int, int] = {}
            for scene_id in scene_values[eligible_reference]:
                reference_per_scene[int(scene_id)] = (
                    reference_per_scene.get(int(scene_id), 0) + 1
                )
            total_references = int(eligible_reference.sum())
            per_query_track = np.asarray(
                [
                    total_references - reference_per_scene.get(int(scene_id), 0)
                    for scene_id in scene_values
                ],
                dtype=np.int64,
            )
            all_anchors = int(np.dot(query_view_counts, per_query_track))
            all_queries = int(query_view_counts[per_query_track > 0].sum())

            positive_mask = (
                reference_view_counts[positive_reference_owners]
                >= int(reference_count)
            )
            focal_mask = positive_mask & anchor_focal_match
            positive_count = int(positive_mask.sum())
            focal_count = int(focal_mask.sum())

            result.all_anchor_pairs[count_index] += all_anchors
            result.all_query_frames[count_index] += all_queries
            result.positive_anchor_pairs[count_index] += positive_count
            result.positive_focal_anchor_pairs[count_index] += focal_count
            if positive_count:
                result.positive_query_frames[count_index] += int(
                    np.unique(positive_query_indices[positive_mask]).size
                )
                result.positive_track_pairs[count_index] += int(
                    np.unique(
                        positive_reference_owners[positive_mask] * track_count
                        + positive_query_owners[positive_mask]
                    ).size
                )
            if focal_count:
                result.positive_focal_query_frames[count_index] += int(
                    np.unique(positive_query_indices[focal_mask]).size
                )
                result.positive_focal_track_pairs[count_index] += int(
                    np.unique(
                        positive_reference_owners[focal_mask] * track_count
                        + positive_query_owners[focal_mask]
                    ).size
                )
            local_all[count_index] = all_anchors
            local_positive[count_index] = positive_count
            local_positive_focal[count_index] = focal_count

        result.all_objects += local_all > 0
        result.positive_objects += local_positive > 0
        result.positive_focal_objects += local_positive_focal > 0
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else float("nan")


def _format_integer(value: int) -> str:
    return f"{int(value):,}"


def _format_percent(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.2f}%"


def _matrix_rows(
    capacity: CapacityMatrix,
    reference_counts: Sequence[int],
    query_counts: Sequence[int],
) -> list[dict[str, int | float]]:
    rows = []
    for ref_index, reference_count in enumerate(reference_counts):
        for query_index, query_count in enumerate(query_counts):
            current = int(capacity.current[ref_index, query_index])
            matched = int(capacity.matched[ref_index, query_index])
            rows.append(
                {
                    "reference_count": int(reference_count),
                    "query_count": int(query_count),
                    "current_track_pairs": current,
                    "matched_track_pairs": matched,
                    "retained_fraction": _ratio(matched, current),
                    "current_object_count": int(
                        capacity.current_objects[ref_index, query_index]
                    ),
                    "matched_object_count": int(
                        capacity.matched_objects[ref_index, query_index]
                    ),
                    "current_object_scene_pairs": int(
                        capacity.current_scene_pairs[ref_index, query_index]
                    ),
                    "matched_object_scene_pairs": int(
                        capacity.matched_scene_pairs[ref_index, query_index]
                    ),
                }
            )
    return rows


def _lookup(
    capacity: CapacityMatrix,
    reference_counts: Sequence[int],
    query_counts: Sequence[int],
    reference_count: int,
    query_count: int,
) -> dict[str, int | float]:
    ref_index = list(reference_counts).index(int(reference_count))
    query_index = list(query_counts).index(int(query_count))
    current = int(capacity.current[ref_index, query_index])
    matched = int(capacity.matched[ref_index, query_index])
    return {
        "current": current,
        "matched": matched,
        "retained_fraction": _ratio(matched, current),
        "current_objects": int(capacity.current_objects[ref_index, query_index]),
        "matched_objects": int(capacity.matched_objects[ref_index, query_index]),
        "current_scene_pairs": int(capacity.current_scene_pairs[ref_index, query_index]),
        "matched_scene_pairs": int(capacity.matched_scene_pairs[ref_index, query_index]),
    }


def _positive_reference_rows(
    capacity: PositiveReferenceCapacity,
    reference_counts: Sequence[int],
) -> list[dict[str, int | float]]:
    rows = []
    for index, reference_count in enumerate(reference_counts):
        all_anchors = int(capacity.all_anchor_pairs[index])
        positive_anchors = int(capacity.positive_anchor_pairs[index])
        focal_anchors = int(capacity.positive_focal_anchor_pairs[index])
        rows.append(
            {
                "reference_count": int(reference_count),
                "all_reference_track_query_frame_anchors": all_anchors,
                "positive_reference_track_query_frame_anchors": positive_anchors,
                "positive_and_focal_matched_anchors": focal_anchors,
                "positive_retained_fraction": _ratio(
                    positive_anchors, all_anchors
                ),
                "positive_and_focal_retained_fraction": _ratio(
                    focal_anchors, all_anchors
                ),
                "focal_fraction_of_positive": _ratio(
                    focal_anchors, positive_anchors
                ),
                "all_query_frames": int(capacity.all_query_frames[index]),
                "positive_query_frames": int(
                    capacity.positive_query_frames[index]
                ),
                "positive_and_focal_query_frames": int(
                    capacity.positive_focal_query_frames[index]
                ),
                "all_object_count": int(capacity.all_objects[index]),
                "positive_object_count": int(capacity.positive_objects[index]),
                "positive_and_focal_object_count": int(
                    capacity.positive_focal_objects[index]
                ),
                "positive_reference_track_query_track_pairs": int(
                    capacity.positive_track_pairs[index]
                ),
                "positive_and_focal_track_pairs": int(
                    capacity.positive_focal_track_pairs[index]
                ),
            }
        )
    return rows


def _scheduled_retention(
    capacity: CapacityMatrix,
    reference_counts: Sequence[int],
    query_counts: Sequence[int],
) -> float:
    """Average count retention under the current total-view/count scheduler."""

    totals = range(
        min(reference_counts) + min(query_counts),
        max(reference_counts) + max(query_counts) + 1,
    )
    total_values = []
    for total in totals:
        pair_ratios = []
        for ref_index, reference_count in enumerate(reference_counts):
            for query_index, query_count in enumerate(query_counts):
                if reference_count + query_count != total:
                    continue
                current = int(capacity.current[ref_index, query_index])
                if current:
                    pair_ratios.append(
                        int(capacity.matched[ref_index, query_index]) / current
                    )
        if pair_ratios:
            total_values.append(float(np.mean(pair_ratios)))
    return float(np.mean(total_values)) if total_values else float("nan")


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        label: float(value)
        for label, value in zip(
            ("min", "p10", "p25", "median", "p75", "p90", "max"),
            np.quantile(values, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)),
        )
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    data_root = args.data_root.expanduser().resolve()
    scene_index = (
        args.scene_index.expanduser().resolve()
        if args.scene_index
        else data_root / "pi3_index" / "megapose_gso.sqlite"
    )
    split_path = (
        args.split_path.expanduser().resolve()
        if args.split_path
        else scene_index.with_suffix(".splits.json")
    )
    if args.references_index:
        references_index = args.references_index.expanduser().resolve()
    else:
        if args.assets_root is None:
            raise ValueError("Provide --assets-root or --references-index")
        references_index = (
            args.assets_root.expanduser().resolve()
            / "renders"
            / "pi3_index"
            / "references.sqlite"
        )
    for path in (scene_index, split_path, references_index):
        if not path.is_file():
            raise FileNotFoundError(path)

    target_resolution = tuple(int(value) for value in args.target_resolution)
    object_ids, scene_ids = load_split_ids(split_path, args.scene_split)
    scene_cameras = load_scene_cameras(scene_index, target_resolution)
    tracks = load_tracks(
        scene_index,
        object_ids=object_ids,
        scene_ids=scene_ids,
        visibility_min=args.visibility_min,
        visibility_inclusive=not args.visibility_exclusive,
        reference_visibility_min=args.reference_visibility_min,
        reference_visibility_inclusive=not args.reference_visibility_exclusive,
        min_visible_pixels=args.min_visible_pixels,
        depth_corruption_policy=args.depth_corruption_policy,
    )
    render_counts, render_cameras = load_render_banks(
        references_index, target_resolution
    )
    reference_counts = tuple(
        range(args.reference_range[0], args.reference_range[1] + 1)
    )
    query_counts = tuple(range(args.query_range[0], args.query_range[1] + 1))
    principal_tolerance = (
        None if args.ignore_principal_point else args.principal_point_tolerance
    )

    render_capacity = count_render_to_scene_capacity(
        tracks,
        scene_cameras,
        render_counts,
        render_cameras,
        reference_counts,
        query_counts,
        focal_log_tolerance=args.focal_log_tolerance,
        principal_point_tolerance=principal_tolerance,
        comparison_space=args.comparison_space,
    )
    scene_capacity = count_scene_to_scene_capacity(
        tracks,
        scene_cameras,
        reference_counts,
        query_counts,
        focal_log_tolerance=args.focal_log_tolerance,
        principal_point_tolerance=principal_tolerance,
        comparison_space=args.comparison_space,
    )
    positive_reference_capacity = count_positive_reference_capacity(
        iter_object_pose_tracks(
            scene_index,
            object_ids=object_ids,
            scene_ids=scene_ids,
            visibility_min=args.visibility_min,
            visibility_inclusive=not args.visibility_exclusive,
            reference_visibility_min=args.reference_visibility_min,
            reference_visibility_inclusive=(
                not args.reference_visibility_exclusive
            ),
            min_visible_pixels=args.min_visible_pixels,
            depth_corruption_policy=args.depth_corruption_policy,
        ),
        scene_cameras,
        reference_counts,
        positive_view_angle_degrees=args.positive_view_angle_degrees,
        minimum_positive_references=args.minimum_positive_references,
        focal_log_tolerance=args.focal_log_tolerance,
        principal_point_tolerance=principal_tolerance,
        comparison_space=args.comparison_space,
    )

    selected_scene_cameras = [
        camera
        for scene_id, camera in scene_cameras.items()
        if not scene_ids or scene_id in scene_ids
    ]
    render_profiles = list(render_cameras.values())
    unique_render_focal = sorted(
        {
            tuple(round(value, 12) for value in camera.final_focal)
            for camera in render_profiles
        }
    )
    summaries = []
    requested_summaries = {
        (reference_counts[0], query_counts[0]),
        (5, 1),
        (reference_counts[-1], query_counts[-1]),
    }
    for reference_count, query_count in sorted(requested_summaries):
        if reference_count not in reference_counts or query_count not in query_counts:
            continue
        summaries.append(
            {
                "reference_count": reference_count,
                "query_count": query_count,
                "render_to_scene": _lookup(
                    render_capacity,
                    reference_counts,
                    query_counts,
                    reference_count,
                    query_count,
                ),
                "scene_to_scene": _lookup(
                    scene_capacity,
                    reference_counts,
                    query_counts,
                    reference_count,
                    query_count,
                ),
            }
        )

    return {
        "definition": {
            "render_to_scene_unit": "one reference-bank/query-track pair",
            "scene_to_scene_unit": (
                "one ordered reference-track/query-track pair in different scenes"
            ),
            "frame_combinations_counted": False,
            "current_denominator": (
                "current unrestricted sampler using visibility_min for both roles"
            ),
            "matched_numerator": (
                "intrinsics-matched sampler using reference_visibility_min only "
                "for scene-derived references and visibility_min for queries"
            ),
            "positive_reference_unit": (
                "one ordered reference-track/query-frame anchor; all references "
                "come from that one reference scene"
            ),
            "positive_reference_selection": (
                "choose N distinct reference frames from one track and force at "
                "least minimum_positive_references frames within the configured "
                "object-view angle of the query"
            ),
        },
        "inputs": {
            "scene_index": str(scene_index),
            "split_path": str(split_path),
            "references_index": str(references_index),
            "scene_split": args.scene_split,
            "target_resolution": list(target_resolution),
            "reference_range": list(args.reference_range),
            "query_range": list(args.query_range),
            "visibility_min": args.visibility_min,
            "visibility_inclusive": not args.visibility_exclusive,
            "reference_visibility_min": args.reference_visibility_min,
            "reference_visibility_inclusive": (
                not args.reference_visibility_exclusive
            ),
            "min_visible_pixels": args.min_visible_pixels,
            "depth_corruption_policy": args.depth_corruption_policy,
            "positive_query_count": 1,
        },
        "matching": {
            "comparison_space": args.comparison_space,
            "focal_log_tolerance": args.focal_log_tolerance,
            "accepted_focal_ratio": [
                math.exp(-args.focal_log_tolerance),
                math.exp(args.focal_log_tolerance),
            ],
            "principal_point_tolerance": principal_tolerance,
            "rule": (
                "max(|log(fx1_norm/fx2_norm)|, |log(fy1_norm/fy2_norm)|) "
                "<= focal_log_tolerance"
            ),
            "positive_view_angle_degrees": args.positive_view_angle_degrees,
            "minimum_positive_references": args.minimum_positive_references,
            "positive_view_rule": (
                "angle between object-to-camera-center unit directions; this is "
                "independent of the focal matching rule"
            ),
        },
        "population": {
            "eligible_tracks": len(tracks),
            "tracks_with_reference_visible_frames": sum(
                track.reference_view_count > 0 for track in tracks
            ),
            "eligible_objects": len({track.object_id for track in tracks}),
            "eligible_scenes": len({track.scene_id for track in tracks}),
            "scene_camera_count": len(selected_scene_cameras),
            "reference_object_count": len(render_counts),
            "unique_reference_final_focal_profiles": len(unique_render_focal),
            "reference_final_focal_profiles": [
                list(value) for value in unique_render_focal
            ],
            "scene_final_fx_over_width": _quantiles(
                [camera.final_focal[0] for camera in selected_scene_cameras]
            ),
            "scene_final_fy_over_height": _quantiles(
                [camera.final_focal[1] for camera in selected_scene_cameras]
            ),
        },
        "summaries": summaries,
        "scheduler_weighted_retention": {
            "render_to_scene": _scheduled_retention(
                render_capacity, reference_counts, query_counts
            ),
            "scene_to_scene": _scheduled_retention(
                scene_capacity, reference_counts, query_counts
            ),
            "equal_mixture": float(
                np.mean(
                    [
                        _scheduled_retention(
                            render_capacity, reference_counts, query_counts
                        ),
                        _scheduled_retention(
                            scene_capacity, reference_counts, query_counts
                        ),
                    ]
                )
            ),
        },
        "render_to_scene": _matrix_rows(
            render_capacity, reference_counts, query_counts
        ),
        "scene_to_scene": _matrix_rows(scene_capacity, reference_counts, query_counts),
        "positive_reference_scene_to_scene": _positive_reference_rows(
            positive_reference_capacity, reference_counts
        ),
    }


def print_report(report: Mapping[str, object]) -> None:
    inputs = report["inputs"]
    matching = report["matching"]
    population = report["population"]
    assert isinstance(inputs, Mapping)
    assert isinstance(matching, Mapping)
    assert isinstance(population, Mapping)
    ratio = matching["accepted_focal_ratio"]
    assert isinstance(ratio, list)
    print("MegaPose-GSO matched-intrinsics capacity audit")
    print(f"  split: {inputs['scene_split']}")
    print(f"  target resolution: {inputs['target_resolution']}")
    print(
        "  focal rule: max log-distance over fx/W and fy/H <= "
        f"{float(matching['focal_log_tolerance']):.4f} "
        f"(ratio {float(ratio[0]):.4f}..{float(ratio[1]):.4f})"
    )
    print(f"  comparison space: {matching['comparison_space']}")
    print(f"  principal-point tolerance: {matching['principal_point_tolerance']}")
    visibility_operator = ">=" if inputs["visibility_inclusive"] else ">"
    reference_visibility_operator = (
        ">=" if inputs["reference_visibility_inclusive"] else ">"
    )
    print(
        "  visibility: queries/current denominator "
        f"{visibility_operator} {float(inputs['visibility_min']):.2f}; "
        "scene-derived references in matched numerator "
        f"{reference_visibility_operator} "
        f"{float(inputs['reference_visibility_min']):.2f}"
    )
    print(
        "  eligible population: "
        f"{_format_integer(int(population['eligible_tracks']))} tracks, "
        f"{_format_integer(int(population['eligible_objects']))} objects, "
        f"{_format_integer(int(population['eligible_scenes']))} scenes"
    )
    print(
        "  sample units exclude combinatorial choices of individual frames "
        "inside a track/reference bank"
    )

    print("\nCapacity relative to the current unrestricted sampler")
    header = (
        "  protocol                 N   K          current          matched   retained  "
        "objects"
    )
    print(header)
    for summary in report["summaries"]:  # type: ignore[index]
        assert isinstance(summary, Mapping)
        for label, key in (
            ("render -> scene", "render_to_scene"),
            ("scene -> scene", "scene_to_scene"),
        ):
            values = summary[key]
            assert isinstance(values, Mapping)
            print(
                f"  {label:<24}"
                f"{int(summary['reference_count']):>3}"
                f"{int(summary['query_count']):>4}"
                f"{_format_integer(int(values['current'])):>17}"
                f"{_format_integer(int(values['matched'])):>17}"
                f"{_format_percent(float(values['retained_fraction'])):>11}  "
                f"{int(values['matched_objects']):,}/{int(values['current_objects']):,}"
            )

    print(
        "\nOne-scene positive-reference capacity (one query image, forced positive)"
    )
    print(
        "  All N references come from one reference track/scene. A positive "
        f"reference is within {float(matching['positive_view_angle_degrees']):g}° "
        "in object-view direction."
    )
    print(
        "    N        all anchors       >=positive   retained   +focal-10%   "
        "retained  query coverage"
    )
    for row in report["positive_reference_scene_to_scene"]:  # type: ignore[index]
        assert isinstance(row, Mapping)
        positive_queries = int(row["positive_query_frames"])
        all_queries = int(row["all_query_frames"])
        print(
            f"  {int(row['reference_count']):>3}"
            f"{_format_integer(int(row['all_reference_track_query_frame_anchors'])):>19}"
            f"{_format_integer(int(row['positive_reference_track_query_frame_anchors'])):>19}"
            f"{_format_percent(float(row['positive_retained_fraction'])):>11}"
            f"{_format_integer(int(row['positive_and_focal_matched_anchors'])):>14}"
            f"{_format_percent(float(row['positive_and_focal_retained_fraction'])):>11}  "
            f"{positive_queries:,}/{all_queries:,}"
        )
    print(
        "  '+focal-10%' additionally applies the normalized focal and "
        "principal-point rules; angular proximity alone does not match intrinsics."
    )

    weighted = report["scheduler_weighted_retention"]
    assert isinstance(weighted, Mapping)
    print("\nAverage retention under the current dynamic N/K scheduler")
    print(
        "  render -> scene: "
        f"{_format_percent(float(weighted['render_to_scene']))}"
    )
    print(
        "  scene  -> scene: "
        f"{_format_percent(float(weighted['scene_to_scene']))}"
    )
    print(
        "  current 50/50 mixture (equal component average): "
        f"{_format_percent(float(weighted['equal_mixture']))}"
    )
    print(
        "\nThe configured epoch/step count would not shrink automatically; the "
        "trainer samples with replacement. A smaller retained percentage means "
        "the surviving physical tracks/pairs are revisited more often."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--scene-index", type=Path)
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--references-index", type=Path)
    parser.add_argument(
        "--scene-split", choices=("train", "val", "all"), default="train"
    )
    parser.add_argument("--target-resolution", type=int, nargs=2, default=(336, 252))
    parser.add_argument("--reference-range", type=int, nargs=2, default=(2, 8))
    parser.add_argument("--query-range", type=int, nargs=2, default=(1, 1))
    parser.add_argument("--visibility-min", type=float, default=0.1)
    parser.add_argument("--visibility-exclusive", action="store_true")
    parser.add_argument("--reference-visibility-min", type=float, default=0.3)
    parser.add_argument("--reference-visibility-exclusive", action="store_true")
    parser.add_argument("--positive-view-angle-degrees", type=float, default=10.0)
    parser.add_argument("--minimum-positive-references", type=int, default=1)
    parser.add_argument("--min-visible-pixels", type=int, default=64)
    parser.add_argument(
        "--depth-corruption-policy",
        choices=("clean_only", "exclude_known_bad", "all"),
        default="clean_only",
    )
    parser.add_argument("--focal-log-tolerance", type=float, default=0.10)
    parser.add_argument("--principal-point-tolerance", type=float, default=0.01)
    parser.add_argument("--ignore-principal-point", action="store_true")
    parser.add_argument(
        "--comparison-space", choices=("postprocess", "raw"), default="postprocess"
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if (
        args.reference_range[0] <= 0
        or args.reference_range[1] < args.reference_range[0]
    ):
        parser.error("--reference-range must be a positive MIN MAX pair")
    if args.query_range[0] <= 0 or args.query_range[1] < args.query_range[0]:
        parser.error("--query-range must be a positive MIN MAX pair")
    if args.focal_log_tolerance < 0 or args.principal_point_tolerance < 0:
        parser.error("intrinsics tolerances must be non-negative")
    if not 0 <= args.visibility_min <= 1:
        parser.error("--visibility-min must be in [0, 1]")
    if not 0 <= args.reference_visibility_min <= 1:
        parser.error("--reference-visibility-min must be in [0, 1]")
    if args.reference_visibility_min < args.visibility_min:
        parser.error(
            "--reference-visibility-min must be at least --visibility-min so "
            "the matched sampler remains a subset of the current denominator"
        )
    if not 0 < args.positive_view_angle_degrees <= 180:
        parser.error("--positive-view-angle-degrees must be in (0, 180]")
    if args.minimum_positive_references <= 0:
        parser.error("--minimum-positive-references must be positive")
    if args.minimum_positive_references > args.reference_range[0]:
        parser.error(
            "--minimum-positive-references cannot exceed the smallest requested "
            "reference count"
        )
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    print_report(report)
    if args.output_json:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_path)
        print(f"\nWrote machine-readable report: {output_path}")


if __name__ == "__main__":
    main()
