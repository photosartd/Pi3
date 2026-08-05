#!/usr/bin/env python3
"""Build a random-access index and entity split for MegaPose-GSO shards.

The source tar files are deliberately left untouched.  The index stores the
small JSON metadata required for sampling and byte ranges for every payload,
so a later dataset adapter can seek directly to selected RGB/depth/mask data.
This file uses only the Python standard library and can also be executed
directly without importing the rest of Pi3.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import struct
import sys
import tarfile
import tempfile
import time
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
SPLIT_FORMAT = "pi3_entity_split_v1"
DEFAULT_SPLIT_SEED = 2026
DEFAULT_VALIDATION_OBJECT_FRACTION = 0.2
DEFAULT_VALIDATION_SCENE_FRACTION = 0.2
EXPECTED_VIEWS_PER_SCENE = 40
MM_TO_M = 0.001

SHARD_RE = re.compile(r"^shard-(\d+)\.tar$")
FRAME_RE = re.compile(r"^(\d{6})_(\d{6})$")
MANIFEST_TEMPLATE = "gso_1M_extr_shard_{shard_number:06d}_corrupted_depths.txt"

REQUIRED_SUFFIXES = frozenset(
    {
        "camera.json",
        "depth.png",
        "gt.json",
        "gt_info.json",
        "mask.json",
        "mask_visib.json",
    }
)
RGB_SUFFIXES = frozenset({"rgb.jpg", "rgb.jpeg", "rgb.png"})
PAYLOAD_SUFFIXES = (
    "rgb",
    "depth",
    "camera",
    "gt",
    "gt_info",
    "mask",
    "mask_visib",
)

SECONDARY_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_frames_scene_view "
    "ON frames(scene_id, view_id)",
    "CREATE INDEX IF NOT EXISTS idx_instances_object_sampling "
    "ON instances(object_id, depth_corrupt, visib_fract, scene_id, frame_id)",
    "CREATE INDEX IF NOT EXISTS idx_instances_track_sampling "
    "ON instances(scene_id, gt_id, depth_corrupt, visib_fract, view_id)",
    "CREATE INDEX IF NOT EXISTS idx_scene_tracks_object "
    "ON scene_tracks(object_id, clean_visible_frame_count, scene_id, gt_id)",
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS shards (
    id INTEGER PRIMARY KEY,
    shard_number INTEGER NOT NULL UNIQUE,
    relative_path TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    instance_count INTEGER NOT NULL,
    manifest_relative_path TEXT,
    manifest_entry_count INTEGER,
    manifest_available INTEGER NOT NULL,
    manifest_size_bytes INTEGER,
    manifest_mtime_ns INTEGER,
    processed_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY,
    frame_key TEXT NOT NULL UNIQUE,
    scene_id INTEGER NOT NULL,
    view_id INTEGER NOT NULL,
    shard_id INTEGER NOT NULL REFERENCES shards(id) ON DELETE CASCADE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    depth_scale_raw REAL NOT NULL,
    depth_unit_m REAL NOT NULL,
    K_f32 BLOB NOT NULL,
    T_C_W_f32 BLOB,
    rgb_format TEXT NOT NULL,
    rgb_offset INTEGER NOT NULL,
    rgb_size INTEGER NOT NULL,
    depth_offset INTEGER NOT NULL,
    depth_size INTEGER NOT NULL,
    camera_offset INTEGER NOT NULL,
    camera_size INTEGER NOT NULL,
    gt_offset INTEGER NOT NULL,
    gt_size INTEGER NOT NULL,
    gt_info_offset INTEGER NOT NULL,
    gt_info_size INTEGER NOT NULL,
    mask_offset INTEGER NOT NULL,
    mask_size INTEGER NOT NULL,
    mask_visib_offset INTEGER NOT NULL,
    mask_visib_size INTEGER NOT NULL,
    UNIQUE(scene_id, view_id)
);

CREATE TABLE IF NOT EXISTS instances (
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    gt_id INTEGER NOT NULL,
    scene_id INTEGER NOT NULL,
    view_id INTEGER NOT NULL,
    object_id INTEGER NOT NULL,
    T_C_O_f32 BLOB NOT NULL,
    bbox_obj_x INTEGER NOT NULL,
    bbox_obj_y INTEGER NOT NULL,
    bbox_obj_w INTEGER NOT NULL,
    bbox_obj_h INTEGER NOT NULL,
    bbox_visib_x INTEGER NOT NULL,
    bbox_visib_y INTEGER NOT NULL,
    bbox_visib_w INTEGER NOT NULL,
    bbox_visib_h INTEGER NOT NULL,
    px_count_all INTEGER NOT NULL,
    px_count_valid INTEGER NOT NULL,
    px_count_visib INTEGER NOT NULL,
    visib_fract REAL NOT NULL,
    depth_corrupt INTEGER,
    PRIMARY KEY(frame_id, gt_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS scene_tracks (
    scene_id INTEGER NOT NULL,
    gt_id INTEGER NOT NULL,
    object_id INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    visible_frame_count INTEGER NOT NULL,
    clean_visible_frame_count INTEGER NOT NULL,
    corrupt_frame_count INTEGER NOT NULL,
    min_view_id INTEGER NOT NULL,
    max_view_id INTEGER NOT NULL,
    mean_visib_fract REAL NOT NULL,
    max_visib_fract REAL NOT NULL,
    PRIMARY KEY(scene_id, gt_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS scenes (
    scene_id INTEGER PRIMARY KEY,
    frame_count INTEGER NOT NULL,
    track_count INTEGER NOT NULL,
    object_count INTEGER NOT NULL,
    min_view_id INTEGER NOT NULL,
    max_view_id INTEGER NOT NULL,
    is_complete INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS objects (
    object_id INTEGER PRIMARY KEY,
    observation_count INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    clean_visible_observation_count INTEGER NOT NULL,
    scene_count INTEGER NOT NULL,
    track_count INTEGER NOT NULL
);
"""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _pack_f32(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *(float(value) for value in values))


def unpack_f32_matrix(blob: bytes, rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
    """Decode one matrix blob without requiring NumPy."""

    values = struct.unpack(f"<{rows * cols}f", blob)
    return tuple(
        tuple(values[row * cols : (row + 1) * cols]) for row in range(rows)
    )


def _make_transform(rotation: Sequence[float], translation_mm: Sequence[float]) -> bytes:
    if len(rotation) != 9 or len(translation_mm) != 3:
        raise ValueError("A pose must contain a 3x3 rotation and a 3-vector translation")
    values = [
        rotation[0], rotation[1], rotation[2], translation_mm[0] * MM_TO_M,
        rotation[3], rotation[4], rotation[5], translation_mm[1] * MM_TO_M,
        rotation[6], rotation[7], rotation[8], translation_mm[2] * MM_TO_M,
        0.0, 0.0, 0.0, 1.0,
    ]
    return _pack_f32(values)


def _json_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> Any:
    stream = tar.extractfile(member)
    if stream is None:
        raise ValueError(f"Cannot read tar member {member.name!r}")
    try:
        return json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in tar member {member.name!r}: {exc}") from exc


def _png_dimensions(tar: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[int, int]:
    stream = tar.extractfile(member)
    if stream is None:
        raise ValueError(f"Cannot read tar member {member.name!r}")
    header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Depth member {member.name!r} is not a valid PNG")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _relative_to_root(path: Path, data_root: Path) -> str:
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return str(path)


def _parse_corruption_manifest(path: Path) -> dict[str, frozenset[int]]:
    entries: dict[str, frozenset[int]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            key, separator, raw_ids = line.partition(":")
            if not separator or FRAME_RE.fullmatch(key) is None:
                raise ValueError(f"Malformed line {line_number} in {path}: {line!r}")
            try:
                parsed = ast.literal_eval(raw_ids)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Malformed instance list at line {line_number} in {path}"
                ) from exc
            if not isinstance(parsed, (list, tuple)) or any(
                not isinstance(value, int) for value in parsed
            ):
                raise ValueError(
                    f"Expected a list of integer gt_ids at line {line_number} in {path}"
                )
            if key in entries:
                raise ValueError(f"Duplicate frame key {key!r} in {path}")
            entries[key] = frozenset(int(value) for value in parsed)
    return entries


def _discover_shards(data_root: Path, shard_glob: str) -> list[tuple[int, Path]]:
    shards = []
    for path in data_root.glob(shard_glob):
        match = SHARD_RE.fullmatch(path.name)
        if path.is_file() and match is not None:
            shards.append((int(match.group(1)), path.resolve()))
    shards.sort(key=lambda item: item[0])
    numbers = [item[0] for item in shards]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Duplicate shard numbers were discovered")
    return shards


def _connect(index_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -262144")
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.executescript(SCHEMA_SQL)
    current = connection.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if current is not None and int(current[0]) != SCHEMA_VERSION:
        connection.close()
        raise RuntimeError(
            f"Index schema version {current[0]} is incompatible with version "
            f"{SCHEMA_VERSION}; rerun with --rebuild"
        )
    with connection:
        values = {
            "schema_version": str(SCHEMA_VERSION),
            "translation_unit": "metre",
            "depth_unit_formula": "raw_depth * depth_scale_raw * 0.001",
            "pose_convention": "T_C_O maps object coordinates to camera coordinates",
            "expected_views_per_scene": str(EXPECTED_VIEWS_PER_SCENE),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", values.items()
        )
    return connection


def _frame_members(tar: tarfile.TarFile) -> dict[str, dict[str, tarfile.TarInfo]]:
    frames: dict[str, dict[str, tarfile.TarInfo]] = {}
    for member in tar:
        if not member.isfile():
            continue
        key, separator, suffix = member.name.partition(".")
        if not separator or FRAME_RE.fullmatch(key) is None:
            raise ValueError(f"Unexpected member name {member.name!r}")
        bucket = frames.setdefault(key, {})
        if suffix in bucket:
            raise ValueError(f"Duplicate member {member.name!r}")
        bucket[suffix] = member
    return frames


def _validate_members(frame_key: str, members: dict[str, tarfile.TarInfo]) -> str:
    missing = REQUIRED_SUFFIXES.difference(members)
    rgb = RGB_SUFFIXES.intersection(members)
    unexpected = set(members).difference(REQUIRED_SUFFIXES).difference(RGB_SUFFIXES)
    if missing or len(rgb) != 1 or unexpected:
        raise ValueError(
            f"Frame {frame_key} has invalid members: missing={sorted(missing)}, "
            f"rgb={sorted(rgb)}, unexpected={sorted(unexpected)}"
        )
    return next(iter(rgb))


def _bbox(info: dict[str, Any], name: str) -> tuple[int, int, int, int]:
    values = info.get(name)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"Expected {name} to contain four integers")
    return tuple(int(value) for value in values)


def _read_shard(
    shard_path: Path,
    manifest: dict[str, frozenset[int]] | None,
) -> tuple[list[dict[str, Any]], int]:
    frame_rows: list[dict[str, Any]] = []
    instance_count = 0
    with tarfile.open(shard_path, mode="r:") as tar:
        frames = _frame_members(tar)
        if manifest is not None:
            missing_manifest = set(frames).difference(manifest)
            extra_manifest = set(manifest).difference(frames)
            if missing_manifest or extra_manifest:
                raise ValueError(
                    f"Corruption manifest mismatch for {shard_path.name}: "
                    f"{len(missing_manifest)} missing and {len(extra_manifest)} extra keys"
                )

        for frame_key in sorted(frames):
            match = FRAME_RE.fullmatch(frame_key)
            assert match is not None
            scene_id, view_id = (int(value) for value in match.groups())
            members = frames[frame_key]
            rgb_suffix = _validate_members(frame_key, members)
            camera = _json_member(tar, members["camera.json"])
            ground_truth = _json_member(tar, members["gt.json"])
            gt_info = _json_member(tar, members["gt_info.json"])
            if not isinstance(camera, dict):
                raise ValueError(f"Camera metadata for {frame_key} is not an object")
            if not isinstance(ground_truth, list) or not isinstance(gt_info, list):
                raise ValueError(f"GT metadata for {frame_key} is not a list")
            if len(ground_truth) != len(gt_info):
                raise ValueError(
                    f"GT/GT-info length mismatch for {frame_key}: "
                    f"{len(ground_truth)} != {len(gt_info)}"
                )
            K = camera.get("cam_K")
            if not isinstance(K, list) or len(K) != 9:
                raise ValueError(f"Invalid cam_K for {frame_key}")
            depth_scale = float(camera.get("depth_scale"))
            width, height = _png_dimensions(tar, members["depth.png"])
            rotation_w2c = camera.get("cam_R_w2c")
            translation_w2c = camera.get("cam_t_w2c")
            T_C_W = None
            if rotation_w2c is not None or translation_w2c is not None:
                if not isinstance(rotation_w2c, list) or not isinstance(
                    translation_w2c, list
                ):
                    raise ValueError(f"Incomplete world camera pose for {frame_key}")
                T_C_W = _make_transform(rotation_w2c, translation_w2c)

            offsets = {}
            suffix_by_payload = {
                "rgb": rgb_suffix,
                "depth": "depth.png",
                "camera": "camera.json",
                "gt": "gt.json",
                "gt_info": "gt_info.json",
                "mask": "mask.json",
                "mask_visib": "mask_visib.json",
            }
            for payload, suffix in suffix_by_payload.items():
                member = members[suffix]
                offsets[f"{payload}_offset"] = int(member.offset_data)
                offsets[f"{payload}_size"] = int(member.size)

            corrupt_gt_ids = (
                manifest.get(frame_key, frozenset()) if manifest is not None else None
            )
            instance_rows = []
            for gt_id, (pose, info) in enumerate(zip(ground_truth, gt_info)):
                if not isinstance(pose, dict) or not isinstance(info, dict):
                    raise ValueError(f"Invalid instance metadata for {frame_key}/{gt_id}")
                bbox_obj = _bbox(info, "bbox_obj")
                bbox_visib = _bbox(info, "bbox_visib")
                instance_rows.append(
                    (
                        int(gt_id),
                        scene_id,
                        view_id,
                        int(pose["obj_id"]),
                        _make_transform(pose["cam_R_m2c"], pose["cam_t_m2c"]),
                        *bbox_obj,
                        *bbox_visib,
                        int(info["px_count_all"]),
                        int(info["px_count_valid"]),
                        int(info["px_count_visib"]),
                        float(info["visib_fract"]),
                        None if corrupt_gt_ids is None else int(gt_id in corrupt_gt_ids),
                    )
                )

            frame_rows.append(
                {
                    "frame_key": frame_key,
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "width": width,
                    "height": height,
                    "depth_scale_raw": depth_scale,
                    "depth_unit_m": depth_scale * MM_TO_M,
                    "K_f32": _pack_f32(K),
                    "T_C_W_f32": T_C_W,
                    "rgb_format": rgb_suffix.rsplit(".", 1)[-1],
                    **offsets,
                    "instances": instance_rows,
                }
            )
            instance_count += len(instance_rows)
    return frame_rows, instance_count


FRAME_INSERT_SQL = """
INSERT INTO frames(
    frame_key, scene_id, view_id, shard_id, width, height,
    depth_scale_raw, depth_unit_m, K_f32, T_C_W_f32, rgb_format,
    rgb_offset, rgb_size, depth_offset, depth_size,
    camera_offset, camera_size, gt_offset, gt_size,
    gt_info_offset, gt_info_size, mask_offset, mask_size,
    mask_visib_offset, mask_visib_size
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""

INSTANCE_INSERT_SQL = """
INSERT INTO instances(
    frame_id, gt_id, scene_id, view_id, object_id, T_C_O_f32,
    bbox_obj_x, bbox_obj_y, bbox_obj_w, bbox_obj_h,
    bbox_visib_x, bbox_visib_y, bbox_visib_w, bbox_visib_h,
    px_count_all, px_count_valid, px_count_visib, visib_fract, depth_corrupt
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _replace_shard(
    connection: sqlite3.Connection,
    *,
    data_root: Path,
    shard_number: int,
    shard_path: Path,
    manifest_path: Path | None,
    manifest_entries: int | None,
    frame_rows: list[dict[str, Any]],
    instance_count: int,
) -> None:
    stat = shard_path.stat()
    with connection:
        existing = connection.execute(
            "SELECT id FROM shards WHERE shard_number = ?", (shard_number,)
        ).fetchone()
        if existing is not None:
            connection.execute("DELETE FROM shards WHERE id = ?", (existing[0],))
        cursor = connection.execute(
            """
            INSERT INTO shards(
                shard_number, relative_path, size_bytes, mtime_ns,
                frame_count, instance_count, manifest_relative_path,
                manifest_entry_count, manifest_available, manifest_size_bytes,
                manifest_mtime_ns, processed_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                shard_number,
                _relative_to_root(shard_path, data_root),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                len(frame_rows),
                int(instance_count),
                None if manifest_path is None else _relative_to_root(manifest_path, data_root),
                manifest_entries,
                int(manifest_path is not None),
                None if manifest_path is None else int(manifest_path.stat().st_size),
                None if manifest_path is None else int(manifest_path.stat().st_mtime_ns),
                _utc_now(),
            ),
        )
        shard_id = int(cursor.lastrowid)
        for row in frame_rows:
            values = [
                row["frame_key"], row["scene_id"], row["view_id"], shard_id,
                row["width"], row["height"], row["depth_scale_raw"],
                row["depth_unit_m"], row["K_f32"], row["T_C_W_f32"],
                row["rgb_format"],
            ]
            for payload in PAYLOAD_SUFFIXES:
                values.extend((row[f"{payload}_offset"], row[f"{payload}_size"]))
            frame_cursor = connection.execute(FRAME_INSERT_SQL, values)
            frame_id = int(frame_cursor.lastrowid)
            connection.executemany(
                INSTANCE_INSERT_SQL,
                ((frame_id, *instance_row) for instance_row in row["instances"]),
            )


def _drop_secondary_indexes(connection: sqlite3.Connection) -> None:
    with connection:
        for statement in SECONDARY_INDEX_SQL:
            name = statement.split()[5]
            connection.execute(f"DROP INDEX IF EXISTS {name}")


def _create_secondary_indexes(connection: sqlite3.Connection) -> None:
    with connection:
        for statement in SECONDARY_INDEX_SQL:
            connection.execute(statement)


def _rebuild_summaries(connection: sqlite3.Connection) -> None:
    conflict = connection.execute(
        """
        SELECT scene_id, gt_id, COUNT(DISTINCT object_id)
        FROM instances GROUP BY scene_id, gt_id
        HAVING COUNT(DISTINCT object_id) != 1 LIMIT 1
        """
    ).fetchone()
    if conflict is not None:
        raise ValueError(
            "A stable (scene_id, gt_id) track maps to multiple object IDs: "
            f"scene={conflict[0]}, gt_id={conflict[1]}"
        )

    with connection:
        connection.execute("DELETE FROM scene_tracks")
        connection.execute("DELETE FROM scenes")
        connection.execute("DELETE FROM objects")
        connection.execute(
            """
            INSERT INTO scene_tracks(
                scene_id, gt_id, object_id, frame_count, visible_frame_count,
                clean_visible_frame_count, corrupt_frame_count,
                min_view_id, max_view_id, mean_visib_fract, max_visib_fract
            )
            SELECT
                scene_id, gt_id, MIN(object_id), COUNT(*),
                SUM(px_count_visib > 0),
                COALESCE(SUM(px_count_visib > 0 AND depth_corrupt = 0), 0),
                COALESCE(SUM(depth_corrupt = 1), 0), MIN(view_id), MAX(view_id),
                AVG(visib_fract), MAX(visib_fract)
            FROM instances GROUP BY scene_id, gt_id
            """
        )
        connection.execute(
            """
            INSERT INTO scenes(
                scene_id, frame_count, track_count, object_count,
                min_view_id, max_view_id, is_complete
            )
            SELECT
                f.scene_id, COUNT(DISTINCT f.id), COUNT(DISTINCT st.gt_id),
                COUNT(DISTINCT st.object_id), MIN(f.view_id), MAX(f.view_id),
                COUNT(DISTINCT f.id) = ?
            FROM frames AS f
            LEFT JOIN scene_tracks AS st ON st.scene_id = f.scene_id
            GROUP BY f.scene_id
            """,
            (EXPECTED_VIEWS_PER_SCENE,),
        )
        connection.execute(
            """
            INSERT INTO objects(
                object_id, observation_count, frame_count,
                clean_visible_observation_count, scene_count, track_count
            )
            SELECT
                i.object_id, COUNT(*), COUNT(DISTINCT i.frame_id),
                COALESCE(SUM(i.px_count_visib > 0 AND i.depth_corrupt = 0), 0),
                COUNT(DISTINCT i.scene_id),
                (SELECT COUNT(*) FROM scene_tracks AS st
                 WHERE st.object_id = i.object_id)
            FROM instances AS i GROUP BY i.object_id
            """
        )


def _database_counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts = {}
    for table in ("shards", "frames", "instances", "scenes", "scene_tracks", "objects"):
        counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    counts["corrupt_instance_observations"] = int(
        connection.execute(
            "SELECT COUNT(*) FROM instances WHERE depth_corrupt = 1"
        ).fetchone()[0]
    )
    counts["unknown_corruption_instance_observations"] = int(
        connection.execute(
            "SELECT COUNT(*) FROM instances WHERE depth_corrupt IS NULL"
        ).fetchone()[0]
    )
    counts["complete_scenes"] = int(
        connection.execute("SELECT COUNT(*) FROM scenes WHERE is_complete = 1").fetchone()[0]
    )
    return counts


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _id_digest(ids: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in sorted(ids)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validation_ids(
    ids: Sequence[int],
    *,
    fraction: float,
    seed: int,
    entity_name: str,
) -> tuple[list[int], list[int]]:
    """Split sorted entity IDs by a stable seeded hash ranking."""

    ids = sorted({int(value) for value in ids})
    if not 0.0 <= float(fraction) < 1.0:
        raise ValueError(f"validation {entity_name} fraction must be in [0, 1)")
    validation_count = int(round(len(ids) * float(fraction)))
    if len(ids) > 1 and fraction > 0:
        validation_count = min(len(ids) - 1, max(1, validation_count))
    else:
        validation_count = 0

    def rank(value: int) -> tuple[bytes, int]:
        digest = hashlib.blake2b(
            f"{int(seed)}:{entity_name}:{int(value)}".encode("ascii"),
            digest_size=16,
        ).digest()
        return digest, int(value)

    validation = set(sorted(ids, key=rank)[:validation_count])
    train_ids = [value for value in ids if value not in validation]
    validation_ids = [value for value in ids if value in validation]
    return train_ids, validation_ids


def _prepare_split_manifest(
    connection: sqlite3.Connection,
    *,
    index_path: Path,
    split_path: Path,
    split_seed: int,
    validation_object_fraction: float,
    validation_scene_fraction: float,
) -> dict[str, Any]:
    object_ids = [
        int(row[0])
        for row in connection.execute("SELECT object_id FROM objects ORDER BY object_id")
    ]
    scene_ids = [
        int(row[0])
        for row in connection.execute("SELECT scene_id FROM scenes ORDER BY scene_id")
    ]
    train_objects, validation_objects = _validation_ids(
        object_ids,
        fraction=validation_object_fraction,
        seed=split_seed,
        entity_name="object",
    )
    train_scenes, validation_scenes = _validation_ids(
        scene_ids,
        fraction=validation_scene_fraction,
        seed=split_seed,
        entity_name="scene",
    )
    manifest = {
        "format": SPLIT_FORMAT,
        "dataset": "MegaPose-GSO",
        "generated_at_utc": _utc_now(),
        "seed": int(split_seed),
        "selection": {
            "algorithm": "blake2b_seeded_rank_v1",
            "validation_object_fraction": float(validation_object_fraction),
            "validation_scene_fraction": float(validation_scene_fraction),
        },
        "membership_policy": {
            "train": "train_object AND train_scene",
            "val": "val_object AND val_scene",
            "cross_partition_pairs": "excluded",
        },
        "membership_scope": {
            "object_ids": "sampled target objects",
            "scene_ids": "source rendered scenes",
        },
        "source": {
            "index_path": str(index_path),
            "index_schema_version": SCHEMA_VERSION,
            "object_count": len(object_ids),
            "scene_count": len(scene_ids),
            "object_ids_sha256": _id_digest(object_ids),
            "scene_ids_sha256": _id_digest(scene_ids),
        },
        "splits": {
            "train": {
                "object_ids": train_objects,
                "scene_ids": train_scenes,
            },
            "val": {
                "object_ids": validation_objects,
                "scene_ids": validation_scenes,
            },
        },
    }
    _write_summary(split_path, manifest)
    return manifest


def preprocess_dataset(
    data_root: str | Path,
    *,
    index_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    split_path: str | Path | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
    validation_object_fraction: float = DEFAULT_VALIDATION_OBJECT_FRACTION,
    validation_scene_fraction: float = DEFAULT_VALIDATION_SCENE_FRACTION,
    manifest_dir: str | Path = "GSO_broken_depth_maps",
    shard_glob: str = "shard-*.tar",
    allow_missing_corruption_manifests: bool = False,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Index every discovered shard and return the generated summary."""

    started = time.monotonic()
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    index_path = (
        data_root / "pi3_index" / "megapose_gso.sqlite"
        if index_path is None
        else Path(index_path).expanduser().resolve()
    )
    summary_path = (
        index_path.with_suffix(".summary.json")
        if summary_path is None
        else Path(summary_path).expanduser().resolve()
    )
    split_path = (
        index_path.with_suffix(".splits.json")
        if split_path is None
        else Path(split_path).expanduser().resolve()
    )
    if split_path in {index_path, summary_path}:
        raise ValueError("split_path must differ from index_path and summary_path")
    if not 0.0 <= float(validation_object_fraction) < 1.0:
        raise ValueError("validation_object_fraction must be in [0, 1)")
    if not 0.0 <= float(validation_scene_fraction) < 1.0:
        raise ValueError("validation_scene_fraction must be in [0, 1)")
    manifest_dir = Path(manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = data_root / manifest_dir

    shards = _discover_shards(data_root, shard_glob)
    if not shards:
        raise FileNotFoundError(f"No {shard_glob!r} files found under {data_root}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and index_path.exists():
        index_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{index_path}{suffix}")
            if sidecar.exists():
                sidecar.unlink()

    connection = _connect(index_path)
    processed: list[int] = []
    skipped: list[int] = []
    removed: list[int] = []
    try:
        stale = []
        for shard_number, shard_path in shards:
            stat = shard_path.stat()
            candidate = manifest_dir / MANIFEST_TEMPLATE.format(
                shard_number=shard_number
            )
            if candidate.is_file():
                candidate = candidate.resolve()
                manifest_size = int(candidate.stat().st_size)
                manifest_mtime = int(candidate.stat().st_mtime_ns)
                manifest_available = 1
            elif allow_missing_corruption_manifests:
                manifest_size = None
                manifest_mtime = None
                manifest_available = 0
            else:
                raise FileNotFoundError(
                    f"Missing corruption manifest for {shard_path.name}: {candidate}. "
                    "Use --allow-missing-corruption-manifests only if NULL/unknown "
                    "corruption status is acceptable."
                )
            row = connection.execute(
                """
                SELECT size_bytes, mtime_ns, manifest_available,
                       manifest_size_bytes, manifest_mtime_ns
                FROM shards WHERE shard_number = ?
                """,
                (shard_number,),
            ).fetchone()
            signature = (
                int(stat.st_size),
                int(stat.st_mtime_ns),
                manifest_available,
                manifest_size,
                manifest_mtime,
            )
            if row == signature:
                skipped.append(shard_number)
            else:
                stale.append((shard_number, shard_path, candidate if manifest_available else None))

        discovered_numbers = {shard_number for shard_number, _ in shards}
        removed = [
            int(row[0])
            for row in connection.execute("SELECT shard_number FROM shards")
            if int(row[0]) not in discovered_numbers
        ]
        complete_row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'index_complete'"
        ).fetchone()
        needs_finalize = bool(stale or removed or complete_row != ("1",))

        if needs_finalize:
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                    ("index_complete", "0"),
                )
            _drop_secondary_indexes(connection)
            if removed:
                with connection:
                    connection.executemany(
                        "DELETE FROM shards WHERE shard_number = ?",
                        ((shard_number,) for shard_number in removed),
                    )
        for position, (shard_number, shard_path, manifest_path) in enumerate(stale, start=1):
            if manifest_path is not None:
                manifest = _parse_corruption_manifest(manifest_path)
            else:
                manifest = None
            print(
                f"[{position}/{len(stale)}] indexing {shard_path.name}",
                flush=True,
            )
            frame_rows, instance_count = _read_shard(shard_path, manifest)
            _replace_shard(
                connection,
                data_root=data_root,
                shard_number=shard_number,
                shard_path=shard_path,
                manifest_path=manifest_path,
                manifest_entries=None if manifest is None else len(manifest),
                frame_rows=frame_rows,
                instance_count=instance_count,
            )
            processed.append(shard_number)

        if needs_finalize:
            print("building aggregate scene/object/track tables and query indexes", flush=True)
            _rebuild_summaries(connection)
            _create_secondary_indexes(connection)
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                    ("index_complete", "1"),
                )
        else:
            print("all discovered shards and manifests are unchanged", flush=True)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        counts = _database_counts(connection)
        indexed_numbers = [
            int(row[0])
            for row in connection.execute("SELECT shard_number FROM shards ORDER BY shard_number")
        ]
        with connection:
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("data_root_at_preprocess_time", str(data_root)),
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("updated_at_utc", _utc_now()),
            )
        split_manifest = _prepare_split_manifest(
            connection,
            index_path=index_path,
            split_path=split_path,
            split_seed=int(split_seed),
            validation_object_fraction=float(validation_object_fraction),
            validation_scene_fraction=float(validation_scene_fraction),
        )
        split_counts = {
            split_name: {
                "objects": len(values["object_ids"]),
                "scenes": len(values["scene_ids"]),
            }
            for split_name, values in split_manifest["splits"].items()
        }
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _utc_now(),
            "data_root": str(data_root),
            "index_path": str(index_path),
            "split_path": str(split_path),
            "split_seed": int(split_seed),
            "split_counts": split_counts,
            "partial_dataset": len(indexed_numbers) < 1040,
            "indexed_shard_numbers": indexed_numbers,
            "processed_shard_numbers_this_run": processed,
            "skipped_unchanged_shard_numbers_this_run": skipped,
            "removed_missing_shard_numbers_this_run": sorted(removed),
            "counts": counts,
            "units": {
                "translation": "metre",
                "depth": "raw_depth * depth_scale_raw * 0.001 metres",
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        _write_summary(summary_path, summary)
        return summary
    finally:
        connection.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="MegaPose-GSO-fixed directory")
    parser.add_argument("--index-path", help="Default: DATA_ROOT/pi3_index/megapose_gso.sqlite")
    parser.add_argument("--summary-path", help="Default: INDEX_PATH with .summary.json suffix")
    parser.add_argument("--split-path", help="Default: INDEX_PATH with .splits.json suffix")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--validation-object-fraction",
        type=float,
        default=DEFAULT_VALIDATION_OBJECT_FRACTION,
    )
    parser.add_argument(
        "--validation-scene-fraction",
        type=float,
        default=DEFAULT_VALIDATION_SCENE_FRACTION,
    )
    parser.add_argument(
        "--manifest-dir",
        default="GSO_broken_depth_maps",
        help="Absolute path or path relative to DATA_ROOT",
    )
    parser.add_argument("--shard-glob", default="shard-*.tar")
    parser.add_argument(
        "--allow-missing-corruption-manifests",
        action="store_true",
        help="Index missing-manifest instances with depth_corrupt=NULL",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Delete and recreate only the requested index file"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = preprocess_dataset(
            args.data_root,
            index_path=args.index_path,
            summary_path=args.summary_path,
            split_path=args.split_path,
            split_seed=args.split_seed,
            validation_object_fraction=args.validation_object_fraction,
            validation_scene_fraction=args.validation_scene_fraction,
            manifest_dir=args.manifest_dir,
            shard_glob=args.shard_glob,
            allow_missing_corruption_manifests=args.allow_missing_corruption_manifests,
            rebuild=args.rebuild,
        )
    except (FileNotFoundError, ValueError, RuntimeError, sqlite3.Error, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
