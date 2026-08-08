#!/usr/bin/env python3
"""Validate and index a Pi3 BOP-style object reference bank once.

The resulting immutable SQLite catalogue is intentionally image-format neutral:
training workers query compact pose/path rows and decode only selected payloads
instead of reparsing every per-object JSON manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
from PIL import Image


INDEX_FORMAT = "pi3_object_reference_index_v1"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _mapping_ids(bank: dict[str, Any], explicit_path: Path | None) -> dict[int, str]:
    raw_path = explicit_path or Path(str(bank.get("mapping_path", "")))
    if not raw_path.is_file():
        return {}
    payload = _load_json(raw_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list mapping in {raw_path}")
    mapping = {int(row["obj_id"]): str(row["gso_id"]) for row in payload}
    if len(mapping) != len(payload):
        raise ValueError(f"Duplicate object IDs in {raw_path}")
    return mapping


def _required_object_ids(
    mapping: dict[int, str], explicit_path: Path | None
) -> set[int]:
    if explicit_path is None:
        return set(mapping)
    raw = _load_json(explicit_path.expanduser().resolve())
    if isinstance(raw, dict):
        values = {int(value) for value in raw}
    elif isinstance(raw, list):
        values = {
            int(row["obj_id"]) if isinstance(row, dict) else int(row)
            for row in raw
        }
    else:
        raise ValueError(
            f"Expected models_info dict or object-ID list in {explicit_path}"
        )
    if not values:
        raise ValueError(f"Required object set is empty: {explicit_path}")
    absent = sorted(values.difference(mapping))
    if mapping and absent:
        raise ValueError(
            f"Required object IDs are absent from mapping: {absent[:20]}"
        )
    return values


def _object_id_digest(values: set[int]) -> str:
    payload = ",".join(str(value) for value in sorted(values)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE objects (
            object_id INTEGER PRIMARY KEY,
            external_id TEXT NOT NULL,
            view_count INTEGER NOT NULL,
            manifest_relpath TEXT NOT NULL
        );
        CREATE TABLE views (
            object_id INTEGER NOT NULL,
            view_id INTEGER NOT NULL,
            coverage_view_id INTEGER NOT NULL,
            T_C_O_f32 BLOB NOT NULL,
            K_f32 BLOB NOT NULL,
            depth_scale REAL NOT NULL,
            rgb_relpath TEXT NOT NULL,
            depth_relpath TEXT NOT NULL,
            mask_relpath TEXT NOT NULL,
            mask_visib_relpath TEXT NOT NULL,
            bbox_obj_json TEXT NOT NULL,
            bbox_visib_json TEXT NOT NULL,
            px_count_all INTEGER NOT NULL,
            px_count_visib INTEGER NOT NULL,
            PRIMARY KEY (object_id, view_id),
            FOREIGN KEY (object_id) REFERENCES objects(object_id)
        ) WITHOUT ROWID;
        CREATE INDEX views_coverage
            ON views(object_id, coverage_view_id);
        """
    )
    return connection


def _check_image(path: Path, expected_size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        image.load()
        if image.size != expected_size:
            raise ValueError(
                f"Unexpected image size for {path}: {image.size}, expected {expected_size}"
            )


def _check_image_job(args: tuple[Path, tuple[int, int]]) -> None:
    _check_image(*args)


def build_reference_index(
    bank_root: Path,
    output_path: Path,
    *,
    namespace: str,
    mapping_path: Path | None = None,
    required_object_ids_path: Path | None = None,
    allow_incomplete: bool = False,
    verify_images: str = "sample",
    verify_workers: int = 1,
) -> dict[str, int]:
    bank_root = bank_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    bank = _load_json(bank_root / "reference_bank.json")
    fingerprint = str(bank.get("config_fingerprint", ""))
    if not fingerprint:
        raise ValueError("Reference bank has no config_fingerprint")
    width, height = (int(value) for value in bank["resolution_wh"])
    expected_views = int(bank["num_views"])
    mapping = _mapping_ids(bank, mapping_path)
    required_ids = _required_object_ids(mapping, required_object_ids_path)
    verify_workers = int(verify_workers)
    if verify_workers <= 0:
        raise ValueError("verify_workers must be positive")

    object_dirs = sorted(
        path for path in bank_root.iterdir() if path.is_dir() and path.name.isdigit()
    )
    complete: list[tuple[Path, dict[str, Any]]] = []
    for object_dir in object_dirs:
        manifest_path = object_dir / "reference_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _load_json(manifest_path)
        if manifest.get("complete"):
            complete.append((object_dir, manifest))

    complete_ids = {int(manifest["object_id"]) for _, manifest in complete}
    if required_ids:
        missing = sorted(required_ids.difference(complete_ids))
        extra = sorted(complete_ids.difference(required_ids))
        if extra:
            raise ValueError(
                f"Reference bank contains objects outside the required set: {extra[:20]}"
            )
        if missing and not allow_incomplete:
            raise ValueError(
                f"Reference bank is incomplete: {len(missing)} required objects are missing; "
                f"first={missing[:20]}"
            )
    if not complete:
        raise ValueError(f"No complete objects found in {bank_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    connection = _open_database(temporary)
    executor = (
        ThreadPoolExecutor(max_workers=verify_workers)
        if verify_workers > 1 and verify_images != "none"
        else None
    )
    view_total = 0
    decoded = 0
    try:
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("format", INDEX_FORMAT),
                ("index_complete", "0"),
                ("namespace", str(namespace)),
                ("bank_root", str(bank_root)),
                ("bank_fingerprint", fingerprint),
                ("bank_format", str(bank.get("format", ""))),
                ("view_order", str(bank.get("view_order", "coverage"))),
                ("resolution_width", str(width)),
                ("resolution_height", str(height)),
                ("expected_views_per_object", str(expected_views)),
                ("required_object_count", str(len(required_ids))),
                ("required_object_ids_sha256", _object_id_digest(required_ids)),
            ],
        )
        for object_position, (object_dir, manifest) in enumerate(complete):
            object_id = int(manifest["object_id"])
            if object_dir.name != f"{object_id:06d}":
                raise ValueError(f"Object directory/manifest mismatch: {object_dir}")
            if manifest.get("config_fingerprint") != fingerprint:
                raise ValueError(f"Fingerprint mismatch in {object_dir}")
            poses = manifest.get("poses")
            if not isinstance(poses, list) or len(poses) != expected_views:
                raise ValueError(
                    f"Expected {expected_views} poses in {object_dir}, got "
                    f"{0 if not isinstance(poses, list) else len(poses)}"
                )
            scene_camera = _load_json(object_dir / "scene_camera.json")
            scene_gt = _load_json(object_dir / "scene_gt.json")
            scene_gt_info = _load_json(object_dir / "scene_gt_info.json")
            external_id = str(manifest.get("gso_id", mapping.get(object_id, object_id)))
            if mapping and external_id != mapping[object_id]:
                raise ValueError(f"External ID mismatch for object {object_id}")
            connection.execute(
                "INSERT INTO objects VALUES (?, ?, ?, ?)",
                (
                    object_id,
                    external_id,
                    expected_views,
                    str((object_dir / "reference_manifest.json").relative_to(bank_root)),
                ),
            )
            rows = []
            image_checks: list[tuple[Path, tuple[int, int]]] = []
            for expected_view_id, pose in enumerate(poses):
                view_id = int(pose["view_id"])
                if view_id != expected_view_id:
                    raise ValueError(
                        f"Non-contiguous view IDs in {object_dir}: {view_id} at "
                        f"position {expected_view_id}"
                    )
                key = str(view_id)
                camera = scene_camera[key]
                gt = scene_gt[key][0]
                info = scene_gt_info[key][0]
                if int(gt["obj_id"]) != object_id:
                    raise ValueError(f"scene_gt object mismatch in {object_dir}/{key}")
                T_manifest = np.asarray(pose["T_C_O"], dtype=np.float64).reshape(4, 4)
                T_bop = np.eye(4, dtype=np.float64)
                T_bop[:3, :3] = np.asarray(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                T_bop[:3, 3] = np.asarray(gt["cam_t_m2c"], dtype=np.float64) * 0.001
                if not np.allclose(T_manifest, T_bop, rtol=1e-6, atol=1e-7):
                    raise ValueError(f"Pose mismatch in {object_dir}/{key}")
                K = np.asarray(camera["cam_K"], dtype=np.float32).reshape(3, 3)
                stem = f"{view_id:06d}"
                paths = {
                    "rgb": object_dir / "rgb" / f"{stem}.png",
                    "depth": object_dir / "depth" / f"{stem}.png",
                    "mask": object_dir / "mask" / f"{stem}_000000.png",
                    "mask_visib": object_dir / "mask_visib" / f"{stem}_000000.png",
                }
                missing_paths = [str(path) for path in paths.values() if not path.is_file()]
                if missing_paths:
                    raise FileNotFoundError(missing_paths[0])
                should_decode = verify_images == "all" or (
                    verify_images == "sample"
                    and view_id in {0, expected_views // 2, expected_views - 1}
                )
                if should_decode:
                    for path in paths.values():
                        image_checks.append((path, (width, height)))
                rows.append(
                    (
                        object_id,
                        view_id,
                        int(pose.get("coverage_view_id", view_id)),
                        np.asarray(T_manifest, dtype="<f4").tobytes(),
                        np.asarray(K, dtype="<f4").tobytes(),
                        float(camera["depth_scale"]),
                        *(str(paths[name].relative_to(bank_root)) for name in paths),
                        json.dumps(info["bbox_obj"], separators=(",", ":")),
                        json.dumps(info["bbox_visib"], separators=(",", ":")),
                        int(info["px_count_all"]),
                        int(info["px_count_visib"]),
                    )
                )
            if executor is None:
                for image_check in image_checks:
                    _check_image_job(image_check)
            else:
                # Bound queued work to one object (at most views * 4 files),
                # keeping memory flat while overlapping independent PNG reads.
                tuple(executor.map(_check_image_job, image_checks))
            decoded += len(image_checks) // 4
            connection.executemany(
                """
                INSERT INTO views(
                    object_id, view_id, coverage_view_id, T_C_O_f32, K_f32,
                    depth_scale, rgb_relpath, depth_relpath, mask_relpath,
                    mask_visib_relpath, bbox_obj_json, bbox_visib_json,
                    px_count_all, px_count_visib
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            view_total += len(rows)
            if (object_position + 1) % 50 == 0:
                print(f"Indexed {object_position + 1}/{len(complete)} objects")
        connection.execute(
            "UPDATE metadata SET value='1' WHERE key='index_complete'"
        )
        connection.execute(
            "INSERT INTO metadata VALUES ('object_count', ?)", (str(len(complete)),)
        )
        connection.execute(
            "INSERT INTO metadata VALUES ('view_count', ?)", (str(view_total),)
        )
        connection.commit()
    except Exception:
        connection.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if temporary.exists():
            temporary.unlink()
        raise
    connection.close()
    if executor is not None:
        executor.shutdown(wait=True)
    os.replace(temporary, output_path)
    return {"objects": len(complete), "views": view_total, "decoded_views": decoded}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--namespace", default="object")
    parser.add_argument("--mapping-path", type=Path)
    parser.add_argument(
        "--required-object-ids-path",
        type=Path,
        help=(
            "models_info.json dict or object-ID list defining required coverage; "
            "defaults to every ID in --mapping-path"
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--verify-images", choices=("none", "sample", "all"), default="sample"
    )
    parser.add_argument(
        "--verify-workers",
        type=int,
        default=1,
        help="Parallel PNG decoders; database writes remain single-threaded",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bank_root = args.bank_root.expanduser().resolve()
    output = (
        bank_root / "pi3_index" / "references.sqlite"
        if args.output is None
        else args.output
    )
    result = build_reference_index(
        bank_root,
        output,
        namespace=args.namespace,
        mapping_path=args.mapping_path,
        required_object_ids_path=args.required_object_ids_path,
        allow_incomplete=args.allow_incomplete,
        verify_images=args.verify_images,
        verify_workers=args.verify_workers,
    )
    print(
        f"Wrote {output}: objects={result['objects']}, views={result['views']}, "
        f"decoded_views={result['decoded_views']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
