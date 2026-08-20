#!/usr/bin/env python3
"""Pack a Pi3 BOP-style object reference bank into tar shards and index it.

A freshly rendered bank (see ``megapose_gso_references.py``) stores one loose
PNG file per RGB/depth/mask/mask_visib payload per view -- for the default
512-view/946-object MegaPose-GSO bank that is roughly 1.9 million files for
41 GB of data, which is unfriendly to inode-quota- or backup-limited cluster
filesystems. This script packs every payload into a handful of uncompressed
tar shards (image-content unchanged, byte-for-byte) and writes a compact
SQLite index that stores, per view, the shard and exact byte offset/size of
each payload. A loader can then ``os.pread`` a payload directly with no
per-sample ``open()`` and no tar member scan, using the same technique
``datasets/preprocess/megapose_gso.py`` already uses for the (separately
downloaded, pre-shipped-as-tar) MegaPose-GSO scene corpus.

This always produces the packed ``pi3_object_reference_index_v2`` format.
There is no longer a loose-file / relative-path index mode: rerun this
script to migrate an older ``pi3_object_reference_index_v1`` bank.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tarfile
from typing import Any

import numpy as np
from PIL import Image


INDEX_FORMAT = "pi3_object_reference_index_v2"
DEFAULT_SHARD_TARGET_BYTES = 512 * 1024 * 1024
_SHARD_NAME_RE = r"^shard-(\d{6})\.tar$"


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
        CREATE TABLE shards (
            id INTEGER PRIMARY KEY,
            shard_number INTEGER NOT NULL UNIQUE,
            relative_path TEXT NOT NULL UNIQUE,
            size_bytes INTEGER NOT NULL,
            object_count INTEGER NOT NULL,
            view_count INTEGER NOT NULL
        );
        CREATE TABLE objects (
            object_id INTEGER PRIMARY KEY,
            external_id TEXT NOT NULL,
            view_count INTEGER NOT NULL
        );
        CREATE TABLE views (
            object_id INTEGER NOT NULL,
            view_id INTEGER NOT NULL,
            coverage_view_id INTEGER NOT NULL,
            T_C_O_f32 BLOB NOT NULL,
            K_f32 BLOB NOT NULL,
            depth_scale REAL NOT NULL,
            shard_id INTEGER NOT NULL REFERENCES shards(id),
            rgb_offset INTEGER NOT NULL,
            rgb_size INTEGER NOT NULL,
            depth_offset INTEGER NOT NULL,
            depth_size INTEGER NOT NULL,
            mask_offset INTEGER NOT NULL,
            mask_size INTEGER NOT NULL,
            mask_visib_offset INTEGER NOT NULL,
            mask_visib_size INTEGER NOT NULL,
            bbox_obj_json TEXT NOT NULL,
            bbox_visib_json TEXT NOT NULL,
            px_count_all INTEGER NOT NULL,
            px_count_visib INTEGER NOT NULL,
            PRIMARY KEY (object_id, view_id),
            FOREIGN KEY (object_id) REFERENCES objects(object_id)
        ) WITHOUT ROWID;
        CREATE INDEX views_coverage
            ON views(object_id, coverage_view_id);
        CREATE INDEX views_shard
            ON views(shard_id);
        """
    )
    return connection


def _check_image_bytes(data: bytes, expected_size: tuple[int, int], where: str) -> None:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.size != expected_size:
            raise ValueError(
                f"Unexpected image size for {where}: {image.size}, expected {expected_size}"
            )


def _check_image_job(args: tuple[bytes, tuple[int, int], str]) -> None:
    _check_image_bytes(*args)


class _ShardWriter:
    """Writes complete, uncompressed tar shards with object-atomic boundaries.

    Objects are never split across two shards: a shard is only closed between
    objects, once its running byte total reaches ``target_bytes``. Each shard
    is written to a process-specific temporary name and only renamed into
    place once fully written, so a crash mid-shard never leaves a half-written
    file at its final name.
    """

    def __init__(self, shard_root: Path, *, target_bytes: int):
        self.shard_root = shard_root
        self.target_bytes = int(target_bytes)
        self.shard_number = 0
        self.finalized: list[tuple[int, Path, int, int, int]] = []
        # (shard_number, final_path, size_bytes, object_count, view_count)
        self._tar: tarfile.TarFile | None = None
        self._temp_path: Path | None = None
        self._final_path: Path | None = None
        self._bytes = 0
        self._object_count = 0
        self._view_count = 0
        self._open_shard()

    def _open_shard(self) -> None:
        self._final_path = self.shard_root / f"shard-{self.shard_number:06d}.tar"
        self._temp_path = self._final_path.with_name(
            self._final_path.name + f".tmp.{os.getpid()}"
        )
        self._tar = tarfile.open(self._temp_path, "w")
        self._bytes = 0
        self._object_count = 0
        self._view_count = 0

    def add_payload(self, path: Path, arcname: str) -> int:
        with path.open("rb") as stream:
            tarinfo = self._tar.gettarinfo(arcname=arcname, fileobj=stream)
            self._tar.addfile(tarinfo, stream)
        self._bytes += int(tarinfo.size)
        return int(tarinfo.size)

    def note_object_written(self, view_count: int) -> None:
        self._object_count += 1
        self._view_count += view_count

    @property
    def current_bytes(self) -> int:
        return self._bytes

    def should_close(self) -> bool:
        return self._bytes >= self.target_bytes

    def close_shard(self) -> tuple[int, Path]:
        """Finalize the current shard and return (shard_number, final_path)."""

        self._tar.close()
        os.replace(self._temp_path, self._final_path)
        size_bytes = self._final_path.stat().st_size
        result = (self.shard_number, self._final_path)
        self.finalized.append(
            (self.shard_number, self._final_path, size_bytes, self._object_count, self._view_count)
        )
        self.shard_number += 1
        self._open_shard()
        return result

    def discard_empty_shard(self) -> None:
        if self._object_count != 0:
            raise RuntimeError("Refusing to discard a non-empty shard")
        self._tar.close()
        if self._temp_path.exists():
            self._temp_path.unlink()

    def abort(self) -> None:
        try:
            if self._tar is not None:
                self._tar.close()
        except Exception:
            pass
        if self._temp_path is not None and self._temp_path.exists():
            self._temp_path.unlink()
        for _, final_path, *_ in self.finalized:
            if final_path.exists():
                final_path.unlink()


def _read_shard_offsets(shard_path: Path) -> dict[str, tuple[int, int]]:
    with tarfile.open(shard_path, "r") as tar:
        return {
            member.name: (int(member.offset_data), int(member.size))
            for member in tar.getmembers()
        }


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
    shard_root: Path | None = None,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
    delete_source_after_verify: bool = False,
) -> dict[str, int]:
    bank_root = bank_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    shard_root = (
        bank_root if shard_root is None else Path(shard_root).expanduser().resolve()
    )
    shard_target_bytes = int(shard_target_bytes)
    if shard_target_bytes <= 0:
        raise ValueError("shard_target_bytes must be positive")
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

    shard_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_db = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    if temporary_db.exists():
        temporary_db.unlink()
    connection = _open_database(temporary_db)
    executor = (
        ThreadPoolExecutor(max_workers=verify_workers)
        if verify_workers > 1 and verify_images != "none"
        else None
    )
    writer = _ShardWriter(shard_root, target_bytes=shard_target_bytes)
    view_total = 0
    decoded = 0
    object_total = 0
    reclaimed_bytes = 0
    deleted_objects: list[Path] = []
    # Objects packed into the shard currently being written, awaiting their
    # true byte offsets once that shard is closed and read back.
    pending: list[tuple[Path, int, str, list[dict[str, Any]]]] = []

    def _flush_pending_shard() -> None:
        nonlocal view_total, decoded
        if not pending:
            return
        shard_number, shard_path = writer.close_shard()
        _, _, size_bytes, object_count, view_count = writer.finalized[-1]
        offsets = _read_shard_offsets(shard_path)
        shard_relative = str(shard_path.relative_to(bank_root)) if _is_within(
            shard_path, bank_root
        ) else str(shard_path)
        cursor = connection.execute(
            "INSERT INTO shards(shard_number, relative_path, size_bytes, "
            "object_count, view_count) VALUES (?, ?, ?, ?, ?)",
            (shard_number, shard_relative, size_bytes, object_count, view_count),
        )
        shard_id = int(cursor.lastrowid)
        for object_dir, object_id, external_id, pending_views in pending:
            image_checks: list[tuple[bytes, tuple[int, int], str]] = []
            rows = []
            for view in pending_views:
                arcnames = view["arcnames"]
                payload_ranges = {}
                for kind, arcname in arcnames.items():
                    payload_ranges[kind] = offsets[arcname]
                view_id = view["view_id"]
                should_decode = verify_images == "all" or (
                    verify_images == "sample"
                    and view_id in {0, expected_views // 2, expected_views - 1}
                )
                if should_decode:
                    for kind, (offset, size) in payload_ranges.items():
                        with shard_path.open("rb") as stream:
                            stream.seek(offset)
                            data = stream.read(size)
                        image_checks.append(
                            (data, (width, height), f"{shard_path}:{arcnames[kind]}")
                        )
                rows.append(
                    (
                        object_id,
                        view_id,
                        view["coverage_view_id"],
                        view["T_C_O_bytes"],
                        view["K_bytes"],
                        view["depth_scale"],
                        shard_id,
                        *payload_ranges["rgb"],
                        *payload_ranges["depth"],
                        *payload_ranges["mask"],
                        *payload_ranges["mask_visib"],
                        view["bbox_obj_json"],
                        view["bbox_visib_json"],
                        view["px_count_all"],
                        view["px_count_visib"],
                    )
                )
            if executor is None:
                for image_check in image_checks:
                    _check_image_job(image_check)
            else:
                tuple(executor.map(_check_image_job, image_checks))
            decoded += len(image_checks) // 4
            connection.executemany(
                """
                INSERT INTO views(
                    object_id, view_id, coverage_view_id, T_C_O_f32, K_f32,
                    depth_scale, shard_id, rgb_offset, rgb_size, depth_offset,
                    depth_size, mask_offset, mask_size, mask_visib_offset,
                    mask_visib_size, bbox_obj_json, bbox_visib_json,
                    px_count_all, px_count_visib
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            view_total += len(rows)
        pending.clear()

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
                ("shard_target_bytes", str(shard_target_bytes)),
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
                "INSERT INTO objects VALUES (?, ?, ?)",
                (object_id, external_id, expected_views),
            )
            object_total += 1
            pending_views = []
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
                arcnames = {
                    "rgb": f"{object_id:06d}/rgb/{stem}.png",
                    "depth": f"{object_id:06d}/depth/{stem}.png",
                    "mask": f"{object_id:06d}/mask/{stem}_000000.png",
                    "mask_visib": f"{object_id:06d}/mask_visib/{stem}_000000.png",
                }
                for kind, path in paths.items():
                    writer.add_payload(path, arcnames[kind])
                pending_views.append(
                    {
                        "view_id": view_id,
                        "coverage_view_id": int(pose.get("coverage_view_id", view_id)),
                        "T_C_O_bytes": np.asarray(T_manifest, dtype="<f4").tobytes(),
                        "K_bytes": np.asarray(K, dtype="<f4").tobytes(),
                        "depth_scale": float(camera["depth_scale"]),
                        "arcnames": arcnames,
                        "bbox_obj_json": json.dumps(info["bbox_obj"], separators=(",", ":")),
                        "bbox_visib_json": json.dumps(info["bbox_visib"], separators=(",", ":")),
                        "px_count_all": int(info["px_count_all"]),
                        "px_count_visib": int(info["px_count_visib"]),
                    }
                )
            writer.note_object_written(len(pending_views))
            pending.append((object_dir, object_id, external_id, pending_views))
            if writer.should_close():
                _flush_pending_shard()
            if (object_position + 1) % 50 == 0:
                print(f"Packed {object_position + 1}/{len(complete)} objects", flush=True)
        _flush_pending_shard()
        # The writer always leaves one open (possibly empty) shard behind
        # after the loop; discard it rather than shipping a zero-object shard.
        writer.discard_empty_shard()

        # Remove stale shards from a previous, larger run at this shard_root.
        written_numbers = {number for number, *_ in writer.finalized}
        for existing in sorted(shard_root.glob("shard-*.tar")):
            match = _shard_number_from_name(existing.name)
            if match is not None and match not in written_numbers:
                existing.unlink()

        connection.execute(
            "UPDATE metadata SET value='1' WHERE key='index_complete'"
        )
        connection.execute(
            "INSERT INTO metadata VALUES ('object_count', ?)", (str(object_total),)
        )
        connection.execute(
            "INSERT INTO metadata VALUES ('view_count', ?)", (str(view_total),)
        )
        connection.execute(
            "INSERT INTO metadata VALUES ('shard_count', ?)", (str(len(writer.finalized)),)
        )
        connection.commit()
    except Exception:
        connection.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        writer.abort()
        if temporary_db.exists():
            temporary_db.unlink()
        raise
    connection.close()
    if executor is not None:
        executor.shutdown(wait=True)
    os.replace(temporary_db, output_path)

    if delete_source_after_verify:
        for object_dir, _ in complete:
            reclaimed_bytes += _delete_object_source(object_dir)
            deleted_objects.append(object_dir)

    return {
        "objects": object_total,
        "views": view_total,
        "decoded_views": decoded,
        "shards": len(writer.finalized),
        "deleted_objects": len(deleted_objects),
        "reclaimed_bytes": reclaimed_bytes,
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _shard_number_from_name(name: str) -> int | None:
    if not (name.startswith("shard-") and name.endswith(".tar")):
        return None
    digits = name[len("shard-") : -len(".tar")]
    return int(digits) if digits.isdigit() else None


def _delete_object_source(object_dir: Path) -> int:
    reclaimed = 0
    for sub in ("rgb", "depth", "mask", "mask_visib"):
        sub_dir = object_dir / sub
        if not sub_dir.is_dir():
            continue
        for file in sub_dir.iterdir():
            reclaimed += file.stat().st_size
            file.unlink()
        sub_dir.rmdir()
    for name in (
        "reference_manifest.json",
        "scene_camera.json",
        "scene_gt.json",
        "scene_gt_info.json",
    ):
        file = object_dir / name
        if file.is_file():
            reclaimed += file.stat().st_size
            file.unlink()
    try:
        object_dir.rmdir()
    except OSError:
        pass
    return reclaimed


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
        help="Parallel PNG decoders; database and shard writes remain single-threaded",
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        help="Directory to write shard-NNNNNN.tar into; defaults to --bank-root",
    )
    parser.add_argument(
        "--shard-target-bytes",
        type=int,
        default=DEFAULT_SHARD_TARGET_BYTES,
        help="Approximate shard size; a shard only closes between objects "
        f"(default: {DEFAULT_SHARD_TARGET_BYTES} bytes = 512 MiB)",
    )
    parser.add_argument(
        "--delete-source-after-verify",
        action="store_true",
        help="After the index is written successfully, delete each packed "
        "object's loose rgb/depth/mask/mask_visib files and per-object JSON "
        "manifests (reference_bank.json at the bank root is kept). Only the "
        "packed shards and this index are the source of truth afterward.",
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
        shard_root=args.shard_root,
        shard_target_bytes=args.shard_target_bytes,
        delete_source_after_verify=args.delete_source_after_verify,
    )
    print(
        f"Wrote {output}: objects={result['objects']}, views={result['views']}, "
        f"decoded_views={result['decoded_views']}, shards={result['shards']}"
    )
    if result["deleted_objects"]:
        reclaimed_gib = result["reclaimed_bytes"] / (1024 ** 3)
        print(
            f"Deleted loose source files for {result['deleted_objects']} objects, "
            f"reclaiming {reclaimed_gib:.2f} GiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
