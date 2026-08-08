#!/usr/bin/env python3
"""Download and selectively extract MegaPose's prepared GSO mesh archive.

The default path keeps only the textured ``models_normalized`` representation
needed for rendering.  Optional flags add the compact BOP PLY and point-cloud
representations needed by the Pi3 model catalogue.  The script is
dependency-free and does not modify the MegaPose scene shards or their index.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import sys
import time
from typing import Iterable
import urllib.error
import urllib.request
import zipfile


DEFAULT_ARCHIVE_URL = (
    "https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/"
    "tars/google_scanned_objects.zip"
)
DEFAULT_MAPPING_URL = (
    "https://huggingface.co/datasets/bop-benchmark/megapose/resolve/"
    "main/MegaPose-GSO/gso_models.json"
)
DEFAULT_ARCHIVE_BYTES = 24_977_819_586
ARCHIVE_NAME = "google_scanned_objects.zip"
MAPPING_NAME = "gso_models.json"
COPY_CHUNK_BYTES = 8 * 1024 * 1024
FREE_SPACE_MARGIN_BYTES = 2 * 1024**3


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _response_total_bytes(response, existing_bytes: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        if total.isdigit():
            return int(total)
    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return existing_bytes + int(content_length)
    return None


def _require_free_space(path: Path, required_bytes: int, purpose: str) -> None:
    free = shutil.disk_usage(path).free
    required_with_margin = int(required_bytes) + FREE_SPACE_MARGIN_BYTES
    if free < required_with_margin:
        raise OSError(
            f"Insufficient free space for {purpose}: need approximately "
            f"{human_bytes(required_with_margin)}, have {human_bytes(free)} at {path}"
        )


def download_file(
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None = None,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download to a temporary file and atomically publish on success.

    A partial file is resumed when the server honors HTTP Range.  If it does
    not, downloading restarts cleanly instead of appending duplicate bytes.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if destination.exists() and not force:
        size = destination.stat().st_size
        if expected_bytes is not None and size != expected_bytes:
            raise ValueError(
                f"Existing {destination} has size {size}, expected {expected_bytes}; "
                "use --force-download to replace it"
            )
        print(f"Using existing {destination} ({human_bytes(size)})")
        return destination

    if force:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Pi3-MegaPose-GSO-downloader/1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot download {url}: {exc}") from exc

    with response:
        status = getattr(response, "status", response.getcode())
        append = bool(existing and status == 206)
        if existing and not append:
            print(
                f"Server did not honor resume for {destination.name}; "
                f"restarting the {human_bytes(existing)} partial download"
            )
            existing = 0

        total = _response_total_bytes(response, existing)
        if expected_bytes is not None and total is not None and total != expected_bytes:
            raise ValueError(
                f"Remote size for {url} is {total}, expected {expected_bytes}. "
                "Use --expected-archive-bytes 0 only for a verified alternate archive."
            )
        if total is not None:
            _require_free_space(
                destination.parent,
                max(0, total - existing),
                f"downloading {destination.name}",
            )

        mode = "ab" if append else "wb"
        downloaded = existing
        last_report = time.monotonic()
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                stream.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 10.0:
                    suffix = f"/{human_bytes(total)}" if total is not None else ""
                    print(f"Downloaded {human_bytes(downloaded)}{suffix}")
                    last_report = now

    size = partial.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise IOError(
            f"Incomplete download at {partial}: got {size} bytes, "
            f"expected {expected_bytes}"
        )
    if total is not None and size != total:
        raise IOError(
            f"Incomplete download at {partial}: got {size} bytes, expected {total}"
        )
    os.replace(partial, destination)
    print(f"Downloaded {destination} ({human_bytes(size)})")
    return destination


def load_object_mapping(path: Path) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {path}")

    mapping: dict[int, str] = {}
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or "obj_id" not in row or "gso_id" not in row:
            raise ValueError(f"Invalid GSO mapping row in {path}: {row!r}")
        object_id = int(row["obj_id"])
        gso_id = str(row["gso_id"])
        if object_id in mapping or gso_id in names:
            raise ValueError(f"Duplicate GSO mapping in {path}: {row!r}")
        mapping[object_id] = gso_id
        names.add(gso_id)
    return mapping


def load_index_object_ids(index_path: Path) -> set[int]:
    index_path = Path(index_path).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"MegaPose-GSO index does not exist: {index_path}")
    uri = f"file:{index_path}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        try:
            rows = connection.execute(
                "SELECT object_id FROM objects ORDER BY object_id"
            ).fetchall()
        except sqlite3.Error as exc:
            raise ValueError(f"Invalid MegaPose-GSO index {index_path}: {exc}") from exc
    object_ids = {int(row[0]) for row in rows}
    if not object_ids:
        raise ValueError(f"No object IDs found in {index_path}")
    return object_ids


def _safe_relative_path(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    return Path(*path.parts)


def select_archive_members(
    archive: zipfile.ZipFile,
    *,
    extract_mode: str,
    selected_gso_ids: set[str] | None,
    include_bop_meshes: bool = False,
    include_pointclouds: bool = False,
) -> list[zipfile.ZipInfo]:
    if extract_mode not in {"normalized", "all"}:
        raise ValueError(f"Unsupported extraction mode: {extract_mode}")

    selected: list[zipfile.ZipInfo] = []
    expected_leaf = {
        "models_normalized": ("meshes", "model.obj"),
    }
    if include_bop_meshes:
        expected_leaf["models_bop-renderer_scale=0.1"] = ("meshes", "model.ply")
    if include_pointclouds:
        expected_leaf["models_pointcloud"] = ("meshes", "model.obj")
    models_found: dict[str, set[str]] = {
        representation: set() for representation in expected_leaf
    }
    for info in archive.infolist():
        relative = _safe_relative_path(info.filename)
        if info.is_dir():
            continue
        parts = relative.parts
        include = extract_mode == "all"
        if extract_mode == "normalized":
            include = False
            for representation, leaf in expected_leaf.items():
                if representation not in parts:
                    continue
                index = parts.index(representation)
                if index + 1 >= len(parts):
                    break
                gso_id = parts[index + 1]
                include = selected_gso_ids is None or gso_id in selected_gso_ids
                if include and parts[-2:] == leaf:
                    models_found[representation].add(gso_id)
                break
            if not include and relative.name == "invalid_meshes.json":
                include = True
        if include:
            selected.append(info)

    if extract_mode == "normalized":
        for representation, found in models_found.items():
            if selected_gso_ids is not None:
                missing = sorted(selected_gso_ids - found)
                if missing:
                    preview = ", ".join(missing[:10])
                    raise FileNotFoundError(
                        f"Archive is missing {representation} models for "
                        f"{len(missing)} selected GSO objects: {preview}"
                    )
            elif not found:
                raise FileNotFoundError(
                    f"Archive contains no usable models in {representation}"
                )
    return selected


def _additional_extraction_bytes(
    members: Iterable[zipfile.ZipInfo], destination: Path
) -> int:
    total = 0
    for info in members:
        target = destination / _safe_relative_path(info.filename)
        if not target.is_file() or target.stat().st_size != info.file_size:
            total += int(info.file_size)
    return total


def extract_members(
    archive: zipfile.ZipFile,
    members: list[zipfile.ZipInfo],
    destination: Path,
) -> tuple[int, int]:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    additional_bytes = _additional_extraction_bytes(members, destination)
    _require_free_space(destination, additional_bytes, "extracting GSO meshes")

    extracted = 0
    skipped = 0
    for index, info in enumerate(members, start=1):
        relative = _safe_relative_path(info.filename)
        target = destination / relative
        if target.is_file() and target.stat().st_size == info.file_size:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        with archive.open(info) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=COPY_CHUNK_BYTES)
        if temporary.stat().st_size != info.file_size:
            raise IOError(f"Short extraction for {info.filename}")
        os.replace(temporary, target)
        extracted += 1
        if index % 250 == 0:
            print(f"Extracted/checked {index}/{len(members)} archive members")
    return extracted, skipped


def _write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download MegaPose's prepared Google Scanned Objects meshes."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Asset directory receiving downloads/, gso_models.json, and extracted meshes.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        help="Extract only object IDs present in a prepared Pi3 MegaPose-GSO SQLite index.",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        action="append",
        default=[],
        help="Extract one numeric GSO object ID; repeat for a small render smoke test.",
    )
    parser.add_argument(
        "--extract",
        choices=("normalized", "all", "none"),
        default="normalized",
        help="Extraction scope (default: normalized textured meshes only).",
    )
    parser.add_argument(
        "--include-bop-meshes",
        action="store_true",
        help="Also extract compact models_bop-renderer_scale=0.1 PLYs for metrics.",
    )
    parser.add_argument(
        "--include-pointclouds",
        action="store_true",
        help="Also extract compact models_pointcloud OBJs for future point sampling.",
    )
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--mapping-url", default=DEFAULT_MAPPING_URL)
    parser.add_argument(
        "--expected-archive-bytes",
        type=int,
        default=DEFAULT_ARCHIVE_BYTES,
        help="Expected ZIP size; set to 0 only for a verified repackaged mirror.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace existing archive and mapping downloads.",
    )
    parser.add_argument(
        "--delete-archive-after-extract",
        action="store_true",
        help="Delete the ZIP only after successful extraction and manifest writing.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm that you are authorized to download and use the GSO assets.",
    )
    args = parser.parse_args(argv)
    if not args.accept_license:
        parser.error(
            "--accept-license is required; review the MegaPose/GSO terms linked in "
            "datasets/preprocess/download/README.md"
        )
    if args.index_path is not None and args.object_id:
        parser.error("Use either --index-path or repeated --object-id, not both")
    if args.extract == "none" and args.delete_archive_after_extract:
        parser.error("--delete-archive-after-extract requires extraction")
    if args.extract == "none" and (args.include_bop_meshes or args.include_pointclouds):
        parser.error("Compact representation flags require extraction")
    if args.extract == "all" and (args.index_path is not None or args.object_id):
        parser.error("Object filtering is supported only with --extract normalized")
    if args.expected_archive_bytes < 0:
        parser.error("--expected-archive-bytes must be non-negative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    archive_path = output_root / "downloads" / ARCHIVE_NAME
    mapping_path = output_root / MAPPING_NAME

    download_file(
        args.mapping_url,
        mapping_path,
        force=args.force_download,
        timeout=args.timeout,
    )
    mapping = load_object_mapping(mapping_path)

    selected_object_ids: set[int] | None = None
    if args.index_path is not None:
        selected_object_ids = load_index_object_ids(args.index_path)
    elif args.object_id:
        selected_object_ids = set(args.object_id)
    if selected_object_ids is not None:
        unknown = sorted(selected_object_ids - mapping.keys())
        if unknown:
            raise ValueError(f"Numeric object IDs are absent from gso_models.json: {unknown}")
        selected_gso_ids = {mapping[object_id] for object_id in selected_object_ids}
    else:
        selected_gso_ids = None

    expected_bytes = args.expected_archive_bytes or None
    download_file(
        args.archive_url,
        archive_path,
        expected_bytes=expected_bytes,
        force=args.force_download,
        timeout=args.timeout,
    )

    extracted = skipped = selected_members = extracted_bytes = 0
    if args.extract != "none":
        with zipfile.ZipFile(archive_path) as archive:
            members = select_archive_members(
                archive,
                extract_mode=args.extract,
                selected_gso_ids=selected_gso_ids,
                include_bop_meshes=args.include_bop_meshes,
                include_pointclouds=args.include_pointclouds,
            )
            selected_members = len(members)
            extracted_bytes = sum(int(info.file_size) for info in members)
            print(
                f"Selected {selected_members} files "
                f"({human_bytes(extracted_bytes)} uncompressed)"
            )
            extracted, skipped = extract_members(archive, members, output_root)

    manifest_path = output_root / "megapose_gso_meshes.download.json"
    manifest = {
        "format": "pi3_megapose_gso_mesh_download_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "archive_url": args.archive_url,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "mapping_url": args.mapping_url,
        "mapping_path": str(mapping_path),
        "mapping_objects": len(mapping),
        "extract_mode": args.extract,
        "include_bop_meshes": args.include_bop_meshes,
        "include_pointclouds": args.include_pointclouds,
        "selected_numeric_object_ids": (
            None if selected_object_ids is None else sorted(selected_object_ids)
        ),
        "selected_archive_members": selected_members,
        "selected_uncompressed_bytes": extracted_bytes,
        "files_extracted_this_run": extracted,
        "files_reused_this_run": skipped,
        "normalized_mesh_scale_for_megapose": 0.1,
        "archive_retained": True,
    }
    _write_manifest(manifest_path, manifest)

    if args.delete_archive_after_extract:
        archive_path.unlink()
        manifest["archive_retained"] = False
        _write_manifest(manifest_path, manifest)
        print(f"Deleted archive after successful extraction: {archive_path}")

    print(f"Wrote {manifest_path}")
    if selected_object_ids is not None:
        print(f"Selected {len(selected_object_ids)} numeric GSO object IDs")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
