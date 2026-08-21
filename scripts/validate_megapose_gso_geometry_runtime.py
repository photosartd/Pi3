#!/usr/bin/env python3
"""Validate a transferred MegaPose-GSO geometry training bundle.

The check is intentionally read-only and uses only the Python standard
library so it can run on a Slurm login node before submitting a GPU job.  It
verifies runtime files, SQLite completion/pairing, packed TAR shard sizes, and
resolved BOP model symlinks.  Surface bitset sidecars are reported but are not
required: training consumes the materialized geometry/plan SQLite files and
only index analysis or plan regeneration needs the bitsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any


PLAN_FORMAT = "pi3_object_geometry_plan_v1"
REFERENCE_FORMAT = "pi3_object_reference_index_v2"


def _require_file(path: Path, description: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    return path


def _metadata(path: Path) -> dict[str, str]:
    path = _require_file(path, "SQLite database")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60.0)
    try:
        return {
            str(key): str(value)
            for key, value in connection.execute("SELECT key,value FROM metadata")
        }
    finally:
        connection.close()


def _first_row_exists(path: Path, table: str) -> bool:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60.0)
    try:
        return connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
    finally:
        connection.close()


def _runtime_path(root: Path, stored_path: str) -> Path:
    path = Path(stored_path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _legacy_relocated_path(owner: Path, stored_path: str) -> Path:
    path = Path(stored_path).expanduser()
    if not path.is_absolute():
        return (owner.parent / path).resolve()
    resolved = path.resolve()
    if resolved.is_file():
        return resolved
    sibling = (owner.parent / path.name).resolve()
    return sibling if sibling.is_file() else resolved


def _validate_shards(root: Path, index_path: Path) -> tuple[int, int]:
    connection = sqlite3.connect(
        f"file:{index_path}?mode=ro", uri=True, timeout=60.0
    )
    try:
        rows = connection.execute(
            "SELECT relative_path,size_bytes FROM shards ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"No packed shards are indexed by {index_path}")
    total_bytes = 0
    failures = []
    for stored_path, expected_size in rows:
        shard_path = _runtime_path(root, str(stored_path))
        if not shard_path.is_file():
            failures.append(f"missing {shard_path}")
            continue
        actual_size = shard_path.stat().st_size
        if actual_size != int(expected_size):
            failures.append(
                f"size mismatch {shard_path}: {actual_size} != {expected_size}"
            )
        total_bytes += actual_size
    if failures:
        raise ValueError("Packed shard validation failed: " + "; ".join(failures[:10]))
    return len(rows), total_bytes


def _validate_base_index(data_root: Path) -> dict[str, Any]:
    index_path = _require_file(
        data_root / "pi3_index" / "megapose_gso.sqlite",
        "MegaPose-GSO scene index",
    )
    split_path = _require_file(
        data_root / "pi3_index" / "megapose_gso.splits.json",
        "MegaPose-GSO split manifest",
    )
    metadata = _metadata(index_path)
    if metadata.get("index_complete") != "1":
        raise ValueError(f"Scene index is incomplete: {index_path}")
    if metadata.get("translation_unit") != "metre":
        raise ValueError(f"Unexpected scene translation unit in {index_path}")
    json.loads(split_path.read_text(encoding="utf-8"))
    shard_count, shard_bytes = _validate_shards(data_root, index_path)
    if not _first_row_exists(index_path, "instances"):
        raise ValueError(f"Scene index contains no instances: {index_path}")
    return {
        "index": str(index_path),
        "shards": shard_count,
        "shard_bytes": shard_bytes,
    }


def _validate_reference_index(references_root: Path) -> dict[str, Any]:
    index_path = _require_file(
        references_root / "pi3_index" / "references.sqlite",
        "GSO packed-reference index",
    )
    metadata = _metadata(index_path)
    if metadata.get("format") != REFERENCE_FORMAT:
        raise ValueError(f"Unsupported packed-reference index format: {index_path}")
    if metadata.get("index_complete") != "1":
        raise ValueError(f"Packed-reference index is incomplete: {index_path}")
    shard_count, shard_bytes = _validate_shards(references_root, index_path)
    if shard_count != int(metadata.get("shard_count", -1)):
        raise ValueError(f"Packed-reference shard count disagrees with metadata: {index_path}")
    if not _first_row_exists(index_path, "views"):
        raise ValueError(f"Packed-reference index contains no views: {index_path}")
    return {
        "index": str(index_path),
        "shards": shard_count,
        "shard_bytes": shard_bytes,
        "objects": int(metadata.get("object_count", 0)),
        "views": int(metadata.get("view_count", 0)),
    }


def _validate_models(assets_root: Path) -> dict[str, Any]:
    models_root = assets_root / "models_eval"
    info_path = _require_file(models_root / "models_info.json", "GSO model catalogue")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict) or not info:
        raise ValueError(f"Empty or invalid model catalogue: {info_path}")
    missing = []
    for raw_object_id in info:
        model_path = models_root / f"obj_{int(raw_object_id):06d}.ply"
        if not model_path.is_file():
            missing.append(str(model_path))
    if missing:
        raise FileNotFoundError(
            "GSO model files/symlink targets are missing: " + ", ".join(missing[:10])
        )
    return {"catalogue": str(info_path), "models": len(info)}


def _validate_geometry_index(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    path = _require_file(path, "Geometry feature index")
    metadata = _metadata(path)
    if metadata.get("index_complete") != "1":
        raise ValueError(f"Geometry feature index is incomplete: {path}")
    if not _first_row_exists(path, "frame_features"):
        raise ValueError(f"Geometry feature index contains no rows: {path}")
    fingerprint = json.loads(metadata["fingerprint"])
    sidecar = _legacy_relocated_path(path, metadata["bits_path"])
    return metadata, {
        "path": str(path),
        "surface_bits": str(sidecar),
        "surface_bits_available": sidecar.is_file(),
        "fingerprint": fingerprint,
    }


def _validate_plan(
    path: Path,
    geometry_path: Path,
    *,
    expected_source_kind: str,
) -> dict[str, Any]:
    path = _require_file(path, "Geometry plan catalogue")
    plan_metadata = _metadata(path)
    if plan_metadata.get("format") != PLAN_FORMAT:
        raise ValueError(f"Unsupported geometry plan format: {path}")
    if plan_metadata.get("index_complete") != "1":
        raise ValueError(f"Geometry plan catalogue is incomplete: {path}")
    if plan_metadata.get("reference_source_kind") != expected_source_kind:
        raise ValueError(
            f"Geometry plan source kind mismatch in {path}: "
            f"{plan_metadata.get('reference_source_kind')} != {expected_source_kind}"
        )
    if not _first_row_exists(path, "reference_plans"):
        raise ValueError(f"Geometry plan catalogue contains no plans: {path}")

    geometry_metadata, geometry_report = _validate_geometry_index(geometry_path)
    if json.loads(plan_metadata["geometry_fingerprint"]) != json.loads(
        geometry_metadata["fingerprint"]
    ):
        raise ValueError(
            f"Plan/geometry fingerprint mismatch: {path} vs {geometry_path}"
        )
    constraints = json.loads(plan_metadata["constraints"])
    if int(constraints.get("reference_count", -1)) != 5:
        raise ValueError(f"Expected N=5 plans in {path}")
    embedded = str(plan_metadata["geometry_index_path"])
    embedded_resolved = _legacy_relocated_path(path, embedded)
    return {
        "path": str(path),
        "geometry": geometry_report,
        "embedded_geometry_path": embedded,
        "embedded_path_resolves_to_runtime_geometry": (
            embedded_resolved == geometry_path.resolve()
        ),
    }


def validate_bundle(
    data_root: Path,
    assets_root: Path,
    references_root: Path,
) -> dict[str, Any]:
    data_root = Path(data_root).expanduser().resolve()
    assets_root = Path(assets_root).expanduser().resolve()
    references_root = Path(references_root).expanduser().resolve()
    for root, description in (
        (data_root, "MegaPose-GSO scene root"),
        (assets_root, "MegaPose-GSO assets root"),
        (references_root, "MegaPose-GSO reference root"),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{description} is missing: {root}")

    scene_index_dir = data_root / "pi3_index"
    render_index_dir = references_root / "pi3_index"
    report = {
        "data_root": str(data_root),
        "assets_root": str(assets_root),
        "references_root": str(references_root),
        "scene_payloads": _validate_base_index(data_root),
        "models": _validate_models(assets_root),
        "reference_payloads": _validate_reference_index(references_root),
        "plans": {
            "scene_train": _validate_plan(
                scene_index_dir / "megapose_gso_geometry_train_n5_plans.sqlite",
                scene_index_dir / "megapose_gso_geometry_v1.sqlite",
                expected_source_kind="scene",
            ),
            "scene_val": _validate_plan(
                scene_index_dir / "megapose_gso_geometry_val_n5_plans.sqlite",
                scene_index_dir / "megapose_gso_geometry_val_v1.sqlite",
                expected_source_kind="scene",
            ),
            "render": _validate_plan(
                render_index_dir / "render_geometry_n5_plans.sqlite",
                render_index_dir / "render_geometry_v1.sqlite",
                expected_source_kind="render",
            ),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--references-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    try:
        report = validate_bundle(args.data_root, args.assets_root, args.references_root)
    except (FileNotFoundError, ValueError, sqlite3.Error, json.JSONDecodeError) as error:
        print(f"ERROR: MegaPose-GSO runtime bundle validation failed: {error}", file=sys.stderr)
        return 2

    missing_sidecars = [
        entry["geometry"]["surface_bits"]
        for entry in report["plans"].values()
        if not entry["geometry"]["surface_bits_available"]
    ]
    if missing_sidecars:
        report["warnings"] = [
            "Surface bitset sidecars are absent. Training is supported, but plan "
            "regeneration/capacity analysis requires them: " + ", ".join(missing_sidecars)
        ]
    if args.json_output is not None:
        output = args.json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("MegaPose-GSO runtime bundle: OK")
    print(f"  scene root: {report['data_root']}")
    print(
        "  scene shards: "
        f"{report['scene_payloads']['shards']} "
        f"({report['scene_payloads']['shard_bytes'] / 2**30:.1f} GiB)"
    )
    print(
        "  reference shards: "
        f"{report['reference_payloads']['shards']} "
        f"({report['reference_payloads']['shard_bytes'] / 2**30:.1f} GiB), "
        f"{report['reference_payloads']['objects']} objects, "
        f"{report['reference_payloads']['views']} views"
    )
    print(f"  resolved object models: {report['models']['models']}")
    for name, entry in report["plans"].items():
        mode = (
            "portable/relocatable"
            if entry["embedded_path_resolves_to_runtime_geometry"]
            else "legacy absolute metadata; explicit runtime pairing will be used"
        )
        print(f"  {name} plans: OK ({mode})")
    for warning in report.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
