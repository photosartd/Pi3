#!/usr/bin/env python3
"""Prepare a Pi3/BOP-compatible catalogue for downloaded MegaPose-GSO models."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError


CATALOG_FORMAT = "pi3_megapose_gso_model_catalog_v1"
DIAMETER_METHOD = "exact_convex_hull_pairwise_v1"
NORMALIZED_SCALE_M = 0.1
BOP_PLY_UNIT_SCALE_M = 0.001


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def load_mapping(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {path}")
    mapping: dict[int, str] = {}
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or "obj_id" not in row or "gso_id" not in row:
            raise ValueError(f"Invalid mapping row in {path}: {row!r}")
        object_id = int(row["obj_id"])
        gso_id = str(row["gso_id"])
        if object_id in mapping or gso_id in names:
            raise ValueError(f"Duplicate mapping row in {path}: {row!r}")
        mapping[object_id] = gso_id
        names.add(gso_id)
    return mapping


def load_index_object_ids(path: Path) -> list[int]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MegaPose-GSO index does not exist: {path}")
    uri = f"file:{path}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        try:
            rows = connection.execute(
                "SELECT object_id FROM objects ORDER BY object_id"
            ).fetchall()
        except sqlite3.Error as exc:
            raise ValueError(f"Invalid MegaPose-GSO index {path}: {exc}") from exc
    object_ids = [int(row[0]) for row in rows]
    if not object_ids:
        raise ValueError(f"No object IDs found in {path}")
    return object_ids


def load_split_membership(path: Path | None, object_ids: set[int]) -> dict[int, str]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("format") != "pi3_entity_split_v1":
        raise ValueError(f"Unsupported split manifest format in {path}")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"Missing splits in {path}")

    membership: dict[int, str] = {}
    for split_name in ("train", "val"):
        split = splits.get(split_name)
        if not isinstance(split, dict) or not isinstance(split.get("object_ids"), list):
            raise ValueError(f"Missing {split_name}.object_ids in {path}")
        for raw_object_id in split["object_ids"]:
            object_id = int(raw_object_id)
            if object_id in membership:
                raise ValueError(f"Object {object_id} occurs in multiple splits in {path}")
            membership[object_id] = split_name
    if set(membership) != object_ids:
        missing = sorted(object_ids - set(membership))
        extra = sorted(set(membership) - object_ids)
        raise ValueError(
            f"Split/index object mismatch in {path}: missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )
    return membership


def load_ascii_ply_vertices(path: Path) -> np.ndarray:
    """Load only XYZ vertices from a simple ASCII PLY without reading faces."""

    with path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        if stream.readline().strip() != "format ascii 1.0":
            raise ValueError(f"Only ASCII PLY 1.0 is supported: {path}")

        vertex_count: int | None = None
        active_element: str | None = None
        vertex_properties: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            fields = line.strip().split()
            if not fields or fields[0] in {"comment", "obj_info"}:
                continue
            if fields[0] == "element":
                if len(fields) != 3:
                    raise ValueError(f"Invalid element declaration in {path}: {line!r}")
                active_element = fields[1]
                if active_element == "vertex":
                    vertex_count = int(fields[2])
            elif fields[0] == "property" and active_element == "vertex":
                if len(fields) != 3 or fields[1] == "list":
                    raise ValueError(f"Unsupported vertex property in {path}: {line!r}")
                vertex_properties.append(fields[2])
            elif fields[0] == "end_header":
                break

        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"PLY has no vertices: {path}")
        try:
            xyz_indices = [vertex_properties.index(axis) for axis in ("x", "y", "z")]
        except ValueError as exc:
            raise ValueError(f"PLY vertex properties do not contain XYZ: {path}") from exc

        vertices = np.empty((vertex_count, 3), dtype=np.float64)
        for index in range(vertex_count):
            line = stream.readline()
            if not line:
                raise ValueError(f"Short PLY vertex section in {path}")
            values = line.split()
            try:
                vertices[index] = [float(values[position]) for position in xyz_indices]
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid PLY vertex in {path}: {line!r}") from exc
    if not np.isfinite(vertices).all():
        raise ValueError(f"PLY contains non-finite vertices: {path}")
    return vertices


def _hull_vertices(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(points.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank == 0:
        return points[:1]
    projected = centered @ right_vectors[:rank].T
    if rank == 1:
        extrema = [int(np.argmin(projected[:, 0])), int(np.argmax(projected[:, 0]))]
        return points[np.unique(extrema)]
    try:
        hull = ConvexHull(projected)
    except QhullError:
        return points
    return points[np.asarray(hull.vertices, dtype=np.int64)]


def exact_point_diameter(points: np.ndarray, chunk_size: int = 256) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Expected nonempty Nx3 points, got {points.shape}")
    hull = _hull_vertices(points)
    maximum_squared = 0.0
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(hull), chunk_size):
        difference = hull[start : start + chunk_size, None, :] - hull[None, :, :]
        squared = np.einsum("ijk,ijk->ij", difference, difference)
        maximum_squared = max(maximum_squared, float(squared.max(initial=0.0)))
    return float(np.sqrt(maximum_squared))


def model_info(vertices_mm: np.ndarray, diameter_chunk_size: int = 256) -> dict[str, Any]:
    minimum = vertices_mm.min(axis=0)
    maximum = vertices_mm.max(axis=0)
    size = maximum - minimum
    diameter = exact_point_diameter(vertices_mm, chunk_size=diameter_chunk_size)
    if diameter <= 0:
        raise ValueError("Model diameter must be positive")
    return {
        "diameter": diameter,
        "min_x": float(minimum[0]),
        "min_y": float(minimum[1]),
        "min_z": float(minimum[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }


def _catalog_path(path: Path, root: Path) -> str:
    """Prefer a portable relative path, but allow an explicit external input."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _link_or_copy(source: Path, destination: Path, mode: str) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target = os.path.relpath(source, start=destination.parent)
        if destination.is_symlink() and os.readlink(destination) == target:
            return False
        if destination.exists() and not destination.is_symlink():
            raise FileExistsError(f"Refusing to replace non-symlink {destination}")
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target)
        os.replace(temporary, destination)
        return True
    if mode == "copy":
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return False
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return True
    raise ValueError(f"Unsupported materialization mode: {mode}")


def prepare_models(
    *,
    assets_root: Path,
    index_path: Path,
    split_path: Path | None,
    mapping_path: Path | None = None,
    models_folder: str = "models_eval",
    materialization: str = "symlink",
    diameter_chunk_size: int = 256,
) -> dict[str, Any]:
    assets_root = assets_root.expanduser().resolve()
    if not assets_root.is_dir():
        raise FileNotFoundError(f"GSO assets root does not exist: {assets_root}")
    mapping_path = (
        assets_root / "gso_models.json" if mapping_path is None else mapping_path.expanduser().resolve()
    )
    mapping = load_mapping(mapping_path)
    object_ids = load_index_object_ids(index_path)
    object_id_set = set(object_ids)
    membership = load_split_membership(split_path, object_id_set)
    unknown = sorted(object_id_set - set(mapping))
    if unknown:
        raise ValueError(f"Indexed object IDs absent from {mapping_path}: {unknown}")

    normalized_root = assets_root / "google_scanned_objects" / "models_normalized"
    bop_root = (
        assets_root
        / "google_scanned_objects"
        / "models_bop-renderer_scale=0.1"
    )
    pointcloud_root = assets_root / "google_scanned_objects" / "models_pointcloud"
    models_eval = assets_root / models_folder
    catalog_path = assets_root / "pi3_gso_models.json"

    models_info: dict[str, dict[str, Any]] = {}
    catalog_objects: list[dict[str, Any]] = []
    materialized = reused = 0
    for position, object_id in enumerate(object_ids, start=1):
        gso_id = mapping[object_id]
        normalized_obj = normalized_root / gso_id / "meshes" / "model.obj"
        normalized_mtl = normalized_obj.with_name("model.mtl")
        texture = normalized_obj.with_name("texture.png")
        bop_ply = bop_root / gso_id / "meshes" / "model.ply"
        pointcloud_obj = pointcloud_root / gso_id / "meshes" / "model.obj"
        required = (normalized_obj, normalized_mtl, texture, bop_ply, pointcloud_obj)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing assets for numeric object {object_id} ({gso_id}): {missing}"
            )

        vertices_mm = load_ascii_ply_vertices(bop_ply)
        info = model_info(vertices_mm, diameter_chunk_size=diameter_chunk_size)
        models_info[str(object_id)] = info
        destination = models_eval / f"obj_{object_id:06d}.ply"
        if _link_or_copy(bop_ply, destination, materialization):
            materialized += 1
        else:
            reused += 1

        catalog_objects.append(
            {
                "object_id": object_id,
                "gso_id": gso_id,
                "split": membership.get(object_id),
                "normalized_obj": _catalog_path(normalized_obj, assets_root),
                "normalized_mtl": _catalog_path(normalized_mtl, assets_root),
                "texture": _catalog_path(texture, assets_root),
                "normalized_mesh_scale_m": NORMALIZED_SCALE_M,
                "bop_ply": _catalog_path(bop_ply, assets_root),
                "bop_ply_unit_scale_m": BOP_PLY_UNIT_SCALE_M,
                "pointcloud_obj": _catalog_path(pointcloud_obj, assets_root),
                "pointcloud_scale_m": NORMALIZED_SCALE_M,
                "vertex_count": int(len(vertices_mm)),
                "diameter_m": float(info["diameter"] * BOP_PLY_UNIT_SCALE_M),
                "symmetry": "unknown",
            }
        )
        if position % 50 == 0 or position == len(object_ids):
            print(f"Prepared {position}/{len(object_ids)} GSO models")

    _write_json_atomic(models_eval / "models_info.json", models_info)
    catalog = {
        "format": CATALOG_FORMAT,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "object_count": len(catalog_objects),
        "assets_root": ".",
        "mapping_path": _catalog_path(mapping_path, assets_root),
        "index_path": str(index_path.expanduser().resolve()),
        "split_path": None if split_path is None else str(split_path.expanduser().resolve()),
        "models_folder": models_folder,
        "materialization": materialization,
        "diameter_method": DIAMETER_METHOD,
        "coordinate_units": {
            "normalized_obj_scale_m": NORMALIZED_SCALE_M,
            "bop_ply_unit_scale_m": BOP_PLY_UNIT_SCALE_M,
            "models_info": "millimetres (BOP convention)",
        },
        "symmetry_policy": "unknown; report ADD and ADD-S separately",
        "objects": catalog_objects,
    }
    _write_json_atomic(catalog_path, catalog)
    print(
        f"Wrote {catalog_path} and {models_eval / 'models_info.json'}; "
        f"materialized={materialized}, reused={reused}"
    )
    return catalog


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare a Pi3/BOP-compatible model catalogue for MegaPose-GSO."
    )
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument(
        "--split-path",
        type=Path,
        help="Prepared split manifest; defaults to megapose_gso.splits.json beside the index.",
    )
    parser.add_argument("--mapping-path", type=Path)
    parser.add_argument("--models-folder", default="models_eval")
    parser.add_argument(
        "--materialization",
        choices=("symlink", "copy"),
        default="symlink",
        help="How models_eval/obj_*.ply refers to the downloaded BOP PLYs.",
    )
    parser.add_argument("--diameter-chunk-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.split_path is None:
        args.split_path = args.index_path.parent / "megapose_gso.splits.json"
    if args.diameter_chunk_size <= 0:
        parser.error("--diameter-chunk-size must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    prepare_models(
        assets_root=args.assets_root,
        index_path=args.index_path,
        split_path=args.split_path,
        mapping_path=args.mapping_path,
        models_folder=args.models_folder,
        materialization=args.materialization,
        diameter_chunk_size=args.diameter_chunk_size,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
