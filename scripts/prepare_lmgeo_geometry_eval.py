#!/usr/bin/env python3
"""Prepare exact LM-O render-to-new_val geometry evaluation sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from datasets.preprocess.lmgeo_geometry import (
    bop_camera_distribution_summary,
    build_bop_geometry_index,
    geometry_index_summary,
)
from datasets.preprocess.megapose_gso_geometry import (
    BuildSettings,
    build_reference_plan_catalog,
)


DEFAULT_DATA_ROOT = Path(
    "/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o"
)


def _paths(data_root: Path) -> dict[str, Path]:
    index_root = data_root / "pi3_index"
    return {
        "atlas": index_root / "lmgeo_surface_atlas_8192.f32",
        "references": index_root / "lmgeo_train_geometry_v1.sqlite",
        "reference_bits": index_root / "lmgeo_train_geometry_v1.sqlite.surface_bits",
        "queries": index_root / "lmgeo_new_val_geometry_v1.sqlite",
        "query_bits": index_root / "lmgeo_new_val_geometry_v1.sqlite.surface_bits",
        "plans": index_root / "lmgeo_train_geometry_n5_plans.sqlite",
    }


def _settings(args) -> BuildSettings:
    return BuildSettings(
        split="all",
        visibility_floor=float(args.visibility_floor),
        min_visible_pixels=int(args.min_visible_pixels),
        depth_corruption_policy="clean_only",
        surface_point_count=int(args.surface_points),
        surface_seed=int(args.surface_seed),
        crop_aspect=float(args.crop_aspect),
        crop_margin_fraction=float(args.crop_margin),
        depth_tolerance_m=float(args.depth_tolerance_m),
        depth_tolerance_relative=float(args.depth_tolerance_relative),
        pixel_radius=int(args.pixel_radius),
    )


def _build(args, *, split: str, compute_surface: bool):
    data_root = args.data_root.expanduser().resolve()
    paths = _paths(data_root)
    output = paths["references"] if split == "train" else paths["queries"]
    bits = paths["reference_bits"] if split == "train" else paths["query_bits"]
    result = build_bop_geometry_index(
        data_root=data_root,
        split=split,
        output_path=output,
        bits_path=bits,
        atlas_path=paths["atlas"],
        settings=_settings(args),
        models_folder=args.models_folder,
        compute_surface=compute_surface,
        device=args.device,
        workers=args.workers,
        batch_frames=args.batch_frames,
        maximum_frames=args.max_frames,
        progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _plans(args):
    paths = _paths(args.data_root.expanduser().resolve())
    result = build_reference_plan_catalog(
        paths["references"],
        paths["plans"],
        reference_count=args.reference_count,
        reference_visibility_min=args.reference_visibility_min,
        per_reference_surface_min=args.per_reference_surface_min,
        union_surface_min=args.union_surface_min,
        focal_shape_log_tolerance=args.focal_shape_log_tolerance,
        reference_source_kind="render",
        variants_per_track=args.variants_per_object,
        progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _summary(args):
    paths = _paths(args.data_root.expanduser().resolve())
    output = {
        "references": geometry_index_summary(paths["references"]),
        "queries": geometry_index_summary(paths["queries"]),
        "reference_camera_distribution": bop_camera_distribution_summary(
            args.data_root,
            "train",
            paths["references"],
            models_folder=args.models_folder,
        ),
        "query_camera_distribution": bop_camera_distribution_summary(
            args.data_root,
            "new_val",
            paths["queries"],
            models_folder=args.models_folder,
        ),
    }
    connection = sqlite3.connect(paths["plans"])
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        rows = connection.execute(
            """
            SELECT object_id,COUNT(*),MIN(union_coverage),AVG(union_coverage),
                   MAX(union_coverage)
            FROM reference_plans GROUP BY object_id ORDER BY object_id
            """
        ).fetchall()
    finally:
        connection.close()
    output["plans"] = {
        "path": str(paths["plans"]),
        "index_complete": metadata.get("index_complete") == "1",
        "constraints": json.loads(metadata["constraints"]),
        "per_object": {
            str(row[0]): {
                "plans": int(row[1]),
                "minimum_union_coverage": float(row[2]),
                "mean_union_coverage": float(row[3]),
                "maximum_union_coverage": float(row[4]),
            }
            for row in rows
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def _add_common(parser):
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models-folder", default="models_eval")
    parser.add_argument("--surface-points", type=int, default=8192)
    parser.add_argument("--surface-seed", type=int, default=20260814)
    parser.add_argument("--visibility-floor", type=float, default=0.1)
    parser.add_argument("--min-visible-pixels", type=int, default=64)
    parser.add_argument("--crop-aspect", type=float, default=4.0 / 3.0)
    parser.add_argument("--crop-margin", type=float, default=0.05)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.002)
    parser.add_argument("--depth-tolerance-relative", type=float, default=0.002)
    parser.add_argument("--pixel-radius", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--reference-count", type=int, default=5)
    parser.add_argument("--reference-visibility-min", type=float, default=0.3)
    parser.add_argument("--per-reference-surface-min", type=float, default=0.0)
    parser.add_argument("--union-surface-min", type=float, default=0.5)
    parser.add_argument("--focal-shape-log-tolerance", type=float, default=0.03)
    parser.add_argument("--variants-per-object", type=int, default=512)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("all", "build-references", "build-queries", "plans", "summary"):
        subparser = subparsers.add_parser(name)
        _add_common(subparser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "build-references":
        _build(args, split="train", compute_surface=True)
    elif args.command == "build-queries":
        _build(args, split="new_val", compute_surface=False)
    elif args.command == "plans":
        _plans(args)
    elif args.command == "summary":
        _summary(args)
    elif args.command == "all":
        if args.max_frames is not None:
            raise ValueError("The all command requires complete indexes; omit --max-frames")
        _build(args, split="train", compute_surface=True)
        _build(args, split="new_val", compute_surface=False)
        _plans(args)
        _summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
