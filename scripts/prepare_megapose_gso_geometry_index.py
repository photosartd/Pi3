#!/usr/bin/env python3
"""Build, benchmark and query the derived MegaPose-GSO geometry index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from datasets.preprocess.megapose_gso_geometry import (
    BuildSettings,
    build_geometry_index,
    build_render_geometry_index,
    build_reference_plan_catalog,
    collect_capacity_statistics,
    index_summary,
)


DEFAULT_DATA_ROOT = Path("/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed")
DEFAULT_ASSETS_ROOT = Path("/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets")


def _paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--model-catalog", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bits", type=Path)
    parser.add_argument("--atlas", type=Path)


def _resolved(args):
    data_root = args.data_root.expanduser().resolve()
    assets_root = args.assets_root.expanduser().resolve()
    output = (
        data_root / "pi3_index" / "megapose_gso_geometry_v1.sqlite"
        if args.output is None
        else args.output.expanduser().resolve()
    )
    return {
        "data_root": data_root,
        "index_path": (
            data_root / "pi3_index" / "megapose_gso.sqlite"
            if args.index_path is None
            else args.index_path.expanduser().resolve()
        ),
        "split_path": (
            data_root / "pi3_index" / "megapose_gso.splits.json"
            if args.split_path is None
            else args.split_path.expanduser().resolve()
        ),
        "model_catalog_path": (
            assets_root / "pi3_gso_models.json"
            if args.model_catalog is None
            else args.model_catalog.expanduser().resolve()
        ),
        "output_path": output,
        "bits_path": (
            Path(str(output) + ".surface_bits")
            if args.bits is None
            else args.bits.expanduser().resolve()
        ),
        "atlas_path": (
            data_root / "pi3_index" / f"gso_surface_atlas_{args.surface_points}.f32"
            if args.atlas is None
            else args.atlas.expanduser().resolve()
        ),
    }


def _settings(args) -> BuildSettings:
    return BuildSettings(
        split=args.split,
        visibility_floor=args.visibility_floor,
        min_visible_pixels=args.min_visible_pixels,
        depth_corruption_policy=args.depth_policy,
        surface_point_count=args.surface_points,
        surface_seed=args.surface_seed,
        crop_aspect=args.crop_aspect,
        crop_margin_fraction=args.crop_margin,
        depth_tolerance_m=args.depth_tolerance_m,
        depth_tolerance_relative=args.depth_tolerance_relative,
        pixel_radius=args.pixel_radius,
    )


def _build_options(parser: argparse.ArgumentParser) -> None:
    _paths(parser)
    parser.add_argument("--split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--visibility-floor", type=float, default=0.1)
    parser.add_argument("--min-visible-pixels", type=int, default=64)
    parser.add_argument(
        "--depth-policy",
        choices=("clean_only", "exclude_known_bad", "all"),
        default="clean_only",
    )
    parser.add_argument("--surface-points", type=int, default=8192)
    parser.add_argument("--surface-seed", type=int, default=20260814)
    parser.add_argument("--crop-aspect", type=float, default=4.0 / 3.0)
    parser.add_argument("--crop-margin", type=float, default=0.05)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.002)
    parser.add_argument("--depth-tolerance-relative", type=float, default=0.002)
    parser.add_argument("--pixel-radius", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-progress", action="store_true")


def build_command(args) -> None:
    paths = _resolved(args)
    result = build_geometry_index(
        **paths,
        settings=_settings(args),
        device=args.device,
        workers=args.workers,
        batch_frames=args.batch_frames,
        maximum_frames=args.max_frames,
        progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_render_command(args) -> None:
    bank_root = args.bank_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    settings = BuildSettings(
        split="all",
        visibility_floor=args.visibility_floor,
        min_visible_pixels=args.min_visible_pixels,
        depth_corruption_policy="clean_only",
        surface_point_count=args.surface_points,
        surface_seed=args.surface_seed,
        crop_aspect=args.crop_aspect,
        crop_margin_fraction=args.crop_margin,
        depth_tolerance_m=args.depth_tolerance_m,
        depth_tolerance_relative=args.depth_tolerance_relative,
        pixel_radius=args.pixel_radius,
    )
    result = build_render_geometry_index(
        bank_root=bank_root,
        reference_index_path=(
            bank_root / "pi3_index" / "references.sqlite"
            if args.reference_index is None
            else args.reference_index.expanduser().resolve()
        ),
        model_catalog_path=args.model_catalog.expanduser().resolve(),
        output_path=output,
        bits_path=(
            Path(str(output) + ".surface_bits")
            if args.bits is None
            else args.bits.expanduser().resolve()
        ),
        atlas_path=args.atlas.expanduser().resolve(),
        settings=settings,
        device=args.device,
        workers=args.workers,
        batch_frames=args.batch_frames,
        maximum_frames=args.max_frames,
        progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def benchmark_command(args) -> None:
    base_paths = _resolved(args)
    results = []
    counts = [int(value) for value in args.frame_counts]
    with tempfile.TemporaryDirectory(prefix="gso-geometry-benchmark-") as temporary:
        root = Path(temporary)
        for index, count in enumerate(counts, 1):
            paths = dict(base_paths)
            paths["output_path"] = root / f"pilot_{count}.sqlite"
            paths["bits_path"] = root / f"pilot_{count}.bits"
            started = time.monotonic()
            value = build_geometry_index(
                **paths,
                settings=_settings(args),
                device=args.device,
                workers=args.workers,
                batch_frames=args.batch_frames,
                maximum_frames=count,
                progress=not args.no_progress,
            )
            value["pilot"] = index
            value["requested_frames"] = count
            value["wall_seconds"] = time.monotonic() - started
            results.append(value)
            print(json.dumps(value, indent=2, sort_keys=True))
    stable = results[-1]
    estimate = stable["estimated_total_seconds"]
    fit_results = results[-min(3, len(results)) :]
    frame_values = np.asarray(
        [value["processed_frames"] for value in fit_results], dtype=np.float64
    )
    elapsed_values = np.asarray(
        [value["elapsed_seconds"] for value in fit_results], dtype=np.float64
    )
    slope, intercept = np.polyfit(frame_values, elapsed_values, 1)
    total_frames = int(stable["total_source_frames"])
    fitted_estimate = max(0.0, float(intercept + slope * total_frames))
    print(
        json.dumps(
            {
                "pilots": results,
                "last_pilot_full_estimate_seconds": estimate,
                "last_pilot_full_estimate_hours": estimate / 3600.0,
                "linear_fit_full_estimate_seconds": fitted_estimate,
                "linear_fit_full_estimate_hours": fitted_estimate / 3600.0,
                "linear_fit_fixed_overhead_seconds": float(intercept),
                "linear_fit_steady_frames_per_second": (
                    1.0 / float(slope) if slope > 0 else None
                ),
                "note": "The linear fit uses the largest three pilots and removes fixed SQL/GPU startup overhead.",
            },
            indent=2,
            sort_keys=True,
        )
    )


def summary_command(args) -> None:
    print(json.dumps(index_summary(args.index.expanduser().resolve()), indent=2, sort_keys=True))


def stats_command(args) -> None:
    result = collect_capacity_statistics(
        args.index.expanduser().resolve(),
        n_values=args.n,
        k_values=args.k,
        query_visibility_min=args.query_visibility,
        reference_visibility_min=args.reference_visibility,
        per_reference_surface_min=args.per_reference_surface,
        union_surface_min=args.union_surface,
        positive_angle_degrees=args.positive_angle,
        focal_relative_tolerance=args.focal_tolerance,
        common_focal_target=args.common_focal_target,
        progress=not args.no_progress,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.expanduser().resolve().write_text(encoded, encoding="utf-8")
    displayed = result if args.include_per_object else {
        key: value for key, value in result.items() if key != "per_object"
    }
    print(json.dumps(displayed, indent=2, sort_keys=True))


def plan_command(args) -> None:
    result = build_reference_plan_catalog(
        args.index.expanduser().resolve(),
        args.output.expanduser().resolve(),
        reference_count=args.n,
        reference_visibility_min=args.reference_visibility,
        per_reference_surface_min=args.per_reference_surface,
        union_surface_min=args.union_surface,
        focal_shape_log_tolerance=args.focal_shape_tolerance,
        reference_source_kind=args.reference_source_kind,
        variants_per_track=args.variants_per_track,
        progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    commands = output.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build or resume the geometry index")
    _build_options(build)
    build.set_defaults(handler=build_command)
    build_render = commands.add_parser(
        "build-render", help="Build or resume exact geometry for a BOP render bank"
    )
    build_render.add_argument(
        "--bank-root",
        type=Path,
        default=DEFAULT_ASSETS_ROOT / "renders",
    )
    build_render.add_argument("--reference-index", type=Path)
    build_render.add_argument(
        "--model-catalog",
        type=Path,
        default=DEFAULT_ASSETS_ROOT / "pi3_gso_models.json",
    )
    build_render.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ASSETS_ROOT / "renders" / "pi3_index" / "render_geometry_v1.sqlite",
    )
    build_render.add_argument("--bits", type=Path)
    build_render.add_argument(
        "--atlas",
        type=Path,
        default=DEFAULT_DATA_ROOT / "pi3_index" / "gso_surface_atlas_8192.f32",
    )
    build_render.add_argument("--visibility-floor", type=float, default=0.1)
    build_render.add_argument("--min-visible-pixels", type=int, default=64)
    build_render.add_argument("--surface-points", type=int, default=8192)
    build_render.add_argument("--surface-seed", type=int, default=20260814)
    build_render.add_argument("--crop-aspect", type=float, default=4.0 / 3.0)
    build_render.add_argument("--crop-margin", type=float, default=0.05)
    build_render.add_argument("--depth-tolerance-m", type=float, default=0.002)
    build_render.add_argument("--depth-tolerance-relative", type=float, default=0.002)
    build_render.add_argument("--pixel-radius", type=int, default=1)
    build_render.add_argument("--device", default="cuda")
    build_render.add_argument("--workers", type=int, default=8)
    build_render.add_argument("--batch-frames", type=int, default=16)
    build_render.add_argument("--max-frames", type=int)
    build_render.add_argument("--no-progress", action="store_true")
    build_render.set_defaults(handler=build_render_command)
    benchmark = commands.add_parser(
        "benchmark", help="Run four or more disposable pilots and extrapolate ETA"
    )
    _build_options(benchmark)
    benchmark.add_argument(
        "--frame-counts", type=int, nargs="+", default=(8, 32, 128, 512)
    )
    benchmark.set_defaults(handler=benchmark_command)
    summary = commands.add_parser("summary", help="Show index progress and contents")
    summary.add_argument("--index", type=Path, required=True)
    summary.set_defaults(handler=summary_command)
    stats = commands.add_parser("stats", help="Count constructible N/K scene-pair units")
    stats.add_argument("--index", type=Path, required=True)
    stats.add_argument("--n", type=int, nargs="+", default=(2, 5, 8))
    stats.add_argument("--k", type=int, nargs="+", default=(1, 5, 10))
    stats.add_argument("--query-visibility", type=float, default=0.1)
    stats.add_argument("--reference-visibility", type=float, default=0.3)
    stats.add_argument("--per-reference-surface", type=float, default=0.0)
    stats.add_argument("--union-surface", type=float, default=0.5)
    stats.add_argument("--positive-angle", type=float, default=10.0)
    stats.add_argument("--focal-tolerance", type=float, default=0.1)
    stats.add_argument("--common-focal-target", action="store_true")
    stats.add_argument("--json-output", type=Path)
    stats.add_argument(
        "--include-per-object",
        action="store_true",
        help="Also print the large per-object section; --json-output always stores it.",
    )
    stats.add_argument("--no-progress", action="store_true")
    stats.set_defaults(handler=stats_command)
    plan = commands.add_parser(
        "plan", help="Build a compact coverage-valid reference-plan catalogue"
    )
    plan.add_argument("--index", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--n", type=int, default=5)
    plan.add_argument("--reference-visibility", type=float, default=0.3)
    plan.add_argument("--per-reference-surface", type=float, default=0.0)
    plan.add_argument("--union-surface", type=float, default=0.5)
    plan.add_argument("--focal-shape-tolerance", type=float, default=0.03)
    plan.add_argument(
        "--reference-source-kind", choices=("scene", "render"), default="scene"
    )
    plan.add_argument(
        "--variants-per-track",
        type=int,
        default=1,
        help=(
            "Number of positive-view-anchored variants per track. Use 1 for "
            "scene tracks and a larger value for dense render banks."
        ),
    )
    plan.add_argument("--no-progress", action="store_true")
    plan.set_defaults(handler=plan_command)
    return output


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
