#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi3.metrics.failure_modes import aggregate_failure_modes


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, float) and value != value:
        return None
    return value


def parse_hard_object_ids(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item) for item in value.replace(" ", "").split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join raw ADD(-S) predictions with BOP covariates and emit failure-mode diagnostics.",
    )
    parser.add_argument("--predictions", required=True, help="Raw predictions parquet produced by ObjectPoseMetric")
    parser.add_argument("--covariates", required=True, help="Covariate parquet produced by build_covariate_table.py")
    parser.add_argument("--out", required=True, help="Output analysis directory")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Checkpoint/global step to analyze. Default: latest in the file.")
    parser.add_argument("--split", default=None, help="Prediction split filter. Default: infer from covariates.")
    parser.add_argument("--val-name", default=None, help="Optional validation-loader name filter.")
    parser.add_argument(
        "--hard-object-ids",
        default=None,
        help="Comma-separated object ids to force into the hard class, e.g. textureless objects.",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Write only the 14 online scalars instead of full CSV/PNG artifacts.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSVs only. Useful on a minimal environment without matplotlib.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scalars = aggregate_failure_modes(
        args.predictions,
        args.covariates,
        args.out,
        coarse=args.coarse,
        plot=not args.no_plots,
        checkpoint_step=args.checkpoint_step,
        split=args.split,
        val_name=args.val_name,
        hard_object_ids=parse_hard_object_ids(args.hard_object_ids),
    )
    if args.coarse:
        print(json.dumps(json_ready(scalars), indent=2, sort_keys=True))
    else:
        print(f"Wrote failure-mode analysis to {args.out}")


if __name__ == "__main__":
    main()
