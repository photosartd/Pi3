#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi3.metrics.failure_modes import build_covariate_rows, write_covariate_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one BOP/LMGeo failure-mode covariate row per GT object instance.",
    )
    parser.add_argument("--dataset-root", required=True, help="BOP dataset root, for example /vol/.../datasets/lm-o")
    parser.add_argument("--split", required=True, help="BOP split directory name, for example train_pbr, new_val, or test")
    parser.add_argument("--out", required=True, help="Output parquet path, for example covariates_test.parquet")
    parser.add_argument(
        "--model-resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(518, 518),
        help="Pi3 model input resolution used to derive amodal patch counts. Default: 518 518.",
    )
    parser.add_argument("--models-folder", default="models_eval")
    parser.add_argument("--depth-unit-scale", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_covariate_rows(
        args.dataset_root,
        args.split,
        model_resolution=tuple(args.model_resolution),
        models_folder=args.models_folder,
        depth_unit_scale=args.depth_unit_scale,
    )
    write_covariate_table(rows, args.out)
    print(f"Wrote {len(rows)} covariate rows to {args.out}")


if __name__ == "__main__":
    main()
