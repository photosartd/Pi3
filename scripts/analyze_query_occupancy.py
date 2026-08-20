#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi3.metrics.query_occupancy import write_query_occupancy_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-bin query pose metrics by exact post-crop visible occupancy."
        )
    )
    parser.add_argument("--rows", required=True, help="query_rows parquet or CSV")
    parser.add_argument("--out", required=True, help="output analysis directory")
    parser.add_argument(
        "--bin-edges",
        default="0,0.005,0.01,0.02,0.04,0.08,1.000001",
        help="comma-separated occupancy fractions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("query occupancy analysis requires pandas") from exc

    rows_path = Path(args.rows)
    if rows_path.suffix == ".parquet":
        frame = pd.read_parquet(rows_path)
    else:
        frame = pd.read_csv(rows_path)
    edges = [float(value) for value in args.bin_edges.split(",")]
    write_query_occupancy_artifacts(
        frame.to_dict(orient="records"),
        args.out,
        edges,
    )
    print(f"Wrote query occupancy analysis to {args.out}")


if __name__ == "__main__":
    main()
