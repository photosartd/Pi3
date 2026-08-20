import json
from pathlib import Path
import tempfile
import unittest

from pi3.metrics.query_occupancy import (
    occupancy_summary_scalars,
    query_occupancy_correlations,
    summarize_query_occupancy,
    validate_occupancy_bin_edges,
    write_query_occupancy_artifacts,
)


def make_row(occupancy, pose, rotation, translation, visibility=0.8):
    return {
        "obj_id": 1,
        "query_visible_fraction": occupancy,
        "query_visible_patches": occupancy * 336 * 252 / (14 * 14),
        "query_visibility_fraction": visibility,
        "query_used_d": pose,
        "query_rot_deg": rotation,
        "query_trans_m": translation,
        "query_trans_lateral_m": translation * 0.6,
        "query_trans_depth_m": translation * 0.8,
        "query_center_m": translation * 2.0,
        "query_center_radial_m": translation * 0.7,
        "query_center_tangential_m": translation * 1.8,
        "query_direction_deg": rotation * 0.7,
        "query_depth_abs_m": translation * 0.4,
        "query_depth_relative": translation * 0.2,
    }


class QueryOccupancyMetricTest(unittest.TestCase):
    def test_summary_uses_exact_fraction_bins_and_pose_thresholds(self):
        rows = [
            make_row(0.004, 1.2, 40.0, 0.20),
            make_row(0.007, 0.8, 20.0, 0.10),
            make_row(0.015, 0.4, 10.0, 0.05),
            make_row(0.030, 0.05, 2.0, 0.01),
            make_row(0.090, 0.02, 1.0, 0.005),
        ]
        summary = summarize_query_occupancy(rows)
        self.assertEqual([row["query_count"] for row in summary], [1, 1, 1, 1, 0, 1])
        self.assertAlmostEqual(summary[0]["query_used_median_d"], 1.2)
        self.assertEqual(summary[0]["query_used_0_5d"], 0.0)
        self.assertEqual(summary[2]["query_used_0_5d"], 1.0)
        self.assertAlmostEqual(summary[3]["query_trans_median_m"], 0.01)
        scalars = occupancy_summary_scalars(rows, summary)
        self.assertEqual(scalars["occupancy/query_count"], 5.0)
        self.assertIn(
            "occupancy/bin_00_0to0p5pct/query_rot_median_deg", scalars
        )

    def test_correlations_include_high_visibility_subset(self):
        rows = [
            make_row(0.004, 1.2, 40.0, 0.20, visibility=0.9),
            make_row(0.010, 0.7, 20.0, 0.10, visibility=0.8),
            make_row(0.030, 0.2, 5.0, 0.02, visibility=0.6),
            make_row(0.080, 0.1, 2.0, 0.01, visibility=0.95),
        ]
        correlations = query_occupancy_correlations(rows)
        self.assertLess(
            correlations["all_queries"]["query_used_d"]["spearman_r"], 0.0
        )
        self.assertEqual(
            correlations["visibility_ge_0_7"]["query_used_d"]["n"], 3
        )

    def test_artifact_writer_emits_reusable_rows_and_summary(self):
        rows = [
            make_row(0.004, 1.2, 40.0, 0.20),
            make_row(0.015, 0.4, 10.0, 0.05),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            scalars = write_query_occupancy_artifacts(rows, output)
            self.assertTrue((output / "query_rows.csv").exists())
            self.assertTrue((output / "summary.csv").exists())
            self.assertTrue((output / "summary_visibility_ge_0_7.csv").exists())
            self.assertTrue((output / "object_bin_counts.csv").exists())
            self.assertTrue((output / "correlations.json").exists())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(sum(row["query_count"] for row in summary), 2)
            self.assertEqual(scalars["occupancy/query_count"], 2.0)

    def test_invalid_edges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_occupancy_bin_edges([0.0, 0.1, 0.1, 1.1])
        with self.assertRaisesRegex(ValueError, "cover fraction 1.0"):
            validate_occupancy_bin_edges([0.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
