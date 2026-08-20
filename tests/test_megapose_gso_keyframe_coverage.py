import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "megapose_gso_keyframe_coverage.py"
)
SPEC = importlib.util.spec_from_file_location("megapose_gso_keyframe_coverage", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


class MegaPoseGsoKeyframeCoverageTest(unittest.TestCase):
    def test_load_surface_pointcloud_applies_metric_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "points.obj"
            path.write_text("# points\nv 1 2 3\nv -1 0 0\n", encoding="utf-8")
            points = coverage.load_surface_pointcloud(path, 0.1)
        np.testing.assert_allclose(points, [[0.1, 0.2, 0.3], [-0.1, 0.0, 0.0]])

    def test_depth_and_visible_mask_reject_occluded_surface_samples(self):
        points = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.1],
                [1.0, 0.0, 1.0],
            ]
        )
        depth = np.zeros((5, 5), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=bool)
        depth[2, 2] = 1.0
        mask[2, 2] = True
        K = np.asarray([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
        visible = coverage.visible_surface_sample_mask(
            points,
            np.eye(4),
            K,
            depth,
            mask,
            depth_tolerance_m=0.001,
            depth_tolerance_relative=0.0,
            pixel_radius=0,
        )
        np.testing.assert_array_equal(visible, [True, False, False])

    def test_greedy_curve_prefers_complementary_views(self):
        masks = np.asarray(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, True],
            ]
        )
        directions = np.eye(3)
        result = coverage.greedy_coverage_curve(masks, directions, maximum_views=2)
        self.assertEqual(result["selected_indices"], [0, 2])
        self.assertEqual(result["coverage_fractions"], [0.5, 1.0])
        self.assertEqual(result["marginal_sample_gains"], [2, 2])

    def test_surface_mask_blob_round_trip(self):
        value = np.asarray([True, False, True, True, False, False, True])
        restored = coverage.unpack_surface_mask(
            coverage.pack_surface_mask(value), len(value)
        )
        np.testing.assert_array_equal(restored, value)

    def test_view_spread_metrics(self):
        directions = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertAlmostEqual(
            coverage.max_pairwise_angle_degrees(directions), 90.0, places=6
        )
        cap = coverage.minimum_enclosing_cap(directions)
        self.assertAlmostEqual(cap["radius_degrees"], 45.0, places=4)
        self.assertGreater(cap["solid_angle_steradians"], 0.0)


if __name__ == "__main__":
    unittest.main()
