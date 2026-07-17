import math
import unittest

import numpy as np

from pi3.metrics.failure_modes import (
    coarse_failure_mode_scalars,
    preprocessing_resize_scale,
)
from pi3.metrics.utils import (
    add_error,
    chamfer_components,
    estimate_world_to_object_sim3,
    invert_se3,
    rotation_error_deg,
)


def make_pose(rotation, translation):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose


def rot_z(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class MetricsUtilsTest(unittest.TestCase):
    def test_sim3_alignment_recovers_known_transform(self):
        scale = 2.5
        rotation = rot_z(math.radians(35.0))
        translation = np.array([0.4, -0.2, 1.1], dtype=np.float64)

        pred_T_W_C = np.stack(
            [
                make_pose(np.eye(3), [0.0, 0.0, 0.0]),
                make_pose(rot_z(0.2), [0.2, 0.0, 0.1]),
                make_pose(rot_z(-0.4), [-0.1, 0.3, 0.2]),
                make_pose(rot_z(0.5), [0.4, -0.1, -0.2]),
            ],
            axis=0,
        )

        gt_T_O_C = []
        for pose in pred_T_W_C:
            aligned = np.eye(4, dtype=np.float64)
            aligned[:3, :3] = rotation @ pose[:3, :3]
            aligned[:3, 3] = scale * (rotation @ pose[:3, 3]) + translation
            gt_T_O_C.append(aligned)
        gt_T_O_C = np.stack(gt_T_O_C, axis=0)
        gt_T_C_O = invert_se3(gt_T_O_C)

        estimated = estimate_world_to_object_sim3(pred_T_W_C[:3], gt_T_C_O[:3])
        pred_query_T_O_C = estimated.camera_to_object_pose(pred_T_W_C[3])
        pred_query_T_C_O = estimated.object_to_camera_pose(pred_T_W_C[3])

        self.assertAlmostEqual(estimated.scale, scale, places=6)
        self.assertLess(rotation_error_deg(estimated.rotation, rotation), 1e-5)
        np.testing.assert_allclose(estimated.translation, translation, atol=1e-6)
        np.testing.assert_allclose(pred_query_T_O_C, gt_T_O_C[3], atol=1e-6)
        np.testing.assert_allclose(pred_query_T_C_O, gt_T_C_O[3], atol=1e-6)

    def test_add_zero_for_identical_pose_and_positive_for_translation(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        gt = np.eye(4, dtype=np.float64)
        pred = np.eye(4, dtype=np.float64)
        self.assertAlmostEqual(add_error(pred, gt, points), 0.0, places=7)

        pred[:3, 3] = [0.1, 0.0, 0.0]
        self.assertAlmostEqual(add_error(pred, gt, points), 0.1, places=7)

    def test_chamfer_identical_cloud_is_zero(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        pred_to_gt, gt_to_pred, chamfer = chamfer_components(points, points, device="cpu")
        self.assertAlmostEqual(pred_to_gt, 0.0, places=7)
        self.assertAlmostEqual(gt_to_pred, 0.0, places=7)
        self.assertAlmostEqual(chamfer, 0.0, places=7)

    def test_preprocessing_resize_scale_matches_center_crop_geometry(self):
        K = np.array(
            [
                [572.0, 0.0, 320.0],
                [0.0, 572.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        scale = preprocessing_resize_scale((640, 480), K, (518, 518))
        self.assertAlmostEqual(scale, 518 / 480, places=6)

    def test_coarse_failure_mode_scalars_use_requested_buckets(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed")

        df = pd.DataFrame(
            [
                {"obj_id": 1, "add_err_norm": 0.05, "size_bin_index": 0, "visib_fract": 0.8, "symmetric": False},
                {"obj_id": 1, "add_err_norm": 0.20, "size_bin_index": 0, "visib_fract": 0.4, "symmetric": False},
                {"obj_id": 10, "add_err_norm": 0.08, "size_bin_index": 3, "visib_fract": 0.9, "symmetric": True},
                {"obj_id": 12, "add_err_norm": 0.30, "size_bin_index": 4, "visib_fract": 0.9, "symmetric": False},
            ]
        )
        scalars = coarse_failure_mode_scalars(df)
        self.assertEqual(scalars["small_visible_recall@0.1d"], 1.0)
        self.assertEqual(scalars["small_occluded_recall@0.1d"], 0.0)
        self.assertEqual(scalars["big_visible_recall@0.1d"], 0.5)
        self.assertEqual(scalars["hard_class_recall@0.1d"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
