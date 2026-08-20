import math
import unittest

import numpy as np

from pi3.metrics.failure_modes import (
    coarse_failure_mode_scalars,
    preprocessing_resize_scale,
)
from pi3.metrics.utils import (
    add_error,
    camera_center_error_components,
    chamfer_components,
    estimate_world_to_object_sim3,
    invert_se3,
    metric_depth_error_summary,
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
    def test_camera_center_error_decomposes_radial_and_tangential_parts(self):
        components = camera_center_error_components(
            np.array([0.3, 0.0, 2.4]),
            np.array([0.0, 0.0, 2.0]),
        )
        self.assertAlmostEqual(components["total_m"], 0.5, places=7)
        self.assertAlmostEqual(components["radial_m"], 0.4, places=7)
        self.assertAlmostEqual(components["tangential_m"], 0.3, places=7)
        self.assertAlmostEqual(
            components["radius_m"], math.sqrt(0.3**2 + 2.4**2) - 2.0,
            places=7,
        )
        self.assertGreater(components["direction_deg"], 0.0)

    def test_metric_depth_summary_uses_supplied_reference_scale(self):
        summary = metric_depth_error_summary(
            np.array([[1.0, 2.0]]),
            np.array([[2.0, 5.0]]),
            np.array([[True, True]]),
            scale=2.0,
        )
        self.assertAlmostEqual(summary["mae_m"], 0.5, places=7)
        self.assertAlmostEqual(summary["median_abs_m"], 0.5, places=7)
        self.assertAlmostEqual(summary["median_relative"], 0.1, places=7)
        self.assertAlmostEqual(summary["median_bias_m"], -0.5, places=7)
        self.assertAlmostEqual(
            summary["scale_log_error"],
            abs(0.5 * math.log(2.5 / 2.0)),
            places=7,
        )

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

    def test_reference_depth_changes_only_scale_source(self):
        rotation = rot_z(math.radians(20.0))
        pred_T_W_C = np.stack(
            [
                make_pose(np.eye(3), [-1.0, 0.0, 0.0]),
                make_pose(rot_z(0.1), [0.0, 0.0, 0.0]),
                make_pose(rot_z(-0.2), [1.0, 0.0, 0.0]),
            ],
            axis=0,
        )
        # Camera centers imply scale 2, while metric reference depths supply
        # scale 3. Rotation still comes exclusively from camera poses.
        gt_T_O_C = []
        for pose in pred_T_W_C:
            aligned = np.eye(4, dtype=np.float64)
            aligned[:3, :3] = rotation @ pose[:3, :3]
            aligned[:3, 3] = 2.0 * (rotation @ pose[:3, 3]) + [0.2, -0.1, 1.0]
            gt_T_O_C.append(aligned)
        gt_T_C_O = invert_se3(np.stack(gt_T_O_C, axis=0))

        pred_local_points = np.asarray(
            [
                [[[-1.0, -1.0, 4.0], [1.0, -1.0, 4.5]],
                 [[-1.0, 1.0, 5.0], [1.0, 1.0, 5.5]]]
            ] * 3,
            dtype=np.float64,
        )
        gt_local_points = 3.0 * pred_local_points
        gt_points_object = np.empty_like(gt_local_points)
        for view_idx, T_C_O in enumerate(gt_T_C_O):
            T_O_C = invert_se3(T_C_O)
            gt_points_object[view_idx] = (
                gt_local_points[view_idx] @ T_O_C[:3, :3].T
                + T_O_C[:3, 3]
            )
        valid_masks = np.ones((3, 2, 2), dtype=bool)
        estimated = estimate_world_to_object_sim3(
            pred_T_W_C,
            gt_T_C_O,
            scale_estimation="reference_depth",
            pred_local_points_refs=pred_local_points,
            gt_points_object_refs=gt_points_object,
            valid_masks_refs=valid_masks,
            min_depth_pixels_per_view=1,
        )

        self.assertAlmostEqual(estimated.scale, 3.0, places=7)
        self.assertEqual(estimated.scale_estimation, "reference_depth")
        self.assertEqual(estimated.depth_scale_valid_views, 3)
        self.assertAlmostEqual(estimated.depth_scale_log_mad, 0.0, places=7)
        self.assertFalse(estimated.underconstrained_scale)
        self.assertLess(rotation_error_deg(estimated.rotation, rotation), 1e-5)
        gt_centers = np.stack(gt_T_O_C, axis=0)[:, :3, 3]
        expected_translation = gt_centers.mean(axis=0) - 3.0 * (
            rotation @ pred_T_W_C[:, :3, 3].mean(axis=0)
        )
        np.testing.assert_allclose(
            estimated.translation, expected_translation, atol=1e-7
        )

    def test_reference_depth_requires_explicit_depth_inputs(self):
        poses = np.stack(
            [make_pose(np.eye(3), [0.0, 0.0, 0.0])], axis=0
        )
        with self.assertRaisesRegex(ValueError, "requires predicted local points"):
            estimate_world_to_object_sim3(
                poses,
                invert_se3(poses),
                scale_estimation="reference_depth",
            )

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
