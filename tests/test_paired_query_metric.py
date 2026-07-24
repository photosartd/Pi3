import unittest

import numpy as np
import torch

from pi3.metrics.paired_query import PairedQueryConsistencyMetric
from pi3.models.ray_conditioning import intrinsics_to_ray_map


class _FakeModelCache:
    def points(self, obj_id):
        return np.array(
            [
                [-0.05, -0.05, 0.0],
                [0.05, -0.05, 0.0],
                [0.05, 0.05, 0.0],
                [-0.05, 0.05, 0.0],
            ],
            dtype=np.float64,
        )

    def diameter(self, obj_id):
        return 1.0


def _invert_pose(T):
    return torch.linalg.inv(T)


def _local_points(intrinsics, height, width, depth=2.0):
    rays = intrinsics_to_ray_map(
        intrinsics[None, None],
        height,
        width,
    )[0, 0]
    z = torch.full((height, width, 1), float(depth), dtype=torch.float32)
    return torch.cat((rays * z, z), dim=-1)


def _collated_view(
    *,
    intrinsics,
    T_C_O,
    is_reference=False,
    is_crop=False,
    is_original=False,
    pair_index=-1,
):
    identity = torch.eye(3, dtype=torch.float32)
    return {
        "camera_intrinsics": intrinsics[None],
        "valid_mask": torch.ones((1, 14, 28), dtype=torch.bool),
        "T_C_O": T_C_O[None],
        "object_id": torch.tensor([1], dtype=torch.int64),
        "is_reference": torch.tensor([is_reference]),
        "is_query": torch.tensor([is_crop]),
        "is_query_context": torch.tensor([is_original]),
        "is_cropped_query": torch.tensor([is_crop]),
        "is_original_query": torch.tensor([is_original]),
        "query_pair_index": torch.tensor([pair_index], dtype=torch.int64),
        "query_recenter_R_old_to_new": identity[None],
        "query_crop_from_original_homography": identity[None],
    }


class PairedQueryConsistencyMetricTest(unittest.TestCase):
    def setUp(self):
        self.height = 14
        self.width = 28
        self.intrinsics = torch.tensor(
            [
                [100.0, 0.0, 14.0],
                [0.0, 100.0, 7.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.points = _local_points(
            self.intrinsics,
            self.height,
            self.width,
        )

        T_O_C_ref0 = torch.eye(4, dtype=torch.float32)
        T_O_C_ref0[:3, 3] = torch.tensor([-0.1, 0.0, -1.0])
        T_O_C_ref1 = torch.eye(4, dtype=torch.float32)
        T_O_C_ref1[:3, 3] = torch.tensor([0.1, 0.0, -1.0])
        T_O_C_query = torch.eye(4, dtype=torch.float32)
        T_O_C_query[:3, 3] = torch.tensor([0.0, 0.0, -1.0])
        self.pred_poses = torch.stack(
            (T_O_C_ref0, T_O_C_ref1, T_O_C_query, T_O_C_query),
            dim=0,
        )[None]
        self.pred_points = self.points[None, None].repeat(1, 4, 1, 1, 1)
        self.batch = [
            _collated_view(
                intrinsics=self.intrinsics,
                T_C_O=_invert_pose(T_O_C_ref0),
                is_reference=True,
            ),
            _collated_view(
                intrinsics=self.intrinsics,
                T_C_O=_invert_pose(T_O_C_ref1),
                is_reference=True,
            ),
            _collated_view(
                intrinsics=self.intrinsics,
                T_C_O=_invert_pose(T_O_C_query),
                is_crop=True,
                pair_index=0,
            ),
            _collated_view(
                intrinsics=self.intrinsics,
                T_C_O=_invert_pose(T_O_C_query),
                is_original=True,
                pair_index=0,
            ),
        ]

    @staticmethod
    def _metric():
        metric = object.__new__(PairedQueryConsistencyMetric)
        metric.model_cache = _FakeModelCache()
        metric.symmetric_ids = set()
        metric.solve_scale = True
        metric.nn_chunk_size = 128
        metric.device = "cpu"
        metric.max_points_per_pair = 4096
        metric.eps = 1e-8
        metric.reset()
        return metric

    def test_exact_pair_has_zero_canonical_pose_and_equivariance_errors(self):
        metric = self._metric()
        metric.update(
            {
                "camera_poses": self.pred_poses,
                "local_points": self.pred_points,
            },
            self.batch,
            mode="val",
        )
        stats = metric.compute()

        self.assertEqual(stats["pair_count"], 1.0)
        self.assertEqual(stats["query_crop_canonical/count"], 1.0)
        self.assertEqual(stats["query_original/count"], 1.0)
        self.assertEqual(stats["query_crop_canonical/add_or_adds_0_1d"], 1.0)
        self.assertEqual(stats["query_original/add_or_adds_0_1d"], 1.0)
        self.assertLess(
            stats["query_pair_pose/relative_rotation_error_deg"],
            1e-5,
        )
        self.assertLess(
            stats["query_pair_pose/relative_translation_m"],
            1e-6,
        )
        self.assertLess(
            stats["query_pair_pose/reprojection_px_mean"],
            1e-4,
        )
        self.assertLess(
            stats["query_pair_dense/angular_deg_mean"],
            1e-4,
        )
        self.assertLess(
            stats["query_pair_dense/relative_3d_mean"],
            1e-5,
        )
        self.assertLess(
            stats["query_pair_world/relative_3d_mean"],
            1e-5,
        )
        self.assertEqual(stats["query_pair_dense/points"], 392.0)

    def test_original_camera_perturbation_is_visible_without_hurting_crop_pose(self):
        prediction = {
            "camera_poses": self.pred_poses.clone(),
            "local_points": self.pred_points,
        }
        prediction["camera_poses"][0, 3, 0, 3] += 0.1
        metric = self._metric()

        metric.update(prediction, self.batch, mode="val")
        stats = metric.compute()

        self.assertLess(
            stats["query_crop_canonical/translation_mean_m"],
            1e-6,
        )
        self.assertGreater(
            stats["query_original/translation_mean_m"],
            0.05,
        )
        self.assertGreater(
            stats["query_pair_pose/relative_translation_m"],
            0.05,
        )
        self.assertGreater(
            stats["query_pair_pose/reprojection_px_mean"],
            1.0,
        )
        self.assertGreater(
            stats["query_pair_world/relative_3d_mean"],
            0.01,
        )


if __name__ == "__main__":
    unittest.main()
