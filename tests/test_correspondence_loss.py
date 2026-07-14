import unittest

import torch

from pi3.models.correspondence import build_query_reference_correspondences
from pi3.models.loss import CorrespondenceConsistencyLoss


def make_view(*, is_reference: bool, batch_size: int = 1, height: int = 28, width: int = 28):
    transform = torch.eye(4).repeat(batch_size, 1, 1)
    intrinsics = torch.tensor(
        [
            [20.0, 0.0, (width - 1) / 2.0],
            [0.0, 20.0, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    ).repeat(batch_size, 1, 1)
    return {
        "depthmap": torch.ones(batch_size, height, width),
        "valid_mask": torch.ones(batch_size, height, width, dtype=torch.bool),
        "T_C_O": transform,
        "camera_intrinsics": intrinsics,
        "is_reference": torch.full((batch_size,), bool(is_reference)),
        "is_query": torch.full((batch_size,), not bool(is_reference)),
    }


def make_prediction(*, query_offset: float = 0.0, batch_size: int = 1, height: int = 28, width: int = 28):
    camera_poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch_size, 2, 1, 1)
    local_points = torch.zeros(batch_size, 2, height, width, 3)
    local_points[..., 2] = 1.0
    local_points[:, 1, ..., 0] += float(query_offset)
    return {
        "local_points": local_points,
        "camera_poses": camera_poses,
    }


class CorrespondenceLossTest(unittest.TestCase):
    def test_builder_accepts_all_valid_patches_for_identical_views(self):
        batch = [make_view(is_reference=True), make_view(is_reference=False)]
        correspondences = build_query_reference_correspondences(
            batch,
            patch_size=14,
            max_reference_per_query=1,
            depth_abs_tol=0.01,
            depth_rel_tol=0.05,
            max_pairs=0,
        )
        self.assertEqual(correspondences.num_pairs, 4)

    def test_loss_is_zero_for_identical_predicted_points(self):
        loss_fn = CorrespondenceConsistencyLoss(
            patch_size=14,
            max_reference_per_query=1,
            max_pairs=0,
            use_dino_weights=False,
        )
        loss, details = loss_fn(make_prediction(query_offset=0.0), [make_view(is_reference=True), make_view(is_reference=False)])
        self.assertAlmostEqual(float(loss), 0.0, places=7)
        self.assertEqual(float(details["correspondence_num_pairs"]), 4.0)

    def test_loss_is_positive_for_query_point_offset(self):
        loss_fn = CorrespondenceConsistencyLoss(
            patch_size=14,
            max_reference_per_query=1,
            max_pairs=0,
            use_dino_weights=False,
            huber_beta=0.01,
        )
        loss, _ = loss_fn(make_prediction(query_offset=0.2), [make_view(is_reference=True), make_view(is_reference=False)])
        self.assertGreater(float(loss), 0.0)

    def test_dino_weights_are_detached_confidence_terms(self):
        loss_fn = CorrespondenceConsistencyLoss(
            patch_size=14,
            max_reference_per_query=1,
            max_pairs=0,
            use_dino_weights=True,
            dino_layer=17,
        )
        pred = make_prediction(query_offset=0.2)
        features = torch.randn(1, 2, 4, 8, requires_grad=True)
        pred["dino_features"] = {"17": features}
        loss, details = loss_fn(pred, [make_view(is_reference=True), make_view(is_reference=False)])
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(details["correspondence_dino_weight"]), 0.0)
        self.assertFalse(details["correspondence_dino_similarity"].requires_grad)


if __name__ == "__main__":
    unittest.main()
