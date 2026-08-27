import unittest

import torch

from pi3.models.loss import Pi3Loss


def _make_batch(*, batch_size=1, views=2, height=4, width=5):
    batch = []
    for _ in range(views):
        points = torch.zeros(batch_size, height, width, 3)
        points[..., 0] = torch.linspace(-0.2, 0.2, width).view(1, 1, width)
        points[..., 1] = torch.linspace(-0.1, 0.1, height).view(1, height, 1)
        points[..., 2] = 1.25
        batch.append(
            {
                "pts3d": points,
                "valid_mask": torch.ones(
                    batch_size, height, width, dtype=torch.bool
                ),
                "camera_pose": torch.eye(4).repeat(batch_size, 1, 1),
                "img": torch.zeros(batch_size, 3, height, width),
                "dataset": ["MetricScaleUnitTest"] * batch_size,
            }
        )
    return batch


def _make_prediction(batch, scale):
    local = torch.stack([view["pts3d"] for view in batch], dim=1) * float(scale)
    poses = torch.stack([view["camera_pose"] for view in batch], dim=1).clone()
    poses[..., :3, 3] *= float(scale)
    return {"local_points": local, "camera_poses": poses}


class MetricScaleLossTest(unittest.TestCase):
    def test_unknown_scale_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scale_mode"):
            Pi3Loss(scale_mode="per_view")

    def test_aligned_mode_hides_global_point_scale_but_metric_mode_does_not(self):
        batch = _make_batch()
        aligned = Pi3Loss(scale_mode="aligned")
        metric = Pi3Loss(scale_mode="metric")

        _, aligned_details = aligned(_make_prediction(batch, 2.0), batch)
        _, metric_details = metric(_make_prediction(batch, 2.0), batch)

        self.assertLess(float(aligned_details["local_pts_loss"]), 1e-5)
        self.assertGreater(float(metric_details["local_pts_loss"]), 0.3)

    def test_metric_mode_is_zero_for_exact_metric_points(self):
        batch = _make_batch()
        _, details = Pi3Loss(scale_mode="metric")(
            _make_prediction(batch, 1.0),
            batch,
        )
        self.assertLess(float(details["local_pts_loss"]), 1e-7)

    def test_metric_gt_keeps_camera_baseline_in_metres(self):
        batch = _make_batch()
        batch[1]["camera_pose"] = torch.eye(4).repeat(1, 1, 1)
        batch[1]["camera_pose"][:, 0, 3] = 1.75
        metric_gt = Pi3Loss(scale_mode="metric").prepare_gt(batch)

        self.assertAlmostEqual(
            float(metric_gt["camera_poses"][0, 1, 0, 3]),
            1.75,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
