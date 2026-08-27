import unittest

import torch

from pi3.metrics.metric_depth import MetricDepthConditioningMetric
from pi3.models.depth_conditioning import FactoredMetricDepthConditioner


class FactoredMetricDepthConditionerTest(unittest.TestCase):
    def test_zero_initialization_is_exact_noop_and_scale_is_metric(self):
        module = FactoredMetricDepthConditioner(
            embed_dim=16,
            patch_size=14,
            scale_hidden_dim=8,
            min_valid_pixels=4,
            reference_probability=1.0,
            query_probability=0.0,
        )
        module.eval()
        depth = torch.stack(
            (torch.full((28, 28), 1.5), torch.full((28, 28), 2.5))
        ).unsqueeze(0)
        valid = torch.ones_like(depth, dtype=torch.bool)
        output = module(
            depth_m=depth,
            valid_mask=valid,
            is_reference=torch.tensor([[True, False]]),
            is_query=torch.tensor([[False, True]]),
        )
        self.assertEqual(tuple(output.tokens.shape), (2, 4, 16))
        self.assertEqual(float(output.tokens.abs().sum()), 0.0)
        self.assertEqual(output.known.tolist(), [[True, False]])
        torch.testing.assert_close(
            output.scale_m, torch.tensor([[1.5, 2.5]])
        )

    def test_unknown_views_remain_gated_after_projection_bias_trains(self):
        module = FactoredMetricDepthConditioner(
            embed_dim=8,
            patch_size=14,
            scale_hidden_dim=4,
            min_valid_pixels=1,
        )
        module.eval()
        with torch.no_grad():
            module.spatial_embed.proj.bias.fill_(2.0)
            module.scale_embed[-1].bias.fill_(3.0)
        output = module(
            depth_m=torch.ones(1, 2, 28, 28),
            valid_mask=torch.ones(1, 2, 28, 28, dtype=torch.bool),
            is_reference=torch.tensor([[True, False]]),
            is_query=torch.tensor([[False, True]]),
        )
        self.assertGreater(float(output.tokens[0].abs().sum()), 0.0)
        self.assertEqual(float(output.tokens[1].abs().sum()), 0.0)

    def test_query_conditioning_is_independently_configurable(self):
        module = FactoredMetricDepthConditioner(
            embed_dim=8,
            patch_size=14,
            scale_hidden_dim=4,
            min_valid_pixels=1,
            reference_probability=0.0,
            query_probability=1.0,
            eval_reference_probability=0.0,
            eval_query_probability=1.0,
        )
        module.eval()
        output = module(
            depth_m=torch.ones(1, 2, 28, 28),
            valid_mask=torch.ones(1, 2, 28, 28, dtype=torch.bool),
            is_reference=torch.tensor([[True, False]]),
            is_query=torch.tensor([[False, True]]),
        )
        self.assertEqual(output.known.tolist(), [[False, True]])
        self.assertEqual(
            float(
                output.statistics[
                    "metric_depth_reference_conditioned_fraction"
                ]
            ),
            0.0,
        )
        self.assertEqual(
            float(
                output.statistics[
                    "metric_depth_query_conditioned_fraction"
                ]
            ),
            1.0,
        )


class MetricDepthConditioningMetricTest(unittest.TestCase):
    def test_reports_reference_conditioned_and_query_unconditioned(self):
        height = width = 14
        intrinsics = torch.tensor(
            [[[10.0, 0.0, 6.5], [0.0, 10.0, 6.5], [0.0, 0.0, 1.0]]]
        )

        def view(depth, *, reference):
            return {
                "img": torch.zeros(1, 3, height, width),
                "depthmap": torch.full((1, height, width), depth),
                "valid_mask": torch.ones(1, height, width, dtype=torch.bool),
                "camera_intrinsics": intrinsics.clone(),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "pts3d": torch.zeros(1, height, width, 3),
                "view_role": ["reference" if reference else "query"],
                "is_reference": torch.tensor([reference]),
                "is_query": torch.tensor([not reference]),
            }

        batch = [view(1.0, reference=True), view(2.0, reference=False)]
        local = torch.zeros(1, 2, height, width, 3)
        local[:, 0, ..., 2] = 1.0
        local[:, 1, ..., 2] = 2.2
        prediction = {
            "local_points": local,
            "metric_depth_conditioning_known": torch.tensor([[True, False]]),
        }
        metric = MetricDepthConditioningMetric()
        metric.update(prediction, batch)
        values = metric.compute()
        self.assertEqual(values["reference/conditioned/views"], 1.0)
        self.assertEqual(values["query/unconditioned/views"], 1.0)
        self.assertAlmostEqual(
            values["reference/conditioned/z_mae_mean_m_mean"], 0.0
        )
        self.assertAlmostEqual(
            values["query/unconditioned/z_mae_mean_m_mean"], 0.2, places=5
        )

    def test_reports_query_conditioned_partition(self):
        height = width = 14
        intrinsics = torch.tensor(
            [[[10.0, 0.0, 6.5], [0.0, 10.0, 6.5], [0.0, 0.0, 1.0]]]
        )
        batch = [
            {
                "img": torch.zeros(1, 3, height, width),
                "depthmap": torch.full((1, height, width), 2.0),
                "valid_mask": torch.ones(
                    1, height, width, dtype=torch.bool
                ),
                "camera_intrinsics": intrinsics,
                "is_reference": torch.tensor([False]),
                "is_query": torch.tensor([True]),
            }
        ]
        local = torch.zeros(1, 1, height, width, 3)
        local[..., 2] = 2.0
        metric = MetricDepthConditioningMetric()
        metric.update(
            {
                "local_points": local,
                "metric_depth_conditioning_known": torch.tensor([[True]]),
            },
            batch,
        )
        values = metric.compute()
        self.assertEqual(values["query/conditioned/views"], 1.0)
        self.assertEqual(values["query/unconditioned/views"], 0.0)
        self.assertEqual(
            values["query/conditioned/z_mae_mean_m_mean"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
