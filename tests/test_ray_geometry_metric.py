import unittest

import torch

from pi3.metrics.ray_geometry import RayGeometryMetric
from pi3.models.ray_conditioning import intrinsics_to_ray_map


def make_view(
    intrinsics,
    valid_mask,
    *,
    is_reference=False,
    is_query=False,
    is_query_context=False,
):
    return {
        "camera_intrinsics": intrinsics[None],
        "valid_mask": valid_mask[None],
        "is_reference": torch.tensor([is_reference]),
        "is_query": torch.tensor([is_query]),
        "is_query_context": torch.tensor([is_query_context]),
    }


def points_from_intrinsics(intrinsics, height, width, depth):
    ray_xy = intrinsics_to_ray_map(intrinsics[None, None], height, width)[0, 0]
    depth_map = torch.full((height, width, 1), float(depth))
    return torch.cat((ray_xy * depth_map, depth_map), dim=-1)


class RayGeometryMetricTest(unittest.TestCase):
    def setUp(self):
        self.height = 14
        self.width = 28
        self.intrinsics = torch.tensor(
            [
                [120.0, 0.0, 13.0],
                [0.0, 100.0, 6.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.valid = torch.ones(self.height, self.width, dtype=torch.bool)

    def test_exact_points_have_zero_ray_and_intrinsics_error(self):
        points = points_from_intrinsics(
            self.intrinsics,
            self.height,
            self.width,
            depth=2.0,
        )
        prediction = {"local_points": points[None, None]}
        batch = [make_view(self.intrinsics, self.valid, is_query=True)]
        metric = RayGeometryMetric(min_fit_points=16)

        metric.update(prediction, batch, mode="val")
        stats = metric.compute()

        self.assertLess(stats["query/angular_mean_deg"], 1e-3)
        self.assertLess(stats["query/reprojection_mean_px"], 1e-4)
        self.assertLess(stats["query/fx_relative_error_mean"], 1e-5)
        self.assertLess(stats["query/fy_relative_error_mean"], 1e-5)
        self.assertLess(stats["query/cx_absolute_error_px_mean"], 1e-4)
        self.assertLess(stats["query/cy_absolute_error_px_mean"], 1e-4)
        self.assertEqual(stats["query/intrinsics_fit_valid_fraction"], 1.0)

    def test_metric_is_invariant_to_local_point_scale(self):
        points = points_from_intrinsics(
            self.intrinsics,
            self.height,
            self.width,
            depth=1.0,
        )
        batch = [make_view(self.intrinsics, self.valid, is_reference=True)]

        metric_a = RayGeometryMetric(min_fit_points=16)
        metric_a.update({"local_points": points[None, None]}, batch)
        metric_b = RayGeometryMetric(min_fit_points=16)
        metric_b.update({"local_points": (points * 7.0)[None, None]}, batch)

        stats_a = metric_a.compute()
        stats_b = metric_b.compute()
        self.assertAlmostEqual(
            stats_a["reference/angular_mean_deg"],
            stats_b["reference/angular_mean_deg"],
            places=6,
        )
        self.assertAlmostEqual(
            stats_a["reference/reprojection_mean_px"],
            stats_b["reference/reprojection_mean_px"],
            places=6,
        )

    def test_wrong_intrinsics_increase_errors(self):
        wrong = self.intrinsics.clone()
        wrong[0, 0] *= 2.0
        wrong[1, 1] *= 2.0
        points = points_from_intrinsics(
            wrong,
            self.height,
            self.width,
            depth=2.0,
        )
        prediction = {"local_points": points[None, None]}
        batch = [make_view(self.intrinsics, self.valid, is_query=True)]
        metric = RayGeometryMetric(min_fit_points=16)

        metric.update(prediction, batch)
        stats = metric.compute()

        self.assertGreater(stats["query/angular_mean_deg"], 0.1)
        self.assertGreater(stats["query/reprojection_mean_px"], 0.1)
        self.assertGreater(stats["query/fx_relative_error_mean"], 0.5)
        self.assertGreater(stats["query/fy_relative_error_mean"], 0.5)

    def test_reference_and_query_are_reported_separately(self):
        points = points_from_intrinsics(
            self.intrinsics,
            self.height,
            self.width,
            depth=2.0,
        )
        prediction = {"local_points": points[None, None].repeat(1, 2, 1, 1, 1)}
        batch = [
            make_view(self.intrinsics, self.valid, is_reference=True),
            make_view(self.intrinsics, self.valid, is_query=True),
        ]
        metric = RayGeometryMetric(min_fit_points=16)

        metric.update(prediction, batch)
        stats = metric.compute()

        self.assertEqual(stats["reference/views"], 1.0)
        self.assertEqual(stats["query/views"], 1.0)
        self.assertEqual(len(metric.values["reference"]["angular_mean_deg"]), 1)
        self.assertEqual(len(metric.values["query"]["angular_mean_deg"]), 1)
        self.assertEqual(len(metric.values["reference"]["reprojection_mean_px"]), 1)
        self.assertEqual(len(metric.values["query"]["reprojection_mean_px"]), 1)

    def test_original_query_context_is_reported_separately(self):
        points = points_from_intrinsics(
            self.intrinsics,
            self.height,
            self.width,
            depth=2.0,
        )
        batch = [
            make_view(
                self.intrinsics,
                self.valid,
                is_query_context=True,
            )
        ]
        metric = RayGeometryMetric(min_fit_points=16)

        metric.update({"local_points": points[None, None]}, batch)
        stats = metric.compute()

        self.assertEqual(stats["query_context/views"], 1.0)
        self.assertLess(stats["query_context/angular_mean_deg"], 1e-3)
        self.assertEqual(stats["query/views"], 0.0)


if __name__ == "__main__":
    unittest.main()
