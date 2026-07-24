import unittest

import torch

from pi3.models.dinov2.layers import PatchEmbed
from pi3.models.ray_conditioning import intrinsics_to_ray_map, pixel_center_grid


class RayConditioningTest(unittest.TestCase):
    def test_canonical_intrinsics_produce_expected_pixel_center_rays(self):
        intrinsics = torch.tensor(
            [[
                [100.0, 0.0, 20.0],
                [0.0, 80.0, 10.0],
                [0.0, 0.0, 1.0],
            ]]
        )
        rays = intrinsics_to_ray_map(intrinsics, 28, 42)

        self.assertEqual(tuple(rays.shape), (1, 28, 42, 2))
        self.assertAlmostEqual(float(rays[0, 0, 0, 0]), (0.5 - 20.0) / 100.0)
        self.assertAlmostEqual(float(rays[0, 0, 0, 1]), (0.5 - 10.0) / 80.0)

    def test_multiple_views_keep_their_own_intrinsics(self):
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
        intrinsics[0, 0, 0, 0] = 100.0
        intrinsics[0, 0, 1, 1] = 100.0
        intrinsics[0, 1, 0, 0] = 400.0
        intrinsics[0, 1, 1, 1] = 400.0

        rays = intrinsics_to_ray_map(intrinsics, 14, 14)

        self.assertEqual(tuple(rays.shape), (1, 2, 14, 14, 2))
        self.assertGreater(
            float(rays[0, 0, -1, -1].norm()),
            float(rays[0, 1, -1, -1].norm()),
        )

    def test_dynamic_patch_projection_matches_560x420_image_tokens(self):
        encoder = PatchEmbed(
            img_size=224,
            patch_size=14,
            in_chans=2,
            embed_dim=16,
        )
        torch.nn.init.zeros_(encoder.proj.weight)
        torch.nn.init.zeros_(encoder.proj.bias)
        ray_map = torch.randn(2, 2, 420, 560)

        tokens = encoder(ray_map)

        self.assertEqual(tuple(tokens.shape), (2, 30 * 40, 16))
        self.assertTrue(torch.equal(tokens, torch.zeros_like(tokens)))

    def test_exact_pi3_large_ray_parameter_count(self):
        encoder = PatchEmbed(
            img_size=224,
            patch_size=14,
            in_chans=2,
            embed_dim=1024,
        )
        parameter_count = sum(parameter.numel() for parameter in encoder.parameters())

        self.assertEqual(parameter_count, 402_432)

    def test_pixel_grid_uses_half_pixel_centers(self):
        grid = pixel_center_grid(2, 3, device=torch.device("cpu"))

        torch.testing.assert_close(grid[0, 0], torch.tensor([0.5, 0.5, 1.0]))
        torch.testing.assert_close(grid[-1, -1], torch.tensor([2.5, 1.5, 1.0]))

    def test_invalid_intrinsics_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ending in"):
            intrinsics_to_ray_map(torch.eye(4), 14, 14)


if __name__ == "__main__":
    unittest.main()
