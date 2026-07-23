import unittest

import numpy as np
import torch
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from pi3.models.layers.camera_head import VisibilityGuidedTokenPool


class VisibilityGuidedTokenPoolTest(unittest.TestCase):
    def test_alpha_zero_matches_uniform_pooling(self):
        feat = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
        mask = torch.zeros(2, 6, dtype=torch.float32)
        mask[:, 2] = 1.0
        pool = VisibilityGuidedTokenPool()

        pooled, stats = pool(feat, 2, 3, visibility_patch_mask=mask, alpha=0.0)

        torch.testing.assert_close(pooled, feat.mean(dim=1))
        self.assertEqual(float(stats["visibility_pool_alpha"]), 0.0)

    def test_all_one_mask_matches_uniform_pooling(self):
        feat = torch.randn(2, 6, 4)
        mask = torch.ones(2, 6)
        pool = VisibilityGuidedTokenPool()

        pooled, stats = pool(feat, 2, 3, visibility_patch_mask=mask, alpha=1.0)

        torch.testing.assert_close(pooled, feat.mean(dim=1), atol=1e-6, rtol=1e-6)
        self.assertEqual(float(stats["visibility_pool_nonempty_fraction"]), 1.0)

    def test_partial_mask_selects_visible_tokens_and_empty_rows_fallback(self):
        feat = torch.arange(2 * 6 * 1, dtype=torch.float32).reshape(2, 6, 1)
        mask = torch.zeros(2, 6)
        mask[0, [1, 3]] = 1.0
        pool = VisibilityGuidedTokenPool()

        pooled, stats = pool(feat, 2, 3, visibility_patch_mask=mask, alpha=1.0)

        self.assertAlmostEqual(float(pooled[0, 0]), 2.0)
        self.assertAlmostEqual(float(pooled[1, 0]), float(feat[1].mean()))
        self.assertAlmostEqual(float(stats["visibility_pool_nonempty_fraction"]), 0.5)


class ObjectVisibilityMaskPreprocessingTest(unittest.TestCase):
    def test_crop_resize_carries_mask_like_far_mask(self):
        dataset = BaseDataset(resolution=[(4, 4)], aug_crop=False, aug_focal=False)
        image = Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8))
        depth = np.ones((6, 8), dtype=np.float32)
        intrinsics = np.array(
            [
                [6.0, 0.0, 4.0],
                [0.0, 6.0, 3.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[2:4, 3:5] = 1

        image_out, depth_out, intrinsics_out, mask_out = dataset._crop_resize_if_necessary(
            image,
            depth,
            intrinsics,
            (4, 4),
            rng=np.random.default_rng(0),
            far_mask=mask,
        )

        self.assertEqual(image_out.size, (4, 4))
        self.assertEqual(depth_out.shape, (4, 4))
        self.assertEqual(mask_out.shape, (4, 4))
        self.assertGreater(int(mask_out.sum()), 0)
        self.assertTrue(np.isfinite(intrinsics_out).all())


if __name__ == "__main__":
    unittest.main()
