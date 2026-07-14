import unittest

import torch
from PIL import Image

from pi3.visualizations.correspondence import CorrespondenceVisualizer


def make_view(*, is_reference: bool, batch_size: int = 1, height: int = 28, width: int = 28):
    transform = torch.eye(4).repeat(batch_size, 1, 1)
    intrinsics = torch.tensor(
        [
            [20.0, 0.0, (width - 1) / 2.0],
            [0.0, 20.0, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    ).repeat(batch_size, 1, 1)
    image = torch.zeros(batch_size, 3, height, width)
    image[:, 0 if is_reference else 1] = 1.0
    return {
        "img": image,
        "depthmap": torch.ones(batch_size, height, width),
        "valid_mask": torch.ones(batch_size, height, width, dtype=torch.bool),
        "T_C_O": transform,
        "camera_intrinsics": intrinsics,
        "is_reference": torch.full((batch_size,), bool(is_reference)),
        "is_query": torch.full((batch_size,), not bool(is_reference)),
    }


class CorrespondenceVisualizerTest(unittest.TestCase):
    def test_render_returns_tensorboard_ready_image(self):
        batch = [make_view(is_reference=True), make_view(is_reference=False)]
        pred = {
            "dino_features": {
                "17": torch.randn(1, 2, 4, 8),
            }
        }
        visualizer = CorrespondenceVisualizer(patch_size=14, max_pairs=0, dino_layer=17)
        rendered = visualizer.render(pred, batch, batch_idx=0)
        self.assertIn("matches", rendered)
        self.assertIsInstance(rendered["matches"], Image.Image)
        self.assertGreater(rendered["matches"].width, 0)
        self.assertGreater(rendered["matches"].height, 0)


if __name__ == "__main__":
    unittest.main()
