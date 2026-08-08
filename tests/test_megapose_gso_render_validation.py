import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "preprocess"
    / "render"
    / "megapose_gso_validate.py"
)
SPEC = importlib.util.spec_from_file_location("megapose_gso_validate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
render_validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render_validation)


def encode_uncompressed_rle(mask: np.ndarray) -> dict:
    flat = np.asarray(mask, dtype=bool).reshape(-1, order="F")
    counts = []
    current = False
    count = 0
    for value in flat:
        if bool(value) == current:
            count += 1
        else:
            counts.append(count)
            count = 1
            current = bool(value)
    counts.append(count)
    return {"size": list(mask.shape), "counts": counts}


class MegaPoseGsoRenderValidationTest(unittest.TestCase):
    def test_uncompressed_rle_round_trip(self):
        mask = np.asarray(
            [[False, True, False], [True, True, False]], dtype=bool
        )
        decoded = render_validation.decode_uncompressed_rle(
            encode_uncompressed_rle(mask)
        )
        np.testing.assert_array_equal(decoded, mask)

    def test_geometry_metrics_use_full_mask_and_visible_depth(self):
        full = np.asarray(
            [[False, True, True], [False, True, True], [False, False, False]]
        )
        visible = full.copy()
        visible[0, 2] = False
        rendered = full.copy()
        source_depth = np.zeros((3, 3), dtype=np.float32)
        render_depth = np.zeros((3, 3), dtype=np.float32)
        source_depth[visible] = 1.0
        render_depth[rendered] = 1.002
        metrics = render_validation.geometry_metrics(
            rendered_mask=rendered,
            rendered_depth_m=render_depth,
            source_full_mask=full,
            source_visible_mask=visible,
            source_depth_m=source_depth,
        )
        self.assertEqual(metrics["full_mask_iou"], 1.0)
        self.assertEqual(metrics["visible_mask_render_recall"], 1.0)
        self.assertEqual(metrics["depth_comparison_pixels"], 3)
        self.assertAlmostEqual(metrics["depth_abs_median_m"], 0.002, places=6)


if __name__ == "__main__":
    unittest.main()
