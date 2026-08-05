import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from datasets.lmgeo_dataset import LMGeoDataset
from datasets.object_centric import ObjectDatasetAdapter


class LMGeoAdapterTest(unittest.TestCase):
    def _dataset(self):
        dataset = LMGeoDataset.__new__(LMGeoDataset)
        dataset.dataset_label = "LMGeoSynthetic"
        dataset.object_id = 8
        dataset.query_scene_id = 3
        dataset.query_subscene_id = 1
        dataset.depth_unit_scale = 0.001
        dataset.depth_masking = True
        dataset.visibility_mask_conditioning = True
        dataset.condition_reference_visibility = True
        dataset.condition_query_visibility = False
        dataset.visibility_condition_corruption = "none"
        dataset.visibility_condition_shift_fraction = 0.5
        dataset.photometric_augmentation = False
        dataset._photometric_role_specs = {}
        dataset.aug_crop = False
        dataset.aug_focal = False
        dataset._current_resolution = [28, 28]
        dataset._rng = np.random.default_rng(12)
        return dataset

    def _record(self, directory):
        directory = Path(directory)
        rgb_path = directory / "rgb.png"
        depth_path = directory / "depth.png"
        mask_path = directory / "mask.png"

        rgb = np.full((28, 28, 3), 127, dtype=np.uint8)
        depth = np.full((28, 28), 1000, dtype=np.uint16)
        mask = np.zeros((28, 28), dtype=np.uint8)
        mask[7:21, 8:20] = 255
        Image.fromarray(rgb).save(rgb_path)
        cv2.imwrite(str(depth_path), depth)
        cv2.imwrite(str(mask_path), mask)

        T_C_O = np.eye(4, dtype=np.float32)
        T_C_O[2, 3] = 1.0
        return {
            "split": "train_pbr",
            "scene_dir": str(directory),
            "im_id": 4,
            "gt_id": 0,
            "object_id": 8,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "mask_path": str(mask_path),
            "K": np.array(
                [[20.0, 0.0, 14.0], [0.0, 20.0, 14.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            "image_size": (28, 28),
            "depth_scale": 1.0,
            "T_C_O": T_C_O,
            "camera_pose": np.linalg.inv(T_C_O).astype(np.float32),
            "visib_fract": 1.0,
            "query_scene_id": 3,
            "query_subscene_id": 1,
        }

    def test_lmgeo_is_an_extension_of_base_dataset(self):
        self.assertTrue(issubclass(LMGeoDataset, ObjectDatasetAdapter))

    def test_raw_adapter_converts_depth_to_meters(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset()
            raw = dataset.load_raw_object_view(self._record(directory))
            self.assertAlmostEqual(float(raw.depthmap.mean()), 1.0, places=6)
            np.testing.assert_allclose(
                raw.camera_pose,
                np.linalg.inv(raw.T_C_O),
                atol=1e-6,
            )

    def test_processor_keeps_mask_targets_separate_from_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset()
            record = self._record(directory)
            reference = dataset._load_view(
                record, rgb_masking=True, view_role="reference"
            )
            query = dataset._load_view(
                record, rgb_masking=False, view_role="query"
            )

            reference_mask = reference["object_visibility_mask"] > 0.5
            self.assertEqual(int((reference["depthmap"] > 0).sum()), int(reference_mask.sum()))
            self.assertEqual(
                int((reference["visibility_mask_condition"] > 0.5).sum()),
                int(reference_mask.sum()),
            )
            self.assertTrue(
                bool((reference["visibility_mask_known"] == 1.0).all())
            )
            self.assertEqual(
                float(query["visibility_mask_condition"].sum()), 0.0
            )
            self.assertEqual(float(query["visibility_mask_known"].sum()), 0.0)
            np.testing.assert_allclose(
                reference["camera_pose"],
                np.linalg.inv(reference["T_C_O"]),
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()

