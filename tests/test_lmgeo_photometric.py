import unittest

import numpy as np
from PIL import Image

from datasets.lmgeo_dataset import LMGeoSequenceDataset


class LMGeoPhotometricAugmentationTest(unittest.TestCase):
    def make_dataset_shell(self):
        dataset = object.__new__(LMGeoSequenceDataset)
        dataset.photometric_augmentation = True
        dataset.photometric_brightness = (0.7, 1.3)
        dataset.photometric_contrast = (0.7, 1.3)
        dataset.photometric_saturation = (0.7, 1.3)
        dataset.photometric_hue = (-0.1, 0.1)
        dataset.photometric_gamma = (0.7, 1.3)
        dataset.photometric_jpeg_prob = 1.0
        dataset.photometric_jpeg_quality = (20, 100)
        dataset.photometric_blur_prob = 1.0
        dataset.photometric_blur_resize_ratio = (0.25, 1.0)
        dataset._photometric_role_specs = {}
        return dataset

    def test_role_specs_are_sampled_independently_and_reused_within_role(self):
        dataset = self.make_dataset_shell()
        rng = np.random.default_rng(11)
        dataset._prepare_sample_photometric_augmentation(rng)

        reference_spec = dataset._photometric_role_specs["reference"]
        query_spec = dataset._photometric_role_specs["query"]
        self.assertNotEqual(reference_spec, query_spec)

        gradient = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        image = Image.fromarray(gradient, mode="RGB")

        reference_a = np.asarray(dataset._apply_sample_photometric_augmentation(image, "reference"))
        reference_b = np.asarray(dataset._apply_sample_photometric_augmentation(image, "reference"))
        query = np.asarray(dataset._apply_sample_photometric_augmentation(image, "query"))

        self.assertTrue(np.array_equal(reference_a, reference_b))
        self.assertFalse(np.array_equal(reference_a, query))

    def test_disabled_augmentation_leaves_image_object_unchanged(self):
        dataset = self.make_dataset_shell()
        dataset.photometric_augmentation = False
        dataset._prepare_sample_photometric_augmentation(np.random.default_rng(3))
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")

        self.assertIs(dataset._apply_sample_photometric_augmentation(image, "reference"), image)


if __name__ == "__main__":
    unittest.main()
