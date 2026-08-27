import unittest

import numpy as np

from datasets.object_centric import PlannedObjectCropTransform
from datasets.virtual_camera import (
    VirtualCameraRectificationConfig,
    VirtualCameraRectifier,
    bbox_from_record,
)


class VirtualCameraTest(unittest.TestCase):
    def setUp(self):
        self.height, self.width = 120, 160
        self.K = np.asarray(
            [[120.0, 0.0, 82.0], [0.0, 118.0, 58.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.bbox = np.asarray([105.0, 43.0, 20.0, 24.0], dtype=np.float32)
        self.rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.depth = np.zeros((self.height, self.width), dtype=np.float32)
        self.mask = np.zeros((self.height, self.width), dtype=bool)
        x, y, width, height = self.bbox.astype(int)
        self.rgb[y : y + height, x : x + width] = (200, 80, 30)
        self.depth[y : y + height, x : x + width] = 1.4
        self.mask[y : y + height, x : x + width] = True
        self.T_C_O = np.eye(4, dtype=np.float32)
        self.T_C_O[:3, 3] = (0.25, 0.0, 1.4)

    def _rectifier(self, **overrides):
        values = {
            "roles": ("query",),
            "zoom_range": (1.0, 5.0),
            "eval_zoom": 4.0,
            "bbox_margin_fraction": 0.05,
            "min_valid_depth_pixels": 8,
        }
        values.update(overrides)
        return VirtualCameraRectifier(VirtualCameraRectificationConfig(**values))

    def test_bbox_reader_accepts_geometry_index_columns(self):
        record = {
            "bbox_obj_x": 1,
            "bbox_obj_y": 2,
            "bbox_obj_w": 30,
            "bbox_obj_h": 40,
        }
        np.testing.assert_allclose(
            bbox_from_record(record),
            [1.0, 2.0, 30.0, 40.0],
        )

    def test_reference_is_unchanged_and_gets_collatable_defaults(self):
        outputs = self._rectifier().transform(
            record={"bbox_obj": self.bbox},
            rgb=self.rgb,
            depthmap=self.depth,
            mask=self.mask,
            intrinsics=self.K,
            T_C_O=self.T_C_O,
            camera_pose=np.linalg.inv(self.T_C_O).astype(np.float32),
            view_role="reference",
            rng=np.random.default_rng(1),
            mode="train",
        )
        rgb, depth, mask, K, T_C_O, _, meta = outputs
        self.assertIs(rgb, self.rgb)
        self.assertIs(depth, self.depth)
        self.assertIs(mask, self.mask)
        np.testing.assert_allclose(K, self.K)
        np.testing.assert_allclose(T_C_O, self.T_C_O)
        self.assertFalse(meta["virtual_camera_applied"])

    def test_query_is_centered_safe_and_pose_consistent(self):
        outputs = self._rectifier().transform(
            record={"bbox_obj": self.bbox},
            rgb=self.rgb,
            depthmap=self.depth,
            mask=self.mask,
            intrinsics=self.K,
            T_C_O=self.T_C_O,
            camera_pose=np.linalg.inv(self.T_C_O).astype(np.float32),
            view_role="query",
            rng=np.random.default_rng(2),
            mode="val",
        )
        _, depth, mask, K, T_C_O, camera_pose, meta = outputs
        self.assertTrue(meta["virtual_camera_applied"])
        self.assertLessEqual(
            float(meta["virtual_camera_zoom"]),
            float(meta["virtual_camera_zoom_safe_max"]) + 1e-6,
        )
        self.assertAlmostEqual(float(K[0, 2]), 0.5 * (self.width - 1), places=5)
        self.assertAlmostEqual(float(K[1, 2]), 0.5 * (self.height - 1), places=5)
        self.assertGreater(int(mask.sum()), 0)
        self.assertFalse(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        self.assertTrue(np.all(depth[~(depth > 0)] == 0.0))
        np.testing.assert_allclose(camera_pose, np.linalg.inv(T_C_O), atol=1e-5)
        source_center = -self.T_C_O[:3, :3].T @ self.T_C_O[:3, 3]
        virtual_center = -T_C_O[:3, :3].T @ T_C_O[:3, 3]
        np.testing.assert_allclose(virtual_center, source_center, atol=1e-5)

    def test_training_zoom_is_reproducible_and_diverse(self):
        config = self._rectifier().config
        first = np.random.default_rng(7)
        second = np.random.default_rng(7)
        values_a = [config.sample_zoom(rng=first, mode="train") for _ in range(64)]
        values_b = [config.sample_zoom(rng=second, mode="train") for _ in range(64)]
        np.testing.assert_allclose(values_a, values_b)
        self.assertGreater(max(values_a), 4.0)
        self.assertLess(min(values_a), 1.2)
        self.assertGreater(len({round(value, 3) for value in values_a}), 50)
        self.assertEqual(config.sample_zoom(rng=first, mode="val"), 4.0)

    def test_safe_fill_policy_tracks_object_preserving_limit(self):
        rectifier = self._rectifier(
            zoom_policy="safe_fill",
            safe_fill_fraction_range=(0.8, 1.0),
            eval_safe_fill_fraction=0.9,
        )
        outputs = rectifier.transform(
            record={"bbox_obj": self.bbox},
            rgb=self.rgb,
            depthmap=self.depth,
            mask=self.mask,
            intrinsics=self.K,
            T_C_O=self.T_C_O,
            camera_pose=np.linalg.inv(self.T_C_O).astype(np.float32),
            view_role="query",
            rng=np.random.default_rng(5),
            mode="val",
        )
        _, _, mask, _, _, _, metadata = outputs
        self.assertAlmostEqual(
            float(metadata["virtual_camera_safe_fill_fraction_requested"]),
            0.9,
            places=6,
        )
        self.assertAlmostEqual(
            float(metadata["virtual_camera_safe_fill_fraction"]),
            0.9,
            places=5,
        )
        self.assertAlmostEqual(
            float(metadata["virtual_camera_zoom"]),
            0.9 * float(metadata["virtual_camera_zoom_safe_max"]),
            places=5,
        )
        self.assertFalse(
            mask[0].any()
            or mask[-1].any()
            or mask[:, 0].any()
            or mask[:, -1].any()
        )

    def test_virtual_query_bypasses_only_its_planned_crop(self):
        state = {
            "record": {"planned_object_crop": {"format": "not-read"}},
            "transform_meta": {"virtual_camera_replaces_planned_crop": True},
        }
        PlannedObjectCropTransform().apply(
            state,
            adapter=object(),
            request=object(),
            rng=np.random.default_rng(0),
        )
        self.assertFalse(state["transform_meta"]["object_crop_applied"])
        self.assertTrue(state["transform_meta"]["planned_object_crop_bypassed"])


if __name__ == "__main__":
    unittest.main()
