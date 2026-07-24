import unittest
from types import MethodType

import numpy as np

from datasets.lmgeo_recenter import (
    LMGeoRecenterZoomSequenceDataset,
    compose_recentered_object_pose,
    corrected_source_z_depth,
    rotation_to_optical_axis,
    warp_recenter_zoom_view,
)


class LMGeoRecenterGeometryTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array(
            [
                [500.0, 0.0, 320.0],
                [0.0, 500.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def test_rotation_maps_bbox_center_ray_to_optical_axis(self):
        center_xy = np.array([430.0, 180.0], dtype=np.float32)
        R = rotation_to_optical_axis(self.K, center_xy)

        ray = np.linalg.inv(self.K) @ np.array([center_xy[0], center_xy[1], 1.0])
        ray = ray / np.linalg.norm(ray)
        mapped = R @ ray

        self.assertGreater(mapped[2], 0.999)
        self.assertAlmostEqual(float(mapped[0]), 0.0, places=5)
        self.assertAlmostEqual(float(mapped[1]), 0.0, places=5)

    def test_pose_composition_left_multiplies_camera_frame_rotation(self):
        T_C_O = np.eye(4, dtype=np.float32)
        T_C_O[:3, 3] = np.array([0.1, -0.2, 1.5], dtype=np.float32)
        R = rotation_to_optical_axis(self.K, np.array([430.0, 240.0], dtype=np.float32))

        T_Cnew_O = compose_recentered_object_pose(T_C_O, R)

        np.testing.assert_allclose(T_Cnew_O[:3, :3], R @ T_C_O[:3, :3], atol=1e-6)
        np.testing.assert_allclose(T_Cnew_O[:3, 3], R @ T_C_O[:3, 3], atol=1e-6)

    def test_corrected_source_z_depth_uses_new_camera_z(self):
        depth = np.full((5, 7), 2.0, dtype=np.float32)
        angle = np.deg2rad(20.0)
        R = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=np.float32,
        )

        corrected, valid = corrected_source_z_depth(depth, self.K, R)

        self.assertTrue(valid.all())
        pixel = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        expected = 2.0 * float(R[2] @ (np.linalg.inv(self.K) @ pixel))
        self.assertAlmostEqual(float(corrected[2, 3]), expected, places=6)

    def test_warp_recenter_zoom_updates_pose_intrinsics_depth_and_mask(self):
        height, width = 480, 640
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=bool)

        bbox = [410.0, 210.0, 50.0, 60.0]
        x0, y0, bw, bh = [int(v) for v in bbox]
        rgb[y0 : y0 + bh, x0 : x0 + bw] = np.array([200, 40, 30], dtype=np.uint8)
        depth[y0 : y0 + bh, x0 : x0 + bw] = 1.7
        mask[y0 : y0 + bh, x0 : x0 + bw] = True

        T_C_O = np.eye(4, dtype=np.float32)
        T_C_O[:3, 3] = np.array([0.2, 0.0, 1.7], dtype=np.float32)

        rgb_w, depth_w, mask_w, K_w, T_w, camera_pose_w, meta = warp_recenter_zoom_view(
            rgb=rgb,
            depthmap=depth,
            mask=mask,
            intrinsics=self.K,
            T_C_O=T_C_O,
            bbox=bbox,
            zoom_target_fraction=0.5,
            zoom_max=3.0,
            min_valid_depth_pixels=10,
        )

        self.assertEqual(rgb_w.shape, rgb.shape)
        self.assertEqual(depth_w.shape, depth.shape)
        self.assertEqual(mask_w.shape, mask.shape)
        self.assertGreater(int((depth_w > 0).sum()), 10)
        self.assertGreater(float(K_w[0, 0]), float(self.K[0, 0]))
        self.assertTrue(meta["query_recenter_applied"])
        self.assertGreater(float(meta["query_recenter_zoom"]), 1.0)
        np.testing.assert_allclose(camera_pose_w, np.linalg.inv(T_w), atol=1e-5)

        ys, xs = np.nonzero(mask_w)
        self.assertLess(abs(float(xs.mean()) - float(K_w[0, 2])), 4.0)
        self.assertLess(abs(float(ys.mean()) - float(K_w[1, 2])), 4.0)

    def test_query_recenter_valid_depth_is_not_limited_to_object_mask(self):
        height, width = 480, 640
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.full((height, width), 2.0, dtype=np.float32)
        mask = np.zeros((height, width), dtype=bool)
        bbox = [410.0, 210.0, 50.0, 60.0]
        x0, y0, bw, bh = [int(v) for v in bbox]
        mask[y0 : y0 + bh, x0 : x0 + bw] = True
        T_C_O = np.eye(4, dtype=np.float32)

        _, depth_w, mask_w, _, _, _, _ = warp_recenter_zoom_view(
            rgb=rgb,
            depthmap=depth,
            mask=mask,
            intrinsics=self.K,
            T_C_O=T_C_O,
            bbox=bbox,
            zoom_target_fraction=0.5,
            zoom_max=3.0,
            min_valid_depth_pixels=10,
        )

        self.assertGreater(int((depth_w > 0).sum()), int(mask_w.sum()) * 10)

    def test_mixin_leaves_reference_views_unchanged(self):
        dataset = object.__new__(LMGeoRecenterZoomSequenceDataset)
        dataset.query_recenter_bbox_key = "bbox_obj"
        dataset.query_recenter_zoom_target_fraction = 0.55
        dataset.query_recenter_zoom_max = 4.0
        dataset.query_recenter_zoom_min = 1.0
        dataset.query_recenter_depth_interpolation = "nearest"
        dataset.query_recenter_min_valid_depth_pixels = 1
        dataset.query_recenter_include_original_query_view = False

        rgb = np.zeros((8, 10, 3), dtype=np.uint8)
        depth = np.ones((8, 10), dtype=np.float32)
        mask = np.ones((8, 10), dtype=bool)
        T_C_O = np.eye(4, dtype=np.float32)

        rgb_out, depth_out, mask_out, K_out, T_out, camera_pose_out, meta = dataset._maybe_transform_raw_view(
            record={"bbox_obj": [1.0, 1.0, 4.0, 4.0]},
            rgb=rgb,
            depthmap=depth,
            mask=mask,
            intrinsics=self.K,
            T_C_O=T_C_O,
            camera_pose=np.linalg.inv(T_C_O).astype(np.float32),
            view_role="reference",
        )

        self.assertFalse(meta["query_recenter_applied"])
        self.assertIs(rgb_out, rgb)
        self.assertIs(depth_out, depth)
        self.assertIs(mask_out, mask)
        np.testing.assert_allclose(K_out, self.K)
        np.testing.assert_allclose(T_out, T_C_O)
        np.testing.assert_allclose(camera_pose_out, np.linalg.inv(T_C_O))

    def test_paired_mixin_puts_collatable_geometry_defaults_on_references(self):
        dataset = object.__new__(LMGeoRecenterZoomSequenceDataset)
        dataset.query_recenter_include_original_query_view = True
        rgb = np.zeros((8, 10, 3), dtype=np.uint8)
        depth = np.ones((8, 10), dtype=np.float32)
        mask = np.ones((8, 10), dtype=bool)
        T_C_O = np.eye(4, dtype=np.float32)

        *_, meta = dataset._maybe_transform_raw_view(
            record={},
            rgb=rgb,
            depthmap=depth,
            mask=mask,
            intrinsics=self.K,
            T_C_O=T_C_O,
            camera_pose=T_C_O,
            view_role="reference",
        )

        np.testing.assert_allclose(
            meta["query_recenter_R_old_to_new"],
            np.eye(3),
        )
        np.testing.assert_allclose(
            meta["query_crop_from_original_homography"],
            np.eye(3),
        )
        np.testing.assert_allclose(meta["query_original_T_C_O"], T_C_O)

    def test_mixin_does_not_apply_baseline_center_crop_filter_to_queries(self):
        dataset = object.__new__(LMGeoRecenterZoomSequenceDataset)

        self.assertTrue(dataset._record_passes_center_crop({"bbox_obj": [-100.0, 20.0, 5.0, 5.0]}))

    def test_mixin_keeps_full_scene_depth_for_queries_only(self):
        dataset = object.__new__(LMGeoRecenterZoomSequenceDataset)
        dataset.depth_masking = True

        self.assertFalse(dataset._should_depth_mask_view(view_role="query", reference_source="query"))
        self.assertTrue(dataset._should_depth_mask_view(view_role="reference", reference_source="render"))

    def test_paired_counting_treats_one_query_record_as_two_model_views(self):
        dataset = object.__new__(LMGeoRecenterZoomSequenceDataset)
        dataset.query_recenter_include_original_query_view = True
        dataset.num_reference_range = (2, 16)
        dataset.num_query_range = (1, 1)
        dataset.allow_repeat = True
        rng = np.random.default_rng(7)

        self.assertEqual(
            dataset._sample_counts(4, [object()], [object()], rng),
            (2, 1),
        )
        self.assertEqual(
            dataset._sample_counts(18, [object()], [object()], rng),
            (16, 1),
        )
        with self.assertRaisesRegex(ValueError, "paired query-record range"):
            dataset._sample_counts(3, [object()], [object()], rng)

    def test_paired_query_expansion_preserves_both_cameras_and_final_homography(self):
        dataset = object.__new__(LMGeoRecenterZoomSequenceDataset)
        dataset.query_recenter_include_original_query_view = True
        dataset.query_rgb_masking = False
        angle = np.deg2rad(12.0)
        R_old_to_new = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=np.float32,
        )
        K_original = np.array(
            [[100.0, 0.0, 14.0], [0.0, 105.0, 7.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        K_crop = np.array(
            [[180.0, 0.0, 14.0], [0.0, 185.0, 7.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        T_C_original_O = np.eye(4, dtype=np.float32)
        T_C_original_O[:3, 3] = np.array([0.1, -0.2, 1.4], dtype=np.float32)
        T_C_crop_O = compose_recentered_object_pose(
            T_C_original_O,
            R_old_to_new,
        )
        load_calls = []

        def fake_load_view(self, record, rgb_masking, view_role):
            is_original = bool(record.get("_lmgeo_skip_query_recenter", False))
            load_calls.append(is_original)
            T_C_O = T_C_original_O if is_original else T_C_crop_O
            view = {
                "camera_intrinsics": (
                    K_original.copy() if is_original else K_crop.copy()
                ),
                "T_C_O": T_C_O.copy(),
                "camera_pose": np.linalg.inv(T_C_O).astype(np.float32),
                "view_role": view_role,
                "is_reference": False,
                "is_query": True,
                "is_query_context": False,
                "is_cropped_query": False,
                "is_original_query": False,
                "query_pair_index": np.int64(-1),
                "query_recenter_applied": not is_original,
            }
            if not is_original:
                view["query_recenter_R_old_to_new"] = R_old_to_new.copy()
            return view

        dataset._load_view = MethodType(fake_load_view, dataset)
        crop, original = dataset._load_query_views({"record": 1})

        self.assertEqual(load_calls, [False, True])
        self.assertTrue(crop["is_query"])
        self.assertTrue(crop["is_cropped_query"])
        self.assertFalse(crop["is_original_query"])
        self.assertFalse(crop["is_query_context"])
        self.assertFalse(original["is_query"])
        self.assertFalse(original["is_cropped_query"])
        self.assertTrue(original["is_original_query"])
        self.assertTrue(original["is_query_context"])
        self.assertEqual(original["view_role"], "query_context")
        np.testing.assert_allclose(crop["T_C_O"], T_C_crop_O, atol=1e-6)
        np.testing.assert_allclose(original["T_C_O"], T_C_original_O, atol=1e-6)
        np.testing.assert_allclose(
            crop["query_original_T_C_O"],
            T_C_original_O,
            atol=1e-6,
        )
        expected_H = K_crop @ R_old_to_new @ np.linalg.inv(K_original)
        expected_H /= expected_H[2, 2]
        np.testing.assert_allclose(
            crop["query_crop_from_original_homography"],
            expected_H,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            original["query_crop_from_original_homography"],
            expected_H,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
