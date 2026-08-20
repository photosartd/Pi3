import unittest

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from scripts.analyze_megapose_gso_focal_capacity import (
    PoseTrack,
    Track,
    cameras_match,
    count_positive_reference_capacity,
    count_render_to_scene_capacity,
    count_scene_to_scene_capacity,
    effective_pi3_intrinsics,
    make_camera_profile,
    object_view_direction,
)


class MegaPoseGSOFocalCapacityTest(unittest.TestCase):
    target = (336, 252)

    @staticmethod
    def camera(focal, *, cx=360.0, cy=270.0):
        K = np.asarray(
            [[focal, 0.0, cx], [0.0, focal * 4.0 / 3.0, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return make_camera_profile(K, 720, 540, (336, 252), label="test")

    def test_effective_intrinsics_preserve_centered_normalized_focal(self):
        K = np.asarray(
            [[720.0, 0.0, 360.0], [0.0, 720.0, 270.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        final = effective_pi3_intrinsics(K, 720, 540, self.target)

        self.assertAlmostEqual(final[0, 0] / 336.0, 1.0, places=6)
        self.assertAlmostEqual(final[1, 1] / 252.0, 4.0 / 3.0, places=6)

    def test_object_view_direction_uses_camera_center_in_object_frame(self):
        T_C_O = np.eye(4, dtype=np.float64)
        T_C_O[:3, 3] = [0.0, 0.0, 2.0]
        np.testing.assert_allclose(
            object_view_direction(T_C_O), [0.0, 0.0, -1.0]
        )

    def test_metadata_intrinsics_match_real_pi3_crop_resize(self):
        K = np.asarray(
            [
                [643.9628, 0.0, 365.9187],
                [0.0, 645.2667, 272.3051],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected = effective_pi3_intrinsics(K, 720, 540, self.target)
        dataset = BaseDataset(resolution=[self.target], aug_crop=False, aug_focal=False)
        image = Image.new("RGB", (720, 540))
        depth = np.ones((540, 720), dtype=np.float32)
        _, _, actual = dataset._crop_resize_if_necessary(
            image,
            depth,
            K.copy(),
            self.target,
        )[:3]

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)

    def test_focal_match_uses_both_axes_and_principal_point(self):
        baseline = self.camera(720.0)
        close = self.camera(720.0 * np.exp(0.04))
        far = self.camera(720.0 * np.exp(0.06))
        shifted = self.camera(720.0, cx=340.0)

        self.assertTrue(
            cameras_match(
                baseline,
                close,
                focal_log_tolerance=0.05,
                principal_point_tolerance=0.01,
                comparison_space="postprocess",
            )
        )
        self.assertFalse(
            cameras_match(
                baseline,
                far,
                focal_log_tolerance=0.05,
                principal_point_tolerance=0.01,
                comparison_space="postprocess",
            )
        )
        self.assertFalse(
            cameras_match(
                baseline,
                shifted,
                focal_log_tolerance=0.05,
                principal_point_tolerance=0.01,
                comparison_space="raw",
            )
        )

    def test_render_to_scene_counts_tracks_not_frame_combinations(self):
        cameras = {1: self.camera(720.0), 2: self.camera(720.0 * np.exp(0.08))}
        tracks = [
            Track(1, 1, 0, query_view_count=6, reference_view_count=1),
            Track(1, 1, 1, query_view_count=2, reference_view_count=0),
            Track(1, 2, 0, query_view_count=6, reference_view_count=0),
        ]
        render_camera = self.camera(720.0)
        result = count_render_to_scene_capacity(
            tracks,
            cameras,
            {1: 10},
            {1: render_camera},
            (5,),
            (1, 5),
            focal_log_tolerance=0.05,
            principal_point_tolerance=0.01,
            comparison_space="postprocess",
        )

        np.testing.assert_array_equal(result.current, [[3, 2]])
        np.testing.assert_array_equal(result.matched, [[2, 1]])
        np.testing.assert_array_equal(result.current_scene_pairs, [[2, 2]])
        np.testing.assert_array_equal(result.matched_scene_pairs, [[1, 1]])

    def test_scene_to_scene_counts_ordered_cross_scene_track_pairs(self):
        cameras = {
            1: self.camera(720.0),
            2: self.camera(720.0 * np.exp(0.04)),
            3: self.camera(720.0 * np.exp(0.10)),
        }
        tracks = [
            Track(1, 1, 0, query_view_count=6, reference_view_count=6),
            Track(1, 1, 1, query_view_count=2, reference_view_count=2),
            Track(1, 2, 0, query_view_count=6, reference_view_count=1),
            Track(1, 3, 0, query_view_count=6, reference_view_count=0),
        ]
        result = count_scene_to_scene_capacity(
            tracks,
            cameras,
            (5,),
            (1,),
            focal_log_tolerance=0.05,
            principal_point_tolerance=0.01,
            comparison_space="postprocess",
        )

        # The current denominator has three reference-capable tracks. The 0.4
        # reference threshold leaves only scene 1 reference-capable, so the
        # matched numerator contains only scene 1 -> scene 2.
        np.testing.assert_array_equal(result.current, [[8]])
        np.testing.assert_array_equal(result.matched, [[1]])
        np.testing.assert_array_equal(result.current_scene_pairs, [[6]])
        np.testing.assert_array_equal(result.matched_scene_pairs, [[1]])

    def test_positive_references_stay_in_one_track_and_focal_is_separate(self):
        query_direction = np.asarray([[0.0, 0.0, 1.0]])
        far_direction = np.asarray([0.0, 1.0, 0.0])
        close_direction = np.asarray(
            [0.0, np.sin(np.deg2rad(5.0)), np.cos(np.deg2rad(5.0))]
        )
        tracks = [
            PoseTrack(1, 1, 0, query_direction, np.empty((0, 3))),
            PoseTrack(
                1,
                2,
                0,
                np.asarray([far_direction]),
                np.stack(
                    [close_direction, far_direction, far_direction, far_direction, far_direction]
                ),
            ),
        ]
        cameras = {
            1: self.camera(720.0),
            2: self.camera(720.0 * np.exp(0.20)),
        }
        result = count_positive_reference_capacity(
            [tracks],
            cameras,
            (2, 5),
            positive_view_angle_degrees=10.0,
            minimum_positive_references=1,
            focal_log_tolerance=0.10,
            principal_point_tolerance=0.01,
            comparison_space="postprocess",
        )

        # The query in scene 1 can use the five-frame track from scene 2, and
        # one of those five frames is a positive. The scene focal profiles are
        # deliberately incompatible, so angle matching does not imply focal matching.
        np.testing.assert_array_equal(result.all_anchor_pairs, [1, 1])
        np.testing.assert_array_equal(result.positive_anchor_pairs, [1, 1])
        np.testing.assert_array_equal(result.positive_focal_anchor_pairs, [0, 0])
        np.testing.assert_array_equal(result.positive_query_frames, [1, 1])
        np.testing.assert_array_equal(result.positive_track_pairs, [1, 1])


if __name__ == "__main__":
    unittest.main()
