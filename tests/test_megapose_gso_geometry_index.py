import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from datasets.preprocess.megapose_gso_geometry import (
    BuildSettings,
    FEATURE_COLUMNS,
    SCHEMA,
    collect_capacity_statistics,
    crop_feasibility,
    decode_uncompressed_rle,
    focal_band_feasible,
    object_view_direction,
    packed_union_curve,
    sample_mesh_surface,
)


class MegaPoseGSOGeometryIndexTest(unittest.TestCase):
    def test_area_weighted_surface_sampling_is_deterministic(self):
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [12, 0, 0], [10, 2, 0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        first = sample_mesh_surface(vertices, faces, 2000, seed=7)
        second = sample_mesh_surface(vertices, faces, 2000, seed=7)
        np.testing.assert_array_equal(first, second)
        # The second triangle has four times the area of the first.
        self.assertGreater(float(np.mean(first[:, 0] > 5)), 0.72)

    def test_rle_decoder_uses_coco_column_major_order(self):
        decoded = decode_uncompressed_rle({"size": [2, 3], "counts": [1, 2, 3]})
        np.testing.assert_array_equal(
            decoded,
            np.asarray([[False, True, False], [True, False, False]], dtype=np.uint8),
        )

    def test_view_direction_uses_object_frame_camera_center(self):
        pose = np.eye(4)
        pose[:3, 3] = [0, 0, 2]
        np.testing.assert_allclose(object_view_direction(pose), [0, 0, -1])

    def test_crop_interval_keeps_expanded_object_bbox(self):
        row = {"bbox_obj_x": 200, "bbox_obj_y": 150, "bbox_obj_w": 200, "bbox_obj_h": 100}
        K = np.asarray([[600, 0, 360], [0, 600, 270], [0, 0, 1]], dtype=float)
        result = crop_feasibility(
            row, K, width=720, height=540, aspect=4 / 3, margin_fraction=0.05
        )
        self.assertEqual(result["crop_feasible"], 1)
        self.assertAlmostEqual(result["crop_width_min"], 220.0)
        self.assertAlmostEqual(result["crop_width_max"], 720.0)
        self.assertGreater(result["max_norm_fx"], result["base_norm_fx"])

    def test_crop_interval_rejects_margin_outside_source(self):
        row = {"bbox_obj_x": 0, "bbox_obj_y": 20, "bbox_obj_w": 200, "bbox_obj_h": 100}
        K = np.eye(3)
        result = crop_feasibility(
            row, K, width=720, height=540, aspect=4 / 3, margin_fraction=0.05
        )
        self.assertEqual(result["crop_feasible"], 0)
        self.assertEqual(result["bbox_obj_touches_border"], 1)

    def test_packed_greedy_curve_selects_complementary_views(self):
        masks = np.packbits(
            np.asarray(
                [[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]],
                dtype=np.uint8,
            ),
            axis=1,
            bitorder="little",
        )
        selected, coverage = packed_union_curve(masks, 2)
        self.assertEqual(selected, [0, 2])
        self.assertEqual(coverage, [0.25, 0.5])

    @staticmethod
    def feature(low, high, shape=1.0):
        return {
            "base_norm_fx": low,
            "base_norm_fy": low / shape,
            "max_norm_fx": high,
            "max_norm_fy": high / shape,
            "crop_feasible": 1,
        }

    def test_independent_focal_band_and_common_target_are_distinct(self):
        query = self.feature(1.0, 1.0)
        references = [self.feature(0.90, 0.95), self.feature(1.05, 1.10)]
        self.assertTrue(
            focal_band_feasible(
                references, query, relative_tolerance=0.1, common_target=False
            )
        )
        self.assertFalse(
            focal_band_feasible(
                references, query, relative_tolerance=0.1, common_target=True
            )
        )

    def test_capacity_grid_counts_a_constructible_cross_scene_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "geometry.sqlite"
            bits = root / "geometry.bits"
            bits.write_bytes(bytes([0b00000011, 0b00001100, 0b00000001]))
            connection = sqlite3.connect(index)
            connection.executescript(SCHEMA)
            fingerprint = {
                "settings": {"surface_point_count": 8},
            }
            connection.executemany(
                "INSERT INTO metadata VALUES (?,?)",
                (
                    ("fingerprint", json.dumps(fingerprint)),
                    ("bits_path", str(bits)),
                    ("index_complete", "1"),
                ),
            )

            def row(feature_id, scene_id, view_id, observed, offset):
                value = {name: 0 for name in FEATURE_COLUMNS}
                value.update(
                    feature_id=feature_id,
                    frame_id=feature_id,
                    gt_id=0,
                    object_id=1,
                    scene_id=scene_id,
                    view_id=view_id,
                    visib_fract=0.8,
                    px_count_all=100,
                    px_count_valid=100,
                    px_count_visib=80,
                    depth_corrupt=0,
                    width=100,
                    height=100,
                    view_x=0.0,
                    view_y=0.0,
                    view_z=1.0,
                    fx=100.0,
                    fy=100.0,
                    cx=50.0,
                    cy=50.0,
                    bbox_obj_w=20.0,
                    bbox_obj_h=20.0,
                    bbox_visib_w=20.0,
                    bbox_visib_h=20.0,
                    crop_width_min=20.0,
                    crop_width_max=100.0,
                    crop_feasible=1,
                    base_norm_fx=1.0,
                    base_norm_fy=1.0,
                    max_norm_fx=5.0,
                    max_norm_fy=5.0,
                    surface_point_count=8,
                    observed_point_count=observed,
                    observed_fraction=observed / 8,
                    bit_offset=offset,
                )
                return tuple(value[name] for name in FEATURE_COLUMNS)

            connection.executemany(
                f"INSERT INTO frame_features ({','.join(FEATURE_COLUMNS)}) "
                f"VALUES ({','.join('?' for _ in FEATURE_COLUMNS)})",
                (
                    row(0, 1, 0, 2, 0),
                    row(1, 1, 1, 2, 1),
                    row(2, 2, 0, 1, 2),
                ),
            )
            connection.commit()
            connection.close()
            report = collect_capacity_statistics(
                index,
                n_values=(2,),
                k_values=(1,),
                query_visibility_min=0.1,
                reference_visibility_min=0.3,
                union_surface_min=0.5,
                progress=False,
            )
            self.assertEqual(report["grid"]["N2_K1"]["reference_tracks"], 1)
            self.assertEqual(report["grid"]["N2_K1"]["ordered_scene_track_pairs"], 1)
            self.assertEqual(report["grid"]["N2_K1"]["query_frames_supported"], 1)


if __name__ == "__main__":
    unittest.main()
