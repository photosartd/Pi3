import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from hydra import compose, initialize_config_dir
import numpy as np
from PIL import Image

from datasets.lmgeo_dataset import (
    LMGeoDataset,
    LMGeoGeometryMatchedSequenceDataset,
)
from datasets.object_geometry import GeometryReferencePlan
from datasets.preprocess.lmgeo_geometry import (
    bop_camera_distribution_summary,
    bop_frame_id,
    build_bop_geometry_index,
)
from datasets.preprocess.megapose_gso_geometry import BuildSettings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPOSITORY_ROOT / "configs"


class LMGeoGeometryMatchedTest(unittest.TestCase):
    def test_bop_frame_identity_is_stable_and_legacy_record_is_additive(self):
        dataset = object.__new__(LMGeoDataset)
        dataset.object_id = 1
        dataset.mask_type = "mask_visib"
        dataset.depth_unit_scale = 0.001
        scene_dir = Path("/tmp/example/new_val/000045")
        gt = {
            "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
            "cam_t_m2c": [0.0, 0.0, 1000.0],
        }
        record = dataset._record_from_gt(
            scene_dir,
            17,
            2,
            gt,
            {"bbox_obj": [10, 11, 20, 21], "bbox_visib": [10, 11, 20, 21]},
            {"cam_K": np.eye(3).reshape(-1).tolist(), "depth_scale": 1.0},
            "new_val",
            image_size=(64, 48),
            rgb_ext=".png",
            trust_paths=True,
        )
        self.assertEqual(record["scene_id"], 45)
        self.assertEqual(record["view_id"], 17)
        self.assertEqual(record["frame_id"], bop_frame_id(45, 17))
        np.testing.assert_allclose(record["camera_pose"], np.linalg.inv(record["T_C_O"]))

    @staticmethod
    def _write_metadata_fixture(root: Path):
        models = root / "models_eval"
        models.mkdir(parents=True)
        (models / "models_info.json").write_text(
            json.dumps({"1": {"diameter": 100.0}}), encoding="utf-8"
        )
        scene = root / "new_val" / "000045"
        for name in ("depth", "mask_visib"):
            (scene / name).mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((48, 64), 1000, dtype=np.uint16)).save(
            scene / "depth" / "000000.png"
        )
        Image.fromarray(np.full((48, 64), 255, dtype=np.uint8)).save(
            scene / "mask_visib" / "000000_000000.png"
        )
        K = [50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0]
        (scene / "scene_camera.json").write_text(
            json.dumps({"0": {"cam_K": K, "depth_scale": 1.0}}),
            encoding="utf-8",
        )
        (scene / "scene_gt.json").write_text(
            json.dumps(
                {
                    "0": [
                        {
                            "obj_id": 1,
                            "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
                            "cam_t_m2c": [0.0, 0.0, 1000.0],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        info = {
            "bbox_obj": [20, 14, 24, 20],
            "bbox_visib": [20, 14, 24, 20],
            "px_count_all": 480,
            "px_count_valid": 480,
            "px_count_visib": 480,
            "visib_fract": 1.0,
        }
        (scene / "scene_gt_info.json").write_text(
            json.dumps({"0": [info]}), encoding="utf-8"
        )

    def test_metadata_only_bop_index_matches_geometry_reader_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_metadata_fixture(root)
            output = root / "geometry.sqlite"
            bits = root / "geometry.bits"
            result = build_bop_geometry_index(
                data_root=root,
                split="new_val",
                output_path=output,
                bits_path=bits,
                atlas_path=root / "unused.f32",
                settings=BuildSettings(
                    split="all",
                    visibility_floor=0.1,
                    min_visible_pixels=1,
                    surface_point_count=8,
                ),
                compute_surface=False,
                device="cpu",
                workers=1,
                progress=False,
            )
            self.assertEqual(result["processed_frames"], 1)
            connection = sqlite3.connect(output)
            try:
                metadata = dict(connection.execute("SELECT key,value FROM metadata"))
                row = connection.execute(
                    "SELECT frame_id,scene_id,view_id,object_id,crop_feasible,"
                    "observed_fraction FROM frame_features"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(metadata["index_complete"], "1")
            self.assertEqual(row[:4], (bop_frame_id(45, 0), 45, 0, 1))
            self.assertEqual(row[4], 1)
            self.assertEqual(row[5], 0.0)
            self.assertEqual(bits.stat().st_size, 1)
            summary = bop_camera_distribution_summary(root, "new_val", output)
            self.assertEqual(summary["unique_intrinsics"], 1)
            self.assertAlmostEqual(summary["object_center_z_m"]["median"], 1.0)
            self.assertAlmostEqual(
                summary["camera_distance_over_diameter"]["median"], 10.0
            )

    def test_matcher_enforces_focal_and_positive_angle(self):
        dataset = object.__new__(LMGeoGeometryMatchedSequenceDataset)
        dataset.positive_angle_degrees = 10.0
        dataset.focal_relative_tolerance = 0.1
        dataset.geometry_plan_selection = "first"
        query = {
            "crop_feasible": 1,
            "base_norm_fx": 1.0,
            "base_norm_fy": 1.0,
            "max_norm_fx": 1.5,
            "max_norm_fy": 1.5,
            "view_x": 0.0,
            "view_y": 0.0,
            "view_z": 1.0,
        }
        directions = np.asarray(
            [[0.0, 0.0, 1.0]] + [[1.0, 0.0, 0.0]] * 4,
            dtype=np.float32,
        )
        plan = GeometryReferencePlan(
            plan_id=0,
            object_id=1,
            scene_id=1,
            gt_id=0,
            reference_count=5,
            union_coverage=0.6,
            feature_ids=np.arange(5, dtype=np.int64),
            frame_ids=np.arange(5, dtype=np.int64),
            view_ids=np.arange(5, dtype=np.int64),
            directions=directions,
            focal_low=1.0,
            focal_high=1.1,
            focal_shape_low=-0.03,
            focal_shape_high=0.03,
        )
        matches, reason = dataset._compatible_plan_matches(query, (plan,))
        self.assertIsNone(reason)
        self.assertEqual(len(matches), 1)
        self.assertLessEqual(matches[0][3], 1e-5)
        bad_query = dict(query, base_norm_fx=2.0, base_norm_fy=2.0)
        matches, reason = dataset._compatible_plan_matches(bad_query, (plan,))
        self.assertEqual(matches, [])
        self.assertEqual(reason, "focal")

    def test_eval_profile_is_opt_in_and_composes_exact_contract(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_pi3_lmgeo_eval_rtxpro6000_70gb_336x252",
                    "data=lmgeo_new_val_geometry_render_n5_k1_masked",
                    "general=lmgeo_geometry_eval",
                ],
            )
        self.assertTrue(cfg.train.eval_only)
        self.assertTrue(cfg.model.use_ray_conditioning)
        self.assertFalse(cfg.model.use_visibility_mask_conditioning)
        dataset = cfg.val_datasets.lmo_new_val_geometry_render_n5_k1.dataset
        self.assertEqual(
            dataset._target_,
            "datasets.lmgeo_dataset.LMGeoGeometryMatchedSequenceDataset",
        )
        self.assertTrue(dataset.reference_rgb_masking)
        self.assertTrue(dataset.query_rgb_masking)
        self.assertEqual(list(dataset.num_reference_range), [5, 5])
        self.assertEqual(list(dataset.num_query_range), [1, 1])
        self.assertEqual(dataset.positive_angle_degrees, 10.0)
        self.assertEqual(dataset.focal_relative_tolerance, 0.1)
        self.assertFalse(dataset.photometric_augmentation)
        self.assertEqual(list(cfg.train.resolution[0]), [336, 252])
        self.assertFalse(cfg.log.save_best)
        self.assertFalse(cfg.log.save_checkpoints)
        self.assertIsNone(cfg.visuals["items"].visibility_condition)
        pose_metric = cfg.metrics["items"].object_pose
        self.assertTrue(pose_metric.query_occupancy_analysis)
        self.assertEqual(
            list(pose_metric.query_occupancy_bin_edges),
            [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 1.000001],
        )
        self.assertEqual(
            list(cfg.metrics["items"].object_pose.symmetric_ids), [10, 11]
        )
        self.assertEqual(
            cfg.metrics["items"].object_pose.scale_estimation,
            "reference_depth",
        )
        self.assertEqual(
            cfg.metrics["items"].camera.scale_estimation,
            "reference_depth",
        )
        self.assertEqual(
            cfg.visuals["items"].query_pose_overlay.scale_estimation,
            "reference_depth",
        )


if __name__ == "__main__":
    unittest.main()
