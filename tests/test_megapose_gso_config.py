import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class MegaPoseGSOConfigTest(unittest.TestCase):
    def test_mesh_free_cross_scene_profile_composes(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_finetune_560",
                    "data=megapose_gso_cross_scene",
                ],
            )
        self.assertEqual(
            OmegaConf.to_container(cfg.train.resolution), [[560, 420]]
        )
        self.assertEqual(list(cfg.train.image_num_range), [6, 28])
        self.assertEqual(
            cfg.train_dataset.MegaPoseGSO._target_,
            "datasets.megapose_gso_dataset.MegaPoseGSOObjectDataset",
        )
        self.assertEqual(cfg.gso.sampling_regime, "independent_scenes")
        self.assertEqual(cfg.train_dataset.MegaPoseGSO.scene_split, "train")
        self.assertEqual(
            cfg.train_dataset.MegaPoseGSO.sampling_regime,
            "independent_scenes",
        )
        self.assertEqual(cfg.val_datasets.gso_val.dataset.scene_split, "val")
        self.assertEqual(
            cfg.val_datasets.gso_val.dataset.sampling_regime,
            "independent_scenes",
        )
        self.assertEqual(cfg.test_dataset.scene_split, "val")
        self.assertEqual(list(cfg.val_datasets), ["gso_val"])
        self.assertEqual(
            list(cfg.val_datasets.gso_val.runtime.image_num_range), [6, 6]
        )
        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertTrue(cfg.gso.condition_query_visibility)
        self.assertTrue(
            cfg.train_dataset.MegaPoseGSO.condition_query_visibility
        )
        self.assertIsNone(cfg.metrics["items"].object_pose)
        self.assertIsNone(cfg.metrics["items"].chamfer)
        self.assertIsNone(cfg.visuals["items"].query_pose_overlay)
        self.assertIn("camera", cfg.metrics["items"])
        self.assertIn("correspondence", cfg.metrics["items"])
        self.assertIn("visibility_condition", cfg.visuals["items"])

    def test_a40_anchor_pair_profile_composes_with_anchor_validation(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_finetune_a40_46gb",
                    "data=megapose_gso_anchor_scene_pairs",
                ],
            )
        self.assertEqual(
            OmegaConf.to_container(cfg.train.resolution), [[560, 420]]
        )
        self.assertEqual(list(cfg.train.image_num_range), [3, 26])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertEqual(list(cfg.gso_profile.num_reference_range), [2, 16])
        self.assertEqual(list(cfg.gso_profile.num_query_range), [1, 10])
        self.assertEqual(cfg.gso.sampling_regime, "anchor_pair")
        self.assertEqual(
            cfg.train_dataset.MegaPoseGSO.sampling_regime, "anchor_pair"
        )
        self.assertEqual(
            cfg.val_datasets.gso_val.dataset.sampling_regime,
            "anchor_pair",
        )
        self.assertFalse(cfg.gso.allow_repeat)
        self.assertTrue(cfg.gso.condition_reference_visibility)
        self.assertTrue(cfg.gso.condition_query_visibility)
        self.assertIsNone(cfg.metrics["items"].object_pose)
        self.assertIsNone(cfg.metrics["items"].chamfer)

    def test_336_n5_k1_profile_is_fixed_and_has_one_anchor_validation(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_finetune_a40_40gb_336x252_n5_k1",
                    "data=megapose_gso_anchor_scene_pairs",
                ],
            )
        self.assertEqual(
            OmegaConf.to_container(cfg.train.resolution), [[336, 252]]
        )
        self.assertEqual(list(cfg.train.image_num_range), [6, 6])
        self.assertEqual(cfg.train.max_img_per_gpu, 72)
        self.assertEqual(list(cfg.gso_profile.num_reference_range), [5, 5])
        self.assertEqual(list(cfg.gso_profile.num_query_range), [1, 1])
        self.assertEqual(list(cfg.val_datasets), ["gso_val"])
        self.assertEqual(
            cfg.val_datasets.gso_val.dataset.sampling_regime, "anchor_pair"
        )
        self.assertEqual(
            list(cfg.val_datasets.gso_val.dataset.num_reference_range), [5, 5]
        )
        self.assertEqual(
            list(cfg.val_datasets.gso_val.dataset.num_query_range), [1, 1]
        )
        self.assertEqual(cfg.val_datasets.gso_val.runtime.max_img_per_gpu, 256)
        self.assertEqual(cfg.val_datasets.gso_val.runtime.num_workers, 2)
        self.assertFalse(cfg.test.persistent_workers)

    def test_336_dynamic_profile_composes_with_anchor_validation_sweep(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_finetune_a40_40gb_336x252_dynamic",
                    "data=megapose_gso_anchor_scene_pairs_multival",
                ],
            )
        self.assertEqual(
            OmegaConf.to_container(cfg.train.resolution), [[336, 252]]
        )
        self.assertEqual(list(cfg.train.image_num_range), [3, 26])
        self.assertEqual(cfg.train.max_img_per_gpu, 72)
        self.assertEqual(list(cfg.gso_profile.num_reference_range), [2, 16])
        self.assertEqual(list(cfg.gso_profile.num_query_range), [1, 10])
        self.assertEqual(
            list(cfg.val_datasets),
            ["gso_val", "gso_val_ref16", "gso_val_k5", "gso_val_k10"],
        )
        expected_protocols = {
            "gso_val": (5, 1, 6),
            "gso_val_ref16": (16, 1, 17),
            "gso_val_k5": (5, 5, 10),
            "gso_val_k10": (5, 10, 15),
        }
        for name, (num_reference, num_query, total) in expected_protocols.items():
            entry = cfg.val_datasets[name]
            self.assertEqual(entry.dataset.sampling_regime, "anchor_pair")
            self.assertFalse(entry.dataset.allow_repeat)
            self.assertEqual(
                list(entry.dataset.num_reference_range),
                [num_reference, num_reference],
            )
            self.assertEqual(
                list(entry.dataset.num_query_range), [num_query, num_query]
            )
            self.assertEqual(list(entry.runtime.image_num_range), [total, total])
            self.assertEqual(entry.runtime.max_img_per_gpu, 256)
            self.assertEqual(entry.runtime.num_workers, 2)
        self.assertFalse(cfg.test.persistent_workers)


if __name__ == "__main__":
    unittest.main()
