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

    def test_a40_anchor_pair_profile_composes_with_independent_validation(self):
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
            "independent_scenes",
        )
        self.assertFalse(cfg.gso.allow_repeat)
        self.assertTrue(cfg.gso.condition_reference_visibility)
        self.assertTrue(cfg.gso.condition_query_visibility)
        self.assertIsNone(cfg.metrics["items"].object_pose)
        self.assertIsNone(cfg.metrics["items"].chamfer)


if __name__ == "__main__":
    unittest.main()
