import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def compose_job(train: str, data: str):
    with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="default",
            overrides=[f"train={train}", f"data={data}"],
        )


def active_val_loader_names(cfg):
    return [
        str(name)
        for name, entry in cfg.val_datasets.items()
        if entry is not None and bool(entry.get("enabled", True))
    ]


class LMGeoHardwareProfileConfigTest(unittest.TestCase):
    def test_named_4090_profile_uses_canonical_data(self):
        cfg = compose_job(
            "train_lmgeo_finetune_rtx4090_24gb",
            "lmgeo_trainpbr45_real_and_new_val",
        )

        self.assertEqual(list(cfg.train.image_num_range), [6, 26])
        self.assertEqual(cfg.train.max_img_per_gpu, 50)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[224, 224]])
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [5, 6])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 20])
        self.assertEqual(cfg.val_datasets.real_test.runtime.max_img_per_gpu, 96)
        self.assertEqual(cfg.val_datasets.pbr_new_val_k10_subset.runtime.max_img_per_gpu, 96)
        self.assertEqual(cfg.val_datasets.real_test_ref16.runtime.max_img_per_gpu, 96)
        self.assertEqual(list(cfg.val_datasets.real_test_ref16.dataset.num_reference_range), [16, 16])
        self.assertEqual(list(cfg.val_datasets.real_test_ref16.dataset.num_query_range), [1, 1])
        self.assertEqual(list(cfg.val_datasets.real_test_ref16.runtime.image_num_range), [17, 17])
        self.assertEqual(cfg.train_dataset.length, "auto")

    def test_named_a40_profile_uses_canonical_data(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_46gb",
            "lmgeo_trainpbr45_real_and_new_val",
        )

        self.assertEqual(list(cfg.train.image_num_range), [6, 28])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
        self.assertFalse(cfg.train.random_reslution)
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [5, 27])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 25])
        self.assertEqual(cfg.lmgeo.context_reference_fraction, 0.0)
        self.assertEqual(cfg.train_dataset.LMGeoSequence.context_reference_fraction, 0.0)
        self.assertEqual(cfg.primary_val, "real_test")
        self.assertEqual(cfg.val_datasets.real_test.runtime.max_img_per_gpu, 128)
        self.assertEqual(cfg.val_datasets.pbr_new_val_k5_subset.runtime.max_img_per_gpu, 128)
        self.assertEqual(cfg.val_datasets.pbr_new_val_k10_subset.runtime.max_img_per_gpu, 128)
        self.assertEqual(cfg.val_datasets.real_test_ref16.runtime.max_img_per_gpu, 128)
        self.assertEqual(cfg.val_datasets.pbr_new_val_ref16.runtime.max_img_per_gpu, 128)
        for name in ("real_test_ref16", "pbr_new_val_ref16"):
            self.assertNotIn("context_reference_fraction", cfg.val_datasets[name].dataset)
            self.assertEqual(list(cfg.val_datasets[name].dataset.num_reference_range), [16, 16])
            self.assertEqual(list(cfg.val_datasets[name].dataset.num_query_range), [1, 1])
            self.assertEqual(list(cfg.val_datasets[name].runtime.image_num_range), [17, 17])

    def test_named_a40_profile_covers_required_split_extremes(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_46gb",
            "lmgeo_trainpbr45_real_and_new_val",
        )

        ref_min, ref_max = cfg.lmgeo.num_reference_range
        query_min, query_max = cfg.lmgeo.num_query_range

        def valid_splits(total):
            return [
                (refs, total - refs)
                for refs in range(ref_min, ref_max + 1)
                if query_min <= total - refs <= query_max
            ]

        self.assertEqual(valid_splits(6), [(5, 1)])
        self.assertEqual(valid_splits(28)[0], (5, 23))
        self.assertEqual(valid_splits(28)[-1], (27, 1))
        self.assertEqual(len(valid_splits(28)), 23)

    def test_context_reference_data_profile_enables_train_and_pbr_context_val(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_46gb",
            "lmgeo_trainpbr45_real_and_new_val_context_refs",
        )

        self.assertEqual(cfg.lmgeo.context_reference_fraction, 0.5)
        self.assertEqual(cfg.train_dataset.LMGeoSequence.context_reference_fraction, 0.5)
        self.assertNotIn("context_reference_fraction", cfg.val_datasets.real_test.dataset)
        self.assertNotIn("context_reference_fraction", cfg.val_datasets.pbr_new_val.dataset)
        self.assertEqual(
            active_val_loader_names(cfg),
            ["real_test", "pbr_new_val", "pbr_new_val_context_refs"],
        )
        context_dataset = cfg.val_datasets.pbr_new_val_context_refs.dataset
        self.assertEqual(list(context_dataset.num_reference_range), [5, 5])
        self.assertEqual(list(context_dataset.num_query_range), [1, 1])
        self.assertEqual(context_dataset.context_reference_fraction, 1.0)
        self.assertTrue(context_dataset.context_reference_eval)
        self.assertEqual(context_dataset.context_reference_exclude, "subscene")

    def test_a40_correspondence_profile_is_a_hardware_preserving_delta(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_46gb_corr",
            "lmgeo_trainpbr45_real_and_new_val",
        )

        self.assertEqual(list(cfg.train.image_num_range), [6, 28])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [5, 27])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 25])
        self.assertEqual(list(cfg.model.dino_output_layers), [17])
        self.assertEqual(cfg.loss.train_loss.correspondence_weight, 0.3)
        self.assertEqual(cfg.loss.test_loss.correspondence_weight, 0.3)
        self.assertTrue(cfg.loss.train_loss.correspondence_use_dino_weights)
        self.assertTrue(cfg.loss.test_loss.correspondence_use_dino_weights)
        self.assertFalse(cfg.visuals.enabled)

    def test_historical_a40_data_name_is_an_alias(self):
        canonical = compose_job(
            "train_lmgeo_finetune_a40_46gb",
            "lmgeo_trainpbr45_real_and_new_val",
        )
        historical = compose_job(
            "train_lmgeo_finetune_518_a40_dynamic",
            "lmgeo_trainpbr45_real_and_new_val_518_a40_dynamic",
        )

        for key in (
            "lmgeo",
            "train_dataset",
            "test_dataset",
            "val_datasets",
        ):
            self.assertEqual(
                OmegaConf.to_container(canonical[key], resolve=True),
                OmegaConf.to_container(historical[key], resolve=True),
            )

    def test_conservative_a40_profile_is_also_fixed_560x420(self):
        cfg = compose_job(
            "train_lmgeo_finetune_518_a40",
            "lmgeo_all_trainpbr_test_bop_518_a40",
        )

        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
        self.assertFalse(cfg.train.random_reslution)
        self.assertEqual(cfg.test.max_img_per_gpu, 128)

    def test_named_blackwell_profile_uses_larger_budgets(self):
        cfg = compose_job(
            "train_lmgeo_finetune_rtxpro6000_blackwell_96gb",
            "lmgeo_trainpbr45_real_and_new_val",
        )

        self.assertEqual(list(cfg.train.image_num_range), [6, 32])
        self.assertEqual(cfg.train.max_img_per_gpu, 56)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
        self.assertFalse(cfg.train.random_reslution)
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [5, 31])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 25])
        self.assertEqual(cfg.val_datasets.real_test.runtime.max_img_per_gpu, 384)
        self.assertEqual(cfg.val_datasets.pbr_new_val_k5_subset.runtime.max_img_per_gpu, 384)
        self.assertEqual(cfg.val_datasets.pbr_new_val_k10_subset.runtime.max_img_per_gpu, 320)
        self.assertEqual(cfg.val_datasets.real_test_ref16.runtime.max_img_per_gpu, 320)
        self.assertEqual(cfg.val_datasets.pbr_new_val_ref16.runtime.max_img_per_gpu, 320)

    def test_generic_pi3_profile_is_unchanged(self):
        cfg = compose_job("train_pi3_lowres", "example")

        self.assertEqual(list(cfg.train.image_num_range), [2, 24])
        self.assertEqual(cfg.train.max_img_per_gpu, 64)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[224, 224]])
        self.assertNotIn("lmgeo_profile", cfg)


if __name__ == "__main__":
    unittest.main()
