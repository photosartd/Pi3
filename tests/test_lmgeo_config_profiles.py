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
        self.assertTrue(cfg.visuals.enabled)
        self.assertFalse(cfg.visuals.train_enabled)
        self.assertTrue(cfg.visuals.val_enabled)
        self.assertIn("input_reference_frames", cfg.visuals["items"])
        self.assertIn("depth_panel", cfg.visuals["items"])
        for name in ("real_test_ref16", "pbr_new_val_ref16"):
            self.assertNotIn("context_reference_fraction", cfg.val_datasets[name].dataset)
            self.assertEqual(list(cfg.val_datasets[name].dataset.num_reference_range), [16, 16])
            self.assertEqual(list(cfg.val_datasets[name].dataset.num_query_range), [1, 1])
            self.assertEqual(list(cfg.val_datasets[name].runtime.image_num_range), [17, 17])

    def test_all_rgb_masked_profile_masks_references_and_queries(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_46gb",
            "lmgeo_trainpbr45_real_and_new_val_all_rgb_masked",
        )

        self.assertTrue(cfg.lmgeo.reference_rgb_masking)
        self.assertTrue(cfg.lmgeo.query_rgb_masking)
        self.assertTrue(cfg.lmgeo.depth_masking)
        self.assertTrue(cfg.train_dataset.LMGeoSequence.reference_rgb_masking)
        self.assertTrue(cfg.train_dataset.LMGeoSequence.query_rgb_masking)
        self.assertTrue(cfg.val_datasets.real_test.dataset.query_rgb_masking)
        self.assertTrue(cfg.val_datasets.pbr_new_val.dataset.query_rgb_masking)

    def test_all_rgb_masked_context_refs_profile_combines_both_ablations(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_46gb",
            "lmgeo_trainpbr45_real_and_new_val_all_rgb_masked_context_refs",
        )

        self.assertTrue(cfg.lmgeo.reference_rgb_masking)
        self.assertTrue(cfg.lmgeo.query_rgb_masking)
        self.assertTrue(cfg.lmgeo.depth_masking)
        self.assertEqual(cfg.lmgeo.context_reference_fraction, 0.5)
        self.assertEqual(cfg.train_dataset.LMGeoSequence.context_reference_fraction, 0.5)
        self.assertTrue(cfg.train_dataset.LMGeoSequence.reference_rgb_masking)
        self.assertTrue(cfg.train_dataset.LMGeoSequence.query_rgb_masking)
        self.assertEqual(
            active_val_loader_names(cfg),
            ["real_test", "pbr_new_val", "pbr_new_val_context_refs"],
        )
        self.assertTrue(cfg.val_datasets.real_test.dataset.query_rgb_masking)
        self.assertTrue(cfg.val_datasets.pbr_new_val.dataset.query_rgb_masking)
        context_dataset = cfg.val_datasets.pbr_new_val_context_refs.dataset
        self.assertTrue(context_dataset.reference_rgb_masking)
        self.assertTrue(context_dataset.query_rgb_masking)
        self.assertEqual(context_dataset.context_reference_fraction, 1.0)
        self.assertTrue(context_dataset.context_reference_eval)
        self.assertEqual(context_dataset.context_reference_exclude, "subscene")
        for name in active_val_loader_names(cfg):
            dataset = cfg.val_datasets[name].dataset
            runtime = cfg.val_datasets[name].runtime
            self.assertEqual(list(dataset.num_reference_range), [5, 5])
            self.assertEqual(list(dataset.num_query_range), [1, 1])
            self.assertEqual(list(runtime.image_num_range), [6, 6])

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

    def test_recenter_zoom_k1_profile_uses_query_only_transform_and_k1_loaders(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_46gb_recenter_zoom_k1",
            "lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1",
        )

        self.assertEqual(list(cfg.train.image_num_range), [3, 17])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [2, 16])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 1])
        self.assertEqual(
            cfg.train_dataset.LMGeoSequence._target_,
            "datasets.lmgeo_recenter.LMGeoRecenterZoomSequenceDataset",
        )
        self.assertFalse(cfg.train_dataset.LMGeoSequence.filter_target_center_crop_visibility)
        self.assertEqual(cfg.train_dataset.LMGeoSequence.query_recenter_bbox_key, "bbox_obj")
        self.assertEqual(cfg.train_dataset.LMGeoSequence.query_recenter_depth_interpolation, "nearest")
        self.assertEqual(
            active_val_loader_names(cfg),
            ["real_test", "real_test_ref16", "pbr_new_val", "pbr_new_val_ref16"],
        )
        for name in active_val_loader_names(cfg):
            dataset = cfg.val_datasets[name].dataset
            runtime = cfg.val_datasets[name].runtime
            self.assertEqual(dataset._target_, "datasets.lmgeo_recenter.LMGeoRecenterZoomSequenceDataset")
            self.assertEqual(list(dataset.num_query_range), [1, 1])
            self.assertFalse(dataset.filter_target_center_crop_visibility)
            self.assertEqual(list(runtime.image_num_range)[-1] - list(dataset.num_reference_range)[-1], 1)
        self.assertFalse(cfg.val_datasets.pbr_new_val_k5_subset.enabled)
        self.assertFalse(cfg.val_datasets.pbr_new_val_k10_subset.enabled)

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
        self.assertTrue(cfg.visuals.enabled)
        self.assertTrue(cfg.visuals.val_enabled)

    def test_a40_photometric_profile_is_train_only_baseline_delta(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_40gb_photometric",
            "lmgeo_trainpbr45_real_and_new_val_photometric",
        )

        self.assertEqual(list(cfg.train.image_num_range), [6, 28])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
        self.assertFalse(cfg.train.random_reslution)
        self.assertEqual(cfg.train.num_workers, 4)
        self.assertEqual(cfg.test.num_workers, 2)
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [5, 27])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 25])
        self.assertTrue(cfg.train_dataset.LMGeoSequence.photometric_augmentation)
        self.assertFalse(cfg.train_dataset.LMGeoSequence.aug_crop)
        self.assertFalse(cfg.train_dataset.LMGeoSequence.aug_focal)
        self.assertEqual(cfg.loss.train_loss.correspondence_weight, 0.0)
        self.assertNotIn("photometric_augmentation", cfg.val_datasets.real_test.dataset)
        self.assertNotIn("photometric_augmentation", cfg.val_datasets.pbr_new_val.dataset)

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

    def test_named_blackwell_70gb_profile_preserves_masked_context_refs(self):
        cfg = compose_job(
            "train_lmgeo_finetune_rtxpro6000_blackwell_70gb",
            "lmgeo_trainpbr45_real_and_new_val_all_rgb_masked_context_refs",
        )

        self.assertEqual(list(cfg.train.image_num_range), [6, 32])
        self.assertEqual(cfg.train.max_img_per_gpu, 56)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [5, 31])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 25])
        self.assertEqual(cfg.lmgeo.context_reference_fraction, 0.5)
        self.assertTrue(cfg.lmgeo.reference_rgb_masking)
        self.assertTrue(cfg.lmgeo.query_rgb_masking)
        self.assertEqual(
            active_val_loader_names(cfg),
            ["real_test", "pbr_new_val", "pbr_new_val_context_refs"],
        )
        self.assertEqual(cfg.val_datasets.real_test.runtime.max_img_per_gpu, 96)
        self.assertEqual(cfg.val_datasets.pbr_new_val.runtime.max_img_per_gpu, 96)
        self.assertEqual(cfg.val_datasets.pbr_new_val_context_refs.runtime.max_img_per_gpu, 96)
        self.assertTrue(cfg.visuals.enabled)
        self.assertFalse(cfg.visuals.train_enabled)
        self.assertTrue(cfg.visuals.val_enabled)
        self.assertEqual(cfg.visuals.val_every_n_epochs, 1)
        for name in active_val_loader_names(cfg):
            dataset = cfg.val_datasets[name].dataset
            runtime = cfg.val_datasets[name].runtime
            self.assertEqual(list(dataset.num_reference_range), [5, 5])
            self.assertEqual(list(dataset.num_query_range), [1, 1])
            self.assertEqual(list(runtime.image_num_range), [6, 6])

    def test_recenter_zoom_masked_context_refs_profile_uses_k1_masked_vals(self):
        data_name = "lmgeo_trainpbr45_real_and_new_val_recenter_zoom_masked_context_refs_k1"
        a40 = compose_job(
            "train_lmgeo_finetune_a40_40gb_recenter_zoom_masked_k1",
            data_name,
        )
        blackwell = compose_job(
            "train_lmgeo_finetune_rtxpro6000_blackwell_70gb_recenter_zoom_masked_k1",
            data_name,
        )

        self.assertEqual(list(a40.train.image_num_range), [3, 17])
        self.assertEqual(a40.train.max_img_per_gpu, 28)
        self.assertEqual(a40.test.max_img_per_gpu, 128)
        self.assertEqual(list(blackwell.train.image_num_range), [3, 17])
        self.assertEqual(blackwell.train.max_img_per_gpu, 56)
        self.assertEqual(blackwell.test.max_img_per_gpu, 96)

        for cfg in (a40, blackwell):
            self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
            self.assertEqual(list(cfg.lmgeo.num_reference_range), [2, 16])
            self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 1])
            self.assertEqual(cfg.lmgeo.context_reference_fraction, 0.5)
            self.assertTrue(cfg.lmgeo.reference_rgb_masking)
            self.assertTrue(cfg.lmgeo.query_rgb_masking)
            self.assertEqual(
                cfg.train_dataset.LMGeoSequence._target_,
                "datasets.lmgeo_recenter.LMGeoRecenterZoomSequenceDataset",
            )
            self.assertEqual(cfg.train_dataset.LMGeoSequence.context_reference_fraction, 0.5)
            self.assertTrue(cfg.train_dataset.LMGeoSequence.reference_rgb_masking)
            self.assertTrue(cfg.train_dataset.LMGeoSequence.query_rgb_masking)
            self.assertFalse(cfg.train_dataset.LMGeoSequence.filter_preprocessed_query_depth)
            self.assertEqual(
                active_val_loader_names(cfg),
                ["real_test", "pbr_new_val"],
            )
            for name in active_val_loader_names(cfg):
                dataset = cfg.val_datasets[name].dataset
                runtime = cfg.val_datasets[name].runtime
                self.assertEqual(dataset._target_, "datasets.lmgeo_recenter.LMGeoRecenterZoomSequenceDataset")
                self.assertTrue(dataset.reference_rgb_masking)
                self.assertTrue(dataset.query_rgb_masking)
                if name == "real_test":
                    self.assertFalse(dataset.filter_target_preprocessed_depth)
                else:
                    self.assertFalse(dataset.filter_preprocessed_query_depth)
                self.assertFalse(dataset.filter_target_center_crop_visibility)
                self.assertEqual(list(dataset.num_reference_range), [5, 5])
                self.assertEqual(list(dataset.num_query_range), [1, 1])
                self.assertEqual(list(runtime.image_num_range), [6, 6])

    def test_generic_pi3_profile_is_unchanged(self):
        cfg = compose_job("train_pi3_lowres", "example")

        self.assertEqual(list(cfg.train.image_num_range), [2, 24])
        self.assertEqual(cfg.train.max_img_per_gpu, 64)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[224, 224]])
        self.assertNotIn("lmgeo_profile", cfg)


if __name__ == "__main__":
    unittest.main()
