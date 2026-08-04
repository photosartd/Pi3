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

    def test_same_scene_ceiling_eval_profile_uses_heldout_query_dataset(self):
        cfg = compose_job(
            "train_lmgeo_eval_only_a40_40gb_same_scene_ceiling",
            "lmgeo_same_scene_ceiling",
        )

        self.assertTrue(cfg.train.eval_only)
        self.assertEqual(list(cfg.train.image_num_range), [25, 25])
        self.assertEqual(cfg.train.max_img_per_gpu, 25)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
        self.assertFalse(cfg.lmgeo.reference_rgb_masking)
        self.assertFalse(cfg.lmgeo.query_rgb_masking)
        self.assertTrue(cfg.lmgeo.depth_masking)
        self.assertEqual(cfg.lmgeo_ceiling.max_reference, 24)
        self.assertEqual(cfg.lmgeo_ceiling.query_frame_strategy, "random")
        self.assertIsNone(cfg.metrics["items"].correspondence)
        self.assertEqual(
            active_val_loader_names(cfg),
            ["real_same_scene_ceiling", "pbr_same_scene_ceiling"],
        )
        for name in active_val_loader_names(cfg):
            dataset = cfg.val_datasets[name].dataset
            runtime = cfg.val_datasets[name].runtime
            self.assertEqual(
                dataset._target_,
                "datasets.lmgeo_dataset.LMGeoSameSceneCeilingDataset",
            )
            self.assertEqual(dataset.query_source, "windows")
            self.assertEqual(dataset.max_reference, 24)
            self.assertEqual(dataset.min_reference, 2)
            self.assertEqual(list(runtime.image_num_range), [25, 25])
            self.assertEqual(runtime.max_img_per_gpu, 25)

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
        self.assertFalse(
            cfg.train_dataset.LMGeoSequence.query_recenter_include_original_query_view
        )
        self.assertNotIn("paired_query", cfg.metrics["items"])
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

    def test_paired_recenter_zoom_profile_adds_one_model_view_and_metrics_on_demand(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_46gb_recenter_zoom_k1",
            "lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1",
        )

        self.assertFalse(cfg.model.use_ray_conditioning)
        self.assertEqual(list(cfg.train.image_num_range), [4, 18])
        self.assertEqual(list(cfg.test.image_num_range), [7, 7])
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [2, 16])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 1])
        self.assertTrue(
            cfg.train_dataset.LMGeoSequence.query_recenter_include_original_query_view
        )
        self.assertEqual(
            cfg.metrics["items"].paired_query._target_,
            "pi3.metrics.paired_query.PairedQueryConsistencyMetric",
        )
        self.assertNotIn("ray_geometry", cfg.metrics["items"])
        self.assertEqual(
            cfg.visuals["items"].input_query_context_frames.role,
            "query_context",
        )
        expected_totals = {
            "real_test": 7,
            "real_test_ref16": 18,
            "pbr_new_val": 7,
            "pbr_new_val_ref16": 18,
        }
        for name, total_views in expected_totals.items():
            dataset = cfg.val_datasets[name].dataset
            runtime = cfg.val_datasets[name].runtime
            self.assertTrue(
                dataset.query_recenter_include_original_query_view
            )
            self.assertEqual(list(dataset.num_query_range), [1, 1])
            self.assertEqual(
                list(runtime.image_num_range),
                [total_views, total_views],
            )

    def test_paired_recenter_zoom_ray_profile_composes_both_optional_deltas(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1",
            "lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1",
        )

        self.assertTrue(cfg.model.use_ray_conditioning)
        self.assertEqual(float(cfg.train.optimizer.ray_lr), 1e-5)
        self.assertEqual(list(cfg.train.image_num_range), [4, 18])
        self.assertTrue(
            cfg.train_dataset.LMGeoSequence.query_recenter_include_original_query_view
        )
        self.assertIn("paired_query", cfg.metrics["items"])
        self.assertIn("ray_geometry", cfg.metrics["items"])

    def test_recenter_zoom_ray_k1_is_an_isolated_model_and_metric_delta(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1",
            "lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1",
        )

        self.assertTrue(cfg.model.use_ray_conditioning)
        self.assertEqual(float(cfg.train.optimizer.lr), 5e-6)
        self.assertEqual(float(cfg.train.optimizer.encoder_lr), 0.0)
        self.assertEqual(float(cfg.train.optimizer.ray_lr), 1e-5)
        self.assertEqual(list(cfg.train.image_num_range), [3, 17])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [2, 16])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 1])
        self.assertEqual(
            cfg.train_dataset.LMGeoSequence._target_,
            "datasets.lmgeo_recenter.LMGeoRecenterZoomSequenceDataset",
        )
        self.assertEqual(
            cfg.metrics["items"].ray_geometry._target_,
            "pi3.metrics.ray_geometry.RayGeometryMetric",
        )
        self.assertFalse(cfg.metrics.train_enabled)
        self.assertTrue(cfg.metrics.val_enabled)
        self.assertEqual(
            active_val_loader_names(cfg),
            ["real_test", "real_test_ref16", "pbr_new_val", "pbr_new_val_ref16"],
        )

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

    def test_anchor_scene_pair_mask_conditioning_profiles_are_leak_free(self):
        train_name = "train_lmgeo_finetune_a40_40gb_anchor_mask_conditioning"
        masked = compose_job(
            train_name,
            "lmgeo_trainpbr45_anchor_scene_pairs_masked_depth",
        )
        full = compose_job(
            train_name,
            "lmgeo_trainpbr45_anchor_scene_pairs_full_depth",
        )

        for cfg in (masked, full):
            self.assertTrue(cfg.model.use_visibility_mask_conditioning)
            self.assertEqual(cfg.model.visibility_mask_conditioning_alpha, 1.0)
            self.assertEqual(cfg.train.optimizer.visibility_mask_lr, 5e-5)
            self.assertEqual(list(cfg.train.image_num_range), [3, 26])
            self.assertEqual(cfg.train.max_img_per_gpu, 28)
            self.assertEqual(list(cfg.lmgeo.num_reference_range), [2, 16])
            self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 10])
            self.assertFalse(cfg.lmgeo.reference_rgb_masking)
            self.assertFalse(cfg.lmgeo.query_rgb_masking)
            self.assertEqual(
                cfg.train_dataset.LMGeoAnchorScenePair._target_,
                "datasets.lmgeo_dataset.LMGeoAnchorScenePairSequenceDataset",
            )
            self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.visibility_mask_conditioning)
            self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.condition_reference_visibility)
            self.assertFalse(cfg.train_dataset.LMGeoAnchorScenePair.condition_query_visibility)
            self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.anchor_allow_same_scene)
            self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.anchor_allow_same_subscene)
            self.assertIn("visibility_condition", cfg.visuals["items"])
            self.assertEqual(
                active_val_loader_names(cfg),
                ["real_anchor_pairs", "pbr_anchor_pairs"],
            )
            for name in active_val_loader_names(cfg):
                dataset = cfg.val_datasets[name].dataset
                runtime = cfg.val_datasets[name].runtime
                self.assertEqual(
                    dataset._target_,
                    "datasets.lmgeo_dataset.LMGeoAnchorScenePairSequenceDataset",
                )
                self.assertEqual(list(dataset.num_reference_range), [5, 5])
                self.assertEqual(list(dataset.num_query_range), [1, 1])
                self.assertFalse(dataset.condition_query_visibility)
                self.assertEqual(list(runtime.image_num_range), [6, 6])

        self.assertTrue(masked.lmgeo.depth_masking)
        self.assertTrue(masked.train_dataset.LMGeoAnchorScenePair.depth_masking)
        self.assertFalse(full.lmgeo.depth_masking)
        self.assertFalse(full.train_dataset.LMGeoAnchorScenePair.depth_masking)

    def test_anchor_scene_pair_query_mask_oracle_conditions_queries(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_40gb_anchor_mask_conditioning",
            "lmgeo_trainpbr45_anchor_scene_pairs_masked_depth_query_masks",
        )

        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertTrue(cfg.lmgeo.depth_masking)
        self.assertEqual(list(cfg.train.image_num_range), [3, 26])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertTrue(cfg.lmgeo_anchor.condition_reference_visibility)
        self.assertTrue(cfg.lmgeo_anchor.condition_query_visibility)
        self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.condition_reference_visibility)
        self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.condition_query_visibility)
        for name in active_val_loader_names(cfg):
            self.assertTrue(cfg.val_datasets[name].dataset.condition_query_visibility)

    def test_anchor_scene_pair_query_mask_ref_sweep_uses_pbr_only(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_40gb_anchor_mask_conditioning_n25_k1",
            "lmgeo_trainpbr45_anchor_scene_pairs_masked_depth_query_masks_pbr_ref_sweep",
        )

        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertTrue(cfg.lmgeo.depth_masking)
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [2, 25])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 1])
        self.assertEqual(list(cfg.train.image_num_range), [3, 26])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertTrue(cfg.lmgeo_anchor.condition_reference_visibility)
        self.assertTrue(cfg.lmgeo_anchor.condition_query_visibility)
        self.assertEqual(cfg.lmgeo_anchor.anchor_selection_attempts, 200)
        self.assertFalse(cfg.val_datasets.real_anchor_pairs.enabled)
        self.assertFalse(cfg.val_datasets.pbr_anchor_pairs.enabled)
        self.assertEqual(
            active_val_loader_names(cfg),
            [
                "pbr_anchor_pairs_ref5",
                "pbr_anchor_pairs_ref10",
                "pbr_anchor_pairs_ref15",
            ],
        )

        expected_refs = {
            "pbr_anchor_pairs_ref5": 5,
            "pbr_anchor_pairs_ref10": 10,
            "pbr_anchor_pairs_ref15": 15,
        }
        for name, ref_count in expected_refs.items():
            dataset = cfg.val_datasets[name].dataset
            runtime = cfg.val_datasets[name].runtime
            self.assertEqual(
                dataset._target_,
                "datasets.lmgeo_dataset.LMGeoAnchorScenePairSequenceDataset",
            )
            self.assertEqual(dataset.query_split, cfg.lmgeo.new_val_query_split)
            self.assertEqual(list(dataset.num_reference_range), [ref_count, ref_count])
            self.assertEqual(list(dataset.num_query_range), [1, 1])
            self.assertEqual(list(runtime.image_num_range), [ref_count + 1, ref_count + 1])
            self.assertEqual(runtime.max_img_per_gpu, 26)
            self.assertTrue(dataset.condition_reference_visibility)
            self.assertTrue(dataset.condition_query_visibility)
            self.assertTrue(dataset.depth_masking)
            self.assertFalse(dataset.reference_rgb_masking)
            self.assertFalse(dataset.query_rgb_masking)

    def test_anchor_scene_pair_full_depth_ref_mask_ref_sweep_has_no_query_masks(self):
        cfg = compose_job(
            "train_lmgeo_finetune_a40_40gb_anchor_mask_conditioning_n5_25_k1",
            "lmgeo_trainpbr45_anchor_scene_pairs_full_depth_ref_masks_pbr_ref_sweep",
        )

        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertFalse(cfg.lmgeo.depth_masking)
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [5, 25])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 1])
        self.assertEqual(list(cfg.train.image_num_range), [6, 26])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertTrue(cfg.lmgeo_anchor.condition_reference_visibility)
        self.assertFalse(cfg.lmgeo_anchor.condition_query_visibility)
        self.assertEqual(cfg.lmgeo_anchor.anchor_selection_attempts, 200)
        self.assertFalse(cfg.train_dataset.LMGeoAnchorScenePair.depth_masking)
        self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.condition_reference_visibility)
        self.assertFalse(cfg.train_dataset.LMGeoAnchorScenePair.condition_query_visibility)
        self.assertFalse(cfg.val_datasets.real_anchor_pairs.enabled)
        self.assertFalse(cfg.val_datasets.pbr_anchor_pairs.enabled)
        self.assertEqual(
            active_val_loader_names(cfg),
            [
                "pbr_anchor_pairs_ref5",
                "pbr_anchor_pairs_ref10",
                "pbr_anchor_pairs_ref15",
            ],
        )

        expected_refs = {
            "pbr_anchor_pairs_ref5": 5,
            "pbr_anchor_pairs_ref10": 10,
            "pbr_anchor_pairs_ref15": 15,
        }
        for name, ref_count in expected_refs.items():
            dataset = cfg.val_datasets[name].dataset
            runtime = cfg.val_datasets[name].runtime
            self.assertEqual(dataset.query_split, cfg.lmgeo.new_val_query_split)
            self.assertEqual(list(dataset.num_reference_range), [ref_count, ref_count])
            self.assertEqual(list(dataset.num_query_range), [1, 1])
            self.assertEqual(list(runtime.image_num_range), [ref_count + 1, ref_count + 1])
            self.assertEqual(runtime.max_img_per_gpu, 26)
            self.assertFalse(dataset.depth_masking)
            self.assertTrue(dataset.condition_reference_visibility)
            self.assertFalse(dataset.condition_query_visibility)
            self.assertFalse(dataset.reference_rgb_masking)
            self.assertFalse(dataset.query_rgb_masking)

    def test_scratch_dinos_small_anchor_mask_profile_is_isolated(self):
        cfg = compose_job(
            "train_lmgeo_scratch_dinos_small_a40_40gb_anchor_mask_conditioning",
            "lmgeo_trainpbr45_anchor_scene_pairs_masked_depth_query_masks",
        )

        self.assertEqual(cfg.model.encoder_size, "small")
        self.assertTrue(cfg.model.encoder_pretrained)
        self.assertIsNone(cfg.model.encoder_ckpt)
        self.assertEqual(cfg.model.decoder_size, "small")
        self.assertEqual(cfg.model.head_dim, 384)
        self.assertEqual(cfg.model.point_decoder_dim, 384)
        self.assertEqual(cfg.model.point_decoder_heads, 6)
        self.assertEqual(cfg.model.camera_decoder_dim, 384)
        self.assertEqual(cfg.model.camera_decoder_heads, 6)
        self.assertEqual(cfg.model.camera_head_dim, 256)
        self.assertFalse(cfg.model.load_vggt)
        self.assertTrue(cfg.model.freeze_encoder)
        self.assertIsNone(cfg.model.ckpt)
        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertEqual(float(cfg.train.optimizer.lr), 5e-5)
        self.assertEqual(float(cfg.train.optimizer.encoder_lr), 0.0)
        self.assertEqual(float(cfg.train.optimizer.visibility_mask_lr), 5e-4)
        self.assertEqual(list(cfg.train.image_num_range), [3, 26])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertTrue(cfg.lmgeo.depth_masking)
        self.assertTrue(cfg.lmgeo_anchor.condition_reference_visibility)
        self.assertTrue(cfg.lmgeo_anchor.condition_query_visibility)
        self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.condition_reference_visibility)
        self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.condition_query_visibility)
        self.assertEqual(
            active_val_loader_names(cfg),
            ["real_anchor_pairs", "pbr_anchor_pairs"],
        )
        for name in active_val_loader_names(cfg):
            self.assertTrue(cfg.val_datasets[name].dataset.condition_query_visibility)

    def test_scratch_dinos_small_70gb_profile_increases_train_only_budget(self):
        cfg = compose_job(
            "train_lmgeo_scratch_dinos_small_rtxpro6000_70gb_anchor_mask_conditioning",
            "lmgeo_trainpbr45_anchor_scene_pairs_masked_depth_query_masks",
        )

        self.assertEqual(cfg.model.encoder_size, "small")
        self.assertEqual(cfg.model.decoder_size, "small")
        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertTrue(cfg.lmgeo_anchor.condition_reference_visibility)
        self.assertTrue(cfg.lmgeo_anchor.condition_query_visibility)
        self.assertEqual(list(cfg.train.image_num_range), [3, 26])
        self.assertEqual(cfg.train.max_img_per_gpu, 140)
        self.assertEqual(cfg.test.max_img_per_gpu, 128)
        for name in active_val_loader_names(cfg):
            runtime = cfg.val_datasets[name].runtime
            self.assertEqual(list(runtime.image_num_range), [6, 6])
            self.assertEqual(runtime.max_img_per_gpu, 128)
            self.assertTrue(cfg.val_datasets[name].dataset.condition_query_visibility)

    def test_anchor_scene_overfit_query_masks_profile_has_mask_controls(self):
        cfg = compose_job(
            "train_lmgeo_overfit_a40_40gb_anchor_mask_conditioning",
            "lmgeo_anchor_scene_overfit_scene13_sub19_query_masks",
        )

        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[560, 420]])
        self.assertEqual(list(cfg.train.image_num_range), [6, 6])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertEqual(cfg.train.iters_per_epoch, 20)
        self.assertEqual(list(cfg.lmgeo.query_scene_ids), [13])
        self.assertEqual(list(cfg.lmgeo.query_subscene_ids), [19])
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [5, 5])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 1])
        self.assertFalse(cfg.lmgeo.reference_rgb_masking)
        self.assertFalse(cfg.lmgeo.query_rgb_masking)
        self.assertTrue(cfg.lmgeo.depth_masking)
        self.assertEqual(cfg.primary_val, "overfit_scene_correct_masks")
        self.assertEqual(
            cfg.train_dataset.LMGeoAnchorScenePair._target_,
            "datasets.lmgeo_dataset.LMGeoAnchorScenePairSequenceDataset",
        )
        self.assertEqual(list(cfg.train_dataset.LMGeoAnchorScenePair.query_scene_ids), [13])
        self.assertEqual(list(cfg.train_dataset.LMGeoAnchorScenePair.query_subscene_ids), [19])
        self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.condition_reference_visibility)
        self.assertTrue(cfg.train_dataset.LMGeoAnchorScenePair.condition_query_visibility)
        self.assertEqual(cfg.train_dataset.LMGeoAnchorScenePair.visibility_condition_corruption, "none")
        self.assertEqual(
            active_val_loader_names(cfg),
            [
                "overfit_scene_correct_masks",
                "overfit_scene_shifted_masks",
                "overfit_scene_no_masks",
            ],
        )

        correct = cfg.val_datasets.overfit_scene_correct_masks.dataset
        shifted = cfg.val_datasets.overfit_scene_shifted_masks.dataset
        no_masks = cfg.val_datasets.overfit_scene_no_masks.dataset
        self.assertTrue(correct.condition_reference_visibility)
        self.assertTrue(correct.condition_query_visibility)
        self.assertEqual(correct.visibility_condition_corruption, "none")
        self.assertTrue(shifted.condition_reference_visibility)
        self.assertTrue(shifted.condition_query_visibility)
        self.assertEqual(shifted.visibility_condition_corruption, "shift")
        self.assertEqual(shifted.visibility_condition_shift_fraction, 0.5)
        self.assertFalse(no_masks.condition_reference_visibility)
        self.assertFalse(no_masks.condition_query_visibility)
        self.assertEqual(no_masks.visibility_condition_corruption, "none")
        for name in active_val_loader_names(cfg):
            runtime = cfg.val_datasets[name].runtime
            dataset = cfg.val_datasets[name].dataset
            self.assertEqual(list(runtime.image_num_range), [6, 6])
            self.assertEqual(runtime.max_img_per_gpu, 28)
            self.assertEqual(list(dataset.num_reference_range), [5, 5])
            self.assertEqual(list(dataset.num_query_range), [1, 1])

        self.assertIsNone(cfg.metrics["items"].chamfer)
        self.assertIsNone(cfg.visuals["items"].query_pose_overlay)
        self.assertIsNone(cfg.visuals["items"].reference_reconstruction)
        self.assertIn("visibility_condition", cfg.visuals["items"])

    def test_anchor_scene_overfit_no_masks_profile_disables_train_conditioning(self):
        cfg = compose_job(
            "train_lmgeo_overfit_a40_40gb_anchor_mask_conditioning",
            "lmgeo_anchor_scene_overfit_scene13_sub19_no_masks",
        )

        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertEqual(cfg.primary_val, "overfit_scene_no_masks")
        self.assertFalse(cfg.lmgeo_anchor.condition_reference_visibility)
        self.assertFalse(cfg.lmgeo_anchor.condition_query_visibility)
        self.assertFalse(cfg.train_dataset.LMGeoAnchorScenePair.condition_reference_visibility)
        self.assertFalse(cfg.train_dataset.LMGeoAnchorScenePair.condition_query_visibility)
        self.assertFalse(cfg.val_datasets.overfit_scene_no_masks.dataset.condition_reference_visibility)
        self.assertFalse(cfg.val_datasets.overfit_scene_no_masks.dataset.condition_query_visibility)

    def test_anchor_cross_scene_overfit_profile_forces_reference_and_query_windows(self):
        cfg = compose_job(
            "train_lmgeo_overfit_a40_40gb_anchor_mask_conditioning_biglr",
            "lmgeo_anchor_cross_scene_overfit_scene13_sub19_to_scene3_sub25_query_masks",
        )

        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertEqual(cfg.train.optimizer.lr, 1e-5)
        self.assertEqual(cfg.train.optimizer.visibility_mask_lr, 1e-4)
        self.assertEqual(list(cfg.train.image_num_range), [6, 6])
        self.assertEqual(cfg.train.max_img_per_gpu, 28)
        self.assertEqual(list(cfg.lmgeo.query_scene_ids), [3])
        self.assertEqual(list(cfg.lmgeo.query_subscene_ids), [25])
        self.assertEqual(list(cfg.lmgeo.num_reference_range), [5, 5])
        self.assertEqual(list(cfg.lmgeo.num_query_range), [1, 1])
        self.assertEqual(cfg.primary_val, "cross_scene_correct_masks")

        train_dataset = cfg.train_dataset.LMGeoAnchorScenePair
        self.assertEqual(train_dataset._target_, "datasets.lmgeo_dataset.LMGeoAnchorScenePairSequenceDataset")
        self.assertFalse(train_dataset.anchor_allow_same_scene)
        self.assertFalse(train_dataset.anchor_allow_same_subscene)
        self.assertTrue(train_dataset.condition_reference_visibility)
        self.assertTrue(train_dataset.condition_query_visibility)
        self.assertEqual(list(train_dataset.query_scene_ids), [3])
        self.assertEqual(list(train_dataset.query_subscene_ids), [25])
        self.assertEqual(len(train_dataset.anchor_reference_windows), 8)
        for window in train_dataset.anchor_reference_windows:
            self.assertEqual(window.query_scene_id, 13)
            self.assertEqual(window.query_subscene_id, 19)

        self.assertEqual(
            active_val_loader_names(cfg),
            [
                "cross_scene_correct_masks",
                "cross_scene_shifted_masks",
                "cross_scene_no_masks",
            ],
        )
        correct = cfg.val_datasets.cross_scene_correct_masks.dataset
        shifted = cfg.val_datasets.cross_scene_shifted_masks.dataset
        no_masks = cfg.val_datasets.cross_scene_no_masks.dataset
        self.assertTrue(correct.condition_reference_visibility)
        self.assertTrue(correct.condition_query_visibility)
        self.assertEqual(correct.visibility_condition_corruption, "none")
        self.assertEqual(shifted.visibility_condition_corruption, "shift")
        self.assertFalse(no_masks.condition_reference_visibility)
        self.assertFalse(no_masks.condition_query_visibility)
        for name in active_val_loader_names(cfg):
            runtime = cfg.val_datasets[name].runtime
            dataset = cfg.val_datasets[name].dataset
            self.assertEqual(list(runtime.image_num_range), [6, 6])
            self.assertEqual(runtime.max_img_per_gpu, 28)
            self.assertEqual(list(dataset.query_scene_ids), [3])
            self.assertEqual(list(dataset.query_subscene_ids), [25])
            self.assertEqual(len(dataset.anchor_reference_windows), 8)

    def test_anchor_cross_scene_overfit_no_masks_profile_disables_train_conditioning(self):
        cfg = compose_job(
            "train_lmgeo_overfit_a40_40gb_anchor_mask_conditioning_biglr",
            "lmgeo_anchor_cross_scene_overfit_scene13_sub19_to_scene3_sub25_no_masks",
        )

        self.assertEqual(cfg.primary_val, "cross_scene_no_masks")
        self.assertFalse(cfg.lmgeo_anchor.condition_reference_visibility)
        self.assertFalse(cfg.lmgeo_anchor.condition_query_visibility)
        self.assertFalse(cfg.train_dataset.LMGeoAnchorScenePair.condition_reference_visibility)
        self.assertFalse(cfg.train_dataset.LMGeoAnchorScenePair.condition_query_visibility)
        self.assertFalse(cfg.val_datasets.cross_scene_no_masks.dataset.condition_reference_visibility)
        self.assertFalse(cfg.val_datasets.cross_scene_no_masks.dataset.condition_query_visibility)

    def test_generic_pi3_profile_is_unchanged(self):
        cfg = compose_job("train_pi3_lowres", "example")

        self.assertEqual(list(cfg.train.image_num_range), [2, 24])
        self.assertEqual(cfg.train.max_img_per_gpu, 64)
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[224, 224]])
        self.assertNotIn("lmgeo_profile", cfg)
        self.assertEqual(cfg.model.encoder_size, "large")
        self.assertFalse(cfg.model.encoder_pretrained)
        self.assertIsNone(cfg.model.encoder_ckpt)
        self.assertEqual(cfg.model.decoder_size, "large")
        self.assertEqual(cfg.model.head_dim, 1024)
        self.assertEqual(cfg.model.point_decoder_dim, 1024)
        self.assertEqual(cfg.model.point_decoder_heads, 16)
        self.assertEqual(cfg.model.camera_decoder_dim, 1024)
        self.assertEqual(cfg.model.camera_decoder_heads, 16)
        self.assertEqual(cfg.model.camera_head_dim, 512)
        self.assertTrue(cfg.model.load_vggt)
        self.assertFalse(cfg.model.use_ray_conditioning)
        self.assertFalse(cfg.model.use_visibility_mask_conditioning)


if __name__ == "__main__":
    unittest.main()
