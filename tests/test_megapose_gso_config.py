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

    def test_blackwell_70gb_dynamic_profile_uses_measured_local_budgets(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_finetune_rtxpro6000_blackwell_70gb_336x252_dynamic",
                    "data=megapose_gso_anchor_scene_pairs_multival",
                ],
            )
        self.assertEqual(
            OmegaConf.to_container(cfg.train.resolution), [[336, 252]]
        )
        self.assertEqual(list(cfg.train.image_num_range), [3, 26])
        self.assertEqual(cfg.train.max_img_per_gpu, 156)
        self.assertEqual(list(cfg.gso_profile.num_reference_range), [2, 16])
        self.assertEqual(list(cfg.gso_profile.num_query_range), [1, 10])
        for entry in cfg.val_datasets.values():
            self.assertEqual(entry.dataset.sampling_regime, "anchor_pair")
            self.assertFalse(entry.dataset.allow_repeat)
            self.assertEqual(entry.runtime.max_img_per_gpu, 768)
            self.assertEqual(entry.runtime.num_workers, 2)
        self.assertFalse(cfg.test.persistent_workers)
        self.assertEqual(cfg.test.prefetch_factor, 2)
        self.assertEqual(cfg.visuals.val_every_n_epochs, 5)

    def test_composable_mixture_exposes_independent_protocol_axes(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_finetune_a40_40gb_336x252_dynamic",
                    "data=megapose_gso_composable_mixed",
                ],
            )
        self.assertEqual(
            cfg.train_dataset.GSORenderToScene._target_,
            "datasets.object_pose_dataset.ComposableObjectPoseDataset",
        )
        self.assertEqual(
            cfg.train_dataset.GSORenderToScene.sampling_policy._target_,
            "datasets.object_sampling.RenderToScenePolicy",
        )
        self.assertEqual(
            cfg.train_dataset.GSOScenePairBothConditioned.sampling_policy._target_,
            "datasets.object_sampling.ScenePairPolicy",
        )
        self.assertEqual(
            cfg.train_dataset.GSOHybridReferences.sampling_policy._target_,
            "datasets.object_sampling.HybridReferencePolicy",
        )
        self.assertEqual(
            OmegaConf.to_container(cfg.train_dataset.weights),
            {
                "GSOScenePairBothConditioned": 4,
                "GSOScenePairRGBMasked": 1,
                "GSORenderToScene": 4,
                "GSOHybridReferences": 2,
            },
        )
        masked = cfg.train_dataset.GSOScenePairRGBMasked
        self.assertEqual(masked.reference_treatment.rgb, "object_only")
        self.assertEqual(masked.query_treatment.mask_condition, "none")
        render = cfg.train_dataset.GSORenderToScene
        self.assertEqual(set(render.sources), {"render", "scene"})
        self.assertEqual(render.sources.render.object_namespace, "gso")
        self.assertEqual(render.object_model_catalog.object_namespace, "gso")
        self.assertEqual(
            list(cfg.val_datasets),
            [
                "gso_val_render_to_scene_coverage",
                "gso_val_scene_pair",
                "gso_val_render_to_scene_contiguous",
                "gso_val_hybrid",
            ],
        )
        self.assertEqual(
            cfg.val_datasets.gso_val_render_to_scene_coverage.dataset.sampling_policy.reference_selection,
            "coverage_uniform",
        )
        self.assertEqual(
            cfg.val_datasets.gso_val_render_to_scene_contiguous.dataset.sampling_policy.reference_selection,
            "first",
        )

    def test_composable_smoke_profile_is_bounded_and_validates(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_composable_smoke",
                    "data=megapose_gso_composable_mixed",
                ],
            )
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[56, 42]])
        self.assertEqual(list(cfg.train.image_num_range), [6, 6])
        self.assertEqual(cfg.train.iters_per_epoch, 14)
        self.assertEqual(cfg.train.num_epoch, 1)
        self.assertEqual(cfg.test.iters_per_test, 1)
        self.assertEqual(cfg.model.encoder_size, "small")
        self.assertEqual(cfg.model.point_decoder_depth, 2)
        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertFalse(cfg.visuals.enabled)
        self.assertEqual(
            cfg.metrics["items"].object_pose._target_,
            "pi3.metrics.object_pose.ObjectPoseMetric",
        )
        self.assertEqual(
            cfg.metrics["items"].object_pose.data_root, cfg.gso.assets_root
        )
        self.assertEqual(
            list(cfg.metrics["items"].object_pose.symmetric_ids), []
        )

    def test_composable_production_profiles_restore_mesh_pose_metric(self):
        profile_expectations = {
            "train_megapose_gso_composable_a40_40gb_336x252_dynamic": (72, 256),
            "train_megapose_gso_composable_rtxpro6000_blackwell_70gb_336x252_dynamic": (
                156,
                768,
            ),
        }
        for profile, (train_budget, val_budget) in profile_expectations.items():
            with self.subTest(profile=profile):
                with initialize_config_dir(
                    version_base="1.2", config_dir=str(CONFIG_DIR)
                ):
                    cfg = compose(
                        config_name="default",
                        overrides=[
                            f"train={profile}",
                            "data=megapose_gso_composable_mixed",
                        ],
                    )
                self.assertEqual(cfg.train.max_img_per_gpu, train_budget)
                self.assertEqual(cfg.test.max_img_per_gpu, val_budget)
                self.assertEqual(
                    cfg.metrics["items"].object_pose._target_,
                    "pi3.metrics.object_pose.ObjectPoseMetric",
                )
                self.assertEqual(
                    cfg.metrics["items"].object_pose.models_folder,
                    cfg.gso.models_folder,
                )

    def test_instance_disambiguated_render_scene_pair_baseline_is_strict_and_routed(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_baseline_rtxpro6000_blackwell_70gb_336x252",
                    "data=megapose_gso_baseline_render_scene_pair_lmo_val",
                ],
            )
        self.assertTrue(cfg.model.use_visibility_mask_conditioning)
        self.assertEqual(cfg.model.visibility_mask_conditioning_alpha, 1.0)
        self.assertEqual(cfg.train.optimizer.lr, 5e-6)
        self.assertEqual(cfg.train.optimizer.visibility_mask_lr, 5e-5)
        self.assertEqual(cfg.train.max_img_per_gpu, 156)
        self.assertEqual(cfg.test.max_img_per_gpu, 768)
        self.assertEqual(
            OmegaConf.to_container(cfg.train_dataset.weights),
            {"GSORenderToScene": 1, "GSOScenePairMaskedReferences": 1},
        )
        render = cfg.train_dataset.GSORenderToScene
        scene_pair = cfg.train_dataset.GSOScenePairMaskedReferences
        self.assertEqual(render.reference_treatment.rgb, "full")
        self.assertEqual(render.query_treatment.rgb, "full")
        self.assertEqual(scene_pair.reference_treatment.rgb, "object_only")
        self.assertEqual(scene_pair.query_treatment.rgb, "full")
        for dataset in (render, scene_pair):
            self.assertEqual(
                dataset.query_treatment.mask_condition, "object_if_repeated"
            )
            self.assertFalse(
                dataset.sources.scene.get("exclude_repeated_object_scenes", False)
            )
        self.assertEqual(render.reference_treatment.mask_condition, "none")
        self.assertEqual(scene_pair.reference_treatment.mask_condition, "none")
        self.assertEqual(
            list(cfg.val_datasets),
            [
                "gso_heldout_render_n5_k1",
                "gso_heldout_render_n16_k1",
                "lmo_bop19_render_n5_k1",
            ],
        )
        for name, total in (
            ("gso_heldout_render_n5_k1", 6),
            ("gso_heldout_render_n16_k1", 17),
        ):
            entry = cfg.val_datasets[name]
            self.assertEqual(entry.dataset.sources.scene.scene_split, "val")
            self.assertTrue(entry.dataset.sampling_policy.enumerate_query_groups)
            self.assertEqual(
                entry.dataset.sampling_policy.reference_selection,
                "coverage_uniform",
            )
            self.assertEqual(list(entry.runtime.image_num_range), [total, total])
            self.assertEqual(entry.runtime.iters_per_test, 0)
        lmo = cfg.val_datasets.lmo_bop19_render_n5_k1.dataset
        self.assertEqual(lmo.query_source, "bop_targets")
        self.assertFalse(lmo.query_rgb_masking)
        self.assertEqual(list(lmo.num_reference_range), [5, 5])
        self.assertEqual(list(lmo.num_query_range), [1, 1])
        self.assertIsNone(cfg.metrics["items"].object_pose)
        self.assertEqual(
            list(cfg.metrics["items"].gso_object_pose.object_model_namespaces),
            ["gso"],
        )
        self.assertEqual(
            list(cfg.metrics["items"].lmo_object_pose.object_model_namespaces),
            ["lmo"],
        )
        self.assertFalse(cfg.metrics["items"].gso_object_pose.report_per_object)
        self.assertFalse(cfg.metrics["items"].lmo_object_pose.report_per_object)

    def test_transfer_profile_composes_all_diagnostics_and_train_treatments(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_transfer_rtxpro6000_blackwell_70gb_336x252",
                    "data=megapose_gso_transfer_diagnostics",
                ],
            )
        self.assertEqual(cfg.train.num_epoch, 80)
        self.assertEqual(cfg.train.val_every_n_epochs, 3)
        self.assertEqual(cfg.train.optimizer.lr, 1e-5)
        self.assertEqual(cfg.train.optimizer.encoder_lr, 0.0)
        self.assertEqual(cfg.train.optimizer.visibility_mask_lr, 5e-5)
        self.assertEqual(cfg.train.lr_scheduler.pct_start, 0.0375)
        self.assertTrue(cfg.model.freeze_encoder)
        for name in ("GSORenderToScene", "GSOScenePairMaskedReferences"):
            dataset = cfg.train_dataset[name]
            self.assertEqual(
                dataset.query_treatment.mask_condition,
                "object_if_repeated_else_probability",
            )
            self.assertEqual(
                dataset.query_treatment.mask_condition_probability, 0.5
            )
            self.assertTrue(dataset.photometric_augmentation)
            self.assertEqual(list(dataset.photometric_brightness), [0.7, 1.3])

        self.assertEqual(
            list(cfg.val_datasets),
            [
                "gso_heldout_render_n5_k1",
                "gso_heldout_render_n16_k1",
                "lmo_bop19_render_n5_k1",
                "gso_heldout_render_n5_k1_all_query_conditioned",
                "gso_heldout_render_n5_k1_visibility_gt_0_5",
                "lmo_pbr_new_val_render_n5_k1",
                "lmo_pbr_new_val_render_n5_k1_all_query_conditioned",
                "lmo_bop19_render_n5_k1_all_query_conditioned",
            ],
        )
        conditioned_gso = cfg.val_datasets[
            "gso_heldout_render_n5_k1_all_query_conditioned"
        ].dataset
        self.assertEqual(conditioned_gso.query_treatment.mask_condition, "object")
        strict = cfg.val_datasets[
            "gso_heldout_render_n5_k1_visibility_gt_0_5"
        ].dataset
        self.assertEqual(strict.sources.scene.visibility_min, 0.5)
        self.assertFalse(strict.sources.scene.visibility_min_inclusive)
        self.assertEqual(
            strict.query_treatment.mask_condition, "object_if_repeated"
        )
        pbr = cfg.val_datasets.lmo_pbr_new_val_render_n5_k1.dataset
        pbr_conditioned = cfg.val_datasets[
            "lmo_pbr_new_val_render_n5_k1_all_query_conditioned"
        ].dataset
        real_conditioned = cfg.val_datasets[
            "lmo_bop19_render_n5_k1_all_query_conditioned"
        ].dataset
        self.assertEqual(pbr.query_split, "new_val")
        self.assertEqual(pbr.query_source, "windows")
        self.assertFalse(pbr.filter_preprocessed_query_depth)
        self.assertFalse(
            pbr_conditioned.filter_preprocessed_query_depth
        )
        self.assertFalse(pbr.visibility_mask_conditioning)
        for dataset in (pbr_conditioned, real_conditioned):
            self.assertTrue(dataset.visibility_mask_conditioning)
            self.assertFalse(dataset.condition_reference_visibility)
            self.assertTrue(dataset.condition_query_visibility)
            self.assertEqual(dataset.mode, "val")


if __name__ == "__main__":
    unittest.main()
