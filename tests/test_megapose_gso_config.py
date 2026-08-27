import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class MegaPoseGSOConfigTest(unittest.TestCase):
    def test_geometry_max_batch_overfit_profile_packs_24_fixed_samples(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_small_scratch_ray_336x252_overfit_max_batch",
                    "data=megapose_gso_geometry_n5_k1_masked_overfit_max_batch",
                ],
            )
        self.assertEqual(len(cfg.gso.object_ids), 24)
        self.assertEqual(len(set(cfg.gso.object_ids)), 24)
        self.assertEqual(cfg.train_dataset.length, 25)
        self.assertEqual(cfg.train.image_num_range, [6, 6])
        self.assertEqual(cfg.train.max_img_per_gpu, 144)
        self.assertEqual(cfg.test.max_img_per_gpu, 144)
        self.assertEqual(cfg.train.iters_per_epoch, 1)
        self.assertEqual(cfg.train.num_epoch, 600)
        self.assertEqual(cfg.train.val_every_n_epochs, 10)
        self.assertEqual(cfg.train.optimizer.lr, 1e-4)
        self.assertEqual(cfg.train.optimizer.ray_lr, 1e-3)
        self.assertEqual(
            cfg.train_dataset.GSOSceneGeometryN5K1.protocol_name,
            "geometry_scene_n5_k1_masked_overfit_max_batch_24",
        )
        self.assertFalse(
            cfg.train_dataset.GSOSceneGeometryN5K1.photometric_augmentation
        )
        self.assertEqual(
            cfg.val_datasets.gso_geometry_overfit_train_batch.dataset.protocol_name,
            "geometry_scene_n5_k1_masked_overfit_max_batch_24",
        )
        self.assertEqual(
            cfg.val_datasets.gso_geometry_overfit_train_batch.runtime.max_img_per_gpu,
            144,
        )

    def test_geometry_one_batch_overfit_profile_is_exact_and_isolated(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_small_scratch_ray_336x252_overfit_one_batch",
                    "data=megapose_gso_geometry_n5_k1_masked_overfit_one_batch",
                ],
            )
        self.assertEqual(list(cfg.gso.object_ids), [0])
        self.assertEqual(cfg.train_dataset.length, 11)
        self.assertEqual(cfg.train_dataset.weights.GSOSceneGeometryN5K1, 1)
        self.assertEqual(cfg.train_dataset.weights.GSORenderGeometryN5K1, 0)
        policy = cfg.train_dataset.GSOSceneGeometryN5K1.sampling_policy
        self.assertEqual(policy.plan_selection, "first")
        self.assertEqual(policy.reference_selection, "first")
        self.assertEqual(policy.query_selection, "first")
        self.assertEqual(policy.crop_center_jitter, 0.0)
        self.assertFalse(policy.random_focal_target)
        self.assertFalse(policy.per_view_focal_targets)
        self.assertFalse(
            cfg.train_dataset.GSOSceneGeometryN5K1.photometric_augmentation
        )
        self.assertEqual(cfg.train.max_img_per_gpu, 6)
        self.assertEqual(cfg.train.iters_per_epoch, 10)
        self.assertEqual(cfg.train.num_epoch, 60)
        self.assertEqual(cfg.train.optimizer.lr, 1e-4)
        self.assertEqual(cfg.train.optimizer.ray_lr, 1e-3)
        self.assertEqual(cfg.train.optimizer.weight_decay, 0.0)
        self.assertEqual(cfg.primary_val, "gso_geometry_overfit_train_batch")
        self.assertFalse(cfg.val_datasets.gso_geometry_val_scene_n5_k1.enabled)
        self.assertFalse(cfg.val_datasets.gso_geometry_val_render_n5_k1.enabled)
        overfit_val = cfg.val_datasets.gso_geometry_overfit_train_batch
        self.assertEqual(overfit_val.dataset.sources.scene.scene_split, "train")
        self.assertEqual(
            overfit_val.dataset.protocol_name,
            "geometry_scene_n5_k1_masked_overfit_one_batch",
        )
        self.assertEqual(overfit_val.runtime.iters_per_test, 1)

    def test_geometry_n5_k1_small_scratch_profile_is_additive_and_consistent(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_small_scratch_ray_336x252",
                    "data=megapose_gso_geometry_n5_k1_masked",
                ],
            )
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[336, 252]])
        self.assertEqual(list(cfg.train.image_num_range), [6, 6])
        self.assertEqual(cfg.model.encoder_size, "small")
        self.assertEqual(cfg.model.decoder_size, "small")
        self.assertTrue(cfg.model.freeze_encoder)
        self.assertTrue(cfg.model.use_ray_conditioning)
        self.assertFalse(cfg.model.use_visibility_mask_conditioning)
        self.assertEqual(cfg.train.optimizer.lr, 5e-5)
        self.assertEqual(cfg.train.optimizer.ray_lr, 5e-4)
        self.assertEqual(
            list(cfg.train_dataset.weights),
            ["GSOSceneGeometryN5K1", "GSORenderGeometryN5K1"],
        )
        targets = {
            "GSOSceneGeometryN5K1":
                "datasets.object_sampling.GeometryConstrainedScenePairPolicy",
            "GSORenderGeometryN5K1":
                "datasets.object_sampling.GeometryConstrainedRenderToScenePolicy",
        }
        for name, target in targets.items():
            dataset = cfg.train_dataset[name]
            self.assertEqual(dataset.sampling_policy._target_, target)
            self.assertTrue(dataset.photometric_augmentation)
            self.assertEqual(list(dataset.photometric_brightness), [0.7, 1.3])
            self.assertEqual(dataset.photometric_jpeg_prob, 0.5)
            self.assertEqual(dataset.photometric_blur_prob, 0.5)
            self.assertEqual(list(dataset.sampling_policy.num_reference_range), [5, 5])
            self.assertEqual(list(dataset.sampling_policy.num_query_range), [1, 1])
            self.assertEqual(dataset.sampling_policy.positive_angle_degrees, 10.0)
            self.assertEqual(dataset.sampling_policy.focal_relative_tolerance, 0.1)
            self.assertEqual(dataset.sampling_policy.crop_aspect, 4 / 3)
            self.assertTrue(dataset.sampling_policy.per_view_focal_targets)
            self.assertEqual(dataset.reference_treatment.rgb, "object_only")
            self.assertEqual(dataset.query_treatment.rgb, "object_only")
            self.assertEqual(dataset.reference_treatment.mask_condition, "none")
        self.assertEqual(
            cfg.train_dataset.GSOSceneGeometryN5K1.sampling_policy.reference_geometry_index_path,
            cfg.gso.geometry_train_index,
        )
        self.assertEqual(
            cfg.train_dataset.GSORenderGeometryN5K1.sampling_policy.reference_geometry_index_path,
            cfg.gso.render_geometry_index,
        )
        self.assertEqual(
            list(cfg.val_datasets),
            ["gso_geometry_val_scene_n5_k1", "gso_geometry_val_render_n5_k1"],
        )
        for entry in cfg.val_datasets.values():
            self.assertFalse(
                OmegaConf.select(
                    entry.dataset,
                    "photometric_augmentation",
                    default=False,
                )
            )
            self.assertEqual(entry.dataset.sources.scene.scene_split, "val")
            self.assertEqual(entry.dataset.sampling_policy.crop_center_jitter, 0.0)
            self.assertFalse(entry.dataset.sampling_policy.random_focal_target)
            self.assertFalse(entry.dataset.sampling_policy.per_view_focal_targets)
        self.assertEqual(
            cfg.val_datasets.gso_geometry_val_scene_n5_k1.dataset.sampling_policy.reference_geometry_index_path,
            cfg.gso.geometry_val_index,
        )
        self.assertEqual(
            cfg.val_datasets.gso_geometry_val_render_n5_k1.dataset.sampling_policy.reference_geometry_index_path,
            cfg.gso.render_geometry_index,
        )

    def test_geometry_blackwell_production_profile_uses_measured_batch(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_small_scratch_ray_rtxpro6000_70gb_336x252",
                    "data=megapose_gso_geometry_n5_k1_masked",
                ],
            )
        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[336, 252]])
        self.assertEqual(list(cfg.train.image_num_range), [6, 6])
        self.assertEqual(cfg.train.max_img_per_gpu, 288)
        self.assertEqual(cfg.train.max_img_per_gpu // 6, 48)
        self.assertEqual(cfg.train.num_workers, 8)
        self.assertTrue(cfg.train.persistent_workers)
        self.assertEqual(cfg.train.num_epoch, 80)
        self.assertEqual(cfg.train.iters_per_epoch, 500)
        self.assertEqual(cfg.train.lr_scheduler.pct_start, 0.0125)
        self.assertEqual(cfg.train.optimizer.lr, 1e-4)
        self.assertEqual(cfg.train.optimizer.ray_lr, 1e-4)
        self.assertEqual(cfg.train.optimizer.encoder_lr, 0.0)
        self.assertFalse(cfg.metrics["items"].object_pose.report_per_object)
        self.assertEqual(cfg.test.num_workers, 2)
        self.assertFalse(cfg.test.persistent_workers)
        self.assertEqual(cfg.test.iters_per_test, 0)
        self.assertTrue(cfg.log.save_best)
        self.assertEqual(
            list(cfg.val_datasets),
            ["gso_geometry_val_scene_n5_k1", "gso_geometry_val_render_n5_k1"],
        )
        for entry in cfg.val_datasets.values():
            self.assertEqual(entry.runtime.num_workers, 2)
            self.assertEqual(entry.runtime.iters_per_test, 0)
            self.assertEqual(list(entry.runtime.image_num_range), [6, 6])

    def test_geometry_pretrained_pi3_ray_blackwell_profile_is_consistent(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_pi3_finetune_ray_rtxpro6000_70gb_336x252",
                    "data=megapose_gso_geometry_n5_k1_masked",
                ],
            )

        self.assertEqual(OmegaConf.to_container(cfg.train.resolution), [[336, 252]])
        self.assertEqual(list(cfg.train.image_num_range), [6, 6])
        self.assertEqual(cfg.model.encoder_size, "large")
        self.assertEqual(cfg.model.decoder_size, "large")
        self.assertTrue(cfg.model.freeze_encoder)
        self.assertFalse(cfg.model.load_vggt)
        self.assertEqual(cfg.model.ckpt, "ckpts/Pi3/model.safetensors")
        self.assertTrue(cfg.model.use_ray_conditioning)
        self.assertFalse(cfg.model.use_visibility_mask_conditioning)
        self.assertEqual(cfg.train.max_img_per_gpu, 156)
        self.assertEqual(cfg.train.max_img_per_gpu // 6, 26)
        self.assertEqual(cfg.train.num_workers, 8)
        self.assertTrue(cfg.train.persistent_workers)
        self.assertEqual(cfg.train.num_epoch, 30)
        self.assertEqual(cfg.train.iters_per_epoch, 500)
        self.assertEqual(cfg.train.num_epoch * cfg.train.iters_per_epoch, 15000)
        self.assertAlmostEqual(
            cfg.train.lr_scheduler.pct_start
            * cfg.train.num_epoch
            * cfg.train.iters_per_epoch,
            500.0,
        )
        self.assertEqual(cfg.train.optimizer.lr, 5e-6)
        self.assertEqual(cfg.train.optimizer.ray_lr, 1e-5)
        self.assertEqual(cfg.train.optimizer.encoder_lr, 0.0)
        self.assertEqual(cfg.test.max_img_per_gpu, 768)
        self.assertEqual(cfg.test.max_img_per_gpu // 6, 128)
        self.assertEqual(cfg.test.num_workers, 2)
        self.assertFalse(cfg.test.persistent_workers)
        self.assertFalse(cfg.metrics["items"].object_pose.report_per_object)
        self.assertIn("ray_geometry", cfg.metrics["items"])
        self.assertEqual(
            list(cfg.val_datasets),
            ["gso_geometry_val_scene_n5_k1", "gso_geometry_val_render_n5_k1"],
        )
        for entry in cfg.val_datasets.values():
            self.assertEqual(entry.runtime.max_img_per_gpu, 768)
            self.assertEqual(entry.runtime.num_workers, 2)
            self.assertEqual(entry.runtime.iters_per_test, 0)

    def test_metric_virtual_query_profile_is_isolated_and_scale_fixed(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_pi3_finetune_ray_metric_virtual_a40_40gb_560x420",
                    "data=megapose_gso_geometry_n5_k1_masked_metric_virtual_query",
                ],
            )

        self.assertEqual(cfg.loss.train_loss.scale_mode, "metric")
        self.assertEqual(cfg.loss.test_loss.scale_mode, "metric")
        self.assertEqual(cfg.train.clip_loss, 1000)
        self.assertFalse(cfg.metrics["items"].object_pose.solve_scale)
        self.assertFalse(cfg.metrics["items"].camera.solve_scale)
        self.assertFalse(
            cfg.visuals["items"].reference_reconstruction.solve_scale
        )
        self.assertEqual(
            OmegaConf.to_container(cfg.train.resolution), [[560, 420]]
        )
        datasets = [
            cfg.train_dataset.GSOSceneGeometryN5K1,
            cfg.train_dataset.GSORenderGeometryN5K1,
            cfg.test_dataset,
            *(entry.dataset for entry in cfg.val_datasets.values()),
        ]
        for dataset in datasets:
            virtual = dataset.virtual_camera_rectification
            self.assertEqual(list(virtual.roles), ["query"])
            self.assertEqual(list(virtual.zoom_range), [1.0, 5.0])
            self.assertEqual(virtual.eval_zoom, 3.0)
            self.assertEqual(virtual.zoom_sampling, "log_uniform")
            self.assertTrue(virtual.safe_zoom)
            self.assertTrue(virtual.replace_planned_crop)
        # This first experiment deliberately reuses the existing plan DBs.
        self.assertEqual(
            cfg.train_dataset.GSOSceneGeometryN5K1.sampling_policy.geometry_plan_path,
            cfg.gso.scene_train_plans,
        )
        self.assertEqual(
            cfg.train_dataset.GSORenderGeometryN5K1.sampling_policy.geometry_plan_path,
            cfg.gso.render_plans,
        )

        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            baseline = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_pi3_finetune_ray_a40_40gb_560x420",
                    "data=megapose_gso_geometry_n5_k1_masked",
                ],
            )
        self.assertEqual(baseline.loss.train_loss.scale_mode, "aligned")
        self.assertNotIn(
            "virtual_camera_rectification",
            baseline.train_dataset.GSOSceneGeometryN5K1,
        )

    def test_metric_virtual_query_336_profile_keeps_previous_336_budget(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_pi3_finetune_ray_metric_virtual_rtxpro6000_70gb_336x252",
                    "data=megapose_gso_geometry_n5_k1_masked_metric_virtual_query",
                ],
            )

        self.assertEqual(
            OmegaConf.to_container(cfg.train.resolution), [[336, 252]]
        )
        self.assertEqual(cfg.train.max_img_per_gpu, 156)
        self.assertEqual(cfg.test.max_img_per_gpu, 768)
        self.assertEqual(cfg.train.image_num_range, [6, 6])
        self.assertEqual(cfg.test.image_num_range, [6, 6])
        self.assertTrue(cfg.model.use_ray_conditioning)
        self.assertEqual(cfg.loss.train_loss.scale_mode, "metric")
        self.assertEqual(cfg.loss.test_loss.scale_mode, "metric")
        self.assertTrue(
            cfg.metrics["items"].object_pose.query_occupancy_analysis
        )
        unresolved = OmegaConf.to_container(cfg, resolve=False)
        self.assertEqual(
            unresolved["metrics"]["items"]["object_pose"][
                "query_occupancy_artifact_dir"
            ],
            "${log.output_dir}/query_occupancy",
        )
        self.assertEqual(
            list(
                cfg.train_dataset.GSOSceneGeometryN5K1
                .virtual_camera_rectification.zoom_range
            ),
            [1.0, 5.0],
        )

    def test_metric_depth_safe_fill_profile_is_opt_in_and_relaxes_only_focal_pairing(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_pi3_finetune_ray_metric_depth_safe_fill_rtxpro6000_70gb_336x252",
                    "data=megapose_gso_geometry_n5_k1_masked_metric_safe_fill_depth",
                ],
            )

        self.assertTrue(cfg.model.use_metric_depth_conditioning)
        self.assertEqual(cfg.model.metric_depth_reference_probability, 0.9)
        self.assertEqual(cfg.model.metric_depth_query_probability, 0.0)
        self.assertEqual(cfg.model.metric_depth_dropout_granularity, "sample")
        self.assertEqual(cfg.train.optimizer.metric_depth_lr, 1e-5)
        self.assertIn("metric_depth", cfg.metrics["items"])
        self.assertFalse(
            cfg.visuals["items"].metric_depth_panel.align_scale
        )
        for dataset in (
            cfg.train_dataset.GSOSceneGeometryN5K1,
            cfg.train_dataset.GSORenderGeometryN5K1,
            *(entry.dataset for entry in cfg.val_datasets.values()),
        ):
            self.assertFalse(
                dataset.sampling_policy.enforce_focal_compatibility
            )
            self.assertEqual(
                dataset.virtual_camera_rectification.zoom_policy, "safe_fill"
            )
            self.assertEqual(
                list(
                    dataset.virtual_camera_rectification.safe_fill_fraction_range
                ),
                [0.8, 1.0],
            )
            self.assertEqual(
                dataset.virtual_camera_rectification.eval_safe_fill_fraction,
                0.9,
            )

        # Baseline composition still has neither module nor relaxed policy.
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            baseline = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_pi3_finetune_ray_metric_virtual_rtxpro6000_70gb_336x252",
                    "data=megapose_gso_geometry_n5_k1_masked_metric_virtual_query",
                ],
            )
        self.assertFalse(baseline.model.use_metric_depth_conditioning)
        self.assertNotIn(
            "enforce_focal_compatibility",
            baseline.train_dataset.GSOSceneGeometryN5K1.sampling_policy,
        )

    def test_metric_query_depth_profile_is_isolated_and_conditions_all_views(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_pi3_finetune_ray_metric_depth_query_safe_fill_rtxpro6000_70gb_336x252",
                    "data=megapose_gso_geometry_n5_k1_masked_metric_safe_fill_depth",
                ],
            )
        self.assertTrue(cfg.model.use_metric_depth_conditioning)
        self.assertEqual(cfg.model.metric_depth_reference_probability, 0.9)
        self.assertEqual(cfg.model.metric_depth_query_probability, 0.9)
        self.assertEqual(cfg.model.metric_depth_eval_reference_probability, 1.0)
        self.assertEqual(cfg.model.metric_depth_eval_query_probability, 1.0)
        self.assertEqual(cfg.model.metric_depth_dropout_granularity, "sample")

        # The parent profile remains the reference-depth-only baseline.
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            reference_only = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_geometry_pi3_finetune_ray_metric_depth_safe_fill_rtxpro6000_70gb_336x252",
                    "data=megapose_gso_geometry_n5_k1_masked_metric_safe_fill_depth",
                ],
            )
        self.assertEqual(
            reference_only.model.metric_depth_query_probability, 0.0
        )
        self.assertEqual(
            reference_only.model.metric_depth_eval_query_probability, 0.0
        )

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
        self.assertFalse(cfg.metrics["items"].object_pose.report_per_object)

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
                self.assertFalse(
                    cfg.metrics["items"].object_pose.report_per_object
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
        self.assertFalse(pbr.filter_target_preprocessed_depth)
        self.assertFalse(
            pbr_conditioned.filter_preprocessed_query_depth
        )
        self.assertFalse(pbr_conditioned.filter_target_preprocessed_depth)
        self.assertFalse(
            cfg.val_datasets.lmo_bop19_render_n5_k1.dataset
            .filter_target_preprocessed_depth
        )
        self.assertFalse(real_conditioned.filter_target_preprocessed_depth)
        self.assertFalse(pbr.visibility_mask_conditioning)
        for dataset in (pbr_conditioned, real_conditioned):
            self.assertTrue(dataset.visibility_mask_conditioning)
            self.assertFalse(dataset.condition_reference_visibility)
            self.assertTrue(dataset.condition_query_visibility)
            self.assertEqual(dataset.mode, "val")

    def test_query_full_depth_transfer_profile_is_an_a40_isolated_ablation(self):
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=train_megapose_gso_transfer_a40_40gb_336x252",
                    "data=megapose_gso_transfer_diagnostics_query_full_depth",
                ],
            )

        self.assertEqual(cfg.train.max_img_per_gpu, 72)
        self.assertEqual(cfg.test.max_img_per_gpu, 256)
        self.assertEqual(cfg.test.max_img_per_gpu_ref16, 256)
        self.assertEqual(cfg.train.num_epoch, 80)
        self.assertEqual(cfg.train.val_every_n_epochs, 3)
        self.assertEqual(cfg.train.optimizer.lr, 1e-5)
        self.assertEqual(cfg.train.optimizer.visibility_mask_lr, 5e-5)
        self.assertTrue(cfg.model.use_visibility_mask_conditioning)

        for name in ("GSORenderToScene", "GSOScenePairMaskedReferences"):
            dataset = cfg.train_dataset[name]
            self.assertEqual(dataset.reference_treatment.depth, "object_only")
            self.assertEqual(dataset.query_treatment.depth, "full")
            self.assertEqual(
                dataset.query_treatment.mask_condition,
                "object_if_repeated_else_probability",
            )
            self.assertEqual(
                dataset.query_treatment.mask_condition_probability, 0.5
            )
            self.assertTrue(dataset.photometric_augmentation)

        # Training depth is the only experimental variable. Held-out losses
        # and pose metrics retain the object-only baseline target.
        for name in (
            "gso_heldout_render_n5_k1",
            "gso_heldout_render_n16_k1",
            "gso_heldout_render_n5_k1_all_query_conditioned",
            "gso_heldout_render_n5_k1_visibility_gt_0_5",
        ):
            dataset = cfg.val_datasets[name].dataset
            self.assertEqual(dataset.reference_treatment.depth, "object_only")
            self.assertEqual(dataset.query_treatment.depth, "object_only")
        for name in (
            "lmo_pbr_new_val_render_n5_k1",
            "lmo_pbr_new_val_render_n5_k1_all_query_conditioned",
            "lmo_bop19_render_n5_k1",
            "lmo_bop19_render_n5_k1_all_query_conditioned",
        ):
            self.assertTrue(cfg.val_datasets[name].dataset.depth_masking)


if __name__ == "__main__":
    unittest.main()
