import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from datasets import create_dataloader
from datasets.base.base_dataset import BaseDataset
from datasets.base.batched_sampler import (
    DynamicBatchSampler,
    DynamicDistributedSampler,
    HomogeneousDynamicBatchSampler,
)
from datasets.base.observation import (
    ObservationCapability,
    batch_supports_capability,
    infer_batch_capabilities,
)
from datasets.base.utils import unified_collate_fn
from datasets.object_centric import (
    KeyQuerySamplingPolicy,
    ObjectDatasetAdapter,
    ObjectViewProcessor,
    ObjectViewTransform,
    RawObjectView,
)
from pi3.metrics.base import BaseMetric
from pi3.metrics.manager import MetricManager
from pi3.models.input_adapter import Pi3BatchAdapter
from pi3.visualizations.base import BaseVisualizer
from pi3.visualizations.manager import VisualManager
from trainers.pi3_trainer import Pi3Trainer


class _LengthOnlyDataset:
    def __init__(self, length):
        self.length = int(length)

    def __len__(self):
        return self.length


class _MissingDependencyTransform(ObjectViewTransform):
    name = "missing_dependency"
    requires = frozenset({"not_provided"})
    provides = frozenset({"conditioned_view"})

    def apply(self, state, *, adapter, request, rng):
        raise AssertionError("dependency validation should fail before apply")


class _ObjectOnlyMetric(BaseMetric):
    name = "object_only"
    required_capabilities = frozenset({"key_query", "object_pose"})

    def reset(self):
        self.updates = 0

    def update(self, prediction, batch, loss_output=None, *, mode="train"):
        self.updates += 1

    def compute(self):
        return {"updates": float(self.updates)}


class _ObjectOnlyVisualizer(BaseVisualizer):
    name = "object_only"
    required_capabilities = frozenset({"key_query", "object_pose"})

    def render(
        self,
        prediction,
        batch,
        loss_output=None,
        *,
        mode="val",
        batch_idx=0,
        rng=None,
    ):
        return {"image": Image.new("RGB", (8, 8), color=(255, 0, 0))}


class _CapturingPi3(torch.nn.Module):
    def forward(self, imgs, **kwargs):
        self.imgs = imgs
        self.kwargs = kwargs
        batch_size, num_views, _, height, width = imgs.shape
        return {
            "local_points": torch.zeros(
                batch_size, num_views, height, width, 3, device=imgs.device
            )
        }


class _MinimalObjectAdapter(ObjectDatasetAdapter):
    """Example adapter requiring only raw loading and sample planning."""

    def __init__(self, resolution=None, **kwargs):
        super().__init__(
            resolution=[[28, 28]] if resolution is None else resolution,
            frame_num=2,
            shuffle=False,
            **kwargs,
        )
        self.dataset_label = "MinimalObject"

    def __len__(self):
        return 1

    def load_raw_object_view(self, record):
        T_C_O = np.eye(4, dtype=np.float32)
        T_C_O[2, 3] = 1.0
        return RawObjectView(
            rgb=np.full((28, 28, 3), record["rgb"], dtype=np.uint8),
            depthmap=np.ones((28, 28), dtype=np.float32),
            object_mask=np.ones((28, 28), dtype=bool),
            camera_intrinsics=np.array(
                [[20.0, 0.0, 14.0], [0.0, 20.0, 14.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            T_C_O=T_C_O,
            camera_pose=np.linalg.inv(T_C_O).astype(np.float32),
            record=record,
        )

    def build_sample_plan(self, index, resolution, rng):
        return self.key_query_sampling_policy.plan(
            reference_records=[
                {
                    "object_id": 1,
                    "rgb": 64,
                    "source": "reference/0",
                    "instance": "reference_0",
                }
            ],
            query_records=[
                {
                    "object_id": 1,
                    "rgb": 192,
                    "source": "query/0",
                    "instance": "query_0",
                }
            ],
            reference_rgb_masking=False,
            query_rgb_masking=False,
        )


class _MinimalSceneDataset(BaseDataset):
    def __init__(self, resolution=None, **kwargs):
        super().__init__(
            resolution=[[28, 28]] if resolution is None else resolution,
            frame_num=2,
            shuffle=False,
            **kwargs,
        )
        self.dataset_label = "MinimalScene"

    def __len__(self):
        return 1

    def _get_views(self, index, resolution, rng):
        K = np.array(
            [[20.0, 0.0, 14.0], [0.0, 20.0, 14.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return [
            {
                "img": Image.fromarray(
                    np.full((28, 28, 3), 32 + view_index, dtype=np.uint8)
                ),
                "depthmap": np.ones((28, 28), dtype=np.float32),
                "camera_intrinsics": K.copy(),
                "camera_pose": np.eye(4, dtype=np.float32),
                "dataset": self.dataset_label,
                "label": "scene_geometry",
                "instance": f"scene_{view_index}",
            }
            for view_index in range(self.frame_num)
        ]

def _role_view(*, object_fields=False):
    view = {
        "img": torch.zeros(1, 3, 28, 28),
        "depthmap": torch.ones(1, 28, 28),
        "camera_intrinsics": torch.eye(3).unsqueeze(0),
        "camera_pose": torch.eye(4).unsqueeze(0),
        "pts3d": torch.zeros(1, 28, 28, 3),
        "valid_mask": torch.ones(1, 28, 28, dtype=torch.bool),
        "view_role": ["reference"],
        "is_reference": torch.ones(1, dtype=torch.bool),
        "is_query": torch.zeros(1, dtype=torch.bool),
    }
    if object_fields:
        view.update(
            {
                "object_id": torch.ones(1, dtype=torch.int64),
                "T_C_O": torch.eye(4).unsqueeze(0),
                "object_visibility_mask": torch.ones(1, 28, 28),
            }
        )
    return view


class ObservationContractTest(unittest.TestCase):
    def test_capabilities_are_inferred_without_widening_core(self):
        core = [_role_view(object_fields=False)]
        self.assertTrue(
            batch_supports_capability(core, ObservationCapability.CORE_GEOMETRY)
        )
        self.assertTrue(
            batch_supports_capability(core, ObservationCapability.KEY_QUERY)
        )
        self.assertFalse(
            batch_supports_capability(core, ObservationCapability.OBJECT_POSE)
        )

        object_batch = [_role_view(object_fields=True)]
        self.assertIn(
            ObservationCapability.OBJECT_POSE,
            infer_batch_capabilities(object_batch),
        )

    def test_object_model_capability_requires_an_available_model(self):
        view = _role_view(object_fields=True)
        view["object_model_available"] = torch.zeros(1, dtype=torch.bool)
        self.assertFalse(
            batch_supports_capability([view], ObservationCapability.OBJECT_MODEL)
        )
        view["object_model_available"] = torch.ones(1, dtype=torch.bool)
        self.assertTrue(
            batch_supports_capability([view], ObservationCapability.OBJECT_MODEL)
        )

    def test_collator_allows_role_optional_fields_within_a_schema(self):
        sample_a = [
            {"value": torch.tensor(1), "reference_only": torch.tensor(2)},
            {"value": torch.tensor(3)},
        ]
        sample_b = [
            {"value": torch.tensor(4), "reference_only": torch.tensor(5)},
            {"value": torch.tensor(6)},
        ]
        batch = unified_collate_fn([sample_a, sample_b])
        self.assertEqual(batch[0]["value"].tolist(), [1, 4])
        self.assertEqual(batch[1]["value"].tolist(), [3, 6])
        self.assertEqual(batch[1]["reference_only"], [None, None])

    def test_collator_rejects_different_sample_schemas(self):
        sample_a = [{"value": torch.tensor(1)}]
        sample_b = [{"value": torch.tensor(2), "object_id": torch.tensor(1)}]
        with self.assertRaisesRegex(ValueError, "homogeneous batches"):
            unified_collate_fn([sample_a, sample_b])


class ObjectPolicyAndPluginTest(unittest.TestCase):
    def test_record_selection_accepts_converted_numpy_object_arrays(self):
        records = np.asarray([{"id": 1}, {"id": 2}], dtype=object)
        selected = KeyQuerySamplingPolicy.select_records(
            records,
            1,
            "first",
            rng=np.random.default_rng(0),
            allow_repeat=False,
        )
        self.assertEqual(selected[0]["id"], 1)

    def test_count_policy_accounts_for_query_model_view_cost(self):
        policy = KeyQuerySamplingPolicy()
        counts = policy.sample_counts(
            7,
            reference_range=(3, 5),
            query_range=(1, 2),
            reference_available=10,
            query_available=10,
            query_view_cost=2,
            allow_repeat=False,
            rng=np.random.default_rng(4),
        )
        self.assertIn(counts, {(3, 2), (5, 1)})
        self.assertEqual(counts[0] + 2 * counts[1], 7)

    def test_sample_plan_is_declarative_and_costed(self):
        policy = KeyQuerySamplingPolicy()
        plan = policy.plan(
            reference_records=[{"id": 1}, {"id": 2}],
            query_records=[{"id": 3}],
            reference_rgb_masking=True,
            query_rgb_masking=False,
            query_view_cost=2,
            metadata={"object_id": 8},
        )
        self.assertEqual(plan.reference_count, 2)
        self.assertEqual(plan.query_record_count, 1)
        self.assertEqual(plan.model_view_count, 4)
        self.assertEqual(plan.metadata["object_id"], 8)

    def test_plugin_dependencies_are_checked_at_construction(self):
        with self.assertRaisesRegex(ValueError, "requires unavailable"):
            ObjectViewProcessor([_MissingDependencyTransform()])

    def test_minimal_adapter_inherits_full_processing_and_base_enrichment(self):
        sample = _MinimalObjectAdapter()[0]
        self.assertEqual(len(sample), 2)
        self.assertTrue(sample[0]["is_reference"])
        self.assertTrue(sample[1]["is_query"])
        self.assertIn("pts3d", sample[0])
        self.assertIn("valid_mask", sample[0])
        self.assertTrue(sample[0]["capability_object_pose"])
        np.testing.assert_allclose(
            sample[0]["camera_pose"],
            np.linalg.inv(sample[0]["T_C_O"]),
            atol=1e-6,
        )


class Pi3InputAdapterTest(unittest.TestCase):
    def test_general_dataset_gets_unknown_visibility_condition(self):
        batch = []
        for _ in range(2):
            batch.append(
                {
                    "img": torch.rand(3, 3, 28, 28),
                    "camera_intrinsics": torch.eye(3).repeat(3, 1, 1),
                }
            )
        adapted = Pi3BatchAdapter(
            use_visibility_mask_conditioning=True
        ).adapt(batch)
        condition = adapted.kwargs["visibility_mask_condition"]
        known = adapted.kwargs["visibility_mask_known"]
        self.assertEqual(tuple(condition.shape), (3, 2, 28, 28))
        self.assertEqual(float(condition.sum()), 0.0)
        self.assertEqual(float(known.sum()), 0.0)
        self.assertFalse(bool(adapted.role_masks["reference"].any()))

    def test_supplied_visibility_condition_is_preserved(self):
        batch = [
            {
                "img": torch.zeros(1, 3, 28, 28),
                "camera_intrinsics": torch.eye(3).unsqueeze(0),
                "visibility_mask_condition": torch.ones(1, 28, 28),
                "visibility_mask_known": torch.ones(1, 28, 28),
                "is_reference": torch.ones(1, dtype=torch.bool),
            }
        ]
        adapted = Pi3BatchAdapter(
            use_visibility_mask_conditioning=True
        ).adapt(batch)
        self.assertEqual(
            float(adapted.kwargs["visibility_mask_condition"].mean()), 1.0
        )
        self.assertTrue(bool(adapted.role_masks["reference"].all()))

    def test_pi3_trainer_routes_core_only_batch_through_adapter(self):
        trainer = object.__new__(Pi3Trainer)
        trainer.cfg = OmegaConf.create(
            {"model": {"use_visibility_mask_conditioning": True}}
        )
        trainer.model = _CapturingPi3()
        trainer.model_input_adapter = Pi3BatchAdapter(
            use_visibility_mask_conditioning=True
        )
        batch = [
            {
                "img": torch.zeros(1, 3, 28, 28),
                "camera_intrinsics": torch.eye(3).unsqueeze(0),
            },
            {
                "img": torch.zeros(1, 3, 28, 28),
                "camera_intrinsics": torch.eye(3).unsqueeze(0),
            },
        ]
        output = trainer.forward_batch(batch, mode="train")
        prediction = output[0]
        self.assertEqual(
            float(trainer.model.kwargs["visibility_mask_known"].sum()), 0.0
        )
        self.assertFalse(
            prediction["observation_capabilities"]["object_pose"]
        )

        trainer.train_loss = lambda pred, gt: (
            pred["local_points"].sum(),
            {},
        )
        result = trainer.calculate_loss(output, batch, mode="train")
        self.assertEqual(
            float(result.batch_capability_object_pose_loss_stat), 0.0
        )


class CapabilityRoutingTest(unittest.TestCase):
    def test_metric_manager_skips_and_counts_incompatible_batches(self):
        metric = _ObjectOnlyMetric()
        manager = MetricManager([metric], enabled=True)
        manager.reset()
        manager.update({}, [_role_view(object_fields=False)])
        skipped = manager.compute()
        self.assertEqual(metric.updates, 0)
        self.assertEqual(
            skipped["routing/object_only_skipped_batches"], 1.0
        )

        manager.update({}, [_role_view(object_fields=True)])
        eligible = manager.compute()
        self.assertEqual(metric.updates, 1)
        self.assertEqual(
            eligible["routing/object_only_eligible_batches"], 1.0
        )

    def test_visual_manager_skips_incompatible_batches(self):
        manager = VisualManager(
            [_ObjectOnlyVisualizer()], enabled=True, val_enabled=True
        )
        self.assertEqual(
            manager.render({}, [_role_view(object_fields=False)]), {}
        )
        self.assertEqual(
            manager.routing_counts["object_only"]["skipped"], 1
        )
        images = manager.render({}, [_role_view(object_fields=True)])
        self.assertIn("object_only/image", images)
        self.assertIn("selection/target", images)


class HomogeneousMixtureSamplerTest(unittest.TestCase):
    def test_single_dataset_dynamic_sampler_respects_supported_frame_counts(self):
        distributed = DynamicDistributedSampler(
            _LengthOnlyDataset(24),
            num_replicas=1,
            rank=0,
            shuffle=False,
        )
        sampler = DynamicBatchSampler(
            distributed,
            resolution_num=1,
            image_num_range=(2, 6),
            max_img_per_gpu=12,
            frame_num_list=[3, 5],
        )
        frame_counts = {item[2] for batch in sampler for item in batch}
        self.assertTrue(frame_counts)
        self.assertTrue(frame_counts.issubset({3, 5}))

    def test_each_batch_stays_inside_one_component_and_syncs_ranks(self):
        dataset = _LengthOnlyDataset(40)
        kwargs = dict(
            dataset=dataset,
            component_sizes=[16, 24],
            resolution_num=3,
            image_num_range=(2, 4),
            seed=17,
            world_size=2,
            max_img_per_gpu=8,
        )
        rank0 = HomogeneousDynamicBatchSampler(rank=0, **kwargs)
        rank1 = HomogeneousDynamicBatchSampler(rank=1, **kwargs)
        rank0.set_epoch(3, base_seed=101)
        rank1.set_epoch(3, base_seed=101)

        batches0 = list(rank0)
        batches1 = list(rank1)
        self.assertEqual(len(batches0), len(batches1))
        self.assertEqual(len(rank0), len(batches0))
        for batch0, batch1 in zip(batches0, batches1):
            components0 = {0 if item[0] < 16 else 1 for item in batch0}
            components1 = {0 if item[0] < 16 else 1 for item in batch1}
            self.assertEqual(len(components0), 1)
            self.assertEqual(components0, components1)
            self.assertEqual({item[2] for item in batch0}, {item[2] for item in batch1})
            self.assertTrue(set(item[0] for item in batch0).isdisjoint(item[0] for item in batch1))

    def test_epoch_replay_is_deterministic(self):
        sampler = HomogeneousDynamicBatchSampler(
            _LengthOnlyDataset(20),
            component_sizes=[8, 12],
            resolution_num=2,
            image_num_range=(2, 3),
            seed=5,
            rank=0,
            world_size=1,
            max_img_per_gpu=6,
        )
        sampler.set_epoch(9, base_seed=33)
        first = list(sampler)
        sampler.set_epoch(9, base_seed=33)
        second = list(sampler)
        self.assertEqual(first, second)

    def test_component_specific_frame_counts_are_respected(self):
        sampler = HomogeneousDynamicBatchSampler(
            _LengthOnlyDataset(40),
            component_sizes=[20, 20],
            component_frame_num_lists=[[2], [4]],
            resolution_num=1,
            image_num_range=(2, 4),
            seed=8,
            rank=0,
            world_size=1,
            max_img_per_gpu=8,
        )
        for batch in sampler:
            component = 0 if batch[0][0] < 20 else 1
            expected = 2 if component == 0 else 4
            self.assertEqual({item[2] for item in batch}, {expected})

    def test_create_dataloader_uses_homogeneous_batches_end_to_end(self):
        cfg = OmegaConf.create(
            {
                "train": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "image_num_range": [2, 2],
                    "max_img_per_gpu": 4,
                    "iters_per_epoch": 4,
                    "base_seed": 23,
                    "resolution": [[28, 28]],
                },
                "train_dataset": {
                    "length": 40,
                    "weights": {"Object": 1, "Scene": 1},
                    "Object": {
                        "_target_": "tests.test_dataset_architecture._MinimalObjectAdapter"
                    },
                    "Scene": {
                        "_target_": "tests.test_dataset_architecture._MinimalSceneDataset"
                    },
                },
                "train_dataloader": {"shuffle": True, "drop_last": True},
            }
        )
        loader = create_dataloader(cfg, "train")
        loader.dataset.set_epoch(0, base_seed=23)
        loader.batch_sampler.set_epoch(0, base_seed=23)

        seen = set()
        for batch_index, batch in enumerate(loader):
            names = set(batch[0]["dataset"])
            self.assertEqual(len(names), 1)
            name = next(iter(names))
            seen.add(name)
            if name == "MinimalObject":
                self.assertTrue(
                    batch_supports_capability(
                        batch, ObservationCapability.OBJECT_POSE
                    )
                )
            else:
                self.assertFalse(
                    batch_supports_capability(
                        batch, ObservationCapability.OBJECT_POSE
                    )
                )
            if batch_index >= 9:
                break
        self.assertEqual(seen, {"MinimalObject", "MinimalScene"})


if __name__ == "__main__":
    unittest.main()
