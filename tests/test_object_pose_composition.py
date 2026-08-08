import unittest

import numpy as np

from datasets.object_centric import RawObjectView, ViewTreatment
from datasets.object_pose_dataset import ComposableObjectPoseDataset
from datasets.object_sampling import (
    HybridReferencePolicy,
    IndependentScenesPolicy,
    RenderToScenePolicy,
    ScenePairPolicy,
)
from datasets.object_sources import ObjectViewGroup, ObjectViewSource


class _MemoryObjectSource(ObjectViewSource):
    def __init__(self, source_name, groups, *, repeated=False):
        self.source_name = source_name
        self.object_namespace = "synthetic"
        self._groups = tuple(groups)
        self._repeated = bool(repeated)

    @property
    def object_ids(self):
        return tuple(sorted({group.object_id for group in self._groups}))

    @property
    def contains_repeated_object_instances(self):
        return self._repeated

    @property
    def repeated_object_scene_count(self):
        return len({group.scene_id for group in self._groups}) if self._repeated else 0

    def groups_for_object(self, object_id):
        return tuple(group for group in self._groups if group.object_id == object_id)

    def records_for_group(self, group):
        records = []
        for view_id in range(group.view_count):
            records.append(
                {
                    "source_name": self.source_name,
                    "object_id": group.object_id,
                    "scene_id": -1 if group.scene_id is None else group.scene_id,
                    "gt_id": 0,
                    "view_id": view_id,
                    "record_id": f"{group.group_id}:{view_id}",
                    "source": f"{group.group_id}:{view_id}",
                    "instance": f"{group.group_id}:{view_id}",
                    "same_object_scene_track_count": 2 if self._repeated else 1,
                    "same_object_frame_instance_count": 2 if self._repeated else 1,
                    "object_model_available": False,
                }
            )
        return records

    def load_raw_object_view(self, record):
        height = width = 28
        mask = np.zeros((height, width), dtype=bool)
        mask[7:21, 7:21] = True
        rgb = np.full((height, width, 3), 100, dtype=np.uint8)
        depth = np.ones((height, width), dtype=np.float32)
        T_C_O = np.eye(4, dtype=np.float32)
        T_C_O[2, 3] = 0.5
        return RawObjectView(
            rgb=rgb,
            depthmap=depth,
            object_mask=mask,
            camera_intrinsics=np.asarray(
                [[20.0, 0.0, 14.0], [0.0, 20.0, 14.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            T_C_O=T_C_O,
            camera_pose=np.linalg.inv(T_C_O).astype(np.float32),
            record=record,
        )

    def validate_planned_view(self, record, *, view_role, treatment):
        if (
            self._repeated
            and treatment.rgb == "full"
            and treatment.mask_condition not in {"object", "object_if_repeated"}
        ):
            raise ValueError("ambiguous target")


def _render_source():
    return _MemoryObjectSource(
        "render",
        [
            ObjectViewGroup(
                source_name="render",
                object_id=0,
                group_id="render:0",
                view_count=12,
            )
        ],
    )


def _scene_source(*, repeated=False):
    return _MemoryObjectSource(
        "scene",
        [
            ObjectViewGroup(
                source_name="scene",
                object_id=0,
                group_id=f"scene:{scene_id}",
                scene_id=scene_id,
                track_id=0,
                view_count=6,
            )
            for scene_id in range(6)
        ],
        repeated=repeated,
    )


def _dataset(policy, *, sources, reference=None, query=None, source_treatments=None):
    return ComposableObjectPoseDataset(
        sources=sources,
        sampling_policy=policy,
        protocol_name="test_protocol",
        dataset_domain="SyntheticObject",
        reference_treatment=reference
        or ViewTreatment(rgb="full", depth="object_only", mask_condition="object"),
        query_treatment=query
        or ViewTreatment(rgb="full", depth="object_only", mask_condition="none"),
        source_treatments=source_treatments,
        resolution=[[28, 28]],
        frame_num=6,
        shuffle=False,
    )


class ObjectPoseCompositionTest(unittest.TestCase):
    def test_explicit_sample_seed_is_worker_order_independent(self):
        dataset = _dataset(
            RenderToScenePolicy(
                num_reference_range=(5, 5), num_query_range=(1, 1)
            ),
            sources={"render": _render_source(), "scene": _scene_source()},
            query=ViewTreatment(mask_condition="object"),
        )
        first = dataset[(0, 0, 6, 123456)]
        dataset[(0, 0, 6, 999)]
        replay = dataset[(0, 0, 6, 123456)]
        self.assertEqual(
            [view["instance"] for view in first],
            [view["instance"] for view in replay],
        )

    def test_render_to_scene_treatments_are_orthogonal(self):
        dataset = _dataset(
            RenderToScenePolicy(
                num_reference_range=(5, 5),
                num_query_range=(1, 1),
                reference_selection="uniform",
                query_selection="first",
            ),
            sources={"render": _render_source(), "scene": _scene_source()},
            reference=ViewTreatment(
                rgb="object_only", depth="full", mask_condition="object"
            ),
            query=ViewTreatment(
                rgb="full", depth="object_only", mask_condition="none"
            ),
        )
        views = dataset[(0, 0, 6)]
        self.assertEqual(len(views), 6)
        for view in views[:5]:
            mask = view["object_visibility_mask"] > 0.5
            image = np.asarray(view["img"])
            self.assertTrue(np.all(image[:, ~mask] == 0.0))
            self.assertTrue(np.all(view["depthmap"][~mask] == 1.0))
            np.testing.assert_array_equal(
                view["visibility_mask_condition"], view["object_visibility_mask"]
            )
            self.assertEqual(view["source_name"], "render")
        query = views[-1]
        mask = query["object_visibility_mask"] > 0.5
        self.assertTrue(np.all(query["depthmap"][~mask] == 0.0))
        self.assertTrue(np.all(query["visibility_mask_known"] == 0.0))
        self.assertEqual(query["protocol_name"], "test_protocol")
        self.assertEqual(query["source_name"], "scene")
        np.testing.assert_allclose(query["camera_pose"], np.linalg.inv(query["T_C_O"]))

    def test_render_to_scene_can_enumerate_every_query_track_once(self):
        policy = RenderToScenePolicy(
            num_reference_range=(5, 5),
            num_query_range=(1, 1),
            reference_selection="uniform",
            query_selection="first",
            enumerate_query_groups=True,
        )
        dataset = _dataset(
            policy,
            sources={"render": _render_source(), "scene": _scene_source()},
            query=ViewTreatment(mask_condition="object"),
        )
        self.assertEqual(len(dataset), 6)
        query_scenes = {
            int(dataset[(index, 0, 6, 100 + index)][-1]["scene_id"])
            for index in range(len(dataset))
        }
        self.assertEqual(query_scenes, set(range(6)))

    def test_scene_pair_uses_distinct_scenes_and_unique_frames(self):
        dataset = _dataset(
            ScenePairPolicy(
                num_reference_range=(3, 3),
                num_query_range=(3, 3),
                allow_repeat=False,
            ),
            sources={"scene": _scene_source()},
            query=ViewTreatment(mask_condition="object"),
        )
        views = dataset[(0, 0, 6)]
        reference_scenes = {int(view["scene_id"]) for view in views[:3]}
        query_scenes = {int(view["scene_id"]) for view in views[3:]}
        self.assertEqual(len(reference_scenes), 1)
        self.assertEqual(len(query_scenes), 1)
        self.assertNotEqual(reference_scenes, query_scenes)
        self.assertEqual(len({view["instance"] for view in views[:3]}), 3)
        self.assertEqual(len({view["instance"] for view in views[3:]}), 3)

    def test_hybrid_policy_composes_two_reference_sources(self):
        dataset = _dataset(
            HybridReferencePolicy(
                num_render_reference_range=(2, 2),
                num_scene_reference_range=(3, 3),
                num_query_range=(1, 1),
            ),
            sources={"render": _render_source(), "scene": _scene_source()},
            query=ViewTreatment(mask_condition="object"),
            source_treatments={
                "render": {
                    "reference": {
                        "rgb": "full",
                        "depth": "object_only",
                        "mask_condition": "none",
                    }
                }
            },
        )
        views = dataset[(0, 0, 6)]
        references = views[:5]
        self.assertEqual(sum(view["source_name"] == "render" for view in references), 2)
        self.assertEqual(sum(view["source_name"] == "scene" for view in references), 3)
        self.assertEqual(views[-1]["source_name"], "scene")

    def test_independent_policy_uses_one_view_per_scene(self):
        dataset = _dataset(
            IndependentScenesPolicy(
                num_reference_range=(3, 3), num_query_range=(3, 3)
            ),
            sources={"scene": _scene_source()},
            query=ViewTreatment(mask_condition="object"),
        )
        views = dataset[(0, 0, 6)]
        self.assertEqual(len({int(view["scene_id"]) for view in views}), 6)

    def test_ambiguous_scene_target_requires_input_disambiguation(self):
        with self.assertRaisesRegex(ValueError, "same-object repeated scenes"):
            _dataset(
                ScenePairPolicy(
                    num_reference_range=(3, 3), num_query_range=(3, 3)
                ),
                sources={"scene": _scene_source(repeated=True)},
                reference=ViewTreatment(
                    rgb="full", depth="object_only", mask_condition="none"
                ),
                query=ViewTreatment(
                    rgb="full", depth="object_only", mask_condition="none"
                ),
            )

    def test_query_mask_is_known_only_for_repeated_same_object_scenes(self):
        reference_treatment = ViewTreatment(
            rgb="object_only", depth="object_only", mask_condition="none"
        )
        conditional_query = ViewTreatment(
            rgb="full",
            depth="object_only",
            mask_condition="object_if_repeated",
        )
        repeated = _dataset(
            ScenePairPolicy(
                num_reference_range=(3, 3), num_query_range=(3, 3)
            ),
            sources={"scene": _scene_source(repeated=True)},
            reference=reference_treatment,
            query=conditional_query,
        )
        self.assertTrue(repeated.requires_visibility_mask_conditioning)
        repeated_views = repeated[(0, 0, 6)]
        for view in repeated_views[:3]:
            self.assertFalse(view["visibility_mask_condition_applied"])
            self.assertEqual(float(view["visibility_mask_known"].sum()), 0.0)
        for view in repeated_views[3:]:
            self.assertTrue(view["visibility_mask_condition_applied"])
            mask = view["object_visibility_mask"] > 0.5
            self.assertTrue(np.any(np.asarray(view["img"])[:, ~mask] != 0.0))
            np.testing.assert_array_equal(
                view["visibility_mask_condition"],
                view["object_visibility_mask"],
            )
            self.assertTrue(np.all(view["visibility_mask_known"] == 1.0))

        unique = _dataset(
            ScenePairPolicy(
                num_reference_range=(3, 3), num_query_range=(3, 3)
            ),
            sources={"scene": _scene_source(repeated=False)},
            reference=reference_treatment,
            query=conditional_query,
        )
        self.assertFalse(unique.requires_visibility_mask_conditioning)
        for view in unique[(0, 0, 6)]:
            self.assertFalse(view["visibility_mask_condition_applied"])
            self.assertEqual(float(view["visibility_mask_known"].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
