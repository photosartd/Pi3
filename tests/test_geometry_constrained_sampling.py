import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from datasets.object_centric import RawObjectView, ViewTreatment
from datasets.object_geometry import object_preserving_crop_spec
from datasets.object_pose_dataset import ComposableObjectPoseDataset
from datasets.object_sampling import GeometryConstrainedScenePairPolicy
from datasets.object_sources import ObjectViewGroup, ObjectViewSource
from datasets.preprocess.megapose_gso_geometry import (
    FEATURE_COLUMNS,
    SCHEMA,
    build_reference_plan_catalog,
)


class _GeometryMemorySource(ObjectViewSource):
    source_name = "scene"
    object_namespace = "synthetic"

    def __init__(self):
        self.groups = (
            ObjectViewGroup("scene", 1, "scene:1:track:0", 5, 1, 0),
            ObjectViewGroup("scene", 1, "scene:2:track:0", 1, 2, 0),
            ObjectViewGroup("scene", 1, "scene:3:track:0", 1, 3, 0),
        )

    @property
    def object_ids(self):
        return (1,)

    def groups_for_object(self, object_id):
        return self.groups if int(object_id) == 1 else ()

    def records_for_group(self, group):
        if int(group.scene_id) == 1:
            frame_ids = range(5)
        elif int(group.scene_id) == 2:
            frame_ids = (10,)
        else:
            frame_ids = (11,)
        return [
            {
                "source_name": "scene",
                "object_id": 1,
                "scene_id": int(group.scene_id),
                "gt_id": 0,
                "view_id": index,
                "frame_id": int(frame_id),
                "source": f"scene:{group.scene_id}:{index}",
                "instance": f"scene:{group.scene_id}:{index}",
            }
            for index, frame_id in enumerate(frame_ids)
        ]

    def load_raw_object_view(self, record):
        height, width = 90, 120
        mask = np.zeros((height, width), dtype=bool)
        mask[25:65, 40:80] = True
        rgb = np.full((height, width, 3), 180, dtype=np.uint8)
        depth = np.ones((height, width), dtype=np.float32)
        T_C_O = np.eye(4, dtype=np.float32)
        T_C_O[2, 3] = 1.0
        return RawObjectView(
            rgb=rgb,
            depthmap=depth,
            object_mask=mask,
            camera_intrinsics=np.asarray(
                [[90, 0, 60], [0, 90, 45], [0, 0, 1]], dtype=np.float32
            ),
            T_C_O=T_C_O,
            camera_pose=np.linalg.inv(T_C_O).astype(np.float32),
            record=record,
        )


def _feature_row(feature_id, frame_id, scene_id, observed_bits):
    value = {name: 0 for name in FEATURE_COLUMNS}
    value.update(
        feature_id=feature_id,
        frame_id=frame_id,
        gt_id=0,
        object_id=1,
        scene_id=scene_id,
        view_id=feature_id,
        visib_fract=0.8,
        px_count_all=1600,
        px_count_valid=1600,
        px_count_visib=1280,
        depth_corrupt=0,
        width=120,
        height=90,
        view_x=0.0,
        view_y=0.0,
        view_z=1.0,
        fx=90.0,
        fy=90.0,
        cx=60.0,
        cy=45.0,
        bbox_obj_x=40.0,
        bbox_obj_y=25.0,
        bbox_obj_w=40.0,
        bbox_obj_h=40.0,
        bbox_visib_x=40.0,
        bbox_visib_y=25.0,
        bbox_visib_w=40.0,
        bbox_visib_h=40.0,
        crop_width_min=56.0,
        crop_width_max=120.0,
        crop_feasible=1,
        bbox_obj_touches_border=0,
        base_norm_fx=0.75,
        base_norm_fy=1.0,
        max_norm_fx=90.0 / 56.0,
        max_norm_fy=90.0 / 42.0,
        surface_point_count=8,
        observed_point_count=int(observed_bits).bit_count(),
        observed_fraction=int(observed_bits).bit_count() / 8.0,
        bit_offset=feature_id,
    )
    return tuple(value[name] for name in FEATURE_COLUMNS)


def _geometry_fixture(root):
    geometry = root / "geometry.sqlite"
    bits = root / "geometry.bits"
    masks = [0b00000011, 0b00001100, 0b00110000, 0b11000000, 0b00010001, 1, 1]
    bits.write_bytes(bytes(masks))
    connection = sqlite3.connect(geometry)
    connection.executescript(SCHEMA)
    connection.executemany(
        "INSERT INTO metadata VALUES (?,?)",
        (
            ("fingerprint", json.dumps({"settings": {"surface_point_count": 8}})),
            ("bits_path", str(bits)),
            ("index_complete", "1"),
        ),
    )
    rows = [
        _feature_row(index, index, 1, masks[index]) for index in range(5)
    ]
    rows.append(_feature_row(5, 10, 2, masks[5]))
    rows.append(_feature_row(6, 11, 3, masks[6]))
    connection.executemany(
        f"INSERT INTO frame_features ({','.join(FEATURE_COLUMNS)}) "
        f"VALUES ({','.join('?' for _ in FEATURE_COLUMNS)})",
        rows,
    )
    connection.commit()
    connection.close()
    plan = root / "plans.sqlite"
    result = build_reference_plan_catalog(
        geometry,
        plan,
        reference_count=5,
        reference_visibility_min=0.3,
        union_surface_min=0.5,
        progress=False,
    )
    return geometry, plan, result


class GeometryConstrainedSamplingTest(unittest.TestCase):
    def test_seeded_plan_variants_preserve_exact_coverage_constraint(self):
        with tempfile.TemporaryDirectory() as temporary:
            geometry, _, _ = _geometry_fixture(Path(temporary))
            plan = Path(temporary) / "variant_plans.sqlite"
            result = build_reference_plan_catalog(
                geometry,
                plan,
                reference_count=3,
                reference_visibility_min=0.3,
                union_surface_min=0.5,
                reference_source_kind="render",
                variants_per_track=5,
                progress=False,
            )
            self.assertGreater(result["plans"], 1)
            connection = sqlite3.connect(plan)
            try:
                rows = connection.execute(
                    "SELECT reference_count,union_coverage,feature_ids_i64 "
                    "FROM reference_plans"
                ).fetchall()
            finally:
                connection.close()
            for count, coverage, encoded in rows:
                self.assertEqual(int(count), 3)
                self.assertGreaterEqual(float(coverage), 0.5)
                self.assertEqual(len(np.frombuffer(encoded, dtype="<i8")), 3)

    def test_crop_spec_preserves_margin_and_jitters_inside_4_by_3_crop(self):
        feature = {
            "width": 120,
            "height": 90,
            "fx": 90.0,
            "fy": 90.0,
            "bbox_obj_x": 40.0,
            "bbox_obj_y": 25.0,
            "bbox_obj_w": 40.0,
            "bbox_obj_h": 40.0,
        }
        spec = object_preserving_crop_spec(
            feature,
            target_normalized_focal=0.9,
            center_jitter=0.1,
            rng=np.random.default_rng(17),
        )
        l, t, r, b = spec["bbox_xyxy"]
        self.assertAlmostEqual((r - l) / (b - t), 4 / 3)
        left, top, right, bottom = spec["expanded_object_bbox_xyxy"]
        self.assertLessEqual(l, left)
        self.assertLessEqual(t, top)
        self.assertGreaterEqual(r, right)
        self.assertGreaterEqual(b, bottom)

    def test_crop_spec_expands_for_fractional_integer_placement(self):
        feature = {
            "width": 120,
            "height": 90,
            "fx": 90.0,
            "fy": 90.0,
            "bbox_obj_x": 40.4,
            "bbox_obj_y": 25.4,
            "bbox_obj_w": 39.0,
            "bbox_obj_h": 39.0,
        }
        spec = object_preserving_crop_spec(
            feature, target_normalized_focal=1.75, center_jitter=0.2
        )
        l, t, r, b = spec["bbox_xyxy"]
        left, top, right, bottom = spec["expanded_object_bbox_xyxy"]
        self.assertEqual((r - l) * 3, (b - t) * 4)
        self.assertLessEqual(l, left)
        self.assertLessEqual(t, top)
        self.assertGreaterEqual(r, right)
        self.assertGreaterEqual(b, bottom)

    def test_plan_catalog_and_policy_materialize_valid_masked_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            geometry, plan, result = _geometry_fixture(Path(temporary))
            self.assertEqual(result["plans"], 1)
            policy = GeometryConstrainedScenePairPolicy(
                geometry_plan_path=plan,
                query_geometry_index_path=geometry,
                scene_source="scene",
                require_different_scene=True,
                num_reference_range=(5, 5),
                num_query_range=(1, 1),
                reference_selection="random",
                query_selection="first",
                allow_repeat=False,
                plan_selection="first",
                random_focal_target=False,
                crop_center_jitter=0.1,
            )
            treatment = ViewTreatment(
                rgb="object_only", depth="object_only", mask_condition="none"
            )
            dataset = ComposableObjectPoseDataset(
                sources={"scene": _GeometryMemorySource()},
                sampling_policy=policy,
                protocol_name="geometry_scene_pair",
                dataset_domain="SyntheticGeometry",
                reference_treatment=treatment,
                query_treatment=treatment,
                resolution=[[84, 63]],
                frame_num=6,
                shuffle=False,
            )
            views = dataset[(0, 0, 6, 1234)]
            self.assertEqual(len(views), 6)
            self.assertTrue(all(view["object_crop_applied"] for view in views))
            self.assertEqual({int(view["scene_id"]) for view in views[:5]}, {1})
            self.assertEqual(int(views[-1]["scene_id"]), 2)
            self.assertGreaterEqual(
                float(dataset.this_views_info["reference_union_coverage"]), 0.5
            )
            self.assertLessEqual(
                float(dataset.this_views_info["positive_angle_degrees"]), 10.0
            )
            self.assertEqual(
                len({str(view["instance"]) for view in views[:5]}), 5
            )
            for view in views:
                mask = np.asarray(view["object_visibility_mask"]) > 0.5
                image = np.asarray(view["img"])
                self.assertTrue(np.all(image[:, ~mask] == 0.0))
                self.assertTrue(np.all(view["depthmap"][~mask] == 0.0))
                np.testing.assert_allclose(
                    view["camera_pose"], np.linalg.inv(view["T_C_O"]), atol=1e-6
                )
                K = np.asarray(view["camera_intrinsics"])
                self.assertAlmostEqual(
                    (K[0, 0] / 84.0) / (K[1, 1] / 63.0), 0.75, places=5
                )

    def test_enumerate_query_groups_covers_every_cross_scene_group_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            geometry, plan, _ = _geometry_fixture(Path(temporary))
            source = _GeometryMemorySource()

            deterministic = GeometryConstrainedScenePairPolicy(
                geometry_plan_path=plan,
                query_geometry_index_path=geometry,
                scene_source="scene",
                require_different_scene=True,
                query_selection="first",
                plan_selection="first",
                random_focal_target=False,
            )
            self.assertEqual(deterministic.natural_length({"scene": source}), 1)

            enumerated = GeometryConstrainedScenePairPolicy(
                geometry_plan_path=plan,
                query_geometry_index_path=geometry,
                scene_source="scene",
                require_different_scene=True,
                query_selection="first",
                plan_selection="first",
                random_focal_target=False,
                enumerate_query_groups=True,
            )
            # Scene 1 is reference-only (it is the only 5-frame track); scenes 2
            # and 3 each contribute exactly one cross-scene query candidate, so
            # enumeration should find both instead of stopping at the first.
            self.assertEqual(enumerated.natural_length({"scene": source}), 2)
            targets = enumerated._enumerated_query_targets({"scene": source})
            found_scenes = sorted(
                int(query_feature["scene_id"]) for _, _, query_feature, _ in targets
            )
            self.assertEqual(found_scenes, [2, 3])

            capped = GeometryConstrainedScenePairPolicy(
                geometry_plan_path=plan,
                query_geometry_index_path=geometry,
                scene_source="scene",
                require_different_scene=True,
                query_selection="first",
                plan_selection="first",
                random_focal_target=False,
                enumerate_query_groups=True,
                max_query_groups_per_object=1,
            )
            self.assertEqual(capped.natural_length({"scene": source}), 1)

            treatment = ViewTreatment(
                rgb="object_only", depth="object_only", mask_condition="none"
            )
            dataset = ComposableObjectPoseDataset(
                sources={"scene": source},
                sampling_policy=enumerated,
                protocol_name="geometry_scene_pair_enumerated",
                dataset_domain="SyntheticGeometry",
                reference_treatment=treatment,
                query_treatment=treatment,
                resolution=[[84, 63]],
                frame_num=6,
                shuffle=False,
            )
            self.assertEqual(len(dataset), 2)
            seen_query_scenes = set()
            for index in range(len(dataset)):
                views = dataset[(index, 0, 6, 1234)]
                self.assertEqual(len(views), 6)
                self.assertEqual({int(view["scene_id"]) for view in views[:5]}, {1})
                seen_query_scenes.add(int(views[-1]["scene_id"]))
            self.assertEqual(seen_query_scenes, {2, 3})


if __name__ == "__main__":
    unittest.main()
