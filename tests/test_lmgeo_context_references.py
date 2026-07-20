import unittest

import numpy as np

from datasets.lmgeo_dataset import LMGeoSequenceDataset


def make_record(source, scene_id, im_id, object_id=1):
    return {
        "split": source,
        "scene_dir": f"/tmp/{source}/{scene_id:06d}",
        "im_id": int(im_id),
        "gt_id": 0,
        "object_id": int(object_id),
        "query_scene_id": int(scene_id),
        "query_subscene_id": int(im_id // 25),
    }


class LMGeoContextReferenceTest(unittest.TestCase):
    def make_dataset_shell(self, fraction):
        dataset = object.__new__(LMGeoSequenceDataset)
        dataset.context_reference_fraction = float(fraction)
        dataset.context_reference_exclude = "scene"
        dataset.reference_selection = "random"
        dataset.allow_repeat = False
        dataset.reference_records_by_object = {
            1: [make_record("train", 1, im_id) for im_id in range(10)]
        }
        dataset.context_reference_records_by_object_scene = {
            1: {
                10: [make_record("train_pbr", 10, im_id) for im_id in range(4)],
                11: [make_record("train_pbr", 11, im_id) for im_id in range(4, 8)],
            }
        }
        return dataset

    def test_zero_fraction_uses_only_render_references(self):
        dataset = self.make_dataset_shell(0.0)
        records = dataset._select_reference_records_for_sample(
            object_id=1,
            query_scene_id=10,
            query_subscene_id=0,
            count=4,
            rng=np.random.default_rng(1),
        )

        self.assertEqual(len(records), 4)
        self.assertTrue(all(record["reference_source"] == "render" for record in records))
        self.assertTrue(all(record["split"] == "train" for record in records))

    def test_full_fraction_uses_context_references_from_other_scenes(self):
        dataset = self.make_dataset_shell(1.0)
        records = dataset._select_reference_records_for_sample(
            object_id=1,
            query_scene_id=10,
            query_subscene_id=0,
            count=4,
            rng=np.random.default_rng(2),
        )

        self.assertEqual(len(records), 4)
        self.assertTrue(all(record["reference_source"] == "context_scene" for record in records))
        self.assertTrue(all(record["split"] == "train_pbr" for record in records))
        self.assertTrue(all(record["query_scene_id"] == 11 for record in records))

    def test_context_shortage_falls_back_to_render_references(self):
        dataset = self.make_dataset_shell(1.0)
        dataset.context_reference_records_by_object_scene = {
            1: {11: [make_record("train_pbr", 11, 0)]}
        }
        records = dataset._select_reference_records_for_sample(
            object_id=1,
            query_scene_id=10,
            query_subscene_id=0,
            count=4,
            rng=np.random.default_rng(3),
        )

        self.assertEqual(len(records), 4)
        self.assertEqual(
            sum(record["reference_source"] == "context_scene" for record in records),
            1,
        )
        self.assertEqual(
            sum(record["reference_source"] == "render" for record in records),
            3,
        )

    def test_subscene_exclusion_allows_other_subscenes_in_same_scene(self):
        dataset = self.make_dataset_shell(1.0)
        dataset.context_reference_exclude = "subscene"
        dataset.reference_selection = "uniform"
        dataset.context_reference_records_by_object_scene = {
            1: {
                10: [
                    make_record("new_val", 10, 0),
                    make_record("new_val", 10, 1),
                    make_record("new_val", 10, 25),
                    make_record("new_val", 10, 26),
                ],
                11: [make_record("new_val", 11, 0)],
            }
        }

        records = dataset._select_reference_records_for_sample(
            object_id=1,
            query_scene_id=10,
            query_subscene_id=0,
            count=3,
            rng=np.random.default_rng(4),
        )

        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["reference_source"] == "context_scene" for record in records))
        self.assertFalse(
            any(
                record["query_scene_id"] == 10 and record["query_subscene_id"] == 0
                for record in records
            )
        )


if __name__ == "__main__":
    unittest.main()
