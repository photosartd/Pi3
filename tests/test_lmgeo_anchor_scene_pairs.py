import unittest

import numpy as np

from datasets.lmgeo_dataset import LMGeoAnchorScenePairSequenceDataset


def make_record(scene_id, subscene_id, im_id, object_id=1):
    return {
        "split": "train_pbr",
        "scene_dir": f"/tmp/train_pbr/{scene_id:06d}",
        "im_id": int(im_id),
        "gt_id": 0,
        "object_id": int(object_id),
        "query_scene_id": int(scene_id),
        "query_subscene_id": int(subscene_id),
    }


def make_sample(scene_id, subscene_id, im_ids, object_id=1):
    return {
        "object_id": int(object_id),
        "query_scene_id": int(scene_id),
        "query_subscene_id": int(subscene_id),
        "query_records": [
            make_record(scene_id, subscene_id, im_id, object_id=object_id)
            for im_id in im_ids
        ],
    }


class LMGeoAnchorScenePairTest(unittest.TestCase):
    def make_dataset_shell(self):
        dataset = object.__new__(LMGeoAnchorScenePairSequenceDataset)
        dataset.allow_repeat = False
        dataset.reference_selection = "first"
        dataset.query_selection = "first"
        dataset.num_reference_range = (2, 2)
        dataset.num_query_range = (1, 1)
        dataset.anchor_allow_same_scene = True
        dataset.anchor_allow_same_subscene = True
        dataset.anchor_selection_attempts = 10
        dataset.samples = [
            make_sample(1, 0, [0, 1, 2]),
            make_sample(2, 0, [25, 26, 27]),
            make_sample(1, 0, [100, 101, 102], object_id=2),
        ]
        dataset.anchor_reference_samples_by_object = dataset._build_anchor_reference_sample_index()
        return dataset

    def test_candidates_are_same_object_only(self):
        dataset = self.make_dataset_shell()
        candidates = dataset._anchor_reference_candidates(dataset.samples[0])

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate["object_id"] == 1 for candidate in candidates))

    def test_can_exclude_same_scene(self):
        dataset = self.make_dataset_shell()
        dataset.anchor_allow_same_scene = False
        candidates = dataset._anchor_reference_candidates(dataset.samples[0])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["query_scene_id"], 2)

    def test_selected_queries_do_not_leak_into_references(self):
        dataset = self.make_dataset_shell()
        reference_sample, reference_records, query_records, ref_count, query_count = (
            dataset._select_anchor_record_sets(
                dataset.samples[0],
                total_frames=3,
                rng=np.random.default_rng(0),
            )
        )

        self.assertEqual(ref_count, 2)
        self.assertEqual(query_count, 1)
        reference_ids = {dataset._record_identity(record) for record in reference_records}
        query_ids = {dataset._record_identity(record) for record in query_records}
        self.assertFalse(reference_ids & query_ids)
        self.assertEqual(reference_sample["object_id"], 1)


if __name__ == "__main__":
    unittest.main()
