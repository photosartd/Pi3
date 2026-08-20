import unittest
from collections import defaultdict

from pi3.metrics.object_pose import ObjectPoseMetric


class ObjectPoseMetricOutputTest(unittest.TestCase):
    def test_per_object_add_and_adds_recall_are_reported(self):
        metric = ObjectPoseMetric.__new__(ObjectPoseMetric)
        metric.add_d = [0.05, 0.20, 0.30]
        metric.adds_d = [0.04, 0.03, 0.40]
        metric.used_d = [0.05, 0.03, 0.30]
        metric.rot_deg = [1.0, 2.0, 3.0]
        metric.trans_m = [0.01, 0.02, 0.03]
        metric.trans_lateral_m = [0.006, 0.012, 0.018]
        metric.trans_depth_m = [0.008, 0.016, 0.024]
        metric.scales = [1.0]
        metric.num_predictions = 3
        metric.num_underconstrained = 0
        metric.add_d_by_obj = defaultdict(list, {1: [0.05, 0.20], 10: [0.30]})
        metric.adds_d_by_obj = defaultdict(list, {1: [0.04, 0.03], 10: [0.40]})
        metric.used_d_by_obj = defaultdict(list, {1: [0.05, 0.20], 10: [0.40]})
        metric.rot_deg_by_obj = defaultdict(list, {1: [1.0, 2.0], 10: [3.0]})
        metric.trans_m_by_obj = defaultdict(list, {1: [0.01, 0.02], 10: [0.03]})

        stats = metric.compute()

        self.assertEqual(stats["obj_000001_query_add_0_1d"], 0.5)
        self.assertEqual(stats["obj_000001_query_adds_0_1d"], 1.0)
        self.assertEqual(stats["obj_000001_query_add_s_0_1d"], 0.5)
        self.assertEqual(stats["obj_000010_query_add_0_1d"], 0.0)
        self.assertEqual(stats["obj_000010_query_adds_0_1d"], 0.0)
        self.assertEqual(stats["query_add_0_1d_macro_obj"], 0.25)
        self.assertEqual(stats["query_adds_0_1d_macro_obj"], 0.5)
        self.assertEqual(stats["query_add_s_0_1d_macro_obj"], 0.25)
        self.assertEqual(stats["query_trans_lateral_median_m"], 0.012)
        self.assertEqual(stats["query_trans_depth_median_m"], 0.016)


if __name__ == "__main__":
    unittest.main()
