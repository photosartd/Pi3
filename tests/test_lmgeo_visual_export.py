import unittest
import csv
import tempfile
from pathlib import Path

from PIL import Image

from scripts.export_lmgeo_geometry_visuals import (
    compose_gallery_index,
    compose_overview,
    evaluate_prediction_for_export,
    ranked_good_bad_selections,
    stratified_sample_indices,
)


class LMGeoVisualExportTest(unittest.TestCase):
    def test_export_measures_raw_prediction_before_mutating_loss(self):
        calls = []
        prediction = {"coordinate_state": "raw"}

        class VisualManager:
            def render(self, value, batch, mode):
                calls.append(("visual", value["coordinate_state"], mode))
                return {"card": Image.new("RGB", (1, 1), "black")}

        class MetricManager:
            def compute_on_batch(self, value, batch, mode):
                calls.append(("metric", value["coordinate_state"], mode))
                return {"metric": 1.0}

        class Trainer:
            visual_manager = VisualManager()
            metric_manager = MetricManager()

            @staticmethod
            def calculate_loss(value, batch, mode):
                calls.append(("loss", value["coordinate_state"], mode))
                value["coordinate_state"] = "normalized"
                return object()

        rendered, metrics, _ = evaluate_prediction_for_export(
            Trainer(), prediction, []
        )
        self.assertEqual(list(rendered), ["card"])
        self.assertEqual(metrics, {"metric": 1.0})
        self.assertEqual(
            calls,
            [
                ("visual", "raw", "val"),
                ("metric", "raw", "val"),
                ("loss", "raw", "test"),
            ],
        )
        self.assertEqual(prediction["coordinate_state"], "normalized")

    def test_stratified_indices_are_unique_and_cover_objects(self):
        samples = [
            {"object_id": object_id, "item": item}
            for object_id in (1, 5, 6, 8)
            for item in range(12)
        ]
        indices = stratified_sample_indices(samples, 20)
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(set(indices)), 20)
        self.assertEqual({samples[index]["object_id"] for index in indices}, {1, 5, 6, 8})
        counts = {
            object_id: sum(samples[index]["object_id"] == object_id for index in indices)
            for object_id in (1, 5, 6, 8)
        }
        self.assertEqual(set(counts.values()), {5})

    def test_overview_reuses_all_expected_visual_cards(self):
        images = {
            "input_reference_frames/grid": Image.new("RGB", (660, 160), "red"),
            "input_query_frames/grid": Image.new("RGB", (160, 180), "green"),
            "lmo_query_pose_overlay/queries": Image.new("RGB", (336, 270), "blue"),
            "reference_reconstruction/orthographic": Image.new("RGB", (980, 340), "orange"),
            "depth_panel/query": Image.new("RGB", (1700, 300), "purple"),
            "depth_panel/reference": Image.new("RGB", (1700, 300), "cyan"),
        }
        overview = compose_overview(
            images,
            header_lines=["example", "metrics", "constraints"],
            maximum_width=1200,
        )
        self.assertEqual(overview.width, 1200)
        self.assertGreater(overview.height, 900)

    def test_gallery_index_arranges_all_overviews(self):
        images = [Image.new("RGB", (1200, 600 + index * 10), "blue") for index in range(5)]
        gallery = compose_gallery_index(images, columns=4, thumbnail_width=300)
        self.assertEqual(gallery.width, 1230)
        self.assertGreater(gallery.height, 300)

    def test_ranked_gallery_selects_one_good_and_bad_per_object(self):
        samples = []
        rows = []
        for object_id in (1, 5, 6, 8):
            for item in range(3):
                samples.append(
                    {
                        "object_id": object_id,
                        "geometry_query_record": {
                            "object_id": object_id,
                            "scene_id": 40 + object_id,
                            "im_id": item,
                            "gt_id": object_id + item,
                        },
                    }
                )
                rows.append(
                    {
                        "obj_id": object_id,
                        "scene_id": 40 + object_id,
                        "im_id": item,
                        "gt_id": object_id + item,
                        "query_used_d": float(item),
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            selections = ranked_good_bad_selections(samples, path, 8)

        self.assertEqual(len(selections), 8)
        self.assertEqual(
            [selection["selection_role"] for selection in selections],
            ["good"] * 4 + ["bad"] * 4,
        )
        self.assertEqual(
            [selection["rank_value"] for selection in selections],
            [0.0] * 4 + [2.0] * 4,
        )


if __name__ == "__main__":
    unittest.main()
