import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image

from datasets.object_sources import IndexedBOPReferenceSource
from datasets.preprocess.render.object_reference_index import build_reference_index


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_bank(root: Path, *, mapping_objects=1):
    mapping = [
        {"obj_id": object_id, "gso_id": f"object_{object_id}"}
        for object_id in range(mapping_objects)
    ]
    mapping_path = root / "mapping.json"
    _write_json(mapping_path, mapping)
    K = [20.0, 0.0, 4.0, 0.0, 20.0, 4.0, 0.0, 0.0, 1.0]
    _write_json(
        root / "reference_bank.json",
        {
            "format": "test_reference_bank",
            "config_fingerprint": "fingerprint",
            "mapping_path": str(mapping_path),
            "resolution_wh": [8, 8],
            "num_views": 2,
            "view_order": "sequential",
        },
    )
    object_dir = root / "000000"
    poses = []
    cameras = {}
    gt = {}
    info = {}
    T_C_O = np.eye(4, dtype=np.float64)
    T_C_O[2, 3] = 0.5
    for view_id in range(2):
        stem = f"{view_id:06d}"
        poses.append(
            {
                "view_id": view_id,
                "coverage_view_id": 1 - view_id,
                "T_C_O": T_C_O.reshape(-1).tolist(),
            }
        )
        cameras[str(view_id)] = {"cam_K": K, "depth_scale": 1.0}
        gt[str(view_id)] = [
            {
                "obj_id": 0,
                "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
                "cam_t_m2c": [0.0, 0.0, 500.0],
            }
        ]
        bbox = [2, 2, 4, 4]
        info[str(view_id)] = [
            {
                "bbox_obj": bbox,
                "bbox_visib": bbox,
                "px_count_all": 16,
                "px_count_visib": 16,
            }
        ]
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[2:6, 2:6] = 127
        depth = np.zeros((8, 8), dtype=np.uint16)
        depth[2:6, 2:6] = 500
        mask = (depth > 0).astype(np.uint8) * 255
        for folder, array, suffix in (
            ("rgb", rgb, f"{stem}.png"),
            ("depth", depth, f"{stem}.png"),
            ("mask", mask, f"{stem}_000000.png"),
            ("mask_visib", mask, f"{stem}_000000.png"),
        ):
            path = object_dir / folder / suffix
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)
    _write_json(
        object_dir / "reference_manifest.json",
        {
            "complete": True,
            "config_fingerprint": "fingerprint",
            "object_id": 0,
            "gso_id": "object_0",
            "view_count": 2,
            "poses": poses,
        },
    )
    _write_json(object_dir / "scene_camera.json", cameras)
    _write_json(object_dir / "scene_gt.json", gt)
    _write_json(object_dir / "scene_gt_info.json", info)
    return mapping_path


class ObjectReferenceIndexTest(unittest.TestCase):
    def test_index_and_source_round_trip_metric_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = _make_bank(root)
            index = root / "pi3_index" / "references.sqlite"
            result = build_reference_index(
                root,
                index,
                namespace="gso",
                mapping_path=mapping,
                verify_images="all",
                verify_workers=4,
            )
            self.assertEqual(result, {"objects": 1, "views": 2, "decoded_views": 2})
            source = IndexedBOPReferenceSource(
                root, object_namespace="gso", source_name="render"
            )
            group = source.groups_for_object(0)[0]
            records = source.records_for_group(group)
            self.assertEqual([record["coverage_view_id"] for record in records], [1, 0])
            raw = source.load_raw_object_view(records[0])
            self.assertAlmostEqual(float(raw.depthmap.max()), 0.5)
            self.assertEqual(int(raw.object_mask.sum()), 16)
            np.testing.assert_allclose(raw.camera_pose, np.linalg.inv(raw.T_C_O))

    def test_complete_mode_rejects_mapping_objects_without_renders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = _make_bank(root, mapping_objects=2)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                build_reference_index(
                    root,
                    root / "references.sqlite",
                    namespace="gso",
                    mapping_path=mapping,
                    verify_images="none",
                )

    def test_required_model_catalog_can_be_subset_of_raw_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = _make_bank(root, mapping_objects=2)
            models_info = root / "models_info.json"
            _write_json(models_info, {"0": {"diameter": 1.0}})
            result = build_reference_index(
                root,
                root / "references.sqlite",
                namespace="gso",
                mapping_path=mapping,
                required_object_ids_path=models_info,
                verify_images="none",
            )
            self.assertEqual(result["objects"], 1)
            with sqlite3.connect(root / "references.sqlite") as db:
                metadata = dict(db.execute("SELECT key, value FROM metadata"))
            self.assertEqual(metadata["required_object_count"], "1")


if __name__ == "__main__":
    unittest.main()
