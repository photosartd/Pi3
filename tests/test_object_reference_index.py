import json
from pathlib import Path
import pickle
import shutil
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image

from datasets.object_sources import IndexedBOPReferenceSource
from datasets.preprocess.megapose_gso_geometry import (
    BuildSettings,
    _ThreadShardCache,
    decode_render_frame,
    iter_render_jobs,
)
from datasets.preprocess.render.object_reference_index import build_reference_index


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_bank(root: Path, *, object_ids=(0,), mapping_objects=None, resolution=8):
    """Build a tiny loose-file reference bank fixture, one object per ID."""

    if mapping_objects is None:
        mapping_objects = max(object_ids) + 1
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
            "resolution_wh": [resolution, resolution],
            "num_views": 2,
            "view_order": "sequential",
        },
    )
    for object_id in object_ids:
        object_dir = root / f"{object_id:06d}"
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
                    "obj_id": object_id,
                    "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
                    "cam_t_m2c": [0.0, 0.0, 500.0],
                }
            ]
            half = resolution // 4
            center = resolution // 2
            bbox = [center - half, center - half, 2 * half, 2 * half]
            info[str(view_id)] = [
                {
                    "bbox_obj": bbox,
                    "bbox_visib": bbox,
                    "px_count_all": (2 * half) ** 2,
                    "px_count_visib": (2 * half) ** 2,
                }
            ]
            rgb = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            rgb[center - half : center + half, center - half : center + half] = 127
            depth = np.zeros((resolution, resolution), dtype=np.uint16)
            depth[center - half : center + half, center - half : center + half] = 500
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
                "object_id": object_id,
                "gso_id": f"object_{object_id}",
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
            self.assertEqual(result["objects"], 1)
            self.assertEqual(result["views"], 2)
            self.assertEqual(result["decoded_views"], 2)
            self.assertEqual(result["shards"], 1)
            self.assertTrue((root / "shard-000000.tar").is_file())
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

    def test_packed_bytes_match_original_loose_files_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = _make_bank(root, object_ids=(0, 1))
            index = root / "pi3_index" / "references.sqlite"
            build_reference_index(
                root, index, namespace="gso", mapping_path=mapping, verify_images="all"
            )
            with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as db:
                db.row_factory = sqlite3.Row
                rows = db.execute(
                    """
                    SELECT v.object_id, v.view_id, v.rgb_offset, v.rgb_size,
                           v.depth_offset, v.depth_size, v.mask_offset, v.mask_size,
                           v.mask_visib_offset, v.mask_visib_size, s.relative_path
                    FROM views AS v JOIN shards AS s ON s.id = v.shard_id
                    """
                ).fetchall()
            self.assertEqual(len(rows), 4)
            for row in rows:
                object_id, view_id = int(row["object_id"]), int(row["view_id"])
                stem = f"{view_id:06d}"
                originals = {
                    "rgb": root / f"{object_id:06d}" / "rgb" / f"{stem}.png",
                    "depth": root / f"{object_id:06d}" / "depth" / f"{stem}.png",
                    "mask": root / f"{object_id:06d}" / "mask" / f"{stem}_000000.png",
                    "mask_visib": root
                    / f"{object_id:06d}"
                    / "mask_visib"
                    / f"{stem}_000000.png",
                }
                with (root / row["relative_path"]).open("rb") as shard_file:
                    for kind, original_path in originals.items():
                        shard_file.seek(int(row[f"{kind}_offset"]))
                        packed = shard_file.read(int(row[f"{kind}_size"]))
                        self.assertEqual(packed, original_path.read_bytes())

    def test_shard_target_forces_split_without_ever_splitting_one_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = _make_bank(root, object_ids=(0, 1, 2, 3))
            index = root / "pi3_index" / "references.sqlite"
            # Each object packs 8 tiny PNGs; a target of a few hundred bytes
            # forces a new shard roughly every object.
            result = build_reference_index(
                root,
                index,
                namespace="gso",
                mapping_path=mapping,
                verify_images="none",
                shard_target_bytes=200,
            )
            self.assertEqual(result["objects"], 4)
            self.assertGreater(result["shards"], 1)
            with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as db:
                db.row_factory = sqlite3.Row
                shard_by_object = dict(
                    db.execute(
                        "SELECT object_id, GROUP_CONCAT(DISTINCT shard_id) AS shards "
                        "FROM views GROUP BY object_id"
                    ).fetchall()
                )
                shard_count = db.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
            self.assertEqual(shard_count, result["shards"])
            for object_id, shards in shard_by_object.items():
                self.assertNotIn(
                    ",", shards, f"object {object_id} split across shards: {shards}"
                )
            source = IndexedBOPReferenceSource(
                root, object_namespace="gso", source_name="render"
            )
            for object_id in (0, 1, 2, 3):
                group = source.groups_for_object(object_id)[0]
                records = source.records_for_group(group)
                self.assertEqual(len(records), 2)
                for record in records:
                    source.load_raw_object_view(record)

    def test_delete_source_after_verify_removes_loose_files_but_keeps_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = _make_bank(root, object_ids=(0, 1))
            index = root / "pi3_index" / "references.sqlite"
            result = build_reference_index(
                root,
                index,
                namespace="gso",
                mapping_path=mapping,
                verify_images="all",
                delete_source_after_verify=True,
            )
            self.assertEqual(result["deleted_objects"], 2)
            self.assertGreater(result["reclaimed_bytes"], 0)
            self.assertFalse((root / "000000" / "rgb").exists())
            self.assertFalse((root / "000000").exists())
            self.assertFalse((root / "000001").exists())
            # The bank-wide manifest and the packed shards/index survive.
            self.assertTrue((root / "reference_bank.json").is_file())
            self.assertTrue((root / "shard-000000.tar").is_file())
            source = IndexedBOPReferenceSource(
                root, object_namespace="gso", source_name="render"
            )
            for object_id in (0, 1):
                group = source.groups_for_object(object_id)[0]
                for record in source.records_for_group(group):
                    raw = source.load_raw_object_view(record)
                    self.assertEqual(int(raw.object_mask.sum()), 16)

    def test_rerun_with_fewer_objects_removes_stale_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = _make_bank(root, object_ids=(0, 1, 2, 3))
            index = root / "pi3_index" / "references.sqlite"
            first = build_reference_index(
                root,
                index,
                namespace="gso",
                mapping_path=mapping,
                verify_images="none",
                shard_target_bytes=200,
            )
            self.assertGreater(first["shards"], 1)
            # Simulate the bank shrinking between runs (objects pruned from disk).
            shutil.rmtree(root / "000002")
            shutil.rmtree(root / "000003")
            models_info = root / "models_info.json"
            _write_json(models_info, {"0": {}, "1": {}})
            second = build_reference_index(
                root,
                index,
                namespace="gso",
                mapping_path=mapping,
                required_object_ids_path=models_info,
                verify_images="none",
                shard_target_bytes=200,
            )
            self.assertEqual(second["objects"], 2)
            remaining_shards = sorted(root.glob("shard-*.tar"))
            self.assertEqual(len(remaining_shards), second["shards"])
            with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as db:
                indexed_relpaths = {
                    row[0] for row in db.execute("SELECT relative_path FROM shards")
                }
            self.assertEqual(
                indexed_relpaths, {path.name for path in remaining_shards}
            )

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

    def test_reader_rejects_legacy_v1_format_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_bank(root)
            index_dir = root / "pi3_index"
            index_dir.mkdir(parents=True, exist_ok=True)
            legacy = index_dir / "references.sqlite"
            connection = sqlite3.connect(legacy)
            connection.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE objects (object_id INTEGER PRIMARY KEY);
                """
            )
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [
                    ("format", "pi3_object_reference_index_v1"),
                    ("index_complete", "1"),
                    ("namespace", "gso"),
                ],
            )
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "v2"):
                IndexedBOPReferenceSource(root, object_namespace="gso")

    def test_corrupted_shard_payload_fails_loudly_instead_of_silently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = _make_bank(root)
            index = root / "pi3_index" / "references.sqlite"
            build_reference_index(
                root, index, namespace="gso", mapping_path=mapping, verify_images="none"
            )
            with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as db:
                row = db.execute(
                    "SELECT rgb_offset, rgb_size FROM views WHERE view_id=0"
                ).fetchone()
            offset, size = int(row[0]), int(row[1])
            shard_path = root / "shard-000000.tar"
            with shard_path.open("r+b") as stream:
                stream.seek(offset)
                stream.write(b"\x00" * size)
            source = IndexedBOPReferenceSource(
                root, object_namespace="gso", source_name="render"
            )
            group = source.groups_for_object(0)[0]
            records = source.records_for_group(group)
            record = next(r for r in records if int(r["view_id"]) == 0)
            with self.assertRaises(Exception):
                source.load_raw_object_view(record)

    def test_render_geometry_job_iterator_decodes_packed_shard_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = _make_bank(root, object_ids=(0, 1))
            index = root / "pi3_index" / "references.sqlite"
            build_reference_index(
                root, index, namespace="gso", mapping_path=mapping, verify_images="none"
            )
            settings = BuildSettings(min_visible_pixels=1, visibility_floor=0.0)
            jobs = list(iter_render_jobs(index, root, settings))
            self.assertEqual(len(jobs), 4)  # 2 objects x 2 views
            reader = _ThreadShardCache(root)
            for job in jobs:
                decoded = decode_render_frame(job, reader)
                self.assertEqual(decoded.depth.shape, (8, 8))
                self.assertEqual(decoded.masks.shape, (1, 8, 8))
                self.assertAlmostEqual(float(decoded.depth.max()), 0.5)
                self.assertEqual(int(decoded.masks.sum()), 16)

    def test_source_survives_pickling_across_a_simulated_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = _make_bank(root)
            index = root / "pi3_index" / "references.sqlite"
            build_reference_index(
                root, index, namespace="gso", mapping_path=mapping, verify_images="none"
            )
            source = IndexedBOPReferenceSource(
                root, object_namespace="gso", source_name="render"
            )
            group = source.groups_for_object(0)[0]
            records = source.records_for_group(group)
            source.load_raw_object_view(records[0])
            self.assertGreaterEqual(len(source._payload_reader._shard_fds), 1)
            reopened = pickle.loads(pickle.dumps(source))
            self.assertEqual(len(reopened._payload_reader._shard_fds), 0)
            group = reopened.groups_for_object(0)[0]
            records = reopened.records_for_group(group)
            raw = reopened.load_raw_object_view(records[0])
            self.assertEqual(int(raw.object_mask.sum()), 16)


if __name__ == "__main__":
    unittest.main()
