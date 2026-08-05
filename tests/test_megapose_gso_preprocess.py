import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import struct
import tarfile
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "preprocess"
    / "megapose_gso.py"
)
SPEC = importlib.util.spec_from_file_location("megapose_gso_preprocess", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
megapose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(megapose)


def _png_header(width=720, height=540):
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(
        ">II", width, height
    )


def _camera():
    return {
        "cam_K": [500.0, 0.0, 360.0, 0.0, 501.0, 270.0, 0.0, 0.0, 1.0],
        "cam_R_w2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "cam_t_w2c": [10.0, 20.0, 1000.0],
        "depth_scale": 0.1,
    }


def _poses(object_ids):
    return [
        {
            "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "cam_t_m2c": [100.0 + index, 200.0, 1000.0],
            "obj_id": object_id,
        }
        for index, object_id in enumerate(object_ids)
    ]


def _infos(count):
    return [
        {
            "bbox_obj": [1, 2, 30, 40],
            "bbox_visib": [3, 4, 20, 25],
            "px_count_all": 1200,
            "px_count_valid": 1000,
            "px_count_visib": 900 - index,
            "visib_fract": 0.75 - 0.01 * index,
        }
        for index in range(count)
    ]


def _add_member(tar, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _write_shard(root, shard_number, frames, corrupt):
    shard = root / f"shard-{shard_number:06d}.tar"
    with tarfile.open(shard, "w") as tar:
        for key, object_ids in frames:
            members = {
                "camera.json": json.dumps(_camera()).encode(),
                "depth.png": _png_header() + b"synthetic-depth",
                "gt.json": json.dumps(_poses(object_ids)).encode(),
                "gt_info.json": json.dumps(_infos(len(object_ids))).encode(),
                "mask.json": json.dumps([{"counts": "x"}] * len(object_ids)).encode(),
                "mask_visib.json": json.dumps([{"counts": "y"}] * len(object_ids)).encode(),
                "rgb.jpg": f"jpeg-{key}".encode(),
            }
            for suffix, payload in members.items():
                _add_member(tar, f"{key}.{suffix}", payload)
    manifest_dir = root / "GSO_broken_depth_maps"
    manifest_dir.mkdir(exist_ok=True)
    manifest = manifest_dir / megapose.MANIFEST_TEMPLATE.format(
        shard_number=shard_number
    )
    with manifest.open("w", encoding="utf-8") as stream:
        for key, _ in frames:
            stream.write(f"{key}:{list(corrupt.get(key, []))}\n")
    return shard


class MegaPoseGsoPreprocessTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_index_supports_scene_object_track_and_direct_payload_access(self):
        shard = _write_shard(
            self.root,
            0,
            [
                ("000007_000000", [5, 5]),
                ("000007_000001", [5, 5]),
                ("000008_000000", [5]),
            ],
            {"000007_000001": [1]},
        )
        summary = megapose.preprocess_dataset(self.root)
        self.assertEqual(summary["counts"]["shards"], 1)
        self.assertEqual(summary["counts"]["frames"], 3)
        self.assertEqual(summary["counts"]["instances"], 5)
        self.assertEqual(summary["counts"]["scene_tracks"], 3)
        self.assertEqual(summary["counts"]["objects"], 1)
        self.assertEqual(summary["counts"]["corrupt_instance_observations"], 1)

        db = sqlite3.connect(self.root / "pi3_index" / "megapose_gso.sqlite")
        db.row_factory = sqlite3.Row
        tracks = db.execute(
            "SELECT * FROM scene_tracks WHERE object_id = 5 ORDER BY scene_id, gt_id"
        ).fetchall()
        self.assertEqual([(row["scene_id"], row["gt_id"]) for row in tracks], [(7, 0), (7, 1), (8, 0)])
        self.assertEqual(tracks[0]["frame_count"], 2)
        self.assertEqual(tracks[1]["clean_visible_frame_count"], 1)

        row = db.execute(
            """
            SELECT i.T_C_O_f32, i.depth_corrupt, f.*, s.relative_path
            FROM instances AS i JOIN frames AS f ON f.id = i.frame_id
            JOIN shards AS s ON s.id = f.shard_id
            WHERE f.frame_key = '000007_000000' AND i.gt_id = 0
            """
        ).fetchone()
        self.assertAlmostEqual(
            megapose.unpack_f32_matrix(row["T_C_O_f32"], 4, 4)[0][3], 0.1
        )
        self.assertAlmostEqual(row["depth_unit_m"], 0.0001)
        self.assertEqual((row["width"], row["height"]), (720, 540))
        with shard.open("rb") as stream:
            stream.seek(row["rgb_offset"])
            self.assertEqual(stream.read(row["rgb_size"]), b"jpeg-000007_000000")
        db.close()

    def test_resume_skips_unchanged_and_indexes_new_shards(self):
        _write_shard(self.root, 0, [("000001_000000", [9])], {})
        first = megapose.preprocess_dataset(self.root)
        self.assertEqual(first["processed_shard_numbers_this_run"], [0])
        second = megapose.preprocess_dataset(self.root)
        self.assertEqual(second["processed_shard_numbers_this_run"], [])
        self.assertEqual(second["skipped_unchanged_shard_numbers_this_run"], [0])
        second_shard = _write_shard(self.root, 1, [("000002_000000", [9])], {})
        third = megapose.preprocess_dataset(self.root)
        self.assertEqual(third["processed_shard_numbers_this_run"], [1])
        self.assertEqual(third["counts"]["frames"], 2)
        self.assertEqual(third["counts"]["scenes"], 2)

        second_shard.unlink()
        fourth = megapose.preprocess_dataset(self.root)
        self.assertEqual(fourth["removed_missing_shard_numbers_this_run"], [1])
        self.assertEqual(fourth["counts"]["frames"], 1)

    def test_missing_corruption_manifest_is_explicitly_unknown(self):
        _write_shard(self.root, 0, [("000001_000000", [9])], {})
        manifest = (
            self.root
            / "GSO_broken_depth_maps"
            / megapose.MANIFEST_TEMPLATE.format(shard_number=0)
        )
        manifest.unlink()
        with self.assertRaises(FileNotFoundError):
            megapose.preprocess_dataset(self.root)
        summary = megapose.preprocess_dataset(
            self.root, allow_missing_corruption_manifests=True
        )
        self.assertEqual(
            summary["counts"]["unknown_corruption_instance_observations"], 1
        )

    def test_prepared_object_scene_split_is_exact_and_reproducible(self):
        frames = [
            (f"{scene_id:06d}_000000", list(range(10)))
            for scene_id in range(10)
        ]
        _write_shard(self.root, 0, frames, {})
        first = megapose.preprocess_dataset(self.root)
        split_path = Path(first["split_path"])
        with split_path.open("r", encoding="utf-8") as stream:
            first_split = json.load(stream)

        self.assertEqual(first_split["format"], "pi3_entity_split_v1")
        self.assertEqual(first_split["seed"], 2026)
        self.assertEqual(
            first_split["membership_policy"],
            {
                "train": "train_object AND train_scene",
                "val": "val_object AND val_scene",
                "cross_partition_pairs": "excluded",
            },
        )
        self.assertEqual(len(first_split["splits"]["train"]["object_ids"]), 8)
        self.assertEqual(len(first_split["splits"]["val"]["object_ids"]), 2)
        self.assertEqual(len(first_split["splits"]["train"]["scene_ids"]), 8)
        self.assertEqual(len(first_split["splits"]["val"]["scene_ids"]), 2)
        self.assertFalse(
            set(first_split["splits"]["train"]["object_ids"])
            & set(first_split["splits"]["val"]["object_ids"])
        )
        self.assertFalse(
            set(first_split["splits"]["train"]["scene_ids"])
            & set(first_split["splits"]["val"]["scene_ids"])
        )

        megapose.preprocess_dataset(self.root)
        with split_path.open("r", encoding="utf-8") as stream:
            repeated_split = json.load(stream)
        self.assertEqual(first_split["splits"], repeated_split["splits"])

        megapose.preprocess_dataset(self.root, split_seed=77)
        with split_path.open("r", encoding="utf-8") as stream:
            alternate_split = json.load(stream)
        self.assertNotEqual(first_split["splits"], alternate_split["splits"])


if __name__ == "__main__":
    unittest.main()
