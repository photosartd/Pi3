import io
import json
import pickle
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image

from datasets.base.observation import (
    ObservationCapability,
    batch_supports_capability,
)
from datasets.base.transforms import ImgToTensor
from datasets.megapose_gso_dataset import MegaPoseGSOObjectDataset
from datasets.preprocess.megapose_gso import MANIFEST_TEMPLATE, preprocess_dataset


WIDTH = 56
HEIGHT = 42


def _image_bytes(array, format_name):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format=format_name)
    return stream.getvalue()


def _uncompressed_rle(mask):
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts = []
    value = 0
    run = 0
    for pixel in flat:
        pixel = int(pixel > 0)
        if pixel == value:
            run += 1
        else:
            counts.append(run)
            run = 1
            value = pixel
    counts.append(run)
    return {"size": [mask.shape[0], mask.shape[1]], "counts": counts}


def _add_member(tar, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _write_indexed_dataset(root, *, repeated_object_scene=False):
    root = Path(root)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[8:34, 13:43] = 1
    second_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    second_mask[5:19, 2:12] = 1
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb[..., 0] = 80
    rgb[mask > 0] = (180, 120, 60)
    rgb[second_mask > 0] = (60, 170, 100)
    depth = np.full((HEIGHT, WIDTH), 1000, dtype=np.uint16)
    camera = {
        "cam_K": [50.0, 0.0, WIDTH / 2, 0.0, 50.0, HEIGHT / 2, 0.0, 0.0, 1.0],
        "cam_R_w2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "cam_t_w2c": [0.0, 0.0, 1000.0],
        "depth_scale": 1.0,
    }
    pose = {
        "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "cam_t_m2c": [0.0, 0.0, 1000.0],
        "obj_id": 5,
    }
    second_pose = {
        **pose,
        "cam_t_m2c": [100.0, 0.0, 1000.0],
    }
    info = {
        "bbox_obj": [13, 8, 30, 26],
        "bbox_visib": [13, 8, 30, 26],
        "px_count_all": int(mask.sum()),
        "px_count_valid": int(mask.sum()),
        "px_count_visib": int(mask.sum()),
        "visib_fract": 1.0,
    }
    second_info = {
        "bbox_obj": [2, 5, 10, 14],
        "bbox_visib": [2, 5, 10, 14],
        "px_count_all": int(second_mask.sum()),
        "px_count_valid": int(second_mask.sum()),
        "px_count_visib": int(second_mask.sum()),
        "visib_fract": 1.0,
    }
    frames = [(1, 0), (1, 1), (2, 0), (3, 0), (4, 0)]
    shard_path = root / "shard-000000.tar"
    with tarfile.open(shard_path, "w") as tar:
        for scene_id, view_id in frames:
            key = f"{scene_id:06d}_{view_id:06d}"
            poses = [pose]
            infos = [info]
            masks = {"0": _uncompressed_rle(mask)}
            if repeated_object_scene and scene_id == 3:
                poses.append(second_pose)
                infos.append(second_info)
                masks["1"] = _uncompressed_rle(second_mask)
            payloads = {
                "rgb.jpg": _image_bytes(rgb, "JPEG"),
                "depth.png": _image_bytes(depth, "PNG"),
                "camera.json": json.dumps(camera).encode(),
                "gt.json": json.dumps(poses).encode(),
                "gt_info.json": json.dumps(infos).encode(),
                "mask.json": json.dumps(masks).encode(),
                "mask_visib.json": json.dumps(masks).encode(),
            }
            for suffix, payload in payloads.items():
                _add_member(tar, f"{key}.{suffix}", payload)
    manifest_dir = root / "GSO_broken_depth_maps"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / MANIFEST_TEMPLATE.format(shard_number=0)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for scene_id, view_id in frames:
            key = f"{scene_id:06d}_{view_id:06d}"
            stream.write(f"{key}:{[0] if scene_id == 4 else []}\n")
    preprocess_dataset(root)
    return int(mask.sum())


def _write_split_indexed_dataset(root):
    root = Path(root)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[8:34, 13:43] = 1
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb[..., 1] = 60
    rgb[mask > 0] = (170, 110, 50)
    depth = np.full((HEIGHT, WIDTH), 1000, dtype=np.uint16)
    camera = {
        "cam_K": [50.0, 0.0, WIDTH / 2, 0.0, 50.0, HEIGHT / 2, 0.0, 0.0, 1.0],
        "cam_R_w2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "cam_t_w2c": [0.0, 0.0, 1000.0],
        "depth_scale": 1.0,
    }
    poses = [
        {
            "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "cam_t_m2c": [float(object_id), 0.0, 1000.0],
            "obj_id": object_id,
        }
        for object_id in range(10)
    ]
    info = {
        "bbox_obj": [13, 8, 30, 26],
        "bbox_visib": [13, 8, 30, 26],
        "px_count_all": int(mask.sum()),
        "px_count_valid": int(mask.sum()),
        "px_count_visib": int(mask.sum()),
        "visib_fract": 1.0,
    }
    masks = {
        str(gt_id): _uncompressed_rle(mask)
        for gt_id in range(len(poses))
    }
    frames = [(scene_id, 0) for scene_id in range(10)]
    shard_path = root / "shard-000000.tar"
    with tarfile.open(shard_path, "w") as tar:
        for scene_id, view_id in frames:
            key = f"{scene_id:06d}_{view_id:06d}"
            payloads = {
                "rgb.jpg": _image_bytes(rgb, "JPEG"),
                "depth.png": _image_bytes(depth, "PNG"),
                "camera.json": json.dumps(camera).encode(),
                "gt.json": json.dumps(poses).encode(),
                "gt_info.json": json.dumps([info] * len(poses)).encode(),
                "mask.json": json.dumps(masks).encode(),
                "mask_visib.json": json.dumps(masks).encode(),
            }
            for suffix, payload in payloads.items():
                _add_member(tar, f"{key}.{suffix}", payload)
    manifest_dir = root / "GSO_broken_depth_maps"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / MANIFEST_TEMPLATE.format(shard_number=0)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for scene_id, view_id in frames:
            stream.write(f"{scene_id:06d}_{view_id:06d}:[]\n")
    preprocess_dataset(root)


class MegaPoseGSOObjectDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mask_pixels = _write_indexed_dataset(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def _dataset(self, **overrides):
        kwargs = dict(
            data_root=self.root,
            resolution=[[WIDTH, HEIGHT]],
            frame_num=3,
            num_reference_range=(2, 2),
            num_query_range=(1, 1),
            visibility_min=0.1,
            min_visible_pixels=1,
            sampling_regime="independent_scenes",
            visibility_mask_conditioning=True,
            condition_reference_visibility=True,
            condition_query_visibility=False,
            transform=ImgToTensor,
            shuffle=False,
            mode="train",
        )
        kwargs.update(overrides)
        return MegaPoseGSOObjectDataset(**kwargs)

    def test_independent_scenes_decode_metric_geometry_and_conditions(self):
        dataset = self._dataset()
        views = dataset[(0, 0, 3)]
        self.assertEqual(len(views), 3)
        self.assertEqual(len({int(view["scene_id"]) for view in views}), 3)
        self.assertEqual(
            [view["view_role"] for view in views],
            ["reference", "reference", "query"],
        )
        for view in views:
            self.assertEqual(tuple(view["img"].shape), (3, HEIGHT, WIDTH))
            self.assertEqual(int(view["valid_mask"].sum()), self.mask_pixels)
            self.assertAlmostEqual(
                float(view["depthmap"][view["valid_mask"]].mean()), 1.0
            )
            np.testing.assert_allclose(
                view["camera_pose"], np.linalg.inv(view["T_C_O"]), atol=1e-6
            )
            self.assertTrue(view["capability_object_pose"])
            self.assertFalse(view["capability_object_model"])
        self.assertEqual(float(views[0]["visibility_mask_known"].mean()), 1.0)
        self.assertEqual(float(views[1]["visibility_mask_known"].mean()), 1.0)
        self.assertEqual(float(views[2]["visibility_mask_known"].sum()), 0.0)
        self.assertFalse(
            batch_supports_capability(views, ObservationCapability.OBJECT_MODEL)
        )
        self.assertNotIn(4, {track.scene_id for track in dataset._tracks_by_object[5]})
        dataset.close()

    def test_anchor_pair_uses_two_scenes_and_unique_reference_views(self):
        dataset = self._dataset(sampling_regime="anchor_pair")
        views = dataset[(0, 0, 3)]
        reference_scenes = {int(view["scene_id"]) for view in views[:2]}
        query_scenes = {int(view["scene_id"]) for view in views[2:]}
        self.assertEqual(len(reference_scenes), 1)
        self.assertEqual(len(query_scenes), 1)
        self.assertTrue(reference_scenes.isdisjoint(query_scenes))
        self.assertEqual(len({int(view["view_id"]) for view in views[:2]}), 2)
        dataset.close()

    def test_prepared_train_val_object_and_scene_lists_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            split_root = Path(temporary)
            _write_split_indexed_dataset(split_root)
            with (
                split_root / "pi3_index" / "megapose_gso.splits.json"
            ).open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)

            common = dict(
                data_root=split_root,
                resolution=[[WIDTH, HEIGHT]],
                frame_num=2,
                num_reference_range=(1, 1),
                num_query_range=(1, 1),
                visibility_min=0.1,
                min_visible_pixels=1,
                sampling_regime="independent_scenes",
                visibility_mask_conditioning=True,
                condition_reference_visibility=True,
                condition_query_visibility=True,
                transform=ImgToTensor,
                shuffle=False,
                mode="train",
            )
            train = MegaPoseGSOObjectDataset(scene_split="train", **common)
            val = MegaPoseGSOObjectDataset(scene_split="val", **common)
            self.assertEqual(
                set(train.object_ids),
                set(manifest["splits"]["train"]["object_ids"]),
            )
            self.assertEqual(
                set(val.object_ids),
                set(manifest["splits"]["val"]["object_ids"]),
            )
            train_scenes = {
                track.scene_id
                for object_id in train.object_ids
                for track in train._tracks_by_object[object_id]
            }
            val_scenes = {
                track.scene_id
                for object_id in val.object_ids
                for track in val._tracks_by_object[object_id]
            }
            self.assertEqual(
                train_scenes, set(manifest["splits"]["train"]["scene_ids"])
            )
            self.assertEqual(
                val_scenes, set(manifest["splits"]["val"]["scene_ids"])
            )
            self.assertFalse(set(train.object_ids) & set(val.object_ids))
            self.assertFalse(train_scenes & val_scenes)
            train_views = train[(0, 0, 2)]
            val_views = val[(0, 0, 2)]
            self.assertTrue(
                {int(view["object_id"]) for view in train_views}
                <= set(train.object_ids)
            )
            self.assertTrue(
                {int(view["scene_id"]) for view in val_views} <= val_scenes
            )
            train.close()
            val.close()

    def test_dataset_reopens_sqlite_and_tar_handles_after_pickling(self):
        dataset = self._dataset()
        first = dataset[(0, 0, 3)]
        restored = pickle.loads(pickle.dumps(dataset))
        second = restored[(0, 0, 3)]
        self.assertEqual(len(first), len(second))
        self.assertTrue(restored._shard_fds)
        dataset.close()
        restored.close()

    def test_repeated_object_id_scene_requires_query_disambiguation(self):
        with tempfile.TemporaryDirectory() as temporary:
            repeated_root = Path(temporary)
            _write_indexed_dataset(
                repeated_root, repeated_object_scene=True
            )

            with self.assertRaisesRegex(
                ValueError, "repeated object ID require every query"
            ):
                self._dataset(data_root=repeated_root)

            with self.assertRaisesRegex(
                ValueError, "condition_query_visibility=true"
            ):
                self._dataset(
                    data_root=repeated_root,
                    visibility_mask_conditioning=False,
                    condition_query_visibility=True,
                )

            conditioned = self._dataset(
                data_root=repeated_root,
                condition_query_visibility=True,
            )
            self.assertEqual(
                conditioned._same_object_scene_track_counts[(5, 3)], 2
            )
            repeated_track = next(
                track
                for track in conditioned._tracks_by_object[5]
                if track.scene_id == 3 and track.gt_id == 0
            )
            record = conditioned._records_for_track(repeated_track)[0]
            self.assertEqual(record["same_object_scene_track_count"], 2)
            self.assertEqual(record["same_object_frame_instance_count"], 2)
            self.assertEqual(record["same_object_visible_instance_count"], 2)
            conditioned._validate_selected_query_records([record])
            conditioned.close()

            rgb_masked = self._dataset(
                data_root=repeated_root,
                query_rgb_masking=True,
                visibility_mask_conditioning=False,
                condition_reference_visibility=False,
                condition_query_visibility=False,
            )
            self.assertEqual(
                rgb_masked._query_disambiguation_mode(), "rgb_masked"
            )
            rgb_masked.close()


if __name__ == "__main__":
    unittest.main()
