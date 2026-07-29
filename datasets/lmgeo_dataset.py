import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *


PHOTOMETRIC_INTERPOLATIONS = (lanczos, bicubic, bilinear)


def _adjust_hue(image, hue_delta):
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    hue_shift = int(round(float(hue_delta) * 255.0))
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_shift) % 256).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _adjust_gamma(image, gamma):
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.clip(array ** float(gamma), 0.0, 1.0)
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _apply_role_photometric_spec(image, spec):
    """Apply the existing Pi3 photometric recipe with pre-sampled parameters."""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")

    image = TF.adjust_brightness(image, spec["brightness"])
    image = TF.adjust_contrast(image, spec["contrast"])
    image = TF.adjust_saturation(image, spec["saturation"])
    image = _adjust_hue(image, spec["hue"])
    image = _adjust_gamma(image, spec["gamma"])

    if spec["jpeg_enabled"]:
        image_cv = np.asarray(image)[:, :, ::-1]
        _, encoded = cv2.imencode(
            ".jpg",
            image_cv,
            [cv2.IMWRITE_JPEG_QUALITY, int(spec["jpeg_quality"])],
        )
        image_cv = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        image = Image.fromarray(image_cv[:, :, ::-1])

    if spec["blur_enabled"]:
        width, height = image.size
        ratio = float(spec["blur_resize_ratio"])
        resized_small = image.resize(
            (max(1, int(width * ratio)), max(1, int(height * ratio))),
            resample=lanczos,
        )
        image = resized_small.resize(
            (width, height),
            resample=PHOTOMETRIC_INTERPOLATIONS[int(spec["blur_interpolation_index"])],
        )

    return image


class LMGeoDataset(BaseDataset):
    """LM-O object-centric overfit dataset for Pi3 training.

    The dataset builds one fixed multi-view sample from two sources:

    - reference/keyframe views from ``train/<object_id>``, usually clean object
      renders;
    - query views from one 25-frame window inside a ``train_pbr`` scene.

    BOP poses are stored as object-to-camera transforms. Pi3 expects
    camera-to-world poses, so this loader uses the object/CAD frame as the
    world frame and returns ``camera_pose = inv(T_C_O)``. Raw BOP depths are
    converted with ``depth_scale`` and then from millimeters to meters by
    default, matching the pose translation units.
    """

    def __init__(
        self,
        data_root,
        object_id=1,
        reference_split="train",
        query_split="train_pbr",
        query_scene_id=0,
        query_subscene_id=0,
        query_window_size=25,
        num_reference=5,
        num_query=20,
        reference_rgb_masking=True,
        query_rgb_masking=False,
        depth_masking=True,
        mask_type="mask_visib",
        visibility_min=0.1,
        reference_selection="uniform",
        query_selection="first",
        depth_unit_scale=0.001,
        filter_center_crop_visibility=True,
        min_center_crop_bbox_area=1.0,
        filter_preprocessed_query_depth=True,
        min_preprocessed_query_depth_pixels=1,
        allow_repeat=False,
        photometric_augmentation=False,
        photometric_brightness=(0.7, 1.3),
        photometric_contrast=(0.7, 1.3),
        photometric_saturation=(0.7, 1.3),
        photometric_hue=(-0.1, 0.1),
        photometric_gamma=(0.7, 1.3),
        photometric_jpeg_prob=0.5,
        photometric_jpeg_quality=(20, 100),
        photometric_blur_prob=0.5,
        photometric_blur_resize_ratio=(0.25, 1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset_label = "LMGeo"
        self._json_cache = {}
        self.data_root = Path(data_root)
        self.object_id = int(object_id)
        self.reference_split = reference_split
        self.query_split = query_split
        self.query_scene_id = int(query_scene_id)
        self.query_subscene_id = int(query_subscene_id)
        self.query_window_size = int(query_window_size)
        self.num_reference = int(num_reference)
        self.num_query = int(num_query)
        self.reference_rgb_masking = bool(reference_rgb_masking)
        self.query_rgb_masking = bool(query_rgb_masking)
        self.depth_masking = bool(depth_masking)
        self.mask_type = str(mask_type)
        self.visibility_min = float(visibility_min)
        self.reference_selection = str(reference_selection)
        self.query_selection = str(query_selection)
        self.depth_unit_scale = float(depth_unit_scale)
        self.filter_center_crop_visibility = bool(filter_center_crop_visibility)
        self.min_center_crop_bbox_area = float(min_center_crop_bbox_area)
        self.filter_preprocessed_query_depth = bool(filter_preprocessed_query_depth)
        self.min_preprocessed_query_depth_pixels = int(min_preprocessed_query_depth_pixels)
        self.allow_repeat = bool(allow_repeat)
        self.photometric_augmentation = bool(photometric_augmentation)
        self.photometric_brightness = tuple(float(value) for value in photometric_brightness)
        self.photometric_contrast = tuple(float(value) for value in photometric_contrast)
        self.photometric_saturation = tuple(float(value) for value in photometric_saturation)
        self.photometric_hue = tuple(float(value) for value in photometric_hue)
        self.photometric_gamma = tuple(float(value) for value in photometric_gamma)
        self.photometric_jpeg_prob = float(photometric_jpeg_prob)
        self.photometric_jpeg_quality = tuple(int(value) for value in photometric_jpeg_quality)
        self.photometric_blur_prob = float(photometric_blur_prob)
        self.photometric_blur_resize_ratio = tuple(float(value) for value in photometric_blur_resize_ratio)
        self._photometric_role_specs = {}

        if self.mask_type not in {"mask", "mask_visib"}:
            raise ValueError("mask_type must be 'mask' or 'mask_visib'")
        if self.reference_selection not in {"uniform", "first", "random"}:
            raise ValueError("reference_selection must be 'uniform', 'first', or 'random'")
        if self.query_selection not in {"uniform", "first", "random"}:
            raise ValueError("query_selection must be 'uniform', 'first', or 'random'")
        if not 0.0 <= self.photometric_jpeg_prob <= 1.0:
            raise ValueError("photometric_jpeg_prob must be in [0, 1]")
        if not 0.0 <= self.photometric_blur_prob <= 1.0:
            raise ValueError("photometric_blur_prob must be in [0, 1]")

        self.reference_records = self._build_reference_records()
        self.query_records = self._build_query_records()

        print(
            f"[{self.dataset_label}] object={self.object_id}, "
            f"reference_records={len(self.reference_records)}, "
            f"query_records={len(self.query_records)}"
        )

    def __len__(self):
        return 1

    def _load_json(self, path):
        path = Path(path)
        cache = getattr(self, "_json_cache", None)
        cache_key = str(path)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        with open(path, "r") as handle:
            data = json.load(handle)
        if cache is not None:
            cache[cache_key] = data
        return data

    def _find_image(self, stem, exts):
        stem = Path(stem)
        for ext in exts:
            path = stem.with_suffix(ext)
            if path.exists():
                return path
        raise FileNotFoundError(f"No image found for {stem} with extensions {exts}")

    def _bop_pose_to_camera_pose(self, gt):
        T_C_O = np.eye(4, dtype=np.float32)
        T_C_O[:3, :3] = np.asarray(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        T_C_O[:3, 3] = (
            np.asarray(gt["cam_t_m2c"], dtype=np.float32) * self.depth_unit_scale
        )
        return np.linalg.inv(T_C_O).astype(np.float32)

    def _record_from_gt(
        self,
        scene_dir,
        im_id,
        gt_id,
        gt,
        info,
        cam,
        split,
        object_id=None,
        image_size=None,
        rgb_ext=None,
        trust_paths=False,
    ):
        object_id = self.object_id if object_id is None else int(object_id)
        if trust_paths:
            rgb_ext = ".png" if rgb_ext is None else str(rgb_ext)
            rgb_path = scene_dir / "rgb" / f"{im_id:06d}{rgb_ext}"
            depth_path = scene_dir / "depth" / f"{im_id:06d}.png"
            mask_path = scene_dir / self.mask_type / f"{im_id:06d}_{gt_id:06d}.png"
        else:
            rgb_path = self._find_image(scene_dir / "rgb" / f"{im_id:06d}", [".png", ".jpg", ".jpeg"])
            depth_path = self._find_image(scene_dir / "depth" / f"{im_id:06d}", [".png", ".tif"])
            mask_path = self._find_image(
                scene_dir / self.mask_type / f"{im_id:06d}_{gt_id:06d}", [".png", ".jpg", ".jpeg"]
            )
        if image_size is None:
            with Image.open(rgb_path) as image:
                image_size = image.size

        return {
            "split": split,
            "scene_dir": str(scene_dir),
            "im_id": int(im_id),
            "gt_id": int(gt_id),
            "object_id": int(object_id),
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "mask_path": str(mask_path),
            "K": np.asarray(cam["cam_K"], dtype=np.float32).reshape(3, 3),
            "image_size": image_size,
            "bbox_visib": info.get("bbox_visib"),
            "bbox_obj": info.get("bbox_obj"),
            "depth_scale": float(cam.get("depth_scale", 1.0)),
            "T_C_O": self._bop_pose_to_matrix(gt),
            "camera_pose": self._bop_pose_to_camera_pose(gt),
            "visib_fract": float(info.get("visib_fract", 1.0)),
        }

    def _bop_pose_to_matrix(self, gt):
        T_C_O = np.eye(4, dtype=np.float32)
        T_C_O[:3, :3] = np.asarray(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        T_C_O[:3, 3] = (
            np.asarray(gt["cam_t_m2c"], dtype=np.float32) * self.depth_unit_scale
        )
        return T_C_O

    def _center_crop_bbox(self, record):
        width, height = record["image_size"]
        K = record["K"]
        cx, cy = K[:2, 2].round().astype(int)
        min_margin_x = min(cx, width - cx)
        min_margin_y = min(cy, height - cy)
        return (
            cx - min_margin_x,
            cy - min_margin_y,
            cx + min_margin_x,
            cy + min_margin_y,
        )

    def _bbox_intersection_area(self, bbox, crop_bbox):
        x, y, width, height = bbox
        x0 = max(float(x), float(crop_bbox[0]))
        y0 = max(float(y), float(crop_bbox[1]))
        x1 = min(float(x + width), float(crop_bbox[2]))
        y1 = min(float(y + height), float(crop_bbox[3]))
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def _filter_records_visible_after_center_crop(self, records, source):
        if not self.filter_center_crop_visibility:
            return records

        filtered = []
        for record in records:
            bbox = record.get("bbox_visib") or record.get("bbox_obj")
            if bbox is None:
                filtered.append(record)
                continue
            crop_bbox = self._center_crop_bbox(record)
            area = self._bbox_intersection_area(bbox, crop_bbox)
            if area >= self.min_center_crop_bbox_area:
                filtered.append(record)

        dropped = len(records) - len(filtered)
        if dropped:
            print(
                f"[{self.dataset_label}] dropped {dropped}/{len(records)} {source} records "
                "whose object bbox is outside Pi3's center crop"
            )
        return filtered

    def _filter_records_with_preprocessed_depth(self, records, source, log_drops=True):
        if not self.filter_preprocessed_query_depth or source != "query":
            return records

        filtered = []
        for record in records:
            if self._record_has_preprocessed_depth(record):
                filtered.append(record)

        dropped = len(records) - len(filtered)
        if dropped and log_drops:
            print(
                f"[{self.dataset_label}] dropped {dropped}/{len(records)} {source} records "
                "with empty object depth after Pi3 preprocessing"
            )
        return filtered

    def _record_has_preprocessed_depth(self, record):
        resolution = self._resolutions[0]
        depth = self._read_depth_meters(record["depth_path"], record["depth_scale"])
        mask = self._read_mask(record["mask_path"], depth.shape)
        depth[~mask] = 0.0
        with Image.open(record["rgb_path"]) as image:
            image = image.convert("RGB")
            _, depth, _ = self._crop_resize_if_necessary(
                image,
                depth,
                record["K"].copy(),
                resolution,
                rng=self._rng,
                info=record["rgb_path"],
            )
        return int((depth > 0).sum()) >= self.min_preprocessed_query_depth_pixels

    def _build_reference_records(self, object_id=None):
        object_id = self.object_id if object_id is None else int(object_id)
        scene_dir = self.data_root / self.reference_split / f"{object_id:06d}"
        records = self._collect_scene_records(
            scene_dir=scene_dir,
            split=self.reference_split,
            frame_ids=None,
            object_id=object_id,
        )
        return self._filter_records_visible_after_center_crop(records, "reference")

    def _build_query_records(self, object_id=None, query_scene_id=None, query_subscene_id=None):
        object_id = self.object_id if object_id is None else int(object_id)
        query_scene_id = self.query_scene_id if query_scene_id is None else int(query_scene_id)
        query_subscene_id = self.query_subscene_id if query_subscene_id is None else int(query_subscene_id)
        scene_dir = self.data_root / self.query_split / f"{query_scene_id:06d}"
        start = query_subscene_id * self.query_window_size
        frame_ids = range(start, start + self.query_window_size)
        records = self._collect_scene_records(
            scene_dir=scene_dir,
            split=self.query_split,
            frame_ids=frame_ids,
            object_id=object_id,
        )
        records = self._filter_records_visible_after_center_crop(records, "query")
        return self._filter_records_with_preprocessed_depth(records, "query")

    def _collect_scene_records(self, scene_dir, split, frame_ids, object_id=None):
        scene_dir = Path(scene_dir)
        object_id = self.object_id if object_id is None else int(object_id)
        scene_gt = self._load_json(scene_dir / "scene_gt.json")
        scene_gt_info = self._load_json(scene_dir / "scene_gt_info.json")
        scene_camera = self._load_json(scene_dir / "scene_camera.json")

        if frame_ids is None:
            frame_ids = sorted(int(key) for key in scene_gt.keys())

        records = []
        for im_id in frame_ids:
            key = str(int(im_id))
            if key not in scene_gt:
                continue

            gts = scene_gt[key]
            infos = scene_gt_info[key]
            cam = scene_camera[key]
            for gt_id, gt in enumerate(gts):
                if int(gt.get("obj_id", -1)) != object_id:
                    continue

                info = infos[gt_id]
                if float(info.get("visib_fract", 1.0)) < self.visibility_min:
                    continue

                records.append(self._record_from_gt(scene_dir, im_id, gt_id, gt, info, cam, split, object_id=object_id))
                break

        if not records:
            raise ValueError(
                f"No LM-O records found for object {object_id} in {scene_dir} "
                f"with visibility_min={self.visibility_min}"
            )
        return records

    def _select_records(self, records, count, strategy, rng=None):
        if count <= 0:
            return []
        if len(records) < count:
            if not self.allow_repeat:
                raise ValueError(
                    f"Requested {count} records but only {len(records)} are available. "
                    "Lower num_reference/num_query or set allow_repeat=true."
                )
            if strategy == "random":
                rng = self._rng if rng is None else rng
                indices = rng.choice(len(records), size=count, replace=True)
                return [records[int(idx)] for idx in indices]
            repeats = int(np.ceil(count / len(records)))
            return (records * repeats)[:count]

        if strategy == "first":
            return records[:count]
        if strategy == "random":
            rng = self._rng if rng is None else rng
            indices = rng.choice(len(records), size=count, replace=False)
            return [records[int(idx)] for idx in np.sort(indices)]

        idxs = np.linspace(0, len(records) - 1, count).astype(int)
        return [records[idx] for idx in idxs]

    def _read_mask(self, path, shape_hw):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        mask = mask > 0
        if mask.shape != shape_hw:
            raise ValueError(f"Mask shape {mask.shape} does not match image/depth shape {shape_hw}: {path}")
        return mask

    def _read_depth_meters(self, path, depth_scale):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot read depth: {path}")
        return depth.astype(np.float32) * float(depth_scale) * self.depth_unit_scale

    def _sample_photometric_spec(self, rng):
        jpeg_enabled = bool(rng.random() < self.photometric_jpeg_prob)
        blur_enabled = bool(rng.random() < self.photometric_blur_prob)
        return {
            "brightness": float(rng.uniform(*self.photometric_brightness)),
            "contrast": float(rng.uniform(*self.photometric_contrast)),
            "saturation": float(rng.uniform(*self.photometric_saturation)),
            "hue": float(rng.uniform(*self.photometric_hue)),
            "gamma": float(rng.uniform(*self.photometric_gamma)),
            "jpeg_enabled": jpeg_enabled,
            "jpeg_quality": int(rng.integers(*self.photometric_jpeg_quality)) if jpeg_enabled else 100,
            "blur_enabled": blur_enabled,
            "blur_resize_ratio": (
                float(rng.uniform(*self.photometric_blur_resize_ratio))
                if blur_enabled
                else 1.0
            ),
            "blur_interpolation_index": (
                int(rng.integers(0, len(PHOTOMETRIC_INTERPOLATIONS)))
                if blur_enabled
                else 0
            ),
        }

    def _prepare_sample_photometric_augmentation(self, rng):
        if not self.photometric_augmentation:
            self._photometric_role_specs = {}
            return
        self._photometric_role_specs = {
            "reference": self._sample_photometric_spec(rng),
            "query": self._sample_photometric_spec(rng),
        }

    def _apply_sample_photometric_augmentation(self, image, view_role):
        spec = getattr(self, "_photometric_role_specs", {}).get(str(view_role))
        if spec is None:
            return image
        return _apply_role_photometric_spec(image, spec)

    def _maybe_transform_raw_view(
        self,
        *,
        record,
        rgb,
        depthmap,
        mask,
        intrinsics,
        T_C_O,
        camera_pose,
        view_role,
    ):
        return rgb, depthmap, mask, intrinsics, T_C_O, camera_pose, {}

    def _should_depth_mask_view(self, *, view_role, reference_source):
        force_object_masking = bool(view_role == "reference" and reference_source == "context_scene")
        return bool(self.depth_masking or force_object_masking)

    def _load_view(self, record, rgb_masking, view_role):
        object_id = int(record.get("object_id", getattr(self, "object_id", -1)))
        query_scene_id = int(record.get("query_scene_id", getattr(self, "query_scene_id", -1)))
        query_subscene_id = int(record.get("query_subscene_id", getattr(self, "query_subscene_id", -1)))
        source_scene_id = int(record.get("query_scene_id", -1))
        source_subscene_id = int(record.get("query_subscene_id", -1))
        reference_source = str(
            record.get(
                "reference_source",
                "query" if view_role == "query" else "render",
            )
        )

        with Image.open(record["rgb_path"]) as image:
            rgb = np.array(image.convert("RGB"))
        depthmap = self._read_depth_meters(record["depth_path"], record["depth_scale"])
        mask = self._read_mask(record["mask_path"], depthmap.shape)
        force_object_masking = bool(view_role == "reference" and reference_source == "context_scene")

        if self._should_depth_mask_view(view_role=view_role, reference_source=reference_source):
            depthmap = depthmap.copy()
            depthmap[~mask] = 0.0

        if rgb_masking or force_object_masking:
            rgb = rgb.copy()
            rgb[~mask] = 0

        intrinsics = record["K"].copy()
        T_C_O = record["T_C_O"].astype(np.float32)
        camera_pose = record["camera_pose"].astype(np.float32)
        (
            rgb,
            depthmap,
            mask,
            intrinsics,
            T_C_O,
            camera_pose,
            transform_meta,
        ) = self._maybe_transform_raw_view(
            record=record,
            rgb=rgb,
            depthmap=depthmap,
            mask=mask,
            intrinsics=intrinsics,
            T_C_O=T_C_O,
            camera_pose=camera_pose,
            view_role=view_role,
        )

        rgb, depthmap, intrinsics = self._crop_resize_if_necessary(
            rgb,
            depthmap,
            intrinsics,
            self._current_resolution,
            rng=self._rng,
            info=record["rgb_path"],
        )
        rgb = self._apply_sample_photometric_augmentation(rgb, view_role)

        view = {
            "img": rgb,
            "depthmap": depthmap.astype(np.float32),
            "camera_pose": camera_pose.astype(np.float32),
            "T_C_O": T_C_O.astype(np.float32),
            "camera_intrinsics": intrinsics.astype(np.float32),
            "dataset": self.dataset_label,
            "object_id": np.int64(object_id),
            "view_role": str(view_role),
            "is_reference": bool(view_role == "reference"),
            "is_query": bool(view_role == "query"),
            # Optional paired-query experiments override these fields after
            # loading. Defaults keep every historical dataset/config unchanged.
            "is_query_context": False,
            "is_cropped_query": False,
            "is_original_query": False,
            "query_pair_index": np.int64(-1),
            "reference_source": reference_source,
            "is_context_reference": bool(view_role == "reference" and reference_source == "context_scene"),
            "source_scene_id": np.int64(source_scene_id),
            "source_subscene_id": np.int64(source_subscene_id),
            "scene_id": np.int64(query_scene_id if view_role == "query" else object_id),
            "im_id": np.int64(record["im_id"]),
            "gt_id": np.int64(record["gt_id"]),
            "visib_fract": np.float32(record["visib_fract"]),
            "target_id": np.int64(record.get("query_target_index", -1)),
            "query_instance_rank": np.int64(record.get("query_instance_rank", -1)),
            "source": f"{record['split']}/{Path(record['scene_dir']).name}/{record['im_id']:06d}_{record['gt_id']:06d}",
            "label": (
                f"obj{object_id:06d}_"
                f"scene{query_scene_id:06d}_sub{query_subscene_id:04d}"
            ),
            "instance": f"{record['split']}_{record['im_id']:06d}_{record['gt_id']:06d}",
        }
        view.update(transform_meta)
        return view

    def _load_query_views(self, record):
        """Expand one selected query record into one or more model views."""

        return [
            self._load_view(
                record,
                rgb_masking=self.query_rgb_masking,
                view_role="query",
            )
        ]

    def _get_views(self, index, resolution, rng):
        expected = self.num_reference + self.num_query
        if self.frame_num != expected:
            raise ValueError(
                f"LMGeoDataset is configured for {expected} views "
                f"({self.num_reference} reference + {self.num_query} query), "
                f"but the sampler requested frame_num={self.frame_num}. "
                "Set train/test.image_num_range to the same total."
            )

        self._current_resolution = resolution
        self._prepare_sample_photometric_augmentation(rng)
        reference_records = self._select_records(
            self.reference_records,
            self.num_reference,
            self.reference_selection,
            rng=rng,
        )
        query_records = self._select_records(
            self.query_records,
            self.num_query,
            self.query_selection,
            rng=rng,
        )

        self.this_views_info = {
            "object_id": self.object_id,
            "reference": [(r["split"], r["im_id"], r["gt_id"]) for r in reference_records],
            "query": [(r["split"], r["im_id"], r["gt_id"]) for r in query_records],
        }

        views = [
            self._load_view(record, rgb_masking=self.reference_rgb_masking, view_role="reference")
            for record in reference_records
        ]
        for query_index, record in enumerate(query_records):
            query_views = self._load_query_views(record)
            for view in query_views:
                view["query_pair_index"] = np.int64(query_index)
            views.extend(query_views)
        return views


class LMGeoSequenceDataset(LMGeoDataset):
    """Indexed LM-O object-centric sequence dataset.

    Unlike :class:`LMGeoDataset`, this dataset is not tied to one fixed
    scene/subscene. It builds an index of valid object/query windows and samples
    reference/query counts at load time. The actual total number of views is the
    ``frame_num`` requested by Pi3's dynamic batch sampler.
    """

    def __init__(
        self,
        data_root,
        object_ids=(8,),
        reference_split="train",
        query_split="train_pbr",
        query_scene_ids="all",
        query_subscene_ids="all",
        query_windows=None,
        query_source="auto",
        query_target_file=None,
        query_window_size=25,
        num_reference_range=(5, 5),
        num_query_range=(20, 20),
        reference_rgb_masking=True,
        query_rgb_masking=False,
        depth_masking=True,
        mask_type="mask_visib",
        visibility_min=0.1,
        reference_selection="random",
        query_selection="random",
        depth_unit_scale=0.001,
        filter_center_crop_visibility=True,
        min_center_crop_bbox_area=1.0,
        filter_preprocessed_query_depth=True,
        min_preprocessed_query_depth_pixels=1,
        allow_repeat=False,
        photometric_augmentation=False,
        photometric_brightness=(0.7, 1.3),
        photometric_contrast=(0.7, 1.3),
        photometric_saturation=(0.7, 1.3),
        photometric_hue=(-0.1, 0.1),
        photometric_gamma=(0.7, 1.3),
        photometric_jpeg_prob=0.5,
        photometric_jpeg_quality=(20, 100),
        photometric_blur_prob=0.5,
        photometric_blur_resize_ratio=(0.25, 1.0),
        min_query_records=None,
        max_query_subsequences=None,
        sort_query_windows_by_visibility=True,
        filter_target_center_crop_visibility=True,
        filter_target_preprocessed_depth=True,
        context_reference_fraction=0.0,
        context_reference_eval=False,
        context_reference_exclude="scene",
        **kwargs,
    ):
        BaseDataset.__init__(self, **kwargs)

        self.dataset_label = "LMGeoSequence"
        self._json_cache = {}
        self.data_root = Path(data_root)
        self.reference_split = reference_split
        self.query_split = query_split
        self.query_source = str(query_source)
        self.query_target_file = query_target_file
        self.query_window_size = int(query_window_size)
        self.num_reference_range = self._normalize_range(num_reference_range, "num_reference_range")
        self.num_query_range = self._normalize_range(num_query_range, "num_query_range")
        self.reference_rgb_masking = bool(reference_rgb_masking)
        self.query_rgb_masking = bool(query_rgb_masking)
        self.depth_masking = bool(depth_masking)
        self.mask_type = str(mask_type)
        self.visibility_min = float(visibility_min)
        self.reference_selection = str(reference_selection)
        self.query_selection = str(query_selection)
        self.depth_unit_scale = float(depth_unit_scale)
        self.filter_center_crop_visibility = bool(filter_center_crop_visibility)
        self.min_center_crop_bbox_area = float(min_center_crop_bbox_area)
        self.filter_preprocessed_query_depth = bool(filter_preprocessed_query_depth)
        self.min_preprocessed_query_depth_pixels = int(min_preprocessed_query_depth_pixels)
        self.allow_repeat = bool(allow_repeat)
        self.photometric_augmentation = bool(photometric_augmentation)
        self.photometric_brightness = tuple(float(value) for value in photometric_brightness)
        self.photometric_contrast = tuple(float(value) for value in photometric_contrast)
        self.photometric_saturation = tuple(float(value) for value in photometric_saturation)
        self.photometric_hue = tuple(float(value) for value in photometric_hue)
        self.photometric_gamma = tuple(float(value) for value in photometric_gamma)
        self.photometric_jpeg_prob = float(photometric_jpeg_prob)
        self.photometric_jpeg_quality = tuple(int(value) for value in photometric_jpeg_quality)
        self.photometric_blur_prob = float(photometric_blur_prob)
        self.photometric_blur_resize_ratio = tuple(float(value) for value in photometric_blur_resize_ratio)
        self._photometric_role_specs = {}
        default_min_query_records = self.num_query_range[0] if self.allow_repeat else self.num_query_range[1]
        self.min_query_records = int(min_query_records) if min_query_records is not None else default_min_query_records
        self.max_query_subsequences = None if max_query_subsequences is None else int(max_query_subsequences)
        self.sort_query_windows_by_visibility = bool(sort_query_windows_by_visibility)
        self.filter_target_center_crop_visibility = bool(filter_target_center_crop_visibility)
        self.filter_target_preprocessed_depth = bool(filter_target_preprocessed_depth)
        self.context_reference_eval = bool(context_reference_eval)
        self.context_reference_exclude = str(context_reference_exclude)
        self.context_reference_fraction = (
            float(context_reference_fraction)
            if self.mode == "train" or self.context_reference_eval
            else 0.0
        )

        if self.mask_type not in {"mask", "mask_visib"}:
            raise ValueError("mask_type must be 'mask' or 'mask_visib'")
        if self.query_source not in {"auto", "windows", "bop_targets"}:
            raise ValueError("query_source must be 'auto', 'windows', or 'bop_targets'")
        if self.reference_selection not in {"uniform", "first", "random"}:
            raise ValueError("reference_selection must be 'uniform', 'first', or 'random'")
        if self.query_selection not in {"uniform", "first", "random"}:
            raise ValueError("query_selection must be 'uniform', 'first', or 'random'")
        if not 0.0 <= self.photometric_jpeg_prob <= 1.0:
            raise ValueError("photometric_jpeg_prob must be in [0, 1]")
        if not 0.0 <= self.photometric_blur_prob <= 1.0:
            raise ValueError("photometric_blur_prob must be in [0, 1]")
        if not 0.0 <= self.context_reference_fraction <= 1.0:
            raise ValueError(
                f"context_reference_fraction must be in [0, 1], got {context_reference_fraction}"
            )
        if self.context_reference_exclude not in {"scene", "subscene"}:
            raise ValueError(
                f"context_reference_exclude must be 'scene' or 'subscene', got {context_reference_exclude}"
            )

        self.object_ids = self._resolve_object_ids(object_ids)
        self.query_windows = self._normalize_query_windows(query_windows)

        self.reference_records_by_object = {
            int(object_id): self._build_reference_records(object_id=object_id)
            for object_id in self.object_ids
        }
        if self._use_bop_targets():
            self.samples = self._build_bop_target_samples()
        else:
            self.samples = self._build_samples(query_scene_ids, query_subscene_ids)
        if not self.samples:
            raise ValueError("LMGeoSequenceDataset did not find any valid samples")
        self.context_reference_records_by_object_scene = (
            self._build_context_reference_index()
            if self.context_reference_fraction > 0.0
            else {}
        )

        print(
            f"[{self.dataset_label}] objects={self.object_ids}, samples={len(self.samples)}, "
            f"query_source={self._resolved_query_source()}, "
            f"num_reference_range={self.num_reference_range}, num_query_range={self.num_query_range}, "
            f"context_reference_fraction={self.context_reference_fraction}"
        )

    def __len__(self):
        return len(self.samples)

    def convert_attributes(self):
        """Keep nested Python metadata as-is.

        The base conversion is useful for huge flat lists, but this dataset owns
        nested dictionaries of records. Keeping them as Python objects is simpler
        and avoids accidental object-array or Series conversions.
        """

    def _normalize_range(self, value, name):
        if isinstance(value, int):
            lo = hi = int(value)
        else:
            lo, hi = [int(x) for x in value]
        if lo <= 0 or hi < lo:
            raise ValueError(f"{name} must be a positive [min, max] range, got {value}")
        return (lo, hi)

    def _resolved_query_source(self):
        if self.query_source != "auto":
            return self.query_source
        if self.mode in {"test", "val"} and self.query_split == "test":
            return "bop_targets"
        return "windows"

    def _use_bop_targets(self):
        return self._resolved_query_source() == "bop_targets"

    def _resolve_object_ids(self, object_ids):
        if isinstance(object_ids, str) and object_ids == "all":
            split_dir = self.data_root / self.reference_split
            return sorted(int(path.name) for path in split_dir.iterdir() if path.is_dir() and path.name.isdigit())
        return [int(object_id) for object_id in object_ids]

    def _resolve_scene_ids(self, query_scene_ids):
        if isinstance(query_scene_ids, str) and query_scene_ids == "all":
            split_dir = self.data_root / self.query_split
            return sorted(int(path.name) for path in split_dir.iterdir() if path.is_dir() and path.name.isdigit())
        return [int(scene_id) for scene_id in query_scene_ids]

    def _scene_subscene_ids(self, scene_id, query_subscene_ids):
        if not (isinstance(query_subscene_ids, str) and query_subscene_ids == "all"):
            return [int(subscene_id) for subscene_id in query_subscene_ids]

        scene_gt = self._load_json(self.data_root / self.query_split / f"{int(scene_id):06d}" / "scene_gt.json")
        if not scene_gt:
            return []
        max_im_id = max(int(key) for key in scene_gt.keys())
        return list(range(max_im_id // self.query_window_size + 1))

    def _normalize_query_windows(self, query_windows):
        if query_windows is None:
            return None
        output = []
        for window in query_windows:
            output.append(
                {
                    "object_id": int(window.get("object_id", self.object_ids[0] if hasattr(self, "object_ids") else 0)),
                    "query_scene_id": int(window["query_scene_id"]),
                    "query_subscene_id": int(window["query_subscene_id"]),
                }
            )
        return output

    def _candidate_windows(self, query_scene_ids, query_subscene_ids):
        if self.query_windows is not None:
            return self.query_windows

        windows = []
        scene_ids = self._resolve_scene_ids(query_scene_ids)
        for object_id in self.object_ids:
            for scene_id in scene_ids:
                for subscene_id in self._scene_subscene_ids(scene_id, query_subscene_ids):
                    windows.append(
                        {
                            "object_id": int(object_id),
                            "query_scene_id": int(scene_id),
                            "query_subscene_id": int(subscene_id),
                        }
                    )
        return windows

    def _build_samples(self, query_scene_ids, query_subscene_ids):
        if self.query_windows is None:
            return self._build_scene_window_samples(query_scene_ids, query_subscene_ids)

        samples = []
        for window in self._candidate_windows(query_scene_ids, query_subscene_ids):
            object_id = int(window["object_id"])
            scene_id = int(window["query_scene_id"])
            subscene_id = int(window["query_subscene_id"])
            try:
                records = self._build_query_records(
                    object_id=object_id,
                    query_scene_id=scene_id,
                    query_subscene_id=subscene_id,
                )
            except (FileNotFoundError, ValueError):
                continue
            if not records:
                continue
            if len(records) < self.min_query_records and not self.allow_repeat:
                continue
            for record in records:
                record["query_scene_id"] = scene_id
                record["query_subscene_id"] = subscene_id
            mean_visibility = float(np.mean([record["visib_fract"] for record in records])) if records else 0.0
            samples.append(
                {
                    "object_id": object_id,
                    "query_scene_id": scene_id,
                    "query_subscene_id": subscene_id,
                    "query_records": records,
                    "mean_visibility": mean_visibility,
                }
            )

        if self.sort_query_windows_by_visibility:
            samples.sort(key=lambda item: (-item["mean_visibility"], item["object_id"], item["query_scene_id"], item["query_subscene_id"]))
        if self.max_query_subsequences is not None:
            samples = samples[: self.max_query_subsequences]
        return samples

    def _scene_rgb_ext(self, scene_dir):
        rgb_dir = scene_dir / "rgb"
        for path in sorted(rgb_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                return path.suffix
        raise ValueError(f"No RGB images found in {rgb_dir}")

    def _first_scene_image_size(self, scene_dir, scene_gt):
        rgb_ext = self._scene_rgb_ext(scene_dir)
        for im_id in sorted(int(key) for key in scene_gt.keys()):
            rgb_path = scene_dir / "rgb" / f"{im_id:06d}{rgb_ext}"
            if not rgb_path.exists():
                continue
            with Image.open(rgb_path) as image:
                return image.size
        raise ValueError(f"No RGB images found in {scene_dir}")

    def _subscene_filter(self, scene_id, query_subscene_ids):
        if isinstance(query_subscene_ids, str) and query_subscene_ids == "all":
            return None
        return {int(subscene_id) for subscene_id in query_subscene_ids}

    def _record_passes_center_crop(self, record):
        if not self.filter_center_crop_visibility:
            return True
        bbox = record.get("bbox_visib") or record.get("bbox_obj")
        if bbox is None:
            return True
        crop_bbox = self._center_crop_bbox(record)
        return self._bbox_intersection_area(bbox, crop_bbox) >= self.min_center_crop_bbox_area

    def _build_scene_window_samples(self, query_scene_ids, query_subscene_ids):
        scene_ids = self._resolve_scene_ids(query_scene_ids)
        object_filter = set(self.object_ids)
        window_records = {}
        dropped_center_crop = 0
        dropped_visibility = 0
        dropped_preprocessed_depth = 0
        windows_with_preprocessed_depth_drop = 0

        for scene_id in scene_ids:
            scene_dir = self.data_root / self.query_split / f"{int(scene_id):06d}"
            try:
                scene_gt = self._load_json(scene_dir / "scene_gt.json")
                scene_gt_info = self._load_json(scene_dir / "scene_gt_info.json")
                scene_camera = self._load_json(scene_dir / "scene_camera.json")
                image_size = self._first_scene_image_size(scene_dir, scene_gt)
                rgb_ext = self._scene_rgb_ext(scene_dir)
            except (FileNotFoundError, ValueError):
                continue

            allowed_subscenes = self._subscene_filter(scene_id, query_subscene_ids)
            for key in sorted(scene_gt.keys(), key=lambda value: int(value)):
                im_id = int(key)
                subscene_id = im_id // self.query_window_size
                if allowed_subscenes is not None and subscene_id not in allowed_subscenes:
                    continue
                cam = scene_camera[key]
                for gt_id, gt in enumerate(scene_gt[key]):
                    object_id = int(gt.get("obj_id", -1))
                    if object_id not in object_filter:
                        continue
                    info = scene_gt_info[key][gt_id]
                    if float(info.get("visib_fract", 1.0)) < self.visibility_min:
                        dropped_visibility += 1
                        continue
                    record = self._record_from_gt(
                        scene_dir,
                        im_id,
                        gt_id,
                        gt,
                        info,
                        cam,
                        self.query_split,
                        object_id=object_id,
                        image_size=image_size,
                        rgb_ext=rgb_ext,
                        trust_paths=True,
                    )
                    record["query_scene_id"] = int(scene_id)
                    record["query_subscene_id"] = int(subscene_id)
                    if not self._record_passes_center_crop(record):
                        dropped_center_crop += 1
                        continue
                    window_records.setdefault((object_id, int(scene_id), int(subscene_id)), []).append(record)

        samples = []
        for (object_id, scene_id, subscene_id), records in window_records.items():
            if len(records) < self.min_query_records and not self.allow_repeat:
                continue
            if self.filter_preprocessed_query_depth:
                before_filter = len(records)
                records = self._filter_records_with_preprocessed_depth(records, "query", log_drops=False)
                dropped_here = before_filter - len(records)
                dropped_preprocessed_depth += dropped_here
                if dropped_here:
                    windows_with_preprocessed_depth_drop += 1
                if not records:
                    continue
                if len(records) < self.min_query_records and not self.allow_repeat:
                    continue
            samples.append(
                {
                    "object_id": int(object_id),
                    "query_scene_id": int(scene_id),
                    "query_subscene_id": int(subscene_id),
                    "query_records": records,
                    "mean_visibility": float(np.mean([record["visib_fract"] for record in records])),
                }
            )

        if self.sort_query_windows_by_visibility:
            samples.sort(key=lambda item: (-item["mean_visibility"], item["object_id"], item["query_scene_id"], item["query_subscene_id"]))
        if self.max_query_subsequences is not None:
            samples = samples[: self.max_query_subsequences]

        if dropped_visibility or dropped_center_crop or dropped_preprocessed_depth:
            print(
                f"[{self.dataset_label}] dropped train/window records: "
                f"visibility={dropped_visibility}, center_crop={dropped_center_crop}, "
                f"preprocessed_depth={dropped_preprocessed_depth} "
                f"(from {windows_with_preprocessed_depth_drop} windows)"
            )
        return samples

    def _target_file_path(self):
        if self.query_target_file is None:
            return self.data_root / "lmo" / "test_targets_bop19.json"
        path = Path(self.query_target_file)
        return path if path.is_absolute() else self.data_root / path

    def _build_bop_target_samples(self):
        target_path = self._target_file_path()
        targets = self._load_json(target_path)
        samples = []
        dropped_missing = 0
        dropped_center_crop = 0
        dropped_preprocessed_depth = 0
        object_filter = set(self.object_ids)

        for target_index, target in enumerate(targets):
            object_id = int(target["obj_id"])
            if object_id not in object_filter:
                continue

            scene_id = int(target["scene_id"])
            im_id = int(target["im_id"])
            instance_count = int(target.get("inst_count", 1))
            scene_dir = self.data_root / self.query_split / f"{scene_id:06d}"

            scene_gt = self._load_json(scene_dir / "scene_gt.json")
            scene_gt_info = self._load_json(scene_dir / "scene_gt_info.json")
            scene_camera = self._load_json(scene_dir / "scene_camera.json")
            key = str(im_id)
            if key not in scene_gt:
                dropped_missing += 1
                continue

            matching_gt_ids = [
                gt_id
                for gt_id, gt in enumerate(scene_gt[key])
                if int(gt.get("obj_id", -1)) == object_id
            ]
            if len(matching_gt_ids) < instance_count:
                dropped_missing += 1
                continue

            for instance_rank, gt_id in enumerate(matching_gt_ids[:instance_count]):
                gt = scene_gt[key][gt_id]
                info = scene_gt_info[key][gt_id]
                record = self._record_from_gt(
                    scene_dir,
                    im_id,
                    gt_id,
                    gt,
                    info,
                    scene_camera[key],
                    self.query_split,
                    object_id=object_id,
                )
                record["query_scene_id"] = scene_id
                record["query_subscene_id"] = -1
                record["query_target_index"] = int(target_index)
                record["query_instance_rank"] = int(instance_rank)

                if self.filter_target_center_crop_visibility:
                    bbox = record.get("bbox_visib") or record.get("bbox_obj")
                    if bbox is not None:
                        crop_bbox = self._center_crop_bbox(record)
                        if self._bbox_intersection_area(bbox, crop_bbox) < self.min_center_crop_bbox_area:
                            dropped_center_crop += 1
                            continue

                if self.filter_target_preprocessed_depth and not self._record_has_preprocessed_depth(record):
                    dropped_preprocessed_depth += 1
                    continue

                samples.append(
                    {
                        "object_id": object_id,
                        "query_scene_id": scene_id,
                        "query_subscene_id": -1,
                        "query_records": [record],
                        "mean_visibility": float(record["visib_fract"]),
                        "target_index": int(target_index),
                    }
                )

        if self.max_query_subsequences is not None:
            samples = samples[: self.max_query_subsequences]

        print(
            f"[{self.dataset_label}] BOP targets from {target_path}: "
            f"kept={len(samples)}, dropped_missing={dropped_missing}, "
            f"dropped_center_crop={dropped_center_crop}, "
            f"dropped_preprocessed_depth={dropped_preprocessed_depth}"
        )
        return samples

    def _build_context_reference_index(self):
        records_by_object_scene = {}
        seen = set()
        for sample in self.samples:
            for record in sample.get("query_records", []):
                object_id = int(record["object_id"])
                scene_id = int(record.get("query_scene_id", -1))
                key = (
                    object_id,
                    scene_id,
                    int(record["im_id"]),
                    int(record["gt_id"]),
                    str(record["scene_dir"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                context_record = dict(record)
                context_record["reference_source"] = "context_scene"
                records_by_object_scene.setdefault(object_id, {}).setdefault(scene_id, []).append(context_record)

        total = sum(
            len(records)
            for records_by_scene in records_by_object_scene.values()
            for records in records_by_scene.values()
        )
        if total == 0:
            print(
                f"[{self.dataset_label}] context_reference_fraction={self.context_reference_fraction} "
                "but no context reference records were found; falling back to render references."
            )
        else:
            print(f"[{self.dataset_label}] context reference records={total}")
        return records_by_object_scene

    def _context_reference_candidates(self, object_id, query_scene_id, query_subscene_id):
        records_by_scene = self.context_reference_records_by_object_scene.get(int(object_id), {})
        candidates = []
        for scene_id, records in records_by_scene.items():
            if self.context_reference_exclude == "scene" and int(scene_id) == int(query_scene_id):
                continue
            if self.context_reference_exclude == "subscene":
                candidates.extend(
                    record
                    for record in records
                    if not (
                        int(record.get("query_scene_id", -1)) == int(query_scene_id)
                        and int(record.get("query_subscene_id", -1)) == int(query_subscene_id)
                    )
                )
            else:
                candidates.extend(records)
        return candidates

    def _annotate_reference_records(self, records, reference_source):
        annotated = []
        for record in records:
            item = dict(record)
            item["reference_source"] = reference_source
            annotated.append(item)
        return annotated

    def _select_reference_records_for_sample(self, object_id, query_scene_id, query_subscene_id, count, rng):
        render_pool = self.reference_records_by_object[object_id]
        if self.context_reference_fraction <= 0.0:
            records = self._select_records(render_pool, count, self.reference_selection, rng=rng)
            return self._annotate_reference_records(records, "render")

        context_pool = self._context_reference_candidates(object_id, query_scene_id, query_subscene_id)
        context_count = int(rng.binomial(int(count), self.context_reference_fraction))
        if not context_pool:
            context_count = 0
        elif not self.allow_repeat:
            context_count = min(context_count, len(context_pool))

        render_count = int(count) - context_count
        render_records = self._select_records(render_pool, render_count, self.reference_selection, rng=rng)
        context_records = self._select_records(context_pool, context_count, self.reference_selection, rng=rng)

        records = self._annotate_reference_records(render_records, "render")
        records.extend(self._annotate_reference_records(context_records, "context_scene"))
        rng.shuffle(records)
        return records

    def _sample_counts(self, total_frames, reference_records, query_records, rng):
        total_frames = int(total_frames)
        ref_min, ref_max = self.num_reference_range
        query_min, query_max = self.num_query_range
        if not self.allow_repeat:
            ref_max = min(ref_max, len(reference_records))
            query_max = min(query_max, len(query_records))

        valid_refs = [
            ref_count
            for ref_count in range(ref_min, ref_max + 1)
            if query_min <= total_frames - ref_count <= query_max
        ]
        if not valid_refs:
            raise ValueError(
                f"Cannot split frame_num={total_frames} into reference range {self.num_reference_range} "
                f"and query range {self.num_query_range}"
            )
        ref_count = int(rng.choice(valid_refs))
        query_count = total_frames - ref_count
        return ref_count, query_count

    def _get_views(self, index, resolution, rng):
        sample = self.samples[int(index) % len(self.samples)]
        object_id = int(sample["object_id"])
        reference_pool = self.reference_records_by_object[object_id]
        query_pool = sample["query_records"]
        ref_count, query_count = self._sample_counts(self.frame_num, reference_pool, query_pool, rng)

        self._current_resolution = resolution
        self._prepare_sample_photometric_augmentation(rng)
        reference_records = self._select_reference_records_for_sample(
            object_id,
            sample["query_scene_id"],
            sample["query_subscene_id"],
            ref_count,
            rng,
        )
        query_records = self._select_records(
            query_pool,
            query_count,
            self.query_selection,
            rng=rng,
        )

        self.this_views_info = {
            "object_id": object_id,
            "query_scene_id": sample["query_scene_id"],
            "query_subscene_id": sample["query_subscene_id"],
            "reference_count": ref_count,
            "query_count": query_count,
            "render_reference_count": sum(1 for r in reference_records if r.get("reference_source") == "render"),
            "context_reference_count": sum(1 for r in reference_records if r.get("reference_source") == "context_scene"),
            "reference": [
                (
                    r.get("reference_source", "render"),
                    r["split"],
                    int(r.get("query_scene_id", -1)),
                    r["im_id"],
                    r["gt_id"],
                )
                for r in reference_records
            ],
            "query": [(r["split"], r["im_id"], r["gt_id"]) for r in query_records],
        }

        views = [
            self._load_view(record, rgb_masking=self.reference_rgb_masking, view_role="reference")
            for record in reference_records
        ]
        for query_index, record in enumerate(query_records):
            query_views = self._load_query_views(record)
            for view in query_views:
                view["query_pair_index"] = np.int64(query_index)
            views.extend(query_views)
        return views


class LMGeoSameSceneCeilingDataset(LMGeoSequenceDataset):
    """Evaluation-only same-scene ceiling dataset for object-pose alignment.

    Every sample takes one object in one scene window, randomly holds out one
    visible frame as the query, and uses the remaining visible frames from the
    same object/window as references.  This tests Pi3's ceiling when reference
    and query views already come from the same scene distribution and therefore
    avoid the render-to-real/context gap.  The held-out query record is never
    used as a reference, so the existing reference-only Sim(3) alignment metric
    remains leak-free.
    """

    def __init__(
        self,
        *args,
        max_reference=24,
        min_reference=2,
        query_frame_strategy="random",
        skip_ambiguous_instances=True,
        min_query_records=None,
        **kwargs,
    ):
        self.same_scene_max_reference = int(max_reference)
        self.same_scene_min_reference = int(min_reference)
        self.query_frame_strategy = str(query_frame_strategy)
        self.skip_ambiguous_instances = bool(skip_ambiguous_instances)
        self._same_scene_epoch = 0
        self._same_scene_base_seed = 777

        if self.same_scene_min_reference <= 0:
            raise ValueError("min_reference must be positive")
        if self.same_scene_max_reference < self.same_scene_min_reference:
            raise ValueError(
                f"max_reference must be >= min_reference, got "
                f"{self.same_scene_max_reference} < {self.same_scene_min_reference}"
            )
        if self.query_frame_strategy not in {"random", "first"}:
            raise ValueError("query_frame_strategy must be 'random' or 'first'")
        if min_query_records is None:
            min_query_records = self.same_scene_min_reference + 1

        kwargs.setdefault("query_source", "windows")
        kwargs.setdefault("num_reference_range", (self.same_scene_min_reference, self.same_scene_max_reference))
        kwargs.setdefault("num_query_range", (1, 1))
        kwargs.setdefault("allow_repeat", False)
        super().__init__(*args, min_query_records=min_query_records, **kwargs)

        self.dataset_label = "LMGeoSameSceneCeiling"
        self.samples, drop_counts = self._filter_same_scene_samples(self.samples)
        if not self.samples:
            raise ValueError(
                "LMGeoSameSceneCeilingDataset did not find any windows with "
                f"at least {self.same_scene_min_reference + 1} usable same-scene records"
            )

        ref_counts = [
            min(self.same_scene_max_reference, max(0, len(sample["query_records"]) - 1))
            for sample in self.samples
        ]
        print(
            f"[{self.dataset_label}] same-scene ceiling samples={len(self.samples)}, "
            f"query_frame_strategy={self.query_frame_strategy}, "
            f"reference_count_range=({min(ref_counts)}, {max(ref_counts)}), "
            f"dropped_too_few={drop_counts['too_few']}, "
            f"dropped_ambiguous={drop_counts['ambiguous']}"
        )

    def set_epoch(self, epoch, base_seed=None):
        self._same_scene_epoch = int(epoch)
        if base_seed is not None:
            self._same_scene_base_seed = int(base_seed)

    def _build_reference_records(self, object_id=None):
        # This diagnostic never samples render references.  Avoid walking the
        # full train/<object_id> render split during validation-only setup.
        return []

    def _filter_same_scene_samples(self, samples):
        filtered = []
        drop_counts = {"too_few": 0, "ambiguous": 0}
        for sample in samples:
            records = list(sample["query_records"])
            if len(records) < self.same_scene_min_reference + 1:
                drop_counts["too_few"] += 1
                continue
            if self.skip_ambiguous_instances:
                im_ids = [int(record["im_id"]) for record in records]
                if len(set(im_ids)) != len(im_ids):
                    drop_counts["ambiguous"] += 1
                    continue
            filtered.append(sample)
        return filtered, drop_counts

    def _same_scene_rng(self, index):
        seed = (
            int(self._same_scene_base_seed)
            + 1000003 * int(self._same_scene_epoch)
            + 9176 * int(index)
        )
        return np.random.default_rng(seed)

    def _record_identity(self, record):
        return (
            str(record.get("split", "")),
            str(record.get("scene_dir", "")),
            int(record.get("im_id", -1)),
            int(record.get("gt_id", -1)),
            int(record.get("object_id", -1)),
        )

    def _select_same_scene_query_record(self, records, rng):
        if self.query_frame_strategy == "first":
            return records[0]
        return records[int(rng.integers(0, len(records)))]

    def _tag_same_scene_reference(self, record):
        item = dict(record)
        item["reference_source"] = "same_scene"
        return item

    def _load_same_scene_reference_view(self, record, sample):
        view = self._load_view(
            record,
            rgb_masking=self.reference_rgb_masking,
            view_role="reference",
        )
        view["reference_source"] = "same_scene"
        view["is_context_reference"] = False
        view["is_same_scene_reference"] = True
        view["scene_id"] = np.int64(sample["query_scene_id"])
        view["source_scene_id"] = np.int64(sample["query_scene_id"])
        view["source_subscene_id"] = np.int64(sample["query_subscene_id"])
        return view

    def _get_views(self, index, resolution, rng):
        sample_index = int(index) % len(self.samples)
        sample = self.samples[sample_index]
        object_id = int(sample["object_id"])
        query_pool = list(sample["query_records"])
        local_rng = self._same_scene_rng(sample_index)
        query_record = self._select_same_scene_query_record(query_pool, local_rng)
        query_identity = self._record_identity(query_record)

        reference_pool = [
            record for record in query_pool
            if self._record_identity(record) != query_identity
        ]
        max_reference_by_sampler = max(0, int(self.frame_num) - 1)
        ref_count = min(
            self.same_scene_max_reference,
            max_reference_by_sampler,
            len(reference_pool),
        )
        if ref_count < self.same_scene_min_reference:
            raise ValueError(
                f"Same-scene sample has only {len(reference_pool)} reference candidates "
                f"after holding out the query; need at least {self.same_scene_min_reference}"
            )

        self._current_resolution = resolution
        self._prepare_sample_photometric_augmentation(rng)
        reference_records = [
            self._tag_same_scene_reference(record)
            for record in self._select_records(
                reference_pool,
                ref_count,
                self.reference_selection,
                rng=local_rng,
            )
        ]

        reference_identities = {self._record_identity(record) for record in reference_records}
        if query_identity in reference_identities:
            raise AssertionError("Selected same-scene query leaked into references")

        self.this_views_info = {
            "object_id": object_id,
            "query_scene_id": sample["query_scene_id"],
            "query_subscene_id": sample["query_subscene_id"],
            "reference_count": ref_count,
            "query_count": 1,
            "same_scene_reference_count": ref_count,
            "heldout_query": (
                query_record["split"],
                int(query_record.get("query_scene_id", -1)),
                int(query_record.get("query_subscene_id", -1)),
                int(query_record["im_id"]),
                int(query_record["gt_id"]),
            ),
            "reference": [
                (
                    r.get("reference_source", "same_scene"),
                    r["split"],
                    int(r.get("query_scene_id", -1)),
                    int(r.get("query_subscene_id", -1)),
                    int(r["im_id"]),
                    int(r["gt_id"]),
                )
                for r in reference_records
            ],
            "query": [(query_record["split"], query_record["im_id"], query_record["gt_id"])],
            "leak_free": query_identity not in reference_identities,
        }

        views = [
            self._load_same_scene_reference_view(record, sample)
            for record in reference_records
        ]
        query_views = self._load_query_views(query_record)
        for view in query_views:
            view["query_pair_index"] = np.int64(0)
        views.extend(query_views)
        return views
