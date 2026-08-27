import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from datasets.base.transforms import *
from datasets.object_centric import (
    ObjectDatasetAdapter,
    PlannedObjectView,
    RawObjectView,
    ViewTreatment,
)
from datasets.object_geometry import (
    GeometryFeatureIndex,
    GeometryPlanIndex,
    normalized_focal_scalar,
    normalized_focal_shape,
    object_preserving_crop_spec,
)


class LMGeoDataset(ObjectDatasetAdapter):
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
        visibility_mask_conditioning=False,
        condition_reference_visibility=True,
        condition_query_visibility=False,
        visibility_condition_corruption="none",
        visibility_condition_shift_fraction=0.5,
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
        self.visibility_mask_conditioning = bool(visibility_mask_conditioning)
        self.condition_reference_visibility = bool(condition_reference_visibility)
        self.condition_query_visibility = bool(condition_query_visibility)
        self.visibility_condition_corruption = str(visibility_condition_corruption)
        self.visibility_condition_shift_fraction = float(visibility_condition_shift_fraction)
        self._configure_role_photometric_augmentation(
            enabled=photometric_augmentation,
            brightness=photometric_brightness,
            contrast=photometric_contrast,
            saturation=photometric_saturation,
            hue=photometric_hue,
            gamma=photometric_gamma,
            jpeg_prob=photometric_jpeg_prob,
            jpeg_quality=photometric_jpeg_quality,
            blur_prob=photometric_blur_prob,
            blur_resize_ratio=photometric_blur_resize_ratio,
        )

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
        if self.visibility_condition_corruption not in {"none", "shift"}:
            raise ValueError(
                "visibility_condition_corruption must be 'none' or 'shift', "
                f"got {visibility_condition_corruption!r}"
            )
        if self.visibility_condition_shift_fraction < 0.0:
            raise ValueError("visibility_condition_shift_fraction must be non-negative")

        self.reference_records = self._build_reference_records()
        self.query_records = self._build_query_records()

        print(
            f"[{self.dataset_label}] object={self.object_id}, "
            f"reference_records={len(self.reference_records)}, "
            f"query_records={len(self.query_records)}"
        )

    def __len__(self):
        return 1

    def supported_frame_counts(self, image_num_range):
        total = int(self.num_reference + self.num_query)
        lo, hi = [int(value) for value in image_num_range]
        return [total] if lo <= total <= hi else []

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

        scene_id = int(Path(scene_dir).name)
        return {
            "split": split,
            "scene_dir": str(scene_dir),
            # Stable BOP identities used by optional derived-geometry indexes.
            # Legacy LMGeo loaders ignore these additive fields.
            "frame_id": scene_id * 1_000_000 + int(im_id),
            "scene_id": scene_id,
            "view_id": int(im_id),
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
        return self.key_query_sampling_policy.select_records(
            records,
            count,
            strategy,
            rng=self._rng if rng is None else rng,
            allow_repeat=self.allow_repeat,
        )

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
        rng=None,
    ):
        return super()._maybe_transform_raw_view(
            record=record,
            rgb=rgb,
            depthmap=depthmap,
            mask=mask,
            intrinsics=intrinsics,
            T_C_O=T_C_O,
            camera_pose=camera_pose,
            view_role=view_role,
            rng=rng,
        )

    def _should_depth_mask_view(self, *, view_role, reference_source):
        force_object_masking = bool(view_role == "reference" and reference_source == "context_scene")
        return bool(self.depth_masking or force_object_masking)

    def _force_rgb_object_masking(self, *, view_role, reference_source):
        return bool(view_role == "reference" and reference_source == "context_scene")

    def _should_condition_visibility_view(self, *, view_role):
        if not self.visibility_mask_conditioning:
            return False
        if view_role == "reference":
            return self.condition_reference_visibility
        if view_role == "query":
            return self.condition_query_visibility
        return False

    @staticmethod
    def _shift_mask_zero(mask, *, dx, dy):
        mask = np.asarray(mask)
        shifted = np.zeros_like(mask)
        height, width = mask.shape[:2]

        if abs(int(dx)) >= width or abs(int(dy)) >= height:
            return shifted

        src_x0 = max(0, -int(dx))
        src_x1 = min(width, width - int(dx))
        dst_x0 = max(0, int(dx))
        dst_x1 = min(width, width + int(dx))
        src_y0 = max(0, -int(dy))
        src_y1 = min(height, height - int(dy))
        dst_y0 = max(0, int(dy))
        dst_y1 = min(height, height + int(dy))

        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return shifted
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
        return shifted

    def _corrupt_visibility_condition(self, condition, *, view_role, rng):
        """Alter only the model-side mask condition, not GT masks/loss targets."""

        if self.visibility_condition_corruption == "none":
            return condition
        condition = np.asarray(condition, dtype=np.float32)
        if self.visibility_condition_corruption == "shift":
            ys, xs = np.nonzero(condition > 0.5)
            if len(xs) == 0:
                return condition

            bbox_w = int(xs.max() - xs.min() + 1)
            bbox_h = int(ys.max() - ys.min() + 1)
            shift_fraction = float(self.visibility_condition_shift_fraction)
            max_axis = "x" if bbox_w >= bbox_h else "y"

            # Move by about half an object footprint while keeping the mask
            # shape itself intact.  This gives a deliberately wrong object cue
            # without changing any supervision target.
            if max_axis == "x":
                dx = max(1, int(round(bbox_w * shift_fraction)))
                dy = 0
            else:
                dx = 0
                dy = max(1, int(round(bbox_h * shift_fraction)))
            if rng.integers(0, 2):
                dx = -dx
                dy = -dy

            shifted = self._shift_mask_zero(condition, dx=dx, dy=dy).astype(np.float32)
            if shifted.sum() > 0:
                return shifted
            return condition
        raise ValueError(f"Unsupported visibility condition corruption {self.visibility_condition_corruption!r}")

    def load_raw_object_view(self, record):
        """Decode a BOP record in meters without applying training policies."""

        with Image.open(record["rgb_path"]) as image:
            rgb = np.array(image.convert("RGB"))
        depthmap = self._read_depth_meters(
            record["depth_path"], record["depth_scale"]
        )
        mask = self._read_mask(record["mask_path"], depthmap.shape)
        return RawObjectView(
            rgb=rgb,
            depthmap=depthmap,
            object_mask=mask,
            camera_intrinsics=record["K"].copy(),
            T_C_O=record["T_C_O"].astype(np.float32),
            camera_pose=record["camera_pose"].astype(np.float32),
            record=record,
        )

    def assemble_processed_object_view(self, state, request):
        """Assemble the historical LMGeo schema from a processed object view."""

        record = state["record"]
        view_role = request.view_role
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

        rgb = state["rgb"]
        depthmap = state["depthmap"]
        object_mask = state["object_mask"]
        visibility_condition = state["visibility_mask_condition"]
        visibility_known_map = state["visibility_mask_known"]
        camera_pose = state["camera_pose"]
        T_C_O = state["T_C_O"]
        intrinsics = state["camera_intrinsics"]

        view = {
            "img": rgb,
            "depthmap": depthmap.astype(np.float32),
            "object_visibility_mask": object_mask.astype(np.float32),
            "visibility_mask_condition": visibility_condition.astype(np.float32),
            "visibility_mask_known": visibility_known_map.astype(np.float32),
            "visibility_mask_condition_applied": bool(
                state.get("visibility_mask_condition_applied", False)
            ),
            "camera_pose": camera_pose.astype(np.float32),
            "T_C_O": T_C_O.astype(np.float32),
            "camera_intrinsics": intrinsics.astype(np.float32),
            "dataset": self.dataset_label,
            "object_id": np.int64(object_id),
            "object_model_namespace": "lmo",
            "object_model_available": True,
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
        view.update(state["transform_meta"])
        return view

    def _load_view(self, record, rgb_masking, view_role):
        request = PlannedObjectView(
            record=record,
            view_role=str(view_role),
            rgb_masking=bool(rgb_masking),
            reference_source=record.get("reference_source"),
        )
        return self.object_view_processor.process(self, request, rng=self._rng)

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

        plan_metadata = {
            "object_id": self.object_id,
            "reference": [(r["split"], r["im_id"], r["gt_id"]) for r in reference_records],
            "query": [(r["split"], r["im_id"], r["gt_id"]) for r in query_records],
        }
        plan = self.key_query_sampling_policy.plan(
            reference_records=reference_records,
            query_records=query_records,
            reference_rgb_masking=self.reference_rgb_masking,
            query_rgb_masking=self.query_rgb_masking,
            metadata=plan_metadata,
        )
        return self._materialize_sample_plan(plan, rng=rng)


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
        visibility_mask_conditioning=False,
        condition_reference_visibility=True,
        condition_query_visibility=False,
        visibility_condition_corruption="none",
        visibility_condition_shift_fraction=0.5,
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
        ObjectDatasetAdapter.__init__(self, **kwargs)

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
        self.visibility_mask_conditioning = bool(visibility_mask_conditioning)
        self.condition_reference_visibility = bool(condition_reference_visibility)
        self.condition_query_visibility = bool(condition_query_visibility)
        self.visibility_condition_corruption = str(visibility_condition_corruption)
        self.visibility_condition_shift_fraction = float(visibility_condition_shift_fraction)
        self._configure_role_photometric_augmentation(
            enabled=photometric_augmentation,
            brightness=photometric_brightness,
            contrast=photometric_contrast,
            saturation=photometric_saturation,
            hue=photometric_hue,
            gamma=photometric_gamma,
            jpeg_prob=photometric_jpeg_prob,
            jpeg_quality=photometric_jpeg_quality,
            blur_prob=photometric_blur_prob,
            blur_resize_ratio=photometric_blur_resize_ratio,
        )
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
        if self.visibility_condition_corruption not in {"none", "shift"}:
            raise ValueError(
                "visibility_condition_corruption must be 'none' or 'shift', "
                f"got {visibility_condition_corruption!r}"
            )
        if self.visibility_condition_shift_fraction < 0.0:
            raise ValueError("visibility_condition_shift_fraction must be non-negative")
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

    def supported_frame_counts(self, image_num_range):
        lo, hi = [int(value) for value in image_num_range]
        query_cost = (
            2
            if bool(getattr(self, "query_recenter_include_original_query_view", False))
            else 1
        )
        possible = {
            reference_count + query_cost * query_count
            for reference_count in range(
                self.num_reference_range[0], self.num_reference_range[1] + 1
            )
            for query_count in range(
                self.num_query_range[0], self.num_query_range[1] + 1
            )
        }
        return sorted(count for count in possible if lo <= count <= hi)

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

        return self._build_samples_from_query_windows(self._candidate_windows(query_scene_ids, query_subscene_ids))

    def _build_samples_from_query_windows(self, windows, *, apply_limits=True):
        samples = []
        for window in windows:
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
        if apply_limits and self.max_query_subsequences is not None:
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
        # BOP targets revisit the same test scenes many times. Resolve JSON,
        # image dimensions, and the RGB extension once per scene instead of
        # reopening one RGB image for every object target (particularly costly
        # when LM-O resides on network storage).
        scene_cache = {}

        for target_index, target in enumerate(targets):
            object_id = int(target["obj_id"])
            if object_id not in object_filter:
                continue

            scene_id = int(target["scene_id"])
            im_id = int(target["im_id"])
            instance_count = int(target.get("inst_count", 1))
            scene_dir = self.data_root / self.query_split / f"{scene_id:06d}"
            cached_scene = scene_cache.get(scene_id)
            if cached_scene is None:
                scene_gt = self._load_json(scene_dir / "scene_gt.json")
                scene_gt_info = self._load_json(scene_dir / "scene_gt_info.json")
                scene_camera = self._load_json(scene_dir / "scene_camera.json")
                image_size = self._first_scene_image_size(scene_dir, scene_gt)
                rgb_ext = self._scene_rgb_ext(scene_dir)
                cached_scene = (
                    scene_gt,
                    scene_gt_info,
                    scene_camera,
                    image_size,
                    rgb_ext,
                )
                scene_cache[scene_id] = cached_scene
            scene_gt, scene_gt_info, scene_camera, image_size, rgb_ext = cached_scene
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
                    image_size=image_size,
                    rgb_ext=rgb_ext,
                    trust_paths=True,
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
        return self.key_query_sampling_policy.sample_counts(
            total_frames,
            reference_range=self.num_reference_range,
            query_range=self.num_query_range,
            reference_available=len(reference_records),
            query_available=len(query_records),
            query_view_cost=1,
            allow_repeat=self.allow_repeat,
            rng=rng,
        )

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

        plan_metadata = {
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
        query_view_cost = (
            2
            if bool(getattr(self, "query_recenter_include_original_query_view", False))
            else 1
        )
        plan = self.key_query_sampling_policy.plan(
            reference_records=reference_records,
            query_records=query_records,
            reference_rgb_masking=self.reference_rgb_masking,
            query_rgb_masking=self.query_rgb_masking,
            query_view_cost=query_view_cost,
            metadata=plan_metadata,
        )
        return self._materialize_sample_plan(plan, rng=rng)


class LMGeoGeometryMatchedSequenceDataset(LMGeoSequenceDataset):
    """Opt-in LM-O render-to-scene evaluation with GSO geometry constraints.

    The legacy :class:`LMGeoSequenceDataset` selection path is intentionally
    unchanged.  This subclass consumes immutable derived geometry sidecars,
    filters the ordinary LMGeo query-window manifest to constructible samples,
    and attaches the same ``planned_object_crop`` dictionaries used by the GSO
    geometry protocol.

    Reference plans are computed only from the LM-O ``train`` reference bank.
    Query GT is used solely to enforce the configured positive-view condition;
    it is never used by the reference-only Sim(3) fit.  Consequently this is an
    oracle "adequate keyframes supplied" evaluation, not a deployable retrieval
    protocol.
    """

    def _build_reference_records(self, object_id=None):
        """Index a BOP reference scene without reopening every RGB image.

        ``LMGeoSequenceDataset`` deliberately keeps its historical conservative
        path checks.  The matched protocol has a fully validated BOP bank and
        10,504 reference views, so resolving the common size/extension once per
        object avoids thousands of unnecessary network reads while preserving
        the same returned record schema.
        """

        object_id = self.object_id if object_id is None else int(object_id)
        scene_dir = self.data_root / self.reference_split / f"{object_id:06d}"
        scene_gt = self._load_json(scene_dir / "scene_gt.json")
        scene_gt_info = self._load_json(scene_dir / "scene_gt_info.json")
        scene_camera = self._load_json(scene_dir / "scene_camera.json")
        image_size = self._first_scene_image_size(scene_dir, scene_gt)
        rgb_ext = self._scene_rgb_ext(scene_dir)
        records = []
        for key in sorted(scene_gt, key=lambda value: int(value)):
            im_id = int(key)
            for gt_id, gt in enumerate(scene_gt[key]):
                if int(gt.get("obj_id", -1)) != object_id:
                    continue
                info = scene_gt_info[key][gt_id]
                if float(info.get("visib_fract", 1.0)) < self.visibility_min:
                    continue
                records.append(
                    self._record_from_gt(
                        scene_dir,
                        im_id,
                        gt_id,
                        gt,
                        info,
                        scene_camera[key],
                        self.reference_split,
                        object_id=object_id,
                        image_size=image_size,
                        rgb_ext=rgb_ext,
                        trust_paths=True,
                    )
                )
                break
        if not records:
            raise ValueError(
                f"No LM-O reference records for object {object_id} in {scene_dir}"
            )
        return self._filter_records_visible_after_center_crop(records, "reference")

    def __init__(
        self,
        *args,
        reference_geometry_plan_path,
        query_geometry_index_path,
        positive_angle_degrees=10.0,
        focal_relative_tolerance=0.1,
        crop_aspect=4.0 / 3.0,
        crop_margin_fraction=0.05,
        crop_center_jitter=0.0,
        plan_selection="first",
        **kwargs,
    ):
        self.reference_geometry_plan_path = Path(
            reference_geometry_plan_path
        ).expanduser().resolve()
        self.query_geometry_index_path = Path(
            query_geometry_index_path
        ).expanduser().resolve()
        self.positive_angle_degrees = float(positive_angle_degrees)
        self.focal_relative_tolerance = float(focal_relative_tolerance)
        self.geometry_crop_aspect = float(crop_aspect)
        self.geometry_crop_margin_fraction = float(crop_margin_fraction)
        self.geometry_crop_center_jitter = float(crop_center_jitter)
        self.geometry_plan_selection = str(plan_selection)

        if not 0.0 < self.positive_angle_degrees <= 180.0:
            raise ValueError("positive_angle_degrees must be in (0, 180]")
        if not 0.0 <= self.focal_relative_tolerance < 1.0:
            raise ValueError("focal_relative_tolerance must be in [0, 1)")
        if self.geometry_crop_aspect <= 0.0:
            raise ValueError("crop_aspect must be positive")
        if self.geometry_crop_margin_fraction < 0.0:
            raise ValueError("crop_margin_fraction must be non-negative")
        if not 0.0 <= self.geometry_crop_center_jitter <= 1.0:
            raise ValueError("crop_center_jitter must be in [0, 1]")
        if self.geometry_plan_selection not in {"first", "closest"}:
            raise ValueError("plan_selection must be 'first' or 'closest'")

        super().__init__(*args, **kwargs)
        if self.num_reference_range != (5, 5) or self.num_query_range != (1, 1):
            raise ValueError(
                "LMGeoGeometryMatchedSequenceDataset currently implements fixed N=5/K=1"
            )
        if self.query_selection != "first":
            raise ValueError(
                "Matched validation requires query_selection='first' so the query "
                "manifest is deterministic"
            )
        if self.context_reference_fraction != 0.0:
            raise ValueError("Matched render-to-scene evaluation cannot mix context references")
        if self.aug_crop or self.aug_focal:
            raise ValueError(
                "Planned calibrated crops require legacy aug_crop/aug_focal to be false"
            )

        self.reference_geometry_plans = GeometryPlanIndex(
            self.reference_geometry_plan_path
        )
        if self.reference_geometry_plans.reference_source_kind != "render":
            raise ValueError(
                "LM-O matched render-to-scene evaluation requires render reference plans"
            )
        if self.reference_geometry_plans.reference_count != 5:
            raise ValueError(
                "Reference plan catalogue must contain fixed five-view plans"
            )
        constraints = self.reference_geometry_plans.constraints
        if float(constraints.get("reference_visibility_min", 0.0)) < 0.3:
            raise ValueError("Reference plans must enforce visibility >= 0.3")
        if float(constraints.get("union_surface_min", 0.0)) < 0.5:
            raise ValueError("Reference plans must enforce union surface coverage >= 0.5")
        self.query_geometry = GeometryFeatureIndex(
            self.query_geometry_index_path, source_kind="scene"
        )
        self._geometry_plan_vector_cache = {}

        original_samples = list(self.samples)
        matched_samples, rejection_counts = self._build_geometry_matched_samples(
            original_samples
        )
        if not matched_samples:
            raise ValueError(
                "No LM-O new_val query has a compatible five-reference geometry plan"
            )
        self.samples = matched_samples
        self.dataset_label = "LMGeoGeometryMatched"
        self.geometry_eligibility = {
            "candidate_samples": len(original_samples),
            "eligible_samples": len(matched_samples),
            "eligible_fraction": len(matched_samples) / len(original_samples),
            **rejection_counts,
        }
        print(
            f"[{self.dataset_label}] geometry eligibility: "
            f"{len(matched_samples)}/{len(original_samples)} "
            f"({100.0 * self.geometry_eligibility['eligible_fraction']:.2f}%), "
            f"rejections={rejection_counts}"
        )

    def _compatible_plan_matches(self, query_feature, plans):
        if not int(query_feature["crop_feasible"]):
            return [], "query_crop"
        direction = np.asarray(
            [
                query_feature["view_x"],
                query_feature["view_y"],
                query_feature["view_z"],
            ],
            dtype=np.float32,
        )
        cosine = float(np.cos(np.deg2rad(self.positive_angle_degrees)))
        object_id = int(plans[0].object_id)
        cache = getattr(self, "_geometry_plan_vector_cache", None)
        if cache is None:
            cache = self._geometry_plan_vector_cache = {}
        arrays = cache.get(object_id)
        if arrays is None or arrays[0] is not plans:
            arrays = (
                plans,
                np.asarray([plan.focal_low for plan in plans], dtype=np.float64),
                np.asarray([plan.focal_high for plan in plans], dtype=np.float64),
                np.asarray(
                    [plan.focal_shape_low for plan in plans], dtype=np.float64
                ),
                np.asarray(
                    [plan.focal_shape_high for plan in plans], dtype=np.float64
                ),
                np.stack([plan.directions for plan in plans], axis=0),
            )
            cache[object_id] = arrays
        _, focal_lows, focal_highs, shape_lows, shape_highs, directions = arrays
        query_low = normalized_focal_scalar(query_feature)
        query_high = min(
            normalized_focal_scalar(query_feature, maximum=True),
            query_low * (1.0 + self.focal_relative_tolerance),
        )
        lows = np.maximum(query_low, focal_lows)
        highs = np.minimum(query_high, focal_highs)
        shape = normalized_focal_shape(query_feature)
        focal_valid = (
            (lows <= highs + 1e-12)
            & (shape_lows <= shape)
            & (shape <= shape_highs)
        )
        similarities = np.einsum("pnc,c->pn", directions, direction)
        positive_indices = np.argmax(similarities, axis=1)
        best = similarities[np.arange(len(plans)), positive_indices]
        valid = focal_valid & (best + 1e-7 >= cosine)
        matches = []
        for index in np.flatnonzero(valid):
            similarity = float(best[index])
            matches.append(
                (
                    plans[int(index)],
                    (float(lows[index]), float(highs[index])),
                    int(positive_indices[index]),
                    float(
                        np.rad2deg(
                            np.arccos(np.clip(similarity, -1.0, 1.0))
                        )
                    ),
                )
            )
        if matches:
            if self.geometry_plan_selection == "closest":
                matches.sort(key=lambda value: (value[3], value[0].plan_id))
            return matches, None
        return [], "positive_angle" if bool(focal_valid.any()) else "focal"

    def _build_geometry_matched_samples(self, samples):
        matched = []
        rejections = {
            "missing_query_geometry": 0,
            "missing_reference_plans": 0,
            "query_crop": 0,
            "focal": 0,
            "positive_angle": 0,
        }
        for sample in samples:
            object_id = int(sample["object_id"])
            query_record = sample["query_records"][0]
            query_feature = self.query_geometry.feature_for_record(query_record)
            if query_feature is None:
                rejections["missing_query_geometry"] += 1
                continue
            plans = self.reference_geometry_plans.plans_for_object(object_id)
            if not plans:
                rejections["missing_reference_plans"] += 1
                continue
            matches, reason = self._compatible_plan_matches(query_feature, plans)
            if not matches:
                rejections[str(reason)] += 1
                continue
            plan, interval, positive_index, positive_angle = matches[0]
            item = dict(sample)
            item["geometry_query_record"] = query_record
            item["geometry_query_feature"] = query_feature
            item["geometry_reference_plan"] = plan
            item["geometry_target_interval"] = interval
            item["geometry_positive_index"] = positive_index
            item["geometry_positive_angle_degrees"] = positive_angle
            matched.append(item)
        return matched, rejections

    def _planned_reference_records(self, sample, target, rng):
        object_id = int(sample["object_id"])
        plan = sample["geometry_reference_plan"]
        by_view = {
            int(record["view_id"]): record
            for record in self.reference_records_by_object[object_id]
        }
        try:
            records = [by_view[int(view_id)] for view_id in plan.view_ids]
        except KeyError as exc:
            raise ValueError(
                f"Reference plan {plan.plan_id} refers to missing LM-O view {exc}"
            ) from exc
        feature_map = self.reference_geometry_plans.features_by_ids(
            plan.feature_ids
        )
        output = []
        for record, feature_id in zip(records, plan.feature_ids):
            item = dict(record)
            item["reference_source"] = "render"
            item["planned_object_crop"] = object_preserving_crop_spec(
                feature_map[int(feature_id)],
                target_normalized_focal=target,
                aspect=self.geometry_crop_aspect,
                margin_fraction=self.geometry_crop_margin_fraction,
                center_jitter=self.geometry_crop_center_jitter,
                rng=rng,
            )
            output.append(item)
        return output

    def _get_views(self, index, resolution, rng):
        if self.frame_num != 6:
            raise ValueError(
                "LMGeoGeometryMatchedSequenceDataset requires exactly six model views"
            )
        sample = self.samples[int(index) % len(self.samples)]
        low, high = (float(value) for value in sample["geometry_target_interval"])
        target = 0.5 * (low + high)
        self._current_resolution = resolution
        self._prepare_sample_photometric_augmentation(rng)
        reference_records = self._planned_reference_records(sample, target, rng)
        query_record = dict(sample["geometry_query_record"])
        query_record["planned_object_crop"] = object_preserving_crop_spec(
            sample["geometry_query_feature"],
            target_normalized_focal=target,
            aspect=self.geometry_crop_aspect,
            margin_fraction=self.geometry_crop_margin_fraction,
            center_jitter=self.geometry_crop_center_jitter,
            rng=rng,
        )
        plan = sample["geometry_reference_plan"]
        metadata = {
            "sampling_policy": "lmgeo_geometry_render_to_scene",
            "object_id": int(sample["object_id"]),
            "query_scene_id": int(sample["query_scene_id"]),
            "query_subscene_id": int(sample["query_subscene_id"]),
            "reference_count": 5,
            "query_count": 1,
            "reference_plan_id": int(plan.plan_id),
            "reference_union_coverage": float(plan.union_coverage),
            "positive_reference_index": int(sample["geometry_positive_index"]),
            "positive_angle_degrees": float(
                sample["geometry_positive_angle_degrees"]
            ),
            "target_normalized_focal": float(target),
            "per_view_target_normalized_focal": (target,) * 6,
            "reference": [
                (record["split"], record["im_id"], record["gt_id"])
                for record in reference_records
            ],
            "query": [
                (query_record["split"], query_record["im_id"], query_record["gt_id"])
            ],
        }
        sample_plan = self.key_query_sampling_policy.plan(
            reference_records=reference_records,
            query_records=[query_record],
            reference_rgb_masking=self.reference_rgb_masking,
            query_rgb_masking=self.query_rgb_masking,
            reference_treatments=[
                ViewTreatment(
                    rgb="object_only", depth="object_only", mask_condition="none"
                )
                for _ in reference_records
            ],
            query_treatments=[
                ViewTreatment(
                    rgb="object_only", depth="object_only", mask_condition="none"
                )
            ],
            metadata=metadata,
        )
        return self._materialize_sample_plan(sample_plan, rng=rng)

    def _materialize_sample_plan(self, plan, *, rng):
        """Materialize fixed one-cost views without legacy query expansion.

        Historical LMGeo query subclasses can expand one physical query into
        paired model views.  This protocol is fixed K=1 and must retain its
        explicit object-only treatment through the final resize, so every
        request is processed directly.
        """

        self.this_views_info = dict(plan.metadata)
        output = []
        for request in plan.views:
            if int(request.model_view_cost) != 1:
                raise ValueError("Matched LM-O views must each have model_view_cost=1")
            view = self.object_view_processor.process(self, request, rng=rng)
            if request.view_role == "query":
                view["query_pair_index"] = np.int64(request.query_pair_index)
            view.update(request.view_updates)
            output.append(view)
        if len(output) != plan.model_view_count:
            raise RuntimeError("Matched LM-O plan materialized the wrong view count")
        return output

    def close(self):
        """Release process-local SQLite handles owned by the opt-in sidecars."""

        plans = getattr(self, "reference_geometry_plans", None)
        if plans is not None:
            plans.close()
        queries = getattr(self, "query_geometry", None)
        if queries is not None:
            queries.close()


class LMGeoAnchorScenePairSequenceDataset(LMGeoSequenceDataset):
    """Same-object scene-pair dataset with reference-only mask conditioning.

    A sample chooses one anchor object, references from one scene/window where
    that object is visible, and queries from another selected scene/window where
    the same object is visible.  The two windows may be identical when
    configured, but selected query records are always removed from the reference
    candidate pool.  All poses remain in the anchor object's CAD frame, so the
    existing Pi3 loss and reference-only pose alignment metrics keep their
    object-centric meaning.
    """

    def __init__(
        self,
        *args,
        anchor_reference_windows=None,
        anchor_reference_query_split=None,
        anchor_allow_same_scene=True,
        anchor_allow_same_subscene=True,
        anchor_selection_attempts=50,
        **kwargs,
    ):
        kwargs.setdefault("query_source", "windows")
        kwargs.setdefault("visibility_mask_conditioning", True)
        kwargs.setdefault("condition_reference_visibility", True)
        kwargs.setdefault("condition_query_visibility", False)
        super().__init__(*args, **kwargs)

        self.dataset_label = "LMGeoAnchorScenePair"
        self.anchor_reference_windows = (
            None
            if anchor_reference_windows is None
            else self._normalize_query_windows(anchor_reference_windows)
        )
        self.anchor_reference_query_split = (
            self.query_split
            if anchor_reference_query_split is None
            else str(anchor_reference_query_split)
        )
        self.anchor_allow_same_scene = bool(anchor_allow_same_scene)
        self.anchor_allow_same_subscene = bool(anchor_allow_same_subscene)
        self.anchor_selection_attempts = max(1, int(anchor_selection_attempts))
        self.anchor_reference_samples = self._build_anchor_reference_samples()
        self.anchor_reference_samples_by_object = self._build_anchor_reference_sample_index()

        candidate_count = sum(
            len(samples)
            for samples in self.anchor_reference_samples_by_object.values()
        )
        print(
            f"[{self.dataset_label}] query windows={len(self.samples)}, "
            f"anchor reference windows={candidate_count}, "
            f"allow_same_scene={self.anchor_allow_same_scene}, "
            f"allow_same_subscene={self.anchor_allow_same_subscene}, "
            f"condition_reference_visibility={self.condition_reference_visibility}, "
            f"condition_query_visibility={self.condition_query_visibility}, "
            f"depth_masking={self.depth_masking}"
        )

    def _build_reference_records(self, object_id=None):
        # This dataset uses scene frames as references.  Avoid walking the
        # render split and keep render-reference experiments fully separate.
        return []

    def _build_anchor_reference_samples(self):
        if self.anchor_reference_windows is None:
            return list(self.samples)

        original_query_split = self.query_split
        try:
            self.query_split = self.anchor_reference_query_split
            return self._build_samples_from_query_windows(
                self.anchor_reference_windows,
                apply_limits=False,
            )
        finally:
            self.query_split = original_query_split

    def _build_anchor_reference_sample_index(self):
        samples_by_object = {}
        for sample in self.anchor_reference_samples:
            samples_by_object.setdefault(int(sample["object_id"]), []).append(sample)
        return samples_by_object

    @staticmethod
    def _sample_scene_key(sample):
        return int(sample.get("query_scene_id", -1))

    @staticmethod
    def _sample_subscene_key(sample):
        return (
            int(sample.get("query_scene_id", -1)),
            int(sample.get("query_subscene_id", -1)),
        )

    @staticmethod
    def _record_identity(record):
        return (
            str(record.get("split", "")),
            str(record.get("scene_dir", "")),
            int(record.get("im_id", -1)),
            int(record.get("gt_id", -1)),
            int(record.get("object_id", -1)),
        )

    def _anchor_reference_candidates(self, query_sample):
        object_id = int(query_sample["object_id"])
        candidates = []
        for sample in self.anchor_reference_samples_by_object.get(object_id, []):
            if (
                not self.anchor_allow_same_scene
                and self._sample_scene_key(sample) == self._sample_scene_key(query_sample)
            ):
                continue
            if (
                not self.anchor_allow_same_subscene
                and self._sample_subscene_key(sample) == self._sample_subscene_key(query_sample)
            ):
                continue
            candidates.append(sample)
        return candidates

    def _tag_anchor_reference(self, record, reference_sample):
        item = dict(record)
        item["reference_source"] = "anchor_scene"
        item["anchor_reference_scene_id"] = int(reference_sample.get("query_scene_id", -1))
        item["anchor_reference_subscene_id"] = int(reference_sample.get("query_subscene_id", -1))
        return item

    def _load_anchor_reference_view(self, record, reference_sample):
        view = self._load_view(
            record,
            rgb_masking=self.reference_rgb_masking,
            view_role="reference",
        )
        view["reference_source"] = "anchor_scene"
        view["is_anchor_scene_reference"] = True
        view["scene_id"] = np.int64(reference_sample.get("query_scene_id", -1))
        view["source_scene_id"] = np.int64(record.get("query_scene_id", -1))
        view["source_subscene_id"] = np.int64(record.get("query_subscene_id", -1))
        return view

    def _select_anchor_record_sets(self, query_sample, total_frames, rng):
        candidates = self._anchor_reference_candidates(query_sample)
        if not candidates:
            raise ValueError(
                f"No same-object anchor reference candidates for object "
                f"{int(query_sample['object_id'])}"
            )

        errors = []
        query_pool = list(query_sample["query_records"])
        for _ in range(self.anchor_selection_attempts):
            reference_sample = candidates[int(rng.integers(0, len(candidates)))]
            reference_pool = list(reference_sample["query_records"])
            try:
                ref_count, query_count = self._sample_counts(
                    total_frames,
                    reference_pool,
                    query_pool,
                    rng,
                )
                query_records = self._select_records(
                    query_pool,
                    query_count,
                    self.query_selection,
                    rng=rng,
                )
                query_identities = {self._record_identity(record) for record in query_records}
                reference_pool = [
                    record
                    for record in reference_pool
                    if self._record_identity(record) not in query_identities
                ]
                if not reference_pool:
                    errors.append("reference pool empty after removing selected queries")
                    continue
                if len(reference_pool) < ref_count and not self.allow_repeat:
                    errors.append(
                        f"only {len(reference_pool)} reference records remain for ref_count={ref_count}"
                    )
                    continue
                reference_records = self._select_records(
                    reference_pool,
                    ref_count,
                    self.reference_selection,
                    rng=rng,
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue

            reference_identities = {self._record_identity(record) for record in reference_records}
            if reference_identities & query_identities:
                errors.append("query/reference identity overlap")
                continue
            return reference_sample, reference_records, query_records, ref_count, query_count

        joined = "; ".join(errors[-3:]) if errors else "no detailed errors"
        raise ValueError(
            f"Could not select leak-free anchor record sets after "
            f"{self.anchor_selection_attempts} attempts: {joined}"
        )

    def _get_views(self, index, resolution, rng):
        query_sample = self.samples[int(index) % len(self.samples)]
        object_id = int(query_sample["object_id"])
        (
            reference_sample,
            reference_records,
            query_records,
            ref_count,
            query_count,
        ) = self._select_anchor_record_sets(query_sample, self.frame_num, rng)

        self._current_resolution = resolution
        self._prepare_sample_photometric_augmentation(rng)
        reference_records = [
            self._tag_anchor_reference(record, reference_sample)
            for record in reference_records
        ]

        plan_metadata = {
            "object_id": object_id,
            "reference_scene_id": int(reference_sample.get("query_scene_id", -1)),
            "reference_subscene_id": int(reference_sample.get("query_subscene_id", -1)),
            "query_scene_id": int(query_sample.get("query_scene_id", -1)),
            "query_subscene_id": int(query_sample.get("query_subscene_id", -1)),
            "reference_count": ref_count,
            "query_count": query_count,
            "reference_visibility_conditioned": bool(self.condition_reference_visibility),
            "query_visibility_conditioned": bool(self.condition_query_visibility),
            "depth_masking": bool(self.depth_masking),
            "reference": [
                (
                    "anchor_scene",
                    r["split"],
                    int(r.get("query_scene_id", -1)),
                    int(r.get("query_subscene_id", -1)),
                    int(r["im_id"]),
                    int(r["gt_id"]),
                )
                for r in reference_records
            ],
            "query": [
                (
                    r["split"],
                    int(r.get("query_scene_id", -1)),
                    int(r.get("query_subscene_id", -1)),
                    int(r["im_id"]),
                    int(r["gt_id"]),
                )
                for r in query_records
            ],
        }
        reference_updates = [
            {
                "reference_source": "anchor_scene",
                "is_anchor_scene_reference": True,
                "scene_id": np.int64(reference_sample.get("query_scene_id", -1)),
                "source_scene_id": np.int64(record.get("query_scene_id", -1)),
                "source_subscene_id": np.int64(record.get("query_subscene_id", -1)),
            }
            for record in reference_records
        ]
        plan = self.key_query_sampling_policy.plan(
            reference_records=reference_records,
            query_records=query_records,
            reference_rgb_masking=self.reference_rgb_masking,
            query_rgb_masking=self.query_rgb_masking,
            metadata=plan_metadata,
            reference_updates=reference_updates,
        )
        return self._materialize_sample_plan(plan, rng=rng)


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

        plan_metadata = {
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
        reference_updates = [
            {
                "reference_source": "same_scene",
                "is_context_reference": False,
                "is_same_scene_reference": True,
                "scene_id": np.int64(sample["query_scene_id"]),
                "source_scene_id": np.int64(sample["query_scene_id"]),
                "source_subscene_id": np.int64(sample["query_subscene_id"]),
            }
            for _ in reference_records
        ]
        plan = self.key_query_sampling_policy.plan(
            reference_records=reference_records,
            query_records=[query_record],
            reference_rgb_masking=self.reference_rgb_masking,
            query_rgb_masking=self.query_rgb_masking,
            metadata=plan_metadata,
            reference_updates=reference_updates,
        )
        return self._materialize_sample_plan(plan, rng=rng)
