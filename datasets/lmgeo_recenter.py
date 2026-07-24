import cv2
import numpy as np
from PIL import Image

from datasets.lmgeo_dataset import LMGeoSequenceDataset


def bbox_center_xy(bbox):
    if bbox is None:
        raise ValueError("Cannot recenter query without a bbox")
    x, y, width, height = [float(value) for value in bbox]
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Invalid bbox for recentering: {bbox}")
    return np.array([x + 0.5 * width, y + 0.5 * height], dtype=np.float32)


def rotation_between_unit_vectors(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)

    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float32)

    if dot < -1.0 + 1e-8:
        axis = np.cross(source, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis /= np.linalg.norm(axis)
        return rodrigues(axis, np.pi)

    axis = np.cross(source, target)
    axis_norm = np.linalg.norm(axis)
    axis /= axis_norm
    angle = np.arctan2(axis_norm, dot)
    return rodrigues(axis, angle)


def rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R.astype(np.float32)


def rotation_to_optical_axis(intrinsics, center_xy):
    ray = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64)) @ np.array(
        [float(center_xy[0]), float(center_xy[1]), 1.0],
        dtype=np.float64,
    )
    ray /= np.linalg.norm(ray)
    return rotation_between_unit_vectors(ray, np.array([0.0, 0.0, 1.0], dtype=np.float64))


def bbox_zoom_factor(image_size, bbox, target_fraction=0.55, max_zoom=4.0, min_zoom=1.0):
    width, height = [float(v) for v in image_size]
    _, _, bbox_width, bbox_height = [float(value) for value in bbox]
    if bbox_width <= 0.0 or bbox_height <= 0.0:
        raise ValueError(f"Invalid bbox for zooming: {bbox}")

    target_fraction = float(target_fraction)
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError(f"target_fraction must be in (0, 1], got {target_fraction}")

    zoom = min(
        (width * target_fraction) / bbox_width,
        (height * target_fraction) / bbox_height,
    )
    return float(np.clip(zoom, float(min_zoom), float(max_zoom)))


def zoomed_intrinsics(intrinsics, zoom):
    K_zoom = np.asarray(intrinsics, dtype=np.float32).copy()
    K_zoom[0, 0] *= float(zoom)
    K_zoom[1, 1] *= float(zoom)
    return K_zoom


def compose_recentered_object_pose(T_C_O, R_old_to_new):
    T_C_O = np.asarray(T_C_O, dtype=np.float32)
    R_old_to_new = np.asarray(R_old_to_new, dtype=np.float32)
    T_Cnew_O = T_C_O.copy()
    T_Cnew_O[:3, :3] = R_old_to_new @ T_C_O[:3, :3]
    T_Cnew_O[:3, 3] = R_old_to_new @ T_C_O[:3, 3]
    return T_Cnew_O.astype(np.float32)


def corrected_source_z_depth(depthmap, intrinsics, R_old_to_new, eps=1e-6):
    depthmap = np.asarray(depthmap, dtype=np.float32)
    K_inv = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64))
    R_old_to_new = np.asarray(R_old_to_new, dtype=np.float64)

    height, width = depthmap.shape
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0).reshape(3, -1)
    rays = K_inv @ pixels
    z_scale = (R_old_to_new[2:3, :] @ rays).reshape(height, width).astype(np.float32)
    corrected = depthmap * z_scale
    valid = np.isfinite(corrected) & np.isfinite(depthmap) & (depthmap > 0.0) & (z_scale > float(eps))
    corrected = corrected.astype(np.float32, copy=False)
    corrected[~valid] = 0.0
    return corrected, valid


def homography_for_recenter_zoom(intrinsics, R_old_to_new, zoom):
    K = np.asarray(intrinsics, dtype=np.float64)
    K_zoom = zoomed_intrinsics(K, zoom).astype(np.float64)
    H = K_zoom @ np.asarray(R_old_to_new, dtype=np.float64) @ np.linalg.inv(K)
    H /= H[2, 2]
    return H.astype(np.float32), K_zoom.astype(np.float32)


def warp_recenter_zoom_view(
    *,
    rgb,
    depthmap,
    mask,
    intrinsics,
    T_C_O,
    bbox,
    zoom_target_fraction=0.55,
    zoom_max=4.0,
    zoom_min=1.0,
    depth_interpolation="nearest",
    min_valid_depth_pixels=1,
):
    height, width = depthmap.shape
    center = bbox_center_xy(bbox)
    R_old_to_new = rotation_to_optical_axis(intrinsics, center)
    zoom = bbox_zoom_factor(
        (width, height),
        bbox,
        target_fraction=zoom_target_fraction,
        max_zoom=zoom_max,
        min_zoom=zoom_min,
    )
    H, K_new = homography_for_recenter_zoom(intrinsics, R_old_to_new, zoom)

    corrected_depth, corrected_valid = corrected_source_z_depth(depthmap, intrinsics, R_old_to_new)
    source_valid = corrected_valid
    source_in_bounds = np.ones((height, width), dtype=np.uint8)

    if depth_interpolation == "nearest":
        depth_flags = cv2.INTER_NEAREST
    elif depth_interpolation == "linear":
        depth_flags = cv2.INTER_LINEAR
    else:
        raise ValueError("depth_interpolation must be 'nearest' or 'linear'")

    rgb_warped = cv2.warpPerspective(
        np.asarray(rgb),
        H,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    depth_warped = cv2.warpPerspective(
        corrected_depth,
        H,
        (width, height),
        flags=depth_flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_warped = cv2.warpPerspective(
        np.asarray(mask, dtype=np.uint8),
        H,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    valid_warped = cv2.warpPerspective(
        source_valid.astype(np.uint8),
        H,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    in_bounds_warped = cv2.warpPerspective(
        source_in_bounds,
        H,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

    valid_warped &= in_bounds_warped
    depth_warped = depth_warped.astype(np.float32, copy=False)
    depth_warped[~valid_warped] = 0.0
    valid_depth_pixels = int(valid_warped.sum())
    if valid_depth_pixels < int(min_valid_depth_pixels):
        raise ValueError(
            f"recenter/zoom query has only {valid_depth_pixels} valid depth pixels "
            f"after warping, required {min_valid_depth_pixels}"
        )

    T_Cnew_O = compose_recentered_object_pose(T_C_O, R_old_to_new)
    camera_pose = np.linalg.inv(T_Cnew_O).astype(np.float32)

    meta = {
        "query_recenter_applied": True,
        "query_recenter_zoom": np.float32(zoom),
        "query_recenter_valid_depth_pixels_raw": np.int64(valid_depth_pixels),
        "query_recenter_valid_fraction_raw": np.float32(valid_depth_pixels / float(height * width)),
        "query_recenter_R_old_to_new": R_old_to_new.astype(np.float32),
        "query_recenter_H_raw": H.astype(np.float32),
    }
    return (
        rgb_warped,
        depth_warped,
        mask_warped,
        K_new,
        T_Cnew_O,
        camera_pose,
        meta,
    )


class LMGeoQueryRecenterZoomMixin:
    """Query-only oracle recenter + zoom transform for LMGeo diagnostics."""

    def __init__(
        self,
        *args,
        query_recenter_bbox_key="bbox_obj",
        query_recenter_zoom_target_fraction=0.55,
        query_recenter_zoom_max=4.0,
        query_recenter_zoom_min=1.0,
        query_recenter_depth_interpolation="nearest",
        query_recenter_min_valid_depth_pixels=1,
        query_recenter_include_original_query_view=False,
        **kwargs,
    ):
        self.query_recenter_bbox_key = str(query_recenter_bbox_key)
        self.query_recenter_zoom_target_fraction = float(query_recenter_zoom_target_fraction)
        self.query_recenter_zoom_max = float(query_recenter_zoom_max)
        self.query_recenter_zoom_min = float(query_recenter_zoom_min)
        self.query_recenter_depth_interpolation = str(query_recenter_depth_interpolation)
        self.query_recenter_min_valid_depth_pixels = int(query_recenter_min_valid_depth_pixels)
        self.query_recenter_include_original_query_view = bool(
            query_recenter_include_original_query_view
        )
        super().__init__(*args, **kwargs)

    def _bbox_for_recenter(self, record):
        bbox = record.get(self.query_recenter_bbox_key)
        if bbox is None and self.query_recenter_bbox_key != "bbox_visib":
            bbox = record.get("bbox_visib")
        if bbox is None and self.query_recenter_bbox_key != "bbox_obj":
            bbox = record.get("bbox_obj")
        return bbox

    def _should_depth_mask_view(self, *, view_role, reference_source):
        if view_role == "query":
            return False
        return super()._should_depth_mask_view(view_role=view_role, reference_source=reference_source)

    def _record_passes_center_crop(self, record):
        return True

    def _record_has_preprocessed_depth(self, record):
        resolution = self._resolutions[0]
        depthmap = self._read_depth_meters(record["depth_path"], record["depth_scale"])
        mask = self._read_mask(record["mask_path"], depthmap.shape)
        with Image.open(record["rgb_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))

        try:
            rgb, depthmap, _, intrinsics, _, _, _ = self._maybe_transform_raw_view(
                record=record,
                rgb=rgb,
                depthmap=depthmap,
                mask=mask,
                intrinsics=record["K"].copy(),
                T_C_O=record["T_C_O"].astype(np.float32),
                camera_pose=record["camera_pose"].astype(np.float32),
                view_role="query",
            )
            _, depthmap, _ = self._crop_resize_if_necessary(
                rgb,
                depthmap,
                intrinsics,
                resolution,
                rng=self._rng,
                info=record["rgb_path"],
            )
        except Exception:
            return False

        return int((depthmap > 0).sum()) >= self.min_preprocessed_query_depth_pixels

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
        skip_recenter = bool(record.get("_lmgeo_skip_query_recenter", False))
        if view_role != "query" or skip_recenter:
            meta = {
                "query_recenter_applied": False,
                "query_recenter_zoom": np.float32(1.0),
                "query_recenter_valid_depth_pixels_raw": np.int64(0),
                "query_recenter_valid_fraction_raw": np.float32(0.0),
            }
            if getattr(
                self,
                "query_recenter_include_original_query_view",
                False,
            ):
                # unified_collate_fn takes its key set from the first
                # reference view, so paired-only matrices need shape-stable
                # identity defaults on every view in this opt-in regime.
                meta.update(
                    {
                        "query_recenter_R_old_to_new": np.eye(
                            3,
                            dtype=np.float32,
                        ),
                        "query_recenter_H_raw": np.eye(3, dtype=np.float32),
                        "query_crop_from_original_homography": np.eye(
                            3,
                            dtype=np.float32,
                        ),
                        "query_original_T_C_O": np.asarray(
                            T_C_O,
                            dtype=np.float32,
                        ).copy(),
                    }
                )
            return rgb, depthmap, mask, intrinsics, T_C_O, camera_pose, meta

        transformed = warp_recenter_zoom_view(
            rgb=rgb,
            depthmap=depthmap,
            mask=mask,
            intrinsics=intrinsics,
            T_C_O=T_C_O,
            bbox=self._bbox_for_recenter(record),
            zoom_target_fraction=self.query_recenter_zoom_target_fraction,
            zoom_max=self.query_recenter_zoom_max,
            zoom_min=self.query_recenter_zoom_min,
            depth_interpolation=self.query_recenter_depth_interpolation,
            min_valid_depth_pixels=self.query_recenter_min_valid_depth_pixels,
        )
        if not getattr(
            self,
            "query_recenter_include_original_query_view",
            False,
        ):
            return transformed

        rgb_out, depth_out, mask_out, K_out, T_out, pose_out, meta = transformed
        meta.update(
            {
                "query_crop_from_original_homography": np.eye(
                    3,
                    dtype=np.float32,
                ),
                "query_original_T_C_O": np.asarray(
                    T_C_O,
                    dtype=np.float32,
                ).copy(),
            }
        )
        return rgb_out, depth_out, mask_out, K_out, T_out, pose_out, meta

    def _sample_counts(self, total_frames, reference_records, query_records, rng):
        if not self.query_recenter_include_original_query_view:
            return super()._sample_counts(
                total_frames,
                reference_records,
                query_records,
                rng,
            )

        total_frames = int(total_frames)
        ref_min, ref_max = self.num_reference_range
        query_min, query_max = self.num_query_range
        if not self.allow_repeat:
            ref_max = min(ref_max, len(reference_records))
            query_max = min(query_max, len(query_records))

        valid_counts = [
            (ref_count, query_count)
            for ref_count in range(ref_min, ref_max + 1)
            for query_count in range(query_min, query_max + 1)
            if ref_count + 2 * query_count == total_frames
        ]
        if not valid_counts:
            raise ValueError(
                f"Cannot split model frame_num={total_frames} into reference "
                f"range {self.num_reference_range} and paired query-record range "
                f"{self.num_query_range}"
            )
        return valid_counts[int(rng.integers(len(valid_counts)))]

    def _load_query_views(self, record):
        cropped_views = super()._load_query_views(record)
        if not self.query_recenter_include_original_query_view:
            return cropped_views
        if len(cropped_views) != 1:
            raise RuntimeError(
                "LMGeo paired-query expansion expects one base view per query record"
            )

        cropped = cropped_views[0]
        original_record = dict(record)
        original_record["_lmgeo_skip_query_recenter"] = True
        original = self._load_view(
            original_record,
            rgb_masking=self.query_rgb_masking,
            view_role="query",
        )

        R_old_to_new = np.asarray(
            cropped["query_recenter_R_old_to_new"],
            dtype=np.float32,
        )
        K_cropped = np.asarray(cropped["camera_intrinsics"], dtype=np.float32)
        K_original = np.asarray(original["camera_intrinsics"], dtype=np.float32)
        H_final = K_cropped @ R_old_to_new @ np.linalg.inv(K_original)
        H_final = (H_final / H_final[2, 2]).astype(np.float32)

        cropped.update(
            {
                "is_cropped_query": True,
                "is_original_query": False,
                "is_query_context": False,
                "query_variant": "cropped",
                "query_crop_from_original_homography": H_final,
                "query_original_T_C_O": np.asarray(
                    original["T_C_O"],
                    dtype=np.float32,
                ).copy(),
            }
        )
        original.update(
            {
                "view_role": "query_context",
                "is_query": False,
                "is_query_context": True,
                "is_cropped_query": False,
                "is_original_query": True,
                "query_variant": "original",
                "query_recenter_R_old_to_new": R_old_to_new.copy(),
                "query_crop_from_original_homography": H_final.copy(),
                "query_original_T_C_O": np.asarray(
                    original["T_C_O"],
                    dtype=np.float32,
                ).copy(),
            }
        )
        return [cropped, original]


class LMGeoRecenterZoomSequenceDataset(LMGeoQueryRecenterZoomMixin, LMGeoSequenceDataset):
    pass
