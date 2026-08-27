"""Reusable virtual-camera rectification for calibrated object views.

The transform is a pure camera rotation followed by a focal change.  It keeps
the physical camera centre fixed, but changes the camera coordinate frame and
therefore must update ``T_C_O`` and z-depth together with the image and
intrinsics.  Dataset-specific sampling stays outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def bbox_center_xy(bbox: Sequence[float]) -> np.ndarray:
    if bbox is None:
        raise ValueError("Cannot recenter a view without a bbox")
    x, y, width, height = [float(value) for value in bbox]
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Invalid bbox for recentering: {bbox}")
    return np.array([x + 0.5 * width, y + 0.5 * height], dtype=np.float32)


def bbox_from_record(record: Mapping[str, Any], key: str = "bbox_obj") -> np.ndarray:
    """Read a BOP ``[x, y, width, height]`` box from common record layouts."""

    value = record.get(str(key))
    if isinstance(value, str):
        value = json.loads(value)
    if value is not None:
        bbox = np.asarray(value, dtype=np.float32).reshape(-1)
        if bbox.size != 4:
            raise ValueError(f"{key} must contain four values, got {value!r}")
        bbox_center_xy(bbox)
        return bbox

    names = tuple(f"{key}_{suffix}" for suffix in ("x", "y", "w", "h"))
    if all(name in record for name in names):
        bbox = np.asarray([record[name] for name in names], dtype=np.float32)
        bbox_center_xy(bbox)
        return bbox
    raise ValueError(f"Record has no {key!r} bbox")


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    rotation = (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )
    return rotation.astype(np.float32)


def rotation_between_unit_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
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
    return rodrigues(axis, np.arctan2(axis_norm, dot))


def rotation_to_optical_axis(
    intrinsics: np.ndarray,
    center_xy: Sequence[float],
) -> np.ndarray:
    ray = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64)) @ np.array(
        [float(center_xy[0]), float(center_xy[1]), 1.0],
        dtype=np.float64,
    )
    ray /= np.linalg.norm(ray)
    return rotation_between_unit_vectors(
        ray,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )


def bbox_zoom_factor(
    image_size: Sequence[float],
    bbox: Sequence[float],
    target_fraction: float = 0.55,
    max_zoom: float = 4.0,
    min_zoom: float = 1.0,
) -> float:
    """Historical bbox-size zoom used by the LMGeo diagnostic."""

    width, height = [float(value) for value in image_size]
    _, _, bbox_width, bbox_height = [float(value) for value in bbox]
    if bbox_width <= 0.0 or bbox_height <= 0.0:
        raise ValueError(f"Invalid bbox for zooming: {bbox}")
    if not 0.0 < float(target_fraction) <= 1.0:
        raise ValueError(
            f"target_fraction must be in (0, 1], got {target_fraction}"
        )
    zoom = min(
        width * float(target_fraction) / bbox_width,
        height * float(target_fraction) / bbox_height,
    )
    return float(np.clip(zoom, float(min_zoom), float(max_zoom)))


def zoomed_intrinsics(
    intrinsics: np.ndarray,
    zoom: float,
    *,
    image_size: Sequence[int] | None = None,
    principal_point: str = "preserve",
) -> np.ndarray:
    """Construct virtual intrinsics without accidentally scaling the pp."""

    if principal_point not in {"preserve", "image_center"}:
        raise ValueError("principal_point must be 'preserve' or 'image_center'")
    output = np.asarray(intrinsics, dtype=np.float32).copy()
    output[0, 0] *= float(zoom)
    output[0, 1] *= float(zoom)
    output[1, 1] *= float(zoom)
    if principal_point == "image_center":
        if image_size is None:
            raise ValueError("image_size is required for an image-centred pp")
        width, height = (int(value) for value in image_size)
        output[0, 2] = 0.5 * (width - 1)
        output[1, 2] = 0.5 * (height - 1)
    return output


def compose_recentered_object_pose(
    T_C_O: np.ndarray,
    R_old_to_new: np.ndarray,
) -> np.ndarray:
    T_C_O = np.asarray(T_C_O, dtype=np.float32)
    R_old_to_new = np.asarray(R_old_to_new, dtype=np.float32)
    T_Cnew_O = T_C_O.copy()
    T_Cnew_O[:3, :3] = R_old_to_new @ T_C_O[:3, :3]
    T_Cnew_O[:3, 3] = R_old_to_new @ T_C_O[:3, 3]
    return T_Cnew_O.astype(np.float32)


def corrected_source_z_depth(
    depthmap: np.ndarray,
    intrinsics: np.ndarray,
    R_old_to_new: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Express source z-depth along the virtual camera's z axis."""

    depthmap = np.asarray(depthmap, dtype=np.float32)
    K_inv = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64))
    R_old_to_new = np.asarray(R_old_to_new, dtype=np.float64)
    height, width = depthmap.shape
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0).reshape(3, -1)
    rays = K_inv @ pixels
    z_scale = (R_old_to_new[2:3, :] @ rays).reshape(height, width).astype(
        np.float32
    )
    corrected = depthmap * z_scale
    valid = (
        np.isfinite(corrected)
        & np.isfinite(depthmap)
        & (depthmap > 0.0)
        & (z_scale > float(eps))
    )
    corrected = corrected.astype(np.float32, copy=False)
    corrected[~valid] = 0.0
    return corrected, valid


def homography_for_recenter_zoom(
    intrinsics: np.ndarray,
    R_old_to_new: np.ndarray,
    zoom: float,
    *,
    image_size: Sequence[int] | None = None,
    principal_point: str = "preserve",
) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(intrinsics, dtype=np.float64)
    K_zoom = zoomed_intrinsics(
        K,
        zoom,
        image_size=image_size,
        principal_point=principal_point,
    ).astype(np.float64)
    homography = K_zoom @ np.asarray(R_old_to_new, dtype=np.float64) @ np.linalg.inv(K)
    homography /= homography[2, 2]
    return homography.astype(np.float32), K_zoom.astype(np.float32)


def _expanded_bbox_corners(
    bbox: Sequence[float],
    margin_fraction: float,
) -> np.ndarray:
    x, y, width, height = [float(value) for value in bbox]
    if width <= 0.0 or height <= 0.0 or margin_fraction < 0.0:
        raise ValueError("bbox size must be positive and margin non-negative")
    margin_x = float(margin_fraction) * width
    margin_y = float(margin_fraction) * height
    left, right = x - margin_x, x + width + margin_x
    top, bottom = y - margin_y, y + height + margin_y
    return np.asarray(
        [[left, top, 1.0], [right, top, 1.0], [right, bottom, 1.0], [left, bottom, 1.0]],
        dtype=np.float64,
    )


def object_preserving_zoom_limit(
    *,
    intrinsics: np.ndarray,
    R_old_to_new: np.ndarray,
    bbox: Sequence[float],
    image_size: Sequence[int],
    principal_point: str = "image_center",
    bbox_margin_fraction: float = 0.05,
) -> float:
    """Maximum focal multiplier keeping a warped amodal box in frame."""

    width, height = (int(value) for value in image_size)
    H_unit, K_unit = homography_for_recenter_zoom(
        intrinsics,
        R_old_to_new,
        1.0,
        image_size=(width, height),
        principal_point=principal_point,
    )
    corners = _expanded_bbox_corners(bbox, bbox_margin_fraction)
    warped = (np.asarray(H_unit, dtype=np.float64) @ corners.T).T
    if np.any(warped[:, 2] <= 1e-8):
        raise ValueError("Virtual camera puts part of the bbox behind the camera")
    warped = warped[:, :2] / warped[:, 2:3]
    cx, cy = float(K_unit[0, 2]), float(K_unit[1, 2])
    limits: list[float] = []
    for x, y in warped:
        dx, dy = float(x - cx), float(y - cy)
        if dx < -1e-9:
            limits.append(cx / -dx)
        elif dx > 1e-9:
            limits.append((width - 1.0 - cx) / dx)
        if dy < -1e-9:
            limits.append(cy / -dy)
        elif dy > 1e-9:
            limits.append((height - 1.0 - cy) / dy)
    safe = min(limits, default=float("inf"))
    if not np.isfinite(safe) or safe <= 0.0:
        raise ValueError(f"Invalid object-preserving virtual zoom limit: {safe}")
    return float(safe)


def warp_recenter_zoom_view(
    *,
    rgb: np.ndarray,
    depthmap: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    T_C_O: np.ndarray,
    bbox: Sequence[float],
    zoom_target_fraction: float = 0.55,
    zoom_max: float = 4.0,
    zoom_min: float = 1.0,
    depth_interpolation: str = "nearest",
    min_valid_depth_pixels: int = 1,
    zoom: float | None = None,
    safe_fill_fraction: float | None = None,
    principal_point: str = "preserve",
    safe_zoom: bool = False,
    bbox_margin_fraction: float = 0.0,
    replace_planned_crop: bool = False,
    include_virtual_metadata: bool = False,
):
    """Warp one calibrated view into a rotated/zoomed virtual camera."""

    height, width = np.asarray(depthmap).shape
    center = bbox_center_xy(bbox)
    R_old_to_new = rotation_to_optical_axis(intrinsics, center)
    requested_zoom = (
        bbox_zoom_factor(
            (width, height),
            bbox,
            target_fraction=zoom_target_fraction,
            max_zoom=zoom_max,
            min_zoom=zoom_min,
        )
        if zoom is None
        else float(zoom)
    )
    safe_max = float("inf")
    if safe_zoom:
        safe_max = object_preserving_zoom_limit(
            intrinsics=intrinsics,
            R_old_to_new=R_old_to_new,
            bbox=bbox,
            image_size=(width, height),
            principal_point=principal_point,
            bbox_margin_fraction=bbox_margin_fraction,
        )
    if safe_fill_fraction is not None:
        safe_fill_fraction = float(safe_fill_fraction)
        if not safe_zoom:
            raise ValueError("safe_fill_fraction requires safe_zoom=True")
        if not 0.0 < safe_fill_fraction <= 1.0:
            raise ValueError("safe_fill_fraction must be in (0, 1]")
        # ``safe_max`` is the zoom at which the expanded amodal object box
        # touches the virtual image boundary. Sampling a fraction of that
        # limit gives an occupancy-aware augmentation without ever clipping
        # the requested object margin.
        requested_zoom = safe_max * safe_fill_fraction
    if requested_zoom <= 0.0:
        raise ValueError("Virtual camera zoom must be positive")
    applied_zoom = min(requested_zoom, safe_max)
    applied_safe_fill_fraction = (
        float(applied_zoom / safe_max)
        if np.isfinite(safe_max) and safe_max > 0.0
        else 0.0
    )
    homography, K_new = homography_for_recenter_zoom(
        intrinsics,
        R_old_to_new,
        applied_zoom,
        image_size=(width, height),
        principal_point=principal_point,
    )

    corrected_depth, corrected_valid = corrected_source_z_depth(
        depthmap,
        intrinsics,
        R_old_to_new,
    )
    source_in_bounds = np.ones((height, width), dtype=np.uint8)
    if depth_interpolation == "nearest":
        depth_flags = cv2.INTER_NEAREST
    elif depth_interpolation == "linear":
        depth_flags = cv2.INTER_LINEAR
    else:
        raise ValueError("depth_interpolation must be 'nearest' or 'linear'")

    rgb_warped = cv2.warpPerspective(
        np.asarray(rgb),
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    depth_warped = cv2.warpPerspective(
        corrected_depth,
        homography,
        (width, height),
        flags=depth_flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_warped = cv2.warpPerspective(
        np.asarray(mask, dtype=np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    valid_warped = cv2.warpPerspective(
        corrected_valid.astype(np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    in_bounds_warped = cv2.warpPerspective(
        source_in_bounds,
        homography,
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
            f"recenter/zoom view has only {valid_depth_pixels} valid depth pixels "
            f"after warping, required {min_valid_depth_pixels}"
        )

    T_Cnew_O = compose_recentered_object_pose(T_C_O, R_old_to_new)
    camera_pose = np.linalg.inv(T_Cnew_O).astype(np.float32)
    if include_virtual_metadata:
        metadata = {
            "virtual_camera_applied": True,
            "virtual_camera_zoom_requested": np.float32(requested_zoom),
            "virtual_camera_zoom": np.float32(applied_zoom),
            "virtual_camera_zoom_safe_max": np.float32(safe_max),
            "virtual_camera_safe_fill_fraction_requested": np.float32(
                0.0 if safe_fill_fraction is None else safe_fill_fraction
            ),
            "virtual_camera_safe_fill_fraction": np.float32(
                applied_safe_fill_fraction
            ),
            "virtual_camera_valid_depth_pixels": np.int64(valid_depth_pixels),
            "virtual_camera_valid_depth_fraction": np.float32(
                valid_depth_pixels / float(height * width)
            ),
            "virtual_camera_R_source_to_virtual": R_old_to_new.astype(np.float32),
            "virtual_camera_H_source_to_virtual": homography.astype(np.float32),
            "virtual_camera_source_intrinsics": np.asarray(
                intrinsics, dtype=np.float32
            ).copy(),
            "virtual_camera_original_T_C_O": np.asarray(
                T_C_O, dtype=np.float32
            ).copy(),
            "virtual_camera_bbox_xywh": np.asarray(bbox, dtype=np.float32),
            "virtual_camera_replaces_planned_crop": bool(replace_planned_crop),
        }
    else:
        # Preserve the existing LMGeo diagnostic metadata contract.
        metadata = {
            "query_recenter_applied": True,
            "query_recenter_zoom": np.float32(applied_zoom),
            "query_recenter_valid_depth_pixels_raw": np.int64(valid_depth_pixels),
            "query_recenter_valid_fraction_raw": np.float32(
                valid_depth_pixels / float(height * width)
            ),
            "query_recenter_R_old_to_new": R_old_to_new.astype(np.float32),
            "query_recenter_H_raw": homography.astype(np.float32),
        }
    return (
        rgb_warped,
        depth_warped,
        mask_warped,
        K_new,
        T_Cnew_O,
        camera_pose,
        metadata,
    )


@dataclass(frozen=True)
class VirtualCameraRectificationConfig:
    """Role-aware, layout-independent virtual-camera sampling settings."""

    roles: tuple[str, ...] = ("query",)
    bbox_key: str = "bbox_obj"
    zoom_range: tuple[float, float] = (1.0, 5.0)
    eval_zoom: float = 3.0
    zoom_sampling: str = "log_uniform"
    zoom_policy: str = "range"
    safe_fill_fraction_range: tuple[float, float] = (0.8, 1.0)
    eval_safe_fill_fraction: float = 0.9
    principal_point: str = "image_center"
    bbox_margin_fraction: float = 0.05
    safe_zoom: bool = True
    replace_planned_crop: bool = True
    depth_interpolation: str = "nearest"
    min_valid_depth_pixels: int = 64

    def __post_init__(self):
        roles = tuple(str(value) for value in self.roles)
        object.__setattr__(self, "roles", roles)
        zoom_range = tuple(float(value) for value in self.zoom_range)
        object.__setattr__(self, "zoom_range", zoom_range)
        safe_fill_range = tuple(
            float(value) for value in self.safe_fill_fraction_range
        )
        object.__setattr__(self, "safe_fill_fraction_range", safe_fill_range)
        if not roles or not set(roles).issubset({"reference", "query"}):
            raise ValueError("virtual-camera roles must contain reference/query")
        if len(zoom_range) != 2 or zoom_range[0] <= 0 or zoom_range[1] < zoom_range[0]:
            raise ValueError("zoom_range must be a positive [minimum, maximum]")
        if float(self.eval_zoom) <= 0:
            raise ValueError("eval_zoom must be positive")
        if self.zoom_sampling not in {"uniform", "log_uniform"}:
            raise ValueError("zoom_sampling must be uniform or log_uniform")
        if self.zoom_policy not in {"range", "safe_fill"}:
            raise ValueError("zoom_policy must be range or safe_fill")
        if (
            len(safe_fill_range) != 2
            or safe_fill_range[0] <= 0.0
            or safe_fill_range[1] < safe_fill_range[0]
            or safe_fill_range[1] > 1.0
        ):
            raise ValueError(
                "safe_fill_fraction_range must be within (0, 1]"
            )
        if not 0.0 < float(self.eval_safe_fill_fraction) <= 1.0:
            raise ValueError("eval_safe_fill_fraction must be in (0, 1]")
        if self.zoom_policy == "safe_fill" and not self.safe_zoom:
            raise ValueError("zoom_policy=safe_fill requires safe_zoom=True")
        if self.principal_point not in {"preserve", "image_center"}:
            raise ValueError("Unsupported virtual-camera principal point")
        if float(self.bbox_margin_fraction) < 0:
            raise ValueError("bbox_margin_fraction must be non-negative")
        if self.depth_interpolation not in {"nearest", "linear"}:
            raise ValueError("Unsupported virtual-camera depth interpolation")
        if int(self.min_valid_depth_pixels) <= 0:
            raise ValueError("min_valid_depth_pixels must be positive")

    @classmethod
    def from_config(
        cls,
        value: "VirtualCameraRectificationConfig | Mapping[str, Any] | None",
    ) -> "VirtualCameraRectificationConfig | None":
        if value is None or isinstance(value, cls):
            return value
        return cls(**dict(value))

    def sample_zoom(self, *, rng: np.random.Generator, mode: str) -> float:
        if str(mode) != "train":
            return float(self.eval_zoom)
        low, high = self.zoom_range
        if high == low:
            return low
        if self.zoom_sampling == "log_uniform":
            return float(np.exp(rng.uniform(math.log(low), math.log(high))))
        return float(rng.uniform(low, high))

    def sample_safe_fill_fraction(
        self, *, rng: np.random.Generator, mode: str
    ) -> float:
        if str(mode) != "train":
            return float(self.eval_safe_fill_fraction)
        low, high = self.safe_fill_fraction_range
        return low if high == low else float(rng.uniform(low, high))


class VirtualCameraRectifier:
    """Apply a configured virtual camera while emitting collatable metadata."""

    def __init__(self, config: VirtualCameraRectificationConfig | Mapping[str, Any]):
        parsed = VirtualCameraRectificationConfig.from_config(config)
        if parsed is None:
            raise ValueError("VirtualCameraRectifier requires a configuration")
        self.config = parsed

    @staticmethod
    def default_metadata(
        intrinsics: np.ndarray,
        T_C_O: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "virtual_camera_applied": False,
            "virtual_camera_zoom_requested": np.float32(1.0),
            "virtual_camera_zoom": np.float32(1.0),
            "virtual_camera_zoom_safe_max": np.float32(1.0),
            "virtual_camera_safe_fill_fraction_requested": np.float32(0.0),
            "virtual_camera_safe_fill_fraction": np.float32(0.0),
            "virtual_camera_valid_depth_pixels": np.int64(0),
            "virtual_camera_valid_depth_fraction": np.float32(0.0),
            "virtual_camera_R_source_to_virtual": np.eye(3, dtype=np.float32),
            "virtual_camera_H_source_to_virtual": np.eye(3, dtype=np.float32),
            "virtual_camera_source_intrinsics": np.asarray(
                intrinsics, dtype=np.float32
            ).copy(),
            "virtual_camera_original_T_C_O": np.asarray(
                T_C_O, dtype=np.float32
            ).copy(),
            "virtual_camera_bbox_xywh": np.zeros(4, dtype=np.float32),
            "virtual_camera_replaces_planned_crop": False,
        }

    def transform(
        self,
        *,
        record: Mapping[str, Any],
        rgb: np.ndarray,
        depthmap: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        T_C_O: np.ndarray,
        camera_pose: np.ndarray,
        view_role: str,
        rng: np.random.Generator,
        mode: str,
    ):
        if str(view_role) not in self.config.roles:
            return (
                rgb,
                depthmap,
                mask,
                intrinsics,
                T_C_O,
                camera_pose,
                self.default_metadata(intrinsics, T_C_O),
            )
        bbox = bbox_from_record(record, self.config.bbox_key)
        requested_zoom = self.config.sample_zoom(rng=rng, mode=mode)
        safe_fill_fraction = None
        if self.config.zoom_policy == "safe_fill":
            safe_fill_fraction = self.config.sample_safe_fill_fraction(
                rng=rng, mode=mode
            )
        return warp_recenter_zoom_view(
            rgb=rgb,
            depthmap=depthmap,
            mask=mask,
            intrinsics=intrinsics,
            T_C_O=T_C_O,
            bbox=bbox,
            zoom=requested_zoom,
            safe_fill_fraction=safe_fill_fraction,
            principal_point=self.config.principal_point,
            safe_zoom=self.config.safe_zoom,
            bbox_margin_fraction=self.config.bbox_margin_fraction,
            replace_planned_crop=self.config.replace_planned_crop,
            depth_interpolation=self.config.depth_interpolation,
            min_valid_depth_pixels=self.config.min_valid_depth_pixels,
            include_virtual_metadata=True,
        )
