from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import math

import numpy as np
import torch
from plyfile import PlyData


def extract_prediction(prediction: Any) -> dict[str, torch.Tensor]:
    """Return the model prediction dict from trainer-specific wrappers."""

    if isinstance(prediction, dict):
        return prediction
    if isinstance(prediction, (list, tuple)) and prediction and isinstance(prediction[0], dict):
        return prediction[0]
    raise TypeError(f"Cannot extract prediction dict from {type(prediction).__name__}")


def as_numpy(value: Any) -> np.ndarray:
    """Detach tensors and convert lists/scalars to a NumPy array."""

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def stack_view_tensor(batch: list[dict[str, Any]], key: str) -> np.ndarray:
    """Stack a per-view batch key into shape ``(B, N, ...)``."""

    values = [as_numpy(view[key]) for view in batch]
    return np.stack(values, axis=1)


def view_bool_mask(batch: list[dict[str, Any]], key: str) -> np.ndarray:
    """Return a boolean ``(B, N)`` role mask from collated view metadata."""

    return stack_view_tensor(batch, key).astype(bool)


def first_scalar(value: Any) -> Any:
    """Extract a scalar from a default-collated batch field."""

    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def batch_object_ids(batch: list[dict[str, Any]]) -> np.ndarray:
    """Return object ids with shape ``(B,)`` for a collated multi-view batch."""

    return as_numpy(batch[0]["object_id"]).reshape(-1).astype(np.int64)


def invert_se3(poses: np.ndarray) -> np.ndarray:
    """Invert one or more rigid transforms."""

    poses = np.asarray(poses, dtype=np.float64)
    single = poses.ndim == 2
    poses_b = poses[None] if single else poses

    inv = np.zeros_like(poses_b)
    rotation = poses_b[..., :3, :3]
    translation = poses_b[..., :3, 3]
    inv[..., :3, :3] = np.swapaxes(rotation, -1, -2)
    inv[..., :3, 3] = -np.einsum("...ij,...j->...i", inv[..., :3, :3], translation)
    inv[..., 3, 3] = 1.0
    return inv[0] if single else inv


def camera_center_error_components(
    pred_center: np.ndarray,
    gt_center: np.ndarray,
) -> dict[str, float]:
    """Decompose an object-frame camera-center error around the object origin.

    ``radial`` is the component along the GT object-to-camera direction and
    ``tangential`` is the orthogonal component. Their squared sum equals the
    squared Euclidean center error. ``radius`` compares only object distance,
    while ``direction`` measures the orbit/viewing-direction discrepancy.
    """

    pred_center = np.asarray(pred_center, dtype=np.float64).reshape(3)
    gt_center = np.asarray(gt_center, dtype=np.float64).reshape(3)
    delta = pred_center - gt_center
    gt_radius = float(np.linalg.norm(gt_center))
    pred_radius = float(np.linalg.norm(pred_center))
    total = float(np.linalg.norm(delta))
    if gt_radius <= 1e-12:
        return {
            "total_m": total,
            "radial_m": float("nan"),
            "tangential_m": float("nan"),
            "radius_m": abs(pred_radius - gt_radius),
            "direction_deg": float("nan"),
        }

    gt_direction = gt_center / gt_radius
    radial_signed = float(np.dot(delta, gt_direction))
    tangential = float(
        np.linalg.norm(delta - radial_signed * gt_direction)
    )
    direction_deg = float("nan")
    if pred_radius > 1e-12:
        cosine = float(
            np.clip(np.dot(pred_center / pred_radius, gt_direction), -1.0, 1.0)
        )
        direction_deg = float(np.degrees(np.arccos(cosine)))
    return {
        "total_m": total,
        "radial_m": abs(radial_signed),
        "tangential_m": tangential,
        "radius_m": abs(pred_radius - gt_radius),
        "direction_deg": direction_deg,
    }


def metric_depth_error_summary(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    scale: float,
) -> dict[str, float]:
    """Summarize one view's predicted depth after one reference-derived scale."""

    pred_depth = np.asarray(pred_depth, dtype=np.float64)
    gt_depth = np.asarray(gt_depth, dtype=np.float64)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(pred_depth)
        & np.isfinite(gt_depth)
        & (pred_depth > 1e-8)
        & (gt_depth > 1e-8)
    )
    if not np.any(valid):
        return {
            "mae_m": float("nan"),
            "median_abs_m": float("nan"),
            "median_relative": float("nan"),
            "median_bias_m": float("nan"),
            "scale_log_error": float("nan"),
        }
    pred_metric = float(scale) * pred_depth[valid]
    gt_valid = gt_depth[valid]
    error = pred_metric - gt_valid
    abs_error = np.abs(error)
    view_log_scale = float(
        np.median(np.log(gt_valid) - np.log(pred_depth[valid]))
    )
    return {
        "mae_m": float(np.mean(abs_error)),
        "median_abs_m": float(np.median(abs_error)),
        "median_relative": float(np.median(abs_error / gt_valid)),
        "median_bias_m": float(np.median(error)),
        "scale_log_error": abs(view_log_scale - float(np.log(scale))),
    }


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""

    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def average_rotations(rotations: np.ndarray) -> np.ndarray:
    """Average rotations by projecting their matrix sum onto SO(3)."""

    rotations = np.asarray(rotations, dtype=np.float64)
    return project_to_rotation(rotations.sum(axis=0))


class SimilarityTransform:
    """World-to-object Sim(3) estimated from reference cameras."""

    def __init__(
        self,
        *,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
        underconstrained_scale: bool = False,
        scale_estimation: str = "camera_centers",
        depth_scale_valid_views: int = 0,
        depth_scale_log_mad: float = float("nan"),
    ):
        self.scale = float(scale)
        self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)
        self.underconstrained_scale = bool(underconstrained_scale)
        self.scale_estimation = str(scale_estimation)
        self.depth_scale_valid_views = int(depth_scale_valid_views)
        self.depth_scale_log_mad = float(depth_scale_log_mad)

    def transform_points(self, points_world: np.ndarray) -> np.ndarray:
        """Map predicted-world points to object coordinates."""

        points_world = np.asarray(points_world, dtype=np.float64)
        return self.scale * (points_world @ self.rotation.T) + self.translation

    def camera_to_object_pose(self, pred_T_W_C: np.ndarray) -> np.ndarray:
        """Convert predicted camera-to-world poses to object-frame cameras."""

        pred_T_W_C = np.asarray(pred_T_W_C, dtype=np.float64)
        single = pred_T_W_C.ndim == 2
        poses = pred_T_W_C[None] if single else pred_T_W_C

        out = np.zeros_like(poses)
        out[..., :3, :3] = np.einsum("ij,...jk->...ik", self.rotation, poses[..., :3, :3])
        out[..., :3, 3] = self.transform_points(poses[..., :3, 3])
        out[..., 3, 3] = 1.0
        return out[0] if single else out

    def object_to_camera_pose(self, pred_T_W_C: np.ndarray) -> np.ndarray:
        """Convert predicted camera poses to BOP-style object-to-camera poses."""

        return invert_se3(self.camera_to_object_pose(pred_T_W_C))


def estimate_world_to_object_sim3(
    pred_T_W_C_refs: np.ndarray,
    gt_T_C_O_refs: np.ndarray,
    *,
    solve_scale: bool = True,
    scale_estimation: str = "camera_centers",
    pred_local_points_refs: np.ndarray | None = None,
    gt_points_object_refs: np.ndarray | None = None,
    valid_masks_refs: np.ndarray | None = None,
    min_depth_pixels_per_view: int = 64,
) -> SimilarityTransform:
    """Estimate a reference-only Sim(3) from predicted world to object frame.

    ``camera_centers`` is the historical estimator: reference camera
    orientations determine rotation, while corresponding reference camera
    centers determine scale and translation. ``reference_depth`` retains the
    same camera-derived rotation and translation fit, but replaces the scale
    with a robust metric-depth estimate. Reference GT point maps are derived
    from metric depth, intrinsics, and known poses. For each reference view,
    the estimator takes the median log ratio of GT camera-space depth to
    predicted local depth over valid object pixels. View estimates are then
    combined with an equal-weight median in log space. The depths deliberately
    remain camera-origin depths: independently centering each visible surface
    would make the estimate invariant to per-view translations that are not
    part of one global Sim(3). Query depth is never used.

    When fewer than one reference view has enough valid depth pixels, the
    depth estimator falls back to the historical camera-center scale and marks
    the result as underconstrained. Missing depth inputs are a configuration
    error and raise immediately rather than silently changing the protocol.
    """

    valid_scale_estimations = {"camera_centers", "reference_depth"}
    scale_estimation = str(scale_estimation)
    if scale_estimation not in valid_scale_estimations:
        raise ValueError(
            f"Unknown scale_estimation={scale_estimation!r}; expected one of "
            f"{sorted(valid_scale_estimations)}"
        )

    pred_T_W_C_refs = np.asarray(pred_T_W_C_refs, dtype=np.float64)
    gt_T_C_O_refs = np.asarray(gt_T_C_O_refs, dtype=np.float64)
    if pred_T_W_C_refs.ndim == 2:
        pred_T_W_C_refs = pred_T_W_C_refs[None]
    if gt_T_C_O_refs.ndim == 2:
        gt_T_C_O_refs = gt_T_C_O_refs[None]
    if pred_T_W_C_refs.shape != gt_T_C_O_refs.shape:
        raise ValueError(f"Pose shape mismatch: {pred_T_W_C_refs.shape} vs {gt_T_C_O_refs.shape}")

    gt_T_O_C_refs = invert_se3(gt_T_C_O_refs)
    rotation_pairs = np.einsum(
        "...ij,...kj->...ik",
        gt_T_O_C_refs[..., :3, :3],
        pred_T_W_C_refs[..., :3, :3],
    )
    rotation = average_rotations(rotation_pairs)

    pred_centers = pred_T_W_C_refs[..., :3, 3]
    gt_centers = gt_T_O_C_refs[..., :3, 3]
    pred_mean = pred_centers.mean(axis=0)
    gt_mean = gt_centers.mean(axis=0)

    scale = 1.0
    underconstrained = False
    scale_estimation_used = "disabled"
    depth_scale_valid_views = 0
    depth_scale_log_mad = float("nan")
    if solve_scale:
        pred_centered = pred_centers - pred_mean
        gt_centered = gt_centers - gt_mean
        pred_rot_centered = pred_centered @ rotation.T
        denom = float(np.sum(pred_rot_centered * pred_rot_centered))
        camera_center_scale = 1.0
        camera_center_underconstrained = False
        if len(pred_centers) < 2 or denom < 1e-12:
            camera_center_underconstrained = True
        else:
            camera_center_scale = float(
                np.sum(gt_centered * pred_rot_centered) / denom
            )
            if (
                not np.isfinite(camera_center_scale)
                or camera_center_scale <= 1e-12
            ):
                camera_center_scale = 1.0
                camera_center_underconstrained = True

        if scale_estimation == "reference_depth":
            if (
                pred_local_points_refs is None
                or gt_points_object_refs is None
                or valid_masks_refs is None
            ):
                raise ValueError(
                    "reference_depth scale estimation requires predicted local "
                    "points, GT object-frame point maps derived from depth, and "
                    "valid masks for the references"
                )
            pred_local_points_refs = np.asarray(
                pred_local_points_refs, dtype=np.float64
            )
            gt_points_object_refs = np.asarray(
                gt_points_object_refs, dtype=np.float64
            )
            valid_masks_refs = np.asarray(valid_masks_refs, dtype=bool)
            if (
                pred_local_points_refs.ndim < 2
                or pred_local_points_refs.shape[-1] != 3
            ):
                raise ValueError(
                    "pred_local_points_refs must end in XYZ, got "
                    f"{pred_local_points_refs.shape}"
                )
            if (
                gt_points_object_refs.shape != pred_local_points_refs.shape
                or valid_masks_refs.shape != pred_local_points_refs.shape[:-1]
            ):
                raise ValueError(
                    "Reference depth alignment shape mismatch: "
                    f"pred={pred_local_points_refs.shape}, "
                    f"gt={gt_points_object_refs.shape}, "
                    f"mask={valid_masks_refs.shape}"
                )
            if pred_local_points_refs.shape[0] != len(pred_centers):
                raise ValueError(
                    "Reference depth view count does not match reference poses: "
                    f"{pred_local_points_refs.shape[0]} vs {len(pred_centers)}"
                )

            per_view_log_scales = []
            min_pixels = max(1, int(min_depth_pixels_per_view))
            for view_idx, (pred_points, gt_points_object, valid_mask) in enumerate(
                zip(
                    pred_local_points_refs,
                    gt_points_object_refs,
                    valid_masks_refs,
                )
            ):
                gt_T_C_O = gt_T_C_O_refs[view_idx]
                gt_points_camera = (
                    gt_points_object @ gt_T_C_O[:3, :3].T
                    + gt_T_C_O[:3, 3]
                )
                valid = (
                    valid_mask
                    & np.isfinite(pred_points).all(axis=-1)
                    & np.isfinite(gt_points_camera).all(axis=-1)
                    & (pred_points[..., 2] > 1e-8)
                    & (gt_points_camera[..., 2] > 1e-8)
                )
                if int(valid.sum()) < min_pixels:
                    continue
                log_ratios = np.log(gt_points_camera[..., 2][valid]) - np.log(
                    pred_points[..., 2][valid]
                )
                finite_log_ratios = log_ratios[np.isfinite(log_ratios)]
                if len(finite_log_ratios) >= min_pixels:
                    per_view_log_scales.append(
                        float(np.median(finite_log_ratios))
                    )

            depth_scale_valid_views = len(per_view_log_scales)
            if per_view_log_scales:
                per_view_log_scales = np.asarray(
                    per_view_log_scales, dtype=np.float64
                )
                median_log_scale = float(np.median(per_view_log_scales))
                scale = float(np.exp(median_log_scale))
                depth_scale_log_mad = float(
                    np.median(np.abs(per_view_log_scales - median_log_scale))
                )
                if np.isfinite(scale) and scale > 1e-12:
                    scale_estimation_used = "reference_depth"
                else:
                    scale = camera_center_scale
                    scale_estimation_used = "camera_centers_fallback"
                    underconstrained = True
            else:
                scale = camera_center_scale
                scale_estimation_used = "camera_centers_fallback"
                underconstrained = True
        else:
            scale = camera_center_scale
            scale_estimation_used = "camera_centers"
            underconstrained = camera_center_underconstrained

    translation = gt_mean - scale * (rotation @ pred_mean)
    return SimilarityTransform(
        scale=scale,
        rotation=rotation,
        translation=translation,
        underconstrained_scale=underconstrained,
        scale_estimation=scale_estimation_used,
        depth_scale_valid_views=depth_scale_valid_views,
        depth_scale_log_mad=depth_scale_log_mad,
    )


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> np.ndarray:
    """Return SO(3) angular errors in degrees."""

    R_pred = np.asarray(R_pred, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)
    residual = np.einsum("...ji,...jk->...ik", R_pred, R_gt)
    trace = np.trace(residual, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def safe_mean(values: list[float]) -> float:
    """Mean with NaN for empty inputs."""

    return float(np.mean(values)) if values else float("nan")


def safe_median(values: list[float]) -> float:
    """Median with NaN for empty inputs."""

    return float(np.median(values)) if values else float("nan")


def safe_percentile(values: list[float], percentile: float) -> float:
    """Percentile with NaN for empty inputs."""

    return float(np.percentile(values, percentile)) if values else float("nan")


def recall_below(values: list[float], threshold: float) -> float:
    """Fraction of values below a threshold."""

    return float(np.mean(np.asarray(values, dtype=np.float64) < float(threshold))) if values else float("nan")


def load_models_info(data_root: str | Path, models_folder: str) -> dict:
    """Load BOP ``models_info.json``."""

    path = Path(data_root) / models_folder / "models_info.json"
    with path.open("r") as handle:
        return json.load(handle)


class BopModelCache:
    """Lazy BOP model vertex and diameter cache.

    The local LM-O models are stored in millimeters. Metrics use meters because
    LMGeo converts BOP poses/depths to meters, so vertices and diameters are
    scaled by ``unit_scale``.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        models_folder: str = "models_eval",
        unit_scale: float = 0.001,
        max_model_points: int = 20000,
    ):
        self.data_root = Path(data_root)
        self.models_folder = str(models_folder)
        self.unit_scale = float(unit_scale)
        self.max_model_points = int(max_model_points)
        self.models_info = load_models_info(self.data_root, self.models_folder)
        self._points: dict[int, np.ndarray] = {}
        self._diameters: dict[int, float] = {}

    def model_path(self, obj_id: int) -> Path:
        return self.data_root / self.models_folder / f"obj_{int(obj_id):06d}.ply"

    def diameter(self, obj_id: int) -> float:
        obj_id = int(obj_id)
        if obj_id not in self._diameters:
            self._diameters[obj_id] = float(self.models_info[str(obj_id)]["diameter"]) * self.unit_scale
        return self._diameters[obj_id]

    def points(self, obj_id: int) -> np.ndarray:
        obj_id = int(obj_id)
        if obj_id not in self._points:
            ply = PlyData.read(str(self.model_path(obj_id)))
            vertex = ply["vertex"].data
            points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
            points *= self.unit_scale
            if self.max_model_points > 0 and len(points) > self.max_model_points:
                indices = np.linspace(0, len(points) - 1, self.max_model_points, dtype=np.int64)
                points = points[indices]
            if len(points) == 0:
                raise ValueError(f"No vertices loaded for object {obj_id}")
            self._points[obj_id] = points
        return self._points[obj_id]


def transform_points(T_C_O: np.ndarray, points_object: np.ndarray) -> np.ndarray:
    """Transform object-frame points to camera coordinates."""

    return points_object @ T_C_O[:3, :3].T + T_C_O[:3, 3]


def nearest_distances_torch(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Chunked nearest-neighbor distances from source to target."""

    if source.numel() == 0 or target.numel() == 0:
        return torch.empty((0,), device=source.device, dtype=source.dtype)
    nearest = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(source), chunk_size):
        distances = torch.cdist(source[start : start + chunk_size], target)
        nearest.append(distances.min(dim=1).values)
    return torch.cat(nearest, dim=0)


def add_error(T_C_O_pred: np.ndarray, T_C_O_gt: np.ndarray, points_object: np.ndarray) -> float:
    """Average Distance of Model Points."""

    pred = transform_points(T_C_O_pred, points_object)
    gt = transform_points(T_C_O_gt, points_object)
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def adds_error(
    T_C_O_pred: np.ndarray,
    T_C_O_gt: np.ndarray,
    points_object: np.ndarray,
    *,
    device: str = "cpu",
    chunk_size: int = 2048,
) -> float:
    """ADD-S / ADI nearest-neighbor distance."""

    pred = torch.as_tensor(transform_points(T_C_O_pred, points_object), dtype=torch.float32, device=device)
    gt = torch.as_tensor(transform_points(T_C_O_gt, points_object), dtype=torch.float32, device=device)
    dists = nearest_distances_torch(pred, gt, chunk_size=chunk_size)
    return float(dists.mean().detach().cpu()) if len(dists) else float("nan")


def deterministic_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Return at most ``max_points`` rows spread uniformly through the input."""

    points = np.asarray(points)
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
    return points[indices]


def voxel_downsample(points: np.ndarray, voxel_size: float | None) -> np.ndarray:
    """Downsample to one centroid per occupied voxel."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or voxel_size is None or voxel_size <= 0:
        return points
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        return points
    voxels = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(voxels, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.stack(
        [np.bincount(inverse, weights=points[:, dim]) for dim in range(3)],
        axis=1,
    )
    return sums / counts[:, None]


def sample_points(points: np.ndarray, *, voxel_size: float | None, max_points: int) -> np.ndarray:
    """Filter finite points, voxel downsample, then cap deterministically."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = voxel_downsample(points[finite], voxel_size)
    return deterministic_subsample(points, int(max_points))


def chamfer_components(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    device: str = "cpu",
    chunk_size: int = 1024,
) -> tuple[float, float, float]:
    """Return directed and symmetric Chamfer distances."""

    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("nan"), float("nan"), float("nan")
    pred = torch.as_tensor(pred_points, dtype=torch.float32, device=device)
    gt = torch.as_tensor(gt_points, dtype=torch.float32, device=device)
    pred_to_gt = nearest_distances_torch(pred, gt, chunk_size=chunk_size).mean()
    gt_to_pred = nearest_distances_torch(gt, pred, chunk_size=chunk_size).mean()
    chamfer = 0.5 * (pred_to_gt + gt_to_pred)
    return float(pred_to_gt.cpu()), float(gt_to_pred.cpu()), float(chamfer.cpu())


def finite_float(value: float) -> float:
    """Return a Python float, preserving NaN for invalid values."""

    value = float(value)
    return value if math.isfinite(value) else float("nan")
