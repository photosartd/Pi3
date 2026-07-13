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
    ):
        self.scale = float(scale)
        self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)
        self.underconstrained_scale = bool(underconstrained_scale)

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
) -> SimilarityTransform:
    """Estimate a Sim(3) from predicted reference cameras to GT object frame."""

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
    if solve_scale:
        pred_centered = pred_centers - pred_mean
        gt_centered = gt_centers - gt_mean
        pred_rot_centered = pred_centered @ rotation.T
        denom = float(np.sum(pred_rot_centered * pred_rot_centered))
        if len(pred_centers) < 2 or denom < 1e-12:
            underconstrained = True
        else:
            scale = float(np.sum(gt_centered * pred_rot_centered) / denom)
            if not np.isfinite(scale) or abs(scale) < 1e-12:
                scale = 1.0
                underconstrained = True

    translation = gt_mean - scale * (rotation @ pred_mean)
    return SimilarityTransform(
        scale=scale,
        rotation=rotation,
        translation=translation,
        underconstrained_scale=underconstrained,
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
