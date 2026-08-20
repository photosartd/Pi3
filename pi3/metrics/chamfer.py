from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import BaseMetric
from .utils import (
    BopModelCache,
    chamfer_components,
    estimate_world_to_object_sim3,
    extract_prediction,
    nearest_distances_torch,
    safe_mean,
    sample_points,
    stack_view_tensor,
    view_bool_mask,
)


class ReferenceChamferMetric(BaseMetric):
    """Visible-surface Chamfer on reference/keyframe views only.

    Predicted reference point maps are aligned into the object frame with the
    same reference-camera Sim(3) used by the pose metrics. They are compared
    against the visible GT object points from the loaded reference depth maps.
    """

    name = "chamfer"
    required_capabilities = frozenset(
        {"key_query", "object_pose", "object_model"}
    )

    def __init__(
        self,
        data_root: str,
        *,
        models_folder: str = "models_eval",
        unit_scale: float = 0.001,
        solve_scale: bool = True,
        scale_estimation: str = "camera_centers",
        min_depth_pixels_per_view: int = 64,
        voxel_size: float | None = None,
        voxel_size_d: float | None = 0.01,
        max_pred_points: int = 20000,
        max_gt_points: int = 20000,
        chunk_size: int = 1024,
        device: str = "cpu",
        use_gpu_fast_path: bool = False,
    ):
        self.model_cache = BopModelCache(data_root, models_folder=models_folder, unit_scale=unit_scale)
        self.solve_scale = bool(solve_scale)
        self.scale_estimation = str(scale_estimation)
        self.min_depth_pixels_per_view = int(min_depth_pixels_per_view)
        self.voxel_size = voxel_size
        self.voxel_size_d = voxel_size_d
        self.max_pred_points = int(max_pred_points)
        self.max_gt_points = int(max_gt_points)
        self.chunk_size = int(chunk_size)
        self.device = str(device)
        self.use_gpu_fast_path = bool(use_gpu_fast_path)
        self.reset()

    def reset(self) -> None:
        self.pred_to_gt_m: list[float] = []
        self.gt_to_pred_m: list[float] = []
        self.chamfer_m: list[float] = []
        self.chamfer_d: list[float] = []
        self.pred_counts: list[float] = []
        self.gt_counts: list[float] = []

    @torch.no_grad()
    def update(
        self,
        prediction: Any,
        batch: list[dict[str, Any]],
        loss_output: Any | None = None,
        *,
        mode: str = "train",
    ) -> None:
        if self.use_gpu_fast_path:
            self._update_gpu_fast(prediction, batch, mode=mode)
            return

        self._update_cpu_voxel(prediction, batch, mode=mode)

    def _update_cpu_voxel(
        self,
        prediction: Any,
        batch: list[dict[str, Any]],
        *,
        mode: str = "train",
    ) -> None:
        pred = extract_prediction(prediction)
        pred_points = pred["points"].detach().float().cpu().numpy()
        pred_T_W_C = pred["camera_poses"].detach().float().cpu().numpy()
        pred_local_points = (
            pred["local_points"].detach().float().cpu().numpy()
            if self.scale_estimation == "reference_depth"
            else None
        )
        gt_T_C_O = stack_view_tensor(batch, "T_C_O").astype(np.float64)
        gt_points = stack_view_tensor(batch, "pts3d").astype(np.float64)
        valid_masks = stack_view_tensor(batch, "valid_mask").astype(bool)
        ref_mask = view_bool_mask(batch, "is_reference")
        obj_ids = stack_view_tensor(batch, "object_id")[:, 0].astype(np.int64)

        for batch_idx, obj_id in enumerate(obj_ids):
            refs = ref_mask[batch_idx]
            if refs.sum() == 0:
                continue
            alignment = estimate_world_to_object_sim3(
                pred_T_W_C[batch_idx, refs],
                gt_T_C_O[batch_idx, refs],
                solve_scale=self.solve_scale,
                scale_estimation=self.scale_estimation,
                pred_local_points_refs=(
                    pred_local_points[batch_idx, refs]
                    if pred_local_points is not None
                    else None
                ),
                gt_points_object_refs=gt_points[batch_idx, refs],
                valid_masks_refs=valid_masks[batch_idx, refs],
                min_depth_pixels_per_view=self.min_depth_pixels_per_view,
            )

            pred_ref_points = pred_points[batch_idx, refs]
            pred_ref_masks = valid_masks[batch_idx, refs]
            gt_ref_points = gt_points[batch_idx, refs]

            pred_cloud_world = pred_ref_points[pred_ref_masks]
            gt_cloud_object = gt_ref_points[pred_ref_masks]
            pred_cloud_object = alignment.transform_points(pred_cloud_world)

            diameter = self.model_cache.diameter(int(obj_id))
            voxel_size = self._resolved_voxel_size(diameter)
            pred_cloud = sample_points(
                pred_cloud_object,
                voxel_size=voxel_size,
                max_points=self.max_pred_points,
            )
            gt_cloud = sample_points(
                gt_cloud_object,
                voxel_size=voxel_size,
                max_points=self.max_gt_points,
            )
            pred_to_gt, gt_to_pred, chamfer = chamfer_components(
                pred_cloud,
                gt_cloud,
                device=self.device,
                chunk_size=self.chunk_size,
            )
            self.pred_to_gt_m.append(pred_to_gt)
            self.gt_to_pred_m.append(gt_to_pred)
            self.chamfer_m.append(chamfer)
            self.chamfer_d.append(float(chamfer / diameter) if np.isfinite(chamfer) else float("nan"))
            self.pred_counts.append(float(len(pred_cloud)))
            self.gt_counts.append(float(len(gt_cloud)))

    def _update_gpu_fast(
        self,
        prediction: Any,
        batch: list[dict[str, Any]],
        *,
        mode: str = "train",
    ) -> None:
        pred = extract_prediction(prediction)
        pred_points = pred["points"].detach().float()
        if not pred_points.is_cuda or str(self.device) == "cpu":
            self._update_cpu_voxel(prediction, batch, mode=mode)
            return

        device = pred_points.device
        pred_T_W_C = pred["camera_poses"].detach().float().cpu().numpy()
        pred_local_points = (
            pred["local_points"].detach().float().cpu().numpy()
            if self.scale_estimation == "reference_depth"
            else None
        )
        gt_T_C_O = self._stack_views(batch, "T_C_O", device=device).detach().cpu().numpy().astype(np.float64)
        gt_points_object = (
            self._stack_views(batch, "pts3d", device=device)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
            if self.scale_estimation == "reference_depth"
            else None
        )
        ref_mask = self._stack_views(batch, "is_reference", device=device).bool()
        obj_ids = self._stack_views(batch, "object_id", device=device)[:, 0].detach().cpu().numpy().astype(np.int64)

        for batch_idx, obj_id in enumerate(obj_ids):
            refs = ref_mask[batch_idx]
            if not bool(refs.any()):
                continue

            refs_cpu = refs.detach().cpu().numpy().astype(bool)
            alignment = estimate_world_to_object_sim3(
                pred_T_W_C[batch_idx, refs_cpu],
                gt_T_C_O[batch_idx, refs_cpu],
                solve_scale=self.solve_scale,
                scale_estimation=self.scale_estimation,
                pred_local_points_refs=(
                    pred_local_points[batch_idx, refs_cpu]
                    if pred_local_points is not None
                    else None
                ),
                gt_points_object_refs=gt_points_object[batch_idx, refs_cpu],
                valid_masks_refs=(
                    self._stack_views(batch, "valid_mask", device=device)[batch_idx, refs]
                    .detach()
                    .bool()
                    .cpu()
                    .numpy()
                ),
                min_depth_pixels_per_view=self.min_depth_pixels_per_view,
            )

            ref_indices = torch.where(refs)[0]
            ref_indices_list = ref_indices.detach().cpu().tolist()
            pred_ref_points = pred_points[batch_idx, ref_indices]
            pred_ref_masks = torch.stack(
                [batch[view_idx]["valid_mask"][batch_idx].to(device=device, non_blocking=True) for view_idx in ref_indices_list],
                dim=0,
            ).bool()
            gt_ref_points = torch.stack(
                [
                    batch[view_idx]["pts3d"][batch_idx].to(device=device, dtype=torch.float32, non_blocking=True)
                    for view_idx in ref_indices_list
                ],
                dim=0,
            )

            pred_cloud_world = pred_ref_points[pred_ref_masks]
            gt_cloud_object = gt_ref_points[pred_ref_masks]
            pred_cloud_object = self._transform_points_torch(pred_cloud_world, alignment)
            pred_cloud = self._sample_points_torch(pred_cloud_object, self.max_pred_points)
            gt_cloud = self._sample_points_torch(gt_cloud_object, self.max_gt_points)
            pred_to_gt, gt_to_pred, chamfer = self._chamfer_components_torch(pred_cloud, gt_cloud)

            diameter = self.model_cache.diameter(int(obj_id))
            self.pred_to_gt_m.append(pred_to_gt)
            self.gt_to_pred_m.append(gt_to_pred)
            self.chamfer_m.append(chamfer)
            self.chamfer_d.append(float(chamfer / diameter) if np.isfinite(chamfer) else float("nan"))
            self.pred_counts.append(float(len(pred_cloud)))
            self.gt_counts.append(float(len(gt_cloud)))

    @staticmethod
    def _stack_views(
        batch: list[dict[str, Any]],
        key: str,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        values = []
        for view in batch:
            value = view[key]
            if torch.is_tensor(value):
                values.append(value.detach().to(device=device, non_blocking=True))
            else:
                values.append(torch.as_tensor(value, device=device))
        return torch.stack(values, dim=1)

    @staticmethod
    def _transform_points_torch(points_world: torch.Tensor, alignment) -> torch.Tensor:
        rotation = torch.as_tensor(alignment.rotation, dtype=points_world.dtype, device=points_world.device)
        translation = torch.as_tensor(alignment.translation, dtype=points_world.dtype, device=points_world.device)
        return float(alignment.scale) * (points_world @ rotation.T) + translation

    @staticmethod
    def _sample_points_torch(points: torch.Tensor, max_points: int) -> torch.Tensor:
        if points.numel() == 0:
            return points.reshape(0, 3)
        points = points[torch.isfinite(points).all(dim=-1)]
        max_points = int(max_points)
        if max_points <= 0 or len(points) <= max_points:
            return points
        indices = torch.linspace(
            0,
            len(points) - 1,
            steps=max_points,
            device=points.device,
            dtype=torch.float32,
        ).long()
        return points.index_select(0, indices)

    def _chamfer_components_torch(self, pred_points: torch.Tensor, gt_points: torch.Tensor) -> tuple[float, float, float]:
        if pred_points.numel() == 0 or gt_points.numel() == 0:
            return float("nan"), float("nan"), float("nan")
        pred_to_gt = nearest_distances_torch(
            pred_points.float(),
            gt_points.float(),
            chunk_size=self.chunk_size,
        ).mean()
        gt_to_pred = nearest_distances_torch(
            gt_points.float(),
            pred_points.float(),
            chunk_size=self.chunk_size,
        ).mean()
        chamfer = 0.5 * (pred_to_gt + gt_to_pred)
        return float(pred_to_gt.cpu()), float(gt_to_pred.cpu()), float(chamfer.cpu())

    def _resolved_voxel_size(self, diameter: float) -> float | None:
        if self.voxel_size is not None:
            return float(self.voxel_size)
        if self.voxel_size_d is not None:
            return float(self.voxel_size_d) * float(diameter)
        return None

    def compute(self) -> dict[str, float]:
        return {
            "reference_pred_to_gt_mean_m": safe_mean(self.pred_to_gt_m),
            "reference_gt_to_pred_mean_m": safe_mean(self.gt_to_pred_m),
            "reference_chamfer_mean_m": safe_mean(self.chamfer_m),
            "reference_chamfer_mean_d": safe_mean(self.chamfer_d),
            "reference_pred_points_mean": safe_mean(self.pred_counts),
            "reference_gt_points_mean": safe_mean(self.gt_counts),
        }
