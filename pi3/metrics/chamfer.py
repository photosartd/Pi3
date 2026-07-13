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

    def __init__(
        self,
        data_root: str,
        *,
        models_folder: str = "models_eval",
        unit_scale: float = 0.001,
        solve_scale: bool = True,
        voxel_size: float | None = None,
        voxel_size_d: float | None = 0.01,
        max_pred_points: int = 20000,
        max_gt_points: int = 20000,
        chunk_size: int = 1024,
        device: str = "cpu",
    ):
        self.model_cache = BopModelCache(data_root, models_folder=models_folder, unit_scale=unit_scale)
        self.solve_scale = bool(solve_scale)
        self.voxel_size = voxel_size
        self.voxel_size_d = voxel_size_d
        self.max_pred_points = int(max_pred_points)
        self.max_gt_points = int(max_gt_points)
        self.chunk_size = int(chunk_size)
        self.device = str(device)
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
        pred = extract_prediction(prediction)
        pred_points = pred["points"].detach().float().cpu().numpy()
        pred_T_W_C = pred["camera_poses"].detach().float().cpu().numpy()
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
