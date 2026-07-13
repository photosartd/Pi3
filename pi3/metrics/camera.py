from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import BaseMetric
from .utils import (
    estimate_world_to_object_sim3,
    extract_prediction,
    invert_se3,
    rotation_error_deg,
    safe_mean,
    safe_median,
    stack_view_tensor,
    view_bool_mask,
)


class CameraAlignmentMetric(BaseMetric):
    """Aligned camera residuals for reference and query frames.

    A Sim(3) is estimated from reference camera predictions to the GT object
    coordinate frame. The same transform is then applied to reference and query
    predicted cameras. The metric reports camera-center and rotation residuals
    against GT camera-to-object poses.
    """

    name = "camera"

    def __init__(self, *, solve_scale: bool = True):
        self.solve_scale = bool(solve_scale)
        self.reset()

    def reset(self) -> None:
        self.ref_center_m: list[float] = []
        self.ref_rot_deg: list[float] = []
        self.query_center_m: list[float] = []
        self.query_rot_deg: list[float] = []
        self.scales: list[float] = []

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
        pred_T_W_C = pred["camera_poses"].detach().float().cpu().numpy()
        gt_T_C_O = stack_view_tensor(batch, "T_C_O").astype(np.float64)
        gt_T_O_C = invert_se3(gt_T_C_O)
        ref_mask = view_bool_mask(batch, "is_reference")
        query_mask = view_bool_mask(batch, "is_query")

        for batch_idx in range(pred_T_W_C.shape[0]):
            refs = ref_mask[batch_idx]
            if refs.sum() == 0:
                continue
            alignment = estimate_world_to_object_sim3(
                pred_T_W_C[batch_idx, refs],
                gt_T_C_O[batch_idx, refs],
                solve_scale=self.solve_scale,
            )
            self.scales.append(float(alignment.scale))

            pred_T_O_C = alignment.camera_to_object_pose(pred_T_W_C[batch_idx])
            self._accumulate_role(
                pred_T_O_C[refs],
                gt_T_O_C[batch_idx, refs],
                self.ref_center_m,
                self.ref_rot_deg,
            )

            queries = query_mask[batch_idx]
            self._accumulate_role(
                pred_T_O_C[queries],
                gt_T_O_C[batch_idx, queries],
                self.query_center_m,
                self.query_rot_deg,
            )

    @staticmethod
    def _accumulate_role(
        pred_T_O_C: np.ndarray,
        gt_T_O_C: np.ndarray,
        center_store: list[float],
        rot_store: list[float],
    ) -> None:
        for T_pred, T_gt in zip(pred_T_O_C, gt_T_O_C):
            center_store.append(float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])))
            rot_store.append(float(rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3])))

    def compute(self) -> dict[str, float]:
        return {
            "reference_count": float(len(self.ref_center_m)),
            "reference_center_mean_m": safe_mean(self.ref_center_m),
            "reference_center_median_m": safe_median(self.ref_center_m),
            "reference_rot_mean_deg": safe_mean(self.ref_rot_deg),
            "reference_rot_median_deg": safe_median(self.ref_rot_deg),
            "query_count": float(len(self.query_center_m)),
            "query_center_mean_m": safe_mean(self.query_center_m),
            "query_center_median_m": safe_median(self.query_center_m),
            "query_rot_mean_deg": safe_mean(self.query_rot_deg),
            "query_rot_median_deg": safe_median(self.query_rot_deg),
            "alignment_scale_mean": safe_mean(self.scales),
            "alignment_scale_median": safe_median(self.scales),
        }
