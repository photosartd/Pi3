from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import BaseMetric
from .utils import (
    camera_center_error_components,
    estimate_world_to_object_sim3,
    extract_prediction,
    invert_se3,
    metric_depth_error_summary,
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
    required_capabilities = frozenset({"key_query", "object_pose"})

    def __init__(
        self,
        *,
        solve_scale: bool = True,
        scale_estimation: str = "camera_centers",
        min_depth_pixels_per_view: int = 64,
    ):
        self.solve_scale = bool(solve_scale)
        self.scale_estimation = str(scale_estimation)
        self.min_depth_pixels_per_view = int(min_depth_pixels_per_view)
        self.reset()

    def reset(self) -> None:
        self.ref_center_m: list[float] = []
        self.ref_center_radial_m: list[float] = []
        self.ref_center_tangential_m: list[float] = []
        self.ref_radius_m: list[float] = []
        self.ref_direction_deg: list[float] = []
        self.ref_rot_deg: list[float] = []
        self.ref_depth_mae_m: list[float] = []
        self.ref_depth_median_abs_m: list[float] = []
        self.ref_depth_median_relative: list[float] = []
        self.ref_depth_median_bias_m: list[float] = []
        self.ref_depth_scale_log_error: list[float] = []
        self.query_center_m: list[float] = []
        self.query_center_radial_m: list[float] = []
        self.query_center_tangential_m: list[float] = []
        self.query_radius_m: list[float] = []
        self.query_direction_deg: list[float] = []
        self.query_rot_deg: list[float] = []
        self.query_depth_mae_m: list[float] = []
        self.query_depth_median_abs_m: list[float] = []
        self.query_depth_median_relative: list[float] = []
        self.query_depth_median_bias_m: list[float] = []
        self.query_depth_scale_log_error: list[float] = []
        self.scales: list[float] = []
        self.depth_scale_log_mads: list[float] = []
        self.depth_scale_valid_views: list[float] = []
        self.num_underconstrained = 0

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
        pred_local_points = (
            pred["local_points"].detach().float().cpu().numpy()
            if self.scale_estimation == "reference_depth"
            else None
        )
        gt_T_C_O = stack_view_tensor(batch, "T_C_O").astype(np.float64)
        gt_points_object = (
            stack_view_tensor(batch, "pts3d").astype(np.float64)
            if self.scale_estimation == "reference_depth"
            else None
        )
        valid_masks = (
            stack_view_tensor(batch, "valid_mask").astype(bool)
            if self.scale_estimation == "reference_depth"
            else None
        )
        gt_depths = (
            stack_view_tensor(batch, "depthmap").astype(np.float64)
            if self.scale_estimation == "reference_depth"
            else None
        )
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
                scale_estimation=self.scale_estimation,
                pred_local_points_refs=(
                    pred_local_points[batch_idx, refs]
                    if pred_local_points is not None
                    else None
                ),
                gt_points_object_refs=(
                    gt_points_object[batch_idx, refs]
                    if gt_points_object is not None
                    else None
                ),
                valid_masks_refs=(
                    valid_masks[batch_idx, refs]
                    if valid_masks is not None
                    else None
                ),
                min_depth_pixels_per_view=self.min_depth_pixels_per_view,
            )
            self.scales.append(float(alignment.scale))
            self.depth_scale_valid_views.append(
                float(alignment.depth_scale_valid_views)
            )
            if np.isfinite(alignment.depth_scale_log_mad):
                self.depth_scale_log_mads.append(
                    float(alignment.depth_scale_log_mad)
                )
            self.num_underconstrained += int(alignment.underconstrained_scale)

            pred_T_O_C = alignment.camera_to_object_pose(pred_T_W_C[batch_idx])
            self._accumulate_role(
                pred_T_O_C[refs],
                gt_T_O_C[batch_idx, refs],
                self.ref_center_m,
                self.ref_center_radial_m,
                self.ref_center_tangential_m,
                self.ref_radius_m,
                self.ref_direction_deg,
                self.ref_rot_deg,
            )
            if pred_local_points is not None:
                self._accumulate_depth_role(
                    pred_local_points[batch_idx, refs, ..., 2],
                    gt_depths[batch_idx, refs],
                    valid_masks[batch_idx, refs],
                    alignment.scale,
                    self.ref_depth_mae_m,
                    self.ref_depth_median_abs_m,
                    self.ref_depth_median_relative,
                    self.ref_depth_median_bias_m,
                    self.ref_depth_scale_log_error,
                )

            queries = query_mask[batch_idx]
            self._accumulate_role(
                pred_T_O_C[queries],
                gt_T_O_C[batch_idx, queries],
                self.query_center_m,
                self.query_center_radial_m,
                self.query_center_tangential_m,
                self.query_radius_m,
                self.query_direction_deg,
                self.query_rot_deg,
            )
            if pred_local_points is not None:
                self._accumulate_depth_role(
                    pred_local_points[batch_idx, queries, ..., 2],
                    gt_depths[batch_idx, queries],
                    valid_masks[batch_idx, queries],
                    alignment.scale,
                    self.query_depth_mae_m,
                    self.query_depth_median_abs_m,
                    self.query_depth_median_relative,
                    self.query_depth_median_bias_m,
                    self.query_depth_scale_log_error,
                )

    @staticmethod
    def _accumulate_role(
        pred_T_O_C: np.ndarray,
        gt_T_O_C: np.ndarray,
        center_store: list[float],
        radial_store: list[float],
        tangential_store: list[float],
        radius_store: list[float],
        direction_store: list[float],
        rot_store: list[float],
    ) -> None:
        for T_pred, T_gt in zip(pred_T_O_C, gt_T_O_C):
            components = camera_center_error_components(
                T_pred[:3, 3], T_gt[:3, 3]
            )
            center_store.append(components["total_m"])
            radial_store.append(components["radial_m"])
            tangential_store.append(components["tangential_m"])
            radius_store.append(components["radius_m"])
            direction_store.append(components["direction_deg"])
            rot_store.append(float(rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3])))

    @staticmethod
    def _accumulate_depth_role(
        pred_depths: np.ndarray,
        gt_depths: np.ndarray,
        valid_masks: np.ndarray,
        scale: float,
        mae_store: list[float],
        median_abs_store: list[float],
        relative_store: list[float],
        bias_store: list[float],
        scale_log_error_store: list[float],
    ) -> None:
        for pred_depth, gt_depth, valid_mask in zip(
            pred_depths, gt_depths, valid_masks
        ):
            summary = metric_depth_error_summary(
                pred_depth, gt_depth, valid_mask, scale=scale
            )
            if np.isfinite(summary["mae_m"]):
                mae_store.append(summary["mae_m"])
                median_abs_store.append(summary["median_abs_m"])
                relative_store.append(summary["median_relative"])
                bias_store.append(summary["median_bias_m"])
                scale_log_error_store.append(summary["scale_log_error"])

    def compute(self) -> dict[str, float]:
        return {
            "reference_count": float(len(self.ref_center_m)),
            "reference_center_mean_m": safe_mean(self.ref_center_m),
            "reference_center_median_m": safe_median(self.ref_center_m),
            "reference_center_radial_mean_m": safe_mean(self.ref_center_radial_m),
            "reference_center_radial_median_m": safe_median(self.ref_center_radial_m),
            "reference_center_tangential_mean_m": safe_mean(self.ref_center_tangential_m),
            "reference_center_tangential_median_m": safe_median(self.ref_center_tangential_m),
            "reference_radius_error_mean_m": safe_mean(self.ref_radius_m),
            "reference_radius_error_median_m": safe_median(self.ref_radius_m),
            "reference_direction_error_mean_deg": safe_mean(self.ref_direction_deg),
            "reference_direction_error_median_deg": safe_median(self.ref_direction_deg),
            "reference_rot_mean_deg": safe_mean(self.ref_rot_deg),
            "reference_rot_median_deg": safe_median(self.ref_rot_deg),
            "reference_depth_mae_mean_m": safe_mean(self.ref_depth_mae_m),
            "reference_depth_abs_median_m": safe_median(self.ref_depth_median_abs_m),
            "reference_depth_relative_median": safe_median(self.ref_depth_median_relative),
            "reference_depth_bias_median_m": safe_median(self.ref_depth_median_bias_m),
            "reference_depth_scale_log_error_median": safe_median(self.ref_depth_scale_log_error),
            "query_count": float(len(self.query_center_m)),
            "query_center_mean_m": safe_mean(self.query_center_m),
            "query_center_median_m": safe_median(self.query_center_m),
            "query_center_radial_mean_m": safe_mean(self.query_center_radial_m),
            "query_center_radial_median_m": safe_median(self.query_center_radial_m),
            "query_center_tangential_mean_m": safe_mean(self.query_center_tangential_m),
            "query_center_tangential_median_m": safe_median(self.query_center_tangential_m),
            "query_radius_error_mean_m": safe_mean(self.query_radius_m),
            "query_radius_error_median_m": safe_median(self.query_radius_m),
            "query_direction_error_mean_deg": safe_mean(self.query_direction_deg),
            "query_direction_error_median_deg": safe_median(self.query_direction_deg),
            "query_rot_mean_deg": safe_mean(self.query_rot_deg),
            "query_rot_median_deg": safe_median(self.query_rot_deg),
            "query_depth_mae_mean_m": safe_mean(self.query_depth_mae_m),
            "query_depth_abs_median_m": safe_median(self.query_depth_median_abs_m),
            "query_depth_relative_median": safe_median(self.query_depth_median_relative),
            "query_depth_bias_median_m": safe_median(self.query_depth_median_bias_m),
            "query_depth_scale_log_error_median": safe_median(self.query_depth_scale_log_error),
            "alignment_scale_mean": safe_mean(self.scales),
            "alignment_scale_median": safe_median(self.scales),
            "alignment_underconstrained": float(self.num_underconstrained),
            "alignment_depth_valid_views_mean": safe_mean(
                self.depth_scale_valid_views
            ),
            "alignment_depth_scale_log_mad_mean": safe_mean(
                self.depth_scale_log_mads
            ),
        }
