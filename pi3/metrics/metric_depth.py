"""Diagnostics for the optional factored metric-depth condition."""

from __future__ import annotations

from typing import Any

import torch

from pi3.models.ray_conditioning import intrinsics_to_ray_map

from .base import BaseMetric
from .utils import extract_prediction, recall_below, safe_mean, safe_median


class MetricDepthConditioningMetric(BaseMetric):
    """Measure metric local-depth/center accuracy by role and condition state.

    These diagnostics do not replace aligned 6-DoF metrics. They answer the
    narrower causal question introduced by depth conditioning: did conditioned
    references recover their metric camera-space range and object centre, and
    how do unconditioned queries compare?
    """

    name = "metric_depth"
    required_capabilities = frozenset({"core_geometry", "key_query"})
    _ROLES = ("reference", "query")
    _STATES = ("all", "conditioned", "unconditioned")
    _FIELDS = (
        "z_mae_mean_m",
        "z_mae_median_m",
        "z_relative_mae",
        "scale_ratio",
        "scale_abs_log_error",
        "object_center_error_m",
        "gt_scale_m",
        "pred_scale_m",
    )

    def __init__(self, *, eps: float = 1e-6):
        self.eps = float(eps)
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        self.reset()

    def reset(self) -> None:
        self.values = {
            role: {
                state: {field: [] for field in self._FIELDS}
                for state in self._STATES
            }
            for role in self._ROLES
        }
        self.views = {
            role: {state: 0 for state in self._STATES}
            for role in self._ROLES
        }

    @staticmethod
    def _stack(
        batch: list[dict[str, Any]], key: str, *, device: torch.device
    ) -> torch.Tensor:
        values = []
        for view in batch:
            value = view[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            values.append(value.detach().to(device=device, non_blocking=True))
        return torch.stack(values, dim=1)

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
        if "metric_depth_conditioning_known" not in pred:
            raise ValueError(
                "MetricDepthConditioningMetric requires a model with "
                "use_metric_depth_conditioning=true"
            )
        local_points = pred["local_points"].detach().float()
        known = pred["metric_depth_conditioning_known"].detach().bool()
        device = local_points.device
        depth = self._stack(batch, "depthmap", device=device).float()
        valid = self._stack(batch, "valid_mask", device=device).bool()
        intrinsics = self._stack(
            batch, "camera_intrinsics", device=device
        ).float()
        roles = {
            "reference": self._stack(
                batch, "is_reference", device=device
            ).bool(),
            "query": self._stack(batch, "is_query", device=device).bool(),
        }
        if known.shape != depth.shape[:2]:
            raise ValueError("Conditioning known mask must have shape [B, N]")

        height, width = depth.shape[-2:]
        rays = intrinsics_to_ray_map(intrinsics, height, width)
        for batch_idx in range(depth.shape[0]):
            for view_idx in range(depth.shape[1]):
                role = next(
                    (
                        name
                        for name, role_mask in roles.items()
                        if bool(role_mask[batch_idx, view_idx])
                    ),
                    None,
                )
                if role is None:
                    continue
                is_conditioned = bool(known[batch_idx, view_idx])
                states = (
                    "all",
                    "conditioned" if is_conditioned else "unconditioned",
                )
                for state in states:
                    self.views[role][state] += 1

                gt_z = depth[batch_idx, view_idx]
                pred_xyz = local_points[batch_idx, view_idx]
                view_valid = valid[batch_idx, view_idx]
                view_valid &= torch.isfinite(gt_z) & (gt_z > self.eps)
                view_valid &= torch.isfinite(pred_xyz).all(dim=-1)
                view_valid &= pred_xyz[..., 2] > self.eps
                if not bool(view_valid.any()):
                    continue
                gt_values = gt_z[view_valid]
                pred_values = pred_xyz[..., 2][view_valid]
                absolute = (pred_values - gt_values).abs()
                gt_scale = gt_values.mean()
                pred_scale = pred_values.mean()
                scale_ratio = pred_scale / gt_scale.clamp_min(self.eps)

                ray_xy = rays[batch_idx, view_idx][view_valid]
                gt_xyz = torch.cat(
                    (ray_xy * gt_values[:, None], gt_values[:, None]), dim=-1
                )
                gt_center = gt_xyz.mean(dim=0)
                pred_center = pred_xyz[view_valid].mean(dim=0)
                measurements = {
                    "z_mae_mean_m": float(absolute.mean()),
                    "z_mae_median_m": float(absolute.median()),
                    "z_relative_mae": float(
                        (absolute / gt_values.clamp_min(self.eps)).mean()
                    ),
                    "scale_ratio": float(scale_ratio),
                    "scale_abs_log_error": float(
                        torch.log(scale_ratio.clamp_min(self.eps)).abs()
                    ),
                    "object_center_error_m": float(
                        torch.linalg.vector_norm(pred_center - gt_center)
                    ),
                    "gt_scale_m": float(gt_scale),
                    "pred_scale_m": float(pred_scale),
                }
                for state in states:
                    for key, value in measurements.items():
                        self.values[role][state][key].append(value)

    def compute(self) -> dict[str, float]:
        output = {}
        for role in self._ROLES:
            total = self.views[role]["all"]
            conditioned = self.views[role]["conditioned"]
            output[f"{role}/conditioned_fraction"] = (
                float(conditioned) / float(total) if total else float("nan")
            )
            for state in self._STATES:
                values = self.values[role][state]
                prefix = f"{role}/{state}"
                output[f"{prefix}/views"] = float(self.views[role][state])
                for field in self._FIELDS:
                    output[f"{prefix}/{field}_mean"] = safe_mean(values[field])
                    output[f"{prefix}/{field}_median"] = safe_median(values[field])
                output[f"{prefix}/z_mae_mean_recall_1cm"] = recall_below(
                    values["z_mae_mean_m"], 0.01
                )
                output[f"{prefix}/z_mae_mean_recall_5cm"] = recall_below(
                    values["z_mae_mean_m"], 0.05
                )
                output[f"{prefix}/center_recall_5cm"] = recall_below(
                    values["object_center_error_m"], 0.05
                )
                output[f"{prefix}/center_recall_10cm"] = recall_below(
                    values["object_center_error_m"], 0.10
                )
        return output
