from __future__ import annotations

from typing import Any

import torch

from pi3.models.ray_conditioning import intrinsics_to_ray_map, pixel_center_grid

from .base import BaseMetric
from .utils import extract_prediction, safe_mean, safe_median


class RayGeometryMetric(BaseMetric):
    """Measure whether predicted local XYZ agrees with each view's intrinsics."""

    name = "ray_geometry"
    required_capabilities = frozenset({"key_query"})
    _ROLES = ("reference", "query", "query_context")

    def __init__(
        self,
        *,
        max_points_per_view: int = 4096,
        min_fit_points: int = 64,
        min_ray_variance: float = 1e-8,
        eps: float = 1e-8,
    ):
        self.max_points_per_view = int(max_points_per_view)
        self.min_fit_points = int(min_fit_points)
        self.min_ray_variance = float(min_ray_variance)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.values = {
            role: {
                "angular_mean_deg": [],
                "angular_median_deg": [],
                "reprojection_mean_px": [],
                "reprojection_median_px": [],
                "fx_relative_error": [],
                "fy_relative_error": [],
                "cx_absolute_error_px": [],
                "cy_absolute_error_px": [],
            }
            for role in self._ROLES
        }
        self.views = {role: 0 for role in self._ROLES}
        self.fit_valid = {role: 0 for role in self._ROLES}

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
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            values.append(value.detach().to(device=device, non_blocking=True))
        return torch.stack(values, dim=1)

    def _subsample_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if self.max_points_per_view <= 0 or len(indices) <= self.max_points_per_view:
            return indices
        positions = torch.linspace(
            0,
            len(indices) - 1,
            self.max_points_per_view,
            device=indices.device,
        ).round().long()
        return indices[positions]

    def _fit_intrinsics(
        self,
        points: torch.Tensor,
        pixels: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[float, float, float, float] | None:
        if len(points) < self.min_fit_points:
            return None
        # The LMGeo pinhole matrices have zero skew and [0, 0, 1] as their
        # final row. Skip the scalar fit rather than silently misreporting a
        # non-canonical calibration.
        canonical = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0]],
            device=intrinsics.device,
            dtype=intrinsics.dtype,
        )
        actual = torch.stack(
            (
                torch.stack((intrinsics[0, 1], intrinsics[1, 0])),
                torch.stack((intrinsics[2, 0], intrinsics[2, 1])),
            )
        )
        if not bool(torch.allclose(actual, canonical, atol=1e-5, rtol=0.0)):
            return None
        if not bool(torch.isclose(
            intrinsics[2, 2],
            torch.ones((), device=intrinsics.device, dtype=intrinsics.dtype),
            atol=1e-5,
            rtol=0.0,
        )):
            return None

        ray_xy = points[:, :2] / points[:, 2:3]

        def fit_axis(ray: torch.Tensor, pixel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
            ray_centered = ray - ray.mean()
            variance_sum = (ray_centered * ray_centered).sum()
            if not bool(torch.isfinite(variance_sum)) or float(variance_sum) <= self.min_ray_variance:
                return None
            focal = (ray_centered * (pixel - pixel.mean())).sum() / variance_sum
            center = pixel.mean() - focal * ray.mean()
            if not bool(torch.isfinite(torch.stack((focal, center))).all()):
                return None
            return focal, center

        fit_x = fit_axis(ray_xy[:, 0], pixels[:, 0])
        fit_y = fit_axis(ray_xy[:, 1], pixels[:, 1])
        if fit_x is None or fit_y is None:
            return None
        fx, cx = fit_x
        fy, cy = fit_y
        return float(fx), float(fy), float(cx), float(cy)

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
        local_points = pred["local_points"].detach().float()
        device = local_points.device
        batch_size, num_views, height, width, _ = local_points.shape

        intrinsics = self._stack_views(batch, "camera_intrinsics", device=device).float()
        valid_masks = self._stack_views(batch, "valid_mask", device=device).bool()
        is_reference = self._stack_views(batch, "is_reference", device=device).bool()
        is_query = self._stack_views(batch, "is_query", device=device).bool()
        is_query_context = self._stack_views(
            batch,
            "is_query_context",
            device=device,
        ).bool()

        expected_rays = intrinsics_to_ray_map(intrinsics, height, width)
        pixels = pixel_center_grid(
            height,
            width,
            device=device,
            dtype=torch.float32,
        )
        pixels_flat = pixels[..., :2].reshape(-1, 2)

        for batch_idx in range(batch_size):
            for view_idx in range(num_views):
                if bool(is_reference[batch_idx, view_idx]):
                    role = "reference"
                elif bool(is_query[batch_idx, view_idx]):
                    role = "query"
                elif bool(is_query_context[batch_idx, view_idx]):
                    role = "query_context"
                else:
                    continue
                self.views[role] += 1

                points_flat = local_points[batch_idx, view_idx].reshape(-1, 3)
                valid = valid_masks[batch_idx, view_idx].reshape(-1)
                valid &= torch.isfinite(points_flat).all(dim=-1)
                valid &= points_flat[:, 2] > self.eps
                indices = torch.nonzero(valid, as_tuple=False).flatten()
                indices = self._subsample_indices(indices)
                if len(indices) == 0:
                    continue

                points = points_flat[indices]
                view_pixels = pixels_flat[indices]
                ray_xy = expected_rays[batch_idx, view_idx].reshape(-1, 2)[indices]

                expected_direction = torch.cat(
                    (ray_xy, torch.ones_like(ray_xy[:, :1])),
                    dim=-1,
                )
                # atan2(||a x b||, a . b) is stable near zero degrees and
                # remains invariant to arbitrary positive point-map scale.
                cross_norm = torch.linalg.cross(
                    points,
                    expected_direction,
                    dim=-1,
                ).norm(dim=-1)
                dot = (points * expected_direction).sum(dim=-1)
                angular_deg = torch.rad2deg(torch.atan2(cross_norm, dot))

                projected = torch.einsum(
                    "ij,nj->ni",
                    intrinsics[batch_idx, view_idx],
                    points,
                )
                projection_valid = projected[:, 2].abs() > self.eps
                projected_xy = projected[:, :2] / projected[:, 2:3].clamp_min(self.eps)
                reprojection_px = (projected_xy - view_pixels).norm(dim=-1)
                finite = (
                    torch.isfinite(angular_deg)
                    & torch.isfinite(reprojection_px)
                    & projection_valid
                )
                if bool(finite.any()):
                    # Retain one mean/median pair per view rather than every
                    # sampled pixel. Full real/ref16 validation can otherwise
                    # accumulate tens of millions of Python floats.
                    view_angular = angular_deg[finite]
                    view_reprojection = reprojection_px[finite]
                    self.values[role]["angular_mean_deg"].append(
                        float(view_angular.mean())
                    )
                    self.values[role]["angular_median_deg"].append(
                        float(view_angular.median())
                    )
                    self.values[role]["reprojection_mean_px"].append(
                        float(view_reprojection.mean())
                    )
                    self.values[role]["reprojection_median_px"].append(
                        float(view_reprojection.median())
                    )

                fitted = self._fit_intrinsics(
                    points[finite],
                    view_pixels[finite],
                    intrinsics[batch_idx, view_idx],
                )
                if fitted is None:
                    continue
                self.fit_valid[role] += 1
                fx, fy, cx, cy = fitted
                target = intrinsics[batch_idx, view_idx]
                target_fx = float(target[0, 0])
                target_fy = float(target[1, 1])
                target_cx = float(target[0, 2])
                target_cy = float(target[1, 2])
                self.values[role]["fx_relative_error"].append(
                    abs(fx - target_fx) / max(abs(target_fx), self.eps)
                )
                self.values[role]["fy_relative_error"].append(
                    abs(fy - target_fy) / max(abs(target_fy), self.eps)
                )
                self.values[role]["cx_absolute_error_px"].append(abs(cx - target_cx))
                self.values[role]["cy_absolute_error_px"].append(abs(cy - target_cy))

    def compute(self) -> dict[str, float]:
        output = {}
        for role in self._ROLES:
            values = self.values[role]
            output.update(
                {
                    f"{role}/views": float(self.views[role]),
                    f"{role}/angular_mean_deg": safe_mean(values["angular_mean_deg"]),
                    f"{role}/angular_median_deg": safe_median(values["angular_median_deg"]),
                    f"{role}/reprojection_mean_px": safe_mean(values["reprojection_mean_px"]),
                    f"{role}/reprojection_median_px": safe_median(values["reprojection_median_px"]),
                    f"{role}/fx_relative_error_mean": safe_mean(values["fx_relative_error"]),
                    f"{role}/fy_relative_error_mean": safe_mean(values["fy_relative_error"]),
                    f"{role}/cx_absolute_error_px_mean": safe_mean(values["cx_absolute_error_px"]),
                    f"{role}/cy_absolute_error_px_mean": safe_mean(values["cy_absolute_error_px"]),
                    f"{role}/intrinsics_fit_valid_fraction": (
                        float(self.fit_valid[role]) / float(self.views[role])
                        if self.views[role]
                        else float("nan")
                    ),
                }
            )
        return output
