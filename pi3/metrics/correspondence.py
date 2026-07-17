from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from pi3.models.correspondence import build_query_reference_correspondences
from pi3.utils.geometry import homogenize_points

from .base import BaseMetric
from .utils import extract_prediction, finite_float, safe_mean, safe_median, safe_percentile


class CorrespondenceMetric(BaseMetric):
    """Diagnostic GT correspondence consistency without affecting gradients."""

    name = "correspondence"

    def __init__(
        self,
        *,
        patch_size: int = 14,
        max_reference_per_query: int = 1,
        reference_selection: str = "uniform",
        max_pairs: int = 512,
        pair_subsample: str = "uniform",
        depth_abs_tol: float = 0.01,
        depth_rel_tol: float = 0.05,
        huber_beta: float = 0.01,
        use_dino_weights: bool = False,
        dino_layer: int = 17,
        dino_center: float = 0.2,
        dino_temperature: float = 0.1,
        min_dino_weight: float = 0.05,
    ):
        self.patch_size = int(patch_size)
        self.max_reference_per_query = int(max_reference_per_query)
        self.reference_selection = str(reference_selection)
        self.max_pairs = int(max_pairs)
        self.pair_subsample = str(pair_subsample)
        self.depth_abs_tol = float(depth_abs_tol)
        self.depth_rel_tol = float(depth_rel_tol)
        self.huber_beta = float(huber_beta)
        self.use_dino_weights = bool(use_dino_weights)
        self.dino_layer = int(dino_layer)
        self.dino_center = float(dino_center)
        self.dino_temperature = max(float(dino_temperature), 1e-6)
        self.min_dino_weight = float(min_dino_weight)
        self.reset()

    def reset(self) -> None:
        self.geo_l2: list[float] = []
        self.huber: list[float] = []
        self.weighted_huber: list[float] = []
        self.depth_errors: list[float] = []
        self.dino_similarities: list[float] = []
        self.dino_weights: list[float] = []
        self.num_pairs = 0
        self.num_batches = 0
        self.num_batches_with_pairs = 0

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
        pred_points = self._normalized_global_points(pred, batch)
        device = pred_points.device
        self.num_batches += 1

        correspondences = build_query_reference_correspondences(
            batch,
            patch_size=self.patch_size,
            max_reference_per_query=self.max_reference_per_query,
            reference_selection=self.reference_selection,
            max_pairs=self.max_pairs,
            pair_subsample=self.pair_subsample,
            depth_abs_tol=self.depth_abs_tol,
            depth_rel_tol=self.depth_rel_tol,
            device=device,
        )
        if correspondences.num_pairs == 0:
            return

        q_points = pred_points[
            correspondences.batch_indices,
            correspondences.query_view_indices,
            correspondences.query_y,
            correspondences.query_x,
        ]
        r_points = pred_points[
            correspondences.batch_indices,
            correspondences.reference_view_indices,
            correspondences.reference_y,
            correspondences.reference_x,
        ]
        deltas = q_points.float() - r_points.float()
        geo_l2 = deltas.norm(dim=-1)
        per_pair_huber = F.smooth_l1_loss(
            q_points.float(),
            r_points.float(),
            reduction="none",
            beta=self.huber_beta,
        ).mean(dim=-1)

        weights = torch.ones_like(per_pair_huber)
        dino_similarity = self._dino_similarity(pred, correspondences, device=device)
        if self.use_dino_weights and dino_similarity is not None:
            weights = torch.sigmoid((dino_similarity - self.dino_center) / self.dino_temperature)
            weights = weights.clamp_min(self.min_dino_weight).detach()

        weighted_huber = (per_pair_huber * weights).sum() / weights.sum().clamp_min(1e-6)

        self.num_pairs += correspondences.num_pairs
        self.num_batches_with_pairs += 1
        self.geo_l2.extend(self._tolist(geo_l2))
        self.huber.extend(self._tolist(per_pair_huber))
        self.weighted_huber.append(finite_float(weighted_huber.detach().cpu()))
        self.depth_errors.extend(self._tolist(correspondences.depth_errors.float()))
        if dino_similarity is not None:
            self.dino_similarities.extend(self._tolist(dino_similarity))
        self.dino_weights.extend(self._tolist(weights))

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

    def _normalized_global_points(self, pred: dict[str, torch.Tensor], batch: list[dict[str, Any]]) -> torch.Tensor:
        local_points = pred["local_points"].detach().float()
        camera_poses = pred["camera_poses"].detach().float()
        device = local_points.device
        masks = self._stack_views(batch, "valid_mask", device=device).bool()

        batch_size, num_views, height, width, _ = local_points.shape
        masked_points = local_points.clone()
        masked_points[~masks] = 0.0
        distances = masked_points.reshape(batch_size, num_views, -1, 3).norm(dim=-1)
        norm_factor = distances.sum(dim=(1, 2)) / masks.float().sum(dim=(1, 2, 3)).clamp_min(1e-8)

        normalized_local_points = local_points / norm_factor[:, None, None, None, None]
        normalized_camera_poses = camera_poses.clone()
        normalized_camera_poses[..., :3, 3] /= norm_factor[:, None, None]
        return torch.einsum(
            "bnij,bnhwj->bnhwi",
            normalized_camera_poses,
            homogenize_points(normalized_local_points),
        )[..., :3]

    def _dino_similarity(self, pred: dict[str, torch.Tensor], correspondences, *, device: torch.device) -> torch.Tensor | None:
        features = pred.get("dino_features", None)
        if features is None:
            return None
        if isinstance(features, dict):
            features = features.get(str(self.dino_layer), features.get(self.dino_layer, None))
        if features is None:
            return None
        features = features.detach().to(device=device, dtype=torch.float32)
        q_feat = features[
            correspondences.batch_indices,
            correspondences.query_view_indices,
            correspondences.query_token_indices,
        ]
        r_feat = features[
            correspondences.batch_indices,
            correspondences.reference_view_indices,
            correspondences.reference_token_indices,
        ]
        q_feat = F.normalize(q_feat, dim=-1)
        r_feat = F.normalize(r_feat, dim=-1)
        return (q_feat * r_feat).sum(dim=-1).detach()

    @staticmethod
    def _tolist(values: torch.Tensor) -> list[float]:
        return [finite_float(value) for value in values.detach().float().cpu().reshape(-1).tolist()]

    def compute(self) -> dict[str, float]:
        return {
            "num_pairs": float(self.num_pairs),
            "batches": float(self.num_batches),
            "batches_with_pairs": float(self.num_batches_with_pairs),
            "geo_l2_mean": safe_mean(self.geo_l2),
            "geo_l2_median": safe_median(self.geo_l2),
            "geo_l2_p90": safe_percentile(self.geo_l2, 90),
            "huber_mean": safe_mean(self.huber),
            "weighted_huber_mean": safe_mean(self.weighted_huber),
            "depth_error_mean": safe_mean(self.depth_errors),
            "dino_similarity_mean": safe_mean(self.dino_similarities),
            "dino_weight_mean": safe_mean(self.dino_weights),
        }
