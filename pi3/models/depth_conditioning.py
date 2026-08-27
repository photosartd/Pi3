"""Lightweight metric-depth conditioning for Pi3 patch tokens.

The adapter deliberately factors metric depth into a dimensionless spatial
map and one metric scale scalar per view.  This lets the patch projection learn
shape/range variation without asking it to encode absolute units implicitly,
while the scalar branch tells the decoder how many metres that shape represents.
Both output projections are zero-initialized, so enabling the module on a Pi3
checkpoint is an exact no-op before its first optimizer update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .dinov2.layers import PatchEmbed


def _probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


class RoleAwareViewDropout(nn.Module):
    """Sample which eligible reference/query views expose a condition."""

    def __init__(
        self,
        *,
        reference_probability: float = 1.0,
        query_probability: float = 0.0,
        eval_reference_probability: float = 1.0,
        eval_query_probability: float = 0.0,
        granularity: str = "view",
    ):
        super().__init__()
        self.reference_probability = _probability(
            "reference_probability", reference_probability
        )
        self.query_probability = _probability(
            "query_probability", query_probability
        )
        self.eval_reference_probability = _probability(
            "eval_reference_probability", eval_reference_probability
        )
        self.eval_query_probability = _probability(
            "eval_query_probability", eval_query_probability
        )
        self.granularity = str(granularity)
        if self.granularity not in {"view", "sample"}:
            raise ValueError("granularity must be view or sample")

    def forward(
        self,
        *,
        eligible: torch.Tensor,
        is_reference: torch.Tensor,
        is_query: torch.Tensor,
        known_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (
            eligible.shape == is_reference.shape == is_query.shape
            and eligible.ndim == 2
        ):
            raise ValueError(
                "eligible/is_reference/is_query must share shape [B, N]"
            )
        if known_override is not None:
            if known_override.shape != eligible.shape:
                raise ValueError("known_override must have shape [B, N]")
            probabilities = known_override.to(dtype=torch.float32)
            return eligible & known_override.bool(), probabilities

        if self.training:
            ref_probability = self.reference_probability
            query_probability = self.query_probability
        else:
            ref_probability = self.eval_reference_probability
            query_probability = self.eval_query_probability
        probabilities = torch.zeros_like(eligible, dtype=torch.float32)
        probabilities = torch.where(
            is_reference, ref_probability, probabilities
        )
        probabilities = torch.where(is_query, query_probability, probabilities)
        probabilities = probabilities * eligible.float()

        # Avoid consuming RNG for the usual deterministic 0/1 evaluation
        # policy. This keeps repeated validation exactly reproducible.
        deterministic = (probabilities == 0.0) | (probabilities == 1.0)
        sampled = probabilities == 1.0
        if not bool(deterministic.all()):
            random_values = (
                torch.rand(
                    probabilities.shape[0],
                    1,
                    device=probabilities.device,
                    dtype=probabilities.dtype,
                ).expand_as(probabilities)
                if self.granularity == "sample"
                else torch.rand_like(probabilities)
            )
            sampled = torch.where(
                deterministic, sampled, random_values < probabilities
            )
        return eligible & sampled, probabilities


@dataclass(frozen=True)
class MetricDepthConditioningOutput:
    tokens: torch.Tensor
    known: torch.Tensor
    scale_m: torch.Tensor
    valid_pixels: torch.Tensor
    statistics: dict[str, torch.Tensor]


class FactoredMetricDepthConditioner(nn.Module):
    """Embed normalized depth+validity and a per-view metric scale."""

    def __init__(
        self,
        *,
        embed_dim: int,
        patch_size: int = 14,
        scale_hidden_dim: int = 128,
        scale_statistic: str = "mean",
        min_valid_pixels: int = 64,
        metric_unit_m: float = 1.0,
        reference_probability: float = 1.0,
        query_probability: float = 0.0,
        eval_reference_probability: float = 1.0,
        eval_query_probability: float = 0.0,
        dropout_granularity: str = "view",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.scale_hidden_dim = int(scale_hidden_dim)
        self.scale_statistic = str(scale_statistic)
        self.min_valid_pixels = int(min_valid_pixels)
        self.metric_unit_m = float(metric_unit_m)
        self.eps = float(eps)
        if self.embed_dim <= 0 or self.patch_size <= 0:
            raise ValueError("embed_dim and patch_size must be positive")
        if self.scale_hidden_dim <= 0 or self.min_valid_pixels <= 0:
            raise ValueError(
                "scale_hidden_dim and min_valid_pixels must be positive"
            )
        if self.scale_statistic not in {"mean", "median"}:
            raise ValueError("scale_statistic must be mean or median")
        if self.metric_unit_m <= 0.0 or self.eps <= 0.0:
            raise ValueError("metric_unit_m and eps must be positive")

        self.spatial_embed = PatchEmbed(
            img_size=224,
            patch_size=self.patch_size,
            in_chans=2,
            embed_dim=self.embed_dim,
        )
        self.scale_embed = nn.Sequential(
            nn.Linear(1, self.scale_hidden_dim),
            nn.GELU(),
            nn.Linear(self.scale_hidden_dim, self.embed_dim),
        )
        nn.init.zeros_(self.spatial_embed.proj.weight)
        nn.init.zeros_(self.spatial_embed.proj.bias)
        nn.init.zeros_(self.scale_embed[-1].weight)
        nn.init.zeros_(self.scale_embed[-1].bias)
        self.view_dropout = RoleAwareViewDropout(
            reference_probability=reference_probability,
            query_probability=query_probability,
            eval_reference_probability=eval_reference_probability,
            eval_query_probability=eval_query_probability,
            granularity=dropout_granularity,
        )

    def _scale(self, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self.scale_statistic == "mean":
            count = valid.sum(dim=(-2, -1)).clamp_min(1)
            return torch.where(valid, depth, 0.0).sum(dim=(-2, -1)) / count
        flattened = depth.flatten(2).masked_fill(~valid.flatten(2), float("nan"))
        return torch.nanmedian(flattened, dim=-1).values

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = values[mask]
        if selected.numel() == 0:
            return values.sum() * 0.0
        return selected.float().mean()

    def forward(
        self,
        *,
        depth_m: torch.Tensor,
        valid_mask: torch.Tensor,
        is_reference: torch.Tensor,
        is_query: torch.Tensor,
        known_override: torch.Tensor | None = None,
    ) -> MetricDepthConditioningOutput:
        if depth_m.ndim != 4:
            raise ValueError("depth_m must have shape [B, N, H, W]")
        if valid_mask.shape != depth_m.shape:
            raise ValueError("valid_mask must match depth_m")
        batch_size, num_views, height, width = depth_m.shape
        if tuple(is_reference.shape) != (batch_size, num_views) or tuple(
            is_query.shape
        ) != (batch_size, num_views):
            raise ValueError("role masks must have shape [B, N]")

        depth = depth_m.to(dtype=torch.float32)
        valid = valid_mask.bool() & torch.isfinite(depth) & (depth > self.eps)
        valid_pixels = valid.sum(dim=(-2, -1))
        scale = self._scale(depth, valid)
        eligible = (
            (valid_pixels >= self.min_valid_pixels)
            & torch.isfinite(scale)
            & (scale > self.eps)
        )
        safe_scale = torch.where(eligible, scale, torch.ones_like(scale))
        known, probabilities = self.view_dropout(
            eligible=eligible,
            is_reference=is_reference.bool(),
            is_query=is_query.bool(),
            known_override=known_override,
        )

        normalized = torch.where(
            valid,
            depth / safe_scale[..., None, None].clamp_min(self.eps),
            0.0,
        )
        spatial_input = torch.stack((normalized, valid.float()), dim=2).reshape(
            batch_size * num_views, 2, height, width
        )
        spatial_tokens = self.spatial_embed(spatial_input)
        log_scale = torch.log(
            safe_scale.clamp_min(self.eps) / self.metric_unit_m
        ).reshape(batch_size * num_views, 1)
        scale_tokens = self.scale_embed(log_scale).unsqueeze(1)
        gate = known.reshape(batch_size * num_views, 1, 1).to(
            dtype=spatial_tokens.dtype
        )
        spatial_tokens = spatial_tokens * gate
        scale_tokens = scale_tokens * gate
        tokens = spatial_tokens + scale_tokens

        ref = is_reference.bool()
        query = is_query.bool()
        conditioned_valid_fraction = self._masked_mean(
            valid.float().mean(dim=(-2, -1)), known
        )
        stats = {
            "metric_depth_conditioned_view_fraction": known.float().mean(),
            "metric_depth_reference_conditioned_fraction": self._masked_mean(
                known.float(), ref
            ),
            "metric_depth_query_conditioned_fraction": self._masked_mean(
                known.float(), query
            ),
            "metric_depth_requested_probability_mean": probabilities.mean(),
            "metric_depth_conditioned_valid_fraction": conditioned_valid_fraction,
            "metric_depth_reference_scale_mean_m": self._masked_mean(
                safe_scale, known & ref
            ),
            "metric_depth_query_scale_mean_m": self._masked_mean(
                safe_scale, known & query
            ),
            "metric_depth_spatial_token_abs_mean": spatial_tokens.detach()
            .float()
            .abs()
            .mean(),
            "metric_depth_scale_token_abs_mean": scale_tokens.detach()
            .float()
            .abs()
            .mean(),
            "metric_depth_spatial_weight_norm": self.spatial_embed.proj.weight.detach()
            .float()
            .norm(),
            "metric_depth_scale_output_weight_norm": self.scale_embed[-1]
            .weight.detach()
            .float()
            .norm(),
        }
        return MetricDepthConditioningOutput(
            tokens=tokens,
            known=known,
            scale_m=safe_scale,
            valid_pixels=valid_pixels,
            statistics=stats,
        )
