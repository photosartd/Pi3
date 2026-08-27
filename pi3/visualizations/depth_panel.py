from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from pi3.metrics.utils import extract_prediction, stack_view_tensor, view_bool_mask

from .base import BaseVisualizer
from .utils import (
    add_title,
    choose_view_indices,
    colorize_values,
    make_grid,
    mask_from_depth,
    pil_from_array,
    tensor_image_to_uint8,
)


class DepthPanelVisualizer(BaseVisualizer):
    """Show RGB, GT depth, predicted depth, and depth error."""

    name = "depth_panel"
    required_capabilities = frozenset({"key_query"})

    def __init__(
        self,
        *,
        include_reference: bool = True,
        include_query: bool = True,
        align_scale: bool = True,
        show_conditioning_status: bool = False,
        visualizer_name: str | None = None,
    ):
        self.include_reference = bool(include_reference)
        self.include_query = bool(include_query)
        self.align_scale = bool(align_scale)
        self.show_conditioning_status = bool(show_conditioning_status)
        self.name = str(visualizer_name or type(self).name)

    def render(
        self,
        prediction: Any,
        batch: list[dict[str, Any]],
        loss_output: Any | None = None,
        *,
        mode: str = "val",
        batch_idx: int = 0,
        rng: Any | None = None,
    ) -> dict[str, Image.Image]:
        pred = extract_prediction(prediction)
        pred_depths = pred["local_points"].detach().float().cpu().numpy()[..., 2]
        images = stack_view_tensor(batch, "img")
        gt_depths = stack_view_tensor(batch, "depthmap").astype(np.float32)
        valid_masks = stack_view_tensor(batch, "valid_mask").astype(bool)
        ref_mask = view_bool_mask(batch, "is_reference")
        query_mask = view_bool_mask(batch, "is_query")
        conditioning_known = (
            pred.get("metric_depth_conditioning_known").detach().cpu().numpy()
            if self.show_conditioning_status
            and pred.get("metric_depth_conditioning_known") is not None
            else None
        )
        conditioning_scale = (
            pred.get("metric_depth_conditioning_scale_m").detach().cpu().numpy()
            if self.show_conditioning_status
            and pred.get("metric_depth_conditioning_scale_m") is not None
            else None
        )

        def title(role: str, view_idx: int) -> str:
            value = f"b{batch_idx} {role} {view_idx}"
            if conditioning_known is not None:
                value += " conditioned" if conditioning_known[batch_idx, view_idx] else " unconditioned"
            if conditioning_scale is not None:
                value += f" input-scale={conditioning_scale[batch_idx, view_idx]:.3f}m"
            return value

        outputs: dict[str, Image.Image] = {}
        batch_idx = int(batch_idx)
        if self.include_reference:
            ref_indices = choose_view_indices(np.flatnonzero(ref_mask[batch_idx]), 1, rng=rng)
            if len(ref_indices):
                outputs["reference"] = self._render_one(
                    images[batch_idx, ref_indices[0]],
                    gt_depths[batch_idx, ref_indices[0]],
                    pred_depths[batch_idx, ref_indices[0]],
                    valid_masks[batch_idx, ref_indices[0]],
                    title=title("reference", int(ref_indices[0])),
                )

        if self.include_query:
            query_indices = choose_view_indices(np.flatnonzero(query_mask[batch_idx]), 1, rng=rng)
            if len(query_indices):
                outputs["query"] = self._render_one(
                    images[batch_idx, query_indices[0]],
                    gt_depths[batch_idx, query_indices[0]],
                    pred_depths[batch_idx, query_indices[0]],
                    valid_masks[batch_idx, query_indices[0]],
                    title=title("query", int(query_indices[0])),
                )

        return outputs

    def _render_one(
        self,
        image,
        gt_depth: np.ndarray,
        pred_depth: np.ndarray,
        valid_mask: np.ndarray,
        *,
        title: str,
    ) -> Image.Image:
        rgb = tensor_image_to_uint8(image)
        valid = valid_mask & mask_from_depth(gt_depth) & np.isfinite(pred_depth) & (pred_depth > 1e-8)
        pred_aligned = pred_depth.astype(np.float32, copy=True)
        if self.align_scale and np.any(valid):
            ratios = gt_depth[valid] / np.maximum(pred_depth[valid], 1e-8)
            scale = float(np.median(ratios[np.isfinite(ratios)])) if np.isfinite(ratios).any() else 1.0
            pred_aligned = pred_depth * scale

        depth_valid = valid_mask & mask_from_depth(gt_depth)
        combined = np.concatenate([gt_depth[depth_valid], pred_aligned[depth_valid]]) if np.any(depth_valid) else np.array([])
        if len(combined):
            vmin, vmax = float(np.percentile(combined, 2)), float(np.percentile(combined, 98))
        else:
            vmin, vmax = None, None

        error = np.abs(pred_aligned - gt_depth)
        error_mask = depth_valid & np.isfinite(error)
        error_vmax = float(np.percentile(error[error_mask], 95)) if np.any(error_mask) else None

        panels = [
            add_title(pil_from_array(rgb), f"{title} RGB"),
            add_title(pil_from_array(colorize_values(gt_depth, mask=depth_valid, vmin=vmin, vmax=vmax)), "GT depth"),
            add_title(
                pil_from_array(
                    colorize_values(
                        pred_aligned, mask=depth_valid, vmin=vmin, vmax=vmax
                    )
                ),
                "pred depth aligned" if self.align_scale else "pred depth metric",
            ),
            add_title(pil_from_array(colorize_values(error, mask=error_mask, vmin=0.0, vmax=error_vmax)), "abs depth error"),
            add_title(pil_from_array((valid_mask.astype(np.uint8) * 255)), "object valid mask"),
        ]
        return make_grid(panels, columns=len(panels))
