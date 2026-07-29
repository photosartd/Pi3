from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from pi3.metrics.utils import stack_view_tensor, view_bool_mask

from .base import BaseVisualizer
from .utils import (
    add_title,
    choose_view_indices,
    make_grid,
    pil_from_array,
    tensor_image_to_uint8,
)


class VisibilityConditionVisualizer(BaseVisualizer):
    """Show which anchor masks are supplied to the model-side conditioner."""

    name = "visibility_condition"

    def __init__(
        self,
        *,
        max_reference_views: int = 3,
        max_query_views: int = 3,
        thumbnail_size: int = 112,
    ):
        self.max_reference_views = max(1, int(max_reference_views))
        self.max_query_views = max(1, int(max_query_views))
        self.thumbnail_size = max(32, int(thumbnail_size))

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
        if not batch or "visibility_mask_condition" not in batch[0]:
            return {}

        batch_idx = int(batch_idx)
        images = stack_view_tensor(batch, "img")
        condition = stack_view_tensor(batch, "visibility_mask_condition").astype(np.float32)
        known = stack_view_tensor(batch, "visibility_mask_known").astype(np.float32)
        object_mask = stack_view_tensor(batch, "object_visibility_mask").astype(np.float32)
        valid_mask = stack_view_tensor(batch, "valid_mask").astype(np.float32)
        ref_mask = view_bool_mask(batch, "is_reference")
        query_mask = view_bool_mask(batch, "is_query")

        indices = []
        indices.extend(
            choose_view_indices(
                np.flatnonzero(ref_mask[batch_idx]),
                self.max_reference_views,
                rng=rng,
            ).tolist()
        )
        indices.extend(
            choose_view_indices(
                np.flatnonzero(query_mask[batch_idx]),
                self.max_query_views,
                rng=rng,
            ).tolist()
        )
        if not indices:
            return {}

        panels = []
        for view_idx in indices:
            role = "ref" if bool(ref_mask[batch_idx, view_idx]) else "query"
            panels.append(
                self._render_view(
                    images[batch_idx, view_idx],
                    condition[batch_idx, view_idx],
                    known[batch_idx, view_idx],
                    object_mask[batch_idx, view_idx],
                    valid_mask[batch_idx, view_idx],
                    title=f"b{batch_idx} {role} v{int(view_idx)}",
                )
            )

        return {"grid": make_grid(panels, columns=min(3, len(panels)))}

    def _render_view(
        self,
        image,
        condition: np.ndarray,
        known: np.ndarray,
        object_mask: np.ndarray,
        valid_mask: np.ndarray,
        *,
        title: str,
    ) -> Image.Image:
        rgb = self._thumb(Image.fromarray(tensor_image_to_uint8(image)))
        condition_panel = self._thumb(pil_from_array(((condition > 0.5).astype(np.uint8) * 255)))
        known_panel = self._thumb(pil_from_array(((known > 0.5).astype(np.uint8) * 255)))
        object_panel = self._thumb(pil_from_array(((object_mask > 0.5).astype(np.uint8) * 255)))
        valid_panel = self._thumb(pil_from_array(((valid_mask > 0.5).astype(np.uint8) * 255)))
        return make_grid(
            [
                add_title(rgb, f"{title} RGB"),
                add_title(condition_panel, "condition"),
                add_title(known_panel, "known"),
                add_title(object_panel, "object mask"),
                add_title(valid_panel, "loss valid"),
            ],
            columns=5,
        )

    def _thumb(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        image.thumbnail((self.thumbnail_size, self.thumbnail_size), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (self.thumbnail_size, self.thumbnail_size), color=(5, 5, 5))
        offset = ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2)
        canvas.paste(image, offset)
        return canvas
