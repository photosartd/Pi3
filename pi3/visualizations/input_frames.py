from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from pi3.metrics.utils import stack_view_tensor, view_bool_mask

from .base import BaseVisualizer
from .utils import add_title, choose_view_indices, make_grid, tensor_image_to_uint8


class InputFramesVisualizer(BaseVisualizer):
    """Stack a capped set of preprocessed input frames for one view role."""

    def __init__(
        self,
        *,
        role: str,
        max_views: int = 6,
        thumbnail_size: int = 128,
        columns: int = 6,
    ):
        if role not in {"reference", "query"}:
            raise ValueError("role must be 'reference' or 'query'")
        self.role = str(role)
        self.name = f"input_{self.role}_frames"
        self.max_views = min(6, max(1, int(max_views)))
        self.thumbnail_size = max(32, int(thumbnail_size))
        self.columns = max(1, int(columns))

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
        role_mask = view_bool_mask(batch, f"is_{self.role}")
        batch_idx = int(batch_idx)
        indices = np.flatnonzero(role_mask[batch_idx])
        if len(indices) == 0:
            return {}
        indices = choose_view_indices(indices, self.max_views, rng=rng)

        images = stack_view_tensor(batch, "img")
        im_ids = self._optional_view_values(batch, "im_id")
        panels = []
        for view_idx in indices:
            image = Image.fromarray(tensor_image_to_uint8(images[batch_idx, view_idx]))
            image = self._resize_thumbnail(image)
            title = f"b{batch_idx} {self.role} view {int(view_idx)}"
            if im_ids is not None:
                title += f" | im {int(im_ids[batch_idx, view_idx])}"
            panels.append(add_title(image, title))

        return {"grid": make_grid(panels, columns=min(self.columns, len(panels)))}

    def _resize_thumbnail(self, image: Image.Image) -> Image.Image:
        image = image.copy()
        image.thumbnail((self.thumbnail_size, self.thumbnail_size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.thumbnail_size, self.thumbnail_size), color=(5, 5, 5))
        offset = ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2)
        canvas.paste(image, offset)
        return canvas

    @staticmethod
    def _optional_view_values(batch: list[dict[str, Any]], key: str) -> np.ndarray | None:
        if not batch or key not in batch[0]:
            return None
        try:
            return stack_view_tensor(batch, key)
        except Exception:
            return None
