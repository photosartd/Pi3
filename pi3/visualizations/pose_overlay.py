from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from pi3.metrics.utils import stack_view_tensor, view_bool_mask

from .base import BaseVisualizer
from .utils import (
    add_title,
    alignment_from_batch,
    choose_view_indices,
    draw_mask_contour,
    draw_pose_projection,
    make_grid,
    object_id_at,
    object_bbox_corners,
    tensor_image_to_uint8,
)
from pi3.metrics.utils import BopModelCache


class QueryPoseOverlayVisualizer(BaseVisualizer):
    """Draw GT and predicted object poses over query frames."""

    name = "query_pose_overlay"

    def __init__(
        self,
        data_root: str,
        *,
        models_folder: str = "models_eval",
        solve_scale: bool = True,
        max_model_points: int = 20000,
        max_query_views: int = 3,
    ):
        self.model_cache = BopModelCache(
            data_root,
            models_folder=models_folder,
            max_model_points=max_model_points,
        )
        self.solve_scale = bool(solve_scale)
        self.max_query_views = max(1, int(max_query_views))

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
        batch_idx = int(batch_idx)
        alignment, pred_T_W_C, gt_T_C_O, _ = alignment_from_batch(
            prediction,
            batch,
            batch_idx=batch_idx,
            solve_scale=self.solve_scale,
        )
        if alignment is None:
            return {}

        query_mask = view_bool_mask(batch, "is_query")
        query_indices = choose_view_indices(
            np.flatnonzero(query_mask[batch_idx]),
            self.max_query_views,
            rng=rng,
        )
        if len(query_indices) == 0:
            return {}

        images = stack_view_tensor(batch, "img")
        intrinsics = stack_view_tensor(batch, "camera_intrinsics").astype(np.float64)
        valid_masks = stack_view_tensor(batch, "valid_mask").astype(bool)

        obj_id = object_id_at(batch, batch_idx)
        model_points = self.model_cache.points(obj_id)
        bbox_corners = object_bbox_corners(model_points)
        axis_length = 0.35 * self.model_cache.diameter(obj_id)

        panels = []
        for view_idx in query_indices:
            image = Image.fromarray(tensor_image_to_uint8(images[batch_idx, view_idx]))
            pred_T_C_O = alignment.object_to_camera_pose(pred_T_W_C[batch_idx, view_idx])
            K = intrinsics[batch_idx, view_idx]

            draw_pose_projection(
                image,
                T_C_O=gt_T_C_O[batch_idx, view_idx],
                K=K,
                bbox_corners=bbox_corners,
                axis_length=axis_length,
                bbox_color=(40, 255, 90),
                width=2,
            )
            draw_pose_projection(
                image,
                T_C_O=pred_T_C_O,
                K=K,
                bbox_corners=bbox_corners,
                axis_length=axis_length,
                bbox_color=(255, 80, 80),
                width=2,
            )
            draw_mask_contour(image, valid_masks[batch_idx, view_idx], color=(255, 230, 0), width=1)
            panels.append(add_title(image, f"b{batch_idx} obj {obj_id} query {int(view_idx)} | green GT, red pred"))

        return {"queries": make_grid(panels, columns=len(panels))}
