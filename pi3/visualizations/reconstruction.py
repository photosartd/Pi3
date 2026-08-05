from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from pi3.metrics.utils import extract_prediction, sample_points, stack_view_tensor, view_bool_mask

from .base import BaseVisualizer
from .utils import add_title, alignment_from_batch, make_grid, pil_from_array


class ReferenceReconstructionVisualizer(BaseVisualizer):
    """Fixed-view projections of aligned reference reconstruction vs GT points."""

    name = "reference_reconstruction"
    required_capabilities = frozenset({"key_query", "object_pose"})

    def __init__(
        self,
        *,
        solve_scale: bool = True,
        voxel_size: float | None = None,
        max_pred_points: int = 12000,
        max_gt_points: int = 12000,
        size: int = 320,
    ):
        self.solve_scale = bool(solve_scale)
        self.voxel_size = voxel_size
        self.max_pred_points = int(max_pred_points)
        self.max_gt_points = int(max_gt_points)
        self.size = int(size)

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
        batch_idx = int(batch_idx)
        alignment, _, _, ref_mask = alignment_from_batch(
            prediction,
            batch,
            batch_idx=batch_idx,
            solve_scale=self.solve_scale,
        )
        if alignment is None:
            return {}

        pred_points = pred["points"].detach().float().cpu().numpy()
        gt_points = stack_view_tensor(batch, "pts3d").astype(np.float64)
        valid_masks = stack_view_tensor(batch, "valid_mask").astype(bool)

        refs = ref_mask[batch_idx]
        if refs.sum() == 0:
            return {}

        pred_cloud_world = pred_points[batch_idx, refs][valid_masks[batch_idx, refs]]
        gt_cloud_object = gt_points[batch_idx, refs][valid_masks[batch_idx, refs]]
        pred_cloud_object = alignment.transform_points(pred_cloud_world)

        pred_cloud = sample_points(
            pred_cloud_object,
            voxel_size=self.voxel_size,
            max_points=self.max_pred_points,
        )
        gt_cloud = sample_points(
            gt_cloud_object,
            voxel_size=self.voxel_size,
            max_points=self.max_gt_points,
        )

        views = [
            ("XY front", (0, 1)),
            ("XZ side", (0, 2)),
            ("YZ top", (1, 2)),
        ]
        panels = [
            add_title(
                self._project_pair(pred_cloud, gt_cloud, axes=axes),
                f"b{batch_idx} {name} | orange GT, cyan pred",
            )
            for name, axes in views
        ]
        return {"orthographic": make_grid(panels, columns=len(panels))}

    def _project_pair(self, pred_points: np.ndarray, gt_points: np.ndarray, *, axes: tuple[int, int]) -> Image.Image:
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:] = (5, 5, 5)
        both = []
        if len(pred_points):
            both.append(pred_points[:, axes])
        if len(gt_points):
            both.append(gt_points[:, axes])
        if not both:
            return pil_from_array(canvas)

        coords_all = np.concatenate(both, axis=0)
        finite = np.isfinite(coords_all).all(axis=1)
        coords_all = coords_all[finite]
        if len(coords_all) == 0:
            return pil_from_array(canvas)

        mins = coords_all.min(axis=0)
        maxs = coords_all.max(axis=0)
        center = 0.5 * (mins + maxs)
        span = float(np.max(maxs - mins))
        if not np.isfinite(span) or span <= 1e-12:
            span = 1.0

        self._paint_cloud(canvas, gt_points[:, axes] if len(gt_points) else gt_points, center, span, (255, 165, 45))
        self._paint_cloud(canvas, pred_points[:, axes] if len(pred_points) else pred_points, center, span, (65, 210, 255))
        return pil_from_array(canvas)

    def _paint_cloud(
        self,
        canvas: np.ndarray,
        coords: np.ndarray,
        center: np.ndarray,
        span: float,
        color: tuple[int, int, int],
    ) -> None:
        if len(coords) == 0:
            return
        coords = np.asarray(coords, dtype=np.float64)
        finite = np.isfinite(coords).all(axis=1)
        coords = coords[finite]
        if len(coords) == 0:
            return
        xy = (coords - center) / span
        pix = np.empty_like(xy)
        pix[:, 0] = (xy[:, 0] * 0.86 + 0.5) * (self.size - 1)
        pix[:, 1] = (0.5 - xy[:, 1] * 0.86) * (self.size - 1)
        pix = np.round(pix).astype(np.int64)
        valid = (pix[:, 0] >= 0) & (pix[:, 0] < self.size) & (pix[:, 1] >= 0) & (pix[:, 1] < self.size)
        canvas[pix[valid, 1], pix[valid, 0]] = np.asarray(color, dtype=np.uint8)
