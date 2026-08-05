from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from pi3.metrics.utils import extract_prediction, stack_view_tensor
from pi3.models.correspondence import build_query_reference_correspondences

from .base import BaseVisualizer
from .utils import add_title, draw_mask_contour, make_grid, tensor_image_to_uint8


class CorrespondenceVisualizer(BaseVisualizer):
    """Show GT query/reference correspondences used by the auxiliary loss."""

    name = "correspondence"
    required_capabilities = frozenset({"correspondence"})

    def __init__(
        self,
        *,
        patch_size: int = 14,
        max_reference_per_query: int = 1,
        reference_selection: str = "uniform",
        max_pairs: int = 256,
        pair_subsample: str = "uniform",
        depth_abs_tol: float = 0.01,
        depth_rel_tol: float = 0.05,
        dino_layer: int = 17,
        max_draw: int = 120,
        draw_mask: bool = True,
    ):
        self.patch_size = int(patch_size)
        self.max_reference_per_query = int(max_reference_per_query)
        self.reference_selection = str(reference_selection)
        self.max_pairs = int(max_pairs)
        self.pair_subsample = str(pair_subsample)
        self.depth_abs_tol = float(depth_abs_tol)
        self.depth_rel_tol = float(depth_rel_tol)
        self.dino_layer = int(dino_layer)
        self.max_draw = int(max_draw)
        self.draw_mask = bool(draw_mask)

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
        correspondences = build_query_reference_correspondences(
            batch,
            patch_size=self.patch_size,
            max_reference_per_query=self.max_reference_per_query,
            reference_selection=self.reference_selection,
            max_pairs=self.max_pairs,
            pair_subsample=self.pair_subsample,
            depth_abs_tol=self.depth_abs_tol,
            depth_rel_tol=self.depth_rel_tol,
            device=torch.device("cpu"),
        )
        if correspondences.num_pairs == 0:
            return {}

        batch_idx = int(batch_idx)
        selected = correspondences.batch_indices.cpu() == batch_idx
        if not bool(selected.any()):
            return {}

        query_views = correspondences.query_view_indices[selected].cpu().numpy()
        reference_views = correspondences.reference_view_indices[selected].cpu().numpy()
        pair_keys, counts = np.unique(np.stack([query_views, reference_views], axis=1), axis=0, return_counts=True)
        query_idx, reference_idx = pair_keys[int(np.argmax(counts))]
        pair_mask = selected & (correspondences.query_view_indices.cpu() == int(query_idx)) & (
            correspondences.reference_view_indices.cpu() == int(reference_idx)
        )

        indices = torch.nonzero(pair_mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            return {}
        if indices.numel() > self.max_draw:
            if rng is None:
                keep_np = np.linspace(0, indices.numel() - 1, self.max_draw).round().astype(np.int64)
            else:
                keep_np = np.sort(rng.choice(indices.numel(), size=self.max_draw, replace=False).astype(np.int64))
            indices = indices[torch.as_tensor(keep_np, dtype=torch.long)]

        dino_sims = self._dino_similarities(pred, correspondences, indices)

        images = stack_view_tensor(batch, "img")
        masks = stack_view_tensor(batch, "valid_mask").astype(bool)
        query_image = Image.fromarray(tensor_image_to_uint8(images[batch_idx, int(query_idx)]))
        reference_image = Image.fromarray(tensor_image_to_uint8(images[batch_idx, int(reference_idx)]))
        if self.draw_mask:
            draw_mask_contour(query_image, masks[batch_idx, int(query_idx)], color=(255, 230, 0), width=1)
            draw_mask_contour(reference_image, masks[batch_idx, int(reference_idx)], color=(255, 230, 0), width=1)

        match_panel = self._draw_matches(query_image, reference_image, correspondences, indices, dino_sims)
        hist_panel = self._draw_histograms(
            correspondences.depth_errors[indices].detach().cpu().numpy(),
            dino_sims,
            width=match_panel.width,
        )
        query_panel = add_title(query_image, f"query view {int(query_idx)}")
        reference_panel = add_title(reference_image, f"reference view {int(reference_idx)}")
        title = (
            f"b{batch_idx} q{int(query_idx)} -> r{int(reference_idx)} | "
            f"pairs={int(pair_mask.sum())} drawn={int(indices.numel())}"
        )
        if dino_sims is not None and len(dino_sims):
            title += f" | DINO{self.dino_layer} med={float(np.median(dino_sims)):.3f}"

        return {
            "matches": make_grid(
                [
                    add_title(match_panel, title, height=26),
                    hist_panel,
                    make_grid([query_panel, reference_panel], columns=2),
                ],
                columns=1,
            )
        }

    def _dino_similarities(self, pred: dict[str, Any], correspondences, indices: torch.Tensor) -> np.ndarray | None:
        features = pred.get("dino_features", None)
        if features is None:
            return None
        if isinstance(features, dict):
            features = features.get(str(self.dino_layer), features.get(self.dino_layer, None))
        if features is None:
            return None
        features = features.detach().float().cpu()
        q_feat = features[
            correspondences.batch_indices[indices],
            correspondences.query_view_indices[indices],
            correspondences.query_token_indices[indices],
        ]
        r_feat = features[
            correspondences.batch_indices[indices],
            correspondences.reference_view_indices[indices],
            correspondences.reference_token_indices[indices],
        ]
        q_feat = F.normalize(q_feat, dim=-1)
        r_feat = F.normalize(r_feat, dim=-1)
        return (q_feat * r_feat).sum(dim=-1).numpy()

    def _draw_matches(
        self,
        query_image: Image.Image,
        reference_image: Image.Image,
        correspondences,
        indices: torch.Tensor,
        dino_sims: np.ndarray | None,
    ) -> Image.Image:
        wq, hq = query_image.size
        wr, hr = reference_image.size
        canvas = Image.new("RGB", (wq + wr, max(hq, hr)), color=(8, 8, 8))
        canvas.paste(query_image, (0, 0))
        canvas.paste(reference_image, (wq, 0))
        draw = ImageDraw.Draw(canvas)

        qx = correspondences.query_x[indices].detach().cpu().numpy()
        qy = correspondences.query_y[indices].detach().cpu().numpy()
        rx = correspondences.reference_x[indices].detach().cpu().numpy() + wq
        ry = correspondences.reference_y[indices].detach().cpu().numpy()
        for idx, (x0, y0, x1, y1) in enumerate(zip(qx, qy, rx, ry)):
            sim = None if dino_sims is None else float(dino_sims[idx])
            color = self._color_from_similarity(sim)
            draw.line([(float(x0), float(y0)), (float(x1), float(y1))], fill=color, width=1)
            radius = 3
            draw.ellipse((x0 - radius, y0 - radius, x0 + radius, y0 + radius), outline=color, width=2)
            draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), outline=color, width=2)
            if idx < 16:
                draw.text((float(x0) + 4, float(y0) + 2), str(idx), fill=color)
                draw.text((float(x1) + 4, float(y1) + 2), str(idx), fill=color)
        return canvas

    def _draw_histograms(self, depth_errors: np.ndarray, dino_sims: np.ndarray | None, *, width: int) -> Image.Image:
        height = 90
        canvas = Image.new("RGB", (width, height), color=(15, 15, 15))
        draw = ImageDraw.Draw(canvas)
        half = width // 2
        self._draw_hist(draw, depth_errors, (0, 18, half - 6, height - 8), color=(255, 170, 60), title="depth err m")
        if dino_sims is not None and len(dino_sims):
            self._draw_hist(draw, dino_sims, (half + 6, 18, width - 1, height - 8), color=(90, 220, 90), title="DINO sim")
        else:
            draw.text((half + 10, 34), "no DINO features", fill=(180, 180, 180))
        return canvas

    @staticmethod
    def _draw_hist(
        draw: ImageDraw.ImageDraw,
        values: np.ndarray,
        box: tuple[int, int, int, int],
        *,
        color: tuple[int, int, int],
        title: str,
    ) -> None:
        x0, y0, x1, y1 = box
        values = np.asarray(values, dtype=np.float32)
        values = values[np.isfinite(values)]
        draw.text((x0 + 2, 2), title, fill=(235, 235, 235))
        draw.rectangle(box, outline=(80, 80, 80))
        if len(values) == 0:
            return
        bins = min(24, max(4, len(values)))
        hist, edges = np.histogram(values, bins=bins)
        max_count = max(int(hist.max()), 1)
        bin_width = max(1.0, (x1 - x0 - 2) / bins)
        for idx, count in enumerate(hist):
            bar_h = (y1 - y0 - 2) * float(count) / max_count
            bx0 = x0 + 1 + idx * bin_width
            bx1 = x0 + 1 + (idx + 1) * bin_width - 1
            draw.rectangle((bx0, y1 - 1 - bar_h, bx1, y1 - 1), fill=color)
        draw.text((x0 + 2, y1 - 13), f"med {float(np.median(values)):.4f}", fill=(230, 230, 230))

    @staticmethod
    def _color_from_similarity(similarity: float | None) -> tuple[int, int, int]:
        if similarity is None or not np.isfinite(similarity):
            return (60, 200, 255)
        value = float(np.clip((similarity + 1.0) * 0.5, 0.0, 1.0))
        red = int(round(255 * (1.0 - value)))
        green = int(round(255 * value))
        return (red, green, 40)
