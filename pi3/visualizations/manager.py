from __future__ import annotations

from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from .base import BaseVisualizer


class VisualManager:
    """Owns configurable TensorBoard image diagnostics."""

    def __init__(
        self,
        visualizers: list[BaseVisualizer] | None = None,
        *,
        enabled: bool = False,
        train_enabled: bool = False,
        val_enabled: bool = True,
        train_every_n_steps: int = 100,
        val_every_n_epochs: int = 1,
        random_seed: int | None = None,
    ):
        self.visualizers = list(visualizers or [])
        self.enabled = bool(enabled and self.visualizers)
        self.train_enabled = bool(train_enabled)
        self.val_enabled = bool(val_enabled)
        self.train_every_n_steps = max(1, int(train_every_n_steps))
        self.val_every_n_epochs = max(1, int(val_every_n_epochs))
        self.rng = np.random.default_rng(random_seed)
        self.last_render_info: dict[str, Any] = {}

    @classmethod
    def from_config(cls, cfg: DictConfig | dict | None) -> "VisualManager":
        """Instantiate visualizers from a Hydra config section."""

        if cfg is None:
            return cls(enabled=False)
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        enabled = bool(cfg.get("enabled", False))
        items = cfg.get("items", {})
        visualizers = []
        if enabled and items:
            for _, visualizer_cfg in items.items():
                visualizers.append(hydra.utils.instantiate(visualizer_cfg))
        return cls(
            visualizers,
            enabled=enabled,
            train_enabled=bool(cfg.get("train_enabled", False)),
            val_enabled=bool(cfg.get("val_enabled", True)),
            train_every_n_steps=int(cfg.get("train_every_n_steps", 100)),
            val_every_n_epochs=int(cfg.get("val_every_n_epochs", 1)),
            random_seed=cfg.get("random_seed", None),
        )

    def should_render(self, mode: str, *, step: int | None = None, epoch: int | None = None) -> bool:
        """Return whether images should be rendered for this mode."""

        if not self.enabled:
            return False
        if mode == "train":
            if not self.train_enabled:
                return False
            if step is None:
                return True
            return int(step) % self.train_every_n_steps == 0
        if mode in {"val", "test"}:
            if not self.val_enabled:
                return False
            if epoch is None:
                return True
            return int(epoch) % self.val_every_n_epochs == 0
        return False

    def render_iteration(self, mode: str, *, epoch: int | None = None, length: int | None = None) -> int | None:
        """Return which validation/test iterator step should log visuals."""

        if not self.should_render(mode, epoch=epoch):
            return None
        if length is None or int(length) <= 1:
            return 0
        return int(self.rng.integers(int(length)))

    def render(
        self,
        prediction: Any,
        batch: list[dict[str, Any]],
        loss_output: Any | None = None,
        *,
        mode: str = "val",
    ) -> dict[str, Image.Image]:
        """Render and namespace all configured images."""

        images: dict[str, Image.Image] = {}
        if not self.enabled:
            return images
        batch_idx = self._sample_batch_idx(batch)
        self.last_render_info = self._summarize_selection(batch, batch_idx=batch_idx, mode=mode)
        images["selection/target"] = self._selection_image(self.last_render_info)
        for visualizer in self.visualizers:
            for key, image in visualizer.render(
                prediction,
                batch,
                loss_output,
                mode=mode,
                batch_idx=batch_idx,
                rng=self.rng,
            ).items():
                images[f"{visualizer.name}/{key}"] = image
        return images

    def _sample_batch_idx(self, batch: list[dict[str, Any]]) -> int:
        batch_size = self._batch_size(batch)
        if batch_size <= 1:
            return 0
        return int(self.rng.integers(batch_size))

    @staticmethod
    def _batch_size(batch: list[dict[str, Any]]) -> int:
        if not batch or "img" not in batch[0]:
            return 1
        value = batch[0]["img"]
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
        return 1

    @classmethod
    def _summarize_selection(cls, batch: list[dict[str, Any]], *, batch_idx: int, mode: str) -> dict[str, Any]:
        refs = cls._role_views(batch, batch_idx=batch_idx, role_key="is_reference")
        queries = cls._role_views(batch, batch_idx=batch_idx, role_key="is_query")
        return {
            "mode": mode,
            "batch_idx": int(batch_idx),
            "batch_size": cls._batch_size(batch),
            "object_id": cls._scalar_at(batch[0].get("object_id"), batch_idx) if batch else None,
            "reference_views": refs,
            "query_views": queries,
        }

    @classmethod
    def _role_views(cls, batch: list[dict[str, Any]], *, batch_idx: int, role_key: str) -> list[dict[str, Any]]:
        views = []
        for view_idx, view in enumerate(batch):
            if not bool(cls._scalar_at(view.get(role_key, False), batch_idx)):
                continue
            views.append(
                {
                    "view_idx": int(view_idx),
                    "scene_id": cls._scalar_at(view.get("scene_id"), batch_idx),
                    "im_id": cls._scalar_at(view.get("im_id"), batch_idx),
                    "gt_id": cls._scalar_at(view.get("gt_id"), batch_idx),
                    "source": cls._scalar_at(view.get("source"), batch_idx),
                }
            )
        return views

    @staticmethod
    def _scalar_at(value: Any, batch_idx: int) -> Any:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            array = value.reshape(-1)
            return array[int(batch_idx)].item() if len(array) else None
        if isinstance(value, (list, tuple)):
            return value[int(batch_idx)] if len(value) > int(batch_idx) else None
        return value

    @staticmethod
    def _selection_image(info: dict[str, Any]) -> Image.Image:
        refs = info.get("reference_views", [])
        queries = info.get("query_views", [])
        lines = [
            f"mode={info.get('mode')} batch_idx={info.get('batch_idx')}/{info.get('batch_size')} "
            f"object_id={info.get('object_id')}",
            "references: " + VisualManager._format_views(refs),
            "queries: " + VisualManager._format_views(queries),
        ]
        width = 960
        height = 24 + 18 * len(lines)
        image = Image.new("RGB", (width, height), color=(25, 25, 25))
        draw = ImageDraw.Draw(image)
        for row, line in enumerate(lines):
            draw.text((8, 8 + row * 18), line[:160], fill=(240, 240, 240))
        return image

    @staticmethod
    def _format_views(views: list[dict[str, Any]], *, limit: int = 8) -> str:
        parts = []
        for item in views[:limit]:
            parts.append(
                f"v{item.get('view_idx')}:scene{item.get('scene_id')}:im{item.get('im_id')}:gt{item.get('gt_id')}"
            )
        if len(views) > limit:
            parts.append(f"...(+{len(views) - limit})")
        return ", ".join(parts) if parts else "none"
