from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class BaseVisualizer(ABC):
    """Small interface for TensorBoard image diagnostics.

    Visualizers receive the raw model prediction and the collated Pi3 batch.
    They return a dictionary of PIL images. The trainer only logs those images;
    all task-specific drawing stays inside visualizer implementations.
    """

    name: str = "visual"
    required_capabilities: frozenset[str] = frozenset()

    @abstractmethod
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
        """Return TensorBoard-ready PIL images."""
