from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Small interface for trainer-independent metric plugins.

    Metrics receive raw model predictions and the already-collated Pi3 batch.
    They own their own accumulation state and return scalar dictionaries from
    :meth:`compute`. The trainer does not need to know whether the metric is a
    pose metric, a reconstruction metric, or something else.
    """

    name: str = "metric"

    def reset(self) -> None:
        """Clear accumulated metric state."""

    @abstractmethod
    def update(
        self,
        prediction: Any,
        batch: list[dict[str, Any]],
        loss_output: Any | None = None,
        *,
        mode: str = "train",
    ) -> None:
        """Accumulate metric state from one model batch."""

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return scalar metric values accumulated so far."""
