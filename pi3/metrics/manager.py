from __future__ import annotations

import time
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from .base import BaseMetric
from datasets.base.observation import batch_supports_capabilities


class MetricManager:
    """Owns a configurable list of metric plugins."""

    def __init__(
        self,
        metrics: list[BaseMetric] | None = None,
        *,
        enabled: bool = False,
        train_enabled: bool = True,
        val_enabled: bool = True,
        train_every_n_steps: int = 10,
    ):
        self.metrics = list(metrics or [])
        self.enabled = bool(enabled and self.metrics)
        self.train_enabled = bool(train_enabled)
        self.val_enabled = bool(val_enabled)
        self.train_every_n_steps = max(1, int(train_every_n_steps))
        self.last_update_times: dict[str, float] = {}
        self.routing_counts: dict[str, dict[str, int]] = {}

    @classmethod
    def from_config(cls, cfg: DictConfig | dict | None) -> "MetricManager":
        """Instantiate a metric manager from a Hydra config section."""

        if cfg is None:
            return cls(enabled=False)
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        enabled = bool(cfg.get("enabled", False))
        items = cfg.get("items", {})
        metrics = []
        if enabled and items:
            for _, metric_cfg in items.items():
                if metric_cfg is None or metric_cfg is False:
                    continue
                metrics.append(hydra.utils.instantiate(metric_cfg))
        return cls(
            metrics,
            enabled=enabled,
            train_enabled=bool(cfg.get("train_enabled", True)),
            val_enabled=bool(cfg.get("val_enabled", True)),
            train_every_n_steps=int(cfg.get("train_every_n_steps", 10)),
        )

    def should_update(self, mode: str, step: int | None = None) -> bool:
        """Return whether metrics should run for this mode/step."""

        if not self.enabled:
            return False
        if mode == "train":
            if not self.train_enabled:
                return False
            if step is None:
                return True
            return int(step) % self.train_every_n_steps == 0
        if mode in {"val", "test"}:
            return self.val_enabled
        return False

    def reset(self) -> None:
        """Reset all metrics."""

        self.routing_counts = {
            metric.name: {"eligible": 0, "skipped": 0}
            for metric in self.metrics
        }
        for metric in self.metrics:
            metric.reset()

    def set_context(self, **context: Any) -> None:
        """Pass trainer/runtime metadata to metrics that can use it."""

        for metric in self.metrics:
            if hasattr(metric, "set_context"):
                metric.set_context(**context)

    def update(
        self,
        prediction: Any,
        batch: list[dict[str, Any]],
        loss_output: Any | None = None,
        *,
        mode: str = "train",
    ) -> None:
        """Update all metrics from one batch."""

        if not self.enabled:
            return
        self.last_update_times = {}
        for metric in self.metrics:
            required = getattr(metric, "required_capabilities", frozenset())
            counts = self.routing_counts.setdefault(
                metric.name, {"eligible": 0, "skipped": 0}
            )
            if not batch_supports_capabilities(batch, required):
                counts["skipped"] += 1
                continue
            if not metric.supports_batch(batch):
                counts["skipped"] += 1
                continue
            counts["eligible"] += 1
            start = time.perf_counter()
            metric.update(prediction, batch, loss_output, mode=mode)
            self.last_update_times[metric.name] = time.perf_counter() - start

    def compute(self, *, accelerator: Any | None = None) -> dict[str, float]:
        """Compute and merge scalar outputs from all metrics."""

        output: dict[str, float] = {}
        if not self.enabled:
            return output
        for metric in self.metrics:
            counts = self.routing_counts.get(
                metric.name, {"eligible": 0, "skipped": 0}
            )
            if counts["eligible"] > 0:
                for key, value in metric.compute().items():
                    output[f"{metric.name}/{key}"] = float(value)
                if hasattr(metric, "flush_artifacts"):
                    for key, value in metric.flush_artifacts(accelerator).items():
                        output[f"{metric.name}/{key}"] = float(value)
            output[f"routing/{metric.name}_eligible_batches"] = float(
                counts["eligible"]
            )
            output[f"routing/{metric.name}_skipped_batches"] = float(
                counts["skipped"]
            )
        return output

    def compute_on_batch(
        self,
        prediction: Any,
        batch: list[dict[str, Any]],
        loss_output: Any | None = None,
        *,
        mode: str = "train",
    ) -> dict[str, float]:
        """Compute metrics for one batch without carrying state across calls."""

        self.reset()
        self.update(prediction, batch, loss_output, mode=mode)
        return self.compute()
