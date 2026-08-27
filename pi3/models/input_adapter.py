"""Model-specific adapters from the shared multi-view observation contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from datasets.base.observation import (
    ObservationCapability,
    batch_supports_capability,
)


@dataclass(frozen=True)
class AdaptedModelBatch:
    inputs: torch.Tensor
    kwargs: Mapping[str, torch.Tensor]
    role_masks: Mapping[str, torch.Tensor]
    capabilities: Mapping[str, bool]


class ModelInputAdapter(ABC):
    """Translate model-neutral observations to one model's forward contract."""

    @abstractmethod
    def adapt(self, batch: Sequence[Mapping[str, Any]]) -> AdaptedModelBatch:
        pass


class Pi3BatchAdapter(ModelInputAdapter):
    """Stack Pi3 inputs and supply unknown defaults for optional conditions."""

    def __init__(
        self,
        *,
        use_visibility_mask_conditioning: bool = False,
        use_metric_depth_conditioning: bool = False,
    ):
        self.use_visibility_mask_conditioning = bool(
            use_visibility_mask_conditioning
        )
        self.use_metric_depth_conditioning = bool(
            use_metric_depth_conditioning
        )

    @staticmethod
    def _stack_role(
        batch: Sequence[Mapping[str, Any]],
        key: str,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        values = []
        for view in batch:
            value = view.get(key)
            if value is None:
                value = torch.zeros(batch_size, dtype=torch.bool, device=device)
            elif not torch.is_tensor(value):
                value = torch.as_tensor(value, device=device)
            values.append(value.bool())
        return torch.stack(values, dim=1)

    @staticmethod
    def _condition_value(
        view: Mapping[str, Any],
        key: str,
        *,
        image: torch.Tensor,
    ) -> torch.Tensor:
        value = view.get(key)
        if value is None:
            # Unknown is represented by [condition=0, known=0], not by a
            # fabricated known-empty object mask.
            return torch.zeros_like(image[:, 0], dtype=torch.float32)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=image.device)
        return value.to(device=image.device, dtype=torch.float32)

    def adapt(self, batch: Sequence[Mapping[str, Any]]) -> AdaptedModelBatch:
        if not batch:
            raise ValueError("Cannot adapt an empty multi-view batch")
        imgs = torch.stack([view["img"] for view in batch], dim=1)
        intrinsics = torch.stack(
            [view["camera_intrinsics"] for view in batch], dim=1
        )
        kwargs: dict[str, torch.Tensor] = {"intrinsics": intrinsics}
        batch_size = int(imgs.shape[0])
        role_masks = {
            role: self._stack_role(
                batch,
                f"is_{role}",
                batch_size=batch_size,
                device=imgs.device,
            )
            for role in ("reference", "query", "query_context")
        }

        if self.use_visibility_mask_conditioning:
            condition = torch.stack(
                [
                    self._condition_value(
                        view,
                        "visibility_mask_condition",
                        image=view["img"],
                    )
                    for view in batch
                ],
                dim=1,
            )
            known = torch.stack(
                [
                    self._condition_value(
                        view,
                        "visibility_mask_known",
                        image=view["img"],
                    )
                    for view in batch
                ],
                dim=1,
            )
            kwargs["visibility_mask_condition"] = condition
            kwargs["visibility_mask_known"] = known

        if self.use_metric_depth_conditioning:
            kwargs["metric_depth"] = torch.stack(
                [
                    self._condition_value(
                        view, "depthmap", image=view["img"]
                    )
                    for view in batch
                ],
                dim=1,
            )
            kwargs["metric_depth_valid"] = torch.stack(
                [
                    self._condition_value(
                        view, "valid_mask", image=view["img"]
                    ).bool()
                    for view in batch
                ],
                dim=1,
            )
            kwargs["metric_depth_is_reference"] = role_masks["reference"]
            kwargs["metric_depth_is_query"] = role_masks["query"]
        capabilities = {
            capability.value: batch_supports_capability(batch, capability)
            for capability in ObservationCapability
        }
        return AdaptedModelBatch(
            inputs=imgs,
            kwargs=kwargs,
            role_masks=role_masks,
            capabilities=capabilities,
        )
