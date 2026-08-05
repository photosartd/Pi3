"""Model-neutral observation contracts and optional dataset capabilities.

The core Pi3 datasets intentionally share a very small contract.  Object pose,
key/query roles, conditioning signals, and paired-query metadata are extensions
of that contract, not requirements imposed on CO3D, ScanNet, TartanAir, or
future scene datasets.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


class ObservationCapability(str, Enum):
    """Optional semantic capabilities exposed by an observation batch."""

    CORE_GEOMETRY = "core_geometry"
    KEY_QUERY = "key_query"
    OBJECT_POSE = "object_pose"
    OBJECT_MODEL = "object_model"
    VISIBILITY_CONDITION = "visibility_condition"
    CORRESPONDENCE = "correspondence"
    PAIRED_QUERY = "paired_query"


CORE_RAW_FIELDS = frozenset(
    {
        "img",
        "depthmap",
        "camera_intrinsics",
        "camera_pose",
        "dataset",
        "label",
        "instance",
    }
)

CAPABILITY_FIELDS: dict[ObservationCapability, frozenset[str]] = {
    ObservationCapability.CORE_GEOMETRY: frozenset(
        {
            "img",
            "depthmap",
            "camera_intrinsics",
            "camera_pose",
            "pts3d",
            "valid_mask",
        }
    ),
    ObservationCapability.KEY_QUERY: frozenset(
        {"view_role", "is_reference", "is_query"}
    ),
    ObservationCapability.OBJECT_POSE: frozenset(
        {"object_id", "T_C_O", "object_visibility_mask"}
    ),
    # Pose supervision and a CAD model are deliberately separate.  Datasets
    # such as MegaPose-GSO can provide metric T_C_O without shipping meshes,
    # diameters, or symmetry metadata.
    ObservationCapability.OBJECT_MODEL: frozenset(
        {"object_model_available"}
    ),
    ObservationCapability.VISIBILITY_CONDITION: frozenset(
        {"visibility_mask_condition", "visibility_mask_known"}
    ),
    ObservationCapability.CORRESPONDENCE: frozenset(
        {
            "depthmap",
            "valid_mask",
            "camera_intrinsics",
            "T_C_O",
            "is_reference",
            "is_query",
        }
    ),
    ObservationCapability.PAIRED_QUERY: frozenset(
        {
            "query_pair_index",
            "is_cropped_query",
            "is_original_query",
            "query_crop_from_original_homography",
            "query_original_T_C_O",
        }
    ),
}


class ObservationContractError(ValueError):
    """Raised when a raw or collated observation violates its contract."""


def normalize_capabilities(
    capabilities: Iterable[ObservationCapability | str],
) -> frozenset[ObservationCapability]:
    return frozenset(ObservationCapability(value) for value in capabilities)


def validate_raw_observation(view: Mapping[str, Any]) -> None:
    """Validate the fields required before :class:`BaseDataset` enrichment."""

    missing = sorted(CORE_RAW_FIELDS.difference(view))
    if missing:
        raise ObservationContractError(
            f"Raw observation is missing required fields: {missing}"
        )

    depth = np.asarray(view["depthmap"])
    if depth.ndim != 2:
        raise ObservationContractError(
            f"depthmap must have shape (H, W), got {depth.shape}"
        )
    intrinsics = np.asarray(view["camera_intrinsics"])
    if intrinsics.shape != (3, 3):
        raise ObservationContractError(
            f"camera_intrinsics must have shape (3, 3), got {intrinsics.shape}"
        )
    pose = np.asarray(view["camera_pose"])
    if pose.shape != (4, 4):
        raise ObservationContractError(
            f"camera_pose must have shape (4, 4), got {pose.shape}"
        )


def _field_truth(value: Any) -> bool:
    """Return whether an explicit capability flag is true for the full batch."""

    if torch.is_tensor(value):
        return bool(value.numel() > 0 and value.bool().all().item())
    if isinstance(value, np.ndarray):
        return bool(value.size > 0 and value.astype(bool).all())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_field_truth(item) for item in value)
    return bool(value)


def batch_supports_capability(
    batch: Sequence[Mapping[str, Any]],
    capability: ObservationCapability | str,
) -> bool:
    """Check a homogeneous multi-view batch for an optional capability."""

    capability = ObservationCapability(capability)
    if not batch:
        return False
    required = CAPABILITY_FIELDS[capability]
    if not all(required.issubset(view) for view in batch):
        return False

    if capability == ObservationCapability.OBJECT_MODEL:
        if not all(_field_truth(view["object_model_available"]) for view in batch):
            return False

    flag = f"capability_{capability.value}"
    explicit = [view[flag] for view in batch if flag in view]
    return not explicit or all(_field_truth(value) for value in explicit)


def batch_supports_capabilities(
    batch: Sequence[Mapping[str, Any]],
    capabilities: Iterable[ObservationCapability | str],
) -> bool:
    return all(batch_supports_capability(batch, item) for item in capabilities)


def infer_batch_capabilities(
    batch: Sequence[Mapping[str, Any]],
) -> frozenset[ObservationCapability]:
    return frozenset(
        capability
        for capability in ObservationCapability
        if batch_supports_capability(batch, capability)
    )


def capability_flags_for_view(
    view: Mapping[str, Any],
) -> dict[str, bool]:
    """Create stable scalar flags for capabilities present in one view."""

    flags = {
        f"capability_{capability.value}": CAPABILITY_FIELDS[capability].issubset(view)
        for capability in ObservationCapability
    }
    if "object_model_available" in view:
        flags[f"capability_{ObservationCapability.OBJECT_MODEL.value}"] = (
            _field_truth(view["object_model_available"])
        )
    return flags
