"""Composable object-centric dataset layer built above :mod:`BaseDataset`.

This module deliberately does not change the core dataset contract.  It adds a
raw-object adapter, declarative key/query sample plans, and a dependency-checked
view processor for datasets that have canonical object geometry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from datasets.base.base_dataset import BaseDataset
from datasets.base.observation import ObservationCapability
from datasets.role_photometric import RoleConsistentPhotometricAugmentation


@dataclass(frozen=True)
class RawObjectView:
    """Dataset-native object observation converted to canonical metric units."""

    rgb: np.ndarray
    depthmap: np.ndarray
    object_mask: np.ndarray
    camera_intrinsics: np.ndarray
    T_C_O: np.ndarray
    camera_pose: np.ndarray
    record: Mapping[str, Any]


@dataclass(frozen=True)
class ViewTreatment:
    """Orthogonal model-input and geometry treatment for one physical view.

    The object mask remains available as supervision metadata in every mode.
    These switches only control whether it is also used to alter RGB, restrict
    depth/point supervision, or become a known model-side condition.
    """

    rgb: str = "full"
    depth: str = "object_only"
    mask_condition: str = "none"
    mask_condition_probability: float = 0.0

    def __post_init__(self):
        if self.rgb not in {"full", "object_only"}:
            raise ValueError("ViewTreatment.rgb must be 'full' or 'object_only'")
        if self.depth not in {"full", "object_only"}:
            raise ValueError(
                "ViewTreatment.depth must be 'full' or 'object_only'"
            )
        allowed_conditions = {
            "none",
            "object",
            "object_if_repeated",
            "object_if_repeated_else_probability",
        }
        if self.mask_condition not in allowed_conditions:
            raise ValueError(
                "ViewTreatment.mask_condition must be one of "
                f"{sorted(allowed_conditions)}"
            )
        probability = float(self.mask_condition_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("mask_condition_probability must be in [0, 1]")
        if (
            self.mask_condition != "object_if_repeated_else_probability"
            and probability != 0.0
        ):
            raise ValueError(
                "mask_condition_probability is only valid with "
                "mask_condition='object_if_repeated_else_probability'"
            )

    @classmethod
    def from_config(cls, value: "ViewTreatment | Mapping[str, Any] | None"):
        if value is None or isinstance(value, cls):
            return value
        return cls(**dict(value))


@dataclass(frozen=True)
class PlannedObjectView:
    """One physical view requested by a key/query sampling policy."""

    record: Mapping[str, Any]
    view_role: str
    rgb_masking: bool = False
    treatment: ViewTreatment | None = None
    reference_source: str | None = None
    query_pair_index: int = -1
    model_view_cost: int = 1
    view_updates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.view_role not in {"reference", "query"}:
            raise ValueError(
                f"Physical planned view role must be reference/query, got "
                f"{self.view_role!r}"
            )
        if int(self.model_view_cost) <= 0:
            raise ValueError("model_view_cost must be positive")
        if self.view_role == "reference" and int(self.model_view_cost) != 1:
            raise ValueError("A physical reference must cost exactly one model view")
        if self.treatment is not None and not isinstance(
            self.treatment, ViewTreatment
        ):
            object.__setattr__(
                self, "treatment", ViewTreatment.from_config(self.treatment)
            )


@dataclass(frozen=True)
class SamplePlan:
    """Declarative physical-view plan, independent of image decoding."""

    views: tuple[PlannedObjectView, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        query_seen = False
        for view in self.views:
            query_seen = query_seen or view.view_role == "query"
            if query_seen and view.view_role == "reference":
                raise ValueError(
                    "SamplePlan must keep reference views before query views"
                )

    @property
    def model_view_count(self) -> int:
        return sum(int(view.model_view_cost) for view in self.views)

    @property
    def reference_count(self) -> int:
        return sum(view.view_role == "reference" for view in self.views)

    @property
    def query_record_count(self) -> int:
        return sum(view.view_role == "query" for view in self.views)


class KeyQuerySamplingPolicy:
    """Reusable record selection, count splitting, and plan construction."""

    _STRATEGIES = frozenset(
        {"first", "uniform", "random", "contiguous", "coverage_uniform"}
    )

    @classmethod
    def select_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        count: int,
        strategy: str,
        *,
        rng: np.random.Generator,
        allow_repeat: bool,
    ) -> list[Mapping[str, Any]]:
        strategy = str(strategy)
        if strategy not in cls._STRATEGIES:
            raise ValueError(
                f"selection strategy must be one of {sorted(cls._STRATEGIES)}, "
                f"got {strategy!r}"
            )
        count = int(count)
        if count <= 0:
            return []
        if len(records) == 0:
            raise ValueError("Cannot select records from an empty pool")
        if len(records) < count:
            if not allow_repeat:
                raise ValueError(
                    f"Requested {count} records but only {len(records)} are available. "
                    "Lower the requested count or enable repetition."
                )
            if strategy == "random":
                indices = rng.choice(len(records), size=count, replace=True)
                return [records[int(index)] for index in indices]
            repeats = int(np.ceil(count / len(records)))
            return (list(records) * repeats)[:count]

        if strategy == "coverage_uniform":
            if not all("coverage_view_id" in record for record in records):
                raise ValueError(
                    "coverage_uniform requires coverage_view_id on every record"
                )
            records = sorted(records, key=lambda record: int(record["coverage_view_id"]))
            indices = np.linspace(0, len(records) - 1, count).astype(int)
            return [records[int(index)] for index in indices]
        if strategy == "first":
            return list(records[:count])
        if strategy == "contiguous":
            start = int(rng.integers(0, len(records) - count + 1))
            return list(records[start : start + count])
        if strategy == "random":
            indices = rng.choice(len(records), size=count, replace=False)
            return [records[int(index)] for index in np.sort(indices)]
        indices = np.linspace(0, len(records) - 1, count).astype(int)
        return [records[int(index)] for index in indices]

    @staticmethod
    def sample_counts(
        total_model_views: int,
        *,
        reference_range: tuple[int, int],
        query_range: tuple[int, int],
        reference_available: int,
        query_available: int,
        query_view_cost: int,
        allow_repeat: bool,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        ref_min, ref_max = reference_range
        query_min, query_max = query_range
        if not allow_repeat:
            ref_max = min(ref_max, int(reference_available))
            query_max = min(query_max, int(query_available))
        valid = [
            (reference_count, query_count)
            for reference_count in range(ref_min, ref_max + 1)
            for query_count in range(query_min, query_max + 1)
            if reference_count + int(query_view_cost) * query_count
            == int(total_model_views)
        ]
        if not valid:
            raise ValueError(
                f"Cannot split model_view_count={total_model_views} into "
                f"reference_range={reference_range}, query_range={query_range}, "
                f"query_view_cost={query_view_cost}"
            )
        if int(query_view_cost) == 1:
            # Preserve LMGeoSequence's historical RNG call and ordering for
            # exact seeded sample parity.
            reference_counts = [item[0] for item in valid]
            reference_count = int(rng.choice(reference_counts))
            return reference_count, int(total_model_views) - reference_count
        return valid[int(rng.integers(len(valid)))]

    @staticmethod
    def plan(
        *,
        reference_records: Sequence[Mapping[str, Any]],
        query_records: Sequence[Mapping[str, Any]],
        reference_rgb_masking: bool,
        query_rgb_masking: bool,
        query_view_cost: int = 1,
        metadata: Mapping[str, Any] | None = None,
        reference_updates: Sequence[Mapping[str, Any]] | None = None,
        query_updates: Sequence[Mapping[str, Any]] | None = None,
        reference_treatments: Sequence[ViewTreatment | Mapping[str, Any] | None]
        | None = None,
        query_treatments: Sequence[ViewTreatment | Mapping[str, Any] | None]
        | None = None,
    ) -> SamplePlan:
        reference_updates = reference_updates or [{} for _ in reference_records]
        query_updates = query_updates or [{} for _ in query_records]
        reference_treatments = reference_treatments or [
            None for _ in reference_records
        ]
        query_treatments = query_treatments or [None for _ in query_records]
        if len(reference_updates) != len(reference_records):
            raise ValueError("reference_updates must match reference_records")
        if len(query_updates) != len(query_records):
            raise ValueError("query_updates must match query_records")
        if len(reference_treatments) != len(reference_records):
            raise ValueError(
                "reference_treatments must match reference_records"
            )
        if len(query_treatments) != len(query_records):
            raise ValueError("query_treatments must match query_records")

        views = []
        for record, updates, treatment in zip(
            reference_records, reference_updates, reference_treatments
        ):
            views.append(
                PlannedObjectView(
                    record=record,
                    view_role="reference",
                    rgb_masking=bool(reference_rgb_masking),
                    reference_source=str(record.get("reference_source", "render")),
                    view_updates=updates,
                    treatment=ViewTreatment.from_config(treatment),
                )
            )
        for query_index, (record, updates, treatment) in enumerate(
            zip(query_records, query_updates, query_treatments)
        ):
            views.append(
                PlannedObjectView(
                    record=record,
                    view_role="query",
                    rgb_masking=bool(query_rgb_masking),
                    query_pair_index=int(query_index),
                    model_view_cost=int(query_view_cost),
                    view_updates=updates,
                    treatment=ViewTreatment.from_config(treatment),
                )
            )
        return SamplePlan(tuple(views), dict(metadata or {}))


class ObjectViewTransform(ABC):
    """One dependency-checked stage of object-view processing."""

    name = "transform"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()

    @abstractmethod
    def apply(
        self,
        state: MutableMapping[str, Any],
        *,
        adapter: "ObjectDatasetAdapter",
        request: PlannedObjectView,
        rng: np.random.Generator,
    ) -> None:
        """Mutate the transient processing state."""


class ObjectMaskTransform(ObjectViewTransform):
    name = "object_masking"
    requires = frozenset({"raw_view"})
    provides = frozenset({"masked_view"})

    def apply(self, state, *, adapter, request, rng):
        mask = state["object_mask"]
        source = state["reference_source"]
        treatment = request.treatment
        if treatment is None:
            rgb_object_only = bool(request.rgb_masking) or bool(
                adapter._force_rgb_object_masking(
                    view_role=request.view_role,
                    reference_source=source,
                )
            )
            depth_object_only = adapter._should_depth_mask_view(
                view_role=request.view_role,
                reference_source=source,
            )
        else:
            rgb_object_only = treatment.rgb == "object_only"
            depth_object_only = treatment.depth == "object_only"
        if depth_object_only:
            state["depthmap"] = state["depthmap"].copy()
            state["depthmap"][~mask] = 0.0
        if rgb_object_only:
            state["rgb"] = state["rgb"].copy()
            state["rgb"][~mask] = 0


class DatasetGeometryTransform(ObjectViewTransform):
    name = "dataset_geometry"
    requires = frozenset({"masked_view"})
    provides = frozenset({"geometry_transformed"})

    def apply(self, state, *, adapter, request, rng):
        (
            state["rgb"],
            state["depthmap"],
            state["object_mask"],
            state["camera_intrinsics"],
            state["T_C_O"],
            state["camera_pose"],
            transform_meta,
        ) = adapter._maybe_transform_raw_view(
            record=state["record"],
            rgb=state["rgb"],
            depthmap=state["depthmap"],
            mask=state["object_mask"],
            intrinsics=state["camera_intrinsics"],
            T_C_O=state["T_C_O"],
            camera_pose=state["camera_pose"],
            view_role=request.view_role,
        )
        state["transform_meta"] = dict(transform_meta)


class CropResizeTransform(ObjectViewTransform):
    name = "crop_resize"
    requires = frozenset({"geometry_transformed"})
    provides = frozenset({"resized_view"})

    def apply(self, state, *, adapter, request, rng):
        outputs = adapter._crop_resize_if_necessary(
            state["rgb"],
            state["depthmap"],
            state["camera_intrinsics"],
            adapter._current_resolution,
            rng=rng,
            info=state["record"].get("rgb_path", state["record"]),
            far_mask=np.asarray(state["object_mask"], dtype=np.uint8),
        )
        (
            state["rgb"],
            state["depthmap"],
            state["camera_intrinsics"],
            object_mask,
        ) = outputs
        state["object_mask"] = np.asarray(object_mask > 0, dtype=np.float32)


class RolePhotometricTransform(ObjectViewTransform):
    name = "role_photometric"
    requires = frozenset({"resized_view"})
    provides = frozenset({"augmented_view"})

    def apply(self, state, *, adapter, request, rng):
        state["rgb"] = adapter._apply_sample_photometric_augmentation(
            state["rgb"], request.view_role
        )


class VisibilityConditionTransform(ObjectViewTransform):
    name = "visibility_condition"
    requires = frozenset({"augmented_view"})
    provides = frozenset({"conditioned_view"})

    def apply(self, state, *, adapter, request, rng):
        object_mask = state["object_mask"]
        if request.treatment is not None:
            condition_mode = request.treatment.mask_condition
            multiplicity = max(
                int(state["record"].get("same_object_scene_track_count", 1)),
                int(state["record"].get("same_object_frame_instance_count", 1)),
            )
            known = condition_mode == "object" or (
                condition_mode == "object_if_repeated" and multiplicity > 1
            )
            if condition_mode == "object_if_repeated_else_probability":
                known = multiplicity > 1 or bool(
                    rng.random() < request.treatment.mask_condition_probability
                )
        else:
            known = adapter._should_condition_visibility_view(
                view_role=request.view_role
            )
        condition = (
            object_mask.copy()
            if known
            else np.zeros_like(object_mask, dtype=np.float32)
        )
        if known:
            condition = adapter._corrupt_visibility_condition(
                condition,
                view_role=request.view_role,
                rng=rng,
            )
        state["visibility_mask_condition"] = np.asarray(
            condition, dtype=np.float32
        )
        state["visibility_mask_known"] = np.full_like(
            object_mask, 1.0 if known else 0.0, dtype=np.float32
        )
        state["visibility_mask_condition_applied"] = bool(known)


class ObjectViewProcessor:
    """Execute an ordered set of independently validated processing stages."""

    def __init__(self, transforms: Sequence[ObjectViewTransform] | None = None):
        self.transforms = list(
            transforms
            or (
                ObjectMaskTransform(),
                DatasetGeometryTransform(),
                CropResizeTransform(),
                RolePhotometricTransform(),
                VisibilityConditionTransform(),
            )
        )
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        available = {"raw_view"}
        names = set()
        for transform in self.transforms:
            if transform.name in names:
                raise ValueError(f"Duplicate object-view transform {transform.name!r}")
            names.add(transform.name)
            missing = transform.requires.difference(available)
            if missing:
                raise ValueError(
                    f"Object-view transform {transform.name!r} requires unavailable "
                    f"stages {sorted(missing)}"
                )
            available.update(transform.provides)
        if "conditioned_view" not in available:
            raise ValueError(
                "Object-view transform pipeline must provide 'conditioned_view'"
            )

    def process(
        self,
        adapter: "ObjectDatasetAdapter",
        request: PlannedObjectView,
        *,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        raw = adapter.load_raw_object_view(request.record)
        self._validate_raw_view(raw)
        record = dict(raw.record)
        if request.reference_source is not None:
            record["reference_source"] = request.reference_source
        state: MutableMapping[str, Any] = {
            "record": record,
            "rgb": np.asarray(raw.rgb),
            "depthmap": np.asarray(raw.depthmap, dtype=np.float32),
            "object_mask": np.asarray(raw.object_mask, dtype=bool),
            "camera_intrinsics": np.asarray(
                raw.camera_intrinsics, dtype=np.float32
            ).copy(),
            "T_C_O": np.asarray(raw.T_C_O, dtype=np.float32).copy(),
            "camera_pose": np.asarray(raw.camera_pose, dtype=np.float32).copy(),
            "reference_source": str(
                record.get(
                    "reference_source",
                    "query" if request.view_role == "query" else "render",
                )
            ),
            "transform_meta": {},
        }
        for transform in self.transforms:
            transform.apply(state, adapter=adapter, request=request, rng=rng)
        view = adapter.assemble_processed_object_view(state, request)
        view.update(request.view_updates)
        return view

    @staticmethod
    def _validate_raw_view(raw: RawObjectView) -> None:
        rgb = np.asarray(raw.rgb)
        depth = np.asarray(raw.depthmap)
        mask = np.asarray(raw.object_mask)
        intrinsics = np.asarray(raw.camera_intrinsics)
        T_C_O = np.asarray(raw.T_C_O)
        camera_pose = np.asarray(raw.camera_pose)
        if rgb.ndim != 3 or rgb.shape[:2] != depth.shape:
            raise ValueError(
                f"Raw RGB/depth shape mismatch: rgb={rgb.shape}, depth={depth.shape}"
            )
        if depth.ndim != 2 or mask.shape != depth.shape:
            raise ValueError(
                f"Raw depth/mask shape mismatch: depth={depth.shape}, mask={mask.shape}"
            )
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Raw camera_intrinsics must be 3x3, got {intrinsics.shape}"
            )
        if T_C_O.shape != (4, 4) or camera_pose.shape != (4, 4):
            raise ValueError(
                f"Raw poses must be 4x4, got {T_C_O.shape} and {camera_pose.shape}"
            )
        if not all(
            np.isfinite(value).all()
            for value in (depth, intrinsics, T_C_O, camera_pose)
        ):
            raise ValueError("Raw object view contains non-finite geometry")
        if not np.allclose(
            camera_pose,
            np.linalg.inv(T_C_O),
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError("Raw camera_pose must equal inv(T_C_O)")


class ObjectDatasetAdapter(BaseDataset, ABC):
    """Upper-set of ``BaseDataset`` for canonical object-centric observations."""

    optional_capabilities = frozenset(
        {
            ObservationCapability.KEY_QUERY,
            ObservationCapability.OBJECT_POSE,
            ObservationCapability.OBJECT_MODEL,
            ObservationCapability.VISIBILITY_CONDITION,
            ObservationCapability.CORRESPONDENCE,
        }
    )
    # Class defaults keep lightweight ``__new__``-constructed test adapters
    # useful without requiring filesystem-heavy dataset initialization.
    key_query_sampling_policy = KeyQuerySamplingPolicy()
    object_view_processor = ObjectViewProcessor()

    def __init__(
        self,
        *args,
        object_view_transforms=None,
        photometric_augmentation=False,
        photometric_brightness=(0.7, 1.3),
        photometric_contrast=(0.7, 1.3),
        photometric_saturation=(0.7, 1.3),
        photometric_hue=(-0.1, 0.1),
        photometric_gamma=(0.7, 1.3),
        photometric_jpeg_prob=0.5,
        photometric_jpeg_quality=(20, 100),
        photometric_blur_prob=0.5,
        photometric_blur_resize_ratio=(0.25, 1.0),
        **kwargs,
    ):
        kwargs.setdefault("shuffle", False)
        super().__init__(*args, **kwargs)
        self.key_query_sampling_policy = KeyQuerySamplingPolicy()
        self.object_view_processor = ObjectViewProcessor(object_view_transforms)
        self._configure_role_photometric_augmentation(
            enabled=photometric_augmentation,
            brightness=photometric_brightness,
            contrast=photometric_contrast,
            saturation=photometric_saturation,
            hue=photometric_hue,
            gamma=photometric_gamma,
            jpeg_prob=photometric_jpeg_prob,
            jpeg_quality=photometric_jpeg_quality,
            blur_prob=photometric_blur_prob,
            blur_resize_ratio=photometric_blur_resize_ratio,
        )

    @abstractmethod
    def load_raw_object_view(
        self, record: Mapping[str, Any]
    ) -> RawObjectView:
        """Decode one record and normalize geometry to the canonical contract."""

    def assemble_processed_object_view(
        self,
        state: Mapping[str, Any],
        request: PlannedObjectView,
    ) -> dict[str, Any]:
        """Add generic identities and optional object capabilities.

        Adapters can override this for richer source metadata, as LMGeo does.
        """

        record = state["record"]
        object_id = int(record.get("object_id", -1))
        source = str(record.get("source", record.get("rgb_path", "unknown")))
        label = str(record.get("label", f"object_{object_id:06d}"))
        instance = str(record.get("instance", source))
        return {
            "img": state["rgb"],
            "depthmap": np.asarray(state["depthmap"], dtype=np.float32),
            "camera_intrinsics": np.asarray(
                state["camera_intrinsics"], dtype=np.float32
            ),
            "camera_pose": np.asarray(state["camera_pose"], dtype=np.float32),
            "T_C_O": np.asarray(state["T_C_O"], dtype=np.float32),
            "object_visibility_mask": np.asarray(
                state["object_mask"], dtype=np.float32
            ),
            "visibility_mask_condition": np.asarray(
                state["visibility_mask_condition"], dtype=np.float32
            ),
            "visibility_mask_known": np.asarray(
                state["visibility_mask_known"], dtype=np.float32
            ),
            "visibility_mask_condition_applied": bool(
                state.get("visibility_mask_condition_applied", False)
            ),
            "dataset": str(getattr(self, "dataset_label", type(self).__name__)),
            "label": label,
            "instance": instance,
            "source": source,
            "object_id": np.int64(object_id),
            "object_model_available": bool(
                record.get("object_model_available", False)
            ),
            "view_role": request.view_role,
            "is_reference": bool(request.view_role == "reference"),
            "is_query": bool(request.view_role == "query"),
            "is_query_context": False,
            "is_cropped_query": False,
            "is_original_query": False,
            "query_pair_index": np.int64(request.query_pair_index),
            "reference_source": state["reference_source"],
            **dict(state.get("transform_meta", {})),
        }

    def build_sample_plan(
        self,
        index: int,
        resolution: Sequence[int],
        rng: np.random.Generator,
    ) -> SamplePlan:
        """Return a plan for generic adapters that do not override ``_get_views``."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement build_sample_plan() or _get_views()"
        )

    def _get_views(self, index, resolution, rng):
        self._current_resolution = resolution
        self._prepare_sample_photometric_augmentation(rng)
        return self._materialize_sample_plan(
            self.build_sample_plan(int(index), resolution, rng),
            rng=rng,
        )

    def _force_rgb_object_masking(
        self, *, view_role: str, reference_source: str
    ) -> bool:
        return False

    def _should_depth_mask_view(
        self, *, view_role: str, reference_source: str
    ) -> bool:
        return bool(getattr(self, "depth_masking", False))

    def _maybe_transform_raw_view(
        self,
        *,
        record,
        rgb,
        depthmap,
        mask,
        intrinsics,
        T_C_O,
        camera_pose,
        view_role,
    ):
        return rgb, depthmap, mask, intrinsics, T_C_O, camera_pose, {}

    def _configure_role_photometric_augmentation(self, *, enabled, **kwargs):
        self._role_photometric = RoleConsistentPhotometricAugmentation(
            enabled=enabled,
            mode=getattr(self, "mode", "train"),
            **kwargs,
        )
        self.photometric_augmentation_requested = bool(enabled)
        self.photometric_augmentation = bool(self._role_photometric.enabled)

    def _ensure_role_photometric_augmentation(self):
        """Support lightweight legacy/test adapters made without ``__init__``."""

        if hasattr(self, "_role_photometric"):
            return
        self._configure_role_photometric_augmentation(
            enabled=bool(getattr(self, "photometric_augmentation", False)),
            brightness=getattr(self, "photometric_brightness", (0.7, 1.3)),
            contrast=getattr(self, "photometric_contrast", (0.7, 1.3)),
            saturation=getattr(self, "photometric_saturation", (0.7, 1.3)),
            hue=getattr(self, "photometric_hue", (-0.1, 0.1)),
            gamma=getattr(self, "photometric_gamma", (0.7, 1.3)),
            jpeg_prob=getattr(self, "photometric_jpeg_prob", 0.5),
            jpeg_quality=getattr(
                self, "photometric_jpeg_quality", (20, 100)
            ),
            blur_prob=getattr(self, "photometric_blur_prob", 0.5),
            blur_resize_ratio=getattr(
                self, "photometric_blur_resize_ratio", (0.25, 1.0)
            ),
        )
        legacy_specs = self.__dict__.pop(
            "_legacy_photometric_role_specs", None
        )
        if legacy_specs is not None:
            self._role_photometric.role_specs = legacy_specs

    @property
    def _photometric_role_specs(self):
        self._ensure_role_photometric_augmentation()
        return self._role_photometric.role_specs

    @_photometric_role_specs.setter
    def _photometric_role_specs(self, value):
        if hasattr(self, "_role_photometric"):
            self._role_photometric.role_specs = dict(value)
        else:
            self.__dict__["_legacy_photometric_role_specs"] = dict(value)

    def _prepare_sample_photometric_augmentation(self, rng):
        self._ensure_role_photometric_augmentation()
        self._role_photometric.begin_sample(rng)

    def _apply_sample_photometric_augmentation(self, image, view_role):
        self._ensure_role_photometric_augmentation()
        return self._role_photometric.apply(image, view_role)

    def _should_condition_visibility_view(self, *, view_role: str) -> bool:
        return False

    def _corrupt_visibility_condition(self, condition, *, view_role, rng):
        return condition

    def _load_view(self, record, rgb_masking, view_role):
        return self.object_view_processor.process(
            self,
            PlannedObjectView(
                record=record,
                view_role=str(view_role),
                rgb_masking=bool(rgb_masking),
                reference_source=record.get("reference_source"),
            ),
            rng=self._rng,
        )

    def _load_query_views(self, record):
        return [self._load_view(record, rgb_masking=False, view_role="query")]

    def _materialize_sample_plan(
        self,
        plan: SamplePlan,
        *,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        self.this_views_info = dict(plan.metadata)
        output = []
        for request in plan.views:
            if request.view_role == "query":
                record = dict(request.record)
                if request.reference_source is not None:
                    record["reference_source"] = request.reference_source
                if type(self)._load_query_views is ObjectDatasetAdapter._load_query_views:
                    query_views = [
                        self.object_view_processor.process(self, request, rng=rng)
                    ]
                else:
                    query_views = self._load_query_views(record)
                if len(query_views) != int(request.model_view_cost):
                    raise RuntimeError(
                        f"SamplePlan expected query record to cost "
                        f"{request.model_view_cost} model views, got {len(query_views)}"
                    )
                for view in query_views:
                    view["query_pair_index"] = np.int64(
                        request.query_pair_index
                    )
                    view.update(request.view_updates)
                output.extend(query_views)
            else:
                output.append(
                    self.object_view_processor.process(self, request, rng=rng)
                )
        if len(output) != plan.model_view_count:
            raise RuntimeError(
                f"SamplePlan declared {plan.model_view_count} model views but "
                f"materialized {len(output)}"
            )
        return output

    def _request_for_record(
        self,
        record: Mapping[str, Any],
        *,
        view_role: str,
        rgb_masking: bool,
    ) -> PlannedObjectView:
        return PlannedObjectView(
            record=record,
            view_role=view_role,
            rgb_masking=rgb_masking,
            reference_source=record.get("reference_source"),
        )
