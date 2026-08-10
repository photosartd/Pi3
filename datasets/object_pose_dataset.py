"""Generic object-pose dataset assembled from sources, policies, and treatments."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from datasets.object_centric import (
    ObjectDatasetAdapter,
    PlannedObjectView,
    SamplePlan,
    ViewTreatment,
)
from datasets.object_sampling import ObjectSamplingPolicy
from datasets.object_sources import BOPObjectModelCatalog, ObjectViewSource


class ComposableObjectPoseDataset(ObjectDatasetAdapter):
    """Thin orchestration layer with no dataset-format or topology assumptions.

    Physical layout belongs to ``ObjectViewSource`` implementations. Record
    selection belongs to an ``ObjectSamplingPolicy``. This class only resolves
    per-view treatments, validates a plan, and materializes the canonical Pi3
    object observation contract.
    """

    def __init__(
        self,
        *,
        sources: Mapping[str, ObjectViewSource],
        sampling_policy: ObjectSamplingPolicy,
        protocol_name: str,
        dataset_domain: str,
        reference_treatment: ViewTreatment | Mapping[str, Any] | None = None,
        query_treatment: ViewTreatment | Mapping[str, Any] | None = None,
        source_treatments: Mapping[str, Mapping[str, Any]] | None = None,
        object_model_catalog: BOPObjectModelCatalog | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sources = dict(sources)
        if not self.sources:
            raise ValueError("ComposableObjectPoseDataset requires at least one source")
        for name, source in self.sources.items():
            if not isinstance(source, ObjectViewSource):
                raise TypeError(f"Source {name!r} is not an ObjectViewSource")
            if str(source.source_name) != str(name):
                raise ValueError(
                    f"Source key {name!r} must match source_name "
                    f"{source.source_name!r}"
                )
        namespaces = {str(source.object_namespace) for source in self.sources.values()}
        if len(namespaces) != 1:
            raise ValueError(
                f"All sources must share one object namespace, got {sorted(namespaces)}"
            )
        if not isinstance(sampling_policy, ObjectSamplingPolicy):
            raise TypeError("sampling_policy must be an ObjectSamplingPolicy")
        self.sampling_policy = sampling_policy
        self.protocol_name = str(protocol_name)
        self.dataset_domain = str(dataset_domain)
        if not self.protocol_name or not self.dataset_domain:
            raise ValueError("protocol_name and dataset_domain must be non-empty")
        self.dataset_label = self.dataset_domain
        self.object_model_namespace = next(iter(namespaces))
        self.object_model_catalog = object_model_catalog
        if object_model_catalog is not None and (
            object_model_catalog.object_namespace != self.object_model_namespace
        ):
            raise ValueError(
                "Object-model namespace does not match observation sources"
            )
        self.reference_treatment = ViewTreatment.from_config(
            reference_treatment
        ) or ViewTreatment()
        self.query_treatment = ViewTreatment.from_config(
            query_treatment
        ) or ViewTreatment()
        self.source_treatments: dict[str, dict[str, ViewTreatment]] = {}
        for source_name, role_values in dict(source_treatments or {}).items():
            if source_name not in self.sources:
                raise ValueError(
                    f"Treatment refers to unknown source {source_name!r}"
                )
            parsed = {}
            for role, value in dict(role_values).items():
                if role not in {"reference", "query"}:
                    raise ValueError(f"Unknown treatment role {role!r}")
                parsed[role] = ViewTreatment.from_config(value)
            self.source_treatments[str(source_name)] = parsed
        self._requires_visibility_mask_conditioning = False
        for view_role in ("reference", "query"):
            for source_name in self.sampling_policy.source_names_for_role(
                view_role, self.sources
            ):
                if source_name not in self.sources:
                    raise ValueError(
                        f"Policy refers to unknown {view_role} source "
                        f"{source_name!r}"
                    )
                treatment = self.treatment_for(view_role, source_name)
                source = self.sources[source_name]
                source.validate_role_treatment(
                    view_role=view_role, treatment=treatment
                )
                if treatment.mask_condition == "object" or (
                    treatment.mask_condition
                    in {
                        "object_if_repeated",
                        "object_if_repeated_else_probability",
                    }
                    and source.contains_repeated_object_instances
                ) or (
                    treatment.mask_condition
                    == "object_if_repeated_else_probability"
                    and treatment.mask_condition_probability > 0.0
                ):
                    self._requires_visibility_mask_conditioning = True
        self.object_ids = self.sampling_policy.eligible_object_ids(self.sources)
        if not self.object_ids:
            raise ValueError(
                f"Protocol {self.protocol_name!r} has no eligible objects"
            )
        self.this_views_info = {}
        print(
            f"[{self.dataset_domain}/{self.protocol_name}] "
            f"sources={list(self.sources)}, objects={len(self.object_ids)}"
        )

    def __len__(self):
        return int(self.sampling_policy.natural_length(self.sources))

    @property
    def requires_visibility_mask_conditioning(self):
        return self._requires_visibility_mask_conditioning

    def convert_attributes(self):
        """Sources already keep compact immutable catalogues and lazy handles."""

    def supported_frame_counts(self, image_num_range):
        return self.sampling_policy.supported_frame_counts(
            image_num_range, self.sources
        )

    def treatment_for(self, view_role: str, source_name: str) -> ViewTreatment:
        source_specific = self.source_treatments.get(str(source_name), {})
        if view_role in source_specific:
            return source_specific[view_role]
        return (
            self.reference_treatment
            if view_role == "reference"
            else self.query_treatment
        )

    def build_sample_plan(self, index, resolution, rng):
        plan = self.sampling_policy.build_plan(
            index=int(index),
            total_views=int(self.frame_num),
            rng=rng,
            sources=self.sources,
            treatment_for=self.treatment_for,
        )
        self._validate_plan(plan)
        return plan

    def _validate_plan(self, plan: SamplePlan):
        if plan.model_view_count != int(self.frame_num):
            raise ValueError(
                f"Policy returned {plan.model_view_count} views, sampler requested "
                f"{self.frame_num}"
            )
        object_ids = {int(view.record["object_id"]) for view in plan.views}
        if len(object_ids) != 1:
            raise ValueError(
                f"One object-centric sample must use one object ID, got {object_ids}"
            )
        for request in plan.views:
            source_name = str(request.record.get("source_name", ""))
            if source_name not in self.sources:
                raise ValueError(f"Plan record has unknown source {source_name!r}")
            treatment = request.treatment or self.treatment_for(
                request.view_role, source_name
            )
            self.sources[source_name].validate_planned_view(
                request.record,
                view_role=request.view_role,
                treatment=treatment,
            )

    def load_raw_object_view(self, record):
        source_name = str(record["source_name"])
        return self.sources[source_name].load_raw_object_view(record)

    def assemble_processed_object_view(self, state, request: PlannedObjectView):
        view = super().assemble_processed_object_view(state, request)
        record = state["record"]
        treatment = request.treatment or self.treatment_for(
            request.view_role, str(record["source_name"])
        )
        scene_id = int(record.get("scene_id", -1))
        view_id = int(record.get("view_id", record.get("im_id", -1)))
        view.update(
            {
                "dataset": self.dataset_domain,
                "dataset_domain": self.dataset_domain,
                "protocol_name": self.protocol_name,
                "mixture_component": self.protocol_name,
                "object_model_namespace": self.object_model_namespace,
                "object_model_available": bool(
                    self.object_model_catalog.has_object(record["object_id"])
                    if self.object_model_catalog is not None
                    else view.get("object_model_available", False)
                ),
                "source_name": str(record["source_name"]),
                "source_scene_id": np.int64(scene_id),
                "source_subscene_id": np.int64(-1),
                "scene_id": np.int64(scene_id),
                "view_id": np.int64(view_id),
                "im_id": np.int64(view_id),
                "gt_id": np.int64(record.get("gt_id", 0)),
                "coverage_view_id": np.int64(
                    record.get("coverage_view_id", -1)
                ),
                "visib_fract": np.float32(record.get("visib_fract", 1.0)),
                "px_count_visib": np.int64(
                    record.get("px_count_visib", 0)
                ),
                "rgb_treatment": treatment.rgb,
                "depth_treatment": treatment.depth,
                "mask_condition_treatment": treatment.mask_condition,
                "has_repeated_object_id": bool(
                    max(
                        int(record.get("same_object_scene_track_count", 1)),
                        int(record.get("same_object_frame_instance_count", 1)),
                    )
                    > 1
                ),
                "same_object_scene_track_count": np.int64(
                    record.get("same_object_scene_track_count", 1)
                ),
                "same_object_frame_instance_count": np.int64(
                    record.get("same_object_frame_instance_count", 1)
                ),
                "same_object_visible_instance_count": np.int64(
                    record.get("same_object_visible_instance_count", 1)
                ),
                "depth_corrupt": (
                    None
                    if record.get("depth_corrupt") is None
                    else bool(record["depth_corrupt"])
                ),
                "source_shard": str(record.get("relative_path", "")),
                "is_context_reference": bool(
                    request.view_role == "reference"
                    and str(state["reference_source"])
                    not in {"render", "query_scene"}
                ),
                "target_id": np.int64(-1),
                "query_instance_rank": np.int64(-1),
            }
        )
        return view

    def _get_views(self, index, resolution, rng):
        self._current_resolution = resolution
        self._prepare_sample_photometric_augmentation(rng)
        plan = self.build_sample_plan(int(index), resolution, rng)
        return self._materialize_sample_plan(plan, rng=rng)

    def close(self):
        for source in getattr(self, "sources", {}).values():
            source.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
