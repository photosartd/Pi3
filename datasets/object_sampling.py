"""Composable sample-topology strategies for object-centric datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from datasets.object_centric import (
    KeyQuerySamplingPolicy,
    SamplePlan,
    ViewTreatment,
)
from datasets.object_geometry import (
    GeometryFeatureIndex,
    GeometryPlanIndex,
    normalized_focal_scalar,
    normalized_focal_shape,
    object_preserving_crop_spec,
)
from datasets.object_sources import ObjectViewGroup, ObjectViewSource


TreatmentResolver = Callable[[str, str], ViewTreatment]


def _range(value, name):
    if isinstance(value, int):
        lower = upper = int(value)
    else:
        lower, upper = (int(item) for item in value)
    if lower <= 0 or upper < lower:
        raise ValueError(f"{name} must be a positive [min, max] range")
    return lower, upper


def _tag_records(records, source_name, reference_source=None):
    output = []
    for record in records:
        item = dict(record)
        item["source_name"] = str(source_name)
        item["reference_source"] = str(reference_source or source_name)
        output.append(item)
    return output


class ObjectSamplingPolicy(ABC):
    """Strategy that chooses physical records and returns a declarative plan."""

    @abstractmethod
    def supported_frame_counts(
        self,
        image_num_range: Sequence[int],
        sources: Mapping[str, ObjectViewSource],
    ) -> list[int]:
        pass

    @abstractmethod
    def eligible_object_ids(
        self,
        sources: Mapping[str, ObjectViewSource],
        total_views: int | None = None,
    ) -> tuple[int, ...]:
        pass

    @abstractmethod
    def build_plan(
        self,
        *,
        index: int,
        total_views: int,
        rng: np.random.Generator,
        sources: Mapping[str, ObjectViewSource],
        treatment_for: TreatmentResolver,
    ) -> SamplePlan:
        pass

    def natural_length(self, sources):
        return len(self.eligible_object_ids(sources))

    def source_names_for_role(self, view_role, sources):
        """Return sources from which this policy can draw a model-view role."""

        if view_role not in {"reference", "query"}:
            raise ValueError(f"Unknown physical view role {view_role!r}")
        return tuple(sources)


class _ReferenceQueryPolicy(ObjectSamplingPolicy):
    def __init__(
        self,
        *,
        num_reference_range=(5, 5),
        num_query_range=(1, 1),
        reference_selection="random",
        query_selection="random",
        allow_repeat=False,
    ):
        self.num_reference_range = _range(
            num_reference_range, "num_reference_range"
        )
        self.num_query_range = _range(num_query_range, "num_query_range")
        self.reference_selection = str(reference_selection)
        self.query_selection = str(query_selection)
        self.allow_repeat = bool(allow_repeat)
        for value in (self.reference_selection, self.query_selection):
            if value not in KeyQuerySamplingPolicy._STRATEGIES:
                raise ValueError(f"Invalid selection strategy {value!r}")
        self.selector = KeyQuerySamplingPolicy()

    def _count_pairs(self, total_views=None):
        return [
            (reference_count, query_count)
            for reference_count in range(
                self.num_reference_range[0], self.num_reference_range[1] + 1
            )
            for query_count in range(
                self.num_query_range[0], self.num_query_range[1] + 1
            )
            if total_views is None
            or reference_count + query_count == int(total_views)
        ]

    @abstractmethod
    def _eligible_for_counts(self, object_id, counts, sources):
        pass

    def eligible_object_ids(self, sources, total_views=None):
        pairs = self._count_pairs(total_views)
        if not pairs:
            return ()
        common = None
        for source in sources.values():
            values = set(int(value) for value in source.object_ids)
            common = values if common is None else common.intersection(values)
        common = common or set()
        return tuple(
            object_id
            for object_id in sorted(common)
            if any(
                self._eligible_for_counts(object_id, pair, sources)
                for pair in pairs
            )
        )

    def _valid_pairs(self, total_views, sources):
        pairs = []
        for pair in self._count_pairs(total_views):
            if any(
                self._eligible_for_counts(object_id, pair, sources)
                for object_id in self.eligible_object_ids(sources)
            ):
                pairs.append(pair)
        return pairs

    def supported_frame_counts(self, image_num_range, sources):
        lower, upper = (int(value) for value in image_num_range)
        return [
            total
            for total in range(lower, upper + 1)
            if self._valid_pairs(total, sources)
        ]

    def _choose_object(self, index, counts, sources):
        eligible = tuple(
            object_id
            for object_id in self.eligible_object_ids(sources)
            if self._eligible_for_counts(object_id, counts, sources)
        )
        if not eligible:
            raise ValueError(f"No object can satisfy counts {counts}")
        return int(eligible[int(index) % len(eligible)])

    def _select(self, records, count, strategy, rng):
        return self.selector.select_records(
            records,
            count,
            strategy,
            rng=rng,
            allow_repeat=self.allow_repeat,
        )

    @staticmethod
    def _plan(
        *,
        references,
        queries,
        metadata,
        treatment_for,
    ):
        return KeyQuerySamplingPolicy.plan(
            reference_records=references,
            query_records=queries,
            reference_rgb_masking=False,
            query_rgb_masking=False,
            reference_treatments=[
                treatment_for("reference", str(record["source_name"]))
                for record in references
            ],
            query_treatments=[
                treatment_for("query", str(record["source_name"]))
                for record in queries
            ],
            metadata=metadata,
        )


class ScenePairPolicy(_ReferenceQueryPolicy):
    """References and queries come from distinct same-object scene tracks."""

    def __init__(self, *, scene_source="scene", require_different_scene=True, **kwargs):
        super().__init__(**kwargs)
        self.scene_source = str(scene_source)
        self.require_different_scene = bool(require_different_scene)

    def _source(self, sources):
        if self.scene_source not in sources:
            raise KeyError(f"Missing scene source {self.scene_source!r}")
        return sources[self.scene_source]

    def source_names_for_role(self, view_role, sources):
        super().source_names_for_role(view_role, sources)
        return (self.scene_source,)

    def eligible_object_ids(self, sources, total_views=None):
        source = self._source(sources)
        pairs = self._count_pairs(total_views)
        return tuple(
            object_id
            for object_id in source.object_ids
            if any(self._eligible_for_counts(object_id, pair, sources) for pair in pairs)
        )

    def _eligible_for_counts(self, object_id, counts, sources):
        reference_count, query_count = counts
        groups = self._source(sources).groups_for_object(object_id)
        reference_need = 1 if self.allow_repeat else reference_count
        query_need = 1 if self.allow_repeat else query_count
        references = [group for group in groups if group.view_count >= reference_need]
        queries = [group for group in groups if group.view_count >= query_need]
        return any(
            reference.group_id != query.group_id
            and (
                not self.require_different_scene
                or reference.scene_id != query.scene_id
            )
            for reference in references
            for query in queries
        )

    def build_plan(self, *, index, total_views, rng, sources, treatment_for):
        pairs = self._valid_pairs(total_views, sources)
        if not pairs:
            raise ValueError(f"ScenePairPolicy cannot materialize {total_views} views")
        counts = pairs[int(rng.integers(len(pairs)))]
        object_id = self._choose_object(index, counts, sources)
        source = self._source(sources)
        reference_count, query_count = counts
        reference_need = 1 if self.allow_repeat else reference_count
        query_need = 1 if self.allow_repeat else query_count
        candidates = [
            (reference, query)
            for reference in source.groups_for_object(object_id)
            if reference.view_count >= reference_need
            for query in source.groups_for_object(object_id)
            if query.view_count >= query_need
            and reference.group_id != query.group_id
            and (
                not self.require_different_scene
                or reference.scene_id != query.scene_id
            )
        ]
        reference_group, query_group = candidates[int(rng.integers(len(candidates)))]
        references = _tag_records(
            self._select(
                source.records_for_group(reference_group),
                reference_count,
                self.reference_selection,
                rng,
            ),
            self.scene_source,
            "anchor_scene",
        )
        queries = _tag_records(
            self._select(
                source.records_for_group(query_group),
                query_count,
                self.query_selection,
                rng,
            ),
            self.scene_source,
            "query_scene",
        )
        return self._plan(
            references=references,
            queries=queries,
            treatment_for=treatment_for,
            metadata={
                "sampling_policy": "scene_pair",
                "object_id": object_id,
                "reference_count": reference_count,
                "query_count": query_count,
                "reference_group": reference_group.group_id,
                "query_group": query_group.group_id,
            },
        )


class _GeometryConstrainedReferencePolicy(_ReferenceQueryPolicy):
    """Shared query-first implementation for scene and render references."""

    reference_source_kind = "scene"

    def __init__(
        self,
        *,
        geometry_plan_path,
        query_geometry_index_path,
        reference_geometry_index_path=None,
        positive_angle_degrees=10.0,
        focal_relative_tolerance=0.1,
        crop_aspect=4.0 / 3.0,
        crop_margin_fraction=0.05,
        crop_center_jitter=0.0,
        plan_selection="random",
        random_focal_target=True,
        per_view_focal_targets=False,
        enforce_focal_compatibility=True,
        max_query_attempts=64,
        enumerate_query_groups=False,
        max_query_groups_per_object=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.geometry_plans = GeometryPlanIndex(
            geometry_plan_path,
            geometry_index_path=reference_geometry_index_path,
        )
        self.query_geometry = GeometryFeatureIndex(
            query_geometry_index_path, source_kind="scene"
        )
        if self.geometry_plans.reference_source_kind != self.reference_source_kind:
            raise ValueError(
                f"Expected {self.reference_source_kind!r} reference plans, got "
                f"{self.geometry_plans.reference_source_kind!r}"
            )
        if self.num_reference_range != (
            self.geometry_plans.reference_count,
            self.geometry_plans.reference_count,
        ):
            raise ValueError(
                "Geometry-constrained policies currently require one fixed reference "
                f"count matching the plan catalogue ({self.geometry_plans.reference_count})"
            )
        if self.num_query_range != (1, 1):
            raise ValueError("Geometry-constrained policies currently implement K=1")
        self.positive_angle_degrees = float(positive_angle_degrees)
        self.focal_relative_tolerance = float(focal_relative_tolerance)
        self.crop_aspect = float(crop_aspect)
        self.crop_margin_fraction = float(crop_margin_fraction)
        self.crop_center_jitter = float(crop_center_jitter)
        self.plan_selection = str(plan_selection)
        self.random_focal_target = bool(random_focal_target)
        self.per_view_focal_targets = bool(per_view_focal_targets)
        self.enforce_focal_compatibility = bool(enforce_focal_compatibility)
        self.max_query_attempts = int(max_query_attempts)
        self.enumerate_query_groups = bool(enumerate_query_groups)
        self.max_query_groups_per_object = (
            None
            if max_query_groups_per_object is None
            else int(max_query_groups_per_object)
        )
        self._plan_vector_cache = {}
        self._enumerated_query_cache = {}
        if not 0 < self.positive_angle_degrees <= 180:
            raise ValueError("positive_angle_degrees must be in (0,180]")
        if not 0 <= self.focal_relative_tolerance < 1:
            raise ValueError("focal_relative_tolerance must be in [0,1)")
        if not 0 <= self.crop_center_jitter <= 1:
            raise ValueError("crop_center_jitter must be in [0,1]")
        if self.plan_selection not in {"first", "random"}:
            raise ValueError("plan_selection must be first or random")
        if self.max_query_attempts <= 0:
            raise ValueError("max_query_attempts must be positive")
        if (
            self.max_query_groups_per_object is not None
            and self.max_query_groups_per_object <= 0
        ):
            raise ValueError("max_query_groups_per_object must be positive")

    def supported_frame_counts(self, image_num_range, sources):
        lower, upper = (int(value) for value in image_num_range)
        total = self.geometry_plans.reference_count + 1
        return [total] if lower <= total <= upper else []

    def _eligible_for_counts(self, object_id, counts, sources):
        return (
            tuple(int(value) for value in counts)
            == (self.geometry_plans.reference_count, 1)
            and bool(self.geometry_plans.plans_for_object(object_id))
        )

    def _query_source(self, sources):
        raise NotImplementedError

    def _reference_source(self, sources):
        raise NotImplementedError

    def source_names_for_role(self, view_role, sources):
        super().source_names_for_role(view_role, sources)
        source = self._reference_source(sources) if view_role == "reference" else self._query_source(sources)
        return (source.source_name,)

    def eligible_object_ids(self, sources, total_views=None):
        expected = self.geometry_plans.reference_count + 1
        if total_views is not None and int(total_views) != expected:
            return ()
        reference_ids = set(int(value) for value in self._reference_source(sources).object_ids)
        query_ids = set(int(value) for value in self._query_source(sources).object_ids)
        return tuple(
            value
            for value in self.geometry_plans.object_ids
            if value in reference_ids and value in query_ids
        )

    @staticmethod
    def _ordered(values, *, selection, rng):
        values = list(values)
        if selection == "random" and len(values) > 1:
            order = rng.permutation(len(values))
            return [values[int(index)] for index in order]
        return values

    def _candidate_plans(self, query_feature, plans, *, different_scene):
        if not plans or (
            self.enforce_focal_compatibility
            and not int(query_feature["crop_feasible"])
        ):
            return []
        direction = np.asarray(
            [
                query_feature["view_x"],
                query_feature["view_y"],
                query_feature["view_z"],
            ],
            dtype=np.float32,
        )
        cosine = np.cos(np.deg2rad(self.positive_angle_degrees))
        object_id = int(plans[0].object_id)
        arrays = self._plan_vector_cache.get(object_id)
        if arrays is None or arrays[0] is not plans:
            arrays = (
                plans,
                np.asarray([plan.scene_id for plan in plans], dtype=np.int64),
                np.asarray([plan.focal_low for plan in plans], dtype=np.float64),
                np.asarray([plan.focal_high for plan in plans], dtype=np.float64),
                np.asarray(
                    [plan.focal_shape_low for plan in plans], dtype=np.float64
                ),
                np.asarray(
                    [plan.focal_shape_high for plan in plans], dtype=np.float64
                ),
                np.stack([plan.directions for plan in plans], axis=0),
            )
            self._plan_vector_cache[object_id] = arrays
        (
            _,
            scene_ids,
            focal_lows,
            focal_highs,
            shape_lows,
            shape_highs,
            directions,
        ) = arrays
        similarities = np.einsum("pnc,c->pn", directions, direction)
        positive_indices = np.argmax(similarities, axis=1)
        best = similarities[np.arange(len(plans)), positive_indices]
        if self.enforce_focal_compatibility:
            query_low = normalized_focal_scalar(query_feature)
            query_high = min(
                normalized_focal_scalar(query_feature, maximum=True),
                query_low * (1.0 + self.focal_relative_tolerance),
            )
            lows = np.maximum(query_low, focal_lows)
            highs = np.minimum(query_high, focal_highs)
            shape = normalized_focal_shape(query_feature)
            valid = (
                (lows <= highs + 1e-12)
                & (shape_lows <= shape)
                & (shape <= shape_highs)
                & (best + 1e-7 >= cosine)
            )
        else:
            # Metric supervision and per-view calibrated ray conditioning make
            # a shared focal target unnecessary. Keep the plan catalogue's
            # coverage/visibility constraints and positive-view requirement,
            # but choose a crop feasible for the reference set independently
            # of the query camera.
            lows = focal_lows
            highs = focal_highs
            valid = best + 1e-7 >= cosine
        if different_scene:
            valid &= scene_ids != int(query_feature["scene_id"])
        output = []
        for index in np.flatnonzero(valid):
            similarity = float(best[index])
            angle = float(
                np.rad2deg(np.arccos(np.clip(similarity, -1.0, 1.0)))
            )
            output.append(
                (
                    plans[int(index)],
                    (float(lows[index]), float(highs[index])),
                    int(positive_indices[index]),
                    angle,
                )
            )
        return output

    def _choose_query_and_plan(self, object_id, sources, rng, *, different_scene):
        """Stream query records (group order set by `query_selection`) and return
        the first one with >=1 valid reference candidate, plus its candidates."""
        query_source = self._query_source(sources)
        groups = self._ordered(
            query_source.groups_for_object(object_id),
            selection=self.query_selection,
            rng=rng,
        )
        plans = self.geometry_plans.plans_for_object(object_id)
        attempted = 0
        for group in groups:
            records = self._ordered(
                query_source.records_for_group(group),
                selection=self.query_selection,
                rng=rng,
            )
            for query_record in records:
                attempted += 1
                query_feature = self.query_geometry.feature_for_record(query_record)
                if query_feature is None:
                    continue
                candidates = self._candidate_plans(
                    query_feature, plans, different_scene=different_scene
                )
                if candidates:
                    return dict(query_record), query_feature, candidates
                if attempted >= self.max_query_attempts:
                    break
            if attempted >= self.max_query_attempts:
                break
        raise ValueError(
            f"No query/reference geometry plan found for object {object_id} after "
            f"{attempted} query candidates"
        )

    def _enumerate_query_groups_for_object(self, object_id, sources, *, different_scene):
        """One feasible query per scene group (natural group order, capped by
        `max_query_groups_per_object`), for exhaustive/enumerated validation."""
        query_source = self._query_source(sources)
        plans = self.geometry_plans.plans_for_object(object_id)
        limit = self.max_query_groups_per_object
        output = []
        for group in query_source.groups_for_object(object_id):
            if limit is not None and len(output) >= limit:
                break
            for query_record in query_source.records_for_group(group):
                query_feature = self.query_geometry.feature_for_record(query_record)
                if query_feature is None:
                    continue
                candidates = self._candidate_plans(
                    query_feature, plans, different_scene=different_scene
                )
                if candidates:
                    output.append((dict(query_record), query_feature, candidates))
                    break
        return output

    def _enumerated_query_targets(self, sources):
        cache_key = id(sources)
        cached = self._enumerated_query_cache.get(cache_key)
        if cached is not None:
            return cached
        different_scene = self.reference_source_kind == "scene"
        targets = []
        for object_id in self.eligible_object_ids(sources):
            for query_record, query_feature, candidates in self._enumerate_query_groups_for_object(
                object_id, sources, different_scene=different_scene
            ):
                targets.append((object_id, query_record, query_feature, candidates))
        targets = tuple(targets)
        self._enumerated_query_cache[cache_key] = targets
        return targets

    def natural_length(self, sources):
        if not self.enumerate_query_groups:
            return super().natural_length(sources)
        return len(self._enumerated_query_targets(sources))

    def _reference_records(self, object_id, plan, sources):
        raise NotImplementedError

    def _tag_reference_records(self, records):
        raise NotImplementedError

    def build_plan(self, *, index, total_views, rng, sources, treatment_for):
        expected = self.geometry_plans.reference_count + 1
        if int(total_views) != expected:
            raise ValueError(
                f"Geometry-constrained policy requires {expected} model views"
            )
        if self.enumerate_query_groups:
            targets = self._enumerated_query_targets(sources)
            if not targets:
                raise ValueError("No object has compatible geometry plans and sources")
            object_id, query_record, query_feature, candidates = targets[
                int(index) % len(targets)
            ]
            query_record = dict(query_record)
        else:
            eligible = self.eligible_object_ids(sources, total_views=total_views)
            if not eligible:
                raise ValueError("No object has compatible geometry plans and sources")
            object_id = int(eligible[int(index) % len(eligible)])
            query_record, query_feature, candidates = self._choose_query_and_plan(
                object_id,
                sources,
                rng,
                different_scene=self.reference_source_kind == "scene",
            )
        if self.plan_selection == "random":
            chosen = candidates[int(rng.integers(len(candidates)))]
        else:
            chosen = candidates[0]
        reference_plan, interval, positive_index, positive_angle = chosen
        low, high = interval
        target_count = (
            expected
            if self.enforce_focal_compatibility and self.per_view_focal_targets
            else self.geometry_plans.reference_count
            if self.per_view_focal_targets
            else 1
        )
        targets = [
            (
                float(rng.uniform(low, high))
                if self.random_focal_target and high > low
                else float((low + high) * 0.5)
            )
            for _ in range(target_count)
        ]
        reference_targets = targets[: self.geometry_plans.reference_count]
        query_target = (
            targets[-1]
            if self.enforce_focal_compatibility
            else float(normalized_focal_scalar(query_feature))
        )
        if not self.per_view_focal_targets:
            reference_targets = targets * self.geometry_plans.reference_count
        reference_records = self._reference_records(
            object_id, reference_plan, sources
        )
        feature_map = self.geometry_plans.features_by_ids(
            reference_plan.feature_ids
        )
        cropped_references = []
        for record, feature_id, target in zip(
            reference_records, reference_plan.feature_ids, reference_targets
        ):
            item = dict(record)
            item["planned_object_crop"] = object_preserving_crop_spec(
                feature_map[int(feature_id)],
                target_normalized_focal=target,
                aspect=self.crop_aspect,
                margin_fraction=self.crop_margin_fraction,
                center_jitter=self.crop_center_jitter,
                rng=rng,
            )
            cropped_references.append(item)
        if self.enforce_focal_compatibility:
            query_record["planned_object_crop"] = object_preserving_crop_spec(
                query_feature,
                target_normalized_focal=query_target,
                aspect=self.crop_aspect,
                margin_fraction=self.crop_margin_fraction,
                center_jitter=self.crop_center_jitter,
                rng=rng,
            )
        references = self._tag_reference_records(cropped_references)
        queries = _tag_records(
            [query_record], self._query_source(sources).source_name, "query_scene"
        )
        return self._plan(
            references=references,
            queries=queries,
            treatment_for=treatment_for,
            metadata={
                "sampling_policy": f"geometry_{self.reference_source_kind}_to_scene",
                "object_id": object_id,
                "reference_count": self.geometry_plans.reference_count,
                "query_count": 1,
                "reference_plan_id": reference_plan.plan_id,
                "reference_group": (
                    f"scene:{reference_plan.scene_id}:track:{reference_plan.gt_id}"
                    if self.reference_source_kind == "scene"
                    else f"render:object:{object_id}"
                ),
                "query_group": (
                    int(query_feature["scene_id"]), int(query_feature["gt_id"])
                ),
                "reference_union_coverage": reference_plan.union_coverage,
                "positive_reference_index": positive_index,
                "positive_angle_degrees": positive_angle,
                "focal_compatibility_enforced": self.enforce_focal_compatibility,
                "target_normalized_focal": query_target,
                "per_view_target_normalized_focal": tuple(
                    reference_targets + [query_target]
                ),
            },
        )


class GeometryConstrainedScenePairPolicy(_GeometryConstrainedReferencePolicy):
    """Coverage/focal/positive-view constrained cross-scene sampling."""

    reference_source_kind = "scene"

    def __init__(
        self,
        *,
        scene_source="scene",
        require_different_scene=True,
        **kwargs,
    ):
        self.scene_source = str(scene_source)
        if not bool(require_different_scene):
            raise ValueError("Geometry constrained scene pairs require different scenes")
        super().__init__(**kwargs)

    def _query_source(self, sources):
        return sources[self.scene_source]

    def _reference_source(self, sources):
        return sources[self.scene_source]

    def _reference_records(self, object_id, plan, sources):
        source = self._reference_source(sources)
        groups = [
            group
            for group in source.groups_for_object(object_id)
            if int(group.scene_id) == int(plan.scene_id)
            and int(group.track_id) == int(plan.gt_id)
        ]
        if len(groups) != 1:
            raise ValueError(f"Cannot resolve geometry reference track for plan {plan.plan_id}")
        by_frame = {
            int(record["frame_id"]): record
            for record in source.records_for_group(groups[0])
        }
        try:
            return [by_frame[int(frame_id)] for frame_id in plan.frame_ids]
        except KeyError as exc:
            raise ValueError(
                f"Plan {plan.plan_id} refers to an ineligible scene frame {exc}"
            ) from exc

    def _tag_reference_records(self, records):
        return _tag_records(records, self.scene_source, "anchor_scene")


class GeometryConstrainedRenderToScenePolicy(_GeometryConstrainedReferencePolicy):
    """Coverage/focal/positive-view constrained render-to-scene sampling."""

    reference_source_kind = "render"

    def __init__(
        self,
        *,
        reference_source="render",
        query_source="scene",
        **kwargs,
    ):
        self.reference_source = str(reference_source)
        self.query_source = str(query_source)
        super().__init__(**kwargs)

    def _query_source(self, sources):
        return sources[self.query_source]

    def _reference_source(self, sources):
        return sources[self.reference_source]

    def _reference_records(self, object_id, plan, sources):
        source = self._reference_source(sources)
        groups = list(source.groups_for_object(object_id))
        if len(groups) != 1:
            raise ValueError(f"Expected one render bank group for object {object_id}")
        by_view = {
            int(record["view_id"]): record
            for record in source.records_for_group(groups[0])
        }
        try:
            return [by_view[int(view_id)] for view_id in plan.view_ids]
        except KeyError as exc:
            raise ValueError(
                f"Render plan {plan.plan_id} refers to a missing view {exc}"
            ) from exc

    def _tag_reference_records(self, records):
        return _tag_records(records, self.reference_source, self.reference_source)


class RenderToScenePolicy(_ReferenceQueryPolicy):
    """Clean reference-bank views aligned with a scene-track query set."""

    def __init__(
        self,
        *,
        reference_source="render",
        query_source="scene",
        enumerate_query_groups=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reference_source = str(reference_source)
        self.query_source = str(query_source)
        self.enumerate_query_groups = bool(enumerate_query_groups)
        self._enumerated_target_cache = {}

    def _sources(self, sources):
        try:
            return sources[self.reference_source], sources[self.query_source]
        except KeyError as exc:
            raise KeyError(f"Missing source for RenderToScenePolicy: {exc}") from exc

    def source_names_for_role(self, view_role, sources):
        super().source_names_for_role(view_role, sources)
        return (
            (self.reference_source,)
            if view_role == "reference"
            else (self.query_source,)
        )

    def _eligible_for_counts(self, object_id, counts, sources):
        reference_count, query_count = counts
        reference_source, query_source = self._sources(sources)
        reference_need = 1 if self.allow_repeat else reference_count
        query_need = 1 if self.allow_repeat else query_count
        return (
            any(
                group.view_count >= reference_need
                for group in reference_source.groups_for_object(object_id)
            )
            and any(
                group.view_count >= query_need
                for group in query_source.groups_for_object(object_id)
            )
        )

    def _enumerated_targets(self, sources, counts=None):
        reference_source, query_source = self._sources(sources)
        cache_key = (
            id(reference_source),
            id(query_source),
            None if counts is None else tuple(int(value) for value in counts),
        )
        cached = self._enumerated_target_cache.get(cache_key)
        if cached is not None:
            return cached
        pairs = self._count_pairs() if counts is None else [counts]
        targets = []
        for object_id in sorted(
            set(reference_source.object_ids).intersection(query_source.object_ids)
        ):
            for query_group in query_source.groups_for_object(object_id):
                for reference_count, query_count in pairs:
                    reference_need = 1 if self.allow_repeat else reference_count
                    query_need = 1 if self.allow_repeat else query_count
                    if query_group.view_count < query_need:
                        continue
                    if any(
                        group.view_count >= reference_need
                        for group in reference_source.groups_for_object(object_id)
                    ):
                        targets.append((int(object_id), query_group))
                        break
        targets = tuple(targets)
        self._enumerated_target_cache[cache_key] = targets
        return targets

    def natural_length(self, sources):
        if not self.enumerate_query_groups:
            return super().natural_length(sources)
        return len(self._enumerated_targets(sources))

    def build_plan(self, *, index, total_views, rng, sources, treatment_for):
        pairs = self._valid_pairs(total_views, sources)
        if not pairs:
            raise ValueError(f"RenderToScenePolicy cannot materialize {total_views} views")
        counts = pairs[int(rng.integers(len(pairs)))]
        if self.enumerate_query_groups:
            targets = self._enumerated_targets(sources, counts)
            if not targets:
                raise ValueError(f"No query group can satisfy counts {counts}")
            object_id, enumerated_query_group = targets[int(index) % len(targets)]
        else:
            object_id = self._choose_object(index, counts, sources)
            enumerated_query_group = None
        reference_count, query_count = counts
        reference_source, query_source = self._sources(sources)
        ref_groups = [
            group
            for group in reference_source.groups_for_object(object_id)
            if self.allow_repeat or group.view_count >= reference_count
        ]
        query_groups = [
            group
            for group in query_source.groups_for_object(object_id)
            if self.allow_repeat or group.view_count >= query_count
        ]
        reference_group = ref_groups[int(rng.integers(len(ref_groups)))]
        query_group = (
            enumerated_query_group
            if enumerated_query_group is not None
            else query_groups[int(rng.integers(len(query_groups)))]
        )
        references = _tag_records(
            self._select(
                reference_source.records_for_group(reference_group),
                reference_count,
                self.reference_selection,
                rng,
            ),
            self.reference_source,
            self.reference_source,
        )
        queries = _tag_records(
            self._select(
                query_source.records_for_group(query_group),
                query_count,
                self.query_selection,
                rng,
            ),
            self.query_source,
            "query_scene",
        )
        return self._plan(
            references=references,
            queries=queries,
            treatment_for=treatment_for,
            metadata={
                "sampling_policy": "render_to_scene",
                "object_id": object_id,
                "reference_count": reference_count,
                "query_count": query_count,
                "reference_group": reference_group.group_id,
                "query_group": query_group.group_id,
            },
        )


class IndependentScenesPolicy(_ReferenceQueryPolicy):
    """Select each physical view from a distinct same-object source scene."""

    def __init__(self, *, scene_source="scene", **kwargs):
        super().__init__(**kwargs)
        self.scene_source = str(scene_source)

    def _source(self, sources):
        return sources[self.scene_source]

    def source_names_for_role(self, view_role, sources):
        super().source_names_for_role(view_role, sources)
        return (self.scene_source,)

    def eligible_object_ids(self, sources, total_views=None):
        source = self._source(sources)
        pairs = self._count_pairs(total_views)
        return tuple(
            object_id
            for object_id in source.object_ids
            if any(self._eligible_for_counts(object_id, pair, sources) for pair in pairs)
        )

    def _eligible_for_counts(self, object_id, counts, sources):
        groups = self._source(sources).groups_for_object(object_id)
        return len({group.scene_id for group in groups}) >= sum(counts)

    def build_plan(self, *, index, total_views, rng, sources, treatment_for):
        pairs = self._valid_pairs(total_views, sources)
        if not pairs:
            raise ValueError(
                f"IndependentScenesPolicy cannot materialize {total_views} views"
            )
        counts = pairs[int(rng.integers(len(pairs)))]
        object_id = self._choose_object(index, counts, sources)
        source = self._source(sources)
        groups_by_scene: dict[int, list[ObjectViewGroup]] = {}
        for group in source.groups_for_object(object_id):
            groups_by_scene.setdefault(int(group.scene_id), []).append(group)
        total = sum(counts)
        chosen_scenes = rng.choice(sorted(groups_by_scene), size=total, replace=False)
        selected = []
        selected_groups = []
        for scene_id in chosen_scenes:
            groups = groups_by_scene[int(scene_id)]
            group = groups[int(rng.integers(len(groups)))]
            records = source.records_for_group(group)
            record = self.selector.select_records(
                records,
                1,
                self.reference_selection
                if len(selected) < counts[0]
                else self.query_selection,
                rng=rng,
                allow_repeat=False,
            )[0]
            selected.append(record)
            selected_groups.append(group.group_id)
        references = _tag_records(
            selected[: counts[0]], self.scene_source, "independent_scene"
        )
        queries = _tag_records(
            selected[counts[0] :], self.scene_source, "query_scene"
        )
        return self._plan(
            references=references,
            queries=queries,
            treatment_for=treatment_for,
            metadata={
                "sampling_policy": "independent_scenes",
                "object_id": object_id,
                "reference_count": counts[0],
                "query_count": counts[1],
                "groups": selected_groups,
            },
        )


class HybridReferencePolicy(ObjectSamplingPolicy):
    """Combine clean-render and scene-track references in each sample."""

    def __init__(
        self,
        *,
        render_source="render",
        scene_source="scene",
        num_render_reference_range=(2, 2),
        num_scene_reference_range=(3, 3),
        num_query_range=(1, 1),
        render_selection="random",
        scene_selection="random",
        query_selection="random",
        allow_repeat=False,
        require_different_scene=True,
    ):
        self.render_source = str(render_source)
        self.scene_source = str(scene_source)
        self.num_render_reference_range = _range(
            num_render_reference_range, "num_render_reference_range"
        )
        self.num_scene_reference_range = _range(
            num_scene_reference_range, "num_scene_reference_range"
        )
        self.num_query_range = _range(num_query_range, "num_query_range")
        self.render_selection = str(render_selection)
        self.scene_selection = str(scene_selection)
        self.query_selection = str(query_selection)
        self.allow_repeat = bool(allow_repeat)
        self.require_different_scene = bool(require_different_scene)
        self.selector = KeyQuerySamplingPolicy()
        for value in (
            self.render_selection,
            self.scene_selection,
            self.query_selection,
        ):
            if value not in KeyQuerySamplingPolicy._STRATEGIES:
                raise ValueError(f"Invalid selection strategy {value!r}")

    def source_names_for_role(self, view_role, sources):
        super().source_names_for_role(view_role, sources)
        return (
            (self.render_source, self.scene_source)
            if view_role == "reference"
            else (self.scene_source,)
        )

    def _count_triples(self, total_views=None):
        return [
            (render_count, scene_count, query_count)
            for render_count in range(
                self.num_render_reference_range[0],
                self.num_render_reference_range[1] + 1,
            )
            for scene_count in range(
                self.num_scene_reference_range[0],
                self.num_scene_reference_range[1] + 1,
            )
            for query_count in range(
                self.num_query_range[0], self.num_query_range[1] + 1
            )
            if total_views is None
            or render_count + scene_count + query_count == int(total_views)
        ]

    def _eligible(self, object_id, counts, sources):
        render_count, scene_count, query_count = counts
        render = sources[self.render_source]
        scene = sources[self.scene_source]
        render_need = 1 if self.allow_repeat else render_count
        scene_need = 1 if self.allow_repeat else scene_count
        query_need = 1 if self.allow_repeat else query_count
        if not any(
            group.view_count >= render_need
            for group in render.groups_for_object(object_id)
        ):
            return False
        references = [
            group
            for group in scene.groups_for_object(object_id)
            if group.view_count >= scene_need
        ]
        queries = [
            group
            for group in scene.groups_for_object(object_id)
            if group.view_count >= query_need
        ]
        return any(
            reference.group_id != query.group_id
            and (
                not self.require_different_scene
                or reference.scene_id != query.scene_id
            )
            for reference in references
            for query in queries
        )

    def eligible_object_ids(self, sources, total_views=None):
        common = set(sources[self.render_source].object_ids).intersection(
            sources[self.scene_source].object_ids
        )
        triples = self._count_triples(total_views)
        return tuple(
            object_id
            for object_id in sorted(common)
            if any(self._eligible(object_id, counts, sources) for counts in triples)
        )

    def supported_frame_counts(self, image_num_range, sources):
        lower, upper = (int(value) for value in image_num_range)
        return [
            total
            for total in range(lower, upper + 1)
            if self.eligible_object_ids(sources, total)
        ]

    def _select(self, source, group, count, strategy, rng):
        return self.selector.select_records(
            source.records_for_group(group),
            count,
            strategy,
            rng=rng,
            allow_repeat=self.allow_repeat,
        )

    def build_plan(self, *, index, total_views, rng, sources, treatment_for):
        valid = [
            counts
            for counts in self._count_triples(total_views)
            if any(
                self._eligible(object_id, counts, sources)
                for object_id in set(sources[self.render_source].object_ids).intersection(
                    sources[self.scene_source].object_ids
                )
            )
        ]
        if not valid:
            raise ValueError(f"HybridReferencePolicy cannot materialize {total_views} views")
        counts = valid[int(rng.integers(len(valid)))]
        eligible = tuple(
            object_id
            for object_id in self.eligible_object_ids(sources, total_views)
            if self._eligible(object_id, counts, sources)
        )
        object_id = int(eligible[int(index) % len(eligible)])
        render_count, scene_count, query_count = counts
        render_source = sources[self.render_source]
        scene_source = sources[self.scene_source]
        render_groups = [
            group
            for group in render_source.groups_for_object(object_id)
            if self.allow_repeat or group.view_count >= render_count
        ]
        pairs = [
            (reference, query)
            for reference in scene_source.groups_for_object(object_id)
            if self.allow_repeat or reference.view_count >= scene_count
            for query in scene_source.groups_for_object(object_id)
            if (self.allow_repeat or query.view_count >= query_count)
            and reference.group_id != query.group_id
            and (
                not self.require_different_scene
                or reference.scene_id != query.scene_id
            )
        ]
        render_group = render_groups[int(rng.integers(len(render_groups)))]
        scene_group, query_group = pairs[int(rng.integers(len(pairs)))]
        render_records = _tag_records(
            self._select(
                render_source,
                render_group,
                render_count,
                self.render_selection,
                rng,
            ),
            self.render_source,
            self.render_source,
        )
        scene_records = _tag_records(
            self._select(
                scene_source,
                scene_group,
                scene_count,
                self.scene_selection,
                rng,
            ),
            self.scene_source,
            "anchor_scene",
        )
        references = render_records + scene_records
        rng.shuffle(references)
        queries = _tag_records(
            self._select(
                scene_source,
                query_group,
                query_count,
                self.query_selection,
                rng,
            ),
            self.scene_source,
            "query_scene",
        )
        return KeyQuerySamplingPolicy.plan(
            reference_records=references,
            query_records=queries,
            reference_rgb_masking=False,
            query_rgb_masking=False,
            reference_treatments=[
                treatment_for("reference", str(record["source_name"]))
                for record in references
            ],
            query_treatments=[
                treatment_for("query", str(record["source_name"]))
                for record in queries
            ],
            metadata={
                "sampling_policy": "hybrid_references",
                "object_id": object_id,
                "render_reference_count": render_count,
                "scene_reference_count": scene_count,
                "query_count": query_count,
                "render_group": render_group.group_id,
                "scene_reference_group": scene_group.group_id,
                "query_group": query_group.group_id,
            },
        )
