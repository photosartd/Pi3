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
