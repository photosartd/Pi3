"""Object-centric Pi3 adapter for indexed MegaPose-GSO WebDataset shards.

The one-time preprocessor in :mod:`datasets.preprocess.megapose_gso` keeps the
payloads inside their uncompressed tar shards and stores exact byte ranges plus
sampling metadata in SQLite.  This loader uses that index without extracting
the dataset and exposes the same key/query object contract as LMGeo.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from datasets.object_centric import ObjectDatasetAdapter, RawObjectView


@dataclass(frozen=True)
class _Track:
    object_id: int
    scene_id: int
    gt_id: int
    eligible_view_count: int


class MegaPoseGSOObjectDataset(ObjectDatasetAdapter):
    """Sample one GSO object across indexed MegaPose scene observations.

    Two sampling regimes are supported:

    ``independent_scenes``
        Every reference and query record comes from a different scene.  This
        is the useful regime for a sparse shard subset and is analogous to
        using scene observations as diverse object keyframes.

    ``anchor_pair``
        All references come from one physical ``(scene_id, gt_id)`` track and
        all queries from another scene track of the same GSO object.  This is
        the true two-scene analogue of LMGeo's anchor-scene-pair dataset, but
        it requires multiple downloaded views of the reference track.

    In both cases poses are metric ``T_C_O`` and Pi3 receives
    ``camera_pose = inv(T_C_O) = T_O_C``.  No GSO mesh is required.
    """

    _REGIMES = frozenset({"independent_scenes", "anchor_pair"})
    _DEPTH_POLICIES = frozenset({"clean_only", "exclude_known_bad", "all"})
    _SPLIT_FORMAT = "pi3_entity_split_v1"
    _SPLITS = frozenset({"all", "train", "val"})

    def __init__(
        self,
        data_root,
        *,
        index_path=None,
        split_path=None,
        object_ids="all",
        sampling_regime="independent_scenes",
        num_reference_range=(5, 5),
        num_query_range=(1, 1),
        visibility_min=0.1,
        min_visible_pixels=64,
        depth_corruption_policy="clean_only",
        scene_split="all",
        split_ratios=None,
        split_seed=None,
        scene_ids=None,
        reference_selection="random",
        query_selection="random",
        allow_repeat=False,
        reference_rgb_masking=False,
        query_rgb_masking=False,
        depth_masking=True,
        mask_type="mask_visib",
        visibility_mask_conditioning=False,
        condition_reference_visibility=True,
        condition_query_visibility=False,
        visibility_condition_corruption="none",
        max_open_shards=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_label = "MegaPoseGSO"
        self.data_root = Path(data_root).expanduser().resolve()
        self.index_path = (
            self.data_root / "pi3_index" / "megapose_gso.sqlite"
            if index_path is None
            else Path(index_path).expanduser().resolve()
        )
        self.split_path = (
            self.index_path.with_suffix(".splits.json")
            if split_path is None
            else Path(split_path).expanduser().resolve()
        )
        self.sampling_regime = str(sampling_regime)
        self.num_reference_range = self._normalize_range(
            num_reference_range, "num_reference_range"
        )
        self.num_query_range = self._normalize_range(
            num_query_range, "num_query_range"
        )
        self.visibility_min = float(visibility_min)
        self.min_visible_pixels = int(min_visible_pixels)
        self.depth_corruption_policy = str(depth_corruption_policy)
        self.scene_split = str(scene_split)
        self._legacy_split_ratios = split_ratios
        self._legacy_split_seed = split_seed
        self.reference_selection = str(reference_selection)
        self.query_selection = str(query_selection)
        self.allow_repeat = bool(allow_repeat)
        self.reference_rgb_masking = bool(reference_rgb_masking)
        self.query_rgb_masking = bool(query_rgb_masking)
        self.depth_masking = bool(depth_masking)
        self.mask_type = str(mask_type)
        self.visibility_mask_conditioning = bool(
            visibility_mask_conditioning
        )
        self.condition_reference_visibility = bool(
            condition_reference_visibility
        )
        self.condition_query_visibility = bool(condition_query_visibility)
        self.visibility_condition_corruption = str(
            visibility_condition_corruption
        )
        self.max_open_shards = int(max_open_shards)
        self._requested_object_ids = self._normalize_ids(object_ids, "object_ids")
        self._requested_scene_ids = self._normalize_ids(scene_ids, "scene_ids")
        self._validate_configuration()
        self._split_manifest, self._prepared_split_ids = (
            self._load_prepared_split_manifest()
        )

        self._runtime_pid: int | None = None
        self._db: sqlite3.Connection | None = None
        self._shard_fds: OrderedDict[str, int] = OrderedDict()
        self._eligibility_cache: dict[tuple[int, int], tuple[int, ...]] = {}
        self._tracks_by_object = self._load_candidate_tracks()

        available_ids = sorted(self._tracks_by_object)
        if self._requested_object_ids is not None:
            requested = set(self._requested_object_ids)
            missing = sorted(requested.difference(available_ids))
            if missing:
                raise ValueError(
                    "Requested GSO object IDs have no eligible observations in "
                    f"scene_split={self.scene_split!r}: {missing[:20]}"
                )
            available_ids = [value for value in available_ids if value in requested]
        self.object_ids = tuple(available_ids)
        self._eligibility_cache.clear()

        usable_ids = set()
        for reference_count in range(
            self.num_reference_range[0], self.num_reference_range[1] + 1
        ):
            for query_count in range(
                self.num_query_range[0], self.num_query_range[1] + 1
            ):
                usable_ids.update(
                    self._eligible_object_ids(reference_count, query_count)
                )
        self.object_ids = tuple(
            object_id for object_id in self.object_ids if object_id in usable_ids
        )
        self._eligibility_cache.clear()
        if not self.object_ids:
            raise ValueError(
                "MegaPoseGSO found no object that can satisfy the configured "
                f"reference/query ranges in split {self.scene_split!r}"
            )

        self._same_object_scene_track_counts = (
            self._load_same_object_scene_track_counts()
        )
        self._validate_repeated_object_query_configuration()

        track_count = sum(
            len(self._tracks_by_object[object_id])
            for object_id in self.object_ids
        )
        print(
            f"[{self.dataset_label}] split={self.scene_split}, "
            f"regime={self.sampling_regime}, objects={len(self.object_ids)}, "
            f"eligible_tracks={track_count}, visibility_min={self.visibility_min}, "
            f"depth_policy={self.depth_corruption_policy}, "
            f"split_seed={self._split_manifest['seed']}, "
            f"repeated_object_scene_groups="
            f"{len(self._same_object_scene_track_counts)}, "
            f"query_disambiguation={self._query_disambiguation_mode()}"
        )

    @staticmethod
    def _normalize_range(value, name):
        if isinstance(value, int):
            lower = upper = int(value)
        else:
            lower, upper = (int(item) for item in value)
        if lower <= 0 or upper < lower:
            raise ValueError(f"{name} must be a positive [min, max] range, got {value}")
        return (lower, upper)

    @staticmethod
    def _normalize_ids(value, name):
        if value is None or (isinstance(value, str) and value == "all"):
            return None
        if isinstance(value, (int, np.integer)):
            return (int(value),)
        try:
            normalized = tuple(sorted({int(item) for item in value}))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be 'all' or integer IDs") from exc
        if not normalized:
            raise ValueError(f"{name} cannot be empty")
        return normalized

    def _validate_configuration(self):
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"MegaPose-GSO root does not exist: {self.data_root}")
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"MegaPose-GSO index does not exist: {self.index_path}. Run "
                "datasets/preprocess/megapose_gso.py first."
            )
        if not self.split_path.is_file():
            raise FileNotFoundError(
                f"MegaPose-GSO prepared split manifest does not exist: "
                f"{self.split_path}. Rerun datasets/preprocess/megapose_gso.py."
            )
        if self.sampling_regime not in self._REGIMES:
            raise ValueError(
                f"sampling_regime must be one of {sorted(self._REGIMES)}"
            )
        if self.depth_corruption_policy not in self._DEPTH_POLICIES:
            raise ValueError(
                "depth_corruption_policy must be one of "
                f"{sorted(self._DEPTH_POLICIES)}"
            )
        if self.scene_split not in self._SPLITS:
            raise ValueError(f"scene_split must be one of {sorted(self._SPLITS)}")
        if self._legacy_split_ratios is not None or self._legacy_split_seed is not None:
            raise ValueError(
                "Runtime split_ratios/split_seed are no longer supported. "
                "Generate deterministic object/scene lists during MegaPose-GSO "
                "preprocessing and pass split_path if it is non-default."
            )
        if not 0.0 <= self.visibility_min <= 1.0:
            raise ValueError("visibility_min must be in [0, 1]")
        if self.min_visible_pixels <= 0:
            raise ValueError("min_visible_pixels must be positive")
        if self.mask_type not in {"mask", "mask_visib"}:
            raise ValueError("mask_type must be 'mask' or 'mask_visib'")
        if self.reference_selection not in {"first", "uniform", "random"}:
            raise ValueError("Invalid reference_selection")
        if self.query_selection not in {"first", "uniform", "random"}:
            raise ValueError("Invalid query_selection")
        if self.visibility_condition_corruption != "none":
            raise ValueError(
                "MegaPoseGSO currently supports visibility_condition_corruption='none'"
            )
        if self.max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")

    @staticmethod
    def _readonly_connection(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro&immutable=1", uri=True, timeout=60.0
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    @staticmethod
    def _id_digest(ids: Sequence[int]) -> str:
        payload = ",".join(
            str(int(value)) for value in sorted(ids)
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def _load_prepared_split_manifest(self):
        try:
            with self.split_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid MegaPose-GSO split manifest {self.split_path}: {exc}"
            ) from exc
        if not isinstance(manifest, dict) or manifest.get("format") != self._SPLIT_FORMAT:
            raise ValueError(
                f"Unsupported MegaPose-GSO split manifest format in {self.split_path}"
            )
        if manifest.get("membership_policy") != {
            "train": "train_object AND train_scene",
            "val": "val_object AND val_scene",
            "cross_partition_pairs": "excluded",
        }:
            raise ValueError(
                "MegaPose-GSO split manifest must use strict joint object/scene "
                "membership with cross-partition pairs excluded"
            )

        raw_splits = manifest.get("splits")
        if not isinstance(raw_splits, dict) or set(raw_splits) != {"train", "val"}:
            raise ValueError("MegaPose-GSO split manifest must define train and val")
        prepared = {}
        for split_name in ("train", "val"):
            values = raw_splits[split_name]
            if not isinstance(values, dict):
                raise ValueError(f"Invalid {split_name} split entry")
            parsed = {}
            for entity_key in ("object_ids", "scene_ids"):
                raw_ids = values.get(entity_key)
                if not isinstance(raw_ids, list) or any(
                    not isinstance(value, int) for value in raw_ids
                ):
                    raise ValueError(
                        f"Prepared {split_name}.{entity_key} must be an integer list"
                    )
                if len(raw_ids) != len(set(raw_ids)):
                    raise ValueError(
                        f"Prepared {split_name}.{entity_key} contains duplicates"
                    )
                parsed[entity_key] = frozenset(int(value) for value in raw_ids)
            prepared[split_name] = {
                "object_ids": parsed["object_ids"],
                "scene_ids": parsed["scene_ids"],
            }
        for entity_key in ("object_ids", "scene_ids"):
            overlap = prepared["train"][entity_key].intersection(
                prepared["val"][entity_key]
            )
            if overlap:
                raise ValueError(
                    f"Prepared MegaPose-GSO {entity_key} overlap across train/val: "
                    f"{sorted(overlap)[:20]}"
                )

        connection = self._readonly_connection(self.index_path)
        try:
            index_objects = tuple(
                int(row[0])
                for row in connection.execute(
                    "SELECT object_id FROM objects ORDER BY object_id"
                )
            )
            index_scenes = tuple(
                int(row[0])
                for row in connection.execute(
                    "SELECT scene_id FROM scenes ORDER BY scene_id"
                )
            )
        finally:
            connection.close()
        source = manifest.get("source", {})
        checks = (
            (
                "object",
                index_objects,
                prepared["train"]["object_ids"]
                | prepared["val"]["object_ids"],
            ),
            (
                "scene",
                index_scenes,
                prepared["train"]["scene_ids"]
                | prepared["val"]["scene_ids"],
            ),
        )
        for entity_name, index_ids, manifest_ids in checks:
            digest_key = f"{entity_name}_ids_sha256"
            if set(index_ids) != set(manifest_ids) or source.get(
                digest_key
            ) != self._id_digest(index_ids):
                raise ValueError(
                    f"MegaPose-GSO split manifest is stale for {entity_name} IDs; "
                    "rerun datasets/preprocess/megapose_gso.py"
                )
        return manifest, prepared

    def _scene_is_selected(self, scene_id: int) -> bool:
        scene_id = int(scene_id)
        if self._requested_scene_ids is not None:
            if scene_id not in self._requested_scene_ids:
                return False
        if self.scene_split == "all":
            return True
        return scene_id in self._prepared_split_ids[self.scene_split]["scene_ids"]

    def _object_is_selected(self, object_id: int) -> bool:
        if self.scene_split == "all":
            return True
        return int(object_id) in self._prepared_split_ids[self.scene_split][
            "object_ids"
        ]

    def _depth_predicate(self) -> str:
        if self.depth_corruption_policy == "clean_only":
            return "i.depth_corrupt = 0"
        if self.depth_corruption_policy == "exclude_known_bad":
            return "i.depth_corrupt IS NOT 1"
        return "1 = 1"

    def _load_candidate_tracks(self) -> dict[int, tuple[_Track, ...]]:
        connection = self._readonly_connection(self.index_path)
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("index_complete") != "1":
                raise ValueError(
                    f"MegaPose-GSO index is incomplete: {self.index_path}"
                )
            if metadata.get("translation_unit") != "metre":
                raise ValueError("MegaPose-GSO index translations must be metric")
            rows = connection.execute(
                f"""
                SELECT i.object_id, i.scene_id, i.gt_id, COUNT(*) AS eligible_views
                FROM instances AS i
                WHERE {self._depth_predicate()}
                  AND i.visib_fract >= ?
                  AND i.px_count_visib >= ?
                GROUP BY i.object_id, i.scene_id, i.gt_id
                ORDER BY i.object_id, i.scene_id, i.gt_id
                """,
                (self.visibility_min, self.min_visible_pixels),
            )
            grouped: dict[int, list[_Track]] = {}
            for row in rows:
                scene_id = int(row["scene_id"])
                if not self._scene_is_selected(scene_id):
                    continue
                object_id = int(row["object_id"])
                if not self._object_is_selected(object_id):
                    continue
                grouped.setdefault(object_id, []).append(
                    _Track(
                        object_id=object_id,
                        scene_id=scene_id,
                        gt_id=int(row["gt_id"]),
                        eligible_view_count=int(row["eligible_views"]),
                    )
                )
            return {key: tuple(value) for key, value in grouped.items()}
        finally:
            connection.close()

    def _load_same_object_scene_track_counts(self) -> dict[tuple[int, int], int]:
        """Return active ``(object, scene)`` groups with multiple instances.

        ``gt_id`` identifies one physical instance track within a MegaPose
        scene.  A scene can contain several tracks with the same numeric GSO
        object ID, so object ID alone is not sufficient to identify the query
        target.  Count all tracks in each active candidate scene, including
        tracks that fail the visibility/depth sampling filters: a partially
        hidden second copy still makes the unmasked RGB query ambiguous.
        """

        active_pairs = {
            (int(object_id), int(track.scene_id))
            for object_id in self.object_ids
            for track in self._tracks_by_object[object_id]
        }
        connection = self._readonly_connection(self.index_path)
        try:
            rows = connection.execute(
                """
                SELECT object_id, scene_id, COUNT(*) AS track_count
                FROM scene_tracks
                GROUP BY object_id, scene_id
                HAVING COUNT(*) > 1
                ORDER BY object_id, scene_id
                """
            )
            return {
                (int(row["object_id"]), int(row["scene_id"])): int(
                    row["track_count"]
                )
                for row in rows
                if (int(row["object_id"]), int(row["scene_id"]))
                in active_pairs
            }
        finally:
            connection.close()

    def _query_is_disambiguated(self) -> bool:
        return bool(
            self.query_rgb_masking
            or (
                self.visibility_mask_conditioning
                and self.condition_query_visibility
            )
        )

    def _query_disambiguation_mode(self) -> str:
        conditioned = bool(
            self.visibility_mask_conditioning
            and self.condition_query_visibility
        )
        if self.query_rgb_masking and conditioned:
            return "rgb_masked_and_mask_conditioned"
        if self.query_rgb_masking:
            return "rgb_masked"
        if conditioned:
            return "mask_conditioned"
        return "none"

    def _repeated_object_error(self, examples) -> ValueError:
        formatted = ", ".join(
            f"object={int(object_id)}/scene={int(scene_id)}"
            for object_id, scene_id in examples[:5]
        )
        return ValueError(
            "MegaPoseGSO scenes with a repeated object ID require every query "
            "to identify its physical target. Enable query_rgb_masking=true or "
            "enable both visibility_mask_conditioning=true and "
            "condition_query_visibility=true. Active ambiguous examples: "
            f"{formatted}"
        )

    def _validate_repeated_object_query_configuration(self) -> None:
        if (
            self._same_object_scene_track_counts
            and not self._query_is_disambiguated()
        ):
            raise self._repeated_object_error(
                list(self._same_object_scene_track_counts)
            )

    def _validate_selected_query_records(
        self, query_records: Sequence[Mapping[str, Any]]
    ) -> None:
        ambiguous = [
            (int(record["object_id"]), int(record["scene_id"]))
            for record in query_records
            if max(
                int(record.get("same_object_scene_track_count", 1)),
                int(record.get("same_object_frame_instance_count", 1)),
            )
            > 1
        ]
        if ambiguous and not self._query_is_disambiguated():
            raise self._repeated_object_error(ambiguous)

    def __len__(self):
        return len(self.object_ids)

    def convert_attributes(self):
        """The compact immutable track index is already worker-friendly."""

    def _eligible_object_ids(
        self, reference_count: int, query_count: int
    ) -> tuple[int, ...]:
        key = (int(reference_count), int(query_count))
        cached = self._eligibility_cache.get(key)
        if cached is not None:
            return cached
        eligible = []
        for object_id in self.object_ids:
            tracks = self._tracks_by_object[object_id]
            if self.sampling_regime == "independent_scenes":
                if len({track.scene_id for track in tracks}) >= sum(key):
                    eligible.append(object_id)
                continue
            reference_need = 1 if self.allow_repeat else key[0]
            query_need = 1 if self.allow_repeat else key[1]
            reference_tracks = [
                track for track in tracks
                if track.eligible_view_count >= reference_need
            ]
            query_tracks = [
                track for track in tracks
                if track.eligible_view_count >= query_need
            ]
            if any(
                reference.scene_id != query.scene_id
                for reference in reference_tracks
                for query in query_tracks
            ):
                eligible.append(object_id)
        result = tuple(eligible)
        self._eligibility_cache[key] = result
        return result

    def _valid_count_pairs(self, total_views: int | None = None):
        pairs = []
        for reference_count in range(
            self.num_reference_range[0], self.num_reference_range[1] + 1
        ):
            for query_count in range(
                self.num_query_range[0], self.num_query_range[1] + 1
            ):
                if total_views is not None and reference_count + query_count != int(total_views):
                    continue
                if self._eligible_object_ids(reference_count, query_count):
                    pairs.append((reference_count, query_count))
        return pairs

    def supported_frame_counts(self, image_num_range):
        lower, upper = (int(value) for value in image_num_range)
        return [
            total
            for total in range(lower, upper + 1)
            if self._valid_count_pairs(total)
        ]

    def _ensure_runtime(self):
        pid = os.getpid()
        if self._runtime_pid == pid and self._db is not None:
            return
        self.close()
        self._runtime_pid = pid
        self._db = self._readonly_connection(self.index_path)

    def _records_for_track(self, track: _Track) -> list[dict[str, Any]]:
        self._ensure_runtime()
        assert self._db is not None
        rows = self._db.execute(
            f"""
            SELECT
                i.frame_id, i.gt_id, i.scene_id, i.view_id, i.object_id,
                i.T_C_O_f32, i.visib_fract, i.px_count_all,
                i.px_count_valid, i.px_count_visib, i.depth_corrupt,
                i.bbox_obj_x, i.bbox_obj_y, i.bbox_obj_w, i.bbox_obj_h,
                i.bbox_visib_x, i.bbox_visib_y,
                i.bbox_visib_w, i.bbox_visib_h,
                (
                    SELECT COUNT(*)
                    FROM instances AS same_i
                    WHERE same_i.frame_id = i.frame_id
                      AND same_i.object_id = i.object_id
                ) AS same_object_frame_instance_count,
                (
                    SELECT COUNT(*)
                    FROM instances AS visible_i
                    WHERE visible_i.frame_id = i.frame_id
                      AND visible_i.object_id = i.object_id
                      AND visible_i.px_count_visib > 0
                ) AS same_object_visible_instance_count,
                f.frame_key, f.width, f.height, f.depth_unit_m, f.K_f32,
                f.rgb_offset, f.rgb_size, f.depth_offset, f.depth_size,
                f.mask_offset, f.mask_size,
                f.mask_visib_offset, f.mask_visib_size,
                s.relative_path
            FROM instances AS i
            JOIN frames AS f ON f.id = i.frame_id
            JOIN shards AS s ON s.id = f.shard_id
            WHERE i.object_id = ? AND i.scene_id = ? AND i.gt_id = ?
              AND {self._depth_predicate()}
              AND i.visib_fract >= ?
              AND i.px_count_visib >= ?
            ORDER BY i.view_id
            """,
            (
                track.object_id,
                track.scene_id,
                track.gt_id,
                self.visibility_min,
                self.min_visible_pixels,
            ),
        ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record.update(
                {
                    "label": f"gso_obj{track.object_id:06d}",
                    "instance": (
                        f"{record['frame_key']}_gt{track.gt_id:02d}"
                    ),
                    "source": (
                        f"{record['relative_path']}:{record['frame_key']}"
                    ),
                    "object_model_available": False,
                    "same_object_scene_track_count": (
                        self._same_object_scene_track_counts.get(
                            (track.object_id, track.scene_id), 1
                        )
                    ),
                }
            )
            records.append(record)
        if len(records) != track.eligible_view_count:
            raise RuntimeError(
                f"Index changed while reading GSO track {track}: expected "
                f"{track.eligible_view_count} records, got {len(records)}"
            )
        return records

    def _choose_independent_scene_records(
        self,
        object_id: int,
        reference_count: int,
        query_count: int,
        rng: np.random.Generator,
    ):
        tracks_by_scene: dict[int, list[_Track]] = {}
        for track in self._tracks_by_object[object_id]:
            tracks_by_scene.setdefault(track.scene_id, []).append(track)
        scene_ids = np.asarray(sorted(tracks_by_scene), dtype=np.int64)
        chosen = rng.choice(
            scene_ids, size=reference_count + query_count, replace=False
        )
        selected_records = []
        selected_tracks = []
        for selection_index, scene_id in enumerate(chosen):
            tracks = tracks_by_scene[int(scene_id)]
            track = tracks[int(rng.integers(len(tracks)))]
            records = self._records_for_track(track)
            strategy = (
                self.reference_selection
                if selection_index < reference_count
                else self.query_selection
            )
            record = self.key_query_sampling_policy.select_records(
                records,
                1,
                strategy,
                rng=rng,
                allow_repeat=False,
            )[0]
            selected_records.append(record)
            selected_tracks.append(track)
        return (
            selected_records[:reference_count],
            selected_records[reference_count:],
            selected_tracks[:reference_count],
            selected_tracks[reference_count:],
        )

    def _choose_anchor_pair_records(
        self,
        object_id: int,
        reference_count: int,
        query_count: int,
        rng: np.random.Generator,
    ):
        tracks = self._tracks_by_object[object_id]
        reference_need = 1 if self.allow_repeat else reference_count
        query_need = 1 if self.allow_repeat else query_count
        reference_tracks = [
            track for track in tracks
            if track.eligible_view_count >= reference_need
        ]
        query_tracks = [
            track for track in tracks
            if track.eligible_view_count >= query_need
        ]
        reference_order = rng.permutation(len(reference_tracks))
        reference_track = None
        compatible_queries = None
        for index in reference_order:
            candidate = reference_tracks[int(index)]
            compatible = [
                track for track in query_tracks
                if track.scene_id != candidate.scene_id
            ]
            if compatible:
                reference_track = candidate
                compatible_queries = compatible
                break
        if reference_track is None or not compatible_queries:
            raise RuntimeError(
                f"No compatible anchor pair for GSO object {object_id}"
            )
        query_track = compatible_queries[int(rng.integers(len(compatible_queries)))]
        reference_records = self.key_query_sampling_policy.select_records(
            self._records_for_track(reference_track),
            reference_count,
            self.reference_selection,
            rng=rng,
            allow_repeat=self.allow_repeat,
        )
        query_records = self.key_query_sampling_policy.select_records(
            self._records_for_track(query_track),
            query_count,
            self.query_selection,
            rng=rng,
            allow_repeat=self.allow_repeat,
        )
        return (
            reference_records,
            query_records,
            [reference_track],
            [query_track],
        )

    def _get_views(self, index, resolution, rng):
        valid_pairs = self._valid_count_pairs(self.frame_num)
        if not valid_pairs:
            raise ValueError(
                f"MegaPoseGSO cannot materialize frame_num={self.frame_num}; "
                f"supported totals are {self.supported_frame_counts((1, 128))}"
            )
        reference_count, query_count = valid_pairs[
            int(rng.integers(len(valid_pairs)))
        ]
        eligible_objects = self._eligible_object_ids(
            reference_count, query_count
        )
        object_id = eligible_objects[int(index) % len(eligible_objects)]
        if self.sampling_regime == "independent_scenes":
            selected = self._choose_independent_scene_records(
                object_id, reference_count, query_count, rng
            )
        else:
            selected = self._choose_anchor_pair_records(
                object_id, reference_count, query_count, rng
            )
        reference_records, query_records, reference_tracks, query_tracks = selected
        self._validate_selected_query_records(query_records)
        for record in reference_records:
            record["reference_source"] = "megapose_scene"

        self._current_resolution = resolution
        metadata = {
            "object_id": int(object_id),
            "sampling_regime": self.sampling_regime,
            "scene_split": self.scene_split,
            "reference_count": int(reference_count),
            "query_count": int(query_count),
            "reference_tracks": [
                (track.scene_id, track.gt_id) for track in reference_tracks
            ],
            "query_tracks": [
                (track.scene_id, track.gt_id) for track in query_tracks
            ],
            "query_disambiguation": self._query_disambiguation_mode(),
            "query_same_object_scene_track_counts": [
                int(record["same_object_scene_track_count"])
                for record in query_records
            ],
            "query_same_object_frame_instance_counts": [
                int(record["same_object_frame_instance_count"])
                for record in query_records
            ],
            "reference": [record["frame_key"] for record in reference_records],
            "query": [record["frame_key"] for record in query_records],
        }
        plan = self.key_query_sampling_policy.plan(
            reference_records=reference_records,
            query_records=query_records,
            reference_rgb_masking=self.reference_rgb_masking,
            query_rgb_masking=self.query_rgb_masking,
            metadata=metadata,
        )
        return self._materialize_sample_plan(plan, rng=rng)

    def _should_depth_mask_view(self, *, view_role, reference_source):
        return self.depth_masking

    def _should_condition_visibility_view(self, *, view_role):
        if not self.visibility_mask_conditioning:
            return False
        if view_role == "reference":
            return self.condition_reference_visibility
        return self.condition_query_visibility

    def _payload_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else self.data_root / path

    def _read_payload(self, record: Mapping[str, Any], payload: str) -> bytes:
        self._ensure_runtime()
        relative_path = str(record["relative_path"])
        descriptor = self._shard_fds.pop(relative_path, None)
        if descriptor is None:
            descriptor = os.open(self._payload_path(relative_path), os.O_RDONLY)
        self._shard_fds[relative_path] = descriptor
        while len(self._shard_fds) > self.max_open_shards:
            _, old_descriptor = self._shard_fds.popitem(last=False)
            os.close(old_descriptor)
        offset = int(record[f"{payload}_offset"])
        size = int(record[f"{payload}_size"])
        data = os.pread(descriptor, size, offset)
        if len(data) != size:
            raise IOError(
                f"Short read for {relative_path}:{payload}; got {len(data)}/{size} bytes"
            )
        return data

    @staticmethod
    def _decode_uncompressed_rle(rle: Mapping[str, Any]) -> np.ndarray:
        size = rle.get("size")
        counts = rle.get("counts")
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError("MegaPose mask RLE must contain [height, width]")
        if not isinstance(counts, list):
            raise ValueError(
                "MegaPose mask uses compressed COCO RLE; rebuild support with "
                "pycocotools or use the released uncompressed masks"
            )
        height, width = (int(value) for value in size)
        flat = np.zeros(height * width, dtype=bool)
        position = 0
        for run_index, raw_count in enumerate(counts):
            count = int(raw_count)
            if count < 0 or position + count > flat.size:
                raise ValueError("Invalid MegaPose mask RLE run lengths")
            if run_index % 2:
                flat[position : position + count] = True
            position += count
        if position != flat.size:
            raise ValueError(
                f"MegaPose mask RLE covers {position} pixels, expected {flat.size}"
            )
        return flat.reshape((height, width), order="F")

    def _decode_mask(self, record: Mapping[str, Any]) -> np.ndarray:
        payload = json.loads(self._read_payload(record, self.mask_type))
        gt_id = int(record["gt_id"])
        if isinstance(payload, dict):
            rle = payload.get(str(gt_id))
        elif isinstance(payload, list):
            rle = payload[gt_id] if gt_id < len(payload) else None
        else:
            rle = None
        if not isinstance(rle, dict):
            raise ValueError(
                f"Missing {self.mask_type} RLE for {record['frame_key']}/gt{gt_id}"
            )
        mask = self._decode_uncompressed_rle(rle)
        expected = (int(record["height"]), int(record["width"]))
        if mask.shape != expected:
            raise ValueError(
                f"Mask shape {mask.shape} does not match indexed image shape {expected}"
            )
        return mask

    def load_raw_object_view(self, record):
        with Image.open(io.BytesIO(self._read_payload(record, "rgb"))) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        with Image.open(io.BytesIO(self._read_payload(record, "depth"))) as image:
            raw_depth = np.asarray(image).copy()
        depthmap = raw_depth.astype(np.float32) * float(record["depth_unit_m"])
        object_mask = self._decode_mask(record)
        intrinsics = np.frombuffer(
            record["K_f32"], dtype="<f4"
        ).reshape(3, 3).copy()
        T_C_O = np.frombuffer(
            record["T_C_O_f32"], dtype="<f4"
        ).reshape(4, 4).copy()
        return RawObjectView(
            rgb=rgb,
            depthmap=depthmap,
            object_mask=object_mask,
            camera_intrinsics=intrinsics,
            T_C_O=T_C_O,
            camera_pose=np.linalg.inv(T_C_O).astype(np.float32),
            record=record,
        )

    def assemble_processed_object_view(self, state, request):
        view = super().assemble_processed_object_view(state, request)
        record = state["record"]
        scene_id = int(record["scene_id"])
        view_id = int(record["view_id"])
        gt_id = int(record["gt_id"])
        view.update(
            {
                "dataset": self.dataset_label,
                "object_model_available": False,
                "object_model_namespace": "gso",
                "scene_id": np.int64(scene_id),
                "source_scene_id": np.int64(scene_id),
                "source_subscene_id": np.int64(-1),
                "view_id": np.int64(view_id),
                "im_id": np.int64(view_id),
                "gt_id": np.int64(gt_id),
                "frame_id": np.int64(record["frame_id"]),
                "frame_key": str(record["frame_key"]),
                "visib_fract": np.float32(record["visib_fract"]),
                "px_count_visib": np.int64(record["px_count_visib"]),
                "same_object_scene_track_count": np.int64(
                    record["same_object_scene_track_count"]
                ),
                "same_object_frame_instance_count": np.int64(
                    record["same_object_frame_instance_count"]
                ),
                "same_object_visible_instance_count": np.int64(
                    record["same_object_visible_instance_count"]
                ),
                "has_repeated_object_id": bool(
                    int(record["same_object_scene_track_count"]) > 1
                    or int(record["same_object_frame_instance_count"]) > 1
                ),
                "query_disambiguation_mode": (
                    self._query_disambiguation_mode()
                ),
                "depth_corrupt": (
                    None
                    if record["depth_corrupt"] is None
                    else bool(record["depth_corrupt"])
                ),
                "source_shard": str(record["relative_path"]),
                "is_context_reference": False,
                "target_id": np.int64(-1),
                "query_instance_rank": np.int64(-1),
            }
        )
        return view

    def close(self):
        connection = getattr(self, "_db", None)
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass
        self._db = None
        for descriptor in getattr(self, "_shard_fds", {}).values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._shard_fds = OrderedDict()
        self._runtime_pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_db"] = None
        state["_shard_fds"] = OrderedDict()
        state["_runtime_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
