"""Storage adapters for reusable object-centric sampling policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from datasets.object_centric import RawObjectView, ViewTreatment


_MEGAPOSE_CATALOG_CACHE: dict[
    tuple[Any, ...],
    tuple[dict[int, tuple["ObjectViewGroup", ...]], dict[tuple[int, int], int]],
] = {}


@dataclass(frozen=True)
class ObjectViewGroup:
    """One physical view group, such as a scene track or reference bank."""

    source_name: str
    object_id: int
    group_id: str
    view_count: int
    scene_id: int | None = None
    track_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ObjectViewSource(ABC):
    """Minimal storage/catalogue interface used by sampling strategies."""

    source_name: str
    object_namespace: str

    @property
    @abstractmethod
    def object_ids(self) -> Sequence[int]:
        """Object IDs with at least one eligible group."""

    @abstractmethod
    def groups_for_object(self, object_id: int) -> Sequence[ObjectViewGroup]:
        """Return lightweight groups without decoding image payloads."""

    @abstractmethod
    def records_for_group(
        self, group: ObjectViewGroup
    ) -> Sequence[Mapping[str, Any]]:
        """Return selectable records in deterministic source order."""

    @abstractmethod
    def load_raw_object_view(self, record: Mapping[str, Any]) -> RawObjectView:
        """Decode one record and normalize depth/pose to metres."""

    def validate_planned_view(
        self,
        record: Mapping[str, Any],
        *,
        view_role: str,
        treatment: ViewTreatment,
    ) -> None:
        """Apply source-specific safety checks before payload decoding."""

    @property
    def contains_repeated_object_instances(self) -> bool:
        """Whether an eligible scene can contain multiple tracks of one ID."""

        return False

    @property
    def repeated_object_scene_count(self) -> int:
        """Number of eligible ``(object_id, scene_id)`` repeated-target pairs."""

        return 0

    def validate_role_treatment(
        self, *, view_role: str, treatment: ViewTreatment
    ) -> None:
        """Fail early when a role cannot identify its supervised instance."""

        if (
            self.contains_repeated_object_instances
            and treatment.rgb == "full"
            and treatment.mask_condition
            not in {
                "object",
                "object_if_repeated",
                "object_if_repeated_else_probability",
            }
        ):
            raise ValueError(
                f"Source {self.source_name!r} contains "
                f"{self.repeated_object_scene_count} eligible same-object repeated "
                f"scenes, but {view_role} treatment uses full RGB without target "
                "disambiguation. Use rgb='object_only', mask_condition='object', "
                "mask_condition='object_if_repeated', stochastic repeated-safe "
                "conditioning, or explicitly filter repeated "
                "scenes."
            )

    def close(self) -> None:
        """Release process-local file/database handles."""


class BOPObjectModelCatalog:
    """Validated optional CAD-model capability, independent of image sources."""

    def __init__(self, data_root, *, models_folder="models_eval", object_namespace="object"):
        self.data_root = Path(data_root).expanduser().resolve()
        self.models_folder = str(models_folder)
        self.object_namespace = str(object_namespace)
        self.models_root = self.data_root / self.models_folder
        info_path = self.models_root / "models_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Model catalogue does not exist: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        self.object_ids = tuple(sorted(int(value) for value in info))
        missing = [
            object_id
            for object_id in self.object_ids
            if not (self.models_root / f"obj_{object_id:06d}.ply").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} model files; first IDs={missing[:20]}"
            )
        self._object_ids = frozenset(self.object_ids)

    def has_object(self, object_id):
        return int(object_id) in self._object_ids

    def model_path(self, object_id):
        object_id = int(object_id)
        if not self.has_object(object_id):
            raise KeyError(object_id)
        return self.models_root / f"obj_{object_id:06d}.ply"


class IndexedBOPReferenceSource(ObjectViewSource):
    """Read a BOP-style clean reference bank through its compact SQLite index."""

    def __init__(
        self,
        bank_root,
        *,
        index_path=None,
        source_name="render",
        object_namespace="object",
        object_ids="all",
        mask_type="mask_visib",
        expected_fingerprint=None,
        object_model_available=False,
    ):
        self.bank_root = Path(bank_root).expanduser().resolve()
        self.index_path = (
            self.bank_root / "pi3_index" / "references.sqlite"
            if index_path is None
            else Path(index_path).expanduser().resolve()
        )
        self.source_name = str(source_name)
        self.object_namespace = str(object_namespace)
        self.mask_type = str(mask_type)
        self.object_model_available = bool(object_model_available)
        if self.mask_type not in {"mask", "mask_visib"}:
            raise ValueError("mask_type must be 'mask' or 'mask_visib'")
        if not self.bank_root.is_dir():
            raise FileNotFoundError(f"Reference bank does not exist: {self.bank_root}")
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"Reference index does not exist: {self.index_path}. Run "
                "datasets/preprocess/render/object_reference_index.py first."
            )
        connection = self._readonly_connection()
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("format") != "pi3_object_reference_index_v1":
                raise ValueError(f"Unsupported reference index {self.index_path}")
            if metadata.get("index_complete") != "1":
                raise ValueError(f"Incomplete reference index {self.index_path}")
            if metadata.get("namespace") != self.object_namespace:
                raise ValueError(
                    f"Reference namespace {metadata.get('namespace')!r} does not "
                    f"match {self.object_namespace!r}"
                )
            if expected_fingerprint is not None and metadata.get(
                "bank_fingerprint"
            ) != str(expected_fingerprint):
                raise ValueError("Reference-bank fingerprint mismatch")
            rows = connection.execute(
                "SELECT object_id FROM objects ORDER BY object_id"
            ).fetchall()
        finally:
            connection.close()
        available = tuple(int(row[0]) for row in rows)
        requested = self._normalize_ids(object_ids)
        if requested is not None:
            missing = sorted(set(requested).difference(available))
            if missing:
                raise ValueError(
                    f"Requested reference objects absent from index: {missing[:20]}"
                )
            available = tuple(value for value in available if value in requested)
        self._object_ids = available
        self._runtime_pid: int | None = None
        self._db: sqlite3.Connection | None = None

    @staticmethod
    def _normalize_ids(value):
        if value is None or (isinstance(value, str) and value == "all"):
            return None
        if isinstance(value, (int, np.integer)):
            return (int(value),)
        values = tuple(sorted({int(item) for item in value}))
        if not values:
            raise ValueError("object_ids cannot be empty")
        return values

    def _readonly_connection(self):
        connection = sqlite3.connect(
            f"file:{self.index_path}?mode=ro&immutable=1", uri=True, timeout=60.0
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _ensure_runtime(self):
        if self._runtime_pid == os.getpid() and self._db is not None:
            return
        self.close()
        self._runtime_pid = os.getpid()
        self._db = self._readonly_connection()

    @property
    def object_ids(self):
        return self._object_ids

    def groups_for_object(self, object_id):
        object_id = int(object_id)
        if object_id not in self._object_ids:
            return ()
        self._ensure_runtime()
        row = self._db.execute(
            "SELECT view_count, external_id FROM objects WHERE object_id=?",
            (object_id,),
        ).fetchone()
        return (
            ObjectViewGroup(
                source_name=self.source_name,
                object_id=object_id,
                group_id=f"{self.source_name}:object:{object_id}",
                view_count=int(row["view_count"]),
                metadata={"external_id": str(row["external_id"])},
            ),
        )

    def records_for_group(self, group):
        if group.source_name != self.source_name:
            raise ValueError(f"Group belongs to {group.source_name}, not {self.source_name}")
        self._ensure_runtime()
        rows = self._db.execute(
            """
            SELECT object_id, view_id, coverage_view_id, T_C_O_f32, K_f32,
                   depth_scale, rgb_relpath, depth_relpath, mask_relpath,
                   mask_visib_relpath, bbox_obj_json, bbox_visib_json,
                   px_count_all, px_count_visib
            FROM views WHERE object_id=? ORDER BY view_id
            """,
            (int(group.object_id),),
        ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record.update(
                {
                    "source_name": self.source_name,
                    "reference_source": self.source_name,
                    "source": (
                        f"{self.source_name}:{int(row['object_id']):06d}/"
                        f"{int(row['view_id']):06d}"
                    ),
                    "label": f"{self.object_namespace}_{int(row['object_id']):06d}",
                    "instance": f"render_{int(row['view_id']):06d}",
                    "object_model_available": self.object_model_available,
                }
            )
            records.append(record)
        if len(records) != int(group.view_count):
            raise RuntimeError(
                f"Reference index changed for object {group.object_id}: "
                f"{len(records)} != {group.view_count}"
            )
        return records

    def _payload_path(self, record, field):
        return self.bank_root / str(record[field])

    def load_raw_object_view(self, record):
        with Image.open(self._payload_path(record, "rgb_relpath")) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        with Image.open(self._payload_path(record, "depth_relpath")) as image:
            raw_depth = np.asarray(image).copy()
        mask_field = f"{self.mask_type}_relpath"
        with Image.open(self._payload_path(record, mask_field)) as image:
            object_mask = np.asarray(image) > 0
        depthmap = (
            raw_depth.astype(np.float32)
            * float(record["depth_scale"])
            * 0.001
        )
        T_C_O = np.frombuffer(record["T_C_O_f32"], dtype="<f4").reshape(4, 4).copy()
        K = np.frombuffer(record["K_f32"], dtype="<f4").reshape(3, 3).copy()
        return RawObjectView(
            rgb=rgb,
            depthmap=depthmap,
            object_mask=object_mask,
            camera_intrinsics=K,
            T_C_O=T_C_O,
            camera_pose=np.linalg.inv(T_C_O).astype(np.float32),
            record=record,
        )

    def close(self):
        if self._db is not None:
            try:
                self._db.close()
            except sqlite3.Error:
                pass
        self._db = None
        self._runtime_pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_db"] = None
        state["_runtime_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MegaPoseGSOSceneSource(ObjectViewSource):
    """Indexed MegaPose-GSO scene tracks, independent of sampling topology."""

    _SPLIT_FORMAT = "pi3_entity_split_v1"
    _DEPTH_POLICIES = frozenset({"clean_only", "exclude_known_bad", "all"})

    def __init__(
        self,
        data_root,
        *,
        index_path=None,
        split_path=None,
        source_name="scene",
        object_namespace="gso",
        object_ids="all",
        scene_split="all",
        scene_ids=None,
        visibility_min=0.1,
        visibility_min_inclusive=True,
        min_visible_pixels=64,
        depth_corruption_policy="clean_only",
        mask_type="mask_visib",
        max_open_shards=8,
        object_model_available=False,
        exclude_repeated_object_scenes=False,
    ):
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
        self.source_name = str(source_name)
        self.object_namespace = str(object_namespace)
        self.scene_split = str(scene_split)
        self.visibility_min = float(visibility_min)
        self.visibility_min_inclusive = bool(visibility_min_inclusive)
        self.min_visible_pixels = int(min_visible_pixels)
        self.depth_corruption_policy = str(depth_corruption_policy)
        self.mask_type = str(mask_type)
        self.max_open_shards = int(max_open_shards)
        self.object_model_available = bool(object_model_available)
        self.exclude_repeated_object_scenes = bool(exclude_repeated_object_scenes)
        self._requested_object_ids = IndexedBOPReferenceSource._normalize_ids(object_ids)
        self._requested_scene_ids = IndexedBOPReferenceSource._normalize_ids(scene_ids)
        self._validate_configuration()
        self._prepared_split_ids = self._load_split_ids()
        catalogue_key = (
            str(self.index_path),
            str(self.split_path),
            self.source_name,
            self.scene_split,
            self._requested_object_ids,
            self._requested_scene_ids,
            self.visibility_min,
            self.visibility_min_inclusive,
            self.min_visible_pixels,
            self.depth_corruption_policy,
        )
        cached = _MEGAPOSE_CATALOG_CACHE.get(catalogue_key)
        if cached is None:
            groups_by_object = self._load_groups()
            self._groups_by_object = groups_by_object
            repeated_counts = self._load_repeated_track_counts()
            cached = (groups_by_object, repeated_counts)
            _MEGAPOSE_CATALOG_CACHE[catalogue_key] = cached
        groups_by_object, repeated_counts = cached
        if self.exclude_repeated_object_scenes:
            groups_by_object = {
                object_id: tuple(
                    group
                    for group in groups
                    if repeated_counts.get(
                        (int(group.object_id), int(group.scene_id)), 1
                    ) == 1
                )
                for object_id, groups in groups_by_object.items()
            }
            groups_by_object = {
                object_id: groups
                for object_id, groups in groups_by_object.items()
                if groups
            }
        self._groups_by_object = groups_by_object
        self._same_object_scene_track_counts = repeated_counts
        active_scene_keys = {
            (int(group.object_id), int(group.scene_id))
            for groups in self._groups_by_object.values()
            for group in groups
        }
        self._repeated_object_scene_count = sum(
            repeated_counts.get(key, 1) > 1 for key in active_scene_keys
        )
        self._object_ids = tuple(sorted(self._groups_by_object))
        if not self._object_ids:
            raise ValueError(
                f"MegaPose scene source has no eligible objects in split {scene_split!r}"
            )
        self._runtime_pid: int | None = None
        self._db: sqlite3.Connection | None = None
        self._shard_fds: OrderedDict[str, int] = OrderedDict()

    def _validate_configuration(self):
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"MegaPose-GSO root does not exist: {self.data_root}")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"MegaPose-GSO index does not exist: {self.index_path}")
        if self.scene_split not in {"all", "train", "val"}:
            raise ValueError("scene_split must be 'all', 'train', or 'val'")
        if self.scene_split != "all" and not self.split_path.is_file():
            raise FileNotFoundError(f"Split manifest does not exist: {self.split_path}")
        if self.depth_corruption_policy not in self._DEPTH_POLICIES:
            raise ValueError(
                f"Invalid depth_corruption_policy {self.depth_corruption_policy!r}"
            )
        if self.mask_type not in {"mask", "mask_visib"}:
            raise ValueError("mask_type must be 'mask' or 'mask_visib'")
        if not 0.0 <= self.visibility_min <= 1.0:
            raise ValueError("visibility_min must be in [0, 1]")
        if self.min_visible_pixels <= 0 or self.max_open_shards <= 0:
            raise ValueError("Pixel and open-shard limits must be positive")
        connection = self._readonly_connection()
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            profile_table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='sampling_profiles'
                """
            ).fetchone()
            profile = None
            if (
                self.visibility_min_inclusive
                and profile_table is not None
                and metadata.get("sampling_profiles_complete") == "1"
            ):
                profile = connection.execute(
                    """
                    SELECT profile_id FROM sampling_profiles
                    WHERE ABS(visibility_min - ?) < 1e-12
                      AND min_visible_pixels = ?
                      AND depth_corruption_policy = ?
                    LIMIT 1
                    """,
                    (
                        self.visibility_min,
                        self.min_visible_pixels,
                        self.depth_corruption_policy,
                    ),
                ).fetchone()
        finally:
            connection.close()
        if metadata.get("index_complete") != "1":
            raise ValueError(f"MegaPose-GSO index is incomplete: {self.index_path}")
        if metadata.get("translation_unit") != "metre":
            raise ValueError("MegaPose-GSO translations must be stored in metres")
        self._sampling_profile_id = None if profile is None else str(profile[0])

    def _readonly_connection(self):
        connection = sqlite3.connect(
            f"file:{self.index_path}?mode=ro&immutable=1", uri=True, timeout=60.0
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    @staticmethod
    def _id_digest(ids):
        payload = ",".join(str(int(value)) for value in sorted(ids)).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def _load_split_ids(self):
        if self.scene_split == "all":
            return None
        manifest = json.loads(self.split_path.read_text(encoding="utf-8"))
        if manifest.get("format") != self._SPLIT_FORMAT:
            raise ValueError(f"Unsupported split manifest {self.split_path}")
        if manifest.get("membership_policy") != {
            "train": "train_object AND train_scene",
            "val": "val_object AND val_scene",
            "cross_partition_pairs": "excluded",
        }:
            raise ValueError("MegaPose split must use strict joint object/scene membership")
        parsed = {}
        for split_name in ("train", "val"):
            values = manifest.get("splits", {}).get(split_name, {})
            parsed[split_name] = {
                "object_ids": frozenset(
                    int(value) for value in values.get("object_ids", [])
                ),
                "scene_ids": frozenset(
                    int(value) for value in values.get("scene_ids", [])
                ),
            }
        source = manifest.get("source", {})
        for entity in ("object", "scene"):
            key = f"{entity}_ids"
            train_ids = parsed["train"][key]
            val_ids = parsed["val"][key]
            if train_ids.intersection(val_ids):
                raise ValueError(f"MegaPose {entity} split overlap")
            all_ids = train_ids.union(val_ids)
            if len(all_ids) != int(source.get(f"{entity}_count", -1)) or (
                self._id_digest(all_ids)
                != source.get(f"{entity}_ids_sha256")
            ):
                raise ValueError(f"Stale or corrupt MegaPose {entity} split manifest")
        object_ids = parsed[self.scene_split]["object_ids"]
        scene_ids = parsed[self.scene_split]["scene_ids"]
        if not object_ids or not scene_ids:
            raise ValueError(f"Empty prepared MegaPose split {self.scene_split}")
        return {"object_ids": object_ids, "scene_ids": scene_ids}

    def _depth_predicate(self):
        if self.depth_corruption_policy == "clean_only":
            return "i.depth_corrupt = 0"
        if self.depth_corruption_policy == "exclude_known_bad":
            return "i.depth_corrupt IS NOT 1"
        return "1 = 1"

    def _visibility_predicate(self):
        operator = ">=" if self.visibility_min_inclusive else ">"
        return f"i.visib_fract {operator} ?"

    def _selected(self, object_id, scene_id):
        if self._requested_object_ids is not None and int(object_id) not in self._requested_object_ids:
            return False
        if self._requested_scene_ids is not None and int(scene_id) not in self._requested_scene_ids:
            return False
        if self._prepared_split_ids is None:
            return True
        return (
            int(object_id) in self._prepared_split_ids["object_ids"]
            and int(scene_id) in self._prepared_split_ids["scene_ids"]
        )

    def _load_groups(self):
        connection = self._readonly_connection()
        try:
            if self._sampling_profile_id is not None:
                rows = connection.execute(
                    """
                    SELECT object_id, scene_id, gt_id, view_count
                    FROM track_sampling_profiles WHERE profile_id=?
                    ORDER BY object_id, scene_id, gt_id
                    """,
                    (self._sampling_profile_id,),
                )
            else:
                rows = connection.execute(
                    f"""
                    SELECT i.object_id, i.scene_id, i.gt_id, COUNT(*) AS view_count
                    FROM instances AS i
                    WHERE {self._depth_predicate()}
                      AND {self._visibility_predicate()}
                      AND i.px_count_visib >= ?
                    GROUP BY i.object_id, i.scene_id, i.gt_id
                    ORDER BY i.object_id, i.scene_id, i.gt_id
                    """,
                    (self.visibility_min, self.min_visible_pixels),
                )
            grouped: dict[int, list[ObjectViewGroup]] = {}
            for row in rows:
                object_id, scene_id, gt_id = (
                    int(row["object_id"]),
                    int(row["scene_id"]),
                    int(row["gt_id"]),
                )
                if not self._selected(object_id, scene_id):
                    continue
                grouped.setdefault(object_id, []).append(
                    ObjectViewGroup(
                        source_name=self.source_name,
                        object_id=object_id,
                        group_id=f"{self.source_name}:scene:{scene_id}:track:{gt_id}",
                        scene_id=scene_id,
                        track_id=gt_id,
                        view_count=int(row["view_count"]),
                    )
                )
            return {key: tuple(value) for key, value in grouped.items()}
        finally:
            connection.close()

    def _load_repeated_track_counts(self):
        active = {
            (group.object_id, int(group.scene_id))
            for groups in self._groups_by_object.values()
            for group in groups
        }
        connection = self._readonly_connection()
        try:
            rows = connection.execute(
                """
                SELECT object_id, scene_id, COUNT(*) AS track_count
                FROM scene_tracks GROUP BY object_id, scene_id
                HAVING COUNT(*) > 1
                """
            )
            return {
                (int(row["object_id"]), int(row["scene_id"])): int(row["track_count"])
                for row in rows
                if (int(row["object_id"]), int(row["scene_id"])) in active
            }
        finally:
            connection.close()

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def contains_repeated_object_instances(self):
        return self._repeated_object_scene_count > 0

    @property
    def repeated_object_scene_count(self):
        return self._repeated_object_scene_count

    def groups_for_object(self, object_id):
        return self._groups_by_object.get(int(object_id), ())

    def _ensure_runtime(self):
        if self._runtime_pid == os.getpid() and self._db is not None:
            return
        self.close()
        self._runtime_pid = os.getpid()
        self._db = self._readonly_connection()

    def records_for_group(self, group):
        if group.source_name != self.source_name or group.scene_id is None:
            raise ValueError(f"Invalid group for {self.source_name}: {group}")
        self._ensure_runtime()
        rows = self._db.execute(
            f"""
            SELECT
                i.frame_id, i.gt_id, i.scene_id, i.view_id, i.object_id,
                i.T_C_O_f32, i.visib_fract, i.px_count_all,
                i.px_count_valid, i.px_count_visib, i.depth_corrupt,
                i.bbox_obj_x, i.bbox_obj_y, i.bbox_obj_w, i.bbox_obj_h,
                i.bbox_visib_x, i.bbox_visib_y, i.bbox_visib_w, i.bbox_visib_h,
                (SELECT COUNT(*) FROM instances AS same_i
                 WHERE same_i.frame_id=i.frame_id AND same_i.object_id=i.object_id)
                    AS same_object_frame_instance_count,
                (SELECT COUNT(*) FROM instances AS visible_i
                 WHERE visible_i.frame_id=i.frame_id
                   AND visible_i.object_id=i.object_id
                   AND visible_i.px_count_visib > 0)
                    AS same_object_visible_instance_count,
                f.frame_key, f.width, f.height, f.depth_unit_m, f.K_f32,
                f.rgb_offset, f.rgb_size, f.depth_offset, f.depth_size,
                f.mask_offset, f.mask_size, f.mask_visib_offset, f.mask_visib_size,
                s.relative_path
            FROM instances AS i
            JOIN frames AS f ON f.id=i.frame_id
            JOIN shards AS s ON s.id=f.shard_id
            WHERE i.object_id=? AND i.scene_id=? AND i.gt_id=?
              AND {self._depth_predicate()}
              AND {self._visibility_predicate()}
              AND i.px_count_visib >= ?
            ORDER BY i.view_id
            """,
            (
                int(group.object_id),
                int(group.scene_id),
                int(group.track_id),
                self.visibility_min,
                self.min_visible_pixels,
            ),
        ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record.update(
                {
                    "source_name": self.source_name,
                    "reference_source": self.source_name,
                    "source": f"{row['relative_path']}:{row['frame_key']}",
                    "label": f"{self.object_namespace}_{int(row['object_id']):06d}",
                    "instance": f"{row['frame_key']}_gt{int(row['gt_id']):02d}",
                    "object_model_available": self.object_model_available,
                    "same_object_scene_track_count": self._same_object_scene_track_counts.get(
                        (int(row["object_id"]), int(row["scene_id"])), 1
                    ),
                }
            )
            records.append(record)
        if len(records) != int(group.view_count):
            raise RuntimeError(
                f"MegaPose group changed while reading {group.group_id}: "
                f"{len(records)} != {group.view_count}"
            )
        return records

    def _payload_path(self, relative_path):
        path = Path(str(relative_path))
        return path if path.is_absolute() else self.data_root / path

    def _read_payload(self, record, payload):
        self._ensure_runtime()
        relative_path = str(record["relative_path"])
        descriptor = self._shard_fds.pop(relative_path, None)
        if descriptor is None:
            descriptor = os.open(self._payload_path(relative_path), os.O_RDONLY)
        self._shard_fds[relative_path] = descriptor
        while len(self._shard_fds) > self.max_open_shards:
            _, old = self._shard_fds.popitem(last=False)
            os.close(old)
        size = int(record[f"{payload}_size"])
        data = os.pread(descriptor, size, int(record[f"{payload}_offset"]))
        if len(data) != size:
            raise IOError(f"Short read for {relative_path}:{payload}")
        return data

    @staticmethod
    def _decode_uncompressed_rle(rle):
        height, width = (int(value) for value in rle["size"])
        counts = rle.get("counts")
        if not isinstance(counts, list):
            raise ValueError("Compressed COCO RLE is not supported")
        flat = np.zeros(height * width, dtype=bool)
        position = 0
        for run_index, raw_count in enumerate(counts):
            count = int(raw_count)
            if count < 0 or position + count > flat.size:
                raise ValueError("Invalid RLE run")
            if run_index % 2:
                flat[position : position + count] = True
            position += count
        if position != flat.size:
            raise ValueError("RLE does not cover the full mask")
        return flat.reshape((height, width), order="F")

    def _decode_mask(self, record):
        payload = json.loads(self._read_payload(record, self.mask_type))
        gt_id = int(record["gt_id"])
        rle = payload.get(str(gt_id)) if isinstance(payload, dict) else payload[gt_id]
        mask = self._decode_uncompressed_rle(rle)
        expected = (int(record["height"]), int(record["width"]))
        if mask.shape != expected:
            raise ValueError(f"Mask shape {mask.shape} != {expected}")
        return mask

    def load_raw_object_view(self, record):
        with Image.open(io.BytesIO(self._read_payload(record, "rgb"))) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        with Image.open(io.BytesIO(self._read_payload(record, "depth"))) as image:
            raw_depth = np.asarray(image).copy()
        depthmap = raw_depth.astype(np.float32) * float(record["depth_unit_m"])
        T_C_O = np.frombuffer(record["T_C_O_f32"], dtype="<f4").reshape(4, 4).copy()
        K = np.frombuffer(record["K_f32"], dtype="<f4").reshape(3, 3).copy()
        return RawObjectView(
            rgb=rgb,
            depthmap=depthmap,
            object_mask=self._decode_mask(record),
            camera_intrinsics=K,
            T_C_O=T_C_O,
            camera_pose=np.linalg.inv(T_C_O).astype(np.float32),
            record=record,
        )

    def validate_planned_view(self, record, *, view_role, treatment):
        multiplicity = max(
            int(record.get("same_object_scene_track_count", 1)),
            int(record.get("same_object_frame_instance_count", 1)),
        )
        if (
            multiplicity > 1
            and treatment.rgb == "full"
            and treatment.mask_condition
            not in {
                "object",
                "object_if_repeated",
                "object_if_repeated_else_probability",
            }
        ):
            raise ValueError(
                f"Ambiguous {view_role} {record.get('frame_key')}: scene contains "
                f"{multiplicity} instances of object {record.get('object_id')}; use "
                "object-only RGB, known mask conditioning, or filter repeated targets"
            )

    def close(self):
        if getattr(self, "_db", None) is not None:
            try:
                self._db.close()
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
