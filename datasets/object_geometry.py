"""Read-only geometry plans and object-preserving crop specifications.

This module is deliberately independent of MegaPose payload decoding.  A
sampling policy consumes a compact plan catalogue plus a geometry sidecar and
attaches ``planned_object_crop`` dictionaries to ordinary source records.  The
generic object-view processor then applies those dictionaries consistently to
RGB, depth, masks and intrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

import numpy as np


PLAN_FORMAT = "pi3_object_geometry_plan_v1"


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{Path(path).expanduser().resolve()}?mode=ro&immutable=1",
        uri=True,
        timeout=60.0,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _decode_i64(value: bytes) -> np.ndarray:
    return np.frombuffer(value, dtype="<i8").astype(np.int64, copy=True)


def _decode_i32(value: bytes) -> np.ndarray:
    return np.frombuffer(value, dtype="<i4").astype(np.int64, copy=True)


def _decode_f32(value: bytes, shape: tuple[int, ...]) -> np.ndarray:
    output = np.frombuffer(value, dtype="<f4").astype(np.float32, copy=True)
    if output.size != int(np.prod(shape)):
        raise ValueError(f"Geometry-plan array has {output.size} values, expected {shape}")
    return output.reshape(shape)


@dataclass(frozen=True)
class GeometryReferencePlan:
    plan_id: int
    object_id: int
    scene_id: int
    gt_id: int
    reference_count: int
    union_coverage: float
    feature_ids: np.ndarray
    frame_ids: np.ndarray
    view_ids: np.ndarray
    directions: np.ndarray
    focal_low: float
    focal_high: float
    focal_shape_low: float
    focal_shape_high: float

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "GeometryReferencePlan":
        count = int(row["reference_count"])
        return cls(
            plan_id=int(row["plan_id"]),
            object_id=int(row["object_id"]),
            scene_id=int(row["scene_id"]),
            gt_id=int(row["gt_id"]),
            reference_count=count,
            union_coverage=float(row["union_coverage"]),
            feature_ids=_decode_i64(row["feature_ids_i64"]),
            frame_ids=_decode_i64(row["frame_ids_i64"]),
            view_ids=_decode_i32(row["view_ids_i32"]),
            directions=_decode_f32(row["directions_f32"], (count, 3)),
            focal_low=float(row["focal_low"]),
            focal_high=float(row["focal_high"]),
            focal_shape_low=float(row["focal_shape_low"]),
            focal_shape_high=float(row["focal_shape_high"]),
        )


def _resolve_plan_geometry_index_path(
    plan_path: Path,
    stored_path: str | Path,
    override_path: str | Path | None,
) -> tuple[Path, str]:
    """Resolve a plan's geometry DB without tying copied indexes to one host.

    Plan catalogues created before the portable-path fix stored an absolute
    build-host path.  Prefer an explicit runtime pairing when supplied.  New
    relative metadata is interpreted from the plan catalogue directory, while
    a missing legacy absolute path may be relocated to a same-named sibling.
    The sibling rule matches every v1 scene/render plan produced by this repo.
    """

    if override_path is not None:
        return Path(override_path).expanduser().resolve(), "runtime override"

    raw_path = Path(stored_path).expanduser()
    if not raw_path.is_absolute():
        return (plan_path.parent / raw_path).resolve(), "relative metadata"

    resolved = raw_path.resolve()
    if resolved.is_file():
        return resolved, "absolute metadata"

    relocated = (plan_path.parent / raw_path.name).resolve()
    if relocated.is_file():
        return relocated, "relocated legacy metadata"
    return resolved, "missing absolute metadata"


class GeometryPlanIndex:
    """Process-safe reader for a reference-plan catalogue and its geometry."""

    def __init__(
        self,
        plan_path: str | Path,
        *,
        geometry_index_path: str | Path | None = None,
    ):
        self.plan_path = Path(plan_path).expanduser().resolve()
        if not self.plan_path.is_file():
            raise FileNotFoundError(f"Geometry plan catalogue does not exist: {self.plan_path}")
        connection = _readonly_connection(self.plan_path)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            if metadata.get("format") != PLAN_FORMAT:
                raise ValueError(f"Unsupported geometry-plan catalogue: {self.plan_path}")
            if metadata.get("index_complete") != "1":
                raise ValueError(f"Incomplete geometry-plan catalogue: {self.plan_path}")
            stored_geometry_path = metadata["geometry_index_path"]
            (
                self.geometry_index_path,
                self.geometry_index_path_source,
            ) = _resolve_plan_geometry_index_path(
                self.plan_path,
                stored_geometry_path,
                geometry_index_path,
            )
            self.reference_source_kind = str(metadata["reference_source_kind"])
            self.constraints = json.loads(metadata["constraints"])
            self.reference_count = int(self.constraints["reference_count"])
            self.object_ids = tuple(
                int(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT object_id FROM reference_plans ORDER BY object_id"
                )
            )
        finally:
            connection.close()
        if not self.geometry_index_path.is_file():
            raise FileNotFoundError(
                f"Plan geometry index does not exist: {self.geometry_index_path} "
                f"(resolved from {self.geometry_index_path_source}; plan: "
                f"{self.plan_path}). Pass geometry_index_path explicitly, copy "
                "the geometry SQLite next to this legacy plan catalogue, or rebuild "
                "the plan catalogue with portable relative metadata."
            )
        self._runtime_pid: int | None = None
        self._plan_db: sqlite3.Connection | None = None
        self._geometry_db: sqlite3.Connection | None = None
        self._plans_cache: dict[int, tuple[GeometryReferencePlan, ...]] = {}

    def _ensure_runtime(self) -> None:
        if self._runtime_pid == os.getpid() and self._plan_db is not None:
            return
        self.close()
        self._runtime_pid = os.getpid()
        self._plan_db = _readonly_connection(self.plan_path)
        self._geometry_db = _readonly_connection(self.geometry_index_path)

    def plans_for_object(self, object_id: int) -> tuple[GeometryReferencePlan, ...]:
        object_id = int(object_id)
        cached = self._plans_cache.get(object_id)
        if cached is not None:
            return cached
        self._ensure_runtime()
        rows = self._plan_db.execute(
            "SELECT * FROM reference_plans WHERE object_id=? ORDER BY plan_id",
            (object_id,),
        ).fetchall()
        output = tuple(GeometryReferencePlan.from_row(row) for row in rows)
        self._plans_cache[object_id] = output
        return output

    def feature_for_record(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return geometry for an ordinary scene or render source record."""

        self._ensure_runtime()
        if "frame_id" in record and self.reference_source_kind != "render":
            row = self._geometry_db.execute(
                "SELECT * FROM frame_features WHERE frame_id=? AND gt_id=?",
                (int(record["frame_id"]), int(record.get("gt_id", 0))),
            ).fetchone()
        else:
            row = self._geometry_db.execute(
                "SELECT * FROM frame_features WHERE object_id=? AND view_id=?",
                (int(record["object_id"]), int(record["view_id"])),
            ).fetchone()
        return None if row is None else dict(row)

    def features_by_ids(self, feature_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        values = tuple(int(value) for value in feature_ids)
        if not values:
            return {}
        self._ensure_runtime()
        placeholders = ",".join("?" for _ in values)
        rows = self._geometry_db.execute(
            f"SELECT * FROM frame_features WHERE feature_id IN ({placeholders})",
            values,
        ).fetchall()
        output = {int(row["feature_id"]): dict(row) for row in rows}
        if len(output) != len(set(values)):
            missing = sorted(set(values).difference(output))
            raise KeyError(f"Missing geometry feature IDs: {missing[:20]}")
        return output

    def close(self) -> None:
        for connection in (self._plan_db, self._geometry_db):
            if connection is not None:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
        self._plan_db = None
        self._geometry_db = None
        self._runtime_pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_plan_db=None, _geometry_db=None, _runtime_pid=None)
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class GeometryFeatureIndex:
    """Process-safe direct lookup into a completed geometry sidecar."""

    def __init__(self, index_path: str | Path, *, source_kind: str = "scene"):
        self.index_path = Path(index_path).expanduser().resolve()
        self.source_kind = str(source_kind)
        if self.source_kind not in {"scene", "render"}:
            raise ValueError("source_kind must be scene or render")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Geometry index does not exist: {self.index_path}")
        connection = _readonly_connection(self.index_path)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            if metadata.get("index_complete") != "1":
                raise ValueError(f"Incomplete geometry index: {self.index_path}")
        finally:
            connection.close()
        self._runtime_pid: int | None = None
        self._db: sqlite3.Connection | None = None

    def _ensure_runtime(self):
        if self._runtime_pid == os.getpid() and self._db is not None:
            return
        self.close()
        self._runtime_pid = os.getpid()
        self._db = _readonly_connection(self.index_path)

    def feature_for_record(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        self._ensure_runtime()
        if self.source_kind == "scene":
            row = self._db.execute(
                "SELECT * FROM frame_features WHERE frame_id=? AND gt_id=?",
                (int(record["frame_id"]), int(record.get("gt_id", 0))),
            ).fetchone()
        else:
            row = self._db.execute(
                "SELECT * FROM frame_features WHERE object_id=? AND view_id=?",
                (int(record["object_id"]), int(record["view_id"])),
            ).fetchone()
        return None if row is None else dict(row)

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
        state.update(_db=None, _runtime_pid=None)
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def normalized_focal_scalar(feature: Mapping[str, Any], *, maximum: bool = False) -> float:
    prefix = "max" if maximum else "base"
    fx = float(feature[f"{prefix}_norm_fx"])
    fy = float(feature[f"{prefix}_norm_fy"])
    if fx <= 0 or fy <= 0:
        raise ValueError("Normalized focal lengths must be positive")
    return math.sqrt(fx * fy)


def normalized_focal_shape(feature: Mapping[str, Any]) -> float:
    fx = float(feature["base_norm_fx"])
    fy = float(feature["base_norm_fy"])
    if fx <= 0 or fy <= 0:
        raise ValueError("Normalized focal lengths must be positive")
    return math.log(fx / fy)


def common_target_interval(
    query: Mapping[str, Any],
    plan: GeometryReferencePlan,
    *,
    relative_tolerance: float,
) -> tuple[float, float] | None:
    """Return a crop-only shared normalized-focal interval."""

    if not int(query["crop_feasible"]):
        return None
    query_low = normalized_focal_scalar(query)
    query_high = min(
        normalized_focal_scalar(query, maximum=True),
        query_low * (1.0 + float(relative_tolerance)),
    )
    low = max(query_low, float(plan.focal_low))
    high = min(query_high, float(plan.focal_high))
    shape = normalized_focal_shape(query)
    if not plan.focal_shape_low <= shape <= plan.focal_shape_high:
        return None
    if low > high + 1e-12:
        return None
    return float(low), float(high)


def object_preserving_crop_spec(
    feature: Mapping[str, Any],
    *,
    target_normalized_focal: float,
    aspect: float = 4.0 / 3.0,
    margin_fraction: float = 0.05,
    center_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Construct an integer crop that preserves an expanded full-object box.

    Crop-only augmentation cannot decrease focal length.  The returned crop is
    therefore rejected unless ``target_normalized_focal`` is inside the exact
    interval implied by the source focal, image bounds and expanded object box.
    """

    aspect = float(aspect)
    margin_fraction = float(margin_fraction)
    center_jitter = float(center_jitter)
    target = float(target_normalized_focal)
    if aspect <= 0 or target <= 0 or margin_fraction < 0:
        raise ValueError("Crop aspect/focal must be positive and margin non-negative")
    if not 0.0 <= center_jitter <= 1.0:
        raise ValueError("center_jitter must be in [0,1]")
    width, height = int(feature["width"]), int(feature["height"])
    fx, fy = float(feature["fx"]), float(feature["fy"])
    if min(width, height, fx, fy) <= 0:
        raise ValueError("Invalid source image or focal length")

    x, y = float(feature["bbox_obj_x"]), float(feature["bbox_obj_y"])
    box_w, box_h = float(feature["bbox_obj_w"]), float(feature["bbox_obj_h"])
    left = x - margin_fraction * box_w
    top = y - margin_fraction * box_h
    right = x + box_w + margin_fraction * box_w
    bottom = y + box_h + margin_fraction * box_h
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError("Expanded full-object box is outside the source image")

    ideal_width = math.sqrt(fx * fy * aspect) / target
    ratio = Fraction(aspect).limit_denominator(1000)
    if not math.isclose(float(ratio), aspect, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Crop aspect must have a practical integer-pixel ratio")
    minimum_height = max(
        int(round(ideal_width / aspect)), int(math.ceil(bottom - top)), 1
    )
    crop_h = int(math.ceil(minimum_height / ratio.denominator) * ratio.denominator)
    crop_w = crop_h * ratio.numerator // ratio.denominator
    while True:
        if crop_w > width or crop_h > height:
            raise ValueError("Requested focal cannot preserve the full object")
        l_min = max(0, int(math.ceil(right - crop_w)))
        l_max = min(width - crop_w, int(math.floor(left)))
        t_min = max(0, int(math.ceil(bottom - crop_h)))
        t_max = min(height - crop_h, int(math.floor(top)))
        if (
            crop_w + 1e-6 >= right - left
            and crop_h + 1e-6 >= bottom - top
            and l_min <= l_max
            and t_min <= t_max
        ):
            break
        # Fractional box boundaries sometimes need one additional integer
        # placement unit even though their continuous extents fit exactly.
        crop_h += ratio.denominator
        crop_w += ratio.numerator
    nominal_l = int(round((left + right - crop_w) * 0.5))
    nominal_t = int(round((top + bottom - crop_h) * 0.5))
    nominal_l = min(max(nominal_l, l_min), l_max)
    nominal_t = min(max(nominal_t, t_min), t_max)
    if rng is None:
        rng = np.random.default_rng(0)
    max_dx = center_jitter * crop_w
    max_dy = center_jitter * crop_h
    low_l = max(l_min, int(math.ceil(nominal_l - max_dx)))
    high_l = min(l_max, int(math.floor(nominal_l + max_dx)))
    low_t = max(t_min, int(math.ceil(nominal_t - max_dy)))
    high_t = min(t_max, int(math.floor(nominal_t + max_dy)))
    crop_l = int(rng.integers(low_l, high_l + 1)) if high_l > low_l else low_l
    crop_t = int(rng.integers(low_t, high_t + 1)) if high_t > low_t else low_t
    actual = math.sqrt((fx / crop_w) * (fy / crop_h))
    return {
        "format": "pi3_object_crop_v1",
        "bbox_xyxy": [crop_l, crop_t, crop_l + crop_w, crop_t + crop_h],
        "source_size_wh": [width, height],
        "target_normalized_focal": target,
        "actual_normalized_focal": float(actual),
        "center_shift_xy": [crop_l - nominal_l, crop_t - nominal_t],
        "expanded_object_bbox_xyxy": [left, top, right, bottom],
    }
