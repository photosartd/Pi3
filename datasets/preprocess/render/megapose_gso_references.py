#!/usr/bin/env python3
"""Render a clean, BOP-style reference bank for MegaPose-GSO objects.

The source GSO scene corpus is intentionally not modified.  Output object
folders use the same internal contract as ``lm-o/train/<object_id>`` but live
under a split-neutral root chosen by the caller.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image


FORMAT = "pi3_megapose_gso_reference_bank_v1"
OBJECT_FORMAT = "pi3_megapose_gso_reference_object_v1"
CAMERA_GENERATOR_VERSION = "fibonacci_look_at_v2"
LMO_WIDTH = 640
LMO_HEIGHT = 480
LMO_K = np.asarray(
    [
        [572.4114, 0.0, 325.2611],
        [0.0, 573.57043, 242.04899],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Object shards may publish the shared bank manifest concurrently. A
    # process-specific temporary keeps each replace atomic without workers
    # clobbering the same intermediate file.
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _save_image_atomic(
    path: Path,
    array: np.ndarray,
    *,
    png_compress_level: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(array).save(
        temporary,
        format="PNG",
        compress_level=int(png_compress_level),
    )
    os.replace(temporary, path)


def scaled_lmo_intrinsics(width: int, height: int) -> np.ndarray:
    """Scale LM-O's clean-render intrinsics without changing its field of view."""

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    K = LMO_K.copy()
    K[0, :] *= width / LMO_WIDTH
    K[1, :] *= height / LMO_HEIGHT
    K[2, :] = (0.0, 0.0, 1.0)
    return K


def fibonacci_sphere(count: int, sphere: str = "full") -> np.ndarray:
    """Return deterministic, approximately equal-area camera-center directions."""

    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    if sphere not in {"full", "upper"}:
        raise ValueError("sphere must be 'full' or 'upper'")
    indices = np.arange(count, dtype=np.float64)
    if sphere == "full":
        z = 1.0 - 2.0 * (indices + 0.5) / count
    else:
        z = (indices + 0.5) / count
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    azimuth = indices * golden_angle
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.stack(
        [radial * np.cos(azimuth), radial * np.sin(azimuth), z], axis=1
    )
    return directions


def _jitter_direction(
    direction: np.ndarray,
    maximum_degrees: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    if maximum_degrees <= 0:
        return np.asarray(direction, dtype=np.float64), 0.0
    direction = np.asarray(direction, dtype=np.float64)
    helper = np.asarray((0.0, 0.0, 1.0))
    if abs(float(direction @ helper)) > 0.9:
        helper = np.asarray((0.0, 1.0, 0.0))
    tangent_x = np.cross(direction, helper)
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(direction, tangent_x)
    tangent_azimuth = rng.uniform(0.0, 2.0 * math.pi)
    tangent = (
        math.cos(tangent_azimuth) * tangent_x
        + math.sin(tangent_azimuth) * tangent_y
    )
    angle_deg = float(rng.uniform(-maximum_degrees, maximum_degrees))
    angle = math.radians(angle_deg)
    jittered = math.cos(angle) * direction + math.sin(angle) * tangent
    jittered /= np.linalg.norm(jittered)
    return jittered, angle_deg


def look_at_T_C_O(camera_center_O: np.ndarray, roll_rad: float) -> np.ndarray:
    """Make OpenCV ``T_C_O`` for a camera centered at ``camera_center_O``.

    The optical +Z axis looks at the object origin and camera +Y points down.
    ``roll_rad`` rotates the image axes about the optical axis.
    """

    center = np.asarray(camera_center_O, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("camera_center_O must be a finite length-three vector")
    radius = float(np.linalg.norm(center))
    if radius <= 0:
        raise ValueError("Camera center must not coincide with the object origin")
    forward = -center / radius
    up_hint = np.asarray((0.0, 0.0, 1.0))
    if abs(float(forward @ up_hint)) > 0.9:
        up_hint = np.asarray((0.0, 1.0, 0.0))
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    cosine, sine = math.cos(float(roll_rad)), math.sin(float(roll_rad))
    rolled_right = cosine * right + sine * down
    rolled_down = -sine * right + cosine * down
    R_C_O = np.stack([rolled_right, rolled_down, forward], axis=0)
    if not np.allclose(R_C_O @ R_C_O.T, np.eye(3), atol=1e-7):
        raise ValueError("Generated camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(R_C_O), 1.0, atol=1e-7):
        raise ValueError("Generated camera rotation is not right-handed")
    T_C_O = np.eye(4, dtype=np.float64)
    T_C_O[:3, :3] = R_C_O
    T_C_O[:3, 3] = -R_C_O @ center
    return T_C_O


def generate_camera_bank(
    *,
    num_views: int,
    sphere: str,
    radius_m: float,
    view_jitter_deg: float,
    radius_jitter_fraction: float,
    roll_mode: str,
    roll_jitter_deg: float,
    seed: int,
    object_id: int,
) -> list[dict[str, Any]]:
    """Generate an object-specific deterministic sphere/SO(3) camera bank."""

    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if view_jitter_deg < 0 or roll_jitter_deg < 0:
        raise ValueError("Angular jitter must be non-negative")
    if not 0 <= radius_jitter_fraction < 1:
        raise ValueError("radius_jitter_fraction must be in [0, 1)")
    if roll_mode not in {"upright", "low_discrepancy"}:
        raise ValueError("roll_mode must be 'upright' or 'low_discrepancy'")

    directions = fibonacci_sphere(num_views, sphere=sphere)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(object_id)]))
    bank = []
    for view_id, base_direction in enumerate(directions):
        direction, applied_view_jitter_deg = _jitter_direction(
            base_direction, float(view_jitter_deg), rng
        )
        radius_scale = 1.0 + float(
            rng.uniform(-radius_jitter_fraction, radius_jitter_fraction)
        )
        radius = float(radius_m * radius_scale)
        if roll_mode == "upright":
            base_roll_rad = 0.0
        else:
            # A second irrational sequence avoids locking roll to the golden
            # azimuth sequence used by the sphere points.
            fraction = ((view_id + 0.5) * math.sqrt(2.0)) % 1.0
            base_roll_rad = 2.0 * math.pi * (fraction - 0.5)
        applied_roll_jitter_deg = float(
            rng.uniform(-roll_jitter_deg, roll_jitter_deg)
        )
        roll_rad = base_roll_rad + math.radians(applied_roll_jitter_deg)
        center = direction * radius
        T_C_O = look_at_T_C_O(center, roll_rad)
        bank.append(
            {
                "view_id": int(view_id),
                "base_direction_O": base_direction.tolist(),
                "camera_center_O_m": center.tolist(),
                "radius_m": radius,
                "view_jitter_deg": applied_view_jitter_deg,
                "roll_deg": math.degrees(roll_rad),
                "roll_jitter_deg": applied_roll_jitter_deg,
                "T_C_O": T_C_O,
            }
        )
    return bank


def sequential_camera_order(
    camera_bank: list[dict[str, Any]],
    *,
    candidate_starts: int = 64,
    two_opt_passes: int = 8,
) -> list[dict[str, Any]]:
    """Order a fixed pose set as a smooth open path in camera SO(3).

    The uniformly sampled poses are unchanged.  Only their output IDs are
    reassigned.  SO(3) distance accounts for both motion over the view sphere
    and in-plane roll, unlike latitude/azimuth sorting.
    """

    count = len(camera_bank)
    if count <= 1:
        return [
            {**pose, "coverage_view_id": int(pose["view_id"]), "view_id": index}
            for index, pose in enumerate(camera_bank)
        ]
    rotations = np.asarray(
        [np.asarray(pose["T_C_O"], dtype=np.float64)[:3, :3] for pose in camera_bank]
    )
    rotation_dots = np.einsum("aij,bij->ab", rotations, rotations)
    distances = np.arccos(np.clip((rotation_dots - 1.0) / 2.0, -1.0, 1.0))

    starts = np.unique(
        np.linspace(0, count - 1, min(int(candidate_starts), count), dtype=np.int64)
    )
    best_order: list[int] | None = None
    best_score: tuple[float, float] | None = None
    for raw_start in starts:
        current = int(raw_start)
        unused = np.ones(count, dtype=bool)
        unused[current] = False
        order = [current]
        for _ in range(count - 1):
            candidates = distances[current].copy()
            candidates[~unused] = np.inf
            current = int(np.argmin(candidates))
            unused[current] = False
            order.append(current)
        edges = distances[order[:-1], order[1:]]
        score = (float(edges.max()), float(edges.sum()))
        if best_score is None or score < best_score:
            best_order, best_score = order, score
    assert best_order is not None

    # An open-path 2-opt pass removes avoidable crossings. Reversing an
    # interior section preserves all its internal undirected edge lengths, so
    # only the two boundary edges need to be compared.
    for _ in range(int(two_opt_passes)):
        changed = False
        for first in range(count - 3):
            for second in range(first + 2, count - 1):
                # A previous reversal in this loop may have changed the edge
                # at ``first``. Read all four endpoints from the current path
                # for every proposal rather than retaining a stale ``b``.
                a, b = best_order[first], best_order[first + 1]
                c, d = best_order[second], best_order[second + 1]
                old_edges = (distances[a, b], distances[c, d])
                new_edges = (distances[a, c], distances[b, d])
                # The sum comparison guarantees that the total trajectory
                # length never increases; maximum step is a deterministic
                # tie-breaker. The multi-start greedy seed already explicitly
                # minimizes the worst step before total length.
                old_score = (sum(old_edges), max(old_edges))
                new_score = (sum(new_edges), max(new_edges))
                if new_score < old_score:
                    best_order[first + 1 : second + 1] = reversed(
                        best_order[first + 1 : second + 1]
                    )
                    changed = True
        if not changed:
            break

    ordered = []
    for sequential_view_id, coverage_view_id in enumerate(best_order):
        pose = dict(camera_bank[coverage_view_id])
        pose["coverage_view_id"] = int(pose["view_id"])
        pose["view_id"] = int(sequential_view_id)
        ordered.append(pose)
    return ordered


def mask_bbox(mask: np.ndarray) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError("Rendered object mask is empty")
    x0, x1 = int(x.min()), int(x.max())
    y0, y1 = int(y.min()), int(y.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def bop_frame_metadata(
    *,
    object_id: int,
    T_C_O: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    depth_scale: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    mask = np.asarray(mask, dtype=bool)
    pixels = int(np.count_nonzero(mask))
    bbox = mask_bbox(mask)
    T_C_O = np.asarray(T_C_O, dtype=np.float64)
    scene_camera = {
        "cam_K": np.asarray(K, dtype=np.float64).reshape(-1).tolist(),
        "depth_scale": float(depth_scale),
    }
    scene_gt = [
        {
            "cam_R_m2c": T_C_O[:3, :3].reshape(-1).tolist(),
            "cam_t_m2c": (T_C_O[:3, 3] * 1000.0).tolist(),
            "obj_id": int(object_id),
        }
    ]
    scene_gt_info = [
        {
            "bbox_obj": bbox,
            "bbox_visib": bbox,
            "px_count_all": pixels,
            "px_count_valid": pixels,
            "px_count_visib": pixels,
            "visib_fract": 1.0,
        }
    ]
    return scene_camera, scene_gt, scene_gt_info


def load_mapping(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    mapping = {int(row["obj_id"]): str(row["gso_id"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"Duplicate object IDs in {path}")
    return mapping


def load_index_object_ids(path: Path) -> set[int]:
    path = path.expanduser().resolve()
    uri = f"file:{path}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute("SELECT object_id FROM objects").fetchall()
    return {int(row[0]) for row in rows}


def select_object_ids(
    *,
    mapping: Mapping[int, str],
    requested: list[int],
    index_path: Path | None,
    shard_count: int,
    shard_index: int,
    max_objects: int | None,
) -> list[int]:
    if requested:
        object_ids = sorted(set(int(value) for value in requested))
    elif index_path is not None:
        object_ids = sorted(load_index_object_ids(index_path))
    else:
        object_ids = sorted(mapping)
    unknown = sorted(set(object_ids) - set(mapping))
    if unknown:
        raise ValueError(f"Object IDs absent from GSO mapping: {unknown[:20]}")
    object_ids = [
        object_id
        for position, object_id in enumerate(object_ids)
        if position % shard_count == shard_index
    ]
    if max_objects is not None:
        object_ids = object_ids[: int(max_objects)]
    return object_ids


def _fingerprint(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_commit(module_file: Path) -> str | None:
    for parent in module_file.resolve().parents:
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"], text=True
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                return None
    return None


def _expected_paths(object_dir: Path, num_views: int) -> list[Path]:
    paths = []
    for view_id in range(int(num_views)):
        paths.extend(
            [
                object_dir / "rgb" / f"{view_id:06d}.png",
                object_dir / "depth" / f"{view_id:06d}.png",
                object_dir / "mask" / f"{view_id:06d}_000000.png",
                object_dir / "mask_visib" / f"{view_id:06d}_000000.png",
            ]
        )
    paths.extend(
        object_dir / name
        for name in ("scene_camera.json", "scene_gt.json", "scene_gt_info.json")
    )
    return paths


def close_renderer_buffers(renderer) -> None:
    """Release auxiliary offscreen buffers before Panda3D/GLX interpreter exit."""

    for cameras in renderer._cameras_pool.values():
        for camera in cameras:
            camera.node_path.node().setActive(0)
            camera.node_path.removeNode()
            camera.graphics_buffer.clearRenderTextures()
            renderer._app.graphicsEngine.removeWindow(camera.graphics_buffer)
    renderer._cameras_pool.clear()
    renderer._app.graphicsEngine.renderFrame()
    renderer._app.graphicsEngine.syncFrame()


def release_renderer_object(renderer, label: str) -> dict[str, int]:
    """Release one loaded object's CPU and prepared GPU resources.

    MegaPose loads models with ``noCache=True``, but referenced image textures
    still enter Panda3D's global TexturePool. Removing only the Python label
    entry therefore retains roughly one decompressed 4K texture per object in
    the OpenGL context. Explicitly release both textures and prepared geometry
    before dropping the model node.
    """

    from panda3d import core as p3d

    node = renderer._label_to_node.pop(label, None)
    if node is None:
        return {"textures": 0, "geoms": 0}

    texture_collection = node.findAllTextures()
    texture_collection.removeDuplicateTextures()
    textures = [
        texture_collection.getTexture(index)
        for index in range(texture_collection.getNumTextures())
    ]
    geom_count = 0
    for geom_path in node.findAllMatches("**/+GeomNode"):
        geom_node = geom_path.node()
        geom_count += geom_node.getNumGeoms()
        # Panda3D's Python binding exposes loaded Geoms as const objects, so
        # their release methods cannot be called directly. Dropping the
        # GeomNode references lets normal ref-count destruction release their
        # prepared contexts without creating copy-on-write duplicates.
        geom_node.removeAllGeoms()

    node.clearTexture()
    node.removeNode()
    for texture in textures:
        p3d.TexturePool.releaseTexture(texture)
        texture.releaseAll()

    p3d.TexturePool.garbageCollect()
    p3d.ModelPool.garbageCollect()
    for _ in range(3):
        p3d.RenderState.garbageCollect()
        p3d.TransformState.garbageCollect()
    renderer._app.graphicsEngine.renderFrame()
    renderer._app.graphicsEngine.syncFrame()
    return {"textures": len(textures), "geoms": geom_count}


def renderer_driver_info(renderer) -> dict[str, str]:
    gsg = renderer._app.win.getGsg()
    return {
        "vendor": str(gsg.getDriverVendor()),
        "renderer": str(gsg.getDriverRenderer()),
        "version": str(gsg.getDriverVersion()),
    }


def validate_object_output(
    object_dir: Path,
    *,
    num_views: int,
    config_fingerprint: str,
) -> bool:
    manifest_path = object_dir / "reference_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError):
        return False
    if (
        manifest.get("format") != OBJECT_FORMAT
        or not manifest.get("complete")
        or manifest.get("config_fingerprint") != config_fingerprint
        or int(manifest.get("view_count", -1)) != int(num_views)
    ):
        return False
    return all(path.is_file() for path in _expected_paths(object_dir, num_views))


def _write_frame(
    *,
    object_dir: Path,
    view_id: int,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    png_compress_level: int,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
        depth_m = depth_m[..., 0]
    mask = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(mask):
        raise ValueError(f"Empty render for view {view_id}")
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError(f"Object touches image boundary in view {view_id}")
    raw_depth = np.rint(depth_m * 1000.0 / depth_scale)
    if float(raw_depth.max(initial=0.0)) > np.iinfo(np.uint16).max:
        raise ValueError(f"Depth exceeds uint16 range in view {view_id}")
    depth_u16 = raw_depth.astype(np.uint16)
    depth_u16[~mask] = 0
    mask_u8 = mask.astype(np.uint8) * 255
    _save_image_atomic(
        object_dir / "rgb" / f"{view_id:06d}.png",
        np.asarray(rgb, dtype=np.uint8),
        png_compress_level=png_compress_level,
    )
    _save_image_atomic(
        object_dir / "depth" / f"{view_id:06d}.png",
        depth_u16,
        png_compress_level=png_compress_level,
    )
    for folder in ("mask", "mask_visib"):
        _save_image_atomic(
            object_dir / folder / f"{view_id:06d}_000000.png",
            mask_u8,
            png_compress_level=png_compress_level,
        )
    return depth_u16, mask


def render_object(
    *,
    object_id: int,
    gso_id: str,
    output_root: Path,
    config_fingerprint: str,
    renderer,
    lights,
    args,
) -> dict[str, Any]:
    # Imports stay local so camera-generation tests and --dry-run do not need
    # MegaPose, Panda3D, or their Python 3.9 environment.
    from megapose.lib3d.transform import Transform
    from megapose.panda3d_renderer.types import (
        Panda3dCameraData,
        Panda3dObjectData,
    )

    object_dir = output_root / f"{object_id:06d}"
    if validate_object_output(
        object_dir,
        num_views=args.num_views,
        config_fingerprint=config_fingerprint,
    ):
        print(f"Reusing complete object {object_id:06d} ({gso_id})")
        return {"object_id": object_id, "gso_id": gso_id, "status": "reused"}
    existing_manifest = object_dir / "reference_manifest.json"
    if existing_manifest.is_file():
        with existing_manifest.open("r", encoding="utf-8") as stream:
            old_manifest = json.load(stream)
        old_fingerprint = old_manifest.get("config_fingerprint")
        if old_fingerprint not in {None, config_fingerprint}:
            raise ValueError(
                f"Existing {object_dir} uses a different render configuration. "
                "Choose another --output-root instead of mixing reference banks."
            )

    camera_bank = generate_camera_bank(
        num_views=args.num_views,
        sphere=args.sphere,
        radius_m=args.radius_m,
        view_jitter_deg=args.view_jitter_deg,
        radius_jitter_fraction=args.radius_jitter_fraction,
        roll_mode=args.roll_mode,
        roll_jitter_deg=args.roll_jitter_deg,
        seed=args.seed,
        object_id=object_id,
    )
    if args.view_order == "sequential":
        camera_bank = sequential_camera_order(camera_bank)
    K = scaled_lmo_intrinsics(args.width, args.height)
    label = f"gso_{gso_id}"
    object_data = [Panda3dObjectData(label=label, TWO=Transform(np.eye(4)))]

    scene_camera: dict[str, Any] = {}
    scene_gt: dict[str, Any] = {}
    scene_gt_info: dict[str, Any] = {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.write_workers) as pool:
        for start in range(0, args.num_views, args.camera_batch_size):
            batch = camera_bank[start : start + args.camera_batch_size]
            camera_data = []
            for pose in batch:
                T_O_C = np.linalg.inv(pose["T_C_O"])
                camera_data.append(
                    Panda3dCameraData(
                        K=K,
                        resolution=(args.height, args.width),
                        TWC=Transform(T_O_C),
                        z_near=args.z_near,
                        z_far=args.z_far,
                    )
                )
            renderings = renderer.render_scene(
                object_data,
                camera_data,
                lights,
                render_depth=True,
                render_binary_mask=False,
            )
            futures = []
            for pose, rendering in zip(batch, renderings):
                if rendering.depth is None:
                    raise RuntimeError("Panda3D renderer returned no depth")
                futures.append(
                    (
                        pose,
                        pool.submit(
                            _write_frame,
                            object_dir=object_dir,
                            view_id=pose["view_id"],
                            rgb=np.asarray(rendering.rgb).copy(),
                            depth_m=np.asarray(rendering.depth).copy(),
                            png_compress_level=args.png_compress_level,
                            depth_scale=args.depth_scale,
                        ),
                    )
                )
            for pose, future in futures:
                _, mask = future.result()
                camera_meta, gt_meta, info_meta = bop_frame_metadata(
                    object_id=object_id,
                    T_C_O=pose["T_C_O"],
                    K=K,
                    mask=mask,
                    depth_scale=args.depth_scale,
                )
                key = str(pose["view_id"])
                scene_camera[key] = camera_meta
                scene_gt[key] = gt_meta
                scene_gt_info[key] = info_meta
            print(
                f"Object {object_id:06d}: rendered/wrote "
                f"{min(start + len(batch), args.num_views)}/{args.num_views}"
            )

    _write_json_atomic(object_dir / "scene_camera.json", scene_camera)
    _write_json_atomic(object_dir / "scene_gt.json", scene_gt)
    _write_json_atomic(object_dir / "scene_gt_info.json", scene_gt_info)
    elapsed = time.monotonic() - started
    pose_records = []
    for pose in camera_bank:
        record = {key: value for key, value in pose.items() if key != "T_C_O"}
        record["T_C_O"] = pose["T_C_O"].reshape(-1).tolist()
        pose_records.append(record)
    manifest = {
        "format": OBJECT_FORMAT,
        "complete": True,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "config_fingerprint": config_fingerprint,
        "object_id": int(object_id),
        "gso_id": gso_id,
        "view_count": args.num_views,
        "elapsed_seconds": elapsed,
        "poses": pose_records,
    }
    _write_json_atomic(object_dir / "reference_manifest.json", manifest)
    if not validate_object_output(
        object_dir,
        num_views=args.num_views,
        config_fingerprint=config_fingerprint,
    ):
        raise RuntimeError(f"Post-write validation failed for {object_dir}")
    # The renderer and its camera buffers are reused across objects. Explicitly
    # release the just-used mesh and its Panda3D-prepared GPU resources so 944
    # decoded 4K textures do not accumulate in VRAM.
    released = release_renderer_object(renderer, label)
    gc.collect()
    print(
        f"Completed object {object_id:06d} ({gso_id}) in {elapsed:.1f}s "
        f"({args.num_views / max(elapsed, 1e-9):.1f} views/s); "
        f"released={released['textures']} textures/{released['geoms']} geoms"
    )
    return {
        "object_id": object_id,
        "gso_id": gso_id,
        "status": "rendered",
        "elapsed_seconds": elapsed,
    }


def make_bank_config(args, mapping_path: Path) -> dict[str, Any]:
    K = scaled_lmo_intrinsics(args.width, args.height)
    config = {
        "format": FORMAT,
        "coordinate_convention": {
            "T_C_O": "object/CAD coordinates to OpenCV camera coordinates",
            "camera_pose": "T_O_C = inverse(T_C_O)",
            "camera_axes": "+X right, +Y down, +Z forward",
            "translation_unit": "scene_gt millimetres; manifests metres",
            "depth_unit": "uint16 * depth_scale millimetres",
        },
        "layout": "split-neutral BOP object folders",
        "camera_generator_version": CAMERA_GENERATOR_VERSION,
        "mapping_path": str(mapping_path),
        "num_views": args.num_views,
        "sphere": args.sphere,
        "radius_m": args.radius_m,
        "view_jitter_deg": args.view_jitter_deg,
        "radius_jitter_fraction": args.radius_jitter_fraction,
        "roll_mode": args.roll_mode,
        "roll_jitter_deg": args.roll_jitter_deg,
        "seed": args.seed,
        "resolution_wh": [args.width, args.height],
        "K": K.reshape(-1).tolist(),
        "intrinsics_profile": "LM-O train intrinsics scaled from 640x480",
        "lighting": args.lighting,
        "display_backend": args.display_backend,
        "depth_scale": args.depth_scale,
        "z_near": args.z_near,
        "z_far": args.z_far,
    }
    # Keep the historical coverage-order fingerprint unchanged unless the new
    # ordering is explicitly requested, so completed banks remain resumable.
    if args.view_order == "sequential":
        config["view_order"] = "sequential"
        config["view_order_method"] = "multistart_greedy_so3_then_2opt_v1"
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render split-neutral BOP-style clean references for GSO objects."
    )
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mapping-path", type=Path)
    parser.add_argument(
        "--index-path",
        type=Path,
        help="Render only object IDs present in a prepared MegaPose-GSO index.",
    )
    parser.add_argument(
        "--object-id", type=int, action="append", default=[], help="Repeat to render a subset."
    )
    parser.add_argument("--object-shard-count", type=int, default=1)
    parser.add_argument("--object-shard-index", type=int, default=0)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--num-views", type=int, default=256)
    parser.add_argument(
        "--view-order",
        choices=("coverage", "sequential"),
        default="coverage",
        help=(
            "coverage preserves historical Fibonacci IDs; sequential assigns "
            "filenames along a smooth path through the same full pose set"
        ),
    )
    parser.add_argument("--sphere", choices=("full", "upper"), default="full")
    parser.add_argument(
        "--radius-m",
        type=float,
        default=0.5,
        help=(
            "Nominal object-to-camera distance. The 0.5 m default safely frames "
            "the largest normalized GSO bounding boxes at the configured jitter."
        ),
    )
    parser.add_argument("--view-jitter-deg", type=float, default=2.0)
    parser.add_argument("--radius-jitter-fraction", type=float, default=0.05)
    parser.add_argument(
        "--roll-mode",
        choices=("low_discrepancy", "upright"),
        default="low_discrepancy",
    )
    parser.add_argument("--roll-jitter-deg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(720, 540),
    )
    parser.add_argument("--lighting", choices=("studio", "ambient"), default="studio")
    parser.add_argument(
        "--display-backend",
        choices=("p3headlessgl", "pandagl"),
        default="p3headlessgl",
        help="Panda3D display pipe; p3headlessgl uses EGL and exits cleanly headlessly.",
    )
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--z-near", type=float, default=0.01)
    parser.add_argument("--z-far", type=float, default=10.0)
    parser.add_argument("--camera-batch-size", type=int, default=32)
    parser.add_argument("--write-workers", type=int, default=8)
    parser.add_argument("--png-compress-level", type=int, default=3)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.width, args.height = (int(value) for value in args.resolution)
    positive = {
        "--num-views": args.num_views,
        "--radius-m": args.radius_m,
        "--depth-scale": args.depth_scale,
        "--z-near": args.z_near,
        "--z-far": args.z_far,
        "--camera-batch-size": args.camera_batch_size,
        "--write-workers": args.write_workers,
        "--object-shard-count": args.object_shard_count,
        "WIDTH": args.width,
        "HEIGHT": args.height,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if not 0 <= args.object_shard_index < args.object_shard_count:
        parser.error("--object-shard-index must be in [0, --object-shard-count)")
    if args.max_objects is not None and args.max_objects <= 0:
        parser.error("--max-objects must be positive")
    if args.view_jitter_deg < 0 or args.roll_jitter_deg < 0:
        parser.error("Angular jitter must be non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not 0 <= args.radius_jitter_fraction < 1:
        parser.error("--radius-jitter-fraction must be in [0, 1)")
    if not 0 <= args.png_compress_level <= 9:
        parser.error("--png-compress-level must be in [0, 9]")
    if args.z_far <= args.z_near:
        parser.error("--z-far must exceed --z-near")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    assets_root = args.assets_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not assets_root.is_dir():
        raise FileNotFoundError(f"Assets root does not exist: {assets_root}")
    mapping_path = (
        assets_root / "gso_models.json"
        if args.mapping_path is None
        else args.mapping_path.expanduser().resolve()
    )
    mapping = load_mapping(mapping_path)
    object_ids = select_object_ids(
        mapping=mapping,
        requested=args.object_id,
        index_path=args.index_path,
        shard_count=args.object_shard_count,
        shard_index=args.object_shard_index,
        max_objects=args.max_objects,
    )
    if not object_ids:
        raise ValueError("Object selection is empty")
    bank_config = make_bank_config(args, mapping_path)
    config_fingerprint = _fingerprint(bank_config)
    print(
        f"Selected {len(object_ids)} objects, {args.num_views} views/object, "
        f"resolution={args.width}x{args.height}, fingerprint={config_fingerprint[:12]}"
    )
    if args.dry_run:
        print("Object IDs:", " ".join(map(str, object_ids)))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    existing_config_path = output_root / "reference_bank.json"
    if existing_config_path.is_file():
        with existing_config_path.open("r", encoding="utf-8") as stream:
            existing_config = json.load(stream)
        if existing_config.get("config_fingerprint") != config_fingerprint:
            raise ValueError(
                f"Existing bank configuration differs at {existing_config_path}; "
                "choose another --output-root"
            )
    runtime_dir = (
        assets_root / ".megapose_runtime"
        if args.runtime_dir is None
        else args.runtime_dir.expanduser().resolve()
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MEGAPOSE_DATA_DIR", str(runtime_dir))
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
        raise ValueError("MegaPose Panda3D requires exactly one CUDA_VISIBLE_DEVICES entry")

    # Import only after renderer environment variables are set.
    import megapose
    from panda3d.core import ConfigVariableString
    from megapose.datasets.gso_dataset import GoogleScannedObjectDataset
    from megapose.panda3d_renderer.panda3d_scene_renderer import (
        Panda3dSceneRenderer,
        make_scene_lights,
    )
    from megapose.panda3d_renderer.types import Panda3dLightData

    # MegaPose's App requests pandagl, but a Panda ConfigVariable local value
    # has higher precedence. The wheel contains p3headlessgl, which avoids GLX
    # teardown failures in non-interactive/headless jobs.
    ConfigVariableString("load-display").setValue(args.display_backend)

    published_config = {
        **bank_config,
        "config_fingerprint": config_fingerprint,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "megapose_module": str(Path(megapose.__file__).resolve()),
            "megapose_commit": _source_commit(Path(megapose.__file__)),
            "panda3d_version": _package_version("panda3d"),
            "panda3d_gltf_version": _package_version("panda3d-gltf"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "egl_visible_devices": os.environ.get("EGL_VISIBLE_DEVICES"),
        },
    }
    labels = {f"gso_{mapping[object_id]}" for object_id in object_ids}
    object_dataset = GoogleScannedObjectDataset(
        assets_root / "google_scanned_objects", split="normalized"
    ).filter_objects(labels)
    if len(object_dataset) != len(labels):
        raise ValueError(
            f"Renderer model coverage is {len(object_dataset)}/{len(labels)}"
        )
    # Reusing one renderer keeps the camera/render buffers bounded. Individual
    # mesh nodes are evicted after each object in render_object().
    renderer = Panda3dSceneRenderer(object_dataset, preload_labels=set(), verbose=False)
    driver_info = renderer_driver_info(renderer)
    print(
        "Panda3D driver: "
        f"{driver_info['vendor']} | {driver_info['renderer']} | "
        f"{driver_info['version']}"
    )
    published_config["graphics_driver"] = driver_info
    if existing_config_path.exists():
        published_config["created_at_utc"] = existing_config.get(
            "created_at_utc", published_config["created_at_utc"]
        )
    _write_json_atomic(existing_config_path, published_config)
    if args.lighting == "studio":
        lights = make_scene_lights()
    else:
        lights = [
            Panda3dLightData(
                light_type="ambient", color=(1.0, 1.0, 1.0, 1.0)
            )
        ]

    results = []
    started = time.monotonic()
    for position, object_id in enumerate(object_ids, start=1):
        print(f"[{position}/{len(object_ids)}] Starting object {object_id:06d}")
        results.append(
            render_object(
                object_id=object_id,
                gso_id=mapping[object_id],
                output_root=output_root,
                config_fingerprint=config_fingerprint,
                renderer=renderer,
                lights=lights,
                args=args,
            )
        )
    close_renderer_buffers(renderer)
    elapsed = time.monotonic() - started
    rendered = sum(result["status"] == "rendered" for result in results)
    reused = sum(result["status"] == "reused" for result in results)
    print(
        f"Reference bank task complete: rendered={rendered}, reused={reused}, "
        f"elapsed={elapsed:.1f}s, "
        f"peak_process_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.2f} GiB, "
        f"output={output_root}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
