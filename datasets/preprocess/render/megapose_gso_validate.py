#!/usr/bin/env python3
"""Re-render indexed MegaPose-GSO objects at GT poses and compare geometry."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw


REPORT_FORMAT = "pi3_megapose_gso_gt_render_validation_v1"


def decode_uncompressed_rle(rle: Mapping[str, Any]) -> np.ndarray:
    size = rle.get("size")
    counts = rle.get("counts")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("Mask RLE must contain size=[height, width]")
    if not isinstance(counts, list):
        raise ValueError("Only released uncompressed COCO RLE masks are supported")
    height, width = (int(value) for value in size)
    flat = np.zeros(height * width, dtype=bool)
    position = 0
    for run_index, raw_count in enumerate(counts):
        count = int(raw_count)
        if count < 0 or position + count > flat.size:
            raise ValueError("Invalid RLE run length")
        if run_index % 2:
            flat[position : position + count] = True
        position += count
    if position != flat.size:
        raise ValueError(f"RLE covers {position} pixels, expected {flat.size}")
    return flat.reshape((height, width), order="F")


def mask_from_payload(payload: bytes, gt_id: int) -> np.ndarray:
    decoded = json.loads(payload)
    if isinstance(decoded, list):
        rle = decoded[gt_id] if gt_id < len(decoded) else None
    elif isinstance(decoded, dict):
        rle = decoded.get(str(gt_id))
    else:
        rle = None
    if not isinstance(rle, dict):
        raise ValueError(f"Mask payload has no entry for gt_id={gt_id}")
    return decode_uncompressed_rle(rle)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = np.count_nonzero(first | second)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(first & second) / union)


def geometry_metrics(
    *,
    rendered_mask: np.ndarray,
    rendered_depth_m: np.ndarray,
    source_full_mask: np.ndarray,
    source_visible_mask: np.ndarray,
    source_depth_m: np.ndarray,
) -> dict[str, float | int]:
    rendered_mask = np.asarray(rendered_mask, dtype=bool)
    source_full_mask = np.asarray(source_full_mask, dtype=bool)
    source_visible_mask = np.asarray(source_visible_mask, dtype=bool)
    rendered_depth_m = np.asarray(rendered_depth_m, dtype=np.float64)
    source_depth_m = np.asarray(source_depth_m, dtype=np.float64)
    shapes = {
        rendered_mask.shape,
        rendered_depth_m.shape,
        source_full_mask.shape,
        source_visible_mask.shape,
        source_depth_m.shape,
    }
    if len(shapes) != 1:
        raise ValueError(f"Render/source shapes differ: {sorted(shapes)}")

    depth_valid = (
        rendered_mask
        & source_visible_mask
        & np.isfinite(rendered_depth_m)
        & np.isfinite(source_depth_m)
        & (rendered_depth_m > 0)
        & (source_depth_m > 0)
    )
    errors = np.abs(rendered_depth_m[depth_valid] - source_depth_m[depth_valid])
    visible_pixels = int(np.count_nonzero(source_visible_mask))
    full_pixels = int(np.count_nonzero(source_full_mask))
    return {
        "full_mask_iou": mask_iou(rendered_mask, source_full_mask),
        "visible_mask_render_recall": float(
            np.count_nonzero(rendered_mask & source_visible_mask) / max(1, visible_pixels)
        ),
        "rendered_pixels": int(np.count_nonzero(rendered_mask)),
        "source_full_pixels": full_pixels,
        "source_visible_pixels": visible_pixels,
        "depth_comparison_pixels": int(len(errors)),
        "depth_comparison_fraction_visible": float(len(errors) / max(1, visible_pixels)),
        "depth_abs_mean_m": float(np.mean(errors)) if len(errors) else float("nan"),
        "depth_abs_median_m": float(np.median(errors)) if len(errors) else float("nan"),
        "depth_abs_p95_m": float(np.percentile(errors, 95)) if len(errors) else float("nan"),
    }


def _load_split_object_ids(path: Path) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("format") != "pi3_entity_split_v1":
        raise ValueError(f"Unsupported split manifest: {path}")
    return {
        split: [int(value) for value in payload["splits"][split]["object_ids"]]
        for split in ("train", "val")
    }


def select_validation_records(
    *,
    index_path: Path,
    split_path: Path,
    samples_per_split: int,
    seed: int,
    minimum_visibility: float,
    minimum_visible_pixels: int,
) -> list[dict[str, Any]]:
    split_ids = _load_split_object_ids(split_path)
    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    query = """
        SELECT i.frame_id, i.gt_id, i.scene_id, i.view_id, i.object_id,
               i.T_C_O_f32, i.visib_fract, i.px_count_visib,
               f.frame_key, f.width, f.height, f.depth_unit_m, f.K_f32,
               f.rgb_offset, f.rgb_size, f.depth_offset, f.depth_size,
               f.mask_offset, f.mask_size,
               f.mask_visib_offset, f.mask_visib_size,
               s.relative_path
        FROM instances AS i
        JOIN frames AS f ON f.id = i.frame_id
        JOIN shards AS s ON s.id = f.shard_id
        WHERE i.object_id = ? AND i.depth_corrupt = 0
          AND i.visib_fract >= ? AND i.px_count_visib >= ?
        ORDER BY i.visib_fract DESC, i.px_count_visib DESC,
                 i.scene_id, i.view_id, i.gt_id
        LIMIT 1
    """
    uri = f"file:{index_path.expanduser().resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for split_name in ("train", "val"):
            candidates = list(split_ids[split_name])
            rng.shuffle(candidates)
            split_records = []
            for object_id in candidates:
                row = connection.execute(
                    query,
                    (object_id, float(minimum_visibility), int(minimum_visible_pixels)),
                ).fetchone()
                if row is not None:
                    record = dict(row)
                    record["split"] = split_name
                    split_records.append(record)
                if len(split_records) == samples_per_split:
                    break
            if len(split_records) != samples_per_split:
                raise ValueError(
                    f"Only found {len(split_records)}/{samples_per_split} eligible "
                    f"records for split={split_name}"
                )
            selected.extend(split_records)
    return selected


def _read_payload(data_root: Path, record: Mapping[str, Any], payload: str) -> bytes:
    path = data_root / str(record["relative_path"])
    descriptor = os.open(path, os.O_RDONLY)
    try:
        data = os.pread(
            descriptor,
            int(record[f"{payload}_size"]),
            int(record[f"{payload}_offset"]),
        )
    finally:
        os.close(descriptor)
    expected = int(record[f"{payload}_size"])
    if len(data) != expected:
        raise IOError(f"Short read for {path}:{payload}: {len(data)}/{expected}")
    return data


def load_source_observation(
    data_root: Path, record: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    with Image.open(io.BytesIO(_read_payload(data_root, record, "rgb"))) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with Image.open(io.BytesIO(_read_payload(data_root, record, "depth"))) as image:
        depth = np.asarray(image).astype(np.float32) * float(record["depth_unit_m"])
    full_mask = mask_from_payload(_read_payload(data_root, record, "mask"), int(record["gt_id"]))
    visible_mask = mask_from_payload(
        _read_payload(data_root, record, "mask_visib"), int(record["gt_id"])
    )
    expected = (int(record["height"]), int(record["width"]))
    for name, value in {
        "rgb": rgb,
        "depth": depth,
        "full_mask": full_mask,
        "visible_mask": visible_mask,
    }.items():
        if value.shape[:2] != expected:
            raise ValueError(f"{name} shape {value.shape} differs from {expected}")
    return {
        "rgb": rgb,
        "depth_m": depth,
        "full_mask": full_mask,
        "visible_mask": visible_mask,
    }


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def _caption(image: Image.Image, text: str) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + 34), "white")
    result.paste(image, (0, 34))
    ImageDraw.Draw(result).text((8, 10), text, fill="black")
    return result


def save_comparison(
    path: Path,
    *,
    source_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    source_full_mask: np.ndarray,
    source_visible_mask: np.ndarray,
    rendered_mask: np.ndarray,
    source_depth_m: np.ndarray,
    rendered_depth_m: np.ndarray,
    title: str,
) -> None:
    source = np.asarray(source_rgb, dtype=np.uint8)
    rendered = np.asarray(rendered_rgb, dtype=np.uint8)
    overlay = source.copy()
    overlay[_mask_boundary(source_full_mask)] = (255, 64, 64)
    overlay[_mask_boundary(source_visible_mask)] = (64, 128, 255)
    overlay[_mask_boundary(rendered_mask)] = (64, 255, 64)

    valid = rendered_mask & source_visible_mask & (source_depth_m > 0) & (rendered_depth_m > 0)
    error = np.zeros_like(source_depth_m, dtype=np.float32)
    error[valid] = np.abs(source_depth_m[valid] - rendered_depth_m[valid])
    normalized = np.clip(error / 0.02, 0, 1)
    error_rgb = np.zeros((*error.shape, 3), dtype=np.uint8)
    error_rgb[..., 0] = (normalized * 255).astype(np.uint8)
    error_rgb[..., 1] = ((1.0 - normalized) * valid * 255).astype(np.uint8)

    panels = [
        _caption(Image.fromarray(source), "Source RGB"),
        _caption(Image.fromarray(rendered), "Panda3D isolated render"),
        _caption(Image.fromarray(overlay), "Contours: full red / visible blue / render green"),
        _caption(Image.fromarray(error_rgb), "Visible depth error: green=0, red>=2 cm"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height + 28), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 28))
        x += panel.width
    ImageDraw.Draw(canvas).text((8, 7), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


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


def _close_renderer_buffers(renderer) -> None:
    """Release auxiliary offscreen buffers before interpreter teardown."""

    for cameras in renderer._cameras_pool.values():
        for camera in cameras:
            camera.node_path.node().setActive(0)
            camera.node_path.removeNode()
            camera.graphics_buffer.clearRenderTextures()
            renderer._app.graphicsEngine.removeWindow(camera.graphics_buffer)
    renderer._cameras_pool.clear()
    renderer._app.graphicsEngine.renderFrame()
    renderer._app.graphicsEngine.syncFrame()


def _renderer_driver_info(renderer) -> dict[str, str]:
    gsg = renderer._app.win.getGsg()
    return {
        "vendor": str(gsg.getDriverVendor()),
        "renderer": str(gsg.getDriverRenderer()),
        "version": str(gsg.getDriverVersion()),
    }


def run_validation(args) -> dict[str, Any]:
    data_root = args.data_root.expanduser().resolve()
    assets_root = args.assets_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = output_dir / "megapose_runtime"
    runtime_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MEGAPOSE_DATA_DIR", str(runtime_dir))

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
        raise ValueError("MegaPose Panda3D requires exactly one CUDA_VISIBLE_DEVICES entry")

    # Import only after MEGAPOSE_DATA_DIR and CUDA visibility are configured.
    import megapose
    from panda3d.core import ConfigVariableString

    # MegaPose's App requests pandagl, but this local ConfigVariable value has
    # higher precedence. The distributed wheel contains p3headlessgl, which
    # uses EGL and avoids GLX teardown failures in headless jobs.
    ConfigVariableString("load-display").setValue(args.display_backend)

    from megapose.datasets.gso_dataset import GoogleScannedObjectDataset
    from megapose.lib3d.transform import Transform
    from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
    from megapose.panda3d_renderer.types import (
        Panda3dCameraData,
        Panda3dLightData,
        Panda3dObjectData,
    )

    with (assets_root / "gso_models.json").open("r", encoding="utf-8") as stream:
        mapping = {int(row["obj_id"]): str(row["gso_id"]) for row in json.load(stream)}
    index_path = args.index_path or data_root / "pi3_index" / "megapose_gso.sqlite"
    split_path = args.split_path or data_root / "pi3_index" / "megapose_gso.splits.json"
    records = select_validation_records(
        index_path=index_path,
        split_path=split_path,
        samples_per_split=args.samples_per_split,
        seed=args.seed,
        minimum_visibility=args.minimum_visibility,
        minimum_visible_pixels=args.minimum_visible_pixels,
    )
    labels = {f"gso_{mapping[int(record['object_id'])]}" for record in records}
    object_dataset = GoogleScannedObjectDataset(
        assets_root / "google_scanned_objects", split="normalized"
    ).filter_objects(labels)
    if len(object_dataset) != len(labels):
        raise ValueError(f"Renderer object coverage {len(object_dataset)}/{len(labels)}")
    renderer = Panda3dSceneRenderer(object_dataset, preload_labels=labels, verbose=True)
    driver_info = _renderer_driver_info(renderer)
    print(
        "Panda3D driver: "
        f"{driver_info['vendor']} | {driver_info['renderer']} | "
        f"{driver_info['version']}"
    )
    lights = [Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))]

    sample_reports = []
    for number, record in enumerate(records):
        source = load_source_observation(data_root, record)
        height, width = int(record["height"]), int(record["width"])
        K = np.frombuffer(record["K_f32"], dtype="<f4").reshape(3, 3).copy()
        T_C_O = np.frombuffer(record["T_C_O_f32"], dtype="<f4").reshape(4, 4).copy()
        label = f"gso_{mapping[int(record['object_id'])]}"
        rendering = renderer.render_scene(
            [Panda3dObjectData(label=label, TWO=Transform(T_C_O))],
            [
                Panda3dCameraData(
                    K=K,
                    resolution=(height, width),
                    TWC=Transform(np.eye(4)),
                    z_near=0.01,
                    z_far=10.0,
                )
            ],
            lights,
            render_depth=True,
            # MegaPose commit f3b8e124 has an upstream local-variable typo in
            # render_binary_mask=True. Its intended mask is exactly depth > 0,
            # so derive it below without modifying the external checkout.
            render_binary_mask=False,
        )[0]
        if rendering.depth is None:
            raise RuntimeError("Panda3D renderer did not return depth")
        rendered_depth = np.asarray(rendering.depth).squeeze(-1)
        rendered_mask = rendered_depth > 0
        metrics = geometry_metrics(
            rendered_mask=rendered_mask,
            rendered_depth_m=rendered_depth,
            source_full_mask=source["full_mask"],
            source_visible_mask=source["visible_mask"],
            source_depth_m=source["depth_m"],
        )
        sample_name = (
            f"{number:02d}_{record['split']}_obj{int(record['object_id']):04d}_"
            f"scene{int(record['scene_id']):06d}_view{int(record['view_id']):06d}_"
            f"gt{int(record['gt_id']):02d}"
        )
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(source["rgb"]).save(sample_dir / "source_rgb.png")
        Image.fromarray(np.asarray(rendering.rgb, dtype=np.uint8)).save(
            sample_dir / "rendered_rgb.png"
        )
        Image.fromarray(rendered_mask.astype(np.uint8) * 255).save(
            sample_dir / "rendered_mask.png"
        )
        np.savez_compressed(
            sample_dir / "geometry.npz",
            rendered_depth_m=rendered_depth.astype(np.float32),
            rendered_mask=rendered_mask,
            source_depth_m=source["depth_m"].astype(np.float32),
            source_full_mask=source["full_mask"],
            source_visible_mask=source["visible_mask"],
            K=K,
            T_C_O=T_C_O,
        )
        save_comparison(
            sample_dir / "comparison.png",
            source_rgb=source["rgb"],
            rendered_rgb=rendering.rgb,
            source_full_mask=source["full_mask"],
            source_visible_mask=source["visible_mask"],
            rendered_mask=rendered_mask,
            source_depth_m=source["depth_m"],
            rendered_depth_m=rendered_depth,
            title=(
                f"{sample_name} | IoU={metrics['full_mask_iou']:.4f} | "
                f"depth median={1000 * metrics['depth_abs_median_m']:.2f} mm"
            ),
        )
        sample_reports.append(
            {
                "sample": sample_name,
                "split": record["split"],
                "object_id": int(record["object_id"]),
                "gso_id": mapping[int(record["object_id"])],
                "scene_id": int(record["scene_id"]),
                "view_id": int(record["view_id"]),
                "gt_id": int(record["gt_id"]),
                "source_visibility_fraction": float(record["visib_fract"]),
                **metrics,
            }
        )
        print(
            f"[{number + 1}/{len(records)}] {sample_name}: "
            f"IoU={metrics['full_mask_iou']:.4f}, "
            f"depth_median={1000 * metrics['depth_abs_median_m']:.2f} mm"
        )

    _close_renderer_buffers(renderer)

    mask_ious = np.asarray([sample["full_mask_iou"] for sample in sample_reports])
    depth_medians = np.asarray(
        [sample["depth_abs_median_m"] for sample in sample_reports], dtype=np.float64
    )
    failures = [
        sample["sample"]
        for sample in sample_reports
        if sample["full_mask_iou"] < args.minimum_mask_iou
        or not np.isfinite(sample["depth_abs_median_m"])
        or sample["depth_abs_median_m"] > args.maximum_depth_median_m
    ]
    report = {
        "format": REPORT_FORMAT,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "passed": not failures,
        "thresholds": {
            "minimum_mask_iou": args.minimum_mask_iou,
            "maximum_depth_median_m": args.maximum_depth_median_m,
        },
        "summary": {
            "sample_count": len(sample_reports),
            "mask_iou_min": float(mask_ious.min()),
            "mask_iou_median": float(np.median(mask_ious)),
            "depth_abs_median_median_m": float(np.nanmedian(depth_medians)),
            "depth_abs_median_max_m": float(np.nanmax(depth_medians)),
            "failed_samples": failures,
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "megapose_module": str(Path(megapose.__file__).resolve()),
            "megapose_commit": _source_commit(Path(megapose.__file__)),
            "panda3d_version": _package_version("panda3d"),
            "panda3d_gltf_version": _package_version("panda3d-gltf"),
            "display_backend": args.display_backend,
            "graphics_driver": driver_info,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "egl_visible_devices": os.environ.get("EGL_VISIBLE_DEVICES"),
        },
        "data_root": str(data_root),
        "assets_root": str(assets_root),
        "samples": sample_reports,
    }
    report_path = output_dir / "report.json"
    temporary = report_path.with_name(report_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, report_path)
    print(f"Wrote {report_path}; passed={report['passed']}")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate GSO mesh scale/pose/camera conventions by GT re-rendering."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--samples-per-split", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-visibility", type=float, default=0.8)
    parser.add_argument("--minimum-visible-pixels", type=int, default=2000)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.90)
    parser.add_argument("--maximum-depth-median-m", type=float, default=0.005)
    parser.add_argument(
        "--display-backend",
        choices=("p3headlessgl", "pandagl"),
        default="p3headlessgl",
        help="Panda3D display pipe; p3headlessgl uses EGL and exits cleanly headlessly.",
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args(argv)
    if args.samples_per_split <= 0:
        parser.error("--samples-per-split must be positive")
    if not 0 <= args.minimum_visibility <= 1:
        parser.error("--minimum-visibility must be in [0, 1]")
    if args.minimum_visible_pixels <= 0:
        parser.error("--minimum-visible-pixels must be positive")
    if not 0 <= args.minimum_mask_iou <= 1:
        parser.error("--minimum-mask-iou must be in [0, 1]")
    if args.maximum_depth_median_m <= 0:
        parser.error("--maximum-depth-median-m must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    report = run_validation(args)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
