#!/usr/bin/env python3
"""Export per-sample matched LM-O inputs and trained-model outputs.

The script restores the same trainer/checkpoint used by validation, then calls
the repository's TensorBoard visualizer and metric plugins directly. Outputs
are ordinary PNG/JSON files, so inspecting more than TensorBoard's one random
validation example does not require changing the trainer.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
import torch

from datasets.base.utils import unified_collate_fn
from pi3.visualizations.reconstruction import ReferenceReconstructionVisualizer
from scripts.train_pi3 import configure_cuda_memory_limit_from_env
from utils.misc import move_to_device


DEFAULT_OUTPUT = Path(
    "/media/internal/nvme/dtrofimov/spott3r/outputs/"
    "lmgeo_geometry_matched_visual_gallery"
)


def stratified_sample_indices(
    samples: Sequence[Mapping[str, Any]], count: int
) -> list[int]:
    """Select deterministic, spread-out indices while balancing object IDs."""

    count = min(max(0, int(count)), len(samples))
    if count == 0:
        return []
    by_object: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        by_object.setdefault(int(sample["object_id"]), []).append(index)
    object_ids = sorted(by_object)
    quotas = {
        object_id: count // len(object_ids) + int(slot < count % len(object_ids))
        for slot, object_id in enumerate(object_ids)
    }
    selected_by_object: dict[int, list[int]] = {}
    for object_id in object_ids:
        values = by_object[object_id]
        quota = min(quotas[object_id], len(values))
        if quota == 0:
            selected_by_object[object_id] = []
            continue
        positions = np.linspace(0, len(values) - 1, quota + 2)[1:-1]
        positions = np.round(positions).astype(np.int64)
        selected_by_object[object_id] = [values[int(value)] for value in positions]

    # Interleave objects so a partial export is still diverse.
    output = []
    row = 0
    while len(output) < count:
        changed = False
        for object_id in object_ids:
            values = selected_by_object[object_id]
            if row < len(values):
                output.append(values[row])
                changed = True
        if not changed:
            break
        row += 1
    if len(output) != count or len(set(output)) != len(output):
        raise RuntimeError("Could not construct unique stratified sample indices")
    return output


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def compact_metrics(metrics: Mapping[str, float]) -> dict[str, float | None]:
    aliases = {
        "pose_error_d": "lmo_object_pose/query_used_median_d",
        "rotation_error_deg": "lmo_object_pose/query_rot_median_deg",
        "translation_error_m": "lmo_object_pose/query_trans_median_m",
        "camera_center_error_m": "camera/query_center_median_m",
        "reference_center_error_m": "camera/reference_center_median_m",
        "query_ray_error_deg": "ray_geometry/query/angular_median_deg",
        "reference_ray_error_deg": "ray_geometry/reference/angular_median_deg",
    }
    output: dict[str, float | None] = {}
    for name, key in aliases.items():
        value = float(metrics[key]) if key in metrics else float("nan")
        output[name] = value if math.isfinite(value) else None
    return output


def _fit_width(image: Image.Image, maximum_width: int) -> Image.Image:
    image = image.convert("RGB")
    if image.width <= maximum_width:
        return image
    height = max(1, round(image.height * maximum_width / image.width))
    return image.resize((maximum_width, height), Image.Resampling.LANCZOS)


def _title(image: Image.Image, text: str, *, height: int = 26) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), (22, 22, 22))
    canvas.paste(image.convert("RGB"), (0, height))
    ImageDraw.Draw(canvas).text((6, 5), str(text), fill=(245, 245, 245))
    return canvas


def _horizontal(images: Sequence[Image.Image], *, padding: int = 8) -> Image.Image:
    values = [image.convert("RGB") for image in images if image is not None]
    if not values:
        return Image.new("RGB", (1, 1), (0, 0, 0))
    canvas = Image.new(
        "RGB",
        (sum(image.width for image in values) + padding * (len(values) - 1),
         max(image.height for image in values)),
        (10, 10, 10),
    )
    left = 0
    for image in values:
        canvas.paste(image, (left, 0))
        left += image.width + padding
    return canvas


def compose_overview(
    images: Mapping[str, Image.Image],
    *,
    header_lines: Sequence[str],
    maximum_width: int = 1800,
) -> Image.Image:
    """Build one readable input/output sheet from standard visualizer cards."""

    maximum_width = max(640, int(maximum_width))
    input_row = _horizontal(
        [
            _title(images["input_reference_frames/grid"], "Five matched references"),
            _title(images["input_query_frames/grid"], "LM-O query"),
        ]
    )
    geometry_row = _horizontal(
        [
            _title(images["lmo_query_pose_overlay/queries"], "Query pose: green GT, red prediction"),
            _title(images["reference_reconstruction/orthographic"], "Aligned reference reconstruction"),
        ]
    )
    cards = [
        _fit_width(input_row, maximum_width),
        _fit_width(geometry_row, maximum_width),
        _fit_width(_title(images["depth_panel/query"], "Query RGB/depth prediction"), maximum_width),
        _fit_width(_title(images["depth_panel/reference"], "Representative reference RGB/depth prediction"), maximum_width),
    ]
    header_height = 18 + 20 * len(header_lines)
    header = Image.new("RGB", (maximum_width, header_height), (18, 18, 18))
    draw = ImageDraw.Draw(header)
    for row, line in enumerate(header_lines):
        draw.text((10, 8 + row * 20), str(line), fill=(245, 245, 245))
    padding = 10
    total_height = header.height + sum(card.height for card in cards) + padding * len(cards)
    canvas = Image.new("RGB", (maximum_width, total_height), (5, 5, 5))
    top = 0
    canvas.paste(header, (0, top))
    top += header.height + padding
    for card in cards:
        canvas.paste(card, ((maximum_width - card.width) // 2, top))
        top += card.height + padding
    return canvas


def _scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    elif isinstance(value, (list, tuple)) and value:
        value = value[0]
    return value


def _header_lines(
    batch: list[dict[str, Any]],
    sample_info: Mapping[str, Any],
    values: Mapping[str, float | None],
    loss: float,
) -> list[str]:
    query = next(view for view in batch if bool(_scalar(view["is_query"])))
    return [
        (
            f"object={int(_scalar(query['object_id']))}  scene={int(_scalar(query['scene_id']))}  "
            f"image={int(_scalar(query['im_id']))}  gt={int(_scalar(query['gt_id']))}  loss={loss:.4f}"
        ),
        (
            f"pose={values['pose_error_d']:.3f}d  rotation={values['rotation_error_deg']:.2f} deg  "
            f"translation={100.0 * values['translation_error_m']:.2f} cm  "
            f"camera-center={100.0 * values['camera_center_error_m']:.2f} cm"
        ),
        (
            f"coverage={float(sample_info['reference_union_coverage']):.3f}  "
            f"nearest-reference={float(sample_info['positive_angle_degrees']):.2f} deg  "
            f"target-normalized-focal={float(sample_info['target_normalized_focal']):.4f}"
        ),
    ]


def _export_settings(cfg: DictConfig) -> dict[str, Any]:
    section = cfg.get("export", {})
    return {
        "output": Path(section.get("output", DEFAULT_OUTPUT)).expanduser().resolve(),
        "samples": int(section.get("samples", 20)),
        "seed": int(section.get("seed", 20260818)),
        "maximum_width": int(section.get("maximum_width", 1800)),
        "overwrite": bool(section.get("overwrite", False)),
    }


def evaluate_prediction_for_export(trainer, prediction, batch):
    """Render/measure raw predictions before the loss normalizes them in place.

    ``Pi3Loss.normalize_pred`` replaces ``local_points`` and camera translations
    with their sample-normalized forms. The trainer intentionally runs visual
    and metric plugins before that mutation. Keep the offline exporter in the
    same order; otherwise ``points`` remains at its raw forward scale while the
    depth-derived Sim(3) is estimated from normalized ``local_points``.
    """

    rendered = trainer.visual_manager.render(prediction, batch, mode="val")
    metrics = trainer.metric_manager.compute_on_batch(
        prediction, batch, mode="val"
    )
    loss_output = trainer.calculate_loss(prediction, batch, mode="test")
    return rendered, metrics, loss_output


@hydra.main(version_base="1.2", config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    configure_cuda_memory_limit_from_env()
    settings = _export_settings(cfg)
    output = settings["output"]
    if output.exists() and any(output.iterdir()) and not settings["overwrite"]:
        raise FileExistsError(
            f"Visual output directory is not empty: {output}. "
            "Choose another +export.output or set +export.overwrite=true."
        )
    output.mkdir(parents=True, exist_ok=True)

    trainer_class = hydra.utils.get_class(str(cfg.trainer))
    trainer = trainer_class(cfg)
    loader = trainer.val_loaders[trainer.primary_val_name]
    dataset = loader.dataset
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError("Matched LM-O visual export requires a dataset sample manifest")
    indices = stratified_sample_indices(samples, settings["samples"])
    if not any(
        visualizer.name == "reference_reconstruction"
        for visualizer in trainer.visual_manager.visualizers
    ):
        scale_estimation = str(
            OmegaConf.select(
                cfg,
                "metrics.items.object_pose.scale_estimation",
                default="camera_centers",
            )
        )
        trainer.visual_manager.visualizers.append(
            ReferenceReconstructionVisualizer(
                solve_scale=True,
                scale_estimation=scale_estimation,
                size=320,
            )
        )
        trainer.visual_manager.enabled = True

    trainer.model.eval()
    started = time.monotonic()
    reports = []
    try:
        with torch.no_grad():
            for ordinal, dataset_index in enumerate(indices):
                sample_seed = settings["seed"] + ordinal
                views = dataset[(dataset_index, 0, 6, sample_seed)]
                sample_info = dict(dataset.this_views_info)
                batch = unified_collate_fn([views])
                batch = move_to_device(batch, trainer.accelerator.device)
                with trainer.accelerator.autocast():
                    prediction = trainer.forward_batch(batch, mode="test")
                rendered, metrics, loss_output = evaluate_prediction_for_export(
                    trainer, prediction, batch
                )
                values = compact_metrics(metrics)
                required = {
                    "input_reference_frames/grid",
                    "input_query_frames/grid",
                    "lmo_query_pose_overlay/queries",
                    "depth_panel/reference",
                    "depth_panel/query",
                    "reference_reconstruction/orthographic",
                }
                missing = sorted(required.difference(rendered))
                if missing:
                    raise RuntimeError(f"Missing visualizer outputs: {missing}")

                query = next(view for view in batch if bool(_scalar(view["is_query"])))
                name = (
                    f"example_{ordinal:02d}_obj_{int(_scalar(query['object_id'])):06d}_"
                    f"scene_{int(_scalar(query['scene_id'])):06d}_"
                    f"im_{int(_scalar(query['im_id'])):06d}_"
                    f"gt_{int(_scalar(query['gt_id'])):02d}"
                )
                example_dir = output / name
                example_dir.mkdir(parents=True, exist_ok=settings["overwrite"])
                visual_paths = {}
                for key, image in rendered.items():
                    filename = _safe_name(key.replace("/", "__")) + ".png"
                    image.save(example_dir / filename)
                    visual_paths[key] = filename
                loss = float(loss_output.loss.detach().float().cpu())
                overview = compose_overview(
                    rendered,
                    header_lines=_header_lines(batch, sample_info, values, loss),
                    maximum_width=settings["maximum_width"],
                )
                overview.save(example_dir / "overview.png")
                report = {
                    "ordinal": ordinal,
                    "dataset_index": int(dataset_index),
                    "sample_seed": int(sample_seed),
                    "folder": name,
                    "object_id": int(_scalar(query["object_id"])),
                    "scene_id": int(_scalar(query["scene_id"])),
                    "im_id": int(_scalar(query["im_id"])),
                    "gt_id": int(_scalar(query["gt_id"])),
                    "source": str(_scalar(query["source"])),
                    "loss": loss,
                    "metrics": _json_value(metrics),
                    "compact_metrics": _json_value(values),
                    "sample_info": _json_value(sample_info),
                    "visuals": visual_paths,
                }
                (example_dir / "metadata.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                reports.append(report)
                print(
                    f"[{ordinal + 1:02d}/{len(indices):02d}] {name}: "
                    f"pose={values['pose_error_d']:.3f}d "
                    f"rot={values['rotation_error_deg']:.2f}deg "
                    f"trans={100.0 * values['translation_error_m']:.2f}cm"
                )
                del prediction, loss_output, batch
    finally:
        if hasattr(dataset, "close"):
            dataset.close()

    summary = {
        "format": "pi3_lmgeo_geometry_visual_gallery_v1",
        "checkpoint": str(cfg.train.resume),
        "train_config": "train_megapose_gso_geometry_pi3_lmgeo_eval_rtxpro6000_70gb_336x252",
        "data_config": "lmgeo_new_val_geometry_render_n5_k1_masked",
        "sample_count": len(reports),
        "dataset_size": len(samples),
        "selection": "object-stratified deterministic spread",
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(trainer.accelerator.device))
            if torch.cuda.is_available()
            else 0
        ),
        "examples": reports,
    }
    (output / "gallery_report.json").write_text(
        json.dumps(_json_value(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trainer.accelerator.end_training()
    print(json.dumps({"output": str(output), "examples": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
