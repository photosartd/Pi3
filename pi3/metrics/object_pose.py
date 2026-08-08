from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import BaseMetric
from datasets.base.observation import batch_matches_metadata
from .failure_modes import aggregate_failure_modes, append_prediction_rows
from .utils import (
    BopModelCache,
    add_error,
    adds_error,
    batch_object_ids,
    estimate_world_to_object_sim3,
    extract_prediction,
    finite_float,
    recall_below,
    rotation_error_deg,
    safe_mean,
    safe_median,
    safe_percentile,
    stack_view_tensor,
    view_bool_mask,
)


class ObjectPoseMetric(BaseMetric):
    """Query-frame BOP-style object-pose metrics.

    The metric estimates a world-to-object Sim(3) from reference/keyframe camera
    predictions and GT reference poses. It then converts query camera
    predictions into object-to-camera poses and evaluates ADD/ADD-S only on the
    query frames.
    """

    name = "object_pose"
    required_capabilities = frozenset(
        {"key_query", "object_pose", "object_model"}
    )

    def __init__(
        self,
        data_root: str,
        *,
        metric_name: str | None = None,
        object_model_namespaces: list[str] | tuple[str, ...] | None = None,
        report_per_object: bool = True,
        compute_adds_for_all: bool = True,
        models_folder: str = "models_eval",
        unit_scale: float = 0.001,
        symmetric_ids: list[int] | tuple[int, ...] = (10, 11),
        solve_scale: bool = True,
        max_model_points: int = 20000,
        nn_chunk_size: int = 2048,
        device: str = "cpu",
        raw_predictions_path: str | None = None,
        run_id: str | None = None,
        masked: bool | None = None,
        keyframe_seed: int | None = None,
        checkpoint_step: int | None = None,
        split: str | None = None,
        covariates_path: str | None = None,
        analysis_dir: str | None = None,
        coarse_analysis: bool = False,
        hard_object_ids: list[int] | tuple[int, ...] | None = None,
    ):
        self.name = str(metric_name or type(self).name)
        self.object_model_namespaces = (
            None
            if object_model_namespaces is None
            else tuple(str(value) for value in object_model_namespaces)
        )
        self.report_per_object = bool(report_per_object)
        self.compute_adds_for_all = bool(compute_adds_for_all)
        self.model_cache = BopModelCache(
            data_root,
            models_folder=models_folder,
            unit_scale=unit_scale,
            max_model_points=max_model_points,
        )
        self.symmetric_ids = {int(obj_id) for obj_id in symmetric_ids}
        self.solve_scale = bool(solve_scale)
        self.nn_chunk_size = int(nn_chunk_size)
        self.device = str(device)
        self.raw_predictions_path = raw_predictions_path
        self.run_id = run_id
        self.masked = None if masked is None else bool(masked)
        self.keyframe_seed = keyframe_seed
        self.checkpoint_step = checkpoint_step
        self.split = split
        self.covariates_path = covariates_path
        self.analysis_dir = analysis_dir
        self.coarse_analysis = bool(coarse_analysis)
        self.hard_object_ids = [int(obj_id) for obj_id in (hard_object_ids or [])]
        self.context: dict[str, Any] = {}
        self.reset()

    def supports_batch(self, batch: list[dict[str, Any]]) -> bool:
        return batch_matches_metadata(
            batch,
            "object_model_namespace",
            self.object_model_namespaces,
        )

    def reset(self) -> None:
        self.add_d: list[float] = []
        self.adds_d: list[float] = []
        self.used_d: list[float] = []
        self.rot_deg: list[float] = []
        self.trans_m: list[float] = []
        self.scales: list[float] = []
        self.used_d_by_obj: dict[int, list[float]] = defaultdict(list)
        self.add_d_by_obj: dict[int, list[float]] = defaultdict(list)
        self.adds_d_by_obj: dict[int, list[float]] = defaultdict(list)
        self.rot_deg_by_obj: dict[int, list[float]] = defaultdict(list)
        self.trans_m_by_obj: dict[int, list[float]] = defaultdict(list)
        self.num_underconstrained = 0
        self.num_predictions = 0
        self.raw_prediction_rows: list[dict[str, Any]] = []

    def set_context(self, **context: Any) -> None:
        self.context.update({key: value for key, value in context.items() if value is not None})

    def _context_value(self, name: str, default: Any = None) -> Any:
        if name in self.context:
            return self.context[name]
        return getattr(self, name, default)

    def _resolve_path(self, value: str | None) -> Path | None:
        if not value:
            return None
        format_context = {
            "run_id": self._context_value("run_id", self.run_id),
            "split": self._context_value("split", self.split),
            "val_name": self._context_value("val_name", self.split),
            "checkpoint_step": self._context_value("checkpoint_step", self.checkpoint_step),
            "global_step": self._context_value("global_step", self.checkpoint_step),
            "output_dir": self._context_value("output_dir", None),
        }
        try:
            resolved = str(value).format(**format_context)
        except KeyError:
            resolved = str(value)
        return Path(resolved)

    def _append_raw_row(
        self,
        *,
        mode: str,
        batch_idx: int,
        view_idx: int,
        obj_id: int,
        add: float,
        adds: float,
        used: float,
        diameter: float,
        scene_ids: np.ndarray | None,
        im_ids: np.ndarray | None,
        gt_ids: np.ndarray | None,
        n_keyframes: int,
        n_queries: int,
    ) -> None:
        if not self.raw_predictions_path:
            return
        split = self._context_value("split", None) or self._context_value("val_name", None) or mode
        metric_type = "ADD-S" if int(obj_id) in self.symmetric_ids else "ADD"
        self.raw_prediction_rows.append(
            {
                "scene_id": int(scene_ids[batch_idx, view_idx]) if scene_ids is not None else -1,
                "im_id": int(im_ids[batch_idx, view_idx]) if im_ids is not None else -1,
                "gt_id": int(gt_ids[batch_idx, view_idx]) if gt_ids is not None else -1,
                "obj_id": int(obj_id),
                "add_err": finite_float(used),
                "add_err_norm": finite_float(used / diameter),
                "metric_type": metric_type,
                "n_keyframes": int(n_keyframes),
                "n_queries": int(n_queries),
                "keyframe_seed": self._context_value("keyframe_seed", self.keyframe_seed),
                "checkpoint_step": self._context_value("checkpoint_step", self.checkpoint_step),
                "split": split,
                "val_name": self._context_value("val_name", split),
                "masked": self._context_value("masked", self.masked),
                "run_id": self._context_value("run_id", self.run_id),
            }
        )

    @torch.no_grad()
    def update(
        self,
        prediction: Any,
        batch: list[dict[str, Any]],
        loss_output: Any | None = None,
        *,
        mode: str = "train",
    ) -> None:
        pred = extract_prediction(prediction)
        pred_T_W_C = pred["camera_poses"].detach().float().cpu().numpy()
        gt_T_C_O = stack_view_tensor(batch, "T_C_O").astype(np.float64)
        ref_mask = view_bool_mask(batch, "is_reference")
        query_mask = view_bool_mask(batch, "is_query")
        obj_ids = batch_object_ids(batch)
        scene_ids = stack_view_tensor(batch, "scene_id").astype(np.int64) if "scene_id" in batch[0] else None
        im_ids = stack_view_tensor(batch, "im_id").astype(np.int64) if "im_id" in batch[0] else None
        gt_ids = stack_view_tensor(batch, "gt_id").astype(np.int64) if "gt_id" in batch[0] else None

        for batch_idx, obj_id in enumerate(obj_ids):
            refs = ref_mask[batch_idx]
            queries = query_mask[batch_idx]
            if refs.sum() == 0 or queries.sum() == 0:
                continue

            alignment = estimate_world_to_object_sim3(
                pred_T_W_C[batch_idx, refs],
                gt_T_C_O[batch_idx, refs],
                solve_scale=self.solve_scale,
            )
            self.scales.append(float(alignment.scale))
            self.num_underconstrained += int(alignment.underconstrained_scale)

            pred_T_C_O_query = alignment.object_to_camera_pose(pred_T_W_C[batch_idx, queries])
            gt_T_C_O_query = gt_T_C_O[batch_idx, queries]
            points = self.model_cache.points(int(obj_id))
            diameter = self.model_cache.diameter(int(obj_id))
            query_indices = np.flatnonzero(queries)
            n_keyframes = int(refs.sum())
            n_queries = int(queries.sum())

            for view_idx, T_pred, T_gt in zip(query_indices, pred_T_C_O_query, gt_T_C_O_query):
                add = add_error(T_pred, T_gt, points)
                is_symmetric = int(obj_id) in self.symmetric_ids
                if is_symmetric or self.compute_adds_for_all:
                    adds = adds_error(
                        T_pred,
                        T_gt,
                        points,
                        device=self.device,
                        chunk_size=self.nn_chunk_size,
                    )
                else:
                    adds = float("nan")
                used = adds if is_symmetric else add
                self.add_d.append(finite_float(add / diameter))
                if np.isfinite(adds):
                    self.adds_d.append(finite_float(adds / diameter))
                self.used_d.append(finite_float(used / diameter))
                self.rot_deg.append(float(rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3])))
                self.trans_m.append(float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])))
                self.add_d_by_obj[int(obj_id)].append(self.add_d[-1])
                if np.isfinite(adds):
                    self.adds_d_by_obj[int(obj_id)].append(
                        finite_float(adds / diameter)
                    )
                self.used_d_by_obj[int(obj_id)].append(self.used_d[-1])
                self.rot_deg_by_obj[int(obj_id)].append(self.rot_deg[-1])
                self.trans_m_by_obj[int(obj_id)].append(self.trans_m[-1])
                self.num_predictions += 1
                self._append_raw_row(
                    mode=mode,
                    batch_idx=batch_idx,
                    view_idx=int(view_idx),
                    obj_id=int(obj_id),
                    add=add,
                    adds=adds,
                    used=used,
                    diameter=diameter,
                    scene_ids=scene_ids,
                    im_ids=im_ids,
                    gt_ids=gt_ids,
                    n_keyframes=n_keyframes,
                    n_queries=n_queries,
                )

    def compute(self) -> dict[str, float]:
        output = {
            "query_count": float(self.num_predictions),
            "query_add_0_1d": recall_below(self.add_d, 0.1),
            "query_add_s_0_1d": recall_below(self.used_d, 0.1),
            "query_used_0_5d": recall_below(self.used_d, 0.5),
            "query_used_1d": recall_below(self.used_d, 1.0),
            "query_used_2d": recall_below(self.used_d, 2.0),
            "query_used_mean_d": safe_mean(self.used_d),
            "query_used_median_d": safe_median(self.used_d),
            "query_used_p90_d": safe_percentile(self.used_d, 90),
            "query_rot_mean_deg": safe_mean(self.rot_deg),
            "query_rot_median_deg": safe_median(self.rot_deg),
            "query_trans_mean_m": safe_mean(self.trans_m),
            "query_trans_median_m": safe_median(self.trans_m),
            "alignment_scale_mean": safe_mean(self.scales),
            "alignment_scale_median": safe_median(self.scales),
            "alignment_underconstrained": float(self.num_underconstrained),
        }
        if self.adds_d:
            output["query_adds_0_1d"] = recall_below(self.adds_d, 0.1)
        if self.used_d_by_obj:
            output["query_add_0_1d_macro_obj"] = safe_mean(
                [recall_below(values, 0.1) for values in self.add_d_by_obj.values()]
            )
            if self.adds_d_by_obj:
                output["query_adds_0_1d_macro_obj"] = safe_mean(
                    [recall_below(values, 0.1) for values in self.adds_d_by_obj.values()]
                )
            output["query_add_s_0_1d_macro_obj"] = safe_mean(
                [recall_below(values, 0.1) for values in self.used_d_by_obj.values()]
            )
            output["query_used_median_d_macro_obj"] = safe_mean(
                [safe_median(values) for values in self.used_d_by_obj.values()]
            )
            output["query_rot_median_deg_macro_obj"] = safe_mean(
                [safe_median(values) for values in self.rot_deg_by_obj.values()]
            )
            output["query_trans_median_m_macro_obj"] = safe_mean(
                [safe_median(values) for values in self.trans_m_by_obj.values()]
            )
            if getattr(self, "report_per_object", True):
                for obj_id in sorted(self.used_d_by_obj):
                    prefix = f"obj_{obj_id:06d}"
                    output[f"{prefix}_query_count"] = float(len(self.used_d_by_obj[obj_id]))
                    output[f"{prefix}_query_add_0_1d"] = recall_below(self.add_d_by_obj[obj_id], 0.1)
                    if obj_id in self.adds_d_by_obj:
                        output[f"{prefix}_query_adds_0_1d"] = recall_below(self.adds_d_by_obj[obj_id], 0.1)
                    output[f"{prefix}_query_add_s_0_1d"] = recall_below(self.used_d_by_obj[obj_id], 0.1)
                    output[f"{prefix}_query_used_median_d"] = safe_median(self.used_d_by_obj[obj_id])
                    output[f"{prefix}_query_rot_median_deg"] = safe_median(self.rot_deg_by_obj[obj_id])
                    output[f"{prefix}_query_trans_median_m"] = safe_median(self.trans_m_by_obj[obj_id])
        return output

    def flush_artifacts(self, accelerator: Any | None = None) -> dict[str, float]:
        if not self.raw_predictions_path:
            return {}

        rows = list(self.raw_prediction_rows)
        self.raw_prediction_rows = []
        if accelerator is not None and getattr(accelerator, "num_processes", 1) > 1:
            from accelerate.utils import gather_object

            gathered = gather_object(rows)
            if gathered and all(isinstance(item, dict) for item in gathered):
                rows = list(gathered)
            else:
                rows = [
                    row
                    for rank_rows in gathered
                    if isinstance(rank_rows, list)
                    for row in rank_rows
                ]

        is_main = True if accelerator is None else bool(accelerator.is_main_process)
        if not is_main:
            return {}

        predictions_path = self._resolve_path(self.raw_predictions_path)
        if predictions_path is None:
            return {}
        append_prediction_rows(predictions_path, rows)

        if not (self.coarse_analysis and self.covariates_path and self.analysis_dir):
            return {}

        covariates_path = self._resolve_path(self.covariates_path)
        analysis_dir = self._resolve_path(self.analysis_dir)
        if covariates_path is None or analysis_dir is None:
            return {}
        split = self._context_value("split", None)
        val_name = self._context_value("val_name", self._context_value("split", "val"))
        checkpoint_step = self._context_value("checkpoint_step", self.checkpoint_step)
        out_dir = analysis_dir / str(val_name) / f"step_{int(checkpoint_step):08d}" if checkpoint_step is not None else analysis_dir / str(val_name)
        return aggregate_failure_modes(
            predictions_path,
            covariates_path,
            out_dir,
            coarse=True,
            plot=False,
            checkpoint_step=checkpoint_step,
            split=split,
            val_name=val_name,
            hard_object_ids=self.hard_object_ids,
        )
