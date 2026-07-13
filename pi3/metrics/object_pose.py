from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .base import BaseMetric
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

    def __init__(
        self,
        data_root: str,
        *,
        models_folder: str = "models_eval",
        unit_scale: float = 0.001,
        symmetric_ids: list[int] | tuple[int, ...] = (10, 11),
        solve_scale: bool = True,
        max_model_points: int = 20000,
        nn_chunk_size: int = 2048,
        device: str = "cpu",
    ):
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
        self.reset()

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

            for T_pred, T_gt in zip(pred_T_C_O_query, gt_T_C_O_query):
                add = add_error(T_pred, T_gt, points)
                adds = adds_error(
                    T_pred,
                    T_gt,
                    points,
                    device=self.device,
                    chunk_size=self.nn_chunk_size,
                )
                used = adds if int(obj_id) in self.symmetric_ids else add
                self.add_d.append(finite_float(add / diameter))
                self.adds_d.append(finite_float(adds / diameter))
                self.used_d.append(finite_float(used / diameter))
                self.rot_deg.append(float(rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3])))
                self.trans_m.append(float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])))
                self.add_d_by_obj[int(obj_id)].append(self.add_d[-1])
                self.adds_d_by_obj[int(obj_id)].append(self.adds_d[-1])
                self.used_d_by_obj[int(obj_id)].append(self.used_d[-1])
                self.rot_deg_by_obj[int(obj_id)].append(self.rot_deg[-1])
                self.trans_m_by_obj[int(obj_id)].append(self.trans_m[-1])
                self.num_predictions += 1

    def compute(self) -> dict[str, float]:
        output = {
            "query_count": float(self.num_predictions),
            "query_add_0_1d": recall_below(self.add_d, 0.1),
            "query_adds_0_1d": recall_below(self.adds_d, 0.1),
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
        if self.used_d_by_obj:
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
            for obj_id in sorted(self.used_d_by_obj):
                prefix = f"obj_{obj_id:06d}"
                output[f"{prefix}_query_count"] = float(len(self.used_d_by_obj[obj_id]))
                output[f"{prefix}_query_add_s_0_1d"] = recall_below(self.used_d_by_obj[obj_id], 0.1)
                output[f"{prefix}_query_used_median_d"] = safe_median(self.used_d_by_obj[obj_id])
                output[f"{prefix}_query_rot_median_deg"] = safe_median(self.rot_deg_by_obj[obj_id])
                output[f"{prefix}_query_trans_median_m"] = safe_median(self.trans_m_by_obj[obj_id])
        return output
