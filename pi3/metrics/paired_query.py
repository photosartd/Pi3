from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pi3.models.ray_conditioning import pixel_center_grid

from .base import BaseMetric
from .utils import (
    BopModelCache,
    add_error,
    adds_error,
    batch_object_ids,
    estimate_world_to_object_sim3,
    extract_prediction,
    invert_se3,
    recall_below,
    rotation_error_deg,
    safe_mean,
    safe_median,
    stack_view_tensor,
    view_bool_mask,
)


class PairedQueryConsistencyMetric(BaseMetric):
    """Correctness and equivariance metrics for crop/original query pairs.

    One physical query contributes a recentered virtual-camera view and an
    original-camera context view. Both predictions are canonicalized to the
    original camera frame, while dense diagnostics use only the known
    preprocessing rotation/homography and do not require query-pose GT.
    """

    name = "paired_query"
    required_capabilities = frozenset({"paired_query"})

    _POSE_ROLES = ("query_crop_canonical", "query_original")

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
        max_points_per_pair: int = 4096,
        eps: float = 1e-8,
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
        self.max_points_per_pair = int(max_points_per_pair)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.pose_values = {
            role: defaultdict(list)
            for role in self._POSE_ROLES
        }
        self.pair_pose = defaultdict(list)
        self.pair_dense = defaultdict(list)
        self.pair_world = defaultdict(list)
        self.pair_count = 0

    @staticmethod
    def _stack_torch_views(
        batch: list[dict[str, Any]],
        key: str,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        values = []
        for view in batch:
            value = view[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.detach().to(device=device, non_blocking=True)
            if dtype is not None:
                value = value.to(dtype=dtype)
            values.append(value)
        return torch.stack(values, dim=1)

    @staticmethod
    def _collated_item(
        batch: list[dict[str, Any]],
        view_idx: int,
        key: str,
        batch_idx: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        value = batch[int(view_idx)][key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value[int(batch_idx)].detach()
        if device is not None:
            value = value.to(device=device, non_blocking=True)
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value

    def _subsample_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if self.max_points_per_pair <= 0 or len(indices) <= self.max_points_per_pair:
            return indices
        positions = torch.linspace(
            0,
            len(indices) - 1,
            self.max_points_per_pair,
            device=indices.device,
        ).round().long()
        return indices[positions]

    @staticmethod
    def _append_summary(store, prefix: str, values: torch.Tensor) -> None:
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return
        store[f"{prefix}_mean"].append(float(values.mean()))
        store[f"{prefix}_median"].append(float(values.median()))

    def _accumulate_pose(
        self,
        role: str,
        pred_T_C_O: np.ndarray,
        gt_T_C_O: np.ndarray,
        *,
        obj_id: int,
        points: np.ndarray,
        diameter: float,
    ) -> None:
        add = add_error(pred_T_C_O, gt_T_C_O, points)
        if int(obj_id) in self.symmetric_ids:
            used = adds_error(
                pred_T_C_O,
                gt_T_C_O,
                points,
                device=self.device,
                chunk_size=self.nn_chunk_size,
            )
        else:
            used = add
        values = self.pose_values[role]
        values["add_d"].append(float(add / diameter))
        values["add_or_adds_d"].append(float(used / diameter))
        values["rotation_deg"].append(
            float(rotation_error_deg(pred_T_C_O[:3, :3], gt_T_C_O[:3, :3]))
        )
        values["translation_m"].append(
            float(np.linalg.norm(pred_T_C_O[:3, 3] - gt_T_C_O[:3, 3]))
        )

    def _accumulate_dense_pair(
        self,
        *,
        crop_points: torch.Tensor,
        original_points: torch.Tensor,
        crop_valid: torch.Tensor,
        original_valid: torch.Tensor,
        crop_intrinsics: torch.Tensor,
        H_crop_from_original: torch.Tensor,
        R_old_to_new: torch.Tensor,
        pred_T_W_C_crop: torch.Tensor,
        pred_T_W_C_original: torch.Tensor,
    ) -> None:
        height, width, _ = crop_points.shape
        pixels_crop = pixel_center_grid(
            height,
            width,
            device=crop_points.device,
            dtype=torch.float32,
        )
        crop_xy = pixels_crop[..., :2].reshape(-1, 2)
        crop_h = pixels_crop.reshape(-1, 3)

        H_original_from_crop = torch.linalg.inv(H_crop_from_original.float())
        original_h = torch.einsum("ij,nj->ni", H_original_from_crop, crop_h)
        original_xy = original_h[:, :2] / original_h[:, 2:3].clamp_min(self.eps)

        original_x = original_xy[:, 0]
        original_y = original_xy[:, 1]
        in_bounds = (
            (original_x >= 0.5)
            & (original_x <= width - 0.5)
            & (original_y >= 0.5)
            & (original_y <= height - 0.5)
        )
        original_ix = torch.floor(original_x).long().clamp(0, width - 1)
        original_iy = torch.floor(original_y).long().clamp(0, height - 1)
        paired_valid = (
            crop_valid.reshape(-1).bool()
            & in_bounds
            & original_valid[original_iy, original_ix].bool()
        )
        candidate_indices = torch.nonzero(paired_valid, as_tuple=False).flatten()
        valid_crop_count = int(crop_valid.sum())
        self.pair_dense["overlap_fraction"].append(
            float(len(candidate_indices)) / float(max(valid_crop_count, 1))
        )
        candidate_indices = self._subsample_indices(candidate_indices)
        if candidate_indices.numel() == 0:
            return

        sampled_original_xy = original_xy[candidate_indices]
        sample_grid = torch.stack(
            (
                2.0 * sampled_original_xy[:, 0] / float(width) - 1.0,
                2.0 * sampled_original_xy[:, 1] / float(height) - 1.0,
            ),
            dim=-1,
        ).reshape(1, -1, 1, 2)
        sampled_original_points = F.grid_sample(
            original_points.permute(2, 0, 1)[None].float(),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0, :, :, 0].transpose(0, 1)
        sampled_crop_points = crop_points.reshape(-1, 3)[candidate_indices].float()
        sampled_crop_xy = crop_xy[candidate_indices]

        finite = (
            torch.isfinite(sampled_original_points).all(dim=-1)
            & torch.isfinite(sampled_crop_points).all(dim=-1)
            & (sampled_original_points[:, 2] > self.eps)
            & (sampled_crop_points[:, 2] > self.eps)
        )
        sampled_original_points = sampled_original_points[finite]
        sampled_crop_points = sampled_crop_points[finite]
        sampled_crop_xy = sampled_crop_xy[finite]
        if sampled_original_points.numel() == 0:
            return

        expected_crop_points = torch.einsum(
            "ij,nj->ni",
            R_old_to_new.float(),
            sampled_original_points,
        )
        expected_norm = expected_crop_points.norm(dim=-1)
        crop_norm = sampled_crop_points.norm(dim=-1)
        relative_3d = (
            (sampled_crop_points - expected_crop_points).norm(dim=-1)
            / (0.5 * (expected_norm + crop_norm)).clamp_min(self.eps)
        )
        cross_norm = torch.linalg.cross(
            sampled_crop_points,
            expected_crop_points,
            dim=-1,
        ).norm(dim=-1)
        dot = (sampled_crop_points * expected_crop_points).sum(dim=-1)
        angular_deg = torch.rad2deg(torch.atan2(cross_norm, dot))
        log_depth_ratio = (
            sampled_crop_points[:, 2].clamp_min(self.eps).log()
            - expected_crop_points[:, 2].clamp_min(self.eps).log()
        ).abs()
        scale_ratio = crop_norm / expected_norm.clamp_min(self.eps)

        self._append_summary(self.pair_dense, "angular_deg", angular_deg)
        self._append_summary(self.pair_dense, "relative_3d", relative_3d)
        self._append_summary(self.pair_dense, "abs_log_depth_ratio", log_depth_ratio)
        self._append_summary(self.pair_dense, "scale_ratio", scale_ratio)
        self.pair_dense["points"].append(float(len(sampled_crop_points)))

        A_pred = torch.linalg.inv(pred_T_W_C_crop.float()) @ pred_T_W_C_original.float()
        points_crop_from_pred = torch.einsum(
            "ij,nj->ni",
            A_pred[:3, :3],
            sampled_original_points,
        ) + A_pred[:3, 3]
        projected = torch.einsum(
            "ij,nj->ni",
            crop_intrinsics.float(),
            points_crop_from_pred,
        )
        projection_valid = projected[:, 2] > self.eps
        projected_xy = projected[:, :2] / projected[:, 2:3].clamp_min(self.eps)
        reprojection = (projected_xy - sampled_crop_xy).norm(dim=-1)
        self._append_summary(
            self.pair_pose,
            "reprojection_px",
            reprojection[projection_valid],
        )

        world_original = torch.einsum(
            "ij,nj->ni",
            pred_T_W_C_original[:3, :3].float(),
            sampled_original_points,
        ) + pred_T_W_C_original[:3, 3].float()
        world_crop = torch.einsum(
            "ij,nj->ni",
            pred_T_W_C_crop[:3, :3].float(),
            sampled_crop_points,
        ) + pred_T_W_C_crop[:3, 3].float()
        world_relative = (
            (world_crop - world_original).norm(dim=-1)
            / (0.5 * (crop_norm + expected_norm)).clamp_min(self.eps)
        )
        self._append_summary(self.pair_world, "relative_3d", world_relative)

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
        pred_T_W_C_torch = pred["camera_poses"].detach().float()
        pred_points = pred["local_points"].detach().float()
        device = pred_points.device
        pred_T_W_C = pred_T_W_C_torch.cpu().numpy()

        ref_mask = view_bool_mask(batch, "is_reference")
        crop_mask = view_bool_mask(batch, "is_cropped_query")
        original_mask = view_bool_mask(batch, "is_original_query")
        pair_indices = stack_view_tensor(batch, "query_pair_index").astype(np.int64)
        gt_T_C_O = stack_view_tensor(batch, "T_C_O").astype(np.float64)
        obj_ids = batch_object_ids(batch)
        intrinsics = self._stack_torch_views(
            batch,
            "camera_intrinsics",
            device=device,
            dtype=torch.float32,
        )
        valid_masks = self._stack_torch_views(
            batch,
            "valid_mask",
            device=device,
        ).bool()

        for batch_idx, obj_id in enumerate(obj_ids):
            refs = ref_mask[batch_idx]
            if refs.sum() == 0:
                continue
            alignment = estimate_world_to_object_sim3(
                pred_T_W_C[batch_idx, refs],
                gt_T_C_O[batch_idx, refs],
                solve_scale=self.solve_scale,
            )
            pred_T_O_C = alignment.camera_to_object_pose(pred_T_W_C[batch_idx])
            pred_T_C_O = invert_se3(pred_T_O_C)
            points = self.model_cache.points(int(obj_id))
            diameter = self.model_cache.diameter(int(obj_id))

            crop_by_pair = {
                int(pair_indices[batch_idx, idx]): int(idx)
                for idx in np.flatnonzero(crop_mask[batch_idx])
            }
            original_by_pair = {
                int(pair_indices[batch_idx, idx]): int(idx)
                for idx in np.flatnonzero(original_mask[batch_idx])
            }
            for pair_idx in sorted(set(crop_by_pair) & set(original_by_pair)):
                crop_idx = crop_by_pair[pair_idx]
                original_idx = original_by_pair[pair_idx]
                R_old_to_new_t = self._collated_item(
                    batch,
                    crop_idx,
                    "query_recenter_R_old_to_new",
                    batch_idx,
                    device=device,
                    dtype=torch.float32,
                )
                H_crop_from_original = self._collated_item(
                    batch,
                    crop_idx,
                    "query_crop_from_original_homography",
                    batch_idx,
                    device=device,
                    dtype=torch.float32,
                )
                A_old_to_new = np.eye(4, dtype=np.float64)
                A_old_to_new[:3, :3] = R_old_to_new_t.cpu().numpy()

                pred_crop_canonical = (
                    invert_se3(A_old_to_new) @ pred_T_C_O[crop_idx]
                )
                pred_original = pred_T_C_O[original_idx]
                gt_original = gt_T_C_O[batch_idx, original_idx]
                self._accumulate_pose(
                    "query_crop_canonical",
                    pred_crop_canonical,
                    gt_original,
                    obj_id=int(obj_id),
                    points=points,
                    diameter=diameter,
                )
                self._accumulate_pose(
                    "query_original",
                    pred_original,
                    gt_original,
                    obj_id=int(obj_id),
                    points=points,
                    diameter=diameter,
                )

                A_pred = (
                    invert_se3(pred_T_W_C[batch_idx, crop_idx])
                    @ pred_T_W_C[batch_idx, original_idx]
                )
                self.pair_pose["relative_rotation_error_deg"].append(
                    float(rotation_error_deg(A_pred[:3, :3], A_old_to_new[:3, :3]))
                )
                relative_translation_m = (
                    abs(float(alignment.scale))
                    * float(np.linalg.norm(A_pred[:3, 3]))
                )
                self.pair_pose["relative_translation_m"].append(
                    relative_translation_m
                )
                self.pair_pose["relative_translation_d"].append(
                    relative_translation_m / diameter
                )
                center_disagreement = float(
                    np.linalg.norm(
                        pred_T_O_C[crop_idx, :3, 3]
                        - pred_T_O_C[original_idx, :3, 3]
                    )
                )
                self.pair_pose["camera_center_disagreement_m"].append(
                    center_disagreement
                )
                self.pair_pose["camera_center_disagreement_d"].append(
                    center_disagreement / diameter
                )
                self.pair_pose["canonical_rotation_disagreement_deg"].append(
                    float(
                        rotation_error_deg(
                            pred_crop_canonical[:3, :3],
                            pred_original[:3, :3],
                        )
                    )
                )
                self.pair_pose["canonical_translation_disagreement_m"].append(
                    float(
                        np.linalg.norm(
                            pred_crop_canonical[:3, 3]
                            - pred_original[:3, 3]
                        )
                    )
                )
                self.pair_pose["canonical_translation_disagreement_d"].append(
                    self.pair_pose["canonical_translation_disagreement_m"][-1]
                    / diameter
                )

                self._accumulate_dense_pair(
                    crop_points=pred_points[batch_idx, crop_idx],
                    original_points=pred_points[batch_idx, original_idx],
                    crop_valid=valid_masks[batch_idx, crop_idx],
                    original_valid=valid_masks[batch_idx, original_idx],
                    crop_intrinsics=intrinsics[batch_idx, crop_idx],
                    H_crop_from_original=H_crop_from_original,
                    R_old_to_new=R_old_to_new_t,
                    pred_T_W_C_crop=pred_T_W_C_torch[batch_idx, crop_idx],
                    pred_T_W_C_original=pred_T_W_C_torch[batch_idx, original_idx],
                )
                self.pair_count += 1

    def compute(self) -> dict[str, float]:
        output: dict[str, float] = {"pair_count": float(self.pair_count)}
        for role in self._POSE_ROLES:
            values = self.pose_values[role]
            output.update(
                {
                    f"{role}/count": float(len(values["add_or_adds_d"])),
                    f"{role}/add_mean_d": safe_mean(values["add_d"]),
                    f"{role}/add_or_adds_mean_d": safe_mean(
                        values["add_or_adds_d"]
                    ),
                    f"{role}/add_or_adds_0_1d": recall_below(
                        values["add_or_adds_d"],
                        0.1,
                    ),
                    f"{role}/rotation_mean_deg": safe_mean(values["rotation_deg"]),
                    f"{role}/rotation_median_deg": safe_median(
                        values["rotation_deg"]
                    ),
                    f"{role}/translation_mean_m": safe_mean(
                        values["translation_m"]
                    ),
                    f"{role}/translation_median_m": safe_median(
                        values["translation_m"]
                    ),
                }
            )

        for namespace, values in (
            ("query_pair_pose", self.pair_pose),
            ("query_pair_dense", self.pair_dense),
            ("query_pair_world", self.pair_world),
        ):
            for key, entries in values.items():
                output[f"{namespace}/{key}"] = safe_mean(entries)
        return output
