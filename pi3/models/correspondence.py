from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CorrespondenceBatch:
    """Accepted query-to-reference patch correspondences for one loss call."""

    batch_indices: torch.Tensor
    query_view_indices: torch.Tensor
    reference_view_indices: torch.Tensor
    query_y: torch.Tensor
    query_x: torch.Tensor
    reference_y: torch.Tensor
    reference_x: torch.Tensor
    query_token_indices: torch.Tensor
    reference_token_indices: torch.Tensor
    depth_errors: torch.Tensor

    @property
    def num_pairs(self) -> int:
        return int(self.batch_indices.numel())


def stack_view_tensor(batch: list[dict], key: str, *, device: torch.device | None = None) -> torch.Tensor:
    """Stack a collated per-view tensor from ``list[N]`` to ``(B, N, ...)``."""

    values = []
    for view in batch:
        value = view[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if device is not None:
            value = value.to(device)
        values.append(value)
    return torch.stack(values, dim=1)


def _patch_representatives(
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return one valid representative point per image patch.

    The representative uses the mean valid pixel coordinate and mean valid
    depth inside each patch. This is deliberately cheap and matches the
    debug visualizer's patch-level correspondence semantics closely enough for
    training-time consistency.
    """

    height, width = depth.shape[-2:]
    token_h = int(height // patch_size)
    token_w = int(width // patch_size)
    crop_h = token_h * patch_size
    crop_w = token_w * patch_size
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Image is too small for patch_size={patch_size}: {(height, width)}")

    depth = depth[:crop_h, :crop_w]
    valid = (mask[:crop_h, :crop_w].bool() & (depth > 0)).float()
    area = float(patch_size * patch_size)

    depth_4d = depth[None, None]
    valid_4d = valid[None, None]
    count = F.avg_pool2d(valid_4d, patch_size, stride=patch_size)[0, 0] * area
    depth_sum = F.avg_pool2d(depth_4d * valid_4d, patch_size, stride=patch_size)[0, 0] * area

    yy, xx = torch.meshgrid(
        torch.arange(crop_h, device=depth.device, dtype=depth.dtype),
        torch.arange(crop_w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    u_sum = F.avg_pool2d(xx[None, None] * valid_4d, patch_size, stride=patch_size)[0, 0] * area
    v_sum = F.avg_pool2d(yy[None, None] * valid_4d, patch_size, stride=patch_size)[0, 0] * area

    ok = count > 0
    denom = count.clamp_min(1.0)
    u = u_sum / denom + 0.5
    v = v_sum / denom + 0.5
    z = depth_sum / denom
    return u, v, z, ok, token_h, token_w


def _invert_se3(transform: torch.Tensor) -> torch.Tensor:
    """Invert one rigid 4x4 transform."""

    inv = torch.zeros_like(transform)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv[:3, :3] = rotation.transpose(0, 1)
    inv[:3, 3] = -(inv[:3, :3] @ translation)
    inv[3, 3] = 1.0
    return inv


def _select_reference_indices(
    reference_indices: torch.Tensor,
    *,
    max_reference_per_query: int,
    strategy: str,
) -> torch.Tensor:
    if reference_indices.numel() <= max_reference_per_query:
        return reference_indices
    if strategy == "first":
        return reference_indices[:max_reference_per_query]
    if strategy == "random":
        order = torch.randperm(reference_indices.numel(), device=reference_indices.device)
        return reference_indices[order[:max_reference_per_query]]
    if strategy == "uniform":
        positions = torch.linspace(
            0,
            reference_indices.numel() - 1,
            steps=max_reference_per_query,
            device=reference_indices.device,
        ).round().long()
        return reference_indices[positions]
    raise ValueError(f"Unknown reference selection strategy: {strategy}")


def _subsample_indices(num_items: int, max_items: int, *, strategy: str, device: torch.device) -> torch.Tensor:
    if max_items <= 0 or num_items <= max_items:
        return torch.arange(num_items, device=device)
    if strategy == "first":
        return torch.arange(max_items, device=device)
    if strategy == "random":
        return torch.randperm(num_items, device=device)[:max_items]
    if strategy == "uniform":
        return torch.linspace(0, num_items - 1, steps=max_items, device=device).round().long()
    raise ValueError(f"Unknown pair subsample strategy: {strategy}")


def build_query_reference_correspondences(
    batch: list[dict],
    *,
    patch_size: int = 14,
    max_reference_per_query: int = 1,
    reference_selection: str = "uniform",
    max_pairs: int = 512,
    pair_subsample: str = "uniform",
    depth_abs_tol: float = 0.01,
    depth_rel_tol: float = 0.05,
    device: torch.device | None = None,
) -> CorrespondenceBatch:
    """Build GT query-to-reference correspondences from BOP depth and poses.

    Returned correspondences are pixel-level indices into dense Pi3 pointmaps
    plus token-level indices for optional DINO patch-feature weighting.
    """

    if device is None:
        device = stack_view_tensor(batch, "depthmap").device

    depths = stack_view_tensor(batch, "depthmap", device=device).float()
    masks = stack_view_tensor(batch, "valid_mask", device=device).bool()
    intrinsics = stack_view_tensor(batch, "camera_intrinsics", device=device).float()
    transforms = stack_view_tensor(batch, "T_C_O", device=device).float()
    is_reference = stack_view_tensor(batch, "is_reference", device=device).bool()
    is_query = stack_view_tensor(batch, "is_query", device=device).bool()

    batch_size, _, height, width = depths.shape
    records: list[tuple[torch.Tensor, ...]] = []

    for batch_idx in range(batch_size):
        reference_indices = torch.nonzero(is_reference[batch_idx], as_tuple=False).flatten()
        query_indices = torch.nonzero(is_query[batch_idx], as_tuple=False).flatten()
        if reference_indices.numel() == 0 or query_indices.numel() == 0:
            continue

        selected_references = _select_reference_indices(
            reference_indices,
            max_reference_per_query=max(1, int(max_reference_per_query)),
            strategy=reference_selection,
        )

        for query_idx in query_indices.tolist():
            q_depth = depths[batch_idx, query_idx]
            q_mask = masks[batch_idx, query_idx]
            q_K = intrinsics[batch_idx, query_idx]
            T_O_Cq = _invert_se3(transforms[batch_idx, query_idx])
            u, v, z, ok, token_h, token_w = _patch_representatives(q_depth, q_mask, patch_size=patch_size)

            token_y, token_x = torch.meshgrid(
                torch.arange(token_h, device=device),
                torch.arange(token_w, device=device),
                indexing="ij",
            )
            flat_ok = ok.flatten()
            if not bool(flat_ok.any()):
                continue

            u_flat = u.flatten()
            v_flat = v.flatten()
            z_flat = z.flatten()
            token_y_flat = token_y.flatten()
            token_x_flat = token_x.flatten()

            x_cam = (u_flat - q_K[0, 2]) * z_flat / q_K[0, 0]
            y_cam = (v_flat - q_K[1, 2]) * z_flat / q_K[1, 1]
            ones = torch.ones_like(z_flat)
            p_Cq = torch.stack([x_cam, y_cam, z_flat, ones], dim=-1)
            p_O = p_Cq @ T_O_Cq.transpose(0, 1)

            for reference_idx in selected_references.tolist():
                r_depth = depths[batch_idx, reference_idx]
                r_mask = masks[batch_idx, reference_idx]
                r_K = intrinsics[batch_idx, reference_idx]
                T_Cr_O = transforms[batch_idx, reference_idx]

                p_Cr = p_O @ T_Cr_O.transpose(0, 1)
                z_ref = p_Cr[:, 2]
                front = z_ref > 1e-8
                ur = r_K[0, 0] * p_Cr[:, 0] / z_ref.clamp_min(1e-8) + r_K[0, 2]
                vr = r_K[1, 1] * p_Cr[:, 1] / z_ref.clamp_min(1e-8) + r_K[1, 2]
                in_bounds = (ur >= 0) & (ur < width) & (vr >= 0) & (vr < height)
                candidates = flat_ok & front & in_bounds
                if not bool(candidates.any()):
                    continue

                candidate_indices = torch.nonzero(candidates, as_tuple=False).flatten()
                rx = ur[candidate_indices].round().long().clamp(0, width - 1)
                ry = vr[candidate_indices].round().long().clamp(0, height - 1)
                ref_mask_ok = r_mask[ry, rx]
                if not bool(ref_mask_ok.any()):
                    continue

                candidate_indices = candidate_indices[ref_mask_ok]
                rx = rx[ref_mask_ok]
                ry = ry[ref_mask_ok]
                ref_depth = r_depth[ry, rx]
                ref_depth_ok = ref_depth > 0
                if not bool(ref_depth_ok.any()):
                    continue

                candidate_indices = candidate_indices[ref_depth_ok]
                rx = rx[ref_depth_ok]
                ry = ry[ref_depth_ok]
                ref_depth = ref_depth[ref_depth_ok]
                depth_error = (z_ref[candidate_indices] - ref_depth).abs()
                tolerance = torch.maximum(
                    torch.full_like(ref_depth, float(depth_abs_tol)),
                    ref_depth * float(depth_rel_tol),
                )
                depth_ok = depth_error <= tolerance
                if not bool(depth_ok.any()):
                    continue

                candidate_indices = candidate_indices[depth_ok]
                rx = rx[depth_ok]
                ry = ry[depth_ok]
                depth_error = depth_error[depth_ok]

                qx = u_flat[candidate_indices].round().long().clamp(0, width - 1)
                qy = v_flat[candidate_indices].round().long().clamp(0, height - 1)
                ref_token_x = torch.div(rx, patch_size, rounding_mode="floor")
                ref_token_y = torch.div(ry, patch_size, rounding_mode="floor")
                reference_token_w = width // patch_size
                q_token = token_y_flat[candidate_indices] * token_w + token_x_flat[candidate_indices]
                r_token = ref_token_y * reference_token_w + ref_token_x
                count = candidate_indices.numel()

                records.append(
                    (
                        torch.full((count,), batch_idx, device=device, dtype=torch.long),
                        torch.full((count,), query_idx, device=device, dtype=torch.long),
                        torch.full((count,), reference_idx, device=device, dtype=torch.long),
                        qy,
                        qx,
                        ry,
                        rx,
                        q_token.long(),
                        r_token.long(),
                        depth_error,
                    )
                )

    if not records:
        empty = torch.empty(0, device=device, dtype=torch.long)
        empty_float = torch.empty(0, device=device)
        return CorrespondenceBatch(empty, empty, empty, empty, empty, empty, empty, empty, empty, empty_float)

    fields = [torch.cat(values, dim=0) for values in zip(*records)]
    num_pairs = int(fields[0].numel())
    keep = _subsample_indices(num_pairs, int(max_pairs), strategy=pair_subsample, device=device)
    fields = [field[keep] for field in fields]

    return CorrespondenceBatch(*fields)
