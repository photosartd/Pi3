from __future__ import annotations

import torch


def pixel_center_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an ``(H, W, 3)`` homogeneous pixel-center grid."""

    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(height, width)}")
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    return torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)


def intrinsics_to_ray_map(
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert pixel-space intrinsics to Pi3X-style two-channel ray maps.

    The returned tensor has shape ``(..., H, W, 2)`` and stores ``x/z`` and
    ``y/z`` for ``K^{-1} [u + 0.5, v + 0.5, 1]``. The calculation is kept in
    float32 for a stable matrix inverse. Callers may cast the result before
    applying a learned patch projection.
    """

    if not torch.is_tensor(intrinsics):
        intrinsics = torch.as_tensor(intrinsics)
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected intrinsics ending in (3, 3), got {tuple(intrinsics.shape)}"
        )

    intrinsics_f = intrinsics.to(dtype=torch.float32)
    if not bool(torch.isfinite(intrinsics_f).all()):
        raise ValueError("Camera intrinsics contain non-finite values")

    pixels = pixel_center_grid(
        int(height),
        int(width),
        device=intrinsics_f.device,
        dtype=intrinsics_f.dtype,
    )
    inverse_intrinsics = torch.linalg.inv(intrinsics_f)
    rays = torch.einsum("...ij,hwj->...hwi", inverse_intrinsics, pixels)
    ray_z = rays[..., 2:3]
    if bool((ray_z.abs() <= float(eps)).any()):
        raise ValueError("Camera intrinsics produced rays with near-zero z")
    return rays[..., :2] / ray_z
