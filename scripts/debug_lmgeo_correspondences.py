from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from safetensors.torch import safe_open

from datasets.base.transforms import ImgToTensor
from datasets.lmgeo_dataset import LMGeoSequenceDataset
from pi3.models.dinov2.hub.backbones import dinov2_vitl14_reg
from pi3.visualizations.utils import add_title, draw_mask_contour, make_grid, tensor_image_to_uint8


@dataclass
class Correspondence:
    query_token: tuple[int, int]
    reference_token: tuple[int, int]
    query_uv: tuple[float, float]
    reference_uv: tuple[float, float]
    query_depth_m: float
    projected_reference_depth_m: float
    reference_depth_m: float
    depth_error_m: float
    dino_similarity: float | None


@dataclass
class RejectCounts:
    query_invalid: int = 0
    behind_reference: int = 0
    outside_reference: int = 0
    outside_reference_mask: int = 0
    reference_depth_invalid: int = 0
    depth_inconsistent: int = 0

    @property
    def total(self) -> int:
        return sum(asdict(self).values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize GT-depth/pose correspondences between one LMGeo reference "
            "view and one query view, optionally weighted by frozen Pi3 DINO features."
        )
    )
    parser.add_argument("--data-root", default="/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o")
    parser.add_argument("--object-id", type=int, default=8)
    parser.add_argument("--query-split", default="new_val", choices=["train_pbr", "new_val", "test"])
    parser.add_argument("--query-source", default="auto", choices=["auto", "windows", "bop_targets"])
    parser.add_argument("--query-target-file", default="lmo/test_targets_bop19.json")
    parser.add_argument("--query-scene-id", type=int, default=45)
    parser.add_argument("--query-subscene-id", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-reference", type=int, default=5)
    parser.add_argument("--num-query", type=int, default=1)
    parser.add_argument("--reference-view", type=int, default=0, help="Index inside reference views of the sampled tuple.")
    parser.add_argument("--query-view", type=int, default=0, help="Index inside query views of the sampled tuple.")
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--max-draw", type=int, default=120)
    parser.add_argument("--depth-rel-tol", type=float, default=0.05)
    parser.add_argument("--depth-abs-tol", type=float, default=0.01)
    parser.add_argument("--min-dino-sim", type=float, default=None)
    parser.add_argument("--dino-source", default="pi3", choices=["pi3", "none"])
    parser.add_argument("--dino-layer", default="final", help="'final' or a DINO block index, e.g. 17 or 23.")
    parser.add_argument("--pi3-ckpt", default="ckpts/Pi3/model.safetensors")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="outputs/correspondence_debug")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--draw-rejected", action="store_true")
    return parser.parse_args()


def make_dataset(args: argparse.Namespace) -> LMGeoSequenceDataset:
    query_scene_ids: Any = "all"
    query_subscene_ids: Any = "all"
    if args.query_split != "test":
        query_scene_ids = [int(args.query_scene_id)]
        query_subscene_ids = [int(args.query_subscene_id)]

    return LMGeoSequenceDataset(
        data_root=args.data_root,
        object_ids=[int(args.object_id)],
        reference_split="train",
        query_split=args.query_split,
        query_scene_ids=query_scene_ids,
        query_subscene_ids=query_subscene_ids,
        query_source=args.query_source,
        query_target_file=args.query_target_file,
        query_window_size=25,
        num_reference_range=(args.num_reference, args.num_reference),
        num_query_range=(args.num_query, args.num_query),
        reference_rgb_masking=True,
        query_rgb_masking=False,
        depth_masking=True,
        mask_type="mask_visib",
        visibility_min=0.1 if args.query_split != "test" else 0.0,
        reference_selection="uniform",
        query_selection="uniform" if args.num_query > 1 else "first",
        depth_unit_scale=0.001,
        filter_center_crop_visibility=True,
        min_center_crop_bbox_area=1.0,
        filter_preprocessed_query_depth=False,
        filter_target_center_crop_visibility=True,
        filter_target_preprocessed_depth=True,
        allow_repeat=False,
        aug_crop=False,
        aug_focal=False,
        z_far=0,
        resolution=[[int(args.resize), int(args.resize)]],
        transform=ImgToTensor,
        shuffle=False,
        mode="test" if args.query_split == "test" else "val",
    )


def select_views(views: list[dict[str, Any]], reference_view: int, query_view: int) -> tuple[dict[str, Any], dict[str, Any]]:
    references = [view for view in views if bool(view["is_reference"])]
    queries = [view for view in views if bool(view["is_query"])]
    if not references:
        raise ValueError("Sample contains no reference views")
    if not queries:
        raise ValueError("Sample contains no query views")
    if reference_view >= len(references):
        raise IndexError(f"reference-view={reference_view} but only {len(references)} reference views exist")
    if query_view >= len(queries):
        raise IndexError(f"query-view={query_view} but only {len(queries)} query views exist")
    return references[reference_view], queries[query_view]


def load_pi3_dino_encoder(ckpt_path: str | Path, device: str) -> torch.nn.Module:
    encoder = dinov2_vitl14_reg(pretrained=False)
    if hasattr(encoder, "mask_token"):
        del encoder.mask_token

    state = {}
    ckpt_path = Path(ckpt_path)
    with safe_open(str(ckpt_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("encoder."):
                state[key[len("encoder.") :]] = handle.get_tensor(key)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"[DINO] Loaded encoder from {ckpt_path}: missing={len(missing)}, unexpected={len(unexpected)}")
    encoder.to(device)
    encoder.eval()
    return encoder


@torch.no_grad()
def compute_dino_features(
    encoder: torch.nn.Module | None,
    views: list[dict[str, Any]],
    *,
    device: str,
    layer: str,
) -> np.ndarray | None:
    if encoder is None:
        return None

    images = torch.stack([view["img"] for view in views], dim=0).to(device=device, dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    images = (images - mean) / std

    if layer == "final":
        features = encoder(images, is_training=True)["x_norm_patchtokens"]
    else:
        features = encoder.get_intermediate_layers(images, n=[int(layer)], norm=True)[0]
    features = F.normalize(features.float(), dim=-1)
    return features.detach().cpu().numpy()


def token_point_from_patch(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    token_y: int,
    token_x: int,
    patch_size: int,
) -> tuple[float, float, float] | None:
    y0 = token_y * patch_size
    x0 = token_x * patch_size
    patch_depth = depth[y0 : y0 + patch_size, x0 : x0 + patch_size]
    patch_mask = valid_mask[y0 : y0 + patch_size, x0 : x0 + patch_size] & (patch_depth > 0)
    if not np.any(patch_mask):
        return None
    ys, xs = np.nonzero(patch_mask)
    u = float(x0 + np.mean(xs) + 0.5)
    v = float(y0 + np.mean(ys) + 0.5)
    z = float(np.median(patch_depth[patch_mask]))
    return u, v, z


def unproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (u - K[0, 2]) * z / K[0, 0],
            (v - K[1, 2]) * z / K[1, 1],
            z,
            1.0,
        ],
        dtype=np.float64,
    )


def project(point_camera: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    z = float(point_camera[2])
    return (
        float(K[0, 0] * point_camera[0] / z + K[0, 2]),
        float(K[1, 1] * point_camera[1] / z + K[1, 2]),
    )


def nearest_depth(depth: np.ndarray, u: float, v: float) -> float:
    x = int(round(u))
    y = int(round(v))
    if y < 0 or y >= depth.shape[0] or x < 0 or x >= depth.shape[1]:
        return 0.0
    return float(depth[y, x])


def build_correspondences(
    reference: dict[str, Any],
    query: dict[str, Any],
    *,
    patch_size: int,
    depth_rel_tol: float,
    depth_abs_tol: float,
    dino_features: np.ndarray | None = None,
    reference_view_global_idx: int = 0,
    query_view_global_idx: int = 1,
    min_dino_sim: float | None = None,
) -> tuple[list[Correspondence], list[tuple[float, float, str]], RejectCounts]:
    query_depth = np.asarray(query["depthmap"], dtype=np.float32)
    reference_depth = np.asarray(reference["depthmap"], dtype=np.float32)
    query_mask = np.asarray(query["valid_mask"], dtype=bool)
    reference_mask = np.asarray(reference["valid_mask"], dtype=bool)
    Kq = np.asarray(query["camera_intrinsics"], dtype=np.float64)
    Kr = np.asarray(reference["camera_intrinsics"], dtype=np.float64)
    T_Cq_O = np.asarray(query["T_C_O"], dtype=np.float64)
    T_Cr_O = np.asarray(reference["T_C_O"], dtype=np.float64)
    T_O_Cq = np.linalg.inv(T_Cq_O)

    H, W = query_depth.shape
    token_h, token_w = H // patch_size, W // patch_size
    reference_token_h, reference_token_w = reference_depth.shape[0] // patch_size, reference_depth.shape[1] // patch_size

    matches: list[Correspondence] = []
    rejected_for_draw: list[tuple[float, float, str]] = []
    rejects = RejectCounts()

    for ty in range(token_h):
        for tx in range(token_w):
            point = token_point_from_patch(query_depth, query_mask, ty, tx, patch_size)
            if point is None:
                rejects.query_invalid += 1
                continue
            uq, vq, zq = point
            p_Cq = unproject(uq, vq, zq, Kq)
            p_O = T_O_Cq @ p_Cq
            p_Cr_h = T_Cr_O @ p_O
            z_ref_projected = float(p_Cr_h[2])
            if z_ref_projected <= 1e-8:
                rejects.behind_reference += 1
                rejected_for_draw.append((uq, vq, "behind"))
                continue

            ur, vr = project(p_Cr_h[:3], Kr)
            if ur < 0 or ur >= reference_depth.shape[1] or vr < 0 or vr >= reference_depth.shape[0]:
                rejects.outside_reference += 1
                rejected_for_draw.append((uq, vq, "outside"))
                continue

            ref_x = int(round(ur))
            ref_y = int(round(vr))
            if not bool(reference_mask[ref_y, ref_x]):
                rejects.outside_reference_mask += 1
                rejected_for_draw.append((uq, vq, "mask"))
                continue

            ref_depth = nearest_depth(reference_depth, ur, vr)
            if ref_depth <= 0:
                rejects.reference_depth_invalid += 1
                rejected_for_draw.append((uq, vq, "depth0"))
                continue

            depth_error = abs(z_ref_projected - ref_depth)
            tolerance = max(depth_abs_tol, depth_rel_tol * ref_depth)
            if depth_error > tolerance:
                rejects.depth_inconsistent += 1
                rejected_for_draw.append((uq, vq, "depth"))
                continue

            ref_tx = int(np.floor(ur / patch_size))
            ref_ty = int(np.floor(vr / patch_size))
            if not (0 <= ref_tx < reference_token_w and 0 <= ref_ty < reference_token_h):
                rejects.outside_reference += 1
                rejected_for_draw.append((uq, vq, "token"))
                continue

            dino_similarity = None
            if dino_features is not None:
                query_flat = ty * token_w + tx
                reference_flat = ref_ty * reference_token_w + ref_tx
                q_feat = dino_features[query_view_global_idx, query_flat]
                r_feat = dino_features[reference_view_global_idx, reference_flat]
                dino_similarity = float(np.dot(q_feat, r_feat))
                if min_dino_sim is not None and dino_similarity < min_dino_sim:
                    rejected_for_draw.append((uq, vq, "dino"))
                    continue

            matches.append(
                Correspondence(
                    query_token=(ty, tx),
                    reference_token=(ref_ty, ref_tx),
                    query_uv=(uq, vq),
                    reference_uv=(ur, vr),
                    query_depth_m=zq,
                    projected_reference_depth_m=z_ref_projected,
                    reference_depth_m=ref_depth,
                    depth_error_m=depth_error,
                    dino_similarity=dino_similarity,
                )
            )

    return matches, rejected_for_draw, rejects


def color_from_similarity(similarity: float | None) -> tuple[int, int, int]:
    if similarity is None or not np.isfinite(similarity):
        return (60, 200, 255)
    value = float(np.clip((similarity + 1.0) * 0.5, 0.0, 1.0))
    red = int(round(255 * (1.0 - value)))
    green = int(round(255 * value))
    return (red, green, 40)


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], radius: int = 3) -> None:
    draw.line([(x - radius, y - radius), (x + radius, y + radius)], fill=color, width=1)
    draw.line([(x - radius, y + radius), (x + radius, y - radius)], fill=color, width=1)


def visualize_matches(
    reference: dict[str, Any],
    query: dict[str, Any],
    matches: list[Correspondence],
    rejected: list[tuple[float, float, str]],
    *,
    max_draw: int,
    seed: int,
    draw_rejected: bool,
    title: str,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    reference_image = Image.fromarray(tensor_image_to_uint8(reference["img"]))
    query_image = Image.fromarray(tensor_image_to_uint8(query["img"]))
    draw_mask_contour(reference_image, reference["valid_mask"], color=(255, 230, 0), width=1)
    draw_mask_contour(query_image, query["valid_mask"], color=(255, 230, 0), width=1)

    Wq, Hq = query_image.size
    Wr, Hr = reference_image.size
    canvas = Image.new("RGB", (Wq + Wr, max(Hq, Hr)), color=(8, 8, 8))
    canvas.paste(query_image, (0, 0))
    canvas.paste(reference_image, (Wq, 0))
    draw = ImageDraw.Draw(canvas)

    if draw_rejected:
        for uq, vq, reason in rejected[: max_draw * 3]:
            color = (120, 120, 120) if reason != "depth" else (255, 80, 80)
            draw_cross(draw, uq, vq, color=color, radius=2)

    draw_matches = matches
    if len(draw_matches) > max_draw:
        indices = rng.choice(len(draw_matches), size=max_draw, replace=False)
        draw_matches = [draw_matches[int(idx)] for idx in np.sort(indices)]

    for idx, match in enumerate(draw_matches):
        color = color_from_similarity(match.dino_similarity)
        uq, vq = match.query_uv
        ur, vr = match.reference_uv
        ur += Wq
        radius = 3
        draw.line([(uq, vq), (ur, vr)], fill=color, width=1)
        draw.ellipse((uq - radius, vq - radius, uq + radius, vq + radius), outline=color, width=2)
        draw.ellipse((ur - radius, vr - radius, ur + radius, vr + radius), outline=color, width=2)
        if idx < 20:
            draw.text((uq + 4, vq + 2), str(idx), fill=color)
            draw.text((ur + 4, vr + 2), str(idx), fill=color)

    return add_title(canvas, title, height=34)


def view_global_indices(views: list[dict[str, Any]], reference: dict[str, Any], query: dict[str, Any]) -> tuple[int, int]:
    ref_idx = next(idx for idx, view in enumerate(views) if view is reference)
    query_idx = next(idx for idx, view in enumerate(views) if view is query)
    return ref_idx, query_idx


def match_summary(matches: list[Correspondence], rejects: RejectCounts) -> dict[str, Any]:
    sims = [m.dino_similarity for m in matches if m.dino_similarity is not None and np.isfinite(m.dino_similarity)]
    depth_errors = [m.depth_error_m for m in matches]
    return {
        "accepted": len(matches),
        "rejected": asdict(rejects),
        "rejected_total": rejects.total,
        "depth_error_m": {
            "mean": float(np.mean(depth_errors)) if depth_errors else None,
            "median": float(np.median(depth_errors)) if depth_errors else None,
            "p95": float(np.percentile(depth_errors, 95)) if depth_errors else None,
        },
        "dino_similarity": {
            "mean": float(np.mean(sims)) if sims else None,
            "median": float(np.median(sims)) if sims else None,
            "p05": float(np.percentile(sims, 5)) if sims else None,
            "p95": float(np.percentile(sims, 95)) if sims else None,
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(args)
    total_views = int(args.num_reference + args.num_query)
    views = dataset[(int(args.sample_index), 0, total_views)]
    reference, query = select_views(views, args.reference_view, args.query_view)
    ref_global_idx, query_global_idx = view_global_indices(views, reference, query)

    encoder = None
    if args.dino_source == "pi3":
        encoder = load_pi3_dino_encoder(args.pi3_ckpt, args.device)
    dino_features = compute_dino_features(encoder, views, device=args.device, layer=str(args.dino_layer))

    matches, rejected, rejects = build_correspondences(
        reference,
        query,
        patch_size=int(args.patch_size),
        depth_rel_tol=float(args.depth_rel_tol),
        depth_abs_tol=float(args.depth_abs_tol),
        dino_features=dino_features,
        reference_view_global_idx=ref_global_idx,
        query_view_global_idx=query_global_idx,
        min_dino_sim=args.min_dino_sim,
    )

    prefix = args.prefix
    if prefix is None:
        prefix = (
            f"obj{args.object_id:06d}_{args.query_split}_"
            f"sample{args.sample_index:04d}_ref{args.reference_view}_query{args.query_view}_"
            f"dino{args.dino_source}_layer{args.dino_layer}"
        )

    summary = match_summary(matches, rejects)
    summary.update(
        {
            "args": vars(args),
            "reference": {
                "source": str(reference.get("source", "")),
                "im_id": int(reference.get("im_id", -1)),
                "scene_id": int(reference.get("scene_id", -1)),
            },
            "query": {
                "source": str(query.get("source", "")),
                "im_id": int(query.get("im_id", -1)),
                "scene_id": int(query.get("scene_id", -1)),
            },
            "first_matches": [asdict(match) for match in matches[:50]],
        }
    )

    title = (
        f"{prefix} | accepted={len(matches)} rejected={rejects.total} "
        f"| DINO median={summary['dino_similarity']['median']}"
    )
    visualization = visualize_matches(
        reference,
        query,
        matches,
        rejected,
        max_draw=int(args.max_draw),
        seed=int(args.seed),
        draw_rejected=bool(args.draw_rejected),
        title=title,
    )

    reference_panel = add_title(Image.fromarray(tensor_image_to_uint8(reference["img"])), f"reference: {reference.get('source', '')}")
    query_panel = add_title(Image.fromarray(tensor_image_to_uint8(query["img"])), f"query: {query.get('source', '')}")
    image = make_grid([visualization, make_grid([query_panel, reference_panel], columns=2)], columns=1)

    image_path = output_dir / f"{prefix}.png"
    json_path = output_dir / f"{prefix}.json"
    image.save(image_path)
    json_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k != "first_matches"}, indent=2))
    print(f"Wrote {image_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
