import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge
from ..utils.alignment import align_points_scale
from .correspondence import build_query_reference_correspondences

from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__
from datasets.base.observation import (
    ObservationCapability,
    batch_supports_capability,
)

# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

# ---------------------------------------------------------------------------
# PointLoss: Scale-invariant Local Pointmap
# ---------------------------------------------------------------------------

class PointLoss(nn.Module):
    def __init__(self, local_align_res=4096, train_conf=False, expected_dist_thresh=0.02):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')

        self.train_conf = train_conf
        if self.train_conf:
            self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh

    def prepare_segformer(self):
        from pi3.models.segformer.model import EncoderDecoder
        self.segformer = EncoderDecoder()
        self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        self.segformer = self.segformer.cuda()

    def predict_sky_mask(self, imgs):
        with torch.no_grad():
            output = self.segformer.inference_(imgs)
            output = output == 2
        return output

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        # conf loss
        if self.train_conf:
            pred_conf = pred['conf']

            # probability loss
            valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
            local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

            sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
            sky_mask[valid_masks] = False
            if sky_mask.sum() == 0:
                sky_mask_loss = 0.0 * aligned_local_pts.mean()
            else:
                sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
            
            final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
            details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points'] * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local

# ---------------------------------------------------------------------------
# CameraLoss: Affine-invariant Camera Pose
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)

# ---------------------------------------------------------------------------
# CorrespondenceConsistencyLoss: GT-corresponding object patches agree in 3D
# ---------------------------------------------------------------------------

class CorrespondenceConsistencyLoss(nn.Module):
    """Cross-view consistency over GT query-to-reference correspondences.

    The correspondence indices are built from GT object depth, intrinsics, and
    BOP object-to-camera poses. The loss itself is applied to Pi3's predicted
    shared-frame pointmaps, so gradients flow only through the prediction.
    Optional DINO weights are detached and act only as correspondence
    confidence.
    """

    def __init__(
        self,
        patch_size: int = 14,
        max_reference_per_query: int = 1,
        reference_selection: str = "uniform",
        max_pairs: int = 512,
        pair_subsample: str = "uniform",
        depth_abs_tol: float = 0.01,
        depth_rel_tol: float = 0.05,
        huber_beta: float = 0.01,
        use_dino_weights: bool = False,
        dino_layer: int = 17,
        dino_center: float = 0.2,
        dino_temperature: float = 0.1,
        min_dino_weight: float = 0.05,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.max_reference_per_query = int(max_reference_per_query)
        self.reference_selection = str(reference_selection)
        self.max_pairs = int(max_pairs)
        self.pair_subsample = str(pair_subsample)
        self.depth_abs_tol = float(depth_abs_tol)
        self.depth_rel_tol = float(depth_rel_tol)
        self.huber_beta = float(huber_beta)
        self.use_dino_weights = bool(use_dino_weights)
        self.dino_layer = int(dino_layer)
        self.dino_center = float(dino_center)
        self.dino_temperature = max(float(dino_temperature), 1e-6)
        self.min_dino_weight = float(min_dino_weight)

    def _global_points_from_normalized_prediction(self, pred: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.einsum(
            'bnij, bnhwj -> bnhwi',
            pred['camera_poses'],
            homogenize_points(pred['local_points']),
        )[..., :3]

    def _dino_features(self, pred: dict[str, torch.Tensor]) -> torch.Tensor | None:
        features = pred.get('dino_features', None)
        if features is None:
            return None
        if isinstance(features, dict):
            return features.get(str(self.dino_layer), features.get(self.dino_layer, None))
        return features

    def forward(self, pred: dict[str, torch.Tensor], gt_raw: list[dict]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_points = self._global_points_from_normalized_prediction(pred)
        device = pred_points.device
        zero = pred_points.mean() * 0.0

        correspondences = build_query_reference_correspondences(
            gt_raw,
            patch_size=self.patch_size,
            max_reference_per_query=self.max_reference_per_query,
            reference_selection=self.reference_selection,
            max_pairs=self.max_pairs,
            pair_subsample=self.pair_subsample,
            depth_abs_tol=self.depth_abs_tol,
            depth_rel_tol=self.depth_rel_tol,
            device=device,
        )
        if correspondences.num_pairs == 0:
            return zero, {
                'correspondence_num_pairs': torch.zeros((), device=device),
                'correspondence_depth_error': torch.zeros((), device=device),
                'correspondence_dino_similarity': torch.zeros((), device=device),
                'correspondence_dino_weight': torch.zeros((), device=device),
                'correspondence_num_pairs_loss_stat': torch.zeros((), device=device),
                'correspondence_depth_error_loss_stat': torch.zeros((), device=device),
                'correspondence_dino_similarity_loss_stat': torch.zeros((), device=device),
                'correspondence_dino_weight_loss_stat': torch.zeros((), device=device),
            }

        q_points = pred_points[
            correspondences.batch_indices,
            correspondences.query_view_indices,
            correspondences.query_y,
            correspondences.query_x,
        ]
        r_points = pred_points[
            correspondences.batch_indices,
            correspondences.reference_view_indices,
            correspondences.reference_y,
            correspondences.reference_x,
        ]
        per_pair_loss = F.smooth_l1_loss(
            q_points.float(),
            r_points.float(),
            reduction='none',
            beta=self.huber_beta,
        ).mean(dim=-1)

        weights = torch.ones_like(per_pair_loss)
        dino_similarity = torch.zeros_like(per_pair_loss)
        dino = self._dino_features(pred)
        if self.use_dino_weights and dino is not None:
            dino = dino.to(device=device, dtype=torch.float32)
            q_feat = dino[
                correspondences.batch_indices,
                correspondences.query_view_indices,
                correspondences.query_token_indices,
            ]
            r_feat = dino[
                correspondences.batch_indices,
                correspondences.reference_view_indices,
                correspondences.reference_token_indices,
            ]
            q_feat = F.normalize(q_feat, dim=-1)
            r_feat = F.normalize(r_feat, dim=-1)
            dino_similarity = (q_feat * r_feat).sum(dim=-1).detach()
            weights = torch.sigmoid((dino_similarity - self.dino_center) / self.dino_temperature)
            weights = weights.clamp_min(self.min_dino_weight).detach()

        loss = (per_pair_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        details = {
            'correspondence_num_pairs': torch.as_tensor(float(correspondences.num_pairs), device=device),
            'correspondence_depth_error': correspondences.depth_errors.float().mean(),
            'correspondence_dino_similarity': dino_similarity.mean(),
            'correspondence_dino_weight': weights.mean(),
        }
        details.update(
            {
                # The current trainer scalarizes/gathers values whose key
                # contains "loss"; keep aliases so these diagnostics reach
                # TensorBoard without widening trainer behavior.
                'correspondence_num_pairs_loss_stat': details['correspondence_num_pairs'],
                'correspondence_depth_error_loss_stat': details['correspondence_depth_error'],
                'correspondence_dino_similarity_loss_stat': details['correspondence_dino_similarity'],
                'correspondence_dino_weight_loss_stat': details['correspondence_dino_weight'],
            }
        )
        return loss, details

# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

class Pi3Loss(nn.Module):
    def __init__(
        self,
        train_conf=False,
        correspondence_weight: float = 0.0,
        correspondence_patch_size: int = 14,
        correspondence_max_reference_per_query: int = 1,
        correspondence_reference_selection: str = "uniform",
        correspondence_max_pairs: int = 512,
        correspondence_pair_subsample: str = "uniform",
        correspondence_depth_abs_tol: float = 0.01,
        correspondence_depth_rel_tol: float = 0.05,
        correspondence_huber_beta: float = 0.01,
        correspondence_use_dino_weights: bool = False,
        correspondence_dino_layer: int = 17,
        correspondence_dino_center: float = 0.2,
        correspondence_dino_temperature: float = 0.1,
        correspondence_min_dino_weight: float = 0.05,
    ):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf)
        self.camera_loss = CameraLoss()
        self.correspondence_weight = float(correspondence_weight)
        self.correspondence_loss = CorrespondenceConsistencyLoss(
            patch_size=correspondence_patch_size,
            max_reference_per_query=correspondence_max_reference_per_query,
            reference_selection=correspondence_reference_selection,
            max_pairs=correspondence_max_pairs,
            pair_subsample=correspondence_pair_subsample,
            depth_abs_tol=correspondence_depth_abs_tol,
            depth_rel_tol=correspondence_depth_rel_tol,
            huber_beta=correspondence_huber_beta,
            use_dino_weights=correspondence_use_dino_weights,
            dino_layer=correspondence_dino_layer,
            dino_center=correspondence_dino_center,
            dino_temperature=correspondence_dino_temperature,
            min_dino_weight=correspondence_min_dino_weight,
        )

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)

        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        
        dataset_names = gt[0]['dataset']

        return dict(
            imgs = torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            dataset_names=dataset_names
        )
    
    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def forward(self, pred, gt_raw):
        gt = self.prepare_gt(gt_raw)
        pred = self.normalize_pred(pred, gt)

        final_loss = 0.0
        details = dict()

        # Local Point Loss
        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        final_loss += point_loss
        details.update(point_loss_details)

        # Camera Loss
        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        final_loss += camera_loss * 0.1
        details.update(camera_loss_details)

        if self.correspondence_weight > 0:
            correspondence_eligible = batch_supports_capability(
                gt_raw, ObservationCapability.CORRESPONDENCE
            )
            if correspondence_eligible:
                correspondence_loss, correspondence_details = self.correspondence_loss(pred, gt_raw)
            else:
                correspondence_loss = final_loss * 0.0
                correspondence_details = {
                    key: correspondence_loss.detach()
                    for key in (
                        'correspondence_num_pairs',
                        'correspondence_depth_error',
                        'correspondence_dino_similarity',
                        'correspondence_dino_weight',
                        'correspondence_num_pairs_loss_stat',
                        'correspondence_depth_error_loss_stat',
                        'correspondence_dino_similarity_loss_stat',
                        'correspondence_dino_weight_loss_stat',
                    )
                }
            final_loss += correspondence_loss * self.correspondence_weight
            details['correspondence_loss'] = correspondence_loss
            details['correspondence_weighted_loss'] = correspondence_loss * self.correspondence_weight
            details['correspondence_eligible_batch_loss_stat'] = torch.as_tensor(
                float(correspondence_eligible), device=correspondence_loss.device
            )
            details['correspondence_skipped_batch_loss_stat'] = torch.as_tensor(
                float(not correspondence_eligible), device=correspondence_loss.device
            )
            details.update(correspondence_details)

        return final_loss, details
