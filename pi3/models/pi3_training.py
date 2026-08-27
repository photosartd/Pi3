import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp, PatchEmbed
from ..utils.geometry import homogenize_points
from .ray_conditioning import intrinsics_to_ray_map
from .depth_conditioning import FactoredMetricDepthConditioner
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import (
    dinov2_vitb14_reg,
    dinov2_vitl14_reg,
    dinov2_vits14_reg,
)
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file

def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


_ENCODER_FACTORIES = {
    "small": (dinov2_vits14_reg, 384),
    "base": (dinov2_vitb14_reg, 768),
    "large": (dinov2_vitl14_reg, 1024),
}

_DECODER_SPECS = {
    "small": dict(dec_embed_dim=384, dec_num_heads=6, mlp_ratio=4, dec_depth=24),
    "base": dict(dec_embed_dim=768, dec_num_heads=12, mlp_ratio=4, dec_depth=24),
    "large": dict(dec_embed_dim=1024, dec_num_heads=16, mlp_ratio=4, dec_depth=36),
}


def _is_none_like(value):
    return value is None or str(value).lower() in {"none", "null", ""}


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "state_dict", "teacher", "student"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _strip_prefix_if_present(state_dict, prefixes):
    keys = list(state_dict.keys())
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in keys):
            return {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
    return state_dict


def _positive_int(name, value):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

class Pi3(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            encoder_size='large',
            encoder_pretrained=False,
            encoder_ckpt=None,
            decoder_size='large',
            head_dim=None,
            point_decoder_dim=1024,
            point_decoder_heads=16,
            point_decoder_depth=5,
            camera_decoder_dim=1024,
            camera_decoder_heads=16,
            camera_decoder_depth=5,
            camera_head_dim=512,
            global_point_decoder_dim=None,
            global_point_decoder_heads=None,
            global_point_decoder_depth=5,
            load_vggt=True,
            freeze_encoder=True,
            use_global_points=False,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            dino_output_layers=None,
            use_ray_conditioning=False,
            use_visibility_mask_conditioning=False,
            visibility_mask_conditioning_alpha=1.0,
            use_metric_depth_conditioning=False,
            metric_depth_conditioning_alpha=1.0,
            metric_depth_scale_hidden_dim=128,
            metric_depth_scale_statistic="mean",
            metric_depth_min_valid_pixels=64,
            metric_depth_unit_m=1.0,
            metric_depth_reference_probability=1.0,
            metric_depth_query_probability=0.0,
            metric_depth_eval_reference_probability=1.0,
            metric_depth_eval_query_probability=0.0,
            metric_depth_dropout_granularity="view",
            ckpt=None,
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        encoder_size = str(encoder_size).lower()
        if encoder_size not in _ENCODER_FACTORIES:
            raise ValueError(
                f"Unsupported encoder_size={encoder_size!r}; "
                f"expected one of {sorted(_ENCODER_FACTORIES)}"
            )
        encoder_factory, expected_encoder_dim = _ENCODER_FACTORIES[encoder_size]
        encoder_ckpt = None if _is_none_like(encoder_ckpt) else str(encoder_ckpt)
        if load_vggt and (bool(encoder_pretrained) or encoder_ckpt is not None):
            raise ValueError(
                "load_vggt and encoder_pretrained/encoder_ckpt are mutually "
                "exclusive because they both initialize the DINO encoder"
            )
        self.encoder = encoder_factory(
            pretrained=bool(encoder_pretrained) and encoder_ckpt is None
        )
        self.patch_size = 14
        del self.encoder.mask_token
        enc_embed_dim = int(getattr(self.encoder, "embed_dim", expected_encoder_dim))
        if enc_embed_dim != expected_encoder_dim:
            raise RuntimeError(
                f"encoder_size={encoder_size!r} expected dim "
                f"{expected_encoder_dim}, got {enc_embed_dim}"
            )
        if encoder_ckpt is not None:
            checkpoint = torch.load(
                encoder_ckpt,
                weights_only=False,
                map_location=torch.device("cpu"),
            )
            encoder_state = _strip_prefix_if_present(
                _unwrap_state_dict(checkpoint),
                ("encoder.", "module.encoder.", "backbone.", "module.backbone."),
            )
            print(
                "Loading DINO encoder",
                self.encoder.load_state_dict(encoder_state, strict=False),
            )
            del checkpoint
        self.dino_output_layers = [] if dino_output_layers is None else [int(layer) for layer in dino_output_layers]

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        decoder_size = str(decoder_size).lower()
        if decoder_size not in _DECODER_SPECS:
            raise ValueError(
                f"Unsupported decoder_size={decoder_size!r}; "
                f"expected one of {sorted(_DECODER_SPECS)}"
            )
        decoder_spec = _DECODER_SPECS[decoder_size]
        dec_embed_dim = int(decoder_spec["dec_embed_dim"])
        dec_num_heads = int(decoder_spec["dec_num_heads"])
        mlp_ratio = int(decoder_spec["mlp_ratio"])
        dec_depth = int(decoder_spec["dec_depth"])
        if enc_embed_dim != dec_embed_dim:
            raise ValueError(
                "encoder and decoder dimensions must match unless an explicit "
                f"projection is added; got encoder_dim={enc_embed_dim}, "
                f"decoder_dim={dec_embed_dim}"
            )
        if dec_embed_dim % dec_num_heads != 0:
            raise ValueError(
                f"decoder dim {dec_embed_dim} must be divisible by "
                f"decoder heads {dec_num_heads}"
            )

        if head_dim is not None and point_decoder_dim is not None:
            if int(head_dim) != int(point_decoder_dim):
                raise ValueError(
                    "head_dim is a legacy alias for point_decoder_dim; "
                    f"got head_dim={head_dim} and "
                    f"point_decoder_dim={point_decoder_dim}"
                )
        if point_decoder_dim is None:
            point_decoder_dim = head_dim if head_dim is not None else 1024
        point_decoder_dim = _positive_int("point_decoder_dim", point_decoder_dim)
        point_decoder_heads = _positive_int("point_decoder_heads", point_decoder_heads)
        point_decoder_depth = _positive_int("point_decoder_depth", point_decoder_depth)
        camera_decoder_dim = _positive_int("camera_decoder_dim", camera_decoder_dim)
        camera_decoder_heads = _positive_int("camera_decoder_heads", camera_decoder_heads)
        camera_decoder_depth = _positive_int("camera_decoder_depth", camera_decoder_depth)
        camera_head_dim = _positive_int("camera_head_dim", camera_head_dim)
        if global_point_decoder_dim is None:
            global_point_decoder_dim = point_decoder_dim
        if global_point_decoder_heads is None:
            global_point_decoder_heads = point_decoder_heads
        global_point_decoder_dim = _positive_int("global_point_decoder_dim", global_point_decoder_dim)
        global_point_decoder_heads = _positive_int("global_point_decoder_heads", global_point_decoder_heads)
        global_point_decoder_depth = _positive_int("global_point_decoder_depth", global_point_decoder_depth)

        for name, dim, heads in (
            ("point_decoder", point_decoder_dim, point_decoder_heads),
            ("camera_decoder", camera_decoder_dim, camera_decoder_heads),
            ("global_point_decoder", global_point_decoder_dim, global_point_decoder_heads),
        ):
            if dim % heads != 0:
                raise ValueError(
                    f"{name} dim {dim} must be divisible by heads {heads}"
                )
        if load_vggt and (
            encoder_size != "large"
            or decoder_size != "large"
            or point_decoder_dim != 1024
            or point_decoder_heads != 16
            or camera_decoder_dim != 1024
            or camera_decoder_heads != 16
            or camera_head_dim != 512
        ):
            raise ValueError(
                "load_vggt=true is only compatible with the historical large "
                "Pi3 dimensions"
            )
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #  Camera Ray Condition
        # ----------------------
        self.use_ray_conditioning = bool(use_ray_conditioning)
        if self.use_ray_conditioning:
            # This is the Pi3X conditioning mechanism: a full-resolution
            # (x/z, y/z) ray map is projected with the same 14x14 patch layout
            # as the RGB encoder, then added to the RGB patch tokens. The
            # projection is zero-initialized so a Pi3 checkpoint has identical
            # initial behavior before the first optimizer update.
            self.ray_embed = PatchEmbed(
                img_size=224,
                patch_size=self.patch_size,
                in_chans=2,
                embed_dim=self.dec_embed_dim,
            )
            nn.init.zeros_(self.ray_embed.proj.weight)
            nn.init.zeros_(self.ray_embed.proj.bias)

        # ----------------------
        #  Visibility Mask Condition
        # ----------------------
        self.use_visibility_mask_conditioning = bool(use_visibility_mask_conditioning)
        self.visibility_mask_conditioning_alpha = float(visibility_mask_conditioning_alpha)
        if self.use_visibility_mask_conditioning:
            # Two channels mirror sparse-depth conditioning conventions:
            # channel 0 stores the target-object visibility mask, channel 1
            # says whether that mask is intentionally supplied. Unknown views
            # are [0, 0]. The zero init preserves exact Pi3 checkpoint behavior
            # at step 0 while gradients flow from known conditions immediately.
            self.visibility_mask_embed = PatchEmbed(
                img_size=224,
                patch_size=self.patch_size,
                in_chans=2,
                embed_dim=self.dec_embed_dim,
            )
            nn.init.zeros_(self.visibility_mask_embed.proj.weight)
            nn.init.zeros_(self.visibility_mask_embed.proj.bias)

        # ----------------------
        #  Metric Depth Condition
        # ----------------------
        self.use_metric_depth_conditioning = bool(use_metric_depth_conditioning)
        self.metric_depth_conditioning_alpha = float(
            metric_depth_conditioning_alpha
        )
        if self.use_metric_depth_conditioning:
            self.metric_depth_conditioner = FactoredMetricDepthConditioner(
                embed_dim=self.dec_embed_dim,
                patch_size=self.patch_size,
                scale_hidden_dim=metric_depth_scale_hidden_dim,
                scale_statistic=metric_depth_scale_statistic,
                min_valid_pixels=metric_depth_min_valid_pixels,
                metric_unit_m=metric_depth_unit_m,
                reference_probability=metric_depth_reference_probability,
                query_probability=metric_depth_query_probability,
                eval_reference_probability=metric_depth_eval_reference_probability,
                eval_query_probability=metric_depth_eval_query_probability,
                dropout_granularity=metric_depth_dropout_granularity,
            )

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=point_decoder_dim,
            dec_num_heads=point_decoder_heads,
            depth=point_decoder_depth,
            out_dim=point_decoder_dim,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=point_decoder_dim, output_dim=3)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=camera_decoder_dim,
            dec_num_heads=camera_decoder_heads,
            depth=camera_decoder_depth,
            out_dim=camera_head_dim,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=camera_head_dim)
        

        # ----------------------
        #  Global Points Decoder
        # ----------------------
        self.use_global_points = use_global_points
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=global_point_decoder_dim,
                dec_num_heads=global_point_decoder_heads,
                depth=global_point_decoder_depth,
                out_dim=global_point_decoder_dim,
                rope=self.rope,
            )
            self.global_point_head = LinearPts3d(patch_size=14, dec_embed_dim=global_point_decoder_dim, output_dim=3)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        if load_vggt:
            vggt_weight = load_file('ckpts/VGGT-1B/model.safetensors')
            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

            vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
            vggt_dec_weight1 = {}
            for k in list(vggt_dec_weight.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight1[f'{int(idx)*2 + 1}{other}'] = vggt_dec_weight[k]
            vggt_dec_weight = vggt_dec_weight1 

            vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
            for k in list(vggt_dec_weight_frame.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight[f'{int(idx)*2}{other}'] = vggt_dec_weight_frame[k]

            print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))

        self.train_conf = train_conf
        if train_conf:
            assert ckpt is not None

            # ----------------------
            #     Conf Decoder
            # ----------------------
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=point_decoder_dim, output_dim=1)

            freeze_all_params([self.encoder, self.decoder, self.point_decoder, self.point_head, self.camera_decoder,  self.camera_head, self.register_token])
            if use_global_points:
                freeze_all_params([self.global_points_decoder, self.global_point_head])

        if freeze_encoder:
            print('Freezing the encoder.')
            freeze_all_params([self.encoder])

        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        ckpt = None if _is_none_like(ckpt) else ckpt
        if ckpt is not None:
            if str(ckpt).endswith(".safetensors"):
                checkpoint = load_file(ckpt)
            else:
                checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')
                checkpoint = _unwrap_state_dict(checkpoint)

            res = self.load_state_dict(checkpoint, strict=False)
            print(f'[Pi3] Load checkpoints from {ckpt}: {res}')

            del checkpoint
            torch.cuda.empty_cache()

    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def _encode_with_optional_dino_outputs(self, imgs):
        if not self.dino_output_layers:
            hidden = self.encoder(imgs, is_training=True)
            if isinstance(hidden, dict):
                hidden = hidden["x_norm_patchtokens"]
            return hidden, {}

        final_layer = len(self.encoder.blocks) - 1
        layers = sorted(set(self.dino_output_layers + [final_layer]))
        outputs = self.encoder.get_intermediate_layers(imgs, n=layers, norm=True)
        layer_to_tokens = {int(layer): output for layer, output in zip(layers, outputs)}
        hidden = layer_to_tokens[final_layer]
        dino_features = {
            str(layer): layer_to_tokens[int(layer)].detach()
            for layer in self.dino_output_layers
        }
        return hidden, dino_features

    def forward(
        self,
        imgs,
        intrinsics=None,
        visibility_mask_condition=None,
        visibility_mask_known=None,
        metric_depth=None,
        metric_depth_valid=None,
        metric_depth_is_reference=None,
        metric_depth_is_query=None,
        metric_depth_known_override=None,
    ):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden, dino_features = self._encode_with_optional_dino_outputs(imgs)

        if self.use_ray_conditioning:
            if intrinsics is None:
                raise ValueError("intrinsics are required when ray conditioning is enabled")
            if tuple(intrinsics.shape[:2]) != (B, N):
                raise ValueError(
                    "Expected intrinsics with leading shape "
                    f"{(B, N)}, got {tuple(intrinsics.shape)}"
                )
            ray_map = intrinsics_to_ray_map(intrinsics, H, W)
            ray_tokens = self.ray_embed(
                ray_map.reshape(B * N, H, W, 2).permute(0, 3, 1, 2)
            )
            if ray_tokens.shape != hidden.shape:
                raise RuntimeError(
                    "Ray/image token shape mismatch: "
                    f"{tuple(ray_tokens.shape)} vs {tuple(hidden.shape)}"
            )
            hidden = hidden + ray_tokens.to(dtype=hidden.dtype)

        visibility_mask_stats = {}
        if self.use_visibility_mask_conditioning:
            if visibility_mask_condition is None:
                raise ValueError(
                    "visibility_mask_condition is required when "
                    "visibility mask conditioning is enabled"
                )
            if tuple(visibility_mask_condition.shape[:2]) != (B, N):
                raise ValueError(
                    "Expected visibility_mask_condition with leading shape "
                    f"{(B, N)}, got {tuple(visibility_mask_condition.shape)}"
                )
            if visibility_mask_known is None:
                visibility_mask_known = torch.zeros_like(visibility_mask_condition)
            if visibility_mask_known.shape != visibility_mask_condition.shape:
                raise ValueError(
                    "visibility_mask_known must match visibility_mask_condition, got "
                    f"{tuple(visibility_mask_known.shape)} vs "
                    f"{tuple(visibility_mask_condition.shape)}"
                )

            mask_input = torch.stack(
                [
                    visibility_mask_condition.to(device=hidden.device, dtype=torch.float32),
                    visibility_mask_known.to(device=hidden.device, dtype=torch.float32),
                ],
                dim=2,
            ).reshape(B * N, 2, H, W)
            mask_tokens = self.visibility_mask_embed(mask_input)
            if mask_tokens.shape != hidden.shape:
                raise RuntimeError(
                    "Visibility mask/image token shape mismatch: "
                    f"{tuple(mask_tokens.shape)} vs {tuple(hidden.shape)}"
                )
            # A condition with known=0 is absent, not a learned modality token.
            # Gating also suppresses the projection bias after it has trained.
            mask_tokens, known_view = self._gate_visibility_mask_tokens(
                mask_tokens, visibility_mask_known
            )
            alpha = torch.as_tensor(
                self.visibility_mask_conditioning_alpha,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            hidden = hidden + alpha * mask_tokens.to(dtype=hidden.dtype)
            visibility_mask_stats = {
                "visibility_mask_conditioning_alpha": alpha.detach().float(),
                "visibility_mask_token_abs_mean": mask_tokens.detach().float().abs().mean(),
                "visibility_mask_token_norm": mask_tokens.detach().float().norm(),
                "visibility_mask_embed_weight_norm": self.visibility_mask_embed.proj.weight.detach().float().norm(),
                "visibility_mask_conditioned_view_fraction": known_view.detach().float().mean(),
            }

        metric_depth_output = None
        if self.use_metric_depth_conditioning:
            required = {
                "metric_depth": metric_depth,
                "metric_depth_valid": metric_depth_valid,
                "metric_depth_is_reference": metric_depth_is_reference,
                "metric_depth_is_query": metric_depth_is_query,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "Metric depth conditioning requires " + ", ".join(missing)
                )
            metric_depth_output = self.metric_depth_conditioner(
                depth_m=metric_depth.to(device=hidden.device),
                valid_mask=metric_depth_valid.to(device=hidden.device),
                is_reference=metric_depth_is_reference.to(device=hidden.device),
                is_query=metric_depth_is_query.to(device=hidden.device),
                known_override=(
                    None
                    if metric_depth_known_override is None
                    else metric_depth_known_override.to(device=hidden.device)
                ),
            )
            if metric_depth_output.tokens.shape != hidden.shape:
                raise RuntimeError(
                    "Metric depth/image token shape mismatch: "
                    f"{tuple(metric_depth_output.tokens.shape)} vs "
                    f"{tuple(hidden.shape)}"
                )
            depth_alpha = torch.as_tensor(
                self.metric_depth_conditioning_alpha,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            hidden = hidden + depth_alpha * metric_depth_output.tokens.to(
                dtype=hidden.dtype
            )
            metric_depth_output.statistics[
                "metric_depth_conditioning_alpha"
            ] = depth_alpha.detach().float()

        hidden, pos = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        if self.use_global_points:
            context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                conf = None
                
            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # Global points
            if self.use_global_points:
                global_point_hidden = global_point_hidden.float()
                global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                global_points = None
            
            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        output = dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            global_points=global_points
        )
        if dino_features:
            output["dino_features"] = {
                layer: features.reshape(B, N, patch_h * patch_w, -1)
                for layer, features in dino_features.items()
            }
        if visibility_mask_stats:
            output["visibility_mask_conditioning_stats"] = visibility_mask_stats
        if metric_depth_output is not None:
            output["metric_depth_conditioning_stats"] = (
                metric_depth_output.statistics
            )
            output["metric_depth_conditioning_known"] = (
                metric_depth_output.known.detach()
            )
            output["metric_depth_conditioning_scale_m"] = (
                metric_depth_output.scale_m.detach()
            )
            output["metric_depth_conditioning_valid_pixels"] = (
                metric_depth_output.valid_pixels.detach()
            )
        return output

    @staticmethod
    def _gate_visibility_mask_tokens(mask_tokens, visibility_mask_known):
        """Make an unknown ``[mask=0, known=0]`` condition an exact no-op."""

        known_view = visibility_mask_known.to(
            device=mask_tokens.device, dtype=mask_tokens.dtype
        ).amax(dim=(-2, -1)).reshape(mask_tokens.shape[0], 1, 1)
        return mask_tokens * known_view, known_view
