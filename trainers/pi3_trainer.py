from trainers.base_trainer_accelerate import BaseTrainer
from easydict import EasyDict
import torch
from datasets.base.base_dataset import sample_resolutions
import hydra

from pi3.models.loss import Pi3Loss

class Pi3Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_loss = hydra.utils.instantiate(cfg.loss.train_loss)
        self.test_loss = hydra.utils.instantiate(cfg.loss.test_loss)

    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            encoder_params = [param for param in model_.encoder.named_parameters()]
            ray_params = [
                (name, param) for name, param in model_.named_parameters()
                if name.startswith("ray_embed.")
            ]
            visibility_mask_params = [
                (name, param) for name, param in model_.named_parameters()
                if name.startswith("visibility_mask_embed.")
            ]
            other_params = [
                (name, param) for name, param in model_.named_parameters()
                if (
                    not name.startswith("encoder.")
                    and '.encoder.' not in name
                    and not name.startswith("ray_embed.")
                    and not name.startswith("visibility_mask_embed.")
                )
            ]

            print(f'Number of trainable encoder parameters:', sum(p.numel() for _, p in encoder_params if p.requires_grad))
            print(f'Number of trainable ray parameters:', sum(p.numel() for _, p in ray_params if p.requires_grad))
            print(f'Number of trainable visibility mask parameters:', sum(p.numel() for _, p in visibility_mask_params if p.requires_grad))
            print(f'Length of trainable others:', sum(p.numel() for _, p in other_params if p.requires_grad))

            def handle_weight_decay(params, weight_decay, lr, group_name):
                decay = []
                no_decay = []
                for name, param in params:
                    if not param.requires_grad:
                        continue

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                groups = []
                for suffix, values, decay_value in (
                    ("no_decay", no_decay, 0.0),
                    ("decay", decay, weight_decay),
                ):
                    if not values:
                        continue
                    group = {
                        "params": values,
                        "weight_decay": decay_value,
                        "lr": lr,
                        "group_name": group_name,
                    }
                    if group_name in {"ray", "visibility_mask"}:
                        # The OneCycle scheduler otherwise caps every group at
                        # the base decoder LR. Preserve explicitly selected
                        # modality-branch peak LRs without changing historical
                        # groups.
                        group["max_lr"] = lr
                    groups.append(group)
                return groups

            res = []
            res.extend(handle_weight_decay(
                encoder_params,
                cfg_optimizer.weight_decay,
                cfg_optimizer.encoder_lr,
                "encoder",
            ))
            if ray_params:
                ray_lr = cfg_optimizer.get("ray_lr", cfg_optimizer.lr)
                res.extend(handle_weight_decay(
                    ray_params,
                    cfg_optimizer.weight_decay,
                    ray_lr,
                    "ray",
                ))
            if visibility_mask_params:
                visibility_mask_lr = cfg_optimizer.get("visibility_mask_lr", cfg_optimizer.lr)
                res.extend(handle_weight_decay(
                    visibility_mask_params,
                    cfg_optimizer.weight_decay,
                    visibility_mask_lr,
                    "visibility_mask",
                ))
            res.extend(handle_weight_decay(
                other_params,
                cfg_optimizer.weight_decay,
                cfg_optimizer.lr,
                "other",
            ))

            return res
        
        return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def before_epoch(self, epoch):
        self._set_loader_epoch(self.train_loader, epoch)

        for loader in getattr(self, "val_loaders", {"default": self.test_loader}).values():
            self._set_loader_epoch(loader, epoch)

        if 'random_reslution' in self.cfg.train and self.cfg.train.random_reslution and self.cfg.train.num_resolution > 0:
            seed = epoch + self.cfg.train.base_seed
            base_resolution = self.cfg.train.base_resolution if 'base_resolution' in self.cfg.train else []
            resolutions = sample_resolutions(
                aspect_ratio_range=self.cfg.train.aspect_ratio_range,
                pixel_count_range=self.cfg.train.pixel_count_range,
                patch_size=self.cfg.train.patch_size,
                num_resolutions=self.cfg.train.num_resolution,
                seed=seed,
                base_resolution=base_resolution,
            )
            print('[Pi3 Trainer] Sampled new resolutions:', resolutions)
            datasets = []
            recursive_get_dataset(self.train_loader.dataset, datasets)
            for dataset in datasets:
                dataset._set_resolutions(resolutions)

    def _set_loader_epoch(self, loader, epoch):
        if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'set_epoch'):
            loader.dataset.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(loader, 'batch_sampler') and hasattr(loader.batch_sampler, 'batch_sampler') and hasattr(loader.batch_sampler.batch_sampler, 'sampler') and hasattr(loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(loader, 'batch_sampler') and hasattr(loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
            
    def forward_batch(self, batch, mode='train'):
        imgs = torch.stack([view['img'] for view in batch], dim=1)
        intrinsics = torch.stack([view['camera_intrinsics'] for view in batch], dim=1)
        model_kwargs = {"intrinsics": intrinsics}
        if bool(self.cfg.model.get("use_visibility_mask_conditioning", False)):
            if "visibility_mask_condition" not in batch[0] or "visibility_mask_known" not in batch[0]:
                raise ValueError(
                    "model.use_visibility_mask_conditioning=true requires "
                    "visibility_mask_condition and visibility_mask_known in the batch"
                )
            visibility_mask_condition = torch.stack(
                [view["visibility_mask_condition"] for view in batch],
                dim=1,
            )
            visibility_mask_known = torch.stack(
                [view["visibility_mask_known"] for view in batch],
                dim=1,
            )
            model_kwargs["visibility_mask_condition"] = visibility_mask_condition
            model_kwargs["visibility_mask_known"] = visibility_mask_known

        pred = self.model(imgs, **model_kwargs)

        if bool(self.cfg.model.get("use_visibility_mask_conditioning", False)):
            ref_mask = torch.stack([view["is_reference"] for view in batch], dim=1).bool()
            query_mask = torch.stack([view["is_query"] for view in batch], dim=1).bool()
            condition = model_kwargs["visibility_mask_condition"].detach().float()
            known = model_kwargs["visibility_mask_known"].detach().float()

            def role_mean(values, role):
                if not bool(role.any()):
                    return values.mean() * 0.0
                expanded = role[..., None, None].expand_as(values)
                return values[expanded].mean()

            stats = pred.setdefault("visibility_mask_conditioning_stats", {})
            stats.update(
                {
                    "visibility_mask_condition_area": condition.mean(),
                    "visibility_mask_known_area": known.mean(),
                    "visibility_mask_reference_condition_area": role_mean(condition, ref_mask),
                    "visibility_mask_reference_known_area": role_mean(known, ref_mask),
                    "visibility_mask_query_condition_area": role_mean(condition, query_mask),
                    "visibility_mask_query_known_area": role_mean(known, query_mask),
                }
            )

        return [pred, batch]
    
    def calculate_loss(self, output, batch, mode='train'):
        output, batch = output

        if mode == 'train':
            loss, details = self.train_loss(output, batch)
        else:
            loss, details = self.test_loss(output, batch)

        conditioning_stats = output.get("visibility_mask_conditioning_stats", {})
        for key, value in conditioning_stats.items():
            if torch.is_tensor(value):
                details[f"{key}_loss_stat"] = value.detach()

        return EasyDict(
            loss=loss,
            **details
        )


def recursive_get_dataset(dataset, res=[]):
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            recursive_get_dataset(ds, res)
    else:
        if hasattr(dataset, 'dataset'):
            res.append(dataset.dataset)
    return res
