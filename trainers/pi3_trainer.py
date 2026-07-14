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
            other_params = [
                (name, param) for name, param in model_.named_parameters()
                if not name.startswith("encoder.") and not '.encoder.' in name
            ]

            print(f'Number of trainable encoder parameters:', sum(p.numel() for _, p in encoder_params if p.requires_grad))
            print(f'Length of trainable others:', sum(p.numel() for _, p in other_params if p.requires_grad))

            def handle_weight_decay(params, weight_decay, lr):
                decay = []
                no_decay = []
                for name, param in params:
                    if not param.requires_grad:
                        continue

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                return [
                    {"params": no_decay, "weight_decay": 0.0, 'lr': lr},
                    {"params": decay, "weight_decay": weight_decay, 'lr': lr},
                ]

            res = []
            res.extend(handle_weight_decay(encoder_params, cfg_optimizer.weight_decay, cfg_optimizer.encoder_lr))
            res.extend(handle_weight_decay(other_params, cfg_optimizer.weight_decay, cfg_optimizer.lr))

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
        pred = self.model(imgs)

        return [pred, batch]
    
    def calculate_loss(self, output, batch, mode='train'):
        output, batch = output

        if mode == 'train':
            loss, details = self.train_loss(output, batch)
        else:
            loss, details = self.test_loss(output, batch)

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
