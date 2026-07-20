import argparse
import datetime
import itertools
import os
import random
import traceback
from accelerate import Accelerator
import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import yaml
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from copy import deepcopy
from easydict import EasyDict
import time
import json
import math
import sys
from PIL import Image
import shutil
from utils.basic import seed_anything, count_parameters

from datasets import create_dataloader
from pi3.metrics import MetricManager
from pi3.visualizations import VisualManager
# from model.network import Network
from utils.misc import get_logger, is_logging_process, pretty_print_hydra_config, move_to_device, get_rank
from utils.basic import seed_anything
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.dist import (
    MetricLogger,
    SmoothedValue,
    init_distributed_mode,
    setup_for_distributed,
)
from accelerate import DistributedDataParallelKwargs
from transformers.trainer_pt_utils import get_model_param_count
from accelerate import (
    DistributedType,
)
from accelerate.utils import (
    DataLoaderConfiguration,
    DynamoBackend,
    GradientAccumulationPlugin,
    ProjectConfiguration,
    TorchDynamoPlugin,
    set_seed,
)
import numpy as np

class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        with open_dict(cfg):
            cfg.job_logging_cfg = HydraConfig.get().job_logging

        # random seed
        if cfg.random_seed is None:
            cfg.random_seed = random.randint(1, 10000)
        seed_anything(cfg.random_seed, deterministic=False)              # deterministic=True for reproduction

        ## 1. Build accelerator
        self.build_accelerator()

        if is_logging_process():
            pretty_print_hydra_config(cfg)

        ## 2. Prepare model
        self.log_info("Preparing model...")
        self.model = self.prepare_model()
        self.metric_manager = self.prepare_metric_manager()
        self.visual_manager = self.prepare_visual_manager()
        self.n_learnable_parameters = get_model_param_count(
            self.model, trainable_only=True
        )
        self.n_fix_parameters = get_model_param_count(
            self.model, trainable_only=False
        )
        self.accelerator.wait_for_everyone()

        ## 3. Prepare dataloader
        self.log_info("Making train dataloader...")
        self.train_loader = create_dataloader(cfg, 'train')
        self.log_info("Making validation dataloader(s)...")
        self.val_loaders, self.val_runtime_cfgs = self.prepare_val_loaders()
        self.test_loader = next(iter(self.val_loaders.values()))
        self.accelerator.wait_for_everyone()

        ## 5. Prepare optimizer and scheduler (fsdp should after preparing the model using accelerate)
        if self.cfg.get("fsdp_plugin"):
            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")
        else:
            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")

            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

        # Create the LR scheduler
        self.iters_per_epoch = self.cfg.train.iters_per_epoch if self.cfg.train.iters_per_epoch > 0 else len(self.train_loader)
        self.iters_per_val = {
            name: (
                runtime_cfg.iters_per_test
                if "iters_per_test" in runtime_cfg and runtime_cfg.iters_per_test > 0
                else len(loader)
            )
            for name, loader, runtime_cfg in (
                (name, self.val_loaders[name], self.val_runtime_cfgs[name])
                for name in self.val_loaders
            )
        }
        self.iters_per_test = next(iter(self.iters_per_val.values()))
        self.primary_val_name = str(self.cfg.get("primary_val", next(iter(self.val_loaders.keys()))))
        if self.primary_val_name not in self.val_loaders:
            raise ValueError(f"primary_val={self.primary_val_name!r} is not in val loaders {list(self.val_loaders)}")
        self.latest_primary_val_stats = {}
        self.cfg.train.lr_scheduler.total_steps = self.cfg.train.num_epoch * self.iters_per_epoch
        self.log_info(f"Total step for lr scheduler: {self.cfg.train.lr_scheduler.total_steps} ({self.cfg.train.num_epoch} * {self.iters_per_epoch})")
        self.lr_scheduler = build_scheduler(
            self.cfg.train.lr_scheduler, optimizer=self.optimizer
        )
        self.log_info(f"LRScheduler: {self.lr_scheduler}")

        ## 6. Prepare accelerate training
        self.prepare_training()

    def build_optimizer(self, cfg_optimizer, model, param_group_fn=None):
        return build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def prepare_training(self):
        # report model details
        self.log_info(
            f"total number of learnable params: {self.n_learnable_parameters / 1e6} M"
        )
        self.log_info(
            f"total number of fixed params: {self.n_fix_parameters / 1e6} M"
        )

        # Wrap the model, optmizer, and scheduler with accelerate
        self.log_info("before accelerator.prepare")

        # (
        #     self.model,
        #     self.train_loader,
        #     self.test_loader,
        #     self.optimizer,
        #     self.lr_scheduler,
        # ) = self.accelerator.prepare(
        #     self.model, self.train_loader, self.test_loader, self.optimizer, self.lr_scheduler
        # )

        # don't wrap dataloader
        (
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(os.path.basename(self.cfg.log.output_dir))

        # Report the training info
        self.total_batch_size = (
            self.cfg.train.batch_size
            * self.accelerator.num_processes
            * self.cfg.train.gradient_accumulation_steps
        )
        self.log_info("***** Running training *****")
        self.log_info(f"LR = {self.cfg.train.optimizer.lr:.8f}")
        self.log_info(f"Weigth Decay = {self.cfg.train.optimizer.weight_decay:.8f}")
        self.log_info(f"Instantaneous batch size per device = {self.cfg.train.batch_size}")
        self.log_info(f"Total Batch size = {self.total_batch_size}")
        self.log_info(
            f"Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}"
        )
        self.log_info(f"Number of epochs = {self.cfg.train.num_epoch}")
        self.log_info(
            f"Number of training steps per epoch = {self.iters_per_epoch}"
        )
        self.log_info(
            f"Number of total training steps = {self.iters_per_epoch * self.cfg.train.num_epoch}"
        )
        # self.log_info(f"Number of training examples per epoch = {len(self.dataloader.dataset)}")
        self.log_info(
            f"Number of model parameters = {self.n_fix_parameters / 1e6:.2f}M"
        )
        self.log_info(
            f"Number of model trainable parameters = {self.n_learnable_parameters / 1e6:.2f}M"
        )

        # Auto resume the checkpoint
        latest_epoch = self.auto_resume()
        self.initial_global_step = self.iters_per_epoch * latest_epoch
        self.first_epoch = latest_epoch

        os.makedirs(self.cfg.log.ckpt_dir, exist_ok=True)

    def prepare_model(self):
        model = hydra.utils.instantiate(self.cfg.model)
        count_parameters(model)
        return model

    def prepare_metric_manager(self):
        return MetricManager.from_config(self.cfg.get("metrics"))

    def prepare_visual_manager(self):
        return VisualManager.from_config(self.cfg.get("visuals"))

    def prepare_val_loaders(self):
        """Build one or more named validation dataloaders.

        Existing configs with ``test_dataset`` keep the historical single
        loader. New configs can define ``val_datasets.<name>`` entries with
        ``dataset``, optional ``dataloader``, and optional ``runtime`` sections.
        """

        val_cfgs = self.cfg.get("val_datasets")
        if not val_cfgs:
            return {"default": create_dataloader(self.cfg, "test")}, {"default": self.cfg.test}

        loaders = {}
        runtime_cfgs = {}
        for name, entry in val_cfgs.items():
            if entry is None or not bool(entry.get("enabled", True)):
                continue
            self.log_info(f"Making validation dataloader: {name}")
            runtime_cfg = entry.get("runtime", self.cfg.test)
            loaders[str(name)] = create_dataloader(
                self.cfg,
                "test",
                dataset_cfg=entry.dataset,
                dataloader_cfg=entry.get("dataloader", self.cfg.test_dataloader),
                runtime_cfg=runtime_cfg,
            )
            runtime_cfgs[str(name)] = runtime_cfg
        return loaders, runtime_cfgs
    
    def before_epoch(self, epoch):
        pass

    def train(self):
        # Start Train!
        start_time = time.time()
        self.accelerator.wait_for_everyone()

        # Initialize variable to track the best validation metric
        best_val_metric = float('inf')  # For metrics like loss; use -float('inf') for accuracy
        best_model_path = None

        max_checkpoints = self.cfg.log.max_checkpoints  # Maximum number of recent checkpoints to keep
        saved_checkpoints = []  # List to track saved checkpoint paths

        for epoch in range(self.first_epoch, self.cfg.train.num_epoch):
            torch.cuda.reset_peak_memory_stats()

            self.before_epoch(epoch)

            train_stats = self.train_one_epoch(epoch)

            val_every_n_epochs = max(1, int(self.cfg.train.get("val_every_n_epochs", 1)))
            should_validate = (
                (epoch + 1) % val_every_n_epochs == 0
                or epoch + 1 == self.cfg.train.num_epoch
            )
            if should_validate:
                val_stats = self.validate_all(epoch)
            else:
                val_stats = {}
                self.log_info(
                    f"Skipping validation at epoch {epoch}; "
                    f"train.val_every_n_epochs={val_every_n_epochs}"
                )

            current_val_metric = self.latest_primary_val_stats.get("loss", float('inf'))
            if val_stats and self.cfg.log.get("save_best", True) and current_val_metric < best_val_metric:
                best_val_metric = current_val_metric
                best_model_path = os.path.join(
                    self.cfg.log.ckpt_dir,
                    "best_model",
                )
                self.accelerator.save_state(best_model_path, safe_serialization=False)
                self.log_info(f"Saved best model at epoch {epoch} with val_metric: {best_val_metric:.4f}")

            self.accelerator.wait_for_everyone()

            if self.cfg.log.get("save_checkpoints", True) and (
                (epoch + 1) % self.cfg.log.ckpt_interval == 0
                or epoch + 1 == self.cfg.train.num_epoch
            ):
                if self.accelerator.sync_gradients:
                    self.global_step = self.iters_per_epoch * (epoch + 1)
                    save_path = os.path.join(
                        self.cfg.log.ckpt_dir,
                        f"checkpoint_{epoch}",
                    )
                    self.accelerator.save_state(save_path, safe_serialization=False)
                    self.log_info(
                        f"Saved state for global step {self.global_step}"
                    )

                    # Manage saved checkpoints
                    saved_checkpoints.append(save_path)
                    if self.accelerator.is_main_process and len(saved_checkpoints) > max_checkpoints:
                        oldest_checkpoint = saved_checkpoints.pop(0)
                        if os.path.exists(oldest_checkpoint):
                            shutil.rmtree(oldest_checkpoint)
                            self.log_info(f"Removed old checkpoint: {oldest_checkpoint}")

                self.accelerator.wait_for_everyone()

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": self.n_learnable_parameters,
            }

            if self.accelerator.is_main_process:
                with open(
                    os.path.join(self.cfg.log.ckpt_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

                self.log_all(log_stats, step=self.global_step)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.log_info("Training time {}".format(total_time_str))

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def validate_all(self, epoch):
        """Run all configured validation loaders and namespace their stats."""

        if list(self.val_loaders.keys()) == ["default"]:
            stats = self.validate(
                epoch,
                val_name="default",
                loader=self.test_loader,
                iters_per_test=self.iters_per_test,
                runtime_cfg=self.cfg.test,
            )
            self.latest_primary_val_stats = stats
            return stats

        all_stats = {}
        self.latest_primary_val_stats = {}
        for name, loader in self.val_loaders.items():
            stats = self.validate(
                epoch,
                val_name=name,
                loader=loader,
                iters_per_test=self.iters_per_val[name],
                runtime_cfg=self.val_runtime_cfgs[name],
            )
            if name == self.primary_val_name:
                self.latest_primary_val_stats = stats
            for key, value in stats.items():
                all_stats[f"{name}_{key}"] = value
        return all_stats

    def validate(
        self,
        epoch,
        *,
        val_name="default",
        loader=None,
        iters_per_test=None,
        runtime_cfg=None,
    ):
        self.model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Validation {val_name} Epoch: [{epoch}]"
        loader = self.test_loader if loader is None else loader
        iters_per_test = self.iters_per_test if iters_per_test is None else int(iters_per_test)
        runtime_cfg = self.cfg.test if runtime_cfg is None else runtime_cfg
        print_freq = runtime_cfg.print_freq if "print_freq" in runtime_cfg else self.cfg.train.print_freq

        val_loss = 0.0
        total_samples = 0
        val_visuals_logged = False
        val_visual_it = self.visual_manager.render_iteration(
            "val",
            epoch=epoch,
            length=iters_per_test,
        )
        if self.metric_manager.should_update("val"):
            split_name = self._validation_split_name(loader, val_name)
            self.metric_manager.set_context(
                mode="val",
                val_name=val_name,
                split=split_name,
                epoch=int(epoch),
                global_step=int(self.global_step),
                checkpoint_step=int(self.global_step),
                run_id=str(self.cfg.name),
                output_dir=str(self.cfg.log.output_dir),
                keyframe_seed=int(self.cfg.train.base_seed),
                masked=bool(OmegaConf.select(self.cfg, "lmgeo.reference_rgb_masking", default=False)),
            )
            self.metric_manager.reset()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)

        self.log_info(f"Start validation {val_name} for epoch {epoch}")
        with torch.no_grad():
            val_iterable = itertools.islice(loader, iters_per_test)
            for it, batch in enumerate(metric_logger.log_every(
                val_iterable, print_freq, header, total=iters_per_test
            )):
                move_start = time.time()
                batch = move_to_device(batch, self.accelerator.device)
                move_time = time.time() - move_start

                forward_start = time.time()
                with self.accelerator.autocast():
                    forward_outputs = self.forward_batch(batch, mode='test')
                forward_time = time.time() - forward_start

                visual_time = 0.0
                if (
                    not val_visuals_logged
                    and val_visual_it is not None
                    and it == val_visual_it
                    and self.accelerator.is_main_process
                ):
                    visual_start = time.time()
                    val_visuals = self.visual_manager.render(forward_outputs, batch, mode="val")
                    visual_time = time.time() - visual_start
                    if val_visuals:
                        self.log_all(val_visuals, step=self.global_step, prefix=f"val_visuals/{val_name}")
                        self.log_info(
                            f"Logged randomized {val_name} val visuals at iterator {it}: "
                            f"{self.visual_manager.last_render_info}"
                        )
                    val_visuals_logged = True
                metric_time = 0.0
                if self.metric_manager.should_update("val"):
                    metric_start = time.time()
                    self.metric_manager.update(forward_outputs, batch, mode="val")
                    metric_time = time.time() - metric_start
                metric_timing_stats = {
                    f"metric_{name}": value
                    for name, value in getattr(self.metric_manager, "last_update_times", {}).items()
                }
                loss_start = time.time()
                outputs = self.calculate_loss(forward_outputs, batch, mode='test')
                loss_eval_time = time.time() - loss_start
                loss = outputs.loss

                # Gather statistics
                loss_value = loss.item()
                val_loss += loss_value * len(batch)
                total_samples += len(batch)

                # self.log_all(outputs, self.global_step, prefix='val')

                metric_logger.update(**outputs)
                metric_logger.update(
                    move=move_time,
                    forward=forward_time,
                    metric=metric_time,
                    visual=visual_time,
                    loss_eval_time=loss_eval_time,
                    **metric_timing_stats,
                )

        # Average the validation loss
        if total_samples == 0:
            self.log_info(f"Validation {val_name} produced no samples")
            return {"loss": float("nan")}
        val_loss /= total_samples

        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.log_info(f"Validation {val_name} results: {metric_logger}")

        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if self.metric_manager.should_update("val"):
            metric_stats = self.metric_manager.compute(accelerator=self.accelerator)
            if self.accelerator.is_main_process:
                self.log_all(metric_stats, step=self.global_step, prefix=f"val_metrics/{val_name}")
            stats.update({f"metric_{k.replace('/', '_')}": v for k, v in metric_stats.items()})

        return stats

    def _validation_split_name(self, loader, fallback):
        dataset = getattr(loader, "dataset", None)
        visited = set()

        def find_query_split(obj):
            if obj is None:
                return None
            obj_id = id(obj)
            if obj_id in visited:
                return None
            visited.add(obj_id)
            if hasattr(obj, "query_split"):
                return str(getattr(obj, "query_split"))
            if hasattr(obj, "dataset"):
                value = find_query_split(getattr(obj, "dataset"))
                if value is not None:
                    return value
            if hasattr(obj, "datasets"):
                for child in getattr(obj, "datasets"):
                    value = find_query_split(child)
                    if value is not None:
                        return value
            return None

        return find_query_split(dataset) or str(fallback)

    def train_one_epoch(self, epoch):
        self.model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter(
            "min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        # metric_logger.add_meter(
        #     "dataloader", SmoothedValue(window_size=1, fmt="{value:.6f}")
        # )
        header = "Epoch: [{}]".format(epoch)
        loss_details_dict = {}
        start_steps = epoch * self.iters_per_epoch
        self.global_step = start_steps

        self.log_info(
            "Start training epoch {}, {} iters per inner epoch. Training dtype {}".format(
                epoch, self.iters_per_epoch, self.cfg.train.model_dtype
            )
        )

        train_iterable = itertools.islice(self.train_loader, self.iters_per_epoch)
        for it, batch in enumerate(metric_logger.log_every(
            train_iterable, self.cfg.train.print_freq, header, total=self.iters_per_epoch
        )):
            with self.accelerator.accumulate(self.model):
                # Perform the forward using the accerlate
                batch = move_to_device(batch, device=self.accelerator.device)
                batch_shape_stats = {}
                if batch and "img" in batch[0]:
                    img = batch[0]["img"]
                    if torch.is_tensor(img) and img.ndim >= 4:
                        samples_per_rank = int(img.shape[0])
                        views_per_sample = len(batch)
                        batch_shape_stats = {
                            "samples_per_rank": samples_per_rank,
                            "views_per_sample": views_per_sample,
                            "images_per_rank": samples_per_rank * views_per_sample,
                            "height": int(img.shape[-2]),
                            "width": int(img.shape[-1]),
                        }
                if batch_shape_stats and self.accelerator.is_main_process:
                    self.log_info(
                        "Train batch shape: epoch={} iter={} samples/rank={} "
                        "views/sample={} images/rank={} resolution={}x{}".format(
                            epoch,
                            it,
                            batch_shape_stats["samples_per_rank"],
                            batch_shape_stats["views_per_sample"],
                            batch_shape_stats["images_per_rank"],
                            batch_shape_stats["height"],
                            batch_shape_stats["width"],
                        )
                    )
                with self.accelerator.autocast():
                    forward_output = self.forward_batch(batch, mode='train')
                next_step = start_steps + 1
                train_metric_stats = {}
                if self.metric_manager.should_update("train", next_step):
                    train_metric_stats = self.metric_manager.compute_on_batch(
                        forward_output,
                        batch,
                        mode="train",
                    )
                train_visuals = {}
                if (
                    self.visual_manager.should_render("train", step=next_step)
                    and self.accelerator.is_main_process
                ):
                    train_visuals = self.visual_manager.render(forward_output, batch, mode="train")
                batch_output = self.calculate_loss(forward_output, batch, mode='train')
                loss = batch_output.loss
                if loss > self.cfg.train.clip_loss:
                    loss = loss * 0.0

                # Check if the loss is nan
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    rank = get_rank()
                    print(
                        f"Rank {rank}: Loss is {loss_value}, stopping training at iter {it} (epoch {epoch}, global step {self.global_step}).",
                        force=True,
                    )
                    sys.exit(1)

                self.accelerator.backward(loss)

                for item in batch_output:
                    if 'loss' in item:
                        batch_output[item] = self.accelerator.gather(batch_output[item]).mean().item()
                        if item in loss_details_dict:
                            loss_details_dict[item] += batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0
                        else:
                            loss_details_dict[item] = batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0

                # clip the gradient
                if self.accelerator.sync_gradients:
                    params_to_clip = self.model.parameters()
                    self.accelerator.clip_grad_norm_(
                        params_to_clip, self.cfg.train.clip_grad
                    )

                    def get_gradient_norm(parameters):
                        norm = 0
                        for param in parameters:
                            if param.grad is None:
                                continue
                            local_norm = param.grad.detach().data.norm(2)
                            norm += local_norm.item() ** 2
                        norm = norm**0.5
                        return norm

                    grad_norm = get_gradient_norm(self.model.parameters())

                if self.accelerator.state.deepspeed_plugin is None:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.lr_scheduler.step()

            if self.accelerator.sync_gradients:
                start_steps += 1

                # Report to tensorboard
                batch_output.update(loss_details_dict)
                loss_details_dict = {}

                if start_steps % 10 == 0 :
                    self.log_all(batch_output, start_steps, prefix='train')
                if train_metric_stats:
                    self.log_all(train_metric_stats, start_steps, prefix='train_metrics')
                if train_visuals:
                    self.log_all(train_visuals, start_steps, prefix='train_visuals')
                    self.log_info(f"Logged randomized train visuals: {self.visual_manager.last_render_info}")
                metric_logger.update(**batch_shape_stats)
                metric_logger.update(**batch_output)

                min_lr = 10.0
                max_lr = 0.0
                for group in self.optimizer.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])

                metric_logger.update(lr=max_lr)
                metric_logger.update(min_lr=min_lr)
                self.accelerator.log({"lr": max_lr}, step=start_steps)
                self.accelerator.log({"min_lr": min_lr}, step=start_steps)

                weight_decay_value = None
                for group in self.optimizer.param_groups:
                    if group["weight_decay"] > 0:
                        weight_decay_value = group["weight_decay"]
                metric_logger.update(weight_decay=weight_decay_value)
                metric_logger.update(grad_norm=grad_norm)
                self.accelerator.log(
                    {"weight_decay": weight_decay_value}, step=start_steps
                )
                self.accelerator.log({"grad_norm": grad_norm}, step=start_steps)

                self.global_step = start_steps

        # # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def log_all(self, output, step, prefix=""):        
        if 'log_keys' in output:
            log_keys = output.log_keys
        else:
            log_keys = list(output.keys())

        log_scaler = {}
        log_img = {}
        for k in log_keys:
            v = output[k]
            if np.isscalar(v):
                log_scaler[prefix+'/'+k] = v
                continue
            if isinstance(v, Image.Image):
                image = np.asarray(v.convert("RGB"))
                log_img[prefix+'/'+k] = image.transpose(2, 0, 1)[None]

        self.accelerator.log(log_scaler, step)
        for tracker in self.accelerator.trackers:
            tracker.log_images(log_img, step)

    def forward_batch(self, batch, mode='train'):
        output = self.model(batch)
        assert isinstance(output, EasyDict)
        return output

    def calculate_loss(self, output, batch, mode='train'):
        pass

    def build_accelerator(self):
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.cfg.log.output_dir,
            logging_dir=self.cfg.log.output_dir,
            total_limit=4,      # self.cfg.save_total_limit = 4
            # automatic_checkpoint_naming=True,
        )

        # Initialize the Environment variables throught MPI run
        init_distributed_mode(
            self.cfg.train, init_pytorch_ddp=False
        )  # set `init_pytorch_ddp` to False, since the accelerate will do later

        if self.cfg.log.use_wandb:
            log_with = 'wandb'
        elif self.cfg.log.use_tensorboard:
            log_with = 'tensorboard'
        else:
            log_with = 'all'

        mixed_precision = 'no' if self.cfg.train.model_dtype not in ['fp8', 'fp16', 'bf16'] else self.cfg.train.model_dtype

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # dynamic complie
        if self.cfg.train.get("dynamo_backend"):
            if isinstance(self.cfg.train.dynamo_backend, str) and hasattr(
                DynamoBackend, self.cfg.train.dynamo_backend.upper()
            ):
                dynamo_backend = getattr(DynamoBackend, self.cfg.train.dynamo_backend.upper())
            elif isinstance(self.cfg.train.dynamo_backend, DynamoBackend):
                dynamo_backend = self.cfg.train.dynamo_backend
            else:
                print(
                    f"Invalid dynamo_backend {self.cfg.train.dynamo_backend}, using default. Please refer to "
                    "https://huggingface.co/docs/accelerate/v1.2.1/en/package_reference/utilities#accelerate.utils.DynamoBackend for available names."
                )
        else:
            dynamo_backend = DynamoBackend.NO

        print(f"Using dynamo backend: {dynamo_backend}")

        torch._inductor.config.reorder_for_compute_comm_overlap = True

        dynamo_plugin = TorchDynamoPlugin(
            backend=dynamo_backend,
            mode="max-autotune-no-cudagraphs",
            dynamic=self.cfg.train.get("dynamic_compile", True),
        )
        
        accelerate_config = dict(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_config=accelerator_project_config,
            dataloader_config=DataLoaderConfiguration(
                non_blocking=True,
                split_batches=False,
                dispatch_batches=None,
                even_batches=True,
                use_seedable_sampler=False,
            ),
            step_scheduler_with_optimizer=False,             # not to step n_gpus times per step.
            dynamo_plugin=dynamo_plugin,
        )

        # fsdp
        if self.cfg.get("fsdp_plugin"):
            fsdp_plugin_kwargs = {}
            fsdp_plugin_kwargs[
                "mixed_precision_policy"
            ] = torch.distributed.fsdp.MixedPrecision(
                param_dtype=self.weight_dtype,
                reduce_dtype=self.weight_dtype,
                buffer_dtype=self.weight_dtype,
                cast_forward_inputs=True,
                cast_root_forward_inputs=True,
            )

            fsdp_plugin = hydra.utils.instantiate(self.cfg.fsdp_plugin)(**fsdp_plugin_kwargs)
            accelerate_config["fsdp_plugin"] = fsdp_plugin
        else:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=self.cfg.train.find_unused_parameters)
            accelerate_config['kwargs_handlers'] = [ddp_kwargs]

        accelerator = Accelerator(**accelerate_config)

        self.logger = get_logger(self.cfg, os.path.basename(__file__))

        # To block the print on non main process
        setup_for_distributed(accelerator.is_main_process)

        # self.logger.rank_zero_only = False
        self.log_info(accelerator.state)
        # self.logger.rank_zero_only = True
        
        if self.cfg.random_seed is not None:
            set_seed(self.cfg.random_seed, device_specific=True)

        self.device = accelerator.device

        self.accelerator = accelerator

    def auto_resume(self):
        if self.cfg.train.resume:
            path = self.cfg.train.resume
        elif os.path.exists(self.cfg.log.ckpt_dir):
            # Get the most recent checkpoint
            dirs = os.listdir(self.cfg.log.ckpt_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint_")]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            if path is not None:
                path = os.path.join(self.cfg.log.ckpt_dir, path)
        else:
            path = None

        if path is None:
            self.log_info("Checkpoint does not exist. Starting a new training run.")
            
            start_epoch = 0
        else:
            self.log_info(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(
                # os.path.join(self.cfg.log.ckpt_dir, path)
                path
            )
            # Extract epoch number from checkpoint path
            # Handles both "checkpoint_N" and "best_model" formats
            if "checkpoint_" in path:
                # Extract epoch from "checkpoint_N" format
                checkpoint_name = path.rstrip('/').split('/')[-1]
                start_epoch = int(checkpoint_name.split("checkpoint_")[-1])
            else:
                # For "best_model" or other formats, start from epoch 0
                # This is correct for stage transitions where we want to reset the epoch counter
                start_epoch = 0

        return start_epoch

    def log_info(self, info):
        if is_logging_process():
            self.logger.info(info)
