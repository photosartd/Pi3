from .base.transforms import *

from utils.misc import get_world_size, get_rank
from torch.utils.data import DataLoader
import hydra
from datasets.base.base_dataset import sample_resolutions, unified_collate_fn
from datasets.base.batched_sampler import (
    DynamicBatchSampler,
    DynamicDistributedSampler,
    HomogeneousDynamicBatchSampler,
)

__HIGH_QUALITY_DATASETS__ = ['BlinkVision', 'Game', 'GameNew', 'DynamicStereo', 'FlyingThings3D', 'GTA-sfm', 'Hypersim', 'MatrixCity', 'MidAir', 'Monkaa', 'PointOdyssey', 'Sintel', 'Spring', 'TarTanAir', 'Unreal4k', 'VirtualKitti', 'Habitat']
__MIDDLE_QUALITY_DATASETS__ = ['BlendedMVG', 'BlendedMVS', 'DTU', 'ETH3D', 'ScanNet', 'Scannetpp', 'Taskonomy']
__INDOOR_DATASETS__ = ['Hypersim', 'ScanNet', 'Scannetpp', 'Taskonomy', 'ARKitScenes', 'Habitat']

def create_dataloader(cfg, mode, *, dataset_cfg=None, dataloader_cfg=None, runtime_cfg=None):
    data_loader = DataLoader
    num_resolution = 1
    world_size = get_world_size()
    rank = get_rank()
    mixture_component_sizes = None
    mixture_component_names = None
    mixture_component_frame_counts = None
    dataset_frame_counts = None

    # pytorch dataset
    if mode == 'train':
        cfg_dataset = cfg.train_dataset if dataset_cfg is None else dataset_cfg
        cfg_dataloader = cfg.train_dataloader if dataloader_cfg is None else dataloader_cfg
        cfg_runtime = cfg.train if runtime_cfg is None else runtime_cfg
        batch_size = cfg_runtime.batch_size if 'batch_size' in cfg_runtime else cfg.train.batch_size
        num_workers = cfg_runtime.num_workers if 'num_workers' in cfg_runtime else cfg.train.num_workers
        worker_defaults = cfg.train
    else:
        cfg_dataset = cfg.test_dataset if dataset_cfg is None else dataset_cfg
        cfg_dataloader = cfg.test_dataloader if dataloader_cfg is None else dataloader_cfg
        cfg_runtime = cfg.test if runtime_cfg is None else runtime_cfg
        batch_size = cfg_runtime.batch_size if 'batch_size' in cfg_runtime else (
            cfg.test.batch_size if 'batch_size' in cfg.test else cfg.train.batch_size
        )
        num_workers = cfg_runtime.num_workers if 'num_workers' in cfg_runtime else (
            cfg.test.num_workers if 'num_workers' in cfg.test else cfg.train.num_workers
        )
        worker_defaults = cfg.test

    if mode == 'train':
        image_num_range = cfg.train.image_num_range
    else:
        image_num_range = cfg_runtime.image_num_range if 'image_num_range' in cfg_runtime else [8, 8]
    print(f'Sampling frame number range from {image_num_range}')
    # adapte from vggt
    if mode == 'train':
        max_img_per_gpu = cfg.train.max_img_per_gpu if 'max_img_per_gpu' in cfg.train else image_num_range[0]
    else:
        max_img_per_gpu = cfg_runtime.max_img_per_gpu if 'max_img_per_gpu' in cfg_runtime else (
            cfg.train.max_img_per_gpu if 'max_img_per_gpu' in cfg.train else image_num_range[0]
        )
    print(f'Max frame number per rank {max_img_per_gpu}')

    def _needed_train_indices_per_rank():
        if mode != 'train' or cfg.train.iters_per_epoch <= 0:
            return 0
        min_image_num = int(image_num_range[0])
        return (int(max_img_per_gpu) // min_image_num) * int(cfg.train.iters_per_epoch)

    def _is_auto_length(value):
        return isinstance(value, str) and value.lower() == 'auto'

    if isinstance(cfg_dataset, str):
        dataset = eval(cfg_dataset) 
    elif 'weights' in cfg_dataset:
        weights = cfg_dataset.weights
        datasets_all = []

        if mode == 'train' and 'random_reslution' in cfg.train and cfg.train.random_reslution:
            num_resolution = cfg.train.num_resolution if 'num_resolution' in cfg.train else 1
            seed = 777 + 0
            base_resolution = cfg.train.base_resolution if 'base_resolution' in cfg.train else []
            resolutions = sample_resolutions(
                aspect_ratio_range=cfg.train.aspect_ratio_range,
                pixel_count_range=cfg.train.pixel_count_range,
                patch_size=cfg.train.patch_size,
                num_resolutions=num_resolution,
                seed=seed,
                base_resolution=base_resolution,
            )
            print('Initialized resolution', resolutions)
            num_resolution = len(resolutions)
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
                dataset_i.convert_attributes()
                datasets_all.append((dataset_name, weight, dataset_i))
        elif 'resolution' in cfg.train:
            resolutions = cfg.train.resolution
            print('Setting dataset resolution', resolutions)
            num_resolution = len(resolutions) if mode == 'train' else 1
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
                dataset_i.convert_attributes()
                datasets_all.append((dataset_name, weight, dataset_i))
        else:
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name])
                dataset_i.convert_attributes()
                datasets_all.append((dataset_name, weight, dataset_i))

        if 'length' in cfg_dataset:
            dataset_length_cfg = cfg_dataset.length
            if _is_auto_length(dataset_length_cfg):
                natural_length = sum(len(dataset_i) for _, _, dataset_i in datasets_all)
                needed_total = (_needed_train_indices_per_rank() + 1) * world_size
                dataset_length = max(natural_length, needed_total)
                print(
                    f'Auto dataset length resolved to {dataset_length} '
                    f'(natural={natural_length}, needed_total={needed_total}, world_size={world_size})'
                )
            else:
                dataset_length = int(dataset_length_cfg)
            weight_sum = sum([v for k, v in weights.items()])
            new_weights = {}
            for dataset_name, weight in weights.items():
                new_weights[dataset_name] = max(int(weight / weight_sum * dataset_length), 1)
            weights = new_weights
            print(f'New weights for dataset (adjusting to dataset length {dataset_length}): {new_weights}')

        mixture_component_names = [dataset_name for dataset_name, _, _ in datasets_all]
        mixture_component_frame_counts = [
            dataset_i.supported_frame_counts(image_num_range)
            for _, _, dataset_i in datasets_all
        ]
        datasets_all = [weights[dataset_name] @ dataset_i for dataset_name, _, dataset_i in datasets_all]
        mixture_component_sizes = [len(dataset_i) for dataset_i in datasets_all]
        dataset = datasets_all[0]
        for dataset_ in datasets_all[1:]:
            dataset += dataset_
        dataset.component_names = tuple(mixture_component_names)
    else:
        dataset = hydra.utils.instantiate(cfg_dataset)
        dataset.convert_attributes()
        dataset_frame_counts = dataset.supported_frame_counts(image_num_range)

    if mode == 'train' and cfg.train.iters_per_epoch > 0:
        print('Needed batch number per epoch (per rank):', _needed_train_indices_per_rank())
        print('Dataset length per rank:', len(dataset) // world_size)
        assert _needed_train_indices_per_rank() < len(dataset) // world_size

    if mode == 'train' and mixture_component_sizes is not None and len(mixture_component_sizes) > 1:
        print(
            'Using homogeneous mixed-dataset batches:',
            dict(zip(mixture_component_names, mixture_component_sizes)),
        )
        batch_sampler = HomogeneousDynamicBatchSampler(
            dataset,
            component_sizes=mixture_component_sizes,
            component_frame_num_lists=mixture_component_frame_counts,
            resolution_num=num_resolution,
            image_num_range=image_num_range,
            seed=cfg.train.base_seed,
            max_img_per_gpu=max_img_per_gpu,
            rank=rank,
            world_size=world_size,
        )
    else:
        if mixture_component_frame_counts is not None:
            dataset_frame_counts = mixture_component_frame_counts[0]
        sampler = DynamicDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=cfg.train.base_seed,
            shuffle=cfg_dataloader.shuffle,
            drop_last=cfg_dataloader.drop_last,
        )
        batch_sampler = DynamicBatchSampler(
            sampler,
            num_resolution,
            image_num_range,
            seed=cfg.train.base_seed,
            max_img_per_gpu=max_img_per_gpu,
            rank=rank,
            frame_num_list=dataset_frame_counts,
        )

    loader_kwargs = dict(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=unified_collate_fn,
    )
    if num_workers > 0:
        persistent_workers = (
            cfg_runtime.persistent_workers
            if 'persistent_workers' in cfg_runtime
            else worker_defaults.get('persistent_workers', True)
        )
        prefetch_factor = (
            cfg_runtime.prefetch_factor
            if 'prefetch_factor' in cfg_runtime
            else worker_defaults.get('prefetch_factor', 2)
        )
        loader_kwargs.update(
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        )

    return data_loader(**loader_kwargs)
