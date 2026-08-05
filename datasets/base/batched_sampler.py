# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Random sampling under a constraint
# --------------------------------------------------------
import numpy as np
import torch
# from torch.utils.data import BatchSampler

from typing import Any, Callable, Generic, Iterable, List, Optional, TypeVar, Union
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler

class BatchedRandomSampler:
    """ Random sampling under a constraint: each sample in the batch has the same feature, 
    which is chosen randomly from a known pool of 'features' for each batch.

    For instance, the 'feature' could be the image aspect-ratio.

    The index returned is a tuple (sample_idx, feat_idx).
    This sampler ensures that each series of `batch_size` indices has the same `feat_idx`.
    """

    def __init__(self, dataset, batch_size, pool_size, world_size=1, rank=0, drop_last=True, frame_num_list=None):
        self.batch_size = batch_size
        self.pool_size = pool_size

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size*world_size) if drop_last else N
        assert world_size == 1 or drop_last, 'must drop the last batch in distributed mode'

        # distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None

        self.frame_num_list = frame_num_list

    def __len__(self):
        return self.total_size // self.world_size

    def set_epoch(self, epoch, base_seed=777):
        self.epoch = epoch
        self.base_seed = base_seed

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            # seed = self.epoch + 777
            seed = self.epoch + self.base_seed
        rng = np.random.default_rng(seed=seed)

        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # random feat_idxs (same across each batch)
        n_batches = (self.total_size+self.batch_size-1) // self.batch_size
        feat_idxs = rng.integers(self.pool_size, size=n_batches)
        feat_idxs = np.broadcast_to(feat_idxs[:, None], (n_batches, self.batch_size))
        feat_idxs = feat_idxs.ravel()[:self.total_size]

        if self.frame_num_list is not None:
            frame_nums = rng.choice(self.frame_num_list, size=n_batches)
            frame_nums = np.broadcast_to(frame_nums[:, None], (n_batches, self.batch_size))
            frame_nums = frame_nums.ravel()[:self.total_size]

            # put them together
            idxs = np.c_[sample_idxs, feat_idxs, frame_nums]  # shape = (total_size, 3)
        else:
            # put them together
            idxs = np.c_[sample_idxs, feat_idxs]  # shape = (total_size, 2)

        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * ((self.total_size + self.world_size *
                                           self.batch_size-1) // (self.world_size * self.batch_size))
        idxs = idxs[self.rank*size_per_proc: (self.rank+1)*size_per_proc]

        yield from (tuple(idx) for idx in idxs)

def round_by(total, multiple, up=False):
    if up:
        total = total + multiple-1
    return (total//multiple) * multiple


## from vggt

class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 resolution_num,
                 image_num_range,
                 epoch=0,
                 seed=42,
                 rank=0,
                 max_img_per_gpu=48,
                 frame_num_list=None):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
        """
        self.sampler = sampler
        self.resolution_num = resolution_num
        self.image_num_range = image_num_range
        
        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        if frame_num_list is None:
            frame_num_list = self.image_num_weights.keys()
        allowed = set(self.image_num_weights)
        possible_nums = sorted({int(value) for value in frame_num_list})
        if not possible_nums or not set(possible_nums).issubset(allowed):
            raise ValueError(
                f"frame_num_list must be a non-empty subset of "
                f"{sorted(allowed)}, got {possible_nums}"
            )
        self.possible_nums = np.asarray(possible_nums, dtype=np.int64)
        
        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        self.rank = rank

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)


    def set_epoch(self, epoch, base_seed=777):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.epoch = epoch
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        self.rng_rank = np.random.default_rng(epoch * 100 + base_seed + self.rank)
        self.rng = np.random.default_rng(epoch * 100 + base_seed)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number and aspect ratio
                random_image_num = int(self.rng.choice(self.possible_nums, p=self.normalized_weights))             # image number (batch size) should be the same (avoid one rank stop early)
                resolution_idx = self.rng_rank.choice(self.resolution_num)                            # resolution can different between different rank

                # Update sampler parameters
                self.sampler.update_parameters(
                    resolution_idx=resolution_idx,
                    image_num=random_image_num
                )

                # Calculate batch size based on max images per GPU and current image number
                batch_size = self.max_img_per_gpu / random_image_num
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)  # Ensure batch size is at least 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, aspect_ratio, image_num)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Dynamic batches contain as many samples as can fit under
        # max_img_per_gpu for the sampled sequence length. Use the smallest
        # possible sample batch size as a conservative, non-zero length
        # estimate for progress logging and short validation sets.
        max_image_num = int(np.max(self.possible_nums))
        min_sample_batch_size = int(np.floor(self.max_img_per_gpu / max_image_num))
        min_sample_batch_size = max(1, min_sample_batch_size)
        return max(1, int(np.ceil(len(self.sampler) / min_sample_batch_size)))


class HomogeneousDynamicBatchSampler(Sampler):
    """Dynamic sampler that chooses one dataset component per full batch.

    Component choice and view count are synchronized across distributed ranks;
    resolution remains rank-local, matching :class:`DynamicBatchSampler`.
    Samples are drawn from the chosen component without crossing its index
    range, which keeps optional observation schemas homogeneous.
    """

    def __init__(
        self,
        dataset,
        *,
        component_sizes,
        component_frame_num_lists=None,
        resolution_num,
        image_num_range,
        seed=42,
        rank=0,
        world_size=1,
        max_img_per_gpu=48,
    ):
        self.dataset = dataset
        self.component_sizes = np.asarray(component_sizes, dtype=np.int64)
        if self.component_sizes.ndim != 1 or len(self.component_sizes) < 2:
            raise ValueError(
                "HomogeneousDynamicBatchSampler requires at least two components"
            )
        if np.any(self.component_sizes <= 0):
            raise ValueError(
                f"component_sizes must be positive, got {self.component_sizes.tolist()}"
            )
        if int(self.component_sizes.sum()) != len(dataset):
            raise ValueError(
                f"component sizes sum to {int(self.component_sizes.sum())}, "
                f"but dataset length is {len(dataset)}"
            )

        self.component_starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(self.component_sizes)[:-1])
        )
        self.component_weights = self.component_sizes.astype(np.float64)
        self.component_weights /= self.component_weights.sum()
        self.resolution_num = int(resolution_num)
        self.image_num_range = tuple(int(value) for value in image_num_range)
        self.possible_nums = np.arange(
            self.image_num_range[0], self.image_num_range[1] + 1, dtype=np.int64
        )
        if len(self.possible_nums) == 0 or int(self.possible_nums[0]) <= 0:
            raise ValueError(
                f"image_num_range must be positive, got {image_num_range}"
            )
        if component_frame_num_lists is None:
            component_frame_num_lists = [
                self.possible_nums.tolist() for _ in self.component_sizes
            ]
        if len(component_frame_num_lists) != len(self.component_sizes):
            raise ValueError(
                "component_frame_num_lists must match component_sizes"
            )
        allowed = set(int(value) for value in self.possible_nums)
        self.component_frame_num_lists = []
        for component, values in enumerate(component_frame_num_lists):
            values = sorted({int(value) for value in values})
            if not values:
                raise ValueError(
                    f"Dataset component {component} supports no frame count in "
                    f"configured range {self.image_num_range}"
                )
            if not set(values).issubset(allowed):
                raise ValueError(
                    f"Dataset component {component} returned frame counts "
                    f"outside {self.image_num_range}: {values}"
                )
            self.component_frame_num_lists.append(np.asarray(values, dtype=np.int64))
        self.max_img_per_gpu = int(max_img_per_gpu)
        if self.max_img_per_gpu <= 0:
            raise ValueError("max_img_per_gpu must be positive")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.total_samples_per_rank = max(1, len(dataset) // self.world_size)
        self.component_batch_counts = np.zeros(
            len(self.component_sizes), dtype=np.int64
        )
        self.set_epoch(0, base_seed=self.seed)

    def set_epoch(self, epoch, base_seed=777):
        self.epoch = int(epoch)
        self.base_seed = int(base_seed)

    def _schedule(self):
        rng = np.random.default_rng(
            self.base_seed + 1000003 * self.epoch
        )
        emitted = 0
        batch_index = 0
        while emitted < self.total_samples_per_rank:
            component = int(
                rng.choice(len(self.component_sizes), p=self.component_weights)
            )
            image_num = int(rng.choice(self.component_frame_num_lists[component]))
            sample_batch_size = max(1, self.max_img_per_gpu // image_num)
            yield batch_index, component, image_num, sample_batch_size
            emitted += sample_batch_size
            batch_index += 1

    def __iter__(self):
        resolution_rng = np.random.default_rng(
            self.base_seed + 1000003 * self.epoch + 7919 * self.rank
        )
        self.component_batch_counts.fill(0)
        for batch_index, component, image_num, sample_batch_size in self._schedule():
            resolution_idx = int(resolution_rng.integers(self.resolution_num))
            # Every rank constructs the same global draw and takes a disjoint
            # contiguous slice. Tiny components use replacement deliberately,
            # matching DistributedSampler's padding semantics.
            draw_rng = np.random.default_rng(
                self.base_seed
                + 1000003 * self.epoch
                + 104729 * batch_index
                + 15485863 * component
            )
            global_draw_size = sample_batch_size * self.world_size
            component_size = int(self.component_sizes[component])
            offsets = draw_rng.choice(
                component_size,
                size=global_draw_size,
                replace=global_draw_size > component_size,
            )
            start = self.rank * sample_batch_size
            local_offsets = offsets[start : start + sample_batch_size]
            dataset_start = int(self.component_starts[component])
            self.component_batch_counts[component] += 1
            yield [
                (dataset_start + int(offset), resolution_idx, image_num)
                for offset in local_offsets
            ]

    def __len__(self):
        return sum(1 for _ in self._schedule())


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )

        self.resolution_idx = None
        self.image_num = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.resolution_idx, self.image_num)

    def update_parameters(self, resolution_idx, image_num):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.resolution_idx = resolution_idx
        self.image_num = image_num
