# LMGeo 518x518 A40 Training

This repo now has two 518px LMGeo profiles for a 46 GB A40-class GPU.

## Profiles

Conservative fixed-view profile:

```bash
train=train_lmgeo_finetune_518_a40
data=lmgeo_all_trainpbr_test_bop_518_a40
```

This uses 24 total views: 5 reference renders + 19 query images. It was chosen
to avoid repeated query frames and leaves strong A40 memory margin.

High-utilization dynamic profile:

```bash
train=train_lmgeo_finetune_518_a40_dynamic
data=lmgeo_trainpbr45_real_and_new_val_518_a40_dynamic
```

This is the profile to use for the bigger run. It inherits the
`trainpbr45_real_and_new_val` validation structure:

- `real_test`
- `pbr_new_val`
- `pbr_new_val_k5_subset`
- `pbr_new_val_k10_subset`

Training samples draw a total view count from `train.image_num_range: [3, 32]`.
The LMGeo dataset then chooses a valid split inside
`lmgeo.num_reference_range: [2, 7]` and `lmgeo.num_query_range: [1, 25]`.
The two counts are not independently drawn by the current sampler; total views
are sampled first, then the reference/query split is sampled from the valid
choices. The worst case is 7 references + 25 queries = 32 views.

With `train.max_img_per_gpu: 28`, the 32-view case is still allowed as one
sample per GPU because the dynamic batch sampler uses a minimum sample batch
size of one. For smaller sampled view counts, it packs more sequences per GPU
using:

```text
max(1, floor(train.max_img_per_gpu / sampled_total_views))
```

The value is intentionally below 32. A local dynamic run with
`max_img_per_gpu=32` reached about `45.6 GiB` of PyTorch allocated memory on a
packed small-view batch, which is too close to the A40 target.

## Dynamic Pixel Range

The dynamic profile enables the existing Pi3 high-resolution mechanism:

```yaml
train:
  random_reslution: true
  aspect_ratio_range: [0.5, 2.0]
  pixel_count_range: [100000, 268324]
  patch_size: 14
  num_resolution: 16
  base_resolution:
    - [518, 518]
```

`518 * 518 = 268324`, so the largest sampled pixel count is the Pi3 native
518px square. `base_resolution` forces `[518, 518]` to remain in the random
resolution pool every epoch.

## Memory Probes

Measured on the local NVIDIA RTX PRO 6000 Blackwell with PyTorch
`2.7.1+cu128`, BF16, frozen encoder, and checkpoint
`ckpts/Pi3/model.safetensors`.

Training, fixed 518x518, metrics/visuals disabled:

| Total views | Reference + query | Peak allocated |
| --- | --- | --- |
| 24 | 5 + 19 | 31.6 GB |
| 28 | 5 + 23 | 36.2 GB |
| 32 | 7 + 25 | 40.7 GB |
| 36 | repeated-query stress | 45.2 GB |
| 38 | repeated-query stress | 47.4 GB |

The dynamic A40 config allows 32-view samples but uses
`train.max_img_per_gpu: 28` as the packing budget. The 32-view case still runs
as one sample per GPU, while smaller sampled view counts are packed more
conservatively. A local dynamic run with the old 32-image packing budget reached
about 45.6 GiB allocated, which is too close to the A40 target.

Validation, fixed 518x518, metrics enabled and visuals disabled. The table uses
PyTorch peak reserved memory, which is closer to the `nvidia-smi` process value
than peak allocated tensor memory.

| Validation loader | Views/sample | `max_img_per_gpu` | Peak reserved |
| --- | ---: | ---: | ---: |
| `real_test` | 6 | 256 | 39.1 GB |
| `pbr_new_val` | 6 | 256 | 39.1 GB |
| `pbr_new_val_k5_subset` | 10 | 256 | 39.1 GB |
| `pbr_new_val_k10_subset` | 15 | 208 | 39.1 GB |

The old validation budget of 512 packed about 85 six-view sequences into one
batch and killed a local run immediately after epoch 0 began validation. The
dynamic data config now uses `max_img_per_gpu: 256` for the 6- and 10-view
validation loaders and `208` for the 15-view loader. A 256 budget for the
15-view loader reached 47.3 GB reserved and is deliberately not used.

Validation timing diagnostic, 2026-07-15, local RTX PRO 6000 Blackwell,
one train step plus one validation batch per loader, visuals disabled:

| Profile | Validation budget | Loader | Forward | Metrics | Total batch |
| --- | ---: | --- | ---: | ---: | ---: |
| 224px baseline | 96 | `real_test` | 0.36 s | 0.94 s | 2.99 s |
| 224px baseline | 96 | `pbr_new_val` | 0.35 s | 0.76 s | 3.44 s |
| 518px dynamic | 96 | `real_test` | 2.36 s | 4.08 s | 9.99 s |
| 518px dynamic | 96 | `pbr_new_val` | 2.29 s | 3.78 s | 8.58 s |
| 518px dynamic | 256 | `real_test` | 5.99 s | 9.53 s | 22.98 s |
| 518px dynamic | 256 | `pbr_new_val` | 6.02 s | 9.03 s | 22.05 s |

So the high-resolution path is active and substantially slower. It can look
less dramatic in logs because the 518px validation profile intentionally packs
many more sequences into each logged batch: about 42 six-view sequences with a
budget of 256, versus about 16 with the 224px budget of 96. The object-pose
metric also evaluates fixed sampled CAD points, and the Chamfer metric
now uses the fast GPU path in the dynamic 518px profile: reference clouds stay
on CUDA, are deterministically capped to 20k points, and skip the old CPU voxel
downsampling path. Metric values are therefore a fast validation approximation;
override `metrics.items.chamfer.use_gpu_fast_path=false` to recover the old
CPU-voxelized Chamfer behavior.

Synthetic validation-shaped Chamfer timing on the local RTX PRO 6000 Blackwell
after this change, with `B=42`, 6 views/sample, 5 reference views/sample,
518x518 resolution, 10% valid-mask density, and 20k point caps:

| Chamfer path | Time |
| --- | ---: |
| Old CPU voxel prep + GPU nearest-neighbor | 12.53 s |
| New GPU mask/subsample + GPU nearest-neighbor | 1.32 s |

That is a 9.5x Chamfer speedup for this case. Higher valid-mask density should
favor the GPU path even more because the old path spends more time in CPU
voxelization and dense GPU-to-CPU transfers.

## Commands

Single GPU dynamic profile:

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_518_a40_dynamic \
  data=lmgeo_trainpbr45_real_and_new_val_518_a40_dynamic \
  name=lmgeo_518_a40_dynamic
```

Four GPUs on one machine:

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 4 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_518_a40_dynamic \
  data=lmgeo_trainpbr45_real_and_new_val_518_a40_dynamic \
  name=lmgeo_518_a40_dynamic_4xa40
```

No config change is needed for ordinary data-parallel multi-GPU training. Each
GPU keeps its own per-rank view budget. The maximum effective views per
optimizer update are:

```text
32 * num_gpus * train.gradient_accumulation_steps
```

For Slurm, use [slurm_a40.md](slurm_a40.md).

## Fast Smoke Overrides

The production dynamic data config keeps `lmgeo.filter_preprocessed_query_depth:
true`. This is safer because invalid high-res samples are filtered before
training, but it can make startup slow on the full train split. The CITEc Slurm
script overrides this to `false` by default to save allocated GPU time.

For a fast memory smoke, override it:

```bash
lmgeo.filter_preprocessed_query_depth=false
```

For a worst-case 32-view fixed-resolution smoke:

```bash
train.random_reslution=false \
train.image_num_range=[32,32] train.max_img_per_gpu=32 \
lmgeo.num_reference_range=[7,7] lmgeo.num_query_range=[25,25]
```

The measured worst-case command with those overrides peaked at `40692 MB`.

## If An A40 Runs Out Of Memory

Drop total training views first:

```bash
train.image_num_range=[3,28] train.max_img_per_gpu=28
```

For a fixed 24-view fallback:

```bash
train=train_lmgeo_finetune_518_a40
data=lmgeo_all_trainpbr_test_bop_518_a40
```

If validation reserved memory is too high on an A40 node, lower the runtime
budgets:

```bash
val_datasets.real_test.runtime.max_img_per_gpu=128
val_datasets.pbr_new_val.runtime.max_img_per_gpu=128
val_datasets.pbr_new_val_k5_subset.runtime.max_img_per_gpu=128
val_datasets.pbr_new_val_k10_subset.runtime.max_img_per_gpu=128
```
