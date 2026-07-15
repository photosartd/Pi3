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

With `train.max_img_per_gpu: 32`, the 32-view case is one sample per GPU. For
smaller sampled view counts, the existing dynamic batch sampler packs more
sequences per GPU using:

```text
floor(train.max_img_per_gpu / sampled_total_views)
```

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

The dynamic A40 config uses 32 as the practical maximum. It fits the target
with useful headroom; 36 is too close to a 46 GB card, and 38 is over the
target.

Validation, fixed 518x518, metrics enabled and visuals disabled:

| Validation loader | Views/sample | `max_img_per_gpu` | Peak allocated |
| --- | ---: | ---: | ---: |
| `real_test` | 6 | 512 | 39.2 GB |
| `pbr_new_val_k5_subset` | 10 | 512 | 39.2 GB |
| `pbr_new_val_k10_subset` | 15 | 512 | 39.2 GB |
| `pbr_new_val_k5_subset` | 10 | 768 | 54.0 GB |

So the dynamic data config uses `max_img_per_gpu: 512` for all inherited
validation loaders. The measured 768-view-budget probe is deliberately not used
because it exceeds the A40 target.

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

If validation runs out of memory before training, reduce validation runtime
budgets:

```bash
val_datasets.real_test.runtime.max_img_per_gpu=384
val_datasets.pbr_new_val.runtime.max_img_per_gpu=384
val_datasets.pbr_new_val_k5_subset.runtime.max_img_per_gpu=384
val_datasets.pbr_new_val_k10_subset.runtime.max_img_per_gpu=384
```
