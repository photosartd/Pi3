# LMGeo 560x420 A40 Training

> The filename is retained for backward-compatible links. The current LMGeo
> high-resolution target is fixed 560x420, not 518x518.

Use the named A40 profile with the canonical dataset:

```text
train=train_lmgeo_finetune_a40_46gb
data=lmgeo_trainpbr45_real_and_new_val
```

The historical `train_lmgeo_finetune_518_a40_dynamic` and
`lmgeo_trainpbr45_real_and_new_val_518_a40_dynamic` names remain aliases, but
new commands should use the names above.

## Why 560x420

LM-O images are natively 640x480. The 560x420 target has the same 4:3 aspect
ratio and a 40x30 DINO patch grid because both dimensions are divisible by 14.
Under the existing Pi3 crop/resize preprocessing, the rectangular target keeps
roughly 97% of the original image area. A square 224x224 or 518x518 target keeps
roughly 74%, mostly because it removes the left and right borders.

This change only selects a different target resolution. It does not implement
the separate proposed `full_resize` preprocessing mode; intrinsics, RGB, depth,
and masks continue through the current geometrically consistent crop/resize
path.

Training and every validation loader use exactly:

```yaml
train:
  resolution:
    - [560, 420]
  random_reslution: false
```

There is no generated resolution pool for LMGeo A40 or Blackwell runs.

## Views And Packing

Training samples 6-28 views per sequence, with 5-27 references and 1-25
queries. The constraints are applied jointly: a sampled total gets only splits
that satisfy all three ranges. Consequently, the 6-view minimum is always 5
references plus 1 query. At 28 views, the largest query-heavy split is
5 references + 23 queries, and the largest reference-heavy split is
27 references + 1 query. This covers the main validation shape
(5 references + 1 query) during training.

`train.max_img_per_gpu: 28` is a packing budget, not a hard per-sequence view
limit. The sample batch size is:

```text
max(1, floor(max_img_per_gpu / sampled_total_views))
```

Thus a 28-view iteration contains one sequence, while a 6-view iteration packs
4 sequences, or 24 images, on each GPU.

The total count is sampled uniformly from the 23 integers in 6-28. Conditional
on that total, the reference count is sampled uniformly from its valid splits.
At the two relevant edges:

- `5 refs + 1 query` occurs whenever total views is 6: about 1/23 of training
  iterations. Because that iteration packs four sequences, a 500-step epoch
  contains about 21.7 such batches, or 87.0 such sequences, in expectation.
- `27 refs + 1 query` is one of 21 valid splits at total views 28: about 1/483
  of iterations, or 1.04 occurrences per 500-step epoch in expectation.

The nominal query range remains 1-25, but with the 28-view A40 cap and five
minimum references, the largest actually sampled query count is 23. Use a
larger GPU profile if a 7+25, 32-view sequence is required.

Validation budgets are fixed to 128 for K=1, K=5, K=10, and the new
16-reference/1-query ablation. Their actual per-rank batches contain:

| Loader | Views/sequence | Sequences/batch | Images/batch |
| --- | ---: | ---: | ---: |
| `real_test` | 6 | 21 | 126 |
| `real_test_ref16` | 17 | 7 | 119 |
| `pbr_new_val` | 6 | 21 | 126 |
| `pbr_new_val_ref16` | 17 | 7 | 119 |
| `pbr_new_val_k5_subset` | 10 | 12 | 120 |
| `pbr_new_val_k10_subset` | 15 | 8 | 120 |

The two `ref16` loaders keep exactly one query and reuse the same real/PBR query
sets as their corresponding 5-reference loaders. They therefore test more
object keyframes rather than more query context. Each configured LM-O object
has 1,313 eligible reference renders, so selecting 16 does not require repeated
keyframes. `primary_val` remains `real_test`, so checkpoint selection behavior
does not change.

The training sampler can produce the exact 16+1 shape, but it is uncommon:
total views 17 is sampled with probability 1/23, and 16+1 is one of 12 valid
splits at that total. Its per-iteration probability is therefore 1/276, or
about 1.81 occurrences per 500-step epoch in expectation.

Interpret this comparison carefully: object-pose evaluation estimates its
reference-only Sim(3) alignment from every reference view. A better `ref16`
score can therefore come from better model context, a more stable alignment,
or both. It is not a pure context-only ablation unless both predictions are
also scored with a common fixed alignment subset.

## 40 GB-Constrained Probe

Measured on 2026-07-17 on the local RTX PRO 6000 Blackwell with only about
40 GB free. PyTorch was capped at 38 GiB. Runs used BF16, the frozen encoder,
the Pi3 base checkpoint, one indexed LM-O object/scene for fast startup, and
the same tensor shapes and production metrics as the full config.

Fixed one-sequence training:

| Views | Reference + query | Allocated | Reserved |
| ---: | --- | ---: | ---: |
| 24 | 7 + 17 | 28,287 MiB | 28,830 MiB |
| 28 | 7 + 21 | 32,276 MiB | 32,764 MiB |
| 32 | 7 + 25 | 36,262 MiB | 36,634 MiB |
| 32 | 31 + 1 | 36,262 MiB | 36,634 MiB |

Memory rose by about 997 MiB per additional view over this interval.

Current minimum-view packing:

| `max_img_per_gpu` | Packed shape | Reference + query per sequence | Allocated | Reserved |
| ---: | --- | --- | ---: | ---: |
| 32 | 5 x 6 views | 5 + 1 | 34,281 MiB | 34,716 MiB |

These older 32-view/5x6 boundaries completed a training step and a small 5+1
inference batch under the 38 GiB allocator cap. That result explains why the
32-view A40 setting looked plausible locally, but it is no longer an accepted
production limit. Changing the 32-view role split from 7+25 to 31+1 did not
change measured peak tensor memory.

The earlier 3-view packing probes below are retained as historical allocator
measurements; 3-view sequences are no longer sampled by the named A40 or
Blackwell profiles.

Packed 3-view training (historical):

| `max_img_per_gpu` | Packed shape | Allocated | Reserved |
| ---: | --- | ---: | ---: |
| 28 | 9 x 3 views | 31,326 MiB | 31,858 MiB |
| 32 | 10 x 3 views | 34,273 MiB | 34,722 MiB |

Production validation was exercised at budgets 64 and 128 after a 32-view
training step. The four loaders that existed before the Ref16 addition
completed with object-pose, camera, correspondence, and Chamfer metrics enabled.

| Validation budget | Approx. images/batch | Largest allocated | Combined peak reserved |
| ---: | ---: | ---: | ---: |
| 64 | 60 | 12,139 MiB | 38,190 MiB |
| 128 | 120-126 | 15,564 MiB | 38,246 MiB |

Across the comparable loaders, doubling the validation budget added about
3.1-3.4 GiB for 60-66 additional images, or approximately 52 MiB per added
inference image. Forward time rose from roughly 3.0 seconds at budget 64 to
6.3-6.9 seconds at budget 128. Training is much steeper because activations and
gradients are retained: the 24/28/32-view measurements add about 997 MiB per
view.

The full-production-validation acceptance run exited successfully and a
0.2-second `nvidia-smi` sampler observed a 37,370 MiB process peak.

The combined reserve is dominated by allocator state retained from the
preceding 32-view backward pass. Budget 128 is the largest validation setting
tested under the current 40 GB availability and retains about 2.6 GiB below a
40 GiB process-memory target before ordinary CUDA-context variation.

## A40 OOM Correction

On the real CITEc A40 nodes, the old 32-view profile OOMed during training
after only a few steps:

```text
39.07 GiB allocated by PyTorch
4.68 GiB reserved but unallocated
44.23 GiB process memory on a 44.42 GiB GPU
```

The local 38 GiB allocator smoke underestimated production memory because it
tested selected edge shapes on a different GPU and did not run enough random
production samples to expose allocator drift/fragmentation on A40. The current
A40 default therefore uses 28 views and a 28-image packing budget. The Slurm
script also defaults to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Commands

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_a40_46gb \
  data=lmgeo_trainpbr45_real_and_new_val \
  name=lmgeo_a40_560x420
```

Four GPUs on one machine:

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 4 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_a40_46gb \
  data=lmgeo_trainpbr45_real_and_new_val \
  name=lmgeo_a40_560x420_4xa40
```

Each GPU retains the same local resolution and packing budget. Different ranks
receive different sample indices, but no rank selects a different resolution.

For CITEc submission, paths, and checkpoint overrides, see
[slurm_a40.md](slurm_a40.md). For all local profile commands, see
[lmgeo_hardware_profiles.md](lmgeo_hardware_profiles.md).

Failure-mode diagnostics for size, occlusion, and object class remain
documented in [failure_modes.md](failure_modes.md). The A40 profile writes raw
ADD/ADD-S prediction rows to `${log.output_dir}/predictions.parquet`.

## Recenter + Zoom K1 Diagnostic

The object-size diagnostic uses:

```bash
train=train_lmgeo_finetune_a40_46gb_recenter_zoom_k1
data=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1
```

This profile keeps the fixed 560x420 resolution and A40 packing budget, but
changes the view-count contract to `2..16` reference/keyframes plus exactly
`1` query. The query frame is transformed by
`datasets.lmgeo_recenter.LMGeoRecenterZoomSequenceDataset`:

- references/keyframes are unchanged;
- the query uses GT `bbox_obj` as an oracle center;
- a pure-rotation homography recenters the bbox ray onto the optical axis;
- the query is then zoomed with a centered virtual focal increase;
- `T_C_O`, `camera_pose`, intrinsics, depth, and valid supervision masks are
  updated consistently.
- query depth supervision uses the transformed scene depth, not only the object
  mask. Only missing source depth or pixels introduced by the homography/crop
  are invalidated. Reference/keyframe depth remains object-masked.

The active validation loaders are all K=1: `real_test`, `real_test_ref16`,
`pbr_new_val`, and `pbr_new_val_ref16`. Multi-query PBR K5/K10 validation is
disabled because the query recenter+zoom transform creates an object-centric
virtual crop and no longer preserves the original multi-query scene-context
assumption.

An example input grid is written to:

```text
examples/lmgeo_recenter_zoom_k1_input_example.png
```

## Blackwell Status

The RTX PRO 6000 profile inherits fixed 560x420 training and evaluation. Its
existing train/validation budgets remain 56 and 384/384/320. Those budgets were
validated at the larger 518x518 square and therefore remain conservative at the
smaller 560x420 token count. They were not re-maximized in this pass because
only about 40 GB was free on the device.
