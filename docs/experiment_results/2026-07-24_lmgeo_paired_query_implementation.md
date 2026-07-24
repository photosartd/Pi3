# 2026-07-24 Paired Recentered/Original Query Implementation

Status: implementation and local validation complete; production training not
started; real CITEc A40 smoke still pending.

## Purpose

Add two opt-in K=1 regimes without changing the existing crop-only baselines:

- recentered/zoomed query plus its original-camera companion, no ray
  conditioning;
- the same two views with the zero-initialized ray-conditioning branch.

The implementation and metric definitions are documented in
[../lmgeo_paired_query_views.md](../lmgeo_paired_query_views.md).

## Configurations

The paired data profile is:

```text
data=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1
```

Compose it with either:

```text
train=train_lmgeo_finetune_a40_46gb_recenter_zoom_k1
train=train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1
```

The existing
`data=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1` remains crop-only and
does not instantiate the paired-query metric.

All small runs used the real LM-O data and
`ckpts/Pi3/model.safetensors`. Preprocessed-depth filtering was explicitly
disabled for training and validation as required. Checkpoint saving was
disabled.

## Validation

- Both new Hydra pairings composed successfully with `--cfg job`.
- A real 560x420 train sample produced exactly two references, one crop, and
  one original-camera companion. Both `camera_pose = inv(T_C_O)` relations,
  `T_Ccrop_O = A T_Corig_O`, nonempty valid masks, and
  `H = K_crop R K_original^-1` passed numerical checks.
- The full CPU suite passed: 66 tests.
- A real seven-view BOP sample also passed the production collator: five
  references, one crop, and one original-camera context view retained the
  paired rotation, raw/final homographies, original pose, role flags, and
  pair index with shape-stable batched tensors.
- Synthetic exact-pair tests reached effectively zero canonical pose,
  relative-camera, reprojection, dense local equivariance, and shared-world
  residuals. A controlled original-camera perturbation increased only the
  expected original/pair metrics.
- One-step BF16 560x420 GPU runs completed for both ray-off and ray-on
  profiles, including one real BOP validation batch, all configured metrics,
  paired visuals, and TensorBoard.
- TensorBoard contained 79 paired-query scalar tags and the
  `input_query_context_frames` image grid in both runs. The ray run additionally
  contained 63 ray/intrinsics tags and `/train_ray_lr`.

On the one-batch real validation probe, the paired metric took 22-46 ms, the
model forward took 111-113 ms, and the complete metric set took 78-82 ms. These
are workstation smoke timings, not stable benchmark numbers.

## Parameters and Memory

The ray branch adds 402,432 trainable parameters:

```text
baseline total/trainable: 892,366,936 / 587,995,224
ray total/trainable:      892,769,368 / 588,397,656
```

Measured BF16 training peaks on the local RTX PRO 6000 were:

| Shape per rank | Rays | Allocated | Reserved |
| --- | --- | ---: | ---: |
| 1 x 4 views, metric/TensorBoard smoke | off | 12,464 MiB | 12,878 MiB |
| 1 x 4 views, metric/TensorBoard smoke | on | 12,469 MiB | 12,888 MiB |
| 1 x 18 views, Slurm-script maximum | off | 22,333 MiB | 22,494 MiB |
| 1 x 18 views, Slurm-script maximum | on | 22,394 MiB | 22,554 MiB |
| 7 x 4 views, 28-image packed maximum | off | 32,224 MiB | 32,454 MiB |
| 7 x 4 views, 28-image packed maximum | on | 32,317 MiB | 32,574 MiB |

The Slurm-script probes used `PI3_CUDA_MEMORY_LIMIT_GIB=44`, metrics and
visuals disabled, and all depth filters off. Both extremes completed forward,
loss, backward, optimizer step, and one batch from each validation loader.
This supports retaining `max_img_per_gpu: 28`: the paired sampler realizes
batch size 7 at four views and batch size 1 at 18 views.

`sbatch` is unavailable on the workstation. The exact `.sbatch` body was run
locally under the 44 GiB allocator cap, but this is not evidence of a completed
real A40 allocation. Run both documented CITEc smoke jobs before production.

## Conclusion

The code path is ready for a fair four-regime experiment. Pair expansion,
supervision, canonical evaluation, TensorBoard logging, ray LR separation, and
the 28-image dynamic budget all work end-to-end locally. No production-quality
accuracy conclusion is possible from one optimizer step; the next gate is the
real CITEc A40 smoke, followed by matched training runs.
