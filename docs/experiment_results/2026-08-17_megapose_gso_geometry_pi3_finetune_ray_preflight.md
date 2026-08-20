# 2026-08-17 MegaPose-GSO geometry full-Pi3 ray fine-tune preflight

Status: configuration, checkpoint-load, full-budget training, and validation
smoke passed on the local RTX PRO 6000 Blackwell under a 70 GiB allocator cap.
The production run has not been started by this preflight.

## Purpose

Test the same geometry-constrained MegaPose-GSO N=5/K=1 task as the completed
compact scratch run while restoring the released Pi3 model and its pretrained
geometry weights. The only instantiated module without a checkpoint value is
the Pi3X-style ray-map projection.

The profile is
`train_megapose_gso_geometry_pi3_finetune_ray_rtxpro6000_70gb_336x252` with
`data=megapose_gso_geometry_n5_k1_masked`.

## Model and schedule

| Field | Value |
| --- | --- |
| Encoder/decoder | historical large Pi3 architecture |
| Initial weights | `ckpts/Pi3/model.safetensors` |
| Encoder | checkpoint-loaded and frozen |
| Trainable historical Pi3 weights | 587,995,224 |
| New ray projection | 402,432 parameters, zero initialized |
| Total trainable | 588,397,656 |
| Training resolution/views | 336x252, fixed N=5/K=1 |
| Train budget | 156 images = 26 six-view samples/update |
| Validation budget | 768 images = 128 six-view samples/batch |
| Main peak LR | `5e-6` |
| Ray peak LR | `1e-5` |
| Schedule | 30 x 500 = 15,000 updates |
| Warm-up | 1,000 updates (`pct_start=1/15`) |

The main LR is the established completed LM-O fine-tuning rate. The ray branch
uses the existing LM-O ray-conditioning rate because it is the only new,
zero-initialized component; it is only twice the main LR, not the ten-times
multiplier used by some mask-conditioning experiments.

At 26 samples/update the run requests 390,000 generated samples: 390,000 query
views, 1.95 million reference views, and 2.34 million total model images. With
the equal training mixture this is approximately 195,000 scene-to-scene and
195,000 render-to-scene samples. These are stochastic draws rather than finite
dataset epochs.

## Checkpoint-load verification

The released checkpoint contains no `ray_embed.*` keys. The real model load
reported exactly these missing instantiated parameters:

```text
ray_embed.proj.weight
ray_embed.proj.bias
```

It reported no other missing instantiated Pi3 parameters. The checkpoint has
extra confidence-head tensors because this fine-tune retains the historical
`train_conf=false` model, so that optional head is not instantiated; this is
the same behavior as the established LM-O fine-tunes. The model implementation
zeros both ray tensors before loading the checkpoint, making the initial
forward identical to unconditioned Pi3.

## GPU smoke

Run:
`megapose_gso_geometry_pi3_finetune_ray_70gb_smoke_20260817_1`

The smoke used eight persistent train workers and completed:

- three forward/backward/AdamW updates at 26 samples x 6 views = 156 images;
- one 128-sample/768-image scene-to-scene validation batch;
- one 128-sample/768-image render-to-scene validation batch;
- all object-pose, camera, correspondence, and ray metrics;
- all 14 configured TensorBoard image panels.

| Phase | Peak allocated | Peak reserved | Result |
| --- | ---: | ---: | --- |
| First optimizer update | 60,805 MiB | 61,076 MiB | passed |
| Steady train maximum | 66,373 MiB | 66,844 MiB | passed |
| Validation active allocation | 25,095 MiB | allocator retained 66,844 MiB | passed |

The measured train reserve leaves 4,836 MiB below the 70 GiB allocator cap.
All training tensors were 26 samples, six views, and 336x252. After the cold
data-loader batch, complete updates took approximately 3.3-3.5 seconds. The
entire train plus metric/visual validation smoke took 1:46.

The partial validation batches produced scene/render losses of `0.5170` and
`0.7253`. They cover only the first 128 of 189 held-out objects after three
artificial smoke-schedule updates, so their pose values are functional checks,
not baselines for the production learning curve.

Artifacts are under:

```text
/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_geometry_pi3_finetune_ray_70gb_smoke_20260817_1
```

The TensorBoard event is in that root's `tensorboard` directory. The smoke
deliberately disabled checkpoint writes; it validates initialization, optimizer
state allocation, metrics, visuals, and memory rather than producing a model.

## Production launch

The local launcher validates all required paths, sets the 70 GiB allocator cap,
and uses direct single-GPU Python/Accelerate initialization:

```bash
RUN=megapose_gso_geometry_pi3_finetune_ray_n5k1_336_warmup1000_$(date +%Y%m%d_%H%M%S)
mkdir -p logs

PI3_RUN_NAME="$RUN" \
nohup scripts/run_megapose_gso_geometry_pi3_finetune_ray_70gb.sh train \
  > "logs/$RUN.log" 2>&1 &
```

TensorBoard will be written to:

```text
/media/internal/nvme/dtrofimov/spott3r/outputs/<RUN>/tensorboard
```

Production validation uses both complete 189-object loaders every 500 updates.
Visuals run every fifth validation epoch. The primary scene loss selects the
best checkpoint; periodic checkpoints are saved every five epochs so pose
metrics can also be used for retrospective selection.
