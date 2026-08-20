# 2026-08-14 MegaPose-GSO geometry N=5/K=1 maximum-batch probe

Status: intentionally stopped after significant early progress. This is a
fixed-training-batch memorization probe, not a held-out generalization result.

## Question and setup

Does the compact scratch Pi3 configuration still make useful progress when the
fixed overfit batch is expanded from one sample to the maximum measured
336x252 batch?

- train config:
  `train_megapose_gso_geometry_small_scratch_ray_336x252_overfit_max_batch`;
- data config: `megapose_gso_geometry_n5_k1_masked_overfit_max_batch`;
- 24 distinct objects, with one deterministic scene-to-scene sample per
  object;
- each sample has five reference views and one query, for 144 images per
  optimizer step;
- fixed first valid coverage plan/query, shared focal target, no crop-center or
  focal randomness, and byte-identical per-object samples across epochs;
- object-only RGB and depth for both roles, no visibility-mask conditioning;
- calibrated post-crop ray conditioning, 336x252 input resolution;
- pretrained frozen DINOv2-S encoder; 62,101,976 randomly initialized
  trainable decoder/head/ray parameters;
- bf16, AdamW, no weight decay, and the same OneCycle schedule as the
  one-sample control: peak LR `1e-4` for decoder/head and `1e-3` for the
  zero-initialized ray adapter, with 5% warmup;
- validation every 10 updates on the exact same 24 samples.

The dataset was explicitly checked at epochs zero and one. The same 24 object
IDs occurred, and each object's image, intrinsic, and pose tensors were
byte-identical. The collated batch contained 24 samples x 6 views.

An initial launch inherited the parent's six-image named-validation budget and
therefore measured only one validation sample. It made no training update and
was discarded. The clean `_v2` run overrides both train and named-validation
budgets to 144 images, so every reported value below aggregates all 24 samples.

## Results

| Update | Val loss | Local points | Translation loss | Rotation loss | Query rot. median | Query trans. median | Pose median / diameter |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.98106 | 0.26894 | 0.05173 | 1.94856 | 91.819 deg | 1.1061 m | 4.5451d |
| 10 | 0.88852 | - | - | - | 63.280 deg | 1.1317 m | 4.6246d |
| 20 | 0.85233 | - | - | - | 69.044 deg | 1.2152 m | 4.5152d |
| 30 | 0.82608 | - | - | - | 68.754 deg | 0.9305 m | 3.7890d |
| 40 | 0.71684 | 0.12091 | 0.04887 | 1.07251 | 41.258 deg | 0.9195 m | 3.5973d |

By update 40, relative to initialization:

- total validation loss fell by 26.9%;
- local-point loss fell by 55.0%, rotation loss by 45.0%, and translation loss
  by 5.5%;
- median query rotation improved by 55.1%;
- median query translation improved by 16.9%;
- median normalized pose error improved by 20.9%, while its p90 improved by
  29.8%;
- query ray angular error improved from 23.65 to 9.00 degrees, and ray
  reprojection error from 334.2 to 118.2 pixels.

The improvement was broadest in rotation: 22 of 24 objects improved. Pose and
translation improved for 15 of 24 objects, so translation learning had started
but was not yet uniform. One query reached normalized pose error at or below
`2d`; none reached `1d`. Strict ADD and ADD-S at `0.1d` remained zero.
Correspondence median L2 was non-monotonic (`0.6925` initially, `0.6119` at
update 30, and `0.8447` at update 40). The fitted ray focal errors also remained
near 97%, despite the strong angular/reprojection improvement. These are clear
signs that the batch was far from memorized when stopped.

## Interpretation

This is significant early progress, but not convergence. Rotation, ray
geometry, and local geometry learned first and across nearly the whole batch;
translation and complete object pose followed more slowly. Update 40 was also
still near the top of the OneCycle schedule (the warmup peak is update 30), so
this run did not test the lower-LR memorization tail.

The result is substantially behind the one-sample control at the same number
of updates, as expected for gradients shared across 24 objects. The one-sample
run was already close to 2.7 degrees and 80 mm at update 50, whereas this run
was at 41.3 degrees and 920 mm at update 40. It nevertheless shows that the
maximum batch is not preventing optimization: several independent geometric
metrics move decisively in the correct direction.

The run was stopped manually immediately after update-40 validation and the
best-checkpoint write. The resulting `KeyboardInterrupt`/exit code 1 is
intentional, not a crash. The final in-progress updates after that checkpoint
are not used in this report.

## Memory and throughput

- first update: 20,582 MiB peak allocated and 21,784 MiB reserved;
- steady peak: 21,076 MiB allocated and 22,256 MiB reserved;
- physical GPU: 97,887 MiB RTX PRO 6000 Blackwell;
- steady wall time: about 4.4-4.8 seconds/update, or 30-32 images/second.

The 144-image batch therefore fits comfortably. About 3.7-4.0 seconds per
update were spent loading the uncached fixed batch with zero data-loader
workers, while model compute was about 0.7 seconds. For a longer memorization
run, caching/preloading these 24 fixed samples or enabling suitable persistent
workers would improve wall-clock speed substantially; GPU memory is not the
bottleneck.

## Artifacts

Run root:

```text
/media/internal/nvme/dtrofimov/runs/overfit_gso_geometry_scene_n5k1_maxbatch24_small_ray_336_20260814_v2
```

TensorBoard log directory:

```text
/media/internal/nvme/dtrofimov/runs/overfit_gso_geometry_scene_n5k1_maxbatch24_small_ray_336_20260814_v2/tensorboard/overfit_gso_geometry_scene_n5k1_maxbatch24_small_ray_336_20260814_v2
```

The best complete checkpoint is `ckpts/best_model` at update 40. Training logs
are under the run root and `ckpts/log.txt`.

## Launch

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_megapose_gso_geometry_small_scratch_ray_336x252_overfit_max_batch \
  data=megapose_gso_geometry_n5_k1_masked_overfit_max_batch \
  name=overfit_gso_geometry_scene_n5k1_maxbatch24_small_ray_336 \
  train.auto_resume=false
```

