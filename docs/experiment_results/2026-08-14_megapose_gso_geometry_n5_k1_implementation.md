# 2026-08-14 MegaPose-GSO geometry-constrained N=5/K=1 implementation

Status: implementation and data-index validation complete; no convergence
claim (the production training run has not started).

## Hypothesis and protocol

Test whether the object-pose task becomes learnable under a controlled
calibrated-camera regime without changing Pi3's image interface:

- five references and one query per sample;
- all five references from one physical scene/track (or one render bank);
- reference/query scenes differ for scene-to-scene samples;
- every reference has `visib_fract >= 0.3`;
- exact joint visible mesh-surface coverage is at least 50%;
- at least one reference viewing direction is within 10 degrees of the query;
- a crop-only focal target exists for all six views and lies within +10% of the
  native query normalized focal;
- a 5% full-object-box margin remains inside every source image;
- RGB and depth are object-only for references and queries;
- the final model resolution is 336x252 (4:3), and ray conditioning receives
  the updated focal and principal-point offset through the post-crop `K`.

The crop center and focal target vary independently per view during training;
validation uses one shared interval midpoint and no jitter. Extrinsics are not
warped. The model's visibility-mask-conditioning branch is disabled: the
silhouette is supplied by object-only RGB, while calibration is supplied by
the ray map.

## Implementation

- `datasets/preprocess/megapose_gso_geometry.py`: resumable scene/render exact
  geometry sidecars and compact reference-plan catalogues.
- `datasets/object_geometry.py`: process-safe readers, shared focal intervals,
  and integer object-preserving 4:3 crop specifications.
- `datasets/object_sampling.py`: additive constrained scene-to-scene and
  render-to-scene sampling policies. Legacy policies remain available.
- `datasets/object_centric.py`: optional planned crop stage, direct 4:3 resize,
  consistent RGB/depth/mask/intrinsics handling, and exact final object mask.
- `configs/data/megapose_gso_geometry_n5_k1_masked.yaml`: homogeneous 50/50
  scene/render training mixture and matching held-out scene/render validation.
- `configs/train/train_megapose_gso_geometry_small_scratch_ray_336x252.yaml`:
  frozen pretrained DINOv2-S encoder, randomly initialized compact Pi3
  decoders/heads, and zero-initialized ray projection.
- `scripts/visualize_megapose_gso_geometry_inputs.py`: exact RGB/depth/ray-map
  materialization with machine-readable assertions and reports.

## Built indexes and measured results

| Artifact | Frames/views | Objects | Valid plans | Result |
| --- | ---: | ---: | ---: | --- |
| train scene geometry | 798,082 | 755 | — | zero errors |
| train scene N=5 plans | — | 755 | 156,114 | min union 50% |
| validation scene geometry | 169,973 | 189 | — | zero errors, 416.9 frames/s |
| validation scene N=5 plans | — | 189 | 9,970 | min union 50% |
| render geometry | 483,328 | 944 | — | zero errors |
| render N=5 positive-anchored plans | — | 944 | 479,552 | exact union recheck |

An exhaustive one-seed policy construction check succeeded for all 755 train
objects and all 189 validation objects in both the scene-reference and
render-reference regimes. The observed scene-reference minima/maxima were:

- train: minimum selected union coverage `0.50256`, maximum positive angle
  `9.99916 degrees`;
- validation: minimum selected union coverage `0.51624`, maximum positive angle
  `9.99965 degrees`.

The first real-data smoke exposed an important but legitimate geometry case:
an object-centered crop may move the calibrated principal point outside the
cropped image. The legacy resize helper rejected that case because it always
recentered around the principal point. Planned crops now bypass that second
crop and resize their already-4:3 window directly. This preserves the valid
off-center `K`, which the ray map explicitly represents. Legacy samples still
take the original principal-centered path.

The first render compatibility sweep also showed that 256 random query
candidates was too small for objects whose scene focal ranges only sparsely
overlap the render bank: 154/755 first-seed attempts exhausted that cap, even
though tested failures found a valid sample when scanning farther. The render
training component now searches up to 4,096 query candidates before falling
back, avoiding systematic cross-object refetch and under-sampling.

Real sample grids and JSON reports are under:

```text
/media/internal/nvme/dtrofimov/runs/pi3_gso_geometry_input_smoke/
```

The visualization command asserts 336x252 tensors, exact zero RGB/depth outside
the final mask, crop metadata, and measured positive angle before writing a
grid. Inspection shows that the mechanics are correct, but also confirms a
real data-quality difficulty: unoccluded (`visib_fract` high) edge-on or
distant objects can still occupy very few pixels. The 50% constraint is joint
3D surface coverage, not a minimum projected object area.

## Validation performed

- Hydra composition of the new train/data pairing;
- focused geometry/crop/sampling/render-plan tests;
- exhaustive plan construction for every train/validation object;
- full CPU suite: 190 tests passed;
- real train and held-out validation sample materialization and visual review;
- one-step GPU optimization plus both validation loaders: passed;
- production 144-image-per-rank GPU budget smoke: passed.

The initial production-budget smoke packed 24 samples x 6 views = 144 images at
336x252. One bf16 forward/backward/AdamW step completed with loss `1.2830`,
gradient norm `0.6834`, 20,630 MiB peak allocated and 21,784 MiB peak reserved.
Both validation loaders also completed at the configured 128-image budget. The
smaller metric-enabled smoke confirmed gradients and all object-pose/camera/
correspondence/ray metric routes, but its single-sample pose values are not
scientifically meaningful for an untrained random decoder.

A later production preflight doubled the fixed-N batch to 48 samples x 6 views
= 288 images with eight persistent workers. Six complete bf16
forward/backward/AdamW updates passed under `PI3_CUDA_MEMORY_LIMIT_GIB=70`,
peaking at 42,357 MiB allocated and 43,074 MiB reserved. After the cold first
batch, prefetched updates took about 1.4-1.5 seconds. The dedicated
`train_megapose_gso_geometry_small_scratch_ray_rtxpro6000_70gb_336x252`
profile records this measured training budget while keeping two
non-persistent validation workers.

## Schedule interpretation

With the measured local profile's `max_img_per_gpu=288`, fixed six-view samples
pack 48 samples per rank. The 80 x 500 configuration is a 40,000-update ceiling,
not 80 finite-dataset passes: the sampler regenerates the query, compatible
reference plan, focal targets, and crop jitter from a fresh stateless seed.
It requests approximately 1.92 million six-view samples: 1.92 million query
images and 9.6 million reference images (11.52 million total model images) on
one rank. The train index has 3,543,486 crop-feasible object observations in
673,935 distinct RGB frames, so this is about 0.54 query draws per eligible
object observation, or 2.85 query uses per underlying RGB frame before
accounting for the different object crops in a frame.

The two training regimes have equal mixture weight, so the expectation is
960,000 generated samples per regime. Across 755 training objects this is
about 1,272 samples per object per regime (2,543 total). Scene-to-scene has
156,114 reference plans, about 6.15 expected draws per plan; render-to-scene
has 383,540 plans for the 755 train objects, about 2.50 draws per plan. These
are not byte-identical repeats because the compatible query, focal target, and
crop-center jitter remain stochastic.

The production-only schedule uses AdamW peak LR `1e-4` for both the randomly
initialized decoder/head and the zero-initialized ray projection, with a
500-update (1.25%) OneCycle warm-up. Keeping the groups equal makes the first
held-out generalization experiment easier to interpret; a faster ray branch
can be tested later only if ray-geometry metrics show that it lags. Validation
curves and the best checkpoint, rather than the nominal epoch number, determine
whether the architecture has converged.

MegaPose-GSO object-pose validation keeps dataset-wide and macro-object
aggregates but sets `report_per_object: false`. This avoids approximately seven
TensorBoard scalar cards per held-out object (about 2,646 cards across two
189-object loaders) without disabling ADD/ADD-S, normalized pose, rotation,
translation, or alignment summaries.

## Launch

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_megapose_gso_geometry_small_scratch_ray_rtxpro6000_70gb_336x252 \
  data=megapose_gso_geometry_n5_k1_masked \
  name=megapose_gso_geometry_small_scratch_ray_n5k1_336
```
