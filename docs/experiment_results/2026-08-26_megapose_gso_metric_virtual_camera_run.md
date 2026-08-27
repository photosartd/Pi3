# 2026-08-26 MegaPose-GSO metric-scale virtual-camera run

Status: complete. The 336x252 production run finished all 15,000 updates and
was evaluated on its native MegaPose-GSO validation sets and the full matched
LM-O `new_val` render-to-scene set.

## Setup and checkpoint choice

| Field | Value |
| --- | --- |
| Run | `megapose_gso_metric_virtual_ray_336x252_20260825_165554` |
| Train config | `train_megapose_gso_geometry_pi3_finetune_ray_metric_virtual_rtxpro6000_70gb_336x252` |
| Data config | `megapose_gso_geometry_n5_k1_masked_metric_virtual_query` |
| Initialization | released full Pi3 checkpoint; frozen DINOv2-L encoder |
| Trainable parameters | 588,397,656, including the ray projection |
| Views | N=5 references and K=1 query; 50/50 scene/render training mixture |
| Input | masked RGB and depth targets, calibrated rays, 336x252 (432 patches/view) |
| Virtual query | object-centred calibrated warp; log-uniform requested zoom in [1, 5], safely capped |
| Validation query | deterministic requested 3x zoom, safely capped |
| Scale mode | metric metres; no loss-time or primary-eval scale fitting |
| Schedule | 30 x 500 = 15,000 updates; OneCycle; peak LR `5e-6`, ray LR `1e-5` |
| Batch | 26 six-view samples = 156 images/update |
| Runtime / peak GPU | 15:50:01; 66,373 MiB allocated / 67,044 MiB reserved |

The run consumed 390,000 sampled sequences, or 2.34 million model images. It
completed without OOM, NaN, exception, or interruption.

`checkpoint_24` at update 12,500 is used as the downstream render-to-scene
checkpoint. It is jointly better than the final checkpoint on render median
pose (0.325d vs 0.351d), rotation (6.63 vs 6.76 degrees), translation (7.21 vs
7.82 cm), recall below 0.5d (66.67 vs 64.55%), and strict ADD (7.41 vs 6.35%).
The final checkpoint has a slightly lower render validation loss (0.1042 vs
0.1063), so lowest loss alone is not the right selection criterion here.

## Native MegaPose-GSO result

The table reports `checkpoint_24`, evaluated on the same deterministic 189
held-out objects as the historical 336x252 run. Alignment scale is exactly one.

| Metric | Scene references | Render references |
| --- | ---: | ---: |
| Validation loss | 0.1905 | **0.1063** |
| Normalized pose median / mean / p90 | 0.638 / 0.859 / 1.801d | **0.325 / 0.547 / 1.124d** |
| Query rotation median / mean | 8.67 / 24.49 deg | **6.63 / 24.31 deg** |
| Query translation median / mean | 13.83 / 18.71 cm | **7.21 / 11.91 cm** |
| Query camera-center median / mean | 19.87 / 39.47 cm | **14.96 / 35.47 cm** |
| Recall below 0.5d / 1d / 2d | 38.62 / 71.43 / 93.12% | **66.67 / 86.24 / 96.30%** |
| Strict ADD below 0.1d | 0.53% | **7.41%** |
| Reference rotation median | 5.05 deg | **3.74 deg** |
| Reference camera-center median | 11.98 cm | **2.84 cm** |

The final update is almost unchanged: scene median pose/rotation/translation is
0.637d/8.99 degrees/13.92 cm; render is 0.351d/6.76 degrees/7.82 cm. The
checkpoint choice is therefore a modest late-run task-metric regression, not
an unstable collapse.

## Effect relative to the previous 336x252 run

The resolution, released initialization, frozen encoder, N=5/K=1 protocol,
batch, schedule, ray conditioning, and scene/render mixture are the same as in
the [2026-08-18 run](2026-08-18_megapose_gso_geometry_pi3_finetune_ray_run.md).
The important changes are the virtual-camera query treatment and metric-scale
loss/evaluation.

| Metric | Old scene -> new scene | Old render -> new render |
| --- | ---: | ---: |
| Median normalized pose | 0.783 -> **0.638d** (-18.5%) | 0.439 -> **0.325d** (-26.0%) |
| Median query rotation | 15.45 -> **8.67 deg** (-43.9%) | 11.01 -> **6.63 deg** (-39.8%) |
| Median query translation | 17.01 -> **13.83 cm** (-18.7%) | 9.88 -> **7.21 cm** (-27.0%) |
| Query camera-center median | 34.15 -> **19.87 cm** (-41.8%) | 21.32 -> **14.96 cm** (-29.8%) |
| Recall below 0.5d | 33.33 -> **38.62%** (+5.29 pp) | 52.91 -> **66.67%** (+13.76 pp) |
| Strict ADD below 0.1d | **2.12** -> 0.53% | 4.76 -> **7.41%** |

The crop/warp changes the actual visual evidence much more than the nominal
336x252 tensor size suggests:

| Validation protocol | Old median occupancy / patches | New median occupancy / patches | Queries below 1% | Queries at least 8% |
| --- | ---: | ---: | ---: | ---: |
| GSO scene-to-scene | 2.46% / 10.6 | **13.07% / 56.5** (5.31x) | 24.87 -> **0.53%** | 12.17 -> **70.37%** |
| GSO render-to-scene | 1.42% / 6.1 | **10.71% / 46.3** (7.53x) | 35.45 -> **1.59%** | 1.06 -> **60.85%** |
| LM-O render-to-scene | 0.716% / 3.1 | **5.313% / 23.0** (7.42x) | 63.64 -> **3.89%** | 0.59 -> **32.70%** |

This is the central result of the experiment. Centring removes the unobserved
principal-point/rotation ambiguity, and zoom gives the model 5-8x more object
patches at the same network resolution. Rotation improves by about 40% on both
GSO regimes and by 63% on LM-O, exactly the axis expected from those changes.
The occupancy-controlled LM-O correlations remain strong after zoom, so small
objects are still a real failure axis rather than a dataset-identity artifact.

This is not a mathematically pure crop ablation: metric loss was enabled in the
same run. Metric loss is needed to make independent per-view zoom well-posed,
but it also changes optimization. The causal statement supported by the data
is therefore that the combined metric+virtual-camera regime works, with the
large occupancy increase and rotation-specific gain providing strong evidence
that cropping/centring supplies most of the geometric benefit.

## Full LM-O transfer result

The full evaluation covers 1,364 eligible queries and all eight LM-O objects.
References are five matched clean renders and queries use the same deterministic
virtual-camera transform. Primary alignment is rigid and fixes scale to one.
The diagnostic applies only one scalar estimated from GT reference depth; it
never aligns on query GT.

| Metric | Strict metric transfer | Reference-depth scale diagnostic |
| --- | ---: | ---: |
| Alignment scale median | **1.000** | 0.7809 |
| Normalized pose median / mean / p90 | 1.723 / 2.173 / 4.369d | **0.357 / 0.589 / 1.434d** |
| Query rotation median / mean | **4.84 / 14.68 deg** | **4.84 / 14.68 deg** |
| Query translation median / mean | 28.74 / 31.25 cm | **6.52 / 8.80 cm** |
| Recall below 0.5d / 1d / 2d | 7.48 / 24.41 / 57.11% | **61.29 / 81.96 / 95.75%** |
| Official ADD(-S) below 0.1d | 0.15% | **13.49%** |

Against the previous matched, scale-aligned LM-O transfer, the comparable
scale-corrected result improves median pose by 20.8%, rotation by 63.1%,
translation by 19.5%, recall below 0.5d by 7.70 points, and official ADD(-S) by
7.62 points. Detailed per-object results are in the
[dedicated transfer note](2026-08-26_megapose_gso_metric_virtual_to_lmo.md).

The strict LM-O failure is not a query-rotation failure. The predicted geometry
requires a median scale correction of 0.7809, meaning it is about 1.281x too
large. This closely follows the training/evaluation rig difference: GSO clean
references are nominally at 0.5 m, versus about 0.4 m for LM-O references. RGB,
masks, and rays do not reveal CAD size or metric distance for a new object. Once
the one reference-depth scalar is supplied, LM-O is close to native GSO render
validation (0.357 vs 0.325d pose and 6.52 vs 7.21 cm translation).

## Next experiments, in order

1. **Prove the scale mechanism cheaply.** Rectify or render LM-O references at
   the GSO 0.5 m rig and repeat strict scale-fixed evaluation. The decisive
   outcome is whether the required scale moves from 0.781 toward 1.
2. **Add reference-depth conditioning.** This has not yet been trained in this
   experiment: GT reference depth is currently a target and a post-hoc
   diagnostic, not an input. Encode metric `D K^-1 [u,v,1]` points plus a valid
   depth/known-role channel before the shared decoder. Clean render depth is
   available at deployment and the existing diagnostic gives a strong ceiling:
   LM-O translation changes from 28.74 to 6.52 cm when reference depth supplies
   only scale. It should primarily improve absolute radial/depth translation;
   it will not by itself fix large rotation/registration outliers.
3. **Then train the same regime at 560x420.** Patch count rises from 432 to
   1,200 per view (2.78x); the current LM-O median object would rise from about
   23 to about 64 visible patches. Based on the completed historical resolution
   comparison and because virtual zoom already recovers much of the signal, a
   conservative expectation is modest additional rotation/localization gains,
   not a scale cure. Resolution should follow, rather than precede, the cheap
   scale test and reference-depth input.
4. **Query depth is a separate RGB-D oracle/variant.** If available, it can
   directly constrain the query radial centre and is the likely route from the
   reference-depth 6-9 cm range toward roughly 3.5-5.5 cm. Keep it separate so
   the calibrated-RGB+reference-depth claim remains interpretable.

The best next production comparison is therefore reference-depth conditioning
at 336x252, followed by the identical model/data regime at 560x420. Raising the
resolution alone should not be expected to remove the 1.28x cross-dataset scale
prior.

## Artifacts

```text
training output:
/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_metric_virtual_ray_336x252_20260825_165554

TensorBoard:
/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_metric_virtual_ray_336x252_20260825_165554/tensorboard/megapose_gso_metric_virtual_ray_336x252_20260825_165554

selected render-to-scene checkpoint:
/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_metric_virtual_ray_336x252_20260825_165554/ckpts/checkpoint_24

full LM-O evaluation:
/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_new_val_metric_virtual_gso_checkpoint24_full_20260826

LM-O good/bad 3D-box gallery (16 examples; one good and one bad per object):
/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_metric_virtual_checkpoint24_bbox_gallery_good_bad_20260826
```

The gallery is ranked by the full-evaluation reference-depth-corrected pose
error, not by hand and not by the scale-biased strict score. `gallery_index.png`
is the contact sheet. Each example directory contains `overview.png`, separate
strict and scale-corrected GT/predicted 3D-box overlays, depth panels, inputs,
orthographic reconstruction, and machine-readable `metadata.json`.
