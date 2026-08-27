# 2026-08-26 metric-scale virtual-query GSO to LM-O transfer

Status: implementation validated and full evaluation complete.

## Question and protocol

Evaluate the completed 336x252 MegaPose-GSO metric-scale/virtual-query model
on LM-O `new_val` render-to-scene, using the same N=5/K=1 geometry constraints,
masked RGB, ray conditioning, and deterministic 3x virtual-query zoom. LM-O has
only eight objects, so this evaluator deliberately enables per-object metrics.

| Field | Value |
| --- | --- |
| Checkpoint | `megapose_gso_metric_virtual_ray_336x252_20260825_165554/ckpts/checkpoint_24` |
| Train/data configs | `train_megapose_gso_metric_virtual_lmgeo_eval_rtxpro6000_70gb_336x252` / `lmgeo_new_val_geometry_render_n5_k1_masked_metric_virtual_query` |
| Query split | LM-O `new_val` |
| References | five clean LM-O `train/<object_id>` renders selected by the existing geometry plans |
| Query | one masked held-out PBR image, centred by a calibrated virtual-camera rotation and safely zoomed toward 3x |
| Resolution | 336x252, 432 patch-equivalent cells per view |
| Primary alignment | reference-only rigid alignment, scale fixed to 1 |
| Diagnostic alignment | reference-only rotation/translation plus one scalar estimated from GT reference depth |
| Eligible samples | 1,364 / 1,600 (85.25%); all 236 rejections are the established query-crop quality gate |

The primary result is intentionally strict: it tests whether metric scale learned
on GSO transfers to LM-O. The scale-corrected result uses the same predictions
and never uses query GT for alignment. It is a diagnostic of pose geometry after
removing exactly one reference-derived scale; it is not the claimed metric result.

The virtual transform updates RGB, mask, z-depth, intrinsics, rays, `T_C_O`, and
`camera_pose` consistently. Evaluation is against GT expressed in that virtual
camera. Because the same virtual rotation acts on prediction and GT, angular and
Euclidean pose errors remain directly comparable to the historical unwarped run.

## Implementation and validation

The implementation is additive. Historical GSO and LMGeo profiles are unchanged.

- `lmgeo_new_val_geometry_render_n5_k1_masked_metric_virtual_query.yaml` adds the
  same query virtual-camera transform used for training.
- `train_megapose_gso_metric_virtual_lmgeo_eval_rtxpro6000_70gb_336x252.yaml`
  restores `checkpoint_24`, fixes primary alignment scale to one, and adds a
  separately named reference-depth scale diagnostic. Both object-pose plugins
  emit per-object metrics.
- The resolved config is covered by a Hydra regression test.

Validation evidence:

- focused LMGeo/virtual-camera/metric-loss suite: 14/14 passed;
- full CPU suite: 233/233 passed;
- a real materialized sample had five untouched references, one 3x virtual
  query, centred principal point, 6,944 valid query-depth pixels, and
  `camera_pose == inv(T_C_O)` after the transform;
- one-batch GPU smoke passed before the full run;
- the full 1,364-query evaluation completed in 11 batches and 3m10s, with
  24,877 MiB peak allocated and 29,880 MiB peak reserved GPU memory;
- all batches were eligible for all four metrics; there were no underconstrained
  alignments, exceptions, or non-finite values.

## Aggregate results

| Metric | Strict metric scale | Reference-depth scale diagnostic |
| --- | ---: | ---: |
| Alignment scale, median | **1.0000** | 0.7809 |
| Normalized pose, median / mean / p90 | 1.723 / 2.173 / 4.369d | **0.357 / 0.589 / 1.434d** |
| Recall below 0.5d / 1d / 2d | 7.48 / 24.41 / 57.11% | **61.29 / 81.96 / 95.75%** |
| Rotation, median / mean | **4.84 / 14.68 deg** | **4.84 / 14.68 deg** |
| Translation, median / mean | 0.287 / 0.313 m | **0.065 / 0.088 m** |
| Translation lateral / depth median | 0.034 / 0.278 m | **0.023 / 0.054 m** |
| Strict ADD below 0.1d | 0.15% | 6.60% |
| Official LM-O ADD(-S) below 0.1d | 0.15% | **13.49%** |

The metric validation loss is 0.198664 (`local=0.100823`,
`translation=0.007917`, `rotation=0.186702`). The strict camera diagnostics
agree with the pose result: query center is 0.338 m median, with 0.279 m radial
and 0.097 m tangential components. The reference center is 0.0995 m median.
Reference/query local-depth relative errors are both about 28%, with positive
depth bias, which is the signature of one common scale error rather than a
query-only translation collapse.

The reference-depth diagnostic uses all five references in every sample. Its
mean within-sample log-MAD is 0.00718, only about 0.72% multiplicative spread,
so the five independently predicted reference views strongly agree on the
required correction.

## Per-object results

`Pose` is the configured normalized ADD for ordinary objects and ADD-S for the
symmetric objects 10 and 11. `Official` likewise uses ADD-S only for IDs 10/11.
Rotation is identical in the strict and scale-corrected columns because a scale
cannot change rotation.

| Object | N | Median occupancy | Rotation | Strict translation | Corrected translation | Strict pose | Corrected pose | Corrected official <0.1d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 180 | 2.66% | 8.28 deg | 43.75 cm | 12.19 cm | 4.282d | 1.203d | 0.56% |
| 5 | 153 | 8.87% | **2.32 deg** | 24.05 cm | **3.40 cm** | 1.195d | **0.168d** | **26.14%** |
| 6 | 169 | 4.30% | 3.52 deg | 34.04 cm | 6.03 cm | 2.205d | 0.420d | 5.33% |
| 8 | 156 | 12.35% | 2.74 deg | **15.13 cm** | 5.91 cm | **0.578d** | 0.227d | 11.54% |
| 9 | 185 | 3.64% | 5.98 deg | 36.77 cm | 8.37 cm | 3.366d | 0.777d | 0.00% |
| 10 (sym.) | 174 | 7.68% | 7.50 deg | 27.38 cm | 6.41 cm | 1.369d | **0.165d** | **25.29%** |
| 11 (sym.) | 181 | 3.35% | 5.83 deg | 25.46 cm | 4.85 cm | 1.130d | **0.118d** | **39.23%** |
| 12 | 166 | 5.64% | 7.00 deg | 40.26 cm | 7.06 cm | 2.759d | 0.508d | 0.60% |

After scale correction the strongest normalized-pose objects are 11, 10, and
5. Object 5 is strongest without symmetry assistance. Object 1 remains the
clear hardest case: virtual zoom reduces its rotation substantially, but its
translation becomes worse than in the old scale-aligned model. Objects 9 and
12 are the next weakest and have effectively no strict 0.1d success despite
reasonable median rotation. Their remaining errors are translation/tail and
shape-sensitive ADD errors, not failure to estimate coarse orientation.

## Comparison with the previous LM-O transfer

The fair geometry comparison is the new reference-depth diagnostic against the
old reference-depth-aligned run. Comparing the strict metric result to an old
scale-aligned result would mix two different questions.

| Metric | Old matched GSO model | New metric+virtual model | Change |
| --- | ---: | ---: | ---: |
| Pose median / mean / p90 | 0.450 / 0.848 / 2.193d | **0.357 / 0.589 / 1.434d** | -20.8 / -30.6 / -34.6% |
| Rotation median / mean | 13.14 / 33.34 deg | **4.84 / 14.68 deg** | -63.1 / -56.0% |
| Translation median / mean | 8.10 / 12.66 cm | **6.52 / 8.80 cm** | -19.5 / -30.5% |
| Recall below 0.5d / 1d / 2d | 53.59 / 74.12 / 88.64% | **61.29 / 81.96 / 95.75%** | +7.70 / +7.84 / +7.11 pp |
| Official ADD(-S) below 0.1d | 5.87% | **13.49%** | +7.62 pp |

Every object improves median rotation by 46-72%. After scale correction, seven
of eight objects also improve median translation by 10-45%; object 1 is the
exception (+24.5% translation, +23.7% normalized pose). This is much stronger
evidence than an aggregate improvement alone that centring/zoom and metric GSO
training transferred useful geometry to unseen LM-O objects.

The scale-corrected LM-O result is also close to this checkpoint's native GSO
render validation at update 12,500: pose 0.357d versus 0.325d, rotation 4.84
versus 6.63 degrees, translation 6.52 versus 7.21 cm, and recall below 0.5d
61.29 versus 66.67%. LM-O retains a heavier p90 tail (1.434d versus 1.124d), but
there is no geometric cross-dataset collapse once one scalar is supplied.

## Occupancy and calibration audit

The deterministic virtual zoom moves LM-O's median query occupancy from the
historical 0.716% to 5.313%, or from about 3.1 to 23.0 patch-equivalent cells.
Only 53/1,364 queries (3.89%) remain below 1%, while 446 (32.70%) are at least
8%. Query ray recovery is healthy: 0.091-degree median angular error, 1.42-pixel
reprojection error, 2.58/2.27% mean `fx/fy` error, and 100% valid fits.

Size remains a failure axis even after zoom. Spearman correlation between log
occupancy and strict error is -0.508 for rotation, -0.530 for translation,
-0.524 for normalized pose, and -0.722 for camera center. The within-object
coefficients remain -0.437, -0.443, -0.436, and -0.699 respectively, so this is
not merely a difference between easy and hard object identities. The few
queries below 0.5% are still catastrophic (72.84-degree median rotation), while
the well-populated >=8% bin reaches 3.04 degrees.

The strict translation occupancy bins should not be read as pure query
registration quality: their optical-axis component is dominated by the global
1.28x scale bias. The scale-corrected aggregate and per-object translation are
the cleaner geometry diagnostic.

## Main conclusion

The addition works on its intended geometric axis. Virtual centring/zoom plus
the new training regime produces a large LM-O rotation gain and materially
better scale-corrected translation/ADD, including on all eight unseen object
identities.

It does **not** yet demonstrate cross-dataset metric-scale generalization. The
median required correction is 0.7809, so the raw prediction is approximately
`1 / 0.7809 = 1.281x` too large. This closely matches the known reference-rig
shift: GSO clean renders are nominally at 0.5 m, while LM-O train references are
at 0.4 m (`0.4 / 0.5 = 0.8`). The model receives RGB, masks, and rays, but not
CAD size, pose, or depth as input; focal calibration alone cannot resolve the
object-size/distance ambiguity for unseen objects. It appears to retain the GSO
reference-distance prior. This explanation is strongly consistent with the
numbers but remains an inference rather than a controlled causal ablation.

The next decisive test is cheap: render or rectify the LM-O references to the
GSO 0.5 m metric rig and repeat the strict scale-fixed evaluation. If the
alignment scale moves from 0.781 toward 1 and strict translation collapses
toward the 6.5 cm diagnostic value, the reference-rig prior is confirmed. If
metric reference depth is acceptable at deployment, the already implemented
reference-depth scalar is instead a practical calibrated protocol—but then the
claim is depth-calibrated transfer, not pure RGB+ray metric inference.

## Artifacts

```text
output:
/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_new_val_metric_virtual_gso_checkpoint24_full_20260826

TensorBoard:
/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_new_val_metric_virtual_gso_checkpoint24_full_20260826/tensorboard/lmgeo_new_val_metric_virtual_gso_checkpoint24_full_20260826

occupancy rows, summaries, correlations, and plot:
/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_new_val_metric_virtual_gso_checkpoint24_full_20260826/query_occupancy/lmo_new_val_geometry_render_n5_k1/step_00000025

run log:
/home/dtrofimov/repositories/Pi3/logs/lmgeo_new_val_metric_virtual_gso_checkpoint24_full_20260826.log
```
