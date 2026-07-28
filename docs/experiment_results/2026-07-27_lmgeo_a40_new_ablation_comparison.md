# 2026-07-27: A40 Context, Recenter, GT-Visibility Pooling, And Masked-Recenter Ablations

## Status

This note mines four new TensorBoard event-log runs from:

`/vol/coro/dtrofimov/data/projects/gfm-6dof/runs`

The four folders contain TensorBoard event files only. I did not find Hydra
snapshots, raw text logs, JSON epoch logs, checkpoints, or parquet prediction
tables in these folders, so config/setup details below are inferred from run
names, scalar tags, and the current config files. The event scalars are still
enough to compare the main pose, correspondence, and per-object metrics.

Important caveats:

- Several A40 runs log validation counts of about half the full real-test set:
  around `722/723` real queries rather than the `1444/1445` full-set count used
  by local/full validation notes. Treat those A40 values as useful but not
  exact apples-to-apples with full-count runs.
- `lmgeo_a40_gt_vis_pool_alpha1_warmup3k` has two event files in one output
  directory. The first trace reaches epoch 29 but logs half-count validation.
  The second trace covers epochs 9..19 and logs full-count validation. The
  cross-run comparison below uses the full-count trace for this run.
- Epochs are inferred from validation global step as `epoch = step / 500 - 1`,
  matching the 500-step inner epoch convention used by these runs.

Related notes:

- [2026-07-17 high-resolution 12-view baseline](2026-07-17_lmgeo_560x420_rtx4090_12view.md)
- [2026-07-21 masked-query/context-reference run](2026-07-21_lmgeo_rtxpro6000_all_rgb_masked_context_refs.md)
- [2026-07-22 A40 baseline vs correspondence](2026-07-22_lmgeo_a40_baseline_vs_corr.md)
- [2026-07-22 next moves](2026-07-22_next_moves_after_a40_corr_masked.md)
- [2026-07-24 paired-query implementation](2026-07-24_lmgeo_paired_query_implementation.md)

## Run Setup

| Run | Event source | Inferred purpose | Notes |
| --- | --- | --- | --- |
| `lmgeo_a40_context_refs_28v` | `events.out.tfevents.1784676046.worker-10...` | A40 high-resolution context-reference training, roughly the context-ref version of the 28-view/baseline family. | Validation includes render-reference real/PBR plus PBR context-reference eval. |
| `lmgeo_a40_recenter_zoom_k1` | `events.out.tfevents.1784756052.worker-5...` | GT-bbox recenter+zoom K=1 diagnostic, unmasked query RGB. | References unchanged, query is object-centric recentered/zoomed. |
| `lmgeo_a40_gt_vis_pool_alpha1_warmup3k` | `events.out.tfevents.1784889635...` and `events.out.tfevents.1785005377...` | GT visibility mask used to weight `CameraHead` pooling with alpha warmup. | Two overlapping traces; full-count trace is used for main comparison. |
| `lmgeo_a40_recenter_zoom_masked_context_refs_k1` | `events.out.tfevents.1784815424.worker-2...` | Recenter+zoom K=1 plus object-masked RGB and context-reference training. | Narrow validation: real/PBR N=5,K=1 only. |

Logged dynamic train averages at the last event point:

| Run | Train loss | Samples/rank | Views/sample | Images/rank | Resolution |
| --- | ---: | ---: | ---: | ---: | --- |
| `context_refs_28v` | 0.1733 | 1.73 | 16.35 | 22.65 | `420x560` |
| `recenter_zoom_k1` | 0.0091 | 3.43 | 9.61 | 23.90 | `420x560` |
| `gt_vis_pool` | 0.0081 | 1.73 | 16.35 | 22.65 | `420x560` |
| `recenter_zoom_masked_context_refs_k1` | 0.1524 | 3.05 | 10.23 | 23.27 | `420x560` |

For `gt_vis_pool`, the logged visibility-pooling alpha reaches `1.0`; the last
train visibility patch fraction in the longest trace is about `0.052`.

## New Run Results

### A40 Context Refs 28v

This run is close to the A40 baseline family but adds context-reference
training/evaluation. It does not improve the render-reference real/PBR splits
over the A40 baseline, and the explicit context-reference validation remains
weak.

| Split | Query count | Best ADD(-S)<0.1d | Last ADD(-S) | Best median d | Last median d | Best rot med | Best trans med | Best ref Chamfer d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_test` | 722 | 24.7% @e19 | 23.1% @e29 | 0.242 @e25 | 0.249 | 7.5 deg | 0.037 m | 0.013 |
| `pbr_new_val` | 800 | 45.2% @e28 | 44.9% @e29 | 0.120 @e28 | 0.120 | 3.4 deg | 0.021 m | 0.014 |
| `pbr_new_val_context_refs` | 800 | 7.1% @e24 | 6.1% @e29 | 0.617 @e27 | 0.626 | 8.2 deg | 0.100 m | 0.658 |

Real-test per-object metrics at the best aggregate real epoch, epoch 19:

| Object | Count | ADD(-S)<0.1d | Median d | Rot med | Trans med |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 86 | 11.6% | 0.269 | 7.7 deg | 0.026 m |
| 5 | 103 | 45.6% | 0.106 | 2.7 deg | 0.021 m |
| 6 | 82 | 4.9% | 0.619 | 22.3 deg | 0.090 m |
| 8 | 102 | 57.8% | 0.079 | 3.7 deg | 0.020 m |
| 9 | 89 | 14.6% | 0.259 | 7.4 deg | 0.028 m |
| 10 | 94 | 2.1% | 1.812 | 106.6 deg | 0.370 m |
| 11 | 65 | 50.8% | 0.091 | 8.3 deg | 0.033 m |
| 12 | 101 | 9.9% | 0.286 | 4.8 deg | 0.041 m |

### A40 Recenter Zoom K1

This run is the cleanest test of the GT-bbox recenter+zoom idea without query
masking. It keeps held-out PBR K=1 performance in the same range as the A40
baseline, but real-test performance collapses. This points to a real-domain or
preprocessing/camera-conditioning mismatch rather than a general inability to
train on the crop.

| Split | Query count | Best ADD(-S)<0.1d | Last ADD(-S) | Best median d | Last median d | Best rot med | Best trans med | Best ref Chamfer d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_test` | 723 | 6.4% @e17 | 5.5% @e29 | 0.565 @e17 | 0.654 | 4.6 deg | 0.089 m | 0.011 |
| `real_test_ref16` | 723 | 6.1% @e17 | 4.3% @e29 | 0.582 @e0 | 0.631 | 4.2 deg | 0.097 m | 0.007 |
| `pbr_new_val` | 800 | 48.8% @e29 | 48.8% @e29 | 0.103 @e26 | 0.104 | 1.3 deg | 0.019 m | 0.011 |
| `pbr_new_val_ref16` | 800 | 53.1% @e26 | 52.1% @e29 | 0.094 @e26 | 0.095 | 1.1 deg | 0.018 m | 0.007 |

Real-test per-object metrics at the best aggregate real epoch, epoch 17:

| Object | Count | ADD(-S)<0.1d | Median d | Rot med | Trans med |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 86 | 3.5% | 0.391 | 4.1 deg | 0.039 m |
| 5 | 104 | 0.0% | 0.934 | 1.7 deg | 0.186 m |
| 6 | 82 | 8.5% | 0.261 | 9.2 deg | 0.041 m |
| 8 | 101 | 0.0% | 0.809 | 0.9 deg | 0.212 m |
| 9 | 89 | 0.0% | 0.618 | 7.2 deg | 0.067 m |
| 10 | 95 | 18.9% | 0.345 | 64.7 deg | 0.100 m |
| 11 | 65 | 20.0% | 0.358 | 4.6 deg | 0.094 m |
| 12 | 101 | 5.0% | 0.524 | 4.7 deg | 0.076 m |

### A40 GT Visibility Pool

The full-count trace reaches epoch 19. It tests whether GT visibility-weighted
camera pooling can recover the benefit of masked-query training while keeping
the RGB unmasked. It does not: real-test performance is essentially A40
baseline-level, not masked-query-level.

The half-count trace in the same directory reaches epoch 29 and is similar on
real (`24.0%` last ADD(-S)) but stronger on half-count PBR (`46.6%` K=1,
`74.3%` K5, `74.0%` K10). The table below uses the full-count trace.

| Split | Query count | Best ADD(-S)<0.1d | Last ADD(-S) | Best median d | Last median d | Best rot med | Best trans med | Best ref Chamfer d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_test` | 1444 | 24.4% @e19 | 24.4% @e19 | 0.260 @e16 | 0.261 | 7.9 deg | 0.039 m | 0.026 |
| `real_test_ref16` | 1444 | 27.1% @e19 | 27.1% @e19 | 0.221 @e18 | 0.226 | 7.1 deg | 0.034 m | 0.010 |
| `pbr_new_val` | 1600 | 43.4% @e19 | 43.4% @e19 | 0.122 @e18 | 0.123 | 3.2 deg | 0.023 m | 0.025 |
| `pbr_new_val_ref16` | 1600 | 50.7% @e19 | 50.7% @e19 | 0.098 @e19 | 0.098 | 2.7 deg | 0.019 m | 0.010 |
| `pbr_new_val_k5_subset` | 800 | 70.5% @e18 | 70.5% @e18 | 0.060 @e18 | 0.060 | 1.3 deg | 0.013 m | 0.017 |
| `pbr_new_val_k10_subset` | 1600 | 70.4% @e18 | 70.4% @e18 | 0.060 @e18 | 0.060 | 1.2 deg | 0.013 m | 0.017 |

Real-test per-object metrics at the best aggregate real epoch, epoch 19:

| Object | Count | ADD(-S)<0.1d | Median d | Rot med | Trans med |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 175 | 8.6% | 0.281 | 6.7 deg | 0.029 m |
| 5 | 199 | 59.8% | 0.082 | 2.2 deg | 0.016 m |
| 6 | 171 | 0.0% | 1.154 | 21.2 deg | 0.176 m |
| 8 | 200 | 61.0% | 0.077 | 2.5 deg | 0.020 m |
| 9 | 180 | 13.3% | 0.230 | 9.0 deg | 0.024 m |
| 10 | 180 | 2.8% | 2.083 | 69.7 deg | 0.407 m |
| 11 | 139 | 42.4% | 0.139 | 9.8 deg | 0.047 m |
| 12 | 200 | 4.0% | 0.337 | 6.0 deg | 0.047 m |

### A40 Recenter Zoom Masked Context Refs K1

This is the combined crop+mask+context-reference diagnostic. It performs worse
than either the masked-query run or the recenter-only PBR result. The crop and
mask combination appears to remove or distort useful context/calibration cues
under the current implementation.

| Split | Query count | Best ADD(-S)<0.1d | Last ADD(-S) | Best median d | Last median d | Best rot med | Best trans med | Best ref Chamfer d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_test` | 723 | 11.3% @e12 | 9.7% @e14 | 0.373 @e12 | 0.381 | 4.0 deg | 0.065 m | 0.017 |
| `pbr_new_val` | 800 | 25.1% @e11 | 24.0% @e14 | 0.179 @e11 | 0.199 | 1.9 deg | 0.034 m | 0.016 |

Real-test per-object metrics at the best aggregate real epoch, epoch 12:

| Object | Count | ADD(-S)<0.1d | Median d | Rot med | Trans med |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 86 | 12.8% | 0.251 | 5.6 deg | 0.025 m |
| 5 | 104 | 1.9% | 0.588 | 0.9 deg | 0.119 m |
| 6 | 82 | 8.5% | 0.240 | 4.8 deg | 0.037 m |
| 8 | 101 | 5.9% | 0.399 | 0.9 deg | 0.101 m |
| 9 | 89 | 5.6% | 0.529 | 5.4 deg | 0.058 m |
| 10 | 95 | 26.3% | 0.228 | 176.5 deg | 0.086 m |
| 11 | 65 | 26.2% | 0.302 | 8.5 deg | 0.083 m |
| 12 | 101 | 8.9% | 0.345 | 4.1 deg | 0.045 m |

## Cross-Run Main Comparison

The table uses best observed real-test ADD(-S) for each run, and best observed
median normalized error. `Q` is the logged real-test query count at the best
ADD(-S) point. The 224px low-res row is taken from the 2026-07-10 note because
the original event file was not present in the searched run folders.

| Run | Main change | Q | Best real ADD(-S) | Best median d | Best rot med | Best trans med | Best PBR K1 ADD(-S) | Other PBR/context |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-07-10 low-res | 224px first baseline | 1381 | 10.9% @e23 | 0.541 @e13 | 26.5 deg | 0.078 m | - | - |
| 2026-07-17 560x420 12-view | higher res, unmasked query | 1444 | 24.0% @e16 | 0.248 @e16 | 8.1 deg | 0.039 m | 41.1% | K5 68.5%, K10 67.4% |
| 2026-07-21 masked-query context refs | masked query/reference RGB, render refs for main val | 1444 | 41.3% @e24 | 0.124 @e23 | 4.5 deg | 0.021 m | 53.2% | context refs 12.6% |
| A40 baseline | 560x420, larger A40 view budget | 722 | 25.5% @e19 | 0.257 @e20 | 7.4 deg | 0.039 m | 48.8% | K5 76.2%, K10 80.4% |
| A40 corr lambda 0.3 | A40 baseline + DINO-weighted corr loss | 722 | 27.3% @e19 | 0.239 @e16 | 7.7 deg | 0.038 m | 48.1% | K5 77.0%, K10 79.3% |
| A40 context refs 28v | A40 context-ref training/eval | 722 | 24.7% @e19 | 0.242 @e25 | 7.5 deg | 0.037 m | 45.2% | context refs 7.1% |
| A40 recenter zoom K1 | GT-bbox recenter+zoom query | 723 | 6.4% @e17 | 0.565 @e17 | 4.6 deg | 0.089 m | 48.8% | - |
| A40 GT visibility pool | GT visibility-weighted camera pooling | 1444 | 24.4% @e19 | 0.260 @e16 | 7.9 deg | 0.039 m | 43.4% | K5 70.5%, K10 70.4% |
| A40 recenter zoom masked context | recenter+zoom + masked RGB + context refs | 723 | 11.3% @e12 | 0.373 @e12 | 4.0 deg | 0.065 m | 25.1% | - |

## Real-Test Per-Object Comparison

Each cell is the per-object value at that run's representative real-test epoch:
best real-test ADD(-S) epoch for all event-mined runs; final/best documented
snapshot for the 224px row. A40 half-count rows have about half the object
queries of the full-count rows.

### ADD(-S)<0.1d

| Object | 224px | 560x420 | Masked query | A40 base | A40 corr | Context refs | Recenter | GT vis pool | Recenter+masked |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6% | 6.9% | 21.7% | 8.1% | 15.1% | 11.6% | 3.5% | 8.6% | 12.8% |
| 5 | 22.6% | 55.8% | 80.9% | 68.0% | 65.0% | 45.6% | 0.0% | 59.8% | 1.9% |
| 6 | 0.0% | 1.2% | 15.2% | 3.7% | 0.0% | 4.9% | 8.5% | 0.0% | 8.5% |
| 8 | 33.0% | 52.0% | 79.0% | 63.7% | 71.6% | 57.8% | 0.0% | 61.0% | 5.9% |
| 9 | 3.3% | 7.8% | 14.4% | 11.2% | 12.4% | 14.6% | 0.0% | 13.3% | 5.6% |
| 10 | 2.0% | 3.3% | 29.4% | 2.1% | 2.1% | 2.1% | 18.9% | 2.8% | 26.3% |
| 11 | 10.0% | 54.0% | 48.2% | 33.8% | 43.1% | 50.8% | 20.0% | 42.4% | 26.2% |
| 12 | 6.5% | 11.0% | 33.5% | 5.0% | 3.0% | 9.9% | 5.0% | 4.0% | 8.9% |

### Median Normalized Pose Error

| Object | 224px | 560x420 | Masked query | A40 base | A40 corr | Context refs | Recenter | GT vis pool | Recenter+masked |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.901 | 0.356 | 0.172 | 0.299 | 0.285 | 0.269 | 0.391 | 0.281 | 0.251 |
| 5 | 0.212 | 0.092 | 0.059 | 0.073 | 0.068 | 0.106 | 0.934 | 0.082 | 0.588 |
| 6 | 1.294 | 0.792 | 0.181 | 0.819 | 0.743 | 0.619 | 0.261 | 1.154 | 0.240 |
| 8 | 0.160 | 0.095 | 0.057 | 0.071 | 0.065 | 0.079 | 0.809 | 0.077 | 0.399 |
| 9 | 0.573 | 0.250 | 0.198 | 0.275 | 0.223 | 0.259 | 0.618 | 0.230 | 0.529 |
| 10 | 1.545 | 1.104 | 0.235 | 1.873 | 1.365 | 1.812 | 0.345 | 2.083 | 0.228 |
| 11 | 1.129 | 0.091 | 0.104 | 0.126 | 0.116 | 0.091 | 0.358 | 0.139 | 0.302 |
| 12 | 0.447 | 0.280 | 0.131 | 0.366 | 0.498 | 0.286 | 0.524 | 0.337 | 0.345 |

## Held-Out PBR Per-Object Comparison

The tables below mirror the real-test object comparison, but use the
`pbr_new_val` K=1 split. Each run is sampled at its own best aggregate
`pbr_new_val` ADD(-S)<0.1d epoch. The 224px low-resolution run is omitted
because that run did not log held-out PBR validation. A40 half-count rows again
use `800` PBR queries, while full-count rows use `1600`.

Representative PBR K1 epochs:

| Run | Query count | Best PBR ADD(-S) | Epoch |
| --- | ---: | ---: | ---: |
| 560x420 | 1600 | 41.1% | 16 |
| Masked query | 1600 | 53.2% | 25 |
| A40 base | 800 | 48.8% | 21 |
| A40 corr | 800 | 48.1% | 20 |
| Context refs | 800 | 45.2% | 28 |
| Recenter | 800 | 48.8% | 29 |
| GT vis pool | 1600 | 43.4% | 19 |
| Recenter+masked | 800 | 25.1% | 11 |

### ADD(-S)<0.1d

| Object | 560x420 | Masked query | A40 base | A40 corr | Context refs | Recenter | GT vis pool | Recenter+masked |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.0% | 20.5% | 19.6% | 19.6% | 14.0% | 22.5% | 8.5% | 8.1% |
| 5 | 57.0% | 76.0% | 63.5% | 61.5% | 62.5% | 43.3% | 62.5% | 20.0% |
| 6 | 34.0% | 49.5% | 45.0% | 45.0% | 42.0% | 42.3% | 30.0% | 16.3% |
| 8 | 65.0% | 81.0% | 80.2% | 71.4% | 71.4% | 53.9% | 68.5% | 32.6% |
| 9 | 17.5% | 19.0% | 24.8% | 23.8% | 11.4% | 33.3% | 17.0% | 11.8% |
| 10 | 55.0% | 73.5% | 55.2% | 70.8% | 67.7% | 83.3% | 63.5% | 45.1% |
| 11 | 58.5% | 68.5% | 69.9% | 71.8% | 66.0% | 75.0% | 70.5% | 55.8% |
| 12 | 33.0% | 37.5% | 38.2% | 27.5% | 34.3% | 37.8% | 27.0% | 12.2% |

### Median Normalized Pose Error

| Object | 560x420 | Masked query | A40 base | A40 corr | Context refs | Recenter | GT vis pool | Recenter+masked |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.304 | 0.224 | 0.209 | 0.222 | 0.315 | 0.186 | 0.283 | 0.243 |
| 5 | 0.078 | 0.059 | 0.057 | 0.068 | 0.070 | 0.117 | 0.077 | 0.184 |
| 6 | 0.140 | 0.101 | 0.119 | 0.111 | 0.130 | 0.118 | 0.151 | 0.187 |
| 8 | 0.076 | 0.057 | 0.057 | 0.060 | 0.053 | 0.095 | 0.070 | 0.158 |
| 9 | 0.277 | 0.198 | 0.198 | 0.189 | 0.255 | 0.130 | 0.241 | 0.262 |
| 10 | 0.089 | 0.050 | 0.088 | 0.046 | 0.063 | 0.051 | 0.070 | 0.113 |
| 11 | 0.078 | 0.065 | 0.054 | 0.050 | 0.069 | 0.049 | 0.064 | 0.090 |
| 12 | 0.157 | 0.130 | 0.138 | 0.159 | 0.163 | 0.136 | 0.195 | 0.233 |

## Insights

- Query masking is still the strongest signal we have seen. The masked-query
  context-reference run continued beyond the earlier 2026-07-21 note and
  reaches `41.3%` real ADD(-S), `0.124d` median error, and `53.2%` held-out PBR
  K1 ADD(-S). None of the new A40 ablations match it.
- Context references do not yet replace clean render references. The new
  context-ref run is baseline-like on render-reference validation and very weak
  when evaluated with context refs as references: only `7.1%` on
  `pbr_new_val_context_refs`, with reference Chamfer around `0.66d`.
- Recenter+zoom alone is not a win on real images. It preserves/increases PBR
  K1 performance (`48.8%`) but collapses real-test ADD(-S) to `6.4%`. That
  says the crop/virtual-camera experiment is either introducing a real-domain
  shift, losing calibration/context that real images need, or still has a
  conditioning/intrinsics mismatch that is not visible on held-out PBR.
- Recenter+zoom plus query masking/context refs is worse than both useful
  parents. It reaches only `11.3%` real and `25.1%` PBR. This makes the current
  crop+mask combination a poor next baseline until we debug what signal it
  removes.
- GT visibility pooling in `CameraHead` does not reproduce the masked-query
  benefit. Even with GT masks and alpha at `1.0`, the full-count run is
  essentially A40-baseline level on real images (`24.4%`). This argues that the
  problem is not only uniform camera-head pooling over object/background
  tokens; useful/ harmful information is already entangled earlier in the
  representation or in the geometry/alignment pipeline.
- The correspondence loss remains a modest positive regularizer, not the main
  fix. It is still one of the few A40-side changes that improves real ADD(-S)
  over the A40 baseline, but the gain is small compared with oracle query
  masking.
- Object-wise, objects 5 and 8 remain the easiest real-test objects across
  normal high-resolution runs. Objects 6 and 10 remain structurally hard, but
  masking dramatically lowers their median normalized errors. Object 10's
  strict ADD(-S) and rotation remain awkward because the symmetry/pose
  ambiguity is not fully captured by a single aggregate metric.
- The A40 validation-count inconsistency should be fixed or at least explained
  before drawing tiny `1..2 pp` conclusions. Large effects are still meaningful,
  but small deltas between half-count A40 runs and full-count local runs should
  be treated as directional.

## Next Meaningful Moves

- First, fix or audit distributed validation aggregation so A40 runs report the
  full real/PBR counts consistently. This will make future A40-vs-local
  conclusions cleaner.
- Keep the deployable-mask direction alive. The oracle masked-query run is the
  clearest ceiling experiment so far; the next practical question is whether a
  predicted/available mask can recover most of that gain without hiding the
  entire query context from the network.
- Do not promote recenter+zoom as a default yet. Its real/PBR split is too
  suspicious. Use paired original+crop/ray-conditioning diagnostics to find
  whether the issue is loss of context, missing intrinsic conditioning,
  virtual-camera distribution shift, or a remaining crop geometry bug.
- Treat context references as a separate onboarding/reference problem. They may
  still be useful, but current context-reference validation shows the alignment
  side is not robust enough for them to replace clean render refs.
