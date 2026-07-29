# 2026-07-29: A40 Recenter-Zoom, Paired Query, And Ray Conditioning

## Status

This note mines TensorBoard event scalars from four A40 recenter-zoom runs:

- `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/lmgeo_a40_recenter_zoom_k1`
- `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/lmgeo_a40_recenter_zoom_plus_original_k1`
- `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/lmgeo_a40_recenter_zoom_ray_k1`
- `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/lmgeo_a40_recenter_zoom_plus_original_ray_k1`

Only TensorBoard event files were present in the three new run folders. Config
names below are therefore taken from the documented launch profiles in
[`../lmgeo_paired_query_views.md`](../lmgeo_paired_query_views.md) and
[`../lmgeo_ray_conditioning.md`](../lmgeo_ray_conditioning.md). All validation
counts here are the half-count A40 validation traces: `723` real queries and
`800` held-out PBR queries.

Related notes:

- [2026-07-24 paired-query implementation](2026-07-24_lmgeo_paired_query_implementation.md)
- [2026-07-27 A40 ablation comparison](2026-07-27_lmgeo_a40_new_ablation_comparison.md)
- [LMGeo paired query views](../lmgeo_paired_query_views.md)
- [LMGeo ray conditioning](../lmgeo_ray_conditioning.md)

## Run Setup

| Run | Train config | Data config | Purpose |
| --- | --- | --- | --- |
| `lmgeo_a40_recenter_zoom_k1` | `train_lmgeo_finetune_a40_46gb_recenter_zoom_k1` | `lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1` | Control: one recentered+zoomed K=1 query, no ray conditioning. |
| `lmgeo_a40_recenter_zoom_plus_original_k1` | `train_lmgeo_finetune_a40_46gb_recenter_zoom_k1` | `lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1` | Pair the crop query with the original uncropped query, no ray conditioning. |
| `lmgeo_a40_recenter_zoom_ray_k1` | `train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1` | `lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1` | Crop-only control plus ray-map conditioning. |
| `lmgeo_a40_recenter_zoom_plus_original_ray_k1` | `train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1` | `lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1` | Pair original+crop queries and add ray-map conditioning. |

Last logged train scalars:

| Run | Last train step | Train loss | Views/sample | Images/rank | Ray LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `recenter_zoom_k1` | 15000 | 0.0177 | 9.61 | 23.90 | - |
| `plus_original_k1` | 10000 | 0.0194 | 11.09 | 22.69 | - |
| `ray_k1` | 10500 | 0.0185 | 10.15 | 23.23 | `2.50e-6` |
| `plus_original_ray_k1` | 8500 | 0.0203 | 11.29 | 22.49 | `4.59e-6` |

## Main Pose Results

Epoch is inferred as `step / 500 - 1`.

| Run | Split | Count | Best ADD(-S)<0.1d | Epoch | Median d @best | Rot med @best | Trans med @best | Last ADD(-S) | Last median d |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `recenter_zoom_k1` | `real_test` | 723 | 6.4% | 17 | 0.565 | 4.7 deg | 0.089 m | 5.5% | 0.654 |
| `recenter_zoom_k1` | `real_test_ref16` | 723 | 6.1% | 17 | 0.590 | 4.4 deg | 0.097 m | 4.3% | 0.631 |
| `recenter_zoom_k1` | `pbr_new_val` | 800 | 48.8% | 29 | 0.104 | 1.4 deg | 0.019 m | 48.8% | 0.104 |
| `recenter_zoom_k1` | `pbr_new_val_ref16` | 800 | 53.1% | 26 | 0.094 | 1.1 deg | 0.018 m | 52.1% | 0.095 |
| `plus_original_k1` | `real_test` | 723 | 27.1% | 6 | 0.200 | 5.7 deg | 0.034 m | 21.4% | 0.200 |
| `plus_original_k1` | `real_test_ref16` | 723 | 25.6% | 6 | 0.173 | 4.8 deg | 0.030 m | 16.5% | 0.238 |
| `plus_original_k1` | `pbr_new_val` | 800 | 59.9% | 18 | 0.073 | 1.5 deg | 0.015 m | 57.1% | 0.085 |
| `plus_original_k1` | `pbr_new_val_ref16` | 800 | 67.9% | 18 | 0.063 | 1.2 deg | 0.012 m | 63.9% | 0.075 |
| `ray_k1` | `real_test` | 723 | 7.9% | 17 | 0.462 | 4.9 deg | 0.074 m | 5.9% | 0.482 |
| `ray_k1` | `real_test_ref16` | 723 | 8.6% | 17 | 0.410 | 4.5 deg | 0.069 m | 4.7% | 0.472 |
| `ray_k1` | `pbr_new_val` | 800 | 46.0% | 19 | 0.111 | 1.6 deg | 0.020 m | 45.1% | 0.111 |
| `ray_k1` | `pbr_new_val_ref16` | 800 | 51.5% | 20 | 0.095 | 1.2 deg | 0.018 m | 51.5% | 0.095 |
| `plus_original_ray_k1` | `real_test` | 723 | 23.9% | 13 | 0.197 | 5.4 deg | 0.035 m | 18.3% | 0.216 |
| `plus_original_ray_k1` | `real_test_ref16` | 723 | 22.1% | 3 | 0.197 | 5.1 deg | 0.035 m | 15.5% | 0.226 |
| `plus_original_ray_k1` | `pbr_new_val` | 800 | 52.7% | 16 | 0.092 | 1.6 deg | 0.018 m | 52.7% | 0.092 |
| `plus_original_ray_k1` | `pbr_new_val_ref16` | 800 | 61.8% | 16 | 0.077 | 1.3 deg | 0.014 m | 61.8% | 0.077 |

## Paired-Query Diagnostics

These are logged only for the `plus_original` data profile. `query_original`
evaluates the uncropped query branch, and `query_crop_canonical` evaluates the
crop after transforming its prediction into the canonical/original query frame.
The disagreement metrics compare the two query predictions for the same scene
frame.

| Run | Split | Last original ADD(-S) | Last crop ADD(-S) | Canonical rot disagreement | Canonical trans disagreement | Pair reproj median | Dense angular median | World 3D median |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plus_original_k1` | `real_test` | 20.6% | 21.4% | 2.15 deg | 0.132d | 31.7 px | 0.413 deg | 0.022 m |
| `plus_original_k1` | `pbr_new_val` | 53.0% | 57.1% | 1.15 deg | 0.085d | 16.1 px | 0.130 deg | 0.013 m |
| `plus_original_ray_k1` | `real_test` | 13.6% | 18.3% | 2.38 deg | 0.150d | 28.8 px | 0.449 deg | 0.023 m |
| `plus_original_ray_k1` | `pbr_new_val` | 51.8% | 52.7% | 1.13 deg | 0.093d | 16.8 px | 0.131 deg | 0.014 m |

The paired metrics improve strongly from the first validation to the last in
both runs. For example, in `plus_original_k1` real-test canonical rotation
disagreement drops from `8.95 deg` to `2.15 deg`, and canonical translation
disagreement drops from `0.546d` to `0.132d`. The ray-conditioned paired run
also improves, but its final pair disagreement is slightly worse than the
non-ray paired run.

## Ray-Conditioning Diagnostics

These metrics exist only for ray-conditioned runs. `query` is the recentered
crop. `query_context` is the uncropped original query in the paired run.

| Run | Split | Role | Views | Reproj median first -> last | Best reproj median | Angular median last | fx/fy rel last | Valid |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ray_k1` | `real_test` | reference | 3615 | 2.78 -> 0.79 | 0.70 @e16 | 0.088 deg | 0.5%/0.5% | 1.0 |
| `ray_k1` | `real_test` | query | 723 | 21.35 -> 15.79 | 13.08 @e17 | 0.583 deg | 9.1%/9.2% | 1.0 |
| `ray_k1` | `pbr_new_val` | reference | 4000 | 2.32 -> 0.77 | 0.63 @e4 | 0.085 deg | 0.5%/0.5% | 1.0 |
| `ray_k1` | `pbr_new_val` | query | 800 | 17.93 -> 2.92 | 2.63 @e18 | 0.098 deg | 1.9%/1.9% | 1.0 |
| `plus_original_ray_k1` | `real_test` | reference | 3615 | 2.65 -> 0.87 | 0.80 @e14 | 0.097 deg | 1.0%/1.0% | 1.0 |
| `plus_original_ray_k1` | `real_test` | query | 723 | 13.64 -> 4.02 | 3.18 @e15 | 0.139 deg | 2.8%/2.6% | 1.0 |
| `plus_original_ray_k1` | `real_test` | query_context | 723 | 6.55 -> 5.57 | 2.54 @e2 | 0.549 deg | 3.2%/3.5% | 1.0 |
| `plus_original_ray_k1` | `pbr_new_val` | reference | 4000 | 2.09 -> 0.72 | 0.67 @e6 | 0.080 deg | 0.5%/0.5% | 1.0 |
| `plus_original_ray_k1` | `pbr_new_val` | query | 800 | 14.45 -> 2.15 | 2.02 @e15 | 0.071 deg | 1.2%/1.2% | 1.0 |
| `plus_original_ray_k1` | `pbr_new_val` | query_context | 800 | 9.29 -> 1.05 | 1.05 @e16 | 0.107 deg | 0.5%/0.5% | 1.0 |

The ray branch is clearly learning something real: reference rays become
sub-pixel, held-out PBR crop rays become very good, and paired original+crop
training reduces real crop ray reprojection error from `13.64 px` to `4.02 px`.
The remaining gap is specifically real-query crop geometry: ray-only real-query
reprojection remains `15.79 px` at the last point, far worse than PBR.

## Per-Object Real Test

Values are taken at each run's own best aggregate `real_test` ADD(-S) epoch.

### ADD(-S)<0.1d

| Object | Recenter | Plus original | Ray | Plus original + ray |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.5% | 7.0% | 4.7% | 18.6% |
| 5 | 0.0% | 48.1% | 1.9% | 51.0% |
| 6 | 8.5% | 4.9% | 6.1% | 4.9% |
| 8 | 0.0% | 61.4% | 0.0% | 28.7% |
| 9 | 0.0% | 5.6% | 0.0% | 2.2% |
| 10 | 18.9% | 18.9% | 28.4% | 17.9% |
| 11 | 20.0% | 60.0% | 23.1% | 67.7% |
| 12 | 5.0% | 11.9% | 4.0% | 7.9% |

### Median Normalized Pose Error

| Object | Recenter | Plus original | Ray | Plus original + ray |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.391 | 0.304 | 0.388 | 0.205 |
| 5 | 0.934 | 0.106 | 0.603 | 0.098 |
| 6 | 0.261 | 0.378 | 0.286 | 0.296 |
| 8 | 0.809 | 0.078 | 0.482 | 0.148 |
| 9 | 0.618 | 0.295 | 0.590 | 0.350 |
| 10 | 0.345 | 0.514 | 0.281 | 0.496 |
| 11 | 0.358 | 0.082 | 0.197 | 0.075 |
| 12 | 0.524 | 0.255 | 0.483 | 0.218 |

## Per-Object Held-Out PBR

Values are taken at each run's own best aggregate `pbr_new_val` ADD(-S) epoch.

### ADD(-S)<0.1d

| Object | Recenter | Plus original | Ray | Plus original + ray |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 22.5% | 27.0% | 19.8% | 18.0% |
| 5 | 43.3% | 72.2% | 52.2% | 54.4% |
| 6 | 42.3% | 39.4% | 32.7% | 43.3% |
| 8 | 53.9% | 88.8% | 58.4% | 70.8% |
| 9 | 33.3% | 39.2% | 25.5% | 31.4% |
| 10 | 83.3% | 86.3% | 78.4% | 81.4% |
| 11 | 75.0% | 83.7% | 76.9% | 85.6% |
| 12 | 37.8% | 50.0% | 27.6% | 41.8% |

### Median Normalized Pose Error

| Object | Recenter | Plus original | Ray | Plus original + ray |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.186 | 0.170 | 0.185 | 0.163 |
| 5 | 0.117 | 0.057 | 0.096 | 0.090 |
| 6 | 0.118 | 0.117 | 0.128 | 0.112 |
| 8 | 0.095 | 0.045 | 0.088 | 0.063 |
| 9 | 0.130 | 0.117 | 0.167 | 0.134 |
| 10 | 0.051 | 0.039 | 0.051 | 0.059 |
| 11 | 0.049 | 0.046 | 0.048 | 0.052 |
| 12 | 0.136 | 0.103 | 0.173 | 0.116 |

## Interpretation

- The visible winner inside this recenter family is `plus_original_k1`, not ray
  conditioning. It gives the best real ADD(-S), best PBR ADD(-S), and best PBR
  ref16 result among the four runs. It also restores real-test performance from
  the collapsed crop-only control (`6.4%`) back to A40 baseline/corr territory
  (`27.1%`).
- The crop-only `ray_k1` run is not enough. It improves the crop-only real
  median error from `0.565d` to `0.462d`, but strict real ADD(-S) remains very
  low (`7.9%`) and PBR ADD(-S) slightly drops relative to the crop-only control.
- Ray conditioning does what the auxiliary diagnostics ask it to do, especially
  for PBR and for paired original+crop real queries. However, better ray
  geometry does not yet translate into better pose accuracy. The likely reason
  is that the dominant failure is not just missing intrinsics/crop geometry; the
  model also needs the original query context or a stronger object cue to align
  the crop with the rendered references.
- Adding ray conditioning on top of paired original+crop is mixed: ray metrics
  become much cleaner, but pose metrics are lower than non-ray paired training.
  This can mean the ray branch is useful but not tuned yet, or that the current
  conditioning injection shifts the pretrained representation enough to hurt
  pose before the rest of the model adapts.
- Object-wise, paired original+crop mainly rescues objects 5, 8, and 11, which
  are exactly the objects that collapse in the crop-only real run. Object 10 is
  the exception where crop-only plus ray has the highest strict recall, but this
  object is symmetry-sensitive, so it should not drive the global conclusion by
  itself.

## Next Read

The best next diagnostic is not another crop-only ray run. The more useful
comparison is:

1. Re-run or extend `plus_original_k1` to the same final epoch budget as the
   older baselines, because its real peak is early but PBR is still strong late.
2. If ray conditioning is retried, make it a controlled branch on top of
   `plus_original_k1`: lower ray LR or longer warmup, then watch both
   `ray_geometry/*` and `paired_query/*` metrics. The target is not only better
   ray reprojection; it must also reduce crop/original pose disagreement and
   improve ADD(-S).
3. Keep the masked-query result as the upper-bound reference. `plus_original_k1`
   is promising but still far below the oracle masked-query run on real images.
