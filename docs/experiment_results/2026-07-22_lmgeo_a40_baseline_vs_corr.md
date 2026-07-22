# 2026-07-22: A40 Baseline vs DINO-Weighted Correspondence Loss

## Status

Both runs were mined from TensorBoard event files in the cluster output
directories. No Hydra job config, raw text log, JSON epoch log, or parquet
prediction tables were present in these two run folders, so the setup below is
partly inferred from the run names, scalar tags, and the current matching config
files.

The baseline event has complete validation through epoch 21. The correspondence
run has a complete six-split validation through epoch 20; the event also
contains a partial step-11000 validation for `real_test`, `real_test_ref16`, and
`pbr_new_val`, but not for all extended PBR splits. Fair baseline-vs-corr
comparisons below therefore use common complete epoch 20 unless stated
otherwise.

## Setup

| Field | A40 baseline | A40 correspondence |
| --- | --- | --- |
| Run name | `lmgeo_a40_baseline` | `lmgeo_a40_corr_lambda0p3` |
| Output dir | `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3/lmgeo_a40_baseline` | `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3/lmgeo_a40_corr_lambda0p3` |
| TensorBoard event | `events.out.tfevents.1784503328.worker-10.GPU.CIT-EC.NET.2645797.0` | `events.out.tfevents.1784511803.worker-8.GPU.CIT-EC.NET.2287506.0` |
| Train config | inferred `train_lmgeo_finetune_a40_46gb` | inferred `train_lmgeo_finetune_a40_46gb_corr` |
| Data config | inferred `lmgeo_trainpbr45_real_and_new_val` | inferred `lmgeo_trainpbr45_real_and_new_val` |
| Resolution | `560 x 420` | `560 x 420` |
| Query/reference RGB | query unmasked, reference masked | query unmasked, reference masked |
| Training view budget | total views `6..28`, refs `5..27`, queries `1..25` | same |
| Logged train averages | about `17.3` views/sample, `23.0` images/rank | about `17.3` views/sample, `23.0` images/rank |
| Validation splits | real/PBR `N=5,K=1`, ref16, PBR `K=5`, PBR `K=10` | same |
| Extra loss | none | correspondence weight `0.3`, max `512` pairs, up to `2` refs/query, detached DINO-sim weights from layer 17 |

## Baseline Results

The A40 baseline is the same high-resolution unmasked-query family as the
2026-07-17 RTX 4090 run, but it is not a bit-for-bit repeat: the A40 profile
uses the larger 46 GB view budget and the event reached later epochs.

| Val split | Best ADD(-S)<0.1d | Last ADD(-S) | Best median d | Last median d | Best rot med | Best trans med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_test` | 25.5% @e11 | 23.7% @e21 | 0.257 @e20 | 0.269 | 7.4 deg | 0.039 m |
| `real_test_ref16` | 28.8% @e11 | 26.6% @e21 | 0.223 @e19 | 0.243 | 7.1 deg | 0.034 m |
| `pbr_new_val` | 48.8% @e21 | 48.8% @e21 | 0.104 @e21 | 0.104 | 3.2 deg | 0.019 m |
| `pbr_new_val_ref16` | 54.1% @e20 | 53.5% @e21 | 0.086 @e20 | 0.089 | 2.6 deg | 0.016 m |
| `pbr_new_val_k5_subset` | 76.2% @e21 | 76.2% @e21 | 0.049 @e21 | 0.049 | 1.1 deg | 0.010 m |
| `pbr_new_val_k10_subset` | 80.4% @e21 | 80.4% @e21 | 0.047 @e21 | 0.047 | 0.9 deg | 0.009 m |

Compared with the 2026-07-17 12-view RTX 4090 note, the real-test result is
close but a little higher (`25.5%` best ADD(-S) versus `24.0%`). The PBR
multi-query subsets are much stronger here (`76.2%/80.4%` for K5/K10 versus
`68.5%/67.4%`), which is consistent with the larger A40 training view budget
and longer run rather than a pure cluster-vs-local effect.

## Correspondence Results

The correspondence run adds the geometry-based query-to-reference consistency
loss, weighted by detached DINO similarity. The auxiliary loss was active and
learned: logged train correspondence loss dropped from `0.207` to `0.007`;
weighted contribution dropped from `0.062` to `0.002`; accepted pairs stayed
around `186..192` per logged train sample; DINO similarity and weight stayed
stable around `0.34` and `0.72`.

| Val split | Best ADD(-S)<0.1d | Last ADD(-S) | Best median d | Last median d | Best rot med | Best trans med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_test` | 27.3% @e19 | 26.5% @partial e21 | 0.239 @e16 | 0.260 | 7.7 deg | 0.038 m |
| `real_test_ref16` | 30.6% @e20 | 28.1% @partial e21 | 0.204 @e10 | 0.215 | 7.1 deg | 0.033 m |
| `pbr_new_val` | 48.1% @e20 | 47.0% @partial e21 | 0.108 @partial e21 | 0.108 | 3.2 deg | 0.018 m |
| `pbr_new_val_ref16` | 57.7% @e20 | 57.7% @e20 | 0.079 @e20 | 0.079 | 2.6 deg | 0.014 m |
| `pbr_new_val_k5_subset` | 77.0% @e20 | 77.0% @e20 | 0.047 @e20 | 0.047 | 1.1 deg | 0.010 m |
| `pbr_new_val_k10_subset` | 79.3% @e20 | 79.3% @e20 | 0.049 @e20 | 0.049 | 1.0 deg | 0.010 m |

## Direct Comparison

At common complete epoch 20, the correspondence loss gives modest pose gains in
most splits, but almost no strict-recall change on the main real `N=5,K=1`
split.

| Val split | Baseline ADD(-S) | Corr ADD(-S) | Delta | Baseline median d | Corr median d | Baseline rot | Corr rot | Baseline trans | Corr trans |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_test` | 25.1% | 25.2% | +0.1 pp | 0.257 | 0.260 | 7.8 deg | 7.8 deg | 0.039 m | 0.040 m |
| `real_test_ref16` | 27.6% | 30.6% | +3.0 pp | 0.228 | 0.214 | 7.5 deg | 7.3 deg | 0.034 m | 0.033 m |
| `pbr_new_val` | 45.8% | 48.1% | +2.4 pp | 0.117 | 0.108 | 3.3 deg | 3.2 deg | 0.020 m | 0.018 m |
| `pbr_new_val_ref16` | 54.1% | 57.7% | +3.6 pp | 0.086 | 0.079 | 2.8 deg | 2.7 deg | 0.016 m | 0.014 m |
| `pbr_new_val_k5_subset` | 74.3% | 77.0% | +2.7 pp | 0.055 | 0.047 | 1.1 deg | 1.2 deg | 0.010 m | 0.010 m |
| `pbr_new_val_k10_subset` | 78.1% | 79.3% | +1.1 pp | 0.051 | 0.049 | 0.9 deg | 1.1 deg | 0.010 m | 0.010 m |

Best-through-epoch-20 comparison tells the same story:

| Val split | Baseline best ADD(-S) | Corr best ADD(-S) | Delta | Baseline best median d | Corr best median d |
| --- | ---: | ---: | ---: | ---: | ---: |
| `real_test` | 25.5% @e11 | 27.3% @e19 | +1.8 pp | 0.257 @e20 | 0.239 @e16 |
| `real_test_ref16` | 28.8% @e11 | 30.6% @e20 | +1.8 pp | 0.223 @e19 | 0.204 @e10 |
| `pbr_new_val` | 45.8% @e20 | 48.1% @e20 | +2.4 pp | 0.117 @e20 | 0.108 @e20 |
| `pbr_new_val_ref16` | 54.1% @e20 | 57.7% @e20 | +3.6 pp | 0.086 @e20 | 0.079 @e20 |
| `pbr_new_val_k5_subset` | 76.0% @e18 | 77.0% @e20 | +1.0 pp | 0.051 @e13 | 0.047 @e20 |
| `pbr_new_val_k10_subset` | 79.8% @e13 | 79.3% @e20 | -0.5 pp | 0.050 @e13 | 0.049 @e20 |

Real-test object behavior at epoch 20 is mixed:

| Object | Baseline ADD(-S) | Corr ADD(-S) | Delta | Baseline median d | Corr median d | Baseline rot | Corr rot |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.8% | 12.8% | +0.0 pp | 0.279 | 0.302 | 7.1 deg | 6.9 deg |
| 5 | 62.1% | 63.1% | +1.0 pp | 0.072 | 0.066 | 2.0 deg | 1.7 deg |
| 6 | 2.4% | 1.2% | -1.2 pp | 0.738 | 0.670 | 21.7 deg | 23.5 deg |
| 8 | 60.8% | 65.7% | +4.9 pp | 0.077 | 0.062 | 1.7 deg | 2.0 deg |
| 9 | 7.9% | 7.9% | +0.0 pp | 0.281 | 0.239 | 8.7 deg | 8.7 deg |
| 10 | 4.3% | 0.0% | -4.3 pp | 1.733 | 1.183 | 62.7 deg | 146.2 deg |
| 11 | 44.6% | 41.5% | -3.1 pp | 0.125 | 0.120 | 11.9 deg | 9.4 deg |
| 12 | 2.0% | 4.0% | +2.0 pp | 0.327 | 0.449 | 5.0 deg | 6.3 deg |

## Correspondence Metric Effect

The auxiliary loss clearly improves the diagnostic correspondence metric itself.
At common epoch 20, median geometric correspondence error drops on every split.

| Val split | Baseline geo L2 median | Corr geo L2 median | Delta | Baseline weighted Huber | Corr weighted Huber | Corr DINO sim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_test` | 0.046 | 0.035 | -0.011 | 0.051 | 0.041 | 0.333 |
| `real_test_ref16` | 0.047 | 0.035 | -0.012 | 0.061 | 0.046 | 0.335 |
| `pbr_new_val` | 0.030 | 0.019 | -0.012 | 0.023 | 0.012 | 0.340 |
| `pbr_new_val_ref16` | 0.028 | 0.018 | -0.011 | 0.022 | 0.013 | 0.343 |
| `pbr_new_val_k5_subset` | 0.023 | 0.015 | -0.008 | 0.012 | 0.005 | 0.366 |
| `pbr_new_val_k10_subset` | 0.021 | 0.014 | -0.007 | 0.010 | 0.005 | 0.367 |

The baseline correspondence metric was logged without DINO weighting, so its
`dino_similarity_mean` scalar is `nan`. The correspondence run used DINO
weights in the loss and logs finite similarities/weights.

## Comparison To Other Runs

Compared with the 2026-07-10 224px run, both A40 high-resolution runs are far
better. The main real-test ADD(-S) recall is about `25..27%` instead of
`10..11%`, and median normalized pose error is about `0.24..0.26d` instead of
`0.54..0.57d`.

Compared with the 2026-07-17 560x420 RTX 4090 12-view run, the A40 baseline is
close on real images and stronger on held-out PBR, especially for K5/K10. This
means the larger A40 view budget is a real experimental change; it should not
be described as only the same run on a different server.

Compared with the 2026-07-21 masked-query/context-reference run, the current
A40 baseline is an important correction to our earlier interpretation. On
held-out PBR `N=5,K=1`, the unmasked-query A40 baseline already reaches
`48.8%`, essentially matching the masked-query run's `48.3%` snapshot. On real
test, however, the masked-query run is much better (`37.4%` at epoch 17 versus
`25.5%` best here). So query masking seems to help most with real-image clutter,
object support, and occlusion/localization, while the larger high-resolution
view budget is already enough to close much of the single-query PBR gap.

The A40 PBR multi-query subsets remain the strongest pose numbers in this
group: `76.2%` at K5 and `80.4%` at K10 for the baseline. Multi-query context
is therefore still a separate positive axis from both masking and
correspondence loss.

No coarse small/occluded/easy/hard bucket scalars are present in these two A40
event files, so this note cannot compare the failure-mode buckets from the
2026-07-17 run.

## Conclusions

- The correspondence loss produced the clearest effect on the metric it directly
  targets: cross-view geometric correspondence error dropped consistently across
  real, PBR, ref16, and multi-query validation splits.
- Pose metrics improved modestly but not decisively. The strongest consistent
  gains are on ref16 and PBR splits, roughly `+1..+4` ADD(-S) percentage
  points. The main real `N=5,K=1` split is essentially unchanged at the common
  complete epoch and only `+1.8` points better at best.
- The object-level real-test changes are not uniform. Objects 5 and 8 improve,
  objects 6/10/11 lose strict recall, and object 10's rotation remains unstable
  despite a better median normalized distance. This argues against treating the
  current correspondence loss as a solved real-domain pose fix.
- The loss is still useful as a controlled regularizer/diagnostic: it improves
  local query-reference consistency without obvious global collapse or large
  pose regression. The current evidence supports keeping it as an ablation, but
  not making it the new default baseline yet.
- A next correspondence experiment should either lower/schedule the weight or
  combine it with the query-masking/recentered-object diagnostics. As-is, it
  improves geometric consistency but does not by itself remove the real-image
  bottleneck.
