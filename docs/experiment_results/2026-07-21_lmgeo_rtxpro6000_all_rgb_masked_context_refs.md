# 2026-07-21: 560x420 All-RGB-Masked Context-Reference Run

## Status

Running. This note is an intermediate snapshot taken on 2026-07-22 while the
run was inside epoch 18. TensorBoard/logged validation metrics were complete
through epoch 17, global step 9000.

## Setup

| Field | Value |
| --- | --- |
| Run name | `lmgeo_rtxpro6000_70gb_all_rgb_masked_context_refs_20260721_183654` |
| Data config | `configs/data/lmgeo_trainpbr45_real_and_new_val_all_rgb_masked_context_refs.yaml` |
| Train config | `configs/train/train_lmgeo_finetune_rtxpro6000_blackwell_70gb.yaml` |
| Objects | LM-O objects `[1, 5, 6, 8, 9, 10, 11, 12]` |
| References | 50/50 clean object renders and masked same-object scene/context references during training |
| Train queries | `train_pbr` synthetic/PBR windows, RGB-masked |
| Validation queries | real LM-O `test` targets and held-out `new_val` PBR, RGB-masked |
| Reference masking | true |
| Query masking | true |
| Training N/K | random total views `6..32`; references `5..31`, queries `1..25` |
| Validation N/K | `N=5`, `K=1` for all active splits |
| Resolution | `560 x 420` |
| Steps | 500 steps/epoch |
| CUDA cap | `PI3_CUDA_MEMORY_LIMIT_GIB=70` on RTX PRO 6000 Blackwell |
| Checkpoint | `ckpts/Pi3/model.safetensors` |
| Important override | `lmgeo.filter_preprocessed_query_depth=false` |

## Validation Partitions

| Partition | Query split | References | Query RGB | Count at epoch 17 | Purpose |
| --- | --- | --- | --- | ---: | --- |
| `real_test` | real BOP `test` targets | 5 render refs | object-masked | 1444 | Main real-image diagnostic under oracle query masking. |
| `pbr_new_val` | held-out `new_val` PBR | 5 render refs | object-masked | 1600 | Same-style held-out synthetic/PBR comparison. |
| `pbr_new_val_context_refs` | held-out `new_val` PBR | 5 masked PBR context refs from other subscenes | object-masked | 1600 | Test whether masked scene refs can replace clean render refs. |

## Outputs

| Artifact | Path |
| --- | --- |
| Output dir | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_rtxpro6000_70gb_all_rgb_masked_context_refs_20260721_183654` |
| JSON log | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_rtxpro6000_70gb_all_rgb_masked_context_refs_20260721_183654/ckpts/log.txt` |
| TensorBoard event | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_rtxpro6000_70gb_all_rgb_masked_context_refs_20260721_183654/lmgeo_rtxpro6000_70gb_all_rgb_masked_context_refs_20260721_183654/events.out.tfevents.1784651859.polaris.1950541.0` |
| Latest checkpoint at snapshot | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_rtxpro6000_70gb_all_rgb_masked_context_refs_20260721_183654/ckpts/checkpoint_14` |
| Raw predictions | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_rtxpro6000_70gb_all_rgb_masked_context_refs_20260721_183654/predictions.parquet` |

## Main Metrics

The table uses completed TensorBoard/log rows only. Epoch 17 is the latest
completed validation at this snapshot.

| Epoch | Split | Train loss | Val loss | ADD(-S)<0.1d | ADD-S<0.1d | Median d | Rot med | Trans med | Ref Chamfer d |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `real_test` | 0.4030 | 0.1809 | 12.8% | 30.2% | 0.473 | 13.6 deg | 0.065 m | 0.072 |
| 0 | `pbr_new_val` | 0.4030 | 0.1634 | 12.2% | 26.9% | 0.518 | 10.2 deg | 0.082 m | 0.071 |
| 0 | `pbr_context_refs` | 0.4030 | 0.3415 | 0.9% | 4.2% | 1.581 | 19.7 deg | 0.251 m | 1.686 |
| 4 | `real_test` | 0.1518 | 0.1510 | 18.9% | 39.3% | 0.288 | 9.4 deg | 0.047 m | 0.059 |
| 4 | `pbr_new_val` | 0.1518 | 0.1200 | 20.8% | 40.1% | 0.246 | 6.0 deg | 0.046 m | 0.065 |
| 4 | `pbr_context_refs` | 0.1518 | 0.2282 | 4.8% | 12.2% | 0.938 | 10.5 deg | 0.141 m | 1.197 |
| 9 | `real_test` | 0.1252 | 0.0954 | 30.5% | 58.0% | 0.165 | 5.9 deg | 0.028 m | 0.021 |
| 9 | `pbr_new_val` | 0.1252 | 0.0589 | 39.8% | 60.6% | 0.134 | 3.6 deg | 0.026 m | 0.022 |
| 9 | `pbr_context_refs` | 0.1252 | 0.1666 | 9.1% | 17.7% | 0.683 | 8.0 deg | 0.107 m | 0.872 |
| 14 | `real_test` | 0.1103 | 0.0969 | 35.9% | 62.5% | 0.144 | 4.8 deg | 0.024 m | 0.023 |
| 14 | `pbr_new_val` | 0.1103 | 0.0518 | 44.8% | 68.2% | 0.114 | 3.1 deg | 0.022 m | 0.023 |
| 14 | `pbr_context_refs` | 0.1103 | 0.1525 | 8.9% | 20.9% | 0.594 | 7.2 deg | 0.094 m | 0.773 |
| 17 | `real_test` | 0.1062 | 0.0887 | 37.4% | 66.5% | 0.134 | 4.9 deg | 0.022 m | 0.018 |
| 17 | `pbr_new_val` | 0.1062 | 0.0492 | 48.3% | 69.8% | 0.104 | 3.0 deg | 0.020 m | 0.016 |
| 17 | `pbr_context_refs` | 0.1062 | 0.1434 | 10.9% | 23.2% | 0.545 | 6.7 deg | 0.088 m | 0.740 |

Best values observed so far:

| Metric | Epoch | Value |
| --- | ---: | ---: |
| `real_test` best val loss | 17 | 0.0887 |
| `real_test` best ADD(-S)<0.1d | 17 | 37.4% |
| `real_test` best ADD-S<0.1d | 17 | 66.5% |
| `real_test` best median normalized pose error | 17 | 0.134d |
| `real_test` best rotation median | 15 | 4.7 deg |
| `real_test` best translation median | 17 | 0.022 m |
| `real_test` best reference Chamfer | 15 | 0.015d |
| `pbr_new_val` best ADD(-S)<0.1d | 17 | 48.3% |
| `pbr_new_val` best ADD-S<0.1d | 16 | 70.2% |
| `pbr_new_val` best median normalized pose error | 17 | 0.104d |
| `pbr_context_refs` best ADD(-S)<0.1d | 17 | 10.9% |
| `pbr_context_refs` best median normalized pose error | 17 | 0.545d |

## Epoch-17 Real Per-Object Metrics

| Object | Count | ADD(-S)<0.1d | Median d | Rot med | Trans med |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 175 | 19.4% | 0.191 | 6.7 deg | 0.019 m |
| 5 | 199 | 75.4% | 0.062 | 1.3 deg | 0.012 m |
| 6 | 171 | 12.9% | 0.200 | 5.7 deg | 0.030 m |
| 8 | 200 | 76.5% | 0.059 | 1.5 deg | 0.016 m |
| 9 | 180 | 17.8% | 0.184 | 5.9 deg | 0.019 m |
| 10 | 180 | 19.4% | 0.278 | 86.9 deg | 0.092 m |
| 11 | 139 | 45.3% | 0.107 | 10.0 deg | 0.043 m |
| 12 | 200 | 25.5% | 0.139 | 2.7 deg | 0.020 m |

## Comparison To 2026-07-10

The 2026-07-10 run used 224x224 inputs, render references, unmasked real query
RGB, and real-test-only validation. Its final real-test metrics were:
`ADD(-S)<0.1d=10.5%`, `ADD-S<0.1d=27.3%`, median normalized error `0.568d`,
median rotation `27.3 deg`, median translation `0.078 m`, and reference Chamfer
`0.039d`.

At epoch 17, this run's `real_test` metrics are: `ADD(-S)<0.1d=37.4%`,
`ADD-S<0.1d=66.5%`, median normalized error `0.134d`, median rotation
`4.9 deg`, median translation `0.022 m`, and reference Chamfer `0.018d`.

This is a large improvement, but it is not a clean apples-to-apples model
comparison. The current run changes both training and evaluation:

- Query RGB is object-masked during training and validation. This gives the
  network an oracle segmentation/visibility cue and removes much of the
  whole-scene clutter/localization burden from the query image.
- Resolution is much higher and keeps the original LM-O 4:3 aspect ratio
  (`560x420` instead of square `224x224`), which gives small objects many more
  patches.
- Training samples are much broader (`6..32` total views, `5..31` references,
  `1..25` queries) and mix render refs with masked same-object context refs.
- The real-test target count differs (`1444` here versus `1381` in the 07-10
  note), so exact aggregate values are not over the same set of rows.

## Comparison To 2026-07-17

The 2026-07-17 run is the more relevant control than 2026-07-10 for resolution:
it already used `560x420`, render references, held-out PBR validation, and real
LM-O BOP targets. Its important differences were unmasked query RGB, a smaller
training view budget (`3..12` total views; references `2..8`, queries `1..4`),
and extra validation regimes (`N=16`, `K=5`, `K=10`, and coarse failure-mode
buckets). The current run keeps `560x420` but uses oracle object-masked query
RGB, broader `6..32`-view training, and context-reference sampling.

Closest real-test comparison, both at `N=5,K=1`:

| Metric | 2026-07-17 final e17 `real_test` | 2026-07-17 best `real_test` | Current e17 `real_test` | Current best `real_test` |
| --- | ---: | ---: | ---: | ---: |
| ADD(-S)<0.1d | 0.231 | 0.240 @e16 | 0.374 | 0.374 @e17 |
| ADD-S<0.1d | 0.430 | 0.442 @e16 | 0.665 | 0.665 @e17 |
| Median d | 0.263 | 0.248 @e16 | 0.134 | 0.134 @e17 |
| Rot med | 8.21 deg | 8.11 deg @e16 | 4.87 deg | 4.7 deg @e15 |
| Trans med | 0.041 m | 0.039 m @e16 | 0.022 m | 0.022 m @e17 |
| Ref Chamfer d | 0.019 | 0.015 @e16 | 0.018 | 0.015 @e15 |

Object-wise real-test comparison at the last completed validation:

| Object | 2026-07-17 e17 ADD(-S) | 2026-07-17 e17 median d | Current e17 ADD(-S) | Current e17 median d | Current best ADD(-S) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.046 | 0.356 | 0.194 | 0.191 | 0.194 @e17 |
| 5 | 0.573 | 0.091 | 0.754 | 0.062 | 0.764 @e15 |
| 6 | 0.000 | 0.981 | 0.129 | 0.200 | 0.205 @e15 |
| 8 | 0.545 | 0.087 | 0.765 | 0.059 | 0.765 @e15 |
| 9 | 0.044 | 0.278 | 0.178 | 0.184 | 0.178 @e17 |
| 10 | 0.033 | 1.131 | 0.194 | 0.278 | 0.250 @e14 |
| 11 | 0.489 | 0.103 | 0.453 | 0.107 | 0.554 @e15 |
| 12 | 0.100 | 0.284 | 0.255 | 0.139 | 0.320 @e14 |

Held-out PBR at `N=5,K=1` also improves under the current masked-query setup:
2026-07-17 had best `0.411` ADD(-S) and best `0.129d` median error; current
epoch 17 has `0.483` ADD(-S) and `0.104d` median error. This suggests the
masked-query setup helps in both real and held-out PBR regimes, not only on the
real split.

The 2026-07-17 multi-query PBR subsets remain an important separate axis. That
run reached about `0.685` ADD(-S) at `K=5` and `0.674` at `K=10`, both above
the current `K=1` masked-query PBR value of `0.483`. So object masking improves
single-query behavior, but it has not tested or replaced the benefit of
additional query context.

The 2026-07-17 `N=16,K=1` real-test split also has no current counterpart. In
that run more render references mainly improved reference Chamfer and did not
solve the real-test gap. The current run's reference Chamfer is already similar
to the 07-17 best value, while pose metrics improve much more, which points the
current gain more toward query-side masking/association than reference geometry
alone.

The context-reference validation is not comparable to a 2026-07-17 split. It is
a deliberately harder reference/onboarding setup: masked PBR scene references
from other subscenes replace clean render references. It remains far worse than
render-reference validation (`0.109` ADD(-S), `0.545d` median error, `0.740d`
reference Chamfer at epoch 17), so the current improvements should be attributed
to masked query/render-reference evaluation first, not to context references.

The class/visibility bucket axis from 2026-07-17 cannot yet be compared
directly. That run enabled coarse failure-mode analysis and wrote bucketed
reports; the current run has `coarse_analysis: false` and only
`predictions.parquet`, without the covariate tables needed for the same
small/occluded/easy/hard aggregations.

## Conclusions

- The object-masked, high-resolution setting shows that Pi3 can produce much
  stronger object-centric geometry and aligned pose when the query image is
  reduced to the visible object support. This supports the hypothesis that a
  large part of the 07-10 and 07-17 failures was query-side clutter/object
  extent/small-object visibility rather than a total inability to learn the
  object geometry.
- Compared with the 07-17 high-resolution unmasked run, the current gain cannot
  be explained by resolution alone. Both runs are `560x420`, and reference
  Chamfer is similar at the best epochs; the big additional improvement is most
  plausibly from oracle query masking plus the broader training view budget.
- The remaining real-vs-PBR gap under masked queries is present but modest:
  at epoch 17, `pbr_new_val` reaches `48.3%` ADD(-S) and `0.104d` median error,
  while `real_test` reaches `37.4%` and `0.134d`. Under this oracle mask setup,
  real images are worse than held-out PBR, but not catastrophically so.
- The current run does not cover the 07-17 multi-query axis. The 07-17 PBR
  `K=5/K=10` subsets still outperform the current masked `K=1` PBR split, so
  query context remains a separate positive signal to test together with
  masking later.
- The context-reference validation is much weaker than render-reference
  validation: `10.9%` ADD(-S), `0.545d` median error, and reference Chamfer
  `0.740d` at epoch 17. This means masked scene/context references are not yet
  reliable drop-in replacements for clean render refs. The failure is strongly
  reference/alignment-side: the reference reconstruction/onboarding metric is
  much worse before the query pose is even the main issue.
- Object-wise, the old easy objects 5 and 8 become very strong, and object 11
  improves substantially. Objects 1, 6, 9, 10, and 12 still have low strict
  ADD(-S) recall despite much better median errors. Object 10 remains special:
  ADD-S-style behavior improves, but the logged rotation median is still very
  large, consistent with symmetry/ambiguity.
- Treat this run as a diagnostic upper-bound style experiment for masked query
  inputs, not as a final replacement for the unmasked real-query baseline.
  The next fair question is whether we can recover a similar gain with a
  deployable mask source, or by training/evaluating unmasked queries at the same
  resolution and view budget.
