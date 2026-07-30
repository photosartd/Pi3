# 2026-07-30: Same-Scene Unmasked Ceiling vs Masked-Query Context Refs

## Purpose

Compare the eval-only same-scene, all-unmasked ceiling run from
[docs/lmgeo_same_scene_ceiling.md](../lmgeo_same_scene_ceiling.md) with the
2026-07-21 masked-query/context-reference run from
[2026-07-21_lmgeo_rtxpro6000_all_rgb_masked_context_refs.md](2026-07-21_lmgeo_rtxpro6000_all_rgb_masked_context_refs.md).

The question is whether the same-scene unmasked setup gives a meaningful upper
ceiling, and how far the masked-query/context-reference experiment is from that
ceiling.

## Compared Artifacts

| Field | Same-scene all-unmasked ceiling | 2026-07-21 masked-query context refs |
| --- | --- | --- |
| Run name | `lmgeo_rtx6000_70gb_same_scene_ceiling_eval_20260729_155932` | `lmgeo_rtxpro6000_70gb_all_rgb_masked_context_refs_20260721_183654` |
| Type | Eval-only, no fine-tuning | Fine-tuning run |
| Checkpoint | `ckpts/Pi3/model.safetensors` | `ckpts/Pi3/model.safetensors` at init, then trained |
| Data config | `lmgeo_same_scene_ceiling` | `lmgeo_trainpbr45_real_and_new_val_all_rgb_masked_context_refs` |
| RGB masking | Reference and query RGB unmasked | Reference and query RGB object-masked |
| Reference source | Same scene/window as the query, query frame held out | Render refs for `real_test` and `pbr_new_val`; masked PBR context refs for `pbr_new_val_context_refs` |
| Query source | Same scene/window as references | Real BOP targets or held-out `new_val` PBR |
| Alignment | Sim(3) from references only | Sim(3) from references only |
| TensorBoard event | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_rtx6000_70gb_same_scene_ceiling_eval_20260729_155932/lmgeo_rtx6000_70gb_same_scene_ceiling_eval_20260729_155932/events.out.tfevents.1785333584.polaris.1055340.0` | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_rtxpro6000_70gb_all_rgb_masked_context_refs_20260721_183654/lmgeo_rtxpro6000_70gb_all_rgb_masked_context_refs_20260721_183654/events.out.tfevents.1784651859.polaris.1950541.0` |

The older 2026-07-21 note was written while the run had completed validation
through global step 9000. The comparison below uses the fuller TensorBoard file
now available through global step 13500.

## Setup Caveats

This is not a strict apples-to-apples model comparison.

- Same-scene ceiling is much easier: references and query come from the same
  physical scene/window, so background, lighting, camera distribution, and
  object constellation are already consistent.
- The masked-query run is a trained cross-scene/render-reference setup. Its
  `pbr_new_val_context_refs` split is harder still because clean render refs are
  replaced by masked scene/context references from other subscenes.
- Real split counts differ: same-scene real has 303 valid samples, while the
  masked-query real split has 1444 samples.
- The same-scene run is still useful as a ceiling because the query is held out
  from alignment. The metric alignment uses only references, then reports pose
  error on the query.

## Aggregate Metrics

Latest scalar rows from TensorBoard are shown. Higher ADD values are better;
lower median `d`, rotation, translation, Chamfer, and loss are better.

| Split/setup | Count | ADD(-S)<0.1d | ADD-S<0.1d | Median d | Rot med | Trans med | Ref Chamfer d | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Same-scene real, unmasked | 303 | 58.1% | 80.2% | 0.076 | 0.00 deg | 0.016 m | 0.361 | 0.0404 |
| Masked real, render refs, latest | 1444 | 40.8% | 67.0% | 0.127 | 4.49 deg | 0.021 m | 0.016 | 0.0865 |
| Same-scene PBR, unmasked | 1600 | 77.3% | 96.3% | 0.053 | 0.35 deg | 0.010 m | 0.056 | 0.0086 |
| Masked PBR, render refs, latest | 1600 | 52.1% | 73.6% | 0.094 | 2.74 deg | 0.018 m | 0.014 | 0.0436 |
| Masked PBR, context refs, latest | 1600 | 12.6% | 26.4% | 0.479 | 6.07 deg | 0.082 m | 0.666 | 0.1266 |

Best values observed in the 2026-07-21 masked-query run:

| Split | Best ADD(-S)<0.1d | Best ADD-S<0.1d | Best median d | Best ref Chamfer d |
| --- | ---: | ---: | ---: | ---: |
| `real_test`, render refs | 41.3% @ 12500 | 67.3% @ 12500 | 0.124 @ 12000 | 0.014 @ 9500 |
| `pbr_new_val`, render refs | 53.2% @ 13000 | 73.8% @ 13000 | 0.092 @ 13000 | 0.013 @ 9500 |
| `pbr_new_val_context_refs` | 12.6% @ 13500 | 26.4% @ 11500 | 0.466 @ 12000 | 0.642 @ 10500 |

## Real-Test Per-Object Comparison

The masked columns use the latest step 13500 unless marked as best. Counts
differ because the two evaluations construct different sample sets.

| Obj | Ceiling count | Ceiling ADD(-S) | Ceiling median d | Masked count | Masked ADD(-S) | Masked median d | Masked best ADD(-S) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39 | 25.6% | 0.195 | 175 | 21.7% | 0.175 | 24.6% @ 12000 |
| 5 | 43 | 74.4% | 0.050 | 199 | 81.4% | 0.054 | 82.4% @ 13000 |
| 6 | 32 | 59.4% | 0.079 | 171 | 17.5% | 0.180 | 20.5% @ 8000 |
| 8 | 44 | 84.1% | 0.060 | 200 | 76.5% | 0.057 | 79.0% @ 12000 |
| 9 | 37 | 37.8% | 0.225 | 180 | 12.8% | 0.205 | 17.8% @ 9000 |
| 10 | 37 | 94.6% | 0.032 | 180 | 26.1% | 0.227 | 29.4% @ 12500 |
| 11 | 27 | 85.2% | 0.030 | 139 | 50.4% | 0.099 | 55.4% @ 8000 |
| 12 | 44 | 13.6% | 0.194 | 200 | 33.0% | 0.131 | 43.0% @ 10500 |

## PBR Per-Object Comparison

Here all compared splits have 200 samples per object.

| Obj | Ceiling ADD(-S) | Ceiling median d | Masked render-ref ADD(-S) | Masked render-ref median d | Masked context-ref ADD(-S) | Masked context-ref median d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 53.5% | 0.098 | 20.5% | 0.234 | 0.5% | 0.899 |
| 5 | 81.5% | 0.050 | 74.0% | 0.060 | 46.5% | 0.108 |
| 6 | 75.5% | 0.064 | 47.5% | 0.104 | 0.0% | 0.853 |
| 8 | 91.0% | 0.037 | 78.5% | 0.059 | 15.0% | 0.178 |
| 9 | 54.5% | 0.093 | 19.5% | 0.199 | 1.5% | 1.096 |
| 10 | 98.5% | 0.032 | 73.0% | 0.053 | 23.5% | 0.847 |
| 11 | 97.5% | 0.025 | 70.0% | 0.064 | 11.0% | 0.264 |
| 12 | 66.0% | 0.073 | 34.0% | 0.134 | 3.0% | 0.592 |

## Main Conclusions

- The same-scene all-unmasked ceiling is clearly above the masked-query
  render-reference run. On real targets it gives `58.1%` ADD(-S) versus
  `40.8%` latest / `41.3%` best, and median error `0.076d` versus about
  `0.124-0.127d`. On held-out PBR it gives `77.3%` ADD(-S) versus
  `52.1%` latest / `53.2%` best, and median error `0.053d` versus about
  `0.092-0.094d`.
- The same-scene PBR ceiling is especially high: `96.3%` pure ADD-S recall,
  `0.35 deg` median rotation, `10 mm` median translation, and `0.053d` median
  normalized error. This says the current Pi3 checkpoint can produce a very
  consistent camera constellation when the references and query remain inside a
  coherent same-scene setting.
- The real same-scene ceiling is lower than the PBR same-scene ceiling, but it
  is still much stronger than the masked-query render-reference real split.
  The weaker real objects remain object 1, 9, and 12; object 12 is the main
  case where the masked trained run beats the same-scene ceiling.
- The `pbr_new_val_context_refs` split is far below both the same-scene ceiling
  and the masked render-reference split: `12.6%` ADD(-S), `0.479d` latest median
  error, and very high reference Chamfer. This points to reference onboarding /
  cross-scene alignment as the failure mode, not merely query masking.
- The reference Chamfer values are not directly comparable across these setups.
  Same-scene real has strong pose metrics but high Chamfer (`0.361d`), likely
  because real same-scene references are partial/noisy and much less render-like.
  Use Chamfer within a fixed split/setup; use aligned query pose metrics for the
  ceiling comparison.

## Interpretation

The same-scene unmasked eval is a valid upper-ceiling diagnostic for the current
checkpoint and metric pipeline: query frames are not used for alignment, yet
the model performs much better when the reference/query images are from the
same scene. That means a large part of the current difficulty is not simply
"unmasked RGB is impossible"; it is the cross-scene object anchoring problem.

The 2026-07-21 masked-query run showed that oracle object masks help strongly
when using clean render references, but the context-reference validation shows
that masked scene references are not yet reliable replacements for renders.
The next useful curriculum should therefore preserve the render-reference bridge
or same-scene consistency while introducing scene references slowly, instead of
jumping directly to context refs as the only reference source.
