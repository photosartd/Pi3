# 2026-08-03: LMGeo A40 Anchor Mask-Conditioning Ablation

## Status

Event-mined comparison of three complete 30-epoch A40 runs:

| Short name | Run folder | Event source |
| --- | --- | --- |
| `full_depth` | `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/lmgeo_a40_anchor_maskcond_full_depth_20260729_183444` | `events.out.tfevents.1785413188.worker-4.GPU.CIT-EC.NET.664274.0` |
| `masked_depth` | `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/lmgeo_a40_anchor_maskcond_masked_depth_20260729_183439` | `events.out.tfevents.1785391529.worker-11.GPU.CIT-EC.NET.2986271.0` |
| `query_masks` | `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/lmgeo_a40_anchor_maskcond_query_masks_20260730_095637` | `events.out.tfevents.1785522995.worker-1.GPU.CIT-EC.NET.1495432.0` |

Related configs:

- `configs/train/train_lmgeo_finetune_a40_40gb_anchor_mask_conditioning.yaml`
- `configs/data/lmgeo_trainpbr45_anchor_scene_pairs_full_depth.yaml`
- `configs/data/lmgeo_trainpbr45_anchor_scene_pairs_masked_depth.yaml`
- `configs/data/lmgeo_trainpbr45_anchor_scene_pairs_masked_depth_query_masks.yaml`

The event folders only contain TensorBoard files, not saved Hydra snapshots, so
the exact checkpoint path is inferred from the current Slurm/config convention.
The scalar `n_parameters=588,397,632` confirms that all three runs used the
large/common Pi3 fine-tuning model, not the compact DINO-S scratch model.

## Config Mapping

The run folders do not contain saved `.hydra` configs, so this table maps the
run names to the configs they are expected to have used, based on the run names,
the Slurm launcher convention, current config tests, and matching TensorBoard
scalars.

| Run folder | Train config | Data config | Intended difference |
| --- | --- | --- | --- |
| `lmgeo_a40_anchor_maskcond_full_depth_20260729_183444` | `train_lmgeo_finetune_a40_40gb_anchor_mask_conditioning` | `lmgeo_trainpbr45_anchor_scene_pairs_full_depth` | Reference/keyframe masks are supplied; query masks are not supplied; depth supervision is full-scene. |
| `lmgeo_a40_anchor_maskcond_masked_depth_20260729_183439` | `train_lmgeo_finetune_a40_40gb_anchor_mask_conditioning` | `lmgeo_trainpbr45_anchor_scene_pairs_masked_depth` | Same as `full_depth`, but depth supervision is object-mask-only. |
| `lmgeo_a40_anchor_maskcond_query_masks_20260730_095637` | `train_lmgeo_finetune_a40_40gb_anchor_mask_conditioning` | `lmgeo_trainpbr45_anchor_scene_pairs_masked_depth_query_masks` | Same as `masked_depth`, but GT query masks are also supplied as conditioning. |

Resolved Hydra comparison:

| Field | `full_depth` | `masked_depth` | `query_masks` |
| --- | --- | --- | --- |
| `model.use_visibility_mask_conditioning` | `true` | `true` | `true` |
| `model.visibility_mask_conditioning_alpha` | `1.0` | `1.0` | `1.0` |
| `train.optimizer.lr` | `5e-6` | `5e-6` | `5e-6` |
| `train.optimizer.visibility_mask_lr` | `5e-5` | `5e-5` | `5e-5` |
| `train.image_num_range` | `[3, 26]` | `[3, 26]` | `[3, 26]` |
| `train.max_img_per_gpu` | `28` | `28` | `28` |
| `train.resolution` | `[[560, 420]]` | `[[560, 420]]` | `[[560, 420]]` |
| `lmgeo.depth_masking` | `false` | `true` | `true` |
| `lmgeo.reference_rgb_masking` | `false` | `false` | `false` |
| `lmgeo.query_rgb_masking` | `false` | `false` | `false` |
| `lmgeo.num_reference_range` | `[2, 16]` | `[2, 16]` | `[2, 16]` |
| `lmgeo.num_query_range` | `[1, 10]` | `[1, 10]` | `[1, 10]` |
| `lmgeo_anchor.condition_reference_visibility` | `true` | `true` | `true` |
| `lmgeo_anchor.condition_query_visibility` | `false` | `false` | `true` |
| `train_dataset.LMGeoAnchorScenePair.depth_masking` | `false` | `true` | `true` |
| `train_dataset.LMGeoAnchorScenePair.condition_reference_visibility` | `true` | `true` | `true` |
| `train_dataset.LMGeoAnchorScenePair.condition_query_visibility` | `false` | `false` | `true` |
| Active val splits | `real_anchor_pairs`, `pbr_anchor_pairs` | `real_anchor_pairs`, `pbr_anchor_pairs` | `real_anchor_pairs`, `pbr_anchor_pairs` |

So the exact ablation chain is:

1. `full_depth` -> `masked_depth`: only the depth-supervision mask changes.
2. `masked_depth` -> `query_masks`: only query visibility-mask conditioning
   changes.

This confirms that the useful comparison for query masks is
`masked_depth` versus `query_masks`, not `full_depth` versus `query_masks`.

## Setup Check

All three runs use the visibility-mask conditioning branch. The difference is
which views receive the GT anchor-object visibility mask, and whether depth loss
is full-scene or object-mask-only.

| Run | Params | Epochs | Peak min/max group LR | Peak mask LR | Alpha | Train ref/query mask area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_depth` | 588,397,632 | 0-29 | `5e-6` / `5e-5` | `5e-5` | 1.0 | 0.0114 / 0.0000 |
| `masked_depth` | 588,397,632 | 0-29 | `5e-6` / `5e-5` | `5e-5` | 1.0 | 0.0114 / 0.0000 |
| `query_masks` | 588,397,632 | 0-29 | `5e-6` / `5e-5` | `5e-5` | 1.0 | 0.0114 / 0.0114 |

Important details:

- `full_depth` has `lmgeo.depth_masking=false`; references are mask-conditioned,
  queries are not mask-conditioned, and depth supervision is full-scene.
- `masked_depth` has `lmgeo.depth_masking=true`; references are
  mask-conditioned, queries are not mask-conditioned, and depth supervision is
  object-only.
- `query_masks` also has `lmgeo.depth_masking=true`, but both references and
  queries are GT mask-conditioned. This is an oracle/ceiling variant because
  query masks are supplied directly.
- `visibility_mask_conditioning_alpha` is fixed at `1.0` in all three event
  logs. There was no alpha ramp in these runs.
- On validation, the ref-only variants report `known_area=0.8333`, exactly
  matching N=5, K=1 where five of six views carry a known conditioning mask.
  The query-mask run reports `known_area=1.0`.

## Training Scalars

| Run | Train loss first -> last (best) | Last pts | Last trans | Last rot | Last mask token abs | Last mask embed norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_depth` | 0.252 -> 0.212 (0.207) | 0.0112 | 0.0125 | 0.7523 | 0.0898 | 1.4325 |
| `masked_depth` | 0.302 -> 0.273 (0.264) | 0.0073 | 0.0191 | 0.7512 | 0.0976 | 1.0252 |
| `query_masks` | 0.298 -> 0.028 (0.021) | 0.0051 | 0.0016 | 0.0743 | 0.0992 | 0.7552 |

The decisive difference is camera supervision, not just point/depth loss:
without query masks, `rot_loss` stays around `0.75`; with query masks it falls
to `0.074`.

## Aggregate Validation

Real anchor-pair validation is very small here (`count=39`), so use it as a
sanity check rather than as a stable ranking signal.

| Run | Count | Best ADD(-S) 0.1d | Last ADD | Best med d | Last med d | Best 1d | Last 1d | Best rot | Last rot | Best trans | Last trans |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_depth` | 39 | 20.5% e25 | 7.7% | 0.329 | 0.793 | 71.8% | 59.0% | 7.2 | 15.7 | 0.077 | 0.100 |
| `masked_depth` | 39 | 23.1% e0 | 0.0% | 0.418 | 1.989 | 71.8% | 20.5% | 7.3 | 94.2 | 0.065 | 0.318 |
| `query_masks` | 39 | 46.2% e17 | 35.9% | 0.107 | 0.204 | 92.3% | 84.6% | 4.7 | 6.9 | 0.021 | 0.029 |

Held-out PBR anchor-pair validation is much more reliable (`count=800`).

| Run | Count | Best ADD(-S) 0.1d | Last ADD | Best med d | Last med d | Best 1d | Last 1d | Best rot | Last rot | Best trans | Last trans |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_depth` | 800 | 0.6% e17 | 0.4% | 2.345 | 2.904 | 16.1% | 10.0% | 83.8 | 86.3 | 0.382 | 0.476 |
| `masked_depth` | 800 | 0.7% e1 | 0.1% | 2.390 | 3.010 | 18.1% | 9.1% | 84.0 | 88.8 | 0.392 | 0.485 |
| `query_masks` | 800 | 40.9% e26 | 39.0% | 0.136 | 0.143 | 93.3% | 93.0% | 3.8 | 3.8 | 0.025 | 0.025 |

## Reference Chamfer Check

Reference Chamfer is useful here because it asks whether query-mask
conditioning only helps the query-side pose shortcut, or also improves
reference-side reconstruction in the aligned object frame.

Important caveat: `ReferenceChamferMetric` uses each view's `valid_mask`.
Therefore `full_depth` is not fully apples-to-apples with the two masked-depth
runs: `full_depth` includes full-scene/background depth, while `masked_depth`
and `query_masks` use object-mask-only valid pixels.

| Run | Split | First ref Chamfer d | Best ref Chamfer d | Last ref Chamfer d | Best ADD(-S) |
| --- | --- | ---: | ---: | ---: | ---: |
| `full_depth` | `real_anchor_pairs` | 0.541 e0 | 0.327 e1 | 0.979 e29 | 20.5% |
| `masked_depth` | `real_anchor_pairs` | 0.477 e0 | 0.121 e10 | 0.233 e29 | 23.1% |
| `query_masks` | `real_anchor_pairs` | 0.479 e0 | 0.096 e29 | 0.096 e29 | 46.2% |
| `full_depth` | `pbr_anchor_pairs` | 0.698 e0 | 0.206 e25 | 0.226 e29 | 0.6% |
| `masked_depth` | `pbr_anchor_pairs` | 0.086 e0 | 0.066 e24 | 0.071 e29 | 0.7% |
| `query_masks` | `pbr_anchor_pairs` | 0.086 e0 | 0.059 e27 | 0.063 e29 | 40.9% |

The clean comparison is `masked_depth` vs `query_masks`. Query masks do improve
reference Chamfer, but only modestly:

- PBR best ref Chamfer improves from `0.066d` to `0.059d`.
- PBR last ref Chamfer improves from `0.071d` to `0.063d`.
- Real best ref Chamfer improves from `0.121d` to `0.096d`, but the real split
  has only 39 queries and should be read cautiously.

That is much smaller than the PBR pose jump from `0.7%` to `40.9%` ADD(-S), or
from `2.390d` to `0.136d` median normalized query pose error. So query masks do
help the shared/reference geometry a little, but the dominant effect is still
query-side target disambiguation and camera alignment.

For context, useful comparison runs:

| Run / split | Reference type | Best ref Chamfer d | Best ADD(-S) |
| --- | --- | ---: | ---: |
| A40 baseline / `pbr_new_val` | clean render refs | 0.023 | 48.8% |
| A40 context refs / `pbr_new_val` | clean render refs | 0.014 | 45.2% |
| Masked-query run / `pbr_new_val` | clean render refs | 0.013 | 53.2% |
| A40 context refs / `pbr_new_val_context_refs` | PBR scene refs | 0.658 | 7.1% |
| Masked-query run / `pbr_new_val_context_refs` | masked PBR scene refs | 0.642 | 12.6% |
| Anchor `query_masks` / `pbr_anchor_pairs` | PBR scene refs + query mask conditioning | 0.059 | 40.9% |

This makes the anchor `query_masks` result look meaningfully different from the
older context-reference failures: it does not reach clean-render-reference
Chamfer (`~0.013-0.023d`), but it is far better than the previous
context-reference scene-ref Chamfer (`~0.64-0.66d`).

## PBR Per-Object Metrics

Per-object metrics are reported at each run's best aggregate PBR ADD(-S) epoch.

`full_depth`, best PBR ADD(-S) 0.1d = 0.6% at epoch 17:

| Obj | Count | ADD(-S) 0.1d | Med d | Rot med deg | Trans med m |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 107 | 0.0% | 4.539 | 96.5 | 0.461 |
| 5 | 96 | 0.0% | 2.147 | 99.1 | 0.429 |
| 6 | 100 | 0.0% | 2.789 | 85.7 | 0.427 |
| 8 | 91 | 1.1% | 1.673 | 81.1 | 0.432 |
| 9 | 105 | 0.0% | 4.292 | 99.1 | 0.469 |
| 10 | 96 | 2.1% | 2.283 | 84.4 | 0.445 |
| 11 | 103 | 1.0% | 2.439 | 85.6 | 0.457 |
| 12 | 102 | 1.0% | 3.760 | 96.2 | 0.542 |

`masked_depth`, best PBR ADD(-S) 0.1d = 0.7% at epoch 1:

| Obj | Count | ADD(-S) 0.1d | Med d | Rot med deg | Trans med m |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 107 | 0.0% | 4.258 | 96.1 | 0.431 |
| 5 | 96 | 0.0% | 1.856 | 107.5 | 0.371 |
| 6 | 100 | 1.0% | 2.342 | 83.3 | 0.361 |
| 8 | 91 | 2.2% | 1.470 | 73.2 | 0.380 |
| 9 | 105 | 0.0% | 4.529 | 79.2 | 0.486 |
| 10 | 96 | 3.1% | 2.183 | 91.6 | 0.431 |
| 11 | 103 | 0.0% | 2.480 | 95.0 | 0.469 |
| 12 | 102 | 0.0% | 3.449 | 98.7 | 0.502 |

`query_masks`, best PBR ADD(-S) 0.1d = 40.9% at epoch 26:

| Obj | Count | ADD(-S) 0.1d | Med d | Rot med deg | Trans med m |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 107 | 8.4% | 0.368 | 7.3 | 0.035 |
| 5 | 96 | 57.3% | 0.087 | 2.4 | 0.017 |
| 6 | 100 | 23.0% | 0.191 | 4.1 | 0.029 |
| 8 | 91 | 73.6% | 0.058 | 1.6 | 0.015 |
| 9 | 105 | 12.4% | 0.292 | 5.0 | 0.029 |
| 10 | 96 | 60.4% | 0.081 | 5.9 | 0.033 |
| 11 | 103 | 67.0% | 0.064 | 5.5 | 0.025 |
| 12 | 102 | 32.4% | 0.173 | 3.6 | 0.024 |

## Interpretation

The query-mask run is clearly the best run, especially on held-out PBR. It is
not just a small TensorBoard fluctuation: PBR ADD(-S) jumps from below `1%` to
`40.9%`, median normalized pose error drops from about `2.3-3.0d` to `0.136d`,
and median rotation drops from about `84-89 deg` to `3.8 deg`.

The reason is most likely task ambiguity, not that the mask adapter was broken
in the first two runs. In `full_depth` and `masked_depth`, the model knows which
object is the anchor in the reference views, but the query is still an
unconditioned multi-object RGB scene. The target selector exists only on one
side of the correspondence problem. In `query_masks`, the target selector is
present in every view, so the model can learn a consistent object-centric frame
instead of trying to infer which query object should match the reference masks.

The mask branch was active in all three runs. The alpha scalar is fixed at
`1.0`, and the token/embedding diagnostics move in all runs. Interestingly,
`query_masks` has a smaller final mask embedding norm than the ref-only runs
(`0.755` vs `1.03-1.43`). So the improvement is not because the branch became
larger or because alpha ramped up; it is because the input signal is more
complete and easier to use.

`masked_depth` improves the final local point loss compared with `full_depth`,
but it does not improve pose. That suggests object-only depth supervision alone
does not solve the cross-scene anchor assignment problem. Full-scene depth also
does not rescue it; in cross-scene pairs, unrelated background geometry can
become a distracting or inconsistent target.

## Relation To Broader Comparison

The useful row from this note, `query_masks`, has also been added to the broader
comparison table in
[2026-07-27_lmgeo_a40_new_ablation_comparison.md](2026-07-27_lmgeo_a40_new_ablation_comparison.md).
Only this best oracle row is carried over; `full_depth` and `masked_depth` stay
here as negative controls.

On held-out PBR, the `query_masks` anchor-pair result reaches `40.9%` ADD(-S).
That is not the best PBR result overall: render-reference and paired-query runs
are still stronger, with the best K=1 number at `59.9%` for
`recenter_zoom_plus_original_k1`. But it is much stronger than every previous
scene-reference/context-reference attempt: the old PBR context-reference
validations were around `7.1-12.6%`, and the ref-only anchor-mask variants here
stay below `1%`.

So the best PBR insight is narrow but important: query-side target selection is
enough to make cross-scene anchor pairs learnable on synthetic held-out PBR, but
scene-pair anchoring still underperforms the easier render-reference setup.

## Main Takeaway

These runs support the hypothesis that visibility-mask conditioning can be used
by the large Pi3 model, but reference-only masks are not enough for this
cross-scene anchor setup. The strongest next evidence would come from bridging
the oracle query-mask setting toward deployment: predicted query masks,
corrupted/shifted query-mask ablations, or a curriculum that starts with query
masks and gradually removes or weakens them. Ref-only mask conditioning by
itself looks too underconstrained in the current scene-pair formulation.
