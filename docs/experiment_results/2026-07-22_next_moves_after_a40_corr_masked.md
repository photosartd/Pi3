# 2026-07-22: Next Moves After A40 Baseline, Corr, And Masked-Query Results

## Purpose

This is a decision note, not a run report. It records why the next experiment
priority moved toward query-side localization/support and pixel allocation after
comparing the current baseline, correspondence-loss, and masked-query evidence.

## Evidence Base

Primary run notes:

- [2026-07-10 low-resolution warmup](2026-07-10_lmgeo_all_trainpbr_test_bop_15k_warmup.md):
  first complete 224px LM-O fine-tuning run; established that the adaptation
  trains but real-query pose remains weak.
- [2026-07-17 560x420 RTX 4090 12-view](2026-07-17_lmgeo_560x420_rtx4090_12view.md):
  high-resolution unmasked-query run with extended validation and
  small/occluded/easy/hard buckets.
- [2026-07-21 all-RGB-masked context refs](2026-07-21_lmgeo_rtxpro6000_all_rgb_masked_context_refs.md):
  oracle object-masked query/reference diagnostic plus context-reference
  validation.
- [2026-07-22 A40 baseline vs correspondence](2026-07-22_lmgeo_a40_baseline_vs_corr.md):
  A40 high-resolution baseline and DINO-weighted correspondence-loss ablation.

Relevant configs:

- A40 baseline:
  [train_lmgeo_finetune_a40_46gb.yaml](../../configs/train/train_lmgeo_finetune_a40_46gb.yaml),
  [lmgeo_trainpbr45_real_and_new_val.yaml](../../configs/data/lmgeo_trainpbr45_real_and_new_val.yaml)
- A40 correspondence:
  [train_lmgeo_finetune_a40_46gb_corr.yaml](../../configs/train/train_lmgeo_finetune_a40_46gb_corr.yaml)
- Masked-query/context-reference diagnostic:
  [lmgeo_trainpbr45_real_and_new_val_all_rgb_masked_context_refs.yaml](../../configs/data/lmgeo_trainpbr45_real_and_new_val_all_rgb_masked_context_refs.yaml)
- Planned recenter/zoom K1 experiment:
  [train_lmgeo_finetune_a40_46gb_recenter_zoom_k1.yaml](../../configs/train/train_lmgeo_finetune_a40_46gb_recenter_zoom_k1.yaml),
  [lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1.yaml](../../configs/data/lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1.yaml)

## Experiment Map

The current five-experiment plan can now be read like this:

| # | Experiment | Current interpretation |
| ---: | --- | --- |
| 1 | Baseline rerun | The A40 baseline is now the best control for future deltas. It is close to the 2026-07-17 run on real images, but stronger on PBR because it uses the larger A40 view budget and ran longer. |
| 2 | DINO-similarity-weighted correspondence loss | Soft positive. It clearly improves the correspondence metric itself, but only modestly improves pose and barely changes main real `N=5,K=1` at the common complete epoch. |
| 3 | Cross-scene masked keyframes/context refs | Not validated as a replacement for render refs. The context-reference validation in the masked run was weak, so context refs are still only plausible as augmentation unless reference onboarding improves. |
| 4 | Oracle masked query + keyframes | Strong positive ceiling signal for real images. It improves real test much more than the A40 baseline, while A40 baseline already matches masked-query PBR `N=5,K=1`. |
| 5 | Oracle homography-recenter + centered crop/zoom | Still the missing ceiling probe. This is the next clean test of whether small-object pixel budget/resolution is a decisive bottleneck. |

## Updated Conclusions

The A40 baseline should replace the 2026-07-17 12-view run as the main control
for this round. The 07-17 run is still useful historically and for bucketed
failure-mode evidence, but the A40 baseline has the matched high-resolution,
larger-view-budget training regime used by the correspondence ablation.

The correspondence loss is doing what it was designed to do locally:
query-reference geometric correspondence error drops across all validation
splits. However, that local consistency does not translate into a decisive
real-pose gain. The main real `N=5,K=1` split is effectively unchanged at the
common complete epoch, while best real improves only modestly. This means
correspondence supervision is useful as a regularizer/diagnostic, but is not
yet the main answer to the real-image bottleneck.

The masked-query experiment is the strongest current signal. Compared with the
A40 baseline, the gain is clearest on real images, not on held-out PBR K1. That
changes the interpretation: oracle masking is probably not just improving
generic geometry; it is removing real-image clutter, object-support ambiguity,
and localization/visibility confusion.

The A40 baseline's strong PBR K5/K10 results show that multi-query context is
still a separate positive axis. Query masking, correspondence loss, and
multi-query context should not be collapsed into the same explanation.

The context-reference validation remains weak. Masked same-object scene
references from other subscenes are not yet reliable drop-in replacements for
clean render references because the reference/onboarding side degrades before
query pose is even the main issue.

## Next Meaningful Moves

1. Run experiment 5 next: oracle homography-recenter plus centered crop/zoom on
   query frames.

   This is the missing ceiling probe. If it helps especially on `small+visible`
   examples, then object pixel budget is a real bottleneck and the next
   deployable mechanism should both localize the object and allocate more input
   pixels to it.

2. Re-enable or reconstruct comparable failure-mode buckets for the A40
   baseline, corr, masked-query, and recenter/zoom runs.

   The aggregate ADD(-S) numbers are no longer enough. The important question is
   which cells move: `small+visible`, `small+occluded`, `big+visible`,
   `big+occluded`, symmetric/hard objects, and per-object rows. Without this,
   we cannot tell whether a method solves small-object resolution, occlusion, or
   only easy foreground/background separation.

3. Treat learned visibility/localization as the likely engineering target if
   experiment 5 is positive.

   Exp 4 already says oracle object support helps real images. If Exp 5 also
   helps, the follow-up should be one deployable mechanism, not two separate
   projects: a learned visibility/object-support/localization signal that can
   drive masking, pooling, or crop/recenter decisions without GT masks or boxes.

4. Keep correspondence loss as a combination ablation, not as the main next
   branch.

   The next correspondence test should be paired with a stronger input-support
   regime, or use a lower/scheduled weight. Repeating correspondence alone is
   unlikely to answer the main bottleneck question because it already improved
   local consistency without strongly moving real pose.

5. Do not prioritize context refs as replacement onboarding yet.

   They may still be useful as training augmentation, but the current
   context-reference validation says they are not ready to replace clean render
   refs. A more controlled context-ref substitution can be revisited after the
   query-side ceiling probes are clearer.

## Decision Gates

If recenter/zoom succeeds:

- prioritize a deployable object-support/localization head;
- evaluate it first on `K=1`, then combine with multi-query context;
- only then add correspondence loss as a regularizer if it still improves the
  same bucketed metrics.

If recenter/zoom fails:

- deprioritize object pixel budget as the main explanation;
- focus on oracle masking/visibility and occlusion handling;
- check whether `small+occluded` remains near zero even with masking and
  recentering, because neither method can recover pixels that are physically
  occluded.

If only PBR improves but real does not:

- treat the change as improving synthetic geometry but not the real-domain
  bottleneck;
- return to domain/appearance transfer and deployable real-query object support.

## Current Working Hypothesis

The most likely bottleneck is query-side object support: the model struggles to
find and use the visible object region in real clutter, and small objects make
that worse by giving DINO/Pi3 too few useful tokens. Correspondence supervision
can make matched patches more geometrically consistent, but it does not by
itself solve the upstream problem of which real-image evidence should dominate
the object pose.
