# Experiment Results

This folder records LMGeo/Pi3 fine-tuning and evaluation runs. Each file should
capture the run setup, output locations, main metrics, and short conclusions so
we can compare experiments without re-mining TensorBoard/log files every time.

## Runs

| Run | Status | Purpose |
| --- | --- | --- |
| [2026-07-10 30ep warmup](2026-07-10_lmgeo_all_trainpbr_test_bop_15k_warmup.md) | complete | First full LM-O fine-tuning run with real BOP validation. |
| [2026-07-13 real + held-out PBR vals](2026-07-13_lmgeo_trainpbr45_real_and_new_val_30ep.md) | running | Same training style, but adds held-out synthetic/PBR validation splits for in-distribution comparison. |
| [2026-07-17 560x420 RTX 4090 12-view](2026-07-17_lmgeo_560x420_rtx4090_12view.md) | interrupted | High-resolution 560x420 unmasked-query run with 12-view train budget and extended validation/failure-mode buckets. |
| [2026-07-21 all-RGB-masked context refs](2026-07-21_lmgeo_rtxpro6000_all_rgb_masked_context_refs.md) | running | High-resolution RTX PRO 6000 diagnostic with masked query/reference RGB and context-reference validation. |
| [2026-07-22 A40 baseline vs correspondence](2026-07-22_lmgeo_a40_baseline_vs_corr.md) | complete/partial | Cluster A40 high-resolution baseline and DINO-weighted correspondence-loss ablation comparison. |
| [2026-07-22 next moves](2026-07-22_next_moves_after_a40_corr_masked.md) | decision note | Interprets the baseline/corr/masked-query evidence and records the next experiment priorities. |
| [2026-07-24 paired query implementation](2026-07-24_lmgeo_paired_query_implementation.md) | implementation validated | Adds opt-in recentered+original query regimes, canonical/equivariance metrics, and local/Slurm-script smoke evidence. |
| [2026-07-27 A40 ablation comparison](2026-07-27_lmgeo_a40_new_ablation_comparison.md) | event-mined comparison | Analyzes context refs, recenter+zoom, GT visibility pooling, and recenter+masked A40 runs against the existing baselines. |
| [2026-07-29 A40 recenter paired/ray](2026-07-29_lmgeo_a40_recenter_zoom_paired_ray.md) | event-mined comparison | Adds original+crop paired-query and ray-conditioning A40 recenter runs, including ray-geometry and paired-query diagnostics. |
| [2026-07-30 same-scene ceiling vs masked context refs](2026-07-30_same_scene_ceiling_vs_masked_context_refs.md) | event-mined comparison | Compares the same-scene all-unmasked ceiling eval against the 2026-07-21 masked-query/context-reference run. |
