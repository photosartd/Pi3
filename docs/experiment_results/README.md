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
| [2026-08-03 A40 anchor mask conditioning](2026-08-03_lmgeo_a40_anchor_mask_conditioning.md) | event-mined comparison | Compares reference-only mask conditioning with full/object depth supervision against the query-mask oracle anchor-pair run. |
| [2026-08-14 GSO geometry-constrained N=5/K=1](2026-08-14_megapose_gso_geometry_n5_k1_implementation.md) | implementation validated | Adds exact surface/focal/view constrained scene/render sampling, calibrated 4:3 crops, held-out validation indexes, and the compact ray-conditioned scratch profile. |
| [2026-08-14 GSO N=5/K=1 one-batch overfit](2026-08-14_megapose_gso_geometry_n5_k1_one_batch_overfit.md) | complete | Memorization control on one deterministic masked scene-to-scene sample; reaches 0.049-degree/0.66-mm query pose error and strict ADD(-S) success. |
| [2026-08-14 GSO N=5/K=1 maximum-batch probe](2026-08-14_megapose_gso_geometry_n5_k1_max_batch_probe.md) | intentionally stopped | Fixed 24-object/144-image batch makes broad early rotation and geometry progress by update 40 while fitting in 22.3 GiB reserved GPU memory. |
| [2026-08-15 GSO N=5/K=1 compact scratch run](2026-08-15_megapose_gso_geometry_n5_k1_scratch_run.md) | complete | Full 40k-update held-out-object run: local/ray geometry converges, while query/reference registration plateaus at 60-96 degree median rotation. |
| [2026-08-17 GSO N=5/K=1 full-Pi3 ray preflight](2026-08-17_megapose_gso_geometry_pi3_finetune_ray_preflight.md) | smoke passed | Released 588M-trainable Pi3 weights plus only a new zero-init ray adapter pass a 156-image train and 768-image validation smoke at 66.8 GiB reserved. |
| [2026-08-18 GSO N=5/K=1 full-Pi3 ray fine-tune](2026-08-18_megapose_gso_geometry_pi3_finetune_ray_run.md) | complete | Full 15k-update run transfers strongly: 11-15 degree median rotation and 0.10-0.17 m translation, with a remaining difficult registration tail and low strict ADD. |
| [2026-08-18 geometry-matched GSO to LM-O transfer](2026-08-18_lmgeo_geometry_matched_gso_transfer.md) | complete | Additive matched N=5/K=1 LM-O evaluator and full 1,364-query transfer run. GT reference scale does not improve query center. Exact post-crop occupancy is the strongest measured failure axis: sub-1% queries are 63.6% of data but 96.9% of failures above 1d. The offline orthographic exporter scale bug is fixed. |
| [2026-08-20 GSO standalone eval and query-occupancy audit](2026-08-20_megapose_gso_geometry_full_eval_and_query_occupancy.md) | complete | Standalone matched-to-LM-O eval of `checkpoint_29` on its own held-out GSO split, plus a new `enumerate_query_groups` sampling-policy feature that grows render-to-scene validation from 189 to 1,988 queries. Occupancy failure mode reproduces on GSO in both regimes, now with a within-object-controlled correlation GSO previously could not run. |
| [2026-08-20 GSO higher-resolution A40 smoke](2026-08-20_gso_higher_resolution_smoke.md) | complete | Empirical `PI3_CUDA_MEMORY_LIMIT_GIB=40` sweep sizing the GSO N=5/K=1 recipe at 560x420 and near-native 728x546 (MegaPose-GSO's actual 720x540 is not patch-14-aligned). 560x420 recommended: ~2.8x compute/sample, 4 samples/GPU; 728x546 works but only 2 samples/GPU at the memory ceiling. New A40 40gb configs at both resolutions. |
| [2026-08-21 GSO Slurm portability fix](2026-08-21_megapose_gso_slurm_portability_fix.md) | implementation validated | Removes workstation absolute paths from plan runtime resolution, keeps copied legacy plan DBs usable without regeneration, and adds a pre-submit audit of scene/reference TARs, geometry/plan fingerprints, and model symlinks. |
