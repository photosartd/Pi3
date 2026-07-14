# Experiment Results

This folder records LMGeo/Pi3 fine-tuning and evaluation runs. Each file should
capture the run setup, output locations, main metrics, and short conclusions so
we can compare experiments without re-mining TensorBoard/log files every time.

## Runs

| Run | Status | Purpose |
| --- | --- | --- |
| [2026-07-10 30ep warmup](2026-07-10_lmgeo_all_trainpbr_test_bop_15k_warmup.md) | complete | First full LM-O fine-tuning run with real BOP validation. |
| [2026-07-13 real + held-out PBR vals](2026-07-13_lmgeo_trainpbr45_real_and_new_val_30ep.md) | running | Same training style, but adds held-out synthetic/PBR validation splits for in-distribution comparison. |

