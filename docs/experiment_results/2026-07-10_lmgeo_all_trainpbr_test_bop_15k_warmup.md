# 2026-07-10: LMGeo All TrainPBR + Real BOP Test, 30 Epochs

## Setup

| Field | Value |
| --- | --- |
| Run name | `lmgeo_all_trainpbr_test_bop_15k_warmup` |
| Data config | `configs/data/lmgeo_all_trainpbr_test_bop.yaml` |
| Train config | `configs/train/train_lmgeo_finetune_lowres.yaml` |
| Objects | LM-O objects `[1, 5, 6, 8, 9, 10, 11, 12]` |
| References | `train/<object_id>` object renders |
| Train queries | `train_pbr` synthetic/PBR windows |
| Validation queries | real LM-O `test` targets from `lmo/test_targets_bop19.json` |
| Query masking | false |
| Reference masking | true |
| Training N/K | random `N=5..6` references, random `K=1..20` queries |
| Validation N/K | uniform `N=5` references, `K=1` query |
| Resolution | `224 x 224` |
| Epochs | 30 |
| Steps | 500 steps/epoch, 15k total |
| Checkpoint | `ckpts/Pi3/model.safetensors` |
| Important override | `lmgeo.filter_preprocessed_query_depth=false` |

## Outputs

| Artifact | Path |
| --- | --- |
| Log file | `/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/lmgeo_all_trainpbr_test_bop_15k_warmup/ckpts/log.txt` |
| TensorBoard event | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_all_trainpbr_test_bop_15k_warmup/lmgeo_all_trainpbr_test_bop_15k_warmup/events.out.tfevents.1783697388.fafnir.300577.0` |
| Final checkpoint | `/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/lmgeo_all_trainpbr_test_bop_15k_warmup/ckpts/checkpoint_29` |
| Best checkpoint | `/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/lmgeo_all_trainpbr_test_bop_15k_warmup/ckpts/best_model` |

The JSON log has 31 rows because epoch 0 appears once from an earlier restart.
The metrics below use the last row per epoch, i.e. the completed 30-epoch run.

## Main Metrics

| Epoch | Train loss | Val loss | ADD(-S)<0.1d | ADD-S<0.1d | Median d | Rot med | Trans med | Ref Chamfer d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5009 | 0.4526 | 0.0065 | 0.0246 | 3.562 | 95.6 deg | 0.578 m | 0.173 |
| 1 | 0.2628 | 0.3544 | 0.0094 | 0.0536 | 2.144 | 85.1 deg | 0.321 m | 0.085 |
| 3 | 0.1321 | 0.3012 | 0.0471 | 0.1361 | 0.957 | 63.8 deg | 0.143 m | 0.083 |
| 5 | 0.0867 | 0.2463 | 0.0507 | 0.1615 | 0.778 | 42.1 deg | 0.119 m | 0.044 |
| 10 | 0.0599 | 0.2475 | 0.0615 | 0.1897 | 0.647 | 34.9 deg | 0.094 m | 0.047 |
| 15 | 0.0430 | 0.2505 | 0.0731 | 0.2274 | 0.614 | 38.3 deg | 0.088 m | 0.054 |
| 20 | 0.0382 | 0.2336 | 0.0927 | 0.2476 | 0.554 | 28.0 deg | 0.079 m | 0.044 |
| 25 | 0.0342 | 0.2257 | 0.1079 | 0.2679 | 0.590 | 28.9 deg | 0.082 m | 0.041 |
| 29 | 0.0364 | 0.2255 | 0.1050 | 0.2730 | 0.568 | 27.3 deg | 0.078 m | 0.039 |

Best values observed:

| Metric | Epoch | Value |
| --- | ---: | ---: |
| Best val loss | 26 | 0.2254 |
| Best ADD(-S)<0.1d | 23 | 0.1086 |
| Best ADD-S<0.1d | 24 | 0.2766 |
| Best median normalized pose error | 13 | 0.5407d |
| Best rotation median | 23 | 26.49 deg |
| Best translation median | 29 | 0.0777 m |
| Best reference Chamfer | 4 | 0.0377d |

## Final Per-Object Metrics

| Object | Count | ADD(-S)<0.1d | Median d | Rot med | Trans med |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 172 | 0.006 | 0.901 | 45.1 deg | 0.088 m |
| 5 | 199 | 0.226 | 0.212 | 7.5 deg | 0.033 m |
| 6 | 170 | 0.000 | 1.294 | 116.4 deg | 0.188 m |
| 8 | 200 | 0.330 | 0.160 | 5.5 deg | 0.039 m |
| 9 | 180 | 0.033 | 0.573 | 19.9 deg | 0.051 m |
| 10 | 150 | 0.020 | 1.545 | 109.2 deg | 0.325 m |
| 11 | 110 | 0.100 | 1.129 | 51.7 deg | 0.235 m |
| 12 | 200 | 0.065 | 0.447 | 9.8 deg | 0.059 m |

## Conclusions

- Fine-tuning gives a clear signal: median normalized query pose error improved
  from `3.56d` to `0.57d`, median rotation from `95.6 deg` to `27.3 deg`, and
  median translation from `0.58 m` to `0.078 m`.
- Strict BOP-style pose accuracy is still limited: final `ADD(-S)<0.1d` is
  about `10.5%`.
- Object behavior is highly uneven. Objects 8 and 5 are much stronger than
  objects 6, 10, and 11.
- Reference/onboarding geometry improved substantially: reference Chamfer
  dropped from `0.173d` to `0.039d`.
- The run validates that this setup is trainable, but real-image generalization
  remains the bottleneck.

## Notes For Comparison

- This run only validated on real LM-O BOP targets with `K=1`.
- It did not include a same-distribution held-out PBR validation split, so it
  could not separate "training did not learn" from "synthetic-to-real domain gap".
- The follow-up run adds held-out PBR validation splits for exactly this reason.

