# 2026-07-13: LMGeo TrainPBR45 + Real Test + Held-Out PBR Validation

## Status

Running / pending results.

## Setup

| Field | Value |
| --- | --- |
| Run name | `lmgeo_trainpbr45_real_and_new_val_30ep_v3` |
| Data config | `configs/data/lmgeo_trainpbr45_real_and_new_val.yaml` |
| Train config | `configs/train/train_lmgeo_finetune_lowres.yaml` |
| Objects | LM-O objects `[1, 5, 6, 8, 9, 10, 11, 12]` |
| References | `train/<object_id>` object renders |
| Train queries | `train_pbr` synthetic/PBR windows, excluding folders `000045`-`000049` |
| Held-out PBR queries | `new_val/000045`-`new_val/000049` |
| Real validation queries | real LM-O `test` targets from `lmo/test_targets_bop19.json` |
| Query masking | false |
| Reference masking | true |
| Training N/K | random `N=5..6` references, random `K=1..20` queries |
| Resolution | `224 x 224` |
| Epochs | 30 |
| Steps | 500 steps/epoch, 15k total |
| Important config | `lmgeo.filter_preprocessed_query_depth=false` |

## Validation Partitions

| Partition | Query split | N refs | K queries | Cap | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| `real_test` | `test` BOP targets | 5 | 1 | none | Final real-image validation signal. |
| `pbr_new_val` | `new_val` | 5 | 1 | none | In-distribution held-out PBR baseline. |
| `pbr_new_val_k5_subset` | `new_val` | 5 | 5 | 160 windows | Test whether more query context helps. |
| `pbr_new_val_k10_subset` | `new_val` | 5 | 10 | 160 windows | Test stronger query-context/parallax effect. |

## Outputs

| Artifact | Path |
| --- | --- |
| Log file | `logs/lmgeo_trainpbr45_real_and_new_val_30ep_v3_*.log` |
| TensorBoard dir | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_trainpbr45_real_and_new_val_30ep_v3` |
| Checkpoints | `/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/lmgeo_trainpbr45_real_and_new_val_30ep_v3/ckpts` |

## Results

To be filled after the run completes.

### Main Metrics

| Epoch | Split | Train loss | Val loss | ADD(-S)<0.1d | ADD-S<0.1d | Median d | Rot med | Trans med | Ref Chamfer d |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TBD | `real_test` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | `pbr_new_val` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | `pbr_new_val_k5_subset` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | `pbr_new_val_k10_subset` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Conclusions

- TBD.
- Compare `real_test` vs `pbr_new_val` to estimate the synthetic-to-real domain gap.
- Compare `pbr_new_val`, `pbr_new_val_k5_subset`, and `pbr_new_val_k10_subset`
  to estimate whether additional query views help under the same synthetic/PBR
  distribution.

