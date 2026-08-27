# 2026-08-25 MegaPose-GSO metric-scale virtual-query setup

Status: implementation validated; production training not started.

## Question and isolated comparison

Test whether Pi3 can learn calibrated object pose when each GSO query is
rectified to a centred virtual camera and independently zoomed, without hiding
metric errors behind Pi3's fitted sample scale. The experiment deliberately
keeps the completed 560x420 fine-tune's architecture, checkpoint, optimizer,
N=5/K=1 sampling, scene/render mixture, masks, photometric augmentation, ray
conditioning, and 15,000-update schedule.

| Field | Value |
| --- | --- |
| 336x252 train config | `train_megapose_gso_geometry_pi3_finetune_ray_metric_virtual_rtxpro6000_70gb_336x252` |
| 560x420 train config | `train_megapose_gso_geometry_pi3_finetune_ray_metric_virtual_a40_40gb_560x420` |
| Data config | `megapose_gso_geometry_n5_k1_masked_metric_virtual_query` |
| Resolution | 560x420, 1,200 patches/view |
| Views | N=5 references, K=1 query |
| Query zoom | log-uniform requested focal multiplier in [1, 5] during training |
| Validation zoom | deterministic requested multiplier 3 |
| Scale supervision | metric metres; no GT normalization or prediction-to-GT fitted scale |
| Existing indexes | reused unchanged |

## Implementation contract

`Pi3Loss(scale_mode=...)` is the only new loss-level switch. Its default is
`aligned`, which exactly retains historical Pi3 sample normalization and fitted
scale. The new profile uses `metric`: GT points and camera translations remain
in metres, predicted points are not normalized, the point loss receives unit
scale, and camera translation is not rescaled. The first-camera SE(3) gauge
change remains because it removes coordinate-frame choice, not metric scale.

`VirtualCameraRectifier` is a reusable object-dataset transform. For configured
query roles it:

1. rotates the annotated object-box centre ray to the image-centre optical axis;
2. samples a focal multiplier and caps it using the expanded amodal object box;
3. warps RGB, z-depth, object mask, and validity with the same homography;
4. corrects z-depth for the virtual camera rotation;
5. updates intrinsics, `T_C_O`, and `camera_pose=inv(T_C_O)` while preserving the
   physical camera centre;
6. bypasses only the query's historical planned crop, then reapplies the exact
   object-only mask. References retain their existing calibrated planned crops.

The final per-view intrinsics continue into the ray-conditioning input. No
stored examples, SQLite schemas, or plan catalogues change. The existing plans
still impose their historical focal-feasibility filter, so this first test does
not recover the full theoretical capacity of unrestricted metric sampling.

Evaluation is also scale-fixed (`solve_scale=false`). `reference_depth` remains
enabled only so the established depth diagnostic fields can be logged; its
estimated scale is not applied to points or camera poses.

## Validation evidence

- Focused loss/warp/config/LMGeo-regression suite: 40/40 passed.
- Full CPU suite: 231/231 passed.
- Hydra production composition passed and resolves metric train/test losses,
  two virtual-camera train components, two virtual-camera validation loaders,
  560x420 input, and the unchanged geometry-plan paths.
- Real data materialized successfully for both scene-to-scene and
  render-to-scene. Each sample has five references plus one query; all RGB,
  mask, depth, intrinsics, poses, and ray maps have compatible geometry.
- Visual sheets were generated with the resolution-independent diagnostic at
  `/tmp/pi3_metric_virtual_inputs` during implementation.

A 30-query probe per training component measured actual (safe-capped) zoom:

| Training component | Applied zoom min / p25 / median / p75 / max | Median query occupancy | Capped draws |
| --- | --- | ---: | ---: |
| Scene references | 1.01 / 1.32 / 1.87 / 2.57 / 4.09 | 9.45% | 33.3% |
| Render references | 1.03 / 1.81 / 2.21 / 2.92 / 4.37 | 7.02% | 16.7% |

This is meaningful focal diversity rather than a fixed crop. It does not force
every object to a common occupancy: edge-on objects and boxes already near an
image boundary can have a lower safe cap.

One update of full Pi3 at 560x420 on an RTX PRO 6000 Blackwell passed training,
both validation loaders, all enabled metrics, finite backward, and checkpoint
save. It used one six-view sample and reached 12,511 MiB peak allocated and
12,632 MiB peak reserved. The raw base-checkpoint metric train loss was 1.2450
(`local=0.5563`, `translation=0.0552`, `rotation=1.3624`); scene/render
validation losses were 0.7784/1.2646. These are smoke values after one update,
not evidence of final accuracy. TensorBoard for the smoke is at:

```text
/tmp/pi3_metric_virtual_metric_smoke/tensorboard/pi3_metric_virtual_metric_smoke
```

Both scale-aware evaluation plugins logged alignment scale exactly 1.0. This
confirms the new evaluation does not silently restore the removed Sim(3) scale.

## Risks and decision gate

- The released checkpoint was trained scale-invariantly. Its initial metric
  output scale is poor, so this is a harder transition than ordinary fine-tune.
  Run a fixed-batch memorization control before trusting a full run.
- The point error is inverse-depth weighted and the camera loss is pairwise;
  these established choices were intentionally not changed in this minimal
  ablation. Metric scale is identifiable, but the point scalar is not plain
  mean absolute metres.
- Camera distance and GSO object identity/size now provide the only evidence for
  absolute depth. GT depth and GT pose remain targets, not model inputs.
- A 5x request is a ceiling, not a promise. Safe cropping may reduce it.
- The high `clip_loss=1000` prevents the trainer's historical batch-drop guard
  from silently zeroing valid metric transition batches; gradient clipping is
  unchanged.

The recommended gate is: overfit one deterministic 24-sample batch, verify
metric local-point and camera-translation errors collapse without a fitted
scale, then launch the unchanged 15k-update schedule. If it cannot memorize,
fix the metric objective before spending a full cluster run. If it memorizes
but held-out loss stalls, the limitation is generalization/scale evidence, not
the virtual-camera geometry.

## Cluster launch

After synchronizing this code to the cluster, use the existing data and index
bundle unchanged:

```bash
PI3_REPO_DIR=/homes/dtrofimov/repositories/Pi3 \
PI3_CONDA_ROOT=/homes/dtrofimov/miniconda3 \
PI3_CONDA_ENV=pi3-lmgeo \
PI3_DATA_ROOT=/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/MegaPose-GSO-fixed \
PI3_GSO_ASSETS_ROOT=/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/MegaPose-GSO-assets \
PI3_GSO_REFERENCES_ROOT=/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/MegaPose-GSO-assets/renders \
PI3_CKPT=/homes/dtrofimov/repositories/Pi3/ckpts/model.safetensors \
PI3_TRAIN_GPUS=2 PI3_TRAIN_CPUS=32 PI3_TRAIN_MEM=220G \
PI3_WALLTIME=4-00:00:00 PI3_CKPT_INTERVAL=5 PI3_MAX_CHECKPOINTS=5 \
PI3_TRAIN_CONFIG=train_megapose_gso_geometry_pi3_finetune_ray_metric_virtual_a40_40gb_560x420 \
PI3_DATA_CONFIG=megapose_gso_geometry_n5_k1_masked_metric_virtual_query \
PI3_RUN_NAME=megapose_gso_metric_virtual_query_560x420_$(date +%Y%m%d_%H%M%S) \
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train --fresh
```
