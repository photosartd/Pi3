# MegaPose-GSO safe-fill queries and metric reference-depth conditioning

Date: 2026-08-27  
Status: implementation validated; one-batch overfit and conditioning ablation complete

## Question

Can the metric-scale virtual-camera setup benefit from aggressive, object-safe
query crops and metric reference depth without adding a second DINO encoder?
The intended deployment contract is that reference depth is available, while
query depth is not.

This is an additive experiment. Historical configs retain their old focal
matching, 1--5x zoom, model inputs, checkpoint structure, and optimizer groups.

## Implementation

### Safe-fill virtual camera

`datasets/virtual_camera.py` now supports `zoom_policy: safe_fill`. For each
query it computes the largest calibrated zoom that retains the expanded amodal
object box in the 4:3 output, then samples 80--100% of that limit in training.
Validation uses a deterministic 90%. The existing `zoom_policy: range` remains
the default.

Safe-fill is an extent constraint, not a literal area target. A thin or
edge-on object can touch the safe image bounds and still occupy a small fraction
of pixels. The applied zoom and safe-fill fraction are preserved in sample
metadata.

The new data profile disables only query/reference focal compatibility. The
existing plan databases are reused and still enforce:

- all references from one coherent scene/track;
- per-reference visibility/quality constraints;
- reference-set surface coverage;
- at least one reference within the configured viewing-direction angle.

Reference crops remain internally focal-compatible with their own plan. No
index rebuild is needed for this first experiment.

### Factored metric-depth tokens

`pi3/models/depth_conditioning.py` implements a lightweight additive decoder
condition with two factors per view:

1. `depth / mean_valid_depth` plus a validity channel are patch-embedded;
2. `log(mean_valid_depth / 1 metre)` is embedded by a small MLP and broadcast
   to all patches in that view.

Both output projections are zero-initialized. Enabling the branch on the base
Pi3 checkpoint is therefore an exact no-op at step zero. After every projection,
unknown views are hard-gated to zero, including learned biases.
The full-Pi3 instance adds 534,784 trainable depth-adapter parameters, rather
than another image encoder.

The production policy conditions all reference views together in 90% of train
samples and leaves 10% unconditioned. Validation always conditions references.
Queries are never conditioned in either mode. This matches the intended
reference-depth-only use case and prevents query-depth leakage.

The branch has an independent `metric_depth_lr` optimizer group. The production
profile uses `1e-5`, while historical Pi3 weights retain `5e-6`.

### Diagnostics

`pi3/metrics/metric_depth.py` reports reference/query values separately and
splits them into conditioned/unconditioned views:

- dense camera-space Z MAE in metres and relative MAE;
- predicted/GT mean-depth scale ratio and absolute log-scale error;
- camera-space object-centre error;
- 1 cm/5 cm depth recall and 5 cm/10 cm centre recall;
- conditioned fractions and view counts.

These are raw metric predictions; they do not perform per-view scale alignment.
The raw-depth visual panel follows the same rule and labels whether each view
received depth. The ordinary scale-aligned depth visual remains unchanged for
old configs.

## Configs

- data: `megapose_gso_geometry_n5_k1_masked_metric_safe_fill_depth`
- train: `train_megapose_gso_geometry_pi3_finetune_ray_metric_depth_safe_fill_rtxpro6000_70gb_336x252`
- fixed-batch data: `megapose_gso_geometry_n5_k1_masked_metric_safe_fill_depth_overfit_one_batch`
- fixed-batch train: `train_megapose_gso_geometry_pi3_finetune_ray_metric_depth_safe_fill_overfit_one_batch_336x252`

The production run remains a 336x252, N=5/K=1, metric-loss, ray-conditioned,
frozen-encoder full-Pi3 fine-tune. It inherits the established 15,000-update
schedule (30 x 500) and 500-step warmup.

## Visual inspection

Four scene-to-scene and four render-to-scene input grids were generated at:

`/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_metric_depth_safe_fill_implementation_20260827/input_visuals`

Each grid shows RGB, raw metric depth and its scale, normalized depth-adapter
input, and ray maps. Intrinsics, RGB, masks, and depth remain registered after
the virtual-camera warp. Sampled query zooms included 1.55x, 2.42x, 2.79x, and
5.23x; this confirms the new policy is not capped by the historical 5x range.
The corresponding safe-fill fractions were 0.82--0.94.

Literal visible-mask occupancies in those four queries ranged from 4.8% to
37.1%. This is expected for amodal-extent safe-fill with thin and partially
occluded objects and must not be described as 80--100% image-area occupancy.

## One-batch overfit

Run:

`megapose_gso_metric_depth_safe_fill_overfit_20260827`

Artifacts:

- output: `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_metric_depth_safe_fill_overfit_20260827`
- TensorBoard: `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_metric_depth_safe_fill_overfit_20260827/tensorboard/megapose_gso_metric_depth_safe_fill_overfit_20260827`
- final checkpoint: `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_metric_depth_safe_fill_overfit_20260827/ckpts/checkpoint_59`
- log: `logs/megapose_gso_metric_depth_safe_fill_overfit_20260827.log`

The test repeatedly used one fixed scene-to-scene sample: five references from
scene 2809 and one query from scene 31. It ran for 60 optimizer updates in bf16.
Peak memory was 12,446 MiB allocated and 12,734 MiB reserved under the 70 GiB
allocator cap. Wall time was 1 minute 37 seconds including checkpoint saves.

At step zero, both depth-token magnitudes and both output-projection norms were
exactly zero. At step 60, the spatial projection norm was 0.213, the scale
output norm 0.129, and spatial/scale token absolute means were 0.0168/0.0092.
The branch therefore left the initial checkpoint unchanged and then learned.

### Final fixed-batch metrics

| Metric | Step 60 |
| --- | ---: |
| validation loss | 0.00151 |
| local point loss | 0.00120 |
| query rotation | 0.0111 deg |
| query translation | 0.00191 m |
| query camera centre | 0.00208 m |
| query ADD at 0.1d | 100% |
| conditioned-reference Z MAE | 0.00323 m |
| conditioned-reference object-centre error | 0.00214 m |
| conditioned-reference predicted/GT depth scale | 0.99878 |
| unconditioned-query Z MAE | 0.00913 m |
| unconditioned-query object-centre error | 0.00681 m |
| unconditioned-query predicted/GT depth scale | 0.99460 |

The validation loss fell from 0.0451 at step 10 to 0.0067 at step 30, 0.0018
at step 50, and 0.0015 at step 60.

## Conditioning-off checkpoint ablation

The final checkpoint was evaluated on the identical fixed batch with reference
depth disabled (`metric_depth_eval_reference_probability=0`). No weights were
updated.

| Metric | Depth on | Depth off | Off/on |
| --- | ---: | ---: | ---: |
| validation loss | 0.00151 | 0.00801 | 5.3x |
| query rotation | 0.0111 deg | 0.921 deg | 82.7x |
| query translation | 0.00191 m | 0.01781 m | 9.3x |
| query camera centre | 0.00208 m | 0.03171 m | 15.2x |
| reference Z MAE | 0.00323 m | 0.01886 m | 5.8x |
| reference object-centre error | 0.00214 m | 0.01886 m | 8.8x |
| reference predicted/GT depth scale | 0.99878 | 0.98843 | -- |
| query Z MAE | 0.00913 m | 0.02088 m | 2.3x |

This is direct evidence that the memorized checkpoint consumes the depth
condition and propagates its benefit to the unconditioned query. It is not yet
evidence of held-out-object generalization: the decoder, ray adapter, and depth
adapter were optimized together on one sample. The production validation run
must establish that.

## Validation performed

- Full CPU suite: 243 tests passed in 10.339 seconds.
- New safe-fill geometry, focal-relaxation, adapter, zero-init/gating, metric,
  optimizer, and Hydra isolation tests pass.
- New production and overfit configs compose successfully.
- The previous metric-virtual train/data pairing still composes with
  `use_metric_depth_conditioning: false` and its historical zoom/focal policy.
- Real-data input grids were generated and inspected.
- A 60-step GPU batch overfit and depth-off evaluation ablation completed.

## Production command

```bash
RUN=megapose_gso_metric_depth_safe_fill_336x252_$(date +%Y%m%d_%H%M%S)
DATA_ROOT=/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed
ASSETS_ROOT=/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets
REFERENCES_ROOT=/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets/renders
OUTPUT_ROOT=/media/internal/nvme/dtrofimov/spott3r/outputs/$RUN

CUDA_VISIBLE_DEVICES=0 \
PI3_CUDA_MEMORY_LIMIT_GIB=70 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
HYDRA_FULL_ERROR=1 \
nohup /home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python -u scripts/train_pi3.py \
  train=train_megapose_gso_geometry_pi3_finetune_ray_metric_depth_safe_fill_rtxpro6000_70gb_336x252 \
  data=megapose_gso_geometry_n5_k1_masked_metric_safe_fill_depth \
  name="$RUN" \
  gso.data_root="$DATA_ROOT" \
  gso.assets_root="$ASSETS_ROOT" \
  gso.references_root="$REFERENCES_ROOT" \
  gso.geometry_train_index="$DATA_ROOT/pi3_index/megapose_gso_geometry_v1.sqlite" \
  gso.geometry_val_index="$DATA_ROOT/pi3_index/megapose_gso_geometry_val_v1.sqlite" \
  gso.scene_train_plans="$DATA_ROOT/pi3_index/megapose_gso_geometry_train_n5_plans.sqlite" \
  gso.scene_val_plans="$DATA_ROOT/pi3_index/megapose_gso_geometry_val_n5_plans.sqlite" \
  gso.render_geometry_index="$REFERENCES_ROOT/pi3_index/render_geometry_v1.sqlite" \
  gso.render_plans="$REFERENCES_ROOT/pi3_index/render_geometry_n5_plans.sqlite" \
  model.ckpt=/home/dtrofimov/repositories/Pi3/ckpts/Pi3/model.safetensors \
  log.output_dir="$OUTPUT_ROOT" \
  log.tensorboard_dir="$OUTPUT_ROOT/tensorboard" \
  log.ckpt_dir="$OUTPUT_ROOT/ckpts" \
  hydra.run.dir="$OUTPUT_ROOT" \
  log.use_tensorboard=true \
  log.ckpt_interval=5 \
  log.max_checkpoints=3 \
  > "logs/$RUN.log" 2>&1 &
```

TensorBoard:

```bash
tensorboard --logdir "$OUTPUT_ROOT/tensorboard" --port 6006
```

## Query-depth extension

Added later on 2026-08-27 as an isolated child profile:

- production: `train_megapose_gso_geometry_pi3_finetune_ray_metric_depth_query_safe_fill_rtxpro6000_70gb_336x252`
- fixed-batch smoke: `train_megapose_gso_geometry_pi3_finetune_ray_metric_depth_query_safe_fill_overfit_one_batch_336x252`

No model or dataset-layout change was necessary. The conditioner and input
adapter were already role-aware. The child profile changes only:

```yaml
model:
  metric_depth_query_probability: 0.9
  metric_depth_eval_query_probability: 1.0
```

The inherited reference probabilities are 0.9 in training and 1.0 in
evaluation. Because production uses sample-level dropout, 90% of train samples
condition all five references and the query together; 10% expose no depth.
Evaluation deterministically conditions every view. All four probabilities can
be overridden independently from Hydra. The existing reference-only profile
still has both query probabilities set to zero.

The query receives the existing metric, object-valid depth after the calibrated
virtual-camera transform. The stored dataset/TAR/index contents are not changed.

### Query-depth GPU smoke

Run:

`megapose_gso_metric_depth_query_safe_fill_smoke_20260827`

The real fixed N=5/K=1 batch completed one bf16 optimizer step, validation,
metric collection, visual logging, and checkpoint save. Observed routing:

- conditioned reference fraction: 1.0 (5/5 views);
- conditioned query fraction: 1.0 (1/1 view);
- input query mean metric depth: 1.2617 m;
- peak allocated/reserved GPU memory: 12,446/12,618 MiB;
- exit code: 0.

The zero-initialized checkpoint's first validation prediction had query depth
MAE 0.223 m, object-centre error 0.216 m, and predicted/GT mean-depth ratio
1.171. These are baseline measurements, not an expected trained result: one
step is only an integration smoke and the additive depth tokens start at zero.

Artifacts:

- output: `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_metric_depth_query_safe_fill_smoke_20260827`
- TensorBoard: `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_metric_depth_query_safe_fill_smoke_20260827/tensorboard/megapose_gso_metric_depth_query_safe_fill_smoke_20260827`
- checkpoint: `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_metric_depth_query_safe_fill_smoke_20260827/ckpts/checkpoint_0`

After adding the query profile, the full CPU suite passes 246 tests in 11.770
seconds.

To launch the all-view-depth production run, use the production command above
but replace its train config with:

```text
train=train_megapose_gso_geometry_pi3_finetune_ray_metric_depth_query_safe_fill_rtxpro6000_70gb_336x252
```
