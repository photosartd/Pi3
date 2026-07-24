# Pi3 Ray-Map Conditioning for LMGeo Recenter/Zoom

For the optional crop+original query companion experiment and its four-way
comparison matrix, see
[lmgeo_paired_query_views.md](lmgeo_paired_query_views.md).

This experiment adds the small camera-intrinsics conditioning mechanism from
Pi3X to the existing Pi3 training implementation. It is a paired comparison
against the current oracle recenter/zoom K=1 baseline, not a Pi3X training
reconstruction.

## Experimental invariant

The ray-conditioned run starts from the same Pi3 checkpoint and keeps the
existing data, losses, trainable decoder/head modules, resolution, view budget,
and validation protocol. The only model-side change is an optional
zero-initialized ray patch projection.

- Baseline train config:
  `train_lmgeo_finetune_a40_46gb_recenter_zoom_k1`
- Ray train config:
  `train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1`
- Shared data config:
  `lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1`

`model.use_ray_conditioning=false` does not construct the module. Historical
Pi3 checkpoints and baseline model structure therefore remain unchanged.

The implementation is pinned against upstream Pi3X commit
[`9fa3ddb`](https://github.com/yyfz/Pi3/commit/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b):
the [zero-initialized two-channel `PatchEmbed`](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/pi3/models/pi3x.py#L84-L87),
the [half-pixel grid and inverse-intrinsics rays](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/pi3/models/pi3x.py#L304-L313),
and [addition to image patch tokens](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/pi3/models/pi3x.py#L365-L374).
This experiment intentionally ports only that ray branch. It does not add
Pi3X's depth encoder, pose-injection blocks, stochastic conditioning masks, or
Pi3X heads.

## Geometry and tensor shapes

Every reference and query view uses the same conditioning path. References use
their final intrinsics after ordinary Pi3 crop/resize. Queries use their final
intrinsics after oracle recenter/zoom and the same ordinary preprocessing.

For every final pixel center, the model computes:

```text
r = inverse(K) * [u + 0.5, v + 0.5, 1]
ray_map = [r_x / r_z, r_y / r_z]
```

The map is generated from the final `K` inside the model; it is not stored,
warped, or resized by the dataset. A zero-initialized `PatchEmbed` with a 14x14
kernel and stride maps it to the DINO patch layout and adds it to the RGB patch
tokens before the alternating Pi3 decoder.

At the fixed 560x420 resolution:

```text
images:     B x N x 3 x 420 x 560
intrinsics: B x N x 3 x 3
ray maps:   B x N x 420 x 560 x 2
ray tokens: (B*N) x 1200 x 1024
```

The `img_size=224` value in DINO/Pi3X `PatchEmbed` is metadata. Its forward
method accepts any height and width divisible by 14.

## Optimization

The paired A40 setup uses:

```yaml
train:
  optimizer:
    lr: 5e-6
    encoder_lr: 0.0
    ray_lr: 1e-5
```

The ray group carries an explicit OneCycle peak LR so the historical scalar
decoder cap does not reduce it to `5e-6`. TensorBoard logs `ray_lr` separately.

The ray projection has 402,432 parameters:

```text
1024 * 2 * 14 * 14 weights + 1024 biases
```

The large Pi3 parameter counts are:

```text
baseline total/trainable: 892,366,936 / 587,995,224
ray total/trainable:      892,769,368 / 588,397,656
delta:                        402,432 /     402,432
```

Loading the historical Pi3 checkpoint into the conditioned model leaves only
`ray_embed.proj.{weight,bias}` missing, as intended. Before an optimizer step,
zero initialization gives bit-exact equality with the baseline for
`points`, `local_points`, and `camera_poses`.

## Measured memory and view budget

Paired one-step BF16 training probes on an RTX PRO 6000 Blackwell, using the
real LMGeo recenter/zoom data at 560x420, measured:

| Shape per rank | Baseline reserved | Ray reserved | Delta |
| --- | ---: | ---: | ---: |
| 1 sample x 17 views | 21,512 MiB | 21,552 MiB | 40 MiB |
| 9 samples x 3 views | 31,474 MiB | 31,574 MiB | 100 MiB |

Both the maximum-view and packed-min probes completed their forward, loss,
backward, and optimizer steps. The adapter adds less than 0.4% to peak reserved
memory in these probes. Keep `train.max_img_per_gpu: 28`: the K=1 sampler's
realized extrema remain 17 images for one sequence and 27 images for the packed
3-view case. Run the documented A40 smoke before a production cluster job,
because allocator behavior and background usage differ by GPU and host.

## Detached validation metrics

`RayGeometryMetric` is logged under
`val_metrics/<loader>/ray_geometry/...`, independently of visualizations. It
uses predicted local XYZ and the known final `K`, without changing the loss.
Reference and query metrics are separated.

Primary outputs are:

- angular mean/median error in degrees;
- pixel reprojection mean/median error;
- recovered `fx`/`fy` relative error;
- recovered `cx`/`cy` absolute pixel error;
- valid intrinsics-fit fraction.

The angular and reprojection metrics are invariant to point-map scale. The
reported values are view-balanced aggregates of per-view mean/median summaries,
so full ref16 validation does not retain every sampled pixel in memory. The
intrinsics fit uses `u = fx * X/Z + cx` and `v = fy * Y/Z + cy`; it skips
non-canonical or underconstrained views.

## Compose without a GPU

```bash
python scripts/train_pi3.py --cfg job \
  train=train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1 \
  data=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1
```

## Local one-GPU smoke

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1 \
  data=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1 \
  name=lmgeo_recenter_zoom_ray_local_smoke \
  train.image_num_range='[3,3]' \
  train.max_img_per_gpu=3 \
  lmgeo_profile.num_reference_range='[2,2]' \
  lmgeo_profile.num_query_range='[1,1]' \
  lmgeo_profile.filter_preprocessed_query_depth=false \
  lmgeo.filter_preprocessed_query_depth=false \
  train.num_epoch=1 \
  train.iters_per_epoch=1 \
  val_datasets.real_test.runtime.iters_per_test=1 \
  val_datasets.real_test.dataset.filter_target_preprocessed_depth=false \
  test_dataset.filter_target_preprocessed_depth=false \
  +val_datasets.real_test_ref16.enabled=false \
  +val_datasets.pbr_new_val.enabled=false \
  +val_datasets.pbr_new_val_ref16.enabled=false \
  visuals.enabled=false \
  log.save_best=false \
  log.save_checkpoints=false
```

This diagnostic deliberately disables the preprocessed-depth filters so dataset
construction cannot turn into the bottleneck. It keeps one real validation
batch so the ray metrics and TensorBoard paths are exercised. Use explicit
output/checkpoint/TensorBoard paths for a retained run.

## CITEc Slurm workflow

```bash
export PI3_TRAIN_CONFIG=train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1
export PI3_DATA_CONFIG=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1
export PI3_RUN_NAME=lmgeo_a40_recenter_zoom_ray_k1

scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh preflight
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

Because both config names contain `recenter_zoom` and `k1`, the existing smoke
workflow selects the 17-view maximum and packed 3-view K=1 probes.
