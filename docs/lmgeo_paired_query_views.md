# Paired Recentered and Original Query Views

This experiment adds the original-camera query image beside the existing
oracle recentered/zoomed query. It is additive and opt-in. Historical
crop-only runs remain unchanged unless the paired data profile is selected.

## Four Comparable Regimes

| Regime | Train profile | Data profile |
| --- | --- | --- |
| crop only, no rays | `train_lmgeo_finetune_a40_46gb_recenter_zoom_k1` | `lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1` |
| crop only, rays | `train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1` | `lmgeo_trainpbr45_real_and_new_val_recenter_zoom_k1` |
| crop + original, no rays | `train_lmgeo_finetune_a40_46gb_recenter_zoom_k1` | `lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1` |
| crop + original, rays | `train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1` | `lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1` |

### Required Additional Runs

The final two rows are the two additional experiments to run. Use the same
base checkpoint, seed, iteration schedule, splits, and all other overrides as
the completed crop-only comparison.

1. **Paired query, no conditioning**

   ```text
   name=lmgeo-recenter-zoom-k1-plus-original-no-ray
   train=train_lmgeo_finetune_a40_46gb_recenter_zoom_k1
   data=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1
   ```

2. **Paired query, ray conditioning**

   ```text
   name=lmgeo-recenter-zoom-k1-plus-original-ray
   train=train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1
   data=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1
   ```

These are deliberately config compositions rather than two duplicated train
YAML files: the data profile owns the extra original-camera view and paired
metric, while the train profile owns whether the zero-initialized ray branch
exists. This keeps the two tests identical apart from conditioning.

The data profile is the activation boundary. Its
`lmgeo_query_recenter.include_original_query_view=true` expands each selected
physical query record into two model views. Selecting the parent
`recenter_zoom_k1` profile restores the previous experiment, omits the
original-camera view, and does not instantiate the paired metric.

The paired profile keeps `num_query_range: [1, 1]`; this counts physical query
records. Actual Pi3 view totals change from 3-17 to 4-18 during training, from
6 to 7 for the five-reference validation loaders, and from 17 to 18 for the
16-reference loaders. The dynamic sampler therefore budgets the real number
of model images rather than treating the companion as free.

## Data and Supervision

For one source query, the ordered model views are:

1. the existing bbox-oracle pure-rotation recenter and zoom, including its
   transformed RGB, depth, intrinsics, and `T_Ccrop_O`;
2. the original image through normal Pi3 center-crop/resize preprocessing,
   with its independently adjusted intrinsics and original `T_Corig_O`.

Both views retain full geometric supervision and participate in the unchanged
Pi3 point/camera losses. The original companion has `is_query_context=true`
and `is_query=false`, so established query metrics and correspondence
diagnostics still count one physical query rather than pooling two correlated
predictions. Explicit `is_cropped_query`, `is_original_query`, and
`query_pair_index` fields drive the paired metric.

Ray conditioning, when enabled by the train profile, is applied to references,
the crop, and the original companion using each view's final post-resize
intrinsics. No fixed 224px ray tensor is stored: the ray map is generated at
the current input height and width, then patchified by the learned projection.

## On-Demand Paired Metrics

`PairedQueryConsistencyMetric` is present only in the paired data profile and
runs only in validation because the inherited metric manager has
`train_enabled=false`. It is detached/no-grad, accumulates scalars only, and
uses at most 4096 deterministic pixels per pair.

Reference cameras alone estimate the usual world-to-object Sim(3); query GT
never participates in alignment. With the known pure-rotation camera transform

`A = T_Ccrop_Corig = [R_old_to_new, 0; 0, 1]`,

the metric reports:

- crop pose canonicalized to the original camera,
  `T_Corig_O_crop = inv(A) T_Ccrop_O_pred`;
- original-view pose directly in the original camera;
- ADD or ADD-S, rotation, and translation for those two predictions as
  separate rows, never as two independent queries;
- predicted-vs-known relative camera rotation and translation;
- disagreement between the two predicted camera centers and canonical poses;
- crop reprojection error from the predicted relative camera transform;
- dense local point equivariance,
  `X_crop(H p) ~= R_old_to_new X_orig(p)`;
- dense shared-world point consistency after applying each predicted camera
  pose.

The existing `object_pose` metric remains available for continuity and still
evaluates the recentered view using its transformed GT. The paired metric adds
the canonical original-camera comparison required to determine whether the
network recognizes that the companion is the same physical observation.

Metric wall time is already printed per validation iteration as
`metric_paired_query`. Because dense work is capped at 4096 points and ADD-S
nearest-neighbor matching is used only for configured symmetric objects, its
cost should normally remain below the model forward and the existing full
metric set. Use the measured smoke/run timing rather than assuming this at a
new resolution or validation batch size.

## Local Commands

Compose both new regimes without allocating a GPU:

```bash
python scripts/train_pi3.py --cfg job \
  train=train_lmgeo_finetune_a40_46gb_recenter_zoom_k1 \
  data=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1

python scripts/train_pi3.py --cfg job \
  train=train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1 \
  data=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1
```

For a short local run, add the normal checkpoint/output overrides and disable
every preprocessed-depth filter:

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 scripts/train_pi3.py \
  train=train_lmgeo_finetune_a40_46gb_recenter_zoom_k1 \
  data=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1 \
  lmgeo_profile.filter_preprocessed_query_depth=false \
  lmgeo.filter_preprocessed_query_depth=false \
  test_dataset.filter_target_preprocessed_depth=false \
  val_datasets.real_test.dataset.filter_target_preprocessed_depth=false \
  val_datasets.real_test_ref16.dataset.filter_target_preprocessed_depth=false \
  train.image_num_range='[4,4]' \
  lmgeo_profile.num_reference_range='[2,2]' \
  train.num_epoch=1 train.iters_per_epoch=1
```

Use the ray train profile in the same command for the conditioned variant.

### Inspect One Actual Dataloader Batch

The following command uses the production validation dataloader and collator,
with depth filtering forced off, and writes a labeled PNG plus its numerical
summary:

```bash
python scripts/visualize_lmgeo_paired_query_batch.py \
  --train-config train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1 \
  --data-config lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1 \
  --output outputs/paired_query_debug/paired_query_dataloader.png
```

The visual contains all seven ordered RGB inputs, a homography-linked
crop/original query pair, final intrinsics, numerical pose/homography checks,
and the two ray maps generated from the final post-resize intrinsics. The RGB
batch is identical in the no-ray run; only the ray branch is absent.

## Slurm

The standard CITEc wrapper recognizes the paired data profile. Its smoke mode
runs both the 18-view maximum and the packed `7 x 4-view` minimum, while
forcing all preprocessed-depth filters off:

```bash
PI3_TRAIN_CONFIG=train_lmgeo_finetune_a40_46gb_recenter_zoom_k1 \
PI3_DATA_CONFIG=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1 \
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke

PI3_TRAIN_CONFIG=train_lmgeo_finetune_a40_46gb_recenter_zoom_ray_k1 \
PI3_DATA_CONFIG=lmgeo_trainpbr45_real_and_new_val_recenter_zoom_plus_original_k1 \
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke
```

Run `preflight` first on a new allocation. Start production with `train` only
after both memory extremes pass on the actual A40.

The implementation validation, TensorBoard checks, metric timings, and local
44 GiB memory probes are recorded in
[experiment_results/2026-07-24_lmgeo_paired_query_implementation.md](experiment_results/2026-07-24_lmgeo_paired_query_implementation.md).
