# LMGeo Failure-Mode Diagnostics

This diagnostic layer separates pose failures by amodal object size, visible
fraction, and object class. It is evaluation-only: none of these BOP GT
covariates enter model inputs or losses.

## Discovery Locations

- BOP root used by the LMGeo configs:
  `/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o`.
- BOP JSON loading and metadata propagation:
  `datasets/lmgeo_dataset.py`.
- Per-prediction ADD/ADD-S computation:
  `pi3/metrics/object_pose.py`.
- Validation loop:
  `trainers/base_trainer_accelerate.py`.
- Resize/crop logic for patch-count conversion:
  `datasets/base/base_dataset.py` and `pi3/utils/cropping.py`.

The local LM-O root has `scene_gt_info.json` for `train`, `train_pbr`,
`new_val`, and `test`.

## Build Covariate Tables

For the fixed 560x420 high-resolution profile:

```bash
DATA=/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o
RUN_DIR=/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3/<run-name>

python scripts/build_covariate_table.py \
  --dataset-root "$DATA" \
  --split test \
  --model-resolution 560 420 \
  --out "$RUN_DIR/covariates_test.parquet"

python scripts/build_covariate_table.py \
  --dataset-root "$DATA" \
  --split new_val \
  --model-resolution 560 420 \
  --out "$RUN_DIR/covariates_new_val.parquet"
```

The script writes one row per BOP GT instance keyed by
`scene_id, im_id, gt_id, obj_id`. `patch_count_amodal` uses the same
principal-point crop and resize scale as Pi3 preprocessing, then divides by the
14px DINO patch area.

## Raw Prediction Rows

`ObjectPoseMetric` can append raw rows to:

```text
${log.output_dir}/predictions.parquet
```

The 560x420 A40/Blackwell config enables this path. Rows include the join key, chosen
ADD or ADD-S error, normalized error, metric type, number of keyframes, number
of query views, seed, checkpoint step, BOP split, validation loader name,
masking flag, and run id.

The file is append-only across checkpoints. The aggregator defaults to the
latest checkpoint in the file unless `--checkpoint-step` is given.

## Aggregate Offline

```bash
python scripts/aggregate_failure_modes.py \
  --predictions "$RUN_DIR/predictions.parquet" \
  --covariates "$RUN_DIR/covariates_test.parquet" \
  --out "$RUN_DIR/analysis/real_test"
```

Outputs:

- `per_object_metrics.csv` and `per_object_metrics.png`
- `size_visib_heatmap.csv` and `size_visib_heatmap.png`
- `margin_size_visible.csv`
- `margin_visib_middle_size.csv`
- `margin_curves.png`
- `n_views_curve.csv` and `n_views_curve.png`
- `depth_spearman_by_size_bin.csv`
- `join_report.json`

For a specific checkpoint or validation loader:

```bash
python scripts/aggregate_failure_modes.py \
  --predictions "$RUN_DIR/predictions.parquet" \
  --covariates "$RUN_DIR/covariates_new_val.parquet" \
  --checkpoint-step 15000 \
  --val-name pbr_new_val_k10_subset \
  --out "$RUN_DIR/analysis/pbr_new_val_k10_subset_step15000"
```

The hard class is `symmetric OR bottom-two size bins` by default. BOP metadata
does not label textureless objects, so add those manually when needed:

```bash
--hard-object-ids 5,10,11
```

## Coarse Online Scalars

After covariates exist in `log.output_dir`, enable the 14-scalar online view:

```bash
metrics.items.object_pose.coarse_analysis=true
```

The hook writes raw rows, filters to the current checkpoint and validation
loader, runs the same aggregator in `--coarse` mode, and logs only:

- small/big x visible/occluded recall buckets,
- easy/hard class recall,
- per-object recall for the 8 LM-O objects.

## Acceptance Checks

Use `join_report.json` after aggregation:

- `orphan_rows` should be zero for the intended split/checkpoint.
- `rows_after_visibility` should match the sum of heatmap cell counts.
- `metric_type` in `predictions.parquet` should be `ADD-S` for symmetric
  objects and `ADD` otherwise.
- Overall `recall@0.1d` in `per_object_metrics.csv`, weighted by `n`, should
  match the existing `object_pose/query_add_s_0_1d` validation metric on the
  same checkpoint, except for any documented visibility-filter difference.
