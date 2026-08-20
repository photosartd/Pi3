# Geometry-matched LM-O transfer evaluation

This protocol evaluates a geometry-constrained MegaPose-GSO checkpoint on
LM-O `new_val` while keeping the same assumptions that made the GSO N=5/K=1
task well defined. It is additive: historical `LMGeoSequenceDataset`, data
profiles, and GSO profiles retain their existing behavior.

## Contract

Each sample contains five clean LM-O `train/<object_id>` references followed by
one `new_val` query. The reference set must satisfy:

- every reference has BOP visible fraction at least 0.3;
- the exact depth-tested union of 8,192 CAD surface samples is at least 0.5;
- at least one reference camera-center direction is within 10 degrees of the
  GT query direction;
- one 4:3 crop-only focal target is feasible for all six views and is at most
  10% above the query's native normalized focal;
- the expanded full-object box, including a 5% margin, remains in every crop.

RGB and metric depth are masked by the exact visible-object mask after the
final resize to 336x252. The crop updates `K` and leaves `T_C_O` and
`camera_pose=inv(T_C_O)` unchanged. The model receives the resulting ray map;
visibility masks are targets/processing masks, not mask-conditioning channels.
Photometric augmentation is disabled for evaluation.

The within-10-degree lookup uses query GT. This is intentional: the experiment
tests the hypothesis "adequate calibrated keyframes were supplied." It is an
oracle retrieval evaluation, not a deployable retrieval method, and query GT
is never used in the reference-only Sim(3) alignment. The current matched
LM-O profile uses `scale_estimation=reference_depth`: reference camera
orientations still determine Sim(3) rotation, reference camera centers still
determine translation, and only the global scale is replaced by the median
metric-depth ratio from the five references. Historical metric and
visualization configs retain `camera_centers` as their default.

For reference view `i`, the estimator takes the median over valid object pixels
of `GT_depth / predicted_local_z` (implemented as a median log ratio). It then
combines the five view estimates with an equal-view-weight median in log space.
The depths are measured from each camera origin and are not centered around the
visible surface: such centering would make the scale estimate invariant to a
different translation in every reference and would no longer estimate the
scale of one global Sim(3).
At least 64 valid pixels are required from a view. Insufficient reference depth
triggers the explicit underconstrained/fallback counter; query depth is never
inspected.

When reference-depth alignment is active, the camera metric additionally
reports reference/query diagnostics separately:

- total, radial, and tangential camera-center error, camera-radius error, and
  camera-direction error;
- metric local-depth MAE, median absolute/relative/bias error, and per-view
  depth-scale disagreement after applying the one reference-derived scale.

Query GT depth is read only by these diagnostic accumulators, after the
reference-only alignment has already been fixed. It cannot change the Sim(3),
pose prediction, or reported pose error. Historical camera-center alignment
profiles do not require depth and retain their previous behavior.

The reconstruction card is an aggregate of the five aligned reference point
maps. Its outline is therefore not a pure scale diagnostic: it also contains
reference rotation/center residuals and non-Sim(3), potentially anisotropic
point-map errors. The card prints the actual alignment mode and scalar. A
surface-centered extent fit is useful as an offline diagnostic, but it is not
the default pose-evaluation alignment.

The offline exporter must render visuals and compute metrics before calling
`calculate_loss`, matching the trainer's validation order. Pi3 loss
normalization mutates predicted local points and camera translations in place;
rendering the still-raw composed world points after that mutation mixes two
coordinate scales. A regression test protects this ordering.

## Implementation

- `datasets/preprocess/lmgeo_geometry.py` builds BOP geometry sidecars using
  the same schema, surface sampler, visibility rasterizer, crop metadata, and
  plan builder as MegaPose-GSO.
- `scripts/prepare_lmgeo_geometry_eval.py` builds/resumes the reference index,
  lightweight query index, N=5 plans, and reports camera/data distributions.
- `LMGeoGeometryMatchedSequenceDataset` is an opt-in
  `LMGeoSequenceDataset` subclass. It filters the normal LMGeo window manifest,
  matches plans, attaches planned crops, and uses the shared object-view
  processor. Only stable `frame_id`, `scene_id`, and `view_id` fields were added
  to legacy LMGeo records; old consumers ignore them.
- `scripts/visualize_lmgeo_geometry_inputs.py` asserts the role order, masking,
  crop/focal equality, 10-degree positive, and coverage floor, then writes RGB,
  depth, and ray-map sheets.
- `lmgeo_new_val_geometry_render_n5_k1_masked` is the isolated data profile.
  `train_megapose_gso_geometry_pi3_lmgeo_eval_rtxpro6000_70gb_336x252` restores
  the full Pi3+ray model shape. `general=lmgeo_geometry_eval` disables
  checkpoint writes; this separate group is necessary because Hydra composes
  `general` after `train`.

## Build the derived indexes

The command is resumable and verifies source signatures before reusing output:

```bash
/home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python \
  scripts/prepare_lmgeo_geometry_eval.py all \
  --data-root /vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o \
  --device cuda --workers 8 --batch-frames 16
```

Outputs are under `<LM-O>/pi3_index/`:

- `lmgeo_surface_atlas_8192.f32[.json]`;
- `lmgeo_train_geometry_v1.sqlite[.surface_bits]`;
- `lmgeo_new_val_geometry_v1.sqlite[.surface_bits]`;
- `lmgeo_train_geometry_n5_plans.sqlite`.

On the RTX PRO 6000 host, the exact reference coverage pass processed 10,504
frames at 128.7 frames/s in 81.5 seconds and used about 55 MB reserved CUDA
memory. Query metadata and 4,087 plans take only seconds.

## Inspect and visualize before evaluation

```bash
/home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python \
  scripts/prepare_lmgeo_geometry_eval.py summary \
  --data-root /vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o

/home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python \
  scripts/visualize_lmgeo_geometry_inputs.py \
  --samples 4 \
  --output /media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_matched_visual_check
```

The visualizer fails rather than drawing a sheet if RGB/depth leak outside the
mask, a view is empty, focal targets differ, the positive exceeds 10 degrees,
or reference coverage is below 0.5.

To export trained-model inputs and outputs for an object-stratified gallery,
including GT/predicted pose overlays, aligned depth/error panels, reference
reconstruction, per-example metrics, and a combined overview:

```bash
CUDA_VISIBLE_DEVICES=0 PI3_CUDA_MEMORY_LIMIT_GIB=70 \
/home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python \
  scripts/export_lmgeo_geometry_visuals.py \
  train=train_megapose_gso_geometry_pi3_lmgeo_eval_rtxpro6000_70gb_336x252 \
  data=lmgeo_new_val_geometry_render_n5_k1_masked \
  general=lmgeo_geometry_visual_export \
  export.samples=20 \
  export.output=/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_matched_gallery
```

The exporter calls the actual validation trainer, restored checkpoint, metric
plugins, and TensorBoard visualizer plugins. It does not maintain a second
inference or alignment implementation.

## Evaluate the completed GSO checkpoint

```bash
RUN=lmgeo_new_val_geometry_render_n5k1_gso_checkpoint
CUDA_VISIBLE_DEVICES=0 \
PI3_CUDA_MEMORY_LIMIT_GIB=70 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
HYDRA_FULL_ERROR=1 \
/home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python -u scripts/train_pi3.py \
  train=train_megapose_gso_geometry_pi3_lmgeo_eval_rtxpro6000_70gb_336x252 \
  data=lmgeo_new_val_geometry_render_n5_k1_masked \
  general=lmgeo_geometry_eval \
  name="$RUN" \
  log.output_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN" \
  log.tensorboard_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"/tensorboard \
  log.ckpt_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"/ckpts \
  hydra.run.dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"
```

Override `PI3_EVAL_CHECKPOINT` to evaluate a different Accelerate checkpoint.
Add `test.iters_per_test=1` for a one-batch smoke. The validated 768-image
evaluation budget means 128 N=5/K=1 samples per batch.

### Query occupancy diagnostics

This isolated evaluator enables exact post-crop query-occupancy analysis in
`ObjectPoseMetric`. It bins the final visible mask fraction at 0.5%, 1%, 2%,
4%, and 8%, logs a compact TensorBoard view, and writes reusable artifacts to:

`<output_dir>/query_occupancy/<val_name>/step_<checkpoint>/`.

Artifacts include `query_rows.csv`, complete and high-visibility summary
CSV/JSON files, `correlations.json`, `object_bin_counts.csv`, and
`occupancy_metrics.png`. `query_rows.csv` contains the exact mask fraction and
pose/rotation/translation/camera-center/depth error for every query. No index
rebuild is required.

Re-bin an existing row file without repeating inference:

```bash
python scripts/analyze_query_occupancy.py \
  --rows <artifact-dir>/query_rows.csv \
  --out <new-analysis-dir> \
  --bin-edges 0,0.005,0.01,0.02,0.04,0.08,1.000001
```

## Automated checks

```bash
/home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python -m unittest \
  tests.test_lmgeo_geometry_matched \
  tests.test_lmgeo_adapter \
  tests.test_geometry_constrained_sampling -v

/home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python \
  -m unittest discover -s tests -v
```

See the [2026-08-18 result](experiment_results/2026-08-18_lmgeo_geometry_matched_gso_transfer.md)
for the built-index statistics, OOD audit, camera-center baseline, and the
reference-depth-scale comparison.
