# Repository instructions

This file applies to the entire repository. Treat it as a research fork, not as
a pristine copy of upstream `yyfz/Pi3`; the current `origin` is the personal
fork `photosartd/Pi3`. Preserve the original general-purpose Pi3 behavior where
practical, but prioritize the fork's object-centric 6-DoF pose estimation work.
The repository is licensed for non-commercial use; retain the existing notices
when redistributing code or derived binaries.

## Project purpose

Pi3 (π³, "Pi to the power of three") is a multi-view 3D geometric backbone. It
takes a set of images, reasons jointly within and across views, and predicts:

- a dense local 3D point map for every input view;
- a camera-to-common/world-frame pose for every view;
- dense point maps transformed into that shared frame;
- optional confidence, global-point, and detached intermediate DINO features.

The training model is built around a DINOv2 ViT-L/14 encoder, a transformer
decoder that alternates per-view and cross-view processing, and point/camera
heads. Its geometry is scale-ambiguous. Do not assume that the predicted common
frame is already metric or aligned to a dataset's world/object frame.

This fork adapts Pi3 training to object-centric 6-DoF pose estimation. The main
use case is: provide clean reference/keyframe renders with known object poses
and one or more query images, align Pi3's predicted common frame to the CAD
object frame using the references, and recover the object-to-camera pose of each
query. The current dataset is LMGeo, built from the BOP-format LINEMOD-Occluded
(LM-O) data. References normally come from `train/<object_id>`; training queries
come from `train_pbr`, held-out synthetic queries from `new_val`, and real test
queries from official BOP19 targets.

## Coordinate and data invariants

Geometry notation is critical. Follow these conventions consistently:

- BOP `T_C_O` maps points from the object/CAD frame `O` into camera frame `C`
  (object-to-camera pose).
- `LMGeoDataset` stores that matrix as `T_C_O` and supplies
  `camera_pose = inv(T_C_O) = T_O_C`. It deliberately treats the object frame as
  the dataset world frame because Pi3 expects camera-to-world poses.
- BOP translations and model vertices are converted from millimeters to meters.
  Raw depth uses `depth_scale` and then `depth_unit_scale` (normally `0.001`) so
  depth, translation, CAD models, metrics, and overlays use the same units.
- Pi3 outputs `camera_poses` interpreted as predicted `T_W_C`, where `W` is its
  arbitrary shared frame. Pose metrics estimate a reference-only Sim(3) from
  `W` to `O`, apply it to query cameras, and invert to obtain query `T_C_O`.
- Never align on query ground truth and then report query pose accuracy; that is
  evaluation leakage. References determine alignment, queries determine the
  reported 6-DoF error.
- Samples put reference views before query views, and the loss uses the first
  view as its relative coordinate origin. Still use `is_reference`, `is_query`,
  and `view_role` as the authoritative role metadata instead of inferring roles
  from fixed indices.
- Keep `T_C_O`, `camera_pose`, intrinsics, metric depth, object ID, view-role
  flags, source IDs, and masks intact through collation. `BaseDataset` derives
  `pts3d` and `valid_mask` from these fields.
- Pi3 input height and width must be compatible with patch size 14. When adding
  crops or resize policies, validate adjusted intrinsics and make sure the
  object/depth is not removed by the center crop.
- LM-O symmetric object IDs are currently configured as 10 and 11. The
  BOP-style summary uses ADD-S for configured symmetric objects and ADD for the
  others; do not silently change this policy.

## What is fork-specific

The most important local work, beyond upstream training code, is:

- `datasets/lmgeo_dataset.py`: `LMGeoDataset` for a fixed object/window and
  `LMGeoSequenceDataset` for indexed multi-object windows or BOP target files.
  It handles reference/query selection, masking, visibility/crop/depth filters,
  dynamic reference/query counts, and BOP metadata.
- `configs/data/lmgeo_*.yaml`: fixed overfit, sequence, all-TrainPBR/real-BOP,
  held-out-PBR validation, and 518px A40 data profiles.
- `configs/train/train_lmgeo_*.yaml`: low-resolution fine-tuning/overfit,
  correspondence-loss, fixed 518px, and dynamic 518px profiles.
- `pi3/metrics/`: plugin-style object-pose ADD/ADD-S, aligned camera residual,
  and reference reconstruction Chamfer metrics. Alignment and BOP model helpers
  live in `pi3/metrics/utils.py`.
- `pi3/visualizations/`: TensorBoard-ready input grids, depth panels, reference
  reconstruction, query pose overlays, and correspondence visualizations,
  managed by configurable plugins.
- `pi3/models/correspondence.py` and the auxiliary loss in
  `pi3/models/loss.py`: GT-depth/pose query-to-reference patch matches with an
  optional detached DINO-similarity weight. The separate correspondence config
  keeps baseline runs comparable.
- `trainers/base_trainer_accelerate.py`: named validation loaders, metrics,
  visualization logging, checkpointing, and TensorBoard/W&B integration.
- `datasets/base/batched_sampler.py`: dynamic view-count/resolution batching.
  The per-rank sample batch size is
  `floor(max_img_per_gpu / sampled_total_views)`.
- `docs/`, especially `docs/experiment_results/`: setup, checkpoint, GPU/Slurm,
  memory-probe, and experiment records.
- `scripts/slurm/`: CITEc A40 preflight, one-step smoke, and production launch.

Keep inference/demo code (`pi3/models/pi3.py`, `example.py`, `demo_gradio.py`)
distinct from the training implementation (`pi3/models/pi3_training.py`). Most
6-DoF work belongs in the latter, the LMGeo pipeline, metrics, or configs unless
the task explicitly changes public inference behavior.

## Current research state and direction

The completed 224px LM-O run recorded in
`docs/experiment_results/2026-07-10_lmgeo_all_trainpbr_test_bop_15k_warmup.md`
shows that the adaptation learns: median real-query normalized pose error fell
from about `3.56d` to `0.57d`, rotation from about `95.6°` to `27.3°`, and
translation from about `0.58 m` to `0.078 m`. Strict final ADD(-S) recall at
`0.1d` was still only about `10.5%`, with large per-object differences. Treat
real-image generalization—not basic trainability—as the present bottleneck.
The 2026-07-13 held-out-PBR run note is still marked running/pending, and the
518px A40 work currently documents launch preparation and memory probes rather
than a completed result. Check actual run artifacts before changing either
status.

Evidence-backed active questions and follow-up directions are:

- measure the synthetic-to-real gap by comparing real BOP targets against the
  held-out `new_val` PBR split;
- test whether more query context/parallax (`K=1`, `K=5`, `K=10`) improves pose;
- compare the baseline fairly with GT geometric correspondence consistency and
  detached DINO-weighted correspondence variants;
- evaluate whether native 518px inputs and dynamic 3-32-view training improve
  small/occluded object geometry without exceeding the A40 memory budget;
- investigate the especially weak objects and handle symmetry/visibility and
  object-specific failure modes explicitly.

These are research directions, not settled design requirements. Keep new ideas
as isolated configs or small composable components, record the hypothesis and
comparison, and avoid overwriting a reproducible baseline.

## Configuration and training workflow

Hydra composes `configs/default.yaml` from model, train, data, general, and
extras groups. Prefer a new or inherited YAML profile plus command-line
overrides over embedding paths or experiment choices in Python. Pay attention
to composition order: a data config may intentionally override train/test view
counts.

The default research environment is Python 3.11 with PyTorch 2.7.1 CUDA 12.8:

```bash
conda env create -f environment_lmgeo.yaml
conda activate pi3-lmgeo
python -m pip check
python -m unittest discover -s tests -v
```

Compose a run without allocating a GPU or starting training:

```bash
python scripts/train_pi3.py --cfg job \
  train=train_lmgeo_finetune_lowres \
  data=lmgeo_all_trainpbr_test_bop
```

Representative single-GPU low-resolution fine-tuning:

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_lowres \
  data=lmgeo_all_trainpbr_test_bop \
  name=<run-name>
```

The base checkpoint is normally `ckpts/Pi3/model.safetensors`; it is not stored
in Git. Data and run artifacts also live outside the repository. See
`docs/checkpoints.md` and `docs/conda_setup.md`. A warning about the missing
CUDA-compiled RoPE2D extension is expected here; the code uses the slower
PyTorch fallback.

For the current high-resolution cluster workflow, read `docs/lmgeo_518_a40.md`
and `docs/slurm_a40.md`, then use:

```bash
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh preflight
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

The Slurm files contain current CITEc defaults for repository, data, checkpoint,
run, Conda, and email locations. Preserve those defaults unless the user asks
to change them; prefer their documented `PI3_*` environment overrides for
another machine or run. Never commit checkpoints, datasets, generated images,
TensorBoard events, logs, or secrets.

## Change guidelines

- Start by checking `git status` and the relevant experiment/config history.
  User changes may coexist with upstream-derived code; do not discard or broadly
  rewrite unrelated work.
- Favor targeted changes. Follow local style in older upstream files, while
  using type hints and small plugin interfaces in newer metrics/visualization
  code. There is no repository-wide formatter configured, so avoid formatting
  churn.
- Preserve baseline comparability. Add a new train/data config for a meaningful
  loss, resolution, sampling, masking, or validation change unless the task is
  explicitly a baseline correction.
- Keep reference/query selection reproducible via sampler/dataset RNGs and
  `set_epoch`. In distributed training, view counts must remain synchronized
  across ranks while resolution may vary per rank.
- Respect view-budget invariants. If `image_num_range`, reference/query ranges,
  `max_img_per_gpu`, iteration count, or world size changes, re-check dataset
  length assertions, possible reference/query splits, duplicate policy, and
  worst-case memory.
- Metrics and visualization code should be optional and Hydra-configurable.
  Avoid retaining autograd graphs; use detached/no-grad values. High-resolution
  A40 configs intentionally disable visuals to retain memory headroom.
- Be explicit about meters versus millimeters, `T_C_O` versus `T_O_C`, and
  predicted `T_W_C` in variable names, docstrings, tests, and review notes.
- Do not use a decreasing training loss alone as evidence of better 6-DoF pose.
  Compare aligned query ADD/ADD-S, rotation/translation, held-out splits, and
  per-object metrics. Watch for query leakage during Sim(3) alignment.
- Put reusable geometric matching in `pi3/models/correspondence.py`, losses in
  `pi3/models/loss.py`, metrics in `pi3/metrics/`, and rendering in
  `pi3/visualizations/`; do not duplicate their coordinate math in the trainer.
- Update `docs/experiment_results/README.md` and add or finish a dated run note
  when results or conclusions change. Include exact configs/overrides, data
  partitions, checkpoints/output paths, primary metrics, and concise findings.

## Validation expectations

Run validation proportional to the change and report what could not be run:

1. For pure utility, metric, loss, or visualization changes, run the relevant
   test module and then the full CPU suite:

   ```bash
   python -m unittest discover -s tests -v
   ```

2. For Hydra changes, compose every affected train/data pairing with
   `--cfg job`. Check the resolved total-view range, reference/query ranges,
   resolution, dataset paths, validation loaders, loss weights, metrics, and
   visuals.
3. For LMGeo changes, inspect at least one sample when data is available. Verify
   role order/flags, matrix inverses, depth units, nonempty valid masks,
   intrinsics after preprocessing, and source/object metadata.
4. For correspondence work, run `tests/test_correspondence_loss.py` and
   `tests/test_correspondence_visualizer.py`. Use
   `scripts/debug_lmgeo_correspondences.py` on real data for geometric or visual
   changes.
5. For training, sampler, resolution, precision, or memory changes, run a
   one-step GPU smoke before a long job. Use the documented worst-case 32-view
   518px smoke for the A40 profile and do not extrapolate memory from 224px.
6. Do not claim end-to-end validation when the external LM-O data, base
   checkpoint, CUDA GPU, or Slurm cluster was unavailable. CPU tests and Hydra
   composition are useful but are not substitutes for a data/GPU smoke.
