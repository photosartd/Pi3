# MegaPose-GSO 560×420 A40 training

The production pair is:

```text
train=train_megapose_gso_finetune_a40_46gb
data=megapose_gso_anchor_scene_pairs
```

It trains on strict same-object anchor pairs: 2–16 unique reference views from
one physical `(scene_id, gt_id)` track and 1–10 unique query views from another
scene. Total model views vary from 3–26. Inputs are fixed at 560×420 and the
per-rank packing ceiling remains 28 images, leaving the same conservative A40
margin as the established object-centric profile. Both roles receive their
instance-specific visibility condition; this is a mask-conditioned regime, not
an unconditioned detection benchmark.

Validation uses the prepared held-out object and held-out scene lists, but
samples each of its five references and one query from independent scenes. That
keeps the validation protocol available when individual held-out tracks do not
contain six clean views. Neither training nor validation repeats a frame.

Before submission, the dataset directory must contain:

```text
pi3_index/megapose_gso.sqlite
pi3_index/megapose_gso.splits.json
```

Compose the exact job without allocating a GPU:

```bash
python scripts/train_pi3.py --cfg job \
  train=train_megapose_gso_finetune_a40_46gb \
  data=megapose_gso_anchor_scene_pairs \
  gso.data_root=/path/to/MegaPose-GSO-fixed
```

## CITEc Slurm environment

The existing A40 submission backend is dataset-aware. Set the environment once:

```bash
export PI3_REPO_DIR=/homes/dtrofimov/repositories/Pi3
export PI3_CONDA_ROOT=/homes/dtrofimov/miniconda3
export PI3_CONDA_ENV=pi3-lmgeo
export PI3_DATA_FAMILY=megapose_gso
export PI3_DATA_ROOT=/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/MegaPose-GSO-fixed
export PI3_TRAIN_CONFIG=train_megapose_gso_finetune_a40_46gb
export PI3_DATA_CONFIG=megapose_gso_anchor_scene_pairs
export PI3_CKPT=/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors
export PI3_RUNS_ROOT=/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3
export PI3_TENSORBOARD_ROOT=/homes/dtrofimov/logs/spott3r
export PI3_SLURM_JOB_PREFIX=pi3-gso-a40
export PI3_TRAIN_GPUS=4
export PI3_TRAIN_CPUS=32
export PI3_TRAIN_MEM=400G
export PI3_TRAIN_TMP=100G
export PI3_WALLTIME=48:00:00
export PI3_CKPT_INTERVAL=1
export PI3_MAX_CHECKPOINTS=5
```

Then validate the node/environment and the two A40 memory extremes:

```bash
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh preflight

export PI3_RUN_NAME=megapose_gso_a40_anchor_smoke_$(date +%Y%m%d_%H%M%S)
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke --fresh
```

The smoke job is one A40. It runs both the worst logical sample
`16 references + 10 queries` and the packed minimum `9 × (2 references + 1
query)`. The full prepared dataset must contain qualifying clean anchor tracks;
failure during candidate construction is a data/visibility-capacity failure,
not a reason to enable repetition silently.

After the smoke passes, submit production on four A40s with a fresh run name:

```bash
export PI3_RUN_NAME=megapose_gso_a40_anchor_$(date +%Y%m%d_%H%M%S)
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train --fresh
```

To continue that exact run later, restore the same `PI3_RUN_NAME` and use:

```bash
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train --continue
```

The launcher routes `PI3_DATA_ROOT` to `gso.data_root`; it no longer injects
LMGeo-only paths or smoke validation loaders for MegaPose jobs. The base
checkpoint, run directory, TensorBoard directory, checkpoint cadence, and
multi-GPU launch behavior otherwise remain identical to previous A40 jobs.

