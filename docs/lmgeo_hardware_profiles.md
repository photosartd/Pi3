# LMGeo Data And GPU Profiles

LMGeo's main train/validation definition has one canonical data config:

```text
data=lmgeo_trainpbr45_real_and_new_val
```

It owns the `train_pbr`, held-out `new_val`, and real BOP target datasets and
their K=1/K=5/K=10 query-context protocols plus matched 16-reference/1-query
real and PBR ablations. Resolution, training view sampling, and every
`max_img_per_gpu` capacity value come from the selected train profile.

## Named GPU Profiles

| Profile | Resolution | Train views | Train `max_img_per_gpu` | Validation budgets |
| --- | --- | --- | ---: | --- |
| `train_lmgeo_finetune_rtx4090_24gb` | 224x224 | 6-26 | 50 | K1/K5/K10/Ref16: 96/96/96/96 |
| `train_lmgeo_finetune_a40_46gb` | fixed 560x420 | 6-32 | 32 | K1/K5/K10/Ref16: 128/128/128/128 |
| `train_lmgeo_finetune_rtxpro6000_blackwell_96gb` | fixed 560x420 | 6-32 | 56 | K1/K5/K10/Ref16: 384/384/320/320 |

`max_img_per_gpu` keeps its historical name and behavior. It is a sequence
packing budget: the dynamic sampler uses
`max(1, floor(max_img_per_gpu / views_per_sequence))`, so a single sequence may
contain more views than the configured budget.

## Local One-GPU Runs

Run from the repository root. The current local defaults are:

| Setting | Default |
| --- | --- |
| Repository | `/home/dtrofimov/repositories/Pi3` |
| LM-O data | `/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o` |
| Pi3 base checkpoint | `ckpts/Pi3/model.safetensors`, relative to the repository |
| Outputs and Hydra run directory | `outputs/<run-name>` |
| Saved checkpoints | `outputs/<run-name>/ckpts` |

Select the named profile matching the GPU. These are complete commands using
the defaults above:

```bash
cd /home/dtrofimov/repositories/Pi3
conda activate pi3-lmgeo

# GeForce RTX 4090, 24 GB
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_rtx4090_24gb \
  data=lmgeo_trainpbr45_real_and_new_val \
  name=lmgeo_rtx4090_24gb

# NVIDIA A40, 46 GB
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_a40_46gb \
  data=lmgeo_trainpbr45_real_and_new_val \
  name=lmgeo_a40_46gb

# RTX PRO 6000 Blackwell, 96 GB
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_rtxpro6000_blackwell_96gb \
  data=lmgeo_trainpbr45_real_and_new_val \
  name=lmgeo_rtxpro6000_blackwell_96gb
```

Hydra overrides replace any local path without editing YAML. Override all three
output paths together so logs, checkpoints, and Hydra files remain in one run
directory:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_rtx4090_24gb \
  data=lmgeo_trainpbr45_real_and_new_val \
  name=my_local_run \
  model.ckpt=/absolute/path/to/model.safetensors \
  lmgeo.data_root=/absolute/path/to/lm-o \
  log.output_dir=/absolute/path/to/runs/my_local_run \
  log.ckpt_dir=/absolute/path/to/runs/my_local_run/ckpts \
  hydra.run.dir=/absolute/path/to/runs/my_local_run
```

`model.ckpt` is the original Pi3 initialization checkpoint. It is distinct from
the checkpoints written to `log.ckpt_dir` during fine-tuning.

Changing `--num_processes` changes the data-parallel world size, not the
per-GPU capacity values. Each rank receives different dataset indices while
using the same sampled total-view count and local sequence batch size.

## CITEc Slurm A40 Runs

The compatibility-named submit helper now launches the named A40 profile and
canonical data config internally:

```text
train=train_lmgeo_finetune_a40_46gb
data=lmgeo_trainpbr45_real_and_new_val
```

Override only the train profile with `PI3_TRAIN_CONFIG`. For example, the
high-resolution correspondence-loss variant inherits the same A40 hardware and
sampling limits and sets correspondence lambda to 0.3:

```bash
PI3_TRAIN_CONFIG=train_lmgeo_finetune_a40_46gb_corr \
PI3_RUN_NAME=lmgeo_a40_corr_lambda0p3 \
  scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

Its constrained 32-view training smoke reached 36,417 MiB allocated / 36,784
MiB reserved under the 38 GiB cap. This was a one-step loss-path check, not a
full production-validation sweep.

Use it from the CITEc checkout:

```bash
cd /homes/dtrofimov/repositories/Pi3

scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh preflight
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

`train` requests four A40s by default. Set `PI3_TRAIN_GPUS` to `1`, `2`, `3`,
or `4`; Accelerate starts one process per allocated GPU and retains the same
per-GPU A40 memory profile.

The current CITEc path defaults and their overrides are:

| Setting | Default | Override |
| --- | --- | --- |
| Repository | `/homes/dtrofimov/repositories/Pi3` | `PI3_REPO_DIR` |
| Conda installation | `/homes/dtrofimov/miniconda3` | `PI3_CONDA_ROOT` |
| Conda environment | `pi3-lmgeo` | `PI3_CONDA_ENV` |
| LM-O data | `/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o` | `PI3_DATA_ROOT` |
| Pi3 base checkpoint | `/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors` | `PI3_CKPT` |
| Runs | `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3` | `PI3_RUNS_ROOT` |
| One run directory | `<runs>/<run-name>` | `PI3_RUN_DIR` |
| Slurm logs | `<runs>/slurm_logs` | `PI3_SLURM_LOG_DIR` |

For example, a two-A40 job with explicit checkpoint, data, and output paths is:

```bash
PI3_TRAIN_GPUS=2 \
PI3_CKPT=/shared/checkpoints/pi3/model.safetensors \
PI3_DATA_ROOT=/shared/datasets/lm-o \
PI3_RUN_DIR=/shared/runs/pi3/my_a40_run \
PI3_RUN_NAME=my_a40_run \
  scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

`PI3_CKPT` has highest precedence. Without it, the job checks the shared
default checkpoint and then falls back to
`<PI3_REPO_DIR>/ckpts/Pi3/model.safetensors`. See [slurm_a40.md](slurm_a40.md)
for scheduler, mail, RAM, wall-time, and smoke-test overrides.

## Backward-Compatible Names

The historical names remain valid:

```text
train_lmgeo_finetune_lowres
train_lmgeo_finetune_518_a40_dynamic
lmgeo_trainpbr45_real_and_new_val_518_a40_dynamic
lmgeo_all_trainpbr_test_bop_518_a40
```

The hardware-specific data names are aliases of their canonical data configs;
they no longer duplicate dataset or validation definitions.

## Constrained Memory Smokes

Set `PI3_CUDA_MEMORY_LIMIT_GIB` to cap PyTorch's CUDA allocator when exercising
a smaller-GPU profile on a larger device:

```bash
PI3_CUDA_MEMORY_LIMIT_GIB=23.5 python scripts/train_pi3.py ...
PI3_CUDA_MEMORY_LIMIT_GIB=38 python scripts/train_pi3.py ...
```

This is opt-in and does not change ordinary training. A profile is accepted
only after its worst-case training shape and all production validation loaders
finish below the corresponding constrained limit.

## Constrained Smoke Results

Measured on 2026-07-17 on the local RTX PRO 6000 Blackwell with PyTorch
2.7.1+cu128, BF16, the frozen encoder, and the Pi3 base checkpoint. Each run
used one worst-case packed training step and one batch from every production
validation loader. The smoke restricted train/new-val indexing to one scene to
avoid shared-filesystem startup cost; tensor shapes, packing budgets, metrics,
and visual settings were unchanged.

| Profile | Allocator cap | Training shape | Train allocated/reserved | Worst validation reserved | Peak process memory |
| --- | ---: | --- | ---: | ---: | ---: |
| RTX 4090 24 GB | 23.5 GiB | 2 x 25 views at 224px | 15,273 / 15,792 MiB | 16,528 MiB | 17,264 MiB |
| A40 46 GB | 38 GiB | 1 x 32 views at 560x420 | 36,262 / 36,634 MiB | 38,246 MiB | 37,370 MiB |

The current 4090 and A40 rows exited successfully. The A40 run used fixed
560x420 and all current production metrics. It was intentionally capped below
the roughly 40 GB that was free on the shared workstation.

The Blackwell profile also inherits fixed 560x420, but was not re-maximized in
this pass. Its unchanged budgets were previously accepted at the larger
518x518 square: 67,736/68,882 MiB train allocated/reserved and 71,242 MiB worst
validation reserve under a 70 GiB allocator cap. Since 560x420 has 1,200 patch
tokens per image versus 1,369 at 518x518, the unchanged budgets are
conservative; a fresh full-memory probe is still required before increasing
them.

The packing edges were checked separately because small sequences pack more
samples per rank. The RTX 4090 profile's `8 x 6`-view batch reached 14,822 MiB
allocated / 15,360 MiB reserved. The named high-resolution profiles use a
6-view minimum. A40 packs `5 x (5 references + 1 query)` at that edge and
reached 34,281/34,716 MiB allocated/reserved. Its `31 references + 1 query`
32-view edge reached 36,262/36,634 MiB. Blackwell packs nine 6-view sequences
with its larger budget, but that shape still requires a fresh full-memory
probe. The old `9 x 3` and `10 x 3` measurements remain only as historical
results in the detailed A40 document.

Detailed A40 training and inference scaling is recorded in
[lmgeo_518_a40.md](lmgeo_518_a40.md#40-gb-constrained-probe); that filename is
historical.
