# Slurm A40 Setup

This note is for running the fixed 560x420 LMGeo profile on one Slurm node with
4 NVIDIA A40 GPUs. The Slurm filenames retain `518_a40_dynamic` only for
backward compatibility.

## Environment Choice

Use the default `pi3-lmgeo` environment from `environment_lmgeo.yaml`.

It installs PyTorch `2.7.1+cu128` from `requirements_torch_cuda128.txt`. This
single environment supports both relevant GPU generations:

- A40: `sm_86`
- RTX PRO 6000 Blackwell: `sm_120`

On the current workstation, the installed wheel reports:

```text
torch 2.7.1+cu128, CUDA runtime 12.8
compiled archs: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120
```

Keep `requirements_cuda124.txt` only as a legacy 4090 reproduction path. It is
not compatible with Blackwell.

## Install From A Login Node

Installing on a login node without GPUs is fine. The PyTorch CUDA wheels include
the CUDA runtime libraries; they do not require the system CUDA toolkit or
`nvcc`.

From the repo root:

```bash
conda env create -f environment_lmgeo.yaml
conda activate pi3-lmgeo
python -m pip check
python -m unittest discover -s tests -v
```

On a login node, this is expected to print `False` if no GPU is visible:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA visible here:", torch.cuda.is_available())
print("compiled archs:", " ".join(torch.cuda.get_arch_list()))
PY
```

Only run CUDA matmul checks on an allocated GPU node.

If `$HOME` is not shared between login and compute nodes, install Miniconda and
the environment under a shared project/scratch path, or build the environment on
a compatible node and pack it with `conda-pack`.

## Driver Check

Ask the cluster admins which NVIDIA driver branch is on the A40 nodes.

CUDA 12.8 GA shipped with an R570 driver line, and NVIDIA's CUDA 12.8 release
notes list `>=570.26` for the CUDA 12.8 toolkit driver. CUDA 12.x minor-version
compatibility can allow older drivers for older GPUs, but for one environment
that also supports Blackwell, R570 or newer is the clean target.

## Short A40 Preflight

Use the shortest GPU/debug partition available on the cluster. Adapt
`--partition` and `--account` to the local Slurm setup.

```bash
srun --partition=<gpu_or_debug_partition> \
  --account=<account_if_required> \
  --gres=gpu:a40:1 \
  --cpus-per-task=8 \
  --time=00:20:00 \
  --pty bash
```

Inside the allocation:

```bash
conda activate pi3-lmgeo

python - <<'PY'
import torch

assert torch.__version__.startswith("2.7.1")
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()

device = torch.cuda.current_device()
print("GPU:", torch.cuda.get_device_name(device))
print("Compute capability:", torch.cuda.get_device_capability(device))
print("BF16 supported:", torch.cuda.is_bf16_supported())
print("Compiled archs:", " ".join(torch.cuda.get_arch_list()))

x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert y.isfinite().all()
print("BF16 CUDA check: OK")
PY
```

For an A40, `torch.cuda.get_device_capability()` should be `(8, 6)`.

## Short Training Smoke

This is the queue-safe smoke I would run before a long job. It forces the
largest currently allowed 28-view 560x420 case, disables the expensive startup filter, and keeps
validation tiny. The helper script runs five training iterations per forced
shape by default; set `PI3_SMOKE_ITERS=<n>` to change that.

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_a40_46gb \
  data=lmgeo_trainpbr45_real_and_new_val \
  name=preflight_lmgeo_560x420_a40_28view \
  model.ckpt=/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors \
  train.random_reslution=false \
  'train.image_num_range=[28,28]' \
  train.max_img_per_gpu=28 \
  'lmgeo_profile.num_reference_range=[5,5]' \
  'lmgeo_profile.num_query_range=[23,23]' \
  lmgeo.filter_preprocessed_query_depth=false \
  train.num_epoch=1 \
  train.iters_per_epoch=5 \
  val_datasets.real_test.runtime.iters_per_test=1 \
  val_datasets.real_test.runtime.max_img_per_gpu=6 \
  val_datasets.real_test_ref16.runtime.iters_per_test=1 \
  val_datasets.real_test_ref16.runtime.max_img_per_gpu=17 \
  val_datasets.pbr_new_val.runtime.iters_per_test=1 \
  val_datasets.pbr_new_val.runtime.max_img_per_gpu=6 \
  val_datasets.pbr_new_val_ref16.runtime.iters_per_test=1 \
  val_datasets.pbr_new_val_ref16.runtime.max_img_per_gpu=17 \
  val_datasets.pbr_new_val_k5_subset.runtime.iters_per_test=1 \
  val_datasets.pbr_new_val_k5_subset.runtime.max_img_per_gpu=10 \
  val_datasets.pbr_new_val_k10_subset.runtime.iters_per_test=1 \
  val_datasets.pbr_new_val_k10_subset.runtime.max_img_per_gpu=15 \
  metrics.enabled=false \
  visuals.enabled=false \
  log.use_tensorboard=false \
  log.save_best=false \
  log.save_checkpoints=false
```

The helper script's `smoke` mode is preferred because it now tests both this
max-view edge and the packed 4 x 6-view edge for `PI3_SMOKE_ITERS=5` iterations
by default. The old 32-view smoke completed locally but OOMed during real A40
production, so do not use it as an acceptance criterion.

## Production 4xA40 Job

Let `accelerate launch` spawn one worker per GPU. The Slurm job itself should
normally be one task with four GPUs allocated, not four independent
`accelerate launch` tasks.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=lmgeo-560x420-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /path/to/Pi3

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate pi3-lmgeo

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

python - <<'PY'
import torch
assert torch.cuda.is_available()
print("visible GPUs:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(idx, torch.cuda.get_device_name(idx), torch.cuda.get_device_capability(idx))
PY

accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 4 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_a40_46gb \
  data=lmgeo_trainpbr45_real_and_new_val \
  name=lmgeo_a40_46gb_4xa40 \
  model.ckpt=/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors
```

Each GPU keeps the same per-rank memory budget. With 4 GPUs and no gradient
accumulation, the maximum effective view count per optimizer update is:

```text
up to 28 views/sample/rank * 4 ranks = 112 views/update
```

The A40 config uses `train.max_img_per_gpu: 28`. A 28-view sequence uses one
sample per rank; the current packed small-view edge is four sequences x six
views (5 references + 1 query each). The old 32-view budget OOMed on real A40
nodes and is no longer the production default.

## CITEc GPU Cluster Scripts

The CITEc cluster documentation lists A40 nodes in the `gpu` partition as
`worker-[1:11]`, with 4 A40 GPUs per node. For that cluster, use the scripts in
`scripts/slurm/`:

```bash
cd /homes/dtrofimov/repositories/Pi3

# 1. Check conda, CUDA visibility, A40 arch support, and BF16 matmul.
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh preflight

# 2. Run max-view and packed-min 560x420 training smokes on one A40.
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke

# 3. Submit the production 4xA40 training run.
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

Defaults baked into the scripts:

```text
repo:       /homes/dtrofimov/repositories/Pi3
conda:      /homes/dtrofimov/miniconda3
env:        pi3-lmgeo
data:       /vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o
runs/logs:  /vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3
ckpt:       /vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors
filter:     lmgeo.filter_preprocessed_query_depth=false
mail:       dmitrii.trofimov@uni-bielefeld.de, BEGIN,END,FAIL
```

The job selects `train=train_lmgeo_finetune_a40_46gb` and the canonical
`data=lmgeo_trainpbr45_real_and_new_val` by default. Set `PI3_TRAIN_CONFIG` to
select another A40 train profile, or `PI3_DATA_CONFIG` to select a data profile
such as `lmgeo_trainpbr45_real_and_new_val_context_refs`. The submit script
keeps its historical filename so existing commands continue to work.
The job exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default
to reduce allocator fragmentation on A40; override it explicitly only when
debugging allocator behavior.

The production command writes Hydra outputs and checkpoints under
`/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3/<run-name>/`, not under the
login-node repo checkout.

Download the Pi3 base checkpoint before submitting GPU jobs. See
[checkpoints.md](checkpoints.md). The CITEc script first looks for the shared
checkpoint path above, then falls back to `ckpts/Pi3/model.safetensors` inside
the repo.

The CITEc examples do not require a Slurm account, so the submit helper does not
set `--account` by default. If the scheduler rejects a job with an account/QoS
message, resubmit with the account supplied by the cluster admins:

```bash
PI3_SLURM_ACCOUNT=<account> scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh preflight
```

Optional overrides:

```bash
PI3_WALLTIME=72:00:00 scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_TRAIN_GPUS=2 scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_TRAIN_GPUS=2 PI3_RUN_NAME=lmgeo_a40_46gb_2xa40 scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_TRAIN_CONFIG=train_lmgeo_finetune_a40_46gb_corr PI3_RUN_NAME=lmgeo_a40_corr_lambda0p3 scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_DATA_CONFIG=lmgeo_trainpbr45_real_and_new_val_context_refs PI3_RUN_NAME=lmgeo_a40_context_refs scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_RUN_NAME=my_run scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_CKPT=/vol/coro/.../model.safetensors scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke
PI3_FILTER_PREPROCESSED_QUERY_DEPTH=true scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_CONTEXT_REFERENCE_FRACTION=0.0 scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_MAIL_TYPE=END,FAIL scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_MAIL_USER= scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke
```

All path overrides accepted by the job are:

| Variable | Meaning |
| --- | --- |
| `PI3_REPO_DIR` | Repository checkout on the compute node |
| `PI3_CONDA_ROOT` | Conda installation directory |
| `PI3_CONDA_ENV` | Conda environment name |
| `PI3_TRAIN_CONFIG` | Train YAML name without `.yaml`; defaults to the A40 baseline |
| `PI3_DATA_CONFIG` | Data YAML name without `.yaml`; defaults to canonical render-keyframe LMGeo |
| `PI3_DATA_ROOT` | LM-O dataset root |
| `PI3_CKPT` | Exact Pi3 base `model.safetensors`; highest checkpoint precedence |
| `PI3_SHARED_CKPT` | Alternative automatic shared-checkpoint location |
| `PI3_RUNS_ROOT` | Parent directory for default run directories and Slurm logs |
| `PI3_RUN_DIR` | Exact output directory for this run |
| `PI3_SLURM_LOG_DIR` | Directory for Slurm `.out` and `.err` files |

Example with explicit paths:

```bash
PI3_CKPT=/shared/checkpoints/pi3/model.safetensors \
PI3_DATA_ROOT=/shared/datasets/lm-o \
PI3_RUN_DIR=/shared/runs/pi3/my_a40_run \
PI3_RUN_NAME=my_a40_run \
  scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

## Baseline And Correspondence Runs

Run the selected profile's short smoke before submitting its production job.
Smoke mode always requests one A40, runs five training iterations per forced
shape by default, and exercises one batch from every validation loader,
including the `16 references + 1 query` real and PBR loaders:

```bash
cd /homes/dtrofimov/repositories/Pi3

# Baseline smoke.
PI3_TRAIN_CONFIG=train_lmgeo_finetune_a40_46gb \
PI3_RUN_NAME=lmgeo_a40_baseline_smoke \
  scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke

# Correspondence-loss smoke (lambda = 0.3).
PI3_TRAIN_CONFIG=train_lmgeo_finetune_a40_46gb_corr \
PI3_RUN_NAME=lmgeo_a40_corr_lambda0p3_smoke \
  scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke
```

Submit the baseline and correspondence-loss comparison as two independent
four-A40 jobs with distinct output names:

```bash
cd /homes/dtrofimov/repositories/Pi3

# Baseline: fixed 560x420, no correspondence loss.
PI3_TRAIN_GPUS=4 \
PI3_TRAIN_CONFIG=train_lmgeo_finetune_a40_46gb \
PI3_RUN_NAME=lmgeo_a40_baseline \
PI3_CKPT=/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors \
  scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train

# Correspondence: same A40/data profile, correspondence lambda = 0.3.
PI3_TRAIN_GPUS=4 \
PI3_TRAIN_CONFIG=train_lmgeo_finetune_a40_46gb_corr \
PI3_RUN_NAME=lmgeo_a40_corr_lambda0p3 \
PI3_CKPT=/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors \
  scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

Both commands can be submitted one after the other; `sbatch` returns after
queueing each job. `PI3_TRAIN_CONFIG` is the YAML filename from
`configs/train/` without `.yaml` and is validated on the compute node. By
default, the data profile remains the canonical render-keyframe
`lmgeo_trainpbr45_real_and_new_val` for both runs. To run the otherwise same
training with 50% masked same-object context-scene keyframes, add:

```bash
PI3_DATA_CONFIG=lmgeo_trainpbr45_real_and_new_val_context_refs
```

That context-reference data profile evaluates only `real_test`, `pbr_new_val`,
and `pbr_new_val_context_refs`. The last loader uses 5 masked held-out PBR
keyframes from subscenes different from the held-out PBR query subscene, plus
1 held-out PBR query.

The explicit `PI3_CKPT` above equals the current shared default and may be
omitted while that default remains valid.

The correspondence profile inherits the fixed 560x420 A40 profile and changes
only the layer-17 DINO output and correspondence-loss settings. Its train and
validation correspondence lambda is `0.3`. The old 32-view smoke completed
locally but real A40 production OOMed, so smoke both the baseline and
correspondence profiles again after any memory-limit change.

The two run roots and their main contents are:

```text
/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3/lmgeo_a40_baseline/
  ckpts/
  lmgeo_a40_baseline/events.out.tfevents.*

/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3/lmgeo_a40_corr_lambda0p3/
  ckpts/
  lmgeo_a40_corr_lambda0p3/events.out.tfevents.*
```

TensorBoard recursively discovers both runs from their common parent:

```bash
tensorboard \
  --logdir /vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3 \
  --port 6006
```

Slurm stdout/stderr defaults to
`/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3/slurm_logs/`; filenames
contain the Slurm job ID, so the two submissions do not overwrite each other.

For `train` mode, `PI3_TRAIN_GPUS` can be `1`, `2`, `3`, or `4`. The helper
scales the default CPU, RAM, and local tmp requests as `8 CPUs`, `100G RAM`, and
`25G tmp` per requested A40. Override those separately if needed:

```bash
PI3_TRAIN_GPUS=2 PI3_TRAIN_MEM=240G PI3_TRAIN_TMP=80G \
  scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

The per-GPU memory profile is unchanged. A 2xA40 run still allows up to 28 views
per sample per GPU and uses `train.max_img_per_gpu: 28` for packed smaller-view
batches. The effective number of views per optimizer step is about half of the
4xA40 run unless you increase gradient accumulation.

The fixed-resolution training config uses `train_dataset.length: auto`. At dataloader
construction time it resolves the virtual dataset length from the real indexed
LMGeo sample count, current Slurm/Accelerate world size, and the configured
sampler runway. This keeps the same config valid for 1xA40, 2xA40, and 4xA40
without hardcoding a huge local epoch length.

## Avoiding Queue Repeats

Recommended sequence:

1. Install/update the conda environment on the login node.
2. Run package tests on the login node.
3. Run the short A40 CUDA/BF16 preflight on one GPU.
4. Run the short max-view and packed-min training smoke on one A40.
5. Submit the 4xA40 production job.

The production data config keeps `lmgeo.filter_preprocessed_query_depth: true`,
but the CITEc Slurm script overrides it to `false` by default to avoid spending
allocated GPU time on a full `/vol` preprocessing scan. Set
`PI3_FILTER_PREPROCESSED_QUERY_DEPTH=true` for a slower but more conservative
run that filters invalid high-res samples before training.

## References

- PyTorch official installation selector and CUDA verification:
  https://pytorch.org/get-started/locally/
- PyTorch 2.7.1 CUDA 12.8 wheel command:
  https://pytorch.org/get-started/previous-versions/#v271
- NVIDIA CUDA 12.8 driver compatibility:
  https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html
