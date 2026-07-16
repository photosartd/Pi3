# Slurm A40 Setup

This note is for running the 518px LMGeo dynamic profile on one Slurm node with
4 NVIDIA A40 GPUs.

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

## One-Step Training Smoke

This is the queue-safe smoke I would run before a long job. It forces the
largest 32-view 518px case, disables the expensive startup filter, and keeps
validation tiny.

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_lmgeo_finetune_518_a40_dynamic \
  data=lmgeo_trainpbr45_real_and_new_val_518_a40_dynamic \
  name=preflight_lmgeo_518_a40_32view \
  train.random_reslution=false \
  'train.image_num_range=[32,32]' \
  train.max_img_per_gpu=32 \
  'lmgeo.num_reference_range=[7,7]' \
  'lmgeo.num_query_range=[25,25]' \
  lmgeo.filter_preprocessed_query_depth=false \
  train.num_epoch=1 \
  train.iters_per_epoch=1 \
  val_datasets.real_test.runtime.iters_per_test=1 \
  val_datasets.real_test.runtime.max_img_per_gpu=6 \
  val_datasets.pbr_new_val.runtime.iters_per_test=1 \
  val_datasets.pbr_new_val.runtime.max_img_per_gpu=6 \
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

On the local PRO 6000, this peaked at `40692 MB`. On an A40 it should fit, but
it is close enough that this smoke is worth running once.

## Production 4xA40 Job

Let `accelerate launch` spawn one worker per GPU. The Slurm job itself should
normally be one task with four GPUs allocated, not four independent
`accelerate launch` tasks.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=lmgeo-518-a40
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
  train=train_lmgeo_finetune_518_a40_dynamic \
  data=lmgeo_trainpbr45_real_and_new_val_518_a40_dynamic \
  name=lmgeo_518_a40_dynamic_4xa40
```

Each GPU keeps the same per-rank memory budget. With 4 GPUs and no gradient
accumulation, the maximum effective view count per optimizer update is:

```text
up to 32 views/sample/rank * 4 ranks = 128 views/update
```

The A40 dynamic config keeps `train.max_img_per_gpu: 28` while still allowing
32-view samples. Values above 28 can pack more low-view sequences into one GPU
batch; a local `max_img_per_gpu=32` run reached about `45.6 GiB` of PyTorch
allocated memory, before allocator reserve/driver overhead.

## CITEc GPU Cluster Scripts

The CITEc cluster documentation lists A40 nodes in the `gpu` partition as
`worker-[1:11]`, with 4 A40 GPUs per node. For that cluster, use the scripts in
`scripts/slurm/`:

```bash
cd /homes/dtrofimov/repositories/Pi3

# 1. Check conda, CUDA visibility, A40 arch support, and BF16 matmul.
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh preflight

# 2. Run one worst-case 32-view 518px training step on one A40.
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
PI3_TRAIN_GPUS=2 PI3_RUN_NAME=lmgeo_518_a40_dynamic_2xa40 scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_RUN_NAME=my_run scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_CKPT=/vol/coro/.../model.safetensors scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke
PI3_FILTER_PREPROCESSED_QUERY_DEPTH=true scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_MAIL_TYPE=END,FAIL scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
PI3_MAIL_USER= scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke
```

For `train` mode, `PI3_TRAIN_GPUS` can be `1`, `2`, `3`, or `4`. The helper
scales the default CPU, RAM, and local tmp requests as `8 CPUs`, `100G RAM`, and
`25G tmp` per requested A40. Override those separately if needed:

```bash
PI3_TRAIN_GPUS=2 PI3_TRAIN_MEM=240G PI3_TRAIN_TMP=80G \
  scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

The per-GPU memory profile is unchanged. A 2xA40 run still allows up to 32 views
per sample per GPU, but uses `train.max_img_per_gpu: 28` for packed smaller-view
batches. The effective number of views per optimizer step is about half of the
4xA40 run unless you increase gradient accumulation.

The dynamic training config uses `train_dataset.length: auto`. At dataloader
construction time it resolves the virtual dataset length from the real indexed
LMGeo sample count, current Slurm/Accelerate world size, and the configured
sampler runway. This keeps the same config valid for 1xA40, 2xA40, and 4xA40
without hardcoding a huge local epoch length.

## Avoiding Queue Repeats

Recommended sequence:

1. Install/update the conda environment on the login node.
2. Run package tests on the login node.
3. Run the short A40 CUDA/BF16 preflight on one GPU.
4. Run the one-step 32-view training smoke on one A40.
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
