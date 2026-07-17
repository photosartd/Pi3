#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --tmp=100G
#SBATCH --time=48:00:00
#SBATCH --job-name=pi3-lmgeo-560x420

set -euo pipefail

REPO_DIR="${PI3_REPO_DIR:-/homes/dtrofimov/repositories/Pi3}"
CONDA_ROOT="${PI3_CONDA_ROOT:-/homes/dtrofimov/miniconda3}"
CONDA_ENV="${PI3_CONDA_ENV:-pi3-lmgeo}"
DATA_ROOT="${PI3_DATA_ROOT:-/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o}"
RUNS_ROOT="${PI3_RUNS_ROOT:-/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3}"
RUN_MODE="${PI3_RUN_MODE:-train}"
SHARED_CKPT="${PI3_SHARED_CKPT:-/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors}"
FILTER_PREPROCESSED_QUERY_DEPTH="${PI3_FILTER_PREPROCESSED_QUERY_DEPTH:-false}"
TRAIN_CONFIG="${PI3_TRAIN_CONFIG:-train_lmgeo_finetune_a40_46gb}"
DATA_CONFIG="lmgeo_trainpbr45_real_and_new_val"

cd "$REPO_DIR"

if ! [[ "$TRAIN_CONFIG" =~ ^[A-Za-z0-9_]+$ ]] || [[ ! -f "configs/train/$TRAIN_CONFIG.yaml" ]]; then
  echo "ERROR: invalid or missing train config: $TRAIN_CONFIG" >&2
  echo "Expected configs/train/$TRAIN_CONFIG.yaml inside $REPO_DIR." >&2
  exit 2
fi

# The Conda root is intentionally configurable.
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ -n "${SLURM_JOB_TMP:-}" ]]; then
  export TMPDIR="$SLURM_JOB_TMP/tmp"
  export XDG_CACHE_HOME="$SLURM_JOB_TMP/cache"
  export HF_HOME="$SLURM_JOB_TMP/huggingface"
  export TORCH_HOME="$SLURM_JOB_TMP/torch"
  mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME"
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 2
fi

if [[ -z "${PI3_CKPT:-}" ]]; then
  if [[ -f "$SHARED_CKPT" ]]; then
    PI3_CKPT="$SHARED_CKPT"
  else
    PI3_CKPT="$REPO_DIR/ckpts/Pi3/model.safetensors"
  fi
fi

if [[ ! -f "$PI3_CKPT" ]]; then
  echo "ERROR: Pi3 checkpoint not found: $PI3_CKPT" >&2
  echo "Set PI3_CKPT=/path/to/model.safetensors, place it at $SHARED_CKPT, or place it at ckpts/Pi3/model.safetensors." >&2
  exit 2
fi

RUN_NAME="${PI3_RUN_NAME:-lmgeo_a40_46gb_${RUN_MODE}_${SLURM_JOB_ID}}"
RUN_DIR="${PI3_RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"
mkdir -p "$RUN_DIR/ckpts"

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Run mode: $RUN_MODE"
echo "Repo: $REPO_DIR"
echo "Data: $DATA_ROOT"
echo "Checkpoint: $PI3_CKPT"
echo "Output: $RUN_DIR"
echo "Conda env: $CONDA_ENV at $CONDA_ROOT"
echo "Train config: $TRAIN_CONFIG"
echo "Data config: $DATA_CONFIG"
echo "filter_preprocessed_query_depth=$FILTER_PREPROCESSED_QUERY_DEPTH"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_JOB_TMP=${SLURM_JOB_TMP:-unset}"
nvidia-smi

python - <<'PY'
import os
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("compiled archs:", " ".join(torch.cuda.get_arch_list()))
print("cuda visible:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is not visible inside the Slurm allocation"

count = torch.cuda.device_count()
print("device count:", count)
for idx in range(count):
    print(idx, torch.cuda.get_device_name(idx), torch.cuda.get_device_capability(idx))

assert any(arch == "sm_86" for arch in torch.cuda.get_arch_list()), "PyTorch wheel does not contain sm_86 for A40"
assert count >= int(os.environ.get("PI3_EXPECT_GPUS", "1")), "Fewer GPUs visible than expected"

x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert y.isfinite().all()
print("BF16 CUDA check: OK")
PY

if [[ "$RUN_MODE" == "preflight" ]]; then
  echo "Preflight passed."
  exit 0
fi

NUM_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "ERROR: no CUDA GPUs visible after preflight." >&2
  exit 2
fi

COMMON_OVERRIDES=(
  "train=$TRAIN_CONFIG"
  "data=$DATA_CONFIG"
  "name=$RUN_NAME"
  "lmgeo.data_root=$DATA_ROOT"
  "lmgeo.filter_preprocessed_query_depth=$FILTER_PREPROCESSED_QUERY_DEPTH"
  "model.ckpt=$PI3_CKPT"
  "log.output_dir=$RUN_DIR"
  "log.ckpt_dir=$RUN_DIR/ckpts"
  "hydra.run.dir=$RUN_DIR"
)

if [[ "$RUN_MODE" == "smoke" ]]; then
  echo "Running one-step worst-case 32-view 560x420 smoke."
  accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes "$NUM_GPUS" --num_machines 1 \
    scripts/train_pi3.py \
    "${COMMON_OVERRIDES[@]}" \
    'train.image_num_range=[32,32]' \
    train.max_img_per_gpu=32 \
    'lmgeo_profile.num_reference_range=[7,7]' \
    'lmgeo_profile.num_query_range=[25,25]' \
    train.num_epoch=1 \
    train.iters_per_epoch=1 \
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
    log.save_checkpoints=false \
    extras.print_config=false
elif [[ "$RUN_MODE" == "train" ]]; then
  echo "Running production fixed 560x420 training on $NUM_GPUS GPU(s)."
  accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes "$NUM_GPUS" --num_machines 1 \
    scripts/train_pi3.py \
    "${COMMON_OVERRIDES[@]}"
else
  echo "ERROR: unknown PI3_RUN_MODE=$RUN_MODE. Use preflight, smoke, or train." >&2
  exit 2
fi
