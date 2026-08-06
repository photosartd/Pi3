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
RUNS_ROOT="${PI3_RUNS_ROOT:-/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3}"
TENSORBOARD_ROOT="${PI3_TENSORBOARD_ROOT:-/homes/dtrofimov/logs/spott3r}"
RUN_MODE="${PI3_RUN_MODE:-train}"
SHARED_CKPT="${PI3_SHARED_CKPT:-/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors}"
FILTER_PREPROCESSED_QUERY_DEPTH="${PI3_FILTER_PREPROCESSED_QUERY_DEPTH:-false}"
TRAIN_CONFIG="${PI3_TRAIN_CONFIG:-train_lmgeo_finetune_a40_46gb}"
DATA_CONFIG="${PI3_DATA_CONFIG:-lmgeo_trainpbr45_real_and_new_val}"
DATA_FAMILY="${PI3_DATA_FAMILY:-}"
if [[ -z "$DATA_FAMILY" ]]; then
  if [[ "$DATA_CONFIG" == megapose_gso_* ]]; then
    DATA_FAMILY="megapose_gso"
  else
    DATA_FAMILY="lmgeo"
  fi
fi
case "$DATA_FAMILY" in
  lmgeo)
    DEFAULT_DATA_ROOT="/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o"
    DEFAULT_RUN_PREFIX="lmgeo_a40_46gb"
    ;;
  megapose_gso)
    DEFAULT_DATA_ROOT="/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/MegaPose-GSO-fixed"
    DEFAULT_RUN_PREFIX="megapose_gso_a40_46gb"
    ;;
  *)
    echo "ERROR: PI3_DATA_FAMILY must be lmgeo or megapose_gso, got: $DATA_FAMILY" >&2
    exit 2
    ;;
esac
DATA_ROOT="${PI3_DATA_ROOT:-$DEFAULT_DATA_ROOT}"
SMOKE_ITERS="${PI3_SMOKE_ITERS:-5}"
CONTEXT_REFERENCE_FRACTION="${PI3_CONTEXT_REFERENCE_FRACTION:-}"
CONTINUE_MODE="${PI3_CONTINUE:-}"
RESUME_PATH="${PI3_RESUME:-}"
CKPT_INTERVAL="${PI3_CKPT_INTERVAL:-1}"
MAX_CHECKPOINTS="${PI3_MAX_CHECKPOINTS:-5}"

cd "$REPO_DIR"

if ! [[ "$TRAIN_CONFIG" =~ ^[A-Za-z0-9_]+$ ]] || [[ ! -f "configs/train/$TRAIN_CONFIG.yaml" ]]; then
  echo "ERROR: invalid or missing train config: $TRAIN_CONFIG" >&2
  echo "Expected configs/train/$TRAIN_CONFIG.yaml inside $REPO_DIR." >&2
  exit 2
fi
if ! [[ "$DATA_CONFIG" =~ ^[A-Za-z0-9_]+$ ]] || [[ ! -f "configs/data/$DATA_CONFIG.yaml" ]]; then
  echo "ERROR: invalid or missing data config: $DATA_CONFIG" >&2
  echo "Expected configs/data/$DATA_CONFIG.yaml inside $REPO_DIR." >&2
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 2
fi
if [[ "$DATA_FAMILY" == "megapose_gso" ]]; then
  if [[ ! -f "$DATA_ROOT/pi3_index/megapose_gso.sqlite" ]]; then
    echo "ERROR: MegaPose-GSO index is missing: $DATA_ROOT/pi3_index/megapose_gso.sqlite" >&2
    echo "Run datasets/preprocess/megapose_gso.py before submitting training." >&2
    exit 2
  fi
  if [[ ! -f "$DATA_ROOT/pi3_index/megapose_gso.splits.json" ]]; then
    echo "ERROR: MegaPose-GSO split manifest is missing: $DATA_ROOT/pi3_index/megapose_gso.splits.json" >&2
    echo "Rerun datasets/preprocess/megapose_gso.py before submitting training." >&2
    exit 2
  fi
fi

if [[ -z "${PI3_CKPT:-}" ]]; then
  if [[ -f "$SHARED_CKPT" ]]; then
    PI3_CKPT="$SHARED_CKPT"
  else
    PI3_CKPT="$REPO_DIR/ckpts/Pi3/model.safetensors"
  fi
fi

case "${PI3_CKPT,,}" in
  none|null)
    PI3_CKPT="null"
    ;;
esac

if [[ "$PI3_CKPT" != "null" && ! -f "$PI3_CKPT" ]]; then
  echo "ERROR: Pi3 checkpoint not found: $PI3_CKPT" >&2
  echo "Set PI3_CKPT=/path/to/model.safetensors, PI3_CKPT=none for scratch runs, place it at $SHARED_CKPT, or place it at ckpts/Pi3/model.safetensors." >&2
  exit 2
fi

RUN_NAME="${PI3_RUN_NAME:-${DEFAULT_RUN_PREFIX}_${RUN_MODE}_${SLURM_JOB_ID}}"
RUN_DIR="${PI3_RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"
if [[ -n "${PI3_TENSORBOARD_DIR+x}" ]]; then
  TENSORBOARD_DIR="$PI3_TENSORBOARD_DIR"
else
  TENSORBOARD_DIR="$TENSORBOARD_ROOT/$RUN_NAME"
fi
if [[ -z "$TENSORBOARD_DIR" ]]; then
  TENSORBOARD_DIR="$RUN_DIR"
fi
mkdir -p "$RUN_DIR/ckpts" "$TENSORBOARD_DIR"
if [[ -z "$RESUME_PATH" ]]; then
  case "${CONTINUE_MODE,,}" in
    1|true|yes|y|on)
      if ! find "$RUN_DIR/ckpts" -maxdepth 1 -type d -name 'checkpoint_*' -print -quit 2>/dev/null | grep -q .; then
        echo "ERROR: continue requested, but no checkpoint_* directory was found under: $RUN_DIR/ckpts" >&2
        exit 2
      fi
      ;;
  esac
fi

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Run mode: $RUN_MODE"
echo "Repo: $REPO_DIR"
echo "Data: $DATA_ROOT"
echo "Checkpoint: $PI3_CKPT"
echo "Output: $RUN_DIR"
echo "TensorBoard: $TENSORBOARD_DIR"
echo "Conda env: $CONDA_ENV at $CONDA_ROOT"
echo "Train config: $TRAIN_CONFIG"
echo "Data config: $DATA_CONFIG"
echo "Data family: $DATA_FAMILY"
echo "filter_preprocessed_query_depth=$FILTER_PREPROCESSED_QUERY_DEPTH"
echo "context_reference_fraction_override=${CONTEXT_REFERENCE_FRACTION:-config}"
echo "continue_mode=${CONTINUE_MODE:-config}"
echo "resume_path=${RESUME_PATH:-latest/config}"
echo "ckpt_interval=${CKPT_INTERVAL:-config}"
echo "max_checkpoints=${MAX_CHECKPOINTS:-config}"
echo "smoke_iters=$SMOKE_ITERS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_JOB_TMP=${SLURM_JOB_TMP:-unset}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset}"
echo "PI3_CUDA_MEMORY_LIMIT_GIB=${PI3_CUDA_MEMORY_LIMIT_GIB:-unset}"
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
  "model.ckpt=$PI3_CKPT"
  "log.output_dir=$RUN_DIR"
  "log.tensorboard_dir=$TENSORBOARD_DIR"
  "log.ckpt_dir=$RUN_DIR/ckpts"
  "hydra.run.dir=$RUN_DIR"
)
if [[ "$DATA_FAMILY" == "lmgeo" ]]; then
  COMMON_OVERRIDES+=(
    "lmgeo.data_root=$DATA_ROOT"
    "lmgeo.filter_preprocessed_query_depth=$FILTER_PREPROCESSED_QUERY_DEPTH"
  )
else
  COMMON_OVERRIDES+=("gso.data_root=$DATA_ROOT")
fi
if [[ -n "$CONTEXT_REFERENCE_FRACTION" && "$DATA_FAMILY" != "lmgeo" ]]; then
  echo "ERROR: PI3_CONTEXT_REFERENCE_FRACTION is LMGeo-only." >&2
  exit 2
elif [[ -n "$CONTEXT_REFERENCE_FRACTION" ]]; then
  COMMON_OVERRIDES+=("lmgeo.context_reference_fraction=$CONTEXT_REFERENCE_FRACTION")
fi
if [[ -n "$CKPT_INTERVAL" ]]; then
  COMMON_OVERRIDES+=("log.ckpt_interval=$CKPT_INTERVAL")
fi
if [[ -n "$MAX_CHECKPOINTS" ]]; then
  COMMON_OVERRIDES+=("log.max_checkpoints=$MAX_CHECKPOINTS")
fi
if [[ -n "$RESUME_PATH" ]]; then
  if [[ ! -d "$RESUME_PATH" ]]; then
    echo "ERROR: resume checkpoint directory does not exist: $RESUME_PATH" >&2
    exit 2
  fi
  COMMON_OVERRIDES+=("train.resume=$RESUME_PATH" "train.auto_resume=true")
elif [[ -n "$CONTINUE_MODE" ]]; then
  case "${CONTINUE_MODE,,}" in
    1|true|yes|y|on)
      COMMON_OVERRIDES+=("train.resume=auto" "train.auto_resume=true")
      ;;
    0|false|no|n|off)
      COMMON_OVERRIDES+=("train.resume=null" "train.auto_resume=false")
      ;;
    *)
      echo "ERROR: PI3_CONTINUE must be true/false when set, got: $CONTINUE_MODE" >&2
      exit 2
      ;;
  esac
fi
if [[ "$RUN_MODE" == "smoke" && "$DATA_FAMILY" == "lmgeo" ]]; then
  # Dataset construction must not dominate or stall a memory/forward smoke.
  # Keep every preprocessed-depth filter off, including the real BOP target
  # loaders that do not inherit lmgeo.filter_preprocessed_query_depth.
  COMMON_OVERRIDES+=(
    "lmgeo_profile.filter_preprocessed_query_depth=false"
    "lmgeo.filter_preprocessed_query_depth=false"
  )
  if [[ "$DATA_CONFIG" == *"anchor_scene_pairs"* ]] || [[ "$TRAIN_CONFIG" == *"anchor_mask_conditioning"* ]]; then
    COMMON_OVERRIDES+=(
      "test_dataset.filter_preprocessed_query_depth=false"
      "val_datasets.real_anchor_pairs.dataset.filter_preprocessed_query_depth=false"
      "val_datasets.pbr_anchor_pairs.dataset.filter_preprocessed_query_depth=false"
    )
  else
    COMMON_OVERRIDES+=(
      "test_dataset.filter_target_preprocessed_depth=false"
      "val_datasets.real_test.dataset.filter_target_preprocessed_depth=false"
      "val_datasets.real_test_ref16.dataset.filter_target_preprocessed_depth=false"
    )
  fi
fi

if [[ "$RUN_MODE" == "smoke" ]]; then
  if [[ "$DATA_FAMILY" == "megapose_gso" ]]; then
    mapfile -t GSO_SMOKE_PROFILE < <(
      python - "$TRAIN_CONFIG" "$DATA_CONFIG" <<'PY'
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

train_config, data_config = sys.argv[1:]
with initialize_config_dir(version_base="1.2", config_dir=str(Path("configs").resolve())):
    cfg = compose(
        config_name="default",
        overrides=[f"train={train_config}", f"data={data_config}"],
    )

values = (
    cfg.train.max_img_per_gpu,
    cfg.train.image_num_range[0],
    cfg.train.image_num_range[1],
    cfg.gso_profile.num_reference_range[0],
    cfg.gso_profile.num_reference_range[1],
    cfg.gso_profile.num_query_range[0],
    cfg.gso_profile.num_query_range[1],
)
for value in values:
    print(int(value))
PY
    )
    if [[ "${#GSO_SMOKE_PROFILE[@]}" -ne 7 ]]; then
      echo "ERROR: failed to resolve MegaPose-GSO smoke profile from Hydra." >&2
      exit 2
    fi
    GSO_TRAIN_BUDGET="${GSO_SMOKE_PROFILE[0]}"
    GSO_MIN_VIEWS="${GSO_SMOKE_PROFILE[1]}"
    GSO_MAX_VIEWS="${GSO_SMOKE_PROFILE[2]}"
    GSO_MIN_REFS="${GSO_SMOKE_PROFILE[3]}"
    GSO_MAX_REFS="${GSO_SMOKE_PROFILE[4]}"
    GSO_MIN_QUERIES="${GSO_SMOKE_PROFILE[5]}"
    GSO_MAX_QUERIES="${GSO_SMOKE_PROFILE[6]}"
    if (( GSO_MIN_REFS + GSO_MIN_QUERIES != GSO_MIN_VIEWS )) || \
       (( GSO_MAX_REFS + GSO_MAX_QUERIES != GSO_MAX_VIEWS )); then
      echo "ERROR: GSO profile extrema do not match train.image_num_range." >&2
      exit 2
    fi

    mapfile -t GSO_VAL_NAMES < <(
      python - "$TRAIN_CONFIG" "$DATA_CONFIG" <<'PY'
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

train_config, data_config = sys.argv[1:]
with initialize_config_dir(version_base="1.2", config_dir=str(Path("configs").resolve())):
    cfg = compose(
        config_name="default",
        overrides=[f"train={train_config}", f"data={data_config}"],
    )
for name in cfg.val_datasets:
    print(str(name))
PY
    )
    GSO_VAL_SMOKE_OVERRIDES=()
    for val_name in "${GSO_VAL_NAMES[@]}"; do
      GSO_VAL_SMOKE_OVERRIDES+=("val_datasets.${val_name}.runtime.iters_per_test=1")
    done

    GSO_MAX_SAMPLE_BATCH=$((GSO_TRAIN_BUDGET / GSO_MAX_VIEWS))
    (( GSO_MAX_SAMPLE_BATCH > 0 )) || GSO_MAX_SAMPLE_BATCH=1
    GSO_MAX_IMAGES=$((GSO_MAX_SAMPLE_BATCH * GSO_MAX_VIEWS))
    echo "Running ${SMOKE_ITERS}-step ${GSO_MAX_SAMPLE_BATCH} x ${GSO_MAX_VIEWS}-view MegaPose-GSO anchor-pair smoke (${GSO_MAX_IMAGES} images/rank, budget ${GSO_TRAIN_BUDGET})."
    accelerate launch --config_file configs/accelerate/ddp.yaml \
      --num_processes "$NUM_GPUS" --num_machines 1 \
      scripts/train_pi3.py \
      "${COMMON_OVERRIDES[@]}" \
      "train.image_num_range=[$GSO_MAX_VIEWS,$GSO_MAX_VIEWS]" \
      train.max_img_per_gpu="$GSO_TRAIN_BUDGET" \
      "gso_profile.num_reference_range=[$GSO_MAX_REFS,$GSO_MAX_REFS]" \
      "gso_profile.num_query_range=[$GSO_MAX_QUERIES,$GSO_MAX_QUERIES]" \
      train.num_epoch=1 \
      train.iters_per_epoch="$SMOKE_ITERS" \
      "${GSO_VAL_SMOKE_OVERRIDES[@]}" \
      metrics.enabled=false \
      visuals.enabled=false \
      log.use_tensorboard=false \
      log.save_best=false \
      log.save_checkpoints=false \
      extras.print_config=false

    if (( GSO_MIN_VIEWS != GSO_MAX_VIEWS )); then
      GSO_MIN_SAMPLE_BATCH=$((GSO_TRAIN_BUDGET / GSO_MIN_VIEWS))
      (( GSO_MIN_SAMPLE_BATCH > 0 )) || GSO_MIN_SAMPLE_BATCH=1
      GSO_MIN_IMAGES=$((GSO_MIN_SAMPLE_BATCH * GSO_MIN_VIEWS))
      echo "Running ${SMOKE_ITERS}-step packed-min ${GSO_MIN_SAMPLE_BATCH} x ${GSO_MIN_VIEWS}-view MegaPose-GSO anchor-pair smoke (${GSO_MIN_IMAGES} images/rank, budget ${GSO_TRAIN_BUDGET})."
      accelerate launch --config_file configs/accelerate/ddp.yaml \
        --num_processes "$NUM_GPUS" --num_machines 1 \
        scripts/train_pi3.py \
        "${COMMON_OVERRIDES[@]}" \
        "train.image_num_range=[$GSO_MIN_VIEWS,$GSO_MIN_VIEWS]" \
        train.max_img_per_gpu="$GSO_TRAIN_BUDGET" \
        "gso_profile.num_reference_range=[$GSO_MIN_REFS,$GSO_MIN_REFS]" \
        "gso_profile.num_query_range=[$GSO_MIN_QUERIES,$GSO_MIN_QUERIES]" \
        train.num_epoch=1 \
        train.iters_per_epoch="$SMOKE_ITERS" \
        "${GSO_VAL_SMOKE_OVERRIDES[@]}" \
        metrics.enabled=false \
        visuals.enabled=false \
        log.use_tensorboard=false \
        log.save_best=false \
        log.save_checkpoints=false \
        extras.print_config=false
    fi
  elif [[ "$DATA_CONFIG" == *"anchor_scene_pairs"* ]] || [[ "$TRAIN_CONFIG" == *"anchor_mask_conditioning"* ]]; then
    echo "Running ${SMOKE_ITERS}-step max-view 26-view anchor-mask-conditioning smoke."
    accelerate launch --config_file configs/accelerate/ddp.yaml \
      --num_processes "$NUM_GPUS" --num_machines 1 \
      scripts/train_pi3.py \
      "${COMMON_OVERRIDES[@]}" \
      'train.image_num_range=[26,26]' \
      train.max_img_per_gpu=28 \
      'lmgeo_profile.num_reference_range=[16,16]' \
      'lmgeo_profile.num_query_range=[10,10]' \
      train.num_epoch=1 \
      train.iters_per_epoch="$SMOKE_ITERS" \
      val_datasets.real_anchor_pairs.runtime.iters_per_test=1 \
      val_datasets.real_anchor_pairs.runtime.max_img_per_gpu=6 \
      val_datasets.pbr_anchor_pairs.runtime.iters_per_test=1 \
      val_datasets.pbr_anchor_pairs.runtime.max_img_per_gpu=6 \
      metrics.enabled=false \
      visuals.enabled=false \
      log.use_tensorboard=false \
      log.save_best=false \
      log.save_checkpoints=false \
      extras.print_config=false

    echo "Running ${SMOKE_ITERS}-step packed-min 9 x 3-view anchor-mask-conditioning smoke."
    accelerate launch --config_file configs/accelerate/ddp.yaml \
      --num_processes "$NUM_GPUS" --num_machines 1 \
      scripts/train_pi3.py \
      "${COMMON_OVERRIDES[@]}" \
      'train.image_num_range=[3,3]' \
      train.max_img_per_gpu=28 \
      'lmgeo_profile.num_reference_range=[2,2]' \
      'lmgeo_profile.num_query_range=[1,1]' \
      train.num_epoch=1 \
      train.iters_per_epoch="$SMOKE_ITERS" \
      val_datasets.real_anchor_pairs.runtime.iters_per_test=1 \
      val_datasets.real_anchor_pairs.runtime.max_img_per_gpu=6 \
      val_datasets.pbr_anchor_pairs.runtime.iters_per_test=1 \
      val_datasets.pbr_anchor_pairs.runtime.max_img_per_gpu=6 \
      metrics.enabled=false \
      visuals.enabled=false \
      log.use_tensorboard=false \
      log.save_best=false \
      log.save_checkpoints=false \
      extras.print_config=false
  elif [[ "$DATA_CONFIG" == *"recenter_zoom_plus_original_k1"* ]]; then
    echo "Running ${SMOKE_ITERS}-step max-view 18-view paired K=1 recenter+zoom smoke."
    accelerate launch --config_file configs/accelerate/ddp.yaml \
      --num_processes "$NUM_GPUS" --num_machines 1 \
      scripts/train_pi3.py \
      "${COMMON_OVERRIDES[@]}" \
      'train.image_num_range=[18,18]' \
      train.max_img_per_gpu=28 \
      'lmgeo_profile.num_reference_range=[16,16]' \
      'lmgeo_profile.num_query_range=[1,1]' \
      train.num_epoch=1 \
      train.iters_per_epoch="$SMOKE_ITERS" \
      val_datasets.real_test.runtime.iters_per_test=1 \
      val_datasets.real_test.runtime.max_img_per_gpu=7 \
      val_datasets.real_test_ref16.runtime.iters_per_test=1 \
      val_datasets.real_test_ref16.runtime.max_img_per_gpu=18 \
      val_datasets.pbr_new_val.runtime.iters_per_test=1 \
      val_datasets.pbr_new_val.runtime.max_img_per_gpu=7 \
      val_datasets.pbr_new_val_ref16.runtime.iters_per_test=1 \
      val_datasets.pbr_new_val_ref16.runtime.max_img_per_gpu=18 \
      metrics.enabled=false \
      visuals.enabled=false \
      log.use_tensorboard=false \
      log.save_best=false \
      log.save_checkpoints=false \
      extras.print_config=false

    echo "Running ${SMOKE_ITERS}-step packed-min 7 x 4-view paired K=1 recenter+zoom smoke."
    accelerate launch --config_file configs/accelerate/ddp.yaml \
      --num_processes "$NUM_GPUS" --num_machines 1 \
      scripts/train_pi3.py \
      "${COMMON_OVERRIDES[@]}" \
      'train.image_num_range=[4,4]' \
      train.max_img_per_gpu=28 \
      'lmgeo_profile.num_reference_range=[2,2]' \
      'lmgeo_profile.num_query_range=[1,1]' \
      train.num_epoch=1 \
      train.iters_per_epoch="$SMOKE_ITERS" \
      val_datasets.real_test.runtime.iters_per_test=1 \
      val_datasets.real_test.runtime.max_img_per_gpu=7 \
      val_datasets.real_test_ref16.runtime.iters_per_test=1 \
      val_datasets.real_test_ref16.runtime.max_img_per_gpu=18 \
      val_datasets.pbr_new_val.runtime.iters_per_test=1 \
      val_datasets.pbr_new_val.runtime.max_img_per_gpu=7 \
      val_datasets.pbr_new_val_ref16.runtime.iters_per_test=1 \
      val_datasets.pbr_new_val_ref16.runtime.max_img_per_gpu=18 \
      metrics.enabled=false \
      visuals.enabled=false \
      log.use_tensorboard=false \
      log.save_best=false \
      log.save_checkpoints=false \
      extras.print_config=false
  elif { [[ "$TRAIN_CONFIG" == *"recenter_zoom"* ]] && [[ "$TRAIN_CONFIG" == *"k1"* ]]; } || \
     { [[ "$DATA_CONFIG" == *"recenter_zoom"* ]] && [[ "$DATA_CONFIG" == *"k1"* ]]; }; then
    echo "Running ${SMOKE_ITERS}-step max-view 17-view K=1 recenter+zoom smoke."
    accelerate launch --config_file configs/accelerate/ddp.yaml \
      --num_processes "$NUM_GPUS" --num_machines 1 \
      scripts/train_pi3.py \
      "${COMMON_OVERRIDES[@]}" \
      'train.image_num_range=[17,17]' \
      train.max_img_per_gpu=28 \
      'lmgeo_profile.num_reference_range=[16,16]' \
      'lmgeo_profile.num_query_range=[1,1]' \
      train.num_epoch=1 \
      train.iters_per_epoch="$SMOKE_ITERS" \
      val_datasets.real_test.runtime.iters_per_test=1 \
      val_datasets.real_test.runtime.max_img_per_gpu=6 \
      val_datasets.real_test_ref16.runtime.iters_per_test=1 \
      val_datasets.real_test_ref16.runtime.max_img_per_gpu=17 \
      val_datasets.pbr_new_val.runtime.iters_per_test=1 \
      val_datasets.pbr_new_val.runtime.max_img_per_gpu=6 \
      val_datasets.pbr_new_val_ref16.runtime.iters_per_test=1 \
      val_datasets.pbr_new_val_ref16.runtime.max_img_per_gpu=17 \
      metrics.enabled=false \
      visuals.enabled=false \
      log.use_tensorboard=false \
      log.save_best=false \
      log.save_checkpoints=false \
      extras.print_config=false

    echo "Running ${SMOKE_ITERS}-step packed-min 9 x 3-view K=1 recenter+zoom smoke."
    accelerate launch --config_file configs/accelerate/ddp.yaml \
      --num_processes "$NUM_GPUS" --num_machines 1 \
      scripts/train_pi3.py \
      "${COMMON_OVERRIDES[@]}" \
      'train.image_num_range=[3,3]' \
      train.max_img_per_gpu=28 \
      'lmgeo_profile.num_reference_range=[2,2]' \
      'lmgeo_profile.num_query_range=[1,1]' \
      train.num_epoch=1 \
      train.iters_per_epoch="$SMOKE_ITERS" \
      val_datasets.real_test.runtime.iters_per_test=1 \
      val_datasets.real_test.runtime.max_img_per_gpu=6 \
      val_datasets.real_test_ref16.runtime.iters_per_test=1 \
      val_datasets.real_test_ref16.runtime.max_img_per_gpu=17 \
      val_datasets.pbr_new_val.runtime.iters_per_test=1 \
      val_datasets.pbr_new_val.runtime.max_img_per_gpu=6 \
      val_datasets.pbr_new_val_ref16.runtime.iters_per_test=1 \
      val_datasets.pbr_new_val_ref16.runtime.max_img_per_gpu=17 \
      metrics.enabled=false \
      visuals.enabled=false \
      log.use_tensorboard=false \
      log.save_best=false \
      log.save_checkpoints=false \
      extras.print_config=false
  else
    echo "Running ${SMOKE_ITERS}-step max-view 28-view 560x420 smoke."
    accelerate launch --config_file configs/accelerate/ddp.yaml \
      --num_processes "$NUM_GPUS" --num_machines 1 \
      scripts/train_pi3.py \
      "${COMMON_OVERRIDES[@]}" \
      'train.image_num_range=[28,28]' \
      train.max_img_per_gpu=28 \
      'lmgeo_profile.num_reference_range=[5,5]' \
      'lmgeo_profile.num_query_range=[23,23]' \
      train.num_epoch=1 \
      train.iters_per_epoch="$SMOKE_ITERS" \
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

    echo "Running ${SMOKE_ITERS}-step packed-min 4 x 6-view 560x420 smoke."
    accelerate launch --config_file configs/accelerate/ddp.yaml \
      --num_processes "$NUM_GPUS" --num_machines 1 \
      scripts/train_pi3.py \
      "${COMMON_OVERRIDES[@]}" \
      'train.image_num_range=[6,6]' \
      train.max_img_per_gpu=28 \
      'lmgeo_profile.num_reference_range=[5,5]' \
      'lmgeo_profile.num_query_range=[1,1]' \
      train.num_epoch=1 \
      train.iters_per_epoch="$SMOKE_ITERS" \
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
  fi
elif [[ "$RUN_MODE" == "train" ]]; then
  echo "Running production training with $TRAIN_CONFIG and $DATA_CONFIG on $NUM_GPUS GPU(s)."
  accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes "$NUM_GPUS" --num_machines 1 \
    scripts/train_pi3.py \
    "${COMMON_OVERRIDES[@]}"
else
  echo "ERROR: unknown PI3_RUN_MODE=$RUN_MODE. Use preflight, smoke, or train." >&2
  exit 2
fi
