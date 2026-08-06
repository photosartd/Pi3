#!/usr/bin/env bash
set -euo pipefail

MODE="train"
if [[ $# -gt 0 && "$1" != --* ]]; then
  MODE="$1"
  shift
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/citec_lmgeo_518_a40_dynamic.sbatch"

RUNS_ROOT="${PI3_RUNS_ROOT:-/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3}"
LOG_DIR="${PI3_SLURM_LOG_DIR:-$RUNS_ROOT/slurm_logs}"
MAIL_USER="${PI3_MAIL_USER:-dmitrii.trofimov@uni-bielefeld.de}"
MAIL_TYPE="${PI3_MAIL_TYPE:-BEGIN,END,FAIL}"
ACCOUNT="${PI3_SLURM_ACCOUNT:-${SLURM_ACCOUNT:-}}"
TRAIN_GPUS="${PI3_TRAIN_GPUS:-4}"
CONTINUE_MODE="${PI3_CONTINUE:-}"
RESUME_PATH="${PI3_RESUME:-}"
CKPT_INTERVAL="${PI3_CKPT_INTERVAL:-1}"
MAX_CHECKPOINTS="${PI3_MAX_CHECKPOINTS:-5}"
DATA_CONFIG_HINT="${PI3_DATA_CONFIG:-lmgeo_trainpbr45_real_and_new_val}"
TRAIN_CONFIG_HINT="${PI3_TRAIN_CONFIG:-train_lmgeo_finetune_a40_46gb}"
if [[ "$DATA_CONFIG_HINT" == megapose_gso_* ]]; then
  DEFAULT_JOB_PREFIX="pi3-gso-a40"
else
  DEFAULT_JOB_PREFIX="pi3-lmgeo-a40"
fi
JOB_PREFIX="${PI3_SLURM_JOB_PREFIX:-$DEFAULT_JOB_PREFIX}"
mkdir -p "$LOG_DIR"

usage() {
  cat >&2 <<'EOF'
Usage: scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh [preflight|smoke|train] [options]

Options:
  --continue            Resume from the latest checkpoint in this run's ckpt dir.
  --fresh               Force a fresh start even if checkpoints already exist.
  --resume PATH         Resume from an exact Accelerate checkpoint directory.
  --ckpt-interval N     Save checkpoints every N epochs. Default: 1.
  --max-checkpoints N   Keep at most N recent checkpoints. Default: 5.
EOF
}

positive_int() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: $1 must be a positive integer, got: $2" >&2
    exit 2
  }
}

bool_is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --continue)
      CONTINUE_MODE=true
      shift
      ;;
    --fresh)
      CONTINUE_MODE=false
      shift
      ;;
    --resume)
      [[ $# -ge 2 ]] || { echo "ERROR: --resume needs a checkpoint directory" >&2; usage; exit 2; }
      RESUME_PATH="$2"
      CONTINUE_MODE=true
      shift 2
      ;;
    --ckpt-interval)
      [[ $# -ge 2 ]] || { echo "ERROR: --ckpt-interval needs a value" >&2; usage; exit 2; }
      positive_int "--ckpt-interval" "$2"
      CKPT_INTERVAL="$2"
      shift 2
      ;;
    --max-checkpoints)
      [[ $# -ge 2 ]] || { echo "ERROR: --max-checkpoints needs a value" >&2; usage; exit 2; }
      positive_int "--max-checkpoints" "$2"
      MAX_CHECKPOINTS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! [[ "$TRAIN_GPUS" =~ ^[1-4]$ ]]; then
  echo "ERROR: PI3_TRAIN_GPUS must be an integer from 1 to 4, got: $TRAIN_GPUS" >&2
  exit 2
fi
if [[ -n "$CKPT_INTERVAL" ]]; then
  positive_int "PI3_CKPT_INTERVAL" "$CKPT_INTERVAL"
fi
if [[ -n "$MAX_CHECKPOINTS" ]]; then
  positive_int "PI3_MAX_CHECKPOINTS" "$MAX_CHECKPOINTS"
fi
if [[ -n "$RESUME_PATH" && ! -d "$RESUME_PATH" ]]; then
  echo "ERROR: resume checkpoint directory does not exist: $RESUME_PATH" >&2
  exit 2
fi
if bool_is_true "$CONTINUE_MODE" && [[ -z "$RESUME_PATH" ]]; then
  RUN_DIR_FOR_CHECK="${PI3_RUN_DIR:-}"
  if [[ -z "$RUN_DIR_FOR_CHECK" && -n "${PI3_RUN_NAME:-}" ]]; then
    RUN_DIR_FOR_CHECK="$RUNS_ROOT/$PI3_RUN_NAME"
  fi
  if [[ -z "$RUN_DIR_FOR_CHECK" ]]; then
    echo "ERROR: --continue needs PI3_RUN_NAME or PI3_RUN_DIR so the previous run folder can be found." >&2
    exit 2
  fi
  if ! find "$RUN_DIR_FOR_CHECK/ckpts" -maxdepth 1 -type d -name 'checkpoint_*' -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: --continue requested, but no checkpoint_* directory was found under: $RUN_DIR_FOR_CHECK/ckpts" >&2
    exit 2
  fi
fi

case "$MODE" in
  preflight)
    SBATCH_OPTS=(
      --job-name="$JOB_PREFIX-preflight"
      --gres=gpu:a40:1
      --cpus-per-task=8
      --mem=80G
      --tmp=20G
      --time=00:20:00
    )
    EXPECT_GPUS=1
    ;;
  smoke)
    if [[ "$DATA_CONFIG_HINT" == megapose_gso_* && "$TRAIN_CONFIG_HINT" == *"336x252"* ]]; then
      DEFAULT_SMOKE_GPUS=2
      DEFAULT_SMOKE_MEM="80G"
    else
      DEFAULT_SMOKE_GPUS=1
      DEFAULT_SMOKE_MEM="160G"
    fi
    SMOKE_GPUS="${PI3_SMOKE_GPUS:-$DEFAULT_SMOKE_GPUS}"
    if ! [[ "$SMOKE_GPUS" =~ ^[1-4]$ ]]; then
      echo "ERROR: PI3_SMOKE_GPUS must be an integer from 1 to 4, got: $SMOKE_GPUS" >&2
      exit 2
    fi
    SBATCH_OPTS=(
      --job-name="$JOB_PREFIX-smoke"
      --gres=gpu:a40:"$SMOKE_GPUS"
      --cpus-per-task="${PI3_SMOKE_CPUS:-$((SMOKE_GPUS * 8))}"
      --mem="${PI3_SMOKE_MEM:-$DEFAULT_SMOKE_MEM}"
      --tmp=50G
      --time=02:00:00
    )
    EXPECT_GPUS="$SMOKE_GPUS"
    ;;
  train)
    TRAIN_CPUS="${PI3_TRAIN_CPUS:-$((TRAIN_GPUS * 8))}"
    if [[ "$DATA_CONFIG_HINT" == megapose_gso_* && "$TRAIN_CONFIG_HINT" == *"336x252"* ]]; then
      DEFAULT_TRAIN_MEM="110G"
    else
      DEFAULT_TRAIN_MEM="$((TRAIN_GPUS * 100))G"
    fi
    TRAIN_MEM="${PI3_TRAIN_MEM:-$DEFAULT_TRAIN_MEM}"
    TRAIN_TMP="${PI3_TRAIN_TMP:-$((TRAIN_GPUS * 25))G}"
    SBATCH_OPTS=(
      --job-name="$JOB_PREFIX-train"
      --gres=gpu:a40:"$TRAIN_GPUS"
      --cpus-per-task="$TRAIN_CPUS"
      --mem="$TRAIN_MEM"
      --tmp="$TRAIN_TMP"
      --time="${PI3_WALLTIME:-48:00:00}"
    )
    EXPECT_GPUS="$TRAIN_GPUS"
    ;;
  *)
    echo "Usage: $0 [preflight|smoke|train]" >&2
    exit 2
    ;;
esac

if [[ -n "$MAIL_USER" ]]; then
  SBATCH_OPTS+=(--mail-user="$MAIL_USER")
fi

if [[ -n "$MAIL_TYPE" ]]; then
  SBATCH_OPTS+=(--mail-type="$MAIL_TYPE")
fi

if [[ -n "$ACCOUNT" ]]; then
  SBATCH_OPTS+=(--account="$ACCOUNT")
fi

sbatch \
  --partition=gpu \
  --output="$LOG_DIR/%x-%j.out" \
  --error="$LOG_DIR/%x-%j.err" \
  --export=ALL,PI3_RUN_MODE="$MODE",PI3_EXPECT_GPUS="$EXPECT_GPUS",PI3_CONTINUE="$CONTINUE_MODE",PI3_RESUME="$RESUME_PATH",PI3_CKPT_INTERVAL="$CKPT_INTERVAL",PI3_MAX_CHECKPOINTS="$MAX_CHECKPOINTS" \
  "${SBATCH_OPTS[@]}" \
  "$SBATCH_SCRIPT"
