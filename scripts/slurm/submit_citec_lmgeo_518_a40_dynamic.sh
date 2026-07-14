#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/citec_lmgeo_518_a40_dynamic.sbatch"

RUNS_ROOT="${PI3_RUNS_ROOT:-/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3}"
LOG_DIR="${PI3_SLURM_LOG_DIR:-$RUNS_ROOT/slurm_logs}"
MAIL_USER="${PI3_MAIL_USER:-dmitrii.trofimov@uni-bielefeld.de}"
MAIL_TYPE="${PI3_MAIL_TYPE:-BEGIN,END,FAIL}"
ACCOUNT="${PI3_SLURM_ACCOUNT:-${SLURM_ACCOUNT:-}}"
mkdir -p "$LOG_DIR"

case "$MODE" in
  preflight)
    SBATCH_OPTS=(
      --job-name=pi3-preflight-a40
      --gres=gpu:a40:1
      --cpus-per-task=8
      --mem=80G
      --tmp=20G
      --time=00:20:00
    )
    EXPECT_GPUS=1
    ;;
  smoke)
    SBATCH_OPTS=(
      --job-name=pi3-smoke-518-a40
      --gres=gpu:a40:1
      --cpus-per-task=8
      --mem=160G
      --tmp=50G
      --time=02:00:00
    )
    EXPECT_GPUS=1
    ;;
  train)
    SBATCH_OPTS=(
      --job-name=pi3-lmgeo-518
      --gres=gpu:a40:4
      --cpus-per-task=32
      --mem=400G
      --tmp=100G
      --time="${PI3_WALLTIME:-48:00:00}"
    )
    EXPECT_GPUS=4
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
  --export=ALL,PI3_RUN_MODE="$MODE",PI3_EXPECT_GPUS="$EXPECT_GPUS" \
  "${SBATCH_OPTS[@]}" \
  "$SBATCH_SCRIPT"
