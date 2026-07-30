#!/usr/bin/env bash
#SBATCH --job-name=sink-pilot
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PILOT_STAGE="full"
export RUN_LABEL="full-${SLURM_JOB_ID}"

source "${SLURM_DIR}/pilot_common.sh"

status_file="${RUN_DIR}/job_status.txt"
trap 'code=$?; printf "exit_code=%s\nfinished_at=%s\n" "$code" "$(date --iso-8601=seconds)" > "${status_file}"; exit "$code"' EXIT

printf 'Executing production pilot command:\n%s\n' "${PILOT_CMD}"
srun --unbuffered bash -lc "${PILOT_CMD}"
