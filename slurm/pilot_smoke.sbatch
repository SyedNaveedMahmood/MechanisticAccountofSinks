#!/usr/bin/env bash
#SBATCH --job-name=sink-pilot-smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PILOT_STAGE="smoke"
export RUN_LABEL="smoke-${SLURM_JOB_ID}"

source "${SLURM_DIR}/pilot_common.sh"

status_file="${RUN_DIR}/job_status.txt"
trap 'code=$?; printf "exit_code=%s\nfinished_at=%s\n" "$code" "$(date --iso-8601=seconds)" > "${status_file}"; exit "$code"' EXIT

printf 'Executing smoke command:\n%s\n' "${PILOT_CMD}"
srun --unbuffered bash -lc "${PILOT_CMD}"
