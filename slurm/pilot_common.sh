#!/usr/bin/env bash
# Shared runtime initialization for bwUniCluster 3.0 pilot jobs.

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SLURM_DIR}/.." && pwd)}"
WORKSPACE_NAME="${WORKSPACE_NAME:-sink-inheritance}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sinks}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "ERROR: PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p logs

module purge
module load devel/miniforge

eval "$(conda shell.bash hook)"

if [[ -n "${SINKS_ENV_PATH:-}" ]]; then
  if [[ ! -d "${SINKS_ENV_PATH}" ]]; then
    echo "ERROR: SINKS_ENV_PATH does not exist: ${SINKS_ENV_PATH}" >&2
    exit 2
  fi
  conda activate "${SINKS_ENV_PATH}"
else
  if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
    echo "ERROR: Conda environment '${CONDA_ENV_NAME}' was not found." >&2
    echo "Create it first or export SINKS_ENV_PATH=/absolute/path/to/env." >&2
    exit 2
  fi
  conda activate "${CONDA_ENV_NAME}"
fi

WORKSPACE_PATH=""
if command -v ws_find >/dev/null 2>&1; then
  WORKSPACE_PATH="$(ws_find "${WORKSPACE_NAME}" 2>/dev/null || true)"
fi

if [[ -n "${WORKSPACE_PATH}" && -d "${WORKSPACE_PATH}" ]]; then
  CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_PATH}/cache}"
  RESULTS_ROOT="${RESULTS_ROOT:-${WORKSPACE_PATH}/results/sink-inheritance}"
else
  CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
  RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/sink-inheritance}"
  echo "WARNING: workspace '${WORKSPACE_NAME}' was not found; using project-local cache/results." >&2
fi

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

mkdir -p \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${RESULTS_ROOT}"

RUN_LABEL="${RUN_LABEL:-${SLURM_JOB_NAME:-pilot}-${SLURM_JOB_ID:-manual}}"
RUN_DIR="${RUN_DIR:-${RESULTS_ROOT}/${RUN_LABEL}}"
mkdir -p "${RUN_DIR}"
export PROJECT_ROOT WORKSPACE_PATH RESULTS_ROOT RUN_DIR

if [[ -z "${PILOT_CMD:-}" ]]; then
  echo "ERROR: PILOT_CMD is empty." >&2
  echo "Submit with: bash slurm/submit_pilot.sh smoke -- python <pilot-entrypoint> <args>" >&2
  exit 2
fi

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "job_name=${SLURM_JOB_NAME:-none}"
  echo "partition=${SLURM_JOB_PARTITION:-none}"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-none}"
  echo "project_root=${PROJECT_ROOT}"
  echo "workspace_path=${WORKSPACE_PATH:-none}"
  echo "run_dir=${RUN_DIR}"
  echo "python=$(command -v python)"
  echo "pilot_stage=${PILOT_STAGE:-unspecified}"
  echo "pilot_cmd=${PILOT_CMD}"
  git rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || true
  python --version 2>&1 | sed 's/^/python_version=/'
} | tee "${RUN_DIR}/run_metadata.txt"

module list 2>"${RUN_DIR}/modules.txt" || true

python - <<'PY' | tee "${RUN_DIR}/accelerator_check.txt"
import sys

try:
    import torch
except Exception as exc:
    print(f"PyTorch import failed: {exc}", file=sys.stderr)
    raise

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"visible_gpus={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"gpu_{index}={torch.cuda.get_device_name(index)}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the allocated GPU job")
PY

nvidia-smi | tee "${RUN_DIR}/nvidia-smi.txt"
