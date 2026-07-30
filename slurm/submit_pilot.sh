#!/usr/bin/env bash
# Submit one exact command from the pilot Markdown to bwUniCluster 3.0.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash slurm/submit_pilot.sh smoke -- <pilot command and arguments>
  bash slurm/submit_pilot.sh full  -- <pilot command and arguments>

Examples:
  bash slurm/submit_pilot.sh smoke -- python path/to/pilot.py --help
  bash slurm/submit_pilot.sh full  -- python path/to/pilot.py --config configs/pilot.yaml

Optional environment variables:
  PROJECT_ROOT      Repository root. Defaults to the parent of slurm/.
  CONDA_ENV_NAME    Named Conda environment. Defaults to sinks.
  SINKS_ENV_PATH    Absolute Conda environment path; overrides CONDA_ENV_NAME.
  WORKSPACE_NAME    bwHPC workspace name. Defaults to sink-inheritance.
  RESULTS_ROOT      Override the results directory.
  CACHE_ROOT        Override model/package cache root.

SLURM command-line options override directives in the job file. For example:
  SBATCH_ARGS='--time=48:00:00 --partition=gpu_a100_il' \
    bash slurm/submit_pilot.sh full -- python path/to/pilot.py --config configs/pilot.yaml
USAGE
}

if [[ $# -lt 3 ]]; then
  usage >&2
  exit 2
fi

mode="$1"
shift

if [[ "$1" != "--" ]]; then
  echo "ERROR: expected -- before the pilot command." >&2
  usage >&2
  exit 2
fi
shift

if [[ $# -eq 0 ]]; then
  echo "ERROR: no pilot command was provided." >&2
  exit 2
fi

case "${mode}" in
  smoke) job_file="pilot_smoke.sbatch" ;;
  full)  job_file="pilot_run.sbatch" ;;
  *)
    echo "ERROR: mode must be 'smoke' or 'full'." >&2
    usage >&2
    exit 2
    ;;
esac

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SLURM_DIR}/.." && pwd)}"
mkdir -p "${PROJECT_ROOT}/logs"

printf -v PILOT_CMD '%q ' "$@"
export PILOT_CMD PROJECT_ROOT

read -r -a extra_sbatch_args <<< "${SBATCH_ARGS:-}"

printf 'Submitting %s pilot job\n' "${mode}"
printf 'Project root: %s\n' "${PROJECT_ROOT}"
printf 'Command: %s\n' "${PILOT_CMD}"

sbatch "${extra_sbatch_args[@]}" "${SLURM_DIR}/${job_file}"
