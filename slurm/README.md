# Sink-inheritance pilot jobs for bwUniCluster 3.0

These files submit the exact commands specified by the sink-inheritance pilot document without embedding workstation-specific paths in the repository.

## Included jobs

| File | Queue | Limit | Purpose |
| --- | --- | ---: | --- |
| `pilot_smoke.sbatch` | `dev_gpu_h100` | 30 minutes | Imports, model loading, path validation, and a tiny pilot run. |
| `pilot_run.sbatch` | `gpu_h100` | 12 hours | One production pilot command using one H100 GPU and eight CPU cores. |
| `pilot_common.sh` | n/a | n/a | Conda, workspace, caches, metadata, CUDA validation, and result paths. |
| `submit_pilot.sh` | n/a | n/a | Quotes and exports the exact pilot command, creates `logs/`, and submits the selected job. |

The production wall time and partition can be overridden at submission. UC3 command-line options take precedence over `#SBATCH` directives.

## One-time setup

From the repository root:

```bash
module purge
module load devel/miniforge
conda create -n sinks python=3.11 -y
conda activate sinks
pip install -r requirements.txt

# Recommended for model caches and generated artifacts.
ws_allocate sink-inheritance 60
```

A path-based environment is also supported:

```bash
export SINKS_ENV_PATH="$(ws_find sink-inheritance)/conda/envs/sinks"
```

## Submit the pilot

Copy a complete Python command from the pilot Markdown after `--`.

First run a smoke test:

```bash
bash slurm/submit_pilot.sh smoke -- python path/to/pilot_entrypoint.py --help
```

Then submit the real pilot command:

```bash
bash slurm/submit_pilot.sh full -- \
  python path/to/pilot_entrypoint.py \
  --config path/to/pilot_config.yaml
```

The wrapper shell-quotes every argument before exporting it as `PILOT_CMD`; the batch job executes that command unchanged on the allocated compute node.

## Override resources

Use an A100 or a longer wall time without editing committed files:

```bash
SBATCH_ARGS='--partition=gpu_a100_il --time=24:00:00' \
  bash slurm/submit_pilot.sh full -- \
  python path/to/pilot_entrypoint.py --config path/to/pilot_config.yaml
```

For a different Conda environment or workspace:

```bash
CONDA_ENV_NAME=my-env WORKSPACE_NAME=my-workspace \
  bash slurm/submit_pilot.sh smoke -- python path/to/pilot_entrypoint.py --help
```

## Outputs

SLURM stdout and stderr are written to:

```text
logs/<job-name>-<job-id>.out
logs/<job-name>-<job-id>.err
```

Run metadata and status are written under:

```text
<workspace>/results/sink-inheritance/smoke-<job-id>/
<workspace>/results/sink-inheritance/full-<job-id>/
```

When the named workspace does not exist, the scripts fall back to:

```text
results/sink-inheritance/
```

Each run records the Git commit, modules, Python and PyTorch versions, allocated GPU information, command, timestamps, and exit code.

## Monitor

```bash
squeue
squeue --start
scontrol show job <job-id>
tail -f logs/<job-name>-<job-id>.out
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS
```

Do not execute the pilot directly on a login node.
