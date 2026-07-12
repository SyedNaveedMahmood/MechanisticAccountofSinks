# E5 Implementation and Usage Guide

This is the operational guide for the E5 implementation on branch
`E5_Implimentation`. For the scientific motivation, hypotheses, and formal metric
definitions, see [`E5.md`](E5.md).

> The implementation has passed source compilation and Git whitespace checks, but
> the smoke test, parity check, fidelity check, model downloads, dataset downloads,
> and GPU experiments were intentionally not run during implementation.

## 1. What was implemented

The entry point is:

```text
evaluation_robustness_analysis.py
```

It provides four experiment modes and automatic statistical aggregation:

| Mode | Purpose | Main outputs |
|---|---|---|
| `metrics` | Alternative attention-sink metrics for all ten Table 1 interventions | `metrics_per_example.csv`, metric concordance, head-cell analysis, E5-A/B figures |
| `length` | Paired length generalization from 40 to 1024 tokens | length summaries, logit slopes, invariance checks, E5-C figure |
| `cost` | Cross-entropy cost and mitigation frontier | CE tables, Pareto frontier, position profiles, E5-D/E figures |
| `content` | Synthetic and optional multilingual content controls | content summaries, paired tests, E5-F figure |
| `all` | Run or aggregate all four modes | Complete E5 output set |

The implementation also includes:

- `--smoke-test`: an offline tiny random GPT-2 test using synthetic token IDs;
- `--verify-parity`: comparison with the existing Table 1 intervention registry;
- `--plot-only`: aggregation and plotting from cached CSV/JSON files without loading
  a model or dataset;
- deterministic per-seed manifests and configuration files;
- fixed seeded random-`W_k` coordinates;
- exact token-ID nested prefixes for length comparisons;
- hierarchy-preserving bootstrap calibration and MDE calculations.

## 2. Code architecture

### Declarative interventions

`InterventionSpec` describes every intervention without duplicating its semantics
between attention and CE paths. A specification can control:

- token embeddings;
- positional embeddings;
- query-bias scale;
- layer-0 MLP transformations;
- all-layer MLP skipping;
- massive-coordinate `W_k` edits;
- fixed random-coordinate `W_k` edits.

`build_intervention_specs()` creates the ten Table 1 interventions, the four E4
combinations, and dose-response variants.

### Unified execution

`execute_spec()` performs the manual GPT-2 forward pass. The same executor is used
for:

- attention-map collection;
- metric calculation;
- cross-entropy calculation;
- length experiments;
- content experiments;
- parity verification.

This prevents an intervention from having one meaning in attention analysis and a
different meaning in functional-cost analysis.

### Shared-file changes

- `intervention_analysis.py`: the non-diagnostic path now skips unused diagnostic
  allocations and loops while preserving tuple positions and default behavior.
- `residual_sink_analysis.py`: `run_config` and `forward_to_logits` accept optional
  token- and PE-transform hooks with default-preserving behavior.
- `datasets_loader.py`: adds exact-token long-context sampling, deterministic
  synthetic domains, and optional FLORES loading.

## 3. Environment setup

From the repository root:

```bash
conda create -n sinks python=3.11 -y
conda activate sinks
pip install -r requirements.txt
```

The existing requirements already contain the needed libraries. E5 adds no new
dependency.

Confirm that you are on the correct branch:

```bash
git checkout E5_Implimentation
git pull --ff-only origin E5_Implimentation
```

## 4. Recommended validation order

Run validation in this order before launching a full experiment.

### Step 1 — Offline smoke test

This constructs a tiny random GPT-2 locally. It does not require a model or dataset
download.

```bash
python evaluation_robustness_analysis.py --smoke-test --output-dir results/_e5_smoke
```

Check that the command ends successfully and that
`results/_e5_smoke/e5_smoke/aggregate/` contains the E5-A through E5-F figures.

### Step 2 — Intervention parity

This loads GPT-2 but does not load benchmark datasets:

```bash
python evaluation_robustness_analysis.py --verify-parity \
  --model-name gpt2 \
  --output-dir results/_e5_parity
```

Inspect `parity_report.json`. Interventions a–i should pass the configured numerical
tolerances. Intervention j is intentionally labeled as a controlled deviation because
E5 fixes seeded random coordinates instead of drawing different coordinates per layer.

### Step 3 — Small fidelity run

```bash
python evaluation_robustness_analysis.py --mode metrics \
  --model-name gpt2 \
  --seeds 0 \
  --sample-size 10 \
  --output-dir results/_e5_fidelity
```

Inspect the baseline and Nullify Query Bias rows before scaling up. Do not treat this
small sample as a reported result.

## 5. Running each mode

### Metric multiverse

```bash
python evaluation_robustness_analysis.py --mode metrics \
  --model-name gpt2 \
  --seeds 0,1,2 \
  --sample-size 100 \
  --cut-length 40 \
  --experiment-name e5_full \
  --output-dir results
```

### Length generalization

```bash
python evaluation_robustness_analysis.py --mode length \
  --model-name gpt2 \
  --seeds 0,1,2 \
  --lengths 40,128,256,512,1024 \
  --sample-size 100 \
  --length-sample-size 50 \
  --experiment-name e5_full \
  --output-dir results
```

The first 50 examples per domain are used at 1024 tokens. Shorter lengths use all
100. Short inputs are prefixes of the same long token sequences.

### Functional cost

```bash
python evaluation_robustness_analysis.py --mode cost \
  --model-name gpt2 \
  --seeds 0,1,2 \
  --sample-size 100 \
  --cut-length 40 \
  --alphas 0,0.25,0.5,0.75,1,1.25,1.5 \
  --experiment-name e5_full \
  --output-dir results
```

This mode also creates 40- and 1024-token position-level CE profiles for a/b/c/i.

### Content generality

```bash
python evaluation_robustness_analysis.py --mode content \
  --model-name gpt2 \
  --seeds 0,1,2 \
  --sample-size 100 \
  --cut-length 40 \
  --experiment-name e5_full \
  --output-dir results
```

To attempt Bangla and Chinese FLORES-200 in addition to the synthetic controls:

```bash
python evaluation_robustness_analysis.py --mode content \
  --with-multilingual \
  --experiment-name e5_full \
  --output-dir results
```

If FLORES or a language configuration is unavailable, the run continues and records
the reason in mode metadata.

## 6. Four-GPU execution

Recommended assignment:

| GPU | Mode |
|---|---|
| RTX 5090 | `length` |
| RTX 4090 | `cost` |
| RTX 3090 | `metrics` |
| RTX 4080 Super | `content` |

Run one command on each machine, always using the same experiment name:

```bash
python evaluation_robustness_analysis.py --mode length  --experiment-name e5_full --output-dir results
python evaluation_robustness_analysis.py --mode cost    --experiment-name e5_full --output-dir results
python evaluation_robustness_analysis.py --mode metrics --experiment-name e5_full --output-dir results
python evaluation_robustness_analysis.py --mode content --experiment-name e5_full --output-dir results
```

If the machines do not share storage, copy the mode CSVs and
`run_config_<mode>.json` files into the matching directories:

```text
results/e5_full/seed_000/
results/e5_full/seed_001/
results/e5_full/seed_002/
```

Do not rename files while merging. Keep every manifest and mode-specific config.

## 7. Regenerating plots from cache

After all mode outputs have been merged:

```bash
python evaluation_robustness_analysis.py --mode all \
  --model-name gpt2 \
  --plot-only \
  --experiment-name e5_full \
  --output-dir results
```

`--plot-only` does not load a model or dataset. It validates cached model, dtype,
layer mode, sample settings, lengths, alphas, registry version, layer band, and
coordinate metadata before aggregation.

To regenerate one family only, replace `--mode all` with the relevant mode.

## 8. GPT-2 medium spot-check

Run metrics and functional cost separately into the same experiment directory:

```bash
python evaluation_robustness_analysis.py --mode metrics \
  --model-name gpt2-medium \
  --seeds 0 \
  --sample-size 20 \
  --experiment-name e5_medium_spot \
  --output-dir results

python evaluation_robustness_analysis.py --mode cost \
  --model-name gpt2-medium \
  --seeds 0 \
  --sample-size 20 \
  --experiment-name e5_medium_spot \
  --output-dir results
```

Do not combine GPT-2 medium CSVs with GPT-2 small outputs.

## 9. Important CLI options

| Option | Default | Meaning |
|---|---:|---|
| `--model-name` | `gpt2` | Hugging Face GPT-2 checkpoint or compatible local path |
| `--seeds` | `0,1,2` | Data-resampling seeds |
| `--sample-size` | `100` | Examples per benchmark domain |
| `--cut-length` | `40` | Standard metric/cost/content token length |
| `--lengths` | `40,128,256,512,1024` | Length experiment grid |
| `--length-sample-size` | `50` | Examples/domain at 1024 tokens |
| `--alphas` | `0,...,1.5` | Dose-response values |
| `--layer-mode` | `scaled` | Depth-aware band; `fixed` forces GPT-2-small layers 4–11 |
| `--dtype` | `float32` | Scientific default; smaller dtypes are optional |
| `--random-wk-seed` | `1729` | Fixed random-`W_k` coordinate seed |
| `--bootstrap-repetitions` | `2000` | Within-seed bootstrap draws |
| `--with-multilingual` | off | Attempt optional FLORES domains |
| `--plot-only` | off | Aggregate existing cache only |

Requested lengths are checked against the model context limit. Invalid or empty
domain selections fail with a descriptive error.

## 10. Output layout

Each seed directory contains raw data and enough metadata to audit or regenerate
the result:

```text
results/e5_full/
  seed_000/
    sample_manifest.csv
    source_manifest.csv
    content_manifest.csv
    run_config.json
    run_config_metrics.json
    run_config_length.json
    run_config_cost.json
    run_config_content.json
    metrics_*.csv
    length_*.csv
    cost_*.csv
    position_ce_profiles.csv
    content_per_example.csv
  seed_001/
  seed_002/
  aggregate/
```

Important aggregate tables:

- `metrics_summary.csv`: cross-seed metric estimates and normalization;
- `metric_concordance.csv`: oriented Kendall ranking agreement;
- `head_cell_effects.csv`: paired baseline/post cell values;
- `length_summary.csv`: length-dependent intervention effects;
- `logit_competition_slopes.csv`: raw fitted slopes;
- `length_invariance_checks.csv`: nested-prefix `k0` and `Delta1` discrepancies;
- `cost_summary.csv`: CE and sink effects with uncertainty;
- `mitigation_frontier.csv`: Pareto-efficient configurations;
- `position_ce_damage_profiles.csv`: per-position delta-CE against baseline;
- `content_summary.csv`: domain/intervention results and sample counts;
- `bootstrap_reseed_calibration.csv`: bootstrap/reseed comparison;
- `mde_power.csv`: paired minimum detectable effects.

Figures are named `fig_E5A_...png` through `fig_E5F_...png`.

## 11. Interpreting statuses safely

- `status=ok`: the recorded value was finite and eligible for aggregation.
- `status=nonfinite_ce`: one or more CE values were non-finite. The code records
  valid counts and warning text; it never invents replacements.
- `intentional_deviation` in parity: expected only for fixed seeded random-`W_k`.

Negative `delta_ce` is retained. It must not be clipped to zero. A Pareto point is
marked efficient only if no other configuration has at least as much sink reduction
and no greater CE cost, with one strict improvement.

## 12. Common problems

### Missing Python package

Install the repository requirements in the active environment:

```bash
pip install -r requirements.txt
```

Do not add a new dependency unless the implementation actually requires it.

### Requested length exceeds context limit

Reduce `--lengths` or use a GPT-2 checkpoint whose configured context limit supports
the requested length. E5 fails before starting the experiment.

### Incompatible cache error

Use a new `--experiment-name`, or rerun with arguments matching the cached metadata.
Do not delete or overwrite incompatible results without first preserving them.

### Missing files during `--plot-only`

Confirm that every requested seed has completed every requested mode and that files
from separate machines were copied into the correct `seed_NNN` directories.

### FLORES warning

This does not invalidate the synthetic content experiment. Read
`multilingual_skip_reason` in `run_config_content.json` and report multilingual data
as skipped.

### CUDA out of memory at length 1024

Run length mode alone on the largest GPU, close other GPU processes, and avoid
launching several long-context workers on one device. Keep FP32 for reported results
unless a deliberate lower-precision sensitivity run is being performed.

## 13. Reproducibility checklist

Before interpreting results, confirm:

- the branch and commit are recorded;
- `run_config_<mode>.json` exists for each mode and seed;
- massive and random coordinates are identical where expected;
- every input has the declared token length;
- 1024-token inputs use the first configured long-context sample subset;
- the same example IDs are paired across interventions;
- non-finite CE statuses are inspected rather than silently dropped;
- final figures were regenerated with `--plot-only` after caches were merged;
- no numerical claim is made from smoke or fidelity outputs.
