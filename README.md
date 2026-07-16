Code for reproducing the experiments in the paper "A Mechanistic Account of Attention Sinks in GPT-2: One Circuit, Broader Implications for Mitigation".

## Repository layout

The code is organized into one folder per experiment group. The analysis modules that
every group shares (`datasets_loader.py`, `intervention_analysis.py`,
`residual_sink_analysis.py`, `experiments_single_input.py`) are kept as a single master
copy under [`common/`](common/); each experiment script adds `common/` to its import
path automatically, so you still `cd` into a folder and run its commands from there.

| Folder | Purpose |
| --- | --- |
| [`common/`](common/) | Shared master modules imported by every experiment group: dataset loading, the GPT-2 intervention harness (also the Fig 5 / Table 1 entry point), the residual-sink harness (E4), and single-input helpers (Fig 2 / Fig 7). |
| [`reproduce_paper/`](reproduce_paper/) | Reproduce the original paper's figures and tables (GPT-2 small). |
| [`cross_scale_and_architecture/`](cross_scale_and_architecture/) | Cross-scale + cross-architecture Table 1 sweep (GPT-2 / OPT / GPT-Neo / Qwen2.5) via the multiseed runner. |
| [`emergence_dynamics/`](emergence_dynamics/) | **E3** — training-time emergence dynamics of the sink circuit. |
| [`residual_sink/`](residual_sink/) | **E4** — anatomy of the residual sink. |
| [`evaluation_robustness/`](evaluation_robustness/) | **E5** — evaluation robustness & functional cost of sink removal. |

Extended write-ups live in [`docs/`](docs/) (e.g. [`docs/E3.md`](docs/E3.md),
[`docs/E4_Plan.md`](docs/E4_Plan.md), [`docs/E5.md`](docs/E5.md),
[`docs/E5_IMPLEMENTATION_GUIDE.md`](docs/E5_IMPLEMENTATION_GUIDE.md), and
[`docs/E5_Summary.md`](docs/E5_Summary.md)).

## Setup

Install once from the repository root (the requirements file stays at the root):

```bash
conda create -n sinks python=3.11 -y
conda activate sinks
pip install -r requirements.txt
```

Then `cd` into the folder for the experiment you want to run.

---

## 1. Reproduce the original paper

```bash
cd reproduce_paper
```

All outputs are written under `reproduce_paper/results/`.

### Figure 1 — Source-Agnostic Shift Histogram (truncated) + Appendix Full Histogram

```bash
python experiments_statistical.py --mode bias-term --output-dir results
```

Outputs:
- `results/bias_term_statistical/bq_k_aggregate_plot_truncated.png` → **Fig 1**
- `results/bias_term_statistical/bq_k_aggregate_plot.png` → **Fig 6** (appendix, full histogram)

### Figure 2 — EPE-Bias Projection Alignment

```bash
python ../common/experiments_single_input.py --mode epe-bias-proj --output-dir results
```

Output:
- `results/epe_bias_proj/epe_alignment.png` → **Fig 2**

### Figure 3 — EPE Captures the Net Positional Contribution

```bash
python experiments_statistical.py --mode epe-validation --output-dir results
```

Outputs:
- `results/epe_validation_statistical/epe_validation_plot.png` → **Fig 3**
- `results/epe_validation_statistical/epe_validation_precentiles` (numerical values for experiments)

### Figure 4 — Coordinate-Level Alignment Histogram (truncated) + Appendix Full Histogram

```bash
python experiments_statistical.py --mode coord-alignment --output-dir results
```

Outputs:
- `results/coord_alignment_statistical/coord_alignment_histogram_truncated.png` → **Fig 4**
- `results/coord_alignment_statistical/coord_alignment_histogram.png` → **Fig 8** (appendix, full histogram)

### Figure 5 — Intervention Attention Maps

```bash
python ../common/intervention_analysis.py --mode sentence --output-dir results
```

Outputs:
- `results/sentence_analysis/layer_04_avg.png` through `layer_11_avg.png` → **Fig 5** (layers 4--11)

### Table 1 — BOS Attention Statistics

```bash
python ../common/intervention_analysis.py --mode dataset --output-dir results
```

Outputs:
- `results/dataset_analysis/bos_attention_summary_mid_layers.txt` → **Table 1**
- `results/dataset_analysis/bos_attention_summary_mid_layers.csv`

### Figure 7 (appendix) — Massive Activations in EPE_1

```bash
python ../common/experiments_single_input.py --mode massive-activations --output-dir results
```

Output:
- `results/massive_activations/massive_activations_in_ppe.png` → **Fig 7**

The forward-dependent Figure 1 and Figure 3 statistical paths also accept
`--engine {manual,nnsight}` (default `manual`). Figure 2, Figure 4/8, and Figure 7 are
pure weight-space calculations; Figure 5 and Table 1 already use the shared E1/E2 engine.
See the [E3/E4/E5 NNsight implementation guide](docs/NNsight_E3_E4_E5_IMPLEMENTATION.md#8-reproduction-entry-audit).

---

## 2. Cross-scale + cross-architecture (all models)

```bash
cd cross_scale_and_architecture
```

`run_table1_multiseed.py` runs the Table 1 dataset analysis across seeds `0,1,2` and
aggregates the results. It drives the per-architecture harness in each model subfolder
(`gpt2/`, `opt/`, `neo/`, `qwen/`) via `--architecture`. Each command below writes to its
own model-named directory under `results/` (`--experiment-name`), so runs never overwrite
each other:

- Seed outputs: `results/<experiment-name>/seed_*/`
- Combined CSVs and scatter plots: `results/<experiment-name>/aggregate/`

`--dtype` defaults to `float32` (matching the paper's standard errors); drop to `bfloat16`
only where noted below to fit the larger checkpoints in memory. Use
`--layer-mode scaled` (the default) to scale the mid-layer band with model depth, or
`--layer-mode fixed` to keep layers 4--11 for direct same-layer comparisons.

Intervention (i) identifies the massive effective-positional-embedding coordinates for
each model instead of assuming GPT-2 small's fixed top three. The random control zeros the
same number of coordinates, and each dataset run records the selected coordinates and model
geometry in `run_config.json`.

#### Execution engines (`--engine`)

Every command in this section accepts `--engine {manual,nnsight}`.

| | |
|---|---|
| `manual` (default) | Re-implements the transformer forward pass by hand, reading raw weight tensors off the modules. This is what produced the numbers in [`docs/E1-E2_Summary.md`](docs/E1-E2_Summary.md); leave it alone to reproduce them. |
| `nnsight` | Runs the **real** HuggingFace forward under [NNsight](https://nnsight.net/), applying all ten interventions as activation edits inside `.trace()` and reading attention probabilities from the model itself. |

Under `--engine nnsight` the model's own attention, RoPE and GQA are used instead of
re-implementations, and each run records the engine plus the `nnsight`/`transformers`/
`torch` versions in `run_config.json`. NNsight runs write to a separate
`..._nnsight` directory, so the two engines never overwrite each other.

`--verify-parity` cross-checks the two engines per intervention and writes
`parity_report.json`. On real weights (fp32) the BOS metric agrees to ~1e-7:

```bash
python ../common/intervention_analysis.py     --model-name gpt2                    --verify-parity
python opt/intervention_analysis_opt.py       --model-name facebook/opt-125m       --verify-parity
python neo/intervention_analysis_neo.py       --model-name EleutherAI/gpt-neo-125m --verify-parity
python qwen/intervention_analysis_qwen.py     --model-name Qwen/Qwen2.5-0.5B       --verify-parity
```

Parity gates on the **BOS metric** (the number that reaches the CSV), not on raw
attention probabilities, and only in fp32 — in fp16/bf16 the two paths are genuinely
different algorithms (HF upcasts q/k and/or softmax to fp32; the manual path does not),
so the gate downgrades to advisory. See [`docs/E1.md`](docs/E1.md) §7 for what parity
found.

`--remote` runs the NNsight engine on [NDIF](https://ndif.us/) instead of locally. It
needs an NDIF API key and only works for checkpoints NDIF hosts — **none of the 16 models
below are currently hosted**, so this is an escape hatch, not a reproduction path.

### GPT-2 (`--architecture gpt2`)

```bash
python run_table1_multiseed.py --architecture gpt2 --model gpt2         --seeds 0,1,2 --experiment-name table1_multiseed_gpt2
python run_table1_multiseed.py --architecture gpt2 --model gpt2-medium  --seeds 0,1,2 --experiment-name table1_multiseed_gpt2-medium
python run_table1_multiseed.py --architecture gpt2 --model gpt2-large   --seeds 0,1,2 --experiment-name table1_multiseed_gpt2-large
```

### OPT (`--architecture opt`)

```bash
python run_table1_multiseed.py --architecture opt --model facebook/opt-125m  --seeds 0,1,2 --experiment-name table1_multiseed_opt-125m
python run_table1_multiseed.py --architecture opt --model facebook/opt-1.3b  --seeds 0,1,2 --experiment-name table1_multiseed_opt-1.3b
python run_table1_multiseed.py --architecture opt --model facebook/opt-2.7b  --seeds 0,1,2 --dtype bfloat16 --experiment-name table1_multiseed_opt-2.7b
python run_table1_multiseed.py --architecture opt --model facebook/opt-6.7b  --seeds 0,1,2 --dtype bfloat16 --experiment-name table1_multiseed_opt-6.7b
python run_table1_multiseed.py --architecture opt --model facebook/opt-13b   --seeds 0,1,2 --dtype bfloat16 --experiment-name table1_multiseed_opt-13b
```

### GPT-Neo (`--architecture neo`)

```bash
python run_table1_multiseed.py --architecture neo --model EleutherAI/gpt-neo-125m --seeds 0,1,2 --experiment-name table1_multiseed_gpt-neo-125m
python run_table1_multiseed.py --architecture neo --model EleutherAI/gpt-neo-1.3B --seeds 0,1,2 --experiment-name table1_multiseed_gpt-neo-1.3B
python run_table1_multiseed.py --architecture neo --model EleutherAI/gpt-neo-2.7B --seeds 0,1,2 --experiment-name table1_multiseed_gpt-neo-2.7B
```

### Qwen2.5 (`--architecture qwen`)

```bash
python run_table1_multiseed.py --architecture qwen --model Qwen/Qwen2.5-0.5B --seeds 0,1,2 --experiment-name table1_multiseed_qwen2.5-0.5B
python run_table1_multiseed.py --architecture qwen --model Qwen/Qwen2.5-1.5B --seeds 0,1,2 --experiment-name table1_multiseed_qwen2.5-1.5B
python run_table1_multiseed.py --architecture qwen --model Qwen/Qwen2.5-3B   --seeds 0,1,2 --experiment-name table1_multiseed_qwen2.5-3B
python run_table1_multiseed.py --architecture qwen --model Qwen/Qwen2.5-7B   --seeds 0,1,2 --dtype bfloat16 --experiment-name table1_multiseed_qwen2.5-7B
python run_table1_multiseed.py --architecture qwen --model Qwen/Qwen2.5-14B  --seeds 0,1,2 --dtype bfloat16 --experiment-name table1_multiseed_qwen2.5-14B
```

---

## 3. E3 — Emergence dynamics

```bash
cd emergence_dynamics
```

`emergence_dynamics_analysis.py` tracks how the sink circuit emerges over training
checkpoints of a run family (default `gpt2-small`, the Stanford-CRFM Mistral runs). See
[`docs/E3.md`](docs/E3.md) for the protocol and caveats.

```bash
python emergence_dynamics_analysis.py --mode all --run-family gpt2-small --output-dir results/e3_emergence_gpt2-small
```

Real Hugging Face forwards through NNsight:

```bash
python emergence_dynamics_analysis.py --mode all --engine nnsight --run-family gpt2-small --skip-existing --purge-cache --output-dir results/e3_emergence_gpt2-small_nnsight
```

E3's random seed is the single `--data-seed` flag (the data-resample seed), not the
multi-seed `--seeds` used elsewhere. To sweep seeds `0,1,2`, run it three times into
separate output directories:

```bash
python emergence_dynamics_analysis.py --mode all --run-family gpt2-small --data-seed 0 --output-dir results/e3_emergence_gpt2-small_seed0
python emergence_dynamics_analysis.py --mode all --run-family gpt2-small --data-seed 1 --output-dir results/e3_emergence_gpt2-small_seed1
python emergence_dynamics_analysis.py --mode all --run-family gpt2-small --data-seed 2 --output-dir results/e3_emergence_gpt2-small_seed2
```

Useful options: `--list-checkpoints` (preview the revisions), `--plot-only` (rebuild figures
from cached per-checkpoint outputs), `--purge-cache` (delete each revision's weights after
use to save disk).

---

## 4. E4 — Anatomy of the residual sink

```bash
cd residual_sink
```

`residual_sink_analysis.py` dissects the *residual* sink the paper leaves unexplained
(§5.3), running all five core analyses (dose-response, decomposition, combined, surgical,
relocation) over seeds `0,1,2` by default. See [`docs/E4_Plan.md`](docs/E4_Plan.md) for the
full account.

```bash
python ../common/residual_sink_analysis.py --mode all --model-name gpt2 --seeds 0,1,2 --output-dir results/e4_residual_sink_gpt2
```

```bash
python ../common/residual_sink_analysis.py --mode all --engine nnsight --model-name gpt2 --seeds 0,1,2 --with-perplexity --output-dir results/e4_residual_sink_gpt2_nnsight
```

Add `--with-perplexity` to include the optional functional-cost (LM cross-entropy) table.
Every figure is regenerable from cached per-seed outputs with `--plot-only`.

---

## 5. E5 — Evaluation robustness & functional cost

```bash
cd evaluation_robustness
```

`evaluation_robustness_analysis.py` audits whether the Table 1 mechanism is robust to
alternative sink metrics, context lengths, and synthetic/multilingual content, and measures
the LM cross-entropy cost of sink removal (seeds `0,1,2`; fp32 by default). See
[`docs/E5.md`](docs/E5.md) for the protocol and
[`docs/E5_IMPLEMENTATION_GUIDE.md`](docs/E5_IMPLEMENTATION_GUIDE.md) for setup and output
interpretation.

```bash
python evaluation_robustness_analysis.py --mode all --model-name gpt2 --seeds 0,1,2 --output-dir results/e5_eval_robustness_gpt2
```

```bash
python evaluation_robustness_analysis.py --mode all --engine nnsight --model-name gpt2 --seeds 0,1,2 --output-dir results/e5_eval_robustness_gpt2_nnsight
```

Rebuild every aggregate table and figure without loading the model/data with `--plot-only`.
The legacy `--smoke-test` is the manual random-model check; the real NNsight offline checks
use the local-model scripts below. `--verify-parity` compares manual execution with NNsight.

For the offline local-model smoke suite, real-model parity commands, cache/provenance rules,
and implementation details, see
[`docs/NNsight_E3_E4_E5_IMPLEMENTATION.md`](docs/NNsight_E3_E4_E5_IMPLEMENTATION.md).
On Windows, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nnsight_rest_smoke_tests.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_nnsight_rest_parity.ps1
```
