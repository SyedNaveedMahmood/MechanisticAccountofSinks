Code for reproducing the experiments in the paper "A Mechanistic Account of Attention Sinks in GPT-2: One Circuit, Broader Implications for Mitigation".

## Setup

```bash
conda create -n sinks python=3.11 -y
conda activate sinks
pip install -r requirements.txt
```

## Reproducing the Figures and Tables

### Figure 1 — Source-Agnostic Shift Histogram (truncated) + Appendix Full Histogram

```bash
python experiments_statistical.py --mode bias-term --output-dir results
```

Outputs:
- `results/bias_term_statistical/bq_k_aggregate_plot_truncated.png` → **Fig 1**
- `results/bias_term_statistical/bq_k_aggregate_plot.png` → **Fig 6** (appendix, full histogram)

### Figure 2 — EPE-Bias Projection Alignment

```bash
python experiments_single_input.py --mode epe-bias-proj --output-dir results
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
python intervention_analysis.py --mode sentence --output-dir results
```

Outputs:
- `results/sentence_analysis/layer_04_avg.png` through `layer_11_avg.png` → **Fig 5** (layers 4--11)

### Table 1 — BOS Attention Statistics

```bash
python intervention_analysis.py --mode dataset --output-dir results
```

Outputs:
- `results/dataset_analysis/bos_attention_summary_mid_layers.txt` → **Table 1**
- `results/dataset_analysis/bos_attention_summary_mid_layers.csv`

### Table 1 (OPT) — Cross-Architecture Replication

`intervention_analysis_opt.py` reruns the exact Table 1 intervention suite on
Meta's OPT models. OPT is the natural cross-architecture test because it keeps the
two ingredients the sink circuit depends on: a learned **query bias** (`q_proj.bias`)
and **learned absolute positional embeddings** (`embed_positions`). `facebook/opt-125m`
is a structural twin of GPT-2 small (12 layers, 12 heads, hidden 768, pre-LayerNorm),
so the metric and layer range (4–11) transfer directly.

```bash
# opt-125m (default) — GPT-2-small twin
python intervention_analysis_opt.py --mode dataset --model-name facebook/opt-125m --output-dir results

# Larger OPT sizes (all pre-LayerNorm)
python intervention_analysis_opt.py --mode dataset --model-name facebook/opt-1.3b --output-dir results
python intervention_analysis_opt.py --mode dataset --model-name facebook/opt-2.7b --output-dir results
```

Outputs:
- `results/dataset_analysis_opt/bos_attention_summary_mid_layers.txt` → **Table 1 (OPT)**
- `results/dataset_analysis_opt/bos_attention_summary_mid_layers.csv`
- `results/dataset_analysis_opt/bos_attention_stats_{by_dataset,overall}.csv`

Multiseed OPT Table 1 runs for seeds `0,1,2`:

```bash
# opt-125m (default)
python run_table1_multiseed.py --architecture opt --model facebook/opt-125m

# opt-1.3b
python run_table1_multiseed.py --architecture opt --model facebook/opt-1.3b

# opt-2.7b
python run_table1_multiseed.py --architecture opt --model facebook/opt-2.7b
```

Each seed writes to a separate directory under
`results/table1_multiseed_opt_<model>/seed_*/dataset_analysis_opt/`. Combined
CSVs and scatter plots are written under that run's `aggregate/` directory.

Sentence-mode heatmap grids (the OPT analog of **Fig 5**):

```bash
python intervention_analysis_opt.py --mode sentence --model-name facebook/opt-125m --output-dir results
# → results/sentence_analysis_opt/layer_*_avg.png
```

> **Note:** `facebook/opt-350m` uses post-LayerNorm and projects the word-embedding
> dimension, so it is not directly comparable to the pre-LayerNorm circuit and is
> rejected at load time. Use `opt-125m`, `opt-1.3b`, or `opt-2.7b`.

### Table 1 (GPT-Neo) — Cross-Architecture Replication

`intervention_analysis_neo.py` reruns the exact Table 1 intervention suite on
EleutherAI's GPT-Neo models. GPT-Neo keeps the **learned absolute positional
embeddings** the sink circuit relies on (a `wpe` table, exactly like GPT-2) but
**drops the learned query bias** (`q_proj` is `bias=False`) and computes attention
**without** the `1/sqrt(d)` scaling. `EleutherAI/gpt-neo-125m` is a structural twin
of GPT-2 small (12 layers, 12 heads, hidden 768, pre-LayerNorm), so the metric and
layer range (4–11) transfer directly.

```bash
# gpt-neo-125m (default) — GPT-2-small twin
python intervention_analysis_neo.py --mode dataset --model-name EleutherAI/gpt-neo-125m --output-dir results

# Larger GPT-Neo sizes
python intervention_analysis_neo.py --mode dataset --model-name EleutherAI/gpt-neo-1.3B --output-dir results
python intervention_analysis_neo.py --mode dataset --model-name EleutherAI/gpt-neo-2.7B --output-dir results
```

Outputs:
- `results/dataset_analysis_neo/bos_attention_summary_mid_layers.txt` → **Table 1 (GPT-Neo)**
- `results/dataset_analysis_neo/bos_attention_summary_mid_layers.csv`
- `results/dataset_analysis_neo/bos_attention_stats_{by_dataset,overall}.csv`

Multiseed GPT-Neo Table 1 runs for seeds `0,1,2`:

```bash
python run_table1_multiseed.py --architecture neo --model EleutherAI/gpt-neo-125m --seeds 0,1,2
python run_table1_multiseed.py --architecture neo --model EleutherAI/gpt-neo-1.3B --seeds 0,1,2
python run_table1_multiseed.py --architecture neo --model EleutherAI/gpt-neo-2.7B --seeds 0,1,2
```

Each seed writes to a separate directory under
`results/table1_multiseed_neo_<model>/seed_*/dataset_analysis_neo/`. Combined
CSVs and scatter plots are written under that run's `aggregate/` directory.

Sentence-mode heatmap grids (the GPT-Neo analog of **Fig 5**):

```bash
python intervention_analysis_neo.py --mode sentence --model-name EleutherAI/gpt-neo-125m --output-dir results
# → results/sentence_analysis_neo/layer_*_avg.png
```

> **Note:** GPT-Neo has **no query bias**, so intervention **(b) No Query Bias** is a
> structural no-op and matches the baseline **(a)**; GPT-Neo does keep learned absolute
> position embeddings, so (c)–(e), (h), and (i) remain meaningful. The fixed mid range
> (layers 4–11) is calibrated for the 12-layer `gpt-neo-125m`; `gpt-neo-1.3B` (24 layers)
> and `gpt-neo-2.7B` (32 layers) reuse it for comparability, and the `all_layers` summary
> is written alongside for the full-depth view.

### Table 1 (Qwen2.5) — RoPE-Family Replication

`intervention_analysis_qwen.py` reruns the Table 1 intervention suite on Alibaba's
Qwen2.5 models. Qwen2.5 is a useful contrast to GPT-2/OPT because it keeps the
paper's central ingredient — a learned **query bias** (`q_proj.bias`,
`attention_bias=True`, which Llama/Mistral drop) — but delivers positional
information through **Rotary Position Embeddings (RoPE)** rather than learned
additive absolute embeddings. Because nothing positional is added to the residual
stream under RoPE, there is no separable *effective positional embedding* (EPE), so
the positional interventions are reframed as RoPE **position-id** manipulations:

- **(c) Remove First PE** → token 0 is rotated as if at position 1 (`pos_ids = [1,1,2,3,…]`).
- **(d) Swap EPE / (e) Swap PE** → swap the RoPE positions of tokens 0 and 1
  (`pos_ids = [1,0,2,3,…]`). Under RoPE the raw-PE vs effective-PE distinction
  collapses, so (d) and (e) coincide; both rows are kept for column alignment.
- **(h) No PE** → RoPE disabled (`pos_ids = 0` everywhere → identity rotation).
- **(i) Zero Top-3 Wk** → zeros the Wk columns matching the top-3 magnitude dims of
  the position-0 token embedding (the RoPE-model stand-in for the top-|EPE[0]| dims).

The harness also handles Qwen2.5's grouped-query attention (`repeat_kv`), RMSNorm,
and SwiGLU FFN. The metric and layer range (4–11) are unchanged; every listed
checkpoint has ≥24 layers.

```bash
python intervention_analysis_qwen.py --mode dataset --model-name Qwen/Qwen2.5-0.5B --output-dir results
python intervention_analysis_qwen.py --mode dataset --model-name Qwen/Qwen2.5-1.5B --output-dir results
python intervention_analysis_qwen.py --mode dataset --model-name Qwen/Qwen2.5-3B   --output-dir results

# Larger checkpoints usually need reduced precision to fit in memory
python intervention_analysis_qwen.py --mode dataset --model-name Qwen/Qwen2.5-7B  --dtype bfloat16 --output-dir results
python intervention_analysis_qwen.py --mode dataset --model-name Qwen/Qwen2.5-14B --dtype bfloat16 --output-dir results
```

Outputs:
- `results/dataset_analysis_qwen/bos_attention_summary_mid_layers.txt` → **Table 1 (Qwen2.5)**
- `results/dataset_analysis_qwen/bos_attention_summary_mid_layers.csv`
- `results/dataset_analysis_qwen/bos_attention_stats_{by_dataset,overall}.csv`

Multiseed Qwen2.5 Table 1 runs for seeds `0,1,2` (via the shared `--architecture` dispatch):

```bash
python run_table1_multiseed.py --architecture qwen --model Qwen/Qwen2.5-0.5B --seeds 0,1,2
python run_table1_multiseed.py --architecture qwen --model Qwen/Qwen2.5-1.5B --seeds 0,1,2
python run_table1_multiseed.py --architecture qwen --model Qwen/Qwen2.5-3B   --seeds 0,1,2
python run_table1_multiseed.py --architecture qwen --model Qwen/Qwen2.5-7B  --dtype bfloat16 --seeds 0,1,2
python run_table1_multiseed.py --architecture qwen --model Qwen/Qwen2.5-14B --dtype bfloat16 --seeds 0,1,2
```

Each seed writes to a separate directory under
`results/table1_multiseed_qwen_<model>/seed_*/dataset_analysis_qwen/`. Combined
CSVs and scatter plots are written under that run's `aggregate/` directory.

Sentence-mode heatmap grids (the Qwen2.5 analog of **Fig 5**):

```bash
python intervention_analysis_qwen.py --mode sentence --model-name Qwen/Qwen2.5-0.5B --output-dir results
# → results/sentence_analysis_qwen/layer_*_avg.png
```

> **Note:** the harness targets Qwen2/Qwen2.5 checkpoints that keep the learned
> query bias. A checkpoint whose `q_proj.bias` is `None` is rejected at load time,
> since without it the (b) *No Query Bias* intervention is a no-op.

### Figure 7 (appendix) — Massive Activations in EPE_1

```bash
python experiments_single_input.py --mode massive-activations --output-dir results
```

Output:
- `results/massive_activations/massive_activations_in_ppe.png` → **Fig 7**
