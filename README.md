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

### Figure 7 (appendix) — Massive Activations in EPE_1

```bash
python experiments_single_input.py --mode massive-activations --output-dir results
```

Output:
- `results/massive_activations/massive_activations_in_ppe.png` → **Fig 7**
