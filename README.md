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

## Extension — Anatomy of the Residual Sink (E4)

`residual_sink_analysis.py` extends the Table 1 story to the *residual* sink the
paper leaves unexplained (§5.3): nullifying `b_Q` still leaves **44.7%** of the sink,
and zeroing the top-3 `W_k` columns leaves **65.2%**. The paper's pre-softmax score
decomposes as

```
s_{i→j} = x_i W_q W_k^T x_j^T (T1, content)  +  x_i W_q b_K^T (T2)
        + b_Q W_k^T x_j^T (T3 = Δ_j, source-agnostic shift)  +  b_Q b_K^T (T4),
```

and because softmax-over-targets cancels the target-constant terms (T2, T4), the
attention distribution is driven by **T1 + T3** only. Nullifying `b_Q` zeros T3, so the
residual is the **content term T1 routed through `k_1 ≈ EPE_1 W_k`** — a *second*
positional pathway that shares the massive-activation key structure but is driven by
downstream **query** alignment rather than the query bias. This harness tests that
account with graded, decomposed, and combined causal analyses. It reuses the exact
Table 1 forward-pass machinery (`intervention_analysis.py`) and runs over seeds `0,1,2`
by default (data resamples), writing per-seed dirs plus a cross-seed `aggregate/`.

```bash
# Run all five core analyses (dose-response, decomposition, combined, surgical, relocation)
python residual_sink_analysis.py --mode all --model-name gpt2 --output-dir results

# Include the optional functional-cost (LM cross-entropy) table
python residual_sink_analysis.py --mode all --with-perplexity --output-dir results

# A single analysis, or a quick check on fewer examples
python residual_sink_analysis.py --mode dose_response --sample-size 20 --seeds 0
```

Modes (each maps to a paper-ready deliverable):

- **`dose_response` (E4.1)** — BOS-attention vs. scale `α ∈ {0,…,1.5}` on four knobs:
  `scale_bq` (`b_Q`, pathway A — floors at the residual, ≈44% on gpt2); `scale_pe`
  (*delete* `p_1` — empirically the sink **survives** deletion, ≈85–110% at `α=0`: a
  deletion-invariance finding); `interp_pe` (*replace* positional identity,
  `pe[0] = α·p_1 + (1−α)·p_2` — grades Remove-First-PE, both pathways collapse to the
  ≈3% floor at `α=0`); and `scale_wk_massive` (the top-k massive `W_k` columns, graded in
  every layer; k = number of identified massive coords, recorded in `run_config.json`).
  Each knob acts at an **effective locus** — the input (`p_1`) or every layer (`W_k`) —
  because a layer-0-only edit is erased by the intermediate layers before the metric
  window (paper fn. 10). → `aggregate/dose_response.png`, `dose_response_monotonicity.csv`.
- **`decomposition` (E4.2)** — exact T1-vs-T3 attribution of the position-1 advantage
  (with an identity assert `T1+T2+T3+T4 == score`), plus the query–EPE_1 alignment
  histogram (the Fig. 2 analog for downstream queries).
  → `aggregate/query_alignment_hist.png`, `delta_share_heatmap.png`, `decomposition_summary.csv`.
- **`combined` (E4.3)** — stacked interventions (`b∧c`, `b∧i`, `b∧d`, `b∧i∧d`) that
  localize the residual. → `aggregate/combined_summary.csv`.
- **`surgical` (E4.4)** — first-layer-only MLP skip and position-1-only PE zeroing vs.
  the paper's coarse all-layer / all-position ablations. → `aggregate/surgical_summary.csv`.
- **`relocation` (E4.5)** — attention to position 2 (the Swap-EPE transplant target):
  does the sink *move* or merely vanish? → `aggregate/relocation_summary.csv`.
- **`perplexity` (E4.6, optional)** — LM cross-entropy cost of each pathway ablation.

> **Note:** all runs are fp32 to match the paper's tight SEs, and every figure is
> regenerable from cached per-seed outputs via `--plot-only`. The massive-activation
> coordinates (paper: 138, 378, 447) are re-identified per model at load time.

## Extension — Evaluation Robustness & Functional Cost (E5)

`evaluation_robustness_analysis.py` audits whether the Table 1 mechanism is robust
to alternative sink metrics, context lengths through 1024 tokens, and synthetic or
multilingual content, then measures the language-model cross-entropy cost of sink
removal. One declarative intervention registry drives attention maps, metrics, CE,
length tests, content controls, and parity checks, including all ten Table 1 rows,
the four E4 combinations, and graded `b_Q`/PE dose responses.

```bash
# Complete GPT-2-small E5 suite (seeds 0,1,2; fp32 by default)
python evaluation_robustness_analysis.py --mode all --model-name gpt2 --output-dir results

# Rebuild every aggregate table and E5-A...F figure without model/dataset loading
python evaluation_robustness_analysis.py --mode all --model-name gpt2 --plot-only --output-dir results

# Offline random-model smoke test; real-model parity is a separate explicit check
python evaluation_robustness_analysis.py --smoke-test --output-dir results/_e5_smoke
python evaluation_robustness_analysis.py --verify-parity --model-name gpt2 --output-dir results/_e5_parity
```

Length comparisons use deterministic within-domain concatenation and paired nested
token-ID prefixes, so decoding cannot alter lengths and independently resampled data
cannot confound scaling. The random-`W_k` control uses fixed coordinates selected by
`--random-wk-seed` and records them in `run_config.json`. Optional Bangla/Chinese
FLORES data is loaded only with `--with-multilingual` and skips gracefully when
unavailable. See [`.md/E5.md`](.md/E5.md) for metric definitions, cache layout,
four-GPU commands, runtime/RAM guidance, and the verification protocol.

## Cross-scale runs (all harnesses)

Every intervention harness — GPT-2 (`intervention_analysis.py`), OPT
(`intervention_analysis_opt.py`), GPT-Neo (`intervention_analysis_neo.py`),
Qwen2.5 (`intervention_analysis_qwen.py`), and the residual-sink E4 harness
(`residual_sink_analysis.py`) — accepts two flags for running across model scales:

- `--layer-mode {scaled,fixed}` (default `scaled`) — the mid-layer band of the
  BOS metric. `scaled` excludes the first 3 and the last layer, reducing to exactly
  layers 4–11 for a 12-layer model (so gpt2 / opt-125m / gpt-neo-125m are unchanged
  and match the paper) and extending proportionally for deeper checkpoints; `fixed`
  forces layers 4–11 on every size for strict same-layer comparability. The full-depth
  `all_layers` summary is still written alongside.
- `--dtype {float32,float16,bfloat16}` (default `float32`, matching the paper's SEs;
  Qwen also accepts `auto`) — drop to a smaller dtype only if VRAM-constrained on a
  large model (e.g. gpt2-xl, opt-2.7b).

`run_table1_multiseed.py` forwards both flags to whichever harness it drives. Examples:

```bash
python intervention_analysis.py --mode dataset --model-name gpt2-large --output-dir results
python intervention_analysis_opt.py --mode dataset --model-name facebook/opt-2.7b --dtype float16 --output-dir results
python residual_sink_analysis.py --mode all --model-name gpt2-xl --output-dir results
python run_table1_multiseed.py --architecture neo --model EleutherAI/gpt-neo-1.3B --layer-mode scaled
```
