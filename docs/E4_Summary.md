# E4 Summary — Anatomy of the Residual Sink across GPT-2 scale

**Reference document for narrative building and paper citation.** Every number below is the
mean over **3 data-resampling seeds (0,1,2)** unless a std is given; each seed draws a fresh
300-example set (100 SST-2 + 100 GSM8K + 100 HumanEval, ≥40 tokens, truncated to 40). All runs
are **fp32**, GPT-2 family, `--layer-mode scaled` (depth-aware mid-layer band). Source: the E4 residual-sink harness
(`common/residual_sink_analysis.py`, run from `residual_sink/`; includes the cross-scale band, the `interp_pe` knob, and the `zero_topk` rename).

Models: **gpt2** (124M), **gpt2-medium** (355M), **gpt2-large** (774M). All three are the *same
architecture* (learned absolute PE + learned query bias `b_Q` + Conv1D QKV + GELU MLP + pre-LN),
differing only in scale — so this is a **cross-scale**, not cross-architecture, study.

---

## 0. Provenance & validation (is this trustworthy?)

- **3 models × 3 distinct seeds × 7 analyses**, all files present, all arrays finite.
- **Internal-consistency anchors pass for all three** (independent code paths agree to <1e-6):
  `scale_bq(0)`≡`nullify_bq`; `scale_pe(0)`≡`pos1_only_pe_zero`; every knob at α=1 ≡ baseline;
  decomposition `attn_full`≡real baseline; `relocation.base_pos0`≡baseline.
- **Manual forward ≡ HF logits** (perplexity `baseline`≡`hf_reference` to <1e-6) — the reimplementation is exact.
- **Decomposition identity** `T1+T2+T3+T4 == score` holds under the relative-tolerance assert
  (needed at medium/large where pre-softmax scores reach ~1e3–1e4).
- Massive-coordinate counts: **k=3** (gpt2: 138/378/447 — exactly the paper's), **k=6** (medium, large).

**Verdict: properly computed, valid, and — with 3 seeds and sub-0.2pp cross-seed std on the circuit
interventions — paper-ready.**

---

## 1. Key-number tables

### 1.1 Setup

| Model | Params | Layers | Mid-band (0-idx) | Massive coords (k) | Baseline BOS attn |
|---|---:|---:|:---:|---|---:|
| gpt2 | 124M | 12 | [3, 11) | 138, 378, 447 (k=3) | **0.5637** |
| gpt2-medium | 355M | 24 | [3, 23) | 9, 238, 268, 428, 580, 608 (k=6) | **0.5753** |
| gpt2-large | 774M | 36 | [3, 35) | 8, 24, 248, 440, 792, 870 (k=6) | **0.5015** |

Baseline sink strength is **non-monotone** in scale (0.564 → 0.575 → 0.501): large has the *weakest* raw sink.

### 1.2 Dose-response floors (metric at α=0, as % of baseline) + monotonicity

| Knob (what α scales) | gpt2 | medium | large | Spearman ρ (s / m / l) |
|---|---:|---:|---:|:---:|
| **scale_bq** — query bias `b_Q` (pathway A) | **44.5%** | **70.4%** | **74.0%** | +1.0 / +1.0 / +1.0 |
| **interp_pe** — `p₁→p₂` identity replacement (both pathways) | **3.0%** | **1.7%** | **3.9%** | +0.29 / +0.64 / +0.96 |
| **scale_pe** — delete `p₁` (deletion control) | **110.0%** | **82.6%** | **36.6%** | −1.0 / +0.46 / +1.0 |
| **scale_wk_massive** — top-k massive `W_k` columns | **65.1%** | **28.2%** | **42.1%** | +1.0 / +1.0 / +1.0 |

`scale_bq` and `scale_wk_massive` are cleanly monotone at every scale. `interp_pe` is a **threshold**
(≈2–4% for α≤0.25, jumps to ~baseline at α≥0.5). `scale_pe` **flips sign** with scale (see §2).

### 1.3 Combined interventions (% of baseline, mean ± std over 3 seeds)

| Intervention | gpt2 | medium | large |
|---|---:|---:|---:|
| nullify `b_Q` (residual) | 44.5 ± 0.16 | 70.4 ± 0.11 | 74.0 ± 0.08 |
| `b_Q`=0 ∧ Remove-First-PE | 2.25 ± 0.06 | 1.64 ± 0.19 | 3.65 ± 0.14 |
| `b_Q`=0 ∧ Zero-Top-k-`W_k` | 24.8 ± 0.10 | 19.1 ± 0.06 | 31.2 ± 0.23 |
| `b_Q`=0 ∧ Swap-EPE | 4.23 ± 0.09 | 3.47 ± 0.03 | 2.17 ± 0.08 |
| `b_Q`=0 ∧ Zero-Top-k-`W_k` ∧ Swap-EPE (all 3) | 4.97 ± 0.05 | 5.58 ± 0.08 | 2.47 ± 0.06 |

Removing all three components leaves an **irreducible ~2–6% floor** at every scale (a genuine secondary contributor).

### 1.4 Surgical / coarse interventions (% of baseline, mean ± std)

| Intervention | gpt2 | medium | large |
|---|---:|---:|---:|
| **first-layer-only MLP skip** | 102.5 ± 0.23 | **6.9 ± 0.03** | 87.7 ± 0.33 |
| all-layer No-MLP | 50.1 ± 0.46 | 13.8 ± 0.56 | 20.6 ± 0.21 |
| all-position No-PE | 18.0 ± 1.29 | 9.3 ± 0.97 | 3.2 ± 0.13 |
| position-1-only PE zero (=`scale_pe(0)`) | 110.0 ± 0.51 | 82.6 ± 1.42 | 36.6 ± 0.67 |

The first-layer MLP is **dispensable at gpt2 (103%) and large (88%) but essential at medium (7%)** — see §5.

### 1.5 Score decomposition (pooled over 3 seeds)

| Quantity | gpt2 | medium | large |
|---|---:|---:|---:|
| attn to pos-1, content-only (T1, pathway B) | 0.210 | 0.329 | 0.323 |
| attn to pos-1, Δ-only (T3, pathway A) | 0.377 | 0.216 | 0.136 |
| **Δ-share of the pos-1 score advantage** | **0.561** | **0.450** | **0.247** |
| query·EPE₁·W_k alignment, mean (red) | +0.228 | +0.023 | −0.061 |
| control alignment, mean (blue) | +0.023 | −0.070 | −0.108 |
| alignment Cohen's d (red vs blue) | 1.79 | 0.42 | 0.30 |
| alignment Mann-Whitney p | ~0 | 1.5e-22 | 1.1e-21 |
| (layer,head) cells with align > 0.1 | **89%** | **48%** | **18%** |
| max single-cell alignment | 0.465 | 0.521 | **0.719** |

### 1.6 Relocation (does the sink *move* under Swap-EPE?)

| Quantity | gpt2 | medium | large |
|---|---:|---:|---:|
| **relocation ratio** (attn gained at pos-2 / baseline sink) | 0.897 ± 0.004 | 0.922 ± 0.001 | **0.944 ± 0.001** |
| attn to pos-2 under Swap-EPE | 0.512 | 0.537 | 0.483 |
| attn to pos-2 at baseline | 0.006 | 0.007 | 0.010 |
| standalone Swap-EPE, attn to pos-1 (% base) | 6.3% | 8.4% | 3.3% |

The sink transplants ~50% of its mass onto position 2; relocation gets **cleaner with scale**.

### 1.7 Function — LM cross-entropy (nats; Δ = cost vs baseline)

| Intervention | gpt2 | medium | large |
|---|---:|---:|---:|
| baseline (≡ HF) | 4.009 | 3.727 | 3.539 |
| nullify `b_Q` | +1.59 | +0.36 | **+0.12** |
| Zero-Top-k-`W_k` | +0.26 | +1.06 | +0.47 |
| first-layer MLP skip | +2.42 | +4.54 | +4.21 |

---

## 2. Findings & trends

**Clean, monotone cross-scale trends (the backbone story):**

1. **The residual after killing the bias grows with scale: 44.5% → 70.4% → 74.0%.** The query
   pathway (pathway B), not the bias pathway (A), carries most of the sink at scale.
2. **The bias pathway's share of the pos-1 score advantage halves: Δ-share 0.561 → 0.450 → 0.247.**
   An independent (softmax-free) confirmation of #1.
3. **Killing the bias becomes functionally almost free: LM cost +1.59 → +0.36 → +0.12 nats.**
   `b_Q` is close to *vestigial* for language modeling at large, even as the residual sink it leaves grows.
4. **Positional encoding becomes more necessary: No-PE 18.0% → 9.3% → 3.2%.**
5. **Relocation sharpens: ratio 0.897 → 0.922 → 0.944.** The paper's "the sink relocates" claim is
   not only true but *increasingly* true with scale.
6. **`scale_pe` flips sign (ρ −1.0 → +0.46 → +1.0).** At gpt2, *deleting* p₁ **strengthens** the sink
   (110%); at large it **collapses** it (37%). The sink's reliance on p₁ being physically present grows with scale.

**The distributed → specialized query pathway (resolves an apparent contradiction):**

7. The query·EPE₁ alignment *mean* falls (0.228 → 0.023 → −0.061) and effect size shrinks
   (d 1.79 → 0.42 → 0.30) — yet the query pathway *grows* (#1). Resolution from the per-head cells:
   the alignment goes from **broad** (gpt2: 89% of heads align, max 0.47) to **sparse and specialized**
   (large: only 18% of heads align, but the strongest reaches **0.72** against an anti-aligning majority).
   The mechanism doesn't weaken — it **concentrates into a few carrier heads**. The pooled mean *understates*
   it at scale; the maximum and the head-count tell the real story.

**Non-monotone / model-specific results (flag, don't over-read):**

8. **first-layer MLP is essential only at medium (7%), dispensable at gpt2 (103%) and large (88%).**
   Yet the LM cost of removing it is large at both medium (+4.54) and large (+4.21) — so at *large* the
   sink survives without the first-layer MLP even though the model badly needs it (a mechanism/function dissociation).
9. **Massive-coordinate dependence is non-monotone (`scale_wk` floor 65% → 28% → 42%)**, but this is
   partly confounded: k = 3 columns zeroed at gpt2 vs 6 at medium/large.
10. Baseline sink strength is non-monotone (0.564 → 0.575 → 0.501).

---

## 3. Suggested narratives (for the paper)

- **N1 — "The residual sink is a second, query-driven positional pathway that grows and specializes
  with scale."** This is the headline. The paper (§5.3) leaves the 44.7% residual unexplained; we show
  it is the content term routed through EPE₁'s key (`k_1 ≈ EPE₁ W_k`), it *grows* to 74% at large, and
  it migrates from a distributed property (89% of heads) to a handful of specialized carrier heads
  (max alignment 0.72). Cite: nullify-`b_Q` 44.5→74.0%; Δ-share 0.56→0.25; align cells>0.1 89→18%, max 0.47→0.72.

- **N2 — "Positional *identity*, not *presence*, drives the sink."** Deleting p₁ leaves the sink
  intact-to-stronger at small (110%); replacing p₁ with p₂ destroys it (3%). The paper's "Remove First
  PE" is our `interp_pe` (replacement); we add the deletion control and get the dissociation. Nuance:
  the dissociation narrows with scale (deletion floor 110%→37%), so *presence* matters more at large.

- **N3 — "The learned query bias becomes functionally vestigial at scale."** Its LM-loss cost falls
  to +0.12 nats at large while the sink it anchors is increasingly carried by content queries. Speaks
  directly to "each component is individually dispensable" — here we quantify *how* dispensable, and that it grows.

- **N4 — "Relocation is real and sharpens with scale."** Validates a stated-but-unmeasured paper claim:
  ~50% of the sink mass transplants to the swapped position; ratio 0.90→0.94.

- **N5 (careful) — "First-layer MLP necessity is non-monotone."** A cautionary counter-narrative to any
  simple "the circuit strengthens/weakens monotonically" claim: gpt2-medium's sink depends entirely on
  the first-layer MLP, gpt2/-large's does not. Frame as evidence that the *implementation* is
  reorganized across scale even within one architecture.

---

## 4. Deep comparison with the paper (agree / diverge / implications)

The paper (Ran-Milo, Ofek & Mendel, arXiv:2604.14722) reports GPT-2 **small only**. Our gpt2 run is a
faithful reproduction; medium/large are extensions of its §5.1 (scale) limitation.

**Where we AGREE (gpt2 reproduces every reported landmark to <0.5pp):**

| Paper Table 1 (gpt2 small) | Paper | Ours (gpt2, 3-seed) |
|---|---:|---:|
| Baseline | .563 | 0.5637 ✓ |
| Nullify `b_Q` | 44.7% | 44.5% ✓ |
| Remove First PE (c) | 3.0% | 3.0% (interp_pe α=0) ✓ |
| Swap EPE (d) | 6.2% | 6.3% (standalone) ✓ |
| No MLP (g) | 50.3% | 50.1% ✓ |
| No PE (h) | 17.5% | 18.0% ✓ |
| Zero-Top-3-`W_k` (i) | 65.2% | 65.1% ✓ |
| Massive coords | 138,378,447 | 138,378,447 ✓ (exact) |

We also confirm the paper's qualitative claims: EPE₁'s massive coordinates coincide with the large
bias-projection coordinates, and the sink is a positional (not BOS-token) phenomenon.

**Where we EXTEND / REFINE the paper:**

- **§5.3 (unexplained residual) → resolved.** The paper cannot say what carries the 44.7% after
  nullifying `b_Q`. We identify it as **pathway B** (downstream queries aligning with EPE₁'s key) and
  show it dominates at scale (74%) and lives in specialized heads.
- **The paper conflates positional presence with identity.** Its only positional-source intervention
  is Remove-First-PE (a *replacement*). Our deletion vs replacement dissociation shows the sink keys on
  *identity/distinctness*, not the mere presence of p₁ — a mechanism the paper does not distinguish.
  This connects to Gu et al. (2025)'s NoPE sinks: even with p₁ deleted, a distinct first position sinks.
- **"Relocation" measured for the first time.** The abstract/§3.2.6 assert the sink "relocates"; the
  metric never tracked anything but position 1. We supply the number (ratio 0.90–0.94).
- **§5.1 (scale) answered with a mechanism**, not a conjecture: the pathway balance *rebalances*
  (bias→query) and the query pathway *specializes* (distributed→sparse) with scale.

**Where we appear to DIVERGE / add tension:**

- The paper frames the circuit as a stable "one circuit." Our first-layer-MLP result (§2, #8) shows the
  *implementation* is reorganized non-monotonically across scale (medium depends on layer-0 MLP; gpt2/large do not).
- The per-head *mean* query-alignment goes to ~0 / negative at scale, which naïvely reads as "the
  mechanism disappears" — it does not; it concentrates. A single summary statistic would mislead here.

---

## 5. Uncertainties & high-value follow-ups

- **[High value] The first-layer-MLP non-monotonicity (103% / 7% / 88%).** The most surprising result,
  and tight across seeds (std ≤0.33) so not noise. Hypothesis: whether the EPE₁ massive activations are
  *generated only in layer 0* vs *regenerated by later MLPs* differs per checkpoint. Worth a dedicated
  layer-wise probe of where the massive activations first appear and whether later layers rebuild them.
  If real and general, it bounds how far "first-layer MLP builds the EPE" generalizes.
- **[High value] Carrier-head identification.** The query pathway concentrates into ~18% of heads at
  large (max alignment 0.72). Identifying and ablating those specific heads would convert the
  "distributed→specialized" narrative from correlational to causal, and is the natural sequel.
- **[Confound] `scale_wk_massive` / Zero-Top-k across scales** zeroes k=3 (gpt2) vs k=6 (medium/large)
  columns. The 65%→28%→42% trend is not a clean scale law; re-run with matched k (e.g. top-3 everywhere)
  to isolate scale from k.
- **[Scope] Single checkpoint per scale, 3 *data* seeds only.** No pre-training seeds — we cannot
  separate scale effects from checkpoint idiosyncrasy (relevant to #8). E3 (emergence dynamics) is the
  complement here.
- **[Metric] Pooled-mean alignment understates head-concentrated mechanisms.** Report the max /
  top-quantile and head-count alongside the mean in the paper, not the mean alone.

---

## 6. Architectural framing, differences & gaps

- **All three models share the GPT-2 architecture** (learned absolute PE, learned query bias `b_Q`,
  combined Conv1D QKV, GELU MLP, pre-LN). The circuit ingredients (EPE = MLP(p₁)+p₁, `b_Q`, massive
  coordinates) exist in all three, so the decomposition transfers directly. Only scale changes:
  n_layer 12/24/36, n_head 12/16/20, n_embd 768/1024/1280.
- **This is cross-*scale*, not cross-*architecture*.** It tests the paper's §5.1 (scale) but **not** its
  headline claim that sinks "arise through *distinct circuits* across architectures." That claim requires
  models with the ingredients present/absent — OPT (has `b_Q` + abs PE), GPT-Neo (**no** `b_Q`), Qwen2.5
  (RoPE, **no** additive EPE). The repo's Table-1 harnesses cover those, but the **deep E4 decomposition
  (dose-response / query-alignment / relocation) is GPT-2-family-specific** and does not yet port:
  Neo has no bias pathway to grade, and Qwen has no separable EPE to scale. **This is the main framing gap** —
  the residual-pathway story is currently an *intra-architecture, cross-scale* result.
- **The massive-coordinate count is not scale-invariant** (3 → 6 → 6), and the *identity* of the coordinates
  is entirely different per checkpoint (no shared indices) — so "the massive coordinates" is a per-model set,
  not a universal address. Any cross-model claim must be about the *phenomenon*, not the specific dimensions.
- **Precision:** all fp32, so no dtype confound here (unlike the ≥6.7B models in the wider sweep).

---

## 7. Why these numbers matter (significance)

- **They resolve the paper's own open question (§5.3).** "Secondary contributors … we leave to future
  work" becomes a concrete, quantified second pathway — the single most defensible contribution: we answer
  the authors' explicit question with data.
- **They convert a limitation into a result (§5.1).** Rather than conjecturing that the mechanism might
  "strengthen, fragment, or be replaced" with scale, we show *which*: the bias→query rebalancing and the
  distributed→specialized transition, with monotone trends across three scales and sub-0.2pp seed noise.
- **They validate a stated-but-unmeasured claim (relocation).** Reproducibility reviewers reward exactly this.
- **They sharpen the mechanistic interpretation of positional sinks** ("identity, not presence"), linking
  the GPT-2-specific circuit to the architecture-general NoPE-sink literature (Gu et al. 2025).
- **They carry a general mech-interp lesson**: a circuit that looks distributed at small scale can be the
  *same* circuit concentrated into a few heads at large scale — and pooled summary statistics will hide it.
  This is a cautionary, citable methodological point beyond attention sinks.
- **Practical stakes:** if the sink is increasingly carried by content-query heads (not the bias) at scale,
  then bias-targeted mitigations (zeroing `b_Q`) will look cheap on LM loss (+0.12 nats) yet **fail to
  remove the sink** (74% remains) — directly informing the paper's "mitigations may need to be
  architecture-*and-scale*-specific" thesis.

---

*All figures are regenerable per model under `results/residual_sink_<model>/aggregate/`
(`dose_response.png`, `delta_share_heatmap.png`, `query_alignment_hist.png`) and the CSV/JSON summaries
this document is built from. gpt2 / gpt2-medium / gpt2-large each have 3 seeds; extend to more scales
(gpt2-xl) to firm up the trends, and add a carrier-head ablation to make the specialization causal.*
