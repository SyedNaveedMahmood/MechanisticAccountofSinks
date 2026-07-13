# E1–E2 Summary — Cross-Scale & Cross-Architecture Attention-Sink Sweep

**Scope.** This document consolidates every number from the E1/E2 Table-1 intervention sweep
(`cross_scale_and_architecture/`) across **16 checkpoints** in **4 architecture families**, run
for **3 seeds each** (`0,1,2`), 300 examples per seed (100 SST-2 + 100 GSM8K + 100 HumanEval,
each truncated to 40 tokens). It is written to be a self-contained reference for narrative
building and for citing numbers in the paper.

**Metric.** For every intervention we report the mean attention paid to position 0 (the BOS
sink) by the second-half tokens, averaged over all heads and over the model's *mid* layer band
(scaled band `[3, L−1)`; = the paper's layers 4–11 on 12-layer models). We report each
intervention as a **percentage of that model's own Baseline (a)** — i.e. *how much of the sink
survives the intervention*. Lower = the intervention destroys more of the sink.

**Provenance / integrity.** Every run carries a `run_config.json` recording model geometry, dtype,
band, and the exact massive-activation coordinates zeroed by (i)/(j). Audit result: all 16
geometries match the true HF configs; all bands correct; all 3 seeds present with 300 examples;
**every** `bos_attention_stats_overall.csv` is md5-distinct (no duplicated/mislabeled outputs);
massive coordinates are model-specific and identical across a model's 3 seeds; cross-seed
variation is tiny (max relative-% σ ≤ 1.8 over all interventions/models); no NaNs.

---

## 1. Intervention legend (Table-1 columns)

| Key | Name | What it does | What it probes |
|---|---|---|---|
| (a) | Baseline | unmodified forward pass | reference sink (= 100%) |
| (b) | No Query Bias | set `b_Q = 0` in every layer | the **query-bias** pathway |
| (c) | Remove First PE | give position 0 the PE of position 1 | position-0's **positional identity** |
| (d) | Swap EPE | swap the effective-PE component of positions 0↔1 | the **EPE transplant** |
| (e) | Swap PE | swap the raw PE of positions 0↔1 | raw vs effective PE |
| (f) | Nullify BOS Token | zero the position-0 **token embedding** | **source-(token-)agnosticism** |
| (g) | No MLP | skip the MLP block in every layer | MLP's role in building the sink |
| (h) | No PE | remove all positional information | the **positional signal** wholesale |
| (i) | Zero Top-3 Wk | zero the Wk columns of the **massive EPE₀ activations** (mean+3σ; ≥3) | the **massive-activation key channel** |
| (j) | Zero Random Wk | zero an equal number of **random** Wk columns | **size-matched control** for (i) |

> Note on (i): "Top-3" is the paper's GPT-2-small shorthand; the intervention zeros *all* massive
> coordinates of EPE₀ (mean + 3σ outliers, ≥3). Counts are model-specific and recorded per run
> (GPT-2-small = 3 → {138,378,447}; larger models 3–12). (j) always zeros the **same number** of
> random columns, so it is a strict size-matched control at every scale.

---

## 2. Master table — geometry & provenance

| Model | Family | Params | L | H | d_model | dtype | Mid band | #massive | Massive coords (EPE₀ outliers) |
|---|---|---|---:|---:|---:|---|---|---:|---|
| gpt2 | GPT-2 | 117M | 12 | 12 | 768 | fp32 | [3,11) | 3 | 138, 378, 447 |
| gpt2-medium | GPT-2 | 345M | 24 | 16 | 1024 | fp32 | [3,23) | 6 | 9, 238, 268, 428, 608, 580 |
| gpt2-large | GPT-2 | 774M | 36 | 20 | 1280 | fp32 | [3,35) | 6 | 792, 870, 440, 8, 24, 248 |
| opt-125m | OPT | 125M | 12 | 12 | 768 | fp32 | [3,11) | 3 | 422, 353, 130 |
| opt-1.3b | OPT | 1.3B | 24 | 32 | 2048 | fp32 | [3,23) | 5 | 1359, 1177, 572, 236, 1192 |
| opt-2.7b | OPT | 2.7B | 32 | 32 | 2560 | fp32 | [3,31) | 7 | 1454, 319, 13, 1625, 1102, 1802, 198 |
| opt-6.7b | OPT | 6.7B | 32 | 32 | 4096 | fp32 | [3,31) | 12 | 2394, 639, 1778, 3136, … (12) |
| opt-13b | OPT | 13B | 40 | 40 | 5120 | fp32 | [3,39) | 10 | 3964, 902, 4960, 440, 2211, … (10) |
| gpt-neo-125m | GPT-Neo | 125M | 12 | 12 | 768 | fp32 | [3,11) | 3 | 647, 522, 536 |
| gpt-neo-1.3B | GPT-Neo | 1.3B | 24 | 16 | 2048 | fp32 | [3,23) | 6 | 858, 352, 1467, 486, 1448, 1826 |
| gpt-neo-2.7B | GPT-Neo | 2.7B | 32 | 20 | 2560 | fp32 | [3,31) | 5 | 936, 1675, 1635, 1567, 483 |
| Qwen2.5-0.5B | Qwen2.5 | 0.5B | 24 | 14 (2 kv) | 896 | fp32 | [3,23) | per-sent. | RoPE stand-in (per sentence) |
| Qwen2.5-1.5B | Qwen2.5 | 1.5B | 28 | 12 (2 kv) | 1536 | fp32 | [3,27) | per-sent. | RoPE stand-in (per sentence) |
| Qwen2.5-3B | Qwen2.5 | 3B | 36 | 16 (2 kv) | 2048 | fp32 | [3,35) | per-sent. | RoPE stand-in (per sentence) |
| Qwen2.5-7B | Qwen2.5 | 7B | 28 | 28 (4 kv) | 3584 | bf16 | [3,27) | per-sent. | RoPE stand-in (per sentence) |
| Qwen2.5-14B | Qwen2.5 | 14B | 48 | 40 (8 kv) | 5120 | bf16 | [3,47) | per-sent. | RoPE stand-in (per sentence) |

*Qwen is RoPE (no additive EPE), so (i) uses the position-0 token embedding as a per-sentence
stand-in; `run_config.json` records `massive_coords = null` with the method noted.*

---

## 3. Master table — Table 1, all models (% of each model's Baseline)

Bold = the sink is **substantially destroyed** (<40% survives). Every value is a 3-seed mean;
cross-seed σ is ≤ ~1.8 pp everywhere (tightest at the largest models).

| Model | (a) | (b) NoQB | (c) RmPE | (d) SwapEPE | (e) SwapPE | (f) NullBOS | (g) NoMLP | (h) NoPE | (i) ZeroMassiveWk | (j) ZeroRandWk | Baseline BOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | 100 | 44.5 | **3.0** | **6.2** | 98.9 | 99.6 | 50.1 | **18.0** | 65.1 | 100.1 | 0.564 |
| gpt2-medium | 100 | 70.4 | **1.7** | **8.3** | 100.1 | 100.1 | **13.8** | **9.3** | **28.2** | 100.2 | 0.575 |
| gpt2-large | 100 | 74.0 | **3.9** | **3.4** | 100.0 | 100.5 | **20.6** | **3.2** | 42.1 | 99.9 | 0.501 |
| opt-125m | 100 | 82.8 | **2.1** | 60.6 | 46.3 | 99.8 | **17.8** | **3.5** | **39.7** | 99.8 | 0.677 |
| opt-1.3b | 100 | 97.7 | **32.2** | **0.5** | **4.0** | 99.6 | **23.9** | **4.6** | **37.7** | 100.0 | 0.626 |
| opt-2.7b | 100 | 98.2 | 35.7 | **13.5** | **32.6** | 99.9 | **17.9** | **7.6** | **38.6** | 99.8 | 0.624 |
| opt-6.7b | 100 | 98.7 | **5.7** | **2.6** | 53.4 | 99.7 | **11.7** | **2.9** | 55.8 | 99.9 | 0.630 |
| **opt-13b** | 100 | 98.8 | **6.7** | 93.0 | 98.6 | 98.0 | 49.3 | 57.8 | **110.1 ⚠** | 97.6 | **0.347** |
| gpt-neo-125m | 100 | 100.0 | **5.9** | 30.3 | 99.5 | 92.7 | **3.8** | 50.6 | 97.7 | 100.7 | 0.271 |
| gpt-neo-1.3B | 100 | 100.0 | **2.4** | 79.9 | 99.9 | 40.1 | **6.6** | 26.3 | 88.3 | 100.1 | 0.333 |
| gpt-neo-2.7B | 100 | 100.0 | **2.2** | 25.2 | 100.0 | 79.6 | **16.1** | **19.4** | 66.3 | 100.0 | 0.412 |
| Qwen2.5-0.5B | 100 | **6.1** | 99.7 | 95.6 | 95.6 | 97.4 | **2.9** | 77.0 | 99.5 | 99.5 | 0.520 |
| Qwen2.5-1.5B | 100 | **6.7** | 95.0 | 83.6 | 83.6 | **0.5** | **3.1** | 78.8 | 98.9 | 99.9 | 0.520 |
| Qwen2.5-3B | 100 | **14.5** | 99.7 | 93.6 | 93.6 | 64.6 | **2.6** | 89.1 | 99.9 | 99.9 | 0.502 |
| Qwen2.5-7B | 100 | **6.7** | 99.4 | 95.9 | 95.9 | **0.5** | **5.1** | 90.1 | 97.0 | 99.9 | 0.539 |
| Qwen2.5-14B | 100 | 45.4 | 100.0 | 99.7 | 99.7 | **1.8** | **4.2** | 95.9 | 99.7 | 99.6 | 0.639 |

Baseline BOS = absolute mean attention to position 0 over the mid band (per-example SE ≈
0.001–0.006 for all models). The **specificity gap** (j − i) — how much more the massive-Wk
ablation destroys the sink than a size-matched random ablation — is tabulated in §5.

---

## 4. Reproduction fidelity — GPT-2-small vs. the paper

GPT-2-small **is** the paper's model, so its row is a direct reproduction check. All four paper
anchors are reproduced to within measurement noise:

| Quantity | Paper | This run (gpt2) | Δ |
|---|---|---|---|
| Baseline sink strength (BOS attention) | 0.563 | 0.564 | +0.001 |
| Massive EPE₀ coordinates | 138, 378, 447 | 138, 378, 447 | exact |
| Residual after nullifying `b_Q` — **(b)** | 44.7% | 44.5% | −0.2 pp |
| Residual after zeroing top-3 `W_k` — **(i)** | 65.2% | 65.1% | −0.1 pp |

The supporting interventions also behave exactly as the paper's mechanism predicts on GPT-2-small:
Remove-First-PE **(c) = 3.0%** and Swap-EPE **(d) = 6.2%** *collapse* the sink (position-0's
positional identity is necessary); Nullify-BOS-token **(f) = 99.6%** leaves it essentially intact
(the sink is **source-/token-agnostic**); the random control **(j) = 100.1%** does nothing.
**Conclusion: the pipeline is a faithful reproduction of the paper on the paper's own model.**

---

## 5. The specificity control (i vs j) — does the massive-Wk channel carry the sink?

`(j)` zeros the same number of *random* Wk columns as `(i)` zeros *massive* columns. `(j) ≈ 100%`
on all 16 models (range 97.6–100.7), so any effect in `(i)` is specific to the massive
coordinates, not to ablating that many columns. The gap `j − i` is the clean read-out:

| Family | Model (gap `j−i`, pp) | Reading |
|---|---|---|
| **GPT-2** | small **+35**, medium **+72**, large **+58** | strong, and *strengthens* beyond the small model |
| **OPT** | 125m **+60**, 1.3b **+62**, 2.7b **+61**, 6.7b **+44**, 13b **−13 ⚠** | strong through 6.7b; **inverts at 13b** |
| **GPT-Neo** | 125m **+3**, 1.3b **+12**, 2.7b **+34** | weak, but *emerges with scale* |
| **Qwen2.5** | 0.5B–14B all **≈ 0** (−0.2 … +2.9) | **absent** (RoPE — no additive-EPE key channel) |

---

## 6. Findings & trends

### 6.1 The two pillars of the GPT-2 sink circuit dissociate across architectures
The paper attributes the GPT-2 sink to two ingredients: **(P1) the query bias `b_Q`** and **(P2)
the positional / massive-EPE key channel**. Across families these pillars come apart:

| Family | P1 — Query bias (b) | P2 — Positional/EPE (c,d,i,h) | Source-agnostic (f) |
|---|---|---|---|
| **GPT-2** | **driver** (b 44–74%), but *weakens with scale* | **driver** (c,d collapse; i 28–65%) | **yes** (f ≈ 100%) |
| **OPT** | *not* a driver (b 83–99%) | driver via **Wk** (i 38–56%); c,d,h mixed | **yes** (f ≈ 98–100%) |
| **GPT-Neo** | none (no `b_Q`; b ≡ 100%) | positional (c collapses); Wk only **at scale** | **partial** (f 40–93%) |
| **Qwen2.5** | **strong driver** (b 6–45%) | **absent** (RoPE; c,d,h,i inert) | **no for several** (f 0.5–1.8%) |

This is the single most important cross-architecture result: **the query-bias pillar and the
positional pillar are separable, and different architectures lean on different pillars.**

### 6.2 GPT-2 scaling: the query-bias pillar fades, the Wk pillar persists
Within GPT-2, nullifying `b_Q` **(b)** removes progressively *less* of the sink with scale
(44.5% → 70.4% → 74.0% survive), i.e. the query-bias contribution shrinks from ~55% of the sink
in the small model to ~26% in large. Meanwhile the massive-Wk channel **(i)** stays a strong
driver (medium 28%, large 42% survive — an even larger specificity gap than the small model).
**Interpretation:** at larger GPT-2 scale the sink shifts from being query-bias-led toward being
carried by the massive-activation key channel.

### 6.3 OPT: the sink is a *key-side* (Wk) phenomenon, not a query-bias one
Although OPT keeps a learned query bias, nullifying it **(b)** barely dents the sink (83–99%
survive). Instead the massive-Wk ablation **(i)** is the dominant lever (37–56% survive through
6.7b), with a large specificity gap (+44 to +62 pp). So OPT reproduces the paper's *mechanism*
(massive-activation key channel) but not the paper's *primary actuator* (the query bias).

### 6.4 GPT-Neo: a positional sink that only becomes Wk-dependent at scale
GPT-Neo has no query bias, so **(b) ≡ 100%** (a structural no-op, as expected). Its sink is
clearly positional — Remove-First-PE **(c)** collapses it at every scale (2–6%). The massive-Wk
channel **(i)** is nearly inert for the 125M model (97.7%) but **emerges monotonically with
scale** (88.3% → 66.3%), mirroring the "Wk pillar strengthens with scale" trend seen in GPT-2.

### 6.5 Qwen2.5 (RoPE): the query bias carries the sink; the positional channel does not
Qwen keeps a learned query bias and delivers position via RoPE. The result is a near-mirror image
of OPT: nullifying `b_Q` **(b)** *collapses* the sink (6–15% survive for 0.5–7B; 45% for 14B),
while every positional/EPE lever is **inert** — Remove-First-PE **(c) 95–100%**, No-PE **(h)
77–96%**, massive-Wk **(i) 97–100%**. **Qwen is the cleanest evidence that the query-bias pillar
is real and can stand alone**, precisely in the family where the additive-EPE pillar is absent by
construction (there is no EPE to have massive activations).

### 6.6 The MLP builds the sink everywhere
No-MLP **(g)** substantially reduces the sink in **every** family (GPT-2 14–50%, OPT 12–49%,
Neo 4–16%, Qwen 3–5%), and most dramatically in Neo/Qwen (down to 3–5%). The MLP-driven
enrichment of position-0's representation is an architecture-universal ingredient — consistent
with the paper's EPE definition `EPE = PE + MLP⁽⁰⁾(PE)`.

### 6.7 Source-agnosticism is *not* universal
The paper's hallmark "source-agnostic" property — nullifying the BOS *token* leaves the sink
intact — holds cleanly for GPT-2 and OPT **(f ≈ 98–100%)**, but **breaks** for several models:
GPT-Neo-1.3B (40%), and Qwen-1.5B/7B (0.5%) and 14B (1.8%). In those Qwen models the sink is
strongly **token-content dependent** — zeroing the first token's embedding almost eliminates it.
(Qwen-0.5B/3B retain it, 97%/65% — the property is heterogeneous within the family.)

---

## 7. Divergence from / agreement with the paper — deep dive

**Where the results *agree* with and *strengthen* the paper.**
- The exact GPT-2-small numbers (§4) reproduce the paper's headline residuals (`b_Q` → 44.5%,
  top-3 `W_k` → 65.2%) and its massive coordinates (138/378/447) — direct corroboration.
- The **mechanism generalizes in kind**: a *massive-activation key channel* exists and drives the
  sink in GPT-2 (all scales) and OPT (to 6.7b), with a rigorous size-matched control (j ≈ 100%).
  The paper's central mechanistic object is therefore not a GPT-2-small idiosyncrasy.
- **Source-agnosticism** — a specific, falsifiable paper claim — holds for the two additive-PE
  families closest to GPT-2 (OPT, and GPT-2 at all scales).
- The **MLP's role** in constructing EPE is confirmed universally (g reduces the sink everywhere).

**Where the results *diverge* from a naive reading of the paper (the interesting part).**
1. **The query bias is not the universal actuator.** The paper frames `b_Q` as a primary driver.
   That is model-specific: it is a *weakening* driver in GPT-2 (share falls with scale), a
   *non*-driver in OPT (b ≈ 98%), and a *dominant* driver in Qwen (b ≈ 6%). The paper's two
   ingredients are **separable**, and which one dominates depends on the positional scheme.
2. **RoPE removes the positional pillar but keeps the sink.** In Qwen the additive-EPE channel
   does not exist, yet a strong sink persists and is carried entirely by the query bias. This
   extends the paper's account to a regime it does not cover.
3. **Source-agnosticism is conditional, not universal** — it fails for several RoPE models and
   one Neo model (§6.7), indicating a second, *token-content-driven* sink regime the paper's
   GPT-2 analysis does not exhibit.
4. **opt-13b breaks the massive-Wk pillar entirely** (see §8): zeroing the massive columns
   *increases* the sink (110%), the only inversion in the sweep.

**Net:** the paper's *circuit components* are real and transferable, but the paper's *emphasis*
(query bias as the lead actuator, source-agnosticism as intrinsic) is a property of GPT-2-scale,
additive-PE models rather than of attention sinks in general.

---

## 8. Anomalies, uncertainties & the opt-13b outlier

### 8.1 opt-13b — a genuine, reproducible outlier (highest-priority uncertainty)
opt-13b is qualitatively unlike every other model in the sweep:
- **(i) Zero Massive Wk = 110.1%** — the *only* model where removing the massive columns
  *increases* the sink. It is **stable** (per-seed 109.8–110.5) and **consistent across datasets**
  (SST-2 ≈ 103%, GSM8K ≈ 112%, HumanEval ≈ 116%), so it is not noise, not a seed fluke, and not
  the old under-ablation bug (the fix zeros all 10 massive coords).
- Its **baseline sink is much weaker** (0.347 vs. ~0.63 for the other OPTs).
- Its positional levers are anomalously **inert**: Swap-EPE **(d) 93%**, No-PE **(h) 58%** — the
  sink largely survives removing positional information (yet Remove-First-PE **(c) 6.7%** still
  collapses it).

**Two live interpretations, both worth reporting:**
(a) *Emergent shift at scale* — the largest OPT has moved to a different, weaker, less
positional / non-massive-Wk sink regime; removing the massive columns redistributes attention
*back onto* position 0. This would be a real and citable "the circuit changes at the largest
scale" finding.
(b) *Methodological interaction* — the deep scaled band `[3,39)` averages 36 layers of a 40-layer
model, and the massive-coordinate identification / metric could interact differently there.

**Recommended before publishing any opt-13b claim:** (1) re-run with `--layer-mode fixed` (layers
4–11) to remove the deep-band confound; (2) inspect per-layer BOS curves and attention maps;
(3) sweep the massive threshold (`n_std ∈ {2.5,3,3.5}`) to confirm the coordinate set is stable;
(4) sanity-check the checkpoint load. Until then, treat opt-13b `(i)` as **"anomalous — under
investigation,"** not as a headline number.

### 8.2 Qwen (f) Nullify-BOS heterogeneity
Within Qwen, **(f)** ranges from 0.5% (1.5B, 7B) to 97% (0.5B) — stable within each model but
wildly different across the family. This is a genuine finding (some Qwen sinks are token-driven,
others positional) but it needs an explanatory sentence in the paper; it is not a bug (f is an
unmodified intervention and the values are seed-stable).

### 8.3 Qwen / small-Neo `(i)` inertness is expected, not a defect
Qwen `(i) ≈ 100%` and Neo-125m `(i) = 97.7%` are **null results by construction/architecture**
(no additive EPE for Qwen; the Wk channel not yet developed in the smallest Neo). Report them as
"no effect," not as evidence against the mechanism.

### 8.4 Statistical confidence
Cross-seed variation is small (max relative-% σ ≤ 1.8 pp; the largest models are the tightest,
e.g. opt-13b σ ≤ 0.37, Qwen-14B σ ≤ 0.19), and per-example SEs on the absolute sink are
0.001–0.006. Differences of >~3 pp between interventions are well outside noise; sub-1-pp
differences (e.g. gpt2 65.1 vs paper 65.2) are within it. Precision: all runs fp32 except
Qwen-7B/14B (bf16); the **relative** metric is robust to this, but flag the dtype when citing
absolute BOS for those two.

---

## 9. Architectural differences & gaps — the framing

| | **GPT-2** | **OPT** | **GPT-Neo** | **Qwen2.5** |
|---|---|---|---|---|
| Positional scheme | learned absolute PE | learned absolute PE (+2 offset) | learned absolute PE | **RoPE** (rotary) |
| Query bias `b_Q` | **yes** | yes | **no** | **yes** |
| Norm placement | pre-LN | pre-LN | pre-LN | pre-LN (RMSNorm) |
| Attention scaling | `1/√d` | `1/√d` | **none** | `1/√d`, **GQA** |
| Has additive EPE (for massive-Wk) | yes | yes | yes | **no** |
| **Sink actuator observed** | query bias → Wk (shifts to Wk at scale) | **Wk channel** (not `b_Q`) | positional; Wk **at scale** | **query bias** (RoPE ⇒ no Wk) |

**Two structural gaps to name explicitly in the paper:**
1. **The `b_Q`-present-but-inert gap (OPT).** OPT has the same query bias as GPT-2 yet does not
   use it for the sink. So *possessing* `b_Q` is not sufficient; the paper's mechanism is one of
   *several* routes to a sink, and OPT selects the key-side route.
2. **The RoPE gap (Qwen).** With no additive EPE there is no "massive-activation" key channel to
   ablate; the paper's positional analysis has no direct analogue. Interventions (c/d/e/h) become
   RoPE **position-id** manipulations and (i) becomes a per-sentence token-embedding stand-in — a
   deliberately weaker probe. The strong `b_Q` dependence is the transferable part; the positional
   part must be reframed, not directly compared.

---

## 10. Suggested narratives for the paper

1. **"One circuit, two separable pillars, four architectures."** Lead with the §6.1 table. The
   GPT-2 sink circuit = query-bias pillar + massive-EPE-key pillar; the sweep shows these pillars
   *dissociate*: OPT keeps only the key pillar, Qwen keeps only the query-bias pillar, GPT-Neo has
   the key pillar emerge with scale. The mechanism is real and modular, not monolithic.

2. **"The query bias is sufficient but not necessary — and vice-versa."** Qwen (b→6%) proves the
   query-bias pillar can carry the sink alone; OPT (b→98%, i→38%) proves the key channel can carry
   it alone. GPT-2 uses both and trades between them with scale (§6.2).

3. **"Attention sinks are convergent, not mechanistically identical."** Every family builds a
   BOS/first-token sink of comparable strength (baseline 0.27–0.68), but via different circuitry.
   The *phenomenon* is universal; the *mechanism* is architecture-contingent. Good mitigation
   advice must therefore be conditioned on the positional scheme and the query bias.

4. **"Source-agnosticism is a GPT-2/OPT property, not a law."** Use (f) to show the sink is
   token-agnostic for additive-PE models but token-driven for several RoPE models — a caveat that
   sharpens (rather than weakens) the paper's original claim.

5. **"The MLP is the common substrate."** (g) reduces the sink in all four families — the
   `MLP⁽⁰⁾`-built EPE / position-0 enrichment is the one universal ingredient, a clean bridge from
   the paper's EPE definition to the cross-architecture picture.

---

## 11. Why these numbers matter & directions for further research

**Why they matter.**
- **Reviewer-proofing.** GPT-2-small reproduces the paper to ≤0.2 pp on both headline residuals,
  with model-specific coordinates recorded in `run_config.json` and a size-matched control at
  100% — the sweep is auditable end-to-end and the earlier defects (duplicated data, invalid
  `int_i`) are provably gone.
- **Generality.** Showing the mechanism in OPT and Qwen (with the pillars dissociating) turns a
  single-model case study into a mechanistic *theory of sinks* with predictive structure.
- **Mitigation.** Sink-removal / long-context methods that target the query bias will work on
  GPT-2/Qwen but **not** OPT; methods that target the massive-activation key columns will work on
  GPT-2/OPT but **not** Qwen. The dissociation table is directly actionable for practitioners.

**High-value follow-ups.**
1. **Resolve opt-13b (§8.1).** Highest priority — it is either a genuine emergent-at-scale
   regime change (very citable) or a band/threshold interaction (must be excluded). A `fixed`-band
   re-run + per-layer curves settles it.
2. **Trace the GPT-2 pillar hand-off.** The query-bias share falling (55%→26%) while the Wk share
   holds is a clean scaling story; a denser GPT-2 ladder (+ xl) plus E3-style emergence dynamics
   could pin down where/why the hand-off happens.
3. **Explain the Qwen (f) split.** Why do 1.5B/7B/14B have token-driven sinks while 0.5B/3B do
   not? A tokenizer/BOS-handling or training-data explanation would be valuable.
4. **A `b_Q`-present-but-inert probe for OPT.** Why does OPT not use its query bias for the sink?
   Measuring `b_Q · k₁` alignment (the paper's Fig. 2 analog) across families would explain the
   dissociation directly.
5. **Formalize the "sink actuator" taxonomy** (query-bias-led vs key-channel-led vs
   positional-led) and test whether it predicts downstream behaviors (long-context degradation,
   quantization sensitivity, streaming-attention robustness).

---

*Generated from the 16-model × 3-seed E1/E2 sweep. Every number is traceable to a
`run_config.json` + `bos_attention_stats_overall.csv` under
`cross_scale_and_architecture/results/table1_multiseed_*`. `(i)` values reflect the corrected,
paper-faithful massive-activation identification (mean + 3σ, size-matched random control).*
