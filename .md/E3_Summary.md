# E3 Summary — Circuit Emergence During Pre-Training: Results, Validity, and Findings

**Run:** `emergence_dynamics_analysis.py --mode all --run-family gpt2-small --n-checkpoints 18`
**Data:** Stanford CRFM "Mistral" GPT-2 small, **5 independent pre-training seeds** (alias, battlestar, caprica, darkmatter, expanse), **17 log-spaced checkpoints** each (steps 0 → 400 000), fixed 300-example eval set (data-seed 0), layers 4–11, fp32.
**Target paper:** Ran-Milo, Ofek & Mendel, *A Mechanistic Account of Attention Sinks in GPT-2* (arXiv:2604.14722). **Track:** BlackboxNLP 2026 Reproducibility (theme: Generalizability → dynamics; addresses the paper's §5.2 "static single checkpoint" limitation).

> **TL;DR.** The run is valid and complete (85/85 checkpoints, 0 NaN). The `b_Q–EPE₁–W_k` sink circuit **reproduces in all 5 independent runs** as a functional endpoint, but with three findings the original paper could not see: (1) the massive-activation **coordinate indices are completely seed-specific** (zero cross-seed overlap) — a universal mechanism in an arbitrary basis; (2) the circuit emerges **non-monotonically** — a transient *anti-aligned / anti-sink* phase (~step 400–700) and a massive-activation **overshoot** (~10× at step 40–70 k, relaxing to ~1.4× at convergence) while the behavioral sink rises monotonically; (3) in Mistral the sink routes **almost entirely through the query pathway** — Nullify-`b_Q` removes only **~5%** of the sink vs the paper's **55%** — confirming E4's two-pathway redundancy but showing the A/B balance is **not universal**. The one weak deliverable is the formal temporal-ordering test (Wilcoxon *p* = 0.81), which the non-monotonic dynamics break by construction.

---

## 1. Validity assessment — the run is sound

| Check | Result |
|---|---|
| Completeness | 5 runs × 17 checkpoints = **85/85** `metrics.json` present |
| Numerical integrity | **0** NaN / inf across all 85 × 24 numeric fields |
| Step-0 null control (random init) | sink = **0.0344** ≈ chance (1/28 ≈ 0.036); bias-align = **0.000** (b_Q inits to zero); efficacy(Nullify-b_Q) = **0.000**; max\|EPE₁\| = 0.065 (noise) — a clean null ✓ |
| Cross-seed convergence | tight bands (e.g. sink 0.897 ± 0.008; bias-align 0.795 ± 0.027) — 5 independent runs agree ✓ |
| Internal consistency | T1-only attn 0.818 + T3-only attn 0.555 → full (T1+T3) 0.897; the score-decomposition identity `T1+T2+T3+T4 == score` held on every checkpoint (no assert failures) ✓ |
| Figures ↔ CSVs | all 5 figures render and match the aggregate CSVs ✓ |
| Comparability | data sample fixed (seed 0) across all runs/checkpoints, so cross-run bands reflect *pre-training* variance, not sampling noise ✓ |

The three code fixes applied during the run (fp-tolerance for the identity assert, tokenizer reuse, diagnostics speedup) are all **numerically identical / non-value-changing** — verified bit-for-bit — so mixing checkpoints computed under different code versions does not affect any number here.

**Verdict: valid.** Proceed to interpretation.

---

## 2. Headline result — the circuit reproduces across 5 independent runs

**Table E3-2 — converged circuit (step 400 000) vs the paper's OpenAI GPT-2.**

| Signal | alias | battlestar | caprica | darkmatter | expanse | **mean ± std** | Paper (OpenAI) |
|---|---|---|---|---|---|---|---|
| Sink strength S (BOS attn) | 0.893 | 0.904 | 0.899 | 0.886 | 0.902 | **0.897 ± 0.008** | 0.563 |
| Bias–EPE₁ align A = cos(b_Q, W_k·EPE₁) | 0.812 | 0.824 | 0.805 | 0.762 | 0.773 | **0.795 ± 0.027** | (Fig 2, positive) |
| Query–EPE₁ align B = cos(x_iW_q, W_k·EPE₁) | 0.286 | 0.307 | 0.266 | 0.276 | 0.257 | **0.278 ± 0.019** | (E4 residual, positive) |
| Δ₁ dominance D (T3-only attn→pos 0) | 0.553 | 0.556 | 0.536 | 0.559 | 0.571 | **0.555 ± 0.013** | — |
| max\|EPE₁\| (massive activation) | 1.409 | 1.338 | 1.387 | 1.308 | 1.360 | **1.360 ± 0.040** | (Fig 7, present) |

Every seed independently develops the full circuit: a strong first-token sink, positive `b_Q`–`EPE₁ W_k` alignment (paper Fig 2), positive downstream **query**–`EPE₁ W_k` alignment (the E4 pathway-B residual), a dominant source-agnostic shift Δ₁, and massive activations in `EPE₁`. **This establishes the paper's circuit as a canonical, reproducible pre-training solution — not an artifact of OpenAI's single training run.**

*Note:* Mistral converges to a **stronger** sink than OpenAI GPT-2 (0.90 vs 0.56). The circuit *structure* reproduces; the absolute magnitude differs (plausibly OpenWebText vs WebText and training length). Report the relative structure, note the absolute gap.

---

## 3. Emergence dynamics — the interesting part

**Table E3-3 — cross-seed mean trajectory (selected signals × steps).**

| step | S sink | max\|EPE₁\| | A bias-align | B query-align | D Δ₁-dom | eff Nullify-b_Q | eff Remove-1st-PE | eff Zero-Top3-W_k |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.034 | 0.065 | 0.000 | 0.007 | 0.034 | 0.000 | 0.032 | 0.000 |
| 400 | **0.011** | 0.098 | **−0.616** | **−0.467** | 0.029 | 0.002 | 0.598 | −0.012 |
| 1 000 | 0.203 | 0.149 | −0.452 | −0.183 | 0.033 | 0.018 | 0.940 | −0.015 |
| 4 000 | 0.323 | 0.746 | −0.314 | 0.098 | 0.061 | 0.069 | 0.983 | 0.249 |
| 7 000 | 0.475 | 2.498 | −0.135 | 0.219 | 0.121 | 0.062 | 0.992 | **0.449** |
| 20 000 | 0.696 | 7.387 | 0.388 | 0.171 | 0.602 | 0.089 | 0.998 | 0.323 |
| 40 000 | 0.776 | **10.196** | 0.614 | 0.102 | 0.713 | **0.091** | 0.995 | 0.188 |
| 70 000 | 0.812 | 10.173 | 0.698 | 0.075 | **0.735** | 0.093 | 0.996 | 0.160 |
| 100 000 | 0.839 | 8.510 | 0.720 | 0.102 | 0.720 | 0.093 | 0.993 | 0.267 |
| 400 000 | 0.897 | 1.360 | 0.795 | 0.278 | 0.555 | 0.053 | 0.993 | 0.271 |

Three dynamical phenomena stand out:

**(a) Transient anti-sink / anti-alignment (~step 400–700).** Both pathways go strongly **anti-aligned** (bias A ≈ −0.62, query B ≈ −0.47) and the sink *drops below its random-init value* (S = 0.011 < 0.034). Early training briefly pushes attention **away** from position 0 before building the sink. Robust across seeds (bands do not cross zero at the trough).

**(b) Massive-activation overshoot-and-relax.** `max|EPE₁|` is flat until ~step 4 000, ramps to a peak **~10.2 at step 40–70 k**, then **declines to ~1.4** at convergence — a ~7.5× overshoot. Δ₁-dominance (peak 0.735 → 0.555), the bias-align *contrast* (peak 1.44 → 1.03), and Zero-Top-3-W_k efficacy (peak 0.45 → 0.27) overshoot in lockstep. **Meanwhile the behavioral sink S rises monotonically.** Interpretation: the model first establishes the sink with extreme massive activations, then *reorganizes to a more efficient circuit* (smaller activations, better alignment) while holding/strengthening the behavior.

**(c) Monotonic behavior, non-monotonic mechanism.** S(t) is a clean sigmoid (chance → 0.90); the *mechanistic* signals underneath it are not. This is the core "dynamics" contribution the static paper cannot make.

---

## 4. Seed universality — the most novel finding (Fig E3-C)

**Top-3 `|EPE₁|` massive-coordinate indices at convergence:**

| seed | top-3 coords | full >3σ set |
|---|---|---|
| alias | **683, 8, 541** | 8,164,340,541,545,572,654,683,724 |
| battlestar | **618, 318, 341** | 232,284,318,341,545,598,618,644,695 |
| caprica | **115, 229, 439** | 113,115,177,229,411,439,655,710 |
| darkmatter | **610, 470, 192** | 100,192,256,470,532,610,667,700,751 |
| expanse | **529, 233, 41** | 41,209,218,233,435,529,595 |
| *Paper (OpenAI)* | *138, 378, 447* | — |

**Cross-seed Jaccard of the top-3 index sets = 0.00 for every pair (perfect identity matrix).** The only overlap anywhere in the wider >3σ sets is a single coordinate (545, alias ∩ battlestar). **None** of the seeds reproduce the paper's 138/378/447.

⇒ **The sink circuit is functionally universal but expressed in a seed-arbitrary coordinate basis.** Every run solves the same problem with the same mechanism (massive activations in `EPE₁`, keyed by `W_k`), but *which* hidden dimensions carry the massive activations is a symmetry-broken choice of the individual optimization. This directly extends — and bounds — the paper's massive-coordinate claim: the *count* and *role* are canonical; the *identity* is not.

**Within-seed crystallization.** The coordinate identity is unstable early (Jaccard-to-final = 0 through ~step 400–1 000), reaches 2/3 locked (Jaccard 0.5) by ~step 2 k–20 k, and fully locks (1.0) at step 40 k (caprica), 70–100 k (battlestar), or only at 400 k (alias, darkmatter, expanse) — i.e. the basis crystallizes together with the massive-activation ramp, with a last coordinate sometimes still migrating late in training.

---

## 5. Two-pathway emergence & causal efficacy — ties to E4 (Figs E3-D, E3-E)

**Table E3-4 — converged intervention efficacy (fraction of sink removed).**

| Intervention | Mistral mean ± std | Paper (OpenAI) | Reading |
|---|---|---|---|
| Remove-First-PE (kills the shared EPE₁ object) | **0.993 ± 0.003** | ~0.97 | reproduces: the shared key kills the whole sink |
| Zero-Top-3-W_k (coordinate channel) | **0.271 ± 0.134** | ~0.35 | roughly reproduces; tracks massive-activation magnitude over time |
| Nullify-b_Q (pathway A / bias) | **0.053 ± 0.003** | **0.55** | **strong divergence — bias pathway nearly negligible in Mistral** |

**The key contrast:** in the paper, nullifying the query bias removes 55% of the sink; in **all 5 Mistral runs it removes ≤ 9% at any point in training and ~5% at convergence.** Yet the bias is *well aligned* with the key (cos A = 0.80). The resolution is E4's redundancy: T1-only (content/query) attention alone already produces 0.82 of the 0.90 sink, so deleting the bias term T3 barely moves the softmax. **High cosine alignment ≠ high causal contribution** — the bias rotates into the key direction but carries little magnitude; the **query pathway carries the sink.**

⇒ This **confirms E4's two-pathway account dynamically and at scale** (two redundant pathways sharing `EPE₁ W_k`; only the shared object is load-bearing), **and shows the A/B balance is run-dependent** — Mistral routes the sink almost entirely through pathway B, OpenAI splits ~55/45. A clean cross-model contrast.

**Pathway emergence order (Fig E3-E).** Both pathways dip anti-aligned (~step 400–700); the **query pathway B recovers first** (crosses 0 at ~step 3–4 k, peaks ~0.22 at 7 k), the **bias pathway A recovers later** (crosses 0 at ~10–20 k) and then overtakes B in cosine (0.80 vs 0.28) — while remaining causally inert. So the *causally dominant* pathway (query) aligns first; the *causally minor* pathway (bias) aligns later but larger.

---

## 6. Temporal ordering — the inconclusive deliverable (honest) (Fig E3-B)

**Table E3-1 — per-signal emergence step (first crossing of 50% of converged value).**

| Signal | median onset | IQR | note |
|---|---:|---|---|
| align_bias_pos0 (A) | 200 | 100 – 20 000 | **unstable** (bimodal across seeds) |
| query_align_pos0 (B) | 0 | 0 – 0 | **artifact** (see below) |
| sink_strength (S) | 7 000 | 7 000 – 10 000 | clean |
| delta_dominance (D) | 20 000 | 20 000 | clean |
| epe1_max_abs (M) | 20 000 | 20 000 | measures the overshoot half-max |

**Ordering test (does the circuit assemble before the sink?):** align-onset vs sink-onset, paired across 5 runs → **3/5 runs have alignment before sink, Wilcoxon p = 0.812 (not significant).**

**Why it's inconclusive, honestly:** the 50%-of-range onset definition assumes monotone emergence. The alignment signals are **not** monotone — they dip strongly negative then recover, so (i) `query_align` onset collapses to step 0 (its min occurs mid-training, making step-0 look "already high"), and (ii) `align_bias` onset is bimodal. The clean, monotone signals (S at ~7 k; D and M at ~20 k) *do* order sensibly, and the trajectories (§3) tell a coherent qualitative sequence — **alignment reorganization (steep ~step 0.4–7 k) → behavioral sink (~4–20 k) → massive-activation / Δ₁ consolidation (~20–70 k)** — but the *formal statistical* ordering claim does not reach significance and should not be over-stated. **This is the one deliverable to present as a negative/limitation** (which the track explicitly welcomes), ideally paired with a monotone-only onset variant in a revision.

---

## 7. What succeeded, what failed, what to caveat

**Succeeded (publishable):**
- Cross-seed **reproduction** of the entire circuit as a functional endpoint (§2).
- **Seed-specificity of the massive-coordinate indices** — zero-overlap Jaccard (§4). *Most novel.*
- **Non-monotonic emergence**: anti-sink transient + massive-activation overshoot (§3).
- **Query-pathway dominance** and the **run-dependent A/B balance** (Nullify-b_Q 5% vs paper 55%) — dynamic confirmation + extension of E4 (§5).

**Failed / weak:**
- **Formal temporal-ordering test is not significant** (p = 0.81) — broken by the non-monotone alignment dynamics (§6). Present as a limitation, not a headline.

**Caveats to state in the paper:**
- Absolute sink magnitude differs from OpenAI (0.90 vs 0.56); structure reproduces, magnitude does not.
- Early-training efficacy ratios (steps ≤ 400) are noisy because the baseline sink is ≈ 0 (division by near-zero); ignore < step 1 000 for the efficacy curves.
- Massive coords are **re-identified per checkpoint** (never hardcoded); the >3σ set size varies (6–9), so universality is reported on the fixed **top-3** for comparability.
- n = 5 seeds → non-parametric tests are low-powered; treat ordering p-values as indicative.

---

## 8. Publishable findings, ranked

1. **Universal mechanism, arbitrary basis.** The sink circuit reproduces across 5 independent pre-training runs, but the massive-activation coordinate indices are completely disjoint (cross-seed Jaccard = 0) and none match the paper's — the *function* is canonical, the *coordinates* are a symmetry-broken per-run choice. *(New; directly extends the paper's coordinate claim.)*
2. **The sink is a redundant two-pathway mechanism, and the pathway balance is not universal.** Nullify-b_Q removes ~5% in Mistral vs 55% in OpenAI; only ablating the shared `EPE₁` object removes the sink (~99%). Confirms E4 dynamically and reframes the paper's b_Q-centric account as run-dependent. *(Confirmation + extension; ties E3↔E4.)*
3. **Non-monotonic emergence: transient anti-sink + massive-activation overshoot.** The behavioral sink is monotone but the mechanism over-forms (~10× massive activations at mid-training) then relaxes, and both pathways pass through an anti-aligned phase. *(New dynamics; addresses §5.2.)*
4. **Reproduction at convergence** of sink, bias/query alignment, Δ₁-dominance, and massive activations across 5 seeds with tight bands. *(Generalizability/robustness.)*
5. **(Negative)** A simple onset-ordering test cannot establish "circuit precedes sink" because the alignment dynamics are non-monotone — a methodological caution for training-dynamics interpretability. *(Welcomed negative result.)*

---

## 9. Framing for the paper (draft claims, all backed by the tables above)

- "Across five independently pre-trained GPT-2 seeds, the `b_Q–EPE₁–W_k` sink circuit re-emerges as a canonical solution (sink 0.90 ± 0.01; bias–`EPE₁ W_k` alignment 0.80 ± 0.03), yet the massive-activation coordinate indices carrying it are **entirely seed-specific** (pairwise Jaccard = 0) — the mechanism is universal but its coordinate basis is arbitrary."
- "The circuit does not assemble monotonically: every seed passes through a transient **anti-aligned, anti-sink** phase (~0.5 k steps) and a massive-activation **overshoot** (~10× at ~50 k steps) that relaxes ~7× by convergence, while the behavioral sink increases monotonically — the model first over-builds, then consolidates, the circuit."
- "Nullifying the query bias removes only ~5% of the Mistral sink (vs 55% reported for OpenAI GPT-2); the sink is carried by the **downstream query pathway** sharing the same `EPE₁ W_k` key, confirming the two-pathway account and showing the pathway balance is a property of the individual training run, not of the architecture."
- "(Limitation) A 50%-onset temporal-ordering test does not reach significance (Wilcoxon p = 0.81) because the alignment signals are non-monotone; establishing causal precedence over training requires a monotonicity-robust onset estimator, which we leave to future work."

---

*Artifacts: `results_e3/emergence_gpt2-small/aggregate/` — `fig_E3A_trajectories.png` … `fig_E3E_two_pathway.png`, `table_E3_1_onsets.csv`, `table_E3_2_converged_vs_paper.csv`, `trajectories.csv`, `onsets.csv`, `measurements_long.csv`, `final_massive_coords.json`, `coord_temporal_stability.csv`, `ordering_test.json`.*
