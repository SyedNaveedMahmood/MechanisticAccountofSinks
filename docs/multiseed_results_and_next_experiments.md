# Multi-Seed Table 1 Study — Results & Next Experiments
## BlackboxNLP 2026 Reproducibility Challenge · Companion to `blackboxnlp_repro_plan.md`
**Status date:** July 7, 2026 · **Deadline:** July 17, 2026 (AoE)

---

## 1. What was run

Seven independent seeds (000–006), each drawing a **fresh 300-example evaluation sample** (100 each from SST-2 / GSM8K / HumanEval, ≥40-token filter, truncated to 40; per-seed `sample_manifest.csv` confirms distinct HF source indices). The full ten-intervention Table 1 pipeline (BOS-attention metric, layers 4–11 and all-layers variants, per-domain splits) was rerun per seed. Model fixed (GPT-2 124M); seeds vary **data selection only**. Cost: ~7 GPU-hours total.

This addresses the reviewer critique that "the evaluation dataset is relatively small (300 examples)... sensitivity not thoroughly explored," and extends E5 (Evaluation robustness) with a sampling-variance axis the original plan did not include.

---

## 2. Results

### 2.1 Headline: all Table 1 conclusions are robust to data resampling

Cross-seed statistics of the relative-to-baseline percentage (mid-layers 4–11, all datasets pooled, n=7 seeds):

| Intervention | Paper % | Multi-seed mean % | Cross-seed std (pp) | Verdict |
|---|---|---|---|---|
| (a) Baseline (abs. BOS attn) | .563 | .5644 | ±0.0008 (abs) | ✅ stable |
| (b) Nullify b_Q | 44.7 | 44.54 | **0.15** | ✅ stable |
| (c) Remove First PE | 3.0 | 2.90 | **0.09** | ✅ stable |
| (d) Swap EPE | 6.2 | 6.21 | **0.08** | ✅ stable |
| (e) Swap PE (control) | 99.0 | 98.95 | 0.28 | ✅ stable |
| (f) Nullify BOS (control) | 99.7 | 99.66 | 0.03 | ✅ stable |
| (g) No MLP | 50.3 | 50.08 | 0.65 | ✅ but noisier |
| (h) No PE | 17.5 | 18.39 | **1.10** | ✅ but noisiest |
| (i) Zero Top-3 W_k | 65.2 | 65.15 | 0.10 | ✅ stable |
| (j) Zero Random W_k (control) | 100.0 | 100.06 | 0.005 | ✅ stable |

Cross-seed std of the mean BOS attention is ≤ the within-sample stderr for all surgical interventions and controls → the paper's single-sample SEs are honest estimates of total uncertainty, and **n=300 is adequately powered** for the reported effect sizes.

### 2.2 New finding 1 — Surgical vs. coarse interventions differ ~10× in sampling stability

Circuit-targeted interventions (b, c, d, i): cross-seed std 0.08–0.15 pp. Coarse interventions (g No MLP: 0.65 pp; h No PE: 1.10 pp): an order of magnitude larger. This **quantitatively substantiates** the reviewer's qualitative concern that No-MLP/No-PE "induce global distribution shifts, potentially confounding interpretability of necessity": their measured effect depends on which data you evaluate on; the circuit interventions' effect does not.

### 2.3 New finding 2 — Coarse interventions are domain-sensitive; circuit interventions are domain-invariant

Per-domain relative % (multi-seed means; seed stds in parentheses):

| Intervention | SST-2 | GSM8K | HumanEval | Max gap |
|---|---|---|---|---|
| No MLP (g) | 41.57 (±1.04) | 52.40 (±1.27) | 57.21 (±1.16) | **15.6 pp** |
| No PE (h) | 17.44 (±2.81) | 23.05 (±2.16) | 14.72 (±0.26) | **8.3 pp** |
| Nullify b_Q (b) | 44.96 (±0.24) | 43.24 (±0.33) | 45.41 (±0.15) | 2.2 pp |
| Swap EPE (d) | 6.50 (±0.22) | 7.00 (±0.27) | 5.07 (±0.02) | 1.9 pp |
| Remove First PE (c) | 2.25 (±0.17) | 2.29 (±0.26) | 4.26 (±0.14) | 2.0 pp |

The No-MLP domain gap (15.6 pp against ~1 pp seed noise) is real, not sampling artifact. **Interpretation for the paper:** a content-agnostic positional circuit predicts exactly this pattern — interventions that surgically target the circuit produce content-independent effects, while interventions that disrupt broad model computation entangle content-dependent pathways. The multi-seed data thus does double duty: robustness check *and* independent evidence that the identified circuit is the content-general component of the sink.

### 2.4 Minor observations

- Baseline sink strength is itself domain-ordered and seed-stable: SST-2 .6027 > GSM8K .5497 > HumanEval .5410 — natural language sinks hardest; worth one sentence + a hypothesis (tokenization/content entropy) in the paper.
- All-layers vs. mid-layers metric variants agree in relative terms across seeds (as in the single-seed run), further de-risking the metric-choice critique.

---

## 3. Where this leaves the schedule (Day 1 of 10)

Complete: **R0** (faithful reproduction, exact match), **E5(e)** (per-domain + now multi-seed), and an unplanned **sampling-robustness study** worth its own subsection. Ahead of plan. Remaining core: E1 (cross-scale), E2 (cross-architecture), E3 (checkpoint dynamics — start downloads now), E4 (dose-response/decomposition), E5(a,b,d).

---

## 4. New experiments unlocked by the multi-seed harness

The harness (seeded resampling → full intervention sweep → aggregate) generalizes along three axes: *what data*, *what model*, *what analysis*. Five new experiments, ranked by expected paper improvement:

### N1 — Multi-seed error bars for the generalization claims (fold into E1/E2) — **Rank 1**
Run **3 seeds per model** for the cross-scale (GPT-2 M/L/XL) and cross-architecture (OPT, GPT-Neo) sweeps instead of a single fixed sample. Every generalization table then carries cross-seed error bars, and claims like "the circuit weakens/fragments at scale" or "OPT forms a different circuit" become statistically defensible rather than anecdotal — the single strongest upgrade available to the submission's two most important sections.
**Compute:** 3× the planned E1/E2 cost → ~35–60 GPU-h total; still fits comfortably across 4 PCs in 2–3 days. **Risk:** none beyond E1/E2 themselves. **Data:** new forward passes.

### N2 — Content-generality stress test: out-of-distribution and degenerate domains — **Rank 2**
The domain-invariance of surgical interventions (§2.3) generates a falsifiable prediction: because the circuit is positional, it should survive **any** content. Add four extreme domains to the seeded sampler: (i) **random token IDs** (uniform over vocab), (ii) **shuffled natural text** (destroys syntax, preserves unigram stats), (iii) **repeated single token**, (iv) **non-English text** (e.g., Bangla/Chinese Wikipedia — heavy multi-byte tokenization). Prediction: baseline sink and surgical-intervention percentages match the natural-text values; coarse interventions drift further. A confirmed prediction is strong causal-account evidence; a violation localizes a content-dependence the paper missed.
**Compute:** ~2 GPU-h (4 domains × 2 seeds × 10 interventions on GPT-2 small). **Risk:** low. **Value:** turns §2.3 from an observation into a tested prediction.

### N3 — Sampling-variance × sequence-length interaction (merge with E5(d)) — **Rank 3**
Run the multi-seed sweep (3 seeds suffice) at 128 / 512 / 1024 tokens. Two questions: (i) do the surgical-intervention percentages stay length-invariant (the circuit predicts yes — Δ₁ is per-target-position, softmax competition grows with length, so absolute BOS attention should fall but *relative* intervention effects persist); (ii) does the coarse-intervention noise grow with length? Reports the reviewer-requested length sensitivity with error bars in one shot.
**Compute:** ~6–10 GPU-h (1024-token maps dominate). **Risk:** low. **Value:** merges two planned items (E5(d) + seed robustness) into a single stronger figure.

### N4 — Bootstrap-vs-reseed calibration (methodological, zero GPU) — **Rank 4**
Using seed 000's per-example BOS-attention values, compute bootstrap CIs over the 300 examples and compare to the empirical cross-seed distribution from all 7 seeds. If bootstrap ≈ reseed (expected, since examples are i.i.d. draws), we can state that future single-sample studies can report bootstrap CIs instead of multi-seed reruns — a small, citable methodological contribution for the reproducibility track, and it retroactively justifies error bars everywhere we can't afford reseeding (e.g., GPT-2 XL).
**Compute:** 0 (CPU, existing data — requires per-example CSVs; if only summary stats were saved, add per-example dumping to the next run). **Risk:** none. **Value:** modest but free.

### N5 — Power analysis and minimum detectable effect (zero GPU) — **Rank 5**
From the cross-seed and within-seed variances, compute the minimum detectable difference in BOS-attention (at α=0.05, power 0.8) for n ∈ {50, 100, 300, 1000}. Deliverable: one table stating what effect sizes the paper's n=300 design can and cannot resolve — e.g., whether the 0.7 pp gap between Swap PE (99.0%) and Nullify BOS (99.7%) is resolvable or noise. Directly answers the reviewer's "statistical testing is light" critique and guides our own E1/E2 sample-size choices.
**Compute:** 0 (CPU). **Risk:** none. **Value:** cheap rigor; informs N1's seed count.

### Execution notes
- N4 + N5 today (CPU, while E3 downloads run in background).
- N2 tomorrow morning on the 4080S (2 GPU-h) — its result decides how strongly to phrase the content-generality claim before E1/E2 write-ups.
- N1 replaces the single-sample plan for E1/E2 starting Day 3; pin 3 seeds (000–002) project-wide for comparability.
- N3 rides along with E5(d) on whichever PC is free from Day 4.
- **Harness to-do carried over:** log attention-to-position-2 (relocation metric, E5(a)) in every future sweep — it is still not in these outputs.

### Updated compute ledger
Multi-seed additions: N1 ≈ +25–40 GPU-h over single-sample E1/E2, N2 ≈ 2 h, N3 ≈ 6–10 h, N4/N5 ≈ 0. New core total ≈ **65–100 GPU-h** ≈ 1.5–2.5 days wall-clock across 4 PCs. Still engineering-bound, not compute-bound.

---

## 5. Framing for the paper (draft claims this study supports)

1. "All ten causal-intervention results of the original Table 1 replicate exactly and are stable under seven independent resamplings of the evaluation set (cross-seed std ≤ 0.15 pp for all circuit-targeted interventions)."
2. "Circuit-targeted interventions are an order of magnitude more sampling-stable and domain-stable than the coarse No-MLP/No-PE ablations, quantitatively confirming that the latter entangle content-dependent computation — and, conversely, that the b_Q–EPE₁–W_k pathway is the content-general component of the sink."
3. (Pending N2) "The circuit's intervention signature is invariant even on random-token and shuffled inputs, as a positional mechanism predicts."
