# BlackboxNLP 2026 Reproducibility Challenge — Project Plan
## Target paper: "A Mechanistic Account of Attention Sinks in GPT-2: One Circuit, Broader Implications for Mitigation" (Ran-Milo, Ofek & Mendel, arXiv:2604.14722)

**Submission deadline:** July 17, 2026, 11:59 PM AoE (OpenReview, BlackboxNLP Special Track)
**Today:** July 7, 2026 → **10 working days**
**Hardware:** 4 independent PCs — RTX 5090 (32 GB), RTX 4090 (24 GB), RTX 3090 (24 GB), RTX 4080 Super (16 GB)

---

## 1. What the track rewards

The Reproducibility Challenge asks for rigorous robustness checks of published interpretability results along four themes: **Baselines**, **Ablation**, **Generalizability**, and **Benchmarking/Evaluation**. Negative results are explicitly welcomed as evidence about the boundary conditions of claims. Accepted papers appear in the ACL Anthology workshop proceedings (cannot be published elsewhere).

**NDIF Best Submission Award ($500):** requires (i) a public codebase enabling easy replication, (ii) NNsight as the primary tool for accessing model internals (NDIF remote execution optional), and (iii) novel insights beyond the original work. **Decision: implement our entire intervention harness in NNsight from day one.** This costs almost nothing (the original interventions are forward-pass patches, which map 1:1 onto NNsight's tracing API) and makes us award-eligible while satisfying the "tooling environments (NDIF/nnsight)" bullet of the Generalizability theme.

---

## 2. Paper summary (what we must understand cold)

**Claim.** In GPT-2 (124M, learned absolute positional embeddings + learned query biases), the first-position attention sink is implemented by one circuit: the interaction of
1. the learned **query bias** b_Q,
2. **EPE₁** = MLP⁽¹⁾(p₁) + p₁, the first-layer MLP's transformation of the first positional embedding, which carries *massive activations* at coordinates 138, 378, 447, and
3. structure in the **key projection** W_k whose bias-projection magnitude |b_Q · W_k[:,d]| is largest at exactly those coordinates.

The source-agnostic score shift Δ⁽ˡ⁾_{j,h} = b_Q W_kᵀ x_jᵀ is anomalously large for j=1, creating a content-independent prior to attend to position 1.

**Evidence.** Four correlational analyses (Figs. 1–4) + ten causal interventions (Table 1, BOS-attention metric = mean attention from second-half tokens to position 1, layers 4–11). Key numbers to match: Baseline .563; Nullify b_Q .251 (44.7%); Remove First PE .017 (3.0%); Swap EPE .035 (6.2%); Swap PE .557 (99.0%, control); Nullify BOS token .561 (99.7%, control); No MLP .283; No PE .099; Zero Top-3 W_k .367; Zero Random W_k .563 (control).

**Setup.** GPT-2 124M (HF `openai-community/gpt2`), 300 examples (100 each from SST-2, GSM8K, HumanEval), truncated to exactly 40 tokens, layers 4–11. Code: github.com/YuvMilo/MechanisticAccountofSinks.

**Stated limitations (our extension surface):**
- §5.1 Scope: only GPT-2 small; scale and other architectures unexamined.
- §5.2 Learning dynamics: static post-hoc analysis of one checkpoint.
- §5.3 Secondary contributors: interventions leave a residual sink (e.g., 44.7% after nullifying b_Q) — unexplained.
- §5.4 Mechanism vs. function: no downstream-impact analysis.

---

## 3. Fit assessment: **9/10**

- **Why it fits:** Core mech-interp with causal interventions on a 124M model — the paper's own limitations map one-to-one onto the track's Ablation/Generalizability/Evaluation themes; public code + data make faithful reproduction near-certain; all follow-ups are inference-only.
- **Why it fits:** The headline claim ("sinks arise through *distinct circuits* across architectures; each component is individually dispensable") is a strong, falsifiable generalization claim the authors support only by citing other papers — we can test it empirically, and either outcome (circuit transfers / circuit differs) is a publishable result under the track's negative-results policy.
- **Risks:** The paper is recent and already rigorous (controls included), so our value must come from extension, not error-hunting; the circuit is tied to GPT-2-specific components (learned absolute PE, query bias), constraining which transfer targets are informative — mitigated by choosing OPT/GPT-Neo, which share exactly those ingredients.

---

## 4. Compute feasibility: **Yes**

| Workload | Model(s) | VRAM (fp16 inference + hooks) | Fits on |
|---|---|---|---|
| Faithful reproduction | GPT-2 124M | < 1 GB | any PC (even CPU) |
| Cross-scale | GPT-2 355M / 774M / 1.5B | 0.7–3.2 GB | any PC |
| Cross-architecture | OPT-125M/350M/1.3B/2.7B, GPT-Neo-125M/1.3B | ≤ 6 GB | any PC |
| (Optional stretch) | OPT-6.7B | ~13 GB | 3090/4090/5090 |
| Training dynamics | Stanford CRFM "Mistral" GPT-2 small+medium checkpoints (5 seeds, 600+ ckpts; we subsample ~20/run) | ≤ 1 GB per ckpt | any PC; parallelize seeds across PCs |

The full 300-sequence × 40-token intervention suite is **minutes per model per intervention** — the entire project is bounded by engineering and writing time, not GPU-hours. Four independent PCs = four parallel model families. **Verdict: comfortably viable before July 17.**

Note on the training-dynamics workload: we do **not** train anything. Stanford CRFM's Mistral project publicly released GPT-2-small and GPT-2-medium pre-training runs with multiple seeds and hundreds of intermediate checkpoints on Hugging Face — same architecture family as the paper (learned absolute PEs, query biases), so the paper's exact analysis pipeline applies unchanged. **Day-1 action item: verify checkpoint availability and license before committing to E3; if unavailable, E3's slot goes to the OPT-6.7B stretch and deeper E4.**

---

## 5. Core results to recreate (faithful reproduction — Days 1–3)

Reproduce, in order of importance:

1. **Table 1 (all ten interventions + BOS-attention metric).** This is the paper's causal backbone. Match within reported ±2 SE. Reimplement in **NNsight** rather than raw PyTorch hooks (award requirement); sanity-check 2–3 interventions against the authors' original codebase to confirm equivalence.
2. **Figure 1 / §3.2.2:** Δ⁽ˡ⁾_{j,h} histograms (first position vs. rest), with the min-subtraction re-centering.
3. **Figure 2 / §3.2.4:** cosine alignment of b_Q with EPE_j W_k (j=1 vs. j>1).
4. **Figure 3 / §3.2.3:** EPE_i vs. net added positional signal N_i (median cosine > 0.82 everywhere, > 0.99 at position 1).
5. **Figures 4/7 / §3.2.5, App. A.2:** massive-activation coordinates of EPE₁ — confirm indices **138, 378, 447** and the coordinate-level γ_h[d] separation.
6. Reconstruct the exact evaluation set (SST-2/GSM8K/HumanEval, ≥40-token filter, truncate to 40) from their released data; verify seed-level determinism of the sampling.

Deliverable: a "Reproduction" section reporting per-intervention deltas vs. published numbers, plus any discrepancies (data filtering, layer averaging, re-centering choices).

---

## 6. Extension strategy

Attack the paper along its own limitations, mapped to track themes:

- **Generalizability (§5.1):** scale (E1) and architecture (E2).
- **Ablation (§5.3):** the unexplained residual sink (E4).
- **Generalizability→dynamics (§5.2):** circuit emergence over pre-training via public checkpoints (E3).
- **Benchmarking/Evaluation (§5.4 + metric robustness):** downstream cost of interventions and alternative sink metrics (E5).
- **Cross-cutting:** everything in NNsight, released as an open repo with configs/seeds → NDIF award eligibility.

---

## 7. Proposed experiments (5)

### E1 — Cross-scale replication: GPT-2 medium / large / XL
**Theme: Generalizability.** Run the full pipeline (Δ analysis, EPE construction, alignment, massive-coordinate identification, all ten interventions) on GPT-2 355M/774M/1.5B. Questions: Does the b_Q–EPE₁–W_k pathway persist? Do massive-activation coordinates remain few and identifiable? Does the % of sink removed by each intervention shift with scale (their §5.1 conjecture: strengthen / fragment / be replaced)?
**Compute:** trivial; one PC, ~1 day including plotting. **Risk:** low. **Outcome value:** either result extends the paper.

### E2 — Cross-architecture ingredient test: OPT and GPT-Neo
**Theme: Generalizability; directly tests the headline claim.** The paper asserts the circuit is architecture-specific but never tests a *second* architecture with the *same* ingredients. OPT (125M/350M/1.3B/2.7B) has learned absolute PEs **and** attention biases — do the same three components co-adapt into the same circuit, or does optimization find a different one even when the GPT-2 ingredients are available? GPT-Neo-125M/1.3B (learned absolute PEs, **no** attention biases) serves as the dispensability control: with b_Q absent, what carries the source-agnostic shift (e.g., a constant-feature slice of W_Q, as the authors speculate in fn. 11)? Adapt the Δ decomposition accordingly (Δ generalizes to any rank-1 source-independent term).
**Stretch (only if E1–E3 land early):** attempt an "effective positional signal" proxy for a RoPE model (e.g., Pythia-160M) — construct the net first-layer positional contribution empirically (with-rotation vs. without-rotation forward difference) and test whether an analogous alignment structure exists. High conceptual value, high risk; scope to one model, one figure.
**Compute:** ≤ 6 GB per model; two PCs, ~2–3 days. **Risk:** medium (analysis adaptation for no-bias case needs care). **Outcome value:** highest — confirms or bounds the paper's central generalization claim; a negative result ("same ingredients, different circuit") is arguably *more* interesting.

### E3 — Circuit emergence during pre-training (Stanford Mistral checkpoints)
**Theme: Generalizability/dynamics; addresses §5.2 with zero training compute.** Using public GPT-2-small/medium pre-training checkpoints (5 seeds, subsample ~20 checkpoints log-spaced per run): track (i) when massive activations appear in EPE₁, (ii) when b_Q–(EPE₁W_k) alignment forms, (iii) when the BOS-attention sink appears, and (iv) their temporal ordering — does alignment precede the sink (circuit causes sink) or co-emerge? Test seed-consistency of the massive-activation coordinate indices.
**Compute:** ~100–200 forward-analysis passes total; parallelize seeds across 2 PCs, ~2 days. **Risk:** medium (checkpoint availability — verify Day 1). **Outcome value:** most *novel* contribution; nothing in the original paper touches dynamics.

### E4 — Anatomy of the residual sink: surgical, graded & decomposed interventions
**Theme: Ablation; addresses §5.3 and the coarse-intervention critique.** Nullifying b_Q leaves 44.7% of the sink. Systematically:
(a) **Combined interventions** (b_Q=0 ∧ zero top-3 W_k ∧ swap EPE) — is the residual additive or a second circuit?
(b) **Dose-response curves**: scale EPE₁ by α ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5} and, separately, reproject only the massive-activation coordinates (138/378/447) of m₁ at graded strengths; plot BOS attention vs. α. A monotone dose-response is a classic causality criterion the original paper never establishes.
(c) **Surgical replacements for the coarse interventions**: the paper's "No MLP" (all layers) and "No PE" (all positions) induce global distribution shift, confounding "necessity." Run first-layer-only MLP skip and position-1-only PE removal, and compare against the coarse variants to separate circuit-specific from global effects.
(d) **Variance decomposition / regression**: per (layer, head), regress BOS-attention (or pre-softmax score to position 1) on the four score terms — content term x W_q W_kᵀ xᵀ, source-dependent bias term x W_q b_Kᵀ, Δ₁ = b_Q W_kᵀ x₁ᵀ, and constant b_Q b_Kᵀ — reporting variance explained by Δ₁ vs. the rest, to quantitatively attribute the residual sink.
(e) **Layer-wise** b_Q nullification and **head-wise** residual decomposition to identify carrier heads; report per-head/per-layer distributions (not just pooled means) with effect sizes for every intervention.
**Compute:** GPT-2 small only; one PC, ~2 days. **Risk:** low mechanically, medium interpretively. **Outcome value:** turns the paper's acknowledged gap into a concrete, quantitative finding and pre-empts the strongest methodological critiques of the original interventions.

### E5 — Evaluation robustness and functional cost
**Theme: Benchmarking/Evaluation; addresses §5.4 + metric fragility.** (a) **Relocation validation** — the paper claims transplanting components "relocates" the sink but only ever measures attention to position 1. Define a **swap-target attention metric** (average second-half attention to position 2, or to whichever position receives the transplanted EPE) and verify the sink actually *moves* under the EPE swap rather than merely vanishing. This is a missing validation of a stated claim — exactly what the track wants. (b) Report each intervention's effect on **language-modeling loss/perplexity** on the same 300 examples (does killing the sink hurt the model? — connects to sink-necessity results they cite); (c) alternative sink metrics (ε-threshold sink rate à la Gu et al.; attention-entropy; sink mass at positions >1) to check conclusions aren't metric artifacts; (d) sequence-length sensitivity across 40 → 128 → 512 → **1024** tokens (GPT-2's full context) since all published numbers use exactly 40 tokens — test whether Δ₁ dominance and intervention effect sizes persist at long context; (e) per-domain breakdown (SST-2 vs. GSM8K vs. HumanEval) instead of pooled statistics, with paired statistical tests and effect sizes across heads/layers rather than histograms + SEs alone.
**Compute:** trivial; folds into E1/E2 runs. **Risk:** low. **Outcome value:** strengthens every other section's credibility and directly closes a validation gap in the original paper.

---

## 7b. Reviewer-informed additions (from independent AI review of the target paper)

An independent review of the paper (est. 6.3/10, ICLR-calibrated) flags gaps that map directly onto cheap experiments. We adopt them because a reproducibility submission that *answers a real reviewer's open questions with data* has a built-in significance argument:

| Reviewer critique / question | Where we address it | Cost |
|---|---|---|
| "Relocation" of the sink is claimed but never measured (BOS metric only tracks position 1) | E5(a): swap-target attention metric | ~free |
| No dose-response evidence; interventions are all-or-nothing | E4(b): graded EPE₁ scaling + massive-coordinate-only reprojection | ~free |
| "No MLP" / "No PE" are coarse, globally disruptive interventions | E4(c): first-layer-only MLP skip; position-1-only PE removal | low |
| Residual sink unattributed; how much does Δ₁ explain vs. other score terms? | E4(d): variance decomposition / regression over the four score terms | low |
| Pooled layer/head statistics may mask heterogeneity; light statistical testing | E4(e) + E5(e): per-head/per-layer distributions, effect sizes, paired tests | ~free |
| Sequence-length sensitivity unexplored (all results at 40 tokens) | E5(d): 40 → 1024 sweep | low |
| No other GPT-2 sizes; no b_Q-free variants; what substitutes for b_Q? | E1 + E2 (already core) | — |
| EPE generalization to RoPE/ALiBi? Training-time emergence of alignment? | E2 stretch (Pythia RoPE proxy) + E3 (already core) | medium |

Framing for the paper: our submission does not merely reproduce — it operationalizes the open questions an expert reviewer raised about the original work, and answers most of them empirically.

---

## 8. Ranking (expected improvement to our submission)

| Rank | Experiment | Why |
|---|---|---|
| **1** | **E2 — Cross-architecture (OPT / GPT-Neo)** | Directly stress-tests the paper's most consequential claim ("distinct circuits across architectures") with the same ingredients present/absent; both positive and negative outcomes are headline results for a reproducibility track. |
| **2** | **E1 — Cross-scale (GPT-2 M/L/XL)** | Lowest-risk, reviewer-expected generalization; addresses stated limitation §5.1; establishes credibility that our harness is faithful before bolder claims. |
| **3** | **E3 — Emergence over pre-training** | Highest novelty per GPU-hour; addresses §5.2 with public checkpoints; multi-seed consistency of massive-coordinate indices is a genuinely new datapoint. Ranked below E1/E2 only because of the availability risk. |
| **4** | **E5 — Evaluation robustness, relocation & downstream cost** | Cheap, folds into other runs; the relocation metric closes a *stated-but-unmeasured* claim in the original paper, and the perplexity-cost analysis touches §5.4. Rank rises if relocation turns out non-trivial (sink vanishes rather than moves). |
| **5** | **E4 — Residual sink anatomy (dose-response, surgical, decomposed)** | Now the most methodologically rich section thanks to reviewer-informed additions, but still the most open-ended. Do the dose-response and variance-decomposition sub-parts first — they are cheap and near-guaranteed to yield a figure; the head-wise attribution can be cut if time runs short. |

**Minimum viable submission:** Reproduction + E1 + E2 + E5. **Target submission:** all five. **Award play:** everything in NNsight, public repo, README with exact configs/seeds.

---

## 9. Ten-day schedule (July 7–17, AoE)

| Days | Work | Machine allocation |
|---|---|---|
| 1–2 | Clone authors' repo; verify data reconstruction; verify Mistral checkpoint availability (E3 go/no-go); build NNsight harness; reproduce Table 1 + Figs. 1–4 on GPT-2 small | 4080S (repro), others idle/setup |
| 3–4 | E1 on GPT-2 M/L/XL; start E5 metrics inside the same runs | 5090: XL; 4090: L; 3090: M |
| 4–6 | E2: OPT family + GPT-Neo (adapt Δ decomposition for no-bias case) | 5090 + 4090 |
| 5–7 | E3: checkpoint sweeps, 5 seeds × ~20 ckpts | 3090 + 4080S |
| 7–8 | E4: combined/layer/head interventions on GPT-2 small; finish E5 (length sweep, per-domain) | 4080S |
| 8–10 | Freeze results; write paper (repro section, per-theme extension sections, negative results stated plainly); clean + release repo; OpenReview submission with ≥1 day buffer | — |

**Reproducibility discipline for our own paper:** fixed seeds per experiment, config files per run, exact HF model revisions pinned, fp32 vs. fp16 numerics noted (their Table 1 SEs are tight — run our repro in fp32 to match), all figures regenerable via one script.

---

## 10. Key risks & mitigations

1. **Authors' repo missing/incomplete** → reimplement from the paper's equations (fully specified in §2–3); note deviations transparently. Verify Day 1.
2. **Mistral checkpoints unavailable/licensing issue** → drop E3, promote OPT-6.7B stretch + deeper E4.
3. **Exact-number mismatch in reproduction** → report both numbers with a discrepancy analysis (data filter, layer range, dtype); the track values this as a finding, not a failure.
4. **E2 no-bias adaptation ambiguity (GPT-Neo)** → pre-register the generalized decomposition (any source-independent rank-1 score term) before looking at results; state it as a methodological contribution.
5. **Scope creep** → E4 is the designated cut if time runs short.
