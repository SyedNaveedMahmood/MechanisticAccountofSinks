# E4 — Anatomy of the Residual Sink: A Graded, Decomposed Causal Account

**Target venue:** BlackboxNLP 2026 Reproducibility Track (theme: **Ablation**, with spillover into **Benchmarking/Evaluation**).
**Target paper:** Ran-Milo, Ofek & Mendel, *A Mechanistic Account of Attention Sinks in GPT-2* (arXiv:2604.14722).
**Scope:** GPT-2 small (124M) primary; GPT-2 medium spot-check. Inference-only. ~a few GPU-hours across 3 seeds.

---

## Context — why this experiment

Our generalizability axis (cross-scale GPT-2 + cross-architecture OPT/GPT-Neo/Qwen2.5, all multi-seeded) is complete and strong. The paper's remaining soft spots are all *on GPT-2 itself*, and its own **Limitation §5.3 (Secondary contributors)** names the gap explicitly:

> "Our interventions reduce the sink substantially but do not eliminate it entirely: nullifying b_Q or zeroing the massive-activation rows of W_k leaves a residual sink … Identifying [the secondary contributors] is a natural extension of our work which we leave to future work."

Concretely, Table 1 leaves **44.7%** of the sink after Nullify b_Q and **65.2%** after Zero-Top-3-W_k, unexplained. Our own multi-seed study confirmed these residuals are robust (44.54%, 65.15%), so they are a real phenomenon, not noise. Three further weaknesses compound this:

1. **All interventions are binary** (all-or-nothing). The paper never establishes a *dose-response* — the standard graded-causality criterion.
2. **Coarse interventions** (No-MLP over *all* layers, No-PE over *all* positions) induce global distribution shift; our multi-seed study already showed they are ~10× noisier and domain-sensitive, so their "necessity" is confounded.
3. **The "relocation" claim is asserted but never measured.** The abstract and §3.2.6 say disrupting a component "weakens, removes, **or relocates**" the sink and that transplanting it "transfers the sink to a new position," yet the BOS-attention metric only ever measures attention to position 1 (`intervention_analysis.py:537`). Table 1 shows the sink *vanishing* under Swap-EPE (6.2%), never *moving*.

**Intended outcome.** A self-contained paper section + reproducible code module + 3 figures and 2 tables that (i) convert the binary interventions into graded dose-response curves, (ii) *decompose and explain* the residual sink, (iii) separate circuit-specific from global effects with surgical interventions, and (iv) validate (or correct) the relocation claim. Either polarity of result is publishable under the track's negative-results policy.

---

## Scientific framing

### The four-term score decomposition (from the paper's §3.1)
The pre-softmax score from source *i* to target *j* (head *h*, layer *l*), before the 1/√D_h scale, is:

```
s_{i→j} = (x_i W_q + b_Q)(x_j W_k + b_K)^T
        = x_i W_q W_k^T x_j^T   (T1: content — depends on i and j)
        + x_i W_q b_K^T         (T2: source-dependent key-bias — depends on i only)
        + b_Q W_k^T x_j^T       (T3 = Δ_j: source-agnostic shift — depends on j only)  ← paper's circuit
        + b_Q b_K^T             (T4: constant)
```

**Key observation the paper does not exploit:** softmax over targets *j* (for a fixed source *i*) is invariant to terms constant in *j*, so **T2 and T4 cancel**. The attention distribution over targets is determined entirely by **T1(i,j) + T3(j)**. This is exactly why the paper re-centers Δ by subtracting min_j (§3.2.2). It has a direct consequence the paper misses:

- **Nullify b_Q (Table 1b)** zeros T3 (and T4). The surviving attention to position 1 is driven *purely by T1(i,1)* — the content term. So **the 44.7% residual IS the content term T1 to position 1**, not a mysterious separate circuit.
- But T1(i,1) = q_i^content · k_1, and k_1 = x_1 W_k is dominated by **EPE_1 W_k** (the same massive-activation key structure the paper identifies). So the residual is *still positional* — it just routes EPE_1 through the **downstream query** side instead of the bias side.

### Central hypothesis: the sink is TWO positional pathways sharing one key object
Both pathways deposit attention on position 1 through the **same** key-side object `k_1 ≈ EPE_1 W_k` (massive coords 138/378/447), but differ on the query side:

| Pathway | Score term | Query side | Killed by | Table 1 evidence |
|---|---|---|---|---|
| **A — bias pathway** (paper's circuit) | T3 = Δ_1 = b_Q·(EPE_1 W_k)^T | fixed query bias b_Q | Nullify b_Q (b) | 100%→44.7% |
| **B — query pathway** (the residual) | T1(i,1) = (x_i W_q)·(EPE_1 W_k)^T | input-dependent queries x_i W_q | remove EPE_1 from x_1 (c) | b leaves 44.7%; c → 3.0% |

This cleanly reconciles the whole Table 1 pattern: **Remove-First-PE (c, 3.0%)** is the strongest intervention precisely because it removes the *shared* object EPE_1 and so kills **both** pathways, whereas **Nullify b_Q (b, 44.7%)** kills only pathway A. It also reconciles the cited literature: Dial (2025) observed downstream queries align with the sink key direction (= pathway B); Sun et al. 2026 ("Spike, Sparse, Sink") showed a sink can survive massive-activation removal — our two-pathway account is the GPT-2 mechanistic instance of that decoupling.

### Falsifiable predictions (what we will test)
- **H1 (dose-response).** BOS-attention rises monotonically as we scale each knob. Scaling **b_Q** floors at ≈44.7% at α=0 (pathway B remains); scaling **EPE_1** floors at ≈3% at α=0 (both pathways die). The *different floors* are the signature of two pathways sharing EPE_1.
- **H2 (query alignment).** Downstream content-queries x_i W_q are positively aligned with EPE_1 W_k across heads/layers 4–11 (the direct analog of the paper's Fig. 2, which showed this only for b_Q). If false, pathway B is not query-alignment and the residual has another source (still publishable).
- **H3 (residual is positional).** Combining Nullify-b_Q with Remove-First-PE collapses the 44.7% residual to ≈ the (c) floor (~3%); combining with the massive-coordinate interventions reduces it further. If the residual instead survives all EPE_1 removals, it is genuinely content/token-driven — a distinct, reportable finding.
- **H4 (relocation).** Under Swap-EPE, attention to **position 2** (the transplant target) rises to near the original position-1 sink level; the sink *moves*, it does not merely vanish. If attention-to-2 stays low, the paper's "relocation" claim is **overstated** — a clean reproducibility correction.

---

## Experiments

All run on the paper's exact setup (300 examples = 100 SST-2 + 100 GSM8K + 100 HumanEval, ≥40 tokens truncated to 40; layers 4–11; second-half BOS metric) via the existing sampler, and repeated over **seeds 0,1,2** (data resamples, matching our project-wide multi-seed convention) for error bars.

### E4.1 — Graded dose-response (H1) — *the headline figure*
Replace binary interventions with a scale parameter α ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5} on three knobs:
- **(k1) Scale b_Q** — `bq = α·bq` in every layer (α=0 reproduces Table 1b; α=1 reproduces baseline).
- **(k2) Scale the EPE_1 direction** in the layer-0 MLP output at position 0 — reuse the projection machinery of `intervention_d_swap_epe_normalized` (`intervention_analysis.py:348`) but *scale* the EPE_1 component by α instead of swapping it.
- **(k3) Scale only the massive coordinates** {138,378,447} of the layer-0 MLP output at position 0 by α — isolates the massive-activation channel directly.

**Deliverable:** one 3-panel dose-response figure (BOS-attn vs α, one curve per knob, seed-mean ± band). **Analysis:** Spearman ρ per curve (monotonicity test); report the α=0 floors and contrast k1's floor (~44.7%) with k2/k3's floor (~3%) as the two-pathway signature.

### E4.2 — Residual decomposition + query–EPE_1 alignment (H2) — *the mechanism*
Two complementary measurements, both from instrumented forward passes (no intervention → no numerical ambiguity):
- **(a) Exact score attribution.** For each (layer, head, second-half source i), compute T1(i,·) and T3(·) explicitly and the position-1 *advantage* `adv_j = [T1(i,j)+T3(j)] − mean_j[...]`. Decompose adv_1 into its T1 (content) and T3 (Δ) components; report the per-(layer,head) share of the position-1 advantage carried by T3 vs T1. Sanity assert: T1+T2+T3+T4 reproduces the full HF score to < 1e-4.
- **(b) Query-alignment histogram** — the analog of the paper's Fig. 2 for *downstream queries*: per head/layer, cos(x_i W_q, EPE_1 W_k) for second-half sources i (vs. the same for target positions j>1 as control). Extends `_compute_perhead_epe_cosine` (`experiments_single_input.py:255`) by substituting real content-query projections for b_Q.

**Deliverable:** one figure (query-alignment dual histogram, mirroring Fig. 2) + a per-(layer,head) heatmap of the T3-vs-T1 share; a table row stating "the residual after Nullify-b_Q is X% attributable to the content term routed through EPE_1's key."

### E4.3 — Residual localization via combined interventions (H3)
Compose *existing, validated* interventions (the harness already supports stacking `attn_kwargs` + `modify_mlp_fn` + `pos_enc` edits in `run_intervention_loop`, `intervention_analysis.py:260`), so no risky new surgery:
- **b ∧ c** (Nullify b_Q + Remove-First-PE) → predicted ≈ 3%.
- **b ∧ i** (Nullify b_Q + Zero-Top-3-W_k) → predicted well below 44.7%.
- **b ∧ d** (Nullify b_Q + Swap-EPE) → predicted position-1 collapse + relocation to position 2 (feeds E4.5).
- **b ∧ i ∧ d** (all three main components) → tests whether a floor remains (a genuine third contributor) or the sink is fully accounted for.

**Deliverable:** a combined-intervention table (BOS-attn, %base, seed-SE), interpreted against the two-pathway prediction.

### E4.4 — Surgical vs. coarse interventions
Disentangle circuit-specific from global effects (addresses the confound our multi-seed study quantified):
- **First-layer-only MLP skip** (`modify_mlp_fn` zeros the layer-0 MLP output only) vs. the paper's all-layer No-MLP (g).
- **Position-1-only PE zeroing** (`pos_enc[0][0] = 0`) vs. the paper's all-position No-PE (h), and vs. the existing swap-based Remove-First-PE (c).

**Deliverable:** two extra table rows placed beside the coarse originals, with the seed-variance gap (surgical should be far more stable), quantifying how much of the coarse effect is global rather than circuit-specific.

### E4.5 — Relocation validation (H4)
Generalize the metric from "attention to position 1" to **"attention to an arbitrary target position"** (add a `target_pos` argument to `compute_bos_attention_metric`, `intervention_analysis.py:504`). Under Swap-EPE (and b∧d), measure attention to **position 2** (the transplant target) from second-half tokens. Define a **relocation ratio** = (attn-to-2 under swap − attn-to-2 baseline) / (baseline attn-to-1). 

**Deliverable:** a small figure/table showing whether the sink moves to position 2 or merely disappears — directly validating or correcting a stated-but-unmeasured claim.

### E4.6 — (Optional, if time) Functional cost of each pathway
Retain LM logits during the intervention passes and report per-intervention cross-entropy on the same 300 examples. Question (connecting to §5.4 and Barbero et al. 2025 on over-mixing): does removing pathway A vs. pathway B cost the model differently? Cheap (same passes, keep logits), cut first if the schedule tightens.

---

## Implementation

**Design principle:** do **not** touch the validated Table 1 code paths. Add a new driver module and make only *backward-compatible* parameter additions to the shared attention function.

### New file: `residual_sink_analysis.py`
The E4 driver + analyses + plotting. Imports and reuses:
- `get_initial_embeddings`, `run_intervention_loop`, `manual_self_attention_new`, `compute_bos_attention_metric`, `INTERVENTIONS` (from `intervention_analysis.py`).
- `sample_benchmark_datasets`, `DEFAULT_*` (from `datasets_loader.py`).
- massive-coordinate identification (`massive_activations_analysis` logic, `experiments_single_input.py:367`) and the per-head alignment machinery (`_compute_perhead_epe_cosine`, `experiments_single_input.py:255`).
- multi-seed aggregation/plot pattern from `run_table1_multiseed.py`.

CLI mirrors the existing harnesses: `--mode {dose_response, decomposition, combined, surgical, relocation}`, `--seeds 0,1,2`, `--model-name gpt2`, `--output-dir results`, `--sample-size`, `--cut-length`. Outputs under `results/residual_sink_analysis/…` with per-seed dirs + an `aggregate/` dir (CSVs + figures), matching current conventions.

### Minimal backward-compatible edits to `intervention_analysis.py`
- `manual_self_attention_new`: add `query_bias_scale: float = 1.0` (applied as `bq = query_bias_scale * bq`; default preserves behavior; `intervene_query_bias=True` stays equivalent to scale 0). Add an optional instrumentation flag that returns the per-head T1/T3 score tensors for E4.2 (guarded, off by default).
- `compute_bos_attention_metric`: add `target_pos: int = 0` (default preserves the position-0/BOS behavior); E4.5 passes `target_pos=1`.
- New `modify_mlp_fn` **factories** in the new module (not edits): `scale_epe_direction(alpha)`, `scale_massive_coords(alpha, coords)`, `zero_layer0_mlp()` — all follow the exact pattern of the existing `_modify` closures in `intervention_d/e`.

### Reproducibility discipline
fp32 (to match the paper's tight SEs); pinned seeds 0,1,2; per-run config + `sample_manifest.csv`; every figure regenerable by a single `--plot-only` pass; HF revision pinned. Raw-PyTorch harness kept for consistency with the rest of the project (an optional NNsight port for the NDIF award is out of scope for E4 and tracked separately).

---

## Deliverables (paper-ready)
- **Fig E4-A:** dose-response curves (3 knobs) — monotonicity + contrasting floors.
- **Fig E4-B:** query–EPE_1 alignment histogram (Fig. 2 analog) + T3-vs-T1 per-head heatmap.
- **Fig E4-C:** relocation (attention-to-position-2 under Swap-EPE).
- **Table E4-1:** combined + surgical interventions (BOS-attn, %base, seed-SE), vs. Table 1 originals.
- **Table E4-2:** residual attribution summary (fraction of residual explained by the query pathway) + optional perplexity column.
- **Prose result:** "The residual sink is a second positional pathway that shares the massive-activation key structure (EPE_1 W_k) but is driven by downstream query alignment rather than the query bias; it is graded, monotone, and collapses when EPE_1 is removed from position 1 — resolving the paper's §5.3 open question and providing a GPT-2 instance of the massive-activation/sink decoupling reported by Sun et al. (2026)."

## Statistical rigor
- Seeds 0,1,2 → mean ± cross-seed band on every number.
- Report **per-(layer,head) distributions** (96 cells), not just pooled means; effect sizes (Cohen's d) for pathway-A-only vs. pathway-B-only attention; paired Wilcoxon across cells with Holm/BH correction for the alignment and decomposition claims.
- Spearman ρ for each dose-response curve.

## Verification (how we'll know it works)
1. **Smoke test** (1 seed, `--sample-size 10`): assert baseline BOS-attn ≈ .563 and Nullify-b_Q ≈ .251/44.7% (reproduces Table 1a/b), confirming the reused harness is intact.
2. **Decomposition identity check:** assert T1+T2+T3+T4 equals the HF eager-attention pre-softmax score to < 1e-4 on a fixed sentence; assert `query_bias_scale=0` reproduces intervention (b) and `=1` reproduces (a) exactly.
3. **Dose-response sanity:** verify k1 curve passes through the (b) point at α=0 and the (a) point at α=1; verify monotonicity (ρ>0.95 expected).
4. **Full run** (seeds 0,1,2, full 300) on one card (4080S or 3090); regenerate all figures/tables via one `--plot-only` command.
5. **Spot-check** the k1/k2 dose-response on GPT-2 **medium** (reuse cross-scale checkpoints) to confirm the two-floor structure is not GPT-2-small-specific.

## Compute & scheduling
GPT-2 small, 40-token inputs, ~two dozen extra forward passes per example → **~1–2 GPU-h per seed**, ≈ a few GPU-h total across 3 seeds; fits any of the four cards (even CPU). Assign E4 to one card (e.g., RTX 4080 Super / RTX 3090) and leave the 5090/4090 free for any remaining generalization or dynamics work. **Engineering is the binding cost (~2–3 days):** the score-decomposition instrumentation and the query-alignment measurement are the only non-trivial pieces; everything else composes existing, validated machinery.

## Risks & mitigations
- **Numerical fidelity of new instrumentation** → the decomposition identity assert (Verification 2) gates correctness before any analysis.
- **EPE_1-direction scaling only well-defined at layer 0** (consistent with the paper's own Swap-EPE) → state this scope explicitly; k3 (massive-coord scaling) is the layer-0-agnostic robustness variant.
- **Query-alignment confound** (queries at layer *l* carry accumulated positional signal) → measure the *content* projection x_i W_q only, keep the bias pathway separate, and restrict to second-half sources as the paper does.
- **A residual could survive all EPE_1 removals** (H3 false) → this is itself a clean finding (a genuine third, non-positional contributor); pre-register both outcomes.
- **Scope creep** → E4.6 (perplexity) is the designated cut; E4.1–E4.5 are the committed core.

## Addendum — Cross-scale support (GPT-2 small / medium / large / XL)

**Context.** E4's account is GPT-2-family-specific, so GPT-2 small stays the headline, but the same harness should run on medium/large/XL to test whether the two-pathway signature strengthens or fragments with scale (paper §5.1). The harness already accepts any `--model-name`, but it hardcodes the layer band to 4–11 (calibrated for the 12-layer small) and forces fp32. Four small, backward-compatible changes make cross-scale runs correct and convenient; **GPT-2 small results are unchanged** (the new default band reduces to exactly 4–11 for a 12-layer model).

**Changes.**
1. **Depth-aware layer band.** Add `--layer-mode {scaled,fixed}` (default `scaled`). `scaled` excludes the first 3 and last 1 layers → band `[3, num_layers−1)`, which reduces to exactly `[3,11)` for small (paper-faithful, small unchanged) and extends appropriately for deeper models; `fixed` forces `[3,11)` for strict same-layer comparability. Implement via optional `layer_start`/`layer_end` on `compute_bos_attention_metric` (`intervention_analysis.py`; defaults `None` → existing scope logic, so Table-1 callers are unaffected) and thread a `band=(start,end)` tuple through the E4 runners (`run_dose_response`, `run_intervention_set`, `run_relocation`, and `collect_decomposition_and_alignment`, which already accepts `layer_start`/`layer_end`). Add a `compute_band(num_layers, layer_mode)` helper.
2. **`--dtype {float32,float16,bfloat16}`** (default `float32`, matching the paper's tight SEs). `load_model` applies it, so XL (1.5B) fits comfortably on the 16 GB card if needed. Keep fp32 for the reported numbers.
3. **Robust massive-coord detection.** `identify_massive_coords` keeps the >3σ criterion (→ 138/378/447 for small) but falls back to top-3-by-|EPE₁| when fewer than 3 coordinates exceed the threshold, so the `scale_massive` / `zero_top3_wk` knobs are well-defined on every checkpoint.
4. **Record run config** (model, dtype, layer band, massive coords) to a per-seed `run_config.json` for reproducibility.

**README.** Add a cross-scale block with copy-paste commands for all four sizes:
```
python residual_sink_analysis.py --mode all --model-name gpt2        --output-dir results
python residual_sink_analysis.py --mode all --model-name gpt2-medium --output-dir results
python residual_sink_analysis.py --mode all --model-name gpt2-large   --output-dir results
python residual_sink_analysis.py --mode all --model-name gpt2-xl --dtype float32 --output-dir results
```
Note the default `scaled` band (with `--layer-mode fixed` for strict comparability) and that each model writes to its own `results/residual_sink_<model>/` directory.

**Verification.** Extend the offline smoke test with a random ~24-layer/1024-dim (medium-shaped) model: assert `compute_band(12,'scaled')==(3,11)` and `compute_band(24,'scaled')==(3,23)`; run one dose-response + decomposition pass end-to-end with no shape errors and the decomposition identity assert passing. (Real GPT-2 numbers run on the user's GPUs — HF is network-blocked in this container.)

## Git workflow
Branch `claude/blackboxnlp-reproducibility-planning-1nxkos` (per project rules). Pull latest `main` first, implement, commit with descriptive messages, push with `-u origin` (retry with backoff on network errors). No PR unless requested.
