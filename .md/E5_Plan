# E5 — Evaluation Robustness & the Functional Cost of Sink Removal
## Plan for the BlackboxNLP 2026 Reproducibility Track submission (deadline July 17 AoE)

---

## 1. Context

**Target paper:** Ran-Milo, Ofek & Mendel, *A Mechanistic Account of Attention Sinks in GPT-2* (arXiv:2604.14722). The paper attributes the first-position sink to one circuit (`b_Q – EPE_1 – W_k`) via ten causal interventions, all evaluated with **one metric** (BOS-attention: mean second-half attention to position 0, layers 4–11), at **one sequence length** (exactly 40 tokens), with **no measurement of what interventions cost the model functionally** (its own Limitation §5.4, and its subtitle — "Broader Implications for **Mitigation**" — is never operationalized).

**Project state:** R0 (exact reproduction), E1 (cross-scale GPT-2), E2 (cross-arch OPT/GPT-Neo/Qwen2.5), E3 (emergence dynamics over Mistral checkpoints), E4 (residual-sink anatomy → the two-pathway account), and a 7-seed sampling-robustness study are complete on branch `E3-E4`. Of the originally scoped E5 items, **(a) relocation** is done (E4.5), **(e) per-domain** is done (multiseed §2.3), and E4.6 has a minimal 4-config perplexity table. Still open: **(c) alternative sink metrics**, **(d) sequence-length sensitivity**, a real **(b) functional-cost analysis**, plus the multiseed doc's proposed **N2 content-generality stress test** and free statistical upgrades **N4/N5**.

**Why this E5 maximizes publishability:** it is the *Benchmarking/Evaluation* theme section (the one track theme our submission doesn't yet own), it attacks the paper's evaluation methodology itself (single metric, single length, mechanism-without-function), and every sub-experiment carries a pre-registered, mechanism-derived falsifiable prediction — so either polarity is a publishable result under the track's negative-results policy. It also converts the paper's un-cashed "mitigation" claim into a measured cost–benefit frontier, which no one in the sink literature has published for these interventions.

**Compute:** RTX 5090 (32 GB), 4090 (24 GB), 3090 (24 GB), 4080 Super (16 GB). Total E5 budget ≈ 8–14 GPU-h → half a day wall-clock across the 4 PCs.

**Scope decisions (defaults adopted — user question dialog was unavailable; override anytime):**
- **Models:** GPT-2 small (124M) is the primary for all four sub-experiments (paper-faithful); a **GPT-2 medium spot-check** of the metric battery + mitigation frontier (~1 extra GPU-h, zero extra engineering — the module takes `--model-name`) is included in the run protocol. OPT/Neo/Qwen ports are explicitly out of scope for E5 (deadline risk).
- **Branches:** develop on the designated `claude/blackboxnlp-e5-experiment-kiuw0v` and also push the identical commits to a new **`E5`** branch, as explicitly requested (satisfies both the session rules and the E3/E3-E4 naming convention).

---

## 2. The four sub-experiments (+ free statistics)

### E5.1 — Metric multiverse: is Table 1 metric-invariant? *(theme core)*
**Gap:** every conclusion rests on the single BOS-attention mean. Gu et al. (2025) use an ε-threshold sink *rate*; entropy and redistribution views exist in the literature; pooled means can mask head heterogeneity.
**Method:** one pass over the ten Table-1 interventions (300 examples × seeds 0,1,2, 40 tokens), computing a **battery of 6 metrics** from the same attention maps:
1. `bos_attn` — the paper's metric (reference point; must reproduce Table 1).
2. `sink_rate@ε` — fraction of (layer, head) cells whose mean second-half attention to pos 0 exceeds ε ∈ {0.2, 0.3, 0.5} (Gu-et-al-style prevalence).
3. `attn_entropy` — mean normalized Shannon entropy of second-half attention rows (∈[0,1], length-comparable).
4. `bos_rank` — mean rank of position 0 in each query's attention distribution (ordinal, outlier-robust).
5. `local_mass` — attention mass on positions 1–4 vs. 5+ (redistribution profile: where does the mass *go* when the sink dies? subsumes the carried-over "attention-to-position-2" to-do).
6. `head_gini` — concentration of sink mass across heads (is the sink carried by few heads or all?).

**Analyses:** (i) metric × intervention matrix (normalized %-of-baseline) + **Kendall τ concordance** of intervention rankings between every metric pair; (ii) **intensive-vs-extensive decomposition**: per-(layer,head) scatter of baseline vs. post-intervention BOS attention for key interventions — does Nullify-b_Q shrink all sink heads proportionally (intensive) or kill a subset (extensive)? The pooled mean cannot distinguish these; either answer is a new mechanistic datapoint.
**Pre-registered predictions:** H-M1: rankings concordant (τ ≥ 0.8) across metrics — conclusions are not metric artifacts. H-M2: circuit interventions act on the intensive margin (a graded logit shift predicts proportional shrinkage, not head death).

### E5.2 — Length generalization & the logit-competition law *(the "scientific law" figure)*
**Gap:** all published numbers use exactly 40 tokens; GPT-2's context is 1024.
**Key theory hook (novel):** the paper's Δ₁ is a *fixed additive logit advantage*, and x₀ is **exactly length-invariant** under causal masking (position 0 attends only to itself). So the mechanism predicts a quantitative law: with a constant advantage δ over i competitors, attention to the sink ≈ σ(δ − log i), i.e. **logit(BOS-attention) should fall linearly in log(#valid positions) with slope ≈ −1**, both across lengths and across query positions within one length. Testing a derived scaling law is far stronger than "does it still hold at 1024".
**Method:** lengths {40, 128, 256, 512, 1024} × core interventions (a, b, c, d, i, j) × seeds 0,1,2. Full 300 examples at ≤512; 50/domain (150) at 1024 (configurable). Compute per length: metric battery (from E5.1 code) + per-query-position BOS-attention bins.
**Analyses:** (i) fitted slope of logit(a₀) vs log(i) per intervention with cross-seed CI vs. the −1 reference; (ii) numerical check that k₀ (= x₀W_k) and Δ₁ are length-invariant (they should be *exactly*, validating the framing); (iii) relative intervention effects (%-of-baseline) vs. length — the circuit account predicts length-stability for surgical interventions; (iv) sampling noise vs. length (merges N3).
**Data caveat solved:** SST-2/GSM8K/HumanEval have essentially no ≥1024-token examples → new **within-domain concatenation sampler** (shuffle domain examples with the seed, join, truncate to exactly `cut_length`, verify re-tokenized length). Documented as a necessary, transparent deviation.

### E5.3 — The mitigation frontier: what does sink removal cost? *(the headline figure)*
**Gap:** §5.4 — mechanism, no function. The paper's subtitle promises "implications for mitigation" but never measures a single downstream number.
**Method:** LM cross-entropy on the same examples under **all ten Table-1 interventions + the E4 combined interventions** (b∧c, b∧i, b∧d, b∧i∧d) + **dose-response CE** along the `scale_bq` and `interp_pe` α-sweeps (reusing E4 knobs). Additionally **per-position CE profiles** for baseline / Nullify-b_Q / Remove-First-PE / Zero-Top-k-Wk at 40 and 1024 tokens (rides on E5.2's long inputs).
**Analyses:** (i) **the frontier scatter** — x: sink reduction (1 − %base), y: ΔCE (nats) — with a Pareto-efficiency reading: which circuit component buys the most sink reduction per nat of damage? (ii) graded-damage curves (is functional cost monotone in α like the sink is?); (iii) **over-mixing test**: Barbero et al. (2025) predict the sink prevents representational over-mixing → sink removal should damage *late* positions more, and more so at 1024 than at 40. Controls: Zero-Random-Wk should sit at (≈0, ≈0); No-PE is expected to be functionally catastrophic — quantifying *how* catastrophic vs. surgical alternatives is exactly the mitigation point.
**Pre-registered predictions:** H-C1: circuit-targeted interventions Pareto-dominate coarse ones. H-C2: damage concentrates on late positions and grows with length.

### E5.4 — Content-generality stress test (N2)
**Gap → prediction:** the multiseed study found surgical interventions are domain-invariant (max gap ~2 pp) while coarse ones drift (8–16 pp). A *positional* circuit predicts the signature survives **any** content.
**Method:** four dataset-free degenerate domains, built from the seeded natural sample: (i) `random_uniform` (iid tokens over vocab, excl. `<|endoftext|>`), (ii) `random_zipf` (iid from the natural sample's empirical unigram distribution — isolates syntax from frequency), (iii) `shuffled_natural` (permute tokens within example), (iv) `repeat_token` (one token × 40). Optional `--with-multilingual` (FLORES-200 Bangla/Chinese via `datasets`; graceful skip if unavailable). Ten interventions × domains × seeds 0,1,2 at 40 tokens; battery metrics.
**Analyses:** domain × intervention table vs. the natural-text reference with cross-seed error bars; paired tests. H-D1: baseline sink and surgical %s match natural text within seed noise; coarse ones drift further. Confirmation = strong causal evidence for content-agnosticism; violation = a localized content dependence the paper missed (both publishable).

### E5.5 — Statistical upgrades (free, CPU; runs automatically in `aggregate`)
- **N4 bootstrap-vs-reseed calibration:** the E5 module saves **per-example** metric values (long format) → bootstrap CIs over 300 examples compared to the empirical cross-seed spread; if they agree, future single-sample studies can use bootstrap CIs (small citable methodological point).
- **N5 power/MDE table:** minimum detectable effect on the BOS metric at n ∈ {50, 100, 300, 1000} from within/between-seed variance — states which Table-1 gaps (e.g., Swap-PE 99.0 vs Nullify-BOS 99.7) are resolvable at all.

---

## 3. Implementation

### New file: `evaluation_robustness_analysis.py` (the E5 driver, ~1000–1300 lines)
Mirrors `residual_sink_analysis.py` scaffolding exactly (per-seed dirs + `aggregate/`, `run_config.json`, `sample_manifest.csv`, tqdm, `--plot-only`, fp32 default). Imports and reuses: `run_config`, `iter_examples`, `count_examples`, `forward_to_logits`, `sequence_cross_entropy`, `identify_massive_coords`, `compute_ppes`, `make_swap_direction`, `_pe_remove_first`, `_pe_interp_first`, `load_model` (from `residual_sink_analysis.py`); `run_intervention_loop`, `compute_band`, `get_initial_embeddings`, `INTERVENTIONS` (from `intervention_analysis.py`); samplers from `datasets_loader.py`.

Key internal pieces:
- **`TABLE1_SPECS`** — the ten interventions re-expressed as fast `run_config` calls (b–i map directly; (f) = token-embedding edit before `run_config`; (j) = seeded fixed random W_k columns, documented as a controlled variant of the per-layer-random original). A `--verify-parity` step asserts each spec's BOS metric equals the registry implementation's on sample inputs (fp tolerance) — validated equivalence + speed.
- **`metric_battery(attn_maps, band, ...)`** — computes all 6 metrics + per-(layer,head) BOS grid + per-query-position bins in one pass over a map list, returns a flat dict; maps freed immediately (RAM caveat at 1024: ~0.6 GB transient per forward).
- **`run_metrics` / `run_length` / `run_cost` / `run_content`** mode runners emitting tidy per-example CSVs; `aggregate_and_plot` produces Figs E5-A…F + Tables + the N4/N5 stats.
- CLI: `--mode {metrics,length,cost,content,all}`, `--seeds 0,1,2`, `--lengths 40,128,256,512,1024`, `--length-sample-size 50`, `--sample-size 100`, `--cut-length 40`, `--alphas`, `--layer-mode`, `--dtype`, `--domains`, `--with-multilingual`, `--output-dir`, `--plot-only`, `--smoke-test`, `--verify-parity`.

### Backward-compatible edits (all default-preserving; validated paths untouched)
1. **`intervention_analysis.py` → `manual_self_attention_new(..., light=False)`**: when `light=True`, skip the pure-diagnostic Python loops (`similarities_bq`, the O(seq²) `value_of_dot_product...` loop at lines 208–212, `similarities_ppes_bq`, the 768-iteration `similarities_rows_of_wk`, `result_vector`), returning empty placeholders for those slots. **Required**: at 1024 tokens the seq² loop is ~12M Python iterations per forward — the length sweep is infeasible without it. Default `False` keeps every existing harness byte-identical; smoke test asserts attention outputs identical light vs. full.
2. **`residual_sink_analysis.py` → `run_config(..., light=False, te_transform=None)`**: threads `light` into `attn_kwargs`; optional token-embedding hook for intervention (f). Defaults preserve behavior.
3. **`datasets_loader.py`** (additive): `sample_long_benchmark_datasets(tokenizer, sample_size, cut_length, seed)` — within-domain concatenation for cut_length > ~64 with exact-length verification; `build_degenerate_domains(tokenizer, natural_sampled, cut_length, seed, domains)`; optional FLORES loader.

### Docs
- **`.md/E5.md`** — full experiment doc in the E3.md style: context, design + hypothesis table, implementation & reuse map, caveats/gotchas (light-flag rationale, concatenation deviation, RAM at 1024, (j) variant, FLORES optionality, fp32), Day-1 verification, **PowerShell and bash commands**, per-GPU allocation, outputs, paper-framing claims.
- **`README.md`** — new "Extension — Evaluation Robustness & Functional Cost (E5)" section mirroring the E4 section.
- `requirements.txt` — unchanged (no new deps).

### Suggested GPU allocation (goes into E5.md)
- **5090:** `--mode length` (heaviest: 1024-token maps)
- **4090:** `--mode cost` (all-intervention CE + dose-CE + per-position)
- **3090:** `--mode metrics` (3 seeds × 10 interventions battery)
- **4080S:** `--mode content` (+ medium spot-check afterwards)

---

## 4. Verification

1. **Offline smoke test** (`--smoke-test`, CPU, no network): random-weight GPT-2 + tiny synthetic sample → all modes run end-to-end; asserts: light≡full attention outputs; TABLE1_SPECS↔registry parity; entropy ∈ [0,1]; metric halves/band correct at two lengths; CE finite; all figures/tables written. Run in this container after `pip install` from PyPI (network caveat noted if blocked).
2. **Fidelity check on user GPU:** `--mode metrics --seeds 0 --sample-size 10` must reproduce baseline ≈ .563 and Nullify-b_Q ≈ 44.7% via the fast specs.
3. **Full runs** per GPU allocation; single `--plot-only` regenerates everything from cached CSVs.
4. **Law robustness:** slope estimated per-seed and per-domain; reported with CI.

## 5. Git workflow

Base = `origin/E3-E4` (the current designated branch `claude/blackboxnlp-e5-experiment-kiuw0v` already points at it). Develop + commit there with descriptive messages, push `-u origin claude/blackboxnlp-e5-experiment-kiuw0v` (retry w/ backoff). Per the user's explicit request, also push the same commits to a new **`E5`** branch (matching the E3/E3-E4 naming convention). No PR unless requested.

## 6. Risks & caveats
- **Long-context data is synthetic-by-concatenation** — unavoidable (benchmarks lack ≥1024-token examples); disclosed, seeded, within-domain.
- **RAM at 1024 tokens:** ~0.6 GB transient per forward for CPU-side maps — fine on the user's PCs; noted in E5.md.
- **(j) Zero-Random-Wk fast variant** uses fixed seeded columns (vs. per-layer random in the registry); parity check quantifies any gap and metrics mode can fall back to the registry function at 40 tokens.
- **FLORES multilingual** is optional and degrades gracefully (skip + warning).
- **No-PE CE will explode** — expected, and part of the mitigation story (coarse interventions are functionally catastrophic).
- **Scope trims if time runs short:** drop 256 from the length grid, drop `--with-multilingual`, drop the medium spot-check — the four core analyses survive.
