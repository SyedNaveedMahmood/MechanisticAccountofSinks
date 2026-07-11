# E3 — Circuit Emergence During Pre-Training (Training Dynamics of the GPT-2 Attention-Sink Circuit)

## Context — why this experiment

**Target paper:** Ran-Milo, Ofek & Mendel, *A Mechanistic Account of Attention Sinks in GPT-2* (arXiv:2604.14722).
**Venue:** BlackboxNLP 2026 Reproducibility Track — theme **Generalizability → dynamics**; directly addresses the paper's **Limitation §5.2** ("static account of a single trained checkpoint; how/when the circuit emerges during pre-training is left to future work").

The paper explains the first-position sink in *converged* OpenAI GPT-2 124M as one circuit: the interaction of the query bias `b_Q`, the effective positional embedding `EPE_1 = MLP⁽¹⁾(p_1) + p_1` (massive activations at coords 138/378/447), and key structure `W_k` whose bias-projection peaks at those coords, producing an anomalous source-agnostic shift `Δ_1 = b_Q W_kᵀ x_1ᵀ`. **E1 (cross-scale), E2 (cross-arch OPT/GPT-Neo/Qwen2.5), and E4 (residual-sink anatomy) are already complete on `E4-with-tqdm`.** E3 is the **dynamics axis** — the single most *novel* contribution left, because it connects the *mechanistic circuit* to its *emergence trajectory across pre-training, over multiple independent seeds* — something neither the paper nor the sink literature has done at this mechanistic resolution.

**Zero training compute.** We reuse Stanford CRFM's public **"Mistral" GPT-2 small** runs (5 independent pre-training seeds, hundreds of intermediate checkpoints each on the HF Hub via `revision="checkpoint-<step>"`). Mistral is **standard GPT-2 architecture** (learned absolute PEs + query bias), so the paper's *entire* pipeline — and all our validated primitives — apply unchanged. (Pythia/OLMo are **not** viable: no learned absolute PE, no query bias ⇒ the circuit does not exist there.)

**Scope (confirmed):** GPT-2 **small only**, **5 Mistral seeds**, ~18 log-spaced checkpoints, **full 5-analysis suite** (Figs A–E, Tables 1–2). Branch: **`E3`** off `E4-with-tqdm`.

**Positioning / novelty wedge.** Gu et al. (2025, "When Attention Sink Emerges") studied *whether/why* the sink phenomenon appears and its data/optimizer dependence. E3 is finer-grained and orthogonal: it tracks the *specific* circuit (massive-coordinate identity, `b_Q`–`EPE_1 W_k` alignment, Δ_1 dominance), its **temporal ordering relative to the behavioral sink**, its **causal-intervention efficacy over training**, and its **multi-seed coordinate-level universality** — none of which are in Gu et al. or the target paper.

**Intended outcome.** A self-contained paper section + additive code module (`emergence_dynamics_analysis.py`) + `.md/E3.md`, answering: (1) *when* each ingredient and the sink appear; (2) does circuit **assembly precede the sink** (assembly-before-behavior ⇒ the circuit *causes* the sink); (3) are massive-coordinate **indices consistent across seeds** (canonical circuit) or seed-specific (universal function, arbitrary basis); (4) *when* do the paper's interventions start to "bite," and does efficacy track alignment onset; (5) does the bias pathway (A) or the E4 query pathway (B) form first. Every result is publishable either polarity under the track's negative-results policy.

---

## Scientific design

### Axes (note the change from E4)
- **Pretraining run = error-bar axis.** The **5 Mistral runs are the seed axis** (genuinely independent optimizations) — unlike E4 where "seeds" were data resamples.
- **Checkpoint step = trajectory axis** (inner sweep; slots in exactly where α sat in E4's dose-response).
- **Data sample is FIXED** (one 300-example draw, data-seed 0, from the standard GPT-2 tokenizer — identical BPE across all Mistral runs) so trajectories are directly comparable and the cross-run band reflects *pretraining* variability, not sampling noise. Sampled **once**, outside the run/checkpoint loops.

### Per-checkpoint measurements (all reuse existing, validated primitives)
For each (run, step): compute six signals. Two are **pure-weight** (no data needed → nearly free); the rest need the fixed 300-example forward passes.

| Signal | Definition | Reused primitive (file:line) | Data? |
|---|---|---|---|
| **S(t) sink strength** | baseline BOS-attention: mean second-half→pos-0 attn over heads×band | `INTERVENTIONS` `int_a` + `compute_bos_attention_metric` (`intervention_analysis.py:489,533`) | yes |
| **M(t) massive activation** | `max_d|EPE_1[d]|`, mean-of-top-3, **#coords>3σ**, and **coord indices** | `identify_massive_coords` + EPE_1 idiom (`residual_sink_analysis.py:103`) | no |
| **A(t) bias alignment** (pathway A / Fig 2) | mean over band×heads of `cos(b_Q, W_k·EPE_1)`; + pos-0-vs-(j>0) contrast | `_compute_perhead_epe_cosine` → `[L,H,P]` (`experiments_single_input.py:255`) | no |
| **D(t) Δ_1 dominance** (Fig 1) | position-0 advantage `Δ_1 − mean_{j>0}Δ_j`, pooled over band×heads (post-LN1 x_j; min-subtraction convention) | `collect_similarities_for_sentence` + min-subtract one-liner (`experiments_single_input.py:148`, `experiments_statistical.py:99`) | yes |
| **B(t) query alignment** (pathway B / E4) | mean `cos(x_i W_q, W_k·EPE_1)`, second-half sources | `collect_decomposition_and_alignment` (`residual_sink_analysis.py:225`) | yes |
| **Efficacy_X(t)** | `(S_base − S_X)/S_base` for X∈{Nullify b_Q, Remove-First-PE, Zero-Top-3-W_k} | `INTERVENTIONS` `int_b/int_c/int_i` via `run_intervention_loop` | yes |

**Standardize massive coords on `identify_massive_coords` (>3σ, top-3 fallback) at every step** — the Explore map flagged that `coord_alignment_analysis` uses a k=2 fallback and `massive_activations_analysis` has none; using one definition everywhere keeps the trajectory consistent. **Never hardcode 138/378/447** — those are OpenAI-GPT-2-specific; each Mistral seed will land elsewhere, which is precisely the E3.3 finding.

**Emergence-step extraction.** For each signal & run: min–max normalize over the trajectory (converged = last checkpoint), define `t_onset` = first step crossing 0.5 of converged; also report the max-slope step of a smoothed curve as a robustness variant. Compare `t_onset` across signals for the ordering claim.

### The five analyses / deliverables
- **E3.1 Emergence trajectories → Fig E3-A (headline).** 4-panel S/M/A/D vs. step (log-x), 5-run mean ± band. Visual test: do circuit signals (M/A/D) rise *before* the sink (S)? + **Table E3-2**: converged Mistral values vs. the paper's OpenAI-GPT-2 numbers (does the circuit reproduce across independent trainings?).
- **E3.2 Temporal ordering → Fig E3-B + Table E3-1.** Per-signal `t_onset` timeline/distribution; alignment-vs-sink scatter; paired ordering test across the 5 runs (Wilcoxon signed-rank + sign/direction, n=5).
- **E3.3 Seed universality of massive coords → Fig E3-C.** Top-k `|EPE_1|` index set per run; cross-run **Jaccard** overlap matrix; within-run temporal stability of the indices. Interesting either way (same indices = canonical; different indices, same function = universal-in-arbitrary-basis).
- **E3.4 Causal efficacy over time → Fig E3-D.** Efficacy_X(t) (3 curves) overlaid on the A(t) alignment onset — does intervention efficacy emerge in lockstep with the circuit?
- **E3.5 Two-pathway emergence → Fig E3-E.** Pathway-A onset A(t) vs pathway-B onset B(t) — which forms first (ties back to E4).

---

## Implementation

**Design principle: purely additive.** One new file + `.md/E3.md` (+ one line in `requirements.txt`). **No edits to any validated harness** — every measurement primitive is already module-level importable, and the only net-new capability (checkpoint loading at a `revision`) lives in the new module.

### New file: `emergence_dynamics_analysis.py`
Mirrors `residual_sink_analysis.py` scaffolding (argparse block `:786-815`; `_seed_dir` `:644`; `save_seed_results` `:678`; `aggregate_and_plot` `:695`; `plot_dose_response` band-plot `:582` → becomes the step-trajectory plot **with `ax.set_xscale("log")`**; `run_config.json` `:838`; tqdm `:408`; `--plot-only` guard `:826`).

**Net-new loader (in this module):**
```python
def load_checkpoint(model_name, device, dtype=torch.float32, revision=None, cache_dir=None):
    model = GPT2LMHeadModel.from_pretrained(model_name, revision=revision,
                                            cache_dir=cache_dir, attn_implementation="eager")
    try:    tok = GPT2Tokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
    except Exception: tok = GPT2Tokenizer.from_pretrained("gpt2")   # identical BPE fallback
    model.to(device); model.eval(); model.to(dtype)
    return model, tok
```
`attn_implementation="eager"` is mandatory (attention-weight extraction). `revision` is the crux the whole repo currently lacks.

**Runtime checkpoint discovery** (robust to whatever schedule is actually on the Hub):
```python
from huggingface_hub import list_repo_refs   # add to requirements.txt
def discover_checkpoints(repo_id):
    steps=[int(m.group(1)) for b in list_repo_refs(repo_id).branches
           if (m:=re.fullmatch(r"checkpoint-(\d+)", b.name))]
    return sorted(set(steps))
# select_log_spaced(steps, n): keep min+max, log-space the rest, snap to nearest available
```
`--list-checkpoints` prints available steps and exits (Day-1 verification). `--checkpoints auto` (default) log-spaces `--n-checkpoints` (default 18) from the discovered set; explicit `--checkpoints 0,100,...` snaps to nearest available. Internally track step→branch-name mapping.

**Mistral run mapping** (`--run-family gpt2-small` expands to; **verify Day-1** — ids from memory):
`stanford-crfm/{alias-x21, battlestar-x49, caprica-x81, darkmatter-x343, expanse-x777}-gpt2-small` (form `stanford-crfm/<name>-gpt2-small-x<seed>`). `--runs <comma repo ids>` overrides.

**Streaming download→analyze→purge** (`--purge-cache`): after each checkpoint, `del model; gc.collect(); torch.cuda.empty_cache()` and delete that revision's blobs (`huggingface_hub.scan_cache_dir().delete_revisions(...)`), bounding peak disk to ≈1 checkpoint (~0.5 GB) instead of ~45 GB.

**CLI (mirror E4 + additions):** `--run-family {gpt2-small}` · `--runs` · `--checkpoints {auto|list}` · `--n-checkpoints 18` · `--list-checkpoints` · `--model-name` (single-run fidelity check) · `--mode {trajectories,universality,efficacy,two_pathway,all}` · `--seeds 0` (data seed, single) · `--sample-size 100` · `--cut-length 40` · `--layer-mode scaled` · `--dtype float32` · `--purge-cache` · `--plot-only` · `--output-dir results`.

**Output tree:**
```
results/emergence_gpt2_small/
  run_alias/ step_00000000/ … step_00400000/   (per-checkpoint CSVs + run_config.json)
  run_battlestar/ …  (×5 runs)
  aggregate/   trajectories.csv, onsets.csv, jaccard.csv, efficacy.csv,
               fig_E3A..E3E.png, table_E3_1.csv, table_E3_2.csv
```

**Measurement fns** (each returns tidy rows; iterate via `iter_examples`/`count_examples` from the E4 module, band from `compute_band`): `measure_weight_signals` (M,A — no data), `measure_data_signals` (S,D,B), `measure_efficacy`. `run_all_modes` dispatches per `--mode`; `all` runs everything. Reuse the **post-LN1 x_j** convention for Δ (or magnitudes won't match Fig-1/2/4 baselines).

**Aggregation → trajectories:** read per-(run,step) CSVs `.assign(run=…, step=…)`, `pd.concat`, `groupby(["measurement","step"]).agg(mean,std)`; band = across-run std via `fill_between`. Onset extraction + ordering tests via `scipy.stats`. `--plot-only` re-renders from cached CSVs only.

### `requirements.txt`
Add `huggingface_hub` explicitly (transitive via `transformers` but needed for `list_repo_refs`). Only additive change to an existing file.

### Reproducibility discipline
fp32; fixed data-seed 0 across all checkpoints; 5 runs as the seed axis; per-checkpoint `run_config.json` (model, revision, dtype, band, massive coords, data-seed); every figure regenerable via `--plot-only`.

---

## Caveats & risks (also go into `.md/E3.md`)
1. **Checkpoint availability & schedule = primary risk** (unverifiable here — HF is proxy-blocked 403). **Day-1 user task on a networked GPU box:** `--list-checkpoints --run-family gpt2-small` to confirm the 5 repo ids resolve and print the real step schedule; the driver adapts to whatever exists. **Fallback** if Mistral is gone/renamed: (a) degrade to any available GPT-2-arch multi-checkpoint run(s); (b) if only a single run is available, keep E3.1/E3.2/E3.4/E3.5 and drop E3.3 (universality needs ≥2 runs); document transparently.
2. **Massive-coord indices are seed-specific** — a *result*, not config; re-identify per checkpoint via `identify_massive_coords`.
3. **Step-0 (random init)** is the natural null control — all circuit signals should sit at chance.
4. **Disk/IO dominates, not FLOPs** — stream+purge; parallelize the 5 runs across the 4 GPUs.
5. **This container cannot execute** (no torch/HF). During coding I'll `pip install` torch+deps from **PyPI (allowed)** and run the offline smoke test on a random GPT-2 (no download); the gpt2-fidelity check runs on the user's networked box.

---

## Verification (how we'll know it works)
1. **Offline smoke test** (added as `--smoke-test` or a small script; CPU): build a random 12-layer `GPT2LMHeadModel(GPT2Config(...))`, monkeypatch `load_checkpoint` to return it for 2 stub "checkpoints," run `--mode all` → assert all measurement CSVs written, values finite, band=[3,11), figures render. Run it here after installing deps from PyPI.
2. **Reuse-fidelity check** (user's networked box): `--model-name gpt2 --checkpoints main` reproduces the known baseline **S≈.563**, massive coords **{138,378,447}**, and positive pos-0 alignment — confirming the E3 wrappers match the validated harness.
3. **Full run** (user GPUs): `--mode all --run-family gpt2-small`; regenerate all figures via `--plot-only`.
4. **Ordering robustness:** `t_onset` ordering stable under the max-slope onset definition and across the 5 runs.

## `.md/E3.md` deliverable (created during coding)
Context · scientific design + measurement table · implementation details & reuse map · caveats (seed-specific coords, network, availability + fallback) · Day-1 verification · **PowerShell *and* bash** run commands · GPU allocation (5 runs across 4 PCs) · smoke-test + fidelity-check instructions · outputs · paper-framing claims.

**Representative commands (PowerShell shown; bash analogous):**
```powershell
# Day-1: confirm availability + real step schedule
python emergence_dynamics_analysis.py --list-checkpoints --run-family gpt2-small
# Full small suite (one box, all 5 runs, streaming)
python emergence_dynamics_analysis.py --mode all --run-family gpt2-small --n-checkpoints 18 --purge-cache --output-dir results
# Parallel: one run per GPU/PC
$env:CUDA_VISIBLE_DEVICES=0
python emergence_dynamics_analysis.py --mode all --runs stanford-crfm/alias-gpt2-small-x21 --purge-cache --output-dir results
# Re-aggregate + plot from cache
python emergence_dynamics_analysis.py --mode all --run-family gpt2-small --plot-only --output-dir results
# Reuse-fidelity on OpenAI gpt2
python emergence_dynamics_analysis.py --mode trajectories --model-name gpt2 --checkpoints main --output-dir results/_fidelity
```

## Git workflow
`git fetch origin E4-with-tqdm` → `git checkout -B E3 origin/E4-with-tqdm` → add `emergence_dynamics_analysis.py`, `.md/E3.md`, the `requirements.txt` line → run offline smoke test → commit (descriptive message) → `git push -u origin E3` (retry w/ backoff on network errors). No PR unless requested.
