# Complete results synthesis: *A Mechanistic Account of Attention Sinks in GPT-2*

**Audit date:** 2026-07-13  
**Scope:** the source paper and its eight figures; every planning/results note in `.md/`; all experiment groups under `results/`; and the standalone completed E5 report.  
**Authority rule:** numerical CSV/JSON artifacts are authoritative when prose summaries disagree. Means are shown with cross-seed SD unless a table explicitly says SE. Positions are zero-indexed in code (`position 0` is the paper's “position 1”/BOS).

## What “complete” means in this report

The local `results/` tree contains **1,960 files and 3,397,209,817 bytes** (about **3.40 GB**, or **3.16 GiB**): **871 CSV**, **281 JSON**, **9 NPZ**, **404 PNG**, and **395 TXT** files. The E5 cache alone contains millions of per-token/per-head observations. Copying every raw observation into Markdown would create a multi-gigabyte data dump, not a scientific summary. This document therefore preserves every experiment, configuration, intervention, reported aggregate, negative result, and integrity qualification; it gives complete experiment-level tables and points to the raw files that preserve full floating-point precision. No experiment group is silently omitted.

Two directories are duplicated exports, not independent replications: top-level `results/e5_full/` and `results/results_e5/results/e5_full/` have the same **99 relative filenames and file sizes**; the two `e5_timing_length` trees likewise match (**19 files**). They are counted by the filesystem census but only once scientifically. The timing tree contains only **5 examples** and is explicitly excluded from inference.

## Bottom line

The original GPT-2-small result reproduces extremely closely, but the new experiments refine “one circuit” into a more general and more defensible account:

1. **A shared positional anchor determines where the sink forms.** In GPT-2, the first-layer-MLP-processed positional signal (`EPE₀`) creates an unusually strong key-side object. Removing that identity nearly eliminates the sink; swapping it relocates **89.7–94.4%** of the original BOS mass to position 1 across GPT-2 small/medium/large.
2. **At least two query-side routes address that anchor.** Pathway A is the paper's source-agnostic `b_Q` route; pathway B is a source-dependent content-query route through the same key. The source-agnostic share falls from **0.561** in GPT-2 small to **0.450** in medium and **0.247** in large, while nullifying `b_Q` leaves **44.5%, 70.4%, and 74.0%** of baseline respectively.
3. **The function is much more universal than its parameterization.** GPT-Neo has no query bias and its `b` intervention is exactly a no-op, yet removing the first positional identity leaves only **5.9%, 2.4%, and 2.2%** of baseline. Qwen/RoPE often depends strongly on query bias but not on the additive-PE interventions. OPT is heterogeneous, and OPT-13B's targeted `W_k` edit *increases* the sink to **110.1%**. There is no architecture-independent intervention signature.
4. **The mechanism is learned non-monotonically.** Across **5 Mistral training runs × 17 checkpoints = 85/85 successful checkpoints**, the sink rises from **0.0344±0.0006** to **0.8968±0.0076**. Massive activations overshoot to **10.1958±3.7878** at 40k steps and relax to **1.3601±0.0397** by 400k. Early training even shows strong anti-alignment (**−0.6160±0.1551** at step 400).
5. **Exact coordinates are arbitrary.** The final top-3 massive-coordinate sets have pairwise Jaccard **0** across all five training runs. The learned computational role is stable; the coordinate labels are not.
6. **The sink survives long context, but not by a simple inverse-competition law.** Baseline BOS attention falls only **23.7%**, from **0.5625** at length 40 to **0.4294** at 1024 while context grows **25.6×**. The across-length logit slope is **−0.166**, not −1; within-length mean slopes flatten from **−0.336** to **−0.107**.
7. **Relocation is much cheaper than deletion.** Swap-EPE removes **93.8%** of BOS mass for **+0.938 nat**, whereas Remove-First-PE removes **97.0%** for **+3.757 nats**. The model appears to benefit from retaining a concentrated anchor more than from anchoring specifically at BOS.
8. **Location is positional; strength and routing are content-sensitive.** Remove-First-PE remains devastating on every tested content regime (**2.1–4.3%** of each regime's baseline), but the residual after nullifying `b_Q` rises from **44.4%** on natural text to **55.1%** on uniform random tokens and **57.4%** on repeated tokens.
9. **Coarse ablations are poor mechanistic evidence.** No-MLP and No-PE are about **10×** more sampling-variable than surgical edits, strongly domain-dependent, and functionally costly. E5 measures costs of **+9.497** and **+5.536 nats** respectively.

The revised claim is therefore: **attention sinks are a robust positional anchoring function implemented by architecture- and training-dependent circuits; in GPT-2, two query-side pathways converge on a shared EPE-derived key object.**

## Common setup and intervention key

Unless noted otherwise, the evaluation uses **300 sequences**: **100 SST-2 + 100 GSM8K + 100 HumanEval**, filtered to at least **40 tokens** and truncated to exactly **40**. The BOS-attention metric averages attention paid to target position 0 by source tokens in the second half, over the specified layer band and all heads.

| Label | Intervention | GPT-2/OPT/Neo meaning | Qwen/RoPE qualification |
|---|---|---|---|
| a | Baseline | No edit | No edit |
| b | Nullify query bias | Set `b_Q=0`; structural no-op in bias-free GPT-Neo | Set learned Q-projection bias to zero |
| c | Remove first PE | Replace first absolute PE by the second | Additive-PE analogue is unavailable; RoPE harness leaves this effectively control-like |
| d | Swap EPE | Swap the MLP-processed positional direction between positions 0 and 1 | Swap RoPE position IDs |
| e | Swap PE | Raw-PE direction control | Same RoPE position operation as d; hence d=e numerically |
| f | Nullify BOS token | Zero token embedding before positional signal | Token-identity edit |
| g | No MLP | Skip all MLP blocks | Same coarse ablation |
| h | No PE | Remove positional encoding | RoPE removal/neutralization analogue |
| i | Zero top-`k` `W_k` coordinates | Zero columns selected from the model's massive-coordinate signal | Per-sentence token-embedding stand-in; not the same EPE object |
| j | Random `W_k` control | E1/E2 resamples matched coordinates per layer with seed 42; E5 uses one fixed draw `{46,420,669}` with seed 1729 | Same architecture-specific matched control |

Because the semantics of c–e and i differ across architecture families, cross-family equality of labels is **not** equality of causal objects. This is a boundary condition, not a footnote.

---

## 1. Original paper and local faithful reproduction

### 1.1 Original mechanistic proposal

Expanding an attention score gives

\[
s_{i\to j}=\underbrace{x_iW_qW_k^Tx_j^T}_{T1:\;\text{content/query route}}+
\underbrace{x_iW_qb_K^T}_{T2}+
\underbrace{b_QW_k^Tx_j^T}_{T3=\Delta_j:\;\text{source-agnostic route}}+
\underbrace{b_Qb_K^T}_{T4}.
\]

For fixed source `i`, T2 and T4 are constant over target `j` and cancel under the softmax. Only **T1 + T3** determine target selection. The paper isolates T3 and argues that GPT-2's sink arises because:

- `EPE₀ = p₀ + MLP⁽¹⁾(p₀)` approximates the positional signal written by the first block;
- `EPE₀ W_k` aligns strongly with `b_Q`, yielding an unusually large source-agnostic shift `Δ₀`;
- `EPE₀` has massive coordinates **{138, 378, 447}**, each more than **15 SD** from the mean under the paper's diagnostic;
- those same coordinates receive unusually large `|b_Q W_kᵀ|` weights.

The new E4 result does not reject this pathway. It shows that it is **Pathway A**, while T1 supplies a second **Pathway B** through the same positional key.

### 1.2 Structural diagnostics

The local source-agnostic-shift run contains **300 sentences × 12 layers × 12 heads × 40 positions**. Per-layer percentiles are:

| Layer | pos0 p10 | pos0 p50 | pos0 p90 | maximum p90 among other positions | strict `pos0 p10 > other p90` |
|---:|---:|---:|---:|---:|:---:|
| 1 | 9.7652 | 16.2639 | 55.2037 | 33.0834 | no |
| 2 | 2.5398 | 16.2146 | 30.8142 | 22.8880 | no |
| 3 | 11.2801 | 40.2938 | 67.3202 | 41.3985 | no |
| 4 | 25.2140 | 50.5024 | 91.1919 | 29.4913 | no |
| 5 | 17.1184 | 38.2684 | 62.9083 | 20.2277 | no |
| 6 | 5.6449 | 17.7579 | 23.2277 | 8.6853 | no |
| 7 | 11.1521 | 22.5356 | 37.3813 | 13.4012 | no |
| 8 | 13.5780 | 26.3035 | 40.2251 | 15.8692 | no |
| 9 | 15.9952 | 22.2114 | 37.9145 | 14.3649 | yes |
| 10 | 14.5095 | 24.9926 | 35.0153 | 13.3608 | yes |
| 11 | 18.1812 | 26.2388 | 35.4180 | 15.5597 | yes |
| 12 | 5.6719 | 18.5610 | 22.2911 | 11.2073 | no |

The pooled histogram visibly separates position 0 from other positions, but the deliberately stringent per-layer test passes only **3/12 layers**. Thus “systematically larger when pooled” is supported; “complete layerwise separation” is not.

Coordinate alignment over paper layers 4–11 and 12 heads gives:

| Coordinate group | Mean `\|γ\|` | SD | Median |
|---|---:|---:|---:|
| Massive `{138,378,447}` | 1.3695 | 1.2750 | 1.0923 |
| Other coordinates | 0.1757 | 0.2127 | 0.1237 |

The massive-coordinate mean is **7.79×** the other-coordinate mean and the median is **8.83×**, supporting co-adaptation rather than a plot-only impression.

### 1.3 EPE validation, all positions

MLP-only validation has global min/median/max **0.6679/0.8473/0.9975**. Full-first-layer validation, which includes normalization and attention, is lower at **0.1987/0.3230/0.8684**. The complete per-position medians are:

| pos | MLP-only | full layer | pos | MLP-only | full layer | pos | MLP-only | full layer | pos | MLP-only | full layer |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9962 | 0.7569 | 10 | 0.8537 | 0.3258 | 20 | 0.8492 | 0.3323 | 30 | 0.8436 | 0.3160 |
| 1 | 0.8251 | 0.3442 | 11 | 0.8546 | 0.3253 | 21 | 0.8517 | 0.3355 | 31 | 0.8404 | 0.3132 |
| 2 | 0.8345 | 0.3397 | 12 | 0.8501 | 0.3260 | 22 | 0.8550 | 0.3297 | 32 | 0.8385 | 0.3069 |
| 3 | 0.8556 | 0.3399 | 13 | 0.8513 | 0.3275 | 23 | 0.8545 | 0.3304 | 33 | 0.8336 | 0.3136 |
| 4 | 0.8685 | 0.3233 | 14 | 0.8482 | 0.3304 | 24 | 0.8530 | 0.3281 | 34 | 0.8302 | 0.3130 |
| 5 | 0.8689 | 0.3257 | 15 | 0.8421 | 0.3372 | 25 | 0.8542 | 0.3223 | 35 | 0.8274 | 0.3130 |
| 6 | 0.8665 | 0.3256 | 16 | 0.8438 | 0.3277 | 26 | 0.8504 | 0.3174 | 36 | 0.8282 | 0.3057 |
| 7 | 0.8598 | 0.3203 | 17 | 0.8487 | 0.3398 | 27 | 0.8495 | 0.3152 | 37 | 0.8218 | 0.3018 |
| 8 | 0.8631 | 0.3234 | 18 | 0.8452 | 0.3322 | 28 | 0.8470 | 0.3199 | 38 | 0.8262 | 0.3011 |
| 9 | 0.8571 | 0.3233 | 19 | 0.8484 | 0.3374 | 29 | 0.8488 | 0.3159 | 39 | 0.8221 | 0.2963 |

This both validates and narrows the EPE abstraction: it is excellent for the MLP-only positional contribution, uniquely strong at position 0, and only a rough proxy for the entire first block.

### 1.4 Exact local Table 1 reproduction

The paper reports:

| int. | paper BOS attention (mean±2SE) | paper % baseline |
|---|---:|---:|
| a | .563±.004 | 100.0% |
| b | .251±.003 | 44.7% |
| c | .017±.001 | 3.0% |
| d | .035±.002 | 6.2% |
| e | .557±.006 | 99.0% |
| f | .561±.004 | 99.7% |
| g | .283±.009 | 50.3% |
| h | .099±.015 | 17.5% |
| i | .367±.004 | 65.2% |
| j | .563±.004 | 100.0% |

The local CSV stores **mean ± SE**; the paper prints approximately **mean ± 2SE**. All local rows have `n=300`.

| int. | all domains | SST-2 | GSM8K | HumanEval |
|---|---:|---:|---:|---:|
| a | 0.563006±0.002098 | 0.598815±0.002928 | 0.551644±0.002674 | 0.538558±0.001962 |
| b | 0.251443±0.001624 | 0.271680±0.002685 | 0.238725±0.002295 | 0.243925±0.002257 |
| c | 0.016841±0.000583 | 0.014420±0.000782 | 0.013387±0.000864 | 0.022715±0.001090 |
| d | 0.035188±0.000879 | 0.037324±0.001224 | 0.040705±0.002114 | 0.027535±0.000297 |
| e | 0.557317±0.002757 | 0.581456±0.006773 | 0.550504±0.003096 | 0.539992±0.002007 |
| f | 0.561375±0.002096 | 0.596156±0.002914 | 0.552250±0.002772 | 0.535719±0.001982 |
| g | 0.283242±0.004668 | 0.255337±0.007798 | 0.284445±0.009307 | 0.309943±0.005907 |
| h | 0.098742±0.007325 | 0.088733±0.014791 | 0.124613±0.014726 | 0.082881±0.006342 |
| i | 0.367183±0.001852 | 0.394746±0.003001 | 0.365128±0.002380 | 0.341675±0.001454 |
| j | 0.563296±0.002102 | 0.599384±0.002910 | 0.551773±0.002677 | 0.538730±0.001954 |

The paper's rounded all-layer values are **a=.44, b=.20, c=.02, d=.04, e=.44, f=.44, g=.24, h=.08, i=.29, j=.44**. The 156 per-layer/per-head attention-map PNGs contain no additional numerical table; visually, the sink is weak in the earliest layers, strongest through middle/deeper layers, persists under a/e/f/j, and collapses or moves under b/c/d/g/h/i as the aggregate metrics predict.

### 1.5 Reproduction verdict

The original GPT-2-small causal result is reproduced. The exact baseline differs from the paper's rounded **0.563** by only **0.000006**. The strongest causal effects—c, d, b, and i—match the paper. The structural diagnostics also reproduce, with two useful qualifications: EPE is much less exact for the full first layer than for the isolated MLP, and strict layerwise source-agnostic separation occurs in only **3/12** layers.

---

## 2. Seven-seed sampling robustness study

This documented study used data-resampling seeds **000–006**, **300 new examples/seed**, and the same GPT-2-small model. Its per-seed artifacts are not present in the current `results/` export, so these values are retained as completed-study results from `.md/multiseed_results_and_next_experiments.md`, not independently re-derived here.

| int. | paper target | 7-seed result | cross-seed SD |
|---|---:|---:|---:|
| a | 0.563 absolute | 0.5644 absolute | 0.0008 absolute |
| b | 44.7% | 44.54% | 0.15 pp |
| c | 3.0% | 2.90% | 0.09 pp |
| d | 6.2% | 6.21% | 0.08 pp |
| e | 99.0% | 98.95% | 0.28 pp |
| f | 99.7% | 99.66% | 0.03 pp |
| g | 50.3% | 50.08% | 0.65 pp |
| h | 17.5% | 18.39% | 1.10 pp |
| i | 65.2% | 65.15% | 0.10 pp |
| j | 100.0% | 100.06% | 0.005 pp |

Per-domain percentages, reported as mean±cross-seed SD:

| int. | SST-2 | GSM8K | HumanEval | maximum domain gap |
|---|---:|---:|---:|---:|
| g | 41.57±1.04% | 52.40±1.27% | 57.21±1.16% | 15.6 pp |
| h | 17.44±2.81% | 23.05±2.16% | 14.72±0.26% | 8.3 pp |
| b | 44.96±0.24% | 43.24±0.33% | 45.41±0.15% | 2.2 pp |
| d | 6.50±0.22% | 7.00±0.27% | 5.07±0.02% | 1.9 pp |
| c | 2.25±0.17% | 2.29±0.26% | 4.26±0.14% | 2.0 pp |

Baseline domain means are **SST-2 0.6027 > GSM8K 0.5497 > HumanEval 0.5410**. Surgical b/c/d/i edits vary only **0.08–0.15 pp** across samples; coarse g/h vary **0.65/1.10 pp** and have much larger domain gaps. This is independent evidence that broad ablations entangle content-dependent computation.

---

## 3. E1/E2: cross-scale and cross-architecture sweep

### 3.1 Coverage and recorded configuration

There are **16 models × 3 data seeds × 300 examples × 10 interventions**, with scaled layer band `[3,L−1)`. Contrary to the planning recommendation, the recorded run configurations show that **all OPT checkpoints, including 6.7B and 13B, ran in float32**. Qwen 7B/14B ran in bfloat16; all other runs used float32.

| Model | L/H/D | dtype | band | massive coordinates used by i |
|---|---|---|---|---|
| GPT-2 small | 12/12/768 | fp32 | `[3,11)` | 138,378,447 |
| GPT-2 medium | 24/16/1024 | fp32 | `[3,23)` | 9,238,268,428,608,580 |
| GPT-2 large | 36/20/1280 | fp32 | `[3,35)` | 792,870,440,8,24,248 |
| GPT-Neo 125M | 12/12/768 | fp32 | `[3,11)` | 647,522,536 |
| GPT-Neo 1.3B | 24/16/2048 | fp32 | `[3,23)` | 858,352,1467,486,1448,1826 |
| GPT-Neo 2.7B | 32/20/2560 | fp32 | `[3,31)` | 936,1675,1635,1567,483 |
| OPT 125M | 12/12/768 | fp32 | `[3,11)` | 422,353,130 |
| OPT 1.3B | 24/32/2048 | fp32 | `[3,23)` | 1359,1177,572,236,1192 |
| OPT 2.7B | 32/32/2560 | fp32 | `[3,31)` | 1454,319,13,1625,1102,1802,198 |
| OPT 6.7B | 32/32/4096 | fp32 | `[3,31)` | 2394,639,1778,3136,2972,2717,2255,314,1450,2553,480,2731 |
| OPT 13B | 40/40/5120 | fp32 | `[3,39)` | 3964,902,4960,440,2211,2050,3689,4711,635,1211 |
| Qwen2.5 0.5B | 24/14/896 | fp32 | `[3,23)` | no fixed EPE coordinates; per-sentence stand-in |
| Qwen2.5 1.5B | 28/12/1536 | fp32 | `[3,27)` | no fixed EPE coordinates; per-sentence stand-in |
| Qwen2.5 3B | 36/16/2048 | fp32 | `[3,35)` | no fixed EPE coordinates; per-sentence stand-in |
| Qwen2.5 7B | 28/28/3584 | bf16 | `[3,27)` | no fixed EPE coordinates; per-sentence stand-in |
| Qwen2.5 14B | 48/40/5120 | bf16 | `[3,47)` | no fixed EPE coordinates; per-sentence stand-in |

E1/E2 random controls use seed **42**, match the number of selected coordinates, and are **resampled per layer**. This differs from E5's fixed random draw and explains why the two j controls are not directly comparable.

### 3.2 Complete pooled intervention table

Each cell is **absolute BOS attention ± cross-seed SD (percent of that model's baseline)**; `n_seeds=3` throughout.

| model | a | b | c | d | e | f | g | h | i | j |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 small | 0.5637±0.0001 (100.0%) | 0.2506±0.0009 (44.5%) | 0.0167±0.0006 (3.0%) | 0.0352±0.0007 (6.2%) | 0.5577±0.0010 (98.9%) | 0.5617±0.0003 (99.6%) | 0.2826±0.0027 (50.1%) | 0.1016±0.0073 (18.0%) | 0.3672±0.0004 (65.1%) | 0.5640±0.0001 (100.1%) |
| GPT-2 medium | 0.5753±0.0013 (100.0%) | 0.4050±0.0015 (70.4%) | 0.0098±0.0013 (1.7%) | 0.0477±0.0007 (8.3%) | 0.5758±0.0013 (100.1%) | 0.5757±0.0011 (100.1%) | 0.0792±0.0030 (13.8%) | 0.0536±0.0054 (9.3%) | 0.1624±0.0005 (28.2%) | 0.5762±0.0013 (100.2%) |
| GPT-2 large | 0.5015±0.0012 (100.0%) | 0.3713±0.0013 (74.0%) | 0.0195±0.0011 (3.9%) | 0.0171±0.0005 (3.4%) | 0.5013±0.0012 (100.0%) | 0.5038±0.0011 (100.5%) | 0.1034±0.0009 (20.6%) | 0.0161±0.0007 (3.2%) | 0.2111±0.0014 (42.1%) | 0.5008±0.0012 (99.9%) |
| GPT-Neo 125M | 0.2714±0.0011 (100.0%) | 0.2714±0.0011 (100.0%) | 0.0160±0.0008 (5.9%) | 0.0822±0.0012 (30.3%) | 0.2701±0.0011 (99.5%) | 0.2515±0.0010 (92.7%) | 0.0104±0.0009 (3.8%) | 0.1373±0.0014 (50.6%) | 0.2651±0.0009 (97.7%) | 0.2733±0.0011 (100.7%) |
| GPT-Neo 1.3B | 0.3336±0.0009 (100.0%) | 0.3336±0.0009 (100.0%) | 0.0081±0.0003 (2.4%) | 0.2665±0.0029 (79.9%) | 0.3334±0.0009 (99.9%) | 0.1337±0.0063 (40.1%) | 0.0220±0.0010 (6.6%) | 0.0878±0.0012 (26.3%) | 0.2948±0.0007 (88.3%) | 0.3338±0.0009 (100.1%) |
| GPT-Neo 2.7B | 0.4121±0.0007 (100.0%) | 0.4121±0.0007 (100.0%) | 0.0092±0.0004 (2.2%) | 0.1038±0.0016 (25.2%) | 0.4121±0.0007 (100.0%) | 0.3281±0.0040 (79.6%) | 0.0664±0.0021 (16.1%) | 0.0801±0.0090 (19.4%) | 0.2730±0.0008 (66.3%) | 0.4120±0.0007 (100.0%) |
| OPT 125M | 0.6766±0.0006 (100.0%) | 0.5604±0.0010 (82.8%) | 0.0143±0.0003 (2.1%) | 0.4103±0.0036 (60.6%) | 0.3134±0.0002 (46.3%) | 0.6755±0.0005 (99.8%) | 0.1207±0.0010 (17.8%) | 0.0236±0.0035 (3.5%) | 0.2686±0.0007 (39.7%) | 0.6754±0.0007 (99.8%) |
| OPT 1.3B | 0.6268±0.0010 (100.0%) | 0.6123±0.0011 (97.7%) | 0.2015±0.0059 (32.2%) | 0.0029±0.0000 (0.5%) | 0.0253±0.0006 (4.0%) | 0.6245±0.0010 (99.6%) | 0.1499±0.0022 (23.9%) | 0.0289±0.0028 (4.6%) | 0.2360±0.0011 (37.7%) | 0.6270±0.0010 (100.0%) |
| OPT 2.7B | 0.6247±0.0010 (100.0%) | 0.6132±0.0010 (98.2%) | 0.2231±0.0033 (35.7%) | 0.0843±0.0019 (13.5%) | 0.2036±0.0007 (32.6%) | 0.6238±0.0009 (99.9%) | 0.1117±0.0027 (17.9%) | 0.0473±0.0023 (7.6%) | 0.2410±0.0007 (38.6%) | 0.6234±0.0010 (99.8%) |
| OPT 6.7B | 0.6300±0.0012 (100.0%) | 0.6217±0.0012 (98.7%) | 0.0358±0.0032 (5.7%) | 0.0165±0.0000 (2.6%) | 0.3363±0.0006 (53.4%) | 0.6281±0.0012 (99.7%) | 0.0739±0.0041 (11.7%) | 0.0184±0.0021 (2.9%) | 0.3515±0.0003 (55.8%) | 0.6296±0.0012 (99.9%) |
| OPT 13B | 0.3475±0.0021 (100.0%) | 0.3433±0.0021 (98.8%) | 0.0232±0.0006 (6.7%) | 0.3230±0.0019 (93.0%) | 0.3425±0.0021 (98.6%) | 0.3404±0.0009 (98.0%) | 0.1713±0.0022 (49.3%) | 0.2007±0.0020 (57.8%) | 0.3826±0.0011 (**110.1%**) | 0.3391±0.0020 (97.6%) |
| Qwen2.5 0.5B | 0.5200±0.0008 (100.0%) | 0.0317±0.0007 (6.1%) | 0.5186±0.0011 (99.7%) | 0.4972±0.0032 (95.6%) | 0.4972±0.0032 (95.6%) | 0.5064±0.0006 (97.4%) | 0.0153±0.0007 (2.9%) | 0.4005±0.0014 (77.0%) | 0.5175±0.0023 (99.5%) | 0.5174±0.0011 (99.5%) |
| Qwen2.5 1.5B | 0.5226±0.0034 (100.0%) | 0.0351±0.0004 (6.7%) | 0.4966±0.0070 (95.0%) | 0.4370±0.0020 (83.6%) | 0.4370±0.0020 (83.6%) | 0.0024±0.0000 (0.5%) | 0.0162±0.0005 (3.1%) | 0.4115±0.0055 (78.8%) | 0.5169±0.0032 (98.9%) | 0.5222±0.0035 (99.9%) |
| Qwen2.5 3B | 0.4990±0.0034 (100.0%) | 0.0724±0.0003 (14.5%) | 0.4975±0.0027 (99.7%) | 0.4669±0.0078 (93.6%) | 0.4669±0.0078 (93.6%) | 0.3221±0.0011 (64.6%) | 0.0131±0.0007 (2.6%) | 0.4444±0.0024 (89.1%) | 0.4986±0.0027 (99.9%) | 0.4986±0.0034 (99.9%) |
| Qwen2.5 7B | 0.5453±0.0056 (100.0%) | 0.0364±0.0002 (6.7%) | 0.5420±0.0063 (99.4%) | 0.5230±0.0066 (95.9%) | 0.5230±0.0066 (95.9%) | 0.0025±0.0001 (0.5%) | 0.0276±0.0002 (5.1%) | 0.4915±0.0043 (90.1%) | 0.5290±0.0042 (97.0%) | 0.5446±0.0054 (99.9%) |
| Qwen2.5 14B | 0.6396±0.0009 (100.0%) | 0.2907±0.0011 (45.4%) | 0.6398±0.0009 (100.0%) | 0.6376±0.0008 (99.7%) | 0.6376±0.0008 (99.7%) | 0.0114±0.0001 (1.8%) | 0.0267±0.0001 (4.2%) | 0.6136±0.0015 (95.9%) | 0.6378±0.0011 (99.7%) | 0.6368±0.0009 (99.6%) |

### 3.3 Domain boundary conditions

Each cell below is **SST-2/GSM8K/HumanEval**. Baseline cells are absolute BOS attention; b/c/d/i cells are percent of the corresponding domain baseline.

| model | baseline | b | c | d | i |
|---|---:|---:|---:|---:|---:|
| GPT-2 small | .603/.548/.540 | 44.9/43.0/45.4 | 2.4/2.4/4.2 | 6.6/7.0/5.1 | 65.9/66.1/63.4 |
| GPT-2 medium | .627/.556/.543 | 70.7/69.9/70.6 | 2.2/1.3/1.5 | 7.0/9.3/8.8 | 27.6/28.0/29.2 |
| GPT-2 large | .555/.482/.468 | 75.4/72.4/74.1 | 4.3/2.9/4.4 | 3.1/3.1/4.1 | 45.9/40.5/39.2 |
| Neo 125M | .333/.245/.237 | 100/100/100 | 4.5/5.1/8.8 | 25.1/32.1/35.6 | 101.3/98.3/91.9 |
| Neo 1.3B | .384/.316/.301 | 100/100/100 | 1.6/2.0/4.0 | 73.1/90.6/77.2 | 89.3/87.0/88.5 |
| Neo 2.7B | .457/.391/.388 | 100/100/100 | 1.8/2.0/2.9 | 23.6/26.3/25.9 | 64.6/66.8/67.6 |
| OPT 125M | .697/.674/.659 | 83.2/82.2/83.0 | 2.3/2.4/1.7 | 56.8/71.0/54.1 | 38.6/40.4/40.2 |
| OPT 1.3B | .666/.607/.607 | 98.0/97.4/97.6 | 26.8/29.7/40.5 | .4/.3/.7 | 36.1/40.0/37.1 |
| OPT 2.7B | .662/.603/.609 | 98.4/97.9/98.1 | 32.3/42.0/33.2 | 13.9/12.4/14.2 | 36.8/38.9/40.2 |
| OPT 6.7B | .669/.606/.615 | 98.9/98.5/98.7 | 5.5/7.1/4.5 | 2.6/2.8/2.5 | 54.4/56.3/56.9 |
| OPT 13B | .378/.337/.327 | 99.1/98.5/98.8 | 9.5/8.6/1.6 | 92.1/92.4/94.5 | 102.5/112.5/116.5 |
| Qwen 0.5B | .531/.514/.515 | 6.8/5.9/5.5 | 99.4/99.7/100.0 | 90.3/97.1/99.6 | 99.5/99.0/100.1 |
| Qwen 1.5B | .538/.511/.518 | 6.3/6.5/7.3 | 89.3/96.3/99.7 | 74.5/78.2/98.4 | 97.8/98.9/100.1 |
| Qwen 3B | .497/.455/.545 | 13.3/15.3/15.0 | 99.6/99.5/100.0 | 91.4/88.9/99.4 | 99.8/99.8/100.2 |
| Qwen 7B | .560/.520/.556 | 6.4/6.9/6.8 | 98.5/99.6/100.1 | 93.2/95.1/99.4 | 95.9/97.5/97.7 |
| Qwen 14B | .653/.623/.643 | 51.1/40.0/44.9 | 100.0/100.0/100.1 | 99.6/99.7/99.7 | 99.5/99.8/99.9 |

### 3.4 Interpretation by family

- **GPT-2:** c and d remain decisive at every scale, so the positional anchor is conserved. But b weakens from a **55.5% reduction** in small to **29.6%/26.0%** in medium/large, while targeted i is strongest in medium (**71.8% reduction**). Scale shifts weight from the fixed-bias route toward content queries.
- **GPT-Neo:** b is exactly baseline because the architecture has no `b_Q`. Nevertheless, c leaves only **5.9/2.4/2.2%**, proving that a sink can use a strong first-position key without the paper's source-agnostic bias path. d varies sharply (**30.3/79.9/25.2%**) and f becomes important at 1.3B (**40.1%**) and 2.7B (**79.6%**), so token identity and how positional identity is moved are model-specific.
- **OPT:** no single signature survives scale. b is nearly irrelevant above 1.3B (**97.7–98.8%**). d is devastating at 1.3B (**0.5%**), 2.7B (**13.5%**), and 6.7B (**2.6%**), but nearly inert at 13B (**93.0%**). Raw-PE control e is not a neutral control in several OPT models (**4.0%, 32.6%, 53.4%** at 1.3/2.7/6.7B). At 13B, i increases BOS attention to **110.1%**, including **116.5%** on HumanEval. The selected coordinates are correlated with the sink circuit but zeroing them need not reduce it after the circuit reorganizes.
- **Qwen2.5/RoPE:** c is a no-op by construction; d=e because both encode the same RoPE position operation. b is extremely strong at 0.5–7B (**6.1–14.5% residual**) but weaker at 14B (**45.4%**). f nearly eliminates the sink in 1.5B, 7B, and 14B (**0.5%, 0.5%, 1.8%**) but not 0.5B or 3B (**97.4%, 64.6%**). g nearly collapses all Qwen sinks (**2.6–5.1%**). This is evidence for alternative circuits, not a direct replication of the GPT-2 EPE pathway.

E1/E2 therefore supports the paper's own broad caveat more strongly than its title: **sink behavior generalizes; the exact parameter-level circuit does not.**

---

## 4. E3: training-time emergence dynamics

### 4.1 Coverage

E3 follows five Stanford-CRFM Mistral `gpt2-small` runs—`alias`, `battlestar`, `caprica`, `darkmatter`, and `expanse`—at steps **0, 100, 200, 400, 700, 1k, 2k, 4k, 7k, 10k, 20k, 40k, 70k, 100k, 200k, 300k, 400k**. All **85/85** run-checkpoints completed; there are **0 NaNs**. The tables below are the authoritative values from `trajectories.csv`, not rounded prose copied from `E3_Summary.md`.

### 4.2 Complete mean trajectories

Each cell is mean±SD across five independently trained runs. `A align` is query-bias/EPE-key alignment; `A contrast` is its separation from control positions; `B content attn` is attention with only the content-query route; `Δ dominance` is the source-agnostic route's attention contribution diagnostic.

| step | EPE max | top-3 mean | # >3σ | A align | A contrast | sink | Δ dominance | B content attn |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0652±0.0085 | 0.0610±0.0051 | 7.20±2.17 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0344±0.0006 | 0.0340±0.0000 | 0.0344±0.0006 |
| 100 | 0.0681±0.0102 | 0.0631±0.0059 | 6.20±1.64 | −0.0119±0.0696 | 0.0326±0.0704 | 0.0460±0.0179 | 0.0342±0.0001 | 0.0460±0.0178 |
| 200 | 0.0779±0.0100 | 0.0720±0.0066 | 6.20±2.68 | 0.0034±0.1158 | 0.0597±0.1106 | 0.0266±0.0089 | 0.0341±0.0001 | 0.0267±0.0089 |
| 400 | 0.0982±0.0066 | 0.0903±0.0073 | 5.20±2.77 | **−0.6160±0.1551** | 0.0963±0.1122 | 0.0105±0.0064 | 0.0295±0.0011 | 0.0106±0.0064 |
| 700 | 0.1237±0.0080 | 0.1158±0.0066 | 7.00±2.92 | −0.6122±0.1623 | 0.2033±0.1205 | 0.1353±0.0364 | 0.0306±0.0012 | 0.1344±0.0362 |
| 1,000 | 0.1488±0.0141 | 0.1414±0.0101 | 8.00±2.24 | −0.4521±0.1941 | 0.3206±0.1514 | 0.2030±0.0323 | 0.0327±0.0009 | 0.2024±0.0320 |
| 2,000 | 0.2647±0.0457 | 0.2301±0.0275 | 10.00±1.22 | −0.3891±0.1147 | 0.2788±0.1297 | 0.2697±0.0159 | 0.0378±0.0017 | 0.2684±0.0155 |
| 4,000 | 0.7457±0.1882 | 0.6049±0.1537 | 10.80±2.05 | −0.3137±0.0723 | 0.3312±0.0642 | 0.3229±0.0328 | 0.0606±0.0064 | 0.3197±0.0325 |
| 7,000 | 2.4978±1.3809 | 2.0168±1.0445 | 7.40±1.67 | −0.1349±0.0609 | 0.3766±0.1529 | 0.4747±0.0312 | 0.1207±0.0149 | 0.4576±0.0326 |
| 10,000 | 3.1141±0.9093 | 2.5863±0.6211 | 7.80±1.30 | −0.0435±0.0888 | 0.6070±0.1147 | 0.5389±0.0429 | 0.2390±0.0415 | 0.4999±0.0345 |
| 20,000 | 7.3873±0.8556 | 6.5164±0.4113 | 7.00±1.22 | 0.3884±0.0401 | 1.1024±0.0381 | 0.6962±0.0270 | 0.6020±0.0154 | 0.5765±0.0274 |
| 40,000 | **10.1958±3.7878** | 8.7105±2.9189 | 6.00±2.55 | 0.6140±0.0329 | 1.3524±0.0338 | 0.7764±0.0164 | 0.7130±0.0197 | 0.6278±0.0179 |
| 70,000 | 10.1726±3.1776 | **8.7220±2.3068** | 5.40±1.52 | 0.6977±0.0171 | 1.4261±0.0464 | 0.8115±0.0238 | **0.7347±0.0199** | 0.6630±0.0200 |
| 100,000 | 8.5096±2.2398 | 7.3363±1.6048 | 7.00±1.22 | 0.7200±0.0298 | **1.4387±0.0495** | 0.8393±0.0123 | 0.7199±0.0161 | 0.7052±0.0112 |
| 200,000 | 4.2981±0.7840 | 3.8486±0.6004 | 7.20±1.30 | 0.7754±0.0314 | 1.4294±0.0682 | 0.8645±0.0071 | 0.6511±0.0145 | 0.7597±0.0131 |
| 300,000 | 2.2397±0.1481 | 1.9843±0.1785 | 8.20±1.30 | **0.8007±0.0274** | 1.2494±0.0938 | 0.8743±0.0100 | 0.6035±0.0126 | 0.7812±0.0128 |
| 400,000 | 1.3601±0.0397 | 1.1971±0.0768 | 8.40±0.89 | 0.7951±0.0268 | 1.0330±0.1343 | **0.8968±0.0076** | 0.5550±0.0126 | **0.8181±0.0110** |

The intervention/alignment trajectory is:

| step | content-query align pos0 | control align | Δ score share | efficacy: nullify `b_Q` | efficacy: remove first PE | efficacy: zero top-`W_k` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0066±0.0135 | 0.0004±0.0010 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0137±0.0154 | 0.0009±0.0018 |
| 100 | −0.0017±0.0471 | −0.0236±0.0203 | −0.0064±0.0140 | 0.0004±0.0016 | 0.4307±0.1666 | 0.0000±0.0057 |
| 200 | −0.0002±0.0876 | −0.0310±0.0052 | 0.0101±0.0168 | 0.0002±0.0012 | −0.1645±0.3436 | −0.0036±0.0092 |
| 400 | **−0.4669±0.1021** | −0.5596±0.0340 | −0.6216±1.4131 | −0.0025±0.0051 | 0.1461±0.4292 | −0.0119±0.0400 |
| 700 | −0.4094±0.1446 | −0.6045±0.0476 | 0.0091±0.0093 | 0.0060±0.0041 | 0.7886±0.0458 | 0.0054±0.0516 |
| 1,000 | −0.1826±0.1594 | −0.5320±0.0358 | 0.0374±0.0595 | 0.0182±0.0034 | 0.9396±0.0269 | −0.0151±0.0573 |
| 2,000 | −0.0387±0.1102 | −0.3879±0.0720 | 0.0084±0.0450 | 0.0450±0.0074 | 0.9710±0.0091 | −0.0748±0.1080 |
| 4,000 | 0.0977±0.0337 | −0.1391±0.0413 | 0.0440±0.0244 | 0.0687±0.0071 | 0.9826±0.0045 | 0.2485±0.0759 |
| 7,000 | 0.2188±0.0476 | 0.0382±0.0739 | −0.0656±0.3318 | 0.0619±0.0144 | 0.9918±0.0039 | **0.4494±0.0346** |
| 10,000 | 0.2147±0.0343 | 0.0034±0.0346 | 0.0966±0.1037 | 0.0730±0.0086 | 0.9953±0.0021 | 0.4080±0.1135 |
| 20,000 | 0.1713±0.0333 | −0.0362±0.0193 | 0.1991±0.1047 | 0.0888±0.0123 | **0.9981±0.0009** | 0.3230±0.1220 |
| 40,000 | 0.1015±0.0223 | −0.1355±0.0169 | **0.2316±0.0320** | 0.0907±0.0139 | 0.9951±0.0016 | 0.1875±0.0112 |
| 70,000 | 0.0745±0.0212 | −0.1787±0.0186 | 0.2120±0.0494 | **0.0928±0.0191** | 0.9958±0.0019 | 0.1598±0.0563 |
| 100,000 | 0.1017±0.0189 | −0.1702±0.0360 | 0.1667±0.0503 | 0.0927±0.0111 | 0.9931±0.0055 | 0.2667±0.1949 |
| 200,000 | 0.2334±0.0206 | −0.0569±0.0746 | 0.1602±0.0163 | 0.0716±0.0036 | 0.9912±0.0069 | 0.2277±0.0293 |
| 300,000 | 0.2679±0.0236 | 0.0695±0.0920 | 0.1541±0.0019 | 0.0623±0.0035 | 0.9931±0.0022 | 0.2776±0.1453 |
| 400,000 | **0.2784±0.0193** | 0.1338±0.0902 | 0.1383±0.0039 | 0.0531±0.0026 | 0.9934±0.0030 | 0.2707±0.1336 |

Here efficacy is fractional sink reduction, so **0.9934** means a **99.34%** reduction and **0.0531** means only **5.31%**.

### 4.3 Converged runs and coordinate arbitrariness

| run | sink | EPE max | A align | Δ dominance | B query align | all >3σ coordinates | final top-3 |
|---|---:|---:|---:|---:|---:|---|---|
| alias | .892688 | 1.408531 | .812382 | .553083 | .285773 | 8,164,340,541,545,572,654,683,724 | 683,8,541 |
| battlestar | .904391 | 1.337701 | .824056 | .555576 | .307072 | 232,284,318,341,545,598,618,644,695 | 618,318,341 |
| caprica | .899217 | 1.386829 | .804974 | .536121 | .265751 | 113,115,177,229,411,439,655,710 | 115,229,439 |
| darkmatter | .885707 | 1.307611 | .761505 | .558917 | .276382 | 100,192,256,470,532,610,667,700,751 | 610,470,192 |
| expanse | .902059 | 1.359967 | .772573 | .571062 | .257133 | 41,209,218,233,435,529,595 | 529,233,41 |

All **10 pairwise top-3 Jaccards are 0**. Only coordinate **545** overlaps even in the wider >3σ sets (alias/battlestar). Coordinate identity is therefore a symmetry-broken implementation detail, not a reproducible semantic feature.

### 4.4 Onset, maximum-slope timing, and temporal stability

| signal | median onset | IQR | per-run onset (alias/battlestar/caprica/darkmatter/expanse) |
|---|---:|---:|---|
| sink | 7,000 | 7,000–10,000 | 7k / 10k / 10k / 7k / 7k |
| EPE max | 20,000 | 20,000–20,000 | 20k / 20k / 20k / 20k / 20k |
| A alignment | 200 | 100–20,000 | 20k / 200 / 20k / 100 / 0 |
| Δ dominance | 20,000 | 20,000–20,000 | 20k / 20k / 20k / 20k / 20k |
| B query alignment | 0 | 0–0 | 0 / 0 / 0 / 0 / 0 |

Maximum-slope steps are, in the same run order: sink **700/20k/1k/7k/7k**; EPE max **200k/20k/200k/20k/40k**; A alignment **1k/400/400/400/400**; Δ dominance **20k** for all five; B alignment **1k/400/400/400/400**. A preregistered early-vs-late ordering test finds A alignment earlier than the sink in only **3/5** runs: Wilcoxon statistic **6.0**, **p=.8125**. The formal ordering hypothesis is not supported.

Top-coordinate Jaccard to the final set is initially 0 in every run. First non-zero overlap appears at step **1k/4k/7k/4k/2k** for alias/battlestar/caprica/darkmatter/expanse. Caprica reaches 1.0 by **40k**, battlestar by **100k**; alias, darkmatter, and expanse do not reach 1.0 until the final **400k** checkpoint. Even within one training run, coordinate labels can remain unstable long after the sink is strong.

### 4.5 E3 interpretation and source discrepancy

E3 shows a staged but non-monotonic process: an early anti-sink/anti-alignment phase; rapid growth of the content-query route; massive-activation and Δ-route overshoot; then compression/relaxation while sink strength continues to increase. By convergence, removing first PE is almost fully causal (**99.34%**), while nullifying `b_Q` removes only **5.31%**—the reverse of treating the paper's route as the dominant universal cause.

`E3_Summary.md` contains several rounded values that do not match `trajectories.csv`: for example, step-0 Remove-First-PE efficacy is written as **.032** but the CSV mean is **.01365584**; step-400 is written as **.598** but the CSV is **.1461**; and step-400 `b_Q` efficacy is described as positive **.002** while the CSV is **−.0025**. This report uses the CSV throughout.

---

## 5. E4: anatomy of the residual sink

### 5.1 The tested two-pathway account

E4 exploits the exact T1/T3 decomposition. Both paths converge on the same key-side object `k₀≈EPE₀W_k`, but differ on the query side:

| path | score term | query side | selectively removed by |
|---|---|---|---|
| A: source-agnostic bias path | `T3(0)=b_Q·k₀` | fixed learned `b_Q` | b / scaling `b_Q` |
| B: source-dependent content path | `T1(i,0)=(x_iW_q)·k₀` | token/context-dependent query | survives b; removed when the shared EPE key is removed |

All E4 runs use seeds **0,1,2**, 100 examples/domain, length 40, fp32, scaled bands, and alpha grid **{0,.25,.5,.75,1,1.25,1.5}**. Massive coordinates are GPT-2 small **{138,378,447}**, medium **{9,238,268,428,580,608}**, and large **{8,24,248,440,792,870}** (order is irrelevant).

### 5.2 Exact decomposition and query alignment

The path-only attention values are separate softmax counterfactuals and should not be added linearly.

| model | full | B-only attention | A-only attention | mean Δ-score share | query·EPE₀-key | control | Cohen d | Mann–Whitney p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | .563700 | .210465 | .376511 | .560981 | .228143 | .023113 | 1.785054 | 0 |
| medium | .575325 | .329358 | .215574 | .449580 | .023017 | −.070096 | .420871 | 1.54155×10⁻²² |
| large | .501453 | .323353 | .136383 | .246848 | −.060736 | −.107904 | .304943 | 1.08739×10⁻²¹ |

The key result is a **relative** query-alignment enrichment at all scales, not positive raw cosine at all scales. Large-model queries are negative on average (**−.0607**) but still substantially less negative than controls (**−.1079**). The A path's score share falls monotonically with scale, matching the E1 residual after b.

### 5.3 Combined interventions

Cells give percent of baseline ± cross-seed SD, followed by absolute BOS attention.

| model | baseline | b | b∧c | b∧d | b∧i | b∧i∧d |
|---|---:|---:|---:|---:|---:|---:|
| small | 100%; .563700 | 44.455±.162%; .250596 | 2.254±.060%; .012708 | 4.229±.093%; .023838 | 24.773±.102%; .139647 | 4.966±.054%; .027992 |
| medium | 100%; .575325 | 70.397±.114%; .405011 | 1.638±.195%; .009427 | 3.467±.034%; .019943 | 19.125±.065%; .110032 | 5.577±.078%; .032084 |
| large | 100%; .501453 | 74.036±.084%; .371258 | 3.651±.141%; .018308 | 2.173±.075%; .010897 | 31.184±.231%; .156374 | 2.466±.062%; .012364 |

Removing the shared first-position object after b collapses the residual: b∧c leaves **1.64–3.65%** and b∧d **2.17–4.23%**. The massive-coordinate edit explains part, not all, of B: b∧i leaves **19.13–31.18%**. Interactions are non-additive: adding i to b∧d slightly *raises* the residual in small/medium, so “more components ablated” is not a monotonic causal scale.

### 5.4 Complete dose-response results

#### GPT-2 small

| α | PE₀→PE₁ identity interpolation | scale `b_Q` | scale raw PE₀ | scale selected `W_k` columns |
|---:|---:|---:|---:|---:|
| 0 | .016674±.000566 | .250596±.000932 | .620344±.002914 | .367226±.000414 |
| .25 | .025623±.001281 | .322406±.000473 | .586550±.000931 | .422210±.000310 |
| .5 | .566462±.000256 | .411177±.000288 | .565637±.000113 | .473940±.000198 |
| .75 | .565266±.000022 | .489755±.000101 | .564118±.000058 | .521199±.000129 |
| 1 | .563700±.000099 | .563700±.000099 | .563700±.000099 | .563700±.000099 |
| 1.25 | .563159±.000138 | .635021±.000234 | .563557±.000137 | .601655±.000086 |
| 1.5 | .562740±.000156 | .695831±.000229 | .563540±.000166 | .635460±.000094 |

#### GPT-2 medium

| α | PE₀→PE₁ identity interpolation | scale `b_Q` | scale raw PE₀ | scale selected `W_k` columns |
|---:|---:|---:|---:|---:|
| 0 | .009795±.001251 | .405011±.001534 | .475356±.008223 | .162372±.000544 |
| .25 | .018954±.003422 | .442929±.001445 | .580617±.001356 | .272205±.001099 |
| .5 | .566137±.000958 | .483931±.001400 | .574075±.001238 | .389691±.001313 |
| .75 | .575394±.001270 | .528410±.001360 | .574826±.001275 | .491970±.001331 |
| 1 | .575325±.001285 | .575325±.001285 | .575325±.001285 | .575325±.001285 |
| 1.25 | .575115±.001314 | .622843±.001192 | .575575±.001304 | .641310±.001154 |
| 1.5 | .574711±.001326 | .670433±.001095 | .575584±.001313 | .693450±.000985 |

#### GPT-2 large

| α | PE₀→PE₁ identity interpolation | scale `b_Q` | scale raw PE₀ | scale selected `W_k` columns |
|---:|---:|---:|---:|---:|
| 0 | .019480±.001111 | .371258±.001256 | .183469±.003528 | .211136±.001441 |
| .25 | .019299±.001348 | .401424±.001260 | .274137±.008216 | .290133±.001407 |
| .5 | .429995±.006185 | .432867±.001254 | .494511±.001174 | .369519±.001300 |
| .75 | .499204±.001224 | .466484±.001248 | .500067±.001220 | .441179±.001252 |
| 1 | .501453±.001213 | .501453±.001213 | .501453±.001213 | .501453±.001213 |
| 1.25 | .501986±.001200 | .535882±.001162 | .501908±.001210 | .551253±.001172 |
| 1.5 | .502276±.001190 | .569150±.001099 | .502131±.001207 | .592679±.001137 |

Monotonicity tests and alpha-zero floors:

| model | knob | Spearman ρ | p | α=0 floor |
|---|---|---:|---:|---:|
| small | PE₀→PE₁ interpolation | .285714 | .534509 | .016674 |
| medium | PE₀→PE₁ interpolation | .642857 | .119392 | .009795 |
| large | PE₀→PE₁ interpolation | .964286 | .000454 | .019480 |
| small/medium/large | `b_Q` scale | 1 / 1 / 1 | 0 / 0 / 0 | .250596 / .405011 / .371258 |
| small/medium/large | raw PE₀ scale | −1 / .464286 / 1 | 0 / .293934 / 0 | .620344 / .475356 / .183469 |
| small/medium/large | selected `W_k` scale | 1 / 1 / 1 | 0 / 0 / 0 | .367226 / .162372 / .211136 |

`b_Q` and selected-`W_k` scaling give clean graded causality. PE₀→PE₁ identity interpolation is threshold-like around **α=.25–.5**, not smoothly graded. Raw-PE magnitude is not a safe proxy for positional identity: deleting it strengthens the small-model sink to **110.05%**.

### 5.5 Relocation is directly validated

| model | baseline pos0 | baseline pos1 | swap pos0 | swap pos1 | b∧swap pos0 | b∧swap pos1 | relocation ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | .563700±.000099 | .006129±.000064 | .035204±.000748 | .511512±.002313 | .023838±.000527 | .261825±.000996 | .896546±.004000 |
| medium | .575325±.001285 | .007155±.000125 | .047749±.000697 | .537342±.001479 | .019943±.000154 | .259775±.002684 | .921543±.000579 |
| large | .501453±.001213 | .009758±.000173 | .017108±.000492 | .483193±.000704 | .010897±.000384 | .360008±.000850 | .944126±.000656 |

The paper's visual relocation claim is quantitatively correct: nearly all of the anchor moves to the transplant target. After b, the relocated mass remains **.262/.260/.360**, direct evidence that Pathway B also addresses the transplanted EPE-derived key.

### 5.6 Surgical versus coarse edits and functional cost

| model | all MLPs off | all PEs off | first-layer MLP skip | zero only PE₀ |
|---|---:|---:|---:|---:|
| small | 50.139±.464%; .282632 | 18.026±1.291%; .101613 | **102.530±.230%; .577964** | **110.048±.507%; .620344** |
| medium | 13.763±.556%; .079178 | 9.312±.965%; .053568 | **6.911±.033%; .039763** | 82.624±1.416%; .475356 |
| large | 20.611±.206%; .103353 | 3.214±.127%; .016119 | 87.664±.330%; .439594 | 36.587±.674%; .183469 |

The first-layer MLP's effect is non-monotonic with scale: skipping it strengthens small, nearly abolishes medium, and modestly reduces large. Therefore the paper's all-layer No-MLP result cannot be attributed cleanly to the first block alone.

The file is named `perplexity_summary.csv`, but its values are mean token cross-entropy in nats:

| model | baseline | HF reference | nullify `b_Q` | zero selected `W_k` | first-layer MLP skip |
|---|---:|---:|---:|---:|---:|
| small | 4.009122±.007633 | 4.009122±.007633 | 5.595538±.039795 | 4.267715±.013720 | 6.431702±.064333 |
| medium | 3.727259±.009375 | 3.727259±.009375 | 4.082281±.018080 | 4.784833±.020796 | 8.267273±.010801 |
| large | 3.539360±.013142 | 3.539360±.013142 | 3.659364±.014488 | 4.011038±.012580 | 7.744783±.029973 |

The baseline/HF agreement is within **4.48×10⁻⁸**, validating the custom executor. Query-bias removal becomes functionally cheaper with scale (**ΔCE +1.586, +.355, +.120**), even as it removes a smaller fraction of the sink. The first-layer skip is extremely damaging (**+2.423, +4.540, +4.205 nats**).

### 5.7 E4 verdict

E4 resolves the paper's “residual sink” limitation: in GPT-2, the residual after b is largely a second positional route, not unexplained BOS semantics. The shared key object is strongly supported by combined interventions and relocation. What changes with scale is the query-side mixture, not the existence of positional anchoring. At the same time, the nonlinear interactions, negative absolute alignment in large, and scale-dependent first-layer effects prevent an overly tidy additive story.

---

## 6. E5: evaluation robustness, scaling laws, content, and functional cost

### 6.1 Scope and integrity audit

E5 uses GPT-2 small, fp32, band `[3,11)`, seeds **0/1/2**, 100 examples/domain at length 40, lengths **40/128/256/512/1024**, and **50 examples/domain at 1024**. It uses massive coordinates **{138,378,447}**, fixed-random coordinates **{46,420,669}** from seed **1729**, and **2,000** within-seed bootstrap repetitions from seed **1730**.

The completed cache contains:

- **9,000** metric rows;
- **24,300** length rows;
- **25,200** sequence-level cost rows;
- **45,000** content rows;
- **270/270** logit-slope fits with `status=ok`;
- **1,981,800** token-position CE rows, all `status=ok`;
- **0** literal NaN or infinite experimental values;
- exactly **100** unique natural-domain examples in each seed manifest;
- exact length **1,024** for every long-context sample;
- bit-identical shared 40-token BOS measurements across metric, length, cost, and natural-content modes;
- all **6** planned aggregate figures and all planned aggregate tables.

Blank warnings in successful cost rows and undefined reduction-per-nat at exactly zero cost are expected, not missing outputs. HumanEval resamples are not independent: every pair of 100-example seed samples shares **60 anchors**, Jaccard **.429**. SST-2 overlap is **9–11** anchors and GSM8K **0–3**. No explicit smoke-test or parity-report artifact was cached; a–i fidelity is supported empirically by the Table 1 match, not by a preserved formal report. Multilingual/FLORES was not run.

### 6.2 Metric multiverse: complete 40-token summary

Every cell is mean±cross-seed SD. `bos_rank` is oriented oppositely from mass/rate metrics: lower rank means a stronger sink. Entropy measures global concentration and is not BOS-specific.

| int. | BOS attn | rate>.2 | rate>.3 | rate>.5 | BOS rank | entropy | mass pos1–4 | mass pos5+ | redistributed local share | head Gini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a | .5625±.0007 | .9429±.0008 | .8856±.0012 | .6287±.0032 | 1.3618±.0024 | .4192±.0009 | .0213±.0003 | .4161±.0010 | .0486±.0008 | .2063±.0014 |
| b | .2498±.0009 | .5477±.0036 | .3804±.0010 | .1147±.0007 | 4.3802±.0108 | .5628±.0009 | .0220±.0001 | .7281±.0010 | .0294±.0002 | .4214±.0019 |
| c | .0166±.0006 | .0050±.0008 | .0012±.0003 | .0001±.0001 | 19.8148±.1631 | .6066±.0017 | .0533±.0008 | .9301±.0008 | .0543±.0008 | .6790±.0029 |
| d | .0350±.0006 | .0258±.0006 | .0181±.0004 | .0105±.0002 | 13.5347±.1461 | .4879±.0001 | **.5518±.0008** | .4132±.0007 | **.5719±.0007** | .6904±.0023 |
| e | .5552±.0016 | .9467±.0012 | .8868±.0009 | .6147±.0048 | 1.3578±.0029 | .4165±.0013 | .0334±.0008 | .4114±.0008 | .0688±.0013 | .1997±.0011 |
| f | .5606±.0007 | .9419±.0009 | .8841±.0014 | .6266±.0029 | 1.3646±.0028 | .4213±.0008 | .0208±.0003 | .4186±.0010 | .0473±.0007 | .2065±.0014 |
| g | .2836±.0013 | .4706±.0045 | .3750±.0036 | .2381±.0017 | 6.2170±.0462 | .5628±.0009 | .0512±.0015 | .6652±.0017 | .0725±.0020 | .5437±.0023 |
| h | .1024±.0079 | .1573±.0104 | .1071±.0094 | .0560±.0097 | 13.4003±.2664 | .6105±.0080 | .2253±.0079 | .6724±.0060 | .2508±.0067 | .6242±.0060 |
| i | .3659±.0006 | .8455±.0021 | .6299±.0020 | .1953±.0016 | 1.7676±.0041 | .5476±.0008 | .0340±.0007 | .6001±.0010 | .0536±.0012 | .2443±.0012 |
| j | .5820±.0006 | .9599±.0005 | .9110±.0014 | .6726±.0029 | 1.3173±.0020 | .3946±.0007 | .0199±.0003 | .3980±.0009 | .0473±.0008 | .1884±.0012 |

This reproduces Table 1 again: a **.5625**, b **44.4%**, c **3.0%**, d **6.2%**, e **98.7%**, f **99.7%**, g **50.4%**, h **18.2%**, i **65.0%**, j **103.5%**. The fixed random j is not a null: it increases BOS attention by **3.47%**.

The relocation/diffusion distinction is visible numerically. d moves **.5518** mass into positions 1–4 with entropy **.4879**; c sends **.9301** to positions 5+ with entropy **.6066**. Both remove BOS mass, but they do not produce the same attention state.

### 6.3 Metric concordance

The table lists all **15 unique off-diagonal** Kendall comparisons among the six preregistered ranking metrics. The full CSV also stores both matrix orientations and six unit diagonals.

| metric pair | Kendall τ | p |
|---|---:|---:|
| BOS – rate>.2 | .866667 | 1.15190×10⁻⁴ |
| BOS – rate>.3 | .866667 | 1.15190×10⁻⁴ |
| BOS – rate>.5 | .955556 | 5.51146×10⁻⁶ |
| BOS – rank | .866667 | 1.15190×10⁻⁴ |
| BOS – entropy | .644444 | .00914848 |
| rate>.2 – rate>.3 | 1.000000 | 5.51146×10⁻⁷ |
| rate>.2 – rate>.5 | .822222 | 3.57694×10⁻⁴ |
| rate>.2 – rank | 1.000000 | 5.51146×10⁻⁷ |
| rate>.2 – entropy | .777778 | 9.46318×10⁻⁴ |
| rate>.3 – rate>.5 | .822222 | 3.57694×10⁻⁴ |
| rate>.3 – rank | 1.000000 | 5.51146×10⁻⁷ |
| rate>.3 – entropy | .777778 | 9.46318×10⁻⁴ |
| rate>.5 – rank | .822222 | 3.57694×10⁻⁴ |
| rate>.5 – entropy | **.600000** | .0166661 |
| rank – entropy | .777778 | 9.46318×10⁻⁴ |

Mean off-diagonal τ is **.84**, but **5/15** pairs are below the preregistered **.8** threshold, all involving entropy. The core intervention ordering is robust; entropy usefully distinguishes relocation/concentration from BOS-specific sinking.

### 6.4 Intensive versus extensive head-cell effects

The baseline has **81,468 / 76,519 / 54,318** active layer-head-query cells above thresholds **.2/.3/.5**. Slopes and correlations are fit once per intervention; the last two columns vary by threshold.

| int. | threshold | baseline→post slope | r | active falling below | active remaining but shrinking |
|---|---:|---:|---:|---:|---:|
| b | .2 | .5267 | .5879 | 41.94% | 56.10% |
| b | .3 | .5267 | .5879 | 57.30% | 40.95% |
| b | .5 | .5267 | .5879 | 82.77% | 16.56% |
| c | .2 | .0174 | .1103 | 99.47% | .52% |
| c | .3 | .0174 | .1103 | 99.87% | .13% |
| c | .5 | .0174 | .1103 | 99.99% | .01% |
| d | .2 | .0870 | .2060 | 97.27% | 2.06% |
| d | .3 | .0870 | .2060 | 97.96% | 1.32% |
| d | .5 | .0870 | .2060 | 98.43% | .66% |
| e | .2 | .9467 | .9690 | .08% | 47.38% |
| e | .3 | .9467 | .9690 | .59% | 48.90% |
| e | .5 | .9467 | .9690 | 4.13% | 53.46% |
| f | .2 | .9961 | .9988 | .17% | 62.45% |
| f | .3 | .9961 | .9988 | .41% | 61.99% |
| f | .5 | .9961 | .9988 | 1.19% | 61.36% |
| g | .2 | .4983 | .3600 | 50.24% | 31.74% |
| g | .3 | .4983 | .3600 | 58.79% | 23.21% |
| g | .5 | .4983 | .3600 | 68.18% | 13.64% |
| h | .2 | −.0767 | −.0870 | 84.08% | 10.28% |
| h | .3 | −.0767 | −.0870 | 89.60% | 5.60% |
| h | .5 | −.0767 | −.0870 | 95.71% | 1.80% |
| i | .2 | .6081 | .7800 | 10.41% | 88.47% |
| i | .3 | .6081 | .7800 | 28.92% | 70.28% |
| i | .5 | .6081 | .7800 | 68.96% | 30.99% |
| j | .2 | .9442 | .9922 | .01% | 28.40% |
| j | .3 | .9442 | .9922 | .05% | 29.88% |
| j | .5 | .9442 | .9922 | .62% | 35.21% |

The preregistered universal “intensive-margin” hypothesis fails. i is mostly intensive at threshold .3 (**70.28%** stay active and shrink); c/d are overwhelmingly extensive (**99.87%/97.96%** fall below); b is mixed. Residual mass becomes more unequal: head Gini rises from **.2063** to **.4214/.6790/.6904** under b/c/d.

### 6.5 Length generalization

The length suite contains a/b/c/d/i/j. Each cell is BOS attention±seed SD (percent of the same-length baseline).

| length | a | b | c | d | i | j |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | .5625±.0007 (100.0%) | .2498±.0009 (44.4%) | .0166±.0006 (3.0%) | .0350±.0006 (6.2%) | .3659±.0006 (65.0%) | .5820±.0006 (103.5%) |
| 128 | .5001±.0007 | .2240±.0004 (44.8%) | .0070±.0001 (1.4%) | .0230±.0005 (4.6%) | .2965±.0006 (59.3%) | .4867±.0008 (97.3%) |
| 256 | .4685±.0008 | .2142±.0003 (45.7%) | .0034±.0002 (.7%) | .0952±.0004 (20.3%) | .2635±.0003 (56.2%) | .4686±.0008 (100.0%) |
| 512 | .4468±.0008 | .1875±.0007 (42.0%) | .0016±.0001 (.4%) | .0906±.0005 (20.3%) | .2280±.0004 (51.0%) | .4433±.0008 (99.2%) |
| 1024 | .4294±.0003 | .2003±.0018 (46.6%) | .0003±.0000 (**.1%**) | .0576±.0016 (13.4%) | .2000±.0005 (46.6%) | .4167±.0005 (97.1%) |

All baseline metrics across length are:

| length | BOS | rate>.2 | rate>.3 | rate>.5 | rank | entropy | mass1–4 | mass5+ | local share | Gini |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | .5625±.0007 | .9429±.0008 | .8856±.0012 | .6287±.0032 | 1.3618±.0024 | .4192±.0009 | .0213±.0003 | .4161±.0010 | .0486±.0008 | .2063±.0014 |
| 128 | .5001±.0007 | .9387±.0001 | .8624±.0010 | .5157±.0020 | 1.4533±.0011 | .3933±.0007 | .0073±.0002 | .4926±.0009 | .0145±.0004 | .1986±.0005 |
| 256 | .4685±.0008 | .9287±.0014 | .8347±.0007 | .4227±.0014 | 1.5335±.0048 | .3831±.0003 | .0037±.0001 | .5277±.0008 | .0070±.0001 | .2037±.0008 |
| 512 | .4468±.0008 | .9206±.0013 | .8008±.0019 | .3701±.0006 | 1.6378±.0053 | .3720±.0005 | .0015±.0000 | .5518±.0008 | .0026±.0001 | .2122±.0007 |
| 1024 | .4294±.0003 | .9066±.0002 | .7721±.0006 | .3431±.0009 | 1.8917±.0067 | .3603±.0001 | .0009±.0000 | .5698±.0003 | .0015±.0000 | .2207±.0005 |

Context grows **25.6×**, but BOS attention falls only **23.7%**. Even at 1024, **90.66%** of cells exceed .2, **77.21%** exceed .3, and **34.31%** exceed .5. The sink persists broadly while weakening intensively. At 1024, b residual is strongly domain-dependent: **30.5% SST-2, 53.4% GSM8K, 57.9% HumanEval**.

### 6.6 Fixed-object invariance and failed inverse-competition law

Nested-prefix invariance checks:

| quantity | length | max abs discrepancy | max relative discrepancy | mean abs discrepancy | comparisons |
|---|---:|---:|---:|---:|---:|
| Δ₁ | 40 | 0 | 0 | 0 | 900 |
| Δ₁ | 128 | 7.62939×10⁻⁶ | 4.34296×10⁻⁷ | 2.66128×10⁻⁶ | 900 |
| Δ₁ | 256 | 7.62939×10⁻⁶ | 4.34597×10⁻⁷ | 2.58128×10⁻⁶ | 900 |
| Δ₁ | 512 | 7.62939×10⁻⁶ | 4.36297×10⁻⁷ | 2.75665×10⁻⁶ | 900 |
| Δ₁ | 1024 | 7.62939×10⁻⁶ | 4.38645×10⁻⁷ | 3.47773×10⁻⁶ | 450 |
| `k₀` | 40 | 0 | 0 | 0 | 900 |
| `k₀` | 128 | 1.71363×10⁻⁶ | 3.81465×10⁻⁷ | 6.30965×10⁻⁷ | 900 |
| `k₀` | 256 | 1.60187×10⁻⁶ | 3.56874×10⁻⁷ | 5.93828×10⁻⁷ | 900 |
| `k₀` | 512 | 1.19209×10⁻⁶ | 2.64897×10⁻⁷ | 6.63300×10⁻⁷ | 900 |
| `k₀` | 1024 | 1.88500×10⁻⁶ | 4.19193×10⁻⁷ | 8.72695×10⁻⁷ | 450 |

So the causal-mask premise is true to fp32 rounding. The proposed attention law is not. Mean within-length logit slopes across domains are shown with the domain range in brackets:

| int. | L40 | L128 | L256 | L512 | L1024 | across-length slope |
|---|---:|---:|---:|---:|---:|---:|
| a | −.336 [−.384,−.309] | −.213 [−.254,−.181] | −.161 [−.175,−.136] | −.108 [−.124,−.093] | −.107 [−.115,−.094] | **−.166171** |
| b | −.351 [−.383,−.302] | .016 [−.016,.046] | −.152 [−.193,−.097] | −.012 [−.061,.031] | **−2.641 [−3.573,−2.033]** | −.102504 |
| c | −1.145 [−1.391,−.747] | −1.163 [−1.289,−.920] | −.995 [−1.405,−.687] | −2.517 [−3.163,−2.082] | −.106 [−1.047,.474] | **−1.161063** |
| d | −.573 [−.597,−.543] | −.038 [−.106,.050] | **4.390 [4.304,4.448]** | **−7.236 [−7.656,−6.758]** | **3.301 [2.941,3.522]** | .307238 |
| i | −.410 [−.447,−.356] | −.232 [−.263,−.211] | −.262 [−.287,−.242] | −.303 [−.328,−.262] | −.267 [−.313,−.215] | −.258093 |
| j | −.346 [−.405,−.312] | −.337 [−.374,−.289] | −.121 [−.146,−.096] | −.130 [−.143,−.114] | .345 [.287,.391] | −.197378 |

Only c is near −1 across lengths, after the sink is almost gone. Extreme b/d values are symptoms of nonlinear position curves and low-probability logit instability, not credible new scaling laws. Invariance of `k₀` and `Δ₁` does not imply exchangeable or fixed competitor scores.

### 6.7 Functional cost and complete mitigation frontier

Pooled baseline cross-entropy is **4.021 nats**: **4.959 SST-2, 3.530 GSM8K, 3.573 HumanEval**. Positive `sink reduction` means less BOS mass. `ratio` is the stored mean per-stratum reduction/nat, which can disagree in sign with the ratio of aggregate axes near zero cost; it must not be used to rank near-zero-cost controls.

| intervention | ΔCE | sink reduction | stored ratio | global Pareto |
|---|---:|---:|---:|:---:|
| a | 0 | 0% | undefined | yes |
| b | 1.605072 | 55.5863% | .348314 | no |
| b∧c | 4.128601 | 97.7322% | .237614 | yes |
| b∧d | 1.789538 | 95.8125% | .543349 | yes |
| b∧i | 2.680295 | 75.2415% | .282832 | no |
| b∧i∧d | 2.774261 | 95.0743% | .345545 | no |
| c | 3.756831 | 97.0236% | .258846 | yes |
| d | .938172 | 93.7995% | 1.019983 | yes |
| e | .003899 | 1.2348% | **−1.026889** | yes |
| f | .136268 | .3358% | .028709 | no |
| g | 9.496907 | 49.2297% | .052272 | no |
| h | 5.536155 | 81.7751% | .149228 | no |
| i | .257205 | 34.9783% | 2.183704 | yes |
| j | .005153 | **−3.4749%** | 1.859403 | no |
| interpolate PE α=0 | 3.756831 | 97.0236% | .258846 | yes |
| interpolate PE α=.25 | 3.714064 | 95.4774% | .257685 | no |
| interpolate PE α=.5 | .055683 | **−.4257%** | −.053689 | no |
| interpolate PE α=.75 | **−.012541** | **−.2661%** | −.383589 | yes |
| interpolate PE α=1 | 0 | 0% | undefined | yes |
| interpolate PE α=1.25 | .011612 | .0896% | .133989 | no |
| interpolate PE α=1.5 | .022248 | .1595% | .113070 | no |
| scale `b_Q` α=0 | 1.605072 | 55.5863% | .348314 | no |
| scale `b_Q` α=.25 | .774745 | 42.9770% | .561824 | yes |
| scale `b_Q` α=.5 | .187796 | 27.1841% | 1.657201 | yes |
| scale `b_Q` α=.75 | .018953 | 13.1880% | **−.809093** | yes |
| scale `b_Q` α=1 | 0 | 0% | undefined | yes |
| scale `b_Q` α=1.25 | .051487 | **−12.7719%** | −2.694148 | no |
| scale `b_Q` α=1.5 | .215125 | **−23.6700%** | −1.109442 | no |

The aggregate frontier's scientifically useful points are:

- `b_Q=.75`: **13.2%** reduction for **+.019 nat**;
- `b_Q=.5`: **27.2%** for **+.188**;
- i: **35.0%** for **+.257**;
- `b_Q=.25`: **43.0%** for **+.775**;
- d: **93.8%** for **+.938**;
- b∧d: **95.8%** for **+1.790**;
- c: **97.0%** for **+3.757**;
- b∧c: **97.7%** for **+4.129**.

d, i, b∧d, and `b_Q` doses .25/.75 are Pareto-efficient in **all 9 seed×domain strata**. g, h, full b, and random j are efficient in **0/9**. The smooth `b_Q` dose frontier is graded causal evidence for Pathway A. PE interpolation is threshold-like: α=.5 gives **−.43% reduction, +.056 nat**, but α=.25 jumps to **95.48% reduction, +3.714 nats**.

Most importantly, d versus c holds sink removal approximately fixed while changing whether the anchor is relocated or destroyed. Relocation achieves almost the same reduction for about one quarter of the CE cost (**.938/3.757=.250**).

### 6.8 Position-level damage

The following values average ΔCE over seeds and domains within normalized early/middle/late regions. Baseline a is exactly **0** in every cell.

| length | int. | early | middle | late |
|---:|---|---:|---:|---:|
| 40 | b | 2.289 | 1.597 | .929 |
| 40 | c | 5.602 | 3.289 | 2.379 |
| 40 | i | .430 | .199 | .143 |
| 1024 | b | .546 | .521 | **1.093** |
| 1024 | c | **1.746** | 1.161 | 1.428 |
| 1024 | i | .161 | .143 | .147 |

At length 40 all three are early-heavy. At 1024 only b becomes late-heavy and its late damage grows from **.929** to **1.093**. c remains early-heavy, while i is small and flat. Long-context late damage is a Pathway-A-specific result, not a universal cost of sink removal.

### 6.9 Content generality: complete BOS table

Each cell is absolute BOS attention±seed SD (percent of that content regime's own baseline).

| content | a | b | c | d | e | f | g | h | i | j |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SST-2 | .5996±.0022 | .2676±.0005 (44.6%) | .0139±.0009 (2.3%) | .0388±.0011 (6.5%) | .5812±.0053 (96.9%) | .5959±.0022 (99.4%) | .2416±.0028 (40.3%) | .1000±.0172 (16.7%) | .3939±.0018 (65.7%) | .6193±.0021 (103.3%) |
| GSM8K | .5478±.0011 | .2361±.0027 (43.1%) | .0130±.0012 (2.4%) | .0387±.0008 (7.1%) | .5429±.0018 (99.1%) | .5485±.0014 (100.1%) | .2985±.0033 (54.5%) | .1306±.0104 (23.8%) | .3614±.0006 (66.0%) | .5676±.0011 (103.6%) |
| HumanEval | .5402±.0004 | .2459±.0004 (45.5%) | .0229±.0004 (4.2%) | .0274±.0002 (5.1%) | .5416±.0005 (100.3%) | .5374±.0004 (99.5%) | .3108±.0012 (57.5%) | .0765±.0022 (14.2%) | .3425±.0006 (63.4%) | .5592±.0003 (103.5%) |
| random uniform | .6020±.0002 | .3317±.0011 (55.1%) | .0126±.0007 (2.1%) | .0341±.0007 (5.7%) | .5993±.0012 (99.6%) | .5983±.0006 (99.4%) | .4380±.0024 (72.8%) | .1238±.0190 (20.6%) | .4095±.0003 (68.0%) | .6191±.0002 (102.8%) |
| empirical-unigram/Zipf | .5225±.0013 | .2386±.0017 (45.7%) | .0138±.0005 (2.6%) | .0447±.0007 (8.5%) | .5069±.0034 (97.0%) | .5197±.0014 (99.5%) | .2887±.0037 (55.3%) | .1398±.0107 (26.7%) | .3305±.0013 (63.2%) | .5459±.0013 (104.5%) |
| shuffled natural | .5084±.0002 | .2339±.0012 (46.0%) | .0149±.0006 (2.9%) | .0420±.0002 (8.3%) | .4927±.0045 (96.9%) | .5064±.0006 (99.6%) | .2921±.0041 (57.4%) | .1445±.0118 (28.4%) | .3236±.0003 (63.6%) | .5304±.0001 (104.3%) |
| repeated token | .5325±.0021 | .3059±.0016 (57.4%) | .0231±.0001 (4.3%) | .0530±.0017 (9.9%) | **.4332±.0077 (81.3%)** | .5297±.0022 (99.5%) | .3071±.0026 (57.7%) | .0340±.0000 (6.4%) | .3876±.0018 (72.8%) | .5439±.0019 (102.1%) |

Natural pooled values are b/c/d/g/h/i/j = **44.4/3.0/6.2/50.4/18.2/65.0/103.5%**. The anchor survives destruction of syntax and semantics, but strength is not content-independent. Uniform input raises paired baseline BOS by **.03945**; shuffling lowers it by **.05416**. Relative b shifts by about **+10.7 pp** on uniform and **+13.0 pp** on repeats; i shifts **+7.7 pp** on repeats. Even e fails as a neutral control for degenerate repeated content, falling to **81.3%**.

### 6.10 All paired content tests

Each synthetic regime has **900 valid pairs** across seeds. These t-tests do not model seed clustering or HumanEval finite-pool overlap; effect sizes are more trustworthy than tiny p-values.

| content | int. | mean paired Δ | t | p |
|---|---|---:|---:|---:|
| uniform | a | .039449 | 28.349 | 8.154×10⁻¹²⁷ |
| uniform | b | .081883 | 72.958 | 0 (numerical underflow) |
| uniform | c | −.003992 | −8.698 | 1.590×10⁻¹⁷ |
| uniform | d | −.000872 | −1.431 | **.1526** |
| uniform | e | .044056 | 21.925 | 1.080×10⁻⁸⁵ |
| uniform | f | .037709 | 27.191 | 2.738×10⁻¹¹⁹ |
| uniform | g | .154345 | 38.956 | 3.120×10⁻¹⁹⁵ |
| uniform | h | .021452 | 2.699 | .007085 |
| uniform | i | .043592 | 34.307 | 1.491×10⁻¹⁶⁵ |
| uniform | j | .037089 | 26.856 | 4.038×10⁻¹¹⁷ |
| Zipf | a | −.040017 | −27.561 | 1.080×10⁻¹²¹ |
| Zipf | b | −.011289 | −9.921 | 4.379×10⁻²² |
| Zipf | c | −.002840 | −5.762 | 1.141×10⁻⁸ |
| Zipf | d | .009716 | 16.308 | 1.428×10⁻⁵² |
| Zipf | e | −.048350 | −18.988 | 7.278×10⁻⁶⁸ |
| Zipf | f | −.040853 | −28.259 | 3.137×10⁻¹²⁶ |
| Zipf | g | .005097 | 1.197 | **.2318** |
| Zipf | h | .037407 | 4.769 | 2.161×10⁻⁶ |
| Zipf | i | −.035447 | −29.794 | 3.135×10⁻¹³⁶ |
| Zipf | j | −.036112 | −25.362 | 1.797×10⁻¹⁰⁷ |
| repeated | a | −.029984 | −17.476 | 4.252×10⁻⁵⁹ |
| repeated | b | .056089 | 26.207 | 6.434×10⁻¹¹³ |
| repeated | c | .006525 | 16.044 | 3.967×10⁻⁵¹ |
| repeated | d | .018016 | 23.393 | 6.679×10⁻⁹⁵ |
| repeated | e | −.122030 | −34.794 | 1.065×10⁻¹⁶⁸ |
| repeated | f | −.030864 | −18.572 | 2.029×10⁻⁶⁵ |
| repeated | g | .023439 | 4.822 | 1.667×10⁻⁶ |
| repeated | h | −.068318 | −15.606 | 9.152×10⁻⁴⁹ |
| repeated | i | .021701 | 10.412 | 4.786×10⁻²⁴ |
| repeated | j | −.038102 | −22.226 | 1.437×10⁻⁸⁷ |
| shuffled | a | −.054158 | −50.946 | 2.833×10⁻²⁶⁷ |
| shuffled | b | −.015936 | −17.715 | 1.843×10⁻⁶⁰ |
| shuffled | c | −.001690 | −3.664 | 2.628×10⁻⁴ |
| shuffled | d | .007088 | 11.682 | 1.867×10⁻²⁹ |
| shuffled | e | −.062552 | −26.765 | 1.584×10⁻¹¹⁶ |
| shuffled | f | −.054183 | −50.173 | 7.425×10⁻²⁶³ |
| shuffled | g | .008438 | 2.081 | .03776 |
| shuffled | h | .042156 | 5.424 | 7.480×10⁻⁸ |
| shuffled | i | −.042361 | −43.187 | 1.755×10⁻²²¹ |
| shuffled | j | −.051633 | −51.309 | 2.468×10⁻²⁶⁹ |

Only uniform d (**p=.1526**) and Zipf g (**p=.2318**) are conventionally non-significant; the scientific conclusion should still rest on effect sizes, not 900-pair pseudoprecision.

### 6.11 Bootstrap calibration

Each bootstrap SE uses **n=300**, **2,000 repetitions**, bootstrap seed **1730**. The empirical SD column is across only three data seeds. Cells show `bootstrap SE (SE/reseed-SD ratio)`.

| int. | empirical seed SD | seed 0 bootstrap | seed 1 bootstrap | seed 2 bootstrap | mean ratio |
|---|---:|---:|---:|---:|---:|
| a | .000713 | .002188 (3.068) | .002078 (2.914) | .002097 (2.940) | 2.974 |
| b | .000896 | .001671 (1.865) | .001510 (1.684) | .001556 (1.735) | 1.761 |
| c | .000619 | .000684 (1.105) | .000571 (.922) | .000623 (1.006) | 1.011 |
| d | .000591 | .000782 (1.323) | .000770 (1.304) | .000692 (1.170) | 1.265 |
| e | .001636 | .003008 (1.839) | .002880 (1.760) | .002837 (1.735) | 1.778 |
| f | .000735 | .002130 (2.898) | .002034 (2.768) | .002032 (2.764) | 2.810 |
| g | .001347 | .004951 (3.675) | .004705 (3.492) | .005025 (3.730) | 3.632 |
| h | .007876 | .008119 (1.031) | .006547 (.831) | .007509 (.953) | .938 |
| i | .000638 | .001866 (2.925) | .001763 (2.764) | .001724 (2.702) | 2.797 |
| j | .000640 | .002120 (3.313) | .002061 (3.220) | .002037 (3.184) | 3.239 |

A single-seed bootstrap is not a general substitute for reseeding: average ratios range from **.938 to 3.632**. The apparent overconservatism partly reflects the imprecise three-seed denominator and finite source-pool overlap, especially HumanEval.

### 6.12 Power and minimum detectable effects

All rows use α=.05, power=.8, and **900 available paired observations**. Variances come from paired intervention-minus-baseline differences.

| int. | within variance | between-seed mean variance | combined variance | MDE n=50 | n=100 | n=300 | n=1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| b | 6.71009×10⁻⁴ | 6.39162×10⁻⁷ | 6.71648×10⁻⁴ | .010268 | .007261 | .004192 | .002296 |
| c | .00154128 | 3.75636×10⁻⁸ | .00154132 | .015555 | .010999 | .006350 | .003478 |
| d | .00126610 | 6.95894×10⁻⁷ | .00126679 | .014102 | .009971 | .005757 | .003153 |
| e | .00166094 | 8.56337×10⁻⁷ | .00166180 | .016151 | .011421 | .006594 | .003612 |
| f | 1.27633×10⁻⁵ | 6.21720×10⁻¹⁰ | 1.27639×10⁻⁵ | .001416 | .001001 | .000578 | .000317 |
| g | .01016292 | 4.27530×10⁻⁷ | .01016335 | .039943 | .028244 | .016307 | .008931 |
| h | .01870410 | 5.34084×10⁻⁵ | .01875751 | .054263 | .038370 | .022153 | .012134 |
| i | .000181707 | 7.01763×10⁻⁸ | .000181778 | .005342 | .003777 | .002181 | .001194 |
| j | 7.54364×10⁻⁶ | 5.95720×10⁻⁹ | 7.54959×10⁻⁶ | .001089 | .000770 | .000444 | .000243 |

At n=300 all large causal effects are easily resolvable. The observed e–f gap is **.00536**, below its direct paired MDE **.00656**, so it should not be treated as a resolved control difference. j's effect is **+.01952** against MDE **.00044**, so the fixed random draw's sink increase is real for that draw.

### 6.13 E5 hypothesis verdicts

| hypothesis | verdict | decisive number |
|---|---|---|
| Metric rankings have τ≥.8 | partially supported | mean .84; 5/15 below .8; entropy–rate>.5=.60 |
| Interventions act mainly on intensive margin | rejected generally | c/d make 99.87%/97.96% of .3-active cells inactive; i is 70.28% intensive |
| BOS attention follows inverse-competition slope −1 | rejected for baseline | across-length −.166; within-length −.336→−.107 |
| `k₀` and `Δ₁` are length-invariant | supported numerically | maximum relative discrepancy <4.4×10⁻⁷ |
| Surgical points Pareto-dominate coarse ones | strongly supported | d/i/b∧d and graded b points efficient; g/h cost +9.497/+5.536 nats |
| Damage concentrates late at long context | pathway-specific | true for b at 1024 (late 1.093), not c/i |
| Signature is content-invariant | strict version rejected | b shifts +10.7/+13.0 pp on uniform/repeated input |

E5 adds the functional interpretation missing from the original paper: a concentrated anchor appears useful, but its exact location is negotiable. The most effective mitigation should **attenuate or relocate** a sink, not indiscriminately destroy the positional object or whole computation blocks.

---

## 7. Integrated findings and revised narrative

### 7.1 From “one circuit” to “one anchor, multiple routes”

The original paper establishes a real and reproducible GPT-2-small route:

`first PE → first-layer MLP → EPE₀ massive coordinates → W_k → alignment with b_Q → source-agnostic Δ₀ → BOS sink`.

The new experiments show why this is not the whole mechanism. Since only T1 and T3 affect target choice, there are two natural ways to address the same positional key:

`shared positional anchor k₀≈EPE₀W_k`

- `Path A: b_Q · k₀` — fixed, source-agnostic, paper's circuit;
- `Path B: (x_iW_q) · k₀` — source-dependent, content-sensitive residual route.

Four independent observations identify the shared object:

1. c removes both routes and leaves only **1.7–3.9%** in GPT-2 small/medium/large.
2. b removes only A and leaves **44.5/70.4/74.0%**, increasingly dominated by B with scale.
3. b∧c falls to **2.25/1.64/3.65%**, so b's residual still requires the first-position object.
4. Swapping EPE moves **89.65/92.15/94.41%** of baseline sink mass to the new target; b∧swap still moves **.262/.260/.360** absolute attention there.

This is the central mechanistic contribution of the reproduction: **the anchor is shared; the query route is plural.**

### 7.2 The hierarchy of what generalizes

The complete evidence supports a hierarchy rather than a binary generalization verdict:

| level | generalizes? | evidence |
|---|---|---|
| Sink phenomenon | strongly | baseline .271–.677 across all 16 E1/E2 models; .429 at GPT-2 length 1024 |
| Early-position anchoring function | strongly | relocation, content stress tests, c across GPT-2/Neo, sinks in RoPE models |
| Positional key identity in absolute-PE GPT-like models | often | c leaves 2–6% in GPT-2 and Neo; d often relocates/removes |
| Two query-side routes in GPT-2 | strongly | exact score decomposition, combined interventions, scale trend |
| Dominance of `b_Q` route | no | GPT-2 residual grows with scale; Mistral b efficacy is 5.31%; Neo has no `b_Q` |
| Same raw intervention semantics across families | no | Qwen c/d/e and i differ; OPT e is often non-neutral |
| Same massive coordinate labels | decisively no | all 10 E3 top-3 pairwise Jaccards are 0 |

The paper's final architectural caution is therefore validated: optimization can build an anchoring computation from whatever components an architecture supplies.

### 7.3 Emergence explains apparent cross-model disagreement

Static checkpoints can make different mechanisms look categorically unrelated, but E3 shows that pathway balance itself is a moving target. The content route becomes effective early, the bias/EPE route passes through anti-alignment, massive coordinates overshoot by almost an order of magnitude, and the final model compresses their amplitude while strengthening the sink. Two checkpoints with the same sink strength could therefore have different internal balances, and two training runs can implement the same balance in disjoint coordinates.

This makes the cross-architecture results less mysterious. GPT-Neo realizes an anchor without `b_Q`; Qwen often relies heavily on a learned query bias plus token/MLP structure under RoPE; OPT switches among PE/EPE/MLP/key routes with scale. These are alternative points in a broader implementation space, not necessarily unrelated phenomena.

### 7.4 Content independence must be split into location and strength

The paper's “source-agnostic” term is mathematically source-independent, but the total sink is not. E5 gives the clean distinction:

- **Location/necessity is content-general:** c leaves **2.1–4.3%**, d **5.7–9.9%**, across natural, random, shuffled, and repeated input.
- **Strength/path mixture is content-sensitive:** baseline spans **.5084–.6020** across regimes; b spans **43.1–57.4%**; g spans **40.3–72.8%**; h spans **6.4–28.4%**.

The best wording is: **the anchor's address is positional, while the traffic routed to it depends on content.**

### 7.5 Function changes what “mitigation” means

The original work asks how to remove sinks but does not measure whether the anchor is useful. E5's frontier changes the question. Coarse deletion is expensive; targeted weakening can be cheap; relocation removes the BOS signature while preserving an anchor elsewhere.

- Cheap attenuation: `b_Q=.75`, **13.2% reduction for +.019 nat**.
- Surgical coordinate edit: i, **35.0% for +.257 nat**.
- High-removal relocation: d, **93.8% for +.938 nat**.
- Destruction: c, **97.0% for +3.757 nats**.
- Global disruption: h/g, **+5.536/+9.497 nats**.

Thus a low BOS-attention number is not automatically a better model state. Any proposed mitigation must report attention redistribution and language-model cost, not only sink reduction.

### 7.6 Robust claims suitable for the reproducibility paper

1. **Faithful reproduction:** all ten GPT-2-small interventions and all four structural analyses reproduce at the original scale.
2. **Residual mechanism:** GPT-2's residual sink is a second, content-query route to the same EPE-derived key object.
3. **Scale:** the source-agnostic share decreases with GPT-2 scale; the shared positional anchor persists.
4. **Architecture:** no single intervention signature is universal; ingredient-aware interventions reveal alternative circuits.
5. **Learning:** sink formation is non-monotonic and coordinate-degenerate across independently trained runs.
6. **Robustness:** the anchor survives length and severe content perturbations, while effect magnitude is metric-, length-, domain-, and content-sensitive.
7. **Function:** relocation and graded attenuation dominate deletion/global ablation on the sink-removal–CE frontier.

Claims that should **not** be made:

- that position 0 is completely separated in every layer (strictly true in only **3/12**);
- that downstream queries always have positive absolute cosine with the sink key (GPT-2 large mean is **−.0607**);
- that `b_Q` is universally necessary (Neo is the direct counterexample);
- that massive-coordinate identities reproduce (pairwise top-3 Jaccard is **0**);
- that baseline follows a −1 length law (slope **−.166**);
- that every sink-removal intervention causes late-context damage (only b does here);
- that one fixed random-coordinate draw defines a zero-centered null (E5 j increases the sink **3.47%**);
- that bootstrap uncertainty generally substitutes for reseeding (ratio **.94–3.63**).

---

## 8. Qualifications, failed or absent conditions, and interpretation hazards

1. **Data seeds versus model seeds.** E1/E2/E4/E5 vary evaluation samples, not pretrained weights. Only E3 varies training runs. Tiny E1/E2 cross-seed SD therefore establishes sampling stability, not parameter-seed universality.
2. **Only three data seeds in most new runs.** Cross-seed SDs and t-based three-seed CIs are themselves noisy. The older seven-seed result is stronger for GPT-2 small but its raw artifacts are absent from this export.
3. **Scaled layer bands.** E1/E2 and E4 use `[3,L−1)`, so deeper models average many more layers. This is the scientifically intended depth-aware comparison, but it is not a fixed-layer comparison.
4. **Architecture-specific labels.** Qwen's c/e/d and i are analogues, not the GPT-2 operations. OPT raw-PE swap e can be strongly causal, invalidating its use as a universal control.
5. **All-layer E1/E2 outputs are rounded.** The per-seed `bos_attention_summary_all_layers.csv` files store only two decimal places, so the mid-band aggregate remains the precision analysis.
6. **E3 checkpoint family.** Stanford-CRFM Mistral `gpt2-small` runs are not the OpenAI GPT-2 checkpoint and converge to a much stronger sink (**.897** versus about **.563**). E3 establishes learning dynamics and degeneracy, not an exact training replication of OpenAI GPT-2.
7. **E3 onset test is negative.** The preregistered early-alignment ordering has **p=.8125**. Descriptive timing should not be recast as a confirmed sequence.
8. **Nonlinear softmax counterfactuals.** A-only and B-only attention do not add to full attention. Likewise, stacked interventions can interact non-monotonically.
9. **Long-context construction.** The 1024-token inputs are within-domain concatenations, not necessarily coherent long documents. Domain boundaries and incoherence may affect b and positional CE profiles.
10. **Paired content inference.** The 900-pair t-tests ignore clustering by seed/source example and finite HumanEval overlap. Their p-values are descriptive.
11. **Random controls differ.** E1/E2 j resamples matched columns per layer; E5 j is one fixed set. Neither should be compared directly with the paper's original resampled j without noting the null definition.
12. **Missing formal checks.** E5 has no cached `--smoke-test` or `--verify-parity` report, although numerical a–i fidelity is excellent.
13. **Unrun planned conditions.** No GPT-2 XL E4 result, no E5 multilingual/FLORES result, and no dense PE₀→PE₁ interpolation between **.25 and .5** are present.
14. **Timing run is not evidence.** `e5_timing_length` has **5 examples** and is excluded.
15. **Duplicate exports.** The two E5 trees are one scientific run. Treating them as six seeds would be a serious error.
16. **Figure limitations.** E5-A's common linear scale is dominated by >1000%-of-baseline rank values; E5-D labels overlap; E5-C hides nonlinear fit failures. The CSVs, not visual area, should drive claims.

---

## 9. Rounded all-layer E1/E2 appendix

These are averages of the three per-seed values already rounded to two decimals by each harness; columns are a–j. They are included for completeness but should not replace the precise scaled-band table.

| model | a | b | c | d | e | f | g | h | i | j |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 small | .44 | .20 | .02 | .04 | .44 | .44 | .24 | .09 | .29 | .44 |
| GPT-2 medium | .51 | .36 | .01 | .05 | .51 | .51 | .08 | .05 | .16 | .51 |
| GPT-2 large | .45 | .33 | .02 | .02 | .45 | .46 | .10 | .02 | .19 | .45 |
| Neo 125M | .21 | .21 | .01 | .07 | .21 | .20 | .02 | .10 | .21 | .21 |
| Neo 1.3B | .29 | .29 | .01 | .23 | .29 | .13 | .02 | .08 | .26 | .29 |
| Neo 2.7B | .38 | .38 | .01 | .10 | .38 | .32 | .06 | .07 | .26 | .38 |
| OPT 125M | .60 | .47 | .02 | .34 | .29 | .60 | .12 | .02 | .24 | .60 |
| OPT 1.3B | .58 | .57 | .18 | .01 | .03 | .58 | .17 | .03 | .26 | .57 |
| OPT 2.7B | .59 | .58 | .20 | .09 | .20 | .59 | .13 | .04 | .26 | .59 |
| OPT 6.7B | .60 | .59 | .03 | .02 | .32 | .59 | .09 | .02 | .35 | .60 |
| OPT 13B | .34 | .33 | .02 | .31 | .33 | .33 | .18 | .19 | .36 | .33 |
| Qwen 0.5B | .44 | .03 | .44 | .42 | .42 | .43 | .02 | .35 | .43 | .43 |
| Qwen 1.5B | .47 | .03 | .45 | .39 | .39 | .00 | .02 | .37 | .46 | .47 |
| Qwen 3B | .45 | .07 | .45 | .42 | .42 | .29 | .01 | .40 | .45 | .45 |
| Qwen 7B | .49 | .04 | .49 | .47 | .47 | .00 | .03 | .45 | .47 | .49 |
| Qwen 14B | .59 | .27 | .59 | .59 | .59 | .01 | .03 | .57 | .59 | .59 |

---

## 10. Artifact census and source map

### Top-level result directories

| directory | files | bytes | scientific role |
|---|---:|---:|---|
| `results/e5_full` | 99 | 1,657,234,783 | canonical E5 export used here |
| `results/e5_timing_length` | 19 | 12,294,571 | 5-example timing only; excluded |
| `results/results_e3` | 188 | 1,517,452 | E3, 5 runs × 17 checkpoints |
| `results/results_e5` | 118 | 1,669,529,354 | duplicate nested E5/timing export |
| `results/results_gpt2_e1_e2` | 228 | 4,582,046 | GPT-2 small/medium/large E1/E2 |
| `results/results_gpt-neo_e1_e2` | 228 | 4,660,653 | GPT-Neo 125M/1.3B/2.7B E1/E2 |
| `results/results_opt_e1_e2` | 380 | 7,847,093 | OPT 125M/1.3B/2.7B/6.7B/13B E1/E2 |
| `results/results_qwen_e1_e2` | 380 | 8,050,345 | Qwen2.5 .5B/1.5B/3B/7B/14B E1/E2 |
| `results/results_gpt2_e4` | 37 | 464,949 | GPT-2-small E4 |
| `results/results_gpt2-medium_e4` | 37 | 503,069 | GPT-2-medium E4 |
| `results/results_gpt2-large_e4` | 37 | 537,646 | GPT-2-large E4 |
| `results/results_original authors` | 209 | 29,987,856 | local paper figures, structural analyses, Table 1 |

### Authoritative compact files

- Original reproduction: `results/results_original authors/dataset_analysis/bos_attention_stats_{overall,by_dataset}.csv`, `bias_term_statistical/bq_k_percentiles.csv`, `epe_validation_statistical/epe_validation_percentiles.csv`, and the structural `summary.txt` files.
- E1/E2: each model's `aggregate/table1_multiseed_summary_{overall,by_dataset}.csv`; configuration is in each seed's `dataset_analysis_*/run_config.json`.
- E3: `results/results_e3/emergence_gpt2-small/aggregate/{trajectories,onsets,coord_temporal_stability,table_E3_2_converged_vs_paper}.csv` plus `ordering_test.json` and `final_massive_coords.json`.
- E4: each scale's `aggregate/{decomposition,combined,dose_response,relocation,surgical,perplexity}_summary.csv` and `dose_response_monotonicity.csv`.
- E5: `results/e5_full/aggregate/`; the key files are `metrics_summary.csv`, `metric_concordance.csv`, `intensive_extensive_summary.csv`, `length_summary.csv`, `length_invariance_summary.csv`, `logit_competition_slopes_summary.csv`, `mitigation_frontier.csv`, `dose_cost.csv`, `position_ce_damage_region_summary.csv`, `content_summary.csv`, `content_paired_tests.csv`, `bootstrap_reseed_calibration.csv`, and `mde_power.csv`.

The raw per-example, per-head-cell, query-position, and token-position CSVs remain the audit trail behind these aggregates. This document is the portable scientific ledger; the `results/` tree is the machine-readable ledger.

## Final finding

The reproduction does more than confirm the paper. It preserves the original GPT-2 mechanism, explains its residual, identifies its training dynamics and coordinate degeneracy, maps its architectural boundary, rejects an overly simple context-length law, and attaches a functional cost to mitigation. The most coherent account across every experiment is:

> **Attention sinks are learned positional anchors. GPT-2 realizes the anchor through an EPE-derived key reached by both a fixed query-bias route and a content-query route. Training, scale, architecture, context, and content change the route balance and parameterization, while the anchoring function persists. Relocating or gently attenuating that function is substantially safer than deleting it.**
