# NNsight Results Summary and Paper-Narrative Audit

**Audit date:** 2026-07-16

**Result bundle:** `nn_results/nn_results`

**Legacy comparator:** `docs/Complete_Results_Synthesis.md` and the corresponding non-NNsight result trees

**Manuscript checked:** `current_paper.md`

## Executive verdict

The NNsight extension is complete and numerically healthy, but `current_paper.md` is not yet accurate as a description of the NNsight results.

- **E1/E2:** the cross-architecture conclusion survives. GPT-2, GPT-Neo, OPT-125M, and OPT-1.3B reproduce the legacy aggregates to the reported precision. The three larger OPT models and all five Qwen models need updated numerical cells. The largest pooled residual change is 4.29 percentage points and the largest domain-level change is 5.34 points; the qualitative family fingerprints remain the same.
- **E3:** this is the material revision. Weight-derived quantities and massive-coordinate identities are unchanged, but the NNsight/Hugging Face forward pass gives a final sink of **0.650 +/- 0.005**, not **0.897 +/- 0.008**, and final delta-dominance of **0.059 +/- 0.001**, not **0.555 +/- 0.013**. The sink onset moves from 7k to 20k steps and the ordering-test value changes from `p=.8125` to `p=.25`. The disjoint-coordinate result, the early anti-aligned phase, the EPE overshoot, the strong dependence on the first PE, and the negative ordering result all survive.
- **E4:** the GPT-2-small residual-sink anatomy is effectively unchanged at manuscript precision. NNsight validates the 124M run only; the 355M and 774M E4 values in the paper remain legacy-executor results.
- **E5:** the metric, length, content, and functional-cost conclusions reproduce. The only manuscript-level numerical correction is the nested-prefix invariance bound: the new maximum relative discrepancy is `<2e-6`, not `<4.4e-7`. A repeated-token/no-PE BOS-rank value is tie-sensitive, but BOS attention and the associated scientific conclusion are unchanged.

Accordingly, the paper's central claim -- the same sink can be built from architecture-, scale-, and run-specific plumbing -- remains supported. The E3 magnitude and route-decomposition story must be rewritten before the manuscript can be called accurate under the NNsight reference execution.

## Scope, authority, and integrity

For the extended results, the authoritative execution path is NNsight 0.7.0 over the real Hugging Face eager-attention forward pass, with Transformers 5.3.0 and PyTorch 2.10.0+cu128. Eager attention is required because the analyses consume realized attention probabilities, Q/K states, or logits. The legacy results are used only to quantify changes.

The result census is:

| Section | Files | Size (GiB) |
|---|---:|---:|
| E1/E2 | 1,713 | 0.053 |
| E3 | 188 | 0.001 |
| E4 | 55 | <0.001 |
| E5 | 99 | 1.861 |
| Production logs | 3 | 0.004 |
| Parity reports and logs | 11 | 0.001 |
| **Total** | **2,069** | **1.921** |

The physical bundle contains 938 CSV, 536 TXT, 287 JSON, 278 PNG, 27 LOG, and 3 NPZ files, totaling 2,062,645,551 bytes. The audit parsed every CSV, JSON, and NPZ artifact; scanned **20,076,878 CSV data rows**; found no NaN or infinite numerical values; and successfully opened all 278 PNG files. The E1/E2 census includes packaged legacy GPT-2/GPT-Neo copies as well as NNsight runs. Seed-level and aggregate exports also intentionally duplicate some physical observations, so physical row counts are not independent-sample counts.

Completion checks passed:

- all 16 E1/E2 model suites are marked complete;
- E3 contains all 5 runs x 17 checkpoints = 85/85 checkpoints;
- E4 and E5 production logs terminate normally;
- E1/E2, E4, and E5 sample manifests are byte-identical to their legacy counterparts, so executor comparisons use the same examples.

### Precision actually used

| Models | NNsight dtype |
|---|---|
| GPT-2 124M/355M/774M | float32 |
| GPT-Neo 125M/1.3B/2.7B | float32 |
| OPT 125M/1.3B | float32 |
| OPT 2.7B/6.7B/13B | bfloat16 |
| Qwen2.5 0.5B/1.5B/3B | float32 |
| Qwen2.5 7B/14B | bfloat16 |
| E3, E4, and E5 GPT-2-small runs | float32 |

This means the paper's blanket E1/E2 setup statement, "fp32 except Qwen-7B/14B," is false for the NNsight sweep: OPT-2.7B, OPT-6.7B, and OPT-13B are also bfloat16.

## Parity evidence

| Experiment | Gate | Result | Largest observed discrepancy |
|---|---:|---:|---|
| E1/E2 GPT-2 | 10 interventions | 10/10 pass | attention `7.72e-5`; BOS metric absolute `4.51e-7`; BOS metric relative `3.30e-6` |
| E3 GPT-2 | 9 forward quantities | 9/9 pass | absolute `3.52e-5`, relative `3.79e-5`, both for delta-score share |
| E4 GPT-2 | 48 quantities | 48/48 pass after the derived-ratio tolerance fix | delta-share ratio relative `2.953e-4` |
| E5 GPT-2 | 1,090 example/intervention/quantity rows | 1,090/1,090 pass | absolute `5.817e-5` for cross-entropy; relative `3.588e-5` for attention maps |

The initial E4 report passed 47/48 rows at a blanket `rtol=1e-4`. The only failure was the derived `decomposition/share_delta` ratio (`2.953e-4` relative; `0.003327` maximum cellwise absolute difference). Primitive quantities already passed. The corrected report applies a `5e-4` floor to that derived ratio and passes all 48 rows; it does not conceal a failed attention or logit comparison.

These parity gates strongly validate the GPT-2 NNsight traces. They do **not** provide separate family-specific parity gates for OPT or Qwen, so cross-engine changes for those families should be reported as observed differences, not automatically attributed to a single implementation detail.

## Intervention key

All E1/E2 and E5 percentages below are the percentage of that model or condition's own baseline BOS attention that survives.

| Label | Intervention |
|---|---|
| a | Baseline |
| b | Nullify query bias |
| c | Remove first positional encoding |
| d | Swap effective positional encoding |
| e | Swap raw positional encoding control |
| f | Nullify the BOS token embedding control |
| g | Remove MLP contribution |
| h | Remove positional encoding |
| i | Zero massive key-projection columns |
| j | Zero the same number of random key columns |

For Qwen, c/d/e/h are the registered RoPE position-ID analogues, d=e by construction, and i is the registered token-embedding stand-in. Equal labels across families therefore remain route-level analogues rather than identical tensor operations.

## E1/E2: cross-model NNsight sweep

### Authoritative pooled results

| Model | dtype | Baseline | b | c | d | e | f | g | h | i | j |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 124M | fp32 | 0.563700 | 44.46 | 2.96 | 6.25 | 98.94 | 99.65 | 50.14 | 18.03 | 65.15 | 100.06 |
| GPT-2 355M | fp32 | 0.575325 | 70.40 | 1.70 | 8.30 | 100.08 | 100.06 | 13.76 | 9.31 | 28.22 | 100.15 |
| GPT-2 774M | fp32 | 0.501453 | 74.04 | 3.88 | 3.41 | 99.96 | 100.46 | 20.61 | 3.21 | 42.10 | 99.86 |
| GPT-Neo 125M | fp32 | 0.271401 | 100.00 | 5.91 | 30.28 | 99.52 | 92.65 | 3.83 | 50.60 | 97.67 | 100.72 |
| GPT-Neo 1.3B | fp32 | 0.333645 | 100.00 | 2.44 | 79.87 | 99.94 | 40.08 | 6.60 | 26.30 | 88.35 | 100.06 |
| GPT-Neo 2.7B | fp32 | 0.412093 | 100.00 | 2.23 | 25.19 | 100.01 | 79.62 | 16.11 | 19.44 | 66.25 | 99.98 |
| OPT 125M | fp32 | 0.676644 | 82.82 | 2.12 | 60.64 | 46.31 | 99.84 | 17.84 | 3.49 | 39.69 | 99.81 |
| OPT 1.3B | fp32 | 0.626803 | 97.68 | 32.15 | 0.46 | 4.03 | 99.63 | 23.92 | 4.61 | 37.66 | 100.03 |
| OPT 2.7B | bf16 | 0.622245 | 98.18 | 36.48 | 13.30 | 33.50 | 99.78 | 17.90 | 7.54 | 38.67 | 99.78 |
| OPT 6.7B | bf16 | 0.630717 | 98.70 | 5.73 | 2.56 | 53.12 | 99.65 | 11.73 | 2.92 | 55.64 | 99.93 |
| OPT 13B | bf16 | 0.342986 | 98.83 | 6.78 | 93.50 | 98.52 | 99.15 | 50.19 | 58.57 | 111.92 | 98.33 |
| Qwen2.5 0.5B | fp32 | 0.492558 | 5.32 | 99.62 | 96.28 | 96.28 | 96.47 | 5.65 | 81.32 | 99.55 | 99.47 |
| Qwen2.5 1.5B | fp32 | 0.499918 | 6.91 | 95.08 | 83.09 | 83.09 | 1.28 | 5.27 | 82.32 | 99.24 | 99.95 |
| Qwen2.5 3B | fp32 | 0.481748 | 12.23 | 99.66 | 94.69 | 94.69 | 66.47 | 3.41 | 92.25 | 99.92 | 99.90 |
| Qwen2.5 7B | bf16 | 0.534766 | 6.06 | 99.44 | 96.52 | 96.52 | 0.85 | 6.72 | 91.87 | 97.15 | 100.01 |
| Qwen2.5 14B | bf16 | 0.636647 | 41.75 | 100.01 | 99.85 | 99.85 | 3.46 | 5.87 | 96.40 | 99.70 | 99.52 |

### Difference from the non-NNsight sweep

GPT-2 at all three scales, all three GPT-Neo models, OPT-125M, and OPT-1.3B match the legacy pooled tables to six displayed decimal places. The eight changed models are:

| Model | Legacy baseline -> NNsight | Largest residual change |
|---|---:|---|
| OPT 2.7B | 0.624710 -> 0.622245 | e: 32.59 -> 33.50% (`+0.91 pp`) |
| OPT 6.7B | 0.629982 -> 0.630717 | e: 53.38 -> 53.12% (`-0.25 pp`) |
| OPT 13B | 0.347479 -> 0.342986 | i: 110.11 -> 111.92% (`+1.81 pp`) |
| Qwen2.5 0.5B | 0.520009 -> 0.492558 | h: 77.02 -> 81.32% (`+4.29 pp`) |
| Qwen2.5 1.5B | 0.522555 -> 0.499918 | h: 78.75 -> 82.32% (`+3.57 pp`) |
| Qwen2.5 3B | 0.498988 -> 0.481748 | h: 89.06 -> 92.25% (`+3.19 pp`) |
| Qwen2.5 7B | 0.545251 -> 0.534766 | h: 90.14 -> 91.87% (`+1.72 pp`) |
| Qwen2.5 14B | 0.639642 -> 0.636647 | b: 45.44 -> 41.75% (`-3.70 pp`) |

The largest per-domain changes are similarly modest and do not reverse a family fingerprint:

| Model | Domain | Largest domain residual change |
|---|---|---|
| OPT 2.7B | HumanEval | c: 33.24 -> 34.96% (`+1.72 pp`) |
| OPT 6.7B | HumanEval | e: 52.75 -> 52.28% (`-0.47 pp`) |
| OPT 13B | HumanEval | i: 116.46 -> 119.09% (`+2.63 pp`) |
| Qwen2.5 0.5B | HumanEval | g: 4.84 -> 10.09% (`+5.25 pp`) |
| Qwen2.5 1.5B | HumanEval | h: 80.87 -> 85.32% (`+4.45 pp`) |
| Qwen2.5 3B | GSM8K | h: 90.43 -> 93.76% (`+3.33 pp`) |
| Qwen2.5 7B | HumanEval | h: 89.23 -> 91.72% (`+2.49 pp`) |
| Qwen2.5 14B | GSM8K | b: 40.05 -> 34.71% (`-5.34 pp`) |

The manifests are identical, so these are not sampling changes. For OPT-2.7B/6.7B/13B, executor change and fp32-to-bf16 change are confounded. For Qwen, the differences are execution-path differences under the registered actual Hugging Face eager operations; no family-specific parity report isolates the exact cause.

The scientific interpretation remains:

- GPT-2 uses both bias and positional/massive-key routes, with the bias route contributing less at larger scale.
- GPT-Neo has no query bias but still forms a sink through positional/MLP and increasingly important massive-key structure.
- OPT through 6.7B is nearly insensitive to b and much more sensitive to its key/positional channels. OPT-13B remains the qualitative outlier; its new baseline is 0.343 and zeroing massive Wk increases the sink to 111.92%.
- Qwen remains bias-carried: b leaves 5.32--12.23% at 0.5B--7B and 41.75% at 14B, while the registered positional and massive-key analogues are mostly inert.
- The random-column control remains near 100%, while MLP removal is the only intervention that reduces the sink in every family; its updated cross-model range is 3.41--50.19%.

The manuscript's seed-stability ceiling also needs a small update. Under NNsight the largest pooled cross-seed SD is **2.171 pp** (GPT-Neo-2.7B, h), followed by **1.821 pp** (GPT-Neo-1.3B, f), rather than `<=1.8 pp` for every cell. Because the Neo aggregates match the legacy run, that global ceiling was already slightly too tight; it is not a new NNsight disagreement. The largest-model maxima do change to 0.575 pp for OPT-13B and 0.241 pp for Qwen-14B, not 0.37 and 0.19 pp. The results remain sampling-stable; only the stated bounds are wrong.

## E3: emergence dynamics

### Converged results at step 400k

| Quantity (mean +/- run SD, n=5) | Legacy | NNsight |
|---|---:|---:|
| Sink strength | 0.8968 +/- 0.0076 | **0.6496 +/- 0.0049** |
| max abs(EPE0) | 1.3601 +/- 0.0397 | 1.3601 +/- 0.0397 |
| Bias-key alignment at position 0 | 0.7951 +/- 0.0268 | 0.7951 +/- 0.0268 |
| Delta-dominance | 0.5550 +/- 0.0126 | **0.05925 +/- 0.00096** |
| Content-pathway attention | 0.8181 +/- 0.0110 | **0.5520 +/- 0.0042** |
| Query alignment at position 0 | 0.2784 +/- 0.0193 | **0.3039 +/- 0.0206** |
| Query-alignment control | 0.1338 +/- 0.0902 | **0.0876 +/- 0.1127** |
| Delta-score share | 0.1383 +/- 0.0039 | **0.1043 +/- 0.0020** |
| Fraction removed by nullifying b | 5.31 +/- 0.26% | **7.71 +/- 0.48%** |
| Fraction removed by removing first PE | 99.34 +/- 0.30% | **98.52 +/- 0.40%** |
| Fraction removed by zeroing massive Wk | 27.07 +/- 13.36% | **42.02 +/- 11.65%** |

The five NNsight final sinks are 0.64492, 0.65456, 0.65153, 0.64389, and 0.65295. Thus the five runs still produce a stronger sink than the OpenAI GPT-2 reference value of about 0.563, but the difference is modest, not 0.897 versus 0.563.

### Dynamics and onsets

| Step | NNsight sink | Legacy sink | max abs(EPE0) | Bias alignment | Delta-dominance |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0341 | 0.0344 | 0.065 | 0.000 | 0.034 |
| 400 | 0.0201 | 0.0105 | 0.098 | -0.616 | 0.034 |
| 1k | 0.1193 | 0.2030 | 0.149 | -0.452 | 0.035 |
| 10k | 0.3026 | 0.5389 | 3.114 | -0.043 | 0.047 |
| 40k | 0.5139 | 0.7764 | 10.196 | 0.614 | 0.078 |
| 70k | 0.5532 | 0.8115 | 10.173 | 0.698 | 0.079 |
| 100k | 0.5823 | 0.8393 | 8.510 | 0.720 | 0.075 |
| 400k | 0.6496 | 0.8968 | 1.360 | 0.795 | 0.059 |

The NNsight sink falls below its initialization level at step 400, then increases at every stored checkpoint thereafter. The paper should therefore say **post-dip monotone growth**, not that the behavior rises monotonically over the entire run. The EPE overshoot remains: it peaks near 10.20 at 40k--70k and relaxes by about 7.5x to 1.36.

NNsight 50%-of-final onsets are:

| Signal | Median onset | IQR |
|---|---:|---:|
| Sink | 20k | 20k--20k |
| max abs(EPE0) | 20k | 20k--20k |
| Bias-key alignment | 200 | 100--20k |
| Delta-dominance | 20k | 20k--20k |
| Query alignment | 0 | 0--0 |

The preregistered ordering result remains negative: early alignment precedes the sink in 3/5 runs, with Wilcoxon statistic 0 and `p=.25`. This replaces statistic 6 and `p=.8125`.

### What did and did not change

The EPE amplitude, bias-key alignment, massive-coordinate sets, and temporal coordinate-stability files are identical to the legacy outputs because they are weight-derived. Final top-three sets remain:

- alias: `{683, 8, 541}`
- battlestar: `{618, 318, 341}`
- caprica: `{115, 229, 439}`
- darkmatter: `{610, 470, 192}`
- expanse: `{529, 233, 41}`

All ten pairwise Jaccard indices are still zero, and none of the five top-three sets contains the OpenAI GPT-2 set `{138, 378, 447}`. The arbitrary-basis conclusion is therefore unchanged.

The changed quantities all depend on the forward execution path. Identical samples plus identical weight-derived outputs rule out data and checkpoint-selection changes. The E3 GPT-2 parity gate makes an ordinary NNsight tracing error unlikely, but it does not by itself identify the precise legacy-executor semantic difference. The safe reporting rule is to treat the NNsight/Hugging Face eager forward as the reference and retain the legacy numbers only as comparison history.

### Correct E3 interpretation

The content-query pathway is still the dominant route: its isolated attention is 0.552, while nullifying the learned query bias removes only 7.7% of the full sink and the delta-only diagnostic is 0.059. The first-position key remains load-bearing: removing the first PE removes 98.5% of the sink. What no longer holds is the claim that delta-dominance converges near 0.555 or that the full sink converges near 0.897.

A manuscript-ready replacement paragraph is:

> Across five independent Mistral GPT-2-small runs (85/85 checkpoints), the NNsight/Hugging Face forward pass yields a converged BOS sink of 0.650 +/- 0.005. Weight-derived EPE amplitude and bias-key alignment are 1.360 +/- 0.040 and 0.795 +/- 0.027, while query alignment is 0.304 +/- 0.021, content-pathway attention is 0.552 +/- 0.004, delta-dominance is 0.059 +/- 0.001, and delta-score share is 0.104 +/- 0.002. Removing the first PE removes 98.5 +/- 0.4% of the sink, nullifying the query bias removes 7.7 +/- 0.5%, and zeroing massive key columns removes 42.0 +/- 11.7%. The sink dips from 0.034 at initialization to 0.020 at step 400 and then grows monotonically to convergence, while EPE amplitude overshoots to 10.20 near 40k and relaxes to 1.36. Final top-three massive-coordinate sets remain pairwise disjoint. The preregistered onset-ordering test remains negative (3/5 runs, Wilcoxon p=.25).

## E4: residual-sink anatomy

The NNsight E4 run is GPT-2-small only. Its core aggregates reproduce the legacy run at manuscript precision:

| Result | NNsight value |
|---|---:|
| Full baseline BOS attention | 0.56370026 |
| Content-pathway-only attention | 0.21046512 |
| Delta-pathway-only attention | 0.37651098 |
| Delta share of position-0 advantage | 0.56098163 |
| Query-to-position-0 alignment | 0.22814290 |
| Alignment control | 0.02311285 |
| Alignment Cohen's d | 1.7851 |
| b + c residual | 2.2544% |
| b + d residual | 4.2288% |
| b + i residual | 24.7733% |
| b + i + d residual | 4.9657% |
| Swap-EPE relocation ratio | 0.896546 |
| Swap target (position 1) attention | 0.511512 |
| b + swap target attention | 0.261825 |

The dose-response findings also reproduce: b and massive-Wk scaling are perfectly monotone over the stored alpha grid; PE-identity interpolation is threshold-like; raw-PE deletion changes sign with scale in the legacy cross-scale runs.

Functional values at 124M are unchanged at paper precision:

- baseline cross-entropy: 4.00911 nats;
- nullify b: 5.59552 (`+1.58642`);
- zero massive Wk: 4.26770 (`+0.25859`);
- first-layer MLP skip: 6.43170 (`+2.42259`);
- first-layer MLP skip leaves 102.53% of baseline BOS attention, while all-layer no-MLP leaves 50.14%.

The paper may retain the GPT-2-small E4 numbers. Its medium/large decomposition, dose-response, relocation, combined-intervention, and cross-entropy claims are still supported by the legacy executor only and should not be described as NNsight-confirmed.

## E5: metrics, length, content, and cost

### Metric multiverse at 40 tokens

| Intervention | BOS attention | % of baseline |
|---|---:|---:|
| a | 0.562528 | 100.00 |
| b | 0.249849 | 44.42 |
| c | 0.016597 | 2.95 |
| d | 0.034954 | 6.21 |
| e | 0.555230 | 98.70 |
| f | 0.560595 | 99.66 |
| g | 0.283620 | 50.42 |
| h | 0.102358 | 18.20 |
| i | 0.365922 | 65.05 |
| j | 0.582042 | 103.47 |

The concordance matrix is identical to the legacy output. Mean Kendall concordance is high; all sub-0.8 comparisons involve attention entropy. The mechanistic distinction also remains: d relocates 0.552 of mass into positions 1--4 and c diffuses about 0.930 to positions 5+, so BOS reduction alone does not distinguish relocation from deletion.

At the preregistered 0.3 head-cell threshold, c and d deactivate 99.87% and 97.96% of previously active cells; i leaves 70.28% active but shrinking; b deactivates 57.30% and leaves 40.95% active but shrinking. These reproduce the paper's rounded intensive/extensive claims.

### Length generalization

| Length | Baseline | b | c | d | i | j |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.562528 | 44.42% | 2.95% | 6.21% | 65.05% | 103.47% |
| 128 | 0.500068 | 44.80% | 1.39% | 4.60% | 59.28% | 97.32% |
| 256 | 0.468529 | 45.73% | 0.72% | 20.32% | 56.24% | 100.01% |
| 512 | 0.446780 | 41.97% | 0.35% | 20.29% | 51.02% | 99.22% |
| 1024 | 0.429379 | 46.64% | 0.08% | 13.42% | 46.59% | 97.06% |

The baseline falls only 23.7% while context grows 25.6x. c strengthens with length, d is non-monotone, i weakens steadily, and j remains near 100%. The new maximum nested-prefix discrepancy is `1.91e-6` relative for the position-0 key and `6.57e-7` for the delta object. The paper should conservatively report `<2e-6` or "float32-scale agreement."

### Content generality

| Content regime | Baseline | b | c | d | e | h | i |
|---|---:|---:|---:|---:|---:|---:|---:|
| SST-2 | 0.599575 | 44.63 | 2.31 | 6.47 | 96.93 | 16.68 | 65.69 |
| GSM8K | 0.547840 | 43.09 | 2.37 | 7.06 | 99.10 | 23.84 | 65.97 |
| HumanEval | 0.540168 | 45.52 | 4.25 | 5.07 | 100.26 | 14.15 | 63.40 |
| Uniform random | 0.601977 | 55.11 | 2.09 | 5.66 | 99.55 | 20.57 | 68.03 |
| Zipf/unigram | 0.522511 | 45.66 | 2.63 | 8.55 | 97.01 | 26.75 | 63.25 |
| Shuffled natural | 0.508370 | 46.01 | 2.93 | 8.27 | 96.91 | 28.43 | 63.65 |
| Repeated token | 0.532544 | 57.45 | 4.34 | 9.95 | 81.35 | 6.39 | 72.79 |

These values support the existing narrative: the anchor address is content-general (c leaves 2.09--4.34%; d leaves 5.07--9.95%), but route traffic changes with content. In particular, b rises to 57.45% and the raw-PE control e falls to 81.35% on repeated tokens.

The only conspicuous legacy-to-NNsight aggregate difference in E5 is BOS rank for repeated-token h: 14.356 becomes 13.689, or 496.09% becomes 473.05% of its baseline. This condition creates near-ties after positional information is removed, so rank changes under tiny floating-point perturbations. Its BOS attention changes only from 0.034040166 to 0.034040174; the scientific conclusion is unaffected.

### Functional cost and mitigation

| Intervention | BOS reduction | Delta cross-entropy (nats) |
|---|---:|---:|
| b | 55.59% | 1.605 |
| c | 97.02% | 3.757 |
| d | 93.80% | 0.938 |
| i | 34.98% | 0.257 |
| g | 49.23% | 9.497 |
| h | 81.78% | 5.536 |
| b + c | 97.73% | 4.129 |
| b + d | 95.81% | 1.790 |
| b + i | 75.24% | 2.680 |
| b + i + d | 95.07% | 2.774 |

The paper's main cost claim is unchanged: d achieves deletion-level BOS reduction for about one quarter of c's language-model cost (`0.938/3.757 = 0.250`). Scaling b to 0.75 and 0.5 removes 13.19% and 27.18% for only 0.019 and 0.188 nats. Relocation and graded attenuation remain preferable to coarse global ablations.

## Audit of `current_paper.md`

### Required corrections

| Paper location or claim | Status under NNsight | Required change |
|---|---|---|
| Setup: fp32 except Qwen-7B/14B | Incorrect | Add OPT-2.7B/6.7B/13B to the bf16 list. |
| E1/E2: every cross-seed SD `<=1.8 pp`; OPT-13B `<=0.37`, Qwen-14B `<=0.19` | Incorrect bounds | Use `<=2.18 pp` globally; OPT-13B `<=0.58`, Qwen-14B `<=0.25`. |
| Figure 1/Table 4 baselines and residuals | Partly stale | Replace the three large-OPT and five Qwen rows with the NNsight table above. Regenerate Figure 1. |
| Qwen b residual `6.1--14.5%` through 7B and `45.4%` at 14B | Stale | Use `5.3--12.2%` through 7B and `41.8%` at 14B. |
| Qwen f values `0.5/0.5/1.8%` for 1.5B/7B/14B | Stale | Use `1.28/0.85/3.46%`. |
| Cross-family g range `2.6--50.1%` | Stale | Use `3.41--50.19%`. |
| OPT-13B baseline `0.347`, i `110.1%` | Stale | Use `0.343`, i `111.92%`. The anomaly conclusion strengthens. |
| E3 final sink `0.897 +/- 0.008` | Incorrect | Use `0.650 +/- 0.005`. |
| E3 query alignment `0.278 +/- 0.019` | Incorrect | Use `0.304 +/- 0.021`. |
| E3 delta-dominance `0.555 +/- 0.013`, peak `0.735 -> 0.555` | Incorrect | Use final `0.059 +/- 0.001`, with a peak near `0.079` at 70k. |
| E3 b/first-PE/massive-Wk efficacy `5.3/99.3/27.1%` | Incorrect | Use `7.7/98.5/42.0%`. |
| Figure 2 and Table 7 sink/delta columns | Incorrect | Regenerate from NNsight E3 aggregates. Coordinate labels remain valid. |
| Sink "rises monotonically" | Incorrect over the complete trajectory | Say it dips at step 400 and is monotone thereafter. |
| E3 sink onset 7k | Incorrect | Use 20k. |
| Ordering test statistic 6, `p=.8125` | Incorrect | Use statistic 0, `p=.25`; the negative conclusion remains. |
| Limitations: Mistral converges to a much stronger `0.897` sink | Overstated | It converges to `0.650`, still above the OpenAI value `0.563`. |
| E5 nested-prefix bound `<4.4e-7` | Incorrect | Use `<2e-6` or "float32-scale agreement." |
| Reproducibility archive `~3.4 GB, 1,960 files` | Scope-dependent | If `nn_results/nn_results` is the new released archive, use 2,069 files and 2,062,645,551 bytes (1.921 GiB). If the sentence still refers only to the old archive, label it explicitly as the legacy archive. |

### Claims that remain supported

- The GPT-2-small intervention reproduction and GPT-2 NNsight parity are strong.
- Baseline sinks occur across all 16 checkpoints, spanning roughly 0.27--0.68.
- No intervention fingerprint is architecture-independent.
- GPT-2's bias contribution falls with scale; OPT is primarily key/positional through 6.7B; Qwen is primarily bias-carried; GPT-Neo forms a sink without query bias.
- Massive-coordinate labels are not transferable across independent training runs; all ten top-three pairwise Jaccards remain zero.
- Correlational alignment does not imply causal importance: E3 bias-key alignment remains high while b removes only 7.7%.
- The first-position key/positional identity remains universally load-bearing in the E3 runs.
- The metric multiverse, length generalization, and content-general address/content-specific traffic narrative remain supported.
- Relocating the anchor costs about one quarter as much as deleting it.

### Claims that need provenance qualification

- E4 cross-scale results at GPT-2-medium and GPT-2-large were not rerun with NNsight. They remain valid legacy results but are not NNsight-confirmed.
- The special OPT-13B single-sentence two-position probe in Appendix C was not independently revalidated by the aggregate NNsight sweep. Its dataset-level anomaly is reproduced; the probe should retain a legacy-executor label unless rerun.
- OPT/Qwen NNsight changes lack architecture-specific manual-versus-NNsight parity reports. Report the observed new values and the dtype/executor differences without asserting a uniquely identified cause.

## Operational caveats from logs

- The first E1/E2 launch stopped before computation because Windows `cp1252` could not print Unicode box characters. The resumed run completed; this did not alter data.
- The E3 log contains 34 background `Thread-auto_conversion` tracebacks from Transformers' safetensors conversion thread (`KeyError: event_id`). The forward jobs continued and all 85 checkpoint outputs are present, finite, and aggregated. This is a provenance warning, not evidence of missing results; a clean archival rerun could eliminate the noisy background failure.
- The first E4 parity run used an overly strict blanket tolerance for a derived ratio. The corrected ratio-aware gate is documented rather than silently replacing the original failed report.
- Repeated-token/no-PE rank statistics are sensitive to floating-point tie breaking; attention mass is the stable quantity there.

## Authoritative artifact map

- E1/E2 NNsight model runs and aggregate tables: [`nn_results/nn_results/e1_e2/nnsight_results`](../nn_results/nn_results/e1_e2/nnsight_results)
- E3 aggregate trajectories, onsets, coordinates, and figures: [`nn_results/nn_results/e3_emergence_dynamics/aggregate`](../nn_results/nn_results/e3_emergence_dynamics/aggregate)
- E4 GPT-2-small aggregates: [`nn_results/nn_results/e4_residual_sink/aggregate`](../nn_results/nn_results/e4_residual_sink/aggregate)
- E5 aggregates: [`nn_results/nn_results/e5_evaluation_robustness/aggregate`](../nn_results/nn_results/e5_evaluation_robustness/aggregate)
- Parity reports: [`nn_results/nn_results/parity_reports`](../nn_results/nn_results/parity_reports)
- Production logs: [`nn_results/nn_results/logs`](../nn_results/nn_results/logs)
- NNsight implementation specification: [`docs/NNsight_E3_E4_E5_IMPLEMENTATION.md`](NNsight_E3_E4_E5_IMPLEMENTATION.md)
- Prior non-NNsight synthesis: [`docs/Complete_Results_Synthesis.md`](Complete_Results_Synthesis.md)
- Manuscript audited here: [`current_paper.md`](../current_paper.md)
