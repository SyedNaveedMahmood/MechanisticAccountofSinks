# E5 Results: Scientific Evaluation and Summary

## Scope and provenance

This report evaluates the completed GPT-2-small E5 run in
`results/e5_full/`. It interprets the results against the E5 preregistration,
the E4 two-pathway account, the earlier seven-seed Table 1 study, and the broader
reproduction plan.

The run used:

- GPT-2 small (`gpt2`), FP32;
- layers 4–11 (zero-indexed band `[3,11)`);
- data-resampling seeds 0, 1, and 2;
- 100 examples each from SST-2, GSM8K, and HumanEval;
- lengths 40, 128, 256, 512, and 1024, with 50 examples/domain at 1024;
- massive coordinates `{138, 378, 447}`;
- fixed random-`W_k` coordinates `{46, 420, 669}` from seed 1729;
- 2,000 bootstrap repetitions within each seed.

The optional multilingual experiment was not requested. The separate
`results/e5_timing_length/` directory is a five-example timing run and is excluded
from all scientific conclusions below.

## Result-integrity audit

The full cache is internally complete and consistent:

- 9,000 metric rows, 24,300 length rows, 25,200 cost rows, and 45,000 content rows;
- 270/270 logit-slope fits have `status=ok`;
- all 25,200 CE examples and all 1,981,800 token-position CE rows have `status=ok`;
- no literal NaN or infinite experimental values;
- all three manifests contain exactly 100 unique examples per natural domain;
- every long manifest contains exactly 1,024 tokens;
- 40-token BOS measurements are bit-identical across metrics, length, cost, and
  natural-content modes for every shared example/intervention;
- all six aggregate figures and all planned aggregate tables are present.

Blank fields in cost tables are expected: the warning column is empty for successful
runs, and sink-reduction-per-nat is undefined at exact zero cost. No failed value was
silently replaced.

One reproducibility qualification matters for uncertainty estimates: the HumanEval
source pool is small enough that each pair of 100-example seed samples shares 60
anchors (Jaccard 0.429). SST-2 overlaps by 9–11 anchors and GSM8K by 0–3. The seed
samples are therefore resamples, but not fully independent draws from an effectively
infinite population—especially for HumanEval.

No cached smoke-test or parity report is present. The close reproduction of Table 1
provides strong empirical fidelity for interventions a–i, but this report does not
claim that the explicit smoke/parity checks passed.

## Executive conclusions

1. **The original Table 1 result reproduces closely.** Baseline BOS attention is
   0.5625 versus the reported 0.563, and interventions b–i closely match their
   published percentages.
2. **The conclusion is mostly, but not completely, metric-robust.** Mean off-diagonal
   Kendall concordance is 0.84, but 5 of 15 metric pairs fall below the preregistered
   0.8 threshold. Entropy is the main dissenter because it measures concentration
   anywhere, not specifically at BOS.
3. **Sink removal is mechanistically heterogeneous.** Zero-Top-k mostly shrinks active
   cells; Remove-First-PE and Swap-EPE eliminate nearly all previously active cells.
   The preregistered universal “intensive margin” account is not supported.
4. **The fixed `k0`/`Δ1` premise is confirmed, but the slope −1 law is rejected for
   baseline.** Baseline BOS attention declines much more slowly than inverse
   competition, and within-length fitted slopes approach 0 rather than −1 as context
   grows.
5. **A strong mitigation frontier exists.** Surgical interventions Pareto-dominate
   No-MLP and No-PE. Swap-EPE and Zero-Top-k are the clearest efficient points.
6. **Relocation has substantially lower measured cross-entropy cost than deletion.** Swap-EPE removes 93.8% of
   BOS mass for +0.94 nat, whereas Remove-First-PE removes 97.0% for +3.76 nats.
   Swap-EPE redirects attention locally; deletion sends it broadly to later keys.
7. **Late-position damage is pathway-specific, not universal.** Nullifying `b_Q`
   develops late-position damage at 1024 tokens, but Remove-First-PE and Zero-Top-k
   do not show the preregistered general late-position pattern.
8. **The circuit is qualitatively content-general but quantitatively content-sensitive.**
   Remove-First-PE remains devastating everywhere, but the residual after nullifying
   `b_Q` changes markedly on uniform-random and repeated-token inputs.
9. **The fixed random-Wk draw is not a null attention control.** It strengthens BOS
   attention by 3.47% with near-zero CE cost. Multiple fixed random draws are needed
   before treating this control distribution as centered at zero.

## Hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H-M1: intervention rankings are concordant across metrics (τ ≥ 0.8) | **Partially supported** | Mean τ = 0.84, but 5/15 pairs are below 0.8; entropy vs sink-rate@0.5 is only 0.60. |
| H-M2: circuit interventions act mainly on the intensive margin | **Not supported as a general claim** | At threshold 0.3, 99.9% of active cells fall below threshold under c and 98.0% under d; i is mostly intensive, while b is mixed. |
| Length law: logit BOS attention has slope near −1 | **Rejected for baseline** | Mean baseline slopes are −0.336, −0.213, −0.161, −0.108, and −0.107 from length 40 to 1024. |
| Length invariance of `k0` and `Δ1` | **Supported numerically** | Maximum relative discrepancy is below 4.4×10⁻⁷ for both quantities. |
| H-C1: surgical interventions Pareto-dominate coarse interventions | **Strongly supported** | d and i are Pareto-efficient in all 9 seed×domain strata; g and h are efficient in 0/9. |
| H-C2: damage concentrates late and worsens at long context | **Pathway-specific support only** | Supported for b at 1024; contradicted for c and largely absent for i. |
| H-D1: intervention signatures match natural text on arbitrary content | **Partially supported; strict invariance rejected** | c and d stay strongly effective, but b shifts by +10.7 to +13.0 percentage points on uniform/repeated input and i shifts +7.7 points on repeated input. |

## E5.1 — Metric multiverse

### Table 1 fidelity

| Intervention | BOS attention | % baseline | Earlier/paper target |
|---|---:|---:|---:|
| a Baseline | 0.5625 | 100.0% | 0.563 |
| b Nullify Query Bias | 0.2498 | 44.4% | 44.7% |
| c Remove First PE | 0.0166 | 3.0% | 3.0% |
| d Swap EPE | 0.0350 | 6.2% | 6.2% |
| e Swap PE | 0.5552 | 98.7% | 99.0% |
| f Nullify BOS Token | 0.5606 | 99.7% | 99.7% |
| g No MLP | 0.2836 | 50.4% | 50.3% |
| h No PE | 0.1024 | 18.2% | 17.5% (18.4% in the prior multiseed run) |
| i Zero Top-k Wk | 0.3659 | 65.0% | 65.2% |
| j Fixed Random Wk | 0.5820 | 103.5% | Not directly comparable to legacy resampling |

The a–i agreement is strong evidence that the optimized E5 executor preserves the
validated intervention semantics.

### Metric agreement is high but scientifically informative disagreement remains

The direct BOS metrics broadly agree:

- BOS attention vs sink rate@0.5: τ = 0.956;
- BOS attention vs sink rate@0.2 or @0.3: τ = 0.867;
- BOS attention vs BOS rank: τ = 0.867.

Entropy is different:

- entropy vs BOS attention: τ = 0.644;
- entropy vs sink rate@0.5: τ = 0.600.

This disagreement is not merely noise. Swap-EPE (d) almost eliminates BOS attention
(6.2% of baseline) but retains relatively concentrated attention: normalized entropy
is 0.488, lower than b/g/c/h. It moves 0.552 attention mass to positions 1–4, versus
only 0.021 at baseline. By contrast, Remove-First-PE (c) sends 0.930 mass to positions
5+, with entropy 0.607. Thus:

> Removing a BOS sink can either relocate a concentrated anchor locally or diffuse
> attention broadly. BOS mass and global entropy are not interchangeable sink metrics.

This is a useful correction to H-M1: metric disagreement exposes different causal
outcomes rather than invalidating the intervention result.

### Head-cell heterogeneity rejects a single-margin story

At the 0.3 active-cell threshold:

| Intervention | Baseline→post slope | Pearson r | Active cells falling below threshold | Remaining active but shrinking |
|---|---:|---:|---:|---:|
| b Nullify Query Bias | 0.527 | 0.588 | 57.3% | 40.9% |
| c Remove First PE | 0.017 | 0.110 | 99.9% | 0.1% |
| d Swap EPE | 0.087 | 0.206 | 98.0% | 1.3% |
| i Zero Top-k Wk | 0.608 | 0.780 | 28.9% | 70.3% |

The massive-coordinate edit i behaves mainly as an intensive attenuation. Removing
or relocating the shared positional key object (c/d) is extensive: almost every
previously active cell crosses below threshold. Nullifying `b_Q` is mixed. This fits
the E4 two-pathway account better than a universal proportional-shrinkage model.

Residual BOS mass also becomes much more head-concentrated: Gini rises from 0.206 at
baseline to 0.421 under b, 0.679 under c, and 0.690 under d. Small pooled residuals are
therefore carried by a highly unequal subset of layer/head cells.

## E5.2 — Length generalization

### The sink persists, but intervention effects are not length-invariant

| Length | Baseline BOS | b (% base) | c (% base) | d (% base) | i (% base) | j (% base) |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.5625 | 44.4% | 3.0% | 6.2% | 65.0% | 103.5% |
| 128 | 0.5001 | 44.8% | 1.4% | 4.6% | 59.3% | 97.3% |
| 256 | 0.4685 | 45.7% | 0.7% | 20.3% | 56.2% | 100.0% |
| 512 | 0.4468 | 42.0% | 0.4% | 20.3% | 51.0% | 99.2% |
| 1024 | 0.4294 | 46.6% | 0.1% | 13.4% | 46.6% | 97.1% |

The baseline declines by only 23.7% while the number of positions grows 25.6-fold.
Even at 1024, 90.7% of layer/head cells exceed BOS mass 0.2 and 77.2% exceed 0.3.
The sink is therefore not a short-context artifact. It weakens on the intensive
margin: cells above 0.5 fall from 62.9% to 34.3%, while low-threshold prevalence stays
high.

The relative effects are not invariant:

- c becomes steadily stronger, leaving only 0.1% of baseline at 1024;
- i strengthens from a 35% reduction at length 40 to a 53% reduction at 1024;
- d is non-monotonic, rising from 6.2% of baseline at 40 to about 20% at 256/512,
  then falling to 13.4% at 1024;
- pooled b looks roughly stable, but at 1024 it leaves 30.5% of baseline on SST-2,
  53.4% on GSM8K, and 57.9% on HumanEval.

The 1024 domain split is a boundary condition for the content-agnostic claim. It may
reflect genuine content/pathway differences, the concatenation construction, or both.

### The fixed-object premise passes, but the competition law fails

Across identical nested prefixes:

- `k0` maximum absolute discrepancy is at most 1.88×10⁻⁶;
- `Δ1` maximum absolute discrepancy is at most 7.63×10⁻⁶;
- maximum relative discrepancy is below 4.4×10⁻⁷.

This confirms the causal-mask invariance premise to ordinary FP32 rounding error.
Nevertheless, the baseline logit slopes are far from −1 and become shallower with
length: −0.336 at 40, −0.213 at 128, −0.161 at 256, −0.108 at 512, and −0.107 at
1024 (averaged over domains). A post-hoc across-length fit of logit(mean BOS mass)
against log(length) is only −0.166 for baseline. Only c is close to −1 across lengths
(−1.16), after the sink has almost vanished.

Therefore the simple “fixed logit advantage against i exchangeable competitors”
model is insufficient. The other key/query scores, their distribution, and/or the
number of effectively competitive positions must co-vary with context. The result
does not refute invariance of `Δ1`; it refutes the step from invariant `Δ1` to an
inverse-competition attention law.

Extreme d/b slope estimates at some long lengths should not be interpreted as new
laws. Their query-position curves are non-linear, and low probabilities make logits
sensitive. The raw position curves and fit residuals should be shown before assigning
a mechanistic meaning to those slopes.

## E5.3 — Functional cost and mitigation frontier

The pooled baseline cross-entropy is 4.021 nats (SST-2 4.959, GSM8K 3.530,
HumanEval 3.573).

### Main frontier

| Intervention | Sink reduction | ΔCE (nats) | Interpretation |
|---|---:|---:|---|
| scale `b_Q` = 0.75 | 13.2% | +0.019 | Very cheap partial mitigation |
| scale `b_Q` = 0.50 | 27.2% | +0.188 | Smooth dose-response point |
| i Zero Top-k Wk | 35.0% | +0.257 | Efficient surgical coordinate edit |
| scale `b_Q` = 0.25 | 43.0% | +0.775 | Efficient stronger dose |
| b Nullify Query Bias | 55.6% | +1.605 | Dominated by graded/surgical alternatives |
| d Swap EPE | 93.8% | +0.938 | Best high-removal trade-off |
| b∧d | 95.8% | +1.790 | More removal, higher cost |
| c Remove First PE | 97.0% | +3.757 | Deletion is much costlier than relocation |
| b∧c | 97.7% | +4.129 | Maximum removal in this suite |
| h No PE | 81.8% | +5.536 | Coarse and dominated |
| g No MLP | 49.2% | +9.497 | Catastrophic functional damage for modest removal |

Swap-EPE (d), Zero-Top-k (i), b∧d, and the 0.25/0.75 query-bias doses are
Pareto-efficient in all nine seed×domain strata. No-MLP, No-PE, full query-bias
nullification, and the random control are efficient in zero of nine. H-C1 is therefore
supported both in the aggregate and within every seed/domain stratum.

The most mechanistically informative comparison is d versus c. Both attack the
position-0 EPE identity and remove more than 93% of the BOS sink, but relocation costs
about one quarter as much CE as deletion. Combined with d's 0.552 local attention mass,
this suggests that the model benefits from retaining a concentrated positional anchor,
even when that anchor is moved away from BOS. “Mitigation” should therefore distinguish
removing sink mass from destroying the anchoring function it serves.

### Dose response

Scaling `b_Q` gives a clean graded frontier: as alpha falls from 1 to 0, sink reduction
increases smoothly from 0 to 55.6% and CE cost rises from 0 to 1.61 nats. This is strong
graded-causality evidence for pathway A.

PE interpolation is sharply nonlinear. Alpha 0.5 has essentially no sink reduction
(−0.4%) and +0.056 nat, while alpha 0.25 suddenly removes 95.5% with +3.71 nats.
This looks like a threshold transition rather than a smooth dose response. A denser
sweep between 0.25 and 0.5 is needed before fitting or claiming monotonic graded damage.

### Position-level damage

Mean ΔCE by normalized region:

| Length | Intervention | Early | Middle | Late |
|---:|---|---:|---:|---:|
| 40 | b | 2.289 | 1.597 | 0.929 |
| 40 | c | 5.602 | 3.289 | 2.379 |
| 40 | i | 0.430 | 0.199 | 0.143 |
| 1024 | b | 0.546 | 0.521 | 1.093 |
| 1024 | c | 1.746 | 1.161 | 1.428 |
| 1024 | i | 0.161 | 0.143 | 0.147 |

At length 40, damage is early-heavy for all three interventions. At length 1024,
only b develops the predicted late-position concentration; late b damage also exceeds
its length-40 late damage (1.093 vs 0.929). c remains early-heavy and is less damaging
at every normalized region than at length 40. i is small and nearly flat. H-C2 is thus
evidence about the query-bias pathway specifically, not a universal consequence of
sink removal.

### Numerical and aggregation cautions

All No-PE and No-MLP losses are finite; their large costs are real outputs rather than
overflow artifacts. However, the stored `sink_reduction_per_nat` column averages
per-stratum ratios. It is unstable when ΔCE is near zero and can have a sign that does
not match the ratio of aggregate means. Use the two frontier axes directly, or report
the ratio of means with uncertainty, rather than ranking near-zero-cost controls by
the current ratio column.

## E5.4 — Content generality

### Baseline varies with content

| Domain | Baseline BOS attention |
|---|---:|
| SST-2 | 0.5996 |
| GSM8K | 0.5478 |
| HumanEval | 0.5402 |
| Random uniform | 0.6020 |
| Empirical-unigram random | 0.5225 |
| Shuffled natural | 0.5084 |
| Repeat token | 0.5325 |

Shuffling natural tokens lowers the baseline sink by about 0.054 absolute, whereas
uniform vocabulary sampling raises it by about 0.039 relative to the paired natural
reference. The positional sink clearly survives without syntax, but its strength is
not independent of token distribution or order.

### Intervention signatures

Percent of each domain's own baseline:

| Domain | b | c | d | g | h | i | j |
|---|---:|---:|---:|---:|---:|---:|---:|
| Natural pooled | 44.4% | 3.0% | 6.2% | 50.4% | 18.2% | 65.0% | 103.5% |
| Random uniform | 55.1% | 2.1% | 5.7% | 72.8% | 20.6% | 68.0% | 102.8% |
| Empirical unigram | 45.7% | 2.6% | 8.5% | 55.3% | 26.7% | 63.2% | 104.5% |
| Shuffled natural | 46.0% | 2.9% | 8.3% | 57.4% | 28.4% | 63.6% | 104.3% |
| Repeat token | 57.4% | 4.3% | 9.9% | 57.7% | 6.4% | 72.8% | 102.1% |

Remove-First-PE remains decisive in every domain (2.1–4.3% of baseline), and Swap-EPE
remains strong (5.7–9.9%). This supports the shared positional key object from E4.
However, the residual after b rises by +10.7 percentage points on uniform random data
and +13.0 points on repeated tokens. The query-driven pathway B is therefore
content-sensitive even though the shared EPE key remains necessary. This is a sharper
version of the two-pathway account:

> The sink's location is positional, but the balance between bias-driven and
> content-query-driven routes depends on the token regime.

The coarse interventions drift more, as expected: No-MLP shifts +22.3 points on
uniform random input, and No-PE ranges from 6.4% to 28.4% of baseline. But surgical
invariance is not perfect: i rises +7.7 points on repeated tokens, and the nominal
Swap-PE control e falls from 98.7% to 81.3% there. Degenerate repeated content therefore
changes even some controls, not just coarse ablations.

The paired-test p-values are often extremely small because each synthetic domain has
900 paired observations across seeds. They should not substitute for effect sizes:
the current tests do not model seed clustering, and HumanEval examples overlap across
seed samples. The percentage-point shifts above are the more interpretable evidence.

Multilingual generality remains untested in this cache.

## E5.5 — Bootstrap calibration and power

### Bootstrap does not generally equal empirical reseeding

Average bootstrap-SE / empirical cross-seed-SD ratios range from 0.94 to 3.63:

- c: 1.01; d: 1.27; h: 0.94;
- b/e: about 1.76–1.78;
- a/f/i/j: about 2.8–3.2;
- g: 3.63.

Thus the preregistered claim that a single-seed bootstrap can generally replace
reseeding is not supported. Bootstrap is often substantially wider here. With only
three seeds, empirical SD itself is imprecise; finite-population overlap, especially
60/100 shared HumanEval anchors between seed pairs, also suppresses reseed variation
relative to a naive with-replacement bootstrap. More independent seeds or a
domain-stratified finite-population bootstrap are needed for a firm calibration.

### Power

At n=300, conservative paired MDEs are:

- b: 0.00419 BOS attention;
- c: 0.00635;
- d: 0.00576;
- g: 0.01631;
- h: 0.02215;
- i: 0.00218;
- f: 0.00058.

All large causal effects are overwhelmingly resolvable. Small control differences are
more delicate. The observed e–f gap is 0.00536, below its directly computed paired
MDE of 0.00656 at n=300; it is not reliably resolvable under this conservative model.
The fixed-random j effect, by contrast, is +0.01952 versus an MDE of 0.00044 and is far
too large to treat as sampling noise.

## Methodological caveats and recommended follow-ups

1. **Run and archive explicit smoke/parity reports.** The result cache itself is
   faithful for a–i, but the formal reports are absent.
2. **Repeat the random-Wk control over many fixed coordinate seeds.** One fixed draw
   increased sink strength by 3.47%; a single draw cannot define a random-ablation null.
3. **Densify PE interpolation over alpha 0.25–0.5.** The current frontier contains a
   sharp transition that deserves localization.
4. **Refine the competition-law model.** Condition on layer/head, competitor score
   distributions, or effective competitor count; show curve residuals rather than a
   single pooled slope.
5. **Replicate 1024-token domain effects on naturally long documents.** Within-domain
   concatenation is transparent and paired, but document boundaries and incoherence
   may influence long-context query behavior and CE profiles.
6. **Use hierarchical inference.** Treat seed and source domain as levels, and account
   for finite-population overlap in HumanEval.
7. **Run the optional FLORES condition.** The current content claim covers synthetic
   perturbations only.
8. **Improve paper figures before publication.** E5-A's shared linear color scale is
   dominated by the >1000%-of-baseline BOS-rank values; use oriented ranks or per-column
   scaling. E5-D labels overlap heavily; label only frontier points or use a legend.
   E5-C should show slope uncertainty and fit diagnostics because several intervention
   curves are non-linear.
9. **Do not rank near-zero-cost points by mean per-stratum reduction/nat.** Use the
   Pareto relation or a ratio-of-means with uncertainty.

## Suggested paper framing

The most defensible headline is not simply that sinks can be removed. It is:

> GPT-2's first-position sink is a robust positional anchoring mechanism whose causal
> signature survives metric, length, and severe content perturbations, but whose
> quantitative strength and functional role depend on how the anchor is disrupted.
> Relocating the anchor incurs substantially lower measured language-model cross-entropy cost than deleting it; removing the
> query-bias pathway exposes content-sensitive residual routing; and the fixed-logit
> circuit does not obey a simple inverse-competition scaling law.

This framing connects all E5 results back to E4. The shared EPE-derived key determines
where the sink forms, while bias and content-query pathways determine how strongly it
is expressed. E5 adds the functional conclusion: the model appears to need a
concentrated anchor more than it needs that anchor to remain specifically at BOS.

## Local result files used

The primary evidence is in:

- `results/e5_full/aggregate/metrics_summary.csv`;
- `results/e5_full/aggregate/metric_concordance.csv`;
- `results/e5_full/aggregate/intensive_extensive_summary.csv`;
- `results/e5_full/aggregate/length_summary.csv`;
- `results/e5_full/aggregate/logit_competition_slopes_summary.csv`;
- `results/e5_full/aggregate/length_invariance_summary.csv`;
- `results/e5_full/aggregate/mitigation_frontier.csv`;
- `results/e5_full/aggregate/cost_paired_effects.csv`;
- `results/e5_full/aggregate/position_ce_damage_region_summary.csv`;
- `results/e5_full/aggregate/content_summary.csv`;
- `results/e5_full/aggregate/content_paired_tests.csv`;
- `results/e5_full/aggregate/bootstrap_reseed_calibration.csv`;
- `results/e5_full/aggregate/mde_power.csv`.

The `results/` directory is intentionally ignored by Git because the local cache is
approximately 1.67 GB. This `summary.md` contains the portable scientific conclusions;
the raw results should be archived separately with checksums for publication.
