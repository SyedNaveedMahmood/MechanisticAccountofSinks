# NNsight execution for E3, E4, E5, and remaining reproduction forwards

> **Implementation-time restriction:** no smoke test, parity test, model forward, dataset
> test, GPU test, NNsight trace, or experiment was run on the Codex machine. Only static
> source validation was performed. The commands below are for the separate Windows RTX
> machine.

## 1. Scope

- Branch: `e_rest_nnsight`
- Source branch: latest remote `E1-E2-nnsight`
- Production default: `--engine manual`
- Alternative: `--engine nnsight`, using the real Hugging Face eager-attention forward
- Main entry points:
  - `emergence_dynamics/emergence_dynamics_analysis.py` (E3)
  - `common/residual_sink_analysis.py` (E4)
  - `evaluation_robustness/evaluation_robustness_analysis.py` (E5)
  - `reproduce_paper/experiments_statistical.py` (remaining forward-dependent reproduction)

The E1/E2 Table-1 engine remains in `common/nnsight_engine.py`. The implementation extends
that module rather than copying tracing code into each experiment.

## 2. Shared architecture

`GPT2TracePlan` is the small execution vocabulary used by the shared traced-forward method
`NNsightEngine.run_gpt2_trace`. It describes edit loci, not scientific experiment names.
E4 maps `ResidualInterventionSpec` to it. E5 continues to use `InterventionSpec` as its sole
source of truth and adapts each spec immediately before execution.

One trace can selectively return:

- full selected-layer attention maps (parity only);
- compact sufficient statistics for the complete E5 metric battery;
- per-head attention to requested target positions (E3/E4);
- selected-layer pre-LN states and actual fused Q/K projections;
- selected block outputs; and
- actual LM-head logits or compact next-token losses derived from them in-trace.

All saved values are detached and moved to CPU after the trace. A requested NNsight failure
raises a contextual error; no caller catches it to run the manual implementation.

## 3. Implementation map

| Experiment measurement | Classification | NNsight behavior |
|---|---|---|
| EPE construction and massive-coordinate selection | Pure weight-space | Shared direct module/weight utility; checkpoint-specific in E3 |
| Bias-to-EPE-key alignment | Pure weight-space | Shared weight utility; no forward is scientifically needed |
| Coordinate γ alignment | Pure weight-space | Unchanged weight calculation |
| Baseline/alternative sink metrics | Forward-derived | Actual eager HF attention probabilities |
| Query-to-EPE-key alignment | Activation-dependent | Actual traced Q projection minus the query bias |
| T1/T2/T3/T4 decomposition | Activation-dependent plus mathematical decomposition | Actual pre-LN and Q/K captures; component algebra remains shared |
| Delta-1 and position-zero key diagnostics | Activation-dependent | Actual traced key projection with key bias removed |
| Cross-entropy and position CE | Forward-derived | Actual intervened HF logits |
| Aggregation, statistics, onset extraction, bootstrap/MDE, plots | Pure post-processing | Unchanged cached-output consumers |

## 4. Intervention mapping

| Manual operation | NNsight trace edit |
|---|---|
| Nullify/scale query bias | Subtract `(1-alpha) * bq` from the query slice of actual fused QKV output |
| Remove first PE | Replace `wpe.output[0,0]` with `wpe.output[0,1]` |
| Zero/scale first PE | Edit only `wpe.output[0,0]` |
| Interpolate PE1 to PE2 | `alpha*p1 + (1-alpha)*p2` at position zero |
| Zero all PE | Zero the positional-embedding module output |
| Nullify first token | Zero `wte.output[0,0]` |
| Zero/scale Wk coordinates | Subtract the exact selected-column contribution from the key projection |
| Swap EPE/PE | Edit layer-0 MLP output with the same normalized component-swap directions |
| First-layer MLP skip | Zero layer-0 MLP output |
| All-layer MLP removal | Zero every block MLP output |
| Attention measurement | Save eager attention output or compact selected-band summaries |
| Functional cost | Derive next-token CE from `lm_head.output` in the same intervened forward; retain full logits only when explicitly requested |

The registry versions stored in caches are `gpt2-nnsight-trace-v1`,
`e3-emergence-spec-v1`, `e4-residual-spec-v1`, and `e5-spec-v2-nnsight`.

## 5. E3 coverage

`--engine nnsight` supports run-family discovery and explicit `--model-name` workflows,
including exact `--revision` forwarding. The base GPT-2 tokenizer is reused across Mistral
checkpoints. Every checkpoint is released before the next is loaded; `--purge-cache` retains
its existing behavior.

Forward-derived E3 quantities now use:

- actual baseline BOS attention;
- Nullify Query Bias;
- Remove First PE;
- checkpoint-specific Zero Top-k Wk;
- actual Q/K-based query alignment;
- Delta-only/content-only component softmaxes; and
- per-layer/per-head decomposition assertions.

`--verify-parity` is deliberately single-checkpoint and requires `--model-name`. It reports
absolute and relative differences for every forward-derived scalar. Weight-only quantities
are not redundantly compared.

Existing `metrics.json`, aggregate CSV, onset, universality, efficacy, two-pathway, and figure
schemas are retained. `run_config.json` adds engine and checkpoint provenance.

## 6. E4 coverage

NNsight supports `dose_response`, `decomposition`, `combined`, `surgical`, `relocation`,
`perplexity`, and `all`:

- all four dose knobs and arbitrary alpha values;
- actual pre-LN/Q/K decomposition with per-head identity checks;
- all existing combined interventions;
- first/all-layer MLP and first/all-position PE variants;
- attention to both sink and relocation positions; and
- actual intervened-logit cross-entropy.

`--verify-parity` runs every E4 mode on the same sample and writes `parity_report.json`.
The report covers dose rows, decomposition arrays, combined/surgical rows, relocation values,
and CE. NNsight/HF is the reference. Every row uses the requested `--parity-atol` and
`--parity-rtol`, except `decomposition/share_delta`, whose relative tolerance is
`max(--parity-rtol, 5e-4)` while preserving the requested absolute tolerance. The report
records the actual tolerances on every row. This narrow floor reflects that `share_delta` is
a derived ratio and showed a maximum observed relative discrepancy of about `2.95e-4` between
mathematically equivalent manual and Hugging Face/NNsight execution.

Existing CSV, JSON, and NPZ outputs are unchanged. Mode-specific `run_config_<mode>.json`
files are additive and make cache validation possible.

## 7. E5 coverage

E5's `InterventionSpec` remains authoritative. `E5Executor` consumes it through either the
existing manual executor or the shared NNsight adapter.

- `metrics`: all alternative metrics, per-example rows, head cells, query curves,
  concordance, intensive/extensive analysis, bootstrap and MDE inputs.
- `length`: exact token-ID nested prefixes, compact attention summaries, actual k0/Delta1,
  query curves, and invariance diagnostics.
- `cost`: Table-1 interventions, E4 combinations, and dose points; attention and next-token
  CE from the same traced LM-head output; unchanged non-finite handling and
  frontier/profile calculations.
- `content`: exact deterministic natural/synthetic token IDs; optional multilingual remains
  opt-in.
- `all`: all four modes.

At 1024 tokens, NNsight computes the metric sufficient statistics inside each selected layer
and next-token CE at the LM head, then saves only small per-head/per-query tensors and the
`sequence-1` loss vector. It does not retain full selected-band attention maps or
sequence-by-vocabulary logits. Length invariance retains only the position-zero Q/K slice,
not every token's projection. Full maps/logits are requested only by short smoke/fidelity probes.

E5 parity covers raw selected-band maps, every alternative metric, BOS, CE, every Table-1
intervention, all E4 combinations, requested dose points, and baseline key/Delta diagnostics.

## 8. Reproduction-entry audit

| Reproduction command | Classification | Result |
|---|---|---|
| Figure 1 bias-term histogram | Forward-dependent, previously manual-only | Added `--engine nnsight`; bq dot key uses actual traced key projections |
| Figure 2 EPE-bias alignment | Pure weight-space | No NNsight trace needed; unchanged |
| Figure 3 EPE validation, MLP-only row | Pure component/weight calculation | Unchanged utility |
| Figure 3 EPE validation, full first layer | Forward-dependent, previously manual-only | Added actual block-0 capture under NNsight |
| Figure 4/8 coordinate alignment | Pure weight-space | Unchanged; records weight-space classification |
| Figure 5 intervention maps | Already covered by E1/E2 | No duplicated implementation |
| Table 1 dataset analysis | Already covered by E1/E2 | No duplicated implementation |
| Figure 7 massive EPE coordinates | Pure weight-space | No NNsight trace needed; unchanged |

`reproduce_paper/experiments_statistical.py` accepts `--engine {manual,nnsight}` with manual as
default. `common/experiments_single_input.py` remains explicitly weight-only.

## 9. Cache compatibility and provenance

E3 `--skip-existing`, E3/E4/E5 `--plot-only`, and E4/E5 mode caches validate engine plus
scientifically critical settings. A legacy config without engine metadata is rejected rather
than guessed. To plot an NNsight cache, pass `--engine nnsight`; the default manual request will
not read it.

NNsight configs record:

- engine and NNsight/Transformers/PyTorch versions;
- model name and revision;
- dtype, device, and local/remote location;
- eager attention implementation and probability source;
- selected layer band; and
- intervention/trace registry versions.

Manual configs also record an explicit manual engine block.

## 10. Offline smoke tests (Windows PowerShell)

These tests require installed dependencies but make no network request. They create local random
models/tokenizers under temporary directories and delete them afterward.

```powershell
Set-Location X:\Projects\Blackbox\MechanisticAccountofSinks
powershell -ExecutionPolicy Bypass -File .\scripts\run_nnsight_rest_smoke_tests.ps1
```

Individual commands:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python .\tests\nnsight_e3_smoke.py
python .\tests\nnsight_e4_smoke.py
python .\tests\nnsight_e5_smoke.py
```

The central runner stops at the first failure, returns nonzero, and writes separate logs under
the ignored `results/_nnsight_rest_smoke/logs` directory.

Equivalent bash commands:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tests/nnsight_e3_smoke.py
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tests/nnsight_e4_smoke.py
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tests/nnsight_e5_smoke.py
```

## 11. Real-model parity/fidelity (Windows PowerShell)

```powershell
Set-Location X:\Projects\Blackbox\MechanisticAccountofSinks
powershell -ExecutionPolicy Bypass -File .\scripts\run_nnsight_rest_parity.ps1
```

Individual commands:

```powershell
python .\emergence_dynamics\emergence_dynamics_analysis.py --verify-parity --engine nnsight --model-name gpt2 --mode all --sample-size 5 --cut-length 16 --output-dir results\_nnsight_rest_parity --experiment-name e3_gpt2_parity
python .\common\residual_sink_analysis.py --verify-parity --engine nnsight --model-name gpt2 --seeds 0 --sample-size 5 --cut-length 16 --alphas 0,1 --output-dir results\_nnsight_rest_parity --experiment-name e4_gpt2_parity
python .\evaluation_robustness\evaluation_robustness_analysis.py --verify-parity --engine nnsight --model-name gpt2 --seeds 0 --sample-size 5 --cut-length 16 --lengths 8,16 --alphas 0,1 --bootstrap-repetitions 20 --output-dir results\_nnsight_rest_parity --experiment-name e5_gpt2_parity
```

Equivalent bash commands:

```bash
python emergence_dynamics/emergence_dynamics_analysis.py --verify-parity --engine nnsight --model-name gpt2 --mode all --sample-size 5 --cut-length 16 --output-dir results/_nnsight_rest_parity --experiment-name e3_gpt2_parity
python common/residual_sink_analysis.py --verify-parity --engine nnsight --model-name gpt2 --seeds 0 --sample-size 5 --cut-length 16 --alphas 0,1 --output-dir results/_nnsight_rest_parity --experiment-name e4_gpt2_parity
python evaluation_robustness/evaluation_robustness_analysis.py --verify-parity --engine nnsight --model-name gpt2 --seeds 0 --sample-size 5 --cut-length 16 --lengths 8,16 --alphas 0,1 --bootstrap-repetitions 20 --output-dir results/_nnsight_rest_parity --experiment-name e5_gpt2_parity
```

These commands are designed but were not run during implementation.

## 12. Full NNsight runs

```powershell
# E3 checkpoint sweep
python .\emergence_dynamics\emergence_dynamics_analysis.py --mode all --engine nnsight --run-family gpt2-small --n-checkpoints 18 --purge-cache --skip-existing --output-dir results --experiment-name emergence_gpt2-small_nnsight

# E4 core plus functional cost
python .\common\residual_sink_analysis.py --mode all --engine nnsight --model-name gpt2 --seeds 0,1,2 --with-perplexity --output-dir results --experiment-name residual_sink_gpt2_nnsight

# E5 complete audit
python .\evaluation_robustness\evaluation_robustness_analysis.py --mode all --engine nnsight --model-name gpt2 --seeds 0,1,2 --output-dir results --experiment-name evaluation_robustness_gpt2_nnsight
```

Forward-dependent reproduction commands:

```powershell
python .\reproduce_paper\experiments_statistical.py --mode bias-term --engine nnsight --output-dir results
python .\reproduce_paper\experiments_statistical.py --mode epe-validation --engine nnsight --output-dir results
```

## 13. Resume and plot-only

E3 resumes at checkpoint granularity:

```powershell
python .\emergence_dynamics\emergence_dynamics_analysis.py --mode all --engine nnsight --run-family gpt2-small --skip-existing --output-dir results --experiment-name emergence_gpt2-small_nnsight
python .\emergence_dynamics\emergence_dynamics_analysis.py --mode all --engine nnsight --run-family gpt2-small --plot-only --output-dir results --experiment-name emergence_gpt2-small_nnsight
```

E4/E5 validate each requested mode's cache. E5 modes can be run separately and merged exactly as
before. E4 has no `--skip-existing`; use a separate experiment name or run/plot the intended mode.

```powershell
python .\common\residual_sink_analysis.py --mode all --engine nnsight --model-name gpt2 --seeds 0,1,2 --plot-only --output-dir results --experiment-name residual_sink_gpt2_nnsight
python .\evaluation_robustness\evaluation_robustness_analysis.py --mode all --engine nnsight --model-name gpt2 --seeds 0,1,2 --plot-only --output-dir results --experiment-name evaluation_robustness_gpt2_nnsight
```

## 14. Expected outputs

- E3: per-checkpoint `metrics.json` and `run_config.json`; aggregate measurements,
  trajectories, onsets, universality files, two tables, and E3-A through E3-E figures.
- E4: existing dose/decomposition/combined/surgical/relocation/perplexity files plus
  `run_config_<mode>.json`; unchanged aggregate tables and figures.
- E5: existing per-example/head/query/cost/content/position files, mode configs, aggregate
  statistics, and E5-A through E5-F figures.
- Parity: one `parity_report.json` per experiment.
- Smoke: temporary outputs only; central PowerShell logs are retained.

## 15. Completion audit on the RTX machine

1. Confirm all three offline smoke logs end with the ASCII `passed` message.
2. Inspect every smoke config for `engine.name = nnsight`, eager attention, versions, local device,
   band, and registry version.
3. Inspect parity reports; do not merely rely on process exit.
4. If a discrepancy is genuine, treat NNsight/HF as the reference and investigate the manual path.
5. Run one small fidelity job before a full sweep.
6. Confirm `--plot-only --engine nnsight` succeeds and `--engine manual` rejects that cache.
7. Confirm E3 checkpoint directories release and `--purge-cache` behaves as intended.
8. Check non-finite CE statuses and E5 paired example identities.

## 16. Known limitations and unresolved runtime risks

- Runtime correctness and parity have not been established on the implementation machine.
- The new generic E3/E4/E5 executor is GPT-2-compatible; E1/E2 retains the existing GPT-2,
  OPT, GPT-Neo, and Qwen architecture paths.
- NNsight trace behavior is targeted to the repository pin (`nnsight==0.7.0`) and the pinned
  Transformers/PyTorch versions.
- Stanford-CRFM run IDs and checkpoint serialization still require the networked availability
  check documented in `docs/E3.md`.
- NDIF remote execution is not exposed for E3/E4/E5 checkpoint workflows; provenance records
  local execution. E1/E2's existing `--remote` behavior is unchanged.
- Full-map parity uses short contexts by design. Production 1024-token E5 runs use compact
  summaries and therefore exercise a different capture mode with identical metric definitions.

## 17. Static-validation boundary

The implementation is suitable for static compilation, AST/name/CLI inspection, and Git
whitespace checks locally. Do not interpret those checks as evidence that a model loaded, a trace
executed, an intervention matched, or parity passed. Those claims require the commands in
sections 10 and 11 on the remote machine.
