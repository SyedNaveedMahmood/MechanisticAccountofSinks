# E4 — Anatomy of the residual sink

The analysis harness for this experiment is the shared master module
[`../common/residual_sink_analysis.py`](../common/residual_sink_analysis.py). It
was previously duplicated in this folder; the single master copy now lives under
[`common/`](../common/) so every experiment references the same code.

Run E4 from this folder so results are written under `residual_sink/results/`
(the `../common/` prefix points Python at the master copy):

```bash
python ../common/residual_sink_analysis.py --mode all --model-name gpt2 --seeds 0,1,2 --output-dir results/e4_residual_sink_gpt2
```

Add `--with-perplexity` for the optional functional-cost table, and `--plot-only`
to regenerate figures from cached per-seed outputs. See
[`../.md/E4_Plan.md`](../.md/E4_Plan.md) for the full account.
