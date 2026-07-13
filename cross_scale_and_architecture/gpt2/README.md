# GPT-2 harness

There is no GPT-2-specific harness. The GPT-2 Table 1 experiment is the shared
master module
[`../../common/intervention_analysis.py`](../../common/intervention_analysis.py);
the other architectures wrap that module in
`intervention_analysis_{opt,neo,qwen}.py`.

`run_table1_multiseed.py --architecture gpt2` drives
`../common/intervention_analysis.py` directly, and the same module is used for the
paper-reproduction figures (see the repository [`README.md`](../../README.md)). The
module was deduplicated out of this folder so a single master copy is maintained
under [`common/`](../../common/).
