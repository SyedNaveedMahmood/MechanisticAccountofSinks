# -*- coding: utf-8 -*-
"""emergence_dynamics_analysis.py — E3: Circuit emergence during pre-training.

The target paper (Ran-Milo, Ofek & Mendel, arXiv:2604.14722) gives a *static* account of the
GPT-2 attention-sink circuit on a single converged checkpoint. Its Limitation §5.2 explicitly
leaves "how and when this circuit emerges during pre-training" to future work. E3 fills that
gap with **zero training compute** by reusing Stanford CRFM's public "Mistral" GPT-2 small runs
(5 independent pre-training seeds, hundreds of intermediate checkpoints each on the HF Hub,
accessed via ``revision="checkpoint-<step>"``). Mistral is *standard GPT-2 architecture*
(learned absolute PEs + query bias), so every validated primitive in this repo applies unchanged.

For each (run, checkpoint step) we track six emergence signals — two are pure functions of the
weights (nearly free), the rest need the paper's fixed 300-example forward passes:

    S(t)  sink strength          baseline BOS-attention (second-half -> position 0), layers 4-11
    M(t)  massive activation     max_d |EPE_1[d]|, mean-of-top-3, #coords>3sigma, and coord indices
    A(t)  bias  alignment  (A)   mean cos(b_Q,  W_k EPE_1)     — the paper's Fig. 2, pathway A
    B(t)  query alignment  (B)   mean cos(x_i W_q, W_k EPE_1)  — the E4 residual pathway B
    D(t)  Delta_1 dominance      softmax attention to pos 0 under the source-agnostic shift (T3) alone
    Eff_X causal efficacy        fraction of sink removed by {Nullify b_Q, Remove-First-PE, Zero-Top3-W_k}

where EPE_1 = MLP^(1)(p_1) + p_1 and the pre-softmax score decomposes (paper §3.1) as
    s_{i->j} = x_i W_q W_k^T x_j^T (T1) + x_i W_q b_K^T (T2) + b_Q W_k^T x_j^T (T3=Delta_j) + b_Q b_K^T (T4).

Five analyses / deliverables (see .md/E3.md):
  trajectories  E3.1  S/M/A/D vs step (Fig E3-A) + converged-vs-paper table (Table E3-2)
  ordering      E3.2  per-signal emergence step + does assembly precede the sink (Fig E3-B, Table E3-1)
  universality  E3.3  are the massive-coordinate *indices* consistent across seeds (Fig E3-C)
  efficacy      E3.4  when the paper's interventions start to bite, vs alignment onset (Fig E3-D)
  two_pathway   E3.5  pathway-A (bias) vs pathway-B (query) onset order (Fig E3-E), ties to E4

Design principle: *purely additive*. This module imports the validated measurement primitives
(``run_config``, ``collect_decomposition_and_alignment``, ``identify_massive_coords``,
``iter_examples``, ``bos_metric`` from residual_sink_analysis; ``_compute_perhead_epe_cosine``
from experiments_single_input; ``compute_band`` from intervention_analysis). The only net-new
capability is loading a checkpoint at an HF ``revision`` — no other harness threads it.

Axes note (differs from E4): here the **5 pre-training runs are the error-bar axis** (genuinely
independent optimizations); the **checkpoint step is the trajectory axis**; the **data sample is
fixed** (one draw, ``--data-seed 0``) across all runs and checkpoints so trajectories are
comparable and the cross-run band reflects pre-training variability, not sampling noise.
"""

import argparse
import gc
import itertools
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # quiet the Windows no-symlink notice

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Shared analysis modules live in the repository's top-level ``common/`` folder
# (one master copy each). Make them importable regardless of the launch directory.
import sys

_COMMON_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "common")
)
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from datasets_loader import (
    sample_benchmark_datasets,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_CUT_LENGTH,
)
from intervention_analysis import (
    get_initial_embeddings,
    compute_band,
    LAYER_RANGE_START,
    LAYER_RANGE_END,
)
from residual_sink_analysis import (
    identify_massive_coords,
    compute_ppes,
    run_config,
    bos_metric,
    iter_examples,
    count_examples,
    collect_decomposition_and_alignment,
    _pe_remove_first,
    SINK_POS,
)
from experiments_single_input import _compute_perhead_epe_cosine

# ═══════════════════════════════════════════════════════════════════════════════
# Registry & constants  (VERIFY the run ids on a networked box — see .md/E3.md §Caveats)
# ═══════════════════════════════════════════════════════════════════════════════

# Stanford CRFM "Mistral" GPT-2 pre-training runs (5 independent seeds per size). The x<NN>
# suffix is the run's random seed. These ids are from published documentation; confirm they
# resolve with ``--list-checkpoints`` before a full run (HF was network-blocked in dev).
MISTRAL_RUNS = {
    "gpt2-small": [
        "stanford-crfm/alias-gpt2-small-x21",
        "stanford-crfm/battlestar-gpt2-small-x49",
        "stanford-crfm/caprica-gpt2-small-x81",
        "stanford-crfm/darkmatter-gpt2-small-x343",
        "stanford-crfm/expanse-gpt2-small-x777",
    ],
    # medium is out of the confirmed E3 scope but wired for the optional stretch / fallback.
    "gpt2-medium": [
        "stanford-crfm/arwen-gpt2-medium-x21",
        "stanford-crfm/beren-gpt2-medium-x49",
        "stanford-crfm/celebrimbor-gpt2-medium-x81",
        "stanford-crfm/durin-gpt2-medium-x343",
        "stanford-crfm/eowyn-gpt2-medium-x777",
    ],
}

# Log-spaced *target* steps; each is snapped to the nearest available revision at runtime.
# Dense early because attention sinks are known to form in the first few thousand steps.
DEFAULT_TARGET_STEPS = [0, 100, 200, 400, 700, 1000, 2000, 4000, 7000,
                        10000, 20000, 40000, 70000, 100000, 200000, 300000, 400000]

# Paper's converged OpenAI-GPT-2 (124M) reference, for Table E3-2. NOTE: the massive-coordinate
# indices are OpenAI-GPT-2-specific; each Mistral seed will differ — that is precisely the E3.3
# finding, so these indices are a reference point, not an expectation.
PAPER_REFERENCE = {"sink_strength": 0.563, "massive_coords": [138, 378, 447]}

DEFAULT_DATA_SEED = 0
REVISION_PREFIX = "checkpoint-"
ALL_MODES = ["trajectories", "universality", "efficacy", "two_pathway"]
DATA_MODES = {"trajectories", "efficacy", "two_pathway"}   # modes that need forward passes

# Signals plotted as the four Fig E3-A panels, and their human labels / whether "up = more circuit".
TRAJECTORY_PANELS = [
    ("sink_strength",    "S(t): sink strength\n(BOS attention, layers 4-11)"),
    ("epe1_max_abs",     r"M(t): max$_d\,|$EPE$_1[d]|$" + "\n(massive activation)"),
    ("align_bias_pos0",  r"A(t): cos($b_Q$, $W_k$EPE$_1$)" + "\n(pathway A / Fig. 2)"),
    ("delta_dominance",  r"D(t): $\Delta_1$ dominance" + "\n(T3-only attn to pos 0)"),
]
ONSET_SIGNALS = ["sink_strength", "epe1_max_abs", "align_bias_pos0",
                 "delta_dominance", "query_align_pos0"]


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint loading (the one net-new capability) + Hub discovery
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(model_name, device, dtype=torch.float32, revision=None, cache_dir=None,
                    load_tokenizer=True):
    """Load a GPT-2 checkpoint at an optional HF ``revision`` (e.g. 'checkpoint-1000').

    No other harness in the repo threads ``revision``; that is the whole reason E3 needs a
    bespoke loader. ``attn_implementation="eager"`` is mandatory so attention weights are
    extractable.

    Set ``load_tokenizer=False`` to skip the tokenizer download: every Mistral run shares the
    base gpt2 BPE, so the sweep loads one gpt2 tokenizer up front and reuses it across all
    checkpoints. That avoids ~4 tokenizer-file downloads per checkpoint — anonymous, rate-limited
    requests that were the step observed hanging mid-run.
    """
    load_kw = dict(revision=revision, cache_dir=cache_dir, attn_implementation="eager")
    # Load the .bin weights directly (use_safetensors=False) FIRST. On a .bin-only checkpoint,
    # *asking* for safetensors is what makes transformers spawn its background "auto_conversion"
    # thread — it tries to open a .bin->safetensors conversion PR on the Hub and dies with an
    # alarming (but harmless, non-fatal) OSError traceback. Explicit False never asks, so the
    # thread is not spawned; we only fall back to safetensors for the rare safetensors-only
    # revision, where they already exist and no conversion is attempted.
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name, use_safetensors=False, **load_kw)
    except Exception:
        model = GPT2LMHeadModel.from_pretrained(model_name, use_safetensors=True, **load_kw)
    model.to(device)
    model.eval()
    model.to(dtype)
    tokenizer = None
    if load_tokenizer:
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
        except Exception:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer


def discover_checkpoints(repo_id):
    """Return a sorted list of (step:int, revision:str) available on the Hub for ``repo_id``.

    Robust to whatever schedule Mistral actually published — filters branches/tags matching
    ``checkpoint-<int>``. Requires network (run on a networked box, not the dev container).
    """
    from huggingface_hub import list_repo_refs
    refs = list_repo_refs(repo_id)
    pat = re.compile(re.escape(REVISION_PREFIX) + r"(\d+)$")
    step_to_rev = {}
    candidates = list(getattr(refs, "branches", []) or []) + list(getattr(refs, "tags", []) or [])
    for ref in candidates:
        m = pat.fullmatch(ref.name)
        if m:
            step_to_rev[int(m.group(1))] = ref.name
    return sorted(step_to_rev.items())


def select_checkpoints(available, targets, n=None):
    """Snap each target step to the nearest available step, de-dup, optionally thin to ``n``.

    ``available`` is a list of (step, revision). Returns a list of (step, revision) sorted by step.
    """
    if not available:
        return []
    steps = np.array([s for s, _ in available])
    rev_of = dict(available)
    if targets:
        chosen_steps = sorted({int(steps[np.abs(steps - t).argmin()]) for t in targets})
    else:
        chosen_steps = sorted(steps.tolist())
    if n is not None and len(chosen_steps) > n:
        idx = np.unique(np.round(np.linspace(0, len(chosen_steps) - 1, n)).astype(int))
        chosen_steps = [chosen_steps[i] for i in idx]
    return [(s, rev_of[s]) for s in chosen_steps]


def purge_revision_cache(repo_id, revision):
    """Best-effort, quiet deletion of a (repo, revision)'s cached files to bound peak disk.

    Deliberately avoids ``huggingface_hub.delete_revisions().execute()`` which spams per-blob
    tracebacks on Windows (where the cache uses file copies instead of symlinks). Instead we
    remove the revision's blobs, snapshot dir, and ref file directly with ``ignore_errors`` /
    try-except, so cache cleanup never crashes the run or floods the log.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
    except Exception:
        return
    for repo in cache.repos:
        if repo.repo_id != repo_id:
            continue
        repo_path = Path(repo.repo_path)
        for rev in repo.revisions:
            if revision not in set(rev.refs):
                continue
            for f in rev.files:                      # real bytes (blobs; also the copies on Windows)
                try:
                    Path(f.blob_path).unlink()
                except Exception:
                    pass
            shutil.rmtree(repo_path / "snapshots" / rev.commit_hash, ignore_errors=True)
            try:
                (repo_path / "refs" / revision).unlink()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# Per-checkpoint measurements (reuse validated primitives)
# ═══════════════════════════════════════════════════════════════════════════════

def measure_weight_signals(model, device, band, num_positions, n_std=3.0):
    """Pure-weight signals — M(t) massive activation and A(t) bias–EPE_1 alignment. No data."""
    ls, le = band
    with torch.no_grad():
        pe0 = model.transformer.wpe(torch.tensor([[0]], device=device))
        epe0 = pe0[0, 0] + model.transformer.h[0].mlp(pe0)[0, 0]
    a = np.abs(epe0.detach().cpu().float().numpy())
    coords = identify_massive_coords(model, device, n_std=n_std)  # one consistent coord definition
    n_over = int((a > (a.mean() + n_std * a.std())).sum())
    top3_idx = np.argsort(a)[-3:][::-1]           # fixed-k indices (analogue of the paper's 138/378/447)
    top3 = a[top3_idx]

    cos = _compute_perhead_epe_cosine(model, num_positions)  # [L, H, P] = cos(b_Q, W_k EPE_j)
    le_ = min(le, cos.shape[0])
    sel = cos[ls:le_, :, :]
    align_pos0 = float(sel[:, :, 0].mean())
    align_rest = float(sel[:, :, 1:].mean()) if sel.shape[2] > 1 else 0.0
    return {
        "epe1_max_abs": float(a.max()),
        "epe1_top3_mean": float(top3.mean()),
        "epe1_n_coords_3sigma": n_over,
        "align_bias_pos0": align_pos0,
        "align_bias_contrast": align_pos0 - align_rest,
        "massive_coords": [int(c) for c in coords],          # full >3sigma set (variable length)
        "epe1_top3_coords": [int(i) for i in top3_idx],      # fixed-k=3 set for cross-seed comparison
    }


def measure_data_signals(model, tokenizer, sampled, massive_coords, band):
    """Data-dependent signals in one pass: S(t), D(t), B(t), pathway attentions, and efficacy."""
    num_layers = len(model.transformer.h)
    ls, le = band
    acc = {k: [] for k in (
        "S_base", "eff_bq_raw", "eff_pe_raw", "eff_wk_raw",
        "attn_delta", "attn_content", "align_red", "align_blue", "share_delta")}
    first = True
    for _ds, _ids, te, pe in tqdm(iter_examples(model, tokenizer, sampled),
                                  total=count_examples(sampled), desc="  data signals", leave=False):
        acc["S_base"].append(bos_metric(run_config(model, te, pe), num_layers, SINK_POS, band=band))
        acc["eff_bq_raw"].append(
            bos_metric(run_config(model, te, pe, nullify_bq=True), num_layers, SINK_POS, band=band))
        acc["eff_pe_raw"].append(
            bos_metric(run_config(model, te, pe, pe_transform=_pe_remove_first), num_layers, SINK_POS, band=band))
        acc["eff_wk_raw"].append(
            bos_metric(run_config(model, te, pe, wk_zero_coords=massive_coords), num_layers, SINK_POS, band=band))
        stats = collect_decomposition_and_alignment(
            model, te, pe, layer_start=ls, layer_end=le, assert_identity=first)
        first = False
        for k in ("attn_delta", "attn_content", "align_red", "align_blue", "share_delta"):
            acc[k].append(float(np.mean(stats[k])))

    m = {k: float(np.mean(v)) for k, v in acc.items()}
    base = m["S_base"] if abs(m["S_base"]) > 1e-9 else 1e-9
    return {
        "sink_strength": m["S_base"],
        "delta_dominance": m["attn_delta"],           # D(t): T3-only softmax attention to pos 0
        "content_pathway_attn": m["attn_content"],    # pathway-B behavioral (T1-only)
        "query_align_pos0": m["align_red"],           # B(t): cos(content query, W_k EPE_1)
        "query_align_control": m["align_blue"],
        "delta_score_share": m["share_delta"],
        "efficacy_nullify_bq": (base - m["eff_bq_raw"]) / base,
        "efficacy_remove_first_pe": (base - m["eff_pe_raw"]) / base,
        "efficacy_zero_top3_wk": (base - m["eff_wk_raw"]) / base,
    }


def measure_checkpoint(model, tokenizer, sampled, device, band, modes, num_positions):
    """Combine the requested signals for one loaded checkpoint into a single flat row dict."""
    row = measure_weight_signals(model, device, band, num_positions)
    if DATA_MODES & set(modes):
        row.update(measure_data_signals(model, tokenizer, sampled, row["massive_coords"], band))
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# Directory helpers  (results/<experiment>/run_<name>/step_XXXXXXXX/metrics.json)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_short(run_id):
    base = run_id.split("/")[-1]
    return base.split("-")[0] or base


def _run_dir(root, run_id):
    return root / f"run_{_run_short(run_id)}"


def _ckpt_dir(root, run_id, step):
    return _run_dir(root, run_id) / f"step_{int(step):08d}"


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation: build the long table from cached per-checkpoint metrics.json
# ═══════════════════════════════════════════════════════════════════════════════

def load_long_table(root):
    """Read every run_*/step_*/metrics.json under ``root`` into one long DataFrame."""
    rows = []
    for mfile in sorted(root.glob("run_*/step_*/metrics.json")):
        rec = json.loads(mfile.read_text())
        rows.append(rec)
    if not rows:
        raise FileNotFoundError(f"No metrics.json found under {root}. Run without --plot-only first.")
    df = pd.DataFrame(rows)
    return df.sort_values(["run", "step"]).reset_index(drop=True)


def _mean_band(df, signal):
    """Cross-run mean/std of ``signal`` per step; returns (steps, mean, std) as arrays."""
    d = df.dropna(subset=[signal])
    g = d.groupby("step")[signal]
    steps = np.array(sorted(d["step"].unique()))
    mean = g.mean().reindex(steps).values
    std = g.std().reindex(steps).fillna(0.0).values
    return steps, mean, std


def _logx(x):
    """Map step to a log-friendly x (step 0 -> a small positive so it shows on a log axis)."""
    x = np.asarray(x, dtype=float)
    pos = x[x > 0]
    floor = (pos.min() / 2.0) if pos.size else 1.0
    return np.where(x > 0, x, floor)


# ═══════════════════════════════════════════════════════════════════════════════
# Onset extraction + ordering statistics
# ═══════════════════════════════════════════════════════════════════════════════

def extract_onsets(df, signals=ONSET_SIGNALS, frac=0.5):
    """Per (run, signal): the first step reaching ``frac`` of the converged (last) value.

    Also reports a max-slope step (robustness variant) and the final value. Trajectories are
    min-max normalized per run so onset is scale-free.
    """
    recs = []
    for run in sorted(df["run"].unique()):
        d = df[df["run"] == run].sort_values("step")
        x = d["step"].values.astype(float)
        for sig in signals:
            if sig not in d:
                continue
            y = d[sig].values.astype(float)
            if len(y) < 2 or not np.isfinite(y).all():
                continue
            lo, hi = np.min(y), np.max(y)
            if hi - lo < 1e-9:
                onset, maxslope = np.nan, np.nan
            else:
                norm = (y - lo) / (hi - lo)
                onset = float(x[int(np.argmax(norm >= frac))])
                xl = np.log10(np.maximum(x, 1.0))
                slopes = np.diff(y) / (np.diff(xl) + 1e-9)
                maxslope = float(x[1:][int(np.nanargmax(np.abs(slopes)))])
            recs.append({"run": run, "signal": sig, "onset_step": onset,
                         "maxslope_step": maxslope, "final_value": float(y[-1])})
    return pd.DataFrame(recs)


def ordering_test(onsets, early="align_bias_pos0", late="sink_strength"):
    """Do circuit signals emerge before the behavioral sink? Paired across runs."""
    piv = onsets.pivot_table(index="run", columns="signal", values="onset_step")
    if early not in piv or late not in piv:
        return {}
    pair = piv[[early, late]].dropna()
    if len(pair) < 2:
        return {"n": int(len(pair))}
    diff = pair[late].values - pair[early].values  # >0 => early precedes sink
    n_precede = int((diff > 0).sum())
    res = {"n": int(len(pair)), "early": early, "late": late,
           "median_early_step": float(np.median(pair[early])),
           "median_late_step": float(np.median(pair[late])),
           "n_runs_early_precedes_late": n_precede}
    try:
        w = scipy_stats.wilcoxon(pair[late].values, pair[early].values)
        res["wilcoxon_stat"], res["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
    except Exception:
        res["wilcoxon_stat"], res["wilcoxon_p"] = float("nan"), float("nan")
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# Universality (massive-coordinate index consistency across seeds)
# ═══════════════════════════════════════════════════════════════════════════════

# Universality compares the fixed top-3 index set (analogue of the paper's 138/378/447);
# falls back to the full >3sigma set on older outputs that lack the top-3 field.
UNIVERSALITY_COORD_FIELD = "epe1_top3_coords"


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def _coords_of(row, field=UNIVERSALITY_COORD_FIELD):
    c = row.get(field, None)
    if c is None:
        c = row.get("massive_coords", [])
    if isinstance(c, str):
        c = json.loads(c)
    return list(c or [])


def final_coords_by_run(df):
    """Top-3 EPE_1 coordinate index set at each run's last checkpoint. Returns {run: [coords]}."""
    out = {}
    for run in sorted(df["run"].unique()):
        d = df[df["run"] == run].sort_values("step")
        out[run] = _coords_of(d.iloc[-1])
    return out


def coord_temporal_stability(df):
    """Per run: Jaccard(coords(step), coords(final)) over training — do indices lock in and stay?"""
    recs = []
    for run in sorted(df["run"].unique()):
        d = df[df["run"] == run].sort_values("step")
        final = _coords_of(d.iloc[-1])
        for _, r in d.iterrows():
            recs.append({"run": run, "step": r["step"],
                         "jaccard_to_final": _jaccard(_coords_of(r), final)})
    return pd.DataFrame(recs)


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting (Figs E3-A .. E3-E)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectories(df, save_path):
    """Fig E3-A: 4-panel S/M/A/D vs training step (log-x), 5-run mean ± band."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (sig, label) in zip(axes.ravel(), TRAJECTORY_PANELS):
        if sig not in df:
            ax.set_visible(False)
            continue
        steps, mean, std = _mean_band(df, sig)
        x = _logx(steps)
        ax.fill_between(x, mean - std, mean + std, alpha=0.20)
        ax.plot(x, mean, marker="o", linewidth=2)
        for run in sorted(df["run"].unique()):  # faint per-run traces
            d = df[df["run"] == run].sort_values("step")
            ax.plot(_logx(d["step"].values), d[sig].values, alpha=0.25, linewidth=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("E3-A  Emergence trajectories of the sink circuit (Mistral GPT-2 small, 5 seeds)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ordering(onsets, order_stat, save_path):
    """Fig E3-B: per-signal onset step (one point per run) + alignment-vs-sink scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sigs = [s for s in ONSET_SIGNALS if s in set(onsets["signal"])]
    for i, sig in enumerate(sigs):
        vals = onsets[onsets["signal"] == sig]["onset_step"].dropna().values
        ax1.scatter(_logx(vals), np.full_like(vals, i, dtype=float), alpha=0.7, s=60)
        if len(vals):
            ax1.scatter([_logx([np.median(vals)])[0]], [i], marker="|", s=800, color="black")
    ax1.set_yticks(range(len(sigs)))
    ax1.set_yticklabels(sigs, fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlabel("Emergence step (first crossing of 50% of converged value)")
    ax1.set_title("Per-signal onset across the 5 seeds (| = median)")
    ax1.grid(True, which="both", alpha=0.3)

    piv = onsets.pivot_table(index="run", columns="signal", values="onset_step")
    if "align_bias_pos0" in piv and "sink_strength" in piv and \
            len(piv[["align_bias_pos0", "sink_strength"]].dropna()) >= 1:
        pair = piv[["align_bias_pos0", "sink_strength"]].dropna()
        ax2.scatter(_logx(pair["align_bias_pos0"]), _logx(pair["sink_strength"]), s=70)
        allv = np.concatenate([_logx(pair["align_bias_pos0"]), _logx(pair["sink_strength"])])
        lo, hi = allv.min() * 0.7, allv.max() * 1.4
        ax2.plot([lo, hi], [lo, hi], "k--", alpha=0.6, label="y = x")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("Bias-alignment A(t) onset step")
        ax2.set_ylabel("Sink S(t) onset step")
        sub = ""
        if order_stat:
            sub = (f"{order_stat.get('n_runs_early_precedes_late','?')}/{order_stat.get('n','?')} "
                   f"runs: alignment precedes sink (Wilcoxon p="
                   f"{order_stat.get('wilcoxon_p', float('nan')):.3f})")
        ax2.set_title("Does the circuit assemble before the sink?\n" + sub, fontsize=11)
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_universality(coords_by_run, stability_df, save_path):
    """Fig E3-C: cross-seed Jaccard of massive-coord indices + within-seed temporal stability."""
    runs = list(coords_by_run.keys())
    n = len(runs)
    J = np.ones((n, n))
    for i, j in itertools.product(range(n), range(n)):
        J[i, j] = _jaccard(coords_by_run[runs[i]], coords_by_run[runs[j]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    im = ax1.imshow(J, vmin=0, vmax=1, cmap="viridis")
    ax1.set_xticks(range(n)); ax1.set_xticklabels([_run_short(r) for r in runs], rotation=45, ha="right")
    ax1.set_yticks(range(n)); ax1.set_yticklabels([_run_short(r) for r in runs])
    for i, j in itertools.product(range(n), range(n)):
        ax1.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center",
                 color="white" if J[i, j] < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Jaccard of massive-coord indices")
    ax1.set_title("Cross-seed overlap of EPE$_1$ massive-coordinate indices")

    if stability_df is not None and len(stability_df):
        for run in runs:
            d = stability_df[stability_df["run"] == run].sort_values("step")
            ax2.plot(_logx(d["step"].values), d["jaccard_to_final"].values,
                     marker="o", markersize=3, label=_run_short(run))
        ax2.set_xscale("log")
        ax2.set_xlabel("Training step")
        ax2.set_ylabel("Jaccard(coords(step), coords(final))")
        ax2.set_title("Within-seed temporal stability of the indices")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_efficacy(df, onsets, save_path):
    """Fig E3-D: intervention efficacy (fraction of sink removed) vs step, + alignment onset line."""
    curves = [("efficacy_nullify_bq", "Nullify $b_Q$ (pathway A)"),
              ("efficacy_remove_first_pe", "Remove-First-PE (both pathways)"),
              ("efficacy_zero_top3_wk", "Zero-Top-3 $W_k$")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for sig, label in curves:
        if sig not in df:
            continue
        steps, mean, std = _mean_band(df, sig)
        x = _logx(steps)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)
        ax.plot(x, mean, marker="o", linewidth=2, label=label)
    if onsets is not None and "align_bias_pos0" in set(onsets["signal"]):
        ovals = onsets[onsets["signal"] == "align_bias_pos0"]["onset_step"].values
        if np.isfinite(ovals).any():
            med = np.nanmedian(ovals)
            ax.axvline(_logx([med])[0], color="gray", linestyle=":", alpha=0.8,
                       label="median A(t) onset")
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Fraction of sink removed by intervention")
    ax.set_title("E3-D  Causal efficacy emerges with the circuit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_two_pathway(df, save_path):
    """Fig E3-E: pathway-A (bias) vs pathway-B (query) alignment onset — which forms first?"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for sig, label in [("align_bias_pos0", "A: cos($b_Q$, $W_k$EPE$_1$)"),
                       ("query_align_pos0", "B: cos($x_iW_q$, $W_k$EPE$_1$)")]:
        if sig not in df:
            continue
        steps, mean, std = _mean_band(df, sig)
        x = _logx(steps)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)
        ax.plot(x, mean, marker="o", linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean alignment (position 0)")
    ax.set_title("E3-E  Two-pathway emergence (ties to E4)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate driver: build tables + all figures from cached metrics.json
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_and_plot(root, modes):
    agg = root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    df = load_long_table(root)
    df.to_csv(agg / "measurements_long.csv", index=False)

    # Cross-run trajectory summary (mean ± std per step) for every numeric signal.
    numeric = [c for c in df.columns if c not in ("run", "step", "revision", "massive_coords")
               and pd.api.types.is_numeric_dtype(df[c])]
    traj = df.groupby("step")[numeric].agg(["mean", "std"])
    traj.columns = ["_".join(c) for c in traj.columns]
    traj.reset_index().to_csv(agg / "trajectories.csv", index=False)

    onsets = None
    if DATA_MODES & set(modes) or "trajectories" in modes:
        onsets = extract_onsets(df)
        onsets.to_csv(agg / "onsets.csv", index=False)

    if "trajectories" in modes:
        plot_trajectories(df, agg / "fig_E3A_trajectories.png")
        _write_converged_table(df, agg / "table_E3_2_converged_vs_paper.csv")
        if onsets is not None:
            ostat = ordering_test(onsets)
            _write_onset_table(onsets, ostat, agg / "table_E3_1_onsets.csv")
            (agg / "ordering_test.json").write_text(json.dumps(ostat, indent=2))
            plot_ordering(onsets, ostat, agg / "fig_E3B_ordering.png")

    if "universality" in modes:
        coords_by_run = final_coords_by_run(df)
        (agg / "final_massive_coords.json").write_text(json.dumps(coords_by_run, indent=2))
        stability = coord_temporal_stability(df)
        stability.to_csv(agg / "coord_temporal_stability.csv", index=False)
        plot_universality(coords_by_run, stability, agg / "fig_E3C_universality.png")

    if "efficacy" in modes and "efficacy_nullify_bq" in df:
        plot_efficacy(df, onsets, agg / "fig_E3D_efficacy.png")

    if "two_pathway" in modes and "query_align_pos0" in df:
        plot_two_pathway(df, agg / "fig_E3E_two_pathway.png")

    return agg


def _write_onset_table(onsets, ostat, path):
    rows = []
    for sig in ONSET_SIGNALS:
        v = onsets[onsets["signal"] == sig]["onset_step"].dropna().values
        if not len(v):
            continue
        rows.append({"signal": sig, "median_onset_step": float(np.median(v)),
                     "iqr_low": float(np.percentile(v, 25)), "iqr_high": float(np.percentile(v, 75)),
                     "n_runs": int(len(v))})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def _write_converged_table(df, path):
    """Table E3-2: final-checkpoint values per run + mean, next to the paper's reference."""
    rows = []
    for run in sorted(df["run"].unique()):
        last = df[df["run"] == run].sort_values("step").iloc[-1]
        rec = {"run": _run_short(run), "step": int(last["step"])}
        for sig in ("sink_strength", "epe1_max_abs", "align_bias_pos0",
                    "delta_dominance", "query_align_pos0"):
            if sig in last:
                rec[sig] = float(last[sig])
        coords = last.get("massive_coords", [])
        if isinstance(coords, str):
            coords = json.loads(coords)
        rec["massive_coords"] = coords
        rows.append(rec)
    out = pd.DataFrame(rows)
    ref = {"run": "PAPER (OpenAI gpt2)", "step": -1,
           "sink_strength": PAPER_REFERENCE["sink_strength"],
           "massive_coords": PAPER_REFERENCE["massive_coords"]}
    out = pd.concat([out, pd.DataFrame([ref])], ignore_index=True)
    out.to_csv(path, index=False)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Run orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_runs(args):
    if args.model_name:
        return [args.model_name]
    if args.runs:
        return [r.strip() for r in args.runs.split(",") if r.strip()]
    return MISTRAL_RUNS[args.run_family]


def run_experiment(args, root, modes):
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)}), dtype {args.dtype}")
    else:
        print("Device: CPU  [WARNING] CUDA not detected — this will be ~50-100x slower than a GPU. "
              "Install the CUDA build of torch (torch==2.10.0+cu128) to use the 4080 Super.")
    runs = resolve_runs(args)
    targets = ([int(s) for s in args.checkpoints.split(",")]
               if args.checkpoints not in (None, "auto") else DEFAULT_TARGET_STEPS)

    # Fixed evaluation sample (identical across every run + checkpoint). Uses the base gpt2 BPE,
    # which every Mistral run shares, so the 300 examples are literally the same tokens everywhere.
    sampled = manifest = sampling_tok = None
    if DATA_MODES & set(modes):
        sampling_tok = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
        sampled, manifest = sample_benchmark_datasets(
            sampling_tok, sample_size=args.sample_size, cut_length=args.cut_length,
            seed=args.data_seed)
        print(f"Sampled {count_examples(sampled)} fixed examples (data-seed {args.data_seed}).")

    for run_id in runs:
        print(f"\n{'='*72}\nRUN {run_id}\n{'='*72}")
        if args.model_name:                       # single-model fidelity mode: no revisions
            selected = [(0, args.revision)]
        else:
            available = discover_checkpoints(run_id)
            selected = select_checkpoints(available, targets, n=args.n_checkpoints)
            print(f"  {len(available)} checkpoints on Hub; analyzing {len(selected)}: "
                  f"{[s for s, _ in selected]}")
        for step, revision in tqdm(selected, desc=f"  {_run_short(run_id)} checkpoints"):
            out_dir = _ckpt_dir(root, run_id, step)
            if args.skip_existing and (out_dir / "metrics.json").exists():
                continue
            model, _ = load_checkpoint(run_id, device, dtype=dtype, revision=revision,
                                       load_tokenizer=False)  # reuse the one gpt2 tokenizer below
            num_layers = len(model.transformer.h)
            band = compute_band(num_layers, args.layer_mode)
            row = measure_checkpoint(model, sampling_tok, sampled, device, band, modes, args.cut_length)
            row.update({"run": _run_short(run_id), "run_id": run_id, "step": int(step),
                        "revision": revision})
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(json.dumps(row, indent=2))
            (out_dir / "run_config.json").write_text(json.dumps({
                "run_id": run_id, "revision": revision, "step": int(step),
                "dtype": args.dtype, "layer_mode": args.layer_mode,
                "band_start": band[0], "band_end": band[1], "num_layers": num_layers,
                "massive_coords": row.get("massive_coords"), "data_seed": args.data_seed,
                "sample_size": args.sample_size, "cut_length": args.cut_length,
            }, indent=2))
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.purge_cache and not args.model_name:
                purge_revision_cache(run_id, revision)
        if manifest is not None:
            pd.DataFrame(manifest).to_csv(_run_dir(root, run_id) / "sample_manifest.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Offline smoke test — random GPT-2, monkeypatched loader + example iterator (no network)
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test(output_dir="results"):
    """End-to-end check on tiny random GPT-2s. No model/dataset download or network required.

    Builds 3 runs x 2 "checkpoints" of independently-initialized 12-layer GPT-2s, patches only
    this module's example iterator (so no tokenizer/dataset is needed), runs all five analyses,
    and asserts every signal, figure, and table is produced with finite values.
    """
    import sys
    from transformers import GPT2Config

    device = torch.device("cpu")
    mod = sys.modules[__name__]

    def make_model(seed):
        torch.manual_seed(seed)
        cfg = GPT2Config(n_layer=12, n_head=12, n_embd=768, n_positions=64,
                         vocab_size=512, attn_implementation="eager")
        return GPT2LMHeadModel(cfg).to(device).eval()

    def fake_iter(model, tokenizer, sampled):
        g = torch.Generator().manual_seed(sampled["_seed"])
        for _ in range(sampled["_n"]):
            ids = torch.randint(0, model.config.vocab_size, (1, 40), generator=g).to(model.device)
            pos_enc, token_embeddings = get_initial_embeddings(model, {"input_ids": ids})
            yield "fake", ids, token_embeddings, pos_enc

    # Patch the example iterator that measure_data_signals looks up in this module's globals.
    mod.iter_examples = fake_iter
    mod.count_examples = lambda sampled: sampled["_n"]

    modes = ALL_MODES
    root = Path(output_dir) / "emergence_smoke"
    sampled = {"_n": 4, "_seed": 123}          # fixed fake sample (same tokens every checkpoint)
    runs = ["smoke/alpha-run", "smoke/beta-run", "smoke/gamma-run"]
    sid = 0
    for run_id in runs:
        for step in (0, 100):
            model = make_model(seed=sid)
            sid += 1
            band = compute_band(len(model.transformer.h), "scaled")
            row = measure_checkpoint(model, None, sampled, device, band, modes, num_positions=40)
            row.update({"run": _run_short(run_id), "run_id": run_id,
                        "step": int(step), "revision": f"checkpoint-{step}"})
            od = _ckpt_dir(root, run_id, step)
            od.mkdir(parents=True, exist_ok=True)
            (od / "metrics.json").write_text(json.dumps(row, indent=2))

    agg = aggregate_and_plot(root, modes)

    # Assertions
    long = load_long_table(root)
    need = ["sink_strength", "epe1_max_abs", "align_bias_pos0", "delta_dominance",
            "query_align_pos0", "efficacy_nullify_bq", "massive_coords"]
    for col in need:
        assert col in long.columns, f"missing signal column: {col}"
    num = long[[c for c in need if c != "massive_coords"]].to_numpy(dtype=float)
    assert np.isfinite(num).all(), "non-finite measurement encountered"
    for fig in ["fig_E3A_trajectories.png", "fig_E3B_ordering.png", "fig_E3C_universality.png",
                "fig_E3D_efficacy.png", "fig_E3E_two_pathway.png"]:
        assert (agg / fig).exists(), f"missing figure: {fig}"
    for tab in ["table_E3_1_onsets.csv", "table_E3_2_converged_vs_paper.csv",
                "onsets.csv", "trajectories.csv"]:
        assert (agg / tab).exists(), f"missing table: {tab}"
    b0, b1 = compute_band(12, "scaled")
    assert (b0, b1) == (3, 11), f"band wrong: {(b0, b1)}"
    print(f"\nSMOKE TEST PASSED. Outputs under {root}")
    print(long[[c for c in need if c != 'massive_coords']].describe().T[["mean", "min", "max"]])


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="E3: Circuit emergence during pre-training (GPT-2 attention-sink dynamics).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="all", choices=ALL_MODES + ["all"],
                   help="Which analysis to run (default: all five).")
    p.add_argument("--run-family", default="gpt2-small", choices=list(MISTRAL_RUNS.keys()),
                   help="Expand to the 5 Mistral runs for this size (default gpt2-small).")
    p.add_argument("--runs", default=None,
                   help="Comma-separated HF repo ids; overrides --run-family.")
    p.add_argument("--model-name", "--model", dest="model_name", default=None,
                   help="Single model id for the reuse-fidelity check (e.g. 'gpt2'); "
                        "analyzes one checkpoint (--revision) instead of a Mistral sweep.")
    p.add_argument("--revision", default=None, help="HF revision for --model-name (default: main).")
    p.add_argument("--checkpoints", default="auto",
                   help="'auto' (log-spaced targets snapped to available) or a comma list of steps.")
    p.add_argument("--n-checkpoints", type=int, default=18,
                   help="Max checkpoints per run under 'auto' (default 18).")
    p.add_argument("--list-checkpoints", action="store_true",
                   help="Print available checkpoint revisions for each run and exit (Day-1 check).")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--experiment-name", default=None,
                   help="Subdir under --output-dir. Default: emergence_<family>.")
    p.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED,
                   help="Seed for the FIXED evaluation sample (kept constant across all runs).")
    p.add_argument("--tokenizer-name", default="gpt2",
                   help="Tokenizer for sampling (all Mistral runs share the gpt2 BPE).")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--cut-length", type=int, default=DEFAULT_CUT_LENGTH)
    p.add_argument("--layer-mode", choices=["scaled", "fixed"], default="scaled")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--purge-cache", action="store_true",
                   help="Delete each checkpoint's HF cache after analysis (bounds peak disk).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip checkpoints whose metrics.json already exists.")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip all computation; aggregate + plot from cached metrics.json.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run the offline end-to-end smoke test on a random GPT-2 and exit.")
    return p


def main():
    args = build_parser().parse_args()
    if args.smoke_test:
        _smoke_test(args.output_dir)
        return

    modes = ALL_MODES if args.mode == "all" else [args.mode]
    exp = args.experiment_name or f"emergence_{args.run_family}"
    root = Path(args.output_dir) / exp
    root.mkdir(parents=True, exist_ok=True)

    if args.list_checkpoints:
        for run_id in resolve_runs(args):
            try:
                avail = discover_checkpoints(run_id)
                print(f"{run_id}: {len(avail)} checkpoints -> {[s for s, _ in avail]}")
            except Exception as e:
                print(f"{run_id}: ERROR {e}")
        return

    if not args.plot_only:
        run_experiment(args, root, modes)

    agg = aggregate_and_plot(root, modes)
    print(f"\nDone. Aggregated tables + figures → {agg}")


if __name__ == "__main__":
    main()
