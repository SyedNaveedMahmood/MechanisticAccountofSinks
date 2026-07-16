# -*- coding: utf-8 -*-
"""experiments_statistical.py — Statistical analyses across benchmark datasets.

All bq·k computations are per-head: data is collected with shape
[n_sentences, num_layers, num_heads, seq_len].

Modes
-----
  --mode bias-term       Per-head percentile analysis of bq_h·k_h (Δ_j) across
                         benchmark datasets.
                         Main figure: aggregate over layers 3-11 (1-indexed) and
                         all heads.
                         Appendix: per-layer ribbon plots (over all heads).

  --mode epe-validation  Validates EPE across benchmarks by computing
                         cos(EPE_i, R_i - O_i) for every position and
                         sentence, then reporting percentiles.
"""

import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import transformers

# Shared analysis modules live in the repository's top-level ``common/`` folder
# (one master copy each). Make them importable regardless of the launch directory.
import os
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
    DEFAULT_SEED,
)
from experiments_single_input import (
    get_initial_embeddings,
    collect_similarities_for_sentence,
    compute_epe_similarities,
    compute_epe_full_first_layer_similarities,
    LAYER_RANGE_START,  # 0-indexed inclusive  (paper layer 4)
    LAYER_RANGE_END,    # 0-indexed exclusive  (paper layer 11)
)
from nnsight_engine import (
    ARCH_SPECS,
    GPT2TracePlan,
    GPT2_TRACE_REGISTRY_VERSION,
    NNsightEngine,
    load_nnsight_model,
)


REPRO_FORWARD_REGISTRY_VERSION = "reproduction-forward-v1"


def _trace_bias_scores(model, nn_engine, inputs):
    """Per-head bq·k values from the actual traced pre-LN/QK projections."""
    num_layers = len(model.transformer.h)
    hidden = model.config.n_embd
    heads = model.config.n_head
    head_dim = hidden // heads
    traced = nn_engine.run_gpt2_trace(
        GPT2TracePlan("repro_bias_baseline"),
        {"input_ids": inputs["input_ids"],
         "attention_mask": inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))},
        band=(0, num_layers), attention="none", capture_qk=True,
        capture_pre_ln=False)
    rows = []
    for li, projected in enumerate(traced["qk"]):
        bias = model.transformer.h[li].attn.c_attn.bias.detach().float().cpu()
        bq, bk, _ = bias.chunk(3, dim=0)
        key_content = projected[:, hidden:2 * hidden] - bk
        values = torch.einsum(
            "hd,shd->hs", bq.view(heads, head_dim),
            key_content.view(key_content.shape[0], heads, head_dim))
        rows.append(values.numpy())
    return rows


@torch.no_grad()
def _direct_embeddings(model, input_ids):
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    return model.transformer.wpe(positions), model.transformer.wte(input_ids)


def _traced_full_first_layer_similarities(model, nn_engine, inputs, token_embeddings, pos_enc):
    baseline = nn_engine.run_gpt2_trace(
        GPT2TracePlan("epe_validation_R"), inputs, band=(0, 1), attention="none",
        capture_block_outputs=(0,))
    token_only = nn_engine.run_gpt2_trace(
        GPT2TracePlan("epe_validation_O", position_edit="zero_all"), inputs,
        band=(0, 1), attention="none", capture_block_outputs=(0,))
    with torch.no_grad():
        epe = pos_enc + model.transformer.h[0].mlp(pos_enc)
        difference = baseline["block_outputs"][0] - token_only["block_outputs"][0]
        similarities = torch.nn.functional.cosine_similarity(
            epe[0].detach().float().cpu(), difference.float().cpu(), dim=-1)
    return [float(value) for value in similarities]


def _write_engine_config(output_path, *, engine, model, model_name, revision,
                         dtype, nn_engine=None, measurement=None):
    if engine == "nnsight":
        info = nn_engine.engine_info(
            model_name=model_name, revision=revision, dtype=dtype,
            device=next(model.parameters()).device,
            band=(0, len(model.transformer.h)),
            registry_version=REPRO_FORWARD_REGISTRY_VERSION)
    elif engine == "manual":
        info = {
            "name": "manual", "nnsight_version": None,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__, "model_name": model_name,
            "model_revision": revision or "main", "dtype": dtype,
            "device": str(next(model.parameters()).device), "remote": False,
            "execution_location": "local",
            "attn_implementation": "manual_reimplementation",
            "attention_probability_source": None,
            "intervention_registry_version": REPRO_FORWARD_REGISTRY_VERSION,
        }
    else:
        info = {
            "name": "weight_space", "requested_engine": engine,
            "nnsight_version": None,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__, "model_name": model_name,
            "model_revision": revision or "main", "dtype": dtype,
            "device": str(next(model.parameters()).device), "remote": False,
            "execution_location": "local", "attn_implementation": "not_applicable",
            "attention_probability_source": "not_applicable",
            "intervention_registry_version": REPRO_FORWARD_REGISTRY_VERSION,
        }
    (output_path / "run_config.json").write_text(json.dumps({
        "engine_name": engine,
        "engine": info,
        "measurement": measurement,
        "registry_version": REPRO_FORWARD_REGISTRY_VERSION,
        "trace_registry_version": GPT2_TRACE_REGISTRY_VERSION,
    }, indent=2, sort_keys=True), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset-level bq·k analysis (per-head)
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(model, tokenizer, output_dir,
                 sample_size=DEFAULT_SAMPLE_SIZE,
                 cut_length=DEFAULT_CUT_LENGTH,
                 seed=DEFAULT_SEED, engine="manual", nn_engine=None,
                 model_name="gpt2", revision=None, dtype="float32"):
    output_path = Path(output_dir) / "bias_term_statistical"
    output_path.mkdir(parents=True, exist_ok=True)
    _write_engine_config(
        output_path, engine=engine, model=model, model_name=model_name,
        revision=revision, dtype=dtype, nn_engine=nn_engine,
        measurement="forward-dependent bq dot actual content-key projection")

    num_layers = len(model.transformer.h)
    num_heads = model.config.n_head

    sampled, manifest_rows = sample_benchmark_datasets(
        tokenizer, sample_size=sample_size,
        cut_length=cut_length, seed=seed,
    )
    pd.DataFrame(manifest_rows).to_csv(
        output_path / "sample_manifest.csv", index=False
    )

    all_sentences = []
    for ds_name, sentences in sampled.items():
        all_sentences.extend(sentences)

    n_sentences = len(all_sentences)
    print(f"\nTotal sentences: {n_sentences}  "
          f"(each {cut_length} tokens, {num_layers} layers, {num_heads} heads)\n")

    # Shape: [n_sentences, num_layers, num_heads, seq_len]
    bq_k_data = np.zeros((n_sentences, num_layers, num_heads, cut_length))

    for idx, sentence in enumerate(all_sentences):
        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"  [{idx + 1}/{n_sentences}] processing...")
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(model.device)
        if engine == "nnsight":
            bq_k = _trace_bias_scores(model, nn_engine, inputs)
        else:
            pos_enc, token_embeddings = get_initial_embeddings(model, inputs)
            with torch.no_grad():
                bq_k, _bq_ppe = collect_similarities_for_sentence(
                    model, token_embeddings, pos_enc
                )

        for layer_idx in range(num_layers):
            # bq_k[layer_idx] is np.ndarray of shape [num_heads, seq_len]
            bq_k_data[idx, layer_idx, :, :] = bq_k[layer_idx]

    # Normalize: shift each (sentence, layer, head) slice so the minimum
    # over positions is 0.  Equivalent to adding a constant to all Δ_j's,
    # which leaves softmax attention weights unchanged.
    bq_k_data -= bq_k_data.min(axis=3, keepdims=True)

    print("\nComputing percentiles...")

    percentiles = [10, 50, 90]

    # --- Main figure: dual histogram over layers 4-11 (1-indexed), all heads ---
    layer_start = LAYER_RANGE_START
    layer_end = min(LAYER_RANGE_END, num_layers)
    _plot_aggregate_histogram(
        bq_k_data, layer_start, layer_end, cut_length,
        save_path=output_path / "bq_k_aggregate_plot.png",
    )
    _plot_aggregate_histogram(
        bq_k_data, layer_start, layer_end, cut_length,
        save_path=output_path / "bq_k_aggregate_plot_truncated.png",
        x_min=0, x_max=75,
    )

    # --- Per-layer appendix figures (all heads flattened) ---
    per_layer_dir = output_path / "bq_k_per_layer"
    per_layer_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx in range(num_layers):
        _plot_single_percentile_layer(
            bq_k_data, layer_idx, cut_length, per_layer_dir, percentiles
        )

    # --- CSV + summary ---

    bq_k_df = _build_percentile_df(bq_k_data, num_layers, num_heads,
                                    cut_length, percentiles)
    bq_k_df.to_csv(output_path / "bq_k_percentiles.csv", index=False)

    summary_lines = _build_summary(bq_k_data, num_layers, num_heads,
                                    cut_length, n_sentences)
    summary_text = "\n".join(summary_lines) + "\n"
    (output_path / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)

    print(f"\nAll outputs saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting and percentile helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_aggregate_histogram(data, layer_start, layer_end, seq_len, save_path,
                               x_min=None, x_max=None):
    """Dual density histogram comparing position 1 vs all other positions,
    pooled across all relevant layers, heads, and sentences.

    data shape: [n_sentences, num_layers, num_heads, seq_len]

    Parameters
    ----------
    x_min, x_max : float or None
        If given, restrict the x-axis to [x_min, x_max].
    """
    selected = data[:, layer_start:layer_end, :, :]  # [S, L_sel, H, P]

    pos1_vals  = selected[:, :, :, 0].ravel()   # [S * L_sel * H]
    other_vals = selected[:, :, :, 1:].ravel()  # [S * L_sel * H * (P-1)]

    all_vals = np.concatenate([pos1_vals, other_vals])

    xlim_left  = x_min if x_min is not None else (all_vals.min() - 0.05)
    xlim_right = x_max if x_max is not None else (all_vals.max() + 0.05)
    bins = np.linspace(xlim_left, xlim_right, 40)

    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.hist(other_vals, bins=bins, density=True, alpha=0.6,
            color="steelblue", edgecolor="black", linewidth=0.8,
            label="Other positions")
    ax.hist(pos1_vals, bins=bins, density=True, alpha=0.7,
            color="tab:red", edgecolor="black", linewidth=0.8,
            label="Position 1")

    ax.set_xlim(xlim_left, xlim_right)
    ax.set_ylim(bottom=0)

    ax.set_xlabel(r"$\Delta_j$", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.tick_params(labelsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def _plot_single_percentile_layer(data, layer_idx, seq_len, out_dir, percentiles):
    """Save a single-layer percentile ribbon plot (flattened over all heads)."""
    # data shape: [n_sentences, num_layers, num_heads, seq_len]
    vals = data[:, layer_idx, :, :]  # [S, H, P]
    flat = vals.reshape(-1, seq_len)  # [S*H, P]

    positions = np.arange(seq_len)
    p10 = np.percentile(flat, 10, axis=0)
    p50 = np.percentile(flat, 50, axis=0)
    p90 = np.percentile(flat, 90, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(positions, p10, p90, alpha=0.25, color="steelblue",
                     label="p10–p90")
    ax.plot(positions, p50, color="steelblue", linewidth=2.5, label="median")
    ax.scatter([0], [p50[0]], color="red", zorder=5, s=60, label="position 1")

    ax.set_xlabel("Token position", fontsize=18)
    ax.set_ylabel(r"$\Delta_j$", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)

    fig.tight_layout()
    fname = out_dir / f"bq_k_layer_{layer_idx + 1:02d}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname.name}")


def _build_percentile_df(data, num_layers, num_heads, seq_len, percentiles):
    """Build a DataFrame with percentiles per (layer, position), flattened over heads."""
    rows = []
    for layer_idx in range(num_layers):
        for pos in range(seq_len):
            # Flatten over sentences and heads
            values = data[:, layer_idx, :, pos].ravel()
            row = {"layer": layer_idx + 1, "position": pos}
            for p in percentiles:
                row[f"p{p}"] = float(np.percentile(values, p))
            row["mean"] = float(np.mean(values))
            row["std"] = float(np.std(values))
            rows.append(row)
    return pd.DataFrame(rows)


def _build_summary(bq_k_data, num_layers, num_heads, seq_len, n_sentences):
    """Return text lines summarising position-0 dominance for bq·k (per-head)."""
    lines = [
        "=" * 70,
        f"  bq · k analysis (per-head)  —  {n_sentences} sentences, "
        f"{num_layers} layers, {num_heads} heads, {seq_len} positions",
        "=" * 70,
        "",
        f"{'Layer':>6}  {'pos0 p10':>10}  {'pos0 p50':>10}  "
        f"{'pos0 p90':>10}  {'other p90 max':>14}  "
        f"{'pos0_p10 > other_p90':>20}",
    ]

    pos0_dominates_count = 0
    for layer_idx in range(num_layers):
        # Flatten over sentences and heads
        vals = bq_k_data[:, layer_idx, :, :]  # [S, H, P]
        flat = vals.reshape(-1, seq_len)       # [S*H, P]

        pos0_p10 = np.percentile(flat[:, 0], 10)
        pos0_p50 = np.percentile(flat[:, 0], 50)
        pos0_p90 = np.percentile(flat[:, 0], 90)

        other_p90 = np.percentile(flat[:, 1:], 90, axis=0)
        other_p90_max = float(np.max(other_p90))

        dominates = pos0_p10 > other_p90_max
        if dominates:
            pos0_dominates_count += 1

        lines.append(
            f"{layer_idx + 1:>6}  {pos0_p10:>10.4f}  {pos0_p50:>10.4f}  "
            f"{pos0_p90:>10.4f}  {other_p90_max:>14.4f}  "
            f"{'YES' if dominates else 'no':>20}"
        )

    lines.append("")
    lines.append(
        f"Position 0's p10 exceeds all other positions' p90 in "
        f"{pos0_dominates_count}/{num_layers} layers."
    )
    lines.append("")

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# EPE validation — cos(EPE_i, R_i - O_i) across benchmark datasets
# ═══════════════════════════════════════════════════════════════════════════════

def _print_percentile_table(label, data, cut_length):
    """Print per-position p10 / p50 / p90 and the global minimum of p10."""
    p10 = np.percentile(data, 10, axis=0)
    p50 = np.percentile(data, 50, axis=0)
    p90 = np.percentile(data, 90, axis=0)

    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    print(f"  {'pos':>4s}   {'p10':>8s}   {'p50':>8s}   {'p90':>8s}")
    for pos in range(cut_length):
        print(f"  {pos:>4d}   {p10[pos]:>8.4f}   {p50[pos]:>8.4f}   {p90[pos]:>8.4f}")
    print(f"\n  Global min(p10) = {p10.min():.4f}  (at pos {int(p10.argmin())})")
    print(f"  Global min(p50) = {p50.min():.4f}  (at pos {int(p50.argmin())})")
    return p10, p50, p90


def epe_validation_analysis(model, tokenizer, output_dir,
                            sample_size=DEFAULT_SAMPLE_SIZE,
                            cut_length=DEFAULT_CUT_LENGTH,
                            seed=DEFAULT_SEED, engine="manual", nn_engine=None,
                            model_name="gpt2", revision=None, dtype="float32"):
    """Compute cos(EPE_i, R_i - O_i) across benchmark datasets and report
    percentiles per token position.

    Runs two variants:
      1. MLP-only:  R = x + MLP(x), O = e + MLP(e)
      2. Full layer: R = block_0(x),  O = block_0(e)
    """
    output_path = Path(output_dir) / "epe_validation_statistical"
    output_path.mkdir(parents=True, exist_ok=True)
    _write_engine_config(
        output_path, engine=engine, model=model, model_name=model_name,
        revision=revision, dtype=dtype, nn_engine=nn_engine,
        measurement="MLP weight utility plus actual first-block R/O forward")

    sampled, manifest_rows = sample_benchmark_datasets(
        tokenizer, sample_size=sample_size,
        cut_length=cut_length, seed=seed,
    )

    all_sentences = []
    for sentences in sampled.values():
        all_sentences.extend(sentences)

    n_sentences = len(all_sentences)
    print(f"\nTotal sentences: {n_sentences}  (each {cut_length} tokens)\n")

    data_mlp = np.zeros((n_sentences, cut_length))
    data_full = np.zeros((n_sentences, cut_length))

    for idx, sentence in enumerate(all_sentences):
        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"  [{idx + 1}/{n_sentences}] processing...")
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(model.device)
        if engine == "nnsight":
            pos_enc, token_embeddings = _direct_embeddings(model, inputs["input_ids"])
        else:
            pos_enc, token_embeddings = get_initial_embeddings(model, inputs)

        with torch.no_grad():
            sims_mlp = compute_epe_similarities(model, token_embeddings, pos_enc)
            if engine != "nnsight":
                sims_full = compute_epe_full_first_layer_similarities(
                    model, token_embeddings, pos_enc
                )
        if engine == "nnsight":
            sims_full = _traced_full_first_layer_similarities(
                model, nn_engine, inputs, token_embeddings, pos_enc)

        data_mlp[idx, :len(sims_mlp)] = sims_mlp
        data_full[idx, :len(sims_full)] = sims_full

    # ── Print both tables ────────────────────────────────────────────────
    p10_mlp, p50_mlp, p90_mlp = _print_percentile_table(
        "MLP-only: cos(EPE_i, R_i - O_i)", data_mlp, cut_length,
    )
    p10_full, p50_full, p90_full = _print_percentile_table(
        "Full first layer: cos(EPE_i, R_i - O_i)", data_full, cut_length,
    )

    # ── CSV ──────────────────────────────────────────────────────────────
    percentiles = [10, 50, 90]
    rows = []
    for pos in range(cut_length):
        row = {"position": pos}
        for tag, d in [("mlp", data_mlp), ("full", data_full)]:
            vals = d[:, pos]
            for p in percentiles:
                row[f"{tag}_p{p}"] = float(np.percentile(vals, p))
            row[f"{tag}_mean"] = float(np.mean(vals))
            row[f"{tag}_std"] = float(np.std(vals))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path / "epe_validation_percentiles.csv", index=False)

    # ── MLP-only plot (same as before) ───────────────────────────────────
    positions = np.arange(cut_length)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between(positions, p10_mlp, p90_mlp, alpha=0.25, color="steelblue",
                    label="p10–p90")
    ax.plot(positions, p50_mlp, color="steelblue", linewidth=2.5, label="median")
    ax.scatter(positions, p50_mlp, color="steelblue", s=25, zorder=5)
    ax.set_xlabel("Token position", fontsize=18)
    ax.set_ylabel("Cosine similarity", fontsize=18)
    ax.set_ylim(0.4, 1.05)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path / "epe_validation_plot.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # ── Full-layer plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between(positions, p10_full, p90_full, alpha=0.25, color="darkorange",
                    label="p10–p90")
    ax.plot(positions, p50_full, color="darkorange", linewidth=2.5, label="median")
    ax.scatter(positions, p50_full, color="darkorange", s=25, zorder=5)
    ax.set_xlabel("Token position", fontsize=18)
    ax.set_ylabel("Cosine similarity", fontsize=18)
    ax.set_ylim(0.4, 1.05)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path / "epe_validation_full_layer_plot.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # ── Summary text ─────────────────────────────────────────────────────
    lines = []
    for tag, data, p50 in [("MLP-only", data_mlp, p50_mlp),
                            ("Full first layer", data_full, p50_full)]:
        lines.append(f"EPE validation ({tag}): cos(EPE_i, R_i - O_i)")
        lines.append(f"  {n_sentences} sentences, {cut_length} positions")
        lines.append(f"  Global min:    {float(data.min()):.4f}")
        lines.append(f"  Global median: {float(np.median(data)):.4f}")
        lines.append(f"  Global max:    {float(data.max()):.4f}")
        lines.append("")
        lines.append("Per-position median (p50):")
        for pos in range(cut_length):
            lines.append(f"  pos {pos:>3d}: {p50[pos]:.4f}")
        lines.append("")

    summary = "\n".join(lines) + "\n"
    (output_path / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Outputs saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Coordinate-level γ alignment analysis (weight-only)
# ═══════════════════════════════════════════════════════════════════════════════

def coord_alignment_analysis(model, output_dir, *, requested_engine="manual",
                             model_name="gpt2", revision=None, dtype="float32"):
    """Per-head dual histogram of |γ_h^(l)[d]| at massive vs other coordinates.

    γ_h^(l)[d] = |bq_h^(l) · Wk_h^(l)[:,d]|  — the contribution of coordinate d
    to the source-agnostic query-key shift Δ, per head h and layer l.

    Massive coordinates are those of EPE_1 = p_0 + MLP(p_0) with
    |EPE_1[d]| > mean + 3·std (same rule as wk_alignment_analysis).

    Pools over layers 4–11 (1-indexed) and all heads, then plots a dual
    density histogram (massive vs other) in the same style as the bq·k figure.
    """
    output_path = Path(output_dir) / "coord_alignment_statistical"
    output_path.mkdir(parents=True, exist_ok=True)
    _write_engine_config(
        output_path, engine="weight_space", model=model, model_name=model_name,
        revision=revision, dtype=dtype,
        measurement=f"pure weight-space gamma alignment (requested engine: {requested_engine})")

    num_layers = len(model.transformer.h)
    num_heads  = model.config.n_head
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads

    # ── Identify massive coordinates of EPE_1 ────────────────────────────────
    with torch.no_grad():
        pe_first   = model.transformer.wpe(torch.tensor([[0]], device=model.device))          # [1,1,H]
        ppes_first = pe_first[0, 0] + model.transformer.h[0].mlp(pe_first)[0, 0]  # [H]

    abs_vals = ppes_first.abs()
    massive_mask = abs_vals > (abs_vals.mean() + 3 * abs_vals.std())
    massive_indices = torch.where(massive_mask)[0].tolist()
    if not massive_indices:
        _, topk = torch.topk(abs_vals, k=2)
        massive_indices = topk.tolist()

    massive_set = set(massive_indices)
    print(f"EPE_1 massive activation indices: {massive_indices}")
    print(f"  values: {[f'{ppes_first[i].item():.3f}' for i in massive_indices]}\n")

    # ── Compute per-head γ for layers LAYER_RANGE_START..LAYER_RANGE_END ─────
    layer_start = LAYER_RANGE_START
    layer_end   = min(LAYER_RANGE_END, num_layers)

    massive_vals = []
    other_vals   = []

    for layer_idx in range(layer_start, layer_end):
        layer = model.transformer.h[layer_idx]
        qkv_weight = layer.attn.c_attn.weight.t()
        qkv_bias   = layer.attn.c_attn.bias
        wq, wk, _  = qkv_weight.chunk(3, dim=0)
        bq, *_     = qkv_bias.chunk(3, dim=0)

        # Split into per-head components
        bq_heads = bq.view(num_heads, head_dim)               # [H, D_h]
        wk_heads = wk.view(num_heads, head_dim, hidden_size)  # [H, D_h, hidden]

        with torch.no_grad():
            # γ_h[d] = |bq_h · wk_h[:,d]|  →  shape [H, hidden_size]
            gamma = torch.einsum("hd,hdi->hi", bq_heads, wk_heads).abs()  # [H, hidden_size]
            gamma_np = gamma.detach().cpu().numpy()

        for h in range(num_heads):
            for d in range(hidden_size):
                if d in massive_set:
                    massive_vals.append(gamma_np[h, d])
                else:
                    other_vals.append(gamma_np[h, d])

    massive_vals = np.array(massive_vals)
    other_vals   = np.array(other_vals)

    print(f"Collected {len(massive_vals)} massive-coord values "
          f"and {len(other_vals)} other-coord values "
          f"across layers {layer_start+1}-{layer_end} (1-indexed), {num_heads} heads.\n")

    # ── Dual density histogram ────────────────────────────────────────────────
    all_vals = np.concatenate([massive_vals, other_vals])
    bins = np.linspace(all_vals.min() - 0.05, all_vals.max() + 0.05, 25)

    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.hist(other_vals,   bins=bins, density=True, alpha=0.6,
            color="steelblue", edgecolor="black", linewidth=0.5,
            label="Other coordinates")
    ax.hist(massive_vals, bins=bins, density=True, alpha=0.7,
            color="tab:red",   edgecolor="black", linewidth=0.5,
            label="Massive coordinates")
    ax.set_xlabel(r"$|\gamma_h^{(l)}[d]|$", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.tick_params(labelsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = output_path / "coord_alignment_histogram.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {plot_path.name}")

    # Truncated version — clipped at x=4
    x_max_trunc = 4.0
    bins_trunc  = np.linspace(0, x_max_trunc, 25)
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.hist(other_vals,   bins=bins_trunc, density=True, alpha=0.6,
            color="steelblue", edgecolor="black", linewidth=0.5,
            label="Other coordinates")
    ax.hist(massive_vals, bins=bins_trunc, density=True, alpha=0.7,
            color="tab:red",   edgecolor="black", linewidth=0.5,
            label="Massive coordinates")
    ax.set_xlim(0, x_max_trunc)
    ax.set_xlabel(r"$|\gamma_h^{(l)}[d]|$", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.tick_params(labelsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    trunc_path = output_path / "coord_alignment_histogram_truncated.png"
    fig.savefig(trunc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {trunc_path.name}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    lines = [
        f"Coordinate alignment: |γ_h^(l)[d]| — massive vs other",
        f"  Layers {layer_start+1}–{layer_end} (1-indexed), {num_heads} heads",
        f"  Massive indices: {massive_indices}",
        f"",
        f"  Massive coords — mean: {massive_vals.mean():.4f}  "
        f"std: {massive_vals.std():.4f}  "
        f"median: {np.median(massive_vals):.4f}",
        f"  Other coords   — mean: {other_vals.mean():.4f}  "
        f"std: {other_vals.std():.4f}  "
        f"median: {np.median(other_vals):.4f}",
    ]
    summary = "\n".join(lines) + "\n"
    (output_path / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Outputs saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Statistical analyses of per-head bq similarities across "
                    "benchmark datasets on GPT-2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["bias-term", "epe-validation", "coord-alignment"],
        required=True,
        help="'bias-term' — per-head percentile analysis of bq_h·k_h across benchmarks; "
             "'epe-validation' — validates EPE across benchmarks; "
             "'coord-alignment' — per-head |γ_h[d]| histogram: massive vs other coordinates.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Root directory for all outputs.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help="Examples to sample from each benchmark dataset.",
    )
    parser.add_argument(
        "--cut-length", type=int, default=DEFAULT_CUT_LENGTH,
        help="Token count threshold and truncation target.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for dataset sampling.",
    )
    parser.add_argument("--model-name", default="gpt2",
                        help="Hugging Face GPT-2-compatible model or local path.")
    parser.add_argument("--revision", default=None,
                        help="Optional Hugging Face model revision (default: main).")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                        default="float32")
    parser.add_argument("--engine", choices=["manual", "nnsight"], default="manual",
                        help="Forward execution engine for bias-term/epe-validation. "
                             "coord-alignment is pure weight-space under either choice.")
    args = parser.parse_args()

    print("Loading GPT-2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = None
    nn_engine = None
    if args.mode != "coord-alignment" and args.engine == "nnsight":
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, revision=args.revision)
        try:
            lm = load_nnsight_model(
                ARCH_SPECS["gpt2"], args.model_name, dtype=dtype, tokenizer=tokenizer,
                device=device, revision=args.revision)
            model = lm._model
            nn_engine = NNsightEngine(lm, ARCH_SPECS["gpt2"])
        except Exception as exc:
            raise RuntimeError(
                f"NNsight initialization failed for reproduction mode {args.mode!r}; "
                f"the manual implementation was not run: {exc}") from exc
    else:
        model = GPT2LMHeadModel.from_pretrained(
            args.model_name, revision=args.revision, attn_implementation="eager")
        model.to(device)
        model.to(dtype)
        model.eval()
        if args.mode != "coord-alignment":
            tokenizer = GPT2Tokenizer.from_pretrained(
                args.model_name, revision=args.revision)
    print(f"Model loaded on {device}.\n")

    if args.mode == "bias-term":
        run_analysis(
            model, tokenizer, args.output_dir,
            sample_size=args.sample_size,
            cut_length=args.cut_length,
            seed=args.seed,
            engine=args.engine, nn_engine=nn_engine,
            model_name=args.model_name, revision=args.revision, dtype=args.dtype,
        )
    elif args.mode == "epe-validation":
        epe_validation_analysis(
            model, tokenizer, args.output_dir,
            sample_size=args.sample_size,
            cut_length=args.cut_length,
            seed=args.seed,
            engine=args.engine, nn_engine=nn_engine,
            model_name=args.model_name, revision=args.revision, dtype=args.dtype,
        )
    elif args.mode == "coord-alignment":
        coord_alignment_analysis(
            model, args.output_dir, requested_engine=args.engine,
            model_name=args.model_name, revision=args.revision, dtype=args.dtype)


if __name__ == "__main__":
    main()
