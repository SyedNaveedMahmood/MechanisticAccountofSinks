# -*- coding: utf-8 -*-
"""intervention_analysis_opt.py — Attention intervention experiments on OPT.

OPT analog of ``intervention_analysis.py`` (which targets GPT-2).  It recreates
**Table 1** (the ten BOS-attention interventions) and the **Figure 5** analog
(per-(layer, head) attention-heatmap grids) for Meta's OPT models.

Why OPT?
--------
The paper's sink circuit relies on two ingredients that most modern LMs drop:
a learned **query bias** ``b_Q`` and **learned absolute positional embeddings**.
OPT is the natural cross-architecture test because it keeps *both* — every OPT
attention block has ``q_proj.bias`` (``enable_bias=True``) and a learned
``embed_positions`` table.  ``facebook/opt-125m`` is moreover a structural twin
of GPT-2 small (12 layers, 12 heads, hidden 768, pre-LayerNorm), so the whole
metric — mean attention to position 0 from the second-half tokens, averaged over
all heads in layers 4-11 — transfers 1:1.

GPT-2 → OPT module mapping
--------------------------
    transformer.wte / .wpe        →  model.decoder.embed_tokens / .embed_positions
    transformer.h                 →  model.decoder.layers
    layer.attn.c_attn (fused QKV)  →  self_attn.{q,k,v}_proj (separate nn.Linear)
    layer.attn.c_proj             →  self_attn.out_proj
    layer.ln_1 / layer.ln_2       →  self_attn_layer_norm / final_layer_norm
    layer.mlp (GELU FFN)          →  fc2(activation_fn(fc1(·)))  (ReLU FFN)
    config.n_head                 →  config.num_attention_heads

Because ``F.linear(x, W, b) == x @ W.T + b`` and ``nn.Linear.weight`` is stored
as ``[out, in]`` — the same layout GPT-2's fused weight has *after* the transpose
in the original code — OPT's ``q_proj.weight`` drops straight into the same math.
OPT applies the ``head_dim**-0.5`` scale to the query rather than after ``QK^T``;
the two are algebraically identical, so we keep the original's post-``QK^T`` scale.

Only **pre-LayerNorm** OPT models (``do_layer_norm_before=True``: 125m, 1.3b,
2.7b, 6.7b …) match the harness's block structure.  ``facebook/opt-350m`` uses
post-LayerNorm *and* projects the word-embedding dim, so it is not directly
comparable and is rejected at load time.

Two modes
---------
  --mode dataset   (default)   Table 1: BOS-attention statistics over SST-2, GSM8K,
                               HumanEval.  Outputs under
                               ``<output-dir>/dataset_analysis_opt/``.
  --mode sentence              Figure 5 analog: per-(layer, head) and head-averaged
                               2x5 heatmap grids for one sentence.  Outputs under
                               ``<output-dir>/sentence_analysis_opt/``.

Interventions (labelled a-j) — identical semantics to the GPT-2 version
-----------------------------------------------------------------------
  (a) Baseline           token + positional embeddings, all layers intact.
  (b) No Query Bias      bq = 0 in every layer (isolates content-only queries).
  (c) Remove First PE    position 0 receives PE[1] instead of PE[0].
  (d) Swap EPE           after the layer-0 FFN, swap the EPE component of pos 0 and 1.
  (e) Swap PE            same swap as (d) but using the raw PE vectors.
  (f) Nullify BOS Token  zero the position-0 token embedding before processing.
  (g) No MLP             skip the FFN block in every layer.
  (h) No PE              token embeddings only; no positional signal anywhere.
  (i) Zero Top-3 Wk      zero the 3 columns of Wk matching the top-3 |EPE[0]| dims.
  (j) Zero Random Wk     zero 3 randomly chosen columns of Wk (control for (i)).
"""

import argparse
import random
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM

from datasets_loader import (
    sample_benchmark_datasets,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_CUT_LENGTH,
    DEFAULT_SEED,
)
# Architecture-agnostic metric (pure attention-tensor math): reuse as-is.
# It internally averages attention to position 0 over layers 4-11 (the paper's
# "mid" range) — that layer definition lives in intervention_analysis and is
# inherited unchanged here. The massive-activation identification (intervention (i))
# and the run_config.json provenance builder are likewise shared, so all harnesses use
# an identical definition of "massive" (|EPE_0| > mean + 3*std, top-3 fallback).
from intervention_analysis import (
    compute_bos_attention_metric, compute_band,
    select_massive_coords, build_run_config,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "facebook/opt-125m"
DEFAULT_SENTENCE = "It was the best of times, it was the worst of times, it was the age of wisdom."

# ═══════════════════════════════════════════════════════════════════════════════
# Model loading (with pre-LayerNorm guard)
# ═══════════════════════════════════════════════════════════════════════════════

def load_opt(model_name, dtype=torch.float32):
    """Load an OPT model + tokenizer, asserting it matches the pre-LN circuit.

    Rejects post-LayerNorm variants (e.g. ``facebook/opt-350m``) and any model
    that projects the word-embedding dimension, since the harness assumes the
    GPT-2-style ``token_emb + pos_emb`` first-layer input and the
    ``ln → attn → residual → ln → mlp → residual`` block order.
    """
    print(f"Loading {model_name} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OPTForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.to(dtype)
    model.eval()

    cfg = model.config
    if not getattr(cfg, "do_layer_norm_before", True):
        raise ValueError(
            f"{model_name} uses post-LayerNorm (do_layer_norm_before=False). This "
            "harness reproduces the GPT-2 pre-LayerNorm sink circuit; post-LN OPT "
            "models (e.g. facebook/opt-350m) are not directly comparable. "
            "Use facebook/opt-125m, opt-1.3b, or opt-2.7b."
        )
    decoder = model.model.decoder
    if getattr(decoder, "project_in", None) is not None:
        raise ValueError(
            f"{model_name} projects the word-embedding dimension (project_in is set), "
            "so token and positional embeddings do not share the residual space this "
            "harness assumes. Use facebook/opt-125m, opt-1.3b, or opt-2.7b."
        )

    print(f"Model loaded — {cfg.num_hidden_layers} layers, "
          f"{cfg.num_attention_heads} heads, hidden {cfg.hidden_size}.\n")
    print(f"Using device: {device}\n")
    return model, tokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# Embedding retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def get_initial_embeddings_opt(model, inputs):
    """Run one forward pass to capture positional and token embeddings via hooks.

    The forward hook on ``embed_positions`` captures the *effective* positional
    embedding (OPT's internal +2 index offset is already applied in the output),
    so ``pos_enc[0][0]`` is genuinely position 0's PE.
    """
    decoder = model.model.decoder
    captured = {}

    def _pos_hook(_module, _input, output):
        captured["pos_enc"] = output.clone()
        return output

    def _tok_hook(_module, _input, output):
        captured["token_embeddings"] = output.clone()
        return output

    hooks = [
        decoder.embed_positions.register_forward_hook(_pos_hook),
        decoder.embed_tokens.register_forward_hook(_tok_hook),
    ]
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()

    return captured["pos_enc"], captured["token_embeddings"]


def _opt_ffn(layer, x):
    """OPT feed-forward block: fc2(activation_fn(fc1(x)))  (no LayerNorm inside)."""
    return layer.fc2(layer.activation_fn(layer.fc1(x)))

# ═══════════════════════════════════════════════════════════════════════════════
# Manual self-attention (OPT)
# ═══════════════════════════════════════════════════════════════════════════════

def manual_self_attention_opt(hidden_states, layer, num_heads,
                              intervene_query_bias=False,
                              fixed_wk_zero_indices=None,
                              random_wk_zero_rows=False,
                              random_wk_zero_count=3):
    """Manual OPT self-attention for a single (already LayerNorm'd) layer input.

    Mirrors ``manual_self_attention_new`` from the GPT-2 harness but uses OPT's
    separate q/k/v/out projections and returns only what the intervention loop
    consumes: ``(attention_output, attention_weights)``.

    Args:
        hidden_states: [batch, seq, hidden] — the LayerNorm'd input to attention.
        layer:         an ``OPTDecoderLayer``.
        num_heads:     number of attention heads (from ``config.num_attention_heads``).
        intervene_query_bias:  if True, nullify the query bias ``b_Q``.
        fixed_wk_zero_indices: list of Wk *columns* (input coords) to zero, or None.
        random_wk_zero_rows:   if True, zero ``random_wk_zero_count`` random Wk columns.
        random_wk_zero_count:  number of random Wk columns to zero (size-matched to (i)).
    """
    attn = layer.self_attn
    hidden_size = hidden_states.size(-1)
    head_dim = hidden_size // num_heads
    scale = head_dim ** -0.5  # 1 / sqrt(head_dim)

    # nn.Linear weights are [out, in] — the same layout GPT-2's fused weight has
    # after its .t(); F.linear(x, W, b) == x @ W.T + b, so these drop in directly.
    wq, bq = attn.q_proj.weight, attn.q_proj.bias
    wk, bk = attn.k_proj.weight, attn.k_proj.bias
    wv, bv = attn.v_proj.weight, attn.v_proj.bias

    # --- INTERVENTION (b): nullify query bias ---
    if intervene_query_bias:
        bq = torch.zeros_like(bq)

    # --- INTERVENTIONS (i)/(j): zero massive / random Wk columns ---
    wk_active = wk.clone()
    if fixed_wk_zero_indices is not None:
        wk_active[:, fixed_wk_zero_indices] = 0.0
    elif random_wk_zero_rows:
        random_indices_to_zero = [random.randint(0, hidden_size - 1)
                                  for _ in range(random_wk_zero_count)]
        wk_active[:, random_indices_to_zero] = 0.0

    # Project input to Q, K, V.
    query_proj = F.linear(hidden_states, wq, bq)
    key_proj = F.linear(hidden_states, wk_active, bk)
    value_proj = F.linear(hidden_states, wv, bv)

    def reshape_for_multihead(x):
        return x.view(x.size(0), x.size(1), num_heads, head_dim).transpose(1, 2)

    query = reshape_for_multihead(query_proj)
    key = reshape_for_multihead(key_proj)
    value = reshape_for_multihead(value_proj)

    # Scaled dot-product attention with a causal mask.
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    seq_len = hidden_states.size(1)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=hidden_states.device)
    ).view(1, 1, seq_len, seq_len)
    attention_scores = attention_scores.masked_fill(causal_mask == 0, float("-inf"))
    attention_weights = F.softmax(attention_scores, dim=-1)

    context = torch.matmul(attention_weights, value)
    context = context.transpose(1, 2).contiguous().view(
        hidden_states.size(0), seq_len, hidden_size
    )

    attention_output = F.linear(context, attn.out_proj.weight, attn.out_proj.bias)
    return attention_output, attention_weights

# ═══════════════════════════════════════════════════════════════════════════════
# Generic intervention loop (OPT)
# ═══════════════════════════════════════════════════════════════════════════════

def run_intervention_loop_opt(model, layer_input, num_heads,
                              attn_kwargs=None, modify_mlp_fn=None, skip_mlp=False):
    """Run through all OPT decoder layers, collecting per-layer attention weights.

    Block structure (pre-LayerNorm, matches the GPT-2 harness):
        normed      = self_attn_layer_norm(x)
        x           = x + attn(normed)
        normed_ar   = final_layer_norm(x)
        x           = x + FFN(normed_ar)          [unless skip_mlp]

    Returns one ``[num_heads, seq_len, seq_len]`` tensor per layer.
    """
    attn_kwargs = attn_kwargs or {}
    layers = model.model.decoder.layers
    attention_weights_per_layer = []

    for i in range(len(layers)):
        layer = layers[i]

        normalized = layer.self_attn_layer_norm(layer_input.clone())

        attention_output, attention_weights = manual_self_attention_opt(
            normalized, layer, num_heads, **attn_kwargs
        )

        # Keep all heads: shape [num_heads, seq_len, seq_len] (batch dim squeezed).
        attention_weights_per_layer.append(attention_weights.cpu().detach()[0])

        if skip_mlp:
            layer_input = layer_input + attention_output
        else:
            attention_and_residual = layer_input + attention_output
            normalized_ar = layer.final_layer_norm(attention_and_residual)
            mlp_output = _opt_ffn(layer, normalized_ar)
            if modify_mlp_fn is not None:
                mlp_output = modify_mlp_fn(i, mlp_output)
            layer_input = attention_and_residual + mlp_output

    return attention_weights_per_layer


def _epe(model, pos_enc):
    """Effective positional embedding: EPE = p + FFN^(0)(p), using layer-0's FFN.

    Returns a ``[seq, hidden]`` tensor (batch dim squeezed), matching the GPT-2
    definition ``ppes = pos_enc[0] + h[0].mlp(pos_enc)[0]``.
    """
    layer0 = model.model.decoder.layers[0]
    return pos_enc[0] + _opt_ffn(layer0, pos_enc)[0]


def identify_massive_coords_opt(model):
    """Massive-activation coordinates of OPT's EPE_0 (deterministic per model).

    Uses the same forward-hook path the interventions use (so OPT's +2 position-index
    offset is handled) on a short dummy input; EPE_0 is input-independent, so the
    returned coordinate set is a fixed property of the model. Reduces to the paper's
    top-3 when exactly three coordinates are massive (see ``select_massive_coords``).
    """
    device = next(model.parameters()).device
    dummy = {"input_ids": torch.arange(8, device=device).unsqueeze(0)}
    pos_enc, _ = get_initial_embeddings_opt(model, dummy)
    with torch.no_grad():
        epe0 = _epe(model, pos_enc)[0]
    return select_massive_coords(epe0)

# ═══════════════════════════════════════════════════════════════════════════════
# Intervention functions
#
# Each takes (model, token_embeddings, pos_enc, num_heads) and returns a list of
# per-layer attention-weight tensors [num_heads, seq_len, seq_len].
# ═══════════════════════════════════════════════════════════════════════════════

def intervention_a_baseline(model, token_embeddings, pos_enc, num_heads):
    """(a) Baseline: regular transformer with PE added as usual."""
    layer_input = token_embeddings.clone() + pos_enc.clone()
    return run_intervention_loop_opt(model, layer_input, num_heads)


def intervention_b_nullify_query_bias(model, token_embeddings, pos_enc, num_heads):
    """(b) No Query Bias: sets bq = 0 in every layer."""
    layer_input = token_embeddings.clone() + pos_enc.clone()
    return run_intervention_loop_opt(
        model, layer_input, num_heads,
        attn_kwargs={"intervene_query_bias": True},
    )


def intervention_c_remove_first_pe(model, token_embeddings, pos_enc, num_heads):
    """(c) Remove First PE: position 0 receives PE[1] instead of PE[0]."""
    pos_enc_copy = pos_enc.clone()
    pos_enc_copy[0][0] = pos_enc_copy[0][1].clone()
    layer_input = token_embeddings.clone() + pos_enc_copy
    return run_intervention_loop_opt(model, layer_input, num_heads)


def intervention_d_swap_epe_normalized(model, token_embeddings, pos_enc, num_heads):
    """(d) Swap EPE: after the layer-0 FFN, swap the EPE component of pos 0 and pos 1.

    True component swap — pos 0 loses its EPE[0] component and gains EPE[1] in its
    place (same magnitude), pos 1 gets the mirror.  After the swap pos 0 looks
    positionally like pos 1 and vice-versa.
    """
    pos_enc_copy = pos_enc.clone()
    layer_input = token_embeddings.clone() + pos_enc_copy
    ppes = _epe(model, pos_enc_copy)

    epe0_hat = ppes[0] / torch.linalg.norm(ppes[0])
    epe1_hat = ppes[1] / torch.linalg.norm(ppes[1])

    def _modify(layer_idx, mlp_output):
        if layer_idx == 0:
            alpha = torch.dot(mlp_output[0][0], epe0_hat)
            mlp_output[0][0] = mlp_output[0][0] - alpha * epe0_hat + alpha * epe1_hat
            mlp_output[0][1] = mlp_output[0][1] + alpha * epe0_hat - alpha * epe1_hat
        return mlp_output

    return run_intervention_loop_opt(model, layer_input, num_heads, modify_mlp_fn=_modify)


def intervention_e_swap_pe(model, token_embeddings, pos_enc, num_heads):
    """(e) Swap PE: same true component swap as (d) but using the raw PE vectors."""
    pos_enc_copy = pos_enc.clone()
    layer_input = token_embeddings.clone() + pos_enc_copy

    pe0_hat = pos_enc_copy[0][0] / torch.linalg.norm(pos_enc_copy[0][0])
    pe1_hat = pos_enc_copy[0][1] / torch.linalg.norm(pos_enc_copy[0][1])

    def _modify(layer_idx, mlp_output):
        if layer_idx == 0:
            alpha = torch.dot(mlp_output[0][0], pe0_hat)
            mlp_output[0][0] = mlp_output[0][0] - alpha * pe0_hat + alpha * pe1_hat
            mlp_output[0][1] = mlp_output[0][1] + alpha * pe0_hat - alpha * pe1_hat
        return mlp_output

    return run_intervention_loop_opt(model, layer_input, num_heads, modify_mlp_fn=_modify)


def intervention_f_nullify_bos_token(model, token_embeddings, pos_enc, num_heads):
    """(f) Nullify BOS Token: zero out the position-0 token embedding; PE still added."""
    te_copy = token_embeddings.clone()
    te_copy[0][0] = torch.zeros_like(te_copy[0][0])
    layer_input = te_copy + pos_enc.clone()
    return run_intervention_loop_opt(model, layer_input, num_heads)


def intervention_g_no_mlp(model, token_embeddings, pos_enc, num_heads):
    """(g) No MLP: skip the FFN block in every layer."""
    layer_input = token_embeddings.clone() + pos_enc.clone()
    return run_intervention_loop_opt(model, layer_input, num_heads, skip_mlp=True)


def intervention_h_no_pe(model, token_embeddings, pos_enc, num_heads):
    """(h) No PE: input is token embeddings only; no positional signal anywhere."""
    layer_input = token_embeddings.clone()
    return run_intervention_loop_opt(model, layer_input, num_heads)


def intervention_i_zero_top_wk(model, token_embeddings, pos_enc, num_heads, zero_indices=None):
    """(i) Zero Top-3 Wk: zero the Wk columns matching the *massive activations* of
    EPE[0] — coordinates whose |value| exceeds ``mean + 3·std`` of |EPE[0]| (paper: the 3
    coords 138/378/447 on GPT-2 small, hence "Top-3"; larger models generally have a
    different count), see ``select_massive_coords``.

    These large-activation coordinates of the effective positional embedding dominate
    the key projection; removing them tests how much the BOS-attention sink depends on
    those specific dimensions.  ``zero_indices`` (identified once per model) is supplied
    by the caller; when None it is re-derived from EPE[0] here (identical result, since
    EPE[0] is input-independent). The same columns are zeroed in every layer.
    """
    pos_enc_copy = pos_enc.clone()
    layer_input = token_embeddings.clone() + pos_enc_copy
    if zero_indices is None:
        zero_indices = select_massive_coords(_epe(model, pos_enc_copy)[0])

    return run_intervention_loop_opt(
        model, layer_input, num_heads,
        attn_kwargs={"fixed_wk_zero_indices": list(zero_indices)},
    )


def intervention_j_zero_random_wk(model, token_embeddings, pos_enc, num_heads, k=None):
    """(j) Zero Random Wk: zero ``k`` randomly chosen Wk columns — a size-matched control
    for (i), where ``k`` is the number of massive coordinates (i) zeroes (default: the
    model's massive-coordinate count)."""
    if k is None:
        k = len(select_massive_coords(_epe(model, pos_enc.clone())[0]))
    random.seed(DEFAULT_SEED)
    layer_input = token_embeddings.clone() + pos_enc.clone()
    return run_intervention_loop_opt(
        model, layer_input, num_heads,
        attn_kwargs={"random_wk_zero_rows": True, "random_wk_zero_count": k},
    )


# Ordered registry — same 10 keys/labels/descriptions as the GPT-2 version, so the
# OPT summaries line up column-for-column with the original outputs.
INTERVENTIONS = [
    ("int_a", "(a)", "Baseline",          intervention_a_baseline),
    ("int_b", "(b)", "No Query Bias",     intervention_b_nullify_query_bias),
    ("int_c", "(c)", "Remove First PE",   intervention_c_remove_first_pe),
    ("int_d", "(d)", "Swap EPE",          intervention_d_swap_epe_normalized),
    ("int_e", "(e)", "Swap PE",           intervention_e_swap_pe),
    ("int_f", "(f)", "Nullify BOS Token", intervention_f_nullify_bos_token),
    ("int_g", "(g)", "No MLP",            intervention_g_no_mlp),
    ("int_h", "(h)", "No PE",             intervention_h_no_pe),
    ("int_i", "(i)", "Zero Top-3 Wk",     intervention_i_zero_top_wk),
    ("int_j", "(j)", "Zero Random Wk",    intervention_j_zero_random_wk),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_interventions(model, token_embeddings, pos_enc, num_heads, massive_coords=None):
    """Run every registered intervention and return {key: [layer_weights]}.

    ``massive_coords`` (identified once per model) is threaded to the Wk interventions:
    (i) zeroes exactly those columns and (j) zeroes an equal number of random columns.
    """
    extra = {}
    if massive_coords is not None:
        extra["int_i"] = {"zero_indices": list(massive_coords)}
        extra["int_j"] = {"k": len(massive_coords)}
    results = {}
    for key, _label, desc, fn in INTERVENTIONS:
        print(f"  Running {key} ({desc})...")
        with torch.no_grad():
            results[key] = fn(model, token_embeddings, pos_enc, num_heads, **extra.get(key, {}))
    return results


def _write_readable_bos_summaries(output_path, scores_by_key, suffix):
    """Write human-readable CSV and TXT.

    CSV: intervention name + mean score (2 decimal places).
    TXT: same, plus a 'relative' section showing each intervention's score as a
         percentage of the baseline (int_a) score.
    """
    rows_csv = []
    means = {}
    for key, _label, desc, _fn in INTERVENTIONS:
        mean = float(np.array(scores_by_key[key]).mean())
        means[key] = mean
        rows_csv.append({"intervention": desc, "score": f"{mean:.2f}"})

    baseline = means.get("int_a", 1.0) or 1.0   # avoid div-by-zero

    lines_txt = []
    for key, _label, desc, _fn in INTERVENTIONS:
        lines_txt.append(f"{desc}\t{means[key]:.2f}")

    lines_txt.append("")
    lines_txt.append("relative (% of baseline)")
    for key, _label, desc, _fn in INTERVENTIONS:
        pct = 100.0 * means[key] / baseline
        lines_txt.append(f"{desc}\t{pct:.1f}%")

    stem = f"bos_attention_summary_{suffix}"
    pd.DataFrame(rows_csv).to_csv(output_path / f"{stem}.csv", index=False)
    (output_path / f"{stem}.txt").write_text("\n".join(lines_txt) + "\n", encoding="utf-8")


def _stats_rows(scores, ds_name):
    """Turn per-sentence scores into summary rows for the detailed CSV."""
    rows = []
    for key, _label, desc, _fn in INTERVENTIONS:
        vals = np.array(scores[key])
        mean = float(vals.mean())
        stderr = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append({
            "dataset": ds_name,
            "intervention": key,
            "description": desc,
            "n_examples": len(vals),
            "mean_bos_attention": f"{mean:.6f}",
            "stderr": f"{stderr:.6f}",
            "display": f"{mean:.4f} ± {stderr:.4f}",
        })
    return rows

# ═══════════════════════════════════════════════════════════════════════════════
# Mode 1 — Sentence analysis (per-head + head-averaged 2x5 grid figures)
# ═══════════════════════════════════════════════════════════════════════════════

def _save_attention_grid(all_results, layer_idx, matrices_fn, title, save_path):
    """Save a 2x5 grid of attention heatmaps for all 10 interventions."""
    matrices = {key: matrices_fn(key) for key, *_ in INTERVENTIONS}
    vmin = min(m.min() for m in matrices.values())
    vmax = max(m.max() for m in matrices.values())

    fig, axes = plt.subplots(2, 5, figsize=(30, 11))

    for idx, (key, label, desc, _fn) in enumerate(INTERVENTIONS):
        row, col = divmod(idx, 5)
        ax = axes[row][col]
        m = matrices[key]
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        cax = ax.matshow(m, cmap="viridis", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=18)
        ax.set_xlabel(f"{label} {desc}", fontsize=26, labelpad=14)
        ax.tick_params(axis="both", which="both", labelsize=16)
        ax.xaxis.set_ticks_position("bottom")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sentence_analysis(model, tokenizer, sentence, output_dir, num_heads):
    """Generate per-(layer, head) and per-layer-averaged 2x5 heatmap grids (Fig 5 analog)."""
    output_path = Path(output_dir) / "sentence_analysis_opt"
    output_path.mkdir(parents=True, exist_ok=True)

    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(model.device)
    pos_enc, token_embeddings = get_initial_embeddings_opt(model, inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print(f"Sentence: {sentence!r}")
    print(f"Tokens ({len(tokens)}): {tokens}")

    massive_coords = identify_massive_coords_opt(model)
    print(f"Massive-activation coordinates of EPE_0 ({len(massive_coords)}): {massive_coords}")
    all_results = run_all_interventions(model, token_embeddings, pos_enc, num_heads,
                                        massive_coords=massive_coords)

    num_layers = len(model.model.decoder.layers)
    total = num_layers * (num_heads + 1)   # +1 for the head-avg figure per layer
    saved = 0

    for layer_idx in range(num_layers):
        # Per-head figures
        for head_idx in range(num_heads):
            _save_attention_grid(
                all_results, layer_idx,
                matrices_fn=lambda key, h=head_idx: all_results[key][layer_idx][h],
                title=f"Layer {layer_idx + 1}  ·  Head {head_idx + 1}",
                save_path=output_path / f"layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}.png",
            )
            saved += 1
            print(f"  [{saved}/{total}] Saved layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}.png")

        # Head-averaged figure
        _save_attention_grid(
            all_results, layer_idx,
            matrices_fn=lambda key: all_results[key][layer_idx].mean(dim=0),
            title=f"Layer {layer_idx + 1}  ·  Avg over heads",
            save_path=output_path / f"layer_{layer_idx + 1:02d}_avg.png",
        )
        saved += 1
        print(f"  [{saved}/{total}] Saved layer_{layer_idx + 1:02d}_avg.png")

    print(f"\nAll {total} figures saved to {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Mode 2 — Dataset analysis (Table 1: three benchmark datasets, per-dataset + pooled)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sentences(model, tokenizer, sentences, ds_label, num_layers, num_heads, band=None,
                   massive_coords=None):
    """Run all interventions on *sentences*, return {key: (mid_scores, all_scores)}."""
    ls, le = band if band is not None else (None, None)
    scores_mid = {key: [] for key, *_ in INTERVENTIONS}
    scores_all = {key: [] for key, *_ in INTERVENTIONS}
    n = len(sentences)
    for i, sentence in enumerate(sentences):
        print(f"  [{ds_label} {i + 1}/{n}] {sentence[:80]}...")
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(model.device)
        pos_enc, token_embeddings = get_initial_embeddings_opt(model, inputs)
        results = run_all_interventions(model, token_embeddings, pos_enc, num_heads,
                                        massive_coords=massive_coords)
        for key, *_ in INTERVENTIONS:
            scores_mid[key].append(
                compute_bos_attention_metric(results[key], num_layers, "mid",
                                             layer_start=ls, layer_end=le)
            )
            scores_all[key].append(
                compute_bos_attention_metric(results[key], num_layers, "all")
            )
    return scores_mid, scores_all


def dataset_analysis(model, tokenizer, output_dir, num_heads,
                     sample_size=DEFAULT_SAMPLE_SIZE,
                     cut_length=DEFAULT_CUT_LENGTH,
                     seed=DEFAULT_SEED,
                     band=None,
                     model_name=DEFAULT_MODEL,
                     dtype="float32",
                     layer_mode="scaled"):
    """Compute the BOS-attention metric on three standard benchmarks (Table 1).

    Datasets: SST-2 (natural language), GSM8K (math), HumanEval (code).  Each is
    sampled to *sample_size* examples of at least *cut_length* OPT tokens, then
    truncated to exactly *cut_length* tokens.  Outputs under
    ``<output_dir>/dataset_analysis_opt/`` mirror the GPT-2 filenames; the pooled
    ``bos_attention_summary_mid_layers.{txt,csv}`` is **Table 1** (OPT). A
    ``run_config.json`` provenance record (model geometry + massive coords) is written
    alongside so every run is self-describing and auditable.
    """
    output_path = Path(output_dir) / "dataset_analysis_opt"
    output_path.mkdir(parents=True, exist_ok=True)

    num_layers = len(model.model.decoder.layers)

    massive_coords = identify_massive_coords_opt(model)
    print(f"Massive-activation coordinates of EPE_0 ({len(massive_coords)}): {massive_coords}")
    run_cfg = build_run_config(
        model, massive_coords, model_name=model_name, dtype=dtype,
        layer_mode=layer_mode, band=band, seed=seed,
        sample_size=sample_size, cut_length=cut_length,
        harness="intervention_analysis_opt.py")
    (output_path / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    sampled, manifest_rows = sample_benchmark_datasets(
        tokenizer, sample_size=sample_size,
        cut_length=cut_length, seed=seed,
    )
    pd.DataFrame(manifest_rows).to_csv(output_path / "sample_manifest.csv", index=False)
    print(f"Manifest saved ({len(manifest_rows)} examples).\n")

    all_mid = {}    # ds_name -> {key: [scores]}
    all_scope = {}  # ds_name -> {key: [scores]}

    for ds_name, sentences in sampled.items():
        print(f"\n{'═'*60}")
        print(f"  Dataset: {ds_name}  ({len(sentences)} examples)")
        print(f"{'═'*60}")
        s_mid, s_all = _run_sentences(model, tokenizer, sentences, ds_name, num_layers,
                                      num_heads, band=band, massive_coords=massive_coords)
        all_mid[ds_name] = s_mid
        all_scope[ds_name] = s_all

        ds_dir = output_path / ds_name
        ds_dir.mkdir(exist_ok=True)
        _write_readable_bos_summaries(ds_dir, s_all, "all_layers")
        _write_readable_bos_summaries(ds_dir, s_mid, "mid_layers")

    # Detailed per-dataset CSV
    rows_by_ds = []
    for ds_name in sampled:
        rows_by_ds.extend(_stats_rows(all_mid[ds_name], ds_name))
    pd.DataFrame(rows_by_ds).to_csv(
        output_path / "bos_attention_stats_by_dataset.csv", index=False
    )

    # Pool across datasets
    pooled_mid   = {key: [] for key, *_ in INTERVENTIONS}
    pooled_scope = {key: [] for key, *_ in INTERVENTIONS}
    for ds_name in sampled:
        for key, *_ in INTERVENTIONS:
            pooled_mid[key].extend(all_mid[ds_name][key])
            pooled_scope[key].extend(all_scope[ds_name][key])

    rows_overall = _stats_rows(pooled_mid, "all_datasets")
    df_overall = pd.DataFrame(rows_overall)
    df_overall.to_csv(output_path / "bos_attention_stats_overall.csv", index=False)

    _write_readable_bos_summaries(output_path, pooled_scope, "all_layers")
    _write_readable_bos_summaries(output_path, pooled_mid,   "mid_layers")

    print(f"\n{'═'*60}")
    print("Pooled results across all datasets (OPT):")
    print(df_overall[["intervention", "description", "display"]].to_string(index=False))
    print(f"\nAll outputs saved to {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Attention intervention experiments on OPT (Table 1 + Fig 5 analog).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="HF OPT model id. Must be pre-LayerNorm (opt-125m/1.3b/2.7b/6.7b). "
             "opt-350m is rejected (post-LN + word-embed projection).",
    )
    parser.add_argument(
        "--mode",
        choices=["sentence", "dataset"],
        default="dataset",
        help="'dataset' computes Table 1 BOS-attention statistics on 3 benchmarks; "
             "'sentence' produces per-(layer,head) heatmap grids (Fig 5 analog).",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default=DEFAULT_SENTENCE,
        help="Input sentence for sentence mode (ignored in dataset mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Root directory for all outputs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Examples to sample from each benchmark dataset (dataset mode).",
    )
    parser.add_argument(
        "--cut-length",
        type=int,
        default=DEFAULT_CUT_LENGTH,
        help="Token count threshold and truncation target. Examples shorter than "
             "this are discarded; longer ones are truncated to this length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for dataset sampling.",
    )
    parser.add_argument(
        "--layer-mode",
        choices=["scaled", "fixed"],
        default="scaled",
        help="Mid-layer band. 'scaled' (default) excludes the first 3 and last layer "
             "(= layers 4-11 for a 12-layer model like opt-125m; extends for deeper opt-1.3b/2.7b); "
             "'fixed' forces layers 4-11 on every size.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model dtype. Default float32; use a smaller dtype only if VRAM-constrained.",
    )
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    model, tokenizer = load_opt(args.model_name, dtype=dtype)
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    band = compute_band(num_layers, args.layer_mode)
    print(f"Layers: {num_layers}; mid-band [{band[0]}, {band[1]}) (layer-mode {args.layer_mode}).\n")

    if args.mode == "sentence":
        sentence_analysis(model, tokenizer, args.sentence, args.output_dir, num_heads)
    else:
        dataset_analysis(
            model, tokenizer, args.output_dir, num_heads,
            sample_size=args.sample_size,
            cut_length=args.cut_length,
            seed=args.seed,
            band=band,
            model_name=args.model_name,
            dtype=args.dtype,
            layer_mode=args.layer_mode,
        )


if __name__ == "__main__":
    main()
