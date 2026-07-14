# -*- coding: utf-8 -*-
"""intervention_analysis_qwen.py — Attention intervention experiments on Qwen2.5.

Qwen2.5 analog of ``intervention_analysis.py`` (which targets GPT-2).  It recreates
**Table 1** (the ten BOS-attention interventions) and the **Figure 5** analog
(per-(layer, head) attention-heatmap grids) for Alibaba's Qwen2.5 family.

Why Qwen2.5 — and what has to change
------------------------------------
The paper's sink circuit relies on two ingredients:
  1. a learned **query bias** ``b_Q``, and
  2. a **positional signal** that gives position 0 a distinct key.

Qwen2.5 keeps ingredient (1): every attention block still has ``q_proj.bias``
(``attention_bias=True``), unlike Llama/Mistral which drop it.  This makes Qwen2.5
one of the few modern families where the (b) *No Query Bias* intervention is even
meaningful, so the central mechanism of the paper can be probed directly.

Ingredient (2) is delivered differently.  GPT-2/OPT use **learned additive
absolute positional embeddings** that live in the residual stream, so the paper
can define an *effective positional embedding* ``EPE = PE + FFN^(0)(PE)`` and
manipulate it as a vector.  Qwen2.5 instead uses **Rotary Position Embeddings
(RoPE)**: position enters only by rotating the query/key vectors inside attention,
and *nothing positional is ever added to the residual stream*.  There is therefore
no additive ``PE`` vector and no separable ``EPE``.  The positional interventions
are reframed as manipulations of the RoPE **position ids** (see the intervention
table below), which is the faithful RoPE-space equivalent of "which position does
token *t* look like".

GPT-2 → Qwen2.5 module mapping
------------------------------
    transformer.wte              →  model.model.embed_tokens
    transformer.wpe              →  (none — RoPE, applied inside attention)
    transformer.h                →  model.model.layers
    layer.attn.c_attn (fused QKV) →  self_attn.{q,k,v}_proj (separate nn.Linear, bias=True)
    layer.attn.c_proj            →  self_attn.o_proj (bias=False)
    layer.ln_1 / layer.ln_2      →  input_layernorm / post_attention_layernorm  (RMSNorm)
    layer.mlp (GELU FFN)         →  down_proj(SiLU(gate_proj(x)) * up_proj(x))   (SwiGLU)
    config.n_head                →  config.num_attention_heads

Two extra structural facts handled here:
  * **Grouped-query attention** — ``num_key_value_heads`` may be < ``num_attention_heads``;
    keys/values are repeated to the query-head count (``repeat_kv``) before ``QK^T``.
  * **RMSNorm** — used as-is by calling the module (no learned bias / mean subtraction).

Because ``F.linear(x, W, b) == x @ W.T + b`` and ``nn.Linear.weight`` is stored
as ``[out, in]`` — the same layout GPT-2's fused weight has after its ``.t()`` —
Qwen2.5's ``q_proj.weight`` drops straight into the same key/query math.  RoPE and
GQA are the only genuinely new pieces, and both are reimplemented explicitly below
so the harness stays a transparent, from-scratch forward pass.

Two modes
---------
  --mode dataset   (default)   Table 1: BOS-attention statistics over SST-2, GSM8K,
                               HumanEval.  Outputs under
                               ``<output-dir>/dataset_analysis_qwen/``.
  --mode sentence              Figure 5 analog: per-(layer, head) and head-averaged
                               2x5 heatmap grids for one sentence.  Outputs under
                               ``<output-dir>/sentence_analysis_qwen/``.

Interventions (labelled a-j) — same 10 keys/labels as the GPT-2 version
-----------------------------------------------------------------------
  (a) Baseline           token embeddings + standard RoPE positions, all layers intact.
  (b) No Query Bias      bq = 0 in every layer (isolates content-only queries).
  (c) Remove First PE    give token 0 the RoPE position of token 1 (pos_ids = [1,1,2,3,…]).
  (d) Swap EPE           swap the RoPE positions of tokens 0 and 1 (pos_ids = [1,0,2,3,…]).
  (e) Swap PE            same RoPE-position swap as (d).  Under RoPE the raw-PE vs
                         effective-PE distinction collapses (there is no additive PE),
                         so (d) and (e) coincide; both rows are kept for column
                         alignment with the GPT-2/OPT tables.
  (f) Nullify BOS Token  zero the position-0 token embedding before processing (RoPE kept).
  (g) No MLP             skip the SwiGLU FFN block in every layer.
  (h) No PE              disable RoPE entirely (pos_ids = 0 everywhere → identity rotation).
  (i) Zero Top-3 Wk      zero the 3 columns of Wk matching the top-3 |dims| of the
                         position-0 token embedding (the RoPE-model stand-in for the
                         top-|EPE[0]| dimensions).
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
from transformers import AutoTokenizer, AutoModelForCausalLM

# Shared analysis modules live in the repository's top-level ``common/`` folder
# (one master copy each). Make them importable regardless of the launch directory.
import os
import sys

_COMMON_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "common")
)
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from datasets_loader import (
    sample_benchmark_datasets,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_CUT_LENGTH,
    DEFAULT_SEED,
)
# Architecture-agnostic metric (pure attention-tensor math): reuse as-is.  It
# averages attention to position 0 from the second-half tokens over layers 4-11
# (the paper's "mid" range); that definition lives in intervention_analysis and
# is inherited unchanged here. The massive-activation selector and run_config.json
# provenance builder are shared too, so the "massive" rule (|signal| > mean + 3*std,
# top-3 fallback) is identical across all harnesses.
from intervention_analysis import (
    compute_bos_attention_metric, compute_band,
    select_massive_coords, build_run_config,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_SENTENCE = "It was the best of times, it was the worst of times, it was the age of wisdom."

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Model loading (with Qwen2-family structural guard)
# ═══════════════════════════════════════════════════════════════════════════════

def load_qwen(model_name, dtype="float32"):
    """Load a Qwen2.5 model + tokenizer and assert it matches the harness's assumptions.

    The harness assumes the Qwen2 pre-RMSNorm decoder block
    ``x = x + attn(input_layernorm(x)); x = x + mlp(post_attention_layernorm(x))``
    with separate q/k/v/o projections and a learned query bias.  We validate that
    structure up front so an unexpected architecture fails loudly rather than
    silently producing wrong numbers.
    """
    print(f"Loading {model_name} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype == "auto":
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"--dtype must be one of {list(_DTYPE_MAP)} or 'auto', got {dtype!r}")
        torch_dtype = _DTYPE_MAP[dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # --- structural validation -------------------------------------------------
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "layers") or not hasattr(inner, "embed_tokens"):
        raise ValueError(
            f"{model_name} does not expose the expected model.model.{{embed_tokens,layers}} "
            "structure. This harness targets the Qwen2/Qwen2.5 decoder."
        )
    layer0 = inner.layers[0]
    attn = layer0.self_attn
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if not hasattr(attn, proj):
            raise ValueError(f"{model_name}: attention block is missing {proj}; not a Qwen2-style model.")
    if attn.q_proj.bias is None:
        raise ValueError(
            f"{model_name} has no query bias (q_proj.bias is None). The paper's sink "
            "circuit is driven by the learned query bias; without it the (b) No Query "
            "Bias intervention is a no-op. Use a Qwen2.5 checkpoint (attention_bias=True)."
        )
    for norm in ("input_layernorm", "post_attention_layernorm"):
        if not hasattr(layer0, norm):
            raise ValueError(f"{model_name}: decoder layer is missing {norm}; not a Qwen2-style model.")

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    print(
        f"Model loaded — {cfg.num_hidden_layers} layers, "
        f"{cfg.num_attention_heads} query heads / {cfg.num_key_value_heads} kv heads, "
        f"hidden {cfg.hidden_size}, head_dim {head_dim}, "
        f"rope_theta {getattr(cfg, 'rope_theta', 10000.0)}."
    )
    print(f"Using device: {device} (dtype {torch_dtype}).\n")
    return model, tokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# Embedding retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def get_token_embeddings_qwen(model, inputs):
    """Run one forward pass to capture the token embeddings via a hook.

    Unlike GPT-2/OPT there is no additive positional embedding to capture — RoPE
    is applied inside attention — so only the ``embed_tokens`` output is returned.
    Qwen2 does not scale the embeddings, so this tensor is exactly the layer-0 input.
    """
    inner = model.model
    captured = {}

    def _tok_hook(_module, _input, output):
        captured["token_embeddings"] = output.clone()
        return output

    handle = inner.embed_tokens.register_forward_hook(_tok_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return captured["token_embeddings"]

# ═══════════════════════════════════════════════════════════════════════════════
# RoPE + GQA primitives (reimplemented to match Qwen2's HF implementation)
# ═══════════════════════════════════════════════════════════════════════════════

def build_rope_cos_sin(position_ids, head_dim, rope_theta, device):
    """Return (cos, sin), each ``[seq_len, head_dim]``, for the given position ids.

    Matches HF's GPT-NeoX-style RoPE used by Qwen2: the head dim is split into two
    halves and rotated by ``rotate_half``.  Computed in float32 for precision and
    cast at application time.
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = position_ids.to(device=device, dtype=torch.float32)          # [seq]
    freqs = torch.outer(pos, inv_freq)                                  # [seq, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)                            # [seq, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    """Apply RoPE to q, k of shape ``[1, n_heads, seq, head_dim]``.

    ``cos``/``sin`` are ``[seq, head_dim]``; broadcast over batch and head dims.
    """
    cos_b = cos[None, None, :, :].to(q.dtype)
    sin_b = sin[None, None, :, :].to(q.dtype)
    q_rot = q * cos_b + _rotate_half(q) * sin_b
    k_rot = k * cos_b + _rotate_half(k) * sin_b
    return q_rot, k_rot


def _repeat_kv(x, n_rep):
    """Expand ``[1, n_kv, seq, head_dim]`` to ``[1, n_kv*n_rep, seq, head_dim]`` (GQA)."""
    if n_rep == 1:
        return x
    b, n_kv, seq, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, seq, hd)
        .reshape(b, n_kv * n_rep, seq, hd)
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Manual self-attention (Qwen2.5)
# ═══════════════════════════════════════════════════════════════════════════════

def manual_self_attention_qwen(hidden_states, layer, num_heads, num_kv_heads, head_dim,
                               cos, sin,
                               intervene_query_bias=False,
                               fixed_wk_zero_indices=None,
                               random_wk_zero_rows=False,
                               random_wk_zero_count=3):
    """Manual Qwen2 self-attention for a single (already RMSNorm'd) layer input.

    Mirrors ``manual_self_attention_new`` from the GPT-2 harness but uses Qwen2's
    separate q/k/v/o projections, RoPE (via the supplied ``cos``/``sin``), and GQA
    key/value repetition.  Returns only what the intervention loop consumes:
    ``(attention_output, attention_weights)`` where the weights are expanded to the
    full query-head count.

    Args:
        hidden_states: [1, seq, hidden] — the RMSNorm'd input to attention.
        layer:         a ``Qwen2DecoderLayer``.
        num_heads:     number of query heads.
        num_kv_heads:  number of key/value heads (GQA; may be < num_heads).
        head_dim:      per-head dimension.
        cos, sin:      RoPE tables ``[seq, head_dim]`` for this run's position ids.
        intervene_query_bias:  if True, nullify the query bias ``b_Q``.
        fixed_wk_zero_indices: list of Wk *columns* (input coords) to zero, or None.
        random_wk_zero_rows:   if True, zero ``random_wk_zero_count`` random Wk columns.
        random_wk_zero_count:  number of random Wk columns to zero (size-matched to (i)).
    """
    attn = layer.self_attn
    seq_len = hidden_states.size(1)
    hidden_size = hidden_states.size(-1)
    scale = head_dim ** -0.5  # 1 / sqrt(head_dim)

    # nn.Linear weights are [out, in]; F.linear(x, W, b) == x @ W.T + b.
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

    def reshape_heads(x, n):
        return x.view(1, seq_len, n, head_dim).transpose(1, 2)  # [1, n, seq, head_dim]

    query = reshape_heads(query_proj, num_heads)
    key = reshape_heads(key_proj, num_kv_heads)
    value = reshape_heads(value_proj, num_kv_heads)

    # RoPE is applied to Q and K before the (grouped) key repetition.
    query, key = _apply_rope(query, key, cos, sin)

    # GQA: repeat K/V so every query head has a matching key/value head.
    n_rep = num_heads // num_kv_heads
    key = _repeat_kv(key, n_rep)
    value = _repeat_kv(value, n_rep)

    # Scaled dot-product attention with a causal mask.
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=hidden_states.device)
    ).view(1, 1, seq_len, seq_len)
    attention_scores = attention_scores.masked_fill(causal_mask == 0, float("-inf"))
    attention_weights = F.softmax(attention_scores, dim=-1)

    context = torch.matmul(attention_weights, value)
    context = context.transpose(1, 2).contiguous().view(1, seq_len, num_heads * head_dim)

    attention_output = F.linear(context, attn.o_proj.weight, attn.o_proj.bias)
    return attention_output, attention_weights

# ═══════════════════════════════════════════════════════════════════════════════
# SwiGLU FFN
# ═══════════════════════════════════════════════════════════════════════════════

def _qwen_ffn(layer, x):
    """Qwen2 SwiGLU feed-forward block: down_proj(SiLU(gate_proj(x)) * up_proj(x))."""
    mlp = layer.mlp
    return mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))

# ═══════════════════════════════════════════════════════════════════════════════
# Generic intervention loop (Qwen2.5)
# ═══════════════════════════════════════════════════════════════════════════════

def run_intervention_loop_qwen(model, layer_input, geom, position_ids,
                               attn_kwargs=None, skip_mlp=False):
    """Run through all Qwen2 decoder layers, collecting per-layer attention weights.

    Block structure (pre-RMSNorm, matches the GPT-2 harness):
        normed    = input_layernorm(x)
        x         = x + attn(normed)
        normed_ar = post_attention_layernorm(x)
        x         = x + SwiGLU(normed_ar)          [unless skip_mlp]

    ``position_ids`` selects the RoPE geometry for this run (baseline, swapped,
    disabled, …); the same rotation is used in every layer, exactly as RoPE does
    natively.  Returns one ``[num_heads, seq_len, seq_len]`` tensor per layer, with
    keys/values already expanded to the full query-head count.
    """
    attn_kwargs = attn_kwargs or {}
    layers = model.model.layers
    num_heads = geom["num_heads"]
    num_kv_heads = geom["num_kv_heads"]
    head_dim = geom["head_dim"]

    cos, sin = build_rope_cos_sin(
        position_ids, head_dim, geom["rope_theta"], layer_input.device
    )

    attention_weights_per_layer = []
    for i in range(len(layers)):
        layer = layers[i]

        normalized = layer.input_layernorm(layer_input.clone())

        attention_output, attention_weights = manual_self_attention_qwen(
            normalized, layer, num_heads, num_kv_heads, head_dim, cos, sin, **attn_kwargs
        )

        # Keep all heads: shape [num_heads, seq_len, seq_len] (batch dim squeezed).
        attention_weights_per_layer.append(attention_weights.cpu().detach()[0])

        if skip_mlp:
            layer_input = layer_input + attention_output
        else:
            attention_and_residual = layer_input + attention_output
            normalized_ar = layer.post_attention_layernorm(attention_and_residual)
            mlp_output = _qwen_ffn(layer, normalized_ar)
            layer_input = attention_and_residual + mlp_output

    return attention_weights_per_layer


def _baseline_position_ids(seq_len, device):
    return torch.arange(seq_len, device=device)

# ═══════════════════════════════════════════════════════════════════════════════
# Intervention functions
#
# Each takes (model, token_embeddings, geom) and returns a list of per-layer
# attention-weight tensors [num_heads, seq_len, seq_len].
# ═══════════════════════════════════════════════════════════════════════════════

def intervention_a_baseline(model, token_embeddings, geom):
    """(a) Baseline: token embeddings + standard RoPE positions, all layers intact."""
    seq_len = token_embeddings.size(1)
    pos = _baseline_position_ids(seq_len, token_embeddings.device)
    return run_intervention_loop_qwen(model, token_embeddings.clone(), geom, pos)


def intervention_b_nullify_query_bias(model, token_embeddings, geom):
    """(b) No Query Bias: sets bq = 0 in every layer."""
    seq_len = token_embeddings.size(1)
    pos = _baseline_position_ids(seq_len, token_embeddings.device)
    return run_intervention_loop_qwen(
        model, token_embeddings.clone(), geom, pos,
        attn_kwargs={"intervene_query_bias": True},
    )


def intervention_c_remove_first_pe(model, token_embeddings, geom):
    """(c) Remove First PE: give token 0 the RoPE position of token 1 → pos_ids = [1,1,2,3,…].

    RoPE analog of the GPT-2 intervention "position 0 receives PE[1] instead of PE[0]":
    token 0 is rotated as if it sat at position 1.
    """
    seq_len = token_embeddings.size(1)
    pos = _baseline_position_ids(seq_len, token_embeddings.device)
    if seq_len > 1:
        pos = pos.clone()
        pos[0] = pos[1].clone()
    return run_intervention_loop_qwen(model, token_embeddings.clone(), geom, pos)


def intervention_d_swap_epe_normalized(model, token_embeddings, geom):
    """(d) Swap EPE: swap the RoPE positions of tokens 0 and 1 → pos_ids = [1,0,2,3,…].

    GPT-2 swaps the *effective* positional embedding (PE + FFN^(0)(PE)) between the
    first two positions.  Under RoPE there is no additive residual-stream PE to
    separate into "effective" vs "raw", so the swap is enacted directly on the RoPE
    position ids of tokens 0 and 1 — after which token 0 rotates like position 1 and
    vice-versa.
    """
    seq_len = token_embeddings.size(1)
    pos = _baseline_position_ids(seq_len, token_embeddings.device)
    if seq_len > 1:
        pos = pos.clone()
        pos[0], pos[1] = pos[1].clone(), pos[0].clone()
    return run_intervention_loop_qwen(model, token_embeddings.clone(), geom, pos)


def intervention_e_swap_pe(model, token_embeddings, geom):
    """(e) Swap PE: same RoPE-position swap as (d).

    In GPT-2 (e) swaps the raw PE vectors while (d) swaps the effective PE.  RoPE has
    a single notion of position and no additive PE, so the two collapse to the same
    operation.  This row is kept identical to (d) to preserve column alignment with
    the GPT-2/OPT Table 1; matching (d)≈(e) values are the expected signature of a
    RoPE model.
    """
    return intervention_d_swap_epe_normalized(model, token_embeddings, geom)


def intervention_f_nullify_bos_token(model, token_embeddings, geom):
    """(f) Nullify BOS Token: zero the position-0 token embedding; RoPE positions kept."""
    seq_len = token_embeddings.size(1)
    te_copy = token_embeddings.clone()
    te_copy[0][0] = torch.zeros_like(te_copy[0][0])
    pos = _baseline_position_ids(seq_len, token_embeddings.device)
    return run_intervention_loop_qwen(model, te_copy, geom, pos)


def intervention_g_no_mlp(model, token_embeddings, geom):
    """(g) No MLP: skip the SwiGLU FFN block in every layer."""
    seq_len = token_embeddings.size(1)
    pos = _baseline_position_ids(seq_len, token_embeddings.device)
    return run_intervention_loop_qwen(model, token_embeddings.clone(), geom, pos, skip_mlp=True)


def intervention_h_no_pe(model, token_embeddings, geom):
    """(h) No PE: disable RoPE entirely (pos_ids = 0 everywhere → identity rotation).

    RoPE analog of "input is token embeddings only, no positional signal": with all
    position ids equal to 0 the rotation is the identity (cos=1, sin=0), so queries
    and keys carry no positional information anywhere in the network.
    """
    seq_len = token_embeddings.size(1)
    pos = torch.zeros(seq_len, dtype=torch.long, device=token_embeddings.device)
    return run_intervention_loop_qwen(model, token_embeddings.clone(), geom, pos)


def intervention_i_zero_top_wk(model, token_embeddings, geom):
    """(i) Zero Top-3 Wk: zero the Wk columns matching the *massive activations* of the
    position-0 token embedding — coordinates whose |value| exceeds ``mean + 3·std`` of the
    embedding's magnitudes (see ``select_massive_coords``; ≥3 by construction).

    GPT-2 zeros the Wk columns aligned with the massive activations of ``EPE[0]`` — the
    large-activation coordinates of the effective positional embedding that dominate the
    key at position 0.  Qwen2.5 has no additive EPE, so the RoPE-model stand-in for "the
    dimensions that make position 0's key special" is the massive-magnitude coordinates of
    the first token's residual-stream embedding. Because that embedding is token-dependent,
    the set is identified per sentence (recorded in run_config.json as a per-sentence
    method). The same columns are zeroed in Wk for every layer.
    """
    seq_len = token_embeddings.size(1)
    pos = _baseline_position_ids(seq_len, token_embeddings.device)

    fixed_wk_zero_indices = select_massive_coords(token_embeddings[0][0])

    return run_intervention_loop_qwen(
        model, token_embeddings.clone(), geom, pos,
        attn_kwargs={"fixed_wk_zero_indices": fixed_wk_zero_indices},
    )


def intervention_j_zero_random_wk(model, token_embeddings, geom):
    """(j) Zero Random Wk: zero ``k`` randomly chosen Wk columns — a size-matched control
    for (i), where ``k`` is the number of massive coordinates (i) zeroes for this sentence."""
    k = len(select_massive_coords(token_embeddings[0][0]))
    random.seed(DEFAULT_SEED)
    seq_len = token_embeddings.size(1)
    pos = _baseline_position_ids(seq_len, token_embeddings.device)
    return run_intervention_loop_qwen(
        model, token_embeddings.clone(), geom, pos,
        attn_kwargs={"random_wk_zero_rows": True, "random_wk_zero_count": k},
    )


# Ordered registry — same 10 keys/labels/descriptions as the GPT-2 version, so the
# Qwen summaries line up column-for-column with the original outputs.
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

def _model_geometry(model):
    """Collect the head/RoPE geometry the intervention loop needs from the config."""
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return {
        "num_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": head_dim,
        "rope_theta": float(getattr(cfg, "rope_theta", 10000.0)),
    }


def run_all_interventions(model, token_embeddings, geom):
    """Run every registered intervention and return {key: [layer_weights]}."""
    results = {}
    for key, _label, desc, fn in INTERVENTIONS:
        print(f"  Running {key} ({desc})...")
        with torch.no_grad():
            results[key] = fn(model, token_embeddings, geom)
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


def sentence_analysis(model, tokenizer, sentence, output_dir, geom):
    """Generate per-(layer, head) and per-layer-averaged 2x5 heatmap grids (Fig 5 analog)."""
    output_path = Path(output_dir) / "sentence_analysis_qwen"
    output_path.mkdir(parents=True, exist_ok=True)

    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(model.device)
    token_embeddings = get_token_embeddings_qwen(model, inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print(f"Sentence: {sentence!r}")
    print(f"Tokens ({len(tokens)}): {tokens}")

    all_results = run_all_interventions(model, token_embeddings, geom)

    num_layers = len(model.model.layers)
    num_heads = geom["num_heads"]
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

def _run_sentences(model, tokenizer, sentences, ds_label, num_layers, geom, band=None):
    """Run all interventions on *sentences*, return {key: (mid_scores, all_scores)}."""
    ls, le = band if band is not None else (None, None)
    scores_mid = {key: [] for key, *_ in INTERVENTIONS}
    scores_all = {key: [] for key, *_ in INTERVENTIONS}
    n = len(sentences)
    for i, sentence in enumerate(sentences):
        print(f"  [{ds_label} {i + 1}/{n}] {sentence[:80]}...")
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(model.device)
        token_embeddings = get_token_embeddings_qwen(model, inputs)
        results = run_all_interventions(model, token_embeddings, geom)
        for key, *_ in INTERVENTIONS:
            scores_mid[key].append(
                compute_bos_attention_metric(results[key], num_layers, "mid",
                                             layer_start=ls, layer_end=le)
            )
            scores_all[key].append(
                compute_bos_attention_metric(results[key], num_layers, "all")
            )
    return scores_mid, scores_all


def dataset_analysis(model, tokenizer, output_dir, geom,
                     sample_size=DEFAULT_SAMPLE_SIZE,
                     cut_length=DEFAULT_CUT_LENGTH,
                     seed=DEFAULT_SEED,
                     band=None,
                     model_name=DEFAULT_MODEL,
                     dtype="float32",
                     layer_mode="scaled"):
    """Compute the BOS-attention metric on three standard benchmarks (Table 1).

    Datasets: SST-2 (natural language), GSM8K (math), HumanEval (code).  Each is
    sampled to *sample_size* examples of at least *cut_length* Qwen tokens, then
    truncated to exactly *cut_length* tokens.  Outputs under
    ``<output_dir>/dataset_analysis_qwen/`` mirror the GPT-2 filenames; the pooled
    ``bos_attention_summary_mid_layers.{txt,csv}`` is **Table 1** (Qwen2.5). A
    ``run_config.json`` provenance record is written alongside; because Qwen is RoPE
    (no additive EPE), intervention (i)'s massive coordinates are the outliers of the
    *position-0 token embedding* and are identified per sentence, so the record notes the
    method rather than a fixed coordinate list.
    """
    output_path = Path(output_dir) / "dataset_analysis_qwen"
    output_path.mkdir(parents=True, exist_ok=True)

    num_layers = len(model.model.layers)

    run_cfg = build_run_config(
        model, None, model_name=model_name, dtype=dtype,
        layer_mode=layer_mode, band=band, seed=seed,
        sample_size=sample_size, cut_length=cut_length,
        harness="intervention_analysis_qwen.py",
        massive_signal="position-0 token embedding (RoPE stand-in; identified per sentence)")
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
        s_mid, s_all = _run_sentences(model, tokenizer, sentences, ds_name, num_layers, geom, band=band)
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
    print("Pooled results across all datasets (Qwen2.5):")
    print(df_overall[["intervention", "description", "display"]].to_string(index=False))
    print(f"\nAll outputs saved to {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Attention intervention experiments on Qwen2.5 (Table 1 + Fig 5 analog).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        "--model",
        dest="model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="HF Qwen2.5 model id (e.g. Qwen/Qwen2.5-0.5B ... Qwen/Qwen2.5-14B). "
             "Must expose a learned query bias (attention_bias=True).",
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
        "--dtype",
        choices=["float32", "float16", "bfloat16", "auto"],
        default="float32",
        help="Model dtype. 'auto' uses float16 on CUDA and float32 on CPU. "
             "Larger checkpoints (7B/14B) typically need float16/bfloat16 to fit.",
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
             "(= layers 4-11 for a 12-layer model; extends for the deeper Qwen2.5 checkpoints, "
             "all ≥24 layers); 'fixed' forces layers 4-11 on every size.",
    )
    args = parser.parse_args()

    model, tokenizer = load_qwen(args.model_name, dtype=args.dtype)
    geom = _model_geometry(model)
    num_layers = len(model.model.layers)
    band = compute_band(num_layers, args.layer_mode)
    print(f"Layers: {num_layers}; mid-band [{band[0]}, {band[1]}) (layer-mode {args.layer_mode}).\n")

    if args.mode == "sentence":
        sentence_analysis(model, tokenizer, args.sentence, args.output_dir, geom)
    else:
        dataset_analysis(
            model, tokenizer, args.output_dir, geom,
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
