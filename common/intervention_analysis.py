# -*- coding: utf-8 -*-
"""Configuration-aware GPT-2 intervention executor.

The original implementation is retained in ``intervention_analysis_legacy``.
This module re-exports its public API and replaces only
``manual_self_attention_new`` so the manual path honors the checkpoint's
``scale_attn_by_inverse_layer_idx`` and ``reorder_and_upcast_attn`` settings.
"""

import math
import random

import torch
import torch.nn.functional as F

try:  # package import
    from . import intervention_analysis_legacy as _legacy
except ImportError:  # scripts add common/ directly to sys.path
    import intervention_analysis_legacy as _legacy

# Preserve the complete legacy module API, including helper names used by the
# experiment scripts.
for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


def manual_self_attention_new(hidden_states, layer,
                              ppes=None,
                              intervene_query_bias=False,
                              query_bias_scale=1.0,
                              fixed_wk_zero_indices=None,
                              wk_scale_indices=None,
                              wk_scale=1.0,
                              random_wk_zero_rows=False,
                              random_wk_zero_count=3,
                              compute_diagnostics=True):
    """Run GPT-2 self-attention while honoring checkpoint attention flags.

    OpenAI GPT-2 has both relevant flags disabled, so the validated legacy path
    is used unchanged there. CRFM checkpoints enable both flags. For those
    checkpoints this function mirrors Hugging Face GPT-2 eager attention:
    layer-index scaling is applied to the scores and the reordered path performs
    the QK product and softmax in fp32 before casting probabilities back to V's
    dtype.
    """
    attn_layer = layer.attn
    inverse_layer_scale = bool(
        getattr(attn_layer, "scale_attn_by_inverse_layer_idx", False))
    reorder_and_upcast = bool(
        getattr(attn_layer, "reorder_and_upcast_attn", False))

    if not inverse_layer_scale and not reorder_and_upcast:
        return _legacy.manual_self_attention_new(
            hidden_states, layer, ppes=ppes,
            intervene_query_bias=intervene_query_bias,
            query_bias_scale=query_bias_scale,
            fixed_wk_zero_indices=fixed_wk_zero_indices,
            wk_scale_indices=wk_scale_indices,
            wk_scale=wk_scale,
            random_wk_zero_rows=random_wk_zero_rows,
            random_wk_zero_count=random_wk_zero_count,
            compute_diagnostics=compute_diagnostics,
        )

    # Diagnostics are independent of the two score-computation flags except for
    # the returned pre-mask score tensor. Reuse the established diagnostic path
    # and replace the forward-derived tensors below. Preserve RNG state so the
    # size-matched random-column control uses the same columns in both passes.
    legacy_result = None
    if compute_diagnostics:
        rng_state = random.getstate()
        legacy_result = _legacy.manual_self_attention_new(
            hidden_states, layer, ppes=ppes,
            intervene_query_bias=intervene_query_bias,
            query_bias_scale=query_bias_scale,
            fixed_wk_zero_indices=fixed_wk_zero_indices,
            wk_scale_indices=wk_scale_indices,
            wk_scale=wk_scale,
            random_wk_zero_rows=random_wk_zero_rows,
            random_wk_zero_count=random_wk_zero_count,
            compute_diagnostics=True,
        )
        random.setstate(rng_state)

    num_heads = attn_layer.num_heads
    hidden_size = hidden_states.size(-1)
    head_dim = hidden_size // num_heads

    score_scale = 1.0
    if getattr(attn_layer, "scale_attn_weights", True):
        score_scale /= math.sqrt(head_dim)
    if inverse_layer_scale:
        layer_idx = getattr(attn_layer, "layer_idx", None)
        if layer_idx is None:
            raise ValueError(
                "scale_attn_by_inverse_layer_idx=True requires attention.layer_idx")
        score_scale /= float(layer_idx + 1)

    qkv_weight = attn_layer.c_attn.weight.t()
    qkv_bias = attn_layer.c_attn.bias
    wq, wk, wv = qkv_weight.chunk(3, dim=0)
    bq, bk, bv = qkv_bias.chunk(3, dim=0)

    if intervene_query_bias:
        bq = torch.zeros_like(bq)
    elif query_bias_scale != 1.0:
        bq = query_bias_scale * bq

    wk_active = wk.clone()
    if fixed_wk_zero_indices is not None:
        wk_active[:, fixed_wk_zero_indices] = 0.0
    elif random_wk_zero_rows:
        random_indices_to_zero = [
            random.randint(0, hidden_size - 1)
            for _ in range(random_wk_zero_count)
        ]
        wk_active[:, random_indices_to_zero] = 0.0
    if wk_scale_indices is not None and wk_scale != 1.0:
        wk_active[:, wk_scale_indices] = (
            wk_scale * wk_active[:, wk_scale_indices])

    query_proj = F.linear(hidden_states, wq, bq)
    key_proj = F.linear(hidden_states, wk_active, bk)
    value_proj = F.linear(hidden_states, wv, bv)

    def reshape_for_multihead(x):
        return x.view(
            x.size(0), x.size(1), num_heads, head_dim).transpose(1, 2)

    query = reshape_for_multihead(query_proj)
    key = reshape_for_multihead(key_proj)
    value = reshape_for_multihead(value_proj)

    if reorder_and_upcast:
        batch_size, _, query_length, key_dim = query.shape
        key_length = key.size(-2)
        with torch.autocast(device_type=query.device.type, enabled=False):
            query_2d = query.reshape(
                batch_size * num_heads, query_length, key_dim).float()
            key_2d = key.transpose(-2, -1).reshape(
                batch_size * num_heads, key_dim, key_length).float()
            attention_scores = torch.baddbmm(
                torch.empty(
                    batch_size * num_heads, query_length, key_length,
                    dtype=torch.float32, device=query.device),
                query_2d,
                key_2d,
                beta=0,
                alpha=score_scale,
            ).reshape(batch_size, num_heads, query_length, key_length)
    else:
        attention_scores = torch.matmul(
            query, key.transpose(-2, -1)) * score_scale

    attention_scores_before_mask = (
        attention_scores.clone() if compute_diagnostics else None)
    sequence_length = hidden_states.size(1)
    causal_mask = torch.tril(torch.ones(
        sequence_length, sequence_length,
        device=hidden_states.device, dtype=torch.bool,
    )).view(1, 1, sequence_length, sequence_length)
    attention_scores = attention_scores.masked_fill(
        ~causal_mask, float("-inf"))
    attention_weights = F.softmax(
        attention_scores, dim=-1).to(value.dtype)

    context = torch.matmul(attention_weights, value)
    context = context.transpose(1, 2).contiguous().view(
        hidden_states.size(0), sequence_length, hidden_size)
    attention_output = F.linear(
        context, attn_layer.c_proj.weight.t(), attn_layer.c_proj.bias)

    if legacy_result is not None:
        return (
            attention_output,
            attention_weights,
            legacy_result[2],
            legacy_result[3],
            attention_scores_before_mask,
            *legacy_result[5:],
        )

    return (
        attention_output,
        attention_weights,
        None,
        None,
        None,
        None,
        [],
        [],
        [],
        [],
        None,
    )


# Functions defined in the retained implementation resolve this name from that
# module's globals, so patch it there as well as exporting it here.
_legacy.manual_self_attention_new = manual_self_attention_new


if __name__ == "__main__":
    _legacy.main()
