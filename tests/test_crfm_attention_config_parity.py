from pathlib import Path
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from intervention_analysis import manual_self_attention_new
from residual_sink_analysis import _decompose_layer


def _tiny_crfm_style_model():
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=8,
        n_layer=2,
        n_head=2,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        scale_attn_by_inverse_layer_idx=True,
        reorder_and_upcast_attn=True,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def _native_attention(layer, hidden_states):
    attn = layer.attn
    query, key, value = attn.c_attn(hidden_states).split(
        attn.split_size, dim=2)
    shape = (*key.shape[:-1], -1, attn.head_dim)
    key = key.view(shape).transpose(1, 2)
    value = value.view(shape).transpose(1, 2)
    query = query.view(shape).transpose(1, 2)

    seq = hidden_states.size(1)
    mask = torch.triu(
        torch.full(
            (1, 1, seq, seq),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
            device=hidden_states.device,
        ),
        diagonal=1,
    )
    output, weights = attn._upcast_and_reordered_attn(
        query, key, value, mask)
    output = output.reshape(*output.shape[:-2], -1).contiguous()
    return attn.c_proj(output), weights


def test_manual_attention_matches_native_crfm_flags():
    torch.manual_seed(0)
    model = _tiny_crfm_style_model()
    layer = model.transformer.h[1]
    hidden = torch.randn(1, 7, model.config.n_embd)
    normalized = layer.ln_1(hidden)

    manual_output, manual_weights, *_ = manual_self_attention_new(
        normalized, layer, ppes=None, compute_diagnostics=False)
    native_output, native_weights = _native_attention(layer, normalized)

    torch.testing.assert_close(
        manual_weights, native_weights, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        manual_output, native_output, atol=1e-6, rtol=1e-5)


def test_decomposition_full_attention_matches_corrected_manual_path():
    torch.manual_seed(1)
    model = _tiny_crfm_style_model()
    layer = model.transformer.h[1]
    hidden = torch.randn(1, 7, model.config.n_embd)
    normalized = layer.ln_1(hidden)
    ppes = torch.randn(7, model.config.n_embd)

    _, manual_weights, *_ = manual_self_attention_new(
        normalized, layer, ppes=ppes, compute_diagnostics=False)
    stats = _decompose_layer(
        normalized,
        layer,
        ppes,
        model.config.n_head,
        model.config.n_embd // model.config.n_head,
        (model.config.n_embd // model.config.n_head) ** -0.5,
        normalized.device,
        assert_identity=True,
    )

    expected = manual_weights[
        0, :, normalized.size(1) // 2:, 0].mean(dim=1)
    torch.testing.assert_close(
        torch.from_numpy(stats["attn_full"]),
        expected.detach().cpu(),
        atol=1e-6,
        rtol=1e-5,
    )
