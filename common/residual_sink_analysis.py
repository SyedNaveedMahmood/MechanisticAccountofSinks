# -*- coding: utf-8 -*-
"""Configuration-aware residual-sink decomposition.

The complete experiment harness is retained in
``residual_sink_analysis_legacy``. This module re-exports it and replaces only
``_decompose_layer`` so component softmaxes honor the checkpoint's GPT-2
attention scaling and upcasting settings.
"""

import torch

try:  # package import
    from . import residual_sink_analysis_legacy as _legacy
except ImportError:  # scripts add common/ directly to sys.path
    import residual_sink_analysis_legacy as _legacy

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


def _decompose_layer(normalized, layer, ppes, H, Dh, scale, device,
                     assert_identity=False):
    seq = normalized.size(1)
    h0 = normalized[0]

    attn = layer.attn
    score_scale = scale if getattr(
        attn, "scale_attn_weights", True) else 1.0
    if getattr(attn, "scale_attn_by_inverse_layer_idx", False):
        layer_idx = getattr(attn, "layer_idx", None)
        if layer_idx is None:
            raise ValueError(
                "scale_attn_by_inverse_layer_idx=True requires attention.layer_idx")
        score_scale /= float(layer_idx + 1)
    upcast_attention = bool(
        getattr(attn, "reorder_and_upcast_attn", False))

    qkv_w = attn.c_attn.weight.t()
    qkv_b = attn.c_attn.bias
    wq, wk, _ = qkv_w.chunk(3, dim=0)
    bq, bk, _ = qkv_b.chunk(3, dim=0)

    q_c = (h0 @ wq.t()).view(seq, H, Dh)
    k_c = (h0 @ wk.t()).view(seq, H, Dh)
    bq_h = bq.view(H, Dh)
    bk_h = bk.view(H, Dh)

    T1 = torch.einsum("ihd,jhd->hij", q_c, k_c)
    T3 = torch.einsum("hd,jhd->hj", bq_h, k_c)

    if assert_identity:
        T2 = torch.einsum("ihd,hd->hi", q_c, bk_h)
        T4 = torch.einsum("hd,hd->h", bq_h, bk_h)
        full = T1 + T2.unsqueeze(2) + T3.unsqueeze(1) + T4.view(H, 1, 1)
        q_f = (h0 @ wq.t() + bq).view(seq, H, Dh)
        k_f = (h0 @ wk.t() + bk).view(seq, H, Dh)
        full_ref = torch.einsum("ihd,jhd->hij", q_f, k_f)
        max_err = (full - full_ref).abs().max().item()
        ref_scale = full_ref.abs().max().item()
        tol = 1e-4 * ref_scale + 1e-3
        assert max_err < tol, (
            f"score decomposition identity failed: {max_err} "
            f"(tol {tol:.3g}, ref scale {ref_scale:.3g})")

    mask = torch.tril(torch.ones(
        seq, seq, device=device, dtype=torch.bool))
    eff = T1 + T3.unsqueeze(1)
    delta_full = T3.unsqueeze(1).expand(H, seq, seq)

    def attn0(scores):
        scaled_scores = scores * score_scale
        if upcast_attention:
            scaled_scores = scaled_scores.float()
        scaled_scores = scaled_scores.masked_fill(
            ~mask.unsqueeze(0), float("-inf"))
        return torch.softmax(
            scaled_scores, dim=-1)[:, :, _legacy.SINK_POS]

    sh = slice(seq // 2, seq)
    a_full = attn0(eff)[:, sh].mean(dim=1)
    a_content = attn0(T1)[:, sh].mean(dim=1)
    a_delta = attn0(delta_full)[:, sh].mean(dim=1)

    counts = torch.arange(1, seq + 1, device=device).float()
    eff_mean = eff.masked_fill(
        ~mask.unsqueeze(0), 0.0).sum(dim=-1) / counts
    adv_full = eff[:, :, _legacy.SINK_POS] - eff_mean
    t3_mean = torch.cumsum(T3, dim=-1) / counts
    adv_delta = T3[:, _legacy.SINK_POS:_legacy.SINK_POS + 1] - t3_mean
    share = (adv_delta / (adv_full + 1e-9))[:, sh].mean(dim=1)

    epe_keys = (ppes @ wk.t()).view(ppes.size(0), H, Dh)
    q_n = q_c / (q_c.norm(dim=-1, keepdim=True) + 1e-9)
    ek_n = epe_keys / (epe_keys.norm(dim=-1, keepdim=True) + 1e-9)
    cos = torch.einsum("ihd,phd->iph", q_n[sh], ek_n)
    align_red = cos[:, _legacy.SINK_POS, :].mean(dim=0)
    align_blue = cos[:, _legacy.SINK_POS + 1:, :].mean(dim=(0, 1))

    return {
        "attn_full": a_full.detach().cpu().numpy(),
        "attn_content": a_content.detach().cpu().numpy(),
        "attn_delta": a_delta.detach().cpu().numpy(),
        "share_delta": share.detach().cpu().numpy(),
        "align_red": align_red.detach().cpu().numpy(),
        "align_blue": align_blue.detach().cpu().numpy(),
    }


# The retained harness resolves _decompose_layer through its own module globals.
_legacy._decompose_layer = _decompose_layer


if __name__ == "__main__":
    _legacy.main()
