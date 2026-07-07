# -*- coding: utf-8 -*-
"""residual_sink_analysis.py — E4: Anatomy of the residual attention sink in GPT-2.

The original paper (Ran-Milo, Ofek & Mendel, arXiv:2604.14722) attributes the
first-position attention sink to the ``b_Q – EPE_1 – W_k`` circuit, but its
interventions leave a large, unexplained *residual* sink (44.7% after nullifying
b_Q; 65.2% after zeroing the top-3 W_k columns — §5.3, "Secondary contributors").
This module resolves that residual with a graded, decomposed causal account.

Central hypothesis — the sink is TWO positional pathways sharing one key object
`k_1 ≈ EPE_1 W_k` (the massive-activation coordinates), differing on the query side:

    Pathway A (bias)  : T3 = Δ_1 = b_Q·(EPE_1 W_k)^T          killed by Nullify b_Q
    Pathway B (query) : T1(i,1) = (x_i W_q)·(EPE_1 W_k)^T     the residual; killed by removing EPE_1 from x_1

where the pre-softmax score decomposes (paper §3.1) as
    s_{i→j} = x_i W_q W_k^T x_j^T (T1) + x_i W_q b_K^T (T2) + b_Q W_k^T x_j^T (T3) + b_Q b_K^T (T4),
and softmax-over-targets cancels T2, T4 (constant in j), so the target distribution
is driven by T1 (content) + T3 (source-agnostic shift) only.

Modes
-----
  dose_response   E4.1  BOS-attention vs scale α on three knobs (b_Q, EPE_1 direction,
                        massive coords). The differing α=0 floors are the two-pathway signature.
  decomposition   E4.2  Exact T1/T3 attribution of the position-1 advantage + the
                        query–EPE_1 alignment histogram (Fig. 2 analog for downstream queries).
  combined        E4.3  Stacked interventions (b∧c, b∧i, b∧d, b∧i∧d) localizing the residual.
  surgical        E4.4  First-layer-only MLP skip and position-1-only PE zeroing vs. the
                        paper's coarse all-layer / all-position ablations.
  relocation      E4.5  Attention to position 2 (the Swap-EPE transplant target) — does
                        the sink move, or merely vanish?
  perplexity      E4.6  (optional) LM cross-entropy cost of each pathway ablation.
  all                   dose_response + decomposition + combined + surgical + relocation.

Every mode runs over ``--seeds`` (default 0,1,2; data resamples) and writes per-seed
outputs plus a cross-seed ``aggregate/`` directory (CSVs + figures).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from datasets_loader import (
    sample_benchmark_datasets,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_CUT_LENGTH,
)
from intervention_analysis import (
    get_initial_embeddings,
    run_intervention_loop,
    manual_self_attention_new,
    compute_bos_attention_metric,
    LAYER_RANGE_START,
    LAYER_RANGE_END,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
DEFAULT_SEEDS = [0, 1, 2]
SINK_POS = 0          # first-token sink position (0-indexed; paper's "position 1")
RELOCATION_POS = 1    # swap target (0-indexed; paper's "position 2")


# ═══════════════════════════════════════════════════════════════════════════════
# Model / embedding helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name, device):
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model.to(torch.float32)  # match the paper's tight SEs (fp32)
    return model, tokenizer


def compute_ppes(model, pos_enc):
    """EPE_j = p_j + MLP^(1)(p_j) for every position, shape [seq, hidden]."""
    return pos_enc.clone()[0] + model.transformer.h[0].mlp(pos_enc.clone())[0]


def identify_massive_coords(model, device, n_std=3.0):
    """Coordinates of EPE_1 whose |value| exceeds mean+n_std·std (paper: 138,378,447)."""
    with torch.no_grad():
        pe0 = model.transformer.wpe(torch.tensor([[0]], device=device))
        epe0 = pe0[0, 0] + model.transformer.h[0].mlp(pe0)[0, 0]
    vals = epe0.detach().cpu().numpy()
    a = np.abs(vals)
    mask = a > (a.mean() + n_std * a.std())
    return np.where(mask)[0].tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# modify_mlp_fn factories (all act on the layer-0 MLP output at position 0/1)
# Each mirrors the closure style of intervention_d/e in intervention_analysis.py.
# ═══════════════════════════════════════════════════════════════════════════════

def make_scale_epe_direction(alpha, epe0_hat):
    """Scale the EPE_1-direction component of the layer-0 MLP output at position 0 by α."""
    def _modify(layer_idx, mlp_output):
        if layer_idx == 0:
            comp = torch.dot(mlp_output[0][0], epe0_hat)
            mlp_output[0][0] = mlp_output[0][0] + (alpha - 1.0) * comp * epe0_hat
        return mlp_output
    return _modify


def make_scale_massive_coords(alpha, coords):
    """Scale the massive-activation coordinates of the layer-0 MLP output at position 0 by α."""
    coords = list(coords)
    def _modify(layer_idx, mlp_output):
        if layer_idx == 0:
            mlp_output[0][0, coords] = alpha * mlp_output[0][0, coords]
        return mlp_output
    return _modify


def make_swap_direction(u0_hat, u1_hat):
    """Magnitude-preserving swap of the u-direction between positions 0 and 1 (paper App. B.1)."""
    def _modify(layer_idx, mlp_output):
        if layer_idx == 0:
            a = torch.dot(mlp_output[0][0], u0_hat)
            mlp_output[0][0] = mlp_output[0][0] - a * u0_hat + a * u1_hat
            mlp_output[0][1] = mlp_output[0][1] + a * u0_hat - a * u1_hat
        return mlp_output
    return _modify


def make_zero_layer0_mlp():
    """Skip the MLP block at layer 0 only (surgical variant of the paper's all-layer No-MLP)."""
    def _modify(layer_idx, mlp_output):
        if layer_idx == 0:
            return torch.zeros_like(mlp_output)
        return mlp_output
    return _modify


# ═══════════════════════════════════════════════════════════════════════════════
# Declarative config runner — composes the primitives in intervention_analysis.py
# ═══════════════════════════════════════════════════════════════════════════════

def run_config(model, token_embeddings, pos_enc, *,
               nullify_bq=False, scale_bq=None, wk_zero_coords=None,
               mlp_modify=None, skip_mlp=False, pe_transform=None):
    """Run one (possibly combined) intervention and return per-layer attention weights.

    Parameters mirror the paper's intervention vocabulary and stack freely, so
    combined interventions (E4.3) and dose-response knobs (E4.1) are one-liners.
    """
    pe = pos_enc.clone()
    if pe_transform is not None:
        pe = pe_transform(pe)
    layer_input = token_embeddings.clone() + pe
    ppes = compute_ppes(model, pe)

    attn_kwargs = {}
    if nullify_bq:
        attn_kwargs["intervene_query_bias"] = True
    elif scale_bq is not None:
        attn_kwargs["query_bias_scale"] = scale_bq
    if wk_zero_coords is not None:
        attn_kwargs["fixed_wk_zero_indices"] = list(wk_zero_coords)

    return run_intervention_loop(
        model, layer_input, ppes,
        attn_kwargs=attn_kwargs, modify_mlp_fn=mlp_modify, skip_mlp=skip_mlp,
    )


def _pe_remove_first(pe):
    """(c) Remove First PE: position 0 receives PE[1]."""
    pe[0][0] = pe[0][1].clone()
    return pe


def _pe_zero_first(pe):
    """Surgical: zero the positional embedding at position 0 only."""
    pe[0][0] = torch.zeros_like(pe[0][0])
    return pe


def bos_metric(attn_weights, num_layers, target_pos=SINK_POS):
    return compute_bos_attention_metric(attn_weights, num_layers, "mid", target_pos=target_pos)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-sentence forward that collects the score decomposition + query alignment (E4.2)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_decomposition_and_alignment(model, token_embeddings, pos_enc,
                                        layer_start=LAYER_RANGE_START, layer_end=LAYER_RANGE_END,
                                        assert_identity=False):
    """Advance a baseline manual forward, collecting per-(layer,head) statistics.

    Returns a dict of arrays indexed [n_layers_in_range, num_heads]:
        attn_full, attn_content, attn_delta : softmax attention to position 0 under
            the full effective score (T1+T3), content-only (T1), and delta-only (T3);
            averaged over second-half source positions.
        share_delta : score-space fraction of the position-0 advantage carried by T3.
        align_red   : mean_i cos(content-query_i, EPE_1 W_k)      (pathway-B alignment)
        align_blue  : mean_i cos(content-query_i, EPE_{j>1} W_k)  (control)
    """
    device = token_embeddings.device
    num_layers = len(model.transformer.h)
    layer_end = min(layer_end, num_layers)
    H = model.config.n_head
    hidden = model.config.hidden_size
    Dh = hidden // H
    scale = Dh ** -0.5

    layer_input = token_embeddings.clone() + pos_enc.clone()
    ppes = compute_ppes(model, pos_enc)  # [seq, hidden]

    out = {k: [] for k in
           ("attn_full", "attn_content", "attn_delta", "share_delta", "align_red", "align_blue")}

    for li in range(num_layers):
        layer = model.transformer.h[li]
        normalized = layer.ln_1(layer_input.clone())

        if layer_start <= li < layer_end:
            stats = _decompose_layer(normalized, layer, ppes, H, Dh, scale, device,
                                     assert_identity=assert_identity)
            for k in out:
                out[k].append(stats[k])

        # advance the (baseline) forward pass exactly as run_intervention_loop does
        attn_out, *_ = manual_self_attention_new(normalized, layer, ppes=ppes)
        attn_res = layer_input + attn_out
        mlp_out = layer.mlp(layer.ln_2(attn_res))
        layer_input = attn_res + mlp_out

    return {k: np.stack(v, axis=0) for k, v in out.items()}  # each [n_layers, H]


def _decompose_layer(normalized, layer, ppes, H, Dh, scale, device, assert_identity=False):
    seq = normalized.size(1)
    h0 = normalized[0]  # [seq, hidden]

    attn = layer.attn
    qkv_w = attn.c_attn.weight.t()
    qkv_b = attn.c_attn.bias
    wq, wk, _ = qkv_w.chunk(3, dim=0)
    bq, bk, _ = qkv_b.chunk(3, dim=0)

    # F.linear(x, W) == x @ W.t() (HF Conv1D weights are transposed to [out, in] above),
    # so the content projections must use the transpose to match manual_self_attention_new.
    q_c = (h0 @ wq.t()).view(seq, H, Dh)   # content query (no bias)
    k_c = (h0 @ wk.t()).view(seq, H, Dh)   # content key   (no bias)
    bq_h = bq.view(H, Dh)
    bk_h = bk.view(H, Dh)

    # Score terms (unscaled): T1[h,i,j], T2[h,i], T3[h,j], T4[h]
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
        assert max_err < 1e-3, f"score decomposition identity failed: {max_err}"

    # Causal mask (True where target j <= source i)
    mask = torch.tril(torch.ones(seq, seq, device=device)).bool()
    eff = T1 + T3.unsqueeze(1)                       # [H, seq, seq] effective score (T2,T4 cancel)
    delta_full = T3.unsqueeze(1).expand(H, seq, seq)  # [H, seq, seq]

    def attn0(scores):
        s = (scores * scale).masked_fill(~mask.unsqueeze(0), float("-inf"))
        return torch.softmax(s, dim=-1)[:, :, SINK_POS]  # [H, seq]

    sh = slice(seq // 2, seq)
    a_full = attn0(eff)[:, sh].mean(dim=1)
    a_content = attn0(T1)[:, sh].mean(dim=1)
    a_delta = attn0(delta_full)[:, sh].mean(dim=1)

    # Score-space additive share carried by T3 (delta) of the position-0 advantage
    counts = torch.arange(1, seq + 1, device=device).float()
    eff_mean = eff.masked_fill(~mask.unsqueeze(0), 0.0).sum(dim=-1) / counts   # [H, seq]
    adv_full = eff[:, :, SINK_POS] - eff_mean                                   # [H, seq]
    t3_mean = torch.cumsum(T3, dim=-1) / counts                                 # [H, seq]
    adv_delta = T3[:, SINK_POS:SINK_POS + 1] - t3_mean                          # [H, seq]
    share = (adv_delta / (adv_full + 1e-9))[:, sh].mean(dim=1)                  # [H]

    # Query–EPE alignment: cos(content-query_i, EPE_p W_k), red=p=0 vs blue=p>0
    epe_keys = (ppes @ wk.t()).view(ppes.size(0), H, Dh)     # EPE_j W_k, per head [seq, H, Dh]
    q_n = q_c / (q_c.norm(dim=-1, keepdim=True) + 1e-9)
    ek_n = epe_keys / (epe_keys.norm(dim=-1, keepdim=True) + 1e-9)
    cos = torch.einsum("ihd,phd->iph", q_n[sh], ek_n)         # [n_sh, seq, H]
    align_red = cos[:, SINK_POS, :].mean(dim=0)               # [H]
    align_blue = cos[:, SINK_POS + 1:, :].mean(dim=(0, 1))    # [H]

    return {
        "attn_full": a_full.detach().cpu().numpy(),
        "attn_content": a_content.detach().cpu().numpy(),
        "attn_delta": a_delta.detach().cpu().numpy(),
        "share_delta": share.detach().cpu().numpy(),
        "align_red": align_red.detach().cpu().numpy(),
        "align_blue": align_blue.detach().cpu().numpy(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Perplexity forward-to-logits (E4.6, optional)
# ═══════════════════════════════════════════════════════════════════════════════

def forward_to_logits(model, token_embeddings, pos_enc, *, attn_kwargs=None,
                      mlp_modify=None, skip_mlp=False):
    """Manual forward returning LM logits, so the CE cost of an intervention is measurable."""
    attn_kwargs = attn_kwargs or {}
    layer_input = token_embeddings.clone() + pos_enc.clone()
    ppes = compute_ppes(model, pos_enc)
    for li, layer in enumerate(model.transformer.h):
        normalized = layer.ln_1(layer_input.clone())
        attn_out, *_ = manual_self_attention_new(normalized, layer, ppes=ppes, **attn_kwargs)
        attn_res = layer_input + attn_out
        if skip_mlp:
            layer_input = attn_res
        else:
            mlp_out = layer.mlp(layer.ln_2(attn_res))
            if mlp_modify is not None:
                mlp_out = mlp_modify(li, mlp_out)
            layer_input = attn_res + mlp_out
    hidden = model.transformer.ln_f(layer_input)
    return model.lm_head(hidden)


def sequence_cross_entropy(logits, input_ids):
    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]
    return F.cross_entropy(shift_logits, shift_labels).item()


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset iteration helper
# ═══════════════════════════════════════════════════════════════════════════════

def iter_examples(model, tokenizer, sampled):
    """Yield (dataset_name, input_ids, token_embeddings, pos_enc) for every sampled example."""
    for ds_name, sentences in sampled.items():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
            inputs = inputs.to(model.device)
            pos_enc, token_embeddings = get_initial_embeddings(model, inputs)
            yield ds_name, inputs["input_ids"], token_embeddings, pos_enc


def count_examples(sampled):
    return sum(len(sentences) for sentences in sampled.values())


# ═══════════════════════════════════════════════════════════════════════════════
# E4.1 — Dose-response
# ═══════════════════════════════════════════════════════════════════════════════

def run_dose_response(model, tokenizer, sampled, alphas, massive_coords):
    """BOS-attention vs α for three knobs. Returns a tidy DataFrame (one row per α×knob)."""
    num_layers = len(model.transformer.h)
    knobs = ["scale_bq", "scale_epe_dir", "scale_massive"]
    acc = {(k, a): [] for k in knobs for a in alphas}

    for _ds, _ids, te, pe in tqdm(
        iter_examples(model, tokenizer, sampled),
        total=count_examples(sampled),
        desc="dose_response examples",
    ):
        ppes = compute_ppes(model, pe)
        epe0_hat = ppes[0] / torch.linalg.norm(ppes[0])
        for a in alphas:
            acc[("scale_bq", a)].append(
                bos_metric(run_config(model, te, pe, scale_bq=a), num_layers))
            acc[("scale_epe_dir", a)].append(
                bos_metric(run_config(model, te, pe,
                                      mlp_modify=make_scale_epe_direction(a, epe0_hat)), num_layers))
            acc[("scale_massive", a)].append(
                bos_metric(run_config(model, te, pe,
                                      mlp_modify=make_scale_massive_coords(a, massive_coords)), num_layers))

    rows = []
    for (knob, a), vals in acc.items():
        rows.append({"knob": knob, "alpha": a,
                     "bos_attention": float(np.mean(vals)),
                     "n": len(vals)})
    return pd.DataFrame(rows).sort_values(["knob", "alpha"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# E4.2 — Decomposition + query alignment
# ═══════════════════════════════════════════════════════════════════════════════

def run_decomposition(model, tokenizer, sampled, assert_identity=False):
    """Aggregate the per-(layer,head) decomposition and alignment across all examples."""
    per_example = {k: [] for k in
                   ("attn_full", "attn_content", "attn_delta", "share_delta", "align_red", "align_blue")}
    first = assert_identity
    for _ds, _ids, te, pe in tqdm(
        iter_examples(model, tokenizer, sampled),
        total=count_examples(sampled),
        desc="decomposition examples",
    ):
        stats = collect_decomposition_and_alignment(model, te, pe, assert_identity=first)
        first = False  # assert once is enough
        for k in per_example:
            per_example[k].append(stats[k])
    # each stacked: [n_examples, n_layers, H] → mean over examples → [n_layers, H]
    cell = {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in per_example.items()}
    return cell


# ═══════════════════════════════════════════════════════════════════════════════
# E4.3 / E4.4 — Combined and surgical interventions
# ═══════════════════════════════════════════════════════════════════════════════

def _combined_and_surgical_specs(model, massive_coords):
    """Return {name: builder(te, pe) -> attn_weights}. Builders close over the model."""
    def epe_hats(pe):
        ppes = compute_ppes(model, pe)
        return (ppes[0] / torch.linalg.norm(ppes[0]), ppes[1] / torch.linalg.norm(ppes[1]))

    specs = {}
    # references
    specs["baseline"] = lambda te, pe: run_config(model, te, pe)
    specs["nullify_bq"] = lambda te, pe: run_config(model, te, pe, nullify_bq=True)
    # E4.3 combined
    specs["bq0__remove_first_pe"] = lambda te, pe: run_config(
        model, te, pe, nullify_bq=True, pe_transform=_pe_remove_first)
    specs["bq0__zero_top3_wk"] = lambda te, pe: run_config(
        model, te, pe, nullify_bq=True, wk_zero_coords=massive_coords)
    specs["bq0__swap_epe"] = lambda te, pe: run_config(
        model, te, pe, nullify_bq=True, mlp_modify=make_swap_direction(*epe_hats(pe)))
    specs["bq0__zero_top3_wk__swap_epe"] = lambda te, pe: run_config(
        model, te, pe, nullify_bq=True, wk_zero_coords=massive_coords,
        mlp_modify=make_swap_direction(*epe_hats(pe)))
    # E4.4 surgical vs. coarse
    specs["first_layer_mlp_skip"] = lambda te, pe: run_config(
        model, te, pe, mlp_modify=make_zero_layer0_mlp())
    specs["all_layer_no_mlp"] = lambda te, pe: run_config(model, te, pe, skip_mlp=True)
    specs["pos1_only_pe_zero"] = lambda te, pe: run_config(
        model, te, pe, pe_transform=_pe_zero_first)
    specs["all_pos_no_pe"] = lambda te, pe: run_config(
        model, te, pe, pe_transform=lambda pe_: torch.zeros_like(pe_))
    return specs


def run_intervention_set(model, tokenizer, sampled, names, specs):
    """Run a named set of interventions, returning per-name BOS-attention (% of baseline)."""
    num_layers = len(model.transformer.h)
    acc = {n: [] for n in names}
    for _ds, _ids, te, pe in tqdm(
        iter_examples(model, tokenizer, sampled),
        total=count_examples(sampled),
        desc="intervention examples",
    ):
        for n in names:
            acc[n].append(bos_metric(specs[n](te, pe), num_layers))
    means = {n: float(np.mean(v)) for n, v in acc.items()}
    base = means.get("baseline", 1.0) or 1.0
    return pd.DataFrame([
        {"intervention": n, "bos_attention": means[n], "pct_base": 100.0 * means[n] / base}
        for n in names
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# E4.5 — Relocation
# ═══════════════════════════════════════════════════════════════════════════════

def run_relocation(model, tokenizer, sampled):
    """Attention to position 0 (sink) and position 1 (swap target) under Swap-EPE and b∧d."""
    num_layers = len(model.transformer.h)
    acc = {k: [] for k in ("base_pos0", "base_pos1",
                           "swap_pos0", "swap_pos1", "bq0swap_pos0", "bq0swap_pos1")}
    for _ds, _ids, te, pe in tqdm(
        iter_examples(model, tokenizer, sampled),
        total=count_examples(sampled),
        desc="relocation examples",
    ):
        ppes = compute_ppes(model, pe)
        hats = (ppes[0] / torch.linalg.norm(ppes[0]), ppes[1] / torch.linalg.norm(ppes[1]))
        base = run_config(model, te, pe)
        swap = run_config(model, te, pe, mlp_modify=make_swap_direction(*hats))
        bq0swap = run_config(model, te, pe, nullify_bq=True, mlp_modify=make_swap_direction(*hats))
        acc["base_pos0"].append(bos_metric(base, num_layers, SINK_POS))
        acc["base_pos1"].append(bos_metric(base, num_layers, RELOCATION_POS))
        acc["swap_pos0"].append(bos_metric(swap, num_layers, SINK_POS))
        acc["swap_pos1"].append(bos_metric(swap, num_layers, RELOCATION_POS))
        acc["bq0swap_pos0"].append(bos_metric(bq0swap, num_layers, SINK_POS))
        acc["bq0swap_pos1"].append(bos_metric(bq0swap, num_layers, RELOCATION_POS))
    means = {k: float(np.mean(v)) for k, v in acc.items()}
    means["relocation_ratio_swap"] = (
        (means["swap_pos1"] - means["base_pos1"]) / (means["base_pos0"] + 1e-9))
    return means


# ═══════════════════════════════════════════════════════════════════════════════
# E4.6 — Perplexity (optional)
# ═══════════════════════════════════════════════════════════════════════════════

def run_perplexity(model, tokenizer, sampled, massive_coords):
    """LM cross-entropy under each pathway ablation (and an HF baseline sanity check)."""
    configs = {
        "baseline": dict(),
        "nullify_bq": dict(attn_kwargs={"intervene_query_bias": True}),
        "zero_top3_wk": dict(attn_kwargs={"fixed_wk_zero_indices": list(massive_coords)}),
        "first_layer_mlp_skip": dict(mlp_modify=make_zero_layer0_mlp()),
    }
    acc = {n: [] for n in configs}
    acc_hf = []
    for _ds, ids, te, pe in tqdm(
        iter_examples(model, tokenizer, sampled),
        total=count_examples(sampled),
        desc="perplexity examples",
    ):
        with torch.no_grad():
            hf_logits = model(input_ids=ids).logits
        acc_hf.append(sequence_cross_entropy(hf_logits, ids))
        for n, cfg in configs.items():
            logits = forward_to_logits(model, te, pe, **cfg)
            acc[n].append(sequence_cross_entropy(logits, ids))
    rows = [{"intervention": "hf_reference", "cross_entropy": float(np.mean(acc_hf))}]
    for n in configs:
        rows.append({"intervention": n, "cross_entropy": float(np.mean(acc[n]))})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

KNOB_LABELS = {
    "scale_bq": r"Scale $b_Q$ (pathway A)",
    "scale_epe_dir": r"Scale EPE$_1$ direction (shared)",
    "scale_massive": r"Scale massive coords (shared)",
}


def plot_dose_response(df_mean, df_std, save_path, baseline=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for knob in ["scale_bq", "scale_epe_dir", "scale_massive"]:
        m = df_mean[df_mean["knob"] == knob].sort_values("alpha")
        y = m["bos_attention"].values
        x = m["alpha"].values
        if df_std is not None:
            s = df_std[df_std["knob"] == knob].sort_values("alpha")["bos_attention"].values
            ax.fill_between(x, y - s, y + s, alpha=0.18)
        ax.plot(x, y, marker="o", linewidth=2, label=KNOB_LABELS[knob])
    ax.set_xlabel(r"Scale factor $\alpha$", fontsize=15)
    ax.set_ylabel("BOS attention (pos. 1)", fontsize=15)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.6)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_title("Dose-response: differing α=0 floors reveal two pathways", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_hist(align_red, align_blue, save_path):
    red = np.asarray(align_red).ravel()
    blue = np.asarray(align_blue).ravel()
    allv = np.concatenate([red, blue])
    bins = np.linspace(allv.min() - 0.05, allv.max() + 0.05, 30)
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.hist(blue, bins=bins, density=True, alpha=0.6, color="steelblue",
            edgecolor="black", linewidth=0.8, label=r"Other positions ($j>1$)")
    ax.hist(red, bins=bins, density=True, alpha=0.7, color="tab:red",
            edgecolor="black", linewidth=0.8, label=r"Position 1 (EPE$_1$)")
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel(r"cos(content query $x_iW_q$, EPE$_j W_k$)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_share_heatmap(share_cell, save_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(share_cell, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Head", fontsize=13)
    ax.set_ylabel("Layer (4–11)", fontsize=13)
    ax.set_yticks(range(share_cell.shape[0]))
    ax.set_yticklabels(range(LAYER_RANGE_START + 1, LAYER_RANGE_START + 1 + share_cell.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r"T3 (Δ) share of pos-1 advantage")
    ax.set_title("Where the source-agnostic shift dominates vs. the query pathway", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Seed orchestration + aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _seed_dir(root, seed):
    return root / f"seed_{seed:03d}"


def run_all_modes(model, tokenizer, sampled, modes, alphas, massive_coords, with_perplexity):
    """Run the requested modes for one seed; return a dict of results (DataFrames / arrays)."""
    res = {}
    if "dose_response" in modes:
        print("Running mode: dose_response")
        res["dose_response"] = run_dose_response(model, tokenizer, sampled, alphas, massive_coords)
    if "decomposition" in modes:
        print("Running mode: decomposition")
        res["decomposition"] = run_decomposition(model, tokenizer, sampled, assert_identity=True)
    if "combined" in modes or "surgical" in modes:
        specs = _combined_and_surgical_specs(model, massive_coords)
        if "combined" in modes:
            print("Running mode: combined")
            names = ["baseline", "nullify_bq", "bq0__remove_first_pe", "bq0__zero_top3_wk",
                     "bq0__swap_epe", "bq0__zero_top3_wk__swap_epe"]
            res["combined"] = run_intervention_set(model, tokenizer, sampled, names, specs)
        if "surgical" in modes:
            print("Running mode: surgical")
            names = ["baseline", "first_layer_mlp_skip", "all_layer_no_mlp",
                     "pos1_only_pe_zero", "all_pos_no_pe"]
            res["surgical"] = run_intervention_set(model, tokenizer, sampled, names, specs)
    if "relocation" in modes:
        print("Running mode: relocation")
        res["relocation"] = run_relocation(model, tokenizer, sampled)
    if with_perplexity or "perplexity" in modes:
        print("Running mode: perplexity")
        res["perplexity"] = run_perplexity(model, tokenizer, sampled, massive_coords)
    return res


def save_seed_results(res, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if "dose_response" in res:
        res["dose_response"].to_csv(out_dir / "dose_response.csv", index=False)
    if "decomposition" in res:
        d = res["decomposition"]
        # flat scalar summary + per-cell arrays
        summary = {k: float(np.mean(v)) for k, v in d.items()}
        (out_dir / "decomposition_summary.json").write_text(json.dumps(summary, indent=2))
        np.savez(out_dir / "decomposition_cells.npz", **d)
    for key in ("combined", "surgical", "perplexity"):
        if key in res:
            res[key].to_csv(out_dir / f"{key}.csv", index=False)
    if "relocation" in res:
        (out_dir / "relocation.json").write_text(json.dumps(res["relocation"], indent=2))


def aggregate_and_plot(root, seeds, modes, alphas, with_perplexity):
    agg = root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)

    # ---- Dose-response ----
    if "dose_response" in modes:
        frames = [pd.read_csv(_seed_dir(root, s) / "dose_response.csv").assign(seed=s)
                  for s in seeds]
        alld = pd.concat(frames, ignore_index=True)
        g = alld.groupby(["knob", "alpha"])["bos_attention"]
        df_mean = g.mean().reset_index()
        df_std = g.std().fillna(0.0).reset_index()
        df_mean.merge(df_std, on=["knob", "alpha"], suffixes=("_mean", "_std")).to_csv(
            agg / "dose_response_summary.csv", index=False)
        plot_dose_response(df_mean, df_std, agg / "dose_response.png")
        _report_monotonicity(df_mean, agg)

    # ---- Decomposition ----
    if "decomposition" in modes:
        cells = [dict(np.load(_seed_dir(root, s) / "decomposition_cells.npz")) for s in seeds]
        keys = cells[0].keys()
        mean_cell = {k: np.mean([c[k] for c in cells], axis=0) for k in keys}
        plot_alignment_hist(mean_cell["align_red"], mean_cell["align_blue"],
                            agg / "query_alignment_hist.png")
        plot_share_heatmap(mean_cell["share_delta"], agg / "delta_share_heatmap.png")
        summary = _decomposition_summary(cells)
        summary.to_csv(agg / "decomposition_summary.csv", index=False)

    # ---- Combined / surgical / perplexity tables ----
    for key in ("combined", "surgical", "perplexity"):
        if key in modes or (key == "perplexity" and with_perplexity):
            paths = [_seed_dir(root, s) / f"{key}.csv" for s in seeds]
            if not all(p.exists() for p in paths):
                continue
            frames = [pd.read_csv(p).assign(seed=s) for p, s in zip(paths, seeds)]
            allc = pd.concat(frames, ignore_index=True)
            valcol = "cross_entropy" if key == "perplexity" else "pct_base"
            grp = allc.groupby("intervention")
            summ = grp.agg(
                mean=(valcol, "mean"), std=(valcol, "std"),
                bos_mean=("bos_attention", "mean") if key != "perplexity" else (valcol, "mean"),
            ).reset_index()
            summ.to_csv(agg / f"{key}_summary.csv", index=False)

    # ---- Relocation ----
    if "relocation" in modes:
        recs = [json.loads((_seed_dir(root, s) / "relocation.json").read_text()) for s in seeds]
        rdf = pd.DataFrame(recs)
        pd.DataFrame({"metric": rdf.columns,
                      "mean": rdf.mean().values,
                      "std": rdf.std().fillna(0.0).values}).to_csv(
            agg / "relocation_summary.csv", index=False)

    return agg


def _report_monotonicity(df_mean, agg):
    rows = []
    for knob in df_mean["knob"].unique():
        m = df_mean[df_mean["knob"] == knob].sort_values("alpha")
        rho, p = scipy_stats.spearmanr(m["alpha"], m["bos_attention"])
        floor = float(m[m["alpha"] == m["alpha"].min()]["bos_attention"].iloc[0])
        rows.append({"knob": knob, "spearman_rho": rho, "p_value": p, "alpha0_floor": floor})
    pd.DataFrame(rows).to_csv(agg / "dose_response_monotonicity.csv", index=False)


def _decomposition_summary(cells):
    def flat(key):
        return np.concatenate([c[key].ravel() for c in cells])
    red, blue = flat("align_red"), flat("align_blue")
    tstat, pval = scipy_stats.mannwhitneyu(red, blue, alternative="greater")
    d = (red.mean() - blue.mean()) / (np.sqrt((red.std() ** 2 + blue.std() ** 2) / 2) + 1e-9)
    return pd.DataFrame([
        {"quantity": "attn_full", "mean": flat("attn_full").mean()},
        {"quantity": "attn_content_pathwayB", "mean": flat("attn_content").mean()},
        {"quantity": "attn_delta_pathwayA", "mean": flat("attn_delta").mean()},
        {"quantity": "delta_share_mean", "mean": flat("share_delta").mean()},
        {"quantity": "align_red_mean(query·EPE1)", "mean": red.mean()},
        {"quantity": "align_blue_mean(control)", "mean": blue.mean()},
        {"quantity": "align_cohens_d", "mean": d},
        {"quantity": "align_mannwhitney_p", "mean": pval},
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

ALL_MODES = ["dose_response", "decomposition", "combined", "surgical", "relocation"]


def main():
    parser = argparse.ArgumentParser(
        description="E4: Anatomy of the residual attention sink in GPT-2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", default="all",
                        choices=ALL_MODES + ["perplexity", "all"],
                        help="Which analysis to run (default: all = the five core analyses).")
    parser.add_argument("--model-name", "--model", dest="model_name", default="gpt2")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--experiment-name", default=None,
                        help="Subdir under --output-dir. Default: residual_sink_<model tag>.")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated seeds (data resamples). Default: 0,1,2")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--cut-length", type=int, default=DEFAULT_CUT_LENGTH)
    parser.add_argument("--alphas", default=None,
                        help="Comma-separated dose-response scale factors. Default: 0..1.5 step .25")
    parser.add_argument("--with-perplexity", action="store_true",
                        help="Also compute the optional E4.6 LM-loss table.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation; only aggregate + plot existing per-seed outputs.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    alphas = ([float(a) for a in args.alphas.split(",")] if args.alphas else DEFAULT_ALPHAS)
    modes = ALL_MODES if args.mode == "all" else [args.mode]

    tag = args.model_name.replace("/", "_")
    exp = args.experiment_name or f"residual_sink_{tag}"
    root = Path(args.output_dir) / exp
    root.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {args.model_name} on {device} ...")
        model, tokenizer = load_model(args.model_name, device)
        massive_coords = identify_massive_coords(model, device)
        print(f"Massive-activation coordinates of EPE_1: {massive_coords}")

        for seed in seeds:
            print(f"\n{'='*70}\nSeed {seed}: sampling + running modes {modes}\n{'='*70}")
            sampled, manifest = sample_benchmark_datasets(
                tokenizer, sample_size=args.sample_size,
                cut_length=args.cut_length, seed=seed)
            out_dir = _seed_dir(root, seed)
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(manifest).to_csv(out_dir / "sample_manifest.csv", index=False)
            res = run_all_modes(model, tokenizer, sampled, modes, alphas,
                                massive_coords, args.with_perplexity)
            save_seed_results(res, out_dir)
            print(f"Seed {seed} outputs → {out_dir}")

    agg = aggregate_and_plot(root, seeds, modes, alphas, args.with_perplexity)
    print(f"\nDone. Aggregated results + figures → {agg}")


if __name__ == "__main__":
    main()
