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
  dose_response   E4.1  BOS-attention vs scale α on four knobs: b_Q (pathway A); p_1 deletion
                        (scale_pe — empirically the sink SURVIVES deletion, ~85-110% at α=0);
                        p_1 → p_2 interpolation (interp_pe — positional-identity replacement,
                        grades Remove-First-PE, both pathways collapse to ~3% at α=0); and the
                        top-k massive W_k columns (coordinate channel, graded every layer;
                        k = len(massive_coords), recorded per run). The differing α=0 floors are
                        the pathway signature. NOTE: the knobs act at the input (p_1) or in every
                        layer (W_k), NOT on the layer-0 MLP output only — a layer-0-only edit is
                        inert because the massive activations at position 0 are re-established by
                        the intermediate layers (paper fn. 10) before the metric window.
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
import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
import transformers

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
    compute_band,
    LAYER_RANGE_START,
    LAYER_RANGE_END,
)
from nnsight_engine import (
    ARCH_SPECS,
    GPT2TracePlan,
    GPT2_TRACE_REGISTRY_VERSION,
    NNsightEngine,
    load_nnsight_model,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
DEFAULT_SEEDS = [0, 1, 2]
SINK_POS = 0          # first-token sink position (0-indexed; paper's "position 1")
RELOCATION_POS = 1    # swap target (0-indexed; paper's "position 2")
E4_REGISTRY_VERSION = "e4-residual-spec-v1"


@dataclass(frozen=True)
class ResidualInterventionSpec:
    """One E4 intervention definition shared by manual and NNsight executors."""

    key: str
    query_bias_scale: float = 1.0
    token_edit: str | None = None
    position_edit: str | None = None
    position_alpha: float = 1.0
    mlp_edit: str | None = None
    wk_kind: str | None = None
    wk_scale: float = 1.0


def _residual_plan(spec: ResidualInterventionSpec,
                   massive_coords: Sequence[int]) -> GPT2TracePlan:
    coords = tuple(int(c) for c in massive_coords) if spec.wk_kind == "massive" else ()
    return GPT2TracePlan(
        key=spec.key,
        token_edit=spec.token_edit,
        position_edit=spec.position_edit,
        position_alpha=spec.position_alpha,
        query_bias_scale=spec.query_bias_scale,
        mlp_edit=spec.mlp_edit,
        wk_coords=coords,
        wk_scale=spec.wk_scale,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Model / embedding helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name, device, dtype=torch.float32):
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model.to(dtype)  # default fp32 to match the paper's SEs; smaller dtype for large XL runs
    return model, tokenizer


def load_model_for_engine(model_name, device, dtype=torch.float32, engine="manual",
                          revision=None, local_files_only=False):
    """Load one GPT-2 checkpoint for the requested engine without fallback."""
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=local_files_only)
    if engine == "manual":
        kwargs = {"attn_implementation": "eager", "revision": revision,
                  "local_files_only": local_files_only}
        model = GPT2LMHeadModel.from_pretrained(model_name, **kwargs)
        model.to(device)
        model.eval()
        model.to(dtype)
        return model, tokenizer, None
    if engine != "nnsight":
        raise ValueError(f"Unknown execution engine: {engine!r}")
    try:
        lm = load_nnsight_model(
            ARCH_SPECS["gpt2"], model_name, dtype=dtype, tokenizer=tokenizer,
            device=device, revision=revision, local_files_only=local_files_only)
        nn_engine = NNsightEngine(lm, ARCH_SPECS["gpt2"])
    except Exception as exc:
        raise RuntimeError(
            f"NNsight initialization failed for {model_name!r}; the manual engine was not run: {exc}"
        ) from exc
    return lm._model, tokenizer, nn_engine


def engine_provenance(engine, model, *, model_name, dtype, device, band,
                      nn_engine=None, revision=None):
    """Stable E3/E4 provenance block used by configs and cache validation."""
    if engine == "nnsight":
        if nn_engine is None:
            raise ValueError("NNsight provenance requires an initialized engine")
        return nn_engine.engine_info(
            model_name=model_name, revision=revision, dtype=dtype, device=device,
            band=band, registry_version=E4_REGISTRY_VERSION)
    return {
        "name": "manual",
        "nnsight_version": None,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_name": model_name,
        "model_revision": revision or "main",
        "dtype": dtype,
        "device": str(device),
        "remote": False,
        "execution_location": "local",
        "attn_implementation": "manual_reimplementation",
        "attention_probability_source": "common.intervention_analysis.manual_self_attention_new",
        "layer_band": [int(band[0]), int(band[1])],
        "intervention_registry_version": E4_REGISTRY_VERSION,
    }


def compute_ppes(model, pos_enc):
    """EPE_j = p_j + MLP^(1)(p_j) for every position, shape [seq, hidden]."""
    return pos_enc.clone()[0] + model.transformer.h[0].mlp(pos_enc.clone())[0]


def identify_massive_coords(model, device, n_std=3.0, min_coords=3):
    """Coordinates of EPE_1 whose |value| exceeds mean+n_std·std (paper: 138,378,447).

    Falls back to the ``min_coords`` largest-|EPE_1| dimensions if fewer than
    ``min_coords`` exceed the threshold, so the massive-coordinate knobs are
    well-defined on every checkpoint (matches the paper's top-3 intervention).
    """
    with torch.no_grad():
        pe0 = model.transformer.wpe(torch.tensor([[0]], device=device))
        epe0 = pe0[0, 0] + model.transformer.h[0].mlp(pe0)[0, 0]
    a = np.abs(epe0.detach().cpu().float().numpy())
    coords = np.where(a > (a.mean() + n_std * a.std()))[0].tolist()
    if len(coords) < min_coords:
        coords = np.argsort(a)[-min_coords:][::-1].tolist()
    return coords


# ═══════════════════════════════════════════════════════════════════════════════
# modify_mlp_fn factories (all act on the layer-0 MLP output at position 0/1)
# Each mirrors the closure style of intervention_d/e in intervention_analysis.py.
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE: earlier versions grafted the EPE_1 / massive-coordinate dose-response onto the
# *layer-0 MLP output* (via modify_mlp_fn). Empirically those knobs were inert (flat BOS
# attention across alpha) because the massive activations at position 0 are re-established
# by the intermediate layers (1-3) before the metric window (layers 4-11) — the same
# effect the paper notes in fn. 10. The dose-response therefore drives these components at
# an *effective* locus instead: p_1 at the input (run_config pe_transform=_pe_scale_first)
# and the top-3 W_k columns in every layer (run_config wk_scale_coords/wk_scale).


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
               wk_scale_coords=None, wk_scale=1.0,
               mlp_modify=None, skip_mlp=False, pe_transform=None,
               te_transform=None):
    """Run one (possibly combined) intervention and return per-layer attention weights.

    Parameters mirror the paper's intervention vocabulary and stack freely, so
    combined interventions (E4.3) and dose-response knobs (E4.1) are one-liners.
    """
    te = token_embeddings.clone()
    if te_transform is not None:
        te = te_transform(te)
    pe = pos_enc.clone()
    if pe_transform is not None:
        pe = pe_transform(pe)
    layer_input = te + pe
    # EPEs feed only legacy diagnostics in run_intervention_loop; those are
    # disabled for every composable dataset run.
    ppes = None

    attn_kwargs = {}
    if nullify_bq:
        attn_kwargs["intervene_query_bias"] = True
    elif scale_bq is not None:
        attn_kwargs["query_bias_scale"] = scale_bq
    if wk_zero_coords is not None:
        attn_kwargs["fixed_wk_zero_indices"] = list(wk_zero_coords)
    if wk_scale_coords is not None:
        attn_kwargs["wk_scale_indices"] = list(wk_scale_coords)
        attn_kwargs["wk_scale"] = wk_scale

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


def _pe_scale_first(alpha):
    """Input-level p_1 *deletion* knob: scale the position-0 PE (p_1) by alpha.

    alpha=1 is baseline; alpha=0 zeroes the positional signal at position 0.
    Empirically (GPT-2 small/medium) this does NOT collapse the sink — deleting p_1
    leaves position 0 positionally *distinct* and the sink survives (~85-110% of
    baseline). Kept as the deletion-invariance finding; the graded both-pathways
    collapse is measured by :func:`_pe_interp_first` instead.
    """
    def _t(pe):
        pe[0][0] = alpha * pe[0][0]
        return pe
    return _t


def _pe_interp_first(alpha):
    """Positional-identity *interpolation* knob: pe[0] = alpha·p_1 + (1−alpha)·p_2.

    Grades the paper's Remove-First-PE intervention: alpha=1 is baseline (position 0
    keeps p_1); alpha=0 gives position 0 exactly p_2 (== intervention (c), ~3% floor,
    both pathways collapse since EPE_1 never forms and position 0 is relabelled as
    position 2). Unlike :func:`_pe_scale_first`, this replaces positional identity
    rather than deleting it, which is what actually kills the sink.
    """
    def _t(pe):
        pe[0][0] = alpha * pe[0][0] + (1.0 - alpha) * pe[0][1]
        return pe
    return _t


def bos_metric(attn_weights, num_layers, target_pos=SINK_POS, band=None):
    ls, le = band if band is not None else (None, None)
    return compute_bos_attention_metric(attn_weights, num_layers, "mid",
                                        target_pos=target_pos, layer_start=ls, layer_end=le)


def _position_transform_for_spec(spec: ResidualInterventionSpec):
    if spec.position_edit == "remove_first":
        return _pe_remove_first
    if spec.position_edit == "zero_first":
        return _pe_zero_first
    if spec.position_edit == "zero_all":
        return lambda pe: torch.zeros_like(pe)
    if spec.position_edit == "scale_first":
        return _pe_scale_first(spec.position_alpha)
    if spec.position_edit == "interp_first":
        return _pe_interp_first(spec.position_alpha)
    return None


class ResidualExecutor:
    """E4 execution seam with identical declarative specs for both engines."""

    def __init__(self, model, engine="manual", nn_engine=None, massive_coords=()):
        self.model = model
        self.engine = engine
        self.nn_engine = nn_engine
        self.massive_coords = tuple(int(c) for c in massive_coords)
        if engine == "nnsight" and nn_engine is None:
            raise ValueError("engine='nnsight' requires an initialized NNsightEngine")

    def _swap_dirs(self, spec: ResidualInterventionSpec, pe=None):
        if spec.mlp_edit not in {"swap_epe", "swap_pe"}:
            return None
        with torch.no_grad():
            if pe is None:
                device = next(self.model.parameters()).device
                positions = torch.arange(2, device=device).unsqueeze(0)
                pe = self.model.transformer.wpe(positions)
            transform = _position_transform_for_spec(spec)
            pe_work = transform(pe.clone()) if transform else pe.clone()
            vectors = (compute_ppes(self.model, pe_work)
                       if spec.mlp_edit == "swap_epe" else pe_work[0])
            u0 = vectors[0] / torch.linalg.vector_norm(vectors[0]).clamp_min(1e-9)
            u1 = vectors[1] / torch.linalg.vector_norm(vectors[1]).clamp_min(1e-9)
        return u0, u1

    def _manual_kwargs(self, spec: ResidualInterventionSpec, pe):
        attn_kwargs = {"query_bias_scale": spec.query_bias_scale}
        if spec.wk_kind == "massive":
            if spec.wk_scale == 0.0:
                attn_kwargs["fixed_wk_zero_indices"] = list(self.massive_coords)
            elif spec.wk_scale != 1.0:
                attn_kwargs["wk_scale_indices"] = list(self.massive_coords)
                attn_kwargs["wk_scale"] = spec.wk_scale
        mlp_modify = None
        if spec.mlp_edit in {"swap_epe", "swap_pe"}:
            mlp_modify = make_swap_direction(*self._swap_dirs(spec, pe))
        elif spec.mlp_edit == "zero_first":
            mlp_modify = make_zero_layer0_mlp()
        te_transform = None
        if spec.token_edit == "zero_first":
            def te_transform(te):
                te[0, 0] = 0
                return te
        return {
            "attn_kwargs": attn_kwargs,
            "mlp_modify": mlp_modify,
            "skip_mlp": spec.mlp_edit == "zero_all",
            "pe_transform": _position_transform_for_spec(spec),
            "te_transform": te_transform,
        }

    def prepare(self, input_ids):
        """Cache manual embeddings once per example; NNsight needs no pre-forward."""
        if self.engine != "manual":
            return None
        return get_initial_embeddings(self.model, {"input_ids": input_ids})

    def execute(self, input_ids, spec: ResidualInterventionSpec, band, *,
                attention="targets", target_positions=(SINK_POS,),
                capture_qk=False, capture_block_outputs=(), capture_logits=False,
                prepared=None):
        """Execute ``spec`` once and return a common result dictionary."""
        if self.engine == "manual":
            pe, te = prepared if prepared is not None else self.prepare(input_ids)
            kwargs = self._manual_kwargs(spec, pe)
            result = {"attention": [], "target_attention": {}, "pre_ln": [], "qk": [],
                      "block_outputs": {}, "logits": None, "band": band,
                      "plan_key": spec.key, "attention_summary": {}}
            if attention != "none":
                weights = run_config(
                    self.model, te, pe,
                    scale_bq=spec.query_bias_scale,
                    wk_zero_coords=(self.massive_coords if spec.wk_kind == "massive"
                                    and spec.wk_scale == 0.0 else None),
                    wk_scale_coords=(self.massive_coords if spec.wk_kind == "massive"
                                     and spec.wk_scale not in (0.0, 1.0) else None),
                    wk_scale=spec.wk_scale,
                    mlp_modify=kwargs["mlp_modify"], skip_mlp=kwargs["skip_mlp"],
                    pe_transform=kwargs["pe_transform"], te_transform=kwargs["te_transform"])
                selected = [weights[i] for i in range(band[0], band[1])]
                if attention == "full":
                    result["attention"] = selected
                start = input_ids.shape[-1] // 2
                result["target_attention"] = {
                    int(pos): torch.stack([
                        layer[:, start:, int(pos)].mean(dim=1).detach().float().cpu()
                        for layer in selected])
                    for pos in target_positions
                }
            if capture_logits:
                result["logits"] = forward_to_logits(
                    self.model, te, pe, attn_kwargs=kwargs["attn_kwargs"],
                    mlp_modify=kwargs["mlp_modify"], skip_mlp=kwargs["skip_mlp"],
                    pe_transform=kwargs["pe_transform"], te_transform=kwargs["te_transform"])
            return result

        plan = _residual_plan(spec, self.massive_coords)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        try:
            return self.nn_engine.run_gpt2_trace(
                plan, inputs, band=band, attention=attention,
                target_positions=target_positions, capture_qk=capture_qk,
                capture_block_outputs=capture_block_outputs,
                capture_logits=capture_logits, swap_dirs=self._swap_dirs(spec))
        except Exception as exc:
            raise RuntimeError(
                f"NNsight execution failed for E4 intervention {spec.key!r}; "
                f"the manual engine was not run: {exc}") from exc


def target_metric(result, target_pos=SINK_POS):
    values = result["target_attention"].get(int(target_pos))
    if values is None or values.numel() == 0:
        raise ValueError(f"Trace did not capture target position {target_pos}")
    return float(values.mean())


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
        attn_out, *_ = manual_self_attention_new(normalized, layer, ppes=ppes,
                                                 compute_diagnostics=False)
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
        # Relative tolerance: once the massive-activation circuit forms, coordinate values reach
        # the hundreds and the pre-softmax scores reach ~1e4, so a fixed 1e-3 *absolute* bound
        # false-positives on trained checkpoints even though the identity holds to fp32 precision
        # (observed relative error ~1e-7 = a single ulp). Scale by the score magnitude, keeping a
        # 1e-3 absolute floor for the small-score (early-training / random-init) regime.
        ref_scale = full_ref.abs().max().item()
        tol = 1e-4 * ref_scale + 1e-3
        assert max_err < tol, (
            f"score decomposition identity failed: {max_err} (tol {tol:.3g}, ref scale {ref_scale:.3g})")

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


def collect_traced_decomposition_and_alignment(model, trace_result,
                                                assert_identity=False):
    """E4 decomposition from actual NNsight-captured pre-LN and Q/K projections.

    The traced Hugging Face attention probabilities supply ``attn_full``.  T1/T2/T3/T4,
    the component-only softmaxes, and the EPE-key alignment remain the experiment's
    mathematical utilities, but their activation-dependent inputs are the real forward's
    pre-LN states and fused projection outputs.  The identity is checked separately for
    every selected layer and head.
    """
    band = tuple(trace_result["band"])
    if len(trace_result["pre_ln"]) != band[1] - band[0] or \
            len(trace_result["qk"]) != band[1] - band[0]:
        raise ValueError("Trace result lacks the selected band's pre-LN/QK captures")
    full_targets = trace_result["target_attention"].get(SINK_POS)
    if full_targets is None or len(full_targets) != band[1] - band[0]:
        raise ValueError("Trace result lacks per-head position-zero attention")

    H = model.config.n_head
    hidden = model.config.hidden_size
    Dh = hidden // H
    device = next(model.parameters()).device
    with torch.no_grad():
        positions = torch.arange(trace_result["pre_ln"][0].shape[0], device=device).unsqueeze(0)
        pe = model.transformer.wpe(positions)
        ppes = compute_ppes(model, pe).detach().float().cpu()

    out = {key: [] for key in
           ("attn_full", "attn_content", "attn_delta", "share_delta",
            "align_red", "align_blue")}
    for offset, li in enumerate(range(band[0], band[1])):
        normalized = trace_result["pre_ln"][offset].float()
        projected = trace_result["qk"][offset].float()
        seq = normalized.shape[0]
        layer = model.transformer.h[li]
        qkv_w = layer.attn.c_attn.weight.detach().t().float().cpu()
        qkv_b = layer.attn.c_attn.bias.detach().float().cpu()
        _wq, wk, _wv = qkv_w.chunk(3, dim=0)
        bq, bk, _bv = qkv_b.chunk(3, dim=0)

        if assert_identity:
            projected_from_pre_ln = F.linear(
                normalized, qkv_w[:2 * hidden], qkv_b[:2 * hidden])
            projection_error = float((projected_from_pre_ln - projected).abs().max())
            projection_scale = float(projected.abs().max())
            projection_tol = 1e-3 + 1e-4 * projection_scale
            if projection_error >= projection_tol:
                raise AssertionError(
                    f"captured Q/K projection mismatch at layer {li}: "
                    f"error={projection_error:.6g}, tolerance={projection_tol:.6g}")

        q_full = projected[:, :hidden].view(seq, H, Dh)
        k_full = projected[:, hidden:2 * hidden].view(seq, H, Dh)
        q_c = q_full - bq.view(1, H, Dh)
        k_c = k_full - bk.view(1, H, Dh)
        bq_h = bq.view(H, Dh)
        bk_h = bk.view(H, Dh)

        T1 = torch.einsum("ihd,jhd->hij", q_c, k_c)
        T2 = torch.einsum("ihd,hd->hi", q_c, bk_h)
        T3 = torch.einsum("hd,jhd->hj", bq_h, k_c)
        T4 = torch.einsum("hd,hd->h", bq_h, bk_h)
        reconstructed = T1 + T2.unsqueeze(2) + T3.unsqueeze(1) + T4.view(H, 1, 1)
        reference = torch.einsum("ihd,jhd->hij", q_full, k_full)
        if assert_identity:
            per_head_error = (reconstructed - reference).abs().flatten(1).max(dim=1).values
            per_head_scale = reference.abs().flatten(1).max(dim=1).values
            tolerance = 1e-3 + 1e-4 * per_head_scale
            failed = torch.where(per_head_error >= tolerance)[0]
            if failed.numel():
                head = int(failed[0])
                raise AssertionError(
                    f"score decomposition identity failed at layer {li}, head {head}: "
                    f"error={float(per_head_error[head]):.6g}, "
                    f"tolerance={float(tolerance[head]):.6g}")

        score_scale = 1.0
        if getattr(layer.attn, "scale_attn_weights", True):
            score_scale /= np.sqrt(Dh)
        if getattr(layer.attn, "scale_attn_by_inverse_layer_idx", False):
            score_scale /= (li + 1)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))

        def target_attention(scores):
            masked = (scores * score_scale).masked_fill(~mask.unsqueeze(0), float("-inf"))
            return torch.softmax(masked, dim=-1)[:, seq // 2:, SINK_POS].mean(dim=1)

        effective = T1 + T3.unsqueeze(1)
        delta_full = T3.unsqueeze(1).expand(H, seq, seq)
        a_content = target_attention(T1)
        a_delta = target_attention(delta_full)
        counts = torch.arange(1, seq + 1).float()
        eff_mean = effective.masked_fill(~mask.unsqueeze(0), 0.0).sum(dim=-1) / counts
        adv_full = effective[:, :, SINK_POS] - eff_mean
        t3_mean = torch.cumsum(T3, dim=-1) / counts
        adv_delta = T3[:, SINK_POS:SINK_POS + 1] - t3_mean
        share = (adv_delta / (adv_full + 1e-9))[:, seq // 2:].mean(dim=1)

        epe_keys = (ppes @ wk.t()).view(seq, H, Dh)
        q_norm = q_c / q_c.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        key_norm = epe_keys / epe_keys.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cosine = torch.einsum("ihd,phd->iph", q_norm[seq // 2:], key_norm)

        out["attn_full"].append(full_targets[offset].numpy())
        out["attn_content"].append(a_content.numpy())
        out["attn_delta"].append(a_delta.numpy())
        out["share_delta"].append(share.numpy())
        out["align_red"].append(cosine[:, SINK_POS, :].mean(dim=0).numpy())
        out["align_blue"].append(cosine[:, SINK_POS + 1:, :].mean(dim=(0, 1)).numpy())
    return {key: np.stack(values, axis=0) for key, values in out.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Perplexity forward-to-logits (E4.6, optional)
# ═══════════════════════════════════════════════════════════════════════════════

def forward_to_logits(model, token_embeddings, pos_enc, *, attn_kwargs=None,
                      mlp_modify=None, skip_mlp=False, pe_transform=None,
                      te_transform=None):
    """Manual forward returning LM logits, so the CE cost of an intervention is measurable."""
    attn_kwargs = attn_kwargs or {}
    te = token_embeddings.clone()
    pe = pos_enc.clone()
    if te_transform is not None:
        te = te_transform(te)
    if pe_transform is not None:
        pe = pe_transform(pe)
    layer_input = te + pe
    # EPEs feed only legacy diagnostics in manual attention, disabled here.
    ppes = None
    for li, layer in enumerate(model.transformer.h):
        normalized = layer.ln_1(layer_input.clone())
        attn_out, *_ = manual_self_attention_new(normalized, layer, ppes=ppes,
                                                 compute_diagnostics=False, **attn_kwargs)
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
    for ds_name, input_ids in iter_token_ids(model, tokenizer, sampled):
        pos_enc, token_embeddings = get_initial_embeddings(model, {"input_ids": input_ids})
        yield ds_name, input_ids, token_embeddings, pos_enc


def iter_token_ids(model, tokenizer, sampled):
    """Yield the exact token IDs without running a preliminary model forward."""
    for ds_name, sentences in sampled.items():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
            inputs = inputs.to(model.device)
            yield ds_name, inputs["input_ids"]


def count_examples(sampled):
    return sum(len(sentences) for sentences in sampled.values())


# ═══════════════════════════════════════════════════════════════════════════════
# E4.1 — Dose-response
# ═══════════════════════════════════════════════════════════════════════════════

def _dose_specs(alphas):
    specs = {}
    for alpha in alphas:
        specs[("scale_bq", alpha)] = ResidualInterventionSpec(
            f"scale_bq_{alpha:g}", query_bias_scale=float(alpha))
        specs[("scale_pe", alpha)] = ResidualInterventionSpec(
            f"scale_pe_{alpha:g}", position_edit="scale_first",
            position_alpha=float(alpha))
        specs[("interp_pe", alpha)] = ResidualInterventionSpec(
            f"interp_pe_{alpha:g}", position_edit="interp_first",
            position_alpha=float(alpha))
        specs[("scale_wk_massive", alpha)] = ResidualInterventionSpec(
            f"scale_wk_massive_{alpha:g}", wk_kind="massive", wk_scale=float(alpha))
    return specs


def run_dose_response(model, tokenizer, sampled, alphas, massive_coords, band=None,
                      executor=None):
    """BOS-attention vs α for three knobs. Returns a tidy DataFrame (one row per α×knob).

    Knobs are driven at an *effective* locus (see the module header on why layer-0-only
    edits wash out):
      scale_bq         : scale the query bias b_Q             — pathway A; floors at the residual
      scale_pe         : scale (delete) the position-0 PE p_1 — deletion-invariance control; sink survives
      interp_pe        : interpolate p_1 → p_2 at position 0  — positional-identity replacement; grades
                         Remove-First-PE, both pathways collapse (~3% floor at alpha=0)
      scale_wk_massive : scale the top-k massive W_k columns (every layer) — coordinate channel;
                         grades Zero-Top-k-Wk (k = len(massive_coords), recorded in run_config.json)
    """
    num_layers = len(model.transformer.h)
    knobs = ["scale_bq", "scale_pe", "interp_pe", "scale_wk_massive"]
    acc = {(k, a): [] for k in knobs for a in alphas}
    specs = _dose_specs(alphas)
    executor = executor or ResidualExecutor(model, "manual", massive_coords=massive_coords)

    for _ds, ids in tqdm(
        iter_token_ids(model, tokenizer, sampled),
        total=count_examples(sampled),
        desc="dose_response examples",
    ):
        prepared = executor.prepare(ids)
        for key, spec in specs.items():
            result = executor.execute(ids, spec, band, attention="targets",
                                      target_positions=(SINK_POS,), prepared=prepared)
            acc[key].append(target_metric(result, SINK_POS))

    rows = []
    for (knob, a), vals in acc.items():
        rows.append({"knob": knob, "alpha": a,
                     "bos_attention": float(np.mean(vals)),
                     "n": len(vals)})
    return pd.DataFrame(rows).sort_values(["knob", "alpha"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# E4.2 — Decomposition + query alignment
# ═══════════════════════════════════════════════════════════════════════════════

def run_decomposition(model, tokenizer, sampled, assert_identity=False, band=None,
                      executor=None):
    """Aggregate the per-(layer,head) decomposition and alignment across all examples."""
    per_example = {k: [] for k in
                   ("attn_full", "attn_content", "attn_delta", "share_delta", "align_red", "align_blue")}
    ls, le = band if band is not None else (LAYER_RANGE_START, LAYER_RANGE_END)
    first = assert_identity
    executor = executor or ResidualExecutor(model, "manual")
    baseline = ResidualInterventionSpec("baseline")
    for _ds, ids in tqdm(
        iter_token_ids(model, tokenizer, sampled),
        total=count_examples(sampled),
        desc="decomposition examples",
    ):
        if executor.engine == "manual":
            pe, te = executor.prepare(ids)
            stats = collect_decomposition_and_alignment(
                model, te, pe, layer_start=ls, layer_end=le, assert_identity=first)
        else:
            traced = executor.execute(
                ids, baseline, (ls, le), attention="targets",
                target_positions=(SINK_POS,), capture_qk=True)
            stats = collect_traced_decomposition_and_alignment(
                model, traced, assert_identity=first)
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
    """Return the named declarative E4.3/E4.4 intervention registry.

    The ``zero_topk_wk`` interventions zero ALL identified massive-activation columns
    of W_k, so k = len(massive_coords) — 3 for GPT-2 small (coords 138/378/447, matching
    the paper's Zero-Top-3-Wk), but model-dependent in general (e.g. 6 for gpt2-medium).
    The per-run k and coordinate list are recorded in run_config.json.
    """
    return {
        "baseline": ResidualInterventionSpec("baseline"),
        "nullify_bq": ResidualInterventionSpec("nullify_bq", query_bias_scale=0.0),
        "bq0__remove_first_pe": ResidualInterventionSpec(
            "bq0__remove_first_pe", query_bias_scale=0.0, position_edit="remove_first"),
        "bq0__zero_topk_wk": ResidualInterventionSpec(
            "bq0__zero_topk_wk", query_bias_scale=0.0,
            wk_kind="massive", wk_scale=0.0),
        "bq0__swap_epe": ResidualInterventionSpec(
            "bq0__swap_epe", query_bias_scale=0.0, mlp_edit="swap_epe"),
        "bq0__zero_topk_wk__swap_epe": ResidualInterventionSpec(
            "bq0__zero_topk_wk__swap_epe", query_bias_scale=0.0,
            wk_kind="massive", wk_scale=0.0, mlp_edit="swap_epe"),
        "first_layer_mlp_skip": ResidualInterventionSpec(
            "first_layer_mlp_skip", mlp_edit="zero_first"),
        "all_layer_no_mlp": ResidualInterventionSpec(
            "all_layer_no_mlp", mlp_edit="zero_all"),
        "pos1_only_pe_zero": ResidualInterventionSpec(
            "pos1_only_pe_zero", position_edit="zero_first"),
        "all_pos_no_pe": ResidualInterventionSpec(
            "all_pos_no_pe", position_edit="zero_all"),
    }


def run_intervention_set(model, tokenizer, sampled, names, specs, band=None,
                         executor=None):
    """Run a named set of interventions, returning per-name BOS-attention (% of baseline)."""
    num_layers = len(model.transformer.h)
    acc = {n: [] for n in names}
    executor = executor or ResidualExecutor(model, "manual")
    for _ds, ids in tqdm(
        iter_token_ids(model, tokenizer, sampled),
        total=count_examples(sampled),
        desc="intervention examples",
    ):
        prepared = executor.prepare(ids)
        for n in names:
            result = executor.execute(ids, specs[n], band, attention="targets",
                                      target_positions=(SINK_POS,), prepared=prepared)
            acc[n].append(target_metric(result, SINK_POS))
    means = {n: float(np.mean(v)) for n, v in acc.items()}
    base = means.get("baseline", 1.0) or 1.0
    return pd.DataFrame([
        {"intervention": n, "bos_attention": means[n], "pct_base": 100.0 * means[n] / base}
        for n in names
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# E4.5 — Relocation
# ═══════════════════════════════════════════════════════════════════════════════

def run_relocation(model, tokenizer, sampled, band=None, executor=None):
    """Attention to position 0 (sink) and position 1 (swap target) under Swap-EPE and b∧d."""
    num_layers = len(model.transformer.h)
    acc = {k: [] for k in ("base_pos0", "base_pos1",
                           "swap_pos0", "swap_pos1", "bq0swap_pos0", "bq0swap_pos1")}
    executor = executor or ResidualExecutor(model, "manual")
    specs = {
        "base": ResidualInterventionSpec("baseline"),
        "swap": ResidualInterventionSpec("swap_epe", mlp_edit="swap_epe"),
        "bq0swap": ResidualInterventionSpec(
            "bq0__swap_epe", query_bias_scale=0.0, mlp_edit="swap_epe"),
    }
    for _ds, ids in tqdm(
        iter_token_ids(model, tokenizer, sampled),
        total=count_examples(sampled),
        desc="relocation examples",
    ):
        prepared = executor.prepare(ids)
        results = {
            name: executor.execute(
                ids, spec, band, attention="targets",
                target_positions=(SINK_POS, RELOCATION_POS), prepared=prepared)
            for name, spec in specs.items()
        }
        acc["base_pos0"].append(target_metric(results["base"], SINK_POS))
        acc["base_pos1"].append(target_metric(results["base"], RELOCATION_POS))
        acc["swap_pos0"].append(target_metric(results["swap"], SINK_POS))
        acc["swap_pos1"].append(target_metric(results["swap"], RELOCATION_POS))
        acc["bq0swap_pos0"].append(target_metric(results["bq0swap"], SINK_POS))
        acc["bq0swap_pos1"].append(target_metric(results["bq0swap"], RELOCATION_POS))
    means = {k: float(np.mean(v)) for k, v in acc.items()}
    means["relocation_ratio_swap"] = (
        (means["swap_pos1"] - means["base_pos1"]) / (means["base_pos0"] + 1e-9))
    return means


# ═══════════════════════════════════════════════════════════════════════════════
# E4.6 — Perplexity (optional)
# ═══════════════════════════════════════════════════════════════════════════════

def run_perplexity(model, tokenizer, sampled, massive_coords, executor=None):
    """LM cross-entropy under each pathway ablation (and an HF baseline sanity check).

    ``zero_topk_wk`` zeroes all k = len(massive_coords) massive W_k columns (k and the
    coordinate list are recorded in run_config.json).
    """
    specs = {
        "baseline": ResidualInterventionSpec("baseline"),
        "nullify_bq": ResidualInterventionSpec("nullify_bq", query_bias_scale=0.0),
        "zero_topk_wk": ResidualInterventionSpec(
            "zero_topk_wk", wk_kind="massive", wk_scale=0.0),
        "first_layer_mlp_skip": ResidualInterventionSpec(
            "first_layer_mlp_skip", mlp_edit="zero_first"),
    }
    executor = executor or ResidualExecutor(model, "manual", massive_coords=massive_coords)
    acc = {n: [] for n in specs}
    acc_hf = []
    for _ds, ids in tqdm(
        iter_token_ids(model, tokenizer, sampled),
        total=count_examples(sampled),
        desc="perplexity examples",
    ):
        prepared = executor.prepare(ids)
        if executor.engine == "manual":
            with torch.no_grad():
                hf_logits = model(input_ids=ids).logits
            acc_hf.append(sequence_cross_entropy(hf_logits, ids))
        for n, spec in specs.items():
            result = executor.execute(ids, spec, (0, len(model.transformer.h)),
                                      attention="none", capture_logits=True,
                                      prepared=prepared)
            acc[n].append(sequence_cross_entropy(result["logits"], ids.cpu()
                                                  if result["logits"].device.type == "cpu" else ids))
            if executor.engine == "nnsight" and n == "baseline":
                acc_hf.append(acc[n][-1])
    rows = [{"intervention": "hf_reference", "cross_entropy": float(np.mean(acc_hf))}]
    for n in specs:
        rows.append({"intervention": n, "cross_entropy": float(np.mean(acc[n]))})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

KNOB_LABELS = {
    "scale_bq": r"Scale $b_Q$ (pathway A: bias)",
    "scale_pe": r"Scale $p_1$ (deletion — sink survives)",
    "interp_pe": r"Interpolate $p_1 \to p_2$ (identity replacement: both pathways)",
    "scale_wk_massive": r"Scale top-$k$ massive $W_k$ columns (coordinate channel)",
}


def plot_dose_response(df_mean, df_std, save_path, baseline=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    present = set(df_mean["knob"].unique())
    for knob in [k for k in KNOB_LABELS if k in present]:
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


def run_all_modes(model, tokenizer, sampled, modes, alphas, massive_coords, with_perplexity,
                  band=None, executor=None):
    """Run the requested modes for one seed; return a dict of results (DataFrames / arrays)."""
    res = {}
    if "dose_response" in modes:
        print("Running mode: dose_response")
        res["dose_response"] = run_dose_response(
            model, tokenizer, sampled, alphas, massive_coords, band=band, executor=executor)
    if "decomposition" in modes:
        print("Running mode: decomposition")
        res["decomposition"] = run_decomposition(
            model, tokenizer, sampled, assert_identity=True, band=band, executor=executor)
    if "combined" in modes or "surgical" in modes:
        specs = _combined_and_surgical_specs(model, massive_coords)
        if "combined" in modes:
            print("Running mode: combined")
            names = ["baseline", "nullify_bq", "bq0__remove_first_pe", "bq0__zero_topk_wk",
                     "bq0__swap_epe", "bq0__zero_topk_wk__swap_epe"]
            res["combined"] = run_intervention_set(
                model, tokenizer, sampled, names, specs, band=band, executor=executor)
        if "surgical" in modes:
            print("Running mode: surgical")
            names = ["baseline", "first_layer_mlp_skip", "all_layer_no_mlp",
                     "pos1_only_pe_zero", "all_pos_no_pe"]
            res["surgical"] = run_intervention_set(
                model, tokenizer, sampled, names, specs, band=band, executor=executor)
    if "relocation" in modes:
        print("Running mode: relocation")
        res["relocation"] = run_relocation(
            model, tokenizer, sampled, band=band, executor=executor)
    if with_perplexity or "perplexity" in modes:
        print("Running mode: perplexity")
        res["perplexity"] = run_perplexity(
            model, tokenizer, sampled, massive_coords, executor=executor)
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


def _requested_config_modes(modes, with_perplexity):
    out = list(modes)
    if with_perplexity and "perplexity" not in out:
        out.append("perplexity")
    return out


def _critical_e4_config(args, seed):
    return {
        "registry_version": E4_REGISTRY_VERSION,
        "trace_registry_version": GPT2_TRACE_REGISTRY_VERSION,
        "engine_name": args.engine,
        "model_name": args.model_name,
        "model_revision": args.revision or "main",
        "dtype": args.dtype,
        "layer_mode": args.layer_mode,
        "seed": int(seed),
        "sample_size": int(args.sample_size),
        "cut_length": int(args.cut_length),
        "alphas": [float(a) for a in args._parsed_alphas],
    }


def _validate_e4_config(path, requested, *, write=False, extra=None):
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        mismatch = {key: (cached.get(key), value) for key, value in requested.items()
                    if cached.get(key) != value}
        if mismatch:
            detail = ", ".join(
                f"{key}: cached={old!r}, requested={new!r}"
                for key, (old, new) in mismatch.items())
            raise ValueError(f"Incompatible E4 cache at {path}: {detail}")
    elif not write:
        raise FileNotFoundError(
            f"Missing E4 cache metadata {path}; legacy caches without engine metadata "
            "cannot be mixed with a requested engine")
    if write:
        path.write_text(json.dumps({**requested, **(extra or {})}, indent=2,
                                   sort_keys=True), encoding="utf-8")


def _e4_parity_tolerances(quantity, atol, rtol):
    # share_delta is a derived ratio; mathematically equivalent manual and HF/NNsight
    # execution showed a maximum observed relative discrepancy of about 2.95e-4.
    if quantity == "decomposition/share_delta":
        return atol, max(rtol, 5e-4)
    return atol, rtol


def _parity_row(quantity, manual, nnsight, atol, rtol):
    atol, rtol = _e4_parity_tolerances(quantity, atol, rtol)
    manual_arr = np.asarray(manual, dtype=float)
    nnsight_arr = np.asarray(nnsight, dtype=float)
    abs_diff = float(np.max(np.abs(manual_arr - nnsight_arr)))
    scale = float(np.max(np.abs(nnsight_arr)))
    rel_diff = abs_diff / max(scale, 1e-12)
    passed = bool(abs_diff <= atol + rtol * scale)
    return {"quantity": quantity, "manual": float(np.mean(manual_arr)),
            "nnsight_reference": float(np.mean(nnsight_arr)),
            "max_abs_difference": abs_diff, "max_relative_difference": rel_diff,
            "atol": atol, "rtol": rtol, "status": "pass" if passed else "fail"}


def verify_e4_parity(model, tokenizer, sampled, alphas, massive_coords, band,
                     nn_engine, *, atol=1e-5, rtol=1e-4):
    """Compare every E4 mode, treating the real traced HF forward as reference."""
    manual = ResidualExecutor(model, "manual", massive_coords=massive_coords)
    traced = ResidualExecutor(model, "nnsight", nn_engine, massive_coords)
    modes = list(ALL_MODES)
    manual_out = run_all_modes(model, tokenizer, sampled, modes, alphas, massive_coords,
                               True, band=band, executor=manual)
    traced_out = run_all_modes(model, tokenizer, sampled, modes, alphas, massive_coords,
                               True, band=band, executor=traced)
    rows = []
    for key in ("dose_response", "combined", "surgical", "perplexity"):
        left, right = manual_out[key], traced_out[key]
        id_cols = (["knob", "alpha"] if key == "dose_response" else ["intervention"])
        value_cols = (["bos_attention"] if key == "dose_response" else
                      (["cross_entropy"] if key == "perplexity" else
                       ["bos_attention", "pct_base"]))
        merged = left.merge(right, on=id_cols, suffixes=("_manual", "_nnsight"))
        for _, row in merged.iterrows():
            identity = "/".join(str(row[col]) for col in id_cols)
            for value in value_cols:
                rows.append(_parity_row(
                    f"{key}/{identity}/{value}", row[f"{value}_manual"],
                    row[f"{value}_nnsight"], atol, rtol))
    for key in manual_out["decomposition"]:
        rows.append(_parity_row(
            f"decomposition/{key}", manual_out["decomposition"][key],
            traced_out["decomposition"][key], atol, rtol))
    for key in manual_out["relocation"]:
        rows.append(_parity_row(
            f"relocation/{key}", manual_out["relocation"][key],
            traced_out["relocation"][key], atol, rtol))
    return {
        "reference": "NNsight real Hugging Face forward",
        "atol": atol, "rtol": rtol, "rows": rows,
        "all_rows_pass": all(row["status"] == "pass" for row in rows),
    }


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
    parser.add_argument("--revision", default=None,
                        help="Optional Hugging Face model revision (default: main).")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--experiment-name", default=None,
                        help="Subdir under --output-dir. Default: residual_sink_<model tag>.")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated seeds (data resamples). Default: 0,1,2")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--cut-length", type=int, default=DEFAULT_CUT_LENGTH)
    parser.add_argument("--alphas", default=None,
                        help="Comma-separated dose-response scale factors. Default: 0..1.5 step .25")
    parser.add_argument("--layer-mode", choices=["scaled", "fixed"], default="scaled",
                        help="Mid-layer band. 'scaled' (default) excludes the first 3 and last layer "
                             "(= layers 4-11 for the 12-layer small model, extends for deeper models); "
                             "'fixed' forces layers 4-11 on every size.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                        help="Model dtype. Default float32 (matches the paper's SEs); use a smaller "
                             "dtype only if VRAM-constrained on large XL runs.")
    parser.add_argument("--engine", choices=["manual", "nnsight"], default="manual",
                        help="Execution engine. Manual remains the reproduction default; "
                             "nnsight uses the real eager Hugging Face forward.")
    parser.add_argument("--with-perplexity", action="store_true",
                        help="Also compute the optional E4.6 LM-loss table.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation; only aggregate + plot existing per-seed outputs.")
    parser.add_argument("--verify-parity", action="store_true",
                        help="Compare manual and NNsight outputs for every E4 mode, write "
                             "parity_report.json, and exit.")
    parser.add_argument("--parity-atol", type=float, default=1e-5)
    parser.add_argument("--parity-rtol", type=float, default=1e-4)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    alphas = ([float(a) for a in args.alphas.split(",")] if args.alphas else DEFAULT_ALPHAS)
    args._parsed_alphas = alphas
    modes = ALL_MODES if args.mode == "all" else [args.mode]

    tag = args.model_name.replace("/", "_")
    exp = args.experiment_name or f"residual_sink_{tag}"
    root = Path(args.output_dir) / exp
    root.mkdir(parents=True, exist_ok=True)

    config_modes = _requested_config_modes(modes, args.with_perplexity)
    if args.plot_only:
        if args.verify_parity:
            parser.error("--plot-only and --verify-parity are mutually exclusive")
        for seed in seeds:
            requested = _critical_e4_config(args, seed)
            for mode in config_modes:
                _validate_e4_config(
                    _seed_dir(root, seed) / f"run_config_{mode}.json",
                    requested, write=False)
        agg = aggregate_and_plot(root, seeds, modes, alphas, args.with_perplexity)
        print(f"Plots and aggregate tables regenerated from cache: {agg}")
        return

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_name} on {device} (dtype {args.dtype}, engine {args.engine}) ...")
    load_engine = "nnsight" if args.verify_parity else args.engine
    try:
        model, tokenizer, nn_engine = load_model_for_engine(
            args.model_name, device, dtype=dtype, engine=load_engine,
            revision=args.revision)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to initialize requested E4 engine {args.engine!r}: {exc}") from exc
    num_layers = len(model.transformer.h)
    band = compute_band(num_layers, args.layer_mode)
    massive_coords = identify_massive_coords(model, device)
    executor = ResidualExecutor(model, args.engine, nn_engine, massive_coords)
    provenance = engine_provenance(
        args.engine, model, model_name=args.model_name, dtype=args.dtype,
        device=device, band=band, nn_engine=nn_engine, revision=args.revision)
    print(f"Layers: {num_layers}  |  mid-band [{band[0]}, {band[1]})  |  layer-mode {args.layer_mode}")
    print(f"Massive-activation coordinates of EPE_1: {massive_coords}")

    if args.verify_parity:
        seed = seeds[0] if seeds else 0
        sampled, _manifest = sample_benchmark_datasets(
            tokenizer, sample_size=args.sample_size,
            cut_length=args.cut_length, seed=seed)
        report = verify_e4_parity(
            model, tokenizer, sampled, alphas, massive_coords, band, nn_engine,
            atol=args.parity_atol, rtol=args.parity_rtol)
        (root / "parity_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        if not report["all_rows_pass"]:
            raise AssertionError(
                "E4 manual/NNsight parity differences exceeded tolerance; "
                "the NNsight Hugging Face forward is the reference. See parity_report.json.")
        print(f"Parity report written to {root / 'parity_report.json'}")
        return

    run_meta = {
        "registry_version": E4_REGISTRY_VERSION,
        "trace_registry_version": GPT2_TRACE_REGISTRY_VERSION,
        "engine_name": args.engine,
        "engine": provenance,
        "model_name": args.model_name, "model_revision": args.revision or "main",
        "dtype": args.dtype, "layer_mode": args.layer_mode,
        "band_start": band[0], "band_end": band[1], "num_layers": num_layers,
        "massive_coords": massive_coords, "alphas": alphas,
        "sample_size": args.sample_size, "cut_length": args.cut_length,
    }

    for seed in seeds:
        print(f"\n{'='*70}\nSeed {seed}: sampling + running modes {modes}\n{'='*70}")
        sampled, manifest = sample_benchmark_datasets(
            tokenizer, sample_size=args.sample_size,
            cut_length=args.cut_length, seed=seed)
        out_dir = _seed_dir(root, seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        requested = _critical_e4_config(args, seed)
        for mode in config_modes:
            _validate_e4_config(
                out_dir / f"run_config_{mode}.json", requested,
                write=True, extra={**run_meta, "seed": seed, "mode": mode})
        pd.DataFrame(manifest).to_csv(out_dir / "sample_manifest.csv", index=False)
        (out_dir / "run_config.json").write_text(
            json.dumps({**run_meta, "seed": seed, "modes": config_modes}, indent=2,
                       sort_keys=True), encoding="utf-8")
        res = run_all_modes(model, tokenizer, sampled, modes, alphas,
                            massive_coords, args.with_perplexity, band=band,
                            executor=executor)
        save_seed_results(res, out_dir)
        print(f"Seed {seed} outputs: {out_dir}")

    agg = aggregate_and_plot(root, seeds, modes, alphas, args.with_perplexity)
    print(f"\nDone. Aggregated results + figures: {agg}")


if __name__ == "__main__":
    main()
