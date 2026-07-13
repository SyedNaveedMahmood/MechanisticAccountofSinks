# -*- coding: utf-8 -*-
"""E5: Evaluation Robustness and the Functional Cost of Sink Removal.

This driver evaluates the GPT-2 Table-1 intervention family along four axes:
metric robustness, context length, language-model cost, and content generality.
All interventions are represented by :class:`InterventionSpec` and executed by
one manual forward path, ensuring attention and cross-entropy use identical
scientific semantics.  Raw per-example measurements are cached per seed and all
aggregation/figures can be regenerated with ``--plot-only`` without loading a
model or dataset.

The optimized path deliberately uses ``compute_diagnostics=False`` in the
existing manual attention implementation.  It changes neither attention nor
logits; it only suppresses legacy single-sentence diagnostic allocations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

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
    build_degenerate_domains,
    load_optional_flores,
    sample_long_benchmark_datasets,
)
from intervention_analysis import (
    INTERVENTIONS,
    compute_band,
    manual_self_attention_new,
)
from residual_sink_analysis import identify_massive_coords, load_model


REGISTRY_VERSION = "e5-spec-v1"
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_LENGTHS = (40, 128, 256, 512, 1024)
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
TABLE1_KEYS = tuple("abcdefghij")
LENGTH_KEYS = ("a", "b", "c", "d", "i", "j")
COMBINED_KEYS = ("b_and_c", "b_and_i", "b_and_d", "b_and_i_and_d")
PROFILE_KEYS = ("a", "b", "c", "i")
SINK_THRESHOLDS = (0.2, 0.3, 0.5)
NUMERIC_EPS = 1e-8
METRIC_ORIENTATION = {
    # +1 means larger raw values indicate a stronger sink; -1 means smaller.
    "bos_attn": 1, "sink_rate_0_2": 1, "sink_rate_0_3": 1,
    "sink_rate_0_5": 1, "bos_rank": -1, "attn_entropy": -1,
}
STRUCTURAL_METRICS = ("mass_pos_1_4", "mass_pos_5_plus", "redistributed_local_share", "head_gini")


def _zero_first_token(te: torch.Tensor) -> torch.Tensor:
    te[0, 0] = 0
    return te


def _remove_first_pe(pe: torch.Tensor) -> torch.Tensor:
    if pe.shape[1] < 2:
        raise ValueError("Remove First PE requires a sequence of at least two tokens")
    pe[0, 0] = pe[0, 1].clone()
    return pe


def _zero_all_pe(pe: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(pe)


def _interp_first_pe(alpha: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def transform(pe: torch.Tensor) -> torch.Tensor:
        if pe.shape[1] < 2:
            raise ValueError("PE interpolation requires a sequence of at least two tokens")
        pe[0, 0] = alpha * pe[0, 0] + (1.0 - alpha) * pe[0, 1]
        return pe
    return transform


@dataclass(frozen=True)
class InterventionSpec:
    """Declarative description of every intervention locus used by E5."""

    key: str
    label: str
    description: str
    te_transform: str | None = None
    pe_transform: str | None = None
    query_bias_scale: float = 1.0
    mlp_transform: str | None = None
    skip_all_mlp: bool = False
    wk_zero_kind: str | None = None  # ``massive`` or ``random_fixed``
    wk_scale: float | None = None


def build_intervention_specs(alpha: float | None = None) -> dict[str, InterventionSpec]:
    """Return the single intervention registry used by every E5 runner."""
    specs = {
        "a": InterventionSpec("a", "(a)", "Baseline"),
        "b": InterventionSpec("b", "(b)", "Nullify Query Bias", query_bias_scale=0.0),
        "c": InterventionSpec("c", "(c)", "Remove First PE", pe_transform="remove_first"),
        "d": InterventionSpec("d", "(d)", "Swap EPE", mlp_transform="swap_epe"),
        "e": InterventionSpec("e", "(e)", "Swap PE", mlp_transform="swap_pe"),
        "f": InterventionSpec("f", "(f)", "Nullify BOS Token", te_transform="zero_first"),
        "g": InterventionSpec("g", "(g)", "No MLP", skip_all_mlp=True),
        "h": InterventionSpec("h", "(h)", "No PE", pe_transform="zero_all"),
        "i": InterventionSpec("i", "(i)", "Zero Top-k Wk", wk_zero_kind="massive"),
        "j": InterventionSpec("j", "(j)", "Zero Random Wk", wk_zero_kind="random_fixed"),
        "b_and_c": InterventionSpec("b_and_c", "b∧c", "Nullify bQ + Remove First PE",
                                     query_bias_scale=0.0, pe_transform="remove_first"),
        "b_and_i": InterventionSpec("b_and_i", "b∧i", "Nullify bQ + Zero Top-k Wk",
                                     query_bias_scale=0.0, wk_zero_kind="massive"),
        "b_and_d": InterventionSpec("b_and_d", "b∧d", "Nullify bQ + Swap EPE",
                                     query_bias_scale=0.0, mlp_transform="swap_epe"),
        "b_and_i_and_d": InterventionSpec(
            "b_and_i_and_d", "b∧i∧d", "Nullify bQ + Zero Top-k Wk + Swap EPE",
            query_bias_scale=0.0, wk_zero_kind="massive", mlp_transform="swap_epe"),
    }
    if alpha is not None:
        specs[f"scale_bq_{alpha:g}"] = InterventionSpec(
            f"scale_bq_{alpha:g}", f"scale_bq({alpha:g})", "Scale query bias",
            query_bias_scale=float(alpha))
        specs[f"interp_pe_{alpha:g}"] = InterventionSpec(
            f"interp_pe_{alpha:g}", f"interp_pe({alpha:g})", "Interpolate PE1 to PE2",
            pe_transform=f"interp:{float(alpha):.17g}")
    return specs


def _resolve_transforms(spec: InterventionSpec, pe: torch.Tensor,
                        model: GPT2LMHeadModel) -> tuple[torch.Tensor, torch.Tensor | None, tuple]:
    te_transform = _zero_first_token if spec.te_transform == "zero_first" else None
    if spec.pe_transform == "remove_first":
        pe_transform = _remove_first_pe
    elif spec.pe_transform == "zero_all":
        pe_transform = _zero_all_pe
    elif spec.pe_transform and spec.pe_transform.startswith("interp:"):
        pe_transform = _interp_first_pe(float(spec.pe_transform.split(":", 1)[1]))
    else:
        pe_transform = None
    pe_work = pe_transform(pe.clone()) if pe_transform else pe.clone()
    ppes = None

    mlp_modify = None
    if spec.mlp_transform in {"swap_epe", "swap_pe"}:
        if spec.mlp_transform == "swap_epe":
            ppes = pe_work[0] + model.transformer.h[0].mlp(pe_work)[0]
            vectors = ppes
        else:
            vectors = pe_work[0]
        u0 = vectors[0] / torch.linalg.vector_norm(vectors[0]).clamp_min(NUMERIC_EPS)
        u1 = vectors[1] / torch.linalg.vector_norm(vectors[1]).clamp_min(NUMERIC_EPS)

        def swap(layer_idx: int, mlp_output: torch.Tensor) -> torch.Tensor:
            if layer_idx == 0:
                magnitude = torch.dot(mlp_output[0, 0], u0)
                mlp_output[0, 0] = mlp_output[0, 0] - magnitude * u0 + magnitude * u1
                mlp_output[0, 1] = mlp_output[0, 1] + magnitude * u0 - magnitude * u1
            return mlp_output
        mlp_modify = swap
    return pe_work, ppes, (te_transform, mlp_modify)


def _initial_embeddings(model: GPT2LMHeadModel, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq = input_ids.shape[1]
    if seq > model.config.n_positions:
        raise ValueError(f"Requested length {seq} exceeds model context limit {model.config.n_positions}")
    positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
    return model.transformer.wte(input_ids), model.transformer.wpe(positions)


@torch.no_grad()
def execute_spec(model: GPT2LMHeadModel, input_ids: torch.Tensor, spec: InterventionSpec,
                 band: tuple[int, int], massive_coords: Sequence[int],
                 random_coords: Sequence[int], collect_attention: bool = True,
                 collect_logits: bool = False, collect_invariance: bool = False) -> dict[str, Any]:
    """Execute one spec once, optionally retaining only band attention and/or logits."""
    te, pe = _initial_embeddings(model, input_ids)
    pe_work, ppes, transforms = _resolve_transforms(spec, pe, model)
    te_transform, mlp_modify = transforms
    if te_transform:
        te = te_transform(te.clone())
    layer_input = te + pe_work
    coords: Sequence[int] | None = None
    if spec.wk_zero_kind == "massive":
        coords = massive_coords
    elif spec.wk_zero_kind == "random_fixed":
        coords = random_coords
    attn_kwargs: dict[str, Any] = {"query_bias_scale": spec.query_bias_scale}
    if coords is not None:
        attn_kwargs["fixed_wk_zero_indices"] = list(coords)
    if spec.wk_scale is not None:
        attn_kwargs["wk_scale_indices"] = list(massive_coords)
        attn_kwargs["wk_scale"] = float(spec.wk_scale)

    attention: list[torch.Tensor] = []
    k0_values, delta_values = [], []
    with torch.no_grad():
        for li, layer in enumerate(model.transformer.h):
            normalized = layer.ln_1(layer_input)
            if collect_invariance and band[0] <= li < band[1]:
                qkv_w = layer.attn.c_attn.weight.t()
                _, wk, _ = qkv_w.chunk(3, dim=0)
                bq, _, _ = layer.attn.c_attn.bias.chunk(3, dim=0)
                k0 = F.linear(normalized[0, 0], wk, torch.zeros_like(bq))
                k0_values.append(k0.detach().float().cpu())
                delta_values.append((bq * k0).view(model.config.n_head, -1).sum(-1).detach().float().cpu())
            attn_out, weights, *_ = manual_self_attention_new(
                normalized, layer, ppes=ppes, compute_diagnostics=False, **attn_kwargs)
            if collect_attention and band[0] <= li < band[1]:
                attention.append(weights[0].detach().float().cpu())
            attn_residual = layer_input + attn_out
            if spec.skip_all_mlp:
                layer_input = attn_residual
            else:
                mlp_out = layer.mlp(layer.ln_2(attn_residual))
                if mlp_modify:
                    mlp_out = mlp_modify(li, mlp_out)
                layer_input = attn_residual + mlp_out
        logits = None
        if collect_logits:
            logits = model.lm_head(model.transformer.ln_f(layer_input))
    result = {"attention": attention, "logits": logits}
    if collect_invariance:
        result["k0"] = torch.stack(k0_values) if k0_values else torch.empty(0)
        result["delta1"] = torch.stack(delta_values) if delta_values else torch.empty(0)
    return result


def _gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    x = np.clip(x, 0.0, None)
    total = x.sum()
    if x.size == 0 or total <= NUMERIC_EPS:
        return 0.0
    x.sort()
    n = x.size
    value = (2.0 * np.dot(np.arange(1, n + 1), x) / (n * total)) - (n + 1.0) / n
    return float(np.clip(value, 0.0, 1.0))


def metric_battery(attn_maps: Sequence[torch.Tensor]) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Compute the E5 metric battery in one pass over selected-layer maps.

    BOS rank uses competition ranking: ``1 + count(keys with strictly greater
    attention)``.  Ties therefore receive the same deterministic best rank.
    Entropy includes only causal keys 0..q and is normalized by log(q+1).
    """
    if not attn_maps:
        raise ValueError("metric_battery received an empty layer band")
    stack = torch.stack(list(attn_maps)).double()  # [L,H,Q,K], CPU
    length = stack.shape[-1]
    start = length // 2
    if start >= length:
        raise ValueError("No second-half query rows are available")
    second = stack[:, :, start:, :]
    cell_bos = second[..., 0].mean(dim=2)  # [L,H]
    metrics: dict[str, float] = {"bos_attn": float(cell_bos.mean())}
    for threshold in SINK_THRESHOLDS:
        metrics[f"sink_rate_{str(threshold).replace('.', '_')}"] = float((cell_bos > threshold).double().mean())

    entropies, ranks, local, far = [], [], [], []
    query_rows = []
    for local_q, q in enumerate(range(start, length)):
        probs = second[:, :, local_q, :q + 1].clamp_min(0.0)
        denom = probs.sum(-1, keepdim=True).clamp_min(NUMERIC_EPS)
        probs = probs / denom
        entropy = -(torch.where(probs > 0, probs * probs.clamp_min(NUMERIC_EPS).log(), 0.0).sum(-1))
        entropy = entropy / math.log(q + 1) if q + 1 > 1 else torch.zeros_like(entropy)
        bos = probs[..., 0]
        rank = 1 + (probs[..., 1:] > bos.unsqueeze(-1)).sum(-1)
        local_mass = probs[..., 1:min(5, q + 1)].sum(-1) if q >= 1 else torch.zeros_like(bos)
        far_mass = probs[..., 5:].sum(-1) if q >= 5 else torch.zeros_like(bos)
        entropies.append(entropy)
        ranks.append(rank.double())
        local.append(local_mass)
        far.append(far_mass)
        query_rows.append({"query_position": q, "valid_positions": q + 1,
                           "bos_attn": float(bos.mean())})
    metrics["attn_entropy"] = float(torch.stack(entropies).mean())
    metrics["bos_rank"] = float(torch.stack(ranks).mean())
    metrics["mass_pos_1_4"] = float(torch.stack(local).mean())
    metrics["mass_pos_5_plus"] = float(torch.stack(far).mean())
    redistributed = metrics["mass_pos_1_4"] + metrics["mass_pos_5_plus"]
    metrics["redistributed_local_share"] = (
        metrics["mass_pos_1_4"] / redistributed if redistributed > NUMERIC_EPS else 0.0)
    metrics["head_gini"] = _gini(cell_bos.numpy())
    cell_rows = [{"layer_offset": li, "head": hi, "bos_attn": float(cell_bos[li, hi])}
                 for li in range(cell_bos.shape[0]) for hi in range(cell_bos.shape[1])]
    return metrics, pd.DataFrame(cell_rows), pd.DataFrame(query_rows)


def token_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return one next-token CE value per predicted position, in nats."""
    return F.cross_entropy(logits[0, :-1].float(), input_ids[0, 1:], reduction="none")


def _parse_csv(value: str, cast: Callable = str) -> list:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated value")
    try:
        return [cast(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _records_at_length(records: dict[str, list[dict]], length: int,
                       limit: int | None = None) -> dict[str, list[dict]]:
    """Create paired nested prefixes while retaining source/example identity."""
    out: dict[str, list[dict]] = {}
    for domain, rows in records.items():
        selected = rows[:limit] if limit is not None else rows
        made = []
        for row in selected:
            if len(row["input_ids"]) < length:
                raise ValueError(f"{domain} example {row['example_id']} has fewer than {length} tokens")
            new = dict(row)
            new["input_ids"] = list(row["input_ids"][:length])
            new["length"] = length
            if len(new["input_ids"]) != length:
                raise AssertionError("Nested-prefix length invariant failed")
            made.append(new)
        if not made:
            raise ValueError(f"Domain '{domain}' is empty at length {length}")
        out[domain] = made
    return out


def _iter_records(records: dict[str, list[dict]]) -> Iterable[tuple[str, dict]]:
    for domain, rows in records.items():
        for row in rows:
            yield domain, row


def _to_ids(model: GPT2LMHeadModel, row: dict) -> torch.Tensor:
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=model.device)
    expected = int(row.get("length", len(row["input_ids"])))
    if ids.shape[1] != expected:
        raise AssertionError(f"Expected {expected} tokens, found {ids.shape[1]}")
    return ids


def _base_fields(seed: int, model_name: str, domain: str, row: dict,
                 spec: InterventionSpec, length: int) -> dict[str, Any]:
    return {
        "seed": seed, "model": model_name, "domain": domain,
        "example_id": int(row["example_id"]), "length": length,
        "intervention": spec.key, "intervention_label": spec.label,
        "intervention_description": spec.description,
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_metrics_mode(model: GPT2LMHeadModel, model_name: str, seed: int,
                     records: dict[str, list[dict]], band: tuple[int, int],
                     massive: Sequence[int], random_coords: Sequence[int],
                     out_dir: Path) -> None:
    specs = build_intervention_specs()
    metric_rows, cell_rows, query_rows = [], [], []
    total = sum(map(len, records.values())) * len(TABLE1_KEYS)
    for domain, row in tqdm(_iter_records(records), total=sum(map(len, records.values())),
                            desc=f"seed {seed} metrics examples"):
        ids = _to_ids(model, row)
        for key in TABLE1_KEYS:
            spec = specs[key]
            result = execute_spec(model, ids, spec, band, massive, random_coords)
            metrics, cells, queries = metric_battery(result["attention"])
            base = _base_fields(seed, model_name, domain, row, spec, ids.shape[1])
            metric_rows.append({**base, **metrics, "status": "ok"})
            for record in cells.to_dict("records"):
                cell_rows.append({**base, "layer": band[0] + record.pop("layer_offset"), **record})
            for record in queries.to_dict("records"):
                query_rows.append({**base, **record})
            del result
        del ids
    _write_csv(metric_rows, out_dir / "metrics_per_example.csv")
    _write_csv(cell_rows, out_dir / "metrics_head_cells.csv")
    _write_csv(query_rows, out_dir / "metrics_query_positions.csv")


def _fit_logit_slopes(query_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["seed", "model", "domain", "intervention", "length"]
    for keys, group in query_df.groupby(group_cols, dropna=False):
        p = np.clip(group["bos_attn"].to_numpy(float), NUMERIC_EPS, 1.0 - NUMERIC_EPS)
        x = np.log(group["valid_positions"].to_numpy(float))
        y = np.log(p) - np.log1p(-p)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 2 and np.ptp(x[valid]) > 0:
            fit = stats.linregress(x[valid], y[valid])
            slope, intercept, stderr, status = fit.slope, fit.intercept, fit.stderr, "ok"
        else:
            slope = intercept = stderr = np.nan
            status = "insufficient_finite_rows"
        rows.append({**dict(zip(group_cols, keys)), "slope": slope, "intercept": intercept,
                     "slope_stderr": stderr, "theoretical_slope": -1.0,
                     "n_query_rows": int(valid.sum()), "status": status,
                     "probability_clip_epsilon": NUMERIC_EPS})
    return pd.DataFrame(rows)


def run_length_mode(model: GPT2LMHeadModel, model_name: str, seed: int,
                    long_records: dict[str, list[dict]], lengths: Sequence[int],
                    length_sample_size: int, band: tuple[int, int],
                    massive: Sequence[int], random_coords: Sequence[int],
                    out_dir: Path) -> None:
    specs = build_intervention_specs()
    metrics_rows, cell_rows, query_rows, invariance_raw = [], [], [], []
    shortest = min(lengths)
    for length in lengths:
        limit = length_sample_size if length == 1024 else None
        records = _records_at_length(long_records, length, limit)
        for domain, row in tqdm(_iter_records(records), total=sum(map(len, records.values())),
                                desc=f"seed {seed} length={length}"):
            ids = _to_ids(model, row)
            for key in LENGTH_KEYS:
                spec = specs[key]
                result = execute_spec(model, ids, spec, band, massive, random_coords,
                                      collect_invariance=(key == "a"))
                metrics, cells, queries = metric_battery(result["attention"])
                base = _base_fields(seed, model_name, domain, row, spec, length)
                metrics_rows.append({**base, **metrics, "status": "ok"})
                for record in cells.to_dict("records"):
                    cell_rows.append({**base, "layer": band[0] + record.pop("layer_offset"), **record})
                for record in queries.to_dict("records"):
                    query_rows.append({**base, **record})
                if key == "a":
                    invariance_raw.append({"seed": seed, "domain": domain,
                                           "example_id": row["example_id"], "length": length,
                                           "k0": result["k0"].numpy(),
                                           "delta1": result["delta1"].numpy()})
                del result
            del ids
    inv_rows = []
    by_example: dict[tuple, list[dict]] = {}
    for record in invariance_raw:
        by_example.setdefault((record["domain"], record["example_id"]), []).append(record)
    for (domain, example_id), group in by_example.items():
        reference = next((r for r in group if r["length"] == shortest), None)
        if reference is None:
            continue
        for record in group:
            for quantity in ("k0", "delta1"):
                a, b = reference[quantity], record[quantity]
                abs_diff = float(np.max(np.abs(a - b)))
                scale = float(np.max(np.abs(a)))
                inv_rows.append({"seed": seed, "domain": domain, "example_id": example_id,
                                 "reference_length": shortest, "length": record["length"],
                                 "quantity": quantity, "max_abs_discrepancy": abs_diff,
                                 "max_relative_discrepancy": abs_diff / max(scale, NUMERIC_EPS)})
    _write_csv(metrics_rows, out_dir / "length_per_example.csv")
    _write_csv(cell_rows, out_dir / "length_head_cells.csv")
    query_df = pd.DataFrame(query_rows)
    query_df.to_csv(out_dir / "length_query_positions.csv", index=False)
    _fit_logit_slopes(query_df).to_csv(out_dir / "logit_competition_slopes.csv", index=False)
    _write_csv(inv_rows, out_dir / "length_invariance_checks.csv")


def _cost_specs(alphas: Sequence[float]) -> list[InterventionSpec]:
    base = build_intervention_specs()
    ordered = [base[k] for k in TABLE1_KEYS + COMBINED_KEYS]
    for alpha in alphas:
        dynamic = build_intervention_specs(alpha)
        ordered.extend([dynamic[f"scale_bq_{alpha:g}"], dynamic[f"interp_pe_{alpha:g}"]])
    return ordered


def run_cost_mode(model: GPT2LMHeadModel, model_name: str, seed: int,
                  long_records: dict[str, list[dict]], cut_length: int,
                  alphas: Sequence[float], band: tuple[int, int],
                  massive: Sequence[int], random_coords: Sequence[int],
                  out_dir: Path, profile_lengths: Sequence[int] = (40, 1024)) -> None:
    specs = _cost_specs(alphas)
    short = _records_at_length(long_records, cut_length)
    cost_rows, cell_rows = [], []
    for domain, row in tqdm(_iter_records(short), total=sum(map(len, short.values())),
                            desc=f"seed {seed} functional cost"):
        ids = _to_ids(model, row)
        for spec in specs:
            result = execute_spec(model, ids, spec, band, massive, random_coords,
                                  collect_attention=True, collect_logits=True)
            status, warning = "ok", ""
            ce_tokens = token_cross_entropy(result["logits"], ids)
            finite = torch.isfinite(ce_tokens)
            if not bool(finite.all()):
                status = "nonfinite_ce"
                warning = f"{int((~finite).sum())} non-finite token losses"
            ce = float(ce_tokens[finite].mean()) if bool(finite.any()) else np.nan
            metrics, cells, _ = metric_battery(result["attention"])
            base = _base_fields(seed, model_name, domain, row, spec, cut_length)
            cost_rows.append({**base, "cross_entropy": ce, "bos_attn": metrics["bos_attn"],
                              "status": status, "warning": warning,
                              "valid_token_losses": int(finite.sum())})
            for record in cells.to_dict("records"):
                cell_rows.append({**base, "layer": band[0] + record.pop("layer_offset"), **record})
            del result, ce_tokens
        del ids

    profile_rows = []
    table = build_intervention_specs()
    for length in profile_lengths:
        if length > model.config.n_positions:
            continue
        limit = min(len(next(iter(long_records.values()))), 50) if length == max(profile_lengths) else None
        records = _records_at_length(long_records, length, limit)
        for domain, row in tqdm(_iter_records(records), total=sum(map(len, records.values())),
                                desc=f"seed {seed} CE profiles length={length}"):
            ids = _to_ids(model, row)
            for key in PROFILE_KEYS:
                spec = table[key]
                result = execute_spec(model, ids, spec, band, massive, random_coords,
                                      collect_attention=False, collect_logits=True)
                losses = token_cross_entropy(result["logits"], ids).detach().cpu().numpy()
                for pos, loss in enumerate(losses, start=1):
                    norm = pos / max(1, length - 1)
                    bin_id = min(9, int(norm * 10))
                    region = "early" if norm <= 1/3 else ("middle" if norm <= 2/3 else "late")
                    profile_rows.append({**_base_fields(seed, model_name, domain, row, spec, length),
                                         "prediction_position": pos, "normalized_position": norm,
                                         "position_bin": bin_id, "position_region": region,
                                         "cross_entropy": float(loss),
                                         "status": "ok" if np.isfinite(loss) else "nonfinite_ce"})
                del result
            del ids
    _write_csv(cost_rows, out_dir / "cost_per_example.csv")
    _write_csv(cell_rows, out_dir / "cost_head_cells.csv")
    _write_csv(profile_rows, out_dir / "position_ce_profiles.csv")


def run_content_mode(model: GPT2LMHeadModel, model_name: str, seed: int,
                     natural: dict[str, list[dict]], domains: Sequence[str],
                     with_multilingual: bool, cut_length: int, band: tuple[int, int],
                     massive: Sequence[int], random_coords: Sequence[int],
                     out_dir: Path, tokenizer: Any) -> tuple[list[dict], str | None]:
    requested_synthetic = [d for d in domains if d in
                           {"random_uniform", "random_zipf", "shuffled_natural", "repeat_token"}]
    synthetic, synthetic_manifest = build_degenerate_domains(
        tokenizer, natural, cut_length=cut_length, seed=seed, domains=requested_synthetic)
    records = {**natural, **synthetic}
    manifest = synthetic_manifest
    skip_reason = None
    if with_multilingual:
        try:
            multilingual, multi_manifest = load_optional_flores(
                tokenizer, sample_size=len(next(iter(natural.values()))),
                cut_length=cut_length, seed=seed)
            records.update(multilingual)
            manifest.extend(multi_manifest)
        except RuntimeError as exc:
            skip_reason = str(exc)
            warnings.warn(skip_reason)
            print(f"WARNING: {skip_reason}")
    specs = build_intervention_specs()
    metric_rows = []
    for domain, row in tqdm(_iter_records(records), total=sum(map(len, records.values())),
                            desc=f"seed {seed} content"):
        ids = _to_ids(model, {**row, "length": cut_length})
        for key in TABLE1_KEYS:
            spec = specs[key]
            result = execute_spec(model, ids, spec, band, massive, random_coords)
            metrics, _, _ = metric_battery(result["attention"])
            metric_rows.append({**_base_fields(seed, model_name, domain, row, spec, cut_length),
                                **metrics, "source_domain": row.get("source_domain", domain),
                                "source_example_id": row.get("source_example_id", row["example_id"]),
                                "status": "ok"})
            del result
        del ids
    _write_csv(metric_rows, out_dir / "content_per_example.csv")
    return manifest, skip_reason


def _read_seed_frames(root: Path, seeds: Sequence[int], filename: str,
                      required: bool = True) -> pd.DataFrame:
    frames, missing = [], []
    for seed in seeds:
        path = root / f"seed_{seed:03d}" / filename
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            missing.append(str(path))
    if required and missing:
        raise FileNotFoundError("Missing cached E5 files:\n  " + "\n  ".join(missing))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cross_seed_summary(df: pd.DataFrame, group: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Average examples within seed first, then quantify cross-seed uncertainty."""
    present = [m for m in metrics if m in df.columns]
    per_seed = df.groupby(["seed", *group], as_index=False)[present].mean(numeric_only=True)
    rows = []
    for keys, g in per_seed.groupby(list(group), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group, keys))
        for metric in present:
            values = g[metric].dropna().to_numpy(float)
            mean = float(np.mean(values)) if len(values) else np.nan
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            se = sd / math.sqrt(len(values)) if len(values) else np.nan
            critical = stats.t.ppf(0.975, len(values) - 1) if len(values) > 1 else np.nan
            rows.append({**base, "metric": metric, "mean": mean, "seed_sd": sd,
                         "seed_se": se, "ci_low": mean - critical * se if len(values) > 1 else np.nan,
                         "ci_high": mean + critical * se if len(values) > 1 else np.nan,
                         "n_seeds": len(values)})
    return pd.DataFrame(rows)


def _add_baseline_normalization(summary: pd.DataFrame,
                                group_without_intervention: Sequence[str]) -> pd.DataFrame:
    out = summary.copy()
    keys = [*group_without_intervention, "metric"]
    baseline = out[out["intervention"] == "a"][keys + ["mean"]].rename(columns={"mean": "baseline_mean"})
    out = out.merge(baseline, on=keys, how="left")
    out["percent_of_baseline"] = np.where(
        out["baseline_mean"].abs() > NUMERIC_EPS, 100 * out["mean"] / out["baseline_mean"], np.nan)
    out["baseline_normalized_effect"] = np.where(
        out["baseline_mean"].abs() > NUMERIC_EPS,
        (out["mean"] - out["baseline_mean"]) / out["baseline_mean"], np.nan)
    return out


def metric_concordance(metrics_df: pd.DataFrame) -> pd.DataFrame:
    means = metrics_df.groupby("intervention", as_index=True)[list(METRIC_ORIENTATION)].mean()
    oriented = means.copy()
    for metric, direction in METRIC_ORIENTATION.items():
        oriented[metric] *= direction
    rows = []
    for first in oriented.columns:
        for second in oriented.columns:
            tau, p = stats.kendalltau(oriented[first], oriented[second], nan_policy="omit")
            rows.append({"metric_1": first, "metric_2": second, "kendall_tau": tau,
                         "p_value": p, "n_interventions": len(oriented),
                         "orientation_1": METRIC_ORIENTATION[first],
                         "orientation_2": METRIC_ORIENTATION[second]})
    return pd.DataFrame(rows)


def intensive_extensive(cell_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_cols = ["seed", "model", "domain", "example_id", "length", "layer", "head"]
    baseline = cell_df[cell_df["intervention"] == "a"][id_cols + ["bos_attn"]].rename(
        columns={"bos_attn": "baseline_bos_attn"})
    post = cell_df[cell_df["intervention"] != "a"].merge(baseline, on=id_cols, how="inner")
    post = post.rename(columns={"bos_attn": "post_bos_attn"})
    rows = []
    for intervention, group in post.groupby("intervention"):
        x, y = group["baseline_bos_attn"].to_numpy(), group["post_bos_attn"].to_numpy()
        slope = stats.linregress(x, y).slope if len(x) > 1 and np.ptp(x) > 0 else np.nan
        corr = stats.pearsonr(x, y).statistic if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else np.nan
        for threshold in SINK_THRESHOLDS:
            active = x > threshold
            n_active = int(active.sum())
            rows.append({"intervention": intervention, "threshold": threshold,
                         "slope_baseline_to_post": slope, "pearson_r": corr,
                         "baseline_active_cells": n_active,
                         "fraction_active_falling_below": float(np.mean(y[active] <= threshold)) if n_active else np.nan,
                         "fraction_active_remaining_and_shrinking": float(np.mean((y[active] > threshold) & (y[active] < x[active]))) if n_active else np.nan})
    return post, pd.DataFrame(rows)


def bootstrap_reseed_calibration(metrics_df: pd.DataFrame, repetitions: int,
                                 bootstrap_seed: int) -> pd.DataFrame:
    rows = []
    seed_means = metrics_df.groupby(["seed", "intervention"])["bos_attn"].mean().reset_index()
    empirical = seed_means.groupby("intervention")["bos_attn"].std(ddof=1).fillna(0.0)
    for (seed, intervention), group in metrics_df.groupby(["seed", "intervention"]):
        values = group["bos_attn"].dropna().to_numpy(float)
        rng = np.random.default_rng(bootstrap_seed + int(seed) * 100003 + sum(map(ord, intervention)))
        if not len(values):
            continue
        draws = np.empty(repetitions, dtype=float)
        for index in range(repetitions):
            draws[index] = rng.choice(values, size=len(values), replace=True).mean()
        low, high = np.quantile(draws, [0.025, 0.975])
        bootstrap_se = float(draws.std(ddof=1)) if repetitions > 1 else 0.0
        reseed_sd = float(empirical.get(intervention, 0.0))
        rows.append({"seed": seed, "intervention": intervention, "n_examples": len(values),
                     "bootstrap_repetitions": repetitions, "bootstrap_seed": bootstrap_seed,
                     "bootstrap_se": bootstrap_se, "bootstrap_ci_low": low,
                     "bootstrap_ci_high": high, "bootstrap_ci_width": high - low,
                     "empirical_cross_seed_sd": reseed_sd,
                     "bootstrap_se_to_reseed_sd": bootstrap_se / reseed_sd if reseed_sd > 0 else np.nan})
    return pd.DataFrame(rows)


def mde_power_table(metrics_df: pd.DataFrame, alpha: float = 0.05,
                    power: float = 0.80) -> pd.DataFrame:
    """Conservative paired MDE using within-seed difference variance + seed-mean variance."""
    id_cols = ["seed", "domain", "example_id"]
    base = metrics_df[metrics_df["intervention"] == "a"][id_cols + ["bos_attn"]].rename(
        columns={"bos_attn": "baseline"})
    rows = []
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    for intervention, frame in metrics_df[metrics_df["intervention"] != "a"].groupby("intervention"):
        paired = frame[id_cols + ["bos_attn"]].merge(base, on=id_cols, how="inner")
        paired["difference"] = paired["bos_attn"] - paired["baseline"]
        within_vars = paired.groupby("seed")["difference"].var(ddof=1).dropna()
        seed_means = paired.groupby("seed")["difference"].mean()
        within = float(within_vars.mean()) if len(within_vars) else 0.0
        between = float(seed_means.var(ddof=1)) if len(seed_means) > 1 else 0.0
        combined = max(0.0, within + between)
        for n in (50, 100, 300, 1000):
            rows.append({"intervention": intervention, "n": n, "alpha": alpha, "power": power,
                         "within_seed_difference_variance": within,
                         "between_seed_mean_variance": between,
                         "conservative_combined_variance": combined,
                         "mde_bos_attention": z * math.sqrt(combined / n),
                         "paired_observations_available": len(paired)})
    return pd.DataFrame(rows)


def mitigation_frontier(cost_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = cost_df[(cost_df["status"] == "ok") & np.isfinite(cost_df["cross_entropy"])].copy()
    seed_domain = valid.groupby(["seed", "domain", "intervention"], as_index=False)[
        ["cross_entropy", "bos_attn"]].mean()
    baseline = seed_domain[seed_domain["intervention"] == "a"].rename(
        columns={"cross_entropy": "baseline_ce", "bos_attn": "baseline_bos"})[
        ["seed", "domain", "baseline_ce", "baseline_bos"]]
    paired = seed_domain.merge(baseline, on=["seed", "domain"], how="left")
    paired["delta_ce"] = paired["cross_entropy"] - paired["baseline_ce"]
    paired["sink_reduction"] = 1.0 - paired["bos_attn"] / paired["baseline_bos"].replace(0, np.nan)
    paired["sink_reduction_per_nat"] = paired["sink_reduction"] / paired["delta_ce"].where(
        paired["delta_ce"].abs() > NUMERIC_EPS)
    frontier = paired.groupby("intervention", as_index=False)[["delta_ce", "sink_reduction",
                                                                 "sink_reduction_per_nat"]].mean()
    efficient = []
    for i, point in frontier.iterrows():
        dominated = ((frontier["sink_reduction"] >= point["sink_reduction"]) &
                     (frontier["delta_ce"] <= point["delta_ce"]) &
                     ((frontier["sink_reduction"] > point["sink_reduction"]) |
                      (frontier["delta_ce"] < point["delta_ce"]))).any()
        efficient.append(not bool(dominated))
    frontier["pareto_efficient"] = efficient
    return paired, frontier


def _paired_content_tests(content: pd.DataFrame) -> pd.DataFrame:
    rows = []
    natural_domains = {"sst2", "gsm8k", "humaneval"}
    synth = content[~content["domain"].isin(natural_domains)]
    natural = content[content["domain"].isin(natural_domains)].copy()
    natural["source_domain"] = natural["domain"]
    natural["source_example_id"] = natural["example_id"]
    keys = ["seed", "source_domain", "source_example_id", "intervention"]
    for domain, group in synth.groupby("domain"):
        paired = group.merge(natural[keys + ["bos_attn"]], on=keys, suffixes=("", "_natural"))
        for intervention, sub in paired.groupby("intervention"):
            diff = sub["bos_attn"] - sub["bos_attn_natural"]
            if len(diff) > 1:
                test = stats.ttest_rel(sub["bos_attn"], sub["bos_attn_natural"], nan_policy="omit")
                statistic, p = test.statistic, test.pvalue
            else:
                statistic = p = np.nan
            rows.append({"domain": domain, "intervention": intervention, "n_pairs": len(diff),
                         "mean_paired_difference": diff.mean(), "paired_t_statistic": statistic,
                         "paired_p_value": p, "pairing_valid": bool(len(diff))})
    return pd.DataFrame(rows)


def _save_heatmap(matrix: pd.DataFrame, path: Path, title: str,
                  vmin: float | None = None, vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix.to_numpy(float), cmap="coolwarm", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_multiverse(summary: pd.DataFrame, path: Path) -> None:
    selected = summary[summary["metric"].isin(METRIC_ORIENTATION)].copy()
    matrix = selected.pivot(index="intervention", columns="metric", values="percent_of_baseline")
    _save_heatmap(matrix, path, "E5-A: Metric multiverse (% of baseline)")


def plot_metric_behavior(concordance: pd.DataFrame, effects: pd.DataFrame,
                         path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    matrix = concordance.pivot(index="metric_1", columns="metric_2", values="kendall_tau")
    im = axes[0].imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(matrix)), matrix.index)
    axes[0].set_title("Ranking concordance (Kendall τ)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    key = effects[effects["intervention"].isin(["b", "c", "i"])].copy()
    for intervention, group in key.groupby("intervention"):
        axes[1].scatter(group["baseline_bos_attn"], group["post_bos_attn"], s=5, alpha=.25,
                        label=intervention)
    limit = max(effects[["baseline_bos_attn", "post_bos_attn"]].max().max(), NUMERIC_EPS)
    axes[1].plot([0, limit], [0, limit], ":", color="black")
    axes[1].set(xlabel="Baseline cell BOS mass", ylabel="Post-intervention BOS mass",
                title="Intensive vs extensive effects")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_length_scaling(summary: pd.DataFrame, slopes: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bos = summary[summary["metric"] == "bos_attn"]
    for intervention, group in bos.groupby("intervention"):
        group = group.sort_values("length")
        axes[0].errorbar(group["length"], group["percent_of_baseline"],
                         yerr=group["seed_se"] * 100 / group["baseline_mean"].abs().clip(lower=NUMERIC_EPS),
                         marker="o", label=intervention)
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Context length", ylabel="BOS attention (% baseline)",
                title="Length generalization")
    axes[0].legend(ncol=2)
    slope_summary = slopes.groupby(["intervention", "length"], as_index=False)["slope"].mean()
    for intervention, group in slope_summary.groupby("intervention"):
        axes[1].plot(group["length"], group["slope"], marker="o", label=intervention)
    axes[1].axhline(-1, color="black", linestyle=":", label="theory −1")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Context length", ylabel="Fitted logit slope",
                title="Logit-competition law")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_frontier(frontier: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(frontier["pareto_efficient"], "tab:red", "steelblue")
    ax.scatter(frontier["sink_reduction"], frontier["delta_ce"], c=colors, s=55)
    for _, row in frontier.iterrows():
        ax.annotate(row["intervention"], (row["sink_reduction"], row["delta_ce"]), fontsize=8)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set(xlabel="Sink reduction (higher is better)", ylabel="ΔCE, nats (lower is better)",
           title="E5-D: Sink-removal mitigation frontier")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_position_damage(profiles: pd.DataFrame, path: Path) -> None:
    valid = profiles[(profiles["status"] == "ok") & np.isfinite(profiles["delta_ce"])]
    bins = valid.groupby(["length", "intervention", "position_bin"], as_index=False)["delta_ce"].mean()
    fig, axes = plt.subplots(1, len(sorted(bins["length"].unique())), figsize=(13, 5), squeeze=False)
    for ax, length in zip(axes[0], sorted(bins["length"].unique())):
        for intervention, group in bins[bins["length"] == length].groupby("intervention"):
            ax.plot((group["position_bin"] + .5) / 10, group["delta_ce"], marker="o",
                    label=intervention)
        ax.axhline(0, color="gray", linestyle=":")
        ax.set(title=f"Length {length}", xlabel="Normalized position", ylabel="ΔCE vs baseline (nats)")
        ax.legend()
    fig.suptitle("E5-E: Per-position functional damage")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_content(summary: pd.DataFrame, path: Path) -> None:
    bos = summary[summary["metric"] == "bos_attn"].copy()
    domains = list(bos["domain"].drop_duplicates())
    interventions = list(bos["intervention"].drop_duplicates())
    matrix = bos.pivot(index="domain", columns="intervention", values="percent_of_baseline").reindex(
        index=domains, columns=interventions)
    _save_heatmap(matrix, path, "E5-F: Content generality (% of domain baseline)")


def aggregate_and_plot(root: Path, seeds: Sequence[int], modes: Sequence[str],
                       bootstrap_repetitions: int, bootstrap_seed: int) -> Path:
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    if "metrics" in modes:
        metrics = _read_seed_frames(root, seeds, "metrics_per_example.csv")
        cells = _read_seed_frames(root, seeds, "metrics_head_cells.csv")
        queries = _read_seed_frames(root, seeds, "metrics_query_positions.csv")
        metrics.to_csv(aggregate / "metrics_per_example.csv", index=False)
        cells.to_csv(aggregate / "metrics_head_cells.csv", index=False)
        queries.to_csv(aggregate / "metrics_query_positions.csv", index=False)
        summary = _cross_seed_summary(metrics, ["intervention"],
                                      [*METRIC_ORIENTATION, *STRUCTURAL_METRICS])
        summary = _add_baseline_normalization(summary, [])
        summary.to_csv(aggregate / "metrics_summary.csv", index=False)
        concordance = metric_concordance(metrics)
        concordance.to_csv(aggregate / "metric_concordance.csv", index=False)
        effects, intensive = intensive_extensive(cells)
        effects.to_csv(aggregate / "head_cell_effects.csv", index=False)
        intensive.to_csv(aggregate / "intensive_extensive_summary.csv", index=False)
        calibration = bootstrap_reseed_calibration(metrics, bootstrap_repetitions, bootstrap_seed)
        calibration.to_csv(aggregate / "bootstrap_reseed_calibration.csv", index=False)
        mde_power_table(metrics).to_csv(aggregate / "mde_power.csv", index=False)
        plot_metric_multiverse(summary, aggregate / "fig_E5A_metric_multiverse.png")
        plot_metric_behavior(concordance, effects, aggregate / "fig_E5B_concordance_intensive.png")

    if "length" in modes:
        length = _read_seed_frames(root, seeds, "length_per_example.csv")
        length_cells = _read_seed_frames(root, seeds, "length_head_cells.csv")
        length.to_csv(aggregate / "length_per_example.csv", index=False)
        length_cells.to_csv(aggregate / "length_head_cells.csv", index=False)
        summary = _cross_seed_summary(length, ["length", "intervention"],
                                      [*METRIC_ORIENTATION, *STRUCTURAL_METRICS])
        summary = _add_baseline_normalization(summary, ["length"])
        summary.to_csv(aggregate / "length_summary.csv", index=False)
        slopes = _read_seed_frames(root, seeds, "logit_competition_slopes.csv")
        slopes.to_csv(aggregate / "logit_competition_slopes.csv", index=False)
        _cross_seed_summary(slopes[slopes["status"] == "ok"],
                            ["domain", "intervention", "length"], ["slope"]).to_csv(
            aggregate / "logit_competition_slopes_summary.csv", index=False)
        invariance = _read_seed_frames(root, seeds, "length_invariance_checks.csv")
        invariance.to_csv(aggregate / "length_invariance_checks.csv", index=False)
        invariance.groupby(["quantity", "length"], as_index=False).agg(
            max_abs_discrepancy=("max_abs_discrepancy", "max"),
            max_relative_discrepancy=("max_relative_discrepancy", "max"),
            mean_abs_discrepancy=("max_abs_discrepancy", "mean"),
            n_comparisons=("example_id", "count")).to_csv(
                aggregate / "length_invariance_summary.csv", index=False)
        query = _read_seed_frames(root, seeds, "length_query_positions.csv")
        query["position_bin"] = np.minimum(19, (20 * query["query_position"] / query["length"]).astype(int))
        query.groupby(["seed", "domain", "intervention", "length", "position_bin"], as_index=False)[
            "bos_attn"].mean().to_csv(aggregate / "length_position_bins.csv", index=False)
        sampling = length.groupby(["seed", "domain", "intervention", "length"])["bos_attn"].agg(
            sample_mean="mean", sample_se=lambda x: x.std(ddof=1) / math.sqrt(len(x))).reset_index()
        sampling.to_csv(aggregate / "length_sampling_variability.csv", index=False)
        plot_length_scaling(summary, slopes, aggregate / "fig_E5C_length_scaling.png")

    if "cost" in modes:
        cost = _read_seed_frames(root, seeds, "cost_per_example.csv")
        cost.to_csv(aggregate / "cost_per_example.csv", index=False)
        valid = cost[(cost["status"] == "ok") & np.isfinite(cost["cross_entropy"])]
        paired, frontier = mitigation_frontier(cost)
        paired.to_csv(aggregate / "cost_paired_effects.csv", index=False)
        summary = _cross_seed_summary(
            paired, ["domain", "intervention"],
            ["cross_entropy", "bos_attn", "delta_ce", "sink_reduction", "sink_reduction_per_nat"])
        summary.to_csv(aggregate / "cost_summary.csv", index=False)
        frontier.to_csv(aggregate / "mitigation_frontier.csv", index=False)
        dose = frontier[frontier["intervention"].str.startswith(("scale_bq_", "interp_pe_"))].copy()
        dose["dose_family"] = dose["intervention"].str.rsplit("_", n=1).str[0]
        dose["alpha"] = pd.to_numeric(dose["intervention"].str.rsplit("_", n=1).str[1])
        dose.to_csv(aggregate / "dose_cost.csv", index=False)
        profiles = _read_seed_frames(root, seeds, "position_ce_profiles.csv")
        profiles.to_csv(aggregate / "position_ce_profiles.csv", index=False)
        profile_ids = ["seed", "domain", "example_id", "length", "prediction_position"]
        profile_base = profiles[profiles["intervention"] == "a"][profile_ids + ["cross_entropy"]].rename(
            columns={"cross_entropy": "baseline_cross_entropy"})
        damage = profiles.merge(profile_base, on=profile_ids, how="left")
        damage["delta_ce"] = damage["cross_entropy"] - damage["baseline_cross_entropy"]
        damage.to_csv(aggregate / "position_ce_damage_profiles.csv", index=False)
        profiles.groupby(["seed", "domain", "length", "intervention", "position_bin"], as_index=False)[
            "cross_entropy"].mean().to_csv(aggregate / "position_ce_normalized_bins.csv", index=False)
        profiles.groupby(["seed", "domain", "length", "intervention", "position_region"], as_index=False)[
            "cross_entropy"].mean().to_csv(aggregate / "position_ce_region_summary.csv", index=False)
        damage.groupby(["seed", "domain", "length", "intervention", "position_bin"], as_index=False)[
            "delta_ce"].mean().to_csv(aggregate / "position_ce_damage_bins.csv", index=False)
        damage.groupby(["seed", "domain", "length", "intervention", "position_region"], as_index=False)[
            "delta_ce"].mean().to_csv(aggregate / "position_ce_damage_region_summary.csv", index=False)
        plot_frontier(frontier, aggregate / "fig_E5D_mitigation_frontier.png")
        plot_position_damage(damage, aggregate / "fig_E5E_position_damage.png")

    if "content" in modes:
        content = _read_seed_frames(root, seeds, "content_per_example.csv")
        content.to_csv(aggregate / "content_per_example.csv", index=False)
        summary = _cross_seed_summary(content, ["domain", "intervention"],
                                      [*METRIC_ORIENTATION, *STRUCTURAL_METRICS])
        summary = _add_baseline_normalization(summary, ["domain"])
        counts = content.groupby(["seed", "domain"], as_index=False)["example_id"].nunique().rename(
            columns={"example_id": "sample_count"})
        summary = summary.merge(counts.groupby("domain", as_index=False)["sample_count"].mean(),
                                on="domain", how="left")
        summary.to_csv(aggregate / "content_summary.csv", index=False)
        _paired_content_tests(content).to_csv(aggregate / "content_paired_tests.csv", index=False)
        metric_cols = [*METRIC_ORIENTATION, *STRUCTURAL_METRICS]
        natural_mask = content["domain"].isin(["sst2", "gsm8k", "humaneval"])
        natural_ref = content[natural_mask].groupby(["seed", "intervention"], as_index=False)[
            metric_cols].mean()
        domain_means = content[~natural_mask].groupby(
            ["seed", "domain", "intervention"], as_index=False)[metric_cols].mean()
        comparison = domain_means.merge(natural_ref, on=["seed", "intervention"],
                                        suffixes=("", "_natural_reference"))
        comparison_rows = []
        for _, row in comparison.iterrows():
            for metric in metric_cols:
                comparison_rows.append({"seed": row["seed"], "domain": row["domain"],
                                        "intervention": row["intervention"], "metric": metric,
                                        "domain_value": row[metric],
                                        "natural_reference_value": row[f"{metric}_natural_reference"],
                                        "raw_difference": row[metric] - row[f"{metric}_natural_reference"]})
        pd.DataFrame(comparison_rows).to_csv(aggregate / "content_domain_comparisons.csv", index=False)
        plot_content(summary, aggregate / "fig_E5F_content_generality.png")
    return aggregate


def _critical_config(args: argparse.Namespace, mode: str, seed: int,
                     band: tuple[int, int] | None = None) -> dict[str, Any]:
    config = {
        "registry_version": REGISTRY_VERSION, "model_name": args.model_name,
        "dtype": args.dtype, "layer_mode": args.layer_mode, "seed": seed,
        "mode": mode, "sample_size": args.sample_size, "cut_length": args.cut_length,
        "random_wk_seed": args.random_wk_seed,
        "benchmark_domains": args.benchmark_domains,
    }
    if mode == "length":
        config.update(lengths=args.lengths, length_sample_size=args.length_sample_size)
    if mode == "cost":
        config["alphas"] = args.alphas
    if mode == "content":
        config.update(content_domains=args.content_domains,
                      with_multilingual=args.with_multilingual)
    if band is not None:
        config.update(band_start=band[0], band_end=band[1])
    return config


def _validate_or_write_config(path: Path, requested: dict[str, Any],
                              extra: dict[str, Any] | None = None,
                              write: bool = True) -> None:
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {key: (cached.get(key), value) for key, value in requested.items()
                      if cached.get(key) != value}
        if mismatches:
            formatted = ", ".join(f"{k}: cached={a!r}, requested={b!r}"
                                  for k, (a, b) in mismatches.items())
            raise ValueError(f"Incompatible cached run at {path}: {formatted}")
    elif not write:
        raise FileNotFoundError(f"Missing cached metadata: {path}")
    if write:
        payload = {**requested, **(extra or {})}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def verify_parity(model: GPT2LMHeadModel, input_ids: torch.Tensor,
                  band: tuple[int, int], massive: Sequence[int], random_coords: Sequence[int],
                  atol: float = 1e-6, rtol: float = 1e-5) -> dict[str, Any]:
    """Compare optimized specifications with the legacy Table-1 registry.

    Intervention (j) is reported as an intentional controlled deviation: legacy
    code draws new random coordinates inside every layer, whereas E5 fixes one
    seeded coordinate set for the complete run.  All other rows are strict parity
    checks on identical token and positional embeddings.
    """
    te, pe = _initial_embeddings(model, input_ids)
    specs = build_intervention_specs()
    legacy = {key.removeprefix("int_"): fn for key, _label, _desc, fn in INTERVENTIONS}
    rows = []
    for key in TABLE1_KEYS:
        optimized = execute_spec(model, input_ids, specs[key], band, massive, random_coords)["attention"]
        with torch.no_grad():
            if key == "j":
                rows.append({"intervention": key, "status": "intentional_deviation",
                             "max_abs_attention_difference": np.nan,
                             "note": "E5 uses fixed seeded Wk coordinates; legacy resamples per layer."})
                continue
            reference_all = legacy[key](model, te, pe)
        reference = [reference_all[li].float().cpu() for li in range(band[0], band[1])]
        differences = [float((a - b).abs().max()) for a, b in zip(optimized, reference)]
        max_diff = max(differences, default=np.nan)
        equal = all(torch.allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(optimized, reference))
        rows.append({"intervention": key, "status": "pass" if equal else "fail",
                     "max_abs_attention_difference": max_diff, "atol": atol, "rtol": rtol,
                     "note": "strict attention parity"})
    report = {"registry_version": REGISTRY_VERSION, "rows": rows,
              "all_strict_rows_pass": all(r["status"] == "pass" for r in rows if r["intervention"] != "j")}
    if not report["all_strict_rows_pass"]:
        failed = [r["intervention"] for r in rows if r["status"] == "fail"]
        raise AssertionError(f"E5 parity failed for: {failed}")
    return report


class _SmokeTokenizer:
    """Minimal tokenizer surface for the explicitly requested offline smoke test."""
    eos_token_id = 31

    def __len__(self) -> int:
        return 32

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(f"t{int(i)}" for i in ids)


def run_smoke_test(output_dir: Path) -> None:
    """Offline end-to-end check using random GPT-2 weights and synthetic IDs."""
    torch.manual_seed(7)
    np.random.seed(7)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=16, n_ctx=16, n_embd=32, n_layer=4, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))
    model.eval()
    band = compute_band(len(model.transformer.h), "scaled")
    assert band[0] < band[1]
    massive = identify_massive_coords(model, torch.device("cpu"))
    random_coords = random.Random(11).sample(range(model.config.n_embd), len(massive))
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    te, pe = _initial_embeddings(model, ids)
    normalized = model.transformer.h[0].ln_1(te + pe)
    full = manual_self_attention_new(normalized, model.transformer.h[0], compute_diagnostics=True)
    light = manual_self_attention_new(normalized, model.transformer.h[0], compute_diagnostics=False)
    assert torch.allclose(full[0], light[0], atol=1e-7, rtol=1e-6)
    assert torch.allclose(full[1], light[1], atol=1e-7, rtol=1e-6)
    specs = build_intervention_specs()
    baseline = execute_spec(model, ids, specs["a"], band, massive, random_coords,
                            collect_logits=True)
    metrics, _, _ = metric_battery(baseline["attention"])
    assert 0 <= metrics["attn_entropy"] <= 1 + 1e-10
    assert torch.isfinite(token_cross_entropy(baseline["logits"], ids)).all()
    report = verify_parity(model, ids, band, massive, random_coords, atol=1e-6, rtol=1e-5)

    root = output_dir / "e5_smoke"
    tokenizer = _SmokeTokenizer()
    natural = {}
    for domain_no, domain in enumerate(("sst2", "gsm8k", "humaneval")):
        seq = [1 + ((domain_no * 5 + i) % 30) for i in range(12)]
        natural[domain] = [{"dataset": domain, "example_id": 0, "input_ids": seq,
                            "text": tokenizer.decode(seq), "source_indices": [0]}]
    for seed in (0, 1):
        seed_dir = root / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        short = _records_at_length(natural, 8)
        run_metrics_mode(model, "random-gpt2", seed, short, band, massive, random_coords, seed_dir)
        run_length_mode(model, "random-gpt2", seed, natural, (8, 12), 1, band,
                        massive, random_coords, seed_dir)
        run_cost_mode(model, "random-gpt2", seed, natural, 8, (0.0, 1.0), band,
                      massive, random_coords, seed_dir, profile_lengths=(8, 12))
        run_content_mode(model, "random-gpt2", seed, short,
                         ("random_uniform", "random_zipf", "shuffled_natural", "repeat_token"),
                         False, 8, band, massive, random_coords, seed_dir, tokenizer)
    aggregate_and_plot(root, (0, 1), ("metrics", "length", "cost", "content"), 20, 13)
    required = ["fig_E5A_metric_multiverse.png", "fig_E5B_concordance_intensive.png",
                "fig_E5C_length_scaling.png", "fig_E5D_mitigation_frontier.png",
                "fig_E5E_position_damage.png", "fig_E5F_content_generality.png"]
    assert all((root / "aggregate" / name).exists() for name in required)
    (root / "parity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SMOKE TEST PASSED: {root}")


def _manifest_rows(records: dict[str, list[dict]], seed: int, max_length: int) -> list[dict]:
    rows = []
    for domain, domain_rows in records.items():
        for row in domain_rows:
            rows.append({"seed": seed, "dataset": domain, "example_id": row["example_id"],
                         "token_length": len(row["input_ids"]), "max_requested_length": max_length,
                         "source_indices": ";".join(map(str, row.get("source_indices", []))),
                         "source_component_count": row.get("source_component_count", 1),
                         "sampling": "within_domain_concatenation", "text": row.get("text", "")})
    return rows


def _requested_modes(mode: str) -> list[str]:
    return ["metrics", "length", "cost", "content"] if mode == "all" else [mode]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", choices=["metrics", "length", "cost", "content", "all"], default="all")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--lengths", default="40,128,256,512,1024")
    parser.add_argument("--length-sample-size", type=int, default=50,
                        help="Examples/domain at length 1024; shorter lengths use --sample-size")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--cut-length", type=int, default=40)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1,1.25,1.5")
    parser.add_argument("--layer-mode", choices=["scaled", "fixed"], default="scaled")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--domains", default="sst2,gsm8k,humaneval,random_uniform,random_zipf,shuffled_natural,repeat_token",
                        help="Comma-separated benchmark and content-control domains")
    parser.add_argument("--with-multilingual", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--verify-parity", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--random-wk-seed", type=int, default=1729)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    args.seeds = _parse_csv(args.seeds, int)
    args.lengths = sorted(set(_parse_csv(args.lengths, int)))
    args.alphas = _parse_csv(args.alphas, float)
    domains = _parse_csv(args.domains, str)
    benchmark_valid = {"sst2", "gsm8k", "humaneval"}
    content_valid = {"random_uniform", "random_zipf", "shuffled_natural", "repeat_token"}
    unknown = set(domains).difference(benchmark_valid | content_valid)
    if unknown:
        raise ValueError(f"Unsupported domain(s): {sorted(unknown)}")
    args.benchmark_domains = [d for d in domains if d in benchmark_valid]
    args.content_domains = [d for d in domains if d in content_valid]
    if not args.benchmark_domains:
        raise ValueError("At least one benchmark domain is required")
    if "content" in _requested_modes(args.mode) and not args.content_domains:
        raise ValueError("Content mode requires at least one synthetic domain")
    if args.sample_size <= 0 or args.length_sample_size <= 0:
        raise ValueError("Sample sizes must be positive")
    if args.cut_length < 2 or any(length < 2 for length in args.lengths):
        raise ValueError("All sequence lengths must be at least 2")
    if args.bootstrap_repetitions < 2:
        raise ValueError("--bootstrap-repetitions must be at least 2")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)
    if args.smoke_test:
        run_smoke_test(Path(args.output_dir))
        return
    modes = _requested_modes(args.mode)
    tag = args.model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    root = Path(args.output_dir) / (args.experiment_name or f"evaluation_robustness_{tag}")

    if args.plot_only:
        cached_science = None
        for seed in args.seeds:
            for mode in modes:
                config_path = root / f"seed_{seed:03d}" / f"run_config_{mode}.json"
                _validate_or_write_config(config_path, _critical_config(args, mode, seed), write=False)
                cached = json.loads(config_path.read_text(encoding="utf-8"))
                science = {key: cached.get(key) for key in
                           ("band_start", "band_end", "massive_coords", "random_wk_coords")}
                if cached_science is None:
                    cached_science = science
                elif science != cached_science:
                    raise ValueError(f"Cached modes disagree on band/coordinate metadata: "
                                     f"{cached_science!r} vs {science!r} at {config_path}")
        aggregate = aggregate_and_plot(root, args.seeds, modes, args.bootstrap_repetitions,
                                       args.random_wk_seed + 1)
        print(f"Plots and aggregate tables regenerated from cache: {aggregate}")
        return

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, tokenizer = load_model(args.model_name, device, dtype=dtype)
    except Exception as exc:
        raise RuntimeError(f"Unable to load GPT-2-compatible model {args.model_name!r}: {exc}") from exc
    if not isinstance(model, GPT2LMHeadModel):
        raise TypeError("E5 currently supports Hugging Face GPT2LMHeadModel checkpoints only")
    band = compute_band(len(model.transformer.h), args.layer_mode)
    requested_max = max(args.lengths) if "length" in modes else args.cut_length
    if "cost" in modes:
        requested_max = max(requested_max, 1024)
    if requested_max > model.config.n_positions:
        raise ValueError(f"Requested length {requested_max} exceeds {args.model_name} context limit "
                         f"{model.config.n_positions}")
    massive = identify_massive_coords(model, device)
    rng = random.Random(args.random_wk_seed)
    if len(massive) > model.config.n_embd:
        raise ValueError("Massive-coordinate count exceeds hidden dimension")
    random_coords = sorted(rng.sample(range(model.config.n_embd), len(massive)))
    scientific_deviations = {
        "random_wk_control": "Fixed seeded coordinates for all examples/layers instead of legacy per-layer draws",
        "long_context_sampling": "Deterministic within-domain concatenation; nested lengths are token-ID prefixes",
    }
    print(f"Loaded {args.model_name} on {device}; band={band}; massive={massive}; random={random_coords}")

    if args.verify_parity:
        ids = torch.arange(1, min(args.cut_length, 40) + 1, device=device).remainder(
            model.config.vocab_size).unsqueeze(0)
        report = verify_parity(model, ids, band, massive, random_coords)
        root.mkdir(parents=True, exist_ok=True)
        (root / "parity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    for seed in args.seeds:
        seed_dir = root / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        long_records, sampler_manifest = sample_long_benchmark_datasets(
            tokenizer, sample_size=args.sample_size, cut_length=requested_max,
            seed=seed, domains=args.benchmark_domains)
        manifest = _manifest_rows(long_records, seed, requested_max)
        pd.DataFrame(manifest).to_csv(seed_dir / "sample_manifest.csv", index=False)
        pd.DataFrame(sampler_manifest).to_csv(seed_dir / "source_manifest.csv", index=False)
        for mode in modes:
            requested = _critical_config(args, mode, seed, band)
            requested.update(massive_coords=massive, random_wk_coords=random_coords)
            extra = {"num_layers": len(model.transformer.h), "context_limit": model.config.n_positions,
                     "scientific_deviations": scientific_deviations,
                     "bootstrap_repetitions": args.bootstrap_repetitions,
                     "bootstrap_seed": args.random_wk_seed + 1}
            config_path = seed_dir / f"run_config_{mode}.json"
            _validate_or_write_config(config_path, requested, extra=extra)
            # A general run_config.json is convenient for auditors; mode-specific files
            # prevent four parallel modes from concealing incompatible settings.
            (seed_dir / "run_config.json").write_text(
                json.dumps({**requested, **extra}, indent=2, sort_keys=True), encoding="utf-8")
            if mode == "metrics":
                run_metrics_mode(model, args.model_name, seed,
                                 _records_at_length(long_records, args.cut_length), band,
                                 massive, random_coords, seed_dir)
            elif mode == "length":
                run_length_mode(model, args.model_name, seed, long_records, args.lengths,
                                args.length_sample_size, band, massive, random_coords, seed_dir)
            elif mode == "cost":
                run_cost_mode(model, args.model_name, seed, long_records, args.cut_length,
                              args.alphas, band, massive, random_coords, seed_dir)
            elif mode == "content":
                content_manifest, skip = run_content_mode(
                    model, args.model_name, seed,
                    _records_at_length(long_records, args.cut_length), args.content_domains,
                    args.with_multilingual, args.cut_length, band, massive, random_coords,
                    seed_dir, tokenizer)
                pd.DataFrame(content_manifest).to_csv(seed_dir / "content_manifest.csv", index=False)
                if skip:
                    meta = json.loads(config_path.read_text(encoding="utf-8"))
                    meta["multilingual_skip_reason"] = skip
                    config_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregate = aggregate_and_plot(root, args.seeds, modes, args.bootstrap_repetitions,
                                   args.random_wk_seed + 1)
    print(f"E5 complete. Cached data and paper-ready outputs: {aggregate}")


if __name__ == "__main__":
    main()
