"""Offline E5 smoke test for metrics, length, cost, content, cache, and plot-only paths."""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))

from evaluation_robustness import evaluation_robustness_analysis as e5
from intervention_analysis import compute_band
from nnsight_engine import ARCH_SPECS, NNsightEngine, load_nnsight_model
from nnsight_smoke_utils import (
    assert_nnsight_metadata,
    create_tiny_checkpoint,
    synthetic_records,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sinks_e5_nnsight_") as temp:
        base = Path(temp)
        checkpoint = base / "tiny_gpt2"
        tokenizer = create_tiny_checkpoint(checkpoint, 53)
        lm = load_nnsight_model(
            ARCH_SPECS["gpt2"], str(checkpoint), dtype=torch.float32,
            tokenizer=tokenizer, device=torch.device("cpu"), local_files_only=True)
        model = lm._model
        nn_engine = NNsightEngine(lm, ARCH_SPECS["gpt2"])
        executor = e5.E5Executor(model, "nnsight", nn_engine)
        band = compute_band(len(model.transformer.h), "scaled")
        massive = e5.identify_massive_coords(model, torch.device("cpu"))
        random_coords = sorted(random.Random(1729).sample(
            range(model.config.n_embd), len(massive)))
        records = synthetic_records(length=16, examples_per_domain=1)
        short = e5._records_at_length(records, 8)
        root = base / "outputs" / "e5_smoke"
        seed_dir = root / "seed_000"
        seed_dir.mkdir(parents=True)

        probe_ids = e5._to_ids(model, short["sst2"][0])
        probe = executor.execute(
            probe_ids, e5.build_intervention_specs()["a"], band,
            massive, random_coords, collect_attention=True,
            collect_logits=True, collect_token_ce=True)
        assert probe["logits"].shape == (1, 8, model.config.vocab_size)
        assert probe["token_ce"].shape == (7,)
        assert torch.isfinite(probe["logits"]).all()
        assert torch.isfinite(probe["token_ce"]).all()
        del probe, probe_ids

        args = SimpleNamespace(
            engine="nnsight", model_name=str(checkpoint), revision=None,
            dtype="float32", layer_mode="scaled", sample_size=1, cut_length=8,
            random_wk_seed=1729, benchmark_domains=["sst2", "gsm8k", "humaneval"],
            lengths=[8, 16], length_sample_size=1, alphas=[0.0, 1.0],
            content_domains=["random_uniform", "random_zipf", "shuffled_natural", "repeat_token"],
            with_multilingual=False)
        provenance = e5._e5_engine_provenance(
            "nnsight", model, nn_engine=nn_engine, args=args,
            device=torch.device("cpu"), band=band)
        for mode in ("metrics", "length", "cost", "content"):
            requested = e5._critical_config(args, mode, 0, band)
            e5._validate_or_write_config(
                seed_dir / f"run_config_{mode}.json", requested,
                extra={"engine": provenance, "num_layers": len(model.transformer.h)})

        e5.run_metrics_mode(
            model, str(checkpoint), 0, short, band, massive, random_coords,
            seed_dir, executor=executor)
        e5.run_length_mode(
            model, str(checkpoint), 0, records, (8, 16), 1, band,
            massive, random_coords, seed_dir, executor=executor)
        e5.run_cost_mode(
            model, str(checkpoint), 0, records, 8, (0.0, 1.0), band,
            massive, random_coords, seed_dir, profile_lengths=(8, 16),
            executor=executor)
        manifest, skip = e5.run_content_mode(
            model, str(checkpoint), 0, short, args.content_domains, False, 8,
            band, massive, random_coords, seed_dir, tokenizer, executor=executor)
        assert skip is None and manifest
        pd.DataFrame(manifest).to_csv(seed_dir / "content_manifest.csv", index=False)
        general = {**e5._critical_config(args, "content", 0, band), "engine": provenance}
        (seed_dir / "run_config.json").write_text(
            json.dumps(general, indent=2), encoding="utf-8")
        assert_nnsight_metadata(general)

        metrics = pd.read_csv(seed_dir / "metrics_per_example.csv")
        cost = pd.read_csv(seed_dir / "cost_per_example.csv")
        invariance = pd.read_csv(seed_dir / "length_invariance_checks.csv")
        assert set(e5.TABLE1_KEYS).issubset(set(metrics["intervention"]))
        assert set(e5.COMBINED_KEYS).issubset(set(cost["intervention"]))
        assert {"scale_bq_0", "interp_pe_1"}.issubset(set(cost["intervention"]))
        assert np.isfinite(metrics["bos_attn"]).all()
        assert np.isfinite(cost.loc[cost["status"] == "ok", "cross_entropy"]).all()
        assert {"k0", "delta1"} == set(invariance["quantity"])

        # Cache-only validation and aggregation: no model-loading call occurs below.
        for mode in ("metrics", "length", "cost", "content"):
            e5._validate_or_write_config(
                seed_dir / f"run_config_{mode}.json",
                e5._critical_config(args, mode, 0), write=False)
        aggregate = e5.aggregate_and_plot(
            root, (0,), ("metrics", "length", "cost", "content"), 20, 1730)
        e5.aggregate_and_plot(
            root, (0,), ("metrics", "length", "cost", "content"), 20, 1730)
        expected = {
            "fig_E5A_metric_multiverse.png", "fig_E5B_concordance_intensive.png",
            "fig_E5C_length_scaling.png", "fig_E5D_mitigation_frontier.png",
            "fig_E5E_position_damage.png", "fig_E5F_content_generality.png",
            "metrics_summary.csv", "length_summary.csv", "cost_summary.csv",
            "content_summary.csv",
        }
        assert expected.issubset({path.name for path in aggregate.iterdir()})
    print("E5 NNsight offline smoke test passed")


if __name__ == "__main__":
    main()
