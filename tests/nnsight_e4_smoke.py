"""Offline E4 smoke test covering every mode through actual local NNsight traces."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "common"))

import residual_sink_analysis as e4
from intervention_analysis import compute_band
from nnsight_engine import ARCH_SPECS, NNsightEngine, load_nnsight_model
from nnsight_smoke_utils import (
    assert_nnsight_metadata,
    create_tiny_checkpoint,
    synthetic_text,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sinks_e4_nnsight_") as temp:
        base = Path(temp)
        checkpoint = base / "tiny_gpt2"
        tokenizer = create_tiny_checkpoint(checkpoint, 41)
        lm = load_nnsight_model(
            ARCH_SPECS["gpt2"], str(checkpoint), dtype=torch.float32,
            tokenizer=tokenizer, device=torch.device("cpu"), local_files_only=True)
        model = lm._model
        nn_engine = NNsightEngine(lm, ARCH_SPECS["gpt2"])
        band = compute_band(len(model.transformer.h), "scaled")
        massive = e4.identify_massive_coords(model, torch.device("cpu"))
        executor = e4.ResidualExecutor(model, "nnsight", nn_engine, massive)
        sampled = {"synthetic": [synthetic_text(8), synthetic_text(8, 17)]}
        alphas = [0.0, 1.0]

        results = e4.run_all_modes(
            model, tokenizer, sampled, e4.ALL_MODES, alphas, massive,
            True, band=band, executor=executor)
        assert set(e4.ALL_MODES).issubset(results)
        assert "perplexity" in results
        assert set(results["dose_response"]["knob"]) == set(e4.KNOB_LABELS)
        assert set(results["dose_response"]["alpha"]) == set(alphas)
        assert np.isfinite(results["dose_response"]["bos_attention"]).all()
        assert np.isfinite(results["perplexity"]["cross_entropy"]).all()
        assert results["relocation"]["swap_pos1"] >= 0.0

        root = base / "outputs" / "e4_smoke"
        seed_dir = e4._seed_dir(root, 0)
        e4.save_seed_results(results, seed_dir)
        args = SimpleNamespace(
            engine="nnsight", model_name=str(checkpoint), revision=None, dtype="float32",
            layer_mode="scaled", sample_size=2, cut_length=8,
            _parsed_alphas=alphas)
        provenance = e4.engine_provenance(
            "nnsight", model, model_name=str(checkpoint), dtype="float32",
            device=torch.device("cpu"), band=band, nn_engine=nn_engine)
        requested = e4._critical_e4_config(args, 0)
        modes = [*e4.ALL_MODES, "perplexity"]
        for mode in modes:
            e4._validate_e4_config(
                seed_dir / f"run_config_{mode}.json", requested, write=True,
                extra={"engine": provenance, "mode": mode,
                       "band_start": band[0], "band_end": band[1],
                       "num_layers": len(model.transformer.h),
                       "massive_coords": massive})
            e4._validate_e4_config(
                seed_dir / f"run_config_{mode}.json", requested, write=False)
        general = {**requested, "engine": provenance, "modes": modes}
        (seed_dir / "run_config.json").write_text(
            json.dumps(general, indent=2), encoding="utf-8")
        assert_nnsight_metadata(general)

        aggregate = e4.aggregate_and_plot(root, [0], e4.ALL_MODES, alphas, True)
        expected = {
            "dose_response_summary.csv", "decomposition_summary.csv",
            "combined_summary.csv", "surgical_summary.csv",
            "relocation_summary.csv", "perplexity_summary.csv",
            "dose_response.png", "query_alignment_hist.png", "delta_share_heatmap.png",
        }
        assert expected.issubset({path.name for path in aggregate.iterdir()})
        cells = dict(np.load(seed_dir / "decomposition_cells.npz"))
        assert all(value.shape == (1, model.config.n_head) for value in cells.values())
    print("E4 NNsight offline smoke test passed")


if __name__ == "__main__":
    main()
