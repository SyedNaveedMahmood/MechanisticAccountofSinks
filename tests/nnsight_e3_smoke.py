"""Offline E3 smoke test. Intended for the remote Windows RTX machine; do not mock NNsight."""

from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))

from emergence_dynamics import emergence_dynamics_analysis as e3
from intervention_analysis import compute_band
from nnsight_smoke_utils import (
    assert_nnsight_metadata,
    create_tiny_checkpoint,
    synthetic_text,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sinks_e3_nnsight_") as temp:
        base = Path(temp)
        fixtures = [base / "checkpoint_000", base / "checkpoint_100"]
        tokenizer = create_tiny_checkpoint(fixtures[0], 13)
        create_tiny_checkpoint(fixtures[1], 29)
        root = base / "outputs" / "e3_smoke"
        run_id = "offline/e3-smoke-run"
        sampled = {"synthetic": [synthetic_text(8), synthetic_text(8, 9)]}
        args = SimpleNamespace(
            engine="nnsight", dtype="float32", layer_mode="scaled",
            data_seed=0, sample_size=2, cut_length=8)

        for step, checkpoint in zip((0, 100), fixtures):
            model, _, nn_engine = e3.load_checkpoint(
                str(checkpoint), torch.device("cpu"), dtype=torch.float32,
                revision=None, load_tokenizer=False, tokenizer=tokenizer,
                engine="nnsight", local_files_only=True)
            band = compute_band(len(model.transformer.h), "scaled")
            row = e3.measure_checkpoint(
                model, tokenizer, sampled, torch.device("cpu"), band,
                e3.ALL_MODES, 8, engine="nnsight", nn_engine=nn_engine)
            row.update({"run": e3._run_short(run_id), "run_id": run_id,
                        "step": step, "revision": f"checkpoint-{step}"})
            required = {
                "sink_strength", "delta_dominance", "query_align_pos0",
                "efficacy_nullify_bq", "efficacy_remove_first_pe",
                "efficacy_zero_top3_wk", "massive_coords",
            }
            assert required.issubset(row)
            numeric = [row[key] for key in required if key != "massive_coords"]
            assert np.isfinite(np.asarray(numeric, dtype=float)).all()

            out = e3._ckpt_dir(root, run_id, step)
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            requested = e3._e3_critical_config(args, run_id, f"checkpoint-{step}", step)
            provenance = e3._e3_engine_provenance(
                "nnsight", model, nn_engine=nn_engine, model_name=str(checkpoint),
                revision=f"checkpoint-{step}", dtype="float32",
                device=torch.device("cpu"), band=band)
            e3._validate_e3_config(
                out / "run_config.json", requested, write=True,
                extra={"engine": provenance, "band_start": band[0],
                       "band_end": band[1], "num_layers": len(model.transformer.h),
                       "massive_coords": row["massive_coords"]})
            assert_nnsight_metadata(json.loads((out / "run_config.json").read_text(encoding="utf-8")))

            del nn_engine, model
            gc.collect()
            shutil.rmtree(checkpoint)
            assert not checkpoint.exists(), "checkpoint fixture was not releasable after tracing"

        aggregate = e3.aggregate_and_plot(root, e3.ALL_MODES)
        e3.validate_e3_plot_cache(root, args)
        e3.aggregate_and_plot(root, e3.ALL_MODES)  # plot-only-compatible second pass
        expected = {
            "measurements_long.csv", "trajectories.csv", "onsets.csv",
            "fig_E3A_trajectories.png", "fig_E3B_ordering.png",
            "fig_E3C_universality.png", "fig_E3D_efficacy.png",
            "fig_E3E_two_pathway.png",
        }
        assert expected.issubset({path.name for path in aggregate.iterdir()})
    print("E3 NNsight offline smoke test passed")


if __name__ == "__main__":
    main()
