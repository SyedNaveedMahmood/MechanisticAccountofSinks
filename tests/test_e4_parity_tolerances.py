"""Model-free unit coverage for E4 parity tolerance selection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "common"))

from residual_sink_analysis import _e4_parity_tolerances, _parity_row


def test_share_delta_uses_quantity_specific_rtol_floor() -> None:
    assert _e4_parity_tolerances(
        "decomposition/share_delta", 1e-5, 1e-4) == (1e-5, 5e-4)
    assert _e4_parity_tolerances(
        "decomposition/share_delta", 7e-6, 8e-4) == (7e-6, 8e-4)


def test_other_quantities_preserve_requested_tolerances() -> None:
    assert _e4_parity_tolerances(
        "decomposition/attn_full", 1e-5, 1e-4) == (1e-5, 1e-4)
    assert _e4_parity_tolerances(
        "relocation/swap_pos0", 3e-6, 2e-4) == (3e-6, 2e-4)


def test_parity_row_records_actual_selected_tolerances() -> None:
    row = _parity_row(
        "decomposition/share_delta",
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        atol=9e-6,
        rtol=1e-4,
    )
    assert row["atol"] == 9e-6
    assert row["rtol"] == 5e-4
