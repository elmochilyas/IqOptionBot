"""Integration and scientific regression tests for Phase 14 research."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.ml.phase10 import (
    Phase10Config,
    build_walk_forward_folds,
    prepare_clean_dataset,
)
from iqoption_research.ml.phase14 import (
    ResearchRecommendation,
    compute_causal_atr14,
    compute_wilson_ci,
    run_phase14_multi_horizon_research,
    simulate_financials,
)
from iqoption_research.models.market import Candle, CandleSize


def _make_dummy_candle(
    idx: int, base_ts: int, op: float, hi: float, lo: float, cl: float
) -> Candle:
    from_ts = base_ts + idx * 60
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=from_ts + 60,
        at_ts=from_ts + 60,
        open=op,
        high=hi,
        low=lo,
        close=cl,
        volume=100,
        bid=op,
        ask=op + 0.0001,
    )


def test_wilson_score_interval_properties() -> None:
    """Test mathematical invariants of Wilson score confidence interval."""
    # Empty sample
    empty = compute_wilson_ci(0, 0)
    assert empty.proportion == 0.0
    assert empty.ci_lower == 0.0
    assert empty.ci_upper == 0.0

    # 50 wins out of 100
    ci_50 = compute_wilson_ci(50, 100)
    assert ci_50.proportion == 0.50
    assert 0.39 < ci_50.ci_lower < 0.42
    assert 0.58 < ci_50.ci_upper < 0.61
    assert ci_50.ci_lower < ci_50.proportion < ci_50.ci_upper

    # 0 wins out of 50
    ci_0 = compute_wilson_ci(0, 50)
    assert ci_0.proportion == 0.0
    assert ci_0.ci_lower == 0.0
    assert ci_0.ci_upper > 0.0

    # 50 wins out of 50
    ci_full = compute_wilson_ci(50, 50)
    assert ci_full.proportion == 1.0
    assert ci_full.ci_lower < 1.0
    assert ci_full.ci_upper == 1.0


def test_causal_atr14_zero_lookahead() -> None:
    """Verify that causal ATR computation only accesses prior and current candles."""
    base = 1700000000
    candles: list[Candle] = []
    # Build 25 synthetic candles
    for i in range(25):
        # Range is 2 pips (0.0002)
        candles.append(_make_dummy_candle(i, base, 1.1000, 1.1002, 1.1000, 1.1001))

    atr_map = compute_causal_atr14(candles)
    assert len(atr_map) == 25

    # Candle 0 has range 2.0 pips
    first_to_ts = candles[0].to_ts
    assert pytest.approx(atr_map[first_to_ts], rel=1e-4) == 2.0

    # After 14 identical candles, ATR should equal 2.0 pips
    assert pytest.approx(atr_map[candles[15].to_ts], rel=1e-4) == 2.0

    # Verify that changing a future candle does not alter past ATR values
    modified = list(candles)
    modified[-1] = _make_dummy_candle(24, base, 1.1000, 1.1020, 1.1000, 1.1010)  # huge spike
    mod_atr_map = compute_causal_atr14(modified)

    # All candles prior to index 24 must have identical ATR
    for i in range(24):
        to_ts = candles[i].to_ts
        assert atr_map[to_ts] == mod_atr_map[to_ts]

    # Index 24 ATR should be strictly larger due to the current spike
    assert mod_atr_map[candles[24].to_ts] > atr_map[candles[24].to_ts]


def test_multi_horizon_label_purging() -> None:
    """Verify that walk-forward folds strictly purge h bars between partitions."""
    base = 1700000000
    candles = [
        _make_dummy_candle(i, base, 1.1000 + (i % 5) * 0.0001, 1.1010, 1.0990, 1.1000)
        for i in range(200)
    ]

    for h in (1, 3, 5):
        clean = prepare_clean_dataset(candles, candle_size_seconds=60, horizon_bars=h)
        cfg = Phase10Config(
            n_folds=2,
            horizon_bars=h,
            initial_train_ratio=0.50,
            validation_ratio=0.15,
            test_ratio=0.15,
        )
        folds = build_walk_forward_folds(clean, cfg)

        for fold in folds:
            # 1. Purging between train and validation:
            assert fold.train_end == fold.validation_start - h
            last_train_exit_ts = clean.exit_timestamps[fold.train_end - 1]
            first_val_signal_ts = clean.timestamps[fold.validation_start]
            # Exit of last train label must precede or match first val signal
            assert last_train_exit_ts <= first_val_signal_ts

            # 2. Purging between validation and test:
            assert fold.validation_end == fold.test_start - h
            last_val_exit_ts = clean.exit_timestamps[fold.validation_end - 1]
            first_test_signal_ts = clean.timestamps[fold.test_start]
            assert last_val_exit_ts <= first_test_signal_ts


def test_financial_simulation_binary_and_spot_hurdles() -> None:
    """Verify dual payoff modeling and breakeven hurdle mechanics."""
    n = 100
    y_true = np.ones(n, dtype=np.int64)
    # Predicted probabilities: all high confidence UP
    p_up = np.full(n, 0.60)
    entry_prices = np.full(n, 1.1000)
    # Exit price moves up by 2 pips (0.0002)
    exit_prices = np.full(n, 1.1002)

    # 1. Perfect 100% win rate scenario
    sim = simulate_financials(
        y_true,
        p_up,
        entry_prices,
        exit_prices,
        stake=100.0,
        payout=0.85,
        friction_pips=1.20,
    )
    assert sim.trade_count == 100
    assert sim.win_count == 100
    assert sim.win_rate == 1.0
    # Binary profit: 100 * $85 = +$8,500
    assert sim.binary_net_pnl == 8500.0
    assert sim.binary_profitable is True
    # Spot profit: 100 trades * (2.0p - 1.2p) = +80.0 pips ($80.00)
    assert pytest.approx(sim.spot_gross_pnl_pips, rel=1e-4) == 200.0
    assert pytest.approx(sim.spot_friction_costs_pips, rel=1e-4) == 120.0
    assert pytest.approx(sim.spot_net_pnl_pips, rel=1e-4) == 80.0
    assert sim.spot_profitable is True

    # 2. Sub-hurdle win rate (50% win rate on binary options)
    # Breakeven is 54.054%. At 50%, binary should lose heavily.
    y_half = np.array([1 if i % 2 == 0 else 0 for i in range(n)], dtype=np.int64)
    sim_half = simulate_financials(
        y_half,
        p_up,
        entry_prices,
        exit_prices,
        stake=100.0,
        payout=0.85,
        friction_pips=1.20,
    )
    assert sim_half.win_rate == 0.50
    # Binary: 50 * $85 - 50 * $100 = $4250 - $5000 = -$750
    assert sim_half.binary_net_pnl == -750.0
    assert sim_half.binary_profitable is False


def test_volatility_gating_suppression() -> None:
    """Verify that signals with ATR below threshold are suppressed."""
    n = 10
    y_true = np.ones(n, dtype=np.int64)
    probs = np.full(n, 0.60)
    entries = np.full(n, 1.1000)
    exits = np.full(n, 1.1002)
    # 5 low ATR, 5 high ATR
    atrs = np.array([0.8] * 5 + [1.5] * 5, dtype=np.float64)

    # Gate threshold = 1.2 pips
    sim = simulate_financials(
        y_true,
        probs,
        entries,
        exits,
        atr_values=atrs,
        atr_threshold=1.2,
    )
    # Only the 5 high ATR bars should execute
    assert sim.trade_count == 5


def test_holdout_reservation_enforced(tmp_path: Path) -> None:
    """Verify that Phase 14 research strictly preserves the reserved confirmation holdout."""
    # Check authentic confirmation state file
    state_file = Path("data/phase11/confirmation_state.json")
    assert state_file.exists()
    with state_file.open(encoding="utf-8") as f:
        state = json.load(f)

    assert state["evaluation_status"] == "RESERVED"
    assert state["prior_evaluation_cutoff_ts"] == 1794607200

    # Ensure running with an altered (invalid) state raises an error
    bad_state = tmp_path / "bad_confirmation.json"
    bad_state.write_text(json.dumps({"evaluation_status": "CONSUMED"}), encoding="utf-8")

    with pytest.raises(Exception, match="Holdout integrity error"):
        run_phase14_multi_horizon_research(
            storage_dir="data/datasets",
            confirmation_state_path=bad_state,
            output_report_path=None,
            output_json_path=None,
        )


def test_phase14_cli_command_offline(tmp_path: Path) -> None:
    """Verify the CLI phase14-research command executes cleanly offline."""
    runner = CliRunner()
    report_out = tmp_path / "test_report.md"
    json_out = tmp_path / "test_results.json"

    res = runner.invoke(
        app,
        [
            "ml",
            "phase14-research",
            "--storage",
            "data/datasets",
            "--confirmation-state",
            "data/phase11/confirmation_state.json",
            "--report",
            str(report_out),
            "--json",
            str(json_out),
            "--folds",
            "2",
        ],
    )
    assert res.exit_code == 0
    assert "Phase 14" in res.output
    assert "STOP_STRATEGY" in res.output
    assert report_out.exists()
    assert json_out.exists()

    with json_out.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["recommendation"] == ResearchRecommendation.STOP_STRATEGY.value
    assert data["holdout_candles_preserved"] > 0
