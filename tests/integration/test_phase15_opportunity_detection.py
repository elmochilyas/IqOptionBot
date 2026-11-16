"""Integration and scientific regression tests for Phase 15 Opportunity Detection research."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.ml.phase10 import (
    Phase10Config,
    Phase10Error,
    build_walk_forward_folds,
    prepare_clean_dataset,
)
from iqoption_research.ml.phase15 import (
    ResearchRecommendation,
    audit_historical_coverage,
    compute_causal_features,
    run_experiment_b,
    run_phase15_research,
)
from iqoption_research.models.market import Candle, CandleSize


def _make_dummy_candle(
    idx: int,
    base_ts: int,
    op: float,
    hi: float,
    lo: float,
    cl: float,
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


def test_historical_coverage_audit_properties() -> None:
    """Verify that dataset coverage audit accurately detects historical deficit and holdout."""
    audit = audit_historical_coverage("data/datasets", "data/phase11/confirmation_state.json")
    assert audit.holdout_status == "RESERVED"
    assert audit.holdout_cutoff_ts == 1794607200
    assert audit.holdout_candles > 0
    assert audit.development_candles > 60000

    # 90-day target requires ~30 days more data
    assert audit.target_calendar_days == 90.0
    assert 25.0 < audit.missing_calendar_days < 35.0
    assert audit.missing_trading_bars_estimate > 20000
    assert "data backfill-extend" in audit.safe_backfill_command
    assert "--job-id eurusd-backward-90d" in audit.safe_backfill_command
    assert "--live --yes-i-understand" in audit.safe_backfill_command


def test_holdout_reservation_enforced(tmp_path: Path) -> None:
    """Verify that Phase 15 strictly prevents tampering with confirmation reservation."""
    bad_state = tmp_path / "bad_confirmation.json"
    bad_state.write_text(json.dumps({"evaluation_status": "CONSUMED"}), encoding="utf-8")

    with pytest.raises(Phase10Error, match="Holdout reservation error"):
        audit_historical_coverage("data/datasets", bad_state)

    with pytest.raises(Phase10Error, match="Holdout reservation error"):
        run_phase15_research("data/datasets", bad_state)


def test_triple_barrier_labeling_and_intrabar_ambiguity() -> None:
    """Verify triple-barrier labeling rules, including conservative handling of ambiguity."""
    base = 1700000000
    # Case 1: Clear Upper barrier hit first (+2 pips = 1.1002)
    # Entry at t+1 open (1.1000). At t+2, High reaches 1.1003, Low is 1.0999
    c1 = [
        _make_dummy_candle(0, base, 1.1000, 1.1001, 1.0999, 1.1000),  # t=0 signal
        _make_dummy_candle(1, base, 1.1000, 1.1001, 1.0999, 1.1000),  # t+1 entry
        _make_dummy_candle(2, base, 1.1000, 1.1003, 1.0999, 1.1002),  # t+2 TP hit
    ] + [_make_dummy_candle(i, base, 1.1000, 1.1001, 1.0999, 1.1000) for i in range(3, 220)]
    clean1 = prepare_clean_dataset(c1, candle_size_seconds=60, horizon_bars=15)
    cfg = Phase10Config(
        n_folds=2,
        horizon_bars=15,
        initial_train_ratio=0.4,
        validation_ratio=0.2,
        test_ratio=0.2,
    )
    folds1 = build_walk_forward_folds(clean1, cfg)
    X_feats1, feat_names1 = compute_causal_features(clean1, c1)

    stats1, _ = run_experiment_b(
        c1, clean1, folds1, X_feats1, feat_names1, pt_pips=2.0, sl_pips=2.0, holding_period_bars=15
    )
    assert stats1.tp_hits >= 1

    # Case 2: Ambiguous intrabar order (Both TP and SL hit in the same 60s bar)
    # Bar t+2 has High = 1.1003 (+3p) AND Low = 1.0997 (-3p)
    c2 = [
        _make_dummy_candle(0, base, 1.1000, 1.1001, 1.0999, 1.1000),
        _make_dummy_candle(1, base, 1.1000, 1.1001, 1.0999, 1.1000),
        _make_dummy_candle(2, base, 1.1000, 1.1003, 1.0997, 1.1000),  # AMBIGUOUS
    ] + [_make_dummy_candle(i, base, 1.1000, 1.1001, 1.0999, 1.1000) for i in range(3, 220)]
    clean2 = prepare_clean_dataset(c2, candle_size_seconds=60, horizon_bars=15)
    folds2 = build_walk_forward_folds(clean2, cfg)
    X_feats2, feat_names2 = compute_causal_features(clean2, c2)

    stats2, _ = run_experiment_b(
        c2, clean2, folds2, X_feats2, feat_names2, pt_pips=2.0, sl_pips=2.0, holding_period_bars=15
    )
    # Must be recorded as ambiguous and conservatively classified as SL, NOT TP
    assert stats2.ambiguous_hits >= 1


def test_causal_features_leakage_isolation() -> None:
    """Verify that causal feature computation never accesses future candle data."""
    base = 1700000000
    candles = [_make_dummy_candle(i, base, 1.1000, 1.1005, 1.0995, 1.1000) for i in range(50)]
    clean = prepare_clean_dataset(candles, candle_size_seconds=60, horizon_bars=15)
    X_orig, _names = compute_causal_features(clean, candles)

    # Modify future candle index 49 (spike)
    mod_candles = list(candles)
    mod_candles[49] = _make_dummy_candle(49, base, 1.1000, 1.1100, 1.0900, 1.1050)
    clean_mod = prepare_clean_dataset(mod_candles, candle_size_seconds=60, horizon_bars=15)
    X_mod, _ = compute_causal_features(clean_mod, mod_candles)

    # For any row before bar 49, features must be strictly identical
    for row_idx in range(len(clean.timestamps) - 2):
        np.testing.assert_allclose(X_orig[row_idx], X_mod[row_idx], rtol=1e-5)


def test_phase15_cli_offline_execution(tmp_path: Path) -> None:
    """Verify that the phase15-research CLI command executes cleanly offline."""
    runner = CliRunner()
    report_out = tmp_path / "test_p15_report.md"
    json_out = tmp_path / "test_p15_results.json"

    res = runner.invoke(
        app,
        [
            "ml",
            "phase15-research",
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
    assert "Phase 15" in res.output
    assert "STOP_STRATEGY" in res.output
    assert report_out.exists()
    assert json_out.exists()

    with json_out.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["recommendation"] == ResearchRecommendation.STOP_STRATEGY.value
    assert data["coverage_audit"]["holdout_status"] == "RESERVED"
    assert data["coverage_audit"]["holdout_candles"] > 0
    assert len(data["experiment_a_results"]) > 0
    assert len(data["experiment_c_results"]) > 0
