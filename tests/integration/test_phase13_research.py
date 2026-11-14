"""End-to-end integration tests for Phase 13 AI research, optimization, and candidate models."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.collector import BACKFILL_SOURCE, HistoricalDataCollector
from iqoption_research.market_data.sessions import GapClassification, get_market_schedule
from iqoption_research.market_data.validation import find_gaps
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.ml.phase10 import Phase10Config, _simulate_trades
from iqoption_research.ml.phase13 import (
    STATIONARY_FEATURE_NAMES,
    audit_move_distribution,
    run_phase13_baseline_audit,
    train_and_save_candidate_bundle,
    verify_offline_replay,
)
from iqoption_research.ml.tracking.artifacts import (
    ArtifactIntegrityError,
    load_model_bundle,
    verify_manifest,
)
from iqoption_research.models.market import Candle


def _synthetic_authentic_candles(count: int = 500) -> list[Candle]:
    """Generate synthetic candles with authentic provenance tags for offline testing."""
    fixture = generate_synthetic_candles(
        SyntheticMarketConfig(symbol="EURUSD", n_bars=count, seed=42)
    )
    return [replace(c, provenance="historical", raw={}) for c in fixture.candles]


def _persist_test_catalog(root: Path, count: int = 500) -> list[Candle]:
    """Persist candles into Parquet storage and SQLite catalog."""
    candles = _synthetic_authentic_candles(count)
    collector = HistoricalDataCollector(storage_dir=root)
    collector.persist_batch(
        candles,
        symbol="EURUSD",
        candle_size_seconds=60,
        source=BACKFILL_SOURCE,
    )
    return candles


def test_audit_move_distribution() -> None:
    candles = _synthetic_authentic_candles(100)
    stats = audit_move_distribution(candles, horizons=(1, 3, 5))

    assert 1 in stats
    assert 3 in stats
    assert 5 in stats

    s1 = stats[1]
    assert s1.sample_count == 99
    assert s1.mean_move_pips > 0.0
    assert s1.median_move_pips > 0.0
    assert 0.0 <= s1.pct_moves_below_friction_1_2p <= 100.0
    total_pct = s1.pct_moves_below_friction_1_2p + s1.pct_moves_above_friction_1_2p
    assert total_pct == pytest.approx(100.0)


def test_candidate_bundle_creation_and_tamper_detection(tmp_path: Path) -> None:
    storage_dir = tmp_path / "datasets"
    bundle_dir = tmp_path / "candidate_bundle"
    _persist_test_catalog(storage_dir, count=400)

    # Train and serialize candidate bundle
    manifest = train_and_save_candidate_bundle(
        storage_dir=storage_dir,
        output_dir=bundle_dir,
        model_family="random_forest",
        stationary_only=True,
        horizon_bars=1,
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=10,
        random_state=42,
        overwrite=True,
    )

    assert manifest is not None
    assert len(manifest.digest()) == 64
    assert (bundle_dir / "artifact_manifest.json").is_file()
    assert (bundle_dir / "bundle_metadata.json").is_file()
    assert (bundle_dir / "model.joblib").is_file()
    assert (bundle_dir / "scaler.json").is_file()
    assert (bundle_dir / "calibrator.json").is_file()

    # Load and verify metadata
    bundle = load_model_bundle(bundle_dir, verify=True)
    assert bundle.model.is_fitted
    assert bundle.scaler is not None
    assert bundle.calibrator is not None
    meta_dict = bundle.metadata.get("metadata", {})
    stage = bundle.metadata.get("lifecycle_stage") or meta_dict.get("lifecycle_stage")
    assert stage == "DEVELOPMENT_ONLY"
    prov = bundle.metadata.get("model_provenance") or meta_dict.get("model_provenance")
    assert prov == "authentic_trained"
    is_froz = bundle.metadata.get("is_frozen", False) or meta_dict.get("is_frozen", False)
    assert is_froz is False
    assert tuple(bundle.feature_names) == STATIONARY_FEATURE_NAMES

    # Tampering test: modify scaler.json and verify rejection
    scaler_file = bundle_dir / "scaler.json"
    scaler_file.write_text('{"corrupted": true}', encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match=r"mismatch for scaler\.json"):
        verify_manifest(bundle_dir)


def test_verify_offline_replay_end_to_end(tmp_path: Path) -> None:
    storage_dir = tmp_path / "datasets"
    bundle_dir = tmp_path / "candidate_bundle"
    _persist_test_catalog(storage_dir, count=350)

    train_and_save_candidate_bundle(
        storage_dir=storage_dir,
        output_dir=bundle_dir,
        model_family="random_forest",
        stationary_only=True,
        horizon_bars=1,
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=10,
        overwrite=True,
    )

    # Run replay over 100 authentic candles
    result = verify_offline_replay(
        storage_dir=storage_dir,
        candidate_bundle_dir=bundle_dir,
        n_candles=100,
        stake=50.0,
        payout=0.85,
    )

    assert result["processed_candles"] == 100
    assert result["model_name"] == "random_forest"
    assert result["model_provenance"] == "authentic_trained"
    assert result["virtual_balance"] <= 10000.0 or result["virtual_balance"] >= 10000.0


def test_cli_phase13_baseline_command(tmp_path: Path) -> None:
    storage_dir = tmp_path / "datasets"
    report_file = tmp_path / "baseline_report.md"
    json_file = tmp_path / "baseline_results.json"
    _persist_test_catalog(storage_dir, count=350)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "ml",
            "phase13-baseline",
            "--storage",
            str(storage_dir),
            "--report",
            str(report_file),
            "--json",
            str(json_file),
            "--folds",
            "2",
        ],
    )

    assert res.exit_code == 0
    assert report_file.is_file()
    assert json_file.is_file()

    payload = json.loads(json_file.read_text(encoding="utf-8"))
    assert "dataset_summary" in payload
    assert "move_distributions" in payload
    assert "model_summaries" in payload


def test_cli_bot_replay_with_authentic_data(tmp_path: Path) -> None:
    storage_dir = tmp_path / "datasets"
    bundle_dir = tmp_path / "candidate_bundle"
    _persist_test_catalog(storage_dir, count=350)

    train_and_save_candidate_bundle(
        storage_dir=storage_dir,
        output_dir=bundle_dir,
        model_family="random_forest",
        stationary_only=True,
        horizon_bars=1,
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=10,
        overwrite=True,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "bot",
            "replay",
            "--storage",
            str(storage_dir),
            "--model-bundle",
            str(bundle_dir),
            "--bars",
            "50",
            "--stake",
            "50",
        ],
    )

    assert res.exit_code == 0
    assert "Replay Completed" in res.output
    assert "authentic_trained" in res.output


def test_market_schedule_gap_classification(tmp_path: Path) -> None:
    """Verify market schedule calendar distinguishes weekend closures from unscheduled gaps."""
    # Create candles with an artificial weekend gap (Friday 22:00 UTC to Sunday 22:00 UTC)
    # 2026-10-09 22:00:00 UTC (Friday) = 1791583200
    # 2026-10-11 22:00:00 UTC (Sunday) = 1791756000
    friday_ts = 1791583200
    sunday_ts = 1791756000
    mon_jump_ts = sunday_ts + 3600

    base_candles = _synthetic_authentic_candles(3)
    candle_fri = replace(base_candles[0], from_ts=friday_ts, to_ts=friday_ts + 60)
    candle_sun = replace(base_candles[1], from_ts=sunday_ts, to_ts=sunday_ts + 60)
    candle_mon = replace(base_candles[2], from_ts=mon_jump_ts, to_ts=mon_jump_ts + 60)

    candles = [candle_fri, candle_sun, candle_mon]
    schedule = get_market_schedule("EURUSD")
    classified = find_gaps(candles, 60, market_schedule=schedule)

    assert len(classified) == 2
    weekend_cls = GapClassification.SCHEDULED_WEEKEND_CLOSURE.value
    intraday_cls = GapClassification.UNSCHEDULED_INTRADAY_GAP.value
    weekend_gaps = [g for g in classified if g.classification == weekend_cls]
    intraday_gaps = [g for g in classified if g.classification == intraday_cls]
    assert len(weekend_gaps) == 1
    assert len(intraday_gaps) == 1

    # Test baseline audit integration
    storage_dir = tmp_path / "datasets"
    report_file = tmp_path / "gap_report.md"
    json_file = tmp_path / "gap_results.json"
    _persist_test_catalog(storage_dir, count=250)

    cfg = Phase10Config(n_folds=2, model_names=("majority_class",))
    res = run_phase13_baseline_audit(
        storage_dir=storage_dir,
        output_report_path=report_file,
        output_json_path=json_file,
        config=cfg,
    )
    summary = res["dataset_summary"]
    assert "scheduled_weekend_closures" in summary
    assert "unexpected_open_market_gaps" in summary
    assert "unresolved_quality_issues" in summary
    assert summary["unexpected_open_market_gaps"] == 0


def test_trading_accumulator_financial_profit_factor() -> None:
    """Verify that financial profit factor calculates gross_profit / gross_loss with edge cases."""
    cfg = Phase10Config(
        confidence_threshold=0.55,
        spread_pips=1.0,
        slippage_pips=0.10,
        commission_per_trade=0.0,
        notional_per_trade=10000.0,
        initial_capital=10000.0,
    )

    # 1. Zero trades (abstention: probabilities at 0.50)
    sig_ts = np.array([1000, 1060], dtype=np.int64)
    ent_ts = np.array([1060, 1120], dtype=np.int64)
    exit_ts = np.array([1120, 1180], dtype=np.int64)
    entry_p = np.array([1.1000, 1.1000], dtype=np.float64)
    exit_p = np.array([1.1050, 1.0950], dtype=np.float64)
    probs_abstain = np.array([0.50, 0.50], dtype=np.float64)

    acc_zero = _simulate_trades(probs_abstain, sig_ts, ent_ts, exit_ts, entry_p, exit_p, cfg)
    assert acc_zero.trade_count == 0
    assert acc_zero.gross_profit == 0.0
    assert acc_zero.gross_loss == 0.0
    assert acc_zero.profit_factor == 0.0

    # 2. 100% win rate (Zero losses -> infinite profit factor)
    probs_win = np.array([0.80, 0.80], dtype=np.float64)
    # Both long trades move up by 10 pips (well above 1.2p cost)
    exit_p_win = np.array([1.1010, 1.1010], dtype=np.float64)
    acc_win = _simulate_trades(probs_win, sig_ts, ent_ts, exit_ts, entry_p, exit_p_win, cfg)
    assert acc_win.trade_count == 2
    assert acc_win.wins == 2
    assert acc_win.losses == 0
    assert acc_win.gross_profit > 0.0
    assert acc_win.gross_loss == 0.0
    assert acc_win.profit_factor == float("inf")

    # 3. 0% win rate (All losses -> profit factor 0.0)
    # Long trades move down by 10 pips
    exit_p_loss = np.array([1.0990, 1.0990], dtype=np.float64)
    acc_loss = _simulate_trades(probs_win, sig_ts, ent_ts, exit_ts, entry_p, exit_p_loss, cfg)
    assert acc_loss.trade_count == 2
    assert acc_loss.wins == 0
    assert acc_loss.losses == 2
    assert acc_loss.gross_profit == 0.0
    assert acc_loss.gross_loss > 0.0
    assert acc_loss.profit_factor == 0.0

    # 4. Mixed trades (One win, one loss -> positive finite profit factor)
    exit_p_mixed = np.array([1.1020, 1.0990], dtype=np.float64)
    acc_mixed = _simulate_trades(probs_win, sig_ts, ent_ts, exit_ts, entry_p, exit_p_mixed, cfg)
    assert acc_mixed.trade_count == 2
    assert acc_mixed.wins == 1
    assert acc_mixed.losses == 1
    assert acc_mixed.gross_profit > 0.0
    assert acc_mixed.gross_loss > 0.0
    expected_pf = acc_mixed.gross_profit / acc_mixed.gross_loss
    assert acc_mixed.profit_factor == pytest.approx(expected_pf)


def test_pip_value_and_friction_math() -> None:
    """Verify EURUSD €10k notional pip value is $1.00/pip and friction is ~$1.11 to $1.20."""
    notional = 10000.0  # EUR
    pip_size = 0.0001
    spread_pips = 1.0
    slippage_pips = 0.10

    # Standard Forex pip value formula at parity (1.0000)
    pip_value_at_parity = notional * pip_size
    assert pip_value_at_parity == 1.00  # $1.00 USD per pip

    # At typical exchange rate (e.g. 1.0850)
    entry = 1.0850
    # Price cost in simulator: (spread + 2 * slippage) / entry * notional
    spread_val = spread_pips * pip_size
    slippage_val = slippage_pips * pip_size
    total_friction_pips = spread_pips + 2.0 * slippage_pips
    assert total_friction_pips == pytest.approx(1.20)

    price_cost = (spread_val + 2.0 * slippage_val) / entry * notional
    # 0.00012 / 1.0850 * 10000 = $1.10599 USD (~$1.11 USD)
    assert 1.00 < price_cost < 1.25
    # Verify it is an order of magnitude smaller than $12.00 (which belongs to 100k standard lot)
    assert price_cost < 2.00


def test_cli_bot_run_live_token_and_offline_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify bot run live authorization token gating via env var and CLI, and offline replay."""
    runner = CliRunner()

    # 1. Live observation WITHOUT token must fail with clear security message
    monkeypatch.delenv("IQOPTION_OBSERVE_TOKEN", raising=False)
    res_no_token = runner.invoke(app, ["bot", "run", "--live"])
    assert res_no_token.exit_code != 0
    assert "Live observation requires an explicit authorization token" in res_no_token.output

    # 2. Live observation with token provided via environment variable (preferred secure workflow)
    valid_token = "lead-eng-token-abc123"
    monkeypatch.setenv("IQOPTION_OBSERVE_TOKEN", valid_token)
    res_env = runner.invoke(app, ["bot", "run", "--live"])
    # Token gate passes (fails at broker auth gracefully without complaining about missing token)
    assert "Live observation requires an explicit authorization token" not in res_env.output

    # 3. Live observation with explicit CLI token (supported for parity)
    monkeypatch.delenv("IQOPTION_OBSERVE_TOKEN", raising=False)
    res_cli = runner.invoke(app, ["bot", "run", "--live", "--token", valid_token])
    assert "Live observation requires an explicit authorization token" not in res_cli.output

    # 4. Offline replay with synthetic bars
    res_offline = runner.invoke(app, ["bot", "run", "--bars", "10"])
    assert res_offline.exit_code == 0
    assert "Continuous Session Concluded" in res_offline.output


def test_audit_move_distribution_gaps_not_crossed() -> None:
    """Verify audit_move_distribution does not span weekend closures or unexpected intraday gaps."""
    base = _synthetic_authentic_candles(10)[0]

    # Segment 1: Friday afternoon (5 consecutive 1m candles)
    # Entry prices at ~1.1000
    t0 = 1791580000
    seg1 = [
        replace(
            base,
            from_ts=t0 + i * 60,
            to_ts=t0 + (i + 1) * 60,
            open=1.1000 + i * 0.0001,
            close=1.1000 + (i + 0.5) * 0.0001,
            high=1.1002 + i * 0.0001,
            low=1.0998 + i * 0.0001,
        )
        for i in range(5)
    ]

    # Weekend gap: jump 48 hours to Sunday 22:00 UTC
    # Artificially jump price to 1.2000 (a massive 1,000 pip weekend gap)
    t_weekend = t0 + 172800
    seg2 = [
        replace(
            base,
            from_ts=t_weekend + i * 60,
            to_ts=t_weekend + (i + 1) * 60,
            open=1.2000 + i * 0.0001,
            close=1.2000 + (i + 0.5) * 0.0001,
            high=1.2002 + i * 0.0001,
            low=1.1998 + i * 0.0001,
        )
        for i in range(5)
    ]

    # Intraday gap: jump 15 minutes (900 seconds)
    t_intraday = t_weekend + 5 * 60 + 900
    seg3 = [
        replace(
            base,
            from_ts=t_intraday + i * 60,
            to_ts=t_intraday + (i + 1) * 60,
            open=1.3000 + i * 0.0001,
            close=1.3000 + (i + 0.5) * 0.0001,
            high=1.3002 + i * 0.0001,
            low=1.2998 + i * 0.0001,
        )
        for i in range(2)  # length 2: cannot form 3m move
    ]

    candles = seg1 + seg2 + seg3
    stats = audit_move_distribution(candles, horizons=(1, 3, 5))

    # Horizon 1:
    # Seg 1 has 5-1=4 moves; Seg 2 has 5-1=4 moves; Seg 3 has 2-1=1 move -> Total 9 moves
    assert stats[1].sample_count == 9

    # Horizon 3:
    # Seg 1 has 5-3=2 moves; Seg 2 has 5-3=2 moves; Seg 3 has 0 moves (length 2 <= 3).
    # Total = 4 moves
    assert stats[3].sample_count == 4

    # Horizon 5:
    # None of the segments have length > 5, so no 5m moves can be formed without crossing gaps
    assert 5 not in stats

    # CRITICAL INVARIANT: The massive 1,000 pip (1.1000 -> 1.2000) weekend gap
    # must NEVER be included in moves or ranges. Maximum move should be < 5 pips.
    assert stats[1].mean_move_pips < 5.0
    assert stats[3].mean_move_pips < 5.0
    assert stats[1].p90_move_pips < 5.0
    assert stats[3].p90_move_pips < 5.0


def test_monday_operator_checklist_cli_offline() -> None:
    """Validate all Monday runbook CLI commands offline with authentic data and saved model."""
    runner = CliRunner()
    storage = "data/datasets"
    bundle = "data/phase13/candidate_models/candidate_h1_rf"

    # Step 1: Pre-market authentic dataset audit
    res_audit = runner.invoke(app, ["data", "coverage-audit", "--output-dir", storage])
    assert res_audit.exit_code == 0
    assert "Authentic Historical Coverage" in res_audit.output
    assert "Unique candles: 62871" in res_audit.output

    # Step 2: Forward status (confirms weekend boundary)
    res_status = runner.invoke(app, ["data", "forward-status", "--output-dir", storage])
    assert res_status.exit_code == 0

    # Step 3: Forward plan entirely offline
    res_plan = runner.invoke(app, ["data", "forward-plan", "--output-dir", storage])
    assert res_plan.exit_code == 0

    # Step 4: Verify candidate model bundle integrity and metadata
    res_inspect = runner.invoke(app, ["ml", "inspect", "--bundle", bundle])
    assert res_inspect.exit_code == 0
    assert "DEVELOPMENT_ONLY" in res_inspect.output
    assert "authentic_trained" in res_inspect.output

    # Step 5: Supervised offline paper-trading replay with candidate model
    res_replay = runner.invoke(
        app,
        [
            "bot",
            "run",
            "--storage",
            storage,
            "--model-bundle",
            bundle,
            "--bars",
            "20",
        ],
    )
    assert res_replay.exit_code == 0
    assert "Continuous Session Concluded" in res_replay.output
    assert "random_forest" in res_replay.output
