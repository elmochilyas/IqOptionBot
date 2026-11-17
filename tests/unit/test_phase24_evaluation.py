"""Synthetic fixtures test mechanics only; they are never empirical edge evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iqoption_research.ml.authentic_training import AuthenticModelTrainer, AuthenticTrainingConfig
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.ml.models.baselines import MajorityClassModel
from iqoption_research.ml.phase10 import (
    CleanResearchDataset,
    Phase10Config,
    build_walk_forward_folds,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.evaluation_audit import candle_integrity, catalog_coverage
from iqoption_research.research.evaluation_metrics import (
    SimulationPolicy,
    block_interval,
    break_even,
    simulate,
)
from iqoption_research.research.phase24 import (
    EvaluationConfig,
    assert_split,
    evaluate_development,
    model_predictions,
    prepared_data,
    probability_metrics,
    signals,
)


def fixture_candles(count: int = 600) -> list[Candle]:
    fixture = generate_synthetic_candles(
        SyntheticMarketConfig(n_bars=count, seed=24, start_ts=1700000040)
    )
    # Deliberately exercise the storage-normalized contract in unit tests only.
    return [replace(c, provenance="historical", raw={}) for c in fixture.candles]


def cfg(**kwargs: Any) -> EvaluationConfig:
    return EvaluationConfig(
        min_rows=100, min_trades=30, folds=1, models=("majority_class",), **kwargs
    )


def simple_data(outcomes: list[int]) -> CleanResearchDataset:
    n = len(outcomes)
    return CleanResearchDataset(
        X=np.ones((n, 1)),
        y=(np.asarray(outcomes) > 0).astype(int),
        timestamps=np.arange(n) * 60 + 1700000040,
        entry_timestamps=np.arange(n) * 60 + 1700000040,
        exit_timestamps=np.arange(n) * 60 + 1700000100,
        entry_prices=np.ones(n),
        exit_prices=1 + np.asarray(outcomes) * 0.001,
        feature_names=("return_1",),
        segment_count=1,
        dropped_warmup_rows=0,
        dropped_gap_crossing_labels=0,
        fingerprint="unit-fixture",
    )


@pytest.mark.parametrize(("payout", "expected"), [(0.85, 1 / 1.85), (0.7, 1 / 1.7), (1.0, 0.5)])
def test_break_even_tracks_actual_net_return(payout: float, expected: float) -> None:
    assert break_even(payout) == pytest.approx(expected)
    assert break_even(payout, 0.01) > expected


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_payout_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        break_even(value)


def test_payoff_ev_drawdown_profit_factor_streak_and_refund() -> None:
    data = simple_data([1, -1, -1, 0])
    policy = SimulationPolicy(cost_rate=0, slippage_price=0, fixed_stake=10)
    metrics = simulate(data, np.arange(4), np.ones(4), policy, bootstrap=False)
    assert metrics["total_simulated_pnl"] == pytest.approx(-11.5)
    assert metrics["expected_pnl_per_trade"] == pytest.approx(-11.5 / 4)
    assert metrics["expected_unit_return_per_trade"] == pytest.approx(-1.15 / 4)
    assert metrics["max_drawdown"] == pytest.approx(20)
    assert metrics["profit_factor"] == pytest.approx(8.5 / 20)
    assert metrics["longest_losing_streak"] == 2
    assert metrics["refunds"] == 1
    assert metrics["win_rate"] == pytest.approx(1 / 3)


def test_costs_can_destroy_frequent_wins() -> None:
    data = simple_data([1, 1, 1, -1, -1])
    metrics = simulate(
        data,
        np.arange(5),
        np.ones(5),
        SimulationPolicy(
            payout=0.5,
            slippage_price=0,
            cost_rate=0.01,
        ),
        bootstrap=False,
    )
    assert metrics["win_rate"] == 0.6
    assert metrics["total_simulated_pnl"] < 0


def test_overlapping_horizons_and_exposure_are_rejected() -> None:
    data = simple_data([1] * 5)
    data = replace(data, exit_timestamps=data.exit_timestamps + 60)
    metrics = simulate(
        data, np.arange(5), np.ones(5), SimulationPolicy(slippage_price=0), bootstrap=False
    )
    assert metrics["accepted_trades"] == 3
    assert metrics["rejection_reasons"]["overlapping_exposure"] == 2
    capped = simulate(
        data, np.arange(5), np.ones(5), SimulationPolicy(fixed_stake=1000), bootstrap=False
    )
    assert capped["accepted_trades"] == 0


def test_loss_brakes_and_probability_filters_never_increase_stakes() -> None:
    data = simple_data([-1] * 10)
    metrics = simulate(
        data,
        np.arange(10),
        np.ones(10),
        SimulationPolicy(
            slippage_price=0,
            consecutive_loss_limit=2,
        ),
        bootstrap=False,
    )
    assert metrics["accepted_trades"] == 2
    rejected = simulate(
        data,
        np.arange(10),
        np.ones(10),
        SimulationPolicy(
            min_confidence=0.7,
            min_ev=0.02,
        ),
        probabilities=np.full(10, 0.55),
        bootstrap=False,
    )
    assert rejected["accepted_trades"] == 0
    assert rejected["rejection_reasons"]["confidence_filter"] == 10


def test_seeded_block_intervals_and_no_trade_null_metrics() -> None:
    returns = np.tile([0.85, -1.0], 100)
    accepted = np.ones(200)
    assert block_interval(returns, accepted, seed=1) == block_interval(returns, accepted, seed=1)
    data = simple_data([1, -1])
    result = simulate(data, np.arange(2), np.zeros(2), SimulationPolicy())
    assert result["accepted_trades"] == 0
    assert result["win_rate"] is None
    assert result["block_bootstrap_unit_ev_interval"] is None


def test_probability_metrics_calibration_and_validation() -> None:
    metrics = probability_metrics(np.asarray([0, 1]), np.asarray([0.2, 0.8]))
    assert metrics["brier_score"] == pytest.approx(0.04)
    assert metrics["log_loss"] == pytest.approx(-np.log(0.8))
    assert metrics["accuracy"] == 1
    assert metrics["expected_calibration_error"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        probability_metrics(np.asarray([0]), np.asarray([float("nan")]))


@pytest.mark.parametrize(
    "change", ["duplicate", "ordering", "missing", "price", "synthetic", "identity"]
)
def test_candle_integrity_detects_data_defects(change: str) -> None:
    candles = fixture_candles(100)
    if change == "duplicate":
        candles.insert(3, candles[2])
    elif change == "ordering":
        candles[3], candles[4] = candles[4], candles[3]
    elif change == "missing":
        candles.pop(3)
    elif change == "price":
        candles[3] = replace(candles[3], high=0.0)
    elif change == "synthetic":
        candles[3] = replace(candles[3], provenance="synthetic")
    else:
        candles[3] = replace(candles[3], symbol="EURUSD_OP", active_id=1861)
    report = candle_integrity(candles, "EURUSD", 60)
    if change == "missing":
        assert report["counts"]["missing_bars"] == 1
    else:
        assert report["status"] == "INVALID_EVALUATION"


def test_future_price_and_volume_cannot_change_past_features() -> None:
    candles = fixture_candles()
    first = prepared_data(candles, cfg())
    altered = list(candles)
    altered[-1] = replace(
        altered[-1], close=altered[-1].close * 1.1, high=altered[-1].high * 1.2, volume=999999
    )
    second = prepared_data(altered, cfg())
    assert first.feature_names == second.feature_names
    np.testing.assert_array_equal(first.X, second.X)
    assert any(first.y != second.y) or any(first.exit_prices != second.exit_prices)
    assert not any("volume" in name for name in first.feature_names)


def test_forward_labels_latency_and_gap_crossing() -> None:
    candles = fixture_candles()
    clean = prepared_data(candles, cfg(horizon=3, latency_bars=2))
    assert np.all(clean.entry_timestamps >= clean.timestamps + 120)
    assert np.all(clean.exit_timestamps - clean.entry_timestamps == 180)
    assert np.all(clean.exit_timestamps > clean.timestamps)
    removed = candles[:250] + candles[260:]
    segmented = prepared_data(removed, cfg(horizon=3))
    assert segmented.segment_count == 2
    gap_start, gap_end = candles[250].from_ts, candles[260].from_ts
    assert not np.any(
        (segmented.entry_timestamps < gap_start) & (segmented.exit_timestamps > gap_end)
    )


def test_purged_walk_forward_boundaries_and_contamination_rejection() -> None:
    clean = prepared_data(fixture_candles(), cfg(horizon=3))
    folds = build_walk_forward_folds(
        clean,
        Phase10Config(
            n_folds=2,
            initial_train_ratio=0.4,
            validation_ratio=0.1,
            test_ratio=0.25,
            horizon_bars=5,
        ),
    )
    prior_end = -1
    for fold in folds:
        train = np.arange(fold.train_start, fold.train_end)
        val = np.arange(fold.validation_start, fold.validation_end)
        test = np.arange(fold.test_start, fold.test_end)
        assert_split(clean, train, val, test)
        assert test[0] > prior_end
        prior_end = test[-1]
    with pytest.raises(ValueError, match="contamination"):
        assert_split(clean, np.arange(100), np.arange(50, 130), np.arange(140, 160))


def test_test_labels_do_not_affect_model_fit_calibration_or_threshold() -> None:
    configuration = cfg()
    data = prepared_data(fixture_candles(900), configuration)
    train, val, test = np.arange(300), np.arange(305, 500), np.arange(505, 700)
    tuning, predicted, metadata = model_predictions(
        data, train, val, test, "logistic_regression", configuration
    )
    modified = replace(data, y=data.y.copy())
    modified.y[test] = 1 - modified.y[test]
    tuning2, predicted2, metadata2 = model_predictions(
        modified, train, val, test, "logistic_regression", configuration
    )
    np.testing.assert_array_equal(tuning, tuning2)
    np.testing.assert_array_equal(predicted, predicted2)
    assert metadata == metadata2
    assert metadata["calibration_last_row"] < metadata["threshold_tuning_first_row"]


def test_repeatability_small_sample_classification_and_failed_windows_not_hidden() -> None:
    candles = fixture_candles(800)
    configuration = cfg()
    data = prepared_data(candles, configuration)
    first = evaluate_development(data, candles, configuration)
    second = evaluate_development(data, candles, configuration)
    assert first == second
    assert all(
        row["classification"] != "EDGE_SUPPORTED_BY_EVIDENCE" for row in first["leaderboard"]
    )
    assert len(first["folds"]) == 1


def test_unavailable_data_has_no_synthetic_substitute(tmp_path: Path) -> None:
    report = catalog_coverage(tmp_path / "missing")
    assert report["status"] == "INSUFFICIENT_AUTHENTIC_DATA"
    assert report["datasets"] == []


def test_signal_thresholds_are_symmetric() -> None:
    np.testing.assert_array_equal(signals(np.asarray([0.2, 0.5, 0.8]), 0.6), [-1, 0, 1])


def test_legacy_trainer_uses_next_candle_targets_and_purges_boundaries(monkeypatch: Any) -> None:
    candles = fixture_candles(220)
    captured: list[np.ndarray] = []
    original = MajorityClassModel.fit

    def spy(self: Any, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> Any:
        captured.append(y.copy())
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(MajorityClassModel, "fit", spy)
    result = AuthenticModelTrainer(AuthenticTrainingConfig()).train_and_evaluate(
        candles,
        provenance_label="SYNTHETIC_UNIT_TEST_ONLY",
    )
    # Warmup consumes indices 0..19; first feature row is 20, first target is 21.
    forward = np.asarray([int(c.close > c.open) for c in candles[21:]])
    expected_count = int(len(forward) * 0.7) - 1
    np.testing.assert_array_equal(captured[0], forward[:expected_count])
    assert result.provenance == "SYNTHETIC_UNIT_TEST_ONLY"


def test_nonsecond_timestamps_are_reported_without_datetime_overflow() -> None:
    candles = fixture_candles(50)
    candles[0] = replace(candles[0], from_ts=1700000000000, to_ts=1700000060000)
    report = candle_integrity(candles, "EURUSD", 60)
    assert report["status"] == "INVALID_EVALUATION"
    assert report["counts"]["invalid_timestamp_precision"] == 1


def test_refund_costs_raise_decisive_break_even_and_percent_sizing_compounds() -> None:
    data = simple_data([1, 0, -1])
    metrics = simulate(
        data,
        np.arange(3),
        np.ones(3),
        SimulationPolicy(
            cost_rate=0.01,
            slippage_price=0,
            sizing="percent",
            stake_fraction=0.001,
        ),
        bootstrap=False,
    )
    assert metrics["break_even_decisive_win_rate"] == pytest.approx(1.015 / 1.85)
    assert metrics["break_even_no_refund_win_rate"] == pytest.approx(1.01 / 1.85)
    assert metrics["accepted_trades"] == 3
