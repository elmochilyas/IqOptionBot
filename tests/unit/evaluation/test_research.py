"""Mechanics use synthetic fixtures only; test results are not market evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from iqoption_research.research.backtest.payoffs import BinaryPayoffModel
from iqoption_research.research.evaluation_metrics import SimulationPolicy
from iqoption_research.research.phase25 import session_name, shared_clock, unit_payoffs
from iqoption_research.research.phase25_data import aggregate_complete
from iqoption_research.research.phase25_protocol import day_block_inference, register_protocol
from tests.unit.connectors.test_evaluation import fixture_candles, simple_data


def test_registry_reserves_before_evaluation_and_reproductions_do_not_spend_twice(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    first = register_protocol(path, {"seed": 42, "data": "development-only"})
    assert first["slot"] == 1 and first["alpha"] == pytest.approx(0.025)
    assert register_protocol(path, {"seed": 42, "data": "development-only"}) == first
    second = register_protocol(path, {"seed": 43, "data": "development-only"})
    assert second["slot"] == 2
    assert second["alpha"] == pytest.approx(0.05 / 6)
    assert second["total_registered_alpha"] < 0.05
    assert "UNREGISTERED" in second["prior_research"]
    assert not path.with_suffix(".json.lock").exists()


def test_registry_lock_and_tampered_budget_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    register_protocol(path, {"seed": 42})
    lock = path.with_suffix(".json.lock")
    lock.write_text("another process")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        register_protocol(path, {"seed": 43})
    assert path.read_bytes() == before and lock.exists()
    lock.unlink()
    data = json.loads(path.read_text())
    data["runs"][0]["alpha"] = 0.05
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="budget"):
        register_protocol(path, {"seed": 43})
    assert not lock.exists()


def test_new_protocols_cannot_refund_prior_alpha_slots(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    for seed in range(30):
        result = register_protocol(path, {"seed": seed})
    assert result["registered_experiments"] == 30
    assert result["total_registered_alpha"] == pytest.approx(0.05 * (1 - 1 / 31))


def test_complete_aggregation_is_utc_aligned_and_identity_is_preserved() -> None:
    candles = fixture_candles(15)
    offset = (-candles[0].from_ts) % 300
    candles = [replace(c, from_ts=c.from_ts + offset, to_ts=c.to_ts + offset) for c in candles]
    bars, metadata = aggregate_complete(candles, 300)
    assert len(bars) == 3
    assert bars[0].open == candles[0].open
    assert bars[0].close == candles[4].close
    assert bars[0].high == max(c.high for c in candles[:5])
    assert bars[0].volume == sum(c.volume for c in candles[:5])
    assert all(c.symbol == "EURUSD" and c.active_id == 1 for c in bars)
    assert all(c.provenance == "derived_authentic" for c in bars)
    assert metadata["provenance"] == "DERIVED_AUTHENTIC_NOT_BROKER_NATIVE"
    assert bars[0].from_ts % 300 == 0
    missing = candles[:3] + candles[4:]
    incomplete, report = aggregate_complete(missing, 300)
    assert len(incomplete) == 2
    assert report["excluded_partial_bins"] == 1


@pytest.mark.parametrize("defect", ["synthetic", "duplicate", "order", "instrument"])
def test_aggregation_rejects_unverified_or_mixed_input(defect: str) -> None:
    candles = fixture_candles(15)
    if defect == "synthetic":
        candles[0] = replace(candles[0], provenance="synthetic")
    elif defect == "duplicate":
        candles.insert(1, candles[0])
    elif defect == "order":
        candles.reverse()
    else:
        candles[0] = replace(candles[0], symbol="EURUSD_OP", active_id=1861)
    with pytest.raises(ValueError):
        aggregate_complete(candles, 300)


def test_future_aggregation_cannot_revise_past_bars() -> None:
    candles = fixture_candles(30)
    before, _ = aggregate_complete(candles, 300)
    last = candles[-1]
    candles[-1] = replace(last, close=last.close * 1.1, high=last.high * 1.2)
    after, _ = aggregate_complete(candles, 300)
    assert before[:-1] == after[:-1]


def test_shared_clock_is_nonoverlapping_even_when_a_strategy_abstains() -> None:
    data = simple_data([1, -1, 1, 0, -1])
    data = replace(data, exit_timestamps=data.exit_timestamps + 60)
    selected = shared_clock(data, np.arange(5))
    np.testing.assert_array_equal(selected, [0, 2, 4])
    assert np.all(data.timestamps[selected[1:]] >= data.exit_timestamps[selected[:-1]])
    with pytest.raises(ValueError):
        shared_clock(data, np.asarray([1, 0]))


@pytest.mark.parametrize("direction", [-1, 1])
def test_uncensored_unit_payoffs_match_existing_settlement_contract(direction: int) -> None:
    data = simple_data([1, -1, 0, 1])
    policy = SimulationPolicy()
    actual = unit_payoffs(data, np.arange(4), np.full(4, direction), policy)
    payoff = BinaryPayoffModel(payout_rate=policy.payout)
    expected = [
        payoff.settle_trade(
            direction="call" if direction == 1 else "put",
            entry_price=data.entry_prices[i],
            exit_price=data.exit_prices[i],
            stake=1,
            slippage=policy.slippage_price,
        ).net_pnl
        - policy.cost_rate
        for i in range(4)
    ]
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(
        unit_payoffs(data, np.arange(4), np.zeros(4), policy), np.zeros(4)
    )


def test_reference_sessions_follow_dst_and_weekdays() -> None:
    summer = int(datetime(2026, 7, 8, 12, 30, tzinfo=UTC).timestamp())
    winter = int(datetime(2026, 1, 7, 12, 30, tzinfo=UTC).timestamp())
    sunday = int(datetime(2026, 7, 12, 12, 30, tzinfo=UTC).timestamp())
    assert session_name(summer) == "OVERLAP"
    assert session_name(winter) == "LONDON_ONLY"
    assert session_name(sunday) == "OTHER"


def test_joint_day_blocks_detect_planted_mean_but_never_no_trade_edge() -> None:
    rng = np.random.default_rng(25)
    positive = 0.1 + rng.normal(0, 0.03, 100)
    matrix = np.column_stack((positive, np.zeros(100), -positive)) * 100
    counts = np.full_like(matrix, 100)
    result = day_block_inference(matrix, counts, alpha=0.025, seed=1, resamples=1000)
    assert result[0]["simultaneous_lower_bound"] > 0
    assert result[1]["simultaneous_lower_bound"] <= 0
    assert result[2]["simultaneous_lower_bound"] < 0
    assert result == day_block_inference(matrix, counts, alpha=0.025, seed=1, resamples=1000)


def test_adding_hypotheses_cannot_improve_simultaneous_lower_bound() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(0, 1, (60, 8))
    counts = np.full_like(matrix, 100)
    one = day_block_inference(matrix[:, :1], counts[:, :1], alpha=0.025, seed=5)
    many = day_block_inference(matrix, counts, alpha=0.025, seed=5)
    assert many[0]["simultaneous_lower_bound"] <= one[0]["simultaneous_lower_bound"]


@pytest.mark.parametrize("defect", ["shape", "nan", "zero_counts", "few_days"])
def test_invalid_inference_is_not_reported_as_a_significant_result(defect: str) -> None:
    values, counts = np.zeros((20, 2)), np.ones((20, 2))
    if defect == "shape":
        counts = counts[:, :1]
    elif defect == "nan":
        counts[0, 0] = np.nan
    elif defect == "zero_counts":
        counts[:] = 0
    else:
        values, counts = values[:1], counts[:1]
    with pytest.raises(ValueError):
        day_block_inference(values, counts, alpha=0.025, seed=1)


def test_low_monte_carlo_resolution_is_explicit() -> None:
    values = np.ones((20, 2))
    result = day_block_inference(values, values, alpha=0.00001, seed=1, resamples=100)
    assert not result[0]["tail_resolution_adequate"]
