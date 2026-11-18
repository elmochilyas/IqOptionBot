"""Synthetic mathematical and leakage contracts; no empirical research evidence."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from iqoption_research.research.aligned_contract import (
    CONTRACT,
    decision,
    feature_vector,
    outcome_label,
    target_times,
    timing_gate,
)
from iqoption_research.research.aligned_data import build_dataset, chronological_splits
from tests.unit.test_phase24_evaluation import fixture_candles


def candles(n: int = 1600) -> list:
    return [
        replace(c, from_ts=1_700_000_040 + i * 60, to_ts=1_700_000_100 + i * 60)
        for i, c in enumerate(fixture_candles(n))
    ]


@pytest.mark.parametrize(("delta", "label"), [(0.000001, 1), (-0.000001, 0), (0, -1), (1e-10, 1)])
def test_exact_labels(delta: float, label: int) -> None:
    assert outcome_label(1.0, 1.0 + delta) == label


@pytest.mark.parametrize(
    ("observed", "decided", "committed", "valid"),
    [
        (0, 1, 15, True),
        (0, 15, 15.0001, False),
        (-0.1, 0, 1, False),
        (2, 1, 3, False),
        (0, 2, 1, False),
        (15.1, 15.1, 15.1, False),
    ],
)
def test_actual_timing_gate(observed: float, decided: float, committed: float, valid: bool) -> None:
    assert timing_gate(120, 120 + observed, 120 + decided, 120 + committed) is valid


def test_training_and_inference_share_exact_features_and_delayed_target() -> None:
    cs = candles()
    data = build_dataset(cs)
    np.testing.assert_array_equal(data.X[0], feature_vector(cs[:21]))
    for t in (100, 500, 799):
        np.testing.assert_array_equal(data.X[t - 20], feature_vector(cs[t - 20 : t + 1]))
    F = cs[20].to_ts
    assert target_times(F) == (cs[23].from_ts, cs[23].to_ts)
    assert data.y[0] == outcome_label(cs[23].open, cs[23].close)
    assert CONTRACT.to_dict()["probability"].startswith("P(UP | DECISIVE)")


def test_future_prices_cannot_change_features_or_earlier_labels() -> None:
    cs = candles()
    before = build_dataset(cs)
    changed = cs[:800] + [
        replace(c, open=c.open * 1.1, high=c.high * 1.1, low=c.low * 1.1, close=c.close * 1.1)
        for c in cs[800:]
    ]
    after = build_dataset(changed)
    safe = before.expiry <= cs[800].from_ts
    np.testing.assert_array_equal(before.X[safe], after.X[safe])
    np.testing.assert_array_equal(before.y[safe], after.y[safe])


def test_context_and_labels_are_purged_at_every_boundary() -> None:
    data = build_dataset(candles())
    parts = chronological_splits(data)
    groups = list(parts.values())
    for a, b in itertools.pairwise(groups):
        assert data.expiry[a].max() < data.feature_start[b].min()


def test_missing_candles_never_interpolate_or_cross_windows() -> None:
    cs = candles()
    data = build_dataset(cs[:800] + cs[801:])
    removed = cs[800].from_ts
    assert not np.any((data.feature_start <= removed) & (data.expiry > removed))
    assert data.exclusions["insufficient_context"] == 40


def test_threshold_frozen_and_ties_not_directional_losses() -> None:
    assert [decision(p) for p in (0.6, 0.4, 0.5)] == ["CALL", "PUT", "SKIP"]
    cs = [replace(c, close=c.open) for c in candles()]
    assert (build_dataset(cs).y == -1).all()


def test_production_sealed_paths_are_denied_without_reading() -> None:
    from pathlib import Path

    from tests.conftest import SEALED_PATHS

    for path in SEALED_PATHS:
        with pytest.raises(PermissionError, match="SEALED_HOLDOUT"):
            Path(path).read_bytes()
