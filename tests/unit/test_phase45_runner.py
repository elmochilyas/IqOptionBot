from __future__ import annotations

from datetime import UTC, datetime

from iqoption_research.ml.targets.specs import TiePolicy
from iqoption_research.ml.targets_v3 import MultiHorizonTargetEngine
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.phase45.runner import (
    Phase45Progress,
    build_phase45_plan,
)


def _candles(count: int, timeframe_seconds: int) -> list[Candle]:
    start = 1_700_000_000
    rows: list[Candle] = []
    for index in range(count):
        from_ts = start + index * timeframe_seconds
        rows.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize(timeframe_seconds),
                from_ts=from_ts,
                to_ts=from_ts + timeframe_seconds,
                at_ts=from_ts * 1_000_000_000,
                open=1.10 + index * 0.0001,
                high=1.1002 + index * 0.0001,
                low=1.0998 + index * 0.0001,
                close=1.1001 + index * 0.0001,
                volume=10,
                bid=1.0999,
                ask=1.1002,
            )
        )
    return rows


def test_phase45_plan_uses_explicit_timeframe_horizon_names() -> None:
    plan = build_phase45_plan(max_trials=60)
    names = {trial.horizon_name for trial in plan}
    assert {"M1_H15", "M5_H3", "M15_H1"}.issubset(names)
    assert len(plan) <= 60
    assert len({trial.trial_id for trial in plan}) == len(plan)

    durations = {(t.horizon_name, t.prediction_duration_minutes) for t in plan}
    assert ("M1_H15", 15.0) in durations
    assert ("M5_H3", 15.0) in durations
    assert ("M15_H1", 15.0) in durations


def test_phase45_plan_includes_timeframe_ablations() -> None:
    presets = {t.feature_preset for t in build_phase45_plan()}
    assert {"m1_only", "m1_long_history", "m1_m5", "m1_m15", "m1_m5_m15"}.issubset(presets)


def test_progress_records_safety_disposition() -> None:
    payload = Phase45Progress(
        completed_trials=1,
        total_trials=2,
        active_model="p45_001",
        elapsed_seconds=1.5,
        cancelled=True,
        message="cancelled",
    ).to_dict()
    assert payload["disposition"] == "NO_DEMONSTRATED_EDGE"
    assert payload["execution_authority"] == "NONE"
    assert payload["cancelled"] is True


def test_label_maturity_boundary_excludes_late_labels() -> None:
    candles = _candles(30, 60)
    boundary = candles[20].to_ts
    bundle = MultiHorizonTargetEngine.generate_horizon_targets(
        candles,
        horizon_bars=5,
        tie_policy=TiePolicy.EXCLUDE,
        warmup_bars=0,
        enforce_temporal_continuity=True,
        timeframe_seconds=60,
        horizon_name="M1_H5",
        max_label_maturity_ts=boundary,
    )
    assert bundle.horizon_name == "M1_H5"
    assert bundle.boundary_invalidations_count > 0
    for index, is_valid in enumerate(bundle.valid_mask):
        if is_valid:
            assert candles[index + 5].to_ts < boundary


def test_boundary_iso_value_matches_epoch() -> None:
    from iqoption_research.ml.preflight import DEFAULT_PROTECTED_HOLDOUT_START_TS

    assert datetime.fromtimestamp(DEFAULT_PROTECTED_HOLDOUT_START_TS, tz=UTC).isoformat() == (
        "2026-11-12T10:00:00+00:00"
    )
