"""Unit tests for Phase 6 market observation architecture.

Covers clock drift detection, feed heartbeat cadence monitoring,
reconnection state resynchronization, and the market observation adapter.
"""

from __future__ import annotations

import pytest

from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.observation.adapter import (
    MarketObservationAdapter,
    ObservationAuthorizationToken,
    ObservationMode,
    ObservationSecurityError,
)
from iqoption_research.observation.clock import (
    ClockDriftValidator,
    ClockDriftVerdict,
)
from iqoption_research.observation.heartbeat import (
    FeedHealthStatus,
    FeedHeartbeatMonitor,
)
from iqoption_research.observation.reconnect import (
    ReconnectResynchronizer,
    ResyncStatus,
)
from iqoption_research.streaming.events import StreamValidationVerdict


def _make_candle(
    symbol: str = "EURUSD",
    from_ts: int = 1700000000,
    to_ts: int = 1700000060,
    open_price: float = 1.0850,
    high: float = 1.0855,
    low: float = 1.0845,
    close: float = 1.0852,
    is_complete: bool = True,
) -> Candle:
    return Candle(
        active_id=1,
        symbol=symbol,
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=to_ts,
        at_ts=int(to_ts * 1_000_000_000),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=100,
        bid=close,
        ask=close,
        is_complete=is_complete,
    )


def test_clock_drift_validator_synchronized() -> None:
    validator = ClockDriftValidator(warn_drift_seconds=5.0, max_drift_seconds=30.0)
    now = 1700000062.0
    res = validator.evaluate(1700000060.0, current_local_ts=now)
    assert res.verdict == ClockDriftVerdict.SYNCHRONIZED
    assert res.is_acceptable is True
    assert res.drift_seconds == 2.0


def test_clock_drift_validator_warning_and_excessive() -> None:
    validator = ClockDriftValidator(warn_drift_seconds=5.0, max_drift_seconds=30.0)

    # Warning drift (6.0s > 5.0s, but <= 30.0s)
    res_warn = validator.evaluate(1700000060.0, current_local_ts=1700000066.0)
    assert res_warn.verdict == ClockDriftVerdict.WARNING_DRIFT
    assert res_warn.is_acceptable is True

    # Excessive drift (35.0s > 30.0s) -> rejected
    res_exc = validator.evaluate(1700000060.0, current_local_ts=1700000095.0)
    assert res_exc.verdict == ClockDriftVerdict.EXCESSIVE_DRIFT
    assert res_exc.is_acceptable is False

    # Future candle (-5.0s < -1.0s) -> rejected
    res_fut = validator.evaluate(1700000060.0, current_local_ts=1700000055.0)
    assert res_fut.verdict == ClockDriftVerdict.FUTURE_CANDLE
    assert res_fut.is_acceptable is False


def test_clock_drift_replay_bypassed() -> None:
    validator = ClockDriftValidator()
    # Candle from years ago
    res = validator.evaluate(1500000000.0, current_local_ts=1700000000.0, is_replay=True)
    assert res.verdict == ClockDriftVerdict.SYNCHRONIZED
    assert res.is_acceptable is True


def test_feed_heartbeat_monitor_transitions() -> None:
    monitor = FeedHeartbeatMonitor(candle_interval_seconds=60.0)

    # Initial state
    h0 = monitor.evaluate_health(current_ts=1700000000.0)
    assert h0.status == FeedHealthStatus.UNINITIALIZED

    # First candle arrival
    monitor.record_candle_arrival(1700000060.0, arrival_ts=1700000061.0)
    assert monitor.candles_received_count == 1

    # Healthy check 20s later
    h1 = monitor.evaluate_health(current_ts=1700000081.0)
    assert h1.status == FeedHealthStatus.HEALTHY

    # Degraded check (100s elapsed >= 60*1.5=90s)
    h2 = monitor.evaluate_health(current_ts=1700000161.0)
    assert h2.status == FeedHealthStatus.DEGRADED

    # Stale check (160s elapsed >= 60*2.5=150s)
    h3 = monitor.evaluate_health(current_ts=1700000221.0)
    assert h3.status == FeedHealthStatus.STALE

    # Disconnected check (310s elapsed >= 300s)
    h4 = monitor.evaluate_health(current_ts=1700000371.0)
    assert h4.status == FeedHealthStatus.DISCONNECTED


def test_reconnect_resynchronizer_warmup_calculation() -> None:
    resync = ReconnectResynchronizer(warmup_lookback_bars=21, candle_size_seconds=60)
    assert resync.is_resynchronized is True

    # Disconnect at t=1000
    resync.record_disconnect(timestamp=1000.0)
    assert resync.is_resynchronized is False

    # Reconnect 300s later (missed ~5 bars)
    ev = resync.record_reconnect(timestamp=1300.0, last_candle_to_ts=960)
    assert ev.reconnect_index == 1
    assert ev.downtime_seconds == 300.0
    assert ev.bars_required_for_warmup == 26  # 21 + 5 missed bars
    assert ev.status == ResyncStatus.NEEDS_WARMUP
    assert resync.is_resynchronized is False

    # Mark synchronized
    resync.mark_synchronized(latest_candle_to_ts=1380)
    assert resync.is_resynchronized is True


def test_observation_adapter_replay_valid_and_incomplete() -> None:
    adapter = MarketObservationAdapter(mode=ObservationMode.OFFLINE_REPLAY, symbol="EURUSD")

    c1 = _make_candle(from_ts=1000, to_ts=1060, is_complete=True)
    ev1 = adapter.process_incoming_candle(c1, current_now_ts=1060.0)
    assert ev1.is_valid is True
    assert adapter.accepted_count == 1

    # Incomplete candle rejected
    c_inc = _make_candle(from_ts=1060, to_ts=1120, is_complete=False)
    ev_inc = adapter.process_incoming_candle(c_inc, current_now_ts=1120.0)
    assert ev_inc.is_valid is False
    assert ev_inc.validation.verdict == StreamValidationVerdict.INCOMPLETE
    assert adapter.rejected_count == 1


def test_observation_adapter_symbol_identity_separation() -> None:
    adapter = MarketObservationAdapter(mode=ObservationMode.OFFLINE_REPLAY, symbol="EURUSD")

    # Regular EURUSD accepted
    c_std = _make_candle(symbol="EURUSD", from_ts=1000, to_ts=1060)
    assert adapter.process_incoming_candle(c_std, current_now_ts=1060.0).is_valid is True

    # OTC EURUSD-OTC rejected as symbol mismatch against standard EURUSD
    c_otc = _make_candle(symbol="EURUSD-OTC", from_ts=1060, to_ts=1120)
    ev_otc = adapter.process_incoming_candle(c_otc, current_now_ts=1120.0)
    assert ev_otc.is_valid is False
    assert ev_otc.validation.verdict == StreamValidationVerdict.SYMBOL_MISMATCH


def test_observation_adapter_observe_only_requires_token() -> None:
    # Unauthenticated OBSERVE_ONLY must be rejected
    adapter_unauth = MarketObservationAdapter(mode=ObservationMode.OBSERVE_ONLY)
    with pytest.raises(ObservationSecurityError, match="explicit human authorization"):
        adapter_unauth.assert_mode_permission()

    # Valid token permits check
    token = ObservationAuthorizationToken(
        token_hash="tok_auth_99182",
        authorized_by="LeadRiskOfficer",
    )
    adapter_auth = MarketObservationAdapter(
        mode=ObservationMode.OBSERVE_ONLY,
        authorization_token=token,
    )
    # Should not raise
    adapter_auth.assert_mode_permission()
    assert adapter_auth.is_observe_only_authorized is True
