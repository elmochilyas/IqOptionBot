"""Unit tests for Phase 35: Broker Clock Freshness & Diagnostics."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from iqoption_research.connectors.iqoption.ws_client import (
    IQOptionWebSocketClient,
    _to_epoch_ns,
)
from iqoption_research.models.clock import (
    BrokerClockDiagnostics,
    BrokerClockSyncStatus,
)
from iqoption_research.timing.instrumentation import TimingRecorder
from iqoption_research.timing.reliability import evaluate
from iqoption_research.timing.taxonomy import (
    BrokerClockDiagnostics as TaxonomyDiagnostics,
)
from iqoption_research.timing.taxonomy import (
    BrokerClockSyncStatus as TaxonomyStatus,
)


def test_taxonomy_reexport() -> None:
    """Verify models.clock items are seamlessly re-exported by timing.taxonomy."""
    assert TaxonomyStatus is BrokerClockSyncStatus
    assert TaxonomyDiagnostics is BrokerClockDiagnostics


def test_broker_clock_diagnostics_sanitization() -> None:
    """Verify BrokerClockDiagnostics serializes cleanly without sensitive fields."""
    diag = BrokerClockDiagnostics(
        status=BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH,
        is_fresh=True,
        age_seconds=1.234567,
        server_time_ns=1795099080000000000,
        last_synced_mono_ns=100_000_000,
        skew_ms=-454.2721,
        total_uncertainty_ms=479.2721,
        step_detected=False,
    )
    d = diag.to_dict()
    assert d["status"] == "SYNCHRONIZED_AND_FRESH"
    assert d["is_fresh"] is True
    assert d["age_seconds"] == 1.2346
    assert d["skew_ms"] == -454.2721
    assert d["total_uncertainty_ms"] == 479.2721
    assert d["step_detected"] is False
    assert "ssid" not in d
    assert "cookie" not in d
    assert "password" not in d


def test_to_epoch_ns_normalization_and_validation() -> None:
    """Verify _to_epoch_ns parses various unit precisions and rejects malformed values."""
    # Valid seconds (2026-11-19 14:38:00 UTC)
    assert _to_epoch_ns(1795099080) == 1795099080000000000
    # Valid milliseconds
    assert _to_epoch_ns(1795099080000) == 1795099080000000000
    # Valid microseconds
    assert _to_epoch_ns(1795099080000000) == 1795099080000000000
    # Valid nanoseconds
    assert _to_epoch_ns(1795099080000000000) == 1795099080000000000

    # Malformed non-numeric values
    assert _to_epoch_ns("not_a_number") is None
    assert _to_epoch_ns(None) is None
    assert _to_epoch_ns(True) is None
    assert _to_epoch_ns(False) is None
    assert _to_epoch_ns({"time": 1795099080}) is None
    assert _to_epoch_ns([1795099080]) is None

    # Invalid numeric values
    assert _to_epoch_ns(-100) is None
    assert _to_epoch_ns(0) is None
    assert _to_epoch_ns(float("nan")) is None
    assert _to_epoch_ns(float("inf")) is None
    assert _to_epoch_ns(float("-inf")) is None

    # Out of plausible epoch range (e.g. year 1970 or year 3000)
    assert _to_epoch_ns(42) is None
    assert _to_epoch_ns(100_000) is None
    assert _to_epoch_ns(10**11) is None  # year 5138 in seconds


@pytest.mark.asyncio
async def test_ws_client_never_synchronized_state() -> None:
    """Client has NEVER_SYNCHRONIZED status before any timeSync frame arrives."""
    client = IQOptionWebSocketClient(ssid="test-ssid")
    diag = client.clock_diagnostics
    assert diag.status == BrokerClockSyncStatus.NEVER_SYNCHRONIZED
    assert diag.is_fresh is False
    assert diag.age_seconds is None
    assert diag.server_time_ns is None
    assert client.server_time_ns is not None  # property has local fallback estimate
    assert client.is_server_time_fresh is False


@pytest.mark.asyncio
async def test_ws_client_valid_time_sync_and_extrapolation() -> None:
    """Incoming valid timeSync frame transitions client to SYNCHRONIZED_AND_FRESH."""
    client = IQOptionWebSocketClient(ssid="test-ssid")
    base_ts_ms = 1795099080000
    client._dispatch({"name": "timeSync", "msg": base_ts_ms}, recv_mono_ns=1_000_000_000)

    diag = client.clock_diagnostics
    assert diag.status == BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH
    assert diag.is_fresh is True
    assert diag.age_seconds is not None
    assert diag.age_seconds >= 0.0
    assert client.is_server_time_fresh is True

    # Verify monotonic extrapolation advances estimated server time
    est1 = client.server_time_ns
    assert est1 is not None
    assert est1 >= 1795099080000000000


@pytest.mark.asyncio
async def test_ws_client_stale_synchronization_expiry() -> None:
    """Synchronization expires and becomes STALE_SYNCHRONIZATION after 120 seconds."""
    client = IQOptionWebSocketClient(ssid="test-ssid")
    base_ts_ms = 1795099080000

    # Dispatch timeSync at virtual monotonic timestamp t=1000.0
    with patch("time.monotonic", return_value=1000.0):
        client._dispatch({"name": "timeSync", "msg": base_ts_ms}, recv_mono_ns=1_000_000_000_000)

    # Within 120s window: at t=1119.9s (age = 119.9s) -> fresh
    with patch("time.monotonic", return_value=1119.9):
        diag = client.clock_diagnostics
        assert diag.status == BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH
        assert diag.is_fresh is True
        assert pytest.approx(diag.age_seconds, abs=0.01) == 119.9

    # Exactly past 120s window: at t=1120.1s (age = 120.1s) -> stale
    with patch("time.monotonic", return_value=1120.1):
        diag = client.clock_diagnostics
        assert diag.status == BrokerClockSyncStatus.STALE_SYNCHRONIZATION
        assert diag.is_fresh is False
        assert pytest.approx(diag.age_seconds, abs=0.01) == 120.1
        assert client.is_server_time_fresh is False

    # Deep in a 5-minute session: at t=1300.0s (age = 300.0s) -> stale
    with patch("time.monotonic", return_value=1300.0):
        diag = client.clock_diagnostics
        assert diag.status == BrokerClockSyncStatus.STALE_SYNCHRONIZATION
        assert diag.is_fresh is False
        assert pytest.approx(diag.age_seconds, abs=0.01) == 300.0


@pytest.mark.asyncio
async def test_ws_client_irregular_updates_and_refresh() -> None:
    """Irregular updates keep status fresh; refresh restores fresh state after stale."""
    client = IQOptionWebSocketClient(ssid="test-ssid")

    # Sync 1 at t=1000.0
    with patch("time.monotonic", return_value=1000.0):
        client._dispatch({"name": "timeSync", "msg": 1795099080000})

    # Sync 2 at t=1050.0 (age was 50s)
    with patch("time.monotonic", return_value=1050.0):
        client._dispatch({"name": "timeSync", "msg": 1795099130000})
        diag = client.clock_diagnostics
        assert diag.status == BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH
        assert pytest.approx(diag.age_seconds, abs=0.01) == 0.0

    # Let clock become stale at t=1180.0 (130s after sync 2)
    with patch("time.monotonic", return_value=1180.0):
        diag = client.clock_diagnostics
        assert diag.status == BrokerClockSyncStatus.STALE_SYNCHRONIZATION
        assert diag.is_fresh is False

    # Fresh update arrives at t=1185.0 -> RECOVERS to fresh
    with patch("time.monotonic", return_value=1185.0):
        client._dispatch({"name": "timeSync", "msg": 1795099265000})
        diag = client.clock_diagnostics
        assert diag.status == BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH
        assert diag.is_fresh is True
        assert pytest.approx(diag.age_seconds, abs=0.01) == 0.0


@pytest.mark.asyncio
async def test_ws_client_invalid_synchronization_payload() -> None:
    """Malformed timeSync payload triggers INVALID_SYNCHRONIZATION_PAYLOAD."""
    client = IQOptionWebSocketClient(ssid="test-ssid")

    # Dispatch malformed payload before any valid sync
    client._dispatch({"name": "timeSync", "msg": "corrupted_timestamp"})
    diag = client.clock_diagnostics
    assert diag.status == BrokerClockSyncStatus.INVALID_SYNCHRONIZATION_PAYLOAD
    assert diag.is_fresh is False
    assert diag.payload_error == "MALFORMED_NON_NUMERIC_OR_NON_POSITIVE_PAYLOAD"

    # Dispatch valid payload -> recovers to fresh
    client._dispatch({"name": "timeSync", "msg": 1795099080000})
    assert client.clock_diagnostics.status == BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH
    assert client.is_server_time_fresh is True

    # Dispatch malformed payload after valid sync -> flags invalid
    client._dispatch({"name": "timeSync", "msg": -500})
    diag = client.clock_diagnostics
    assert diag.status == BrokerClockSyncStatus.INVALID_SYNCHRONIZATION_PAYLOAD
    assert diag.is_fresh is False


@pytest.mark.asyncio
async def test_ws_client_broker_clock_step_detection() -> None:
    """Discontinuous broker timestamp jump (> 5s) triggers CLOCK_STEP_DETECTED."""
    recorder = TimingRecorder("test-clock-step", provenance="OPERATOR_SUPERVISED_LIVE")
    client = IQOptionWebSocketClient(ssid="test-ssid", timing_recorder=recorder)

    # Initial sync at monotonic t=1000.0, broker=1795099080.0
    with patch("time.monotonic", return_value=1000.0):
        client._dispatch({"name": "timeSync", "msg": 1795099080000}, recv_mono_ns=10**12)
        assert client.clock_diagnostics.status == BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH

    # Next sync at monotonic t=1010.0 (10s later), but broker timestamp jumped forward by 60s
    # Expected broker time: 1795099090.0, received: 1795099140.0 (drift = +50,000 ms > 5000 ms)
    with patch("time.monotonic", return_value=1010.0):
        client._dispatch(
            {"name": "timeSync", "msg": 1795099140000},
            recv_mono_ns=10**12 + 10 * 10**9,
        )
        diag = client.clock_diagnostics
        assert diag.status == BrokerClockSyncStatus.CLOCK_STEP_DETECTED
        assert diag.is_fresh is False
        assert diag.step_detected is True
        assert diag.step_details is not None
        assert pytest.approx(diag.step_details["drift_ms"], abs=1.0) == 50000.0
        assert len(recorder.clock.clock_steps_detected) > 0


@pytest.mark.asyncio
async def test_ws_client_wait_for_server_time_sync() -> None:
    """wait_for_server_time_sync passively waits for timeSync without sending any frames."""
    client = IQOptionWebSocketClient(ssid="test-ssid")

    # Timeout case
    res = await client.wait_for_server_time_sync(timeout=0.02)
    assert res is False

    # Successful wait case
    async def push_sync_delayed() -> None:
        await asyncio.sleep(0.01)
        client._dispatch({"name": "timeSync", "msg": 1795099080000})

    task = asyncio.create_task(push_sync_delayed())
    res = await client.wait_for_server_time_sync(timeout=0.1)
    await task
    assert res is True
    assert client.is_server_time_fresh is True


def test_timing_recorder_acquisition_attempt_audit_record() -> None:
    """record_acquisition_attempt preserves sanitized clock status, age, skew and uncertainty."""
    recorder = TimingRecorder("meas-test", provenance="OPERATOR_SUPERVISED_LIVE")
    obs = recorder.start_candle_observation(1795099080, expected_request_id="candles-1")
    recorder.record_request_sent("candles-1", "sendMessage", 100_000_000)

    attempt = recorder.record_acquisition_attempt(
        attempt_number=1,
        request_id="candles-1",
        broker_clock_fresh=False,
        broker_clock_sync_status="NEVER_SYNCHRONIZED",
        broker_clock_age_seconds=None,
        broker_clock_skew_ms=None,
        broker_clock_total_uncertainty_ms=25.0,
        candle_count=22,
        is_complete=True,
        rejection_classification="BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED",
        retry_attempted=False,
    )
    assert attempt is not None
    assert attempt["broker_clock_fresh"] is False
    assert attempt["broker_clock_sync_status"] == "NEVER_SYNCHRONIZED"
    assert attempt["broker_clock_age_seconds"] is None
    assert attempt["broker_clock_skew_ms"] is None
    assert attempt["broker_clock_total_uncertainty_ms"] == 25.0
    assert attempt["rejection_classification"] == "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED"

    # Confirm record is stored in observation
    assert len(obs.acquisition_attempts) == 1
    assert obs.acquisition_attempts[0] == attempt


def test_offline_reliability_failure_modes_separation() -> None:
    """Offline reliability evaluator cleanly separates clock, network, and data failures."""
    from pathlib import Path

    report = evaluate([Path("data/phase34/operator_supervised_live")])

    assert report["unique_live_boundaries"] == 9
    assert "failure_modes" in report
    fm = report["failure_modes"]
    assert fm["network_receipt_failures"] == 0
    assert fm["candle_completeness_failures"] == 1
    assert fm["broker_clock_freshness_failures"] == 1
    assert fm["end_to_end_decision_failures"] == 0
    assert fm["rejection_reasons_breakdown"] == {
        "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED": 1,
        "INVALID_OR_INCOMPLETE_WINDOW": 1,
    }
