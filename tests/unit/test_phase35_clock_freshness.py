"""Unit tests for Phase 35: Broker Clock Freshness & Diagnostics."""

from __future__ import annotations

import asyncio
import copy
import time
from pathlib import Path
from typing import Any
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
from iqoption_research.models.connection import DisconnectReason
from iqoption_research.timing.evidence import TimingEvidenceRegistry
from iqoption_research.timing.instrumentation import (
    ClockDiscipline,
    TimingRecorder,
    compute_observation_dict_hash,
)
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


@pytest.mark.asyncio
async def test_ws_client_clock_cleared_on_shutdown_and_reconnect() -> None:
    """Transport shutdown and reconnection must clear clock state completely."""
    client = IQOptionWebSocketClient(ssid="test-ssid")

    # 1. Establish valid sync at virtual t=1000.0
    with patch("time.monotonic", return_value=1000.0):
        client._dispatch({"name": "timeSync", "msg": 1795099080000}, recv_mono_ns=10**12)
        assert client.is_server_time_fresh is True
        assert client.clock_diagnostics.status == BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH

    # 2. Simulate transport shutdown at t=1020.0 (only 20s later, well before 120s window)
    with patch("time.monotonic", return_value=1020.0):
        await client._shutdown_transport(DisconnectReason.NETWORK_ERROR)
        assert client._server_time_ns is None
        assert client._server_time_synced_at is None
        assert client._server_time_synced_at_mono_ns is None
        assert client.clock_diagnostics.status == BrokerClockSyncStatus.NEVER_SYNCHRONIZED
        assert client.is_server_time_fresh is False


@pytest.mark.asyncio
async def test_wait_for_server_time_sync_after_synchronization_expiry() -> None:
    """After 120s expiry, wait_for_server_time_sync does NOT return immediately with stale state."""
    client = IQOptionWebSocketClient(ssid="test-ssid")

    # Initial sync
    client._dispatch({"name": "timeSync", "msg": 1795099080000})
    assert client.is_server_time_fresh is True

    # Simulate 130s elapsed without monkeypatching time.monotonic globally
    client._server_time_synced_at = time.monotonic() - 130.0
    assert client.is_server_time_fresh is False
    assert client.clock_diagnostics.status == BrokerClockSyncStatus.STALE_SYNCHRONIZATION

    # Must NOT return immediately! Must actively await and timeout waiting for a new sync
    res = await client.wait_for_server_time_sync(timeout=0.03)
    assert res is False


@pytest.mark.asyncio
async def test_wait_for_server_time_sync_valid_refresh_during_wait() -> None:
    """Valid refresh pushed during active wait wakes waiter and establishes freshness."""
    client = IQOptionWebSocketClient(ssid="test-ssid")

    # Initial sync, then expire
    client._dispatch({"name": "timeSync", "msg": 1795099080000})
    client._server_time_synced_at = time.monotonic() - 130.0
    assert client.is_server_time_fresh is False

    async def push_refresh() -> None:
        await asyncio.sleep(0.01)
        client._dispatch({"name": "timeSync", "msg": 1795099210000})

    bg_task = asyncio.create_task(push_refresh())
    res = await client.wait_for_server_time_sync(timeout=0.1)
    await bg_task
    assert res is True
    assert client.is_server_time_fresh is True
    assert client.clock_diagnostics.status == BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH


@pytest.mark.asyncio
async def test_wait_for_server_time_sync_invalid_payload_fails_closed_immediately() -> None:
    """Invalid timeSync payload during wait terminates waiter fail-closed immediately."""
    client = IQOptionWebSocketClient(ssid="test-ssid")

    async def push_invalid() -> None:
        await asyncio.sleep(0.01)
        client._dispatch({"name": "timeSync", "msg": "malformed_string"})

    bg_task = asyncio.create_task(push_invalid())
    # Generous timeout 1.0s. It should return in ~0.02s upon receiving the invalid frame.
    res = await client.wait_for_server_time_sync(timeout=1.0)
    await bg_task
    assert res is False
    assert client.clock_diagnostics.status == BrokerClockSyncStatus.INVALID_SYNCHRONIZATION_PAYLOAD
    assert client.is_server_time_fresh is False


@pytest.mark.asyncio
async def test_wait_for_server_time_sync_clock_step_fails_closed_immediately() -> None:
    """Clock step detected during wait terminates waiter fail-closed immediately."""
    client = IQOptionWebSocketClient(ssid="test-ssid")

    # Initial sync, then expire so wait_for_server_time_sync actively waits
    client._dispatch({"name": "timeSync", "msg": 1795099080000})
    client._server_time_synced_at = time.monotonic() - 130.0
    assert client.is_server_time_fresh is False

    async def push_step() -> None:
        await asyncio.sleep(0.01)
        # 500s jump relative to recent sync (>5s drift)
        client._dispatch({"name": "timeSync", "msg": 1795099580000})

    bg_task = asyncio.create_task(push_step())
    res = await client.wait_for_server_time_sync(timeout=1.0)
    await bg_task
    assert res is False
    assert client.clock_diagnostics.status == BrokerClockSyncStatus.CLOCK_STEP_DETECTED
    assert client.is_server_time_fresh is False


@pytest.mark.asyncio
async def test_wait_for_server_time_sync_disconnection_fails_closed_immediately() -> None:
    """Transport disconnection during wait terminates waiter fail-closed immediately."""
    client = IQOptionWebSocketClient(ssid="test-ssid")

    async def trigger_shutdown() -> None:
        await asyncio.sleep(0.01)
        await client._shutdown_transport(DisconnectReason.NETWORK_ERROR)

    bg_task = asyncio.create_task(trigger_shutdown())
    res = await client.wait_for_server_time_sync(timeout=1.0)
    await bg_task
    assert res is False
    assert client.is_server_time_fresh is False


def _create_sample_phase35_registry(
    *,
    status: str = BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH.value,
    fresh: bool = True,
    age: float | None = 1.234,
    skew: float | None = -12.345,
    unc: float | None = 37.345,
) -> TimingEvidenceRegistry:
    """Helper to build a valid TimingEvidenceRegistry containing Phase 35 clock diagnostics."""
    clock = ClockDiscipline(anchor_utc_ns=1795000000_000_000_000)
    recorder = TimingRecorder("p35_test_session", clock=clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000060)
    obs.is_synthetic = True
    obs.provenance = "OFFLINE_SIMULATION"
    recorder.record_stage("STAGE_1_REQUEST_INIT", mono_ns=1795000060_000_000_000)
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=1795000061_000_000_000)
    recorder.record_stage("STAGE_3_PARSE_VALIDATE", mono_ns=1795000061_500_000_000)
    recorder.record_stage("STAGE_4_CANDLE_VALIDATE", mono_ns=1795000062_000_000_000)
    recorder.record_candle_validated(1795000062_000_000_000, candle_count=1)
    recorder.record_acquisition_attempt(
        attempt_number=1,
        request_id="req-p35-001",
        broker_clock_fresh=fresh,
        broker_clock_sync_status=status,
        broker_clock_age_seconds=age,
        broker_clock_skew_ms=skew,
        broker_clock_total_uncertainty_ms=unc,
        candle_count=1,
        is_complete=True,
        retry_attempted=False,
    )
    obs.compute_latencies_and_gate(clock)
    obs.compute_hash()

    return TimingEvidenceRegistry.create(
        session_id="p35_test_session",
        operational_mode="OFFLINE_SIMULATION",
        project_root=Path(),
        clock=clock,
        observations=[obs],
    )


def test_compute_observation_dict_hash_includes_phase35_fields() -> None:
    """Verify compute_observation_dict_hash incorporates all Phase 35 clock fields."""
    base_obs: dict[str, Any] = {
        "observation_index": 1,
        "feature_close_F": 1795000060,
        "feature_close_F_ns": 1795000060_000_000_000,
        "request_id": "req-1",
        "symbol": "EURUSD",
        "candle_size_seconds": 60,
        "is_synthetic": True,
        "provenance": "OFFLINE_SIMULATION",
        "decision": "NOT_MEASURED",
        "acquisition_attempts": [
            {
                "attempt_number": 1,
                "request_id": "req-1",
                "request_mono_ns": 1000,
                "receipt_mono_ns": 2000,
                "validation_mono_ns": 3000,
                "broker_clock_fresh": True,
                "broker_clock_sync_status": "SYNCHRONIZED_AND_FRESH",
                "broker_clock_age_seconds": 1.2345,
                "broker_clock_skew_ms": -12.3456,
                "broker_clock_total_uncertainty_ms": 37.3456,
                "candle_count": 1,
                "is_complete": True,
                "retry_attempted": False,
            }
        ],
    }

    base_hash = compute_observation_dict_hash(base_obs)
    assert isinstance(base_hash, str) and len(base_hash) == 64

    # 1. Modifying broker_clock_sync_status changes the hash
    obs_mod_status = copy.deepcopy(base_obs)
    obs_mod_status["acquisition_attempts"][0]["broker_clock_sync_status"] = "STALE_SYNCHRONIZATION"
    assert compute_observation_dict_hash(obs_mod_status) != base_hash

    # 2. Modifying broker_clock_age_seconds changes the hash
    obs_mod_age = copy.deepcopy(base_obs)
    obs_mod_age["acquisition_attempts"][0]["broker_clock_age_seconds"] = 1.2346
    assert compute_observation_dict_hash(obs_mod_age) != base_hash

    # 3. Modifying broker_clock_skew_ms changes the hash
    obs_mod_skew = copy.deepcopy(base_obs)
    obs_mod_skew["acquisition_attempts"][0]["broker_clock_skew_ms"] = -12.3457
    assert compute_observation_dict_hash(obs_mod_skew) != base_hash

    # 4. Modifying broker_clock_total_uncertainty_ms changes the hash
    obs_mod_unc = copy.deepcopy(base_obs)
    obs_mod_unc["acquisition_attempts"][0]["broker_clock_total_uncertainty_ms"] = 37.3457
    assert compute_observation_dict_hash(obs_mod_unc) != base_hash

    # 5. Deleting any of the four fields changes the hash
    for field_name in (
        "broker_clock_sync_status",
        "broker_clock_age_seconds",
        "broker_clock_skew_ms",
        "broker_clock_total_uncertainty_ms",
    ):
        obs_del = copy.deepcopy(base_obs)
        del obs_del["acquisition_attempts"][0][field_name]
        assert compute_observation_dict_hash(obs_del) != base_hash


def test_compute_observation_dict_hash_preserves_historical_compatibility() -> None:
    """Historical observations lacking Phase 35 clock fields retain their canonical hash."""
    historical_obs: dict[str, Any] = {
        "observation_index": 1,
        "feature_close_F": 1795000060,
        "feature_close_F_ns": 1795000060_000_000_000,
        "request_id": "req-1",
        "symbol": "EURUSD",
        "candle_size_seconds": 60,
        "is_synthetic": True,
        "provenance": "OFFLINE_SIMULATION",
        "decision": "NOT_MEASURED",
        "acquisition_attempts": [
            {
                "attempt_number": 1,
                "request_id": "req-1",
                "request_mono_ns": 1000,
                "receipt_mono_ns": 2000,
                "validation_mono_ns": 3000,
                "broker_clock_fresh": True,
                "candle_count": 1,
                "is_complete": True,
                "retry_attempted": False,
            }
        ],
    }

    h1 = compute_observation_dict_hash(historical_obs)
    h2 = compute_observation_dict_hash(copy.deepcopy(historical_obs))
    assert h1 == h2


def test_evidence_registry_verify_integrity_tampering_detection() -> None:
    """verify_integrity detects any tampering with Phase 35 clock diagnostic fields."""
    reg = _create_sample_phase35_registry()
    valid, errors = reg.verify_integrity()
    assert valid is True
    assert errors == []

    # Tampering with each diagnostic field without recomputing the observation hash fails integrity
    for field_name, tampered_val in [
        ("broker_clock_sync_status", "STALE_SYNCHRONIZATION"),
        ("broker_clock_age_seconds", 9.999),
        ("broker_clock_skew_ms", 100.0),
        ("broker_clock_total_uncertainty_ms", 125.0),
    ]:
        data = copy.deepcopy(reg.to_dict())
        data["observations"][0]["acquisition_attempts"][0][field_name] = tampered_val
        tampered_reg = TimingEvidenceRegistry.from_dict(data)
        is_valid, errs = tampered_reg.verify_integrity()
        assert is_valid is False
        assert any("integrity hash mismatch" in e for e in errs)

    # Deleting any diagnostic field without recomputing hash fails integrity
    for field_name in (
        "broker_clock_sync_status",
        "broker_clock_age_seconds",
        "broker_clock_skew_ms",
        "broker_clock_total_uncertainty_ms",
    ):
        data = copy.deepcopy(reg.to_dict())
        del data["observations"][0]["acquisition_attempts"][0][field_name]
        tampered_reg = TimingEvidenceRegistry.from_dict(data)
        is_valid, errs = tampered_reg.verify_integrity()
        assert is_valid is False
        assert any("integrity hash mismatch" in e for e in errs)


def test_evidence_registry_verify_integrity_consistency_checks() -> None:
    """verify_integrity enforces semantic validity of Phase 35 clock fields even if rehashed."""
    reg = _create_sample_phase35_registry()

    invalid_cases: list[tuple[str, dict[str, Any], str]] = [
        (
            "unrecognized status",
            {"broker_clock_sync_status": "UNKNOWN_SYNC_STATE"},
            "unrecognized broker_clock_sync_status",
        ),
        (
            "fresh=True but status=STALE",
            {
                "broker_clock_fresh": True,
                "broker_clock_sync_status": "STALE_SYNCHRONIZATION",
                "broker_clock_age_seconds": 130.0,
            },
            "broker_clock_fresh=True but broker_clock_sync_status='STALE_SYNCHRONIZATION'",
        ),
        (
            "fresh=False but status=SYNCHRONIZED_AND_FRESH",
            {
                "broker_clock_fresh": False,
                "broker_clock_sync_status": "SYNCHRONIZED_AND_FRESH",
                "broker_clock_age_seconds": 1.0,
            },
            "broker_clock_fresh=False but broker_clock_sync_status='SYNCHRONIZED_AND_FRESH'",
        ),
        (
            "status=SYNCHRONIZED_AND_FRESH but age > 120s",
            {
                "broker_clock_fresh": True,
                "broker_clock_sync_status": "SYNCHRONIZED_AND_FRESH",
                "broker_clock_age_seconds": 120.1,
            },
            "broker_clock_sync_status='SYNCHRONIZED_AND_FRESH' but age 120.1s exceeds 120.0s",
        ),
        (
            "status=STALE_SYNCHRONIZATION but age <= 120s",
            {
                "broker_clock_fresh": False,
                "broker_clock_sync_status": "STALE_SYNCHRONIZATION",
                "broker_clock_age_seconds": 50.0,
            },
            "broker_clock_sync_status='STALE_SYNCHRONIZATION' but age 50.0s <= 120.0s",
        ),
        (
            "status=NEVER_SYNCHRONIZED but age present",
            {
                "broker_clock_fresh": False,
                "broker_clock_sync_status": "NEVER_SYNCHRONIZED",
                "broker_clock_age_seconds": 10.0,
            },
            "broker_clock_sync_status='NEVER_SYNCHRONIZED' but age is present",
        ),
        (
            "negative age",
            {"broker_clock_age_seconds": -0.5},
            "invalid broker_clock_age_seconds: expected finite non-negative number",
        ),
        (
            "non-finite age inf",
            {"broker_clock_age_seconds": float("inf")},
            "invalid broker_clock_age_seconds: expected finite non-negative number",
        ),
        (
            "boolean age",
            {"broker_clock_age_seconds": True},
            "invalid broker_clock_age_seconds: expected finite non-negative number",
        ),
        (
            "non-finite skew nan",
            {"broker_clock_skew_ms": float("nan")},
            "invalid broker_clock_skew_ms: expected finite number",
        ),
        (
            "boolean skew",
            {"broker_clock_skew_ms": False},
            "invalid broker_clock_skew_ms: expected finite number",
        ),
        (
            "negative uncertainty",
            {"broker_clock_total_uncertainty_ms": -1.0},
            "invalid broker_clock_total_uncertainty_ms: expected finite non-negative number",
        ),
        (
            "boolean uncertainty",
            {"broker_clock_total_uncertainty_ms": True},
            "invalid broker_clock_total_uncertainty_ms: expected finite non-negative number",
        ),
    ]

    for label, updates, expected_substring in invalid_cases:
        data = copy.deepcopy(reg.to_dict())
        data["observations"][0]["acquisition_attempts"][0].update(updates)
        # Rehash observation and registry to isolate the semantic consistency check
        data["observations"][0]["integrity_hash"] = compute_observation_dict_hash(
            data["observations"][0]
        )
        tampered_reg = TimingEvidenceRegistry.from_dict(data)
        tampered_reg.integrity_hash = tampered_reg.compute_hash()

        is_valid, errs = tampered_reg.verify_integrity()
        assert is_valid is False, f"Case '{label}' unexpectedly passed verification"
        assert any(expected_substring in e for e in errs), (
            f"Case '{label}' did not produce expected error containing '{expected_substring}'. "
            f"Actual errors: {errs}"
        )


def test_historical_evidence_files_pass_integrity() -> None:
    """Historical Phase 33 and Phase 34 evidence registries verify cleanly with zero errors."""
    for base_dir in (Path("data/phase34/operator_supervised_live"), Path("data/phase33")):
        if not base_dir.is_dir():
            continue
        for file_path in base_dir.rglob("*.json"):
            registry = TimingEvidenceRegistry.load_from_file(file_path)
            valid, errors = registry.verify_integrity()
            assert valid is True, f"Historical file {file_path.name} failed integrity: {errors}"
            assert errors == []
