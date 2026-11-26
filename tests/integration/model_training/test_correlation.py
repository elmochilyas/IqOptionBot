"""Comprehensive offline integration tests for Phase 33 request correlation and timing readiness.

Tests the complete pipeline:
Request creation -> actual request ID -> WebSocket frame receipt -> request correlation
-> candle validation -> measurement result -> immutable evidence verification.

Validates:
1. Reproduction of old request ID mismatch where expected_request_id="candles-{F}" caused rejection.
2. Genuine sequential request IDs ("candles-1", "candles-2") across two completed candles.
3. Interleaved messages (heartbeat, timeSync, profile, balances) do not contaminate stages.
4. Mismatched ("candles-999") and missing request IDs are rejected without contamination.
5. Response timeout triggers fail-closed without stage corruption.
6. Delayed candle availability (broker candle to_ts < F) fails closed cleanly.
7. Connection interruption fails closed and preserves valid partial evidence registry.
8. CLI command validation ensuring documented flags remain valid and unsupported flags are rejected.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import (
    ConnectionError,
    RequestTimeoutError,
)
from iqoption_research.connectors.iqoption.protocol import (
    MSG_AUTHENTICATED,
    MSG_BALANCES,
    MSG_CANDLES,
    MSG_HEARTBEAT,
    MSG_PROFILE,
)
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.timing.evidence import TimingEvidenceRegistry
from iqoption_research.timing.instrumentation import ClockDiscipline, TimingRecorder


class MockWebSocketConnection:
    """Mock WebSocket client connection yielding predefined frames."""

    def __init__(self, messages: list[str] | None = None) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        if messages:
            for m in messages:
                self._queue.put_nowait(m)
        self.sent_frames: list[dict[str, Any]] = []
        self.send = AsyncMock(side_effect=self._record_send)
        self._is_closed = False

    async def _record_send(self, data: str | bytes) -> None:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        payload = json.loads(data)
        self.sent_frames.append(payload)

    def push_message(self, message: dict[str, Any] | str) -> None:
        raw = json.dumps(message) if isinstance(message, dict) else message
        self._queue.put_nowait(raw)

    async def close(self, *args: Any, **kwargs: Any) -> None:
        self._is_closed = True
        self._queue.put_nowait(None)

    def __aiter__(self) -> MockWebSocketConnection:
        return self

    async def __anext__(self) -> str:
        msg = await self._queue.get()
        if msg is None:
            raise StopAsyncIteration
        return msg


def make_standard_handshake(server_time_s: int) -> list[str]:
    """Standard handshake sequence: authenticated, profile, balances, timeSync."""
    return [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({
            "name": MSG_PROFILE,
            "msg": {
                "user_id": 1001,
                "email": "operator@example.com",
                "first_name": "Operator",
                "balance_id": "b_practice_100",
            },
        }),
        json.dumps({
            "name": MSG_BALANCES,
            "msg": [
                {"id": "b_practice_100", "type": 4, "amount": 10000.0, "currency": "USD"},
            ],
        }),
        json.dumps({"name": "timeSync", "msg": int(server_time_s * 1000)}),
    ]


def make_candles_frame(
    request_id: str,
    to_ts: int,
    size: int = 60,
    active_id: int = 1,
) -> dict[str, Any]:
    """Build a valid broker candles response frame matching get_candles schema."""
    return {
        "name": MSG_CANDLES,
        "request_id": request_id,
        "msg": {
            "active_id": active_id,
            "candles": [
                {
                    "id": 101,
                    "from": to_ts - size,
                    "to": to_ts,
                    "open": 1.08500,
                    "close": 1.08525,
                    "min": 1.08480,
                    "max": 1.08550,
                    "volume": 25,
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# Test 1: Reproduce old request-ID mismatch
# ---------------------------------------------------------------------------


def test_reproduce_request_id_mismatch_old_design_failure() -> None:
    """Verify that fabricating expected_request_id='candles-{F}' causes frame rejection.

    This explicitly reproduces the bug identified in source review:
    supervised_session set expected_request_id='candles-1795000060', while
    AsyncIQOption generated 'candles-1'. record_correlated_frame rejected the frame.
    """
    clock = ClockDiscipline()
    recorder = TimingRecorder("reproduce_bug", clock=clock, enable_clock_stability_check=False)

    now_epoch = int(time.time())
    f_boundary = (now_epoch // 60) * 60
    # Old faulty pattern: guessing candles-{F}
    obs = recorder.start_candle_observation(f_boundary, expected_request_id=f"candles-{f_boundary}")

    # Actual broker connector generates sequential ID: candles-1
    actual_connector_request_id = "candles-1"

    # Frame arrives with the actual connector request ID
    result = recorder.record_correlated_frame(
        request_id=actual_connector_request_id,
        frame_name="candles",
        recv_mono_ns=time.perf_counter_ns(),
        parsed_mono_ns=time.perf_counter_ns(),
    )

    # In the old design, this was rejected, leaving STAGE_2_NETWORK_RECV missing
    assert result is None
    assert "STAGE_2_NETWORK_RECV" not in obs.stages


# ---------------------------------------------------------------------------
# Test 2: Complete end-to-end genuine request-ID correlation for 2 candles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_genuine_request_id_correlation_two_consecutive_candles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify genuine sequential request IDs link all pipeline stages for two candles."""
    monkeypatch.setattr(
        "iqoption_research.connectors.iqoption.client.async_logout",
        AsyncMock(return_value=True),
    )

    now_s = int(time.time())
    mock_ws = MockWebSocketConnection(make_standard_handshake(now_s))

    async def connect_factory() -> MockWebSocketConnection:
        return mock_ws

    clock = ClockDiscipline()
    recorder = TimingRecorder(
        "e2e_correlation",
        clock=clock,
        provenance="OPERATOR_SUPERVISED_LIVE",
        is_synthetic=False,
        enable_clock_stability_check=False,
    )

    client = AsyncIQOption(
        ssid="mock_ssid",
        offline=False,
        connect_factory=connect_factory,
        timing_recorder=recorder,
    )

    await client.connect()

    # --- Cycle 1: Candle F1 = now_s - 2 (recently closed) ---
    f1 = now_s - 2
    obs1 = recorder.start_candle_observation(f1)  # No guessed request_id!
    obs1.provenance = "OPERATOR_SUPERVISED_LIVE"
    obs1.is_synthetic = False
    obs1.measurement_status = "ACQUISITION_ONLY"

    # Queue mock response for candles-1
    mock_ws.push_message(make_candles_frame("candles-1", f1))

    res1 = await client.get_candles("EURUSD", size=60, count=1, endtime=f1)
    assert res1.request_id == "candles-1"
    obs1.compute_latencies_and_gate(clock)

    # Verify all stages in obs1 genuinely link "candles-1"
    assert obs1.request_id == "candles-1"
    assert "STAGE_1_REQUEST_INIT" in obs1.stages
    assert obs1.stages["STAGE_1_REQUEST_INIT"].details["request_id"] == "candles-1"
    assert "STAGE_2_NETWORK_RECV" in obs1.stages
    assert obs1.stages["STAGE_2_NETWORK_RECV"].details["request_id"] == "candles-1"
    assert "STAGE_3_PARSE_VALIDATE" in obs1.stages
    assert obs1.stages["STAGE_3_PARSE_VALIDATE"].details["request_id"] == "candles-1"
    assert "STAGE_4_CANDLE_VALIDATE" in obs1.stages
    assert obs1.stages["STAGE_4_CANDLE_VALIDATE"].details["request_id"] == "candles-1"
    assert obs1.candle_received_within_15s is True
    assert obs1.candle_validated_within_15s is True
    assert obs1.gate_verdict == "ACQUISITION_15S_COMPLIANT"

    # --- Cycle 2: Candle F2 = now_s - 1 ---
    f2 = now_s - 1
    obs2 = recorder.start_candle_observation(f2)
    obs2.provenance = "OPERATOR_SUPERVISED_LIVE"
    obs2.is_synthetic = False
    obs2.measurement_status = "ACQUISITION_ONLY"

    # Queue mock response for candles-2
    mock_ws.push_message(make_candles_frame("candles-2", f2))

    res2 = await client.get_candles("EURUSD", size=60, count=1, endtime=f2)
    assert res2.request_id == "candles-2"
    obs2.compute_latencies_and_gate(clock)

    # Verify all stages in obs2 genuinely link "candles-2"
    assert obs2.request_id == "candles-2"
    assert obs2.stages["STAGE_1_REQUEST_INIT"].details["request_id"] == "candles-2"
    assert obs2.stages["STAGE_2_NETWORK_RECV"].details["request_id"] == "candles-2"
    assert obs2.stages["STAGE_3_PARSE_VALIDATE"].details["request_id"] == "candles-2"
    assert obs2.stages["STAGE_4_CANDLE_VALIDATE"].details["request_id"] == "candles-2"
    assert obs2.candle_received_within_15s is True
    assert obs2.candle_validated_within_15s is True
    assert obs2.gate_verdict == "ACQUISITION_15S_COMPLIANT"

    await client.close()

    # --- Save and verify evidence registry ---
    evidence_file = tmp_path / "supervised_timing_evidence_e2e.json"
    registry = TimingEvidenceRegistry.create(
        session_id=recorder.session_id,
        operational_mode="OPERATOR_SUPERVISED_LIVE",
        project_root=Path(__file__).resolve().parents[3],
        clock=clock,
        observations=recorder.observations,
    )
    registry.save_to_file(evidence_file, allow_overwrite=False)

    is_valid, errors = registry.verify_integrity()
    assert is_valid is True, f"Integrity errors: {errors}"
    assert len(registry.observations) == 2
    assert registry.observations[0]["request_id"] == "candles-1"
    assert registry.observations[1]["request_id"] == "candles-2"


# ---------------------------------------------------------------------------
# Test 3: Interleaved messages do not contaminate observation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interleaved_heartbeat_timesync_account_messages_do_not_contaminate_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify heartbeat, timeSync, profile, balances do not contaminate active candle recorder."""
    monkeypatch.setattr(
        "iqoption_research.connectors.iqoption.client.async_logout",
        AsyncMock(return_value=True),
    )

    now_s = int(time.time())
    mock_ws = MockWebSocketConnection(make_standard_handshake(now_s))

    async def connect_factory() -> MockWebSocketConnection:
        return mock_ws

    clock = ClockDiscipline()
    recorder = TimingRecorder("interleave_test", clock=clock, enable_clock_stability_check=False)

    client = AsyncIQOption(
        ssid="mock_ssid",
        offline=False,
        connect_factory=connect_factory,
        timing_recorder=recorder,
    )
    await client.connect()

    f = now_s - 1
    obs = recorder.start_candle_observation(f)

    # Interleave non-candle frames before the candle response arrives
    mock_ws.push_message({"name": MSG_HEARTBEAT, "msg": int(time.time())})
    mock_ws.push_message({"name": "timeSync", "msg": int(now_s * 1000)})
    mock_ws.push_message({"name": MSG_PROFILE, "msg": {"user_id": 9999}})
    mock_ws.push_message({"name": MSG_BALANCES, "msg": [{"id": "b1", "amount": 500}]})

    # Finally push correlated candle response
    mock_ws.push_message(make_candles_frame("candles-1", f))

    res = await client.get_candles("EURUSD", size=60, count=1, endtime=f)
    assert res.request_id == "candles-1"
    obs.compute_latencies_and_gate(clock)

    # STAGE_2 and STAGE_3 must be from the candles frame, NOT any interleaved frames
    assert obs.stages["STAGE_2_NETWORK_RECV"].details["request_id"] == "candles-1"
    assert obs.stages["STAGE_3_PARSE_VALIDATE"].details["request_id"] == "candles-1"
    assert obs.stages["STAGE_3_PARSE_VALIDATE"].details["frame_name"] == "candles"

    await client.close()


# ---------------------------------------------------------------------------
# Test 4: Mismatched and missing request IDs are rejected
# ---------------------------------------------------------------------------


def test_mismatched_and_missing_request_ids_are_rejected() -> None:
    """Verify recorder rejects mismatched or missing request IDs without contaminating stages."""
    clock = ClockDiscipline()
    recorder = TimingRecorder("reject_test", clock=clock, enable_clock_stability_check=False)

    obs = recorder.start_candle_observation(int(time.time()))
    recorder.record_request_sent("candles-1", "sendMessage", time.perf_counter_ns())
    assert obs.request_id == "candles-1"

    t_now = time.perf_counter_ns()
    # 1. Non-matching request ID: candles-999
    res_mismatch = recorder.record_correlated_frame(
        request_id="candles-999",
        frame_name="candles",
        recv_mono_ns=t_now,
        parsed_mono_ns=t_now + 500_000,
    )
    assert res_mismatch is None
    assert "STAGE_2_NETWORK_RECV" not in obs.stages

    # 2. Missing/empty request ID
    res_empty = recorder.record_correlated_frame(
        request_id="",
        frame_name="candles",
        recv_mono_ns=t_now + 1_000_000,
        parsed_mono_ns=t_now + 1_500_000,
    )
    assert res_empty is None
    assert "STAGE_2_NETWORK_RECV" not in obs.stages

    # 3. Matching request ID: candles-1 -> accepted
    res_match = recorder.record_correlated_frame(
        request_id="candles-1",
        frame_name="candles",
        recv_mono_ns=t_now + 2_000_000,
        parsed_mono_ns=t_now + 2_500_000,
    )
    assert res_match is not None
    assert "STAGE_2_NETWORK_RECV" in obs.stages
    assert obs.stages["STAGE_2_NETWORK_RECV"].mono_ns == t_now + 2_000_000


# ---------------------------------------------------------------------------
# Test 5: Response timeout fails closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_timeout_fails_closed_without_contamination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a request timeout fails closed without corrupting observation stages."""
    monkeypatch.setattr(
        "iqoption_research.connectors.iqoption.client.async_logout",
        AsyncMock(return_value=True),
    )

    now_s = int(time.time())
    mock_ws = MockWebSocketConnection(make_standard_handshake(now_s))

    async def connect_factory() -> MockWebSocketConnection:
        return mock_ws

    clock = ClockDiscipline()
    recorder = TimingRecorder("timeout_test", clock=clock, enable_clock_stability_check=False)

    config = ConnectionConfig(request_timeout=0.1)
    client = AsyncIQOption(
        ssid="mock_ssid",
        offline=False,
        config=config,
        connect_factory=connect_factory,
        timing_recorder=recorder,
    )
    await client.connect()

    obs = recorder.start_candle_observation(now_s - 1)

    # Do not push any response frame; expect timeout
    with pytest.raises(RequestTimeoutError, match="No response for sendMessage"):
        await client.get_candles("EURUSD", size=60, count=1, endtime=now_s - 1)

    obs.rejection_reason = "TIMEOUT_WAITING_FOR_COMPLETION"
    obs.compute_latencies_and_gate(clock)

    assert "STAGE_1_REQUEST_INIT" in obs.stages
    assert "STAGE_2_NETWORK_RECV" not in obs.stages  # Not measured!
    assert obs.candle_received_within_15s is False
    assert obs.candle_validated_within_15s is False
    assert obs.phase30_gate_satisfied is False
    assert obs.gate_verdict == "FAIL_REJECTED"

    await client.close()


# ---------------------------------------------------------------------------
# Test 6: Delayed candle availability fails closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delayed_candle_availability_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that if broker response has candles with to_ts < F, observation fails closed."""
    monkeypatch.setattr(
        "iqoption_research.connectors.iqoption.client.async_logout",
        AsyncMock(return_value=True),
    )

    now_s = int(time.time())
    mock_ws = MockWebSocketConnection(make_standard_handshake(now_s))

    async def connect_factory() -> MockWebSocketConnection:
        return mock_ws

    clock = ClockDiscipline()
    recorder = TimingRecorder("delayed_test", clock=clock, enable_clock_stability_check=False)

    client = AsyncIQOption(
        ssid="mock_ssid",
        offline=False,
        connect_factory=connect_factory,
        timing_recorder=recorder,
    )
    await client.connect()

    target_F = now_s - 1
    obs = recorder.start_candle_observation(target_F)

    # Broker replies with previous candle (to_ts = target_F - 60), not target_F
    stale_to_ts = target_F - 60
    mock_ws.push_message(make_candles_frame("candles-1", stale_to_ts))

    res = await client.get_candles("EURUSD", size=60, count=1, endtime=target_F)
    assert res.candles[-1].to_ts == stale_to_ts

    # Simulating supervised session arrival check
    if not res.candles or res.candles[-1].to_ts != target_F:
        obs.rejection_reason = "CANDLE_NOT_YET_AVAILABLE_AT_F"

    obs.compute_latencies_and_gate(clock)

    assert obs.rejection_reason == "CANDLE_NOT_YET_AVAILABLE_AT_F"
    assert obs.candle_validated_within_15s is False
    assert obs.phase30_gate_satisfied is False
    assert obs.gate_verdict == "FAIL_REJECTED"

    await client.close()


# ---------------------------------------------------------------------------
# Test 7: Connection interruption fails closed and preserves partial evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_interruption_fails_closed_and_saves_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify socket closure aborts measurement cleanly and saves valid partial registry."""
    monkeypatch.setattr(
        "iqoption_research.connectors.iqoption.client.async_logout",
        AsyncMock(return_value=True),
    )

    now_s = int(time.time())
    mock_ws = MockWebSocketConnection(make_standard_handshake(now_s))

    async def connect_factory() -> MockWebSocketConnection:
        return mock_ws

    clock = ClockDiscipline()
    recorder = TimingRecorder("interrupt_test", clock=clock, enable_clock_stability_check=False)

    client = AsyncIQOption(
        ssid="mock_ssid",
        offline=False,
        connect_factory=connect_factory,
        timing_recorder=recorder,
    )
    await client.connect()

    # Complete 1 valid candle
    f1 = now_s - 2
    obs1 = recorder.start_candle_observation(f1)
    mock_ws.push_message(make_candles_frame("candles-1", f1))
    await client.get_candles("EURUSD", size=60, count=1, endtime=f1)
    obs1.compute_latencies_and_gate(clock)

    # Start 2nd candle, then disconnect client abruptly
    f2 = now_s - 1
    obs2 = recorder.start_candle_observation(f2)
    await client.close()

    try:
        await client.get_candles("EURUSD", size=60, count=1, endtime=f2)
    except (ConnectionError, OSError, asyncio.CancelledError):
        obs2.rejection_reason = "CONNECTION_INTERRUPTED_FAIL_CLOSED"
        obs2.compute_latencies_and_gate(clock)

    await client.close()

    # Save partial evidence
    out_file = tmp_path / "partial_evidence.json"
    registry = TimingEvidenceRegistry.create(
        session_id=recorder.session_id,
        operational_mode="OFFLINE_SIMULATION",
        project_root=Path(__file__).resolve().parents[3],
        clock=clock,
        observations=recorder.observations,
    )
    registry.save_to_file(out_file, allow_overwrite=False)

    is_valid, errors = registry.verify_integrity()
    assert is_valid is True, f"Integrity errors on partial evidence: {errors}"
    assert len(registry.observations) == 2
    assert registry.observations[0]["gate_verdict"] == "ACQUISITION_15S_COMPLIANT"
    assert registry.observations[1]["rejection_reason"] == "CONNECTION_INTERRUPTED_FAIL_CLOSED"


# ---------------------------------------------------------------------------
# Test 8: CLI command and flag regression verification
# ---------------------------------------------------------------------------


def test_cli_documented_flags_and_rejection_of_unsupported_flags(tmp_path: Path) -> None:
    """Verify timing measure only accepts supported flags and rejects unsupported flags."""
    runner = CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})

    # 1. Verify --help contains supported flags
    res_help = runner.invoke(app, ["timing", "measure", "--help"])
    assert res_help.exit_code == 0
    assert "--live" in res_help.output
    assert "--yes-i-understand" in res_help.output
    assert "--duration" in res_help.output
    assert "--max-candles" in res_help.output
    assert "--output" in res_help.output
    assert "--overwrite" in res_help.output
    assert "--simulation" in res_help.output

    # 2. Verify unsupported flags are REJECTED by Click/Typer with exit code 2
    unsupported_flags = [
        ["--symbol", "EURUSD"],
        ["--size", "60"],
        ["--max-seconds", "180"],
        ["--poll-interval", "1.0"],
        ["--out-dir", "data/timing_experiments"],
    ]
    for flag_args in unsupported_flags:
        res_bad = runner.invoke(app, ["timing", "measure", *flag_args])
        assert res_bad.exit_code == 2, f"Expected flag {flag_args[0]} to be rejected"
        assert "No such option" in res_bad.output or "Error" in res_bad.output

    # 3. Verify execution with the exact supported Phase 33 flags in simulation mode
    sim_out = tmp_path / "valid_flags_evidence.json"
    res_sim = runner.invoke(
        app,
        [
            "timing",
            "measure",
            "--simulation",
            "--max-candles",
            "2",
            "--duration",
            "180",
            "--output",
            str(sim_out),
        ],
    )
    assert res_sim.exit_code == 0
    assert sim_out.is_file()

    # 4. Verify the generated evidence file passes verify-evidence command
    res_ver = runner.invoke(
        app,
        ["timing", "verify-evidence", "--path", str(sim_out), "--json"],
    )
    assert res_ver.exit_code == 0
    parsed = json.loads(res_ver.output)
    assert parsed["is_valid"] is True
