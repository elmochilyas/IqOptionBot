"""Integration tests for Phase 32 CLI commands, session orchestration, and connector hooks."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.connectors.iqoption.client import AsyncIQOption, CandleFetchResult
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.models.account import PracticeStatus
from iqoption_research.security.credentials import Credentials
from iqoption_research.timing.instrumentation import ClockDiscipline, TimingRecorder
from iqoption_research.timing.supervised_session import (
    SupervisedSessionConfig,
    SupervisedSessionRunner,
)


def test_cli_pipeline_audit_command() -> None:
    runner = CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})
    res = runner.invoke(app, ["timing", "pipeline-audit", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["pipeline_version"] == "PHASE_32_ACQUISITION_TIMING_AUDIT_V1"
    assert data["total_stages_audited"] == 10
    assert "critical_findings" in data


def test_cli_replay_command() -> None:
    runner = CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})
    res = runner.invoke(app, ["timing", "replay", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert isinstance(data, list)
    assert len(data) == 10
    # Scenario 1 passed
    assert data[0]["phase30_gate_satisfied"] is True
    # Scenarios 2-10 failed closed
    for s in data[1:]:
        assert s["phase30_gate_satisfied"] is False


def test_cli_measure_simulation_command(tmp_path: Path) -> None:
    runner = CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})
    out_file = tmp_path / "sim_output.json"
    res = runner.invoke(
        app,
        ["timing", "measure", "--simulation", "--output", str(out_file), "--json"],
    )
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["operational_mode"] == "OFFLINE_SIMULATION"
    assert out_file.is_file()
    assert len(data["observations"]) == 5


def test_cli_measure_live_refusal_without_operator_confirmation() -> None:
    runner = CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})
    res = runner.invoke(app, ["timing", "measure", "--live"])
    assert res.exit_code == 1
    assert "requires explicit operator confirmation flag" in res.output


@pytest.mark.asyncio
async def test_connector_instrumentation_hooks() -> None:
    """Verify TimingRecorder hooks in IQOptionWebSocketClient and AsyncIQOption."""
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("conn_hook_test", clock=clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000000)

    # 1. Test WebSocketClient hooks
    ws_client = IQOptionWebSocketClient("test_ssid", timing_recorder=recorder)
    assert ws_client._timing_recorder is recorder

    # Simulate dispatching timeSync - must NOT record candle network receipt
    ws_client._dispatch({"name": "timeSync", "msg": 1795000002100}, recv_mono_ns=100_000_000)
    assert "STAGE_2_NETWORK_RECV" not in obs.stages

    # Simulate dispatching correlated candle response
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    req_id = "candles-test-1"
    from iqoption_research.connectors.iqoption.ws_client import _PendingRequest

    ws_client._pending[req_id] = _PendingRequest(future=fut, name="sendMessage", created_at=0.0)

    candle_frame = {
        "name": "candles",
        "request_id": req_id,
        "msg": {"candles": [{"from": 1794999940, "to": 1795000000, "close": 1.0850}]},
    }
    t_recv = 150_000_000
    t_parsed = 151_200_000
    ws_client._dispatch(
        candle_frame,
        recv_mono_ns=t_recv,
        parsed_mono_ns=t_parsed,
        raw_payload_bytes=256,
    )

    assert "STAGE_2_NETWORK_RECV" in obs.stages
    assert "STAGE_3_PARSE_VALIDATE" in obs.stages
    assert obs.stages["STAGE_2_NETWORK_RECV"].mono_ns == t_recv
    assert obs.stages["STAGE_3_PARSE_VALIDATE"].mono_ns == t_parsed
    assert obs.stages["STAGE_2_NETWORK_RECV"].details["request_id"] == req_id
    assert fut.done()

    # 2. Test AsyncIQOption hook configuration
    creds = Credentials(email="test@example.com", password="dummy_password")
    client = AsyncIQOption(
        credentials=creds,
        offline=True,
        timing_recorder=recorder,
    )
    assert client.timing_recorder is recorder
    client.assert_read_only()


def test_cli_measure_rejects_non_positive_parameters() -> None:
    """Verify CLI timing measure rejects zero or negative duration and cycle limits."""
    runner = CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})

    # Duration 0
    res_dur0 = runner.invoke(app, ["timing", "measure", "--duration", "0"])
    assert res_dur0.exit_code == 1
    assert "duration must be positive" in res_dur0.output

    # Duration negative
    res_dur_neg = runner.invoke(app, ["timing", "measure", "--duration", "-10"])
    assert res_dur_neg.exit_code == 1
    assert "duration must be positive" in res_dur_neg.output

    # Max candles 0
    res_c0 = runner.invoke(app, ["timing", "measure", "--max-candles", "0"])
    assert res_c0.exit_code == 1
    assert "max-candles must be positive" in res_c0.output

    # Max candles negative
    res_c_neg = runner.invoke(app, ["timing", "measure", "--max-candles", "-3"])
    assert res_c_neg.exit_code == 1
    assert "max-candles must be positive" in res_c_neg.output


def test_cli_measure_create_only_file_protection(tmp_path: Path) -> None:
    """Verify CLI timing measure prevents accidental overwriting unless --overwrite is passed."""
    runner = CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})
    out_file = tmp_path / "protected_evidence.json"

    # First write succeeds
    res1 = runner.invoke(
        app,
        ["timing", "measure", "--simulation", "--output", str(out_file), "--json"],
    )
    assert res1.exit_code == 0
    assert out_file.is_file()

    # Second write without --overwrite fails closed
    res2 = runner.invoke(
        app,
        ["timing", "measure", "--simulation", "--output", str(out_file), "--json"],
    )
    assert res2.exit_code == 1
    assert "already exists and cannot be overwritten" in res2.output

    # Second write with --overwrite succeeds
    res3 = runner.invoke(
        app,
        ["timing", "measure", "--simulation", "--output", str(out_file), "--overwrite", "--json"],
    )
    assert res3.exit_code == 0


@pytest.mark.asyncio
async def test_live_session_console_reporting_distinguishes_stages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify console output distinguishes receipt, validation, and unmeasured prediction stages."""
    dest = tmp_path / "supervised_timing_evidence.json"
    cfg = SupervisedSessionConfig(
        symbol="EURUSD",
        max_duration_seconds=300,
        max_candle_cycles=1,
        live_mode=True,
        simulation_mode=False,
        operator_confirmed=True,
        output_destination=dest,
    )
    runner = SupervisedSessionRunner(cfg)

    # Mock client and practice verification
    mock_client = AsyncMock()
    mock_client.assert_read_only = MagicMock()
    mock_client.measurement_wallet_status = MagicMock(
        return_value=PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED
    )

    now_wall = int(time.time())
    F = (now_wall // 60) * 60
    mock_candle = MagicMock(to_ts=F, is_complete=True)

    async def mock_get_candles(*args: Any, **kwargs: Any) -> CandleFetchResult:
        now_mono = time.perf_counter_ns()
        # Simulate ws_client correlated frame receipt
        runner.recorder.record_correlated_frame(
            request_id=f"candles-{F}",
            frame_name="candles",
            recv_mono_ns=now_mono,
            parsed_mono_ns=now_mono,
        )
        # Simulate client.py recording validation
        runner.recorder.record_candle_validated(now_mono, candle_count=22, is_complete=True)
        return CandleFetchResult(
            symbol="EURUSD",
            size=60,
            candles=(mock_candle,),
            errors=(),
            quality=MagicMock(),
            fetched_at=datetime.now(UTC),
            raw_count=1,
            source="iqoption",
            recv_mono_ns=now_mono,
            parsed_mono_ns=now_mono + 500_000,
            request_completion_mono_ns=now_mono + 1_500_000,
        )

    mock_client.get_candles.side_effect = mock_get_candles

    mock_iq = MagicMock()
    mock_iq.__aenter__.return_value = mock_client
    mock_iq.__aexit__.return_value = None

    fixed_now = datetime.fromtimestamp(F + 2, tz=UTC)

    with (
        patch("iqoption_research.timing.supervised_session.AsyncIQOption", return_value=mock_iq),
        patch("asyncio.to_thread", side_effect=["test@example.com", "password"]),
        patch("iqoption_research.timing.supervised_session.datetime") as mock_dt,
        patch("asyncio.sleep", return_value=None),
    ):
        mock_dt.now.return_value = fixed_now
        mock_dt.fromtimestamp = datetime.fromtimestamp
        mock_dt.UTC = UTC
        reg = await runner.execute()

    captured = capsys.readouterr().out

    # 1. Verify stages are clearly distinguished in console output
    assert "- Client frame receipt:" in captured
    assert "- Candle validation:" in captured
    assert "- Full prediction contract: NOT_MEASURED (timing measurement only)" in captured
    assert "- Observation verdict:" in captured

    # 2. Verify genuine validation timestamp from client.py was preserved (not overwritten)
    obs = reg.observations[0]
    assert "STAGE_4_CANDLE_VALIDATE" in obs["stages"]
    assert (
        obs["stages"]["STAGE_4_CANDLE_VALIDATE"]["mono_ns"]
        < obs["stages"]["STAGE_5_COMPLETED_RECOG"]["mono_ns"]
    )

    # 3. Verify unique filename generated for live session
    assert runner.output_destination != dest
    assert runner.output_destination.name.startswith("supervised_timing_evidence_meas_")
    assert runner.output_destination.is_file()
