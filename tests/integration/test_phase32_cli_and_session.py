"""Integration tests for Phase 32 CLI commands, session orchestration, and connector hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.security.credentials import Credentials
from iqoption_research.timing.instrumentation import ClockDiscipline, TimingRecorder


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

    # Simulate raw frame arrival and frame decoding
    fake_raw = json.dumps({"name": "timeSync", "msg": 1795000002100})
    recorder.record_raw_receipt(1795000002100000000, fake_raw)
    recorder.record_frame_decoded(1795000002101000000, {"name": "timeSync", "msg": 1795000002100})

    assert "STAGE_2_NETWORK_RECV" in obs.stages
    assert "STAGE_3_PARSE_VALIDATE" in obs.stages
    assert obs.stages["STAGE_2_NETWORK_RECV"].details["payload_bytes"] == len(fake_raw)

    # 2. Test AsyncIQOption hook configuration
    creds = Credentials(email="test@example.com", password="dummy_password")
    client = AsyncIQOption(
        credentials=creds,
        offline=True,
        timing_recorder=recorder,
    )
    assert client.timing_recorder is recorder
    client.assert_read_only()
