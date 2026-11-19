"""Integration tests for Phase 33 readiness, contamination defense, and verification CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.connectors.iqoption.ws_client import (
    IQOptionWebSocketClient,
    _PendingRequest,
)
from iqoption_research.timing.instrumentation import ClockDiscipline, TimingRecorder
from iqoption_research.timing.supervised_session import (
    SupervisedSessionConfig,
    SupervisedSessionRunner,
)


def test_websocket_non_candle_messages_do_not_contaminate_recorder() -> None:
    """Verify profile, balances, and timeSync do not contaminate active candle observation."""
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("no_contam_test", clock=clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000060, expected_request_id="candles-expected-1")

    ws_client = IQOptionWebSocketClient("mock_ssid", timing_recorder=recorder)

    # 1. Dispatch profile response with active observation - must NOT record STAGE_2
    loop = asyncio.new_event_loop()
    try:
        fut_prof = loop.create_future()
        ws_client._pending["profile-req-1"] = _PendingRequest(
            future=fut_prof, name="get-profile", created_at=0.0
        )
        profile_frame = {"name": "profile", "request_id": "profile-req-1", "msg": {"user_id": 123}}
        ws_client._dispatch(profile_frame, recv_mono_ns=100_000_000, parsed_mono_ns=100_500_000)
        assert "STAGE_2_NETWORK_RECV" not in obs.stages

        # 2. Dispatch balances response - must NOT record STAGE_2
        fut_bal = loop.create_future()
        ws_client._pending["bal-req-1"] = _PendingRequest(
            future=fut_bal, name="get-balances", created_at=0.0
        )
        bal_frame = {
            "name": "balances",
            "request_id": "bal-req-1",
            "msg": [{"id": 1, "amount": 1000}],
        }
        ws_client._dispatch(bal_frame, recv_mono_ns=200_000_000, parsed_mono_ns=200_500_000)
        assert "STAGE_2_NETWORK_RECV" not in obs.stages

        # 3. Dispatch correlated frame for candle response - MUST record STAGE_2 and STAGE_3
        fut_candle = loop.create_future()
        ws_client._pending["candles-expected-1"] = _PendingRequest(
            future=fut_candle, name="sendMessage", created_at=0.0
        )
        candle_frame = {
            "name": "candles",
            "request_id": "candles-expected-1",
            "msg": {"candles": [{"from": 1795000000, "to": 1795000060, "close": 1.0850}]},
        }
        ws_client._dispatch(candle_frame, recv_mono_ns=300_000_000, parsed_mono_ns=300_500_000)
        assert "STAGE_2_NETWORK_RECV" in obs.stages
        assert obs.stages["STAGE_2_NETWORK_RECV"].mono_ns == 300_000_000
        assert "STAGE_3_PARSE_VALIDATE" in obs.stages
        assert obs.stages["STAGE_3_PARSE_VALIDATE"].mono_ns == 300_500_000
    finally:
        loop.close()


def test_unexpected_request_id_does_not_contaminate_active_observation() -> None:
    """Verify frames with non-matching request IDs are rejected by record_correlated_frame."""
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("id_mismatch_test", clock=clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000060, expected_request_id="candles-target-99")

    # Frame with unexpected request ID
    result = recorder.record_correlated_frame(
        request_id="candles-other-12",
        frame_name="candles",
        recv_mono_ns=500_000_000,
        parsed_mono_ns=500_100_000,
    )
    assert result is None
    assert "STAGE_2_NETWORK_RECV" not in obs.stages

    # Frame with expected request ID
    result_valid = recorder.record_correlated_frame(
        request_id="candles-target-99",
        frame_name="candles",
        recv_mono_ns=600_000_000,
        parsed_mono_ns=600_100_000,
    )
    assert result_valid is not None
    assert "STAGE_2_NETWORK_RECV" in obs.stages
    assert obs.stages["STAGE_2_NETWORK_RECV"].mono_ns == 600_000_000


def test_cli_verify_evidence_command(tmp_path: Path) -> None:
    """Verify timing verify-evidence validates genuine files and catches tampered files."""
    runner = CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})
    out_file = tmp_path / "test_sim_registry.json"

    # 1. Generate clean simulation evidence file
    res_meas = runner.invoke(
        app,
        ["timing", "measure", "--simulation", "--output", str(out_file), "--json"],
    )
    assert res_meas.exit_code == 0
    assert out_file.is_file()

    # 2. Run verify-evidence on valid file
    res_verify = runner.invoke(app, ["timing", "verify-evidence", "--path", str(out_file)])
    assert res_verify.exit_code == 0
    assert "Evidence Registry Verification PASSED" in res_verify.output
    assert "Session ID:" in res_verify.output

    # 3. Run verify-evidence with --json
    res_json = runner.invoke(app, ["timing", "verify-evidence", "--path", str(out_file), "--json"])
    assert res_json.exit_code == 0
    parsed = json.loads(res_json.output)
    assert parsed["is_valid"] is True
    assert len(parsed["errors"]) == 0

    # 4. Tamper evidence file on disk
    raw_content = json.loads(out_file.read_text(encoding="utf-8"))
    raw_content["observations"][0]["gate_verdict"] = "TAMPERED_VERDICT"
    out_file.write_text(json.dumps(raw_content), encoding="utf-8")

    # 5. Run verify-evidence on tampered file - must fail closed with exit code 1
    res_tampered = runner.invoke(app, ["timing", "verify-evidence", "--path", str(out_file)])
    assert res_tampered.exit_code == 1
    assert "Evidence Registry Verification FAILED" in res_tampered.output
    assert "integrity hash mismatch" in res_tampered.output


def test_cli_measure_rejects_live_overwrite_flag() -> None:
    """Verify timing measure rejects --live with --overwrite flag."""
    runner = CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})
    res = runner.invoke(
        app,
        ["timing", "measure", "--live", "--yes-i-understand", "--overwrite"],
    )
    assert res.exit_code == 1
    assert "Overwriting evidence (--overwrite) is strictly forbidden" in res.output


def test_bounded_trial_config_parameters(tmp_path: Path) -> None:
    """Verify configuration parameters for a 3-minute, 2-candle operator trial."""
    config = SupervisedSessionConfig(
        symbol="EURUSD",
        candle_size_seconds=60,
        max_duration_seconds=180,
        max_candle_cycles=2,
        live_mode=True,
        simulation_mode=False,
        operator_confirmed=True,
        allow_overwrite=False,
        output_destination=tmp_path / "trial_dest.json",
    )
    config.validate()
    assert config.max_duration_seconds == 180
    assert config.max_candle_cycles == 2

    runner = SupervisedSessionRunner(config)
    assert (
        runner.output_destination.name.startswith("trial_dest_meas_")
        or runner.output_destination == tmp_path / "trial_dest.json"
    )
