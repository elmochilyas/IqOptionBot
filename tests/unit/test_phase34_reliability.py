"""Registry-level aggregation and historical presentation regression cases."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.timing.evidence import TimingEvidenceRegistry
from iqoption_research.timing.instrumentation import TimingRecorder
from iqoption_research.timing.reliability import distribution, evaluate


def fixture_registry(dest: Path, session: str, offsets: list[tuple[int, int]]) -> Path:
    # Emulate the existing on-disk live format exclusively inside pytest's temporary directory.
    recorder = TimingRecorder(
        session,
        provenance="OPERATOR_SUPERVISED_LIVE",
        is_synthetic=False,
        enable_clock_stability_check=False,
    )
    for boundary, ms in offsets:
        req = f"candles-{boundary}"
        obs = recorder.start_candle_observation(boundary, expected_request_id=req)
        obs.measurement_status = "ACQUISITION_ONLY"
        mono = boundary * 10**9 - recorder.clock.anchor_utc_ns + ms * 10**6
        recorder.record_request_sent(req, "sendMessage", mono)
        if ms:
            recorder.record_correlated_frame(
                request_id=req, frame_name="candles", recv_mono_ns=mono + 1, parsed_mono_ns=mono + 2
            )
            recorder.record_candle_validated(mono + 3, 22, request_id=req)
            recorder.record_completed_recognized(mono + 4, boundary)
        recorder.finalize_current_observation(None if ms else "DISCONNECT")
    registry = TimingEvidenceRegistry.create(
        session, "OPERATOR_SUPERVISED_LIVE", dest.parent, recorder.clock, recorder.observations
    )
    registry.save_to_file(dest)
    return dest


def test_multisession_report_and_cli(tmp_path: Path) -> None:
    a = fixture_registry(tmp_path / "a.json", "session-a", [(600, 100), (720, 16000)])
    b = fixture_registry(tmp_path / "b.json", "session-b", [(720, 17000), (780, 0)])
    bad = tmp_path / "bad.json"
    data = json.loads(a.read_text())
    data["observations"][0]["gate_verdict"] = "changed"
    bad.write_text(json.dumps(data))
    report = evaluate([a, b, a, bad])
    assert report["unique_live_boundaries"] == 3
    assert report["duplicate_registries"] == 1
    assert len(report["invalid_registries"]) == 1
    assert report["within_session_gaps"][0]["unobserved_boundaries"] == 1
    assert report["observation_issues"] == {
        "duplicate_boundary_attempts": 1,
        "late_receipt": 1,
        "late_validation": 1,
        "rejected": 2,
        "incomplete_or_invalid": 1,
    }
    assert report["receipt_15s"]["rate"] == pytest.approx(1 / 3)
    assert report["validation_ms"]["count"] == 2
    assert report["receipt_ms"]["p90"] is None
    cli = CliRunner()
    display = cli.invoke(app, ["timing", "verify-evidence", "--path", str(a)])
    assert display.exit_code == 0 and "NOT_MEASURED" in display.output
    assert cli.invoke(app, ["timing", "reliability", str(tmp_path)]).exit_code == 1


def test_tail_resolution() -> None:
    assert distribution(list(range(1, 101))) == {
        "count": 100,
        "median": 50.5,
        "maximum": 100,
        "p90": 90,
        "p95": 95,
        "p99": 99,
    }
