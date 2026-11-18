"""Integration tests for Phase 31 timing diagnostics and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.timing.reconstruction import reconstruct_all_timing_evidence


def test_reconstruct_all_timing_evidence_on_repo_checkpoints(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = reconstruct_all_timing_evidence(root)

    assert "checkpoints" in evidence
    assert "prospective_journals" in evidence
    assert "epistemic_findings" in evidence

    # If repo has local checkpoints (e.g. dev environment), verify their timing invariants
    for cp in evidence["checkpoints"]:
        if "error" in cp:
            continue
        assert "invocation_delay_seconds" in cp
        assert "collection_duration_seconds" in cp
        assert "total_offset_seconds" in cp
        # All historical forward syncs completed after F+15s
        assert cp["satisfies_phase30_15s_gate"] is False

    # Also test reconstruction against synthetic fixture directory to guarantee coverage in CI
    mock_dir = tmp_path / "mock_project"
    cp_dir = mock_dir / "data" / "datasets"
    cp_dir.mkdir(parents=True)
    sample_cp = {
        "job_id": "synthetic_cp_001",
        "symbol": "EURUSD",
        "active_id": 1,
        "candle_size_seconds": 60,
        "target_completed_end_ts": 1794942000,
        "created_at_utc": "2026-11-17T19:00:35.000000+00:00",
        "updated_at_utc": "2026-11-17T19:00:37.000000+00:00",
        "requests_made": 1,
        "new_records_saved": 60,
        "state": "completed",
    }
    (cp_dir / "forward_checkpoint_synthetic.json").write_text(
        json.dumps(sample_cp), encoding="utf-8"
    )
    synth_evidence = reconstruct_all_timing_evidence(mock_dir)
    assert len(synth_evidence["checkpoints"]) == 1
    synth_cp = synth_evidence["checkpoints"][0]
    assert synth_cp["invocation_delay_seconds"] == 35.0
    assert synth_cp["collection_duration_seconds"] == 2.0
    assert synth_cp["total_offset_seconds"] == 37.0
    assert synth_cp["satisfies_phase30_15s_gate"] is False


def test_cli_timing_commands() -> None:
    runner = CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})

    # 1. timing audit --json
    res_audit = runner.invoke(app, ["timing", "audit", "--json"])
    assert res_audit.exit_code == 0
    data_audit = json.loads(res_audit.output)
    assert "checkpoints" in data_audit
    assert "epistemic_findings" in data_audit

    # 2. timing feasibility --json
    res_feas = runner.invoke(app, ["timing", "feasibility", "--json"])
    assert res_feas.exit_code == 0
    data_feas = json.loads(res_feas.output)
    assert "architectures" in data_feas
    status = data_feas["phase30_gate_verdict"]["current_verified_workflow_status"]
    assert status == "DEFINITIVE_NO_GO"

    # 3. timing horizons --json
    res_horizons = runner.invoke(app, ["timing", "horizons", "--json"])
    assert res_horizons.exit_code == 0
    data_horizons = json.loads(res_horizons.output)
    assert isinstance(data_horizons, list)
    assert len(data_horizons) == 4

    # 4. timing blitz --json
    res_blitz = runner.invoke(app, ["timing", "blitz", "--json"])
    assert res_blitz.exit_code == 0
    data_blitz = json.loads(res_blitz.output)
    assert "items" in data_blitz
    assert data_blitz["synthesis"]["research_conclusion"] == "NO_DEMONSTRATED_EDGE"

    # 5. timing protocol --json
    res_proto = runner.invoke(app, ["timing", "protocol", "--json"])
    assert res_proto.exit_code == 0
    data_proto = json.loads(res_proto.output)
    assert data_proto["protocol_name"] == "OPERATOR_SUPERVISED_ACQUISITION_TIMING_PROTOCOL_V1"
    assert data_proto["status"] == "SPECIFICATION_ONLY_UNEXECUTED"
