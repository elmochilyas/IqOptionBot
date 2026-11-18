"""Integration tests for Phase 31 timing diagnostics and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.timing.reconstruction import reconstruct_all_timing_evidence


def test_reconstruct_all_timing_evidence_on_repo_checkpoints() -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = reconstruct_all_timing_evidence(root)

    assert "checkpoints" in evidence
    assert len(evidence["checkpoints"]) >= 4
    # None of the single-request historical checkpoints met the 15s gate
    for cp in evidence["checkpoints"]:
        if "error" in cp:
            continue
        assert "invocation_delay_seconds" in cp
        assert "collection_duration_seconds" in cp
        assert "total_offset_seconds" in cp
        # All historical forward syncs completed after F+15s
        assert cp["satisfies_phase30_15s_gate"] is False

    assert "prospective_journals" in evidence
    assert "Phase 28" in evidence["prospective_journals"]
    assert "Phase 29" in evidence["prospective_journals"]
    assert "Phase 30" in evidence["prospective_journals"]


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
