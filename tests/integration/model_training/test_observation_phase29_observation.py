"""Synthetic contract checks only; never evidence of empirical skill."""

from __future__ import annotations

import json
import socket
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.ml.phase10 import prepare_clean_dataset
from iqoption_research.ml.targets.generator import generate_targets
from iqoption_research.ml.targets.specs import TargetSpec
from iqoption_research.research import observation_session as session
from iqoption_research.research.observation_report import observation_report
from iqoption_research.research.prospective import (
    ProspectiveJournal,
    record_prediction,
    resolve_predictions,
)
from iqoption_research.research.target_alignment import target_audit
from tests.integration.collection.test_boundary import immutable_files
from tests.integration.collection.test_prospective import persist, setup_run
from tests.unit.connectors.test_evaluation import fixture_candles


def file_bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, list[Any], list[int], list[float]]:
    old, storage, candles, clock = setup_run(tmp_path, monkeypatch)
    run = tmp_path / "phase29"
    monotonic = [0.0]
    monkeypatch.setattr(session, "now_ts", lambda: clock[0])
    monkeypatch.setattr(session.time, "monotonic", lambda: monotonic[0])
    session.register_observation_run(run, old / "model", storage)
    return run, storage, candles, clock, monotonic


def test_multiple_new_windows_no_network_or_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, storage, candles, clock, monotonic = new_run(tmp_path, monkeypatch)
    original = file_bytes(run / "model")
    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: pytest.fail("network forbidden"))
    position = [70]

    def advance(seconds: float) -> None:
        monotonic[0] += seconds
        clock[0] += int(seconds)
        persist(storage, candles[position[0] : position[0] + 1])
        position[0] += 1

    monkeypatch.setattr(session.time, "sleep", advance)
    result = session.observe_session(run, storage, 300)
    assert result["new_predictions"] == 4  # baseline window never replayed
    assert result["broker_requests"] == result["authenticated_sessions"] == 0
    events = ProspectiveJournal(run).events()
    predictions = [e for e in events if e["kind"] == "PREDICTION"]
    assert len({p["feature_candle_from_ts"] for p in predictions}) == 4
    assert all(p["recorded_at_ts"] < p["entry_ts"] for p in predictions)
    assert all(p["feature_max_available_ts"] <= p["data_cutoff_ts"] for p in predictions)
    receipts = [e for e in events if e["kind"] == "PREDICTION_RECEIPT"]
    assert len(receipts) == 4 and all(e["commit_before_entry_confirmed"] for e in receipts)
    assert file_bytes(run / "model") == original
    # Another operator session with unchanged latest bar cannot duplicate it.
    monkeypatch.setattr(
        session.time,
        "sleep",
        lambda seconds: (
            monotonic.__setitem__(0, monotonic[0] + seconds),
            clock.__setitem__(0, clock[0] + int(seconds)),
        ),
    )
    session.observe_session(run, storage, 1)
    report = observation_report(run)
    assert report["counts"]["skill_eligible"] == 0
    assert report["forecasting_skill"]["accuracy"] is None
    assert report["sample_sufficiency"]["independent_sample_size"] is None


def test_interrupt_recovery_and_lock_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, _, _, _ = new_run(tmp_path, monkeypatch)

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(session.time, "sleep", interrupt)
    assert session.observe_session(run, storage, 60)["status"] == "INTERRUPTED"
    assert session.observe_session(run, storage, 60)["status"] == "INTERRUPTED"
    journal = ProspectiveJournal(run)
    journal.append("crashed", {"kind": "SESSION_START", "session_id": "crashed"})
    assert "crashed" in observation_report(run)["unfinished_session_ids"]
    session.observe_session(run, storage, 60)
    assert not observation_report(run)["unfinished_session_ids"]
    assert any(e["kind"] == "SESSION_RECOVERY" for e in journal.events())


def test_failure_marks_session_and_concurrency_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, storage, _, _, _ = new_run(tmp_path, monkeypatch)
    with sqlite3.connect(run / "session-lock.db") as connection:
        connection.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            session.observe_session(run, storage, 1)
    monkeypatch.setattr(
        session,
        "resolve_predictions",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid source")),
    )
    with pytest.raises(ValueError, match="invalid source"):
        session.observe_session(run, storage, 1)
    assert ProspectiveJournal(run).events()[-1]["status"] == "FAILED"


@pytest.mark.parametrize("duration", [0, -1, 3601, True, 1.5])
def test_duration_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duration: int) -> None:
    run, storage, _, _, _ = new_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duration"):
        session.observe_session(run, storage, duration)


def test_new_registration_required_after_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, _, _, _ = new_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="storage"):
        session.observe_session(run, tmp_path / "different", 1)
    monkeypatch.setattr(session, "session_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="registration"):
        session.observe_session(run, storage, 1)


def test_target_labels_differ_despite_one_bar_horizon() -> None:
    candles = fixture_candles(60)
    # Discontinuous open versus prior close separates generic and execution targets.
    candles = [replace(c, open=c.close + 0.00001, high=c.high + 0.0001) for c in candles]
    clean = prepare_clean_dataset(candles, candle_size_seconds=60)
    generic, _ = generate_targets(candles, TargetSpec.binary())
    assert (clean.y == 0).all()
    assert any(generic)
    # A tie maps to training class 0, while observer treats it as non-directional.
    tied = [replace(c, open=c.close) for c in candles]
    assert not prepare_clean_dataset(tied, candle_size_seconds=60).y.any()


def test_legacy_tie_overlay_preserves_every_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, candles, clock = setup_run(tmp_path, monkeypatch)
    prediction = record_prediction(run, storage)["prediction"]
    target = next(c for c in candles if c.from_ts == prediction["entry_ts"])
    persist(storage, [replace(target, close=target.open)])
    clock[0] = prediction["expiry_ts"]
    resolve_predictions(run, storage)
    before = file_bytes(run)
    source_before = immutable_files(storage)
    report = observation_report(run)
    assert report["counts"]["tied"] == 1
    assert report["counts"]["decisive"] == 0
    assert report["counts"]["target_unverified"] == 1
    assert report["forecasting_skill"]["accuracy"] is None
    for scenario in report["hypothetical_economics"]:
        assert scenario["wins"] == scenario["losses"] == 0
        assert scenario["ties"] == 1
    assert file_bytes(run) == before
    assert immutable_files(storage) == source_before


def test_unknown_lineage_never_certified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _, _, _ = setup_run(tmp_path, monkeypatch)
    audit = target_audit(run / "model")
    assert audit["status"] == "TARGET_UNVERIFIED"
    assert audit["forecasting_skill_proof_allowed"] is False
    assert len(audit["development_targets"]) == 5


def test_future_candles_not_recorded_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, storage, candles, clock, monotonic = new_run(tmp_path, monkeypatch)
    persist(storage, candles[70:])
    monkeypatch.setattr(session.time, "sleep", lambda _seconds: monotonic.__setitem__(0, 1.0))
    assert session.observe_session(run, storage, 1)["new_predictions"] == 0
    assert clock[0] < candles[70].to_ts


def test_cli_requires_operator_and_new_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, _, _, _ = new_run(tmp_path, monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "observation",
            "session",
            "--run",
            str(run),
            "--duration-seconds",
            "1",
            "--storage",
            str(storage),
        ],
    )
    assert result.exit_code != 0 and "operator-supervised" in result.output
    output = tmp_path / "report.json"
    session.export_new_json(output, {"ok": True}, (run, storage))
    with pytest.raises(FileExistsError):
        session.export_new_json(output, {}, (run, storage))
    with pytest.raises(ValueError):
        session.export_new_json(run / "report.json", {}, (run, storage))
    assert json.loads(output.read_text()) == {"ok": True}


def test_late_commit_excluded_and_session_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, candles, clock, _ = new_run(tmp_path, monkeypatch)
    persist(storage, candles[70:71])
    clock[0] = candles[70].to_ts
    original_record = session.record_prediction

    def delayed_commit(run: Path, storage: Path) -> dict[str, Any]:
        result = original_record(run, storage)
        clock[0] = result["prediction"]["entry_ts"]
        return result

    monkeypatch.setattr(session, "record_prediction", delayed_commit)
    with pytest.raises(ValueError, match="pre-entry"):
        session.observe_session(run, storage, 300)
    report = observation_report(run)
    assert report["counts"]["excluded_unconfirmed_commits"] == 1
    assert report["session_statuses"] == {"FAILED": 1}


def test_clock_rollback_stops_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, storage, _, clock, _ = new_run(tmp_path, monkeypatch)
    clock[0] -= 1
    with pytest.raises(ValueError, match="Clock"):
        session.observe_session(run, storage, 60)


@pytest.mark.parametrize("clock_fault", ["rollback", "deadline"])
def test_slow_or_clock_invalid_commit_is_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock_fault: str
) -> None:
    run, storage, candles, clock, monotonic = new_run(tmp_path, monkeypatch)
    persist(storage, candles[70:71])
    clock[0] = candles[70].to_ts
    original_record = session.record_prediction

    def faulty_commit(run: Path, storage: Path) -> dict[str, Any]:
        result = original_record(run, storage)
        if clock_fault == "rollback":
            clock[0] -= 1
        else:
            monotonic[0] += 10
        return result

    monkeypatch.setattr(session, "record_prediction", faulty_commit)
    with pytest.raises(ValueError, match=r"pre-entry|deadline"):
        session.observe_session(run, storage, 5)
    report = observation_report(run)
    assert report["counts"]["excluded_unconfirmed_commits"] == 1
    assert report["session_statuses"] == {"FAILED": 1}


def test_staleness_at_commit_stops_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, storage, candles, clock, _ = new_run(tmp_path, monkeypatch)
    persist(storage, candles[70:71])
    clock[0] = candles[70].to_ts + 119
    original_record = session.record_prediction

    def slow_commit(run: Path, storage: Path) -> dict[str, Any]:
        result = original_record(run, storage)
        clock[0] += 2
        return result

    monkeypatch.setattr(session, "record_prediction", slow_commit)
    with pytest.raises(ValueError, match="stale"):
        session.observe_session(run, storage, 300)
    assert observation_report(run)["counts"]["excluded_unconfirmed_commits"] == 1


def test_runtime_change_requires_new_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, _, _, _ = new_run(tmp_path, monkeypatch)
    monkeypatch.setattr(session, "runtime_versions", lambda: {"changed": True})
    with pytest.raises(ValueError, match="registration"):
        session.observe_session(run, storage, 1)
