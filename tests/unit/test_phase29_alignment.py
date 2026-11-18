"""Target audit and frozen-study guard regression checks."""

from pathlib import Path
from typing import Any

import pytest

from iqoption_research.ml.tracking.artifacts import ArtifactIntegrityError
from iqoption_research.research.observation_contract import (
    confirmed_prediction_keys,
    validate_observation_events,
)
from iqoption_research.research.observation_report import observation_report
from iqoption_research.research.observation_session import checked_registration
from iqoption_research.research.prospective import ProspectiveJournal, record_prediction
from iqoption_research.research.target_alignment import target_audit
from tests.integration.test_phase28_prospective import setup_run


def test_audit_rejects_modified_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _, _, _ = setup_run(tmp_path, monkeypatch)
    (run / "model" / "bundle_metadata.json").write_text("{}")
    with pytest.raises(ArtifactIntegrityError):
        target_audit(run / "model")


def test_old_journal_cannot_become_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, _, _ = setup_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="registration"):
        checked_registration(run, storage)


def test_mixed_latency_never_pooled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, storage, _, _ = setup_run(tmp_path, monkeypatch)
    prediction = record_prediction(run, storage)["prediction"]
    # Append a hash-valid but incompatible definition to test assessment refusal.
    changed = {k: v for k, v in prediction.items() if k != "key"}
    changed["entry_ts"] += 60
    changed["expiry_ts"] += 60
    changed["recorded_at_ts"] += 60
    changed["entry_ts"] += 60
    changed["expiry_ts"] += 60
    ProspectiveJournal(run).append("different-definition", changed)
    with pytest.raises(ValueError, match="latency"):
        observation_report(run)


def test_stale_unscoreable_is_not_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, storage, _, clock = setup_run(tmp_path, monkeypatch)
    clock[0] += 121
    record_prediction(run, storage)
    report = observation_report(run)
    assert report["counts"]["stale"] == report["counts"]["unscoreable"] == 1
    assert report["counts"]["pending"] == report["counts"]["fresh"] == 0


@pytest.mark.parametrize(
    "corruption",
    [
        {"entry_ts": 0},
        {"expiry_ts": 0},
        {"observable_at_ts": 0},
        {"observed_at_ts": 0},
        {"movement": 1},
        {"hypothetical_exit_price": float("nan")},
        {"broker_verified_blitz_settlement": True},
        {"prediction_key": "orphan"},
    ],
)
def test_invalid_outcomes_cannot_enter_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: dict[str, Any]
) -> None:
    run, storage, _, _ = setup_run(tmp_path, monkeypatch)
    p = record_prediction(run, storage)["prediction"]
    events = ProspectiveJournal(run).events()
    outcome = {
        "kind": "OUTCOME",
        "prediction_key": p["key"],
        "entry_ts": p["entry_ts"],
        "expiry_ts": p["expiry_ts"],
        "observable_at_ts": p["expiry_ts"],
        "observed_at_ts": p["expiry_ts"],
        "movement": 0,
        "hypothetical_entry_price": 1.0,
        "hypothetical_exit_price": 1.0,
        "provenance": "HYPOTHETICAL_EURUSD_CANDLE_OUTCOME",
        "broker_verified_blitz_settlement": False,
    }
    validate_observation_events([*events, outcome])
    with pytest.raises(ValueError):
        validate_observation_events([*events, {**outcome, **corruption}])
    with pytest.raises(ValueError, match="duplicate"):
        validate_observation_events([*events, outcome, outcome])


def test_future_inputs_rejected_by_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, storage, _, _ = setup_run(tmp_path, monkeypatch)
    p = record_prediction(run, storage)["prediction"]
    events = ProspectiveJournal(run).events()
    events[-1] = {**p, "feature_max_available_ts": p["recorded_at_ts"] + 1}
    with pytest.raises(ValueError, match="future inputs"):
        validate_observation_events(events)


def test_receipt_requires_causal_order_and_session_deadline() -> None:
    start = {"kind": "SESSION_START", "session_id": "s", "created_at_ts": 100, "deadline_ts": 110}
    prediction = {"kind": "PREDICTION", "key": "p", "recorded_at_ts": 101, "entry_ts": 200}
    receipt = {
        "kind": "PREDICTION_RECEIPT",
        "session_id": "s",
        "prediction_key": "p",
        "observed_at_ts": 102,
        "commit_before_entry_confirmed": True,
        "within_session_deadline": True,
        "fresh_at_commit": True,
    }
    assert confirmed_prediction_keys([start, prediction, receipt]) == {"p"}
    assert not confirmed_prediction_keys([start, receipt, prediction])
    for changes in (
        {"observed_at_ts": 100},
        {"observed_at_ts": 110},
        {"session_id": "unknown"},
        {"within_session_deadline": False},
    ):
        assert not confirmed_prediction_keys([start, prediction, {**receipt, **changes}])
