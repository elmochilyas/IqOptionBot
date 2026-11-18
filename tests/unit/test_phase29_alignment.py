"""Target audit and frozen-study guard regression checks."""

from pathlib import Path

import pytest

from iqoption_research.ml.tracking.artifacts import ArtifactIntegrityError
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
