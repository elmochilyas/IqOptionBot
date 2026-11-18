"""Entirely synthetic study/registration/observer contracts, never empirical performance."""

from __future__ import annotations

import itertools
import json
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.ml.tracking.artifacts import ArtifactIntegrityError, load_model_bundle
from iqoption_research.research import aligned_observer, aligned_study
from iqoption_research.research.aligned_data import inventory, load_selected, protection
from iqoption_research.research.phase28_audit import file_sha
from iqoption_research.research.prospective import ProspectiveJournal
from tests.unit.test_phase30_contract import candles


@pytest.fixture
def study(tmp_path: Path) -> tuple[Path, Path, list[Any], dict[str, Any]]:
    project = tmp_path
    storage = project / "data/datasets"
    cs = candles(1600)
    collector = HistoricalDataCollector(storage_dir=storage)
    collector.persist_batch(cs, symbol="EURUSD", candle_size_seconds=60)
    reserved = project / "data/phase11/confirmation_state.json"
    reserved.parent.mkdir(parents=True)
    reserved.write_text(
        json.dumps({
            "evaluation_status": "RESERVED",
            "prior_evaluation_cutoff_ts": cs[-1].to_ts + 600,
        })
    )
    guard = {
        "candles": 688,
        "original_start_ts": cs[-1].to_ts + 600,
        "original_last_open_ts": cs[-1].to_ts + 1200,
        "reservation_sha256": file_sha(reserved),
        "fingerprint_sha256": "sealed-synthetic-reference",
    }
    reference = project / "docs/phase29/review/dataset_audit.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({"original_holdout": guard}))
    run = project / "data/phase30/study"
    config = aligned_study.register_study(run, storage, project, minimum_rows=200)
    return run, storage, cs, config


def test_register_before_loading_prices_and_no_retry(
    study: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, _, _ = study
    assert [e["kind"] for e in ProspectiveJournal(run).events()] == ["CONFIG"]
    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: pytest.fail("network forbidden"))
    result = aligned_study.train_study(run)
    assert result["evidence_class"].startswith("INTERNAL_DEVELOPMENT")
    assert result["prospective_observations"] == 0 and result["model_promotion_allowed"] is False
    assert len(result["candidates"]) == 4
    assert result["purged_rows"] > 60
    assert (
        len([e for e in ProspectiveJournal(run).events() if e["kind"] == "SELECTION_FROZEN"]) == 1
    )
    with pytest.raises(ValueError, match="already attempted"):
        aligned_study.train_study(run)


def test_selected_sources_cannot_include_sealed_or_unsafe_paths(study: tuple) -> None:
    _run, storage, cs, config = study
    guard = config["protection"]
    selected = config["sources"]
    bad = json.loads(json.dumps(selected))
    bad["selected"][0]["from_ts"] = guard["original_start_ts"]
    bad["selected"][0]["to_ts"] = guard["end_exclusive"]
    with pytest.raises(ValueError, match="sealed"):
        load_selected(storage, bad, guard)
    collector = HistoricalDataCollector(storage_dir=storage)
    protected = [
        replace(
            c,
            from_ts=guard["original_start_ts"] + i * 60,
            to_ts=guard["original_start_ts"] + (i + 1) * 60,
        )
        for i, c in enumerate(cs[:10])
    ]
    collector.persist_batch(protected, symbol="EURUSD", candle_size_seconds=60)
    manifest = inventory(storage, protection(Path(config["project"])))
    assert any(e["reason"].startswith("SEALED") for e in manifest["excluded"])
    assert manifest["selected"] == selected["selected"]


def test_pipeline_scaling_and_calibration_only_fit_their_partitions(
    study: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, _, _ = study
    fit_inputs = []
    original = aligned_study.StandardScaler.fit

    def capture(self: Any, X: np.ndarray, **kwargs: Any) -> Any:
        fit_inputs.append(X.copy())
        return original(self, X, **kwargs)

    monkeypatch.setattr(aligned_study.StandardScaler, "fit", capture)
    report = aligned_study.train_study(run)
    assert len(fit_inputs) == 1
    assert len(fit_inputs[0]) == report["splits"]["train"]["decisive"]
    parts = list(report["splits"].values())
    assert all(a["last_label_expiry"] < b["first_context_ts"] for a, b in itertools.pairwise(parts))


def test_prospective_prediction_before_entry_exact_tie_and_restart(
    study: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, cs, config = study
    aligned_study.train_study(run)
    prospective = run.parent / "prospective"
    clock = [cs[1200].to_ts - 60.0]
    monkeypatch.setattr(aligned_study.time, "time", lambda: clock[0])
    # The historical development registration is in actual time; do not fake an earlier prospective
    # registration through checked_config. Use a later synthetic market window instead.
    clock[0] = config["created_at_ts"] + 60
    aligned_observer.register_prospective(prospective, run)
    F = int(clock[0] // 60) * 60 + 60
    offset = F - cs[1200].to_ts
    future = [replace(c, from_ts=c.from_ts + offset, to_ts=c.to_ts + offset) for c in cs[1100:1220]]
    future = [replace(c, close=c.open) if c.from_ts == F + 120 else c for c in future]
    HistoricalDataCollector(storage_dir=storage).persist_batch(
        future, symbol="EURUSD", candle_size_seconds=60
    )
    clock[0] = F + 1.0
    result = aligned_observer.record_aligned(prospective)
    assert result["eligible"] is True
    p = result["prediction"]
    assert p["entry_ts"] == F + 120 and p["expiry_ts"] == F + 180
    assert aligned_observer.record_aligned(prospective)["duplicate_prevented"] is True
    assert aligned_observer.resolve_aligned(prospective)["newly_resolved"] == 0
    clock[0] = F + 180
    assert aligned_observer.resolve_aligned(prospective)["newly_resolved"] == 1
    assert aligned_observer.resolve_aligned(prospective)["newly_resolved"] == 0
    report = aligned_observer.aligned_report(prospective)
    assert report["resolved"] == 1 and report["promotion_allowed"] is False
    assert report["directional_quality"]["ties"] == 1
    assert report["directional_quality"]["accuracy"] is None
    assert report["verified_blitz_settlements"] == 0


def test_study_source_mutation_and_interrupted_training_fail_closed(
    study: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, _, _ = study
    monkeypatch.setattr(
        aligned_study, "load_selected", lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    with pytest.raises(KeyboardInterrupt):
        aligned_study.train_study(run)
    assert any(e["kind"] == "TRAINING_STARTED" for e in ProspectiveJournal(run).events())
    with pytest.raises(ValueError, match="already attempted"):
        aligned_study.train_study(run)


def test_future_test_prices_do_not_change_fitted_models_or_selection(study: tuple) -> None:
    run, _storage, cs, config = study
    first = aligned_study.train_study(run)
    project = Path(config["project"])
    other_storage = project / "data/other-synthetic-source"
    modified = cs[:-100] + [
        replace(c, open=c.open * 1.05, high=c.high * 1.05, low=c.low * 1.05, close=c.close * 1.05)
        for c in cs[-100:]
    ]
    HistoricalDataCollector(storage_dir=other_storage).persist_batch(
        modified, symbol="EURUSD", candle_size_seconds=60
    )
    other = run.parent / "changed-test-only"
    aligned_study.register_study(other, other_storage, project, minimum_rows=200)
    second = aligned_study.train_study(other)
    assert first["selection_brier"] == second["selection_brier"]
    assert first["selected_development_model"] == second["selected_development_model"]
    from iqoption_research.research.aligned_data import build_dataset

    X = build_dataset(cs).X[:500]
    for name in ("logistic_regression", "random_forest"):
        one = load_model_bundle(run / "models" / name)
        two = load_model_bundle(other / "models" / name)
        np.testing.assert_array_equal(
            aligned_study.predict_bundle(one, X)[1], aligned_study.predict_bundle(two, X)[1]
        )


def test_modified_artifact_cannot_be_registered(study: tuple) -> None:
    run, _, _, _ = study
    result = aligned_study.train_study(run)
    metadata = run / "models" / result["selected_development_model"] / "bundle_metadata.json"
    metadata.write_text(metadata.read_text() + " ")
    with pytest.raises(ArtifactIntegrityError):
        aligned_observer.register_prospective(run.parent / "invalid-prospective", run)


def test_missing_exact_latency_or_outcome_candles_remain_pending(
    study: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, cs, config = study
    aligned_study.train_study(run)
    clock = [config["created_at_ts"] + 60]
    monkeypatch.setattr(aligned_study.time, "time", lambda: clock[0])
    prospective = run.parent / "missing-outcome"
    aligned_observer.register_prospective(prospective, run)
    F = int(clock[0] // 60) * 60 + 60
    offset = F - cs[1200].to_ts
    future = [
        replace(c, from_ts=c.from_ts + offset, to_ts=c.to_ts + offset)
        for c in cs[1100:1220]
        if c.from_ts + offset != F + 120
    ]
    collector = HistoricalDataCollector(storage_dir=storage)
    for segment in (
        [c for c in future if c.from_ts < F + 120],
        [c for c in future if c.from_ts > F + 120],
    ):
        collector.persist_batch(segment, symbol="EURUSD", candle_size_seconds=60)
    clock[0] = F + 1
    result = aligned_observer.record_aligned(prospective)
    assert result.get("eligible") is True, result
    clock[0] = F + 300
    assert aligned_observer.resolve_aligned(prospective)["pending"] == 1
    assert aligned_observer.aligned_report(prospective)["resolved"] == 0


def test_late_commit_is_retained_but_never_eligible(
    study: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, cs, config = study
    aligned_study.train_study(run)
    clock = [config["created_at_ts"] + 60]
    monkeypatch.setattr(aligned_study.time, "time", lambda: clock[0])
    prospective = run.parent / "late-commit"
    aligned_observer.register_prospective(prospective, run)
    F = int(clock[0] // 60) * 60 + 60
    offset = F - cs[1200].to_ts
    future = [replace(c, from_ts=c.from_ts + offset, to_ts=c.to_ts + offset) for c in cs[1100:1220]]
    HistoricalDataCollector(storage_dir=storage).persist_batch(
        future, symbol="EURUSD", candle_size_seconds=60
    )
    original = ProspectiveJournal.append

    def slow(self: Any, key: str, payload: dict) -> Any:
        value = original(self, key, payload)
        if payload["kind"] == "PREDICTION":
            clock[0] = F + 15.1
        return value

    monkeypatch.setattr(ProspectiveJournal, "append", slow)
    clock[0] = F + 1
    result = aligned_observer.record_aligned(prospective)
    assert result["recorded"] is True and result["eligible"] is False
    assert aligned_observer.aligned_report(prospective)["timing_eligible"] == 0
