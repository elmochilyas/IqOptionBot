"""Deterministic frozen-label reporting; no production evidence or network."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from iqoption_research.shadow.baseline import FrozenMajorityBenchmark, resolve_majority_benchmark
from iqoption_research.shadow.evaluator import ShadowOutcomeEvaluator
from iqoption_research.shadow.journal import ShadowPredictionJournal
from iqoption_research.shadow.observer import ObservationSessionConfig, ProspectiveShadowObserver
from iqoption_research.shadow.registry import FrozenRegistrationRecord, ShadowModelRegistry
from tests.desktop.workstation.test_phase40b_prospective_shadow import (
    prospective_workspace as prospective_workspace,
)
from tests.desktop.workstation.test_phase40b_prospective_shadow import (
    test_contract_economics_binary_vs_digital as build_economics_journal,
)


@pytest.fixture
def benchmark_inputs() -> tuple[Mock, FrozenRegistrationRecord, FrozenRegistrationRecord]:
    candidate = FrozenRegistrationRecord(
        registration_id="reg",
        registered_at_utc="2023-01-01T00:00:00+00:00",
        model_id="candidate",
        run_id="run",
        model_family="logistic_regression",
        bundle_sha256="candidate_hash",
        manifest_sha256="manifest",
        snapshot_id="snapshot",
        dataset_content_sha256="dataset_hash",
        data_cutoff_ts=1600000000,
        feature_names=("return_1",),
        horizon_bars=1,
        confidence_threshold=0.52,
        calibration_method="isotonic",
        verdict="NO_DEMONSTRATED_EDGE",
        wallet_classification="UNKNOWN",
        is_baseline=False,
    )
    baseline = replace(
        candidate,
        model_id="majority",
        model_family="majority_class",
        is_baseline=True,
        bundle_sha256="baseline_hash",
        confidence_threshold=0.55,
    )
    registry = Mock(spec=ShadowModelRegistry)
    registry.get_candidate.return_value = baseline
    registry.load_frozen_bundle.return_value = {
        "manifest": {
            "is_synthetic": False,
            "bundle_sha256": "baseline_hash",
            "snapshot_id": "snapshot",
            "horizon_bars": 1,
        },
        "bundle": {"model": SimpleNamespace(class_frequencies_=[0.53, 0.47], majority_class_=0)},
    }
    return registry, candidate, baseline


def test_verified_training_label_does_not_change_abstaining_policy(benchmark_inputs) -> None:
    registry, candidate, baseline = benchmark_inputs
    result = resolve_majority_benchmark(
        registry, candidate, (candidate, baseline), observation_start_ts=1700000000
    )
    assert result.status == "VERIFIED"
    assert (result.direction, result.probability_up) == ("DOWN", 0.47)
    assert result.source_sha256 == baseline.bundle_sha256
    assert baseline.confidence_threshold == 0.55
    assert candidate.confidence_threshold == 0.52
    assert result.semantics == "TRAINING_MAJORITY_LABEL_NOT_TRADING_POLICY"
    registry.load_frozen_bundle.assert_called_once_with("majority")


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "ambiguous",
        "mismatch",
        "future",
        "naive",
        "synthetic",
        "tampered",
        "hash",
        "invalid_probability",
    ],
)
def test_unverified_linkage_is_unavailable(benchmark_inputs, failure: str) -> None:
    registry, candidate, baseline = benchmark_inputs
    observed = (candidate, baseline)
    if failure == "missing":
        observed = (candidate,)
    elif failure == "ambiguous":
        observed += (replace(baseline, model_id="other"),)
    elif failure == "mismatch":
        registry.get_candidate.return_value = replace(baseline, dataset_content_sha256="other")
    elif failure == "future":
        registry.get_candidate.return_value = replace(
            baseline, registered_at_utc="2030-01-01T00:00:00+00:00"
        )
    elif failure == "naive":
        registry.get_candidate.return_value = replace(
            baseline, registered_at_utc="2023-01-01T00:00:00"
        )
    elif failure == "synthetic":
        registry.load_frozen_bundle.return_value["manifest"]["is_synthetic"] = True
    elif failure == "tampered":
        registry.load_frozen_bundle.side_effect = ValueError("tampered")
    elif failure == "hash":
        candidate = replace(
            candidate, baseline_source_model_id="majority", baseline_source_sha256="wrong"
        )
    else:
        registry.load_frozen_bundle.return_value["bundle"]["model"].class_frequencies_ = [
            0.53,
            float("nan"),
        ]
    result = resolve_majority_benchmark(
        registry, candidate, observed, observation_start_ts=1700000000
    )
    assert result.status == "UNAVAILABLE"
    assert result.direction is None and result.probability_up is None


def test_explicit_preverified_linkage_without_coobserved_policy(benchmark_inputs) -> None:
    registry, candidate, _ = benchmark_inputs
    candidate = replace(
        candidate, baseline_source_model_id="majority", baseline_source_sha256="baseline_hash"
    )
    assert (
        resolve_majority_benchmark(
            registry, candidate, (candidate,), observation_start_ts=1700000000
        ).status
        == "VERIFIED"
    )


def test_matched_label_population_and_preobservation_guard(tmp_path: Path) -> None:
    # Existing fixture contains UP win, UP loss, tie, DOWN win with timely attestations.
    build_economics_journal(tmp_path)
    folders = list((tmp_path / "data/shadow_eval/sessions").iterdir())
    journal = ShadowPredictionJournal(folders[0].name, session_dir=folders[0])
    evaluator = ShadowOutcomeEvaluator(
        journal,
        frozen_majority_direction="DOWN",
        frozen_majority_prob_up=0.47,
        baseline_comparison_available=True,
        baseline_source_id="majority",
        baseline_source_sha256="hash",
        baseline_frozen_at_ts=0,
    )
    summary = evaluator.compute_summary(model_id="cand_1")
    majority = summary.baseline_comparisons[0]
    assert majority.sample_count == 4 and majority.decisive_sample_count == 3
    assert majority.tie_count == 1
    assert majority.decisive_directional_accuracy == 0.6667
    assert majority.brier_score == pytest.approx((0.53**2 + 2 * 0.47**2) / 3, abs=5e-5)
    # Cutoffs preceding freeze cannot be promoted to verified baseline evidence.
    evaluator.baseline_frozen_at_ts = 9999999999
    unavailable = evaluator.compute_summary(model_id="cand_1")
    assert not unavailable.baseline_comparison_available
    assert unavailable.baseline_comparisons[0].status == "UNAVAILABLE"


@pytest.mark.parametrize("verified", [True, False])
def test_observer_wires_per_model_benchmark_and_masks_unavailable(
    prospective_workspace,
    monkeypatch: pytest.MonkeyPatch,
    verified: bool,
) -> None:
    root, _, model_id, other_id = prospective_workspace
    benchmark = (
        FrozenMajorityBenchmark(
            status="VERIFIED",
            reason="fixture",
            source_id="majority",
            source_sha256="hash",
            direction="DOWN",
            probability_up=0.47,
            frozen_at_ts=0,
        )
        if verified
        else FrozenMajorityBenchmark()
    )
    resolver = Mock(side_effect=[benchmark, FrozenMajorityBenchmark()])
    monkeypatch.setattr("iqoption_research.shadow.observer.resolve_majority_benchmark", resolver)
    monkeypatch.setattr("iqoption_research.shadow.observer.time.time", lambda: 1800000000.0)
    observer = ProspectiveShadowObserver(
        ObservationSessionConfig(
            session_id="reporting",
            model_ids=(model_id, other_id),
            duration_minutes=15,
            is_synthetic=True,
        ),
        ShadowModelRegistry(root),
        root=root,
    )
    evaluator = observer.reporting_evaluators[model_id]
    assert evaluator.baseline_comparison_available is verified
    assert not observer.reporting_evaluators[other_id].baseline_comparison_available
    assert resolver.call_count == 2
    assert resolver.call_args.kwargs["observation_start_ts"] == 1800000000.0
    build_economics_journal(root)
    folder = next(
        p for p in (root / "data/shadow_eval/sessions").iterdir() if p.name != "reporting"
    )
    journal = ShadowPredictionJournal(folder.name, session_dir=folder)
    summary = ShadowOutcomeEvaluator(
        journal,
        frozen_majority_direction=benchmark.direction,
        frozen_majority_prob_up=benchmark.probability_up,
        baseline_comparison_available=verified,
    ).compute_summary(model_id="cand_1")
    monkeypatch.setattr(evaluator, "compute_summary", lambda **_kwargs: summary)
    observer._update_model_states()
    state = observer.states[model_id]
    assert state.matched_baseline_accuracy == (0.6667 if verified else None)
    assert state.matched_baseline_status == ("VERIFIED" if verified else "UNAVAILABLE")


@pytest.mark.parametrize(
    "excluded",
    [
        "SKIPPED_ABSTAIN",
        "PENDING",
        "INVALID_DISCONTINUITY",
        "QUARANTINED_LATE_COMMITMENT",
        "OTHER_MODEL",
        "LATE",
    ],
)
def test_benchmark_uses_only_matched_eligible_selected_timestamps(
    tmp_path: Path,
    excluded: str,
) -> None:
    build_economics_journal(tmp_path)
    journal = ShadowPredictionJournal("sess_test", root=tmp_path)
    records = journal.list_predictions()
    # In-memory adversarial evaluator input; no mutation of journal files.
    first = records[0]
    if excluded == "OTHER_MODEL":
        records[0] = replace(first, model_id="other")
    elif excluded != "LATE":
        records[0] = replace(first, outcome_status=excluded)
    proxy = Mock(spec=ShadowPredictionJournal)
    proxy.session_id = journal.session_id
    proxy.list_predictions.return_value = records

    def commitment(pid):
        attestation = journal.get_commitment(pid)
        if excluded == "LATE" and pid == first.prediction_id:
            return replace(attestation, is_timely=False)
        return attestation

    proxy.get_commitment.side_effect = commitment
    summary = ShadowOutcomeEvaluator(
        proxy,
        frozen_majority_direction="DOWN",
        frozen_majority_prob_up=0.47,
        baseline_comparison_available=True,
    ).compute_summary(model_id="cand_1")
    majority = summary.baseline_comparisons[0]
    assert majority.sample_count == 3
    assert majority.decisive_sample_count == 2
    assert majority.decisive_directional_accuracy == 1.0
    assert majority.candidate_delta_decisive_accuracy == -0.5
