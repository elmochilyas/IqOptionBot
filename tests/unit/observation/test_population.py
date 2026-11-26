"""Expanded development population tests for Phase 44.

Authorized expansion beyond the legacy 66,464-bar cap is exercised only through
the explicit TEST_ONLY enforcement opt-out because every fixture here is a
synthetic test fixture. Genuine callers must supply a real verified contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iqoption_research.ml.experiment_matrix import (
    MultiHorizonExperimentMatrix,
    build_phase43_experiment_plan,
)
from iqoption_research.ml.preflight import (
    PreflightEnforcement,
    PreflightValidationError,
    compute_candle_content_digest,
    validate_development_population_preflight,
)
from tests.unit.governance.test_contract_phase44_population_contract import (
    PRE_BOUNDARY_START_TS,
    make_candles,
    make_contract,
)


def test_expanded_population_exceeding_66464_bars_is_authorized_under_test_only_contract() -> None:
    """An explicit contract supports 70,000 bars without the legacy hardcoded cap."""
    candles = make_candles(count=70_000, start_ts=PRE_BOUNDARY_START_TS)
    digest = compute_candle_content_digest(candles)
    contract = make_contract(
        candles,
        population_id="dev_eurusd_expanded_70k",
        expected_content_digest=digest,
        snapshot_id="rds_eurusd_m1_test_70000",
        plan_id="phase44_focused_expansion",
    )

    report = validate_development_population_preflight(
        candles,
        contract=contract,
        enforcement=PreflightEnforcement.TEST_ONLY,
    )

    assert report.is_valid is True
    assert report.candle_count == 70_000
    assert report.computed_content_digest == digest
    assert report.status == "TEST_ONLY_AUTHORIZED_POPULATION"
    assert report.snapshot_id == "rds_eurusd_m1_test_70000"
    assert report.plan_id == "phase44_focused_expansion"


def test_unauthorized_contract_fails_closed() -> None:
    """A contract flagged is_development_authorized=False is rejected."""
    candles = make_candles(1200)
    contract = make_contract(candles, is_development_authorized=False)

    with pytest.raises(
        PreflightValidationError,
        match="not authorized for development research",
    ):
        validate_development_population_preflight(
            candles,
            contract=contract,
            enforcement=PreflightEnforcement.TEST_ONLY,
        )


def test_experiment_matrix_requires_genuine_contract(tmp_path: Path) -> None:
    """The matrix cannot execute genuine research without an explicit contract."""
    candles = make_candles(1200)
    matrix = MultiHorizonExperimentMatrix(repo_root=tmp_path, db_path=tmp_path / "matrix.db")
    plan = build_phase43_experiment_plan(max_search_budget=1)

    with pytest.raises(
        PreflightValidationError,
        match="Explicit AuthorizedDevelopmentPopulation contract is required",
    ):
        matrix.run_suite(candles=candles, plan=plan, max_trials=1, bootstrap_replicates=50)


def test_experiment_matrix_uses_full_authorized_population_and_canonical_digest(
    tmp_path: Path,
) -> None:
    """No implicit 85% slicing occurs and exposure records the canonical digest."""
    candles = make_candles(1200)
    expected_digest = compute_candle_content_digest(candles)
    contract = make_contract(candles, population_id="dev_eurusd_1200")
    matrix = MultiHorizonExperimentMatrix(repo_root=tmp_path, db_path=tmp_path / "matrix.db")
    plan = build_phase43_experiment_plan(max_search_budget=1)

    summary = matrix.run_suite(
        candles=candles,
        plan=plan,
        max_trials=1,
        bootstrap_replicates=50,
        dataset_id="test_expanded_dataset",
        dataset_sha256=expected_digest,
        contract=contract,
        preflight_enforcement=PreflightEnforcement.TEST_ONLY,
    )

    assert summary.total_completed_trials == 1
    assert summary.plan_id == "phase44_focused_h15_plan"
    assert summary.global_disposition == "NO_DEMONSTRATED_EDGE"
    assert summary.execution_authority == "NONE"

    _, exposures = matrix.exposure_registry.check_interval_exposure(
        from_ts=candles[0].from_ts, to_ts=candles[-1].to_ts
    )
    assert len(exposures) >= 1
    assert exposures[-1].sample_count == 1200
    assert exposures[-1].dataset_sha256 == expected_digest


def test_experiment_matrix_rejects_parquet_checksum_as_dataset_sha256(tmp_path: Path) -> None:
    """A parquet checksum passed as dataset_sha256 fails closed."""
    candles = make_candles(1200)
    contract = make_contract(candles)
    matrix = MultiHorizonExperimentMatrix(repo_root=tmp_path, db_path=tmp_path / "matrix.db")
    plan = build_phase43_experiment_plan(max_search_budget=1)

    with pytest.raises(
        PreflightValidationError,
        match="dataset_sha256 does not match",
    ):
        matrix.run_suite(
            candles=candles,
            plan=plan,
            max_trials=1,
            bootstrap_replicates=50,
            dataset_sha256="b" * 64,
            contract=contract,
            preflight_enforcement=PreflightEnforcement.TEST_ONLY,
        )
