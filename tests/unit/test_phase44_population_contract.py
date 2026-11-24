"""Genuine population-contract fail-closed tests for Phase 44.

Every candle fixture in this module is explicitly marked as a synthetic test
fixture. It must pass only through the explicit TEST_ONLY enforcement opt-out
and must never be accepted as genuine research evidence.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from iqoption_research.ml.h15_research_v4 import H15ResearchEngineV4
from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
    AuthorizedDevelopmentPopulation,
    PreflightEnforcement,
    PreflightValidationError,
    ProtectedDataExposureError,
    compute_candle_content_digest,
    validate_development_population_preflight,
)
from iqoption_research.models.market import Candle
from tests.fixtures.synthetic_market import create_deterministic_synthetic_candles

PRE_BOUNDARY_START_TS: int = DEFAULT_PROTECTED_HOLDOUT_START_TS - 5_000_000


def make_candles(
    count: int,
    start_ts: int = PRE_BOUNDARY_START_TS,
    *,
    provenance: str = "SYNTHETIC_FIXTURE",
) -> list[Candle]:
    """Create candles explicitly marked as synthetic test fixtures."""
    candles = create_deterministic_synthetic_candles(count=count, start_ts=start_ts)
    return [
        replace(candle, provenance=provenance, identity_provenance="TEST_FIXTURE")
        for candle in candles
    ]


def make_contract(
    candles: list[Candle],
    **overrides: object,
) -> AuthorizedDevelopmentPopulation:
    """Build a complete, explicit contract for a test-only population."""
    values: dict[str, object] = {
        "population_id": "dev_eurusd_m1_expanded_test",
        "symbol": "EURUSD",
        "expected_content_digest": compute_candle_content_digest(candles),
        "from_ts": candles[0].from_ts,
        "to_ts": candles[-1].to_ts,
        "timeframe_seconds": 60,
        "snapshot_id": "rds_eurusd_m1_test_authorized",
        "snapshot_verified": True,
        "provenance": "HISTORICAL_REUSED",
        "plan_id": "phase44_focused_h15_plan",
        "max_allowed_ts": DEFAULT_PROTECTED_HOLDOUT_START_TS,
        "excluded_protected_intervals": (
            (DEFAULT_PROTECTED_HOLDOUT_START_TS, DEFAULT_PROTECTED_HOLDOUT_START_TS + 86_400),
        ),
        "exposure_history": ("PHASE44_TEST_AUTHORIZED",),
        "min_candle_count": 1,
        "max_candle_count": None,
        "is_development_authorized": True,
        "notes": "Explicit test-only contract; never genuine evidence.",
    }
    values.update(overrides)
    return AuthorizedDevelopmentPopulation(**values)  # type: ignore[arg-type]


def test_genuine_preflight_requires_explicit_contract() -> None:
    """A genuine run without a contract fails before any research computation."""
    candles = make_candles(1200)

    with pytest.raises(
        PreflightValidationError,
        match="Explicit AuthorizedDevelopmentPopulation contract is required",
    ):
        validate_development_population_preflight(candles)


def test_empty_expected_content_digest_fails_genuine_preflight() -> None:
    """A truthy contract with an empty digest is not genuine authorization."""
    candles = make_candles(1200)
    contract = make_contract(candles, expected_content_digest="")

    with pytest.raises(PreflightValidationError, match="contract is incomplete") as exc_info:
        validate_development_population_preflight(candles, contract=contract)
    assert "expected_content_digest" in str(exc_info.value)


def test_truthy_but_incomplete_contract_fails_genuine_preflight() -> None:
    """Metadata omissions cannot be bypassed by a truthy contract object."""
    candles = make_candles(1200)
    contract = make_contract(
        candles,
        snapshot_id="",
        snapshot_verified=False,
        provenance="",
        plan_id="",
        timeframe_seconds=0,
        excluded_protected_intervals=(),
        exposure_history=(),
    )

    with pytest.raises(PreflightValidationError, match="contract is incomplete"):
        validate_development_population_preflight(candles, contract=contract)


def test_parquet_checksum_cannot_substitute_for_candle_content_digest() -> None:
    """A file checksum passed as dataset_sha256 fails against the canonical digest."""
    candles = make_candles(1200)
    contract = make_contract(candles)

    with pytest.raises(PreflightValidationError, match="dataset_sha256 does not match"):
        validate_development_population_preflight(
            candles,
            contract=contract,
            expected_digest="a" * 64,
        )


def test_verified_snapshot_id_must_match_contract() -> None:
    """A contract snapshot id must match the snapshot verified by the caller."""
    candles = make_candles(1200)
    contract = make_contract(candles)

    with pytest.raises(PreflightValidationError, match="Verified snapshot id does not match"):
        validate_development_population_preflight(
            candles,
            contract=contract,
            verified_snapshot_id="rds_other_snapshot",
        )


def test_synthetic_fixture_cannot_masquerade_as_genuine() -> None:
    """Even a complete, digest-matching contract rejects explicitly synthetic candles."""
    candles = make_candles(1200)
    contract = make_contract(candles)

    with pytest.raises(
        PreflightValidationError,
        match="Synthetic/test fixture candles cannot masquerade",
    ):
        validate_development_population_preflight(candles, contract=contract)

    report = validate_development_population_preflight(
        candles,
        contract=contract,
        enforcement=PreflightEnforcement.TEST_ONLY,
    )
    assert report.status == "TEST_ONLY_AUTHORIZED_POPULATION"
    assert report.enforcement is PreflightEnforcement.TEST_ONLY


def test_protected_interval_blocked_even_with_expanded_authorized_population() -> None:
    """A contract cannot unlock the protected evaluation boundary."""
    candles = make_candles(
        count=2000,
        start_ts=DEFAULT_PROTECTED_HOLDOUT_START_TS - 60_000,
    )
    contract = make_contract(
        candles,
        max_allowed_ts=DEFAULT_PROTECTED_HOLDOUT_START_TS + 10_000_000,
        excluded_protected_intervals=(
            (
                DEFAULT_PROTECTED_HOLDOUT_START_TS + 10_000_000,
                DEFAULT_PROTECTED_HOLDOUT_START_TS + 20_000_000,
            ),
        ),
    )

    with pytest.raises(
        ProtectedDataExposureError,
        match="Protected evaluation holdout detected",
    ):
        validate_development_population_preflight(
            candles,
            contract=contract,
            enforcement=PreflightEnforcement.TEST_ONLY,
        )


def test_h15_genuine_evaluation_requires_contract_before_model_fitting() -> None:
    """H15 V4 genuine preflight fails before feature extraction without a contract."""
    candles = make_candles(1200)
    engine = H15ResearchEngineV4(bootstrap_replicates=10)

    with pytest.raises(
        PreflightValidationError,
        match="Explicit AuthorizedDevelopmentPopulation contract is required",
    ):
        engine.evaluate_model_v4(candles=candles, k_folds=1, enforce_preflight=True)
