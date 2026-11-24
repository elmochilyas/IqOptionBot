"""Unit tests for Phase 44 protected data preflight validation.

Tests:
1. Valid authorized development population passes preflight.
2. Fail-closed rejection when protected evaluation holdout data is detected.
3. Fail-closed rejection when cryptographic content digest does not match.
4. Fail-closed rejection when timestamp bounds violate contract.
5. Fail-closed rejection when candle count is insufficient.

All genuine-mode contracts are complete: preflight no longer accepts implicit
or partially populated development populations.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_END_TS,
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


def _make_candles(count: int, start_ts: int) -> list[Candle]:
    return create_deterministic_synthetic_candles(count=count, start_ts=start_ts)


def _complete_contract(
    population_id: str,
    candles: list[Candle],
    **overrides: object,
) -> AuthorizedDevelopmentPopulation:
    values: dict[str, object] = {
        "population_id": population_id,
        "symbol": "EURUSD",
        "expected_content_digest": compute_candle_content_digest(candles),
        "from_ts": candles[0].from_ts,
        "to_ts": candles[-1].to_ts,
        "timeframe_seconds": 60,
        "snapshot_id": "rds_phase44_preflight_unit",
        "snapshot_verified": True,
        "provenance": "HISTORICAL_REUSED",
        "plan_id": "PLAN_PHASE44_PREFLIGHT_UNIT",
        "max_allowed_ts": DEFAULT_PROTECTED_HOLDOUT_START_TS,
        "excluded_protected_intervals": (
            (DEFAULT_PROTECTED_HOLDOUT_START_TS, DEFAULT_PROTECTED_HOLDOUT_END_TS),
        ),
        "exposure_history": ("PHASE42_HISTORICAL", "PHASE43_HISTORICAL"),
        "min_candle_count": 1000,
    }
    values.update(overrides)
    return AuthorizedDevelopmentPopulation(**values)  # type: ignore[arg-type]


def test_preflight_valid_development_population() -> None:
    """A series within development boundaries and matching digest passes preflight."""
    start_ts = 1725310800
    candles = _make_candles(1500, start_ts)
    digest = compute_candle_content_digest(candles)

    contract = _complete_contract("pop_dev_01", candles)

    report = validate_development_population_preflight(candles, contract=contract)
    assert report.is_valid is True
    assert report.protected_data_detected is False
    assert report.status == "AUTHORIZED_DEVELOPMENT_POPULATION"
    assert report.computed_content_digest == digest


def test_protected_boundary_epoch_matches_authoritative_iso_timestamp() -> None:
    """The protected epoch must be 2026-11-12T10:00:00Z, never a later boundary."""
    expected = int(datetime(2026, 11, 12, 10, 0, tzinfo=UTC).timestamp())
    assert expected == DEFAULT_PROTECTED_HOLDOUT_START_TS
    assert datetime.fromtimestamp(DEFAULT_PROTECTED_HOLDOUT_START_TS, UTC).isoformat() == (
        "2026-11-12T10:00:00+00:00"
    )

    boundary_candles = _make_candles(10, DEFAULT_PROTECTED_HOLDOUT_START_TS)
    genuine_contract = _complete_contract(
        "pop_exact_boundary",
        boundary_candles,
        min_candle_count=10,
    )
    with pytest.raises(ProtectedDataExposureError):
        validate_development_population_preflight(boundary_candles, contract=genuine_contract)
    with pytest.raises(ProtectedDataExposureError):
        validate_development_population_preflight(
            boundary_candles,
            enforcement=PreflightEnforcement.TEST_ONLY,
        )


def test_preflight_fails_on_protected_evaluation_leakage() -> None:
    """Fail closed when input data extends into the protected evaluation interval."""
    start_ts = DEFAULT_PROTECTED_HOLDOUT_START_TS - 300
    candles = _make_candles(100, start_ts)
    contract = _complete_contract(
        "pop_leaked",
        candles,
        min_candle_count=10,
        to_ts=start_ts + 6000,
    )

    with pytest.raises(ProtectedDataExposureError, match="Protected evaluation holdout detected"):
        validate_development_population_preflight(candles, contract=contract)


def test_preflight_fails_on_content_digest_mismatch() -> None:
    """Fail closed when actual candle content does not match the authorized digest."""
    start_ts = 1725310800
    candles = _make_candles(1200, start_ts)
    contract = _complete_contract("pop_tampered", candles, expected_content_digest="a" * 64)

    with pytest.raises(PreflightValidationError, match="Content digest mismatch"):
        validate_development_population_preflight(candles, contract=contract)


def test_preflight_fails_on_boundary_violations() -> None:
    """Fail closed when dataset timestamps exceed contract bounds."""
    start_ts = 1725310800
    candles = _make_candles(1200, start_ts)
    contract = _complete_contract("pop_bounds", candles, from_ts=start_ts + 3600)

    with pytest.raises(PreflightValidationError, match="exceeds contract boundaries"):
        validate_development_population_preflight(candles, contract=contract)


def test_preflight_fails_on_insufficient_sample_count() -> None:
    """Fail closed when candle count is below contract requirement."""
    start_ts = 1725310800
    candles = _make_candles(500, start_ts)
    contract = _complete_contract(
        "pop_small",
        candles,
        to_ts=start_ts + 30000,
    )

    with pytest.raises(PreflightValidationError, match="below contract minimum"):
        validate_development_population_preflight(candles, contract=contract)
