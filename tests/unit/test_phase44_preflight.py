"""Unit tests for Phase 44 protected data preflight validation.

Tests:
1. Valid authorized development population passes preflight.
2. Fail-closed rejection when protected evaluation holdout data is detected.
3. Fail-closed rejection when cryptographic content digest does not match.
4. Fail-closed rejection when timestamp bounds violate contract.
5. Fail-closed rejection when candle count is insufficient.
"""

from __future__ import annotations

import pytest

from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
    AuthorizedDevelopmentPopulation,
    PreflightValidationError,
    ProtectedDataExposureError,
    compute_candle_content_digest,
    validate_development_population_preflight,
)
from iqoption_research.models.market import Candle
from tests.unit.test_phase43_data_inventory import (
    create_deterministic_synthetic_candles,
)


def _make_candles(count: int, start_ts: int) -> list[Candle]:
    return create_deterministic_synthetic_candles(count=count, start_ts=start_ts)


def test_preflight_valid_development_population() -> None:
    """A series within development boundaries and matching digest passes preflight."""
    start_ts = 1725310800  # 2026-09-02 21:00:00 UTC
    candles = _make_candles(1500, start_ts)
    digest = compute_candle_content_digest(candles)

    contract = AuthorizedDevelopmentPopulation(
        population_id="pop_dev_01",
        symbol="EURUSD",
        expected_content_digest=digest,
        from_ts=candles[0].from_ts,
        to_ts=candles[-1].to_ts,
        max_allowed_ts=DEFAULT_PROTECTED_HOLDOUT_START_TS,
        min_candle_count=1000,
    )

    report = validate_development_population_preflight(candles, contract=contract)
    assert report.is_valid is True
    assert report.protected_data_detected is False
    assert report.status == "AUTHORIZED_DEVELOPMENT_POPULATION"
    assert report.computed_content_digest == digest


def test_preflight_fails_on_protected_evaluation_leakage() -> None:
    """Fail closed when input data extends into the protected evaluation interval."""
    # Start just before protected threshold, extending beyond it
    start_ts = DEFAULT_PROTECTED_HOLDOUT_START_TS - 300  # 5 minutes before
    candles = _make_candles(100, start_ts)  # reaches past threshold

    contract = AuthorizedDevelopmentPopulation(
        population_id="pop_leaked",
        symbol="EURUSD",
        expected_content_digest="",
        from_ts=start_ts,
        to_ts=start_ts + 6000,
        max_allowed_ts=DEFAULT_PROTECTED_HOLDOUT_START_TS,
    )

    with pytest.raises(ProtectedDataExposureError, match="Protected evaluation holdout detected"):
        validate_development_population_preflight(candles, contract=contract)


def test_preflight_fails_on_content_digest_mismatch() -> None:
    """Fail closed when actual candle content does not match the authorized digest."""
    start_ts = 1725310800
    candles = _make_candles(1200, start_ts)
    _ = compute_candle_content_digest(candles)
    tampered_digest = "a" * 64

    contract = AuthorizedDevelopmentPopulation(
        population_id="pop_tampered",
        symbol="EURUSD",
        expected_content_digest=tampered_digest,
        from_ts=candles[0].from_ts,
        to_ts=candles[-1].to_ts,
        min_candle_count=1000,
    )

    with pytest.raises(PreflightValidationError, match="Content digest mismatch"):
        validate_development_population_preflight(candles, contract=contract)


def test_preflight_fails_on_boundary_violations() -> None:
    """Fail closed when dataset timestamps exceed contract bounds."""
    start_ts = 1725310800
    candles = _make_candles(1200, start_ts)
    digest = compute_candle_content_digest(candles)

    contract = AuthorizedDevelopmentPopulation(
        population_id="pop_bounds",
        symbol="EURUSD",
        expected_content_digest=digest,
        from_ts=start_ts + 3600,  # Contract starts 1 hour later than data
        to_ts=candles[-1].to_ts,
        min_candle_count=1000,
    )

    with pytest.raises(PreflightValidationError, match="exceeds contract boundaries"):
        validate_development_population_preflight(candles, contract=contract)


def test_preflight_fails_on_insufficient_sample_count() -> None:
    """Fail closed when candle count is below contract requirement."""
    start_ts = 1725310800
    candles = _make_candles(500, start_ts)

    contract = AuthorizedDevelopmentPopulation(
        population_id="pop_small",
        symbol="EURUSD",
        expected_content_digest="",
        from_ts=start_ts,
        to_ts=start_ts + 30000,
        min_candle_count=1000,
    )

    with pytest.raises(PreflightValidationError, match="below contract minimum"):
        validate_development_population_preflight(candles, contract=contract)
