"""Protected Data Preflight & Development Population Contracts (Phase 44).

Replaces implicit slicing of historical datasets with an explicit authorized
development population contract.

Genuine research invariants:
1. An explicit AuthorizedDevelopmentPopulation contract is mandatory.
2. The contract must be complete: verified snapshot id, canonical candle content
   digest, authorized interval, instrument, timeframe, provenance, protected
   exclusions, exposure history and research plan identity.
3. The canonical candle content digest must be computed from the actual candles;
   a parquet file checksum is never accepted for this field.
4. Protected evaluation intervals remain blocked regardless of any contract.
5. Synthetic/test fixtures must use the explicit TEST_ONLY enforcement mode and
   can never masquerade as genuine research evidence.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from iqoption_research.models.market import Candle

# Authoritative protected boundary: 2026-11-12T10:00:00+00:00.
# Intervals at or after this boundary are "do not use for development" evidence.
# Derive the epoch from the ISO boundary so it cannot silently drift again.
DEFAULT_PROTECTED_HOLDOUT_START_TS: Final[int] = int(
    datetime(2026, 11, 12, 10, 0, tzinfo=UTC).timestamp()
)
DEFAULT_PROTECTED_HOLDOUT_END_TS: Final[int] = 1893456000

_SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]+$")
_SYNTHETIC_MARKERS: Final[tuple[str, ...]] = (
    "SYNTHETIC",
    "FIXTURE",
    "TEST",
    "MOCK",
    "FAKE",
)


class PreflightValidationError(ValueError):
    """Raised when an input dataset fails development population preflight validation."""


class ProtectedDataExposureError(PreflightValidationError):
    """Raised when protected evaluation holdout data is detected in a research dataset."""


class PreflightEnforcement(StrEnum):
    """Explicit enforcement mode for preflight validation.

    GENUINE is the only mode permitted for research conclusions. TEST_ONLY is an
    explicit, auditable opt-out for synthetic fixtures and methodology checks.
    """

    GENUINE = "GENUINE"
    TEST_ONLY = "TEST_ONLY"


@dataclass(frozen=True, slots=True)
class AuthorizedDevelopmentPopulation:
    """Explicit contract specifying authorized development data population boundaries.

    Fields with empty defaults are intentionally incomplete: genuine preflight
    rejects the contract until every required field is explicitly populated.
    """

    population_id: str
    symbol: str
    expected_content_digest: str  # canonical SHA-256 of candle content
    from_ts: int
    to_ts: int
    timeframe_seconds: int = 0
    snapshot_id: str = ""
    snapshot_verified: bool = False
    provenance: str = ""
    plan_id: str = ""
    max_allowed_ts: int = DEFAULT_PROTECTED_HOLDOUT_START_TS
    excluded_protected_intervals: tuple[tuple[int, int], ...] = ()
    exposure_history: tuple[str, ...] = ()
    min_candle_count: int = 1000
    max_candle_count: int | None = None
    is_development_authorized: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreflightValidationReport:
    """Audit report returned by development population preflight verification."""

    is_valid: bool
    population_id: str
    symbol: str
    candle_count: int
    computed_content_digest: str
    expected_content_digest: str | None
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    protected_data_detected: bool
    status: str
    rejection_reason: str | None = None
    snapshot_id: str = ""
    plan_id: str = ""
    provenance: str = ""
    enforcement: PreflightEnforcement = PreflightEnforcement.GENUINE
    snapshot_verified: bool = False
    timeframe_seconds: int = 0
    excluded_protected_intervals: tuple[tuple[int, int], ...] = ()
    exposure_history: tuple[str, ...] = ()
    research_disposition: str = "NO_DEMONSTRATED_EDGE"
    execution_authority: str = "NONE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_candle_content_digest(candles: Sequence[Candle]) -> str:
    """Compute deterministic cryptographic SHA-256 content digest across candle series.

    Digests timestamps and OHLCV canonically so any altered, truncated, or forged
    bar alters the hash. This is deliberately distinct from a parquet file checksum.
    """
    if not candles:
        return hashlib.sha256(b"EMPTY_SERIES").hexdigest()

    hasher = hashlib.sha256()
    for candle in candles:
        line = (
            f"{candle.from_ts}:{candle.to_ts}:{candle.open:.6f}:{candle.high:.6f}:"
            f"{candle.low:.6f}:{candle.close:.6f}:{candle.volume}\n"
        )
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def _normalize_enforcement(enforcement: PreflightEnforcement | str) -> PreflightEnforcement:
    try:
        return PreflightEnforcement(enforcement)
    except ValueError as exc:
        raise PreflightValidationError(
            f"Unknown preflight enforcement mode: {enforcement!r}"
        ) from exc


def _candle_timeframe_seconds(candle: Candle) -> int:
    size = getattr(candle, "size", None)
    seconds = getattr(size, "value", None)
    if isinstance(seconds, int) and seconds > 0:
        return seconds
    return int(candle.to_ts - candle.from_ts)


def _has_synthetic_marker(value: str) -> bool:
    upper = value.upper()
    return any(marker in upper for marker in _SYNTHETIC_MARKERS)


def _validate_genuine_contract(contract: AuthorizedDevelopmentPopulation) -> None:
    """Require a complete, explicit, verified contract for genuine research."""
    missing: list[str] = []

    def require(name: str, value: object) -> None:
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)

    require("population_id", contract.population_id)
    require("symbol", contract.symbol)
    require("expected_content_digest", contract.expected_content_digest)
    require("snapshot_id", contract.snapshot_id)
    require("provenance", contract.provenance)
    require("plan_id", contract.plan_id)

    if not contract.snapshot_verified:
        missing.append("snapshot_verified")
    if contract.timeframe_seconds <= 0:
        missing.append("timeframe_seconds")
    if not contract.excluded_protected_intervals:
        missing.append("excluded_protected_intervals")
    if not contract.exposure_history:
        missing.append("exposure_history")

    if missing:
        raise PreflightValidationError(
            "AuthorizedDevelopmentPopulation contract is incomplete for genuine research: "
            + ", ".join(missing)
            + "."
        )

    if not _SNAPSHOT_ID_RE.fullmatch(contract.snapshot_id.strip()):
        raise PreflightValidationError(
            f"Invalid verified snapshot id {contract.snapshot_id!r}; expected rds-style identifier."
        )

    digest = contract.expected_content_digest.strip().lower()
    if not _SHA256_HEX_RE.fullmatch(digest):
        raise PreflightValidationError(
            "expected_content_digest must be a 64-character SHA-256 hex string "
            "computed with compute_candle_content_digest(); empty or file checksums are rejected."
        )

    if _has_synthetic_marker(contract.provenance):
        raise PreflightValidationError(
            "Synthetic/test provenance cannot be used for genuine research preflight."
        )


def validate_development_population_preflight(
    candles: Sequence[Candle],
    contract: AuthorizedDevelopmentPopulation | None = None,
    *,
    expected_digest: str | None = None,
    max_allowed_ts: int = DEFAULT_PROTECTED_HOLDOUT_START_TS,
    allow_empty: bool = False,
    enforcement: PreflightEnforcement | str = PreflightEnforcement.GENUINE,
    verified_snapshot_id: str | None = None,
) -> PreflightValidationReport:
    """Enforce fail-closed preflight validation before feature generation or model fitting.

    Genuine mode fails closed when:
    1. No explicit AuthorizedDevelopmentPopulation contract is supplied.
    2. The contract is incomplete or its snapshot id is not verified.
    3. The canonical content digest is empty, malformed, or does not match the candles.
    4. Any bar reaches a protected evaluation interval, regardless of contract settings.
    5. Instrument, timeframe, interval or exposure-history metadata is absent or inconsistent.

    TEST_ONLY is an explicit, auditable opt-out for synthetic fixtures and
    methodology checks; its reports are never marked as genuine populations.
    """
    mode = _normalize_enforcement(enforcement)

    if mode is PreflightEnforcement.GENUINE:
        if contract is None:
            raise PreflightValidationError(
                "Explicit AuthorizedDevelopmentPopulation contract is required for genuine research."
            )
        _validate_genuine_contract(contract)
        if verified_snapshot_id is not None and verified_snapshot_id != contract.snapshot_id:
            raise PreflightValidationError(
                "Verified snapshot id does not match AuthorizedDevelopmentPopulation contract: "
                f"{verified_snapshot_id!r} != {contract.snapshot_id!r}."
            )
    elif not candles and not allow_empty:
        raise PreflightValidationError("Input candle series is empty. Preflight rejected.")

    if not candles:
        if allow_empty and mode is PreflightEnforcement.TEST_ONLY:
            return PreflightValidationReport(
                is_valid=True,
                population_id=contract.population_id if contract else "empty_test",
                symbol=contract.symbol if contract else "UNKNOWN",
                candle_count=0,
                computed_content_digest=compute_candle_content_digest(candles),
                expected_content_digest=expected_digest,
                from_ts=0,
                to_ts=0,
                from_utc="",
                to_utc="",
                protected_data_detected=False,
                status="TEST_ONLY_EMPTY",
                snapshot_id=contract.snapshot_id if contract else "",
                plan_id=contract.plan_id if contract else "",
                provenance=contract.provenance if contract else "",
                enforcement=mode,
                snapshot_verified=contract.snapshot_verified if contract else False,
                timeframe_seconds=contract.timeframe_seconds if contract else 0,
                excluded_protected_intervals=(
                    contract.excluded_protected_intervals if contract else ()
                ),
                exposure_history=contract.exposure_history if contract else (),
            )
        raise PreflightValidationError("Input candle series is empty. Preflight rejected.")

    n_candles = len(candles)
    actual_from = candles[0].from_ts
    actual_to = candles[-1].to_ts
    actual_digest = compute_candle_content_digest(candles)
    actual_symbol = getattr(
        candles[0], "symbol", "EURUSD" if mode is PreflightEnforcement.TEST_ONLY else ""
    )

    if contract is not None and not contract.is_development_authorized:
        raise PreflightValidationError(
            f"Population contract '{contract.population_id}' is not authorized for development research."
        )

    # Protected evaluation intervals are an absolute boundary in every mode.
    contract_max_ts = contract.max_allowed_ts if contract else max_allowed_ts
    contract_exclusions = contract.excluded_protected_intervals if contract else ()
    effective_max_ts = min(max_allowed_ts, contract_max_ts, DEFAULT_PROTECTED_HOLDOUT_START_TS)
    excluded_intervals = (
        (DEFAULT_PROTECTED_HOLDOUT_START_TS, DEFAULT_PROTECTED_HOLDOUT_END_TS),
        *contract_exclusions,
    )

    violating = [
        candle
        for candle in candles
        if candle.from_ts >= effective_max_ts
        or any(start <= candle.from_ts <= end for start, end in excluded_intervals)
    ]
    if violating:
        v_min = min(candle.from_ts for candle in violating)
        v_min_utc = datetime.fromtimestamp(v_min, UTC).isoformat()
        raise ProtectedDataExposureError(
            "Protected evaluation holdout detected in input dataset! "
            f"{len(violating)} bars fall in protected interval at or after {v_min_utc} "
            f"(ts {v_min}, threshold {effective_max_ts}). Implicit slicing is strictly prohibited."
        )

    contract_digest = (
        contract.expected_content_digest.strip().lower()
        if contract and contract.expected_content_digest
        else None
    )
    supplied_digest = expected_digest.strip().lower() if expected_digest else None
    if supplied_digest and contract_digest and supplied_digest != contract_digest:
        raise PreflightValidationError(
            "dataset_sha256 does not match the authorized candle content digest. "
            "A parquet file checksum is never valid for this field."
        )
    target_digest = contract_digest or supplied_digest
    if target_digest and actual_digest != target_digest:
        raise PreflightValidationError(
            f"Content digest mismatch! Computed: {actual_digest[:16]}... "
            f"Expected: {target_digest[:16]}... Dataset modified or corrupted."
        )

    if contract is not None:
        if n_candles < contract.min_candle_count:
            raise PreflightValidationError(
                f"Candle count {n_candles} is below contract minimum {contract.min_candle_count}."
            )
        if contract.max_candle_count is not None and n_candles > contract.max_candle_count:
            raise PreflightValidationError(
                f"Candle count {n_candles} exceeds contract maximum {contract.max_candle_count}."
            )

        expected_symbol = contract.symbol.strip().upper()
        mismatched_symbols = {
            str(getattr(candle, "symbol", "")).strip().upper()
            for candle in candles
            if str(getattr(candle, "symbol", "")).strip().upper() != expected_symbol
        }
        if mismatched_symbols:
            raise PreflightValidationError(
                f"Symbol mismatch: dataset contains {sorted(mismatched_symbols)}, "
                f"contract requires {expected_symbol}."
            )

        if actual_from < contract.from_ts or actual_to > contract.to_ts:
            raise PreflightValidationError(
                f"Timestamp range [{actual_from}..{actual_to}] exceeds contract boundaries "
                f"[{contract.from_ts}..{contract.to_ts}]."
            )

        if mode is PreflightEnforcement.GENUINE:
            observed_timeframes = {_candle_timeframe_seconds(candle) for candle in candles}
            if observed_timeframes != {contract.timeframe_seconds}:
                raise PreflightValidationError(
                    f"Timeframe mismatch: dataset contains {sorted(observed_timeframes)}s candles, "
                    f"contract requires {contract.timeframe_seconds}s."
                )
            synthetic_candles = [
                candle
                for candle in candles
                if _has_synthetic_marker(str(getattr(candle, "provenance", "")))
                or _has_synthetic_marker(str(getattr(candle, "identity_provenance", "")))
            ]
            if synthetic_candles:
                raise PreflightValidationError(
                    "Synthetic/test fixture candles cannot masquerade as genuine research data."
                )

    status = (
        "AUTHORIZED_DEVELOPMENT_POPULATION"
        if mode is PreflightEnforcement.GENUINE
        else "TEST_ONLY_AUTHORIZED_POPULATION"
        if contract is not None
        else "TEST_ONLY_AD_HOC"
    )

    return PreflightValidationReport(
        is_valid=True,
        population_id=contract.population_id if contract else "ad_hoc_preflight",
        symbol=actual_symbol,
        candle_count=n_candles,
        computed_content_digest=actual_digest,
        expected_content_digest=target_digest,
        from_ts=actual_from,
        to_ts=actual_to,
        from_utc=datetime.fromtimestamp(actual_from, UTC).isoformat(),
        to_utc=datetime.fromtimestamp(actual_to, UTC).isoformat(),
        protected_data_detected=False,
        status=status,
        snapshot_id=contract.snapshot_id if contract else "",
        plan_id=contract.plan_id if contract else "",
        provenance=contract.provenance if contract else "",
        enforcement=mode,
        snapshot_verified=contract.snapshot_verified if contract else False,
        timeframe_seconds=contract.timeframe_seconds if contract else 0,
        excluded_protected_intervals=(contract.excluded_protected_intervals if contract else ()),
        exposure_history=contract.exposure_history if contract else (),
    )


def validate_exploratory_preflight(
    candles: Sequence[Candle],
    contract: AuthorizedDevelopmentPopulation | None = None,
    **kwargs: Any,
) -> PreflightValidationReport:
    """Explicit test-only/exploratory opt-out; never produces a genuine report."""
    kwargs.pop("enforcement", None)
    return validate_development_population_preflight(
        candles,
        contract,
        enforcement=PreflightEnforcement.TEST_ONLY,
        **kwargs,
    )
