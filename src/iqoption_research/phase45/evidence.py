"""Genuine Phase 45 evidence consolidation over immutable local artifacts.

This module is strictly offline and read-only. It re-verifies every durable
collection receipt, consolidates candle populations by native timeframe, and
keeps the authoritative M1 snapshot separate from all authorized development
data. Missing intervals are classified against the verified market schedule;
no candle or price is ever interpolated or fabricated.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.market_data.collection_engine import (
    PROVENANCE_AUTHENTIC,
    RECEIPT_COMMITTED,
    RECEIPT_SCHEMA_VERSION,
    _compute_raw_payload_sha256,
)
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.market_data.parquet_storage import read_candles_parquet
from iqoption_research.market_data.sessions import GapClassification, get_market_schedule
from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_END_TS,
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
    compute_candle_content_digest,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.prospective import digest

AUTHORITATIVE_SYMBOL: Final[str] = "EURUSD"
AUTHORITATIVE_M1_SNAPSHOT_ID: Final[str] = "rds_eurusd_m1_20261121_105558_215496a1a71f"
AUTHORITATIVE_M1_PARQUET_SHA256: Final[str] = (
    "6398befe8e0da71a27d27c1f6e4b147e8e8ef612d474de7ec9fd168eb2ac3071"
)
AUTHORITATIVE_M1_CONTENT_DIGEST: Final[str] = (
    "dcc628747899af7daa70becdc6a13e0675f85d331c832934e17ffbd6de520d7e"
)
SUPPORTED_EVIDENCE_TIMEFRAMES: Final[tuple[int, ...]] = (60, 300, 900)
PROTECTED_EXPOSURE_CLASS: Final[str] = "PROTECTED_HISTORICALLY_REUSED"
DISPOSITION: Final[str] = "NO_DEMONSTRATED_EDGE"
EXECUTION_AUTHORITY: Final[str] = "NONE"
RESEARCH_PLAN_ID: Final[str] = "phase45_multi_timeframe_research"

_RECEIPTS_REL: Final[Path] = Path("data/collection_receipts")
_CHECKPOINTS_REL: Final[Path] = Path("data/collection_checkpoints")
_SNAPSHOTS_REL: Final[Path] = Path("data/research_datasets")
_EXTENDED_REL: Final[Path] = Path("data/extended_collection")


def _utc_iso(ts: int | None) -> str:
    return "" if ts is None else datetime.fromtimestamp(ts, UTC).isoformat()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _repo_relative(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sanitized_serialized_path(repo_root: Path, raw: object) -> str:
    """Serialize a provenance path without leaking the workstation user or drive."""
    text = str(raw or "").strip()
    if not text:
        return ""
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return "<OUTSIDE_REPOSITORY>"


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _canonical_symbol(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty symbol string")
    try:
        return InstrumentRegistry.normalize_symbol(value)
    except Exception as exc:
        raise ValueError(f"Unsupported {field}: {value!r}") from exc


def _as_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def _same_float(left: Any, right: Any) -> bool:
    """Compare receipt price bindings bit-exactly without floating-point equality lint."""
    try:
        return float(left).hex() == float(right).hex()
    except (TypeError, ValueError):
        return False


def _candle_key(candle: Candle) -> tuple[int, int, str, str, str, str, str]:
    return (
        candle.from_ts,
        candle.to_ts,
        f"{candle.open:.6f}",
        f"{candle.high:.6f}",
        f"{candle.low:.6f}",
        f"{candle.close:.6f}",
        str(candle.volume),
    )


@dataclass(frozen=True, slots=True)
class PopulationInventory:
    """Exact evidence census for one symbol and one native timeframe."""

    timeframe_seconds: int
    symbol: str
    full_unique_candles: int
    development_unique_candles: int
    protected_unique_candles: int
    earliest_ts: int | None
    latest_ts: int | None
    earliest_utc: str
    latest_utc: str
    development_earliest_ts: int | None
    development_latest_ts: int | None
    development_latest_utc: str
    market_days: int
    continuity_segments: int
    missing_intervals: tuple[dict[str, Any], ...]
    duplicate_timestamps: int
    conflicting_timestamps: int
    provenance_breakdown: dict[str, int]
    source_hashes: tuple[dict[str, Any], ...]
    content_digest: str
    development_content_digest: str
    quarantine_records: tuple[dict[str, Any], ...]
    quality_statistics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceInventory:
    """Top-level immutable evidence inventory with enforced development separation."""

    generated_at_utc: str
    symbol: str
    boundary_ts: int
    populations: dict[int, PopulationInventory]
    receipts: dict[int, dict[str, Any]]
    checkpoints: dict[str, Any]
    catalog: dict[str, Any]
    extended_sessions: dict[str, Any]
    snapshots: tuple[dict[str, Any], ...]
    conflicts: tuple[dict[str, Any], ...]
    integrity: dict[str, Any]
    separation: dict[str, Any]
    disposition: str = DISPOSITION
    execution_authority: str = EXECUTION_AUTHORITY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ReceiptEvidence:
    path: Path
    data: dict[str, Any]
    parquet_path: Path
    receipt_sha256: str
    candles: tuple[Candle, ...]


@dataclass(frozen=True, slots=True)
class _Consolidation:
    candles: tuple[Candle, ...]
    duplicate_timestamps: int
    conflicting_timestamps: int
    conflicts: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _AuthoritativeSnapshot:
    snapshot_id: str
    manifest_path: Path
    parquet_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    parquet_sha256: str
    content_digest: str
    candles: tuple[Candle, ...]
    source_hashes: tuple[dict[str, Any], ...]
    quarantine_records: tuple[dict[str, Any], ...]
    provenance_breakdown: dict[str, int]


def _verify_receipt_file(receipt_path: Path, repo_root: Path) -> _ReceiptEvidence:
    """Re-verify one durable receipt and its bound Parquet file; fail closed."""
    data = _load_json_dict(receipt_path)
    required = (
        "receipt_id",
        "batch_id",
        "job_id",
        "schema_version",
        "commit_state",
        "source_provenance",
        "symbol",
        "timeframe_seconds",
        "from_ts",
        "to_ts",
        "candle_count",
        "first_candle_open",
        "last_candle_close",
        "raw_payload_sha256",
        "parquet_sha256",
        "candle_content_sha256",
        "parquet_relative_path",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{receipt_path.name}: missing receipt fields: {sorted(missing)}")

    receipt_id = data["receipt_id"]
    batch_id = data["batch_id"]
    job_id = data["job_id"]
    if not all(isinstance(value, str) and value for value in (receipt_id, batch_id, job_id)):
        raise ValueError(f"{receipt_path.name}: receipt/job identifiers must be non-empty strings")

    timeframe_seconds = _as_positive_int(data["timeframe_seconds"], "timeframe_seconds")
    if timeframe_seconds not in SUPPORTED_EVIDENCE_TIMEFRAMES:
        raise ValueError(f"{receipt_path.name}: unsupported native timeframe {timeframe_seconds}s")
    symbol = _canonical_symbol(data["symbol"], "symbol")
    from_ts = _as_positive_int(data["from_ts"], "from_ts")
    to_ts = _as_positive_int(data["to_ts"], "to_ts")
    if to_ts < from_ts:
        raise ValueError(f"{receipt_path.name}: inverted receipt interval")

    content_sha = data["candle_content_sha256"]
    expected_id = (
        f"rcpt_{job_id}_{from_ts}_{to_ts}_{content_sha[:12].lower()}"
        if isinstance(content_sha, str)
        else ""
    )
    if receipt_path.stem != receipt_id or receipt_id != batch_id or receipt_id != expected_id:
        raise ValueError(
            f"{receipt_path.name}: receipt id/batch id/filename/content binding mismatch"
        )
    if data["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError(f"{receipt_path.name}: unsupported receipt schema_version")
    if data["commit_state"] != RECEIPT_COMMITTED:
        raise ValueError(f"{receipt_path.name}: receipt is not COMMITTED")
    if data["source_provenance"] != PROVENANCE_AUTHENTIC:
        raise ValueError(f"{receipt_path.name}: non-authentic receipt provenance")

    rel_path = data["parquet_relative_path"]
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError(f"{receipt_path.name}: missing parquet_relative_path")
    parquet_path = (repo_root / rel_path).resolve()
    try:
        parquet_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{receipt_path.name}: parquet path escapes repository root") from exc
    if not parquet_path.is_file():
        raise ValueError(f"{receipt_path.name}: bound parquet file does not exist")

    claimed_parquet_sha = str(data["parquet_sha256"]).lower()
    actual_parquet_sha = _sha256_file(parquet_path)
    if actual_parquet_sha != claimed_parquet_sha:
        raise ValueError(f"{receipt_path.name}: parquet SHA-256 mismatch")
    candles = read_candles_parquet(
        parquet_path,
        expected_symbol=symbol,
        verify_checksum=True,
        expected_checksum=actual_parquet_sha,
    )
    if not candles:
        raise ValueError(f"{receipt_path.name}: bound parquet contains no candles")
    if len(candles) != data["candle_count"]:
        raise ValueError(f"{receipt_path.name}: candle count mismatch")
    if candles[0].from_ts != from_ts or candles[-1].to_ts != to_ts:
        raise ValueError(f"{receipt_path.name}: receipt time boundaries do not match parquet")
    if not _same_float(candles[0].open, data["first_candle_open"]):
        raise ValueError(f"{receipt_path.name}: first candle open mismatch")
    if not _same_float(candles[-1].close, data["last_candle_close"]):
        raise ValueError(f"{receipt_path.name}: last candle close mismatch")
    if compute_candle_content_digest(candles) != str(content_sha).lower():
        raise ValueError(f"{receipt_path.name}: candle content digest mismatch")
    if _compute_raw_payload_sha256(candles) != str(data["raw_payload_sha256"]).lower():
        raise ValueError(f"{receipt_path.name}: raw payload SHA-256 mismatch")

    observed: set[int] = set()
    for index, candle in enumerate(candles):
        if (
            candle.size.value != timeframe_seconds
            or candle.to_ts - candle.from_ts != timeframe_seconds
        ):
            raise ValueError(f"{receipt_path.name}: timeframe mismatch at candle {index}")
        if _canonical_symbol(candle.symbol, "candle.symbol") != symbol:
            raise ValueError(f"{receipt_path.name}: symbol mismatch at candle {index}")
        if candle.from_ts in observed:
            raise ValueError(f"{receipt_path.name}: duplicate timestamp inside receipt parquet")
        observed.add(candle.from_ts)

    return _ReceiptEvidence(
        path=receipt_path,
        data=data,
        parquet_path=parquet_path,
        receipt_sha256=_sha256_file(receipt_path),
        candles=tuple(candles),
    )


def _verify_all_receipts_detail(repo_root: Path) -> dict[int, tuple[_ReceiptEvidence, ...]]:
    receipts_dir = repo_root / _RECEIPTS_REL
    grouped: dict[int, list[_ReceiptEvidence]] = {}
    if not receipts_dir.is_dir():
        return {}
    for receipt_path in sorted(receipts_dir.glob("rcpt_*.json")):
        receipt = _verify_receipt_file(receipt_path, repo_root)
        grouped.setdefault(int(receipt.data["timeframe_seconds"]), []).append(receipt)
    return {timeframe: tuple(items) for timeframe, items in grouped.items()}


def _summarize_receipt_group(
    timeframe_seconds: int,
    receipts: tuple[_ReceiptEvidence, ...],
    repo_root: Path,
    boundary_ts: int,
) -> dict[str, Any]:
    by_timestamp: dict[int, Candle] = {}
    duplicates = 0
    for receipt in receipts:
        for candle in receipt.candles:
            if candle.from_ts in by_timestamp:
                duplicates += 1
            else:
                by_timestamp[candle.from_ts] = candle
    sorted_ts = sorted(by_timestamp)
    return {
        "timeframe_seconds": timeframe_seconds,
        "receipt_count": len(receipts),
        "receipt_ids": tuple(str(item.data["receipt_id"]) for item in receipts),
        "receipt_sha256": tuple(item.receipt_sha256 for item in receipts),
        "parquet_sha256": tuple(str(item.data["parquet_sha256"]).lower() for item in receipts),
        "parquet_paths": tuple(_repo_relative(repo_root, item.parquet_path) for item in receipts),
        "unique_candles": len(by_timestamp),
        "development_candles": sum(1 for ts in sorted_ts if ts < boundary_ts),
        "protected_candles": sum(1 for ts in sorted_ts if ts >= boundary_ts),
        "duplicate_timestamps": duplicates,
        "conflicting_timestamps": 0,
        "verification": "PASS",
        "provenance": PROVENANCE_AUTHENTIC,
        "commit_state": RECEIPT_COMMITTED,
        "schema_version": RECEIPT_SCHEMA_VERSION,
    }


def verify_all_receipts(repo_root: Path) -> dict[int, dict[str, Any]]:
    """Re-verify every durable receipt and return a JSON-safe summary by timeframe.

    Checkpoint status is never consulted: only the receipt JSON and its bound
    Parquet file are treated as proof.
    """
    root = repo_root.resolve()
    boundary_ts = DEFAULT_PROTECTED_HOLDOUT_START_TS
    details = _verify_all_receipts_detail(root)
    return {
        timeframe: _summarize_receipt_group(timeframe, receipts, root, boundary_ts)
        for timeframe, receipts in sorted(details.items())
    }


def _consolidate(
    sources: tuple[tuple[str, tuple[Candle, ...]], ...],
    *,
    timeframe_seconds: int,
    known_conflict_timestamps: frozenset[int] = frozenset(),
) -> _Consolidation:
    """Deduplicate candle sources by timestamp with fail-closed conflict detection."""
    by_timestamp: dict[int, Candle] = {}
    duplicates = 0
    conflicts: list[dict[str, Any]] = []
    for source_label, candles in sources:
        for candle in candles:
            if candle.size.value != timeframe_seconds:
                raise ValueError(
                    f"{source_label}: candle timeframe {candle.size.value}s does not match "
                    f"population timeframe {timeframe_seconds}s"
                )
            if candle.to_ts - candle.from_ts != timeframe_seconds:
                raise ValueError(f"{source_label}: non-native candle duration at {candle.from_ts}")
            existing = by_timestamp.get(candle.from_ts)
            if existing is None:
                by_timestamp[candle.from_ts] = replace(
                    candle,
                    provenance=source_label,
                    identity_provenance="explicit_broker_metadata",
                )
                continue
            if _candle_key(existing) == _candle_key(candle):
                duplicates += 1
                continue
            if candle.from_ts not in known_conflict_timestamps:
                raise ValueError(
                    f"Unquarantined OHLCV conflict at timestamp {candle.from_ts} "
                    f"({_candle_key(existing)} != {_candle_key(candle)})"
                )
            conflicts.append({
                "from_ts": candle.from_ts,
                "from_utc": _utc_iso(candle.from_ts),
                "source": source_label,
                "reason": "KNOWN_AUTHORITATIVE_CONFLICT_SKIPPED",
            })
    return _Consolidation(
        candles=tuple(by_timestamp[key] for key in sorted(by_timestamp)),
        duplicate_timestamps=duplicates,
        conflicting_timestamps=len(conflicts),
        conflicts=tuple(conflicts),
    )


def _continuity_statistics(
    candles: tuple[Candle, ...],
    *,
    timeframe_seconds: int,
    symbol: str,
) -> tuple[int, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Compute physical gaps and schedule-aware missing open-market intervals."""
    if not candles:
        return (
            0,
            (),
            {
                "gap_count": 0,
                "scheduled_closure_count": 0,
                "unscheduled_gap_count": 0,
                "missing_open_market_candles": 0,
            },
        )

    schedule = get_market_schedule(symbol)
    gaps: list[dict[str, Any]] = []
    for previous, current in itertools.pairwise(candles):
        if current.from_ts < previous.to_ts:
            raise ValueError(
                f"Overlapping candles at {previous.to_ts}/{current.from_ts}; cannot consolidate"
            )
        if current.from_ts == previous.to_ts:
            continue
        gap_from = previous.to_ts
        gap_to = current.from_ts
        classification = schedule.classify_gap(
            datetime.fromtimestamp(gap_from, UTC),
            datetime.fromtimestamp(gap_to, UTC),
        )
        if classification in (
            GapClassification.SCHEDULED_WEEKEND_CLOSURE,
            GapClassification.UNVERIFIED_GAP,
        ):
            missing_candles = 0
        else:
            missing_candles = 0
            cursor = gap_from
            while cursor < gap_to:
                if schedule.is_market_open(datetime.fromtimestamp(cursor, UTC)):
                    missing_candles += 1
                cursor += timeframe_seconds
        gaps.append({
            "from_ts": gap_from,
            "to_ts": gap_to,
            "from_utc": _utc_iso(gap_from),
            "to_utc": _utc_iso(gap_to),
            "duration_seconds": gap_to - gap_from,
            "classification": classification.value,
            "is_scheduled_closure": classification is GapClassification.SCHEDULED_WEEKEND_CLOSURE,
            "missing_candle_count": missing_candles,
            "expected_candles_count": missing_candles,
        })

    statistics = {
        "gap_count": len(gaps),
        "scheduled_closure_count": sum(1 for gap in gaps if gap["is_scheduled_closure"]),
        "unscheduled_gap_count": sum(1 for gap in gaps if not gap["is_scheduled_closure"]),
        "missing_open_market_candles": sum(int(gap["missing_candle_count"]) for gap in gaps),
    }
    return len(gaps) + 1, tuple(gaps), statistics


def _breakdown(candles: tuple[Candle, ...] | list[Candle]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candle in candles:
        label = str(candle.provenance)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _build_population_inventory(
    *,
    timeframe_seconds: int,
    symbol: str,
    candles: tuple[Candle, ...],
    boundary_ts: int,
    duplicate_timestamps: int,
    conflicting_timestamps: int,
    source_hashes: tuple[dict[str, Any], ...],
    quarantine_records: tuple[dict[str, Any], ...],
    development_candles: tuple[Candle, ...] | None = None,
    continuity_candles: tuple[Candle, ...] | None = None,
    extra_quality: dict[str, Any] | None = None,
) -> PopulationInventory:
    ordered = tuple(sorted(candles, key=lambda candle: candle.from_ts))
    development_source = ordered if development_candles is None else development_candles
    continuity_source = ordered if continuity_candles is None else continuity_candles
    development = tuple(
        candle
        for candle in sorted(development_source, key=lambda candle: candle.from_ts)
        if candle.from_ts < boundary_ts
    )
    assert not any(candle.from_ts >= boundary_ts for candle in development), (  # noqa: S101
        "development population contains protected-boundary candles"
    )

    segments, gaps, gap_statistics = _continuity_statistics(
        tuple(sorted(continuity_source, key=lambda candle: candle.from_ts)),
        timeframe_seconds=timeframe_seconds,
        symbol=symbol,
    )
    earliest_ts = ordered[0].from_ts if ordered else None
    latest_ts = ordered[-1].to_ts if ordered else None
    development_earliest_ts = development[0].from_ts if development else None
    development_latest_ts = development[-1].to_ts if development else None
    quality: dict[str, Any] = {
        **gap_statistics,
        "continuity_segments": segments,
        "duplicate_timestamps": duplicate_timestamps,
        "conflicting_timestamps": conflicting_timestamps,
        "development_boundary_ts": boundary_ts,
        "development_earliest_utc": _utc_iso(development_earliest_ts),
        "development_latest_utc": _utc_iso(development_latest_ts),
        "latest_candle_from_ts": ordered[-1].from_ts if ordered else None,
        "protected_exposure_class": PROTECTED_EXPOSURE_CLASS,
        "development_provenance_breakdown": _breakdown(development),
        "interpolation_permitted": False,
    }
    if extra_quality:
        quality.update(extra_quality)

    return PopulationInventory(
        timeframe_seconds=timeframe_seconds,
        symbol=symbol,
        full_unique_candles=len(ordered),
        development_unique_candles=len(development),
        protected_unique_candles=sum(1 for candle in ordered if candle.from_ts >= boundary_ts),
        earliest_ts=earliest_ts,
        latest_ts=latest_ts,
        earliest_utc=_utc_iso(earliest_ts),
        latest_utc=_utc_iso(latest_ts),
        development_earliest_ts=development_earliest_ts,
        development_latest_ts=development_latest_ts,
        development_latest_utc=_utc_iso(development_latest_ts),
        market_days=len({datetime.fromtimestamp(candle.from_ts, UTC).date() for candle in ordered}),
        continuity_segments=segments,
        missing_intervals=gaps,
        duplicate_timestamps=duplicate_timestamps,
        conflicting_timestamps=conflicting_timestamps,
        provenance_breakdown=_breakdown(ordered),
        source_hashes=source_hashes,
        content_digest=compute_candle_content_digest(ordered),
        development_content_digest=compute_candle_content_digest(development),
        quarantine_records=quarantine_records,
        quality_statistics=quality,
    )


def _verify_manifest_sha256(manifest_path: Path, data: dict[str, Any]) -> str:
    claimed = data.get("manifest_sha256")
    if not isinstance(claimed, str) or not claimed:
        raise ValueError(f"{manifest_path}: missing manifest_sha256")
    clean = {key: value for key, value in data.items() if key != "manifest_sha256"}
    actual = digest(clean)
    if actual != claimed:
        raise ValueError(f"{manifest_path}: manifest self-digest mismatch")
    return claimed


def _load_authoritative_m1_snapshot(repo_root: Path) -> _AuthoritativeSnapshot:
    snapshots_dir = repo_root / _SNAPSHOTS_REL
    candidates: list[_AuthoritativeSnapshot] = []
    if not snapshots_dir.is_dir():
        raise ValueError(f"No research snapshot directory found at {snapshots_dir}")

    for manifest_path in sorted(snapshots_dir.glob("*/manifest.json")):
        data = _load_json_dict(manifest_path)
        if data.get("symbol") != AUTHORITATIVE_SYMBOL:
            continue
        if data.get("candle_size_seconds") != 60:
            continue
        snapshot_id = str(data.get("snapshot_id", manifest_path.parent.name))
        manifest_sha = _verify_manifest_sha256(manifest_path, data)
        parquet_path = manifest_path.parent / "candles.parquet"
        if not parquet_path.is_file():
            continue
        parquet_sha = _sha256_file(parquet_path)
        if parquet_sha != str(data.get("sha256_checksum", "")).lower():
            continue
        expected_count = data.get("total_candles_retained")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise ValueError(f"{manifest_path}: invalid total_candles_retained")
        candles = read_candles_parquet(
            parquet_path,
            expected_symbol=AUTHORITATIVE_SYMBOL,
            verify_checksum=True,
            expected_checksum=parquet_sha,
        )
        if len(candles) != expected_count:
            raise ValueError(f"{manifest_path}: retained candle count mismatch")
        seen: set[int] = set()
        for candle in candles:
            if candle.from_ts in seen:
                raise ValueError(f"{manifest_path}: duplicate timestamp in authoritative snapshot")
            seen.add(candle.from_ts)
        content_digest = compute_candle_content_digest(candles)
        if snapshot_id == AUTHORITATIVE_M1_SNAPSHOT_ID:
            if parquet_sha != AUTHORITATIVE_M1_PARQUET_SHA256:
                raise ValueError("Authoritative M1 snapshot parquet checksum drift detected")
            if content_digest != AUTHORITATIVE_M1_CONTENT_DIGEST:
                raise ValueError("Authoritative M1 snapshot content digest drift detected")

        source_hashes: list[dict[str, Any]] = [
            {
                "path": _repo_relative(repo_root, parquet_path),
                "sha256": parquet_sha,
                "role": "AUTHORITATIVE_SNAPSHOT_PARQUET",
                "snapshot_id": snapshot_id,
            },
            {
                "path": _repo_relative(repo_root, manifest_path),
                "sha256": manifest_sha,
                "role": "AUTHORITATIVE_SNAPSHOT_MANIFEST",
                "snapshot_id": snapshot_id,
            },
        ]
        for source in data.get("source_files", []):
            if not isinstance(source, dict):
                continue
            source_hashes.append({
                "path": _sanitized_serialized_path(repo_root, source.get("path", "")),
                "sha256": str(source.get("sha256", "")),
                "role": "AUTHORITATIVE_SOURCE_FILE",
                "source_id": str(source.get("source_id", "")),
                "provenance": str(source.get("provenance", "")),
            })
        quarantine = tuple(
            item for item in data.get("conflicts_quarantined", []) if isinstance(item, dict)
        )
        breakdown_raw = data.get("provenance_breakdown", {})
        breakdown = (
            {str(key): int(value) for key, value in breakdown_raw.items()}
            if isinstance(breakdown_raw, dict)
            else {}
        )
        candidates.append(
            _AuthoritativeSnapshot(
                snapshot_id=snapshot_id,
                manifest_path=manifest_path,
                parquet_path=parquet_path,
                manifest=data,
                manifest_sha256=manifest_sha,
                parquet_sha256=parquet_sha,
                content_digest=content_digest,
                candles=tuple(sorted(candles, key=lambda candle: candle.from_ts)),
                source_hashes=tuple(source_hashes),
                quarantine_records=quarantine,
                provenance_breakdown=breakdown,
            )
        )

    if not candidates:
        raise ValueError("No verified authoritative EURUSD M1 snapshot found")
    authoritative = next(
        (item for item in candidates if item.snapshot_id == AUTHORITATIVE_M1_SNAPSHOT_ID),
        None,
    )
    if authoritative is None:
        authoritative = min(
            candidates,
            key=lambda item: str(item.manifest.get("created_at_utc", "")),
        )
    return authoritative


def _summarize_checkpoints(repo_root: Path) -> dict[str, Any]:
    checkpoint_dir = repo_root / _CHECKPOINTS_REL
    records: dict[str, Any] = {}
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.glob("*.json")):
            try:
                data = _load_json_dict(path)
            except ValueError:
                continue
            records[path.stem] = {
                "sha256": _sha256_file(path),
                "job_id": str(data.get("job_id", path.stem)),
                "status": str(data.get("status", "UNKNOWN")),
                "symbol": str(data.get("symbol", "")),
                "timeframe_seconds": data.get("timeframe_seconds"),
                "receipt_count": len(data.get("receipt_ids", [])),
                "trusted_as_proof": False,
            }
    return {
        "records": records,
        "checkpoint_count": len(records),
        "status_trusted_as_proof": False,
        "note": "Checkpoint status is informational only; receipts are re-verified independently.",
    }


def _summarize_catalog(repo_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in ("data/datasets/catalog.db", "data/parquet_candles/catalog.db"):
        path = repo_root / relative
        if path.is_file():
            files.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "files": tuple(files),
        "is_authoritative_for_receipts": False,
        "note": "Catalog metadata is informational; source hashes are bound by durable receipts.",
    }


def _summarize_extended_sessions(repo_root: Path) -> dict[str, Any]:
    root = repo_root / _EXTENDED_REL
    records: dict[str, Any] = {}
    if root.is_dir():
        for session_dir in sorted(root.glob("collection_*")):
            manifest_path = session_dir / "manifest.json"
            session_path = session_dir / "session.json"
            selected = manifest_path if manifest_path.is_file() else session_path
            if not selected.is_file():
                continue
            try:
                data = _load_json_dict(selected)
            except ValueError:
                continue
            final_state = data.get("final_state")
            if not isinstance(final_state, dict):
                final_state = {}
            records[session_dir.name] = {
                "manifest_path": _repo_relative(repo_root, selected),
                "sha256": _sha256_file(selected),
                "valid_candles": final_state.get("valid_candles", 0),
                "integrity": str(final_state.get("integrity", data.get("integrity", "UNKNOWN"))),
                "trusted_as_proof": False,
            }
    return {
        "sessions": records,
        "session_count": len(records),
        "trusted_as_proof": False,
    }


def _summarize_snapshots(repo_root: Path, authoritative_id: str) -> tuple[dict[str, Any], ...]:
    snapshots: list[dict[str, Any]] = []
    snapshot_root = repo_root / _SNAPSHOTS_REL
    if not snapshot_root.is_dir():
        return ()
    for manifest_path in sorted(snapshot_root.glob("*/manifest.json")):
        try:
            data = _load_json_dict(manifest_path)
            manifest_sha = _verify_manifest_sha256(manifest_path, data)
            parquet_path = manifest_path.parent / "candles.parquet"
            parquet_sha = _sha256_file(parquet_path) if parquet_path.is_file() else ""
            verified = parquet_sha == str(data.get("sha256_checksum", "")).lower()
        except ValueError:
            manifest_sha = ""
            parquet_sha = ""
            verified = False
        snapshots.append({
            "snapshot_id": str(data.get("snapshot_id", manifest_path.parent.name)),
            "symbol": str(data.get("symbol", "")),
            "timeframe_seconds": data.get("candle_size_seconds"),
            "candle_count": data.get("total_candles_retained", 0),
            "parquet_sha256": parquet_sha,
            "manifest_sha256": manifest_sha,
            "verified": verified,
            "is_authoritative": data.get("snapshot_id") == authoritative_id,
        })
    return tuple(snapshots)


def build_evidence_inventory(
    repo_root: Path,
    *,
    boundary_ts: int = DEFAULT_PROTECTED_HOLDOUT_START_TS,
) -> EvidenceInventory:
    """Build a complete, fail-closed evidence inventory from immutable local artifacts."""
    root = repo_root.resolve()
    receipt_details = _verify_all_receipts_detail(root)
    receipts = {
        timeframe: _summarize_receipt_group(timeframe, items, root, boundary_ts)
        for timeframe, items in sorted(receipt_details.items())
    }
    snapshot = _load_authoritative_m1_snapshot(root)
    authoritative_conflict_ts = frozenset(
        int(item.get("boundary_ts", -1))
        for item in snapshot.quarantine_records
        if isinstance(item.get("boundary_ts", None), int)
    )

    m1_consolidation = _consolidate(
        (
            ("AUTHORITATIVE_SNAPSHOT", snapshot.candles),
            *(
                (
                    "AUTHENTIC_BROKER_COLLECTION",
                    tuple(
                        candle
                        for receipt in receipt_details.get(60, ())
                        for candle in receipt.candles
                    ),
                ),
            ),
        ),
        timeframe_seconds=60,
        known_conflict_timestamps=authoritative_conflict_ts,
    )
    m1_source_hashes = (
        snapshot.source_hashes
        + tuple(
            {
                "path": _repo_relative(root, receipt.path),
                "sha256": receipt.receipt_sha256,
                "role": "RECEIPT_JSON",
                "receipt_id": str(receipt.data["receipt_id"]),
            }
            for receipt in receipt_details.get(60, ())
        )
        + tuple(
            {
                "path": _repo_relative(root, receipt.parquet_path),
                "sha256": str(receipt.data["parquet_sha256"]).lower(),
                "role": "RECEIPT_PARQUET",
                "receipt_id": str(receipt.data["receipt_id"]),
            }
            for receipt in receipt_details.get(60, ())
        )
    )
    m1_extra_quality = {
        "authoritative_snapshot_id": snapshot.snapshot_id,
        "authoritative_parquet_sha256": snapshot.parquet_sha256,
        "authoritative_content_digest": snapshot.content_digest,
        "authoritative_manifest_provenance_breakdown": dict(snapshot.provenance_breakdown),
        "authoritative_snapshot_candles": len(snapshot.candles),
        "receipt_bound_candles": len(m1_consolidation.candles) - len(snapshot.candles),
    }

    populations: dict[int, PopulationInventory] = {
        60: _build_population_inventory(
            timeframe_seconds=60,
            symbol=AUTHORITATIVE_SYMBOL,
            candles=m1_consolidation.candles,
            boundary_ts=boundary_ts,
            duplicate_timestamps=m1_consolidation.duplicate_timestamps,
            conflicting_timestamps=len(
                authoritative_conflict_ts
                | {
                    int(conflict["from_ts"])
                    for conflict in m1_consolidation.conflicts
                    if isinstance(conflict.get("from_ts"), int)
                }
            ),
            source_hashes=m1_source_hashes,
            quarantine_records=snapshot.quarantine_records,
            development_candles=snapshot.candles,
            continuity_candles=snapshot.candles,
            extra_quality=m1_extra_quality,
        )
    }

    for timeframe in (300, 900):
        group = receipt_details.get(timeframe, ())
        if not group:
            continue
        consolidation = _consolidate(
            (
                (
                    "AUTHENTIC_BROKER_COLLECTION",
                    tuple(candle for receipt in group for candle in receipt.candles),
                ),
            ),
            timeframe_seconds=timeframe,
        )
        source_hashes = tuple(
            item
            for receipt in group
            for item in (
                {
                    "path": _repo_relative(root, receipt.path),
                    "sha256": receipt.receipt_sha256,
                    "role": "RECEIPT_JSON",
                    "receipt_id": str(receipt.data["receipt_id"]),
                },
                {
                    "path": _repo_relative(root, receipt.parquet_path),
                    "sha256": str(receipt.data["parquet_sha256"]).lower(),
                    "role": "RECEIPT_PARQUET",
                    "receipt_id": str(receipt.data["receipt_id"]),
                },
            )
        )
        populations[timeframe] = _build_population_inventory(
            timeframe_seconds=timeframe,
            symbol=AUTHORITATIVE_SYMBOL,
            candles=consolidation.candles,
            boundary_ts=boundary_ts,
            duplicate_timestamps=consolidation.duplicate_timestamps,
            conflicting_timestamps=consolidation.conflicting_timestamps,
            source_hashes=source_hashes,
            quarantine_records=(),
        )

    snapshot_summaries = _summarize_snapshots(root, snapshot.snapshot_id)
    conflict_by_timestamp: dict[int, dict[str, Any]] = {}
    for conflict in snapshot.quarantine_records:
        conflict_by_timestamp[int(conflict.get("boundary_ts", -1))] = conflict
    for population in populations.values():
        for conflict in population.quarantine_records:
            conflict_by_timestamp.setdefault(int(conflict.get("boundary_ts", -1)), conflict)
    conflicts = tuple(
        conflict_by_timestamp[key] for key in sorted(conflict_by_timestamp, key=lambda value: value)
    )
    integrity = {
        "receipts_verified": True,
        "receipt_file_count": sum(len(items) for items in receipt_details.values()),
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_provenance": PROVENANCE_AUTHENTIC,
        "receipt_commit_state": RECEIPT_COMMITTED,
        "authoritative_snapshot_id": snapshot.snapshot_id,
        "authoritative_parquet_sha256": snapshot.parquet_sha256,
        "authoritative_content_digest": snapshot.content_digest,
        "authoritative_manifest_sha256": snapshot.manifest_sha256,
        "checkpoint_status_trusted": False,
        "catalog_metadata_trusted_as_proof": False,
        "no_interpolation_or_fabrication": True,
        "all_sources_read_only": True,
    }
    separation = {
        "boundary_ts": boundary_ts,
        "boundary_utc": _utc_iso(boundary_ts),
        "protected_interval": (boundary_ts, DEFAULT_PROTECTED_HOLDOUT_END_TS),
        "development_rule": "from_ts < boundary_ts",
        "boundary_touching_candles": "RETAINED_WHEN_from_ts_LT_boundary_ts",
        "to_ts_equals_boundary_allowed": True,
        "to_ts_strictly_pre_boundary_required": False,
        "label_maturity_invalidation": "max_label_maturity_ts",
        "protected_exposure_class": PROTECTED_EXPOSURE_CLASS,
        "m1_development_authority": snapshot.snapshot_id,
        "m5_m15_authority": "VERIFIED_RECEIPT_BOUND_PARQUETS",
        "plan_id": RESEARCH_PLAN_ID,
    }

    return EvidenceInventory(
        generated_at_utc=datetime.now(UTC).isoformat(),
        symbol=AUTHORITATIVE_SYMBOL,
        boundary_ts=boundary_ts,
        populations=populations,
        receipts=receipts,
        checkpoints=_summarize_checkpoints(root),
        catalog=_summarize_catalog(root),
        extended_sessions=_summarize_extended_sessions(root),
        snapshots=snapshot_summaries,
        conflicts=conflicts,
        integrity=integrity,
        separation=separation,
        disposition=DISPOSITION,
        execution_authority=EXECUTION_AUTHORITY,
    )


def load_population_candles(
    repo_root: Path,
    timeframe_seconds: int,
    *,
    development_only: bool = True,
    boundary_ts: int = DEFAULT_PROTECTED_HOLDOUT_START_TS,
) -> list[Candle]:
    """Load a reconciled native-timeframe population, sorted by ``from_ts``.

    M1 development candles come exclusively from the authoritative snapshot.
    M5/M15 candles come exclusively from verified receipt-bound native Parquet
    files. Protected records are never returned in development mode.

    Membership is determined solely by ``from_ts < boundary_ts``. A retained
    candle may have ``to_ts == boundary_ts``; strict pre-boundary ``to_ts`` is
    not required. Label maturity is invalidated separately downstream via
    ``max_label_maturity_ts``.
    """
    if timeframe_seconds not in SUPPORTED_EVIDENCE_TIMEFRAMES:
        raise ValueError(
            f"Unsupported Phase 45 timeframe {timeframe_seconds}s; "
            f"expected one of {SUPPORTED_EVIDENCE_TIMEFRAMES}"
        )
    root = repo_root.resolve()

    if timeframe_seconds == 60:
        snapshot = _load_authoritative_m1_snapshot(root)
        sources: tuple[tuple[str, tuple[Candle, ...]], ...] = (
            ("AUTHORITATIVE_SNAPSHOT", snapshot.candles),
        )
        if not development_only:
            receipt_details = _verify_all_receipts_detail(root)
            if receipt_details.get(60):
                sources += (
                    (
                        "AUTHENTIC_BROKER_COLLECTION",
                        tuple(
                            candle for receipt in receipt_details[60] for candle in receipt.candles
                        ),
                    ),
                )
        known_conflicts = frozenset(
            int(item["boundary_ts"])
            for item in snapshot.quarantine_records
            if isinstance(item.get("boundary_ts"), int)
        )
        consolidation = _consolidate(
            sources,
            timeframe_seconds=60,
            known_conflict_timestamps=known_conflicts,
        )
        if development_only:
            consolidation = _Consolidation(
                candles=tuple(
                    candle for candle in consolidation.candles if candle.from_ts < boundary_ts
                ),
                duplicate_timestamps=consolidation.duplicate_timestamps,
                conflicting_timestamps=consolidation.conflicting_timestamps,
                conflicts=consolidation.conflicts,
            )
    else:
        receipt_details = _verify_all_receipts_detail(root)
        group = receipt_details.get(timeframe_seconds, ())
        consolidation = _consolidate(
            (
                (
                    "AUTHENTIC_BROKER_COLLECTION",
                    tuple(candle for receipt in group for candle in receipt.candles),
                ),
            ),
            timeframe_seconds=timeframe_seconds,
        )
        if development_only:
            consolidation = _Consolidation(
                candles=tuple(
                    candle for candle in consolidation.candles if candle.from_ts < boundary_ts
                ),
                duplicate_timestamps=consolidation.duplicate_timestamps,
                conflicting_timestamps=consolidation.conflicting_timestamps,
                conflicts=consolidation.conflicts,
            )

    candles = list(consolidation.candles)
    candles.sort(key=lambda candle: candle.from_ts)
    if development_only:
        assert not any(candle.from_ts >= boundary_ts for candle in candles), (  # noqa: S101
            "protected-boundary candle leaked into development population"
        )
    return candles
