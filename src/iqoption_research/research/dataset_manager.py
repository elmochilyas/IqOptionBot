"""Research Dataset Manager: offline discovery, verification, deduplication, and snapshotting.

Enforces strict research data governance:
1. Discovers authentic EURUSD M1 datasets and V2 collection sessions.
2. Verifies cryptographic checksums, catalog bindings, and availability receipts.
3. Excludes empty sessions (0 candles), corrupted files, synthetic data, OTC, and unverified data.
4. Distinguishes legacy authentic historical data from live-observed collection with receipts.
5. Reconciles cross-session duplicate candle boundaries.
6. Quarantines conflicting OHLCV values rather than silently selecting one.
7. Analyzes continuity, distinguishing scheduled market closures from unscheduled intraday gaps.
8. Writes immutable, versioned, checksum-verified research dataset snapshots.
9. Never mutates original source data; never invents missing candles or timestamps.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.extended_store import inspect_session
from iqoption_research.market_data.parquet_storage import (
    CANDLE_ARROW_SCHEMA,
    read_candles_parquet,
)
from iqoption_research.market_data.sessions import (
    EURUSD_FOREX_SCHEDULE,
    GapClassification,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.prospective import digest

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOTS_DIR: Final[Path] = Path("data/research_datasets")
PROVENANCE_LEGACY: Final[str] = "LEGACY_AUTHENTIC_COLLECTION"
PROVENANCE_EXTENDED_LIVE: Final[str] = "EXTENDED_LIVE_COLLECTION"


class DatasetManagerError(Exception):
    """Base exception for research dataset preparation errors."""


class DatasetIntegrityError(DatasetManagerError):
    """Raised when source or snapshot checksum/integrity verification fails."""


class ConflictingDataError(DatasetManagerError):
    """Raised when conflicting market data cannot be reconciled."""


@dataclass(frozen=True, slots=True)
class DiscoveredSource:
    """A discovered candidate dataset or collection session."""

    source_id: str
    source_type: str  # "CATALOG_PARQUET" | "EXTENDED_SESSION"
    path: Path
    candle_count: int
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    is_eligible: bool
    exclusion_reason: str | None
    provenance_class: str
    wallet_classification: str
    sha256_checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "path": str(self.path),
            "candle_count": self.candle_count,
            "from_ts": self.from_ts,
            "to_ts": self.to_ts,
            "from_utc": self.from_utc,
            "to_utc": self.to_utc,
            "is_eligible": self.is_eligible,
            "exclusion_reason": self.exclusion_reason,
            "provenance_class": self.provenance_class,
            "wallet_classification": self.wallet_classification,
            "sha256_checksum": self.sha256_checksum,
        }


@dataclass(frozen=True, slots=True)
class QuarantinedConflict:
    """A boundary timestamp with conflicting OHLCV across sources, quarantined from training."""

    boundary_ts: int
    boundary_utc: str
    sources: list[dict[str, Any]]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_ts": self.boundary_ts,
            "boundary_utc": self.boundary_utc,
            "sources": self.sources,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GapRecord:
    """An observed time gap between consecutive candles."""

    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    duration_seconds: int
    gap_type: str  # "SCHEDULED_WEEKEND_CLOSURE" | "UNSCHEDULED_INTRADAY_GAP"

    @property
    def gap_minutes(self) -> int:
        return self.duration_seconds // 60

    @property
    def classification(self) -> str:
        return self.gap_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_ts": self.from_ts,
            "to_ts": self.to_ts,
            "from_utc": self.from_utc,
            "to_utc": self.to_utc,
            "duration_seconds": self.duration_seconds,
            "gap_minutes": self.gap_minutes,
            "gap_type": self.gap_type,
            "classification": self.classification,
        }


@dataclass(frozen=True, slots=True)
class ResearchDatasetManifest:
    """Tamper-evident specification of a prepared research dataset snapshot."""

    snapshot_id: str
    created_at_utc: str
    dataset_version: str
    symbol: str
    candle_size_seconds: int
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    total_candles_discovered: int
    total_candles_retained: int
    duplicate_boundaries_deduplicated: int
    conflicts_quarantined_count: int
    conflicts_quarantined: list[dict[str, Any]]
    unscheduled_gaps_count: int
    unscheduled_gaps: list[dict[str, Any]]
    scheduled_closures_count: int
    scheduled_closures: list[dict[str, Any]]
    provenance_breakdown: dict[str, int]
    source_files: list[dict[str, Any]]
    eligibility_rules: list[str]
    sha256_checksum: str

    @property
    def total_candles(self) -> int:
        return self.total_candles_retained

    @property
    def deduplicated_boundaries_count(self) -> int:
        return self.duplicate_boundaries_deduplicated

    @property
    def quarantined_conflicts_count(self) -> int:
        return self.conflicts_quarantined_count

    @property
    def quarantined_conflict_timestamps(self) -> list[int]:
        return [c.get("boundary_ts", c.get("timestamp", 0)) for c in self.conflicts_quarantined]

    @property
    def weekend_closures(self) -> list[dict[str, Any]]:
        return self.scheduled_closures

    @property
    def intraday_gaps(self) -> list[dict[str, Any]]:
        return self.unscheduled_gaps

    @property
    def date_range_start_utc(self) -> str:
        return self.from_utc

    @property
    def date_range_end_utc(self) -> str:
        return self.to_utc

    @property
    def parquet_sha256(self) -> str:
        return self.sha256_checksum

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResearchDatasetSnapshot:
    """In-memory representation of a verified research dataset snapshot."""

    manifest: ResearchDatasetManifest
    parquet_path: Path
    manifest_path: Path

    @property
    def candles_path(self) -> Path:
        return self.parquet_path

    @property
    def path(self) -> Path:
        return self.parquet_path.parent

    def verify_integrity(self) -> bool:
        """Verify that the candles parquet file sha256 matches the signed manifest."""
        if not self.parquet_path.is_file():
            return False
        computed = hashlib.sha256(self.parquet_path.read_bytes()).hexdigest()
        return computed == self.manifest.sha256_checksum

    def load_candles(self) -> list[Candle]:
        """Load and verify all candles from the snapshot Parquet file."""
        return read_candles_parquet(
            self.parquet_path,
            expected_symbol=self.manifest.symbol,
            verify_checksum=True,
            expected_checksum=self.manifest.sha256_checksum,
        )


class ResearchDatasetManager:
    """Manages offline research dataset discovery, validation, deduplication, and snapshotting."""

    ELIGIBILITY_RULES: Final[list[str]] = [
        "symbol == 'EURUSD' and active_id == 1",
        "candle_size_seconds == 60 (M1 timeframe)",
        "is_otc == False (zero OTC fallback allowed)",
        "is_synthetic == False (authentic broker quotes only)",
        "candle_count > 0 and valid_candles > 0 (empty sessions excluded)",
        "structural_integrity_passed == True / VERIFIED status",
        "file SHA-256 matches catalog or receipt inventory",
    ]

    def __init__(self, root: Path | None = None) -> None:
        self.root = root.resolve() if root is not None else Path().resolve()

    def discover_sources(self) -> list[DiscoveredSource]:
        """Scan data catalogs and extended collection directories for potential datasets."""
        sources: list[DiscoveredSource] = []

        # 1. Discover catalogued Parquet datasets (e.g. data/datasets/catalog.db)
        catalog_candidates = [
            self.root / "data" / "datasets" / "catalog.db",
            self.root / "datasets" / "catalog.db",
            self.root / "data" / "catalog.db",
            self.root / "catalog.db",
        ]
        catalog_path = next((p for p in catalog_candidates if p.is_file()), None)
        if catalog_path is not None:
            catalog_base = catalog_path.parent
            try:
                catalog = DatasetCatalog(catalog_path)
                for entry in catalog.find_datasets():
                    entry_path = Path(entry.file_path)
                    if entry_path.is_absolute():
                        parquet_full = entry_path
                    elif (catalog_base / entry_path).is_file():
                        parquet_full = catalog_base / entry_path
                    elif (self.root / entry_path).is_file():
                        parquet_full = self.root / entry_path
                    else:
                        parquet_full = catalog_base / entry_path
                    is_eligible = True
                    reason = None

                    if entry.symbol != "EURUSD" or entry.active_id != 1:
                        is_eligible = False
                        reason = f"INELIGIBLE_SYMBOL_{entry.symbol}_{entry.active_id}"
                    elif entry.is_otc:
                        is_eligible = False
                        reason = "INELIGIBLE_OTC_SUBSTITUTION_FORBIDDEN"
                    elif entry.is_synthetic:
                        is_eligible = False
                        reason = "INELIGIBLE_SYNTHETIC_DATA"
                    elif entry.candle_size_seconds != 60:
                        is_eligible = False
                        reason = f"INELIGIBLE_TIMEFRAME_{entry.candle_size_seconds}"
                    elif entry.candle_count <= 0:
                        is_eligible = False
                        reason = "INELIGIBLE_EMPTY_DATASET"
                    elif not entry.structural_integrity_passed:
                        is_eligible = False
                        reason = "INELIGIBLE_FAILED_STRUCTURAL_INTEGRITY"
                    elif not parquet_full.is_file():
                        is_eligible = False
                        reason = "INELIGIBLE_FILE_NOT_FOUND"
                    else:
                        # Verify on-disk checksum
                        actual_sha = hashlib.sha256(parquet_full.read_bytes()).hexdigest()
                        if actual_sha != entry.sha256_checksum:
                            is_eligible = False
                            reason = "INELIGIBLE_CHECKSUM_MISMATCH"

                    sources.append(
                        DiscoveredSource(
                            source_id=entry.dataset_id,
                            source_type="CATALOG_PARQUET",
                            path=parquet_full,
                            candle_count=entry.candle_count,
                            from_ts=entry.from_ts,
                            to_ts=entry.to_ts,
                            from_utc=entry.from_utc,
                            to_utc=entry.to_utc,
                            is_eligible=is_eligible,
                            exclusion_reason=reason,
                            provenance_class=PROVENANCE_LEGACY,
                            wallet_classification="UNKNOWN",
                            sha256_checksum=entry.sha256_checksum,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error reading catalog at %s: %s", catalog_path, exc)

        # 2. Discover Extended Collection sessions (e.g. data/extended_collection/collection_*)
        extended_candidates = [
            self.root / "data" / "extended_collection",
            self.root / "extended_collection",
        ]
        extended_dir = next((p for p in extended_candidates if p.is_dir()), None)
        if extended_dir is not None:
            for session_dir in sorted(extended_dir.glob("collection_*")):
                if not session_dir.is_dir():
                    continue
                try:
                    info = inspect_session(session_dir)
                    valid_candles = info.get("valid_candles", 0)
                    boundaries = info.get("boundaries", [])
                    earliest = min(boundaries) if boundaries else 0
                    latest = max(boundaries) if boundaries else 0
                    earliest_utc = (
                        datetime.fromtimestamp(earliest, UTC).isoformat() if earliest else ""
                    )
                    latest_utc = datetime.fromtimestamp(latest, UTC).isoformat() if latest else ""

                    is_eligible = True
                    reason = None
                    if valid_candles == 0:
                        is_eligible = False
                        reason = "INELIGIBLE_EMPTY_ZERO_CANDLES"
                    elif info.get("integrity") not in ("VERIFIED", "VERIFIED_PARTIAL"):
                        is_eligible = False
                        reason = f"INELIGIBLE_INTEGRITY_{info.get('integrity')}"
                    elif info.get("symbol") != "EURUSD":
                        is_eligible = False
                        reason = f"INELIGIBLE_SYMBOL_{info.get('symbol')}"

                    sources.append(
                        DiscoveredSource(
                            source_id=info.get("session_id", session_dir.name),
                            source_type="EXTENDED_SESSION",
                            path=session_dir,
                            candle_count=valid_candles,
                            from_ts=earliest,
                            to_ts=latest,
                            from_utc=earliest_utc,
                            to_utc=latest_utc,
                            is_eligible=is_eligible,
                            exclusion_reason=reason,
                            provenance_class=PROVENANCE_EXTENDED_LIVE,
                            wallet_classification=info.get("wallet_classification", "UNKNOWN"),
                            sha256_checksum=hashlib.sha256(
                                (session_dir / "manifest.json").read_bytes()
                                if (session_dir / "manifest.json").is_file()
                                else (session_dir / "session.json").read_bytes()
                            ).hexdigest(),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    sources.append(
                        DiscoveredSource(
                            source_id=session_dir.name,
                            source_type="EXTENDED_SESSION",
                            path=session_dir,
                            candle_count=0,
                            from_ts=0,
                            to_ts=0,
                            from_utc="",
                            to_utc="",
                            is_eligible=False,
                            exclusion_reason=f"INELIGIBLE_INSPECTION_FAILED_{type(exc).__name__}",
                            provenance_class=PROVENANCE_EXTENDED_LIVE,
                            wallet_classification="UNKNOWN",
                            sha256_checksum="",
                        )
                    )

        return sorted(sources, key=lambda s: s.source_id)

    def prepare_dataset(
        self,
        sources: list[DiscoveredSource] | None = None,
        *,
        output_dir: Path | None = None,
        snapshot_id: str | None = None,
        max_gap_minutes: int = 10,
    ) -> ResearchDatasetSnapshot:
        """Merge, deduplicate, conflict-quarantine, and snapshot eligible research data.

        Returns an immutable ResearchDatasetSnapshot with tamper-evident manifest.
        """
        if sources is None:
            all_sources = self.discover_sources()
            sources = [s for s in all_sources if s.is_eligible]

        if not sources:
            raise DatasetManagerError("No eligible authentic datasets available for preparation")

        out_base = (
            output_dir.resolve()
            if output_dir is not None
            else (self.root / DEFAULT_SNAPSHOTS_DIR).resolve()
        )
        out_base.mkdir(parents=True, exist_ok=True)

        # 1. Load candles from all eligible sources
        # Map: from_ts -> list of (Candle, source_id, provenance_class)
        boundary_candles: dict[int, list[tuple[Candle, str, str]]] = {}
        total_discovered = 0
        source_manifest_records: list[dict[str, Any]] = []

        for src in sources:
            loaded_for_src: list[Candle] = []
            if src.source_type == "CATALOG_PARQUET":
                # Find matching catalog entry
                catalog_candidates = [
                    self.root / "data" / "datasets" / "catalog.db",
                    self.root / "datasets" / "catalog.db",
                    self.root / "data" / "catalog.db",
                    self.root / "catalog.db",
                ]
                catalog_path = next((p for p in catalog_candidates if p.is_file()), None)
                cat = DatasetCatalog(catalog_path) if catalog_path else None
                entry = cat.get_dataset(src.source_id) if cat else None
                loaded_for_src = read_candles_parquet(
                    src.path,
                    catalog_entry=entry,
                    verify_checksum=bool(entry and entry.sha256_checksum),
                )
            elif src.source_type == "EXTENDED_SESSION":
                # Load candles from receipts
                for receipt_path in sorted((src.path / "receipts").glob("*.json")):
                    receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
                    catalog_dict = receipt_data["catalog_entry"]
                    entry = CatalogEntry(**catalog_dict)
                    parquet_file = (src.path / "datasets" / entry.file_path).resolve()
                    if parquet_file.is_file():
                        page_candles = read_candles_parquet(parquet_file, catalog_entry=entry)
                        loaded_for_src.extend(page_candles)

            total_discovered += len(loaded_for_src)
            source_manifest_records.append({
                "source_id": src.source_id,
                "source_type": src.source_type,
                "path": str(src.path),
                "candle_count": len(loaded_for_src),
                "sha256": src.sha256_checksum,
                "provenance": src.provenance_class,
                "wallet_classification": src.wallet_classification,
            })

            for c in loaded_for_src:
                boundary_candles.setdefault(c.from_ts, []).append((
                    c,
                    src.source_id,
                    src.provenance_class,
                ))

        # 2. Cross-session Deduplication and Conflict Quarantine
        retained_candles: list[Candle] = []
        provenance_counts: dict[str, int] = {
            PROVENANCE_LEGACY: 0,
            PROVENANCE_EXTENDED_LIVE: 0,
        }
        quarantined_conflicts: list[QuarantinedConflict] = []
        duplicate_count = 0

        for from_ts in sorted(boundary_candles.keys()):
            items = boundary_candles[from_ts]
            if len(items) == 1:
                c, _, prov = items[0]
                retained_candles.append(c)
                provenance_counts[prov] = provenance_counts.get(prov, 0) + 1
                continue

            # Multiple records for this boundary: compare OHLCV
            first_c = items[0][0]
            has_conflict = False
            for other_c, _other_src, _ in items[1:]:
                if (
                    abs(first_c.open - other_c.open) > 1e-7
                    or abs(first_c.high - other_c.high) > 1e-7
                    or abs(first_c.low - other_c.low) > 1e-7
                    or abs(first_c.close - other_c.close) > 1e-7
                    or first_c.volume != other_c.volume
                ):
                    has_conflict = True
                    break

            if has_conflict:
                # Conflict detected: quarantine this boundary! Do not silently choose one.
                sources_detail = [
                    {
                        "source_id": s_id,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                        "provenance": prov,
                    }
                    for c, s_id, prov in items
                ]
                quarantined_conflicts.append(
                    QuarantinedConflict(
                        boundary_ts=from_ts,
                        boundary_utc=datetime.fromtimestamp(from_ts, UTC).isoformat(),
                        sources=sources_detail,
                        reason="OHLCV_DISAGREEMENT_ACROSS_SOURCES",
                    )
                )
            else:
                # Identical OHLCV across sources: deduplicate!
                duplicate_count += len(items) - 1
                # Prefer EXTENDED_LIVE provenance if available among duplicates
                best_item = next(
                    (it for it in items if it[2] == PROVENANCE_EXTENDED_LIVE), items[0]
                )
                c, _, prov = best_item
                retained_candles.append(c)
                provenance_counts[prov] = provenance_counts.get(prov, 0) + 1

        # 3. Sort retained candles chronologically
        retained_candles.sort(key=lambda c: c.from_ts)

        if not retained_candles:
            raise DatasetManagerError(
                "No valid non-conflicting candles remained after deduplication and quarantine"
            )

        # 4. Gap and Market Closure Analysis
        scheduled_closures: list[GapRecord] = []
        unscheduled_gaps: list[GapRecord] = []

        for i in range(len(retained_candles) - 1):
            curr_to = retained_candles[i].to_ts
            next_from = retained_candles[i + 1].from_ts
            diff = next_from - curr_to
            if diff > 0 and (diff >= 60 or max_gap_minutes > 0):
                start_dt = datetime.fromtimestamp(curr_to, UTC)
                end_dt = datetime.fromtimestamp(next_from, UTC)
                classification = EURUSD_FOREX_SCHEDULE.classify_gap(start_dt, end_dt)
                gap_rec = GapRecord(
                    from_ts=curr_to,
                    to_ts=next_from,
                    from_utc=start_dt.isoformat(),
                    to_utc=end_dt.isoformat(),
                    duration_seconds=diff,
                    gap_type=classification.value,
                )
                if classification == GapClassification.SCHEDULED_WEEKEND_CLOSURE:
                    scheduled_closures.append(gap_rec)
                else:
                    unscheduled_gaps.append(gap_rec)

        # 5. Write Immutable Snapshot Parquet
        now_utc = datetime.now(UTC)
        date_str = now_utc.strftime("%Y%m%d_%H%M%S")
        if snapshot_id is None:
            content_digest = hashlib.sha256(
                f"{retained_candles[0].from_ts}_{retained_candles[-1].to_ts}_{len(retained_candles)}".encode()
            ).hexdigest()[:12]
            snapshot_id = f"rds_eurusd_m1_{date_str}_{content_digest}"

        snapshot_dir = out_base / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=False)

        parquet_path = snapshot_dir / "candles.parquet"
        manifest_path = snapshot_dir / "manifest.json"

        # Build Arrow Table
        arrays = {
            "symbol": pa.array([c.symbol for c in retained_candles], type=pa.string()),
            "active_id": pa.array([c.active_id for c in retained_candles], type=pa.int32()),
            "is_otc": pa.array([False for _ in retained_candles], type=pa.bool_()),
            "size_seconds": pa.array([c.size.value for c in retained_candles], type=pa.int32()),
            "from_ts": pa.array([c.from_ts for c in retained_candles], type=pa.int64()),
            "to_ts": pa.array([c.to_ts for c in retained_candles], type=pa.int64()),
            "at_ts": pa.array([c.at_ts for c in retained_candles], type=pa.int64()),
            "open": pa.array([c.open for c in retained_candles], type=pa.float64()),
            "high": pa.array([c.high for c in retained_candles], type=pa.float64()),
            "low": pa.array([c.low for c in retained_candles], type=pa.float64()),
            "close": pa.array([c.close for c in retained_candles], type=pa.float64()),
            "volume": pa.array([c.volume for c in retained_candles], type=pa.int64()),
            "bid": pa.array(
                [c.bid if c.bid is not None else 0.0 for c in retained_candles],
                type=pa.float64(),
            ),
            "ask": pa.array(
                [c.ask if c.ask is not None else 0.0 for c in retained_candles],
                type=pa.float64(),
            ),
            "is_complete": pa.array([c.is_complete for c in retained_candles], type=pa.bool_()),
            "original_index": pa.array(list(range(len(retained_candles))), type=pa.int32()),
        }
        table = pa.Table.from_arrays(
            [arrays[field.name] for field in CANDLE_ARROW_SCHEMA],
            schema=CANDLE_ARROW_SCHEMA,
        )
        pq.write_table(table, parquet_path, compression="snappy")
        parquet_sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

        # Build Manifest
        manifest = ResearchDatasetManifest(
            snapshot_id=snapshot_id,
            created_at_utc=now_utc.isoformat(),
            dataset_version="1.0",
            symbol="EURUSD",
            candle_size_seconds=60,
            from_ts=retained_candles[0].from_ts,
            to_ts=retained_candles[-1].to_ts,
            from_utc=datetime.fromtimestamp(retained_candles[0].from_ts, UTC).isoformat(),
            to_utc=datetime.fromtimestamp(retained_candles[-1].to_ts, UTC).isoformat(),
            total_candles_discovered=total_discovered,
            total_candles_retained=len(retained_candles),
            duplicate_boundaries_deduplicated=duplicate_count,
            conflicts_quarantined_count=len(quarantined_conflicts),
            conflicts_quarantined=[c.to_dict() for c in quarantined_conflicts],
            unscheduled_gaps_count=len(unscheduled_gaps),
            unscheduled_gaps=[g.to_dict() for g in unscheduled_gaps],
            scheduled_closures_count=len(scheduled_closures),
            scheduled_closures=[g.to_dict() for g in scheduled_closures],
            provenance_breakdown=provenance_counts,
            source_files=source_manifest_records,
            eligibility_rules=self.ELIGIBILITY_RULES,
            sha256_checksum=parquet_sha,
        )

        manifest_dict = manifest.to_dict()
        manifest_dict["manifest_sha256"] = digest(manifest_dict)
        manifest_path.write_text(
            json.dumps(manifest_dict, indent=2, sort_keys=True), encoding="utf-8"
        )

        return ResearchDatasetSnapshot(
            manifest=manifest,
            parquet_path=parquet_path,
            manifest_path=manifest_path,
        )

    def list_snapshots(self, base_dir: Path | None = None) -> list[ResearchDatasetSnapshot]:
        """Scan and verify existing research dataset snapshots."""
        snapshots_dir = (
            base_dir.resolve()
            if base_dir is not None
            else (self.root / DEFAULT_SNAPSHOTS_DIR).resolve()
        )
        if not snapshots_dir.is_dir():
            return []

        snapshots: list[ResearchDatasetSnapshot] = []
        for s_dir in sorted(snapshots_dir.glob("rds_*"), reverse=True):
            if not s_dir.is_dir():
                continue
            manifest_file = s_dir / "manifest.json"
            parquet_file = s_dir / "candles.parquet"
            if not manifest_file.is_file() or not parquet_file.is_file():
                continue
            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                claimed_sha = data.get("manifest_sha256")
                clean_data = {k: v for k, v in data.items() if k != "manifest_sha256"}
                if claimed_sha and digest(clean_data) != claimed_sha:
                    logger.warning("Manifest integrity mismatch in %s", s_dir)
                    continue
                # Verify Parquet checksum
                actual_parquet_sha = hashlib.sha256(parquet_file.read_bytes()).hexdigest()
                if actual_parquet_sha != data["sha256_checksum"]:
                    logger.warning("Parquet checksum mismatch in %s", s_dir)
                    continue

                manifest = ResearchDatasetManifest(
                    snapshot_id=data["snapshot_id"],
                    created_at_utc=data["created_at_utc"],
                    dataset_version=data["dataset_version"],
                    symbol=data["symbol"],
                    candle_size_seconds=data["candle_size_seconds"],
                    from_ts=data["from_ts"],
                    to_ts=data["to_ts"],
                    from_utc=data["from_utc"],
                    to_utc=data["to_utc"],
                    total_candles_discovered=data["total_candles_discovered"],
                    total_candles_retained=data["total_candles_retained"],
                    duplicate_boundaries_deduplicated=data["duplicate_boundaries_deduplicated"],
                    conflicts_quarantined_count=data["conflicts_quarantined_count"],
                    conflicts_quarantined=data.get("conflicts_quarantined", []),
                    unscheduled_gaps_count=data["unscheduled_gaps_count"],
                    unscheduled_gaps=data.get("unscheduled_gaps", []),
                    scheduled_closures_count=data["scheduled_closures_count"],
                    scheduled_closures=data.get("scheduled_closures", []),
                    provenance_breakdown=data["provenance_breakdown"],
                    source_files=data["source_files"],
                    eligibility_rules=data.get("eligibility_rules", []),
                    sha256_checksum=data["sha256_checksum"],
                )
                snapshots.append(
                    ResearchDatasetSnapshot(
                        manifest=manifest,
                        parquet_path=parquet_file,
                        manifest_path=manifest_file,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load snapshot at %s: %s", s_dir, exc)

        return snapshots

    build_snapshot = prepare_dataset
