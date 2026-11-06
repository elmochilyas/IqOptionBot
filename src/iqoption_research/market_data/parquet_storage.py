"""Partitioned Parquet candle storage with cryptographic provenance and crash safety.

Implements a two-tier persistence workflow:
1. Input validation and canonical chronological ordering.
2. Staging write to isolated temporary files with SHA-256 hash calculation.
3. Atomic publication via atomic filesystem replacement.
4. Synchronized SQLite catalog registration.
5. Strict partition separation between regular EUR/USD and EUR/USD OTC.
6. Path traversal protection and staging file cleanup.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.instruments import (
    BROKER_QUOTE_DISCLAIMER,
    MAPPING_PROVENANCE_REF,
    REGISTRY_VERSION,
    InstrumentRegistry,
)
from iqoption_research.market_data.sessions import get_market_schedule
from iqoption_research.market_data.sufficiency import (
    DEFAULT_RESEARCH_POLICY,
    ResearchSufficiencyPolicy,
    assess_research_sufficiency,
)
from iqoption_research.market_data.validation import (
    DataQualityReport,
    assess_data_quality,
    validate_candle_ohlc_integrity,
)
from iqoption_research.models.market import Candle, CandleSize

if TYPE_CHECKING:
    pass

PARQUET_STORAGE_SCHEMA_VERSION: Final[str] = "2.0"

# Explicit Arrow Schema with fixed types
CANDLE_ARROW_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        ("symbol", pa.string()),
        ("active_id", pa.int32()),
        ("is_otc", pa.bool_()),
        ("size_seconds", pa.int32()),
        ("from_ts", pa.int64()),
        ("to_ts", pa.int64()),
        ("at_ts", pa.int64()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("is_complete", pa.bool_()),
        ("original_index", pa.int32()),
    ]
)


class StorageError(Exception):
    """Base exception for market data storage operations."""


class DataIntegrityError(StorageError):
    """Raised when file checksum or internal record validation fails."""


class ImmutableDatasetError(StorageError):
    """Raised when attempting an unauthorized overwrite of an existing dataset."""


class UnsafePathError(StorageError):
    """Raised when a directory traversal or unsafe path is detected."""


@dataclass(frozen=True, slots=True)
class ParquetWriteResult:
    """Outcome of writing a partitioned Parquet dataset."""

    dataset_id: str
    parquet_path: Path
    metadata_path: Path
    sha256_checksum: str
    metadata_sha256: str
    candle_count: int
    quality_report: DataQualityReport
    catalog_entry: CatalogEntry | None


def _compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sanitize_path_component(component: str) -> str:
    """Ensure path component contains only safe alphanumeric characters."""
    sanitized = re.sub(r"[^\w\-.]", "_", component.strip().lower())
    if not sanitized or ".." in sanitized or sanitized.startswith((".", "/")):
        raise UnsafePathError(f"Unsafe path component rejected: {component!r}")
    return sanitized


def _assert_subpath(parent: Path, child: Path) -> None:
    """Ensure child path resolves strictly within the parent directory."""
    resolved_parent = parent.resolve()
    resolved_child = child.resolve()
    try:
        resolved_child.relative_to(resolved_parent)
    except ValueError as exc:
        raise UnsafePathError(
            f"Path traversal detected: {resolved_child} is outside {resolved_parent}"
        ) from exc


def build_partition_dir(
    root_dir: Path,
    *,
    source: str,
    symbol: str,
    is_otc: bool,
    size_seconds: int,
    from_ts: int,
) -> Path:
    """Construct deterministic partition directory path.

    Structure: {root}/{source}/{symbol_partition}/{timeframe}/year={YYYY}/month={MM}/
    """
    safe_source = _sanitize_path_component(source)
    clean_sym = symbol.lower()
    if is_otc and not clean_sym.endswith("_otc"):
        clean_sym = f"{clean_sym}_otc"
    safe_symbol = _sanitize_path_component(clean_sym)
    safe_timeframe = f"{size_seconds}s"

    dt = datetime.fromtimestamp(from_ts, tz=UTC)
    year_part = f"year={dt.year}"
    month_part = f"month={dt.month:02d}"

    target_dir = root_dir / safe_source / safe_symbol / safe_timeframe / year_part / month_part
    _assert_subpath(root_dir, target_dir)
    return target_dir


def write_candles_parquet(
    candles: list[Candle],
    root_dir: Path | str,
    *,
    source: str = "iqoption",
    is_synthetic: bool = False,
    catalog: DatasetCatalog | None = None,
    sufficiency_policy: ResearchSufficiencyPolicy | None = None,
    on_duplicate: str = "raise",
) -> ParquetWriteResult:
    """Persist a list of normalized candles into partitioned Parquet storage.

    Executes a crash-safe, two-tier persistence workflow with checksums,
    sidecar metadata, and transactional catalog registration.

    Raises:
        ValueError: If candle series is empty or malformed.
        ImmutableDatasetError: If dataset already exists and on_duplicate='raise'.
        StorageError: On filesystem or database failures.
    """
    if not candles:
        raise ValueError("Cannot write an empty candle dataset")

    base_dir = Path(root_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate instrument consistency
    inst = InstrumentRegistry.validate_series_consistency(candles)

    # 2. Check OHLC validity for each record
    for idx, c in enumerate(candles):
        ohlc_errs = validate_candle_ohlc_integrity(c)
        if ohlc_errs:
            raise InvalidCandleError(
                f"Record at index {idx} has invalid OHLC: {'; '.join(ohlc_errs)}"
            )

    # 3. Canonical sort while recording original input index
    indexed_candles = list(enumerate(candles))
    indexed_candles.sort(key=lambda item: (item[1].from_ts, item[1].to_ts))

    earliest_from = indexed_candles[0][1].from_ts
    latest_to = indexed_candles[-1][1].to_ts
    size_seconds = inst.supported_sizes[0].value if not candles else candles[0].size.value

    # Assess quality
    sorted_candles = [c for _, c in indexed_candles]
    schedule = get_market_schedule(inst.symbol)
    quality = assess_data_quality(
        sorted_candles,
        expected_size=size_seconds,
        is_historical=True,
        market_schedule=schedule,
        expected_instrument=inst,
    )

    # Assess sufficiency
    active_policy = sufficiency_policy or DEFAULT_RESEARCH_POLICY
    sufficiency = assess_research_sufficiency(quality, active_policy)

    # 4. Build PyArrow Table
    table_dict: dict[str, list[Any]] = {
        "symbol": [c.symbol for _, c in indexed_candles],
        "active_id": [c.active_id for _, c in indexed_candles],
        "is_otc": [inst.is_otc for _ in indexed_candles],
        "size_seconds": [c.size.value for _, c in indexed_candles],
        "from_ts": [c.from_ts for _, c in indexed_candles],
        "to_ts": [c.to_ts for _, c in indexed_candles],
        "at_ts": [c.at_ts for _, c in indexed_candles],
        "open": [c.open for _, c in indexed_candles],
        "high": [c.high for _, c in indexed_candles],
        "low": [c.low for _, c in indexed_candles],
        "close": [c.close for _, c in indexed_candles],
        "volume": [c.volume for _, c in indexed_candles],
        "bid": [c.bid for _, c in indexed_candles],
        "ask": [c.ask for _, c in indexed_candles],
        "is_complete": [c.is_complete for _, c in indexed_candles],
        "original_index": [idx for idx, _ in indexed_candles],
    }
    table = pa.Table.from_pydict(table_dict, schema=CANDLE_ARROW_SCHEMA)

    # Deterministic content hash identifier
    content_fingerprint = (
        f"{inst.symbol}_{inst.active_id}_{size_seconds}_{earliest_from}_{latest_to}_{len(candles)}"
    )
    content_hash = hashlib.sha256(content_fingerprint.encode("utf-8")).hexdigest()[:12]
    dataset_id = (
        f"ds_{inst.symbol.lower()}_{size_seconds}s_{earliest_from}_{latest_to}_{content_hash}"
    )

    # Staging paths
    staging_dir = base_dir / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    _assert_subpath(base_dir, staging_dir)

    tmp_parquet = staging_dir / f"{dataset_id}.tmp.parquet"
    tmp_meta = staging_dir / f"{dataset_id}.tmp.meta.json"

    # Destination paths
    target_dir = build_partition_dir(
        base_dir,
        source=source,
        symbol=inst.symbol,
        is_otc=inst.is_otc,
        size_seconds=size_seconds,
        from_ts=earliest_from,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    filename_stem = f"candles_{earliest_from}_{latest_to}_{content_hash}"
    target_parquet = target_dir / f"{filename_stem}.parquet"
    target_meta = target_dir / f"{filename_stem}.meta.json"

    # Check existing dataset
    if target_parquet.exists() and on_duplicate == "raise":
        raise ImmutableDatasetError(
            f"Dataset file {target_parquet} already exists and on_duplicate='raise'"
        )

    # 5. Write to temporary staging
    try:
        pq.write_table(table, tmp_parquet, compression="snappy")
        parquet_checksum = _compute_file_sha256(tmp_parquet)

        metadata_dict = {
            "dataset_id": dataset_id,
            "schema_version": PARQUET_STORAGE_SCHEMA_VERSION,
            "source": source,
            "is_synthetic": is_synthetic,
            "instrument": {
                "symbol": inst.symbol,
                "active_id": inst.active_id,
                "is_otc": inst.is_otc,
                "pip_size": inst.pip_size,
                "source_venue": inst.source_venue,
                "mapping_provenance": inst.mapping_provenance,
                "disclaimer": inst.disclaimer,
            },
            "timeframe": {
                "candle_size_seconds": size_seconds,
                "earliest_from_ts": earliest_from,
                "latest_to_ts": latest_to,
                "earliest_from_utc": datetime.fromtimestamp(earliest_from, tz=UTC).isoformat(),
                "latest_to_utc": datetime.fromtimestamp(latest_to, tz=UTC).isoformat(),
            },
            "candle_count": len(candles),
            "sha256_checksum": parquet_checksum,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "quality_audit": quality.to_dict(),
            "research_sufficiency": sufficiency.to_dict(),
        }

        meta_bytes = json.dumps(metadata_dict, indent=2, sort_keys=True).encode("utf-8")
        with tmp_meta.open("wb") as f:
            f.write(meta_bytes)
        meta_checksum = hashlib.sha256(meta_bytes).hexdigest()

        # 6. Atomic publication via os.replace
        Path(tmp_parquet).replace(target_parquet)
        Path(tmp_meta).replace(target_meta)

    except Exception as exc:
        # Cleanup temporary files on failure
        if tmp_parquet.exists():
            tmp_parquet.unlink(missing_ok=True)
        if tmp_meta.exists():
            tmp_meta.unlink(missing_ok=True)
        raise StorageError(f"Failed during dataset writing: {exc}") from exc

    # 7. Catalog registration
    catalog_entry: CatalogEntry | None = None
    if catalog is not None:
        rel_parquet = str(target_parquet.relative_to(base_dir))
        rel_meta = str(target_meta.relative_to(base_dir))

        catalog_entry = CatalogEntry(
            dataset_id=dataset_id,
            schema_version=PARQUET_STORAGE_SCHEMA_VERSION,
            source=source,
            is_synthetic=is_synthetic,
            symbol=inst.symbol,
            active_id=inst.active_id,
            is_otc=inst.is_otc,
            candle_size_seconds=size_seconds,
            from_ts=earliest_from,
            to_ts=latest_to,
            from_utc=datetime.fromtimestamp(earliest_from, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(latest_to, tz=UTC).isoformat(),
            candle_count=len(candles),
            created_at_utc=str(metadata_dict["created_at_utc"]),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary=quality.to_dict(),
            structural_integrity_passed=(quality.acceptance_status == "accepted"),
            coverage_completeness_ratio=quality.completeness_ratio,
            research_sufficiency_status=sufficiency.status.value,
            research_sufficiency_policy=active_policy.name,
            file_path=rel_parquet,
            sha256_checksum=parquet_checksum,
            metadata_path=rel_meta,
            metadata_sha256=meta_checksum,
            provenance_version=REGISTRY_VERSION,
            mapping_provenance=MAPPING_PROVENANCE_REF,
            disclaimer=BROKER_QUOTE_DISCLAIMER,
        )

        try:
            catalog.register_dataset(catalog_entry, on_duplicate=on_duplicate)
        except Exception as exc:
            # If catalog registration fails and on_duplicate='raise', remove published files
            if on_duplicate == "raise":
                target_parquet.unlink(missing_ok=True)
                target_meta.unlink(missing_ok=True)
            raise StorageError(f"Failed to register dataset in catalog: {exc}") from exc

    return ParquetWriteResult(
        dataset_id=dataset_id,
        parquet_path=target_parquet,
        metadata_path=target_meta,
        sha256_checksum=parquet_checksum,
        metadata_sha256=meta_checksum,
        candle_count=len(candles),
        quality_report=quality,
        catalog_entry=catalog_entry,
    )


def read_candles_parquet(
    parquet_path: Path | str,
    *,
    expected_symbol: str | None = None,
    verify_checksum: bool = True,
) -> list[Candle]:
    """Read normalized Candle objects from a Parquet file.

    Optionally verifies cryptographic SHA-256 sidecar checksums.

    Raises:
        FileNotFoundError: If Parquet file does not exist.
        DataIntegrityError: If checksum verification fails or instrument is mismatched.
    """
    path = Path(parquet_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    # Checksum verification against sidecar
    if verify_checksum:
        meta_path = path.with_suffix(".meta.json")
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                expected_sha = meta.get("sha256_checksum")
                if expected_sha:
                    actual_sha = _compute_file_sha256(path)
                    if actual_sha != expected_sha:
                        raise DataIntegrityError(
                            f"Checksum mismatch for {path}: "
                            f"expected {expected_sha}, got {actual_sha}"
                        )
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(f"Corrupt metadata sidecar: {meta_path}") from exc

    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise DataIntegrityError(f"Failed to read Parquet data from {path}: {exc}") from exc

    candles: list[Candle] = []
    pydict = table.to_pydict()
    num_rows = table.num_rows

    target_symbol = (
        InstrumentRegistry.normalize_symbol(expected_symbol) if expected_symbol else None
    )

    for i in range(num_rows):
        row_symbol = pydict["symbol"][i]
        if target_symbol and row_symbol != target_symbol:
            raise DataIntegrityError(
                f"Row {i} symbol {row_symbol} does not match expected {target_symbol}"
            )

        c = Candle(
            active_id=pydict["active_id"][i],
            symbol=row_symbol,
            size=CandleSize(pydict["size_seconds"][i]),
            from_ts=pydict["from_ts"][i],
            to_ts=pydict["to_ts"][i],
            at_ts=pydict["at_ts"][i],
            open=pydict["open"][i],
            high=pydict["high"][i],
            low=pydict["low"][i],
            close=pydict["close"][i],
            volume=pydict["volume"][i],
            bid=pydict["bid"][i],
            ask=pydict["ask"][i],
            is_complete=pydict["is_complete"][i],
            provenance="historical",
        )
        candles.append(c)

    return candles


def read_filtered_candles(
    base_dir: Path | str,
    *,
    symbol: str,
    size_seconds: int,
    is_otc: bool = False,
    from_ts: int | None = None,
    to_ts: int | None = None,
    source: str = "iqoption",
    verify_checksum: bool = True,
) -> list[Candle]:
    """Read and aggregate candles matching partition and timestamp filters."""
    root = Path(base_dir).resolve()
    safe_source = _sanitize_path_component(source)
    clean_sym = symbol.lower()
    if is_otc and not clean_sym.endswith("_otc"):
        clean_sym = f"{clean_sym}_otc"
    safe_symbol = _sanitize_path_component(clean_sym)
    safe_tf = f"{size_seconds}s"

    partition_root = root / safe_source / safe_symbol / safe_tf
    if not partition_root.is_dir():
        return []

    collected: list[Candle] = []
    parquet_files = sorted(partition_root.rglob("*.parquet"))

    for p_file in parquet_files:
        if p_file.name.endswith(".tmp.parquet"):
            continue
        candles = read_candles_parquet(
            p_file, expected_symbol=symbol, verify_checksum=verify_checksum
        )
        for c in candles:
            if from_ts is not None and c.from_ts < from_ts:
                continue
            if to_ts is not None and c.to_ts > to_ts:
                continue
            collected.append(c)

    collected.sort(key=lambda c: (c.from_ts, c.to_ts))
    return collected


def cleanup_staging_files(root_dir: Path | str, max_age_seconds: float = 3600.0) -> int:
    """Scan and clean up orphaned temporary files in .staging older than max_age_seconds."""
    root = Path(root_dir).resolve()
    staging_dir = root / ".staging"
    if not staging_dir.is_dir():
        return 0

    removed = 0
    now = datetime.now(UTC).timestamp()

    for item in staging_dir.glob("*.tmp.*"):
        if item.is_file():
            try:
                mtime = item.stat().st_mtime
                if (now - mtime) > max_age_seconds:
                    item.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue

    return removed
