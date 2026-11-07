"""SQLite dataset catalog for research candle datasets.

Maintains ACID metadata records, file locations, cryptographic SHA-256 checksums,
structural integrity statuses, and research sufficiency assessments.
Uses parameterized queries, explicit constraints, and deterministic indexing.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.market_data.instruments import (
    BROKER_QUOTE_DISCLAIMER,
    MAPPING_PROVENANCE_REF,
    REGISTRY_VERSION,
)
from iqoption_research.market_data.sufficiency import SufficiencyStatus

CURRENT_CATALOG_SCHEMA_VERSION: Final[str] = "2.0"

SCHEMA_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS dataset_catalog (
    dataset_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    source TEXT NOT NULL,
    is_synthetic INTEGER NOT NULL DEFAULT 0,
    symbol TEXT NOT NULL,
    active_id INTEGER NOT NULL,
    is_otc INTEGER NOT NULL,
    candle_size_seconds INTEGER NOT NULL,
    from_ts INTEGER NOT NULL,
    to_ts INTEGER NOT NULL,
    from_utc TEXT NOT NULL,
    to_utc TEXT NOT NULL,
    candle_count INTEGER NOT NULL,
    created_at_utc TEXT NOT NULL,
    imported_at_utc TEXT NOT NULL,
    quality_summary TEXT NOT NULL,
    structural_integrity_passed INTEGER NOT NULL,
    coverage_completeness_ratio REAL NOT NULL,
    research_sufficiency_status TEXT NOT NULL,
    research_sufficiency_policy TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    sha256_checksum TEXT NOT NULL,
    metadata_path TEXT NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    provenance_version TEXT NOT NULL,
    mapping_provenance TEXT NOT NULL,
    disclaimer TEXT NOT NULL,
    CHECK (candle_count >= 0),
    CHECK (from_ts <= to_ts),
    CHECK (candle_size_seconds > 0)
);

CREATE INDEX IF NOT EXISTS idx_dataset_lookup
ON dataset_catalog(symbol, is_otc, candle_size_seconds, from_ts, to_ts);

CREATE INDEX IF NOT EXISTS idx_dataset_checksum
ON dataset_catalog(sha256_checksum);

CREATE INDEX IF NOT EXISTS idx_dataset_source
ON dataset_catalog(source, created_at_utc);
"""


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """A strongly typed dataset catalog record."""

    dataset_id: str
    schema_version: str
    source: str
    is_synthetic: bool
    symbol: str
    active_id: int
    is_otc: bool
    candle_size_seconds: int
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    candle_count: int
    created_at_utc: str
    imported_at_utc: str
    quality_summary: dict[str, Any]
    structural_integrity_passed: bool
    coverage_completeness_ratio: float
    research_sufficiency_status: str
    research_sufficiency_policy: str
    file_path: str
    sha256_checksum: str
    metadata_path: str
    metadata_sha256: str
    provenance_version: str = REGISTRY_VERSION
    mapping_provenance: str = MAPPING_PROVENANCE_REF
    disclaimer: str = BROKER_QUOTE_DISCLAIMER

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> CatalogEntry:
        """Construct a CatalogEntry from an SQLite Row object."""
        quality_dict = json.loads(row["quality_summary"])
        return cls(
            dataset_id=row["dataset_id"],
            schema_version=row["schema_version"],
            source=row["source"],
            is_synthetic=bool(row["is_synthetic"]),
            symbol=row["symbol"],
            active_id=row["active_id"],
            is_otc=bool(row["is_otc"]),
            candle_size_seconds=row["candle_size_seconds"],
            from_ts=row["from_ts"],
            to_ts=row["to_ts"],
            from_utc=row["from_utc"],
            to_utc=row["to_utc"],
            candle_count=row["candle_count"],
            created_at_utc=row["created_at_utc"],
            imported_at_utc=row["imported_at_utc"],
            quality_summary=quality_dict,
            structural_integrity_passed=bool(row["structural_integrity_passed"]),
            coverage_completeness_ratio=float(row["coverage_completeness_ratio"]),
            research_sufficiency_status=row["research_sufficiency_status"],
            research_sufficiency_policy=row["research_sufficiency_policy"],
            file_path=row["file_path"],
            sha256_checksum=row["sha256_checksum"],
            metadata_path=row["metadata_path"],
            metadata_sha256=row["metadata_sha256"],
            provenance_version=row["provenance_version"],
            mapping_provenance=row["mapping_provenance"],
            disclaimer=row["disclaimer"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to sanitized JSON-serializable dictionary."""
        d = asdict(self)
        d["is_synthetic"] = self.is_synthetic
        d["is_otc"] = self.is_otc
        d["structural_integrity_passed"] = self.structural_integrity_passed
        return d


class DatasetCatalog:
    """Thread-safe SQLite dataset catalog for historical candle datasets."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_DDL)

    def register_dataset(
        self,
        entry: CatalogEntry,
        *,
        on_duplicate: str = "raise",
    ) -> None:
        """Register a new completed dataset in the catalog.

        Args:
            entry: Verified catalog entry.
            on_duplicate: "raise", "ignore", or "replace".

        Raises:
            FileExistsError: If dataset_id or file_path already exists and on_duplicate="raise".
        """
        sql: str
        if on_duplicate == "ignore":
            sql = "INSERT OR IGNORE INTO dataset_catalog VALUES ("
        elif on_duplicate == "replace":
            sql = "INSERT OR REPLACE INTO dataset_catalog VALUES ("
        else:
            sql = "INSERT INTO dataset_catalog VALUES ("

        placeholders = ", ".join(["?"] * 27)
        sql = f"{sql}{placeholders});"

        quality_json = json.dumps(entry.quality_summary, sort_keys=True)
        params = (
            entry.dataset_id,
            entry.schema_version,
            entry.source,
            1 if entry.is_synthetic else 0,
            entry.symbol,
            entry.active_id,
            1 if entry.is_otc else 0,
            entry.candle_size_seconds,
            entry.from_ts,
            entry.to_ts,
            entry.from_utc,
            entry.to_utc,
            entry.candle_count,
            entry.created_at_utc,
            entry.imported_at_utc or datetime.now(UTC).isoformat(),
            quality_json,
            1 if entry.structural_integrity_passed else 0,
            entry.coverage_completeness_ratio,
            entry.research_sufficiency_status,
            entry.research_sufficiency_policy,
            entry.file_path,
            entry.sha256_checksum,
            entry.metadata_path,
            entry.metadata_sha256,
            entry.provenance_version,
            entry.mapping_provenance,
            entry.disclaimer,
        )

        try:
            with self._get_connection() as conn:
                conn.execute(sql, params)
        except sqlite3.IntegrityError as exc:
            raise FileExistsError(
                f"Dataset with ID {entry.dataset_id!r} or file path {entry.file_path!r} "
                "already exists in catalog."
            ) from exc

    def get_dataset(self, dataset_id: str) -> CatalogEntry | None:
        """Retrieve a dataset entry by its unique dataset ID."""
        sql = "SELECT * FROM dataset_catalog WHERE dataset_id = ?;"
        with self._get_connection() as conn:
            cursor = conn.execute(sql, (dataset_id,))
            row = cursor.fetchone()
            if row is not None:
                return CatalogEntry.from_row(row)
        return None

    def get_by_checksum(self, sha256_checksum: str) -> CatalogEntry | None:
        """Retrieve a dataset entry by its Parquet SHA-256 checksum."""
        sql = "SELECT * FROM dataset_catalog WHERE sha256_checksum = ?;"
        with self._get_connection() as conn:
            cursor = conn.execute(sql, (sha256_checksum,))
            row = cursor.fetchone()
            if row is not None:
                return CatalogEntry.from_row(row)
        return None

    def find_datasets(
        self,
        *,
        symbol: str | None = None,
        candle_size_seconds: int | None = None,
        is_otc: bool | None = None,
        source: str | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
        min_sufficiency: SufficiencyStatus | str | None = None,
    ) -> list[CatalogEntry]:
        """Search catalog entries with deterministic criteria filtering."""
        clauses = []
        params: list[Any] = []

        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol.strip().upper())

        if candle_size_seconds is not None:
            clauses.append("candle_size_seconds = ?")
            params.append(candle_size_seconds)

        if is_otc is not None:
            clauses.append("is_otc = ?")
            params.append(1 if is_otc else 0)

        if source is not None:
            clauses.append("source = ?")
            params.append(source)

        if from_ts is not None:
            clauses.append("from_ts >= ?")
            params.append(from_ts)

        if to_ts is not None:
            clauses.append("to_ts <= ?")
            params.append(to_ts)

        if min_sufficiency is not None:
            suff_val = (
                min_sufficiency.value
                if isinstance(min_sufficiency, SufficiencyStatus)
                else str(min_sufficiency).lower()
            )
            if suff_val == SufficiencyStatus.SUFFICIENT.value:
                clauses.append("research_sufficiency_status = 'sufficient'")
            elif suff_val == SufficiencyStatus.BORDERLINE.value:
                clauses.append("research_sufficiency_status IN ('sufficient', 'borderline')")

        where_stmt = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT * FROM dataset_catalog {where_stmt} "  # noqa: S608
            "ORDER BY from_ts ASC, created_at_utc ASC;"
        )

        with self._get_connection() as conn:
            cursor = conn.execute(sql, tuple(params))
            return [CatalogEntry.from_row(row) for row in cursor.fetchall()]

    def unregister_dataset(self, dataset_id: str) -> bool:
        """Remove a dataset entry from the catalog."""
        sql = "DELETE FROM dataset_catalog WHERE dataset_id = ?;"
        with self._get_connection() as conn:
            cursor = conn.execute(sql, (dataset_id,))
            return cursor.rowcount > 0

    def count_datasets(self) -> int:
        """Return total number of registered datasets in the catalog."""
        sql = "SELECT COUNT(*) FROM dataset_catalog;"
        with self._get_connection() as conn:
            cursor = conn.execute(sql)
            row = cursor.fetchone()
            return int(row[0]) if row else 0
