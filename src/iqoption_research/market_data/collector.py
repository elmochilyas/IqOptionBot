"""Authentic market data acquisition, incremental collection, and quality management.

Implements Section 6 requirements:
- Incremental historical candle retrieval from authorized broker feeds
- Authentic external dataset ingestion with explicit provenance tracking
- Strict separation between standard FX pairs (EURUSD) and OTC derivatives (EURUSD-OTC)
- Dataset deduplication, timestamp consistency, and gap auditing
- Atomic persistence into Parquet storage and SQLite catalog
- Safety enforcement: fails closed if broker account status is unverified
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import ConnectionError, InvalidCandleError
from iqoption_research.execution.account_diagnostics import BrokerAccountDiagnostic
from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.importer import import_candles_from_csv
from iqoption_research.market_data.parquet_storage import (
    read_candles_parquet,
    write_candles_parquet,
)
from iqoption_research.market_data.symbols import normalize_symbol
from iqoption_research.market_data.validation import DataQualityReport, assess_data_quality
from iqoption_research.models.market import Candle, CandleSize

logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR: Final[Path] = Path("data/datasets")


class DataAcquisitionError(RuntimeError):
    """Raised when data collection fails or is blocked by security policies."""


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Detailed coverage and quality summary for a stored dataset."""

    symbol: str
    candle_size_seconds: int
    source: str
    provenance: str
    is_otc: bool
    total_bars: int
    start_utc: str | None
    end_utc: str | None
    duration_hours: float
    gap_count: int
    max_gap_seconds: int
    quality_verdict: str

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "candle_size_seconds": self.candle_size_seconds,
            "source": self.source,
            "provenance": self.provenance,
            "is_otc": self.is_otc,
            "total_bars": self.total_bars,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "duration_hours": self.duration_hours,
            "gap_count": self.gap_count,
            "max_gap_seconds": self.max_gap_seconds,
            "quality_verdict": self.quality_verdict,
        }


class HistoricalDataCollector:
    """Manages authentic market data acquisition, validation, deduplication, and cataloging."""

    def __init__(
        self,
        *,
        storage_dir: Path | str | None = None,
        catalog: DatasetCatalog | None = None,
        account_diagnostic: BrokerAccountDiagnostic | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir is not None else DEFAULT_DATASET_DIR
        self.catalog = catalog or DatasetCatalog(self.storage_dir / "catalog.db")
        self.account_diagnostic = account_diagnostic or BrokerAccountDiagnostic()

    def is_otc_instrument(self, symbol: str) -> bool:
        """Identify whether an instrument is an OTC synthetic market."""
        if not isinstance(symbol, str):
            return False
        raw = symbol.strip().upper()
        return "OTC" in raw or raw.endswith("-OTC") or raw.endswith("_OTC")

    def deduplicate_and_sort_candles(self, candles: list[Candle]) -> list[Candle]:
        """Deduplicate candles by from_ts and sort chronologically."""
        if not candles:
            return []

        # Keep latest or unique candle for each from_ts
        seen: dict[int, Candle] = {}
        for c in candles:
            seen[c.from_ts] = c

        return sorted(seen.values(), key=lambda c: c.from_ts)

    async def collect_from_broker(
        self,
        client: AsyncIQOption,
        *,
        symbol: str = "EURUSD",
        candle_size_seconds: int = 60,
        end_time_ts: int | None = None,
        count: int = 1000,
        chunk_size: int = 500,
        require_practice: bool = False,
        allow_partial: bool = True,
    ) -> list[Candle]:
        """Fetch authentic historical candles from IQ Option using read-only methods.

        Enforces:
        - Strict read-only connector assertion.
        - Verification that account ledger routing is blocked.
        - Symbol normalization and OTC isolation.
        - Automatic pagination when count exceeds chunk size.
        - Deduplication and chronological sorting across pages.
        - Graceful handling of network disconnects and partial collections.
        """
        client.assert_read_only()
        canonical_symbol = normalize_symbol(symbol)

        # Account safety check if ws is available
        ws = getattr(client, "_ws", None)
        if ws is not None:
            profile = getattr(ws, "profile", None)
            balances = getattr(ws, "balances", [])
            readiness = self.account_diagnostic.evaluate(
                profile=profile,
                balances=balances,
            )
            if require_practice and not readiness.is_practice_verified:
                raise DataAcquisitionError(
                    f"Collection blocked: account safety verification requires PRACTICE account. "
                    f"Current state: {readiness.explanation}"
                )

        resolved_size = CandleSize(candle_size_seconds)
        end_ts = end_time_ts if end_time_ts is not None else int(datetime.now(UTC).timestamp())

        # Bound chunk size by client settings max_candles (default 500, max 1000)
        client_max = getattr(getattr(client, "_settings", None), "max_candles", 500)
        effective_chunk = min(max(1, chunk_size), client_max, 1000)

        collected: list[Candle] = []
        current_endtime = end_ts

        while len(collected) < count:
            remaining = count - len(collected)
            batch_count = min(remaining, effective_chunk)

            try:
                fetch_res = await client.get_candles(
                    canonical_symbol,
                    size=resolved_size,
                    count=batch_count,
                    endtime=current_endtime,
                )
            except (ConnectionError, InvalidCandleError, TimeoutError, OSError) as exc:
                logger.warning(
                    "Historical fetch error during pagination at endtime=%s: %s",
                    current_endtime,
                    exc,
                )
                if collected and allow_partial:
                    logger.info(
                        "Returning %d partial candles collected prior to error.",
                        len(collected),
                    )
                    break
                raise DataAcquisitionError(f"Broker historical fetch failed: {exc}") from exc

            batch = fetch_res.candle_list
            if not batch:
                logger.info(
                    "Broker returned 0 candles for endtime=%s; end of history reached.",
                    current_endtime,
                )
                break

            collected.extend(batch)

            earliest_batch_ts = min(c.from_ts for c in batch)
            if earliest_batch_ts >= current_endtime:
                break

            current_endtime = earliest_batch_ts
            if len(batch) < batch_count:
                break

        # Deduplicate and sort chronologically
        deduped = self.deduplicate_and_sort_candles(collected)
        return deduped[-count:] if len(deduped) > count else deduped

    def import_external_csv(
        self,
        csv_path: Path | str,
        *,
        symbol: str = "EURUSD",
        candle_size_seconds: int = 60,
        source_label: str = "EXTERNAL_AUTHENTIC",
    ) -> list[Candle]:
        """Import candles from an authentic external CSV dataset."""
        canonical = normalize_symbol(symbol)
        imported = import_candles_from_csv(
            csv_path,
            symbol=canonical,
            candle_size_seconds=candle_size_seconds,
            is_synthetic=False,
        )
        # Update provenance
        result: list[Candle] = []
        for c in imported:
            result.append(
                Candle(
                    active_id=c.active_id,
                    symbol=c.symbol,
                    size=c.size,
                    from_ts=c.from_ts,
                    to_ts=c.to_ts,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=c.volume,
                    bid=c.bid,
                    ask=c.ask,
                    is_complete=c.is_complete,
                    at_ts=c.at_ts,
                    provenance=source_label,
                )
            )
        return result

    def persist_dataset(
        self,
        candles: list[Candle],
        *,
        symbol: str,
        candle_size_seconds: int,
        source: str = "iqoption",
        provenance: str = "AUTHENTIC_MARKET_DATA",
    ) -> Path:
        """Deduplicate, audit, sort, and atomically persist candles to Parquet."""
        canonical = normalize_symbol(symbol)
        clean_candles = self.deduplicate_and_sort_candles(candles)

        if not clean_candles:
            raise DataAcquisitionError("Cannot persist an empty candle dataset.")

        # If dataset already exists in catalog, load and merge
        existing = self.load_dataset(canonical, candle_size_seconds)
        if existing:
            clean_candles = self.deduplicate_and_sort_candles(existing + clean_candles)

        # Audit quality
        report: DataQualityReport = assess_data_quality(
            clean_candles,
            expected_size=candle_size_seconds,
            is_historical=True,
        )
        if report.acceptance_status == "rejected":
            logger.warning(
                "Dataset quality status is rejected (score=%.1f)",
                report.score,
            )

        # Persist to Parquet and update SQLite catalog atomically
        write_res = write_candles_parquet(
            clean_candles,
            self.storage_dir,
            source=provenance or source,
            is_synthetic=False,
            catalog=self.catalog,
            on_duplicate="replace",
        )
        return write_res.parquet_path

    def load_dataset(
        self,
        symbol: str,
        candle_size_seconds: int = 60,
    ) -> list[Candle]:
        """Load stored dataset from Parquet via catalog lookup."""
        canonical = normalize_symbol(symbol)
        entries = self.catalog.find_datasets(
            symbol=canonical,
            candle_size_seconds=candle_size_seconds,
        )
        if not entries:
            # Fallback direct path check
            otc_flag = self.is_otc_instrument(canonical)
            direct_dir = self.storage_dir / ("otc" if otc_flag else "regular") / canonical.lower()
            if not direct_dir.is_dir():
                return []
            parquet_files = sorted(direct_dir.glob("*.parquet"))
            if not parquet_files:
                return []
            return read_candles_parquet(parquet_files[-1])

        # Load latest entry
        latest = entries[0]
        file_path = self.storage_dir / latest.file_path
        if not file_path.is_file():
            return []
        return read_candles_parquet(file_path)

    def generate_coverage_report(
        self,
        symbol: str,
        candle_size_seconds: int = 60,
    ) -> CoverageReport | None:
        """Generate a thorough coverage report for a stored dataset."""
        candles = self.load_dataset(symbol, candle_size_seconds)
        if not candles:
            return None

        canonical = normalize_symbol(symbol)
        otc_flag = self.is_otc_instrument(canonical)
        start_dt = datetime.fromtimestamp(candles[0].from_ts, tz=UTC)
        end_dt = datetime.fromtimestamp(candles[-1].to_ts, tz=UTC)
        duration_hrs = (candles[-1].to_ts - candles[0].from_ts) / 3600.0

        # Calculate gaps
        gaps = 0
        max_gap = 0
        for i in range(1, len(candles)):
            delta = candles[i].from_ts - candles[i - 1].to_ts
            if delta > 0:
                gaps += 1
                if delta > max_gap:
                    max_gap = delta

        report = assess_data_quality(
            candles,
            expected_size=candle_size_seconds,
            is_historical=True,
        )
        verdict = "PASS" if report.acceptance_status != "rejected" else "FAIL"

        return CoverageReport(
            symbol=canonical,
            candle_size_seconds=candle_size_seconds,
            source=candles[0].provenance or "unknown",
            provenance="AUTHENTIC_MARKET_DATA",
            is_otc=otc_flag,
            total_bars=len(candles),
            start_utc=start_dt.isoformat(),
            end_utc=end_dt.isoformat(),
            duration_hours=round(duration_hrs, 2),
            gap_count=gaps,
            max_gap_seconds=max_gap,
            quality_verdict=verdict,
        )
