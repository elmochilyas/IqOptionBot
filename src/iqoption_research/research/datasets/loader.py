"""Verified historical dataset loader.

Loads and aggregates partitioned Parquet datasets indexed by the SQLite catalog
with rigorous cryptographic integrity verification, OTC isolation, synthetic
separation, gap detection, and research sufficiency evaluation.

Completely offline: performs no broker connections or network operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.instruments import Instrument, InstrumentRegistry
from iqoption_research.market_data.parquet_storage import (
    DataIntegrityError,
    read_candles_parquet,
)
from iqoption_research.market_data.sessions import get_market_schedule
from iqoption_research.market_data.sufficiency import (
    DEFAULT_RESEARCH_POLICY,
    ResearchSufficiencyPolicy,
    SufficiencyEvaluation,
    SufficiencyStatus,
    assess_research_sufficiency,
)
from iqoption_research.market_data.validation import (
    CandleGap,
    DataQualityReport,
    assess_data_quality,
)
from iqoption_research.models.market import Candle


class DatasetLoadingError(ValueError):
    """Raised when dataset loading encounters integrity or consistency violations."""


class MixedInstrumentError(DatasetLoadingError):
    """Raised when regular Forex and OTC instruments are combined."""


class MixedSyntheticError(DatasetLoadingError):
    """Raised when synthetic fixtures and authentic datasets are combined without policy."""


class InsufficientDataError(DatasetLoadingError):
    """Raised when dataset does not satisfy requested research sufficiency requirements."""


@dataclass(frozen=True, slots=True)
class LoadedHistoricalDataset:
    """An immutable, verified historical dataset ready for causal research."""

    symbol: str
    instrument: Instrument
    active_id: int
    is_otc: bool
    is_synthetic: bool
    source: str
    candle_size_seconds: int
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    candles: tuple[Candle, ...]
    gaps: tuple[CandleGap, ...]
    catalog_entries: tuple[CatalogEntry, ...]
    quality_report: DataQualityReport
    sufficiency_evaluation: SufficiencyEvaluation | None
    provenance: str = "unknown"

    @property
    def candle_count(self) -> int:
        return len(self.candles)

    @property
    def provenance_property(self) -> str:
        """Return the dataset provenance for inspection/tracking."""
        return self.provenance


class HistoricalDatasetLoader:
    """Loads and validates historical candle datasets from SQLite catalog and Parquet files."""

    def __init__(self, catalog: DatasetCatalog, storage_root: Path | str) -> None:
        self.catalog = catalog
        self.storage_root = Path(storage_root).resolve()

    def load_dataset(
        self,
        *,
        symbol: str,
        candle_size_seconds: int,
        source: str | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
        is_otc: bool | None = None,
        synthetic_policy: str = "reject_mixed",
        sufficiency_policy: ResearchSufficiencyPolicy | None = None,
        enforce_sufficiency: bool = False,
        verify_checksums: bool = True,
    ) -> LoadedHistoricalDataset:
        """Query catalog and load matching candle files with end-to-end verification.

        Args:
            symbol: Target symbol (e.g. 'EURUSD' or 'EURUSD_OTC').
            candle_size_seconds: Exact timeframe in seconds (e.g. 60).
            source: Data provider source identity (e.g. 'iqoption').
            from_ts: Optional earliest start timestamp (inclusive).
            to_ts: Optional latest end timestamp (inclusive).
            is_otc: Optional explicit OTC override; if None, inferred from symbol.
            synthetic_policy: Policy for handling synthetic vs authentic datasets.
            sufficiency_policy: Optional policy to assess research sufficiency.
            enforce_sufficiency: If True, raise InsufficientDataError when policy fails.
            verify_checksums: If True, check SHA-256 against catalog and sidecar hashes.

        Raises:
            DatasetLoadingError: On file absence, gaps, overlaps or corrupt data.
            MixedInstrumentError: If regular and OTC datasets are matched.
            MixedSyntheticError: If authentic and synthetic data are blended.
            InsufficientDataError: If enforce_sufficiency=True and sufficiency fails.
        """
        inst = InstrumentRegistry.get(symbol)
        norm_symbol = inst.symbol
        target_otc = inst.is_otc if is_otc is None else is_otc

        if target_otc != inst.is_otc:
            raise MixedInstrumentError(
                f"Requested is_otc={target_otc} contradicts instrument {inst.symbol} "
                f"(is_otc={inst.is_otc})"
            )

        # 1. Find matching entries in SQLite catalog
        matched_entries = self.catalog.find_datasets(
            symbol=norm_symbol,
            candle_size_seconds=candle_size_seconds,
            from_ts=from_ts,
            to_ts=to_ts,
            is_otc=target_otc,
            source=source,
        )

        if not matched_entries:
            raise DatasetLoadingError(
                f"No catalog entries found for symbol={norm_symbol}, size={candle_size_seconds}s, "
                f"is_otc={target_otc}, source={source} in range [{from_ts}, {to_ts}]"
            )

        # 2. Check regular vs OTC mixing
        for entry in matched_entries:
            if entry.is_otc != target_otc:
                raise MixedInstrumentError(
                    f"Catalog entry {entry.dataset_id} is_otc={entry.is_otc} does not match "
                    f"target is_otc={target_otc}"
                )
            if entry.symbol != norm_symbol:
                raise MixedInstrumentError(
                    f"Catalog entry {entry.dataset_id} symbol {entry.symbol} does not match "
                    f"target symbol {norm_symbol}"
                )

        # 3. Check synthetic vs authentic policy
        synth_flags = {entry.is_synthetic for entry in matched_entries}
        if len(synth_flags) > 1:
            raise MixedSyntheticError(
                f"Matched catalog entries contain a mixture of synthetic and authentic datasets: "
                f"{[e.dataset_id for e in matched_entries]}"
            )

        is_dataset_synthetic = next(iter(synth_flags))
        if synthetic_policy == "synthetic_only" and not is_dataset_synthetic:
            raise MixedSyntheticError("synthetic_policy='synthetic_only' but authentic data found")
        if synthetic_policy == "authentic_only" and is_dataset_synthetic:
            raise MixedSyntheticError("synthetic_policy='authentic_only' but synthetic data found")

        # 4. Load candles from Parquet files with catalog hash cross-check
        all_candles: list[Candle] = []

        for entry in matched_entries:
            p_file = self.storage_root / entry.file_path
            if not p_file.is_file():
                # Try fallback relative to storage_root if file_path is already absolute
                p_file = Path(entry.file_path).resolve()
                if not p_file.is_file():
                    raise DatasetLoadingError(
                        f"Parquet file referenced in catalog entry {entry.dataset_id} does not "
                        f"exist: {entry.file_path}"
                    )

            try:
                candles = read_candles_parquet(
                    p_file,
                    expected_symbol=norm_symbol,
                    verify_checksum=verify_checksums,
                    catalog_entry=entry,
                )
            except DataIntegrityError as exc:
                raise DatasetLoadingError(
                    f"Integrity check failed for dataset {entry.dataset_id}: {exc}"
                ) from exc

            # Filter within requested range
            for c in candles:
                if from_ts is not None and c.from_ts < from_ts:
                    continue
                if to_ts is not None and c.to_ts > to_ts:
                    continue
                all_candles.append(c)

        if not all_candles:
            raise DatasetLoadingError("No candles remained after applying timestamp filter")

        # 5. Sort chronologically and verify consistency
        all_candles.sort(key=lambda c: (c.from_ts, c.to_ts))

        # Check for duplicates, overlaps, and interval consistency
        seen_from_timestamps: set[int] = set()
        for idx, c in enumerate(all_candles):
            # Check duplicate timestamps
            if c.from_ts in seen_from_timestamps:
                raise DatasetLoadingError(
                    f"Duplicate candle timestamp {c.from_ts} detected at index {idx}"
                )
            seen_from_timestamps.add(c.from_ts)

            # Check interval duration
            actual_duration = c.to_ts - c.from_ts
            if actual_duration != candle_size_seconds:
                raise DatasetLoadingError(
                    f"Candle at index {idx} has duration {actual_duration}s, expected "
                    f"{candle_size_seconds}s"
                )

            # Check overlaps with preceding candle
            if idx > 0:
                prev = all_candles[idx - 1]
                if c.from_ts < prev.to_ts:
                    raise DatasetLoadingError(
                        f"Overlapping candles between index {idx - 1} "
                        f"(to_ts={prev.to_ts}) and index {idx} (from_ts={c.from_ts})"
                    )

        # 6. Quality and gap assessment (surface unresolved gaps without filling)
        schedule = get_market_schedule(norm_symbol)
        quality = assess_data_quality(
            all_candles,
            expected_size=candle_size_seconds,
            is_historical=True,
            market_schedule=schedule,
            expected_instrument=inst,
        )

        # 7. Sufficiency assessment
        active_policy = sufficiency_policy or DEFAULT_RESEARCH_POLICY
        sufficiency = assess_research_sufficiency(quality, active_policy)

        if enforce_sufficiency and sufficiency.status == SufficiencyStatus.INSUFFICIENT:
            raise InsufficientDataError(
                f"Dataset failed research sufficiency under policy '{active_policy.name}': "
                f"{'; '.join(sufficiency.reasons)}"
            )

        earliest_from = all_candles[0].from_ts
        latest_to = all_candles[-1].to_ts

        return LoadedHistoricalDataset(
            symbol=norm_symbol,
            instrument=inst,
            active_id=inst.active_id,
            is_otc=target_otc,
            is_synthetic=is_dataset_synthetic,
            source=source or matched_entries[0].source,
            candle_size_seconds=candle_size_seconds,
            from_ts=earliest_from,
            to_ts=latest_to,
            from_utc=datetime.fromtimestamp(earliest_from, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(latest_to, tz=UTC).isoformat(),
            candles=tuple(all_candles),
            gaps=tuple(quality.gaps),
            catalog_entries=tuple(matched_entries),
            quality_report=quality,
            provenance=("synthetic" if is_dataset_synthetic else "authentic"),
            sufficiency_evaluation=sufficiency,
        )
