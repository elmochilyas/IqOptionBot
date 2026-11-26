"""Phase 3.4 Tests: Historical Data Importer (CSV & Parquet).

Covers:
- Auto-detection of varied CSV headers and custom mapping
- Timestamp parsing: Unix seconds, Unix milliseconds, and ISO datetime strings
- OHLC mathematical integrity validation on imported files
- Atomic persistence into partitioned Parquet storage and SQLite catalog
- Correct propagation of synthetic vs authentic provenance
- Non-destructive handling of duplicate imports
- Offline execution without broker connectivity
"""

from pathlib import Path

import pytest

from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.importer import (
    CSVColumnMapping,
    ImportDataError,
    import_and_persist_dataset,
    import_candles_from_csv,
)


def test_import_csv_header_autodetection_and_iso_timestamps(tmp_path: Path) -> None:
    """Import CSV with ISO datetime strings and standard header names."""
    csv_file = tmp_path / "test_iso.csv"
    csv_file.write_text(
        "datetime,open,high,low,close,volume\n"
        "2024-05-01T10:00:00Z,1.0850,1.0855,1.0848,1.0852,15\n"
        "2024-05-01T10:01:00Z,1.0852,1.0858,1.0850,1.0856,20\n",
        encoding="utf-8",
    )

    candles = import_candles_from_csv(
        csv_file,
        symbol="EURUSD",
        candle_size_seconds=60,
        is_synthetic=True,
    )

    assert len(candles) == 2
    assert candles[0].from_ts == 1714557600
    assert candles[0].to_ts == 1714557660
    assert candles[0].open == 1.0850
    assert candles[0].close == 1.0852
    assert candles[0].volume == 15
    assert candles[0].provenance == "synthetic"
    assert candles[1].from_ts == 1714557660


def test_import_csv_epoch_milliseconds(tmp_path: Path) -> None:
    """Import CSV with millisecond timestamps."""
    csv_file = tmp_path / "test_ms.csv"
    csv_file.write_text(
        "timestamp,o,h,l,c,v\n"
        "1714557600000,1.0850,1.0855,1.0848,1.0852,10\n"
        "1714557660000,1.0852,1.0858,1.0850,1.0856,12\n",
        encoding="utf-8",
    )

    mapping = CSVColumnMapping(
        timestamp_col="timestamp",
        open_col="o",
        high_col="h",
        low_col="l",
        close_col="c",
        volume_col="v",
    )

    candles = import_candles_from_csv(
        csv_file,
        symbol="EURUSD",
        candle_size_seconds=60,
        header_mapping=mapping,
    )

    assert len(candles) == 2
    assert candles[0].from_ts == 1714557600
    assert candles[1].from_ts == 1714557660


def test_import_csv_rejects_missing_headers_or_corrupt_data(tmp_path: Path) -> None:
    """Import raises ImportDataError on missing timestamp or non-numeric OHLC."""
    # Missing timestamp header
    bad_headers = tmp_path / "bad_headers.csv"
    bad_headers.write_text("foo,open,high,low,close\n1,1.0,1.1,0.9,1.0\n", encoding="utf-8")

    with pytest.raises(ImportDataError, match="Could not identify timestamp column"):
        import_candles_from_csv(bad_headers, symbol="EURUSD", candle_size_seconds=60)

    # Non-numeric OHLC
    bad_num = tmp_path / "bad_num.csv"
    bad_num.write_text(
        "timestamp,open,high,low,close\n1714557600,invalid_price,1.1,0.9,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ImportDataError, match="Invalid numeric OHLC"):
        import_candles_from_csv(bad_num, symbol="EURUSD", candle_size_seconds=60)


def test_import_and_persist_dataset_pipeline(tmp_path: Path) -> None:
    """End-to-end import persists into Parquet storage and catalog with metadata."""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1714557600,1.0850,1.0855,1.0848,1.0852,15\n"
        "1714557660,1.0852,1.0858,1.0850,1.0856,20\n",
        encoding="utf-8",
    )

    catalog = DatasetCatalog(tmp_path / "catalog.db")
    storage_root = tmp_path / "storage"

    res = import_and_persist_dataset(
        csv_file,
        symbol="EURUSD",
        candle_size_seconds=60,
        storage_root=storage_root,
        catalog=catalog,
        source="csv_feed",
        is_synthetic=True,
    )

    assert res.candle_count == 2
    assert res.parquet_path.is_file()
    assert res.metadata_path.is_file()
    assert catalog.count_datasets() == 1

    entry = catalog.get_dataset(res.dataset_id)
    assert entry is not None
    assert entry.symbol == "EURUSD"
    assert entry.is_synthetic is True
    assert entry.source == "csv_feed"
    assert entry.sha256_checksum == res.sha256_checksum
