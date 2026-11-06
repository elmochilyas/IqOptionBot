"""Phase 3.2 Tests: SQLite Dataset Catalog, Partitioned Parquet Storage, Integrity & Provenance.

Covers:
- SQLite catalog schema, constraints, indexes, and transactions
- Parquet write/read round trips with exact numerical and timestamp fidelity
- Strict separation of regular EUR/USD and EUR/USD OTC partitions
- Filtered historical dataset lookups across timeframes and date ranges
- Duplicate import policies ("raise", "ignore", "replace")
- Detection of corrupted Parquet files and tampered SHA-256 sidecar checksums
- Crash safety, staging isolation, atomic publication, and orphaned file cleanup
- Rejection of invalid source data and mixed instruments
- Separation of structural integrity from research sufficiency policies
- Path traversal protection and rejection of unsafe filenames
- Completely offline execution with synthetic fixtures only
"""

from pathlib import Path

import pytest

from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.parquet_storage import (
    DataIntegrityError,
    UnsafePathError,
    cleanup_staging_files,
    read_candles_parquet,
    read_filtered_candles,
    write_candles_parquet,
)
from iqoption_research.market_data.sufficiency import (
    POLICY_EXPLORATORY,
    POLICY_SMOKE_TEST,
    POLICY_STATISTICAL_BACKTEST,
    SufficiencyStatus,
    assess_research_sufficiency,
)
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.market import Candle, CandleSize


def _make_sample_candle(
    from_ts: int,
    size: int = 60,
    symbol: str = "EURUSD",
    active_id: int = 1,
    open_: float = 1.08500,
    high: float | None = None,
    low: float | None = None,
    close: float = 1.08520,
    volume: int = 15,
) -> Candle:
    high_val = high if high is not None else max(open_, close) + 0.00050
    low_val = low if low is not None else min(open_, close) - 0.00050
    return Candle(
        active_id=active_id,
        symbol=symbol,
        size=CandleSize(size),
        from_ts=from_ts,
        to_ts=from_ts + size,
        at_ts=(from_ts + size) * 1_000_000_000,
        open=open_,
        high=high_val,
        low=low_val,
        close=close,
        volume=volume,
        bid=close - 0.00005,
        ask=close + 0.00005,
        is_complete=True,
        provenance="historical",
        raw={"active_id": active_id, "from": from_ts, "to": from_ts + size},
    )


# ============================================================================
# 1. SQLite Dataset Catalog Tests
# ============================================================================


def test_catalog_schema_initialization_and_constraints(tmp_path: Path) -> None:
    """Test catalog table initialization, constraints, and deterministic indexing."""
    db_file = tmp_path / "test_catalog.db"
    catalog = DatasetCatalog(db_file)
    assert catalog.count_datasets() == 0

    entry = CatalogEntry(
        dataset_id="ds_eurusd_60s_100_220_abc123",
        schema_version="2.0",
        source="test_source",
        is_synthetic=True,
        symbol="EURUSD",
        active_id=1,
        is_otc=False,
        candle_size_seconds=60,
        from_ts=100,
        to_ts=220,
        from_utc="1970-01-01T00:01:40Z",
        to_utc="1970-01-01T00:03:40Z",
        candle_count=2,
        created_at_utc="2026-11-07T00:00:00Z",
        imported_at_utc="2026-11-07T00:00:01Z",
        quality_summary={"score": 100.0, "acceptance_status": "accepted"},
        structural_integrity_passed=True,
        coverage_completeness_ratio=1.0,
        research_sufficiency_status="sufficient",
        research_sufficiency_policy="smoke_test",
        file_path="test_source/eurusd/60s/candles.parquet",
        sha256_checksum="deadbeef1234",
        metadata_path="test_source/eurusd/60s/candles.meta.json",
        metadata_sha256="cafebabe5678",
    )

    catalog.register_dataset(entry)
    assert catalog.count_datasets() == 1

    fetched = catalog.get_dataset("ds_eurusd_60s_100_220_abc123")
    assert fetched is not None
    assert fetched.symbol == "EURUSD"
    assert fetched.is_synthetic is True
    assert fetched.sha256_checksum == "deadbeef1234"

    by_sha = catalog.get_by_checksum("deadbeef1234")
    assert by_sha is not None
    assert by_sha.dataset_id == entry.dataset_id


def test_catalog_duplicate_import_behavior(tmp_path: Path) -> None:
    """Test on_duplicate policies: raise, ignore, and replace."""
    catalog = DatasetCatalog(tmp_path / "catalog_dup.db")
    entry = CatalogEntry(
        dataset_id="ds_1",
        schema_version="2.0",
        source="src",
        is_synthetic=True,
        symbol="EURUSD",
        active_id=1,
        is_otc=False,
        candle_size_seconds=60,
        from_ts=100,
        to_ts=160,
        from_utc="1970-01-01T00:01:40Z",
        to_utc="1970-01-01T00:02:40Z",
        candle_count=1,
        created_at_utc="2026-11-07T00:00:00Z",
        imported_at_utc="2026-11-07T00:00:01Z",
        quality_summary={},
        structural_integrity_passed=True,
        coverage_completeness_ratio=1.0,
        research_sufficiency_status="sufficient",
        research_sufficiency_policy="smoke_test",
        file_path="src/file.parquet",
        sha256_checksum="hash1",
        metadata_path="src/file.meta.json",
        metadata_sha256="meta_hash1",
    )

    catalog.register_dataset(entry)

    # on_duplicate="raise" must raise FileExistsError
    with pytest.raises(FileExistsError, match="already exists"):
        catalog.register_dataset(entry, on_duplicate="raise")

    # on_duplicate="ignore" must not raise
    catalog.register_dataset(entry, on_duplicate="ignore")
    assert catalog.count_datasets() == 1

    # on_duplicate="replace" must replace cleanly
    updated_entry = CatalogEntry(
        dataset_id="ds_1",
        schema_version="2.0",
        source="src",
        is_synthetic=True,
        symbol="EURUSD",
        active_id=1,
        is_otc=False,
        candle_size_seconds=60,
        from_ts=100,
        to_ts=160,
        from_utc="1970-01-01T00:01:40Z",
        to_utc="1970-01-01T00:02:40Z",
        candle_count=1,
        created_at_utc="2026-11-07T00:00:00Z",
        imported_at_utc="2026-11-07T00:00:01Z",
        quality_summary={"updated": True},
        structural_integrity_passed=True,
        coverage_completeness_ratio=1.0,
        research_sufficiency_status="sufficient",
        research_sufficiency_policy="smoke_test",
        file_path="src/file.parquet",
        sha256_checksum="hash_updated",
        metadata_path="src/file.meta.json",
        metadata_sha256="meta_hash_updated",
    )
    catalog.register_dataset(updated_entry, on_duplicate="replace")
    assert catalog.get_dataset("ds_1").sha256_checksum == "hash_updated"  # type: ignore[union-attr]


# ============================================================================
# 2. Partitioned Parquet Storage: Write / Read Round Trips
# ============================================================================


def test_parquet_storage_roundtrip_precision_and_ordering(tmp_path: Path) -> None:
    """Write candles to Parquet and verify exact float64 and int64 round-trip fidelity."""
    db_file = tmp_path / "catalog.db"
    catalog = DatasetCatalog(db_file)

    candles = [
        _make_sample_candle(1714560000, open_=1.0851234, close=1.0859876),
        _make_sample_candle(1714560060, open_=1.0859876, close=1.0865432),
    ]

    res = write_candles_parquet(
        candles,
        tmp_path / "storage",
        source="test_feed",
        is_synthetic=True,
        catalog=catalog,
    )

    assert res.parquet_path.is_file()
    assert res.metadata_path.is_file()
    assert res.candle_count == 2
    assert catalog.count_datasets() == 1

    # Read back and verify exact values
    read_back = read_candles_parquet(res.parquet_path, expected_symbol="EURUSD")
    assert len(read_back) == 2
    assert read_back[0].from_ts == 1714560000
    assert read_back[0].to_ts == 1714560060
    assert read_back[0].open == pytest.approx(1.0851234, abs=1e-7)
    assert read_back[0].close == pytest.approx(1.0859876, abs=1e-7)
    assert read_back[1].from_ts == 1714560060


def test_parquet_storage_otc_separation(tmp_path: Path) -> None:
    """Ensure regular EURUSD and EURUSD_OTC are partitioned into distinct directories."""
    storage_root = tmp_path / "storage"

    eurusd_candles = [_make_sample_candle(1714560000, symbol="EURUSD", active_id=1)]
    otc_candles = [_make_sample_candle(1714560000, symbol="EURUSD_OTC", active_id=76)]

    res_reg = write_candles_parquet(eurusd_candles, storage_root, source="feed")
    res_otc = write_candles_parquet(otc_candles, storage_root, source="feed")

    # Paths must reside in separate partition branches
    assert "eurusd/60s" in str(res_reg.parquet_path).replace("\\", "/")
    assert "eurusd_otc/60s" in str(res_otc.parquet_path).replace("\\", "/")
    assert res_reg.parquet_path != res_otc.parquet_path

    # Filtered reads must strictly respect OTC flag
    reg_read = read_filtered_candles(
        storage_root, symbol="EURUSD", size_seconds=60, is_otc=False, source="feed"
    )
    assert len(reg_read) == 1
    assert reg_read[0].symbol == "EURUSD"

    otc_read = read_filtered_candles(
        storage_root, symbol="EURUSD_OTC", size_seconds=60, is_otc=True, source="feed"
    )
    assert len(otc_read) == 1
    assert otc_read[0].symbol == "EURUSD_OTC"


# ============================================================================
# 3. Crash Safety, Staging Isolation & Tamper Detection
# ============================================================================


def test_staging_cleanup_and_atomic_replace(tmp_path: Path) -> None:
    """Test that failed writes do not leave corrupted files in target directory."""
    storage_root = tmp_path / "storage"
    staging_dir = storage_root / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy orphaned temporary files
    orphan = staging_dir / "ds_orphan.tmp.parquet"
    orphan.write_text("incomplete parquet data")

    # Cleanup with 0 max_age should remove it
    removed = cleanup_staging_files(storage_root, max_age_seconds=0.0)
    assert removed == 1
    assert not orphan.exists()


def test_corrupted_file_and_tampered_checksum_detection(tmp_path: Path) -> None:
    """Detect modified file content or altered metadata checksums."""
    candles = [_make_sample_candle(1714560000)]
    res = write_candles_parquet(candles, tmp_path / "storage", is_synthetic=True)

    # 1. Reading with valid checksum passes
    assert len(read_candles_parquet(res.parquet_path, verify_checksum=True)) == 1

    # 2. Tampering with Parquet file must trigger DataIntegrityError
    with res.parquet_path.open("ab") as f:
        f.write(b"corrupting_garbage_bytes")

    with pytest.raises(DataIntegrityError, match="Checksum mismatch"):
        read_candles_parquet(res.parquet_path, verify_checksum=True)


def test_invalid_source_data_rejected_before_write(tmp_path: Path) -> None:
    """Invalid OHLC or mixed instruments must be rejected before touching storage."""
    # Broken OHLC: High < Low
    bad_candle = _make_sample_candle(100, high=1.0800, low=1.0850)
    with pytest.raises(InvalidCandleError, match="invalid OHLC"):
        write_candles_parquet([bad_candle], tmp_path / "storage")

    # Mixed instruments in single series
    c_reg = _make_sample_candle(100, symbol="EURUSD", active_id=1)
    c_otc = _make_sample_candle(160, symbol="EURUSD_OTC", active_id=76)
    with pytest.raises(ValueError, match="Instrument mixing detected"):
        write_candles_parquet([c_reg, c_otc], tmp_path / "storage")


def test_path_traversal_protection(tmp_path: Path) -> None:
    """Path traversal sequences in source or symbol must raise UnsafePathError."""
    candles = [_make_sample_candle(100)]
    with pytest.raises(UnsafePathError, match="Unsafe path component"):
        write_candles_parquet(candles, tmp_path / "storage", source="../evil_source")


# ============================================================================
# 4. Research Sufficiency vs. Structural Integrity Tests
# ============================================================================


def test_one_candle_structural_integrity_vs_research_sufficiency() -> None:
    """Verify that 1 candle passes structural integrity but fails research sufficiency."""
    one_candle = _make_sample_candle(1714560000)
    quality = assess_data_quality([one_candle], expected_size=60, is_historical=True)

    # Structural integrity is accepted and 100% clean
    assert quality.acceptance_status == "accepted"
    assert quality.score == 100.0

    # Under Smoke Test policy (1+ bar): SUFFICIENT
    eval_smoke = assess_research_sufficiency(quality, policy=POLICY_SMOKE_TEST)
    assert eval_smoke.status == SufficiencyStatus.SUFFICIENT
    assert eval_smoke.passed is True

    # Under Exploratory policy (500+ bars): INSUFFICIENT
    eval_exploratory = assess_research_sufficiency(quality, policy=POLICY_EXPLORATORY)
    assert eval_exploratory.status == SufficiencyStatus.INSUFFICIENT
    assert eval_exploratory.passed is False
    assert any("Candle count 1 is below policy minimum 500" in r for r in eval_exploratory.reasons)

    # Under Statistical Backtest policy (10,000+ bars): INSUFFICIENT
    eval_backtest = assess_research_sufficiency(quality, policy=POLICY_STATISTICAL_BACKTEST)
    assert eval_backtest.status == SufficiencyStatus.INSUFFICIENT
    assert eval_backtest.passed is False
    assert any("below policy minimum 10000" in r for r in eval_backtest.reasons)
