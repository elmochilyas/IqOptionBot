"""Unit and regression tests for Phase 39 ResearchDatasetManager."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.parquet_storage import CANDLE_ARROW_SCHEMA
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
)


def _create_dummy_parquet(
    path: Path,
    start_ts: int,
    n_candles: int,
    open_p: float = 1.0500,
    interval_sec: int = 60,
    volume: int = 100,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    from_ts = [start_ts + i * interval_sec for i in range(n_candles)]
    to_ts = [t + interval_sec for t in from_ts]
    at_ts = [t + interval_sec for t in from_ts]
    opens = [open_p + (i % 5) * 0.0001 for i in range(n_candles)]
    highs = [p + 0.0002 for p in opens]
    lows = [p - 0.0002 for p in opens]
    closes = [p + 0.0001 for p in opens]

    table = pa.Table.from_arrays(
        [
            pa.array(["EURUSD"] * n_candles, type=pa.string()),
            pa.array([1] * n_candles, type=pa.int32()),
            pa.array([False] * n_candles, type=pa.bool_()),
            pa.array([interval_sec] * n_candles, type=pa.int32()),
            pa.array(from_ts, type=pa.int64()),
            pa.array(to_ts, type=pa.int64()),
            pa.array(at_ts, type=pa.int64()),
            pa.array(opens, type=pa.float64()),
            pa.array(highs, type=pa.float64()),
            pa.array(lows, type=pa.float64()),
            pa.array(closes, type=pa.float64()),
            pa.array([volume] * n_candles, type=pa.int64()),
            pa.array(closes, type=pa.float64()),
            pa.array([c + 0.00005 for c in closes], type=pa.float64()),
            pa.array([True] * n_candles, type=pa.bool_()),
            pa.array(list(range(n_candles)), type=pa.int32()),
        ],
        schema=CANDLE_ARROW_SCHEMA,
    )
    pq.write_table(table, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_catalog_entry(
    dataset_id: str,
    path: Path,
    sha: str,
    candle_count: int,
    from_ts: int,
    to_ts: int,
    symbol: str = "EURUSD",
    is_otc: bool = False,
    is_synthetic: bool = False,
    candle_size_seconds: int = 60,
) -> CatalogEntry:
    return CatalogEntry(
        dataset_id=dataset_id,
        schema_version="1.0.0",
        source="test",
        is_synthetic=is_synthetic,
        symbol=symbol,
        active_id=1 if symbol == "EURUSD" else 99,
        is_otc=is_otc,
        candle_size_seconds=candle_size_seconds,
        from_ts=from_ts,
        to_ts=to_ts,
        from_utc=datetime.fromtimestamp(from_ts, tz=UTC).isoformat(),
        to_utc=datetime.fromtimestamp(to_ts, tz=UTC).isoformat(),
        candle_count=candle_count,
        created_at_utc=datetime.now(UTC).isoformat(),
        imported_at_utc=datetime.now(UTC).isoformat(),
        quality_summary={},
        structural_integrity_passed=True,
        coverage_completeness_ratio=1.0,
        research_sufficiency_status="SUFFICIENT",
        research_sufficiency_policy="PHASE24",
        file_path=str(path),
        sha256_checksum=sha,
        metadata_path="",
        metadata_sha256="",
    )


def test_discover_sources_catalog_and_extended(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "data" / "catalog"
    catalog_dir.mkdir(parents=True)
    parquet_path = catalog_dir / "eurusd_m1_test.parquet"
    sha = _create_dummy_parquet(parquet_path, start_ts=1700000000, n_candles=100)

    # Register in catalog
    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        _make_catalog_entry(
            dataset_id="test_cat_1",
            path=parquet_path,
            sha=sha,
            candle_count=100,
            from_ts=1700000000,
            to_ts=1700000000 + 99 * 60,
        )
    )

    # Create one valid extended session and one empty extended session
    session_valid = tmp_path / "data" / "extended_collection" / "collection_valid"
    session_valid.mkdir(parents=True)
    (session_valid / "manifest.json").write_text("{}", encoding="utf-8")

    session_empty = tmp_path / "data" / "extended_collection" / "collection_empty"
    session_empty.mkdir(parents=True)
    (session_empty / "manifest.json").write_text("{}", encoding="utf-8")

    def mock_inspect(session_dir: Path):
        if "valid" in session_dir.name:
            return {
                "session_id": "collection_valid",
                "symbol": "EURUSD",
                "valid_candles": 50,
                "boundaries": [1700010000, 1700013000],
                "integrity": "VERIFIED",
                "wallet_classification": "UNKNOWN",
            }
        return {
            "session_id": "collection_empty",
            "symbol": "EURUSD",
            "valid_candles": 0,
            "boundaries": [],
            "integrity": "VERIFIED",
            "wallet_classification": "CONFLICTING_ACCOUNT_INDICATORS",
        }

    with patch(
        "iqoption_research.research.dataset_manager.inspect_session", side_effect=mock_inspect
    ):
        manager = ResearchDatasetManager(tmp_path)
        sources = manager.discover_sources()

    assert len(sources) == 3
    eligible = [s for s in sources if s.is_eligible]
    excluded = [s for s in sources if not s.is_eligible]

    assert len(eligible) == 2
    assert len(excluded) == 1
    assert excluded[0].source_id == "collection_empty"
    assert "INELIGIBLE_EMPTY_ZERO_CANDLES" in excluded[0].exclusion_reason


def test_deduplication_and_quarantine(tmp_path: Path) -> None:
    # Source 1: candles at 1000, 1060, 1120
    p1 = tmp_path / "data" / "catalog" / "p1.parquet"
    sha1 = _create_dummy_parquet(p1, start_ts=1000, n_candles=3, interval_sec=60, open_p=1.1000)

    # Source 2: candle at 1060 identical, but 1120 conflicting
    p2 = tmp_path / "data" / "catalog" / "p2.parquet"
    p2.parent.mkdir(parents=True, exist_ok=True)
    from_ts = [1060, 1120, 1180]
    to_ts = [t + 60 for t in from_ts]
    opens = [1.1001, 1.2000, 1.1003]  # 1120 has open 1.2000 vs 1.1002 in p1
    table = pa.Table.from_arrays(
        [
            pa.array(["EURUSD"] * 3, type=pa.string()),
            pa.array([1] * 3, type=pa.int32()),
            pa.array([False] * 3, type=pa.bool_()),
            pa.array([60] * 3, type=pa.int32()),
            pa.array(from_ts, type=pa.int64()),
            pa.array(to_ts, type=pa.int64()),
            pa.array(to_ts, type=pa.int64()),
            pa.array(opens, type=pa.float64()),
            pa.array([p + 0.0002 for p in opens], type=pa.float64()),
            pa.array([p - 0.0002 for p in opens], type=pa.float64()),
            pa.array([p + 0.0001 for p in opens], type=pa.float64()),
            pa.array([100] * 3, type=pa.int64()),
            pa.array(opens, type=pa.float64()),
            pa.array([p + 0.00005 for p in opens], type=pa.float64()),
            pa.array([True] * 3, type=pa.bool_()),
            pa.array(list(range(3)), type=pa.int32()),
        ],
        schema=CANDLE_ARROW_SCHEMA,
    )
    pq.write_table(table, str(p2))
    sha2 = hashlib.sha256(p2.read_bytes()).hexdigest()

    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        _make_catalog_entry(
            dataset_id="p1",
            path=p1,
            sha=sha1,
            candle_count=3,
            from_ts=1000,
            to_ts=1120,
        )
    )
    cat.register_dataset(
        _make_catalog_entry(
            dataset_id="p2",
            path=p2,
            sha=sha2,
            candle_count=3,
            from_ts=1060,
            to_ts=1180,
        )
    )

    manager = ResearchDatasetManager(tmp_path)
    snapshot = manager.build_snapshot()

    # Total timestamps: 1000, 1060, 1120, 1180
    # 1060 was identical -> deduplicated (count 1)
    # 1120 was conflicting -> quarantined (excluded completely from retained candles)
    # Retained timestamps: 1000, 1060, 1180 -> 3 candles
    assert snapshot.manifest.total_candles == 3
    assert snapshot.manifest.deduplicated_boundaries_count == 1
    assert snapshot.manifest.quarantined_conflicts_count == 1
    assert 1120 in snapshot.manifest.quarantined_conflict_timestamps


def test_gap_classification(tmp_path: Path) -> None:
    # Create candles spanning Friday close and Sunday open
    # 2026-11-20 21:58 UTC (Friday) to 2026-11-20 21:59 UTC
    # Then weekend closure until 2026-11-22 22:05 UTC (Sunday)
    # Then an unexpected intraday gap on Monday: 2026-11-23 10:00 to 10:20
    dt_fri = datetime(2026, 11, 20, 21, 58, tzinfo=UTC)
    ts_fri = int(dt_fri.timestamp())

    dt_sun = datetime(2026, 11, 22, 22, 5, tzinfo=UTC)
    ts_sun = int(dt_sun.timestamp())

    dt_mon1 = datetime(2026, 11, 23, 10, 0, tzinfo=UTC)
    ts_mon1 = int(dt_mon1.timestamp())

    dt_mon2 = datetime(2026, 11, 23, 10, 20, tzinfo=UTC)
    ts_mon2 = int(dt_mon2.timestamp())

    from_ts = [
        ts_fri,
        ts_fri + 60,
        ts_sun,
        ts_sun + 60,
        ts_mon1,
        ts_mon2,
    ]
    to_ts = [t + 60 for t in from_ts]
    p = tmp_path / "data" / "catalog" / "gap_test.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array(["EURUSD"] * len(from_ts), type=pa.string()),
            pa.array([1] * len(from_ts), type=pa.int32()),
            pa.array([False] * len(from_ts), type=pa.bool_()),
            pa.array([60] * len(from_ts), type=pa.int32()),
            pa.array(from_ts, type=pa.int64()),
            pa.array(to_ts, type=pa.int64()),
            pa.array(to_ts, type=pa.int64()),
            pa.array([1.0500] * len(from_ts), type=pa.float64()),
            pa.array([1.0502] * len(from_ts), type=pa.float64()),
            pa.array([1.0498] * len(from_ts), type=pa.float64()),
            pa.array([1.0501] * len(from_ts), type=pa.float64()),
            pa.array([100] * len(from_ts), type=pa.int64()),
            pa.array([1.0501] * len(from_ts), type=pa.float64()),
            pa.array([1.05015] * len(from_ts), type=pa.float64()),
            pa.array([True] * len(from_ts), type=pa.bool_()),
            pa.array(list(range(len(from_ts))), type=pa.int32()),
        ],
        schema=CANDLE_ARROW_SCHEMA,
    )
    pq.write_table(table, str(p))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()

    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        _make_catalog_entry(
            dataset_id="gaps",
            path=p,
            sha=sha,
            candle_count=len(from_ts),
            from_ts=from_ts[0],
            to_ts=to_ts[-1],
        )
    )

    manager = ResearchDatasetManager(tmp_path)
    snapshot = manager.build_snapshot(max_gap_minutes=5)

    assert len(snapshot.manifest.weekend_closures) == 1
    assert snapshot.manifest.weekend_closures[0]["gap_type"] == "scheduled_weekend_closure"

    assert len(snapshot.manifest.intraday_gaps) == 2
    assert snapshot.manifest.intraday_gaps[-1]["gap_minutes"] == 19


def test_tamper_detection(tmp_path: Path) -> None:
    p = tmp_path / "data" / "catalog" / "p.parquet"
    sha = _create_dummy_parquet(p, start_ts=1700000000, n_candles=50)

    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        _make_catalog_entry(
            dataset_id="p",
            path=p,
            sha=sha,
            candle_count=50,
            from_ts=1700000000,
            to_ts=1700000000 + 49 * 60,
        )
    )

    manager = ResearchDatasetManager(tmp_path)
    snapshot = manager.build_snapshot()
    assert snapshot.verify_integrity()

    # Tamper with the candles parquet file
    with Path(snapshot.candles_path).open("ab") as f:
        f.write(b"tamper_bytes")

    assert not snapshot.verify_integrity()
