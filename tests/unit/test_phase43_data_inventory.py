"""Tests for Phase 43 Data Inventory, Capability Assessment, and Collection Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from iqoption_research.market_data.collection_engine import (
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from iqoption_research.market_data.data_inventory import (
    FieldProvenance,
    HorizonReadinessStatus,
    audit_genuine_market_data,
    compute_horizon_readiness,
)
from iqoption_research.models.market import Candle
from tests.fixtures.synthetic_market import (
    create_deterministic_synthetic_candles,
    setup_isolated_research_workspace,
)


def test_horizon_readiness_calculation_empty() -> None:
    """Empty candle sequence results in DATA_INSUFFICIENT for all horizons."""
    readiness = compute_horizon_readiness([], warmup_bars=50)
    assert len(readiness) == 6
    assert all(r.status == HorizonReadinessStatus.DATA_INSUFFICIENT for r in readiness.values())
    assert all(r.rolling_opportunities_count == 0 for r in readiness.values())


def test_horizon_readiness_calculation_sufficient() -> None:
    """Continuous candles produce correct rolling and non-overlapping counts across horizons."""
    # 2500 candles spanning multiple days
    candles = create_deterministic_synthetic_candles(count=2500)
    readiness = compute_horizon_readiness(candles, warmup_bars=50)

    assert "H1" in readiness
    assert "H3" in readiness
    assert "H5" in readiness
    assert "H15" in readiness
    assert "H30" in readiness
    assert "H60" in readiness

    h1 = readiness["H1"]
    assert h1.horizon_bars == 1
    # 2500 - 1 - 50 = 2449 rolling opportunities
    assert h1.rolling_opportunities_count == 2449
    assert h1.non_overlapping_opportunities_count == 2449

    h5 = readiness["H5"]
    assert h5.horizon_bars == 5
    assert h5.rolling_opportunities_count == 2445
    # Non-overlapping count = ceil((2445)/5) = 489
    assert h5.non_overlapping_opportunities_count == 489

    h60 = readiness["H60"]
    assert h60.horizon_bars == 60
    assert h60.rolling_opportunities_count == 2390
    assert h60.non_overlapping_opportunities_count == 40


def test_audit_genuine_market_data(tmp_path: Path) -> None:
    """audit_genuine_market_data generates comprehensive inventory and readiness report."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1200)
    report = audit_genuine_market_data(setup["root"])

    assert report.generated_at_utc is not None
    assert report.capability_assessment is not None
    assert report.capability_assessment.individual_quote_updates.startswith("UNVERIFIED")
    assert report.capability_assessment.genuine_bid_ask_quotes.startswith("UNAVAILABLE")
    assert report.field_provenance_matrix["bid"] == FieldProvenance.UNAVAILABLE_IN_BROKER_FEED.value
    assert report.field_provenance_matrix["open"] == FieldProvenance.OBSERVED_GENUINE.value
    assert len(report.horizon_readiness_matrix) == 6

    md = report.to_markdown()
    assert "# Genuine Market Data Inventory & Readiness Matrix" in md
    assert "Field Provenance Matrix" in md
    assert "Multi-Horizon Dataset Readiness Matrix" in md


def test_historical_collection_coordinator_plan_and_record(tmp_path: Path) -> None:
    """HistoricalCollectionCoordinator plans missing intervals and records durable receipts."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, missing, progress = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-01T00:00:00+00:00",
        end_utc="2026-11-01T02:00:00+00:00",
        timeframe_seconds=60,
    )

    assert job_id.startswith("collect_eurusd_")
    assert len(missing) == 1
    assert missing[0].expected_candles_count == 121
    assert progress.status == CollectionStatus.PENDING

    # Record batch
    candles = create_deterministic_synthetic_candles(count=50)
    receipt = coord.record_collected_batch(job_id, candles, is_synthetic=True)

    assert receipt.receipt_id.startswith("rcpt_collect_eurusd_")
    assert receipt.candle_count == 50
    assert receipt.parquet_sha256 != ""
    assert receipt.source_provenance == "SYNTHETIC_TEST_FIXTURE"
    assert (coord.receipts_dir / f"{receipt.receipt_id}.json").is_file()


@pytest.mark.asyncio
async def test_historical_collection_coordinator_execute_collection(tmp_path: Path) -> None:
    """HistoricalCollectionCoordinator executes collection via collector and supports cancellation."""
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, MagicMock

    from iqoption_research.market_data.fetch import CandleFetchResult
    from iqoption_research.market_data.validation import assess_data_quality
    from iqoption_research.models.market import CandleSize

    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _missing, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-01T00:00:00+00:00",
        end_utc="2026-11-01T00:10:00+00:00",
        timeframe_seconds=60,
    )

    # Build mock authentic candles
    start_ts = int(datetime.fromisoformat("2026-11-01T00:00:00+00:00").timestamp())
    mock_candles = create_deterministic_synthetic_candles(count=10, start_ts=start_ts)
    auth_candles = [
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
            provenance="historical",
            identity_provenance="explicit_broker_metadata",
        )
        for c in mock_candles
    ]

    quality = assess_data_quality(
        auth_candles, expected_size=CandleSize(60), now_utc=datetime.now(UTC)
    )
    fetch_result = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(auth_candles),
        errors=(),
        quality=quality,
        fetched_at=datetime.now(UTC),
        raw_count=len(auth_candles),
        source="iqoption",
    )

    mock_client = MagicMock()
    mock_client.assert_read_only = MagicMock()
    mock_client.get_candles = AsyncMock(return_value=fetch_result)

    progress_reports = []

    def on_progress(p: Any) -> None:
        progress_reports.append(p)

    # 1. Execute collection
    final_progress = await coord.execute_collection(
        job_id=job_id,
        client=mock_client,
        progress_cb=on_progress,
        chunk_size=10,
        is_synthetic=False,
    )

    assert mock_client.assert_read_only.called
    assert final_progress.status == CollectionStatus.COMPLETED
    assert final_progress.candles_collected_this_session > 0
    assert len(progress_reports) > 0

    # Verify receipt provenance
    receipts = list(coord.receipts_dir.glob("*.json"))
    assert len(receipts) > 0
    receipt_data = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt_data["source_provenance"] == "AUTHENTIC_BROKER_COLLECTION"

    # 2. Test Cancellation
    job_id2, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-02T00:00:00+00:00",
        end_utc="2026-11-02T01:00:00+00:00",
        timeframe_seconds=60,
    )

    # Cancellation token that fires immediately
    canc_progress = await coord.execute_collection(
        job_id=job_id2,
        client=mock_client,
        cancel_token=lambda: True,
    )
    assert canc_progress.status == CollectionStatus.CANCELLED
