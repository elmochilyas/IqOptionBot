"""Unit tests for Phase 44 historical collection coordinator corrections.

Tests:
1. Client authorization enforcement (rejects missing assert_read_only or rogue buy/sell methods).
2. False completion prevention on empty broker responses.
3. Scheduled weekend closure recognition.
4. Durable resumption with storage reconciliation (no double-counting).
5. Safe cooperative cancellation at page boundary.
6. Cryptographic job integrity verification.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from iqoption_research.market_data.collection_engine import (
    CollectionCheckpoint,
    CollectionStatus,
    HistoricalCollectionCoordinator,
    is_interval_within_scheduled_closure,
    verify_authorized_client,
)
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.market import Candle, CandleSize
from tests.unit.market_data.test_inventory import (
    create_deterministic_synthetic_candles,
)


def _make_auth_candles(count: int, start_ts: int) -> list[Candle]:
    synthetic = create_deterministic_synthetic_candles(count=count, start_ts=start_ts)
    return [
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
        for c in synthetic
    ]


def test_client_authorization_enforced() -> None:
    """Coordinator enforces assert_read_only() and rejects forbidden execution methods."""

    # 1. Missing assert_read_only
    class ArbitraryClient:
        async def get_candles(self, *args, **kwargs):
            return None

    with pytest.raises(PermissionError, match="missing mandatory assert_read_only"):
        verify_authorized_client(ArbitraryClient())

    # 2. Possesses forbidden buy() method
    class RogueClient:
        def assert_read_only(self) -> None:
            pass

        def buy(self) -> None:
            pass

        async def get_candles(self, *args, **kwargs):
            return None

    with pytest.raises(PermissionError, match="forbidden execution method 'buy'"):
        verify_authorized_client(RogueClient())

    # 3. Legitimate read-only mock
    mock_clean = MagicMock()
    mock_clean.assert_read_only = MagicMock()
    mock_clean.get_candles = AsyncMock()
    # Must not raise
    verify_authorized_client(mock_clean)
    assert mock_clean.assert_read_only.called


@pytest.mark.asyncio
async def test_execute_requires_explicit_operator_authorization(tmp_path: Path) -> None:
    """Unattended execution fails closed before any checkpoint mutation."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-04T10:00:00+00:00",
        end_utc="2026-11-04T10:10:00+00:00",
        timeframe_seconds=60,
    )
    client = MagicMock()
    client.assert_read_only = MagicMock()
    client.get_candles = AsyncMock()

    with pytest.raises(PermissionError, match="operator authorization"):
        await coord.execute_collection(job_id=job_id, client=client)

    ckpt_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.json"
    assert CollectionCheckpoint.load(ckpt_path).status == CollectionStatus.PENDING.value
    assert not list(coord.receipts_dir.glob("*.json"))


@pytest.mark.asyncio
async def test_false_completion_prevented_on_empty_response(tmp_path: Path) -> None:
    """Empty response during open market interval must NOT mark interval or job completed."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    # Wednesday morning (open market)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-04T10:00:00+00:00",
        end_utc="2026-11-04T11:00:00+00:00",
        timeframe_seconds=60,
    )

    # Mock client that returns empty candle list
    empty_result = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=(),
        errors=(),
        quality=MagicMock(),
        fetched_at=datetime.now(UTC),
        raw_count=0,
        source="iqoption",
    )
    mock_client = MagicMock()
    mock_client.assert_read_only = MagicMock()
    mock_client.get_candles = AsyncMock(return_value=empty_result)

    progress = await coord.execute_collection(
        job_id=job_id,
        client=mock_client,
        is_synthetic=False,
        operator_authorized=True,
    )

    # Must be SOURCE_UNAVAILABLE, NOT COMPLETED_VERIFIED!
    assert progress.status == CollectionStatus.SOURCE_UNAVAILABLE
    assert progress.percent_complete < 100.0
    assert "zero candles" in (progress.failure_reason or "").lower()

    # Verify checkpoint on disk
    ckpt = CollectionCheckpoint.load(tmp_path / "data/collection_checkpoints" / f"{job_id}.json")
    assert ckpt.status == CollectionStatus.SOURCE_UNAVAILABLE.value
    assert len(ckpt.completed_interval_indices) == 0


def test_weekend_closure_recognized_correctly() -> None:
    """Verify that is_interval_within_scheduled_closure accurately identifies Forex weekends."""
    # Saturday 12:00 UTC (Saturday is always closed for EURUSD)
    sat_start = int(datetime.fromisoformat("2026-11-07T00:00:00+00:00").timestamp())
    sat_end = int(datetime.fromisoformat("2026-11-07T23:59:00+00:00").timestamp())
    assert is_interval_within_scheduled_closure("EURUSD", sat_start, sat_end) is True

    # Wednesday 12:00 UTC (open market)
    wed_start = int(datetime.fromisoformat("2026-11-04T12:00:00+00:00").timestamp())
    wed_end = int(datetime.fromisoformat("2026-11-04T13:00:00+00:00").timestamp())
    assert is_interval_within_scheduled_closure("EURUSD", wed_start, wed_end) is False

    # OTC symbols have unverified schedules; must return False
    assert is_interval_within_scheduled_closure("EURUSD-OTC", sat_start, sat_end) is False


@pytest.mark.asyncio
async def test_durable_resumption_and_reconciliation(tmp_path: Path) -> None:
    """Interrupted collection can be reconciled and resumed without duplicate counts."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-02T10:00:00+00:00",
        end_utc="2026-11-02T10:20:00+00:00",
        timeframe_seconds=60,
    )

    start_ts = int(datetime.fromisoformat("2026-11-02T10:00:00+00:00").timestamp())
    candles_first_batch = _make_auth_candles(count=10, start_ts=start_ts + 600)

    quality1 = assess_data_quality(candles_first_batch, expected_size=CandleSize(60))
    fetch_result1 = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(candles_first_batch),
        errors=(),
        quality=quality1,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles_first_batch),
        source="iqoption",
    )

    mock_client = MagicMock()
    mock_client.assert_read_only = MagicMock()
    mock_client.get_candles = AsyncMock(return_value=fetch_result1)

    # 1. Execute first batch, then request cancellation before the next fetch.
    cancel_requested = False

    def cancel_after_commit(_progress) -> None:
        nonlocal cancel_requested
        if not cancel_requested:
            cancel_requested = True
            coord.cancel_collection(job_id)

    prog1 = await coord.execute_collection(
        job_id=job_id,
        client=mock_client,
        chunk_size=10,
        progress_cb=cancel_after_commit,
        is_synthetic=False,
        operator_authorized=True,
    )
    assert prog1.status == CollectionStatus.CANCELLED
    assert prog1.candles_collected_this_session == 10

    # 2. Resuming without operator authorization raises PermissionError
    with pytest.raises(PermissionError, match="operator authorization"):
        await coord.resume_collection(
            job_id=job_id,
            client=mock_client,
            operator_authorized=False,
        )

    # 3. Reconcile job state
    reconciled = coord.reconcile_job_state(job_id)
    assert reconciled.total_candles_collected == 10
    assert len(reconciled.receipt_ids) >= 1

    # 4. Resume with remaining candles
    candles_second_batch = _make_auth_candles(count=10, start_ts=start_ts)
    quality2 = assess_data_quality(candles_second_batch, expected_size=CandleSize(60))
    fetch_result2 = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(candles_second_batch),
        errors=(),
        quality=quality2,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles_second_batch),
        source="iqoption",
    )
    mock_client.get_candles = AsyncMock(return_value=fetch_result2)

    prog2 = await coord.resume_collection(
        job_id=job_id,
        client=mock_client,
        operator_authorized=True,
        chunk_size=10,
        is_synthetic=False,
    )
    assert prog2.status == CollectionStatus.COMPLETED_VERIFIED
    assert prog2.percent_complete == 100.0


def test_job_integrity_verification(tmp_path: Path) -> None:
    """Verifies that verify_job_integrity checks receipts and checkpoint validity."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-02T10:00:00+00:00",
        end_utc="2026-11-02T10:10:00+00:00",
        timeframe_seconds=60,
    )

    report_pending = coord.verify_job_integrity(job_id)
    assert report_pending["verified"] is False
    assert report_pending["status"] == CollectionStatus.PENDING.value

    # Write a batch to create a receipt
    start_ts = int(datetime.fromisoformat("2026-11-02T10:00:00+00:00").timestamp())
    candles = _make_auth_candles(10, start_ts)
    coord.record_collected_batch(job_id, candles, is_synthetic=False)

    ckpt_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.json"
    ckpt = CollectionCheckpoint.load(ckpt_path)
    ckpt.status = CollectionStatus.COMPLETED_VERIFIED.value
    ckpt.completed_interval_indices = [0]
    ckpt.save(ckpt_path)

    report_complete = coord.verify_job_integrity(job_id)
    assert report_complete["verified"] is True
    assert report_complete["receipt_count"] >= 1
