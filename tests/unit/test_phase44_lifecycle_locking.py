"""Unit tests for Phase 44 single-worker job locking and persistent IPC cancellation.

Tests:
1. Job lock creation and collision detection (cannot run concurrent collectors on same job).
2. Lock release allows subsequent collection.
3. Persistent IPC cancel file (<job_id>.cancel) stops active collection worker safely at page boundary.
4. Resuming across app restart preserves durable cursor and completes cleanly.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iqoption_research.market_data.collection_engine import (
    CollectionCheckpoint,
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.market import CandleSize
from tests.unit.test_phase43_data_inventory import (
    create_deterministic_synthetic_candles,
)


def test_single_worker_job_locking(tmp_path: Path) -> None:
    """Acquiring a lock on an already-running job raises RuntimeError."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id = "collect_eurusd_test_lock_123"

    # Acquire initial lock
    lock_path = coord.acquire_job_lock(job_id)
    assert lock_path.is_file()

    # Attempting to acquire lock while active PID holds it raises RuntimeError
    with pytest.raises(RuntimeError, match="locked by another active collection process"):
        coord.acquire_job_lock(job_id)

    # Release lock
    coord.release_job_lock(job_id)
    assert not lock_path.is_file()

    # Re-acquire succeeds
    lock_path2 = coord.acquire_job_lock(job_id)
    assert lock_path2.is_file()
    coord.release_job_lock(job_id)


def test_same_process_concurrent_worker_is_rejected(tmp_path: Path) -> None:
    """Two coordinators in one interpreter cannot own the same job lock."""
    coord1 = HistoricalCollectionCoordinator(tmp_path)
    coord2 = HistoricalCollectionCoordinator(tmp_path)
    job_id = "collect_eurusd_test_same_process_lock"

    coord1.acquire_job_lock(job_id)
    try:
        with pytest.raises(RuntimeError, match="in this interpreter"):
            coord2.acquire_job_lock(job_id)
    finally:
        coord1.release_job_lock(job_id)

    coord2.acquire_job_lock(job_id)
    coord2.release_job_lock(job_id)


@pytest.mark.asyncio
async def test_ipc_cancellation_signal_file(tmp_path: Path) -> None:
    """A .cancel IPC file placed in checkpoints dir cancels running execute_collection."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-04T10:00:00+00:00",
        end_utc="2026-11-04T10:30:00+00:00",
        timeframe_seconds=60,
    )

    start_ts = int(datetime.fromisoformat("2026-11-04T10:00:00+00:00").timestamp())
    candles = create_deterministic_synthetic_candles(count=15, start_ts=start_ts + 15 * 60)
    q = assess_data_quality(candles, expected_size=CandleSize(60))
    fetch_res = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(candles),
        errors=(),
        quality=q,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles),
        source="synthetic",
    )

    mock_client = MagicMock()
    mock_client.assert_read_only = MagicMock()

    # Simulate side effect: on first broker fetch, trigger cancel_collection (writing .cancel IPC file)
    def on_fetch(*args, **kwargs):
        coord.cancel_collection(job_id)
        return fetch_res

    mock_client.get_candles = AsyncMock(side_effect=on_fetch)

    prog = await coord.execute_collection(
        job_id=job_id,
        client=mock_client,
        chunk_size=15,
        is_synthetic=True,
        operator_authorized=True,
    )

    assert prog.status == CollectionStatus.CANCELLED
    ckpt = CollectionCheckpoint.load(tmp_path / f"data/collection_checkpoints/{job_id}.json")
    assert ckpt.status == CollectionStatus.CANCELLED.value

    # Verify lock was cleanly released
    lock_file = tmp_path / f"data/collection_checkpoints/{job_id}.lock"
    assert not lock_file.exists()


@pytest.mark.asyncio
async def test_resume_after_app_restart(tmp_path: Path) -> None:
    """Coordinator created in a new process/session can resume an interrupted job cleanly."""
    coord1 = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord1.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-04T10:00:00+00:00",
        end_utc="2026-11-04T10:20:00+00:00",
        timeframe_seconds=60,
    )

    start_ts = int(datetime.fromisoformat("2026-11-04T10:00:00+00:00").timestamp())
    all_candles = create_deterministic_synthetic_candles(count=20, start_ts=start_ts)

    # First batch: 10 candles
    batch1 = all_candles[10:]
    q1 = assess_data_quality(batch1, expected_size=CandleSize(60))
    res1 = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(batch1),
        errors=(),
        quality=q1,
        fetched_at=datetime.now(UTC),
        raw_count=len(batch1),
        source="synthetic",
    )
    mock1 = MagicMock()
    mock1.assert_read_only = MagicMock()
    mock1.get_candles = AsyncMock(return_value=res1)

    cancel_requested = False

    def cancel_after_commit(_progress) -> None:
        nonlocal cancel_requested
        if not cancel_requested:
            cancel_requested = True
            coord1.cancel_collection(job_id)

    prog1 = await coord1.execute_collection(
        job_id=job_id,
        client=mock1,
        chunk_size=10,
        progress_cb=cancel_after_commit,
        is_synthetic=True,
        operator_authorized=True,
    )
    assert prog1.status == CollectionStatus.CANCELLED

    # Simulate app restart: Instantiate fresh coordinator instance
    coord2 = HistoricalCollectionCoordinator(tmp_path)
    jobs = coord2.list_jobs()
    assert any(j.job_id == job_id for j in jobs)

    # Second batch: remaining 10 candles
    batch2 = all_candles[:10]
    q2 = assess_data_quality(batch2, expected_size=CandleSize(60))
    res2 = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(batch2),
        errors=(),
        quality=q2,
        fetched_at=datetime.now(UTC),
        raw_count=len(batch2),
        source="synthetic",
    )
    mock2 = MagicMock()
    mock2.assert_read_only = MagicMock()
    mock2.get_candles = AsyncMock(return_value=res2)

    prog2 = await coord2.resume_collection(
        job_id=job_id,
        client=mock2,
        chunk_size=10,
        is_synthetic=True,
        operator_authorized=True,
    )

    assert prog2.status == CollectionStatus.COMPLETED_VERIFIED
    assert prog2.candles_collected_this_session == 20


@pytest.mark.asyncio
async def test_idle_cancel_signal_does_not_block_explicit_resume(tmp_path: Path) -> None:
    """Cancelling an idle job leaves a stale signal that explicit resume must consume."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-05T10:00:00+00:00",
        end_utc="2026-11-05T10:10:00+00:00",
        timeframe_seconds=60,
    )

    cancelled = coord.cancel_collection(job_id)
    assert cancelled.status == CollectionStatus.CANCELLED
    cancel_file = tmp_path / "data/collection_checkpoints" / f"{job_id}.cancel"
    assert cancel_file.is_file()

    start_ts = int(datetime.fromisoformat("2026-11-05T10:00:00+00:00").timestamp())
    candles = create_deterministic_synthetic_candles(count=10, start_ts=start_ts)
    quality = assess_data_quality(candles, expected_size=CandleSize(60))
    fetch_res = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(candles),
        errors=(),
        quality=quality,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles),
        source="synthetic",
    )
    mock_client = MagicMock()
    mock_client.assert_read_only = MagicMock()
    mock_client.get_candles = AsyncMock(return_value=fetch_res)

    prog = await coord.resume_collection(
        job_id=job_id,
        client=mock_client,
        chunk_size=10,
        is_synthetic=True,
        operator_authorized=True,
    )

    assert prog.status == CollectionStatus.COMPLETED_VERIFIED
    assert not cancel_file.exists()


@pytest.mark.asyncio
async def test_cancel_issued_during_resume_authentication_is_not_lost(tmp_path: Path) -> None:
    """A cancel after request-time token capture survives the authentication gap."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-06T10:00:00+00:00",
        end_utc="2026-11-06T10:10:00+00:00",
        timeframe_seconds=60,
    )
    coord.cancel_collection(job_id)
    cancel_file = tmp_path / "data/collection_checkpoints" / f"{job_id}.cancel"
    # CLI/desktop capture at resume-request time, before broker authentication.
    stale_cancel_token = coord.capture_stale_cancel_token(job_id)
    assert stale_cancel_token is not None

    start_ts = int(datetime.fromisoformat("2026-11-06T10:00:00+00:00").timestamp())
    candles = create_deterministic_synthetic_candles(count=10, start_ts=start_ts)
    quality = assess_data_quality(candles, expected_size=CandleSize(60))
    fetch_res = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(candles),
        errors=(),
        quality=quality,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles),
        source="synthetic",
    )

    def issue_new_cancel() -> None:
        coord._write_cancel_signal(job_id)

    mock_client = MagicMock()
    mock_client.assert_read_only = MagicMock(side_effect=issue_new_cancel)
    mock_client.get_candles = AsyncMock(return_value=fetch_res)

    prog = await coord.resume_collection(
        job_id=job_id,
        client=mock_client,
        chunk_size=10,
        is_synthetic=True,
        operator_authorized=True,
        stale_cancel_token=stale_cancel_token,
        capture_stale_cancel_token=False,
    )

    assert prog.status == CollectionStatus.CANCELLED
    assert mock_client.get_candles.await_count == 0
    assert not cancel_file.exists()


@pytest.mark.asyncio
async def test_cancel_during_final_verification_wins_over_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel after the last batch cannot be overwritten by COMPLETED_VERIFIED."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-07T10:00:00+00:00",
        end_utc="2026-11-07T10:10:00+00:00",
        timeframe_seconds=60,
    )
    start_ts = int(datetime.fromisoformat("2026-11-07T10:00:00+00:00").timestamp())
    candles = create_deterministic_synthetic_candles(count=10, start_ts=start_ts)
    quality = assess_data_quality(candles, expected_size=CandleSize(60))
    fetch_res = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(candles),
        errors=(),
        quality=quality,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles),
        source="synthetic",
    )
    mock_client = MagicMock()
    mock_client.assert_read_only = MagicMock()
    mock_client.get_candles = AsyncMock(return_value=fetch_res)

    real_verify = coord.verify_job_integrity

    def verify_then_cancel(target_job_id: str) -> dict:
        coord._write_cancel_signal(target_job_id)
        return real_verify(target_job_id)

    monkeypatch.setattr(coord, "verify_job_integrity", verify_then_cancel)

    prog = await coord.execute_collection(
        job_id=job_id,
        client=mock_client,
        chunk_size=10,
        is_synthetic=True,
        operator_authorized=True,
    )

    assert prog.status == CollectionStatus.CANCELLED
    assert not (tmp_path / "data/collection_checkpoints" / f"{job_id}.cancel").exists()


@pytest.mark.asyncio
async def test_active_worker_is_the_only_writer_of_cancel_status(tmp_path: Path) -> None:
    """The cancel command signals only; the lock owner persists the final status."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-08T10:00:00+00:00",
        end_utc="2026-11-08T10:10:00+00:00",
        timeframe_seconds=60,
    )
    start_ts = int(datetime.fromisoformat("2026-11-08T10:00:00+00:00").timestamp())
    candles = create_deterministic_synthetic_candles(count=10, start_ts=start_ts)
    quality = assess_data_quality(candles, expected_size=CandleSize(60))
    fetch_res = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(candles),
        errors=(),
        quality=quality,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles),
        source="synthetic",
    )
    signalled_statuses: list[CollectionStatus] = []

    async def cancel_during_fetch(*_args: Any, **_kwargs: Any) -> CandleFetchResult:
        signalled_statuses.append(coord.cancel_collection(job_id).status)
        await asyncio.sleep(0)
        return fetch_res

    mock_client = MagicMock()
    mock_client.assert_read_only = MagicMock()
    mock_client.get_candles = AsyncMock(side_effect=cancel_during_fetch)

    prog = await coord.execute_collection(
        job_id=job_id,
        client=mock_client,
        chunk_size=10,
        is_synthetic=True,
        operator_authorized=True,
    )

    assert signalled_statuses == [CollectionStatus.IN_PROGRESS]
    assert prog.status == CollectionStatus.CANCELLED
    ckpt = CollectionCheckpoint.load(tmp_path / "data/collection_checkpoints" / f"{job_id}.json")
    assert ckpt.status == CollectionStatus.CANCELLED.value
