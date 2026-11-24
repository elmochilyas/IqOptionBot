"""Offline broker boundary regressions; no live connectivity."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from iqoption_research.market_data.collection_engine import (
    CollectionCheckpoint,
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from tests.unit.test_phase44_collection_coordinator import _make_auth_candles


@pytest.mark.asyncio
@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize("count", [1, 15, 500, 501])
async def test_boundary_pages(tmp_path, inclusive, count):
    await exercise(tmp_path, inclusive, count)


async def exercise(tmp_path, inclusive, count, *, missing=False, resume=False):
    start = int(datetime(2026, 11, 23, 14, tzinfo=UTC).timestamp())
    coord = HistoricalCollectionCoordinator(tmp_path)
    job, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc=datetime.fromtimestamp(start, UTC).isoformat(),
        end_utc=datetime.fromtimestamp(start + count * 60, UTC).isoformat(),
        timeframe_seconds=60,
    )
    candles = _make_auth_candles(count + 2, start - 60)
    if missing:
        candles = [c for c in candles if c.from_ts != start]
    calls = []

    class Broker:
        _settings = SimpleNamespace(max_candles=500)

        def assert_read_only(self):
            pass

        async def get_candles(self, symbol, *, size, count, endtime):
            calls.append((count, endtime))
            assert 1 <= count <= 500
            assert len(calls) <= 5
            selected = [
                c for c in candles if c.from_ts <= endtime if inclusive or c.from_ts < endtime
            ]
            return SimpleNamespace(candle_list=selected[-count:])

    old_receipt = None
    old_bytes = None
    if resume:
        receipt = coord.record_collected_batch(
            job_id=job,
            candles=[c for c in candles if start < c.from_ts < start + count * 60],
            is_synthetic=False,
        )
        old_receipt = coord.receipts_dir / f"{receipt.receipt_id}.json"
        coord._finalize_receipt(receipt.receipt_id)
        old_bytes = old_receipt.read_bytes()
        checkpoint_path = coord.checkpoints_dir / f"{job}.json"
        checkpoint = CollectionCheckpoint.load(checkpoint_path)
        checkpoint.status = CollectionStatus.PARTIALLY_COLLECTED.value
        checkpoint.current_interval_cursor = start + 60
        checkpoint.save(checkpoint_path)
        result = await coord.resume_collection(
            job_id=job, client=Broker(), operator_authorized=True
        )
    else:
        result = await coord.execute_collection(
            job_id=job, client=Broker(), operator_authorized=True
        )
    report = coord.verify_job_integrity(job)
    assert not report["errors"]
    assert report["unique_verified_candles"] == count - int(missing)
    if missing:
        assert result.status != CollectionStatus.COMPLETED_VERIFIED
    else:
        assert result.status == CollectionStatus.COMPLETED_VERIFIED
    if resume:
        assert old_receipt.read_bytes() == old_bytes
        assert report["receipt_count"] == 2
        assert calls == [(2, start + 60)]


@pytest.mark.asyncio
@pytest.mark.parametrize("inclusive", [False, True])
async def test_unavailable_boundary_remains_missing(tmp_path, inclusive):
    await exercise(tmp_path, inclusive, 15, missing=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("inclusive", [False, True])
async def test_resume_preserves_fourteen_candles_and_receipt(tmp_path, inclusive):
    await exercise(tmp_path, inclusive, 15, resume=True)
