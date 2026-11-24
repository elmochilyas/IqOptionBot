"""Offline discontinuous-response receipt and resume tests."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from iqoption_research.market_data.collection_engine import (
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from tests.unit.test_phase44_collection_coordinator import _make_auth_candles


@pytest.mark.asyncio
@pytest.mark.parametrize("gaps", [(5,), (3, 7, 10)])
@pytest.mark.parametrize("cancel", [False, True, "crash"])
async def test_segmented_response(tmp_path, gaps, cancel):
    start = int(datetime(2026, 11, 18, 10, tzinfo=UTC).timestamp())
    coord = HistoricalCollectionCoordinator(tmp_path)
    job, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc=datetime.fromtimestamp(start, UTC).isoformat(),
        end_utc=datetime.fromtimestamp(start + 900, UTC).isoformat(),
        timeframe_seconds=60,
    )
    candles = [c for i, c in enumerate(_make_auth_candles(16, start)) if i not in gaps]
    calls = []

    class Broker:
        _settings = SimpleNamespace(max_candles=500)

        def assert_read_only(self):
            pass

        async def get_candles(self, symbol, *, size, count, endtime):
            calls.append(endtime)
            assert len(calls) < 10
            return SimpleNamespace(
                candle_list=[c for c in candles if c.from_ts <= endtime][-count:]
            )

    class SimulatedCrash(BaseException):
        pass

    def progress(_):
        if cancel == "crash":
            raise SimulatedCrash
        if cancel:
            coord.cancel_collection(job)

    try:
        result = await coord.execute_collection(
            job_id=job, client=Broker(), operator_authorized=True, progress_cb=progress
        )
    except SimulatedCrash:
        assert cancel == "crash"
        coord = HistoricalCollectionCoordinator(tmp_path)
        result = await coord.resume_collection(
            job_id=job, client=Broker(), operator_authorized=True
        )
        cancel = False
    if cancel:
        assert result.status == CollectionStatus.CANCELLED
        result = await coord.resume_collection(
            job_id=job, client=Broker(), operator_authorized=True
        )
    assert result.status == CollectionStatus.PARTIALLY_COLLECTED
    report = coord.verify_job_integrity(job)
    assert report["errors"] == []
    assert report["unique_verified_candles"] == 15 - len(gaps)
    assert report["receipt_count"] == len(gaps) + 1
    assert report["coverage_gaps"][0]["uncovered_open_bars"] == len(gaps)
