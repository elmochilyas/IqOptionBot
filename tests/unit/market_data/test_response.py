"""Weekend discontinuities, including 47/49-hour DST closures, in one response."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from iqoption_research.market_data.collection_engine import (
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from tests.unit.market_data.test_coordinator import _make_auth_candles


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("friday", "sunday"),
    [
        ("2026-10-23T20:59:00", "2026-10-25T21:00:00"),
        ("2026-11-20T21:59:00", "2026-11-22T22:00:00"),
        ("2026-10-30T20:59:00", "2026-11-01T22:00:00"),
        ("2026-03-06T21:59:00", "2026-03-08T21:00:00"),
    ],
)
async def test_weekend_response_segments_and_verified_closure(tmp_path, friday, sunday):
    start = int(datetime.fromisoformat(friday).replace(tzinfo=UTC).timestamp())
    reopen = int(datetime.fromisoformat(sunday).replace(tzinfo=UTC).timestamp())
    end = reopen + 7200
    candles = [_make_auth_candles(1, start)[0], *_make_auth_candles(121, reopen)]

    class Broker:
        _settings = SimpleNamespace(max_candles=500)

        def assert_read_only(self):
            pass

        async def get_candles(self, symbol, *, size, count, endtime):
            return SimpleNamespace(
                candle_list=[c for c in candles if c.from_ts <= endtime][-count:]
            )

    c = HistoricalCollectionCoordinator(tmp_path)
    job, _, _ = c.plan_collection(
        symbol="EURUSD",
        start_utc=datetime.fromtimestamp(start, UTC).isoformat(),
        end_utc=datetime.fromtimestamp(end, UTC).isoformat(),
        timeframe_seconds=60,
    )
    result = await c.execute_collection(job_id=job, client=Broker(), operator_authorized=True)
    assert result.status == CollectionStatus.COMPLETED_VERIFIED
    report = c.verify_job_integrity(job)
    assert report["errors"] == []
    assert report["unique_verified_candles"] == 121
    assert report["receipt_count"] == 2
    assert report["coverage_gaps_count"] == 0
