"""Representative multi-batch calendar collection & persistence tests (7 & 30 days)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from iqoption_research.market_data.collection_engine import (
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from iqoption_research.market_data.sessions import get_market_schedule
from tests.support.candles import make_authentic_candles


@pytest.mark.asyncio
@pytest.mark.parametrize("days", [7, 30])
async def test_calendar_scale(tmp_path, days):
    end = datetime(2026, 11, 23, tzinfo=UTC)
    start = end - timedelta(days=days)
    lo, hi = int(start.timestamp()), int(end.timestamp())
    schedule = get_market_schedule("EURUSD")
    expected = sum(
        schedule.is_market_open(datetime.fromtimestamp(ts, UTC)) for ts in range(lo, hi, 60)
    )
    calls = 0

    class Broker:
        _settings = SimpleNamespace(max_candles=500)

        def assert_read_only(self):
            pass

        async def get_candles(self, symbol, *, size, count, endtime):
            nonlocal calls
            calls += 1
            assert calls < expected // 490 + 60
            stamps = []
            cursor = endtime
            while len(stamps) < count and cursor >= lo - 60 * 60 * 72:
                if schedule.is_market_open(datetime.fromtimestamp(cursor, UTC)):
                    stamps.append(cursor)
                cursor -= 60
            candles = [make_authentic_candles(1, ts)[0] for ts in reversed(stamps)]
            return SimpleNamespace(candle_list=candles)

    coordinator = HistoricalCollectionCoordinator(tmp_path)
    job, _, _ = coordinator.plan_collection(
        symbol="EURUSD", start_utc=start.isoformat(), end_utc=end.isoformat(), timeframe_seconds=60
    )
    result = await coordinator.execute_collection(
        job_id=job, client=Broker(), operator_authorized=True
    )
    report = coordinator.verify_job_integrity(job)
    assert result.status == CollectionStatus.COMPLETED_VERIFIED
    assert report["unique_verified_candles"] == expected
    assert report["errors"] == []
