"""Cross-job immutable storage deduplication and conflict quarantine."""

import hashlib
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from iqoption_research.market_data.collection_engine import HistoricalCollectionCoordinator
from iqoption_research.market_data.parquet_storage import ImmutableDatasetError
from tests.unit.test_phase44_collection_coordinator import _make_auth_candles


def test_cross_job_duplicates_and_conflicts(tmp_path):
    c = HistoricalCollectionCoordinator(tmp_path)
    start = int(datetime(2026, 11, 18, 10, tzinfo=UTC).timestamp())

    def plan():
        return c.plan_collection(
            symbol="EURUSD",
            start_utc=datetime.fromtimestamp(start, UTC).isoformat(),
            end_utc=datetime.fromtimestamp(start + 900, UTC).isoformat(),
            timeframe_seconds=60,
        )[0]

    job = plan()
    candles = _make_auth_candles(15, start)
    first = c.record_collected_batch(job_id=job, candles=candles, is_synthetic=False)
    checkpoint = c.checkpoints_dir / f"{job}.json"
    import json

    data = json.loads(checkpoint.read_text())
    second_job = job.replace("collect_", "collect_second_", 1)
    data.update(job_id=second_job, receipt_ids=[], total_candles_collected=0)
    (c.checkpoints_dir / f"{second_job}.json").write_text(json.dumps(data))
    second = c.record_collected_batch(job_id=second_job, candles=candles, is_synthetic=False)
    assert first.parquet_relative_path == second.parquet_relative_path
    path = tmp_path / first.parquet_relative_path
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    changed = [replace(x, volume=x.volume + 1) for x in candles]
    with pytest.raises(ImmutableDatasetError, match="quarantined"):
        c.record_collected_batch(job_id=second_job, candles=changed, is_synthetic=False)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    assert len(list((tmp_path / "data/datasets/quarantine").glob("*.parquet"))) == 1
    assert c.verify_job_integrity(job)["errors"] == []
    assert c.verify_job_integrity(second_job)["errors"] == []
