"""Read-only audit and isolated recovery of the owner's retained failed job."""

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from iqoption_research.market_data.collection_engine import (
    CollectionStatus,
    HistoricalCollectionCoordinator,
)


@pytest.mark.asyncio
async def test_retained_failed_job_recovery_in_copy(tmp_path):
    root = Path(__file__).resolve().parents[2]
    job = "collect_eurusd_20261124_152833_2b516905"
    checkpoint = root / "data/collection_checkpoints" / f"{job}.json"
    if not checkpoint.exists():
        pytest.skip("Owner local job is not distributed with source code")
    paths = [checkpoint, *sorted((root / "data/collection_receipts").glob(f"rcpt_{job}_*.json"))]
    for receipt in paths[1:]:
        paths.append(root / json.loads(receipt.read_text())["parquet_relative_path"])
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in paths:
        target = tmp_path / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    expected_candles = 3723 if len(paths) > 9 else 3603
    expected_receipts = 10 if len(paths) > 9 else 8
    coord = HistoricalCollectionCoordinator(tmp_path)
    report = coord.verify_job_integrity(job)
    assert report["unique_verified_candles"] == expected_candles
    assert report["receipt_count"] == expected_receipts
    assert report["errors"] == []
    receipts = {p.name: p.read_bytes() for p in coord.receipts_dir.glob("*.json")}

    class Broker:
        _settings = SimpleNamespace(max_candles=500)

        def assert_read_only(self):
            pass

        async def get_candles(self, *args, **kwargs):
            return SimpleNamespace(candle_list=[])

    result = await coord.resume_collection(job_id=job, client=Broker(), operator_authorized=True)
    assert result.status == CollectionStatus.PARTIALLY_COLLECTED
    assert coord.verify_job_integrity(job)["unique_verified_candles"] == expected_candles
    assert receipts == {p.name: p.read_bytes() for p in coord.receipts_dir.glob("*.json")}
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
