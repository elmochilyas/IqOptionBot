"""Crash-recovery regressions for the two-phase collection commit protocol.

The durable commit order is:

1. Parquet file is published atomically.
2. A PENDING receipt is written atomically.
3. The receipt ID is registered in the checkpoint atomically.
4. The receipt is finalized to COMMITTED atomically.

Every interrupted prefix of that order must either recover safely or fail
closed. No orphan parquet file may ever become verified coverage.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from iqoption_research.market_data.collection_engine import (
    RECEIPT_COMMIT_PENDING,
    CollectionCheckpoint,
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from iqoption_research.models.market import Candle
from tests.fixtures.synthetic_market import create_deterministic_synthetic_candles


def _auth_candles(count: int, start_ts: int, symbol: str = "EURUSD") -> list[Candle]:
    synthetic = create_deterministic_synthetic_candles(
        count=count, start_ts=start_ts, symbol=symbol
    )
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


def _setup_job(tmp_path: Path) -> tuple[HistoricalCollectionCoordinator, str, Path, Path]:
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-04T10:00:00+00:00",
        end_utc="2026-11-04T10:10:00+00:00",
        timeframe_seconds=60,
    )
    start_ts = int(datetime.fromisoformat("2026-11-04T10:00:00+00:00").timestamp())
    receipt = coord.record_collected_batch(job_id, _auth_candles(10, start_ts), is_synthetic=False)
    ckpt_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.json"
    rcpt_path = tmp_path / "data/collection_receipts" / f"{receipt.receipt_id}.json"
    return coord, job_id, ckpt_path, rcpt_path


def _set_receipt_pending(rcpt_path: Path) -> dict:
    data = json.loads(rcpt_path.read_text(encoding="utf-8"))
    data["commit_state"] = RECEIPT_COMMIT_PENDING
    rcpt_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _unregister_receipt(ckpt_path: Path) -> None:
    ckpt = CollectionCheckpoint.load(ckpt_path)
    ckpt.receipt_ids = []
    ckpt.total_candles_collected = 0
    ckpt.save(ckpt_path)


def test_crash_after_receipt_registration_is_finalized(tmp_path: Path) -> None:
    """A PENDING receipt already registered in the checkpoint is safely finalized."""
    coord, job_id, ckpt_path, rcpt_path = _setup_job(tmp_path)
    ckpt = CollectionCheckpoint.load(ckpt_path)
    ckpt.status = CollectionStatus.COMPLETED_VERIFIED.value
    ckpt.completed_interval_indices = [0]
    ckpt.save(ckpt_path)
    _set_receipt_pending(rcpt_path)

    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is True
    assert json.loads(rcpt_path.read_text(encoding="utf-8"))["commit_state"] == "COMMITTED"


def test_crash_before_receipt_registration_adopts_valid_receipt(tmp_path: Path) -> None:
    """A valid PENDING receipt is adopted when no cancellation intervened."""
    coord, job_id, ckpt_path, rcpt_path = _setup_job(tmp_path)
    receipt_id = rcpt_path.stem
    _set_receipt_pending(rcpt_path)
    _unregister_receipt(ckpt_path)

    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is True
    assert receipt_id in CollectionCheckpoint.load(ckpt_path).receipt_ids
    assert json.loads(rcpt_path.read_text(encoding="utf-8"))["commit_state"] == "COMMITTED"


def test_crash_before_receipt_creation_is_not_verified(tmp_path: Path) -> None:
    """An orphan parquet file without a receipt can never become verified coverage."""
    coord, job_id, _ckpt_path, rcpt_path = _setup_job(tmp_path)
    rcpt_path.unlink()

    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is False
    assert report["receipt_count"] == 0
    assert report["unique_verified_candles"] == 0

    reconciled = coord.reconcile_job_state(job_id)
    assert reconciled.total_candles_collected == 0
    assert reconciled.receipt_ids == []
    assert reconciled.status != CollectionStatus.COMPLETED_VERIFIED.value


def test_committed_receipt_missing_checkpoint_registration_fails_closed(
    tmp_path: Path,
) -> None:
    """A COMMITTED receipt absent from checkpoint registration is not silently adopted."""
    coord, job_id, ckpt_path, rcpt_path = _setup_job(tmp_path)
    receipt_id = rcpt_path.stem
    _unregister_receipt(ckpt_path)

    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is False
    assert any("not registered" in error for error in report["errors"])
    assert receipt_id not in CollectionCheckpoint.load(ckpt_path).receipt_ids


def test_cancellation_prevents_pending_receipt_adoption(tmp_path: Path) -> None:
    """A cancel signal blocks adoption of a receipt that was never committed."""
    coord, job_id, ckpt_path, rcpt_path = _setup_job(tmp_path)
    _set_receipt_pending(rcpt_path)
    _unregister_receipt(ckpt_path)
    (tmp_path / "data/collection_checkpoints" / f"{job_id}.cancel").write_text(
        datetime.now(UTC).isoformat(), encoding="utf-8"
    )

    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is False
    assert report["receipt_count"] == 0
    assert json.loads(rcpt_path.read_text(encoding="utf-8"))["commit_state"] == (
        RECEIPT_COMMIT_PENDING
    )


def test_verify_does_not_recover_while_another_worker_holds_lock(tmp_path: Path) -> None:
    """Read-only verification must not adopt receipts owned by an active worker."""
    coord, job_id, ckpt_path, rcpt_path = _setup_job(tmp_path)
    receipt_id = rcpt_path.stem
    _set_receipt_pending(rcpt_path)
    _unregister_receipt(ckpt_path)
    lock_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.lock"
    lock_path.write_text(
        json.dumps({
            "job_id": job_id,
            "pid": os.getpid(),
            "owner_token": "foreign-worker-token",
        }),
        encoding="utf-8",
    )

    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is False
    assert receipt_id not in CollectionCheckpoint.load(ckpt_path).receipt_ids
    assert json.loads(rcpt_path.read_text(encoding="utf-8"))["commit_state"] == (
        RECEIPT_COMMIT_PENDING
    )


def test_verify_recovers_pending_receipt_behind_stale_dead_lock(tmp_path: Path) -> None:
    """A crashed foreign lock must not block ordinary offline crash recovery."""
    coord, job_id, ckpt_path, rcpt_path = _setup_job(tmp_path)
    receipt_id = rcpt_path.stem
    _set_receipt_pending(rcpt_path)
    _unregister_receipt(ckpt_path)
    lock_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.lock"
    lock_path.write_text(
        json.dumps({
            "job_id": job_id,
            "pid": 999_999,
            "process_create_time": 0,
            "owner_token": "crashed-worker-token",
        }),
        encoding="utf-8",
    )

    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is True
    assert receipt_id in CollectionCheckpoint.load(ckpt_path).receipt_ids
    assert json.loads(rcpt_path.read_text(encoding="utf-8"))["commit_state"] == "COMMITTED"


def test_pending_receipt_adoption_revalidates_parquet(tmp_path: Path) -> None:
    """Recovery never adopts a pending receipt whose bound parquet changed."""
    coord, job_id, ckpt_path, rcpt_path = _setup_job(tmp_path)
    rdata = _set_receipt_pending(rcpt_path)
    _unregister_receipt(ckpt_path)

    parquet_path = tmp_path / rdata["parquet_relative_path"]
    parquet_path.write_bytes(b"tampered parquet payload")

    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is False
    assert rcpt_path.stem not in CollectionCheckpoint.load(ckpt_path).receipt_ids
    assert json.loads(rcpt_path.read_text(encoding="utf-8"))["commit_state"] == (
        RECEIPT_COMMIT_PENDING
    )
