"""Unit tests for Phase 44 cryptographic receipt verification and tamper detection.

Tests:
1. Valid job passes cryptographic integrity verification.
2. Altered candle price in Parquet (valid Parquet syntax, tampered content) is caught by candle content digest.
3. Modified receipt parquet_sha256 is detected.
4. Deleted parquet file fails integrity verification.
5. Injected synthetic receipt into genuine job is rejected.
6. Symbol or timeframe mismatch in receipt fails integrity verification.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from iqoption_research.market_data.collection_engine import (
    CollectionCheckpoint,
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.market_data.parquet_storage import _compute_file_sha256
from iqoption_research.models.market import Candle
from tests.fixtures.synthetic_market import create_deterministic_synthetic_candles


def _auth_candles(count: int, start_ts: int, symbol: str = "EURUSD") -> list[Candle]:
    synthetic = create_deterministic_synthetic_candles(
        count=count, start_ts=start_ts, symbol=symbol
    )
    return [
        Candle(
            active_id=InstrumentRegistry.resolve_active_id(symbol),
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


def _make_auth_candles(count: int, start_ts: int) -> list[Candle]:
    return _auth_candles(count, start_ts)


def _setup_verified_job(tmp_path: Path) -> tuple[HistoricalCollectionCoordinator, str, Path, Path]:
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-02T10:00:00+00:00",
        end_utc="2026-11-02T10:10:00+00:00",
        timeframe_seconds=60,
    )
    start_ts = int(datetime.fromisoformat("2026-11-02T10:00:00+00:00").timestamp())
    candles = _make_auth_candles(10, start_ts)
    rcpt = coord.record_collected_batch(job_id, candles, is_synthetic=False)

    ckpt_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.json"
    ckpt = CollectionCheckpoint.load(ckpt_path)
    ckpt.status = CollectionStatus.COMPLETED_VERIFIED.value
    ckpt.completed_interval_indices = [0]
    ckpt.save(ckpt_path)

    rcpt_path = tmp_path / "data/collection_receipts" / f"{rcpt.receipt_id}.json"
    pq_path = tmp_path / rcpt.parquet_relative_path

    return coord, job_id, rcpt_path, pq_path


def test_tampered_candle_price_detected(tmp_path: Path) -> None:
    """Modifying a candle's price in Parquet file fails integrity verification."""
    coord, job_id, _rcpt_path, pq_path = _setup_verified_job(tmp_path)

    # Sanity check: untampered job passes
    res_clean = coord.verify_job_integrity(job_id)
    assert res_clean["verified"] is True

    # Tamper with candle close price in the parquet file
    table = pq.read_table(pq_path)
    df = table.to_pandas()
    df.loc[0, "close"] = df.loc[0, "close"] + 0.00500  # Alter close price
    tampered_table = pa.Table.from_pandas(df)
    pq.write_table(tampered_table, pq_path)

    # Re-verify: must fail
    res_tampered = coord.verify_job_integrity(job_id)
    assert res_tampered["verified"] is False
    assert any("checksum altered" in e or "mismatch" in e for e in res_tampered.get("errors", []))


def test_tampered_receipt_checksum_detected(tmp_path: Path) -> None:
    """Modifying the parquet_sha256 recorded in the receipt fails integrity verification."""
    coord, job_id, rcpt_path, _ = _setup_verified_job(tmp_path)

    # Tamper with receipt file
    rdata = json.loads(rcpt_path.read_text(encoding="utf-8"))
    rdata["parquet_sha256"] = "0" * 64  # Fake checksum
    rcpt_path.write_text(json.dumps(rdata), encoding="utf-8")

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert any("checksum altered" in e for e in res.get("errors", []))


def test_tampered_candle_digest_in_receipt_detected(tmp_path: Path) -> None:
    """Modifying the candle_content_sha256 recorded in the receipt fails integrity verification."""
    coord, job_id, rcpt_path, _ = _setup_verified_job(tmp_path)

    rdata = json.loads(rcpt_path.read_text(encoding="utf-8"))
    rdata["candle_content_sha256"] = "f" * 64
    rcpt_path.write_text(json.dumps(rdata), encoding="utf-8")

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert any("candle content digest mismatch" in e for e in res.get("errors", []))


def test_deleted_parquet_file_fails_integrity(tmp_path: Path) -> None:
    """Deleting the referenced parquet file causes integrity check to fail."""
    coord, job_id, _, pq_path = _setup_verified_job(tmp_path)

    pq_path.unlink()

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert any("parquet file not found" in e for e in res.get("errors", []))


def test_synthetic_receipt_cannot_satisfy_genuine_job(tmp_path: Path) -> None:
    """A receipt flagged as SYNTHETIC_TEST_FIXTURE cannot satisfy genuine job coverage."""
    coord, job_id, rcpt_path, _ = _setup_verified_job(tmp_path)

    rdata = json.loads(rcpt_path.read_text(encoding="utf-8"))
    rdata["source_provenance"] = "SYNTHETIC_TEST_FIXTURE"
    rcpt_path.write_text(json.dumps(rdata), encoding="utf-8")

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert any(
        "synthetic receipt cannot satisfy genuine coverage" in e for e in res.get("errors", [])
    )


def test_missing_candle_content_digest_rejected(tmp_path: Path) -> None:
    """A modern receipt without a canonical content digest is never trusted."""
    coord, job_id, rcpt_path, _ = _setup_verified_job(tmp_path)
    rdata = json.loads(rcpt_path.read_text(encoding="utf-8"))
    rdata.pop("candle_content_sha256", None)
    rcpt_path.write_text(json.dumps(rdata, indent=2), encoding="utf-8")

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert any("missing candle content digest" in e for e in res["errors"])


def test_duplicate_receipt_timestamps_detected(tmp_path: Path) -> None:
    """Two receipts claiming the same candles cannot both count as coverage."""
    coord, job_id, rcpt_path, _ = _setup_verified_job(tmp_path)
    rdata = json.loads(rcpt_path.read_text(encoding="utf-8"))
    duplicate_id = f"{rcpt_path.stem}_duplicate"
    rdata["receipt_id"] = duplicate_id
    duplicate_path = rcpt_path.with_name(f"{duplicate_id}.json")
    duplicate_path.write_text(json.dumps(rdata, indent=2), encoding="utf-8")

    ckpt_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.json"
    ckpt = CollectionCheckpoint.load(ckpt_path)
    ckpt.receipt_ids.append(duplicate_id)
    ckpt.save(ckpt_path)

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert any("Duplicate verified candle" in e for e in res["errors"])


def test_substituted_parquet_with_matching_hash_still_fails(tmp_path: Path) -> None:
    """A receipt is bound to exact candles, not merely to any valid checksum."""
    coord, job_id, rcpt_path, _ = _setup_verified_job(tmp_path)

    other_job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-02T10:20:00+00:00",
        end_utc="2026-11-02T10:30:00+00:00",
        timeframe_seconds=60,
    )
    other_start_ts = int(datetime.fromisoformat("2026-11-02T10:20:00+00:00").timestamp())
    other_receipt = coord.record_collected_batch(
        other_job_id, _make_auth_candles(10, other_start_ts), is_synthetic=False
    )
    other_parquet = tmp_path / other_receipt.parquet_relative_path

    rdata = json.loads(rcpt_path.read_text(encoding="utf-8"))
    rdata["parquet_relative_path"] = str(
        other_parquet.resolve().relative_to(tmp_path.resolve())
    ).replace("\\", "/")
    rdata["parquet_sha256"] = _compute_file_sha256(other_parquet)
    rcpt_path.write_text(json.dumps(rdata, indent=2), encoding="utf-8")

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert any("candles missing from parquet" in e for e in res["errors"])


def test_synthetic_job_rejects_authentic_receipt(tmp_path: Path) -> None:
    """Synthetic and genuine provenance cannot be mixed in either direction."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-02T10:00:00+00:00",
        end_utc="2026-11-02T10:10:00+00:00",
        timeframe_seconds=60,
        is_synthetic=True,
    )
    start_ts = int(datetime.fromisoformat("2026-11-02T10:00:00+00:00").timestamp())
    coord.record_collected_batch(job_id, _make_auth_candles(10, start_ts), is_synthetic=False)
    ckpt_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.json"
    ckpt = CollectionCheckpoint.load(ckpt_path)
    ckpt.status = CollectionStatus.COMPLETED_VERIFIED.value
    ckpt.completed_interval_indices = [0]
    ckpt.save(ckpt_path)

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert any("synthetic job receipt provenance mismatch" in e for e in res["errors"])


def test_unknown_schedule_missing_bars_are_not_verified(tmp_path: Path) -> None:
    """Unverified broker schedules cannot excuse missing candles."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, missing, _ = coord.plan_collection(
        symbol="EURUSD-OTC",
        start_utc="2026-11-07T10:00:00+00:00",
        end_utc="2026-11-07T10:10:00+00:00",
        timeframe_seconds=60,
    )
    assert len(missing) == 1
    start_ts = int(datetime.fromisoformat("2026-11-07T10:00:00+00:00").timestamp())
    coord.record_collected_batch(
        job_id,
        _auth_candles(5, start_ts, symbol="EURUSD_OTC"),
        is_synthetic=False,
    )
    ckpt_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.json"
    ckpt = CollectionCheckpoint.load(ckpt_path)
    ckpt.status = CollectionStatus.COMPLETED_VERIFIED.value
    ckpt.completed_interval_indices = [0]
    ckpt.save(ckpt_path)

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert res["coverage_gaps_count"] >= 1
    assert res["unique_verified_candles"] == 5


def test_completion_flag_without_receipts_never_verifies(tmp_path: Path) -> None:
    """Checkpoint completion flags are not proof of verified coverage."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-02T10:00:00+00:00",
        end_utc="2026-11-02T10:10:00+00:00",
        timeframe_seconds=60,
    )
    ckpt_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.json"
    ckpt = CollectionCheckpoint.load(ckpt_path)
    ckpt.status = CollectionStatus.COMPLETED_VERIFIED.value
    ckpt.completed_interval_indices = [0]
    ckpt.save(ckpt_path)

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert res["receipt_count"] == 0
    assert "No cryptographic receipts" in res["reason"]


def test_snapshot_whitelists_only_receipt_bound_parquet(tmp_path: Path) -> None:
    """Snapshot creation never silently includes unrelated parquet files."""
    coord, job_id, rcpt_path, _ = _setup_verified_job(tmp_path)
    first_start_ts = int(datetime.fromisoformat("2026-11-02T10:00:00+00:00").timestamp())

    other_job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc="2026-11-02T10:20:00+00:00",
        end_utc="2026-11-02T10:30:00+00:00",
        timeframe_seconds=60,
    )
    other_start_ts = int(datetime.fromisoformat("2026-11-02T10:20:00+00:00").timestamp())
    coord.record_collected_batch(
        other_job_id, _make_auth_candles(10, other_start_ts), is_synthetic=False
    )

    assert rcpt_path.is_file()
    snapshot = coord.create_snapshot_for_job(job_id)
    candles = snapshot.load_candles()
    assert len(candles) == 10
    assert {c.from_ts for c in candles} == set(range(first_start_ts, first_start_ts + 10 * 60, 60))


def test_legacy_receipt_without_new_fields_is_still_verifiable(tmp_path: Path) -> None:
    """Legacy receipts verify with their recorded provenance, without inventing metadata."""
    coord, job_id, rcpt_path, _ = _setup_verified_job(tmp_path)
    rdata = json.loads(rcpt_path.read_text(encoding="utf-8"))
    for field in (
        "schema_version",
        "commit_state",
        "batch_id",
        "parquet_relative_path",
        "candle_content_sha256",
        "account_id",
        "account_label",
    ):
        rdata.pop(field, None)
    rcpt_path.write_text(json.dumps(rdata, indent=2), encoding="utf-8")

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is True
    assert res["legacy_receipt_count"] == 1
    assert res["provenance_status"] == ["AUTHENTIC_BROKER_COLLECTION"]


def test_modern_receipt_requires_explicit_provenance(tmp_path: Path) -> None:
    """A schema-v2 receipt cannot omit provenance and inherit legacy ambiguity."""
    coord, job_id, rcpt_path, _ = _setup_verified_job(tmp_path)
    rdata = json.loads(rcpt_path.read_text(encoding="utf-8"))
    rdata.pop("source_provenance", None)
    rcpt_path.write_text(json.dumps(rdata, indent=2), encoding="utf-8")

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is False
    assert res["legacy_receipt_count"] == 0
    assert any("requires explicit broker provenance" in e for e in res["errors"])


def test_legacy_receipt_reports_unspecified_provenance(tmp_path: Path) -> None:
    """Legacy provenance remains unspecified rather than being silently promoted."""
    coord, job_id, rcpt_path, _ = _setup_verified_job(tmp_path)
    rdata = json.loads(rcpt_path.read_text(encoding="utf-8"))
    for field in (
        "schema_version",
        "commit_state",
        "batch_id",
        "parquet_relative_path",
        "candle_content_sha256",
        "account_id",
        "account_label",
        "source_provenance",
    ):
        rdata.pop(field, None)
    rcpt_path.write_text(json.dumps(rdata, indent=2), encoding="utf-8")

    res = coord.verify_job_integrity(job_id)
    assert res["verified"] is True
    assert res["legacy_receipt_count"] == 1
    assert res["provenance_status"] == ["LEGACY_PROVENANCE_UNSPECIFIED"]
