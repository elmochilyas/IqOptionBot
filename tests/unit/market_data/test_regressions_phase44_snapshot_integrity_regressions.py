"""Fail-closed snapshot integrity and genuine-coverage regressions (Phase 44).

These tests lock down the final acceptance rules for PR #61:

1. A caller-supplied integrity report is never trusted without an internal
   authenticity proof and a current durable-state binding.
2. Every receipt-bound parquet file must be present and exactly bound by path
   and hash before any snapshot is registered.
3. Corrupted, synthetic, OTC, or otherwise ineligible snapshots cannot be
   counted as genuine research coverage.
4. Genuine market evidence collected from a demo/practice account remains
   eligible coverage.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from iqoption_research.market_data.collection_engine import (
    CollectionCheckpoint,
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from iqoption_research.research.prospective import digest
from tests.unit.market_data.test_tampering import _auth_candles

START_UTC = "2026-11-02T10:00:00+00:00"
END_UTC = "2026-11-02T10:10:00+00:00"
START_TS = int(datetime.fromisoformat(START_UTC).timestamp())
END_TS = int(datetime.fromisoformat(END_UTC).timestamp())


def _setup_job(
    tmp_path: Path,
    *,
    end_utc: str = END_UTC,
    batches: int = 1,
    symbol: str = "EURUSD",
) -> tuple[HistoricalCollectionCoordinator, str, list[Path]]:
    """Create a completed genuine job with one or two receipt-bound parquet files."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol=symbol,
        start_utc=START_UTC,
        end_utc=end_utc,
        timeframe_seconds=60,
    )
    parquet_paths: list[Path] = []
    for batch in range(batches):
        batch_start = START_TS + batch * 10 * 60
        receipt = coord.record_collected_batch(
            job_id,
            _auth_candles(10, batch_start, symbol=symbol),
            is_synthetic=False,
        )
        parquet_paths.append(tmp_path / receipt.parquet_relative_path)

    ckpt_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.json"
    ckpt = CollectionCheckpoint.load(ckpt_path)
    ckpt.status = CollectionStatus.COMPLETED_VERIFIED.value
    ckpt.completed_interval_indices = list(range(len(ckpt.missing_intervals)))
    ckpt.save(ckpt_path)
    return coord, job_id, parquet_paths


def _snapshot_dirs(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "data/research_datasets").glob("rds_*"))


def _rewrite_manifest(path: Path, mutate: Any) -> None:
    """Change a manifest while keeping its integrity digest internally consistent."""
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    clean = {key: value for key, value in data.items() if key != "manifest_sha256"}
    data["manifest_sha256"] = digest(clean)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _missing_count(coord: HistoricalCollectionCoordinator, end_ts: int) -> int:
    return len(coord.inspect_coverage("EURUSD", START_TS, end_ts, 60))


def test_modified_parquet_after_verification_forces_fresh_verification(tmp_path: Path) -> None:
    """A verified report cannot authorize a snapshot after its bound file changed."""
    coord, job_id, parquet_paths = _setup_job(tmp_path)
    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is True

    original = parquet_paths[0].read_bytes()
    parquet_paths[0].write_bytes(original + b"\x00")

    with pytest.raises(ValueError, match=r"failed integrity verification|stale"):
        coord.create_snapshot_for_job(job_id, integrity=report)

    assert _snapshot_dirs(tmp_path) == []


def test_stale_checkpoint_after_verification_forces_fresh_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale report is discarded; genuine evidence is re-verified before publication."""
    coord, job_id, _parquet_paths = _setup_job(tmp_path)
    report = coord.verify_job_integrity(job_id)
    ckpt_path = tmp_path / "data/collection_checkpoints" / f"{job_id}.json"
    ckpt = CollectionCheckpoint.load(ckpt_path)
    ckpt.total_candles_collected += 1
    ckpt.save(ckpt_path)

    calls: list[str] = []
    original_verify = coord.verify_job_integrity

    def counting_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("verify")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(coord, "verify_job_integrity", counting_verify)

    snapshot = coord.create_snapshot_for_job(job_id, integrity=report)

    assert calls == ["verify"]
    assert snapshot.manifest.total_candles_retained == 10


def test_stale_receipt_after_verification_rejects_snapshot(tmp_path: Path) -> None:
    """Receipt edits invalidate a report even when the parquet bytes are unchanged."""
    coord, job_id, _parquet_paths = _setup_job(tmp_path)
    report = coord.verify_job_integrity(job_id)
    receipt_path = sorted((tmp_path / "data/collection_receipts").glob(f"rcpt_{job_id}_*.json"))[0]
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    data["created_at_utc"] = "NOT_A_TIMESTAMP"
    receipt_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match=r"stale|failed integrity verification|invalid"):
        coord.create_snapshot_for_job(job_id, integrity=report)

    assert _snapshot_dirs(tmp_path) == []


def test_forged_verified_report_is_never_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copied fingerprint with no internal proof forces fresh verification."""
    coord, job_id, _parquet_paths = _setup_job(tmp_path)
    legitimate = coord.verify_job_integrity(job_id)
    forged = dict(legitimate)
    forged["verification_token"] = "f" * 32
    assert forged["snapshot_state_fingerprint"] == legitimate["snapshot_state_fingerprint"]

    calls: list[str] = []
    original_verify = coord.verify_job_integrity

    def counting_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("verify")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(coord, "verify_job_integrity", counting_verify)

    snapshot = coord.create_snapshot_for_job(job_id, integrity=forged)

    assert calls == ["verify"]
    assert snapshot.manifest.total_candles_retained == 10


def test_swapped_path_hash_pairs_are_rejected(tmp_path: Path) -> None:
    """Path-to-hash bindings are exact; swapping valid hashes between files is fraud."""
    coord, job_id, parquet_paths = _setup_job(
        tmp_path, end_utc="2026-11-02T10:20:00+00:00", batches=2
    )
    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is True

    receipt_paths = sorted((tmp_path / "data/collection_receipts").glob(f"rcpt_{job_id}_*.json"))
    assert len(receipt_paths) == 2
    first = json.loads(receipt_paths[0].read_text(encoding="utf-8"))
    second = json.loads(receipt_paths[1].read_text(encoding="utf-8"))
    first["parquet_relative_path"], second["parquet_relative_path"] = (
        second["parquet_relative_path"],
        first["parquet_relative_path"],
    )
    receipt_paths[0].write_text(json.dumps(first, indent=2), encoding="utf-8")
    receipt_paths[1].write_text(json.dumps(second, indent=2), encoding="utf-8")

    with pytest.raises(ValueError):
        coord.create_snapshot_for_job(job_id, integrity=report)

    assert len(parquet_paths) == 2
    assert _snapshot_dirs(tmp_path) == []


def test_missing_expected_source_never_registers_partial_snapshot(tmp_path: Path) -> None:
    """Deleting one receipt-bound source blocks the whole snapshot."""
    coord, job_id, parquet_paths = _setup_job(
        tmp_path, end_utc="2026-11-02T10:20:00+00:00", batches=2
    )
    report = coord.verify_job_integrity(job_id)
    assert report["verified"] is True

    parquet_paths[1].unlink()

    with pytest.raises(ValueError):
        coord.create_snapshot_for_job(job_id, integrity=report)

    assert _snapshot_dirs(tmp_path) == []


def test_synthetic_job_never_produces_genuine_snapshot(tmp_path: Path) -> None:
    """Synthetic test evidence is not publishable as authentic research data."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    job_id, _, _ = coord.plan_collection(
        symbol="EURUSD",
        start_utc=START_UTC,
        end_utc=END_UTC,
        timeframe_seconds=60,
        is_synthetic=True,
    )
    coord.record_collected_batch(job_id, _auth_candles(10, START_TS), is_synthetic=True)

    with pytest.raises(ValueError, match="synthetic test evidence"):
        coord.create_snapshot_for_job(job_id)

    assert _snapshot_dirs(tmp_path) == []


def test_corrupted_snapshot_is_not_counted_as_coverage(tmp_path: Path) -> None:
    """A checksum-invalid snapshot leaves its bars reported as missing."""
    coord, job_id, _ = _setup_job(tmp_path)
    snapshot = coord.create_snapshot_for_job(job_id)
    assert _missing_count(coord, END_TS) == 0

    path = snapshot.parquet_path
    path.write_bytes(path.read_bytes()[:-1] + b"\x00")

    assert _missing_count(coord, END_TS) > 0


def test_synthetic_snapshot_is_not_counted_as_coverage(tmp_path: Path) -> None:
    """A synthetic-labelled snapshot is ineligible even when its checksums are valid."""
    coord, job_id, _ = _setup_job(tmp_path)
    snapshot = coord.create_snapshot_for_job(job_id)
    assert _missing_count(coord, END_TS) == 0

    _rewrite_manifest(
        snapshot.manifest_path,
        lambda data: data.update({
            "provenance_breakdown": {"SYNTHETIC_TEST_FIXTURE": 10},
            "source_files": [
                dict(source, provenance="SYNTHETIC_TEST_FIXTURE") for source in data["source_files"]
            ],
        }),
    )

    assert _missing_count(coord, END_TS) > 0


def test_otc_snapshot_is_not_counted_as_coverage(tmp_path: Path) -> None:
    """An OTC-labelled snapshot cannot substitute for genuine EURUSD coverage."""
    coord, job_id, _ = _setup_job(tmp_path)
    snapshot = coord.create_snapshot_for_job(job_id)
    assert _missing_count(coord, END_TS) == 0

    _rewrite_manifest(snapshot.manifest_path, lambda data: data.update({"is_otc": True}))

    assert _missing_count(coord, END_TS) > 0


def test_demo_account_snapshot_remains_genuine_coverage(tmp_path: Path) -> None:
    """Demo/practice-account market evidence is authentic, not synthetic fixture data."""
    coord, job_id, _ = _setup_job(tmp_path)
    snapshot = coord.create_snapshot_for_job(job_id)
    assert _missing_count(coord, END_TS) == 0

    _rewrite_manifest(
        snapshot.manifest_path,
        lambda data: data.update({"wallet_classification": "SIMULATED_DEMO_ACCOUNT"}),
    )

    assert _missing_count(coord, END_TS) == 0


def test_otc_symbol_snapshot_is_not_counted_as_coverage(tmp_path: Path) -> None:
    """An OTC symbol is ineligible even when the manifest has no explicit is_otc flag."""
    coord, job_id, _ = _setup_job(tmp_path)
    snapshot = coord.create_snapshot_for_job(job_id)
    assert _missing_count(coord, END_TS) == 0

    _rewrite_manifest(snapshot.manifest_path, lambda data: data.update({"symbol": "EURUSD-OTC"}))

    assert _missing_count(coord, END_TS) > 0
