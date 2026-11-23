"""Unit tests for Phase 42 immutable research data partitions.

Validates:
1. 4-way chronological partitioning (Train 60%, Val 15%, Cal 10%, Test 15%).
2. Purge and embargo boundary buffer integrity.
3. Manifest cryptographic self-digest (SHA-256).
4. Manifest serialization and disk persistence round-trip.
5. Partition candle extraction and integrity verification.
6. Detection of tampered manifests or corrupted candle sequences.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iqoption_research.ml.partitions import (
    PartitionError,
    PartitionIntegrityError,
    PartitionManifest,
    create_partition_manifest,
    extract_partition_candles,
    verify_partition_integrity,
)
from iqoption_research.models.market import Candle, CandleSize


def _make_candle_sequence(count: int, start_ts: int = 1700000000) -> list[Candle]:
    """Generate sequential 1-minute candles."""
    candles = []
    base_price = 1.0850
    for i in range(count):
        ts = start_ts + (i * 60)
        c_val = base_price + (0.0001 if i % 2 == 0 else -0.0001)
        candles.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=ts,
                to_ts=ts + 60,
                at_ts=ts * 1_000_000_000,
                open=base_price,
                high=base_price + 0.0005,
                low=base_price - 0.0005,
                close=c_val,
                volume=100,
                bid=c_val - 0.00005,
                ask=c_val + 0.00005,
            )
        )
    return candles


def test_create_partition_manifest_proportions_and_buffers() -> None:
    """Test standard 4-way chronological partitioning and buffer enforcement."""
    candles = _make_candle_sequence(1000, start_ts=1700000000)
    manifest = create_partition_manifest(
        candles=candles,
        snapshot_id="test_snapshot_01",
        snapshot_sha256="abc123snapsha",
        train_ratio=0.60,
        val_ratio=0.15,
        cal_ratio=0.10,
        test_ratio=0.15,
        purging_horizon_bars=1,
        embargo_bars=5,
    )

    assert manifest.snapshot_id == "test_snapshot_01"
    assert manifest.total_candles == 1000
    assert manifest.purging_horizon_bars == 1
    assert manifest.embargo_bars == 5

    # Check partitions are populated chronologically
    tr = manifest.train_partition
    val = manifest.val_partition
    cal = manifest.calibration_partition
    te = manifest.protected_test_partition

    assert tr.from_ts < tr.to_ts
    assert tr.to_ts <= val.from_ts
    assert val.to_ts <= cal.from_ts
    assert cal.to_ts <= te.from_ts

    # Verify purge and embargo buffer between train and val
    buffer_seconds = (manifest.purging_horizon_bars + manifest.embargo_bars) * 60
    assert val.from_ts - tr.to_ts >= buffer_seconds
    assert cal.from_ts - val.to_ts >= buffer_seconds
    assert te.from_ts - cal.to_ts >= buffer_seconds

    # Verify self-digest is populated
    assert len(manifest.manifest_sha256) == 64


def test_create_partition_manifest_ratio_validation() -> None:
    """Verify that invalid partition ratios raise PartitionError."""
    candles = _make_candle_sequence(1000)
    with pytest.raises(PartitionError, match=r"must sum to 1\.0"):
        create_partition_manifest(
            candles=candles,
            snapshot_id="invalid_snap",
            snapshot_sha256="abc",
            train_ratio=0.50,
            val_ratio=0.20,
            cal_ratio=0.10,
            test_ratio=0.10,  # Sum is 0.90 != 1.0
        )


def test_create_partition_manifest_insufficient_samples() -> None:
    """Verify that fewer than 1000 candles raises PartitionError."""
    candles = _make_candle_sequence(500)
    with pytest.raises(PartitionError, match="Insufficient candles"):
        create_partition_manifest(
            candles=candles,
            snapshot_id="too_few_snap",
            snapshot_sha256="abc",
        )


def test_manifest_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Verify save and load round-trip with cryptographic self-digest verification."""
    candles = _make_candle_sequence(1000)
    manifest = create_partition_manifest(
        candles=candles,
        snapshot_id="roundtrip_snap",
        snapshot_sha256="roundtrip123",
    )

    file_dest = tmp_path / "custom_manifest.json"
    saved_path = manifest.save(destination=file_dest)
    assert saved_path == file_dest
    assert file_dest.is_file()

    loaded = PartitionManifest.load(file_dest)
    assert loaded.manifest_id == manifest.manifest_id
    assert loaded.manifest_sha256 == manifest.manifest_sha256
    assert loaded.train_partition.candle_count == manifest.train_partition.candle_count

    # Also test directory destination
    dir_dest = tmp_path / "manifest_dir"
    saved_dir_path = manifest.save(destination=dir_dest)
    assert saved_dir_path == dir_dest / "manifest.json"
    loaded_dir = PartitionManifest.load(dir_dest)
    assert loaded_dir.manifest_id == manifest.manifest_id


def test_manifest_load_tamper_detection(tmp_path: Path) -> None:
    """Verify that any modification to saved manifest raises PartitionIntegrityError."""
    candles = _make_candle_sequence(1000)
    manifest = create_partition_manifest(
        candles=candles,
        snapshot_id="tamper_snap",
        snapshot_sha256="tamper123",
    )
    dest = tmp_path / "tampered_manifest.json"
    manifest.save(destination=dest)

    # Tamper with total_candles
    data = json.loads(dest.read_text(encoding="utf-8"))
    data["total_candles"] = 999999
    dest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PartitionIntegrityError, match="Manifest SHA-256 mismatch"):
        PartitionManifest.load(dest)


def test_extract_partition_candles_and_verify_integrity() -> None:
    """Verify partition candle extraction and leak-free verification."""
    candles = _make_candle_sequence(1200)
    manifest = create_partition_manifest(
        candles=candles,
        snapshot_id="integrity_snap",
        snapshot_sha256="int123",
    )

    extracted = extract_partition_candles(candles, manifest)
    assert len(extracted["train"]) == manifest.train_partition.candle_count
    assert len(extracted["val"]) == manifest.val_partition.candle_count
    assert len(extracted["calibration"]) == manifest.calibration_partition.candle_count
    assert len(extracted["protected_test"]) == manifest.protected_test_partition.candle_count

    ok, msg = verify_partition_integrity(manifest, candles)
    assert ok is True
    assert msg == "VERIFIED_LEAK_FREE"


def test_verify_partition_integrity_catches_missing_candle() -> None:
    """Verify that removing a candle causes verify_partition_integrity to fail."""
    candles = _make_candle_sequence(1000)
    manifest = create_partition_manifest(
        candles=candles,
        snapshot_id="corrupt_snap",
        snapshot_sha256="corrupt123",
    )

    # Pass candles with one candle omitted from train
    corrupted_candles = candles[1:]
    ok, msg = verify_partition_integrity(manifest, corrupted_candles)
    assert ok is False
    assert "count mismatch" in msg


def test_interval_overlap_logic() -> None:
    """Verify mathematical correctness of interval overlap: max(A1, B1) <= min(A2, B2)."""
    from iqoption_research.ml.partitions import (
        audit_holdout_independence,
        check_interval_overlap,
    )

    # Disjoint intervals
    assert check_interval_overlap(100, 200, 250, 300) is False
    assert check_interval_overlap(250, 300, 100, 200) is False

    # Overlapping intervals
    assert check_interval_overlap(100, 200, 150, 250) is True
    assert check_interval_overlap(150, 250, 100, 200) is True
    assert check_interval_overlap(100, 300, 150, 200) is True
    assert check_interval_overlap(150, 200, 100, 300) is True

    # Boundary touch (closed interval)
    assert check_interval_overlap(100, 200, 200, 300) is True

    # Audit historical holdout: November 12-20, 2026 (approx 1794500000 to 1795190400)
    # overlaps with Phase 41 registry (up to 1795190400)
    is_inspected, prov, notes = audit_holdout_independence(1794500000, 1795190000)
    assert is_inspected is True
    assert prov == "HISTORICAL_REUSED"
    assert "HISTORICAL_REUSED" in notes

    # Far future interval (year 2027+) is genuinely independent prospective
    is_inspected_fut, prov_fut, _notes_fut = audit_holdout_independence(1850000000, 1850100000)
    assert is_inspected_fut is False
    assert prov_fut == "GENUINELY_INDEPENDENT_PROSPECTIVE"

    # Synthetic fixture is classified as synthetic
    is_inspected_syn, prov_syn, _notes_syn = audit_holdout_independence(
        1850000000, 1850100000, is_synthetic=True
    )
    assert is_inspected_syn is True
    assert prov_syn == "SYNTHETIC_TEST_FIXTURE"
