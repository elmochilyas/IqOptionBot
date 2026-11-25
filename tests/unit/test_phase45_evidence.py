"""Offline unit tests for Phase 45 genuine evidence consolidation.

Every artifact created here is a small synthetic fixture inside ``tmp_path``.
No repository data directory, broker, or network resource is read or written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from iqoption_research.market_data.collection_engine import (
    PROVENANCE_AUTHENTIC,
    RECEIPT_COMMITTED,
    RECEIPT_SCHEMA_VERSION,
    _compute_raw_payload_sha256,
)
from iqoption_research.market_data.parquet_storage import write_candles_parquet
from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
    compute_candle_content_digest,
)
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.phase45.evidence import (
    build_evidence_inventory,
    load_population_candles,
    verify_all_receipts,
)
from iqoption_research.research.prospective import digest

BOUNDARY_TS: int = DEFAULT_PROTECTED_HOLDOUT_START_TS
WEEKEND_GAP_FROM_TS: int = 1789765200
WEEKEND_GAP_TO_TS: int = 1789938060
SYMBOL: str = "EURUSD"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candles(
    start_ts: int,
    count: int,
    timeframe_seconds: int,
    *,
    base_price: float = 1.1000,
) -> list[Candle]:
    size = CandleSize(timeframe_seconds)
    result: list[Candle] = []
    for index in range(count):
        from_ts = start_ts + index * timeframe_seconds
        open_price = round(base_price + index * 0.0001, 6)
        close_price = round(open_price + 0.00005, 6)
        result.append(
            Candle(
                active_id=1,
                symbol=SYMBOL,
                size=size,
                from_ts=from_ts,
                to_ts=from_ts + timeframe_seconds,
                at_ts=from_ts * 1_000_000_000,
                open=open_price,
                high=round(close_price + 0.00002, 6),
                low=round(open_price - 0.00002, 6),
                close=close_price,
                volume=100 + index,
                bid=round(close_price - 0.00001, 6),
                ask=round(close_price + 0.00001, 6),
            )
        )
    return result


def _conflicting_candle(candle: Candle, delta: float = 0.0010) -> Candle:
    changed_close = round(candle.close + delta, 6)
    return replace(
        candle,
        close=changed_close,
        high=round(max(candle.high, changed_close) + 0.00002, 6),
    )


def _write_parquet(repo_root: Path, candles: list[Candle], relative_path: str) -> str:
    stage_root = repo_root / "_stage" / uuid4().hex
    staged = write_candles_parquet(
        candles,
        root_dir=stage_root,
        source="offline_fixture",
        is_synthetic=False,
        catalog=None,
        on_duplicate="raise",
    )
    target = repo_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(staged.parquet_path.read_bytes())
    return staged.sha256_checksum


def _write_snapshot(
    repo_root: Path,
    candles: list[Candle],
    *,
    conflicts: tuple[dict[str, Any], ...] = (),
) -> tuple[Path, Path]:
    snapshot_id = "rds_offline_eurusd_m1_fixture"
    parquet_relative = f"data/research_datasets/{snapshot_id}/candles.parquet"
    parquet_sha = _write_parquet(repo_root, candles, parquet_relative)
    manifest: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "created_at_utc": "2026-11-25T00:00:00+00:00",
        "symbol": SYMBOL,
        "candle_size_seconds": 60,
        "from_ts": candles[0].from_ts,
        "to_ts": candles[-1].to_ts,
        "total_candles_discovered": len(candles),
        "total_candles_retained": len(candles),
        "duplicate_boundaries_deduplicated": 0,
        "conflicts_quarantined_count": len(conflicts),
        "conflicts_quarantined": list(conflicts),
        "scheduled_closures_count": 0,
        "scheduled_closures": [],
        "unscheduled_gaps_count": 0,
        "unscheduled_gaps": [],
        "provenance_breakdown": {PROVENANCE_AUTHENTIC: len(candles)},
        "source_files": [
            {
                "source_id": "offline_fixture_source",
                "path": str(repo_root / parquet_relative),
                "sha256": parquet_sha,
                "candle_count": len(candles),
                "provenance": PROVENANCE_AUTHENTIC,
            }
        ],
        "sha256_checksum": parquet_sha,
        "is_synthetic": False,
        "manifest_sha256": "",
    }
    clean_manifest = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = digest(clean_manifest)
    manifest_path = repo_root / f"data/research_datasets/{snapshot_id}/manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path, repo_root / parquet_relative


def _write_receipt(
    repo_root: Path,
    candles: list[Candle],
    *,
    job_id: str,
) -> Path:
    timeframe_seconds = candles[0].size.value
    parquet_relative = f"data/offline_receipt_parquets/{job_id}.parquet"
    parquet_sha = _write_parquet(repo_root, candles, parquet_relative)
    content_sha = compute_candle_content_digest(candles)
    raw_sha = _compute_raw_payload_sha256(candles)
    receipt_id = f"rcpt_{job_id}_{candles[0].from_ts}_{candles[-1].to_ts}_{content_sha[:12]}"
    receipt: dict[str, Any] = {
        "receipt_id": receipt_id,
        "batch_id": receipt_id,
        "job_id": job_id,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "commit_state": RECEIPT_COMMITTED,
        "source_provenance": PROVENANCE_AUTHENTIC,
        "symbol": SYMBOL,
        "timeframe_seconds": timeframe_seconds,
        "from_ts": candles[0].from_ts,
        "to_ts": candles[-1].to_ts,
        "candle_count": len(candles),
        "first_candle_open": candles[0].open,
        "last_candle_close": candles[-1].close,
        "raw_payload_sha256": raw_sha,
        "parquet_sha256": parquet_sha,
        "candle_content_sha256": content_sha,
        "parquet_relative_path": parquet_relative,
    }
    receipts_dir = repo_root / "data/collection_receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"{receipt_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt_path


def _build_boundary_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path.resolve()
    m1 = _candles(BOUNDARY_TS - 180, 5, 60)
    _write_snapshot(repo_root, m1)
    _write_receipt(
        repo_root,
        _candles(BOUNDARY_TS - 600, 4, 300),
        job_id="offline_m5_boundary",
    )
    _write_receipt(
        repo_root,
        _candles(BOUNDARY_TS - 900, 3, 900),
        job_id="offline_m15_boundary",
    )
    return repo_root


def test_verify_all_receipts_re_verifies_every_binding(tmp_path: Path) -> None:
    repo_root = tmp_path.resolve()
    m5 = _candles(BOUNDARY_TS - 600, 3, 300)
    receipt_path = _write_receipt(repo_root, m5, job_id="offline_m5_verify")

    summary = verify_all_receipts(repo_root)
    assert set(summary) == {300}
    assert summary[300]["receipt_count"] == 1
    assert summary[300]["unique_candles"] == 3
    assert summary[300]["verification"] == "PASS"
    assert summary[300]["receipt_ids"] == (receipt_path.stem,)

    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    data["candle_count"] = 99
    receipt_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="candle count mismatch"):
        verify_all_receipts(repo_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "0"),
        ("commit_state", "PENDING"),
        ("source_provenance", "SYNTHETIC_FIXTURE"),
        ("timeframe_seconds", 60),
        ("symbol", "GBPUSD"),
        ("parquet_sha256", "0" * 64),
        ("candle_content_sha256", "0" * 64),
        ("raw_payload_sha256", "0" * 64),
    ],
)
def test_verify_all_receipts_fails_closed_on_tampered_evidence(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    repo_root = tmp_path.resolve()
    receipt_path = _write_receipt(
        repo_root,
        _candles(BOUNDARY_TS - 600, 2, 300),
        job_id=f"offline_tamper_{field}",
    )
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    data[field] = value
    receipt_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError):
        verify_all_receipts(repo_root)


def test_inventory_separates_boundary_touching_and_protected_candles(
    tmp_path: Path,
) -> None:
    repo_root = _build_boundary_repo(tmp_path)
    inventory = build_evidence_inventory(repo_root)

    m1 = inventory.populations[60]
    assert m1.full_unique_candles == 5
    assert m1.development_unique_candles == 3
    assert m1.protected_unique_candles == 2
    assert m1.development_latest_ts == BOUNDARY_TS
    assert m1.development_latest_utc.endswith("10:00:00+00:00")

    m5 = inventory.populations[300]
    assert m5.full_unique_candles == 4
    assert m5.development_unique_candles == 2
    assert m5.development_latest_ts == BOUNDARY_TS

    m15 = inventory.populations[900]
    assert m15.full_unique_candles == 3
    assert m15.development_unique_candles == 1
    assert m15.development_latest_ts == BOUNDARY_TS

    separation = inventory.separation
    assert separation["development_rule"] == "from_ts < boundary_ts"
    assert separation["to_ts_equals_boundary_allowed"] is True
    assert separation["to_ts_strictly_pre_boundary_required"] is False
    assert separation["label_maturity_invalidation"] == "max_label_maturity_ts"

    for timeframe, expected in ((60, 3), (300, 2), (900, 1)):
        candles = load_population_candles(
            repo_root,
            timeframe,
            development_only=True,
            boundary_ts=BOUNDARY_TS,
        )
        assert len(candles) == expected
        assert all(candle.from_ts < BOUNDARY_TS for candle in candles)
        assert candles[-1].to_ts == BOUNDARY_TS
        assert candles == sorted(candles, key=lambda candle: candle.from_ts)

    full_m1 = load_population_candles(repo_root, 60, development_only=False)
    assert len(full_m1) == 5
    assert any(candle.from_ts == BOUNDARY_TS for candle in full_m1)

    serialized = json.dumps(inventory.to_dict())
    assert str(repo_root) not in serialized
    assert "data/research_datasets/rds_offline_eurusd_m1_fixture/candles.parquet" in serialized


def test_schedule_aware_weekend_gap_is_not_a_missing_candle(tmp_path: Path) -> None:
    repo_root = tmp_path.resolve()
    candles = [
        *_candles(WEEKEND_GAP_FROM_TS - 60, 1, 60),
        *_candles(WEEKEND_GAP_TO_TS, 1, 60, base_price=1.1200),
    ]
    _write_snapshot(repo_root, candles)

    inventory = build_evidence_inventory(repo_root)
    m1 = inventory.populations[60]
    assert m1.continuity_segments == 2
    assert len(m1.missing_intervals) == 1
    gap = m1.missing_intervals[0]
    assert gap["from_ts"] == WEEKEND_GAP_FROM_TS
    assert gap["to_ts"] == WEEKEND_GAP_TO_TS
    assert gap["classification"] == "scheduled_weekend_closure"
    assert gap["missing_candle_count"] == 0
    assert m1.quality_statistics["missing_open_market_candles"] == 0


def test_unquarantined_m1_ohlcv_conflict_is_rejected(tmp_path: Path) -> None:
    repo_root = tmp_path.resolve()
    authoritative = _candles(BOUNDARY_TS - 180, 2, 60)
    _write_snapshot(repo_root, authoritative)

    conflicting = [_conflicting_candle(authoritative[0]), authoritative[1]]
    _write_receipt(repo_root, conflicting, job_id="offline_m1_conflict")

    with pytest.raises(ValueError, match="Unquarantined OHLCV conflict"):
        build_evidence_inventory(repo_root)
    with pytest.raises(ValueError, match="Unquarantined OHLCV conflict"):
        load_population_candles(repo_root, 60, development_only=False)


def test_known_m1_conflict_is_retained_as_quarantine_evidence(tmp_path: Path) -> None:
    repo_root = tmp_path.resolve()
    authoritative = _candles(BOUNDARY_TS - 180, 2, 60)
    conflict_ts = authoritative[0].from_ts
    _write_snapshot(
        repo_root,
        authoritative,
        conflicts=(
            {
                "boundary_ts": conflict_ts,
                "classification": "AUTHENTIC_SOURCE_PRICE_CONFLICT",
            },
        ),
    )
    conflicting = [_conflicting_candle(authoritative[0]), authoritative[1]]
    _write_receipt(repo_root, conflicting, job_id="offline_m1_known_conflict")

    inventory = build_evidence_inventory(repo_root)
    assert inventory.populations[60].conflicting_timestamps == 1
    assert len(inventory.conflicts) == 1
    assert inventory.conflicts[0]["boundary_ts"] == conflict_ts

    candles = load_population_candles(repo_root, 60, development_only=False)
    assert candles[0].close == authoritative[0].close


def test_native_m5_timestamp_conflict_is_rejected(tmp_path: Path) -> None:
    repo_root = tmp_path.resolve()
    _write_snapshot(repo_root, _candles(BOUNDARY_TS - 180, 1, 60))

    first = _candles(BOUNDARY_TS - 600, 2, 300)
    second = [_conflicting_candle(first[0]), first[1]]
    _write_receipt(repo_root, first, job_id="offline_m5_first")
    _write_receipt(repo_root, second, job_id="offline_m5_second")

    with pytest.raises(ValueError, match="Unquarantined OHLCV conflict"):
        load_population_candles(repo_root, 300, development_only=True)


def test_unsupported_timeframes_are_rejected(tmp_path: Path) -> None:
    _write_snapshot(tmp_path.resolve(), _candles(BOUNDARY_TS - 180, 1, 60))
    with pytest.raises(ValueError, match="Unsupported Phase 45 timeframe"):
        load_population_candles(tmp_path.resolve(), 3600)
