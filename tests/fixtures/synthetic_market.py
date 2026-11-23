"""Deterministic synthetic market data fixtures for clean test isolation.

Provides helper utilities to create isolated research workspaces under `pytest` `tmp_path`,
guaranteeing tests run completely offline and never depend on private local datasets.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.market_data.parquet_storage import write_candles_parquet
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.prospective import digest
from iqoption_research.shadow.registry import ShadowModelRegistry


def create_deterministic_synthetic_candles(
    count: int = 1200,
    start_ts: int = 1700000000,
    symbol: str = "EURUSD",
) -> list[Candle]:
    """Generate a deterministic sequence of valid M1 candles."""
    candles: list[Candle] = []
    base_price = 1.0850

    for i in range(count):
        ts = start_ts + (i * 60)
        # Alternate small price movements deterministically
        delta = 0.00015 if (i % 3 == 0) else (-0.00010 if (i % 3 == 1) else 0.0)
        c_close = round(base_price + delta, 6)
        c_open = round(base_price, 6)
        c_high = round(max(c_open, c_close) + 0.0002, 6)
        c_low = round(min(c_open, c_close) - 0.0002, 6)
        base_price = c_close

        candles.append(
            Candle(
                active_id=1,
                symbol=symbol,
                size=CandleSize.MIN_1,
                from_ts=ts,
                to_ts=ts + 60,
                at_ts=ts * 1_000_000_000,
                open=c_open,
                high=c_high,
                low=c_low,
                close=c_close,
                volume=150 + (i % 50),
                bid=round(c_close - 0.00005, 6),
                ask=round(c_close + 0.00005, 6),
            )
        )
    return candles


def setup_isolated_research_workspace(
    tmp_path: Path,
    candle_count: int = 1200,
    start_ts: int = 1700000000,
    symbol: str = "EURUSD",
) -> dict[str, Any]:
    """Create a fully compliant, self-contained research workspace in tmp_path.

    Exercises actual DatasetCatalog, Parquet storage, SnapshotManager,
    Partitions, and Shadow Registry while explicitly marking all evidence as synthetic.
    """
    workspace = tmp_path.resolve()
    candles = create_deterministic_synthetic_candles(
        count=candle_count, start_ts=start_ts, symbol=symbol
    )

    # 1. Parquet storage and snapshot directory
    snapshot_id = "rds_synthetic_eurusd_m1_test"
    snap_dir = workspace / "data" / "research_datasets" / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = snap_dir / "candles.parquet"

    # Write parquet file
    res = write_candles_parquet(
        candles,
        root_dir=workspace / "data" / "parquet_candles",
        is_synthetic=True,
    )
    # Copy or link parquet to snapshot path
    parquet_bytes = Path(res.parquet_path).read_bytes()
    parquet_path.write_bytes(parquet_bytes)
    parquet_sha = res.sha256_checksum

    # 2. Write Snapshot Manifest
    now_utc = datetime.now(UTC).isoformat()
    manifest_data: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "created_at_utc": now_utc,
        "dataset_version": "1.0.0-synthetic",
        "symbol": symbol,
        "candle_size_seconds": 60,
        "from_ts": candles[0].from_ts,
        "to_ts": candles[-1].to_ts,
        "from_utc": datetime.fromtimestamp(candles[0].from_ts, tz=UTC).isoformat(),
        "to_utc": datetime.fromtimestamp(candles[-1].to_ts, tz=UTC).isoformat(),
        "total_candles_discovered": len(candles),
        "total_candles_retained": len(candles),
        "duplicate_boundaries_deduplicated": 0,
        "conflicts_quarantined_count": 0,
        "conflicts_quarantined": [],
        "unscheduled_gaps_count": 0,
        "unscheduled_gaps": [],
        "scheduled_closures_count": 0,
        "scheduled_closures": [],
        "provenance_breakdown": {"SYNTHETIC_FIXTURE": len(candles)},
        "source_files": [
            {
                "candle_count": len(candles),
                "path": str(parquet_path),
                "provenance": "SYNTHETIC_FIXTURE",
                "sha256": parquet_sha,
                "source_id": "synthetic_source_test",
            }
        ],
        "eligibility_rules": [
            "symbol == 'EURUSD' and active_id == 1",
            "candle_size_seconds == 60 (M1 timeframe)",
        ],
        "wallet_classification": "SYNTHETIC_TEST_EVIDENCE",
        "distinct_market_days": len({c.from_ts // 86400 for c in candles}),
        "is_continuous": True,
        "has_genuine_volume": False,
        "has_bid_ask_quotes": True,
        "sha256_checksum": parquet_sha,
        "is_synthetic": True,
        "disclaimer": (
            "SYNTHETIC TEST FIXTURE. ENGINEERING VERIFICATION ONLY. "
            "CANNOT PRODUCE SCIENTIFIC ACCEPTANCE."
        ),
        "manifest_sha256": "",
    }
    clean = {k: v for k, v in manifest_data.items() if k != "manifest_sha256"}
    manifest_sha = digest(clean)
    manifest_data["manifest_sha256"] = manifest_sha

    manifest_file = snap_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    # 3. Initialize Shadow Model Registry
    reg_dir = workspace / "data" / "shadow_eval"
    reg_dir.mkdir(parents=True, exist_ok=True)
    registry_file = reg_dir / "registry.json"
    clean_reg: dict[str, Any] = {
        "schema_version": "1.0.0",
        "updated_at_utc": now_utc,
        "records": {},
        "notes": "Test isolated shadow model registry",
    }
    clean_reg["registry_sha256"] = digest(clean_reg)
    registry_file.write_text(json.dumps(clean_reg, indent=2, sort_keys=True), encoding="utf-8")

    # 4. Create remaining directories
    (workspace / "data" / "partitions").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "plans").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "candidate_models").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "experiments").mkdir(parents=True, exist_ok=True)
    (workspace / "reports").mkdir(parents=True, exist_ok=True)

    return {
        "root": workspace,
        "snapshot_id": snapshot_id,
        "snapshot_dir": snap_dir,
        "parquet_path": parquet_path,
        "manifest_path": manifest_file,
        "candles": candles,
        "registry": ShadowModelRegistry(workspace),
    }
