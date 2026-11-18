"""Offline seeding avoids legacy collector audits opening the original sealed catalog."""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.research.aligned_data import inventory, load_selected, protection
from iqoption_research.research.aligned_study import allowed_run, write_new_json
from iqoption_research.research.phase28_audit import file_sha


def seed_collection(project: Path, storage: Path, destination: Path) -> dict[str, Any]:
    allowed_run(destination, project)
    if destination.exists():
        raise ValueError("Use a NEW isolated collection store")
    guard = protection(project)
    F = int(time.time() // 60) * 60
    manifest = inventory(storage, guard, development=False, lower=F - 7 * 86400, upper=F)
    candles, _ = load_selected(storage, manifest, guard, upper=F)
    # Only post-seal seed candles enter the isolated collector's catalog.
    candles = [c for c in candles if c.from_ts >= guard["end_exclusive"]][-64:]
    if len(candles) != 64 or any(b.from_ts != a.to_ts for a, b in itertools.pairwise(candles)):
        raise ValueError("Need 64 contiguous verified post-seal seed candles")
    HistoricalDataCollector(storage_dir=destination).persist_batch(
        candles, symbol="EURUSD", candle_size_seconds=60
    )
    evidence = {
        "source_inventory_hash": manifest["fingerprint"],
        "seed_candles": 64,
        "seed_first_from_ts": candles[0].from_ts,
        "seed_last_close_ts": candles[-1].to_ts,
        "new_broker_requests": 0,
        "prices_modified": False,
        "sealed_files_opened": False,
        "acquisition_status": "REPEATED_AUTHENTICATED_SESSION_REMAINS_BLOCKED",
    }
    write_new_json(destination / "seed_lineage.json", evidence)
    return evidence


def acquisition_audit(project: Path) -> dict[str, Any]:
    paths = [
        "src/iqoption_research/cli/commands/backfill_cmd.py",
        "src/iqoption_research/market_data/forward_sync.py",
        "src/iqoption_research/connectors/iqoption/client.py",
    ]
    return {
        "source_sha256": {p: file_sha(project / p) for p in paths},
        "verified_behavior": [
            "Explicit live flags and interactive local login",
            "Fixed completed target and persisted request budget",
            "Read-only client capability assertion",
            "Client close and auth logout in finally",
        ],
        "not_established": [
            "Broker-approved repeated acquisition rate",
            "Safe persistent subscription/reconnection behavior",
            "Reliable end-to-end acquisition within 15 seconds",
        ],
        "repeated_acquisition": "BLOCKED_NO_NEW_BROKER_REQUEST_TYPES",
        "manual_protocol": "One explicit bounded cycle against NEW isolated post-seal store",
        "agent_authenticated_requests": 0,
        "wallet_switches": 0,
    }
