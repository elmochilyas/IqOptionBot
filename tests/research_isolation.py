"""Legacy CLI smoke tests use synthetic data, never production sealed candles."""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import sys
from collections.abc import MutableMapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.models.market import Candle, CandleSize

CREDENTIAL_ENV_VARS: tuple[str, ...] = (
    "IQOPTION_EMAIL",
    "IQOPTION_PASSWORD",
    "IQOPTION_SSID",
    "IQOPTION_2FA_CODE",
    "IQOPTION_ACCOUNT_ID",
    "OPENAI_API_KEY",
)
_CREDENTIAL_KEY_MARKERS: tuple[str, ...] = (
    "EMAIL",
    "PASS",
    "SECRET",
    "TOKEN",
    "KEY",
    "SSID",
    "CREDENTIAL",
    "AUTH",
)

_NETWORK_GUARD_INSTALLED = False
_SEALED_READ_GUARD_INSTALLED = False
_DENIED_SEALED_PATHS: set[str] = set()


def scrub_credential_environment(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Clear real credential environment variables so local secrets never leak into tests."""
    target = os.environ if environ is None else environ
    scrubbed: list[str] = []
    keys = set(CREDENTIAL_ENV_VARS)
    for key in list(target.keys()):
        upper = key.upper()
        if upper.startswith("IQOPTION_") and any(
            marker in upper for marker in _CREDENTIAL_KEY_MARKERS
        ):
            keys.add(key)
    for key in sorted(keys):
        if target.get(key):
            scrubbed.append(key)
        target[key] = ""
    return scrubbed


def is_network_isolation_installed() -> bool:
    """Return whether the process-wide external network audit guard is active."""
    return _NETWORK_GUARD_INSTALLED


def install_network_isolation_guard() -> bool:
    """Unconditionally block non-loopback socket.connect calls for offline test safety."""
    global _NETWORK_GUARD_INSTALLED
    if _NETWORK_GUARD_INSTALLED:
        return False

    def _network_audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "socket.connect" and len(args) >= 2:
            addr = args[1]
            if isinstance(addr, tuple) and addr:
                host = addr[0]
                if isinstance(host, bytes):
                    host = host.decode("ascii", errors="ignore")
                if host not in ("127.0.0.1", "::1", "localhost", ""):
                    raise PermissionError(f"EXTERNAL_NETWORK_DENIED_BY_OFFLINE_TEST_GUARD: {host}")

    sys.addaudithook(_network_audit)
    _NETWORK_GUARD_INSTALLED = True
    return True


def discover_sealed_holdout_paths(project: Path) -> set[str]:
    """Discover sealed holdout dataset shards from catalog.db when present locally."""
    catalog = project / "data/datasets/catalog.db"
    audit_ref = project / "docs/phase29/review/dataset_audit.json"
    if not catalog.is_file() or not audit_ref.is_file():
        return set()
    reference = json.loads(audit_ref.read_text(encoding="utf-8"))["original_holdout"]
    with closing(sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT file_path FROM dataset_catalog WHERE from_ts<=? AND to_ts>=?",
            (reference["original_last_open_ts"] + 60, reference["original_start_ts"]),
        )
        return {str((catalog.parent / path).resolve()).casefold() for (path,) in rows}


def install_sealed_read_guard(project: Path) -> set[str]:
    """Install offline network/credential guards unconditionally and sealed-shard guards if present."""
    global _SEALED_READ_GUARD_INSTALLED
    scrub_credential_environment()
    install_network_isolation_guard()

    denied = discover_sealed_holdout_paths(project)
    if not denied:
        return set()

    _DENIED_SEALED_PATHS.update(denied)
    if not _SEALED_READ_GUARD_INSTALLED:

        def check(path: Any) -> None:
            if (
                isinstance(path, (str, Path))
                and str(Path(path).resolve()).casefold() in _DENIED_SEALED_PATHS
            ):
                raise PermissionError("SEALED_HOLDOUT_READ_DENIED_BY_TEST_GUARD")

        def audit(event: str, args: tuple[Any, ...]) -> None:
            if event == "open" and args:
                check(args[0])

        original = pq.read_table

        def guarded_read(source: Any, *args: Any, **kwargs: Any) -> Any:
            check(source)
            return original(source, *args, **kwargs)

        sys.addaudithook(audit)
        pq.read_table = guarded_read
        _SEALED_READ_GUARD_INSTALLED = True
    return set(_DENIED_SEALED_PATHS)


def build_legacy_workspace(root: Path, project: Path) -> Path:
    """Synthetic fixtures emulate source identity ONLY for engineering tests, never research."""
    candles = []
    rng = np.random.default_rng(300)
    previous = 1.10
    stamp = 1789448460
    for i in range(62000):
        while datetime.fromtimestamp(stamp, UTC).weekday() >= 5:
            stamp += 60
        op = previous
        close = op + float(rng.normal(0, 0.00012))
        previous = close
        candles.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=stamp,
                to_ts=stamp + 60,
                at_ts=stamp + 60,
                open=op,
                high=max(op, close) + float(rng.uniform(0.00001, 0.00008)),
                low=min(op, close) - float(rng.uniform(0.00001, 0.00008)),
                close=close,
                volume=100 + i % 50,
                bid=op,
                ask=op + 0.00001,
            )
        )
        stamp += 60
    for i in range(688):
        stamp = 1794780060 + i * 60
        candles.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=stamp,
                to_ts=stamp + 60,
                at_ts=stamp + 60,
                open=1.1,
                high=1.10002,
                low=1.09998,
                close=1.1 + 0.00001 * math.sin(i),
                volume=100 + i % 50,
                bid=1.1,
                ask=1.10001,
            )
        )
    HistoricalDataCollector(storage_dir=root / "data/datasets").persist_batch(
        candles, symbol="EURUSD", candle_size_seconds=60
    )
    source = project / "data/phase13/candidate_models/candidate_h1_rf"
    shutil.copytree(source, root / "data/phase13/candidate_models/candidate_h1_rf")
    reservation = root / "data/phase11/confirmation_state.json"
    reservation.parent.mkdir(parents=True, exist_ok=True)
    source_reservation = project / "data/phase11/confirmation_state.json"
    if source_reservation.exists():
        shutil.copyfile(source_reservation, reservation)
    else:
        reservation.write_text(
            json.dumps(
                {
                    "confirmation_dataset_fingerprint": None,
                    "confirmation_days": 14,
                    "confirmation_end_ts": None,
                    "confirmation_start_ts": None,
                    "evaluated_at_utc": None,
                    "evaluation_started_at_utc": None,
                    "evaluation_status": "RESERVED",
                    "full_dataset_fingerprint_at_evaluation": None,
                    "prior_evaluated_dataset_fingerprint": (
                        "70d8b1e906d5b76f97b93346ad4399e23abe1630dca66e14b1a1828a617e6632"
                    ),
                    "prior_evaluation_cutoff_ts": 1794607200,
                    "prior_evaluation_cutoff_utc": "2026-11-13T22:00:00+00:00",
                    "schema_version": 1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    (root / "SYNTHETIC_TEST_FIXTURE_ONLY.json").write_text(
        json.dumps({
            "synthetic": True,
            "independent_evidence": False,
            "broker_acquisition": False,
            "source_identity_emulated_for_legacy_engineering_tests_only": True,
        })
    )
    return root
