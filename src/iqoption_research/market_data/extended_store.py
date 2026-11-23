"""Incremental collection receipts; offline verification never opens a broker connection."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.parquet_storage import (
    read_candles_parquet,
    write_candles_parquet,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.prospective import digest

NAMESPACE = Path("data/extended_collection")


def atomic_json(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(temp, path)  # Atomic create-only publication on the same NTFS volume.
        else:
            temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def sealed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "sha256": digest(value)}


def read_sealed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("sha256")
    if digest(value) != claimed:
        raise ValueError("Collection integrity mismatch")
    return value


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Collection path escapes session")
    return path


def persist_page(
    directory: Path,
    candles: list[Candle],
    provenance: dict[str, Any],
) -> None:
    """Commit data/catalog first, then an immutable availability receipt.

    Orphaned Parquet pages from an interrupted publication remain on disk but are
    reported as uncommitted. No availability timestamp is invented to recover them.
    """
    boundary = int(provenance["requested_boundary"])
    receipt = directory / "receipts" / f"{boundary}.json"
    if receipt.exists():
        raise FileExistsError("This request boundary already has a committed receipt")
    result = write_candles_parquet(
        candles,
        directory / "datasets",
        source="iqoption",
        is_synthetic=False,
        catalog=DatasetCatalog(directory / "catalog.db"),
        on_duplicate="raise",
    )
    if result.catalog_entry is None:
        raise ValueError("Collection catalog registration missing")
    for path in (result.parquet_path, result.metadata_path):
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
    atomic_json(
        receipt,
        sealed({
            "version": 1,
            "kind": "COLLECTION_DATA_NOT_PREDICTIONS",
            "catalog_entry": asdict(result.catalog_entry),
            "provenance": provenance,
            "boundaries": [c.to_ts for c in candles],
        }),
        immutable=True,
    )


def inspect_session(directory: Path) -> dict[str, Any]:
    """Read-only verification, including catalog binding and actual availability receipts."""
    spec = read_sealed(directory / "session.json")
    boundaries: set[int] = set()
    receipts: dict[str, str] = {}
    committed: set[str] = set()
    for path in sorted((directory / "receipts").glob("*.json")):
        receipt = read_sealed(path)
        entry = CatalogEntry(**receipt["catalog_entry"])
        if entry.source != "iqoption" or entry.is_synthetic or entry.is_otc:
            raise ValueError("Collection source is not authentic regular-market data")
        if entry.symbol != "EURUSD" or entry.active_id != 1 or entry.candle_size_seconds != 60:
            raise ValueError("Collection instrument mismatch")
        database = directory / "catalog.db"
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM dataset_catalog WHERE dataset_id = ?", (entry.dataset_id,)
            ).fetchone()
            if row is None or CatalogEntry.from_row(row) != entry:
                raise ValueError("Collection catalog binding mismatch")
        candles = read_candles_parquet(
            contained(directory / "datasets", entry.file_path),
            expected_symbol="EURUSD",
            expected_checksum=entry.sha256_checksum,
            expected_metadata_checksum=entry.metadata_sha256,
            catalog_entry=entry,
        )
        values = [c.to_ts for c in candles]
        if values != receipt["boundaries"] or boundaries.intersection(values):
            raise ValueError("Duplicate or conflicting committed collection boundaries")
        observed = float(receipt["provenance"]["observed_at_utc_ts"])
        if any(value > observed for value in values):
            raise ValueError("Observed availability precedes candle completion")
        boundaries.update(values)
        receipts[path.name] = file_hash(path)
        committed.add(entry.file_path)
    state: dict[str, Any] = {}
    if (directory / "state.json").is_file():
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    final = directory / "manifest.json"
    status = "VERIFIED_PARTIAL / INTERRUPTED_OR_ACTIVE"
    if final.is_file():
        manifest = read_sealed(final)
        if manifest["receipts"] != receipts or manifest["session_sha256"] != file_hash(
            directory / "session.json"
        ):
            raise ValueError("Collection manifest does not match committed files")
        state = manifest["final_state"]
        if state["valid_candles"] != len(boundaries):
            raise ValueError("Collection manifest count mismatch")
        status = "VERIFIED"
    else:
        state["last_recorded_connection"] = state.get("connection", "UNKNOWN")
        state["connection"] = "NOT_LIVE_VERIFIED"
        state["termination_reason"] = "NO_FINAL_MANIFEST_ACTIVE_OR_INTERRUPTED"
    disk = {
        str(p.relative_to(directory / "datasets"))
        for p in (directory / "datasets").rglob("*.parquet")
    }
    return {
        **spec,
        **state,
        "session_id": directory.name,
        "valid_candles": len(boundaries),
        "boundaries": sorted(boundaries),
        "receipts": receipts,
        "integrity": status,
        "uncommitted_files": sorted(disk - committed),
        "destination": str(directory),
    }


def library(root: Path) -> list[dict[str, Any]]:
    rows = []
    for directory in sorted((root / NAMESPACE).glob("collection_*"), reverse=True):
        if not directory.is_dir():
            continue
        try:
            rows.append(inspect_session(directory))
        except Exception as exc:  # noqa: BLE001 - file failures are explicit, never counted as verified
            rows.append({
                "session_id": directory.name,
                "integrity": "FAILED",
                "error": type(exc).__name__,
                "destination": str(directory),
                "valid_candles": 0,
            })
    return rows
