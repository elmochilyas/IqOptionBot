"""Select development shards BEFORE opening prices. Never open sealed/overlapping shards."""

from __future__ import annotations

import itertools
import json
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.market_data.candles import validate_ohlc
from iqoption_research.market_data.catalog import CatalogEntry
from iqoption_research.market_data.history_expansion import signature
from iqoption_research.market_data.parquet_storage import read_candles_parquet
from iqoption_research.models.market import Candle
from iqoption_research.research.aligned_contract import FEATURES, outcome_label, target_times
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.phase28_audit import file_sha
from iqoption_research.research.prospective import digest


def protection(project: Path) -> dict[str, Any]:
    reference = json.loads((project / "docs/phase29/review/dataset_audit.json").read_text())
    guard = reference["original_holdout"]
    reservation = project / "data/phase11/confirmation_state.json"
    state = json.loads(reservation.read_text())
    if (
        guard["candles"] != 688
        or state["evaluation_status"] != "RESERVED"
        or state["prior_evaluation_cutoff_ts"] != guard["original_start_ts"]
        or file_sha(reservation) != guard["reservation_sha256"]
    ):
        raise ValueError("Original sealed holdout reservation cannot be established")
    return {
        **guard,
        "end_exclusive": guard["original_last_open_ts"] + 60,
        "verification": "RESERVATION_BYTES_AND_PRIOR_HASH_REFERENCE_ONLY_NO_SEALED_PRICE_READ",
        "sealed_candle_hash_recomputed": False,
    }


def catalog_entries(storage: Path) -> list[CatalogEntry]:
    path = (storage / "catalog.db").resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        return [
            CatalogEntry.from_row(r)
            for r in connection.execute("SELECT * FROM dataset_catalog ORDER BY from_ts,dataset_id")
        ]


def safe_path(storage: Path, path: str) -> Path:
    resolved = (storage / path).resolve()
    if not resolved.is_relative_to(storage.resolve()):
        raise ValueError("Source path escapes storage")
    return resolved


def inventory(
    storage: Path,
    guard: dict[str, Any],
    *,
    development: bool = True,
    lower: int | None = None,
    upper: int | None = None,
) -> dict[str, Any]:
    selected, excluded = [], []
    for entry in catalog_entries(storage):
        reason = None
        if not (
            entry.source == "broker_authentic_collection"
            and not entry.is_synthetic
            and entry.symbol == "EURUSD"
            and entry.active_id == 1
            and not entry.is_otc
            and entry.candle_size_seconds == 60
            and entry.structural_integrity_passed
        ):
            reason = "INELIGIBLE_IDENTITY_OR_INTEGRITY"
        elif entry.from_ts < guard["end_exclusive"] and entry.to_ts >= guard["original_start_ts"]:
            reason = "SEALED_OR_OVERLAPPING_SHARD_NOT_OPENED"
        elif development and entry.to_ts >= guard["original_start_ts"]:
            reason = "AFTER_DEVELOPMENT_CUTOFF_NOT_OPENED"
        elif (lower is not None and entry.to_ts < lower) or (
            upper is not None and entry.from_ts > upper
        ):
            continue
        if reason:
            excluded.append({
                "dataset_id": entry.dataset_id,
                "reason": reason,
                "catalog_sha256": entry.sha256_checksum,
            })
        else:
            safe_path(storage, entry.file_path)
            selected.append(entry.to_dict())
    return {"selected": selected, "excluded": excluded, "fingerprint": digest(selected)}


def load_selected(
    storage: Path, manifest: dict[str, Any], guard: dict[str, Any], *, upper: int | None = None
) -> tuple[list[Candle], dict[str, int]]:
    catalog = {entry.dataset_id: entry.to_dict() for entry in catalog_entries(storage)}
    candles: dict[int, Candle] = {}
    counts: Counter[str] = Counter()
    for row in manifest["selected"]:
        if row["from_ts"] < guard["end_exclusive"] and row["to_ts"] >= guard["original_start_ts"]:
            raise ValueError("Refusing to open a sealed or overlapping shard")
        if catalog.get(row["dataset_id"]) != row:
            raise ValueError("Source manifest differs from current trusted catalog")
        batch = read_candles_parquet(
            safe_path(storage, row["file_path"]),
            expected_symbol="EURUSD",
            expected_checksum=row["sha256_checksum"],
            expected_metadata_checksum=row["metadata_sha256"],
        )
        if len(batch) != row["candle_count"]:
            raise ValueError("Source row count contradicts registered catalog")
        for c in batch:
            if c.from_ts < row["from_ts"] or c.to_ts > row["to_ts"]:
                raise ValueError("Source timestamps contradict catalog bounds")
            if c.from_ts < guard["end_exclusive"] and c.to_ts > guard["original_start_ts"]:
                raise ValueError("Catalog misdeclared a sealed source")
            if upper is not None and c.to_ts > upper:
                counts["not_yet_observable"] += 1
                continue
            if not c.is_complete:
                counts["incomplete"] += 1
                continue
            try:
                validate_ohlc(c.open, c.high, c.low, c.close)
                valid = (
                    c.active_id == 1
                    and c.symbol == "EURUSD"
                    and c.size.value == 60
                    and c.to_ts == c.from_ts + 60
                    and c.from_ts % 60 == 0
                    and np.isfinite([c.open, c.high, c.low, c.close]).all()
                    and c.low > 0
                )
            except (ValueError, InvalidCandleError):
                valid = False
            if not valid:
                counts["invalid"] += 1
                continue
            previous = candles.get(c.from_ts)
            if previous is not None:
                if signature(previous) != signature(c):
                    raise ValueError("Conflicting authentic duplicate")
                counts["identical_duplicates"] += 1
            candles[c.from_ts] = c
    return sorted(candles.values(), key=lambda c: c.from_ts), dict(counts)


@dataclass
class AlignedDataset:
    X: np.ndarray
    y: np.ndarray
    feature_close: np.ndarray
    feature_start: np.ndarray
    expiry: np.ndarray
    regime: list[str]
    sensitivity: dict[int, np.ndarray]
    exclusions: dict[str, int]
    fingerprint: str


def build_dataset(candles: list[Candle]) -> AlignedDataset:
    if any(b.from_ts <= a.from_ts for a, b in itertools.pairwise(candles)):
        raise ValueError("Require unique chronological candles")
    segments: list[list[Candle]] = []
    for c in candles:
        if not segments or c.from_ts != segments[-1][-1].to_ts:
            segments.append([])
        segments[-1].append(c)
    X, labels, closes, starts, expiries, regimes = [], [], [], [], [], []
    sensitivities: dict[int, list[int]] = {60: [], 180: []}
    counts: Counter[str] = Counter()
    for segment in segments:
        rows = (
            CausalFeaturePipeline(include_volume=False, include_quotes=False)
            .transform(segment)
            .rows
        )
        for row in rows:
            t = row.index
            if row.is_warmup:
                counts["insufficient_context"] += 1
                continue
            if t + 3 >= len(segment):
                counts["missing_latency_or_outcome"] += 1
                continue
            F = segment[t].to_ts
            entry, expiry = target_times(F)
            target = segment[t + 3]
            if target.from_ts != entry or target.to_ts != expiry:
                raise ValueError("Target geometry mismatch")
            X.append([row.features[name] for name in FEATURES])
            labels.append(outcome_label(target.open, target.close))
            closes.append(F)
            starts.append(segment[t - 20].from_ts)
            expiries.append(expiry)
            regimes.append(
                ("UP" if row.features["sma_ratio_20"] >= 0 else "DOWN")
                + ("_HIGH_VOL" if row.features["volatility_20"] >= 0.0001 else "_LOW_VOL")
            )
            for delay, index in ((60, t + 2), (180, t + 4)):
                sensitivities[delay].append(
                    outcome_label(segment[index].open, segment[index].close)
                    if index < len(segment)
                    else -2
                )
    matrix = np.asarray(X, dtype=float)
    if not len(X) or not np.isfinite(matrix).all():
        raise ValueError("No valid aligned development rows")
    return AlignedDataset(
        matrix,
        np.asarray(labels),
        np.asarray(closes),
        np.asarray(starts),
        np.asarray(expiries),
        regimes,
        {k: np.asarray(v) for k, v in sensitivities.items()},
        dict(counts),
        digest([signature(c) for c in candles]),
    )


def chronological_splits(data: AlignedDataset) -> dict[str, np.ndarray]:
    n = len(data.y)
    edges = [0, int(n * 0.60), int(n * 0.75), int(n * 0.85), n]
    result = {}
    for i, name in enumerate(("train", "calibration", "selection", "test")):
        lower = data.feature_start[0] if i == 0 else data.feature_close[edges[i]]
        upper = data.expiry[-1] + 1 if i == 3 else data.feature_close[edges[i + 1]]
        indices = np.arange(edges[i], edges[i + 1])
        indices = indices[(data.feature_start[indices] >= lower) & (data.expiry[indices] < upper)]
        if len(indices) < 30:
            raise ValueError("Insufficient purged chronological partition")
        result[name] = indices
    for left, right in zip(list(result.values()), list(result.values())[1:], strict=False):
        if data.expiry[left].max() >= data.feature_start[right].min():
            raise ValueError("Labels/context overlap chronological partitions")
    return result
