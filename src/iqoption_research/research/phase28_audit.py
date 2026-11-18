"""Read-only dataset/checkpoint audit and minimal, sanitized product-history evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from iqoption_research.market_data.forward_sync import ForwardSyncCheckpoint
from iqoption_research.market_data.history_boundary import verified_index
from iqoption_research.research.prospective import visible_fingerprint


def file_sha(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1_048_576):
            hashed.update(block)
    return hashed.hexdigest()


def audit_phase28(storage: Path, project: Path) -> dict[str, Any]:
    with verified_index(storage) as index:
        coverage = index.report()
        checkpoints = []
        verified_ids = {source["dataset_id"] for source in index.manifest}
        for path in sorted(storage.glob("forward_checkpoint_*.json")):
            checkpoint = ForwardSyncCheckpoint.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
            missing = [
                identifier
                for identifier in checkpoint.committed_dataset_ids
                if identifier not in verified_ids
            ]
            if missing:
                raise ValueError("Forward checkpoint references missing catalog evidence")
            count = index.db.execute(
                "SELECT COUNT(*) FROM bars WHERE ts > ? AND ts + 60 <= ?",
                (checkpoint.baseline_latest_from_ts, checkpoint.target_completed_end_ts),
            ).fetchone()[0]
            if checkpoint.state == "completed" and count != checkpoint.new_records_saved:
                raise ValueError("Completed checkpoint new-unique count contradicts catalog")
            if (
                not 0 <= checkpoint.requests_made <= checkpoint.max_requests
                or checkpoint.active_id != 1
                or checkpoint.symbol != "EURUSD"
                or checkpoint.candle_size_seconds != 60
            ):
                raise ValueError("Forward checkpoint request or identity bounds invalid")
            baseline_hash = hashlib.sha256()
            for entry in index.collector.catalog.find_datasets(
                symbol="EURUSD", candle_size_seconds=60, source="broker_authentic_collection"
            ):
                if (
                    entry.to_ts <= checkpoint.baseline_latest_to_ts
                    and entry.created_at_utc <= checkpoint.created_at_utc
                ):
                    baseline_hash.update(entry.dataset_id.encode())
                    baseline_hash.update(entry.sha256_checksum.encode("ascii"))
            if baseline_hash.hexdigest() != checkpoint.baseline_fingerprint:
                raise ValueError("Forward checkpoint baseline source fingerprint changed")
            checkpoints.append({
                "checkpoint": path.name,
                "sha256": file_sha(path),
                "state": checkpoint.state,
                "cycle": checkpoint.cycle_number,
                "requests_made": checkpoint.requests_made,
                "max_requests": checkpoint.max_requests,
                "new_unique_reported": checkpoint.new_records_saved,
                "verified_new_unique_in_cycle_window": count,
                "duplicates_skipped": checkpoint.duplicates_skipped,
                "unresolved_gaps_reported": checkpoint.unresolved_gap_count,
                "committed_sources_verified": len(checkpoint.committed_dataset_ids),
                "baseline_market_fingerprint": visible_fingerprint(
                    index, checkpoint.baseline_latest_to_ts
                ),
                "baseline_catalog_fingerprint_verified": True,
                "checkpoint_fingerprint_spec": "source_ids_and_checksums_not_market_hash",
            })
        # Preserve the ORIGINAL reserved interval; do not move 688 bars to the new tail.
        reservation_path = project / "data/phase11/confirmation_state.json"
        original_path = project / "docs/phase26/coverage.json"
        protected = {}
        if reservation_path.is_file() and original_path.is_file():
            reservation = json.loads(reservation_path.read_text())
            original = json.loads(original_path.read_text())
            start, end = reservation["prior_evaluation_cutoff_ts"], original["newest_ts"]
            reserved_rows = list(
                index.db.execute(
                    "SELECT signature FROM bars WHERE ts >= ? AND ts <= ? ORDER BY ts", (start, end)
                )
            )
            hashed = hashlib.sha256()
            for (signature,) in reserved_rows:
                hashed.update((signature + "\n").encode())
            protected = {
                "status": reservation.get("evaluation_status"),
                "original_start_ts": start,
                "original_last_open_ts": end,
                "candles": len(reserved_rows),
                "fingerprint_sha256": hashed.hexdigest(),
                "reservation_sha256": file_sha(reservation_path),
                "consumed_by_phase28": False,
            }
            prior_results = project / "docs/phase25/results.json"
            if prior_results.is_file():
                expected = json.loads(prior_results.read_text())["holdout"][
                    "reservation_fingerprint"
                ]
                if file_sha(reservation_path) != expected:
                    raise ValueError("Sealed reservation changed since Phase 25")
            if len(reserved_rows) != 688 or reservation.get("evaluation_status") != "RESERVED":
                raise ValueError("Original holdout reservation is not intact")
    archives = []
    for path in sorted((storage / "phase26_sources").glob("*/*.json")):
        archive_sha = file_sha(path)
        if path.stem.split("_")[-1] != archive_sha:
            raise ValueError("Source archive checksum failure")
        archives.append({"path": path.relative_to(storage).as_posix(), "sha256": archive_sha})
    protected_paths = [
        "data/phase25/registry.json",
        "docs/phase25/registry.json",
        "docs/phase24/results.json",
        "docs/phase25/results.json",
        "data/phase11/confirmation_state.json",
    ]
    return {
        "phase": 28,
        "coverage": coverage,
        "forward_checkpoints": checkpoints,
        "original_holdout": protected,
        "source_archives": archives,
        "protected_file_checksums": {
            name: file_sha(project / name) for name in protected_paths if (project / name).is_file()
        },
        "audit_only": True,
        "new_candles_collected_by_phase28": 0,
    }


def product_history_evidence(project: Path) -> dict[str, Any]:
    sources = []
    observations = []
    for name in ("recovered_history.json", "blitz_history_evidence.json"):
        path = project / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        sources.append({
            "file": name,
            "sha256": file_sha(path),
            "candidate_count": len(data.get("candidates", [])),
            "incomplete_count": len(data.get("incomplete_candidates", [])),
            "history_may_be_truncated": data.get("query_limit_may_truncate_history") is True,
        })
        for row in data.get("candidates", []):
            if row.get("symbol") != "EURUSD_OP" or row.get("product_type") != "blitz-option":
                continue
            duration = row.get("expires_at_utc", 0) - row.get("opened_at_utc", 0)
            observations.append({
                "symbol": "EURUSD_OP",
                "product_type": "blitz-option",
                "duration_seconds": duration if 0 < duration <= 86400 else None,
                "reported_closed": row.get("status") == "closed",
                "reported_win": row.get("result") == "win",
                "direction_present": row.get("direction") in {"call", "put"},
                "payout_present": row.get("payout") is not None,
                "realized_profit_present": row.get("realized_profit") is not None,
                "wallet_link_present": row.get("user_balance_id") is not None,
            })
    return {
        "phase": 28,
        "sources": sources,
        "local_history_observations": observations,
        "EURUSD": {"market_data_active_id": 1, "source": "broker candle quotes"},
        "EURUSD_OP": {
            "history_label_observed": bool(observations),
            "connector_history_reference_active_id": 1861,
            "reference_id_not_native_market_data_mapping": True,
            "equivalent_to_EURUSD": "NOT_ESTABLISHED",
            "native_quote_feed": "UNVERIFIED",
            "strike_formula": "UNVERIFIED",
            "current_payout": "UNVERIFIED",
            "financial_settlement": "UNCONFIRMED",
            "refund_and_tie_rule": "UNVERIFIED",
        },
        "classification": "PRODUCT_EVIDENCE_REQUIRED",
        "broker_requests_made": 0,
        "privacy": "Account amounts, wallet/ticket IDs, credentials and raw payloads omitted",
    }
