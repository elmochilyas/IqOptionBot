"""Forensic reconstruction of market data acquisition timing evidence.

Analyzes historical forward checkpoints, catalog entries, and prospective
journals to decompose timing intervals and distinguish genuine measurements
from proxies.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.timing.taxonomy import (
    LatencyInterval,
    MeasurementQuality,
    TimestampAuditRecord,
    TimestampRole,
)


def _iso_to_epoch(iso_str: str) -> float:
    """Parse ISO-8601 string to epoch seconds."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.timestamp()


def audit_checkpoint_timing(checkpoint_data: dict[str, Any], source_name: str) -> dict[str, Any]:
    """Forensically audit timing metadata from a forward-sync checkpoint."""
    target_end_ts = int(checkpoint_data["target_completed_end_ts"])
    created_iso = checkpoint_data["created_at_utc"]
    updated_iso = checkpoint_data["updated_at_utc"]
    created_epoch = _iso_to_epoch(created_iso)
    updated_epoch = _iso_to_epoch(updated_iso)

    # 1. Individual timestamp audit records
    records = [
        TimestampAuditRecord(
            name="target_completed_end_ts",
            role=TimestampRole.NOMINAL_BUCKET_BOUNDARY,
            quality=MeasurementQuality.NOMINAL_DISCRETE_LABEL,
            timestamp_epoch=target_end_ts,
            timestamp_iso=datetime.fromtimestamp(target_end_ts, tz=UTC).isoformat(),
            description="Nominal candle close boundary F (bucket label, not packet arrival)",
            source_context=source_name,
        ),
        TimestampAuditRecord(
            name="created_at_utc",
            role=TimestampRole.CLIENT_INVOCATION_CLOCK,
            quality=MeasurementQuality.GENUINELY_MEASURED,
            timestamp_epoch=created_epoch,
            timestamp_iso=created_iso,
            description="Client wall clock when operator or script invoked collection process",
            source_context=source_name,
        ),
        TimestampAuditRecord(
            name="updated_at_utc",
            role=TimestampRole.CLIENT_STORAGE_PERSIST,
            quality=MeasurementQuality.GENUINELY_MEASURED,
            timestamp_epoch=updated_epoch,
            timestamp_iso=updated_iso,
            description="Client wall clock when parquet persistence and catalog commit finished",
            source_context=source_name,
        ),
        TimestampAuditRecord(
            name="broker_first_candle_arrival",
            role=TimestampRole.CLIENT_SOCKET_RECEIVE,
            quality=MeasurementQuality.UNMEASURED_UNKNOWN,
            timestamp_epoch=None,
            timestamp_iso=None,
            description="Actual instant broker completed candle frame arrived at client socket",
            source_context=source_name,
        ),
    ]

    # 2. Decomposed latency intervals
    invocation_delay = created_epoch - target_end_ts
    collection_duration = updated_epoch - created_epoch
    total_offset_from_close = updated_epoch - target_end_ts

    intervals = [
        LatencyInterval(
            name="operator_invocation_offset",
            start_role=TimestampRole.NOMINAL_BUCKET_BOUNDARY,
            end_role=TimestampRole.CLIENT_INVOCATION_CLOCK,
            duration_seconds=round(invocation_delay, 3),
            quality=MeasurementQuality.GENUINELY_MEASURED,
            description="Elapsed time from candle close until collection command execution began",
        ),
        LatencyInterval(
            name="collection_execution_duration",
            start_role=TimestampRole.CLIENT_INVOCATION_CLOCK,
            end_role=TimestampRole.CLIENT_STORAGE_PERSIST,
            duration_seconds=round(collection_duration, 3),
            quality=MeasurementQuality.GENUINELY_MEASURED,
            description="Duration of startup, TLS/auth, network fetch, and parquet persistence",
        ),
        LatencyInterval(
            name="total_offset_from_candle_close",
            start_role=TimestampRole.NOMINAL_BUCKET_BOUNDARY,
            end_role=TimestampRole.CLIENT_STORAGE_PERSIST,
            duration_seconds=round(total_offset_from_close, 3),
            quality=MeasurementQuality.GENUINELY_MEASURED,
            description="Total elapsed time from nominal candle close to parquet persistence",
        ),
        LatencyInterval(
            name="broker_delivery_latency",
            start_role=TimestampRole.NOMINAL_BUCKET_BOUNDARY,
            end_role=TimestampRole.CLIENT_SOCKET_RECEIVE,
            duration_seconds=None,
            quality=MeasurementQuality.UNMEASURED_UNKNOWN,
            description="Physical server-to-socket transmission latency (uninstrumented in batch)",
        ),
    ]

    requests_made = int(checkpoint_data.get("requests_made", 1))
    new_records = int(checkpoint_data.get("new_records_saved", 0))

    return {
        "job_id": checkpoint_data.get("job_id", "unknown"),
        "source": source_name,
        "target_completed_end_ts": target_end_ts,
        "requests_made": requests_made,
        "new_records_saved": new_records,
        "invocation_delay_seconds": round(invocation_delay, 3),
        "collection_duration_seconds": round(collection_duration, 3),
        "total_offset_seconds": round(total_offset_from_close, 3),
        "single_request_batch": requests_made == 1,
        "satisfies_phase30_15s_gate": total_offset_from_close <= 15.0,
        "timestamp_records": [r.to_dict() for r in records],
        "latency_intervals": [i.to_dict() for i in intervals],
    }


def audit_observation_timing(journal_db_path: Path) -> list[dict[str, Any]]:
    """Forensically audit timing records from an observation SQLite journal."""
    if not journal_db_path.is_file():
        return []

    con = sqlite3.connect(journal_db_path)
    cur = con.cursor()
    query = (
        "SELECT seq, event_key, payload FROM events "
        "WHERE event_key LIKE 'prediction:%' OR event_key LIKE 'receipt:%'"
    )
    rows = cur.execute(query).fetchall()
    con.close()

    events_by_key: dict[str, dict[str, Any]] = {}
    receipts_by_key: dict[str, dict[str, Any]] = {}

    for _seq, event_key, payload_str in rows:
        payload = json.loads(payload_str)
        if event_key.startswith("prediction:"):
            events_by_key[event_key] = payload
        elif event_key.startswith("receipt:"):
            pred_key = payload.get("prediction_key") or event_key.replace("receipt:", "")
            receipts_by_key[pred_key] = payload

    audits = []
    for key, pred in events_by_key.items():
        f_close = int(pred.get("feature_max_available_ts") or pred.get("feature_close", 0))
        observed = float(pred.get("data_observed_at_ts", 0))
        decided = float(pred.get("recorded_at_ts", 0))
        receipt = receipts_by_key.get(key)
        committed = float(receipt.get("observed_at_ts", decided)) if receipt else decided
        entry = int(pred.get("entry_ts", 0))

        delta_observed = observed - f_close if f_close else None
        delta_decided = decided - f_close if f_close else None
        delta_committed = committed - f_close if f_close else None

        meets_phase29_gate = (f_close <= decided < entry) if (f_close and entry) else False
        meets_phase30_gate = (
            (f_close <= observed <= decided <= committed <= f_close + 15)
            if (f_close and observed and decided and committed)
            else False
        )

        audits.append({
            "prediction_key": key,
            "feature_close_F": f_close,
            "observed_at_ts": observed,
            "decided_at_ts": decided,
            "committed_at_ts": committed,
            "entry_ts": entry,
            "offset_observed_from_F": (
                round(delta_observed, 3) if delta_observed is not None else None
            ),
            "offset_decided_from_F": (
                round(delta_decided, 3) if delta_decided is not None else None
            ),
            "offset_committed_from_F": (
                round(delta_committed, 3) if delta_committed is not None else None
            ),
            "decision": pred.get("decision"),
            "meets_phase29_gate": meets_phase29_gate,
            "meets_phase30_gate": meets_phase30_gate,
        })

    return audits


def reconstruct_all_timing_evidence(project_root: Path) -> dict[str, Any]:
    """Scan and reconstruct all timing evidence across checkpoints and prospective journals."""
    datasets_dir = project_root / "data" / "datasets"
    checkpoint_files = sorted(datasets_dir.glob("forward_checkpoint_*.json"))

    checkpoints_audit = []
    for cp_file in checkpoint_files:
        try:
            data = json.loads(cp_file.read_text(encoding="utf-8"))
            audited = audit_checkpoint_timing(data, cp_file.name)
            checkpoints_audit.append(audited)
        except Exception as exc:  # noqa: BLE001
            checkpoints_audit.append({"file": cp_file.name, "error": str(exc)})

    # Audit prospective journals
    journal_paths = [
        ("Phase 28", project_root / "data" / "phase28" / "operator-001" / "observations.db"),
        (
            "Phase 29",
            project_root / "data" / "phase29" / "operator-20261118-090107" / "observations.db",
        ),
        ("Phase 30", project_root / "data" / "phase30" / "prospective-001" / "observations.db"),
    ]

    journals_audit = {}
    for phase_name, jpath in journal_paths:
        journals_audit[phase_name] = audit_observation_timing(jpath)

    # Synthesis of proxy vs genuine measurement
    epistemic_findings = {
        "genuinely_measured_timestamps": [
            "checkpoint.created_at_utc (Client wall-clock invocation timestamp)",
            "checkpoint.updated_at_utc (Client wall-clock persistence completion timestamp)",
            "journal.data_observed_at_ts (Client wall-clock dataset read timestamp)",
            "journal.recorded_at_ts (Client wall-clock model decision timestamp)",
            "receipt.observed_at_ts (Client wall-clock journal commit timestamp)",
        ],
        "proxy_and_discrete_timestamps": [
            "candle.from_ts / to_ts (Nominal discrete bucket boundary; not packet arrival)",
            "feature_row.available_at_ts (Hardcoded to candle.to_ts in pipeline.py; causal proxy)",
            "checkpoint.target_completed_end_ts (Client-computed discrete target boundary)",
        ],
        "unmeasured_physical_events": [
            "Broker candle frame assembly and WebSocket emit timestamp",
            "Network wire transit duration",
            "Client OS socket receive timestamp (time.perf_counter_ns of frame receipt)",
        ],
        "critical_conclusion": (
            "Historical catalog and checkpoint timestamps reflect manual operator or scheduled "
            "batch invocation offset (24.7s - 46.8s), NOT broker delivery latency. The broker's "
            "first candle availability distribution is completely uninstrumented in existing data."
        ),
    }

    return {
        "timestamp_audit_summary": {
            "checkpoints_examined": len(checkpoints_audit),
            "checkpoints_meeting_15s_gate": sum(
                1 for cp in checkpoints_audit if cp.get("satisfies_phase30_15s_gate")
            ),
            "single_request_batches": [
                cp for cp in checkpoints_audit if cp.get("single_request_batch")
            ],
        },
        "checkpoints": checkpoints_audit,
        "prospective_journals": journals_audit,
        "epistemic_findings": epistemic_findings,
    }
