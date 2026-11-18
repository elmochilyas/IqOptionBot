"""Five-request, non-persisting investigation of historical retrieval boundaries.

Only reports are written. The original catalog is opened read-only and backed up to
a temporary database before reusing Phase 26's checksum verification and disk index.
No retries, alternate endpoints, guessed symbols, or unit-conversion probes occur.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import sqlite3
import tempfile
from collections.abc import Generator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from iqoption_research.connectors.iqoption.exceptions import (
    ConnectionError,
    LoginError,
    RequestTimeoutError,
)
from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.collector import (
    DataAcquisitionError,
    HistoricalBroker,
    HistoricalDataCollector,
)
from iqoption_research.market_data.history_expansion import (
    HistoryIndex,
    atomic_json,
    signature,
    strict_candles,
)


class _ClosingConnection(sqlite3.Connection):
    """Close temporary snapshot handles deterministically, including on Windows."""

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: TracebackType | None,
    ) -> Literal[False]:
        try:
            return super().__exit__(kind, value, trace)
        finally:
            self.close()


class _SnapshotCatalog(DatasetCatalog):
    def _get_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        return connection


@contextmanager
def verified_index(storage: Path) -> Generator[HistoryIndex]:
    """Verify source shards without initializing or writing the original catalog."""
    catalog_path = (storage / "catalog.db").resolve()
    if not catalog_path.is_file():
        raise ValueError("Existing authentic catalog required")
    with tempfile.TemporaryDirectory(prefix="phase27-") as temporary:
        snapshot = Path(temporary) / "catalog.db"
        with (
            closing(sqlite3.connect(catalog_path.as_uri() + "?mode=ro", uri=True)) as source,
            closing(sqlite3.connect(snapshot)) as destination,
        ):
            source.backup(destination)
        collector = HistoricalDataCollector(storage_dir=storage, catalog=_SnapshotCatalog(snapshot))
        index = HistoryIndex(collector, Path(temporary) / "index.db")
        try:
            index.rebuild()
            yield index
        finally:
            index.close()


def _utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


def _inside_storage(path: Path, storage: Path) -> bool:
    return path.resolve().is_relative_to(storage.resolve())


def boundary_plan(index: HistoryIndex, count: int, earlier_windows: int) -> dict[str, Any]:
    if type(count) is not int or not 1 <= count <= 500 or earlier_windows not in (1, 2):
        raise ValueError("Count must be 1..500 and earlier windows 1 or 2")
    coverage = index.report()
    oldest = coverage["oldest_ts"]
    if oldest is None or oldest < 28 * 86400:
        raise ValueError("Verified coverage required for documented diagnostic windows")
    # A more recent known interval separates API health from the old boundary.
    # Prefer the week before the newest source candle, away from the final tail.
    control_anchor = max(oldest, coverage["newest_ts"] - 7 * 86400)
    control = [
        row[0]
        for row in index.db.execute(
            "SELECT ts FROM bars WHERE ts >= ? ORDER BY ts LIMIT ?", (control_anchor, count)
        )
    ]
    if len(control) != count or any(
        right - left != 60 for left, right in itertools.pairwise(control)
    ):
        raise ValueError("Known control interval is not fully covered; choose a smaller count")
    control_end = control[-1] + 59
    requests = [
        {"label": "known_catalog_control", "endtime": control_end},
        {"label": "oldest_exact_boundary", "endtime": oldest},
        {"label": "immediately_before_oldest", "endtime": oldest - 1},
        {"label": "seven_days_earlier", "endtime": oldest - 7 * 86400 - 1},
    ]
    if earlier_windows == 2:
        requests.append({"label": "twenty_eight_days_earlier", "endtime": oldest - 28 * 86400 - 1})
    for request in requests:
        request["endtime_utc"] = _utc(int(request["endtime"]))
        request["count"] = count
    return {
        "schema_version": 1,
        "symbol": "EURUSD",
        "active_id": 1,
        "size_seconds": 60,
        "timestamp_unit": "UTC_EPOCH_SECONDS",
        "oldest_catalog_ts": oldest,
        "known_control_start_ts": control[0],
        "catalog_fingerprint_sha256": coverage["fingerprint_sha256"],
        "requests": requests,
        "maximum_candle_requests": len(requests),
        "retries": 0,
        "interpretation": "Count bars ending at to; nominal window does not guarantee coverage",
    }


def plan_boundary(storage: Path, *, count: int = 20, earlier_windows: int = 2) -> dict[str, Any]:
    with verified_index(storage) as index:
        return boundary_plan(index, count, earlier_windows)


def _safe_evidence(value: Any) -> dict[str, Any]:
    """Accept only the connector's fixed evidence vocabulary, even for injected clients."""
    if not isinstance(value, dict):
        return {"correlation": "UNVERIFIED"}
    choices = {
        "frame_type": {"CANDLES", "ERROR", "OTHER", "INVALID"},
        "correlation": {"ABSENT", "MATCHED", "MISMATCH", "UNVERIFIED"},
        "envelope_identity": {"ABSENT", "MATCHED", "MISMATCH"},
    }
    output: dict[str, Any] = {}
    for key, labels in choices.items():
        entry = value.get(key)
        output[key] = entry if isinstance(entry, str) and entry in labels else "UNVERIFIED"
    statuses = {
        "ABSENT",
        "OK",
        "SUCCESS",
        "ERROR",
        "FAILED",
        "FAIL",
        "DENIED",
        "UNAUTHORIZED",
        "RATE_LIMITED",
        "NO_DATA",
        "NOT_FOUND",
        "PRESENT_UNRECOGNIZED",
    }
    for key in ("envelope_status", "payload_status", "envelope_code", "payload_code"):
        entry = value.get(key)
        output[key] = (
            entry
            if (type(entry) is int and -9999 <= entry <= 9999)
            or (isinstance(entry, str) and entry in statuses)
            else "PRESENT_UNRECOGNIZED"
        )
    for key in ("error_field_present", "error_signal_present", "candles_list_present"):
        output[key] = value.get(key) is True
    return output


def _comparison(index: HistoryIndex, request: dict[str, Any], result: Any) -> dict[str, Any]:
    candles = list(result.candle_list)
    evidence = _safe_evidence(getattr(result, "response_evidence", None))
    output: dict[str, Any] = {
        "response": evidence,
        "returned_candle_count": len(candles),
        "rejected_candle_count": len(result.errors),
        "raw_candle_count": getattr(result, "raw_count", len(candles)),
        "earliest_ts": None,
        "latest_ts": None,
        "contradicts_local_catalog": None,
    }
    if len(candles) > request["count"] or result.errors or str(result.source).lower() != "iqoption":
        output["status"] = "INVALID_RESULT"
        return output
    try:
        strict_candles(candles)
    except DataAcquisitionError:
        output["status"] = "INVALID_RESULT"
        return output
    timestamps = [c.from_ts for c in candles]
    if len(set(timestamps)) != len(timestamps):
        output["status"] = "DUPLICATE_RESPONSE"
        return output
    if timestamps:
        output.update(earliest_ts=min(timestamps), latest_ts=max(timestamps))
        output["earliest_utc"] = _utc(min(timestamps))
        output["latest_utc"] = _utc(max(timestamps))
    expected = dict(
        index.db.execute(
            "SELECT ts, signature FROM bars WHERE ts <= ? ORDER BY ts DESC LIMIT ?",
            (request["endtime"], request["count"]),
        )
    )
    conflicts = matched = unknown = 0
    for candle in candles:
        existing = index.db.execute(
            "SELECT signature FROM bars WHERE ts=?", (candle.from_ts,)
        ).fetchone()
        if existing:
            if existing[0] == signature(candle):
                matched += 1
            else:
                conflicts += 1
        else:
            unknown += 1
    missing = len(expected.keys() - set(timestamps))
    outside = any(ts > request["endtime"] for ts in timestamps)
    output.update(
        matched_catalog_candles=matched,
        conflicting_catalog_candles=conflicts,
        candles_without_local_reference=unknown,
        missing_expected_catalog_candles=missing,
        returned_after_request_end=outside,
        contradicts_local_catalog=bool(conflicts or missing),
    )
    explicit_error = evidence.get("error_signal_present") or any(
        evidence.get(key) in {"ERROR", "FAILED", "FAIL", "DENIED", "UNAUTHORIZED", "RATE_LIMITED"}
        for key in ("envelope_status", "payload_status")
    )
    output["status"] = (
        "RESPONSE_ANOMALY"
        if outside or conflicts or explicit_error or evidence.get("envelope_identity") == "MISMATCH"
        else "CORRELATION_UNVERIFIED"
        if evidence.get("correlation") != "MATCHED"
        else "CATALOG_MISMATCH"
        if missing
        else "CANDLES_RETURNED"
        if candles
        else "EMPTY_CANDLES_LIST"
    )
    oldest = index.db.execute("SELECT MIN(ts) FROM bars").fetchone()[0]
    output["older_than_catalog_count"] = sum(ts < oldest for ts in timestamps)
    if (
        not candles
        and any(
            evidence.get(key) not in {"ABSENT", "OK", "SUCCESS"}
            for key in ("envelope_status", "payload_status", "envelope_code", "payload_code")
        )
        and output["status"] == "EMPTY_CANDLES_LIST"
    ):
        output["status"] = "EMPTY_WITH_UNINTERPRETED_STATUS"
    return output


def _conclusion(results: list[dict[str, Any]]) -> str:
    if any(
        r.get("status") == "CANDLES_RETURNED" and r.get("older_than_catalog_count", 0)
        for r in results
    ):
        return "SUCCESSFULLY_RETRIEVABLE"
    if any(
        r.get("status") in {"RESPONSE_ANOMALY", "DUPLICATE_RESPONSE", "INVALID_RESULT"}
        or r.get("response", {}).get("correlation") == "MISMATCH"
        or r.get("response", {}).get("envelope_identity") == "MISMATCH"
        or r.get("response", {}).get("frame_type") in {"ERROR", "INVALID", "OTHER"}
        or (
            r.get("error_type") == "InvalidCandleError"
            and r.get("response", {}).get("correlation") == "MATCHED"
        )
        for r in results
    ):
        return "PROTOCOL_ANOMALY_OBSERVED_CAUSE_UNVERIFIED"
    if results and results[0].get("status") == "CANDLES_RETURNED":
        older = [
            r
            for r in results
            if r["label"]
            in {"immediately_before_oldest", "seven_days_earlier", "twenty_eight_days_earlier"}
        ]
        if older and all(r.get("status") == "EMPTY_CANDLES_LIST" for r in older):
            return "UNAVAILABLE_IN_TESTED_WINDOWS"
    return "STILL_UNVERIFIED"


async def diagnose_boundary(
    storage: Path,
    broker: HistoricalBroker,
    report_path: Path,
    *,
    count: int = 20,
    earlier_windows: int = 2,
) -> dict[str, Any]:
    """Execute the fixed plan once. Attempt accounting is saved before each request."""
    if _inside_storage(report_path, storage):
        raise ValueError("Diagnostic reports must be outside the immutable source storage")
    broker.assert_read_only()
    with verified_index(storage) as index:
        plan = boundary_plan(index, count, earlier_windows)
        report: dict[str, Any] = {
            "mode": "OPERATOR_READ_ONLY_DIAGNOSTIC",
            "plan": plan,
            "candle_requests_attempted": 0,
            "results": [],
            "state": "RUNNING",
            "historical_availability": "STILL_UNVERIFIED",
            "retention_ceiling_confirmed": False,
            "pagination_defect_confirmed": False,
            "catalog_publications": 0,
            "new_authentic_candles_acquired_into_dataset": 0,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, sort_keys=True))
        try:
            for request in plan["requests"]:
                row = {
                    **request,
                    "status": "ATTEMPT_STARTED",
                    "returned_candle_count": None,
                    "earliest_ts": None,
                    "latest_ts": None,
                    "contradicts_local_catalog": None,
                }
                report["results"].append(row)
                report["candle_requests_attempted"] += 1
                atomic_json(report_path, report)
                await asyncio.sleep(1)
                try:
                    broker.assert_read_only()
                    async with asyncio.timeout(30):
                        result = await broker.get_candles(
                            "EURUSD", 60, count, endtime=request["endtime"]
                        )
                    row.update(_comparison(index, request, result))
                except Exception as exc:  # noqa: BLE001 -- sanitized evidence only; no exception text
                    row.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
                    row["response"] = _safe_evidence(getattr(exc, "response_evidence", None))
                    if isinstance(
                        exc, (ConnectionError, LoginError, RequestTimeoutError, TimeoutError)
                    ):
                        report["state"] = "STOPPED_TRANSPORT_OR_AUTH"
                        break
                atomic_json(report_path, report)
                if "RATE_LIMITED" in row.get("response", {}).values():
                    report["state"] = "STOPPED_RATE_LIMIT"
                    break
            else:
                report["state"] = "COMPLETED"
            report["historical_availability"] = _conclusion(report["results"])
            exact: dict[str, Any] = next(
                (r for r in report["results"] if r["label"] == "oldest_exact_boundary"), {}
            )
            report["boundary_observation"] = (
                "OLDEST_OPEN_RETURNED_AT_EQUAL_ENDTIME"
                if exact.get("earliest_ts") == plan["oldest_catalog_ts"]
                else "OLDEST_OPEN_NOT_OBSERVED_AT_EQUAL_ENDTIME_SEMANTICS_UNPROVEN"
            )
            report["untested_windows"] = len(plan["requests"]) - report["candle_requests_attempted"]
            atomic_json(report_path, report)
            return report
        except BaseException:
            report["state"] = "INTERRUPTED"
            atomic_json(report_path, report)
            raise
