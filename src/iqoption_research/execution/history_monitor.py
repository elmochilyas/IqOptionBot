"""Bounded snapshot comparisons; history discovery is never a live opening event."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from iqoption_research.execution.practice_history import (
    normalize_history_asset,
    normalize_ticket_aliases,
    normalized_history_records,
)

_CAPACITY = 512
_TERMINAL = {"closed", "expired", "settled"}


@dataclass(slots=True)
class HistoryRecord:
    number: int
    fields: dict[str, Any] = field(default_factory=dict)
    conflict: bool = False
    baseline: bool = False


class HistorySnapshotMonitor:
    """Retain validated tickets internally and publish only local numbers/counts.

    First successful snapshot establishes the baseline, even after earlier query
    failures. Later discoveries can be delayed or previously truncated history;
    neither their discovery time nor their lifecycle is a WebSocket event.
    """

    def __init__(self, asset: str, limit: int) -> None:
        self.asset = normalize_history_asset(asset)
        self.limit = limit
        self._records: dict[str, HistoryRecord] = {}
        self._excluded: set[str] = set()
        self._ambiguous: set[str] = set()
        self._incomplete: set[str] = set()
        self._baseline_incomplete: set[str] = set()
        self._identity_capacity_exhausted = False
        self._practice_id: str | None = None
        self._previous: set[str] = set()
        self._baseline = False
        self._gap_pending = False
        self._next_number = 1
        self._counts: Counter[str] = Counter()
        self._recent: deque[dict[str, Any]] = deque(maxlen=20)

    def gap(self) -> None:
        """A disconnect/query failure breaks coverage, not record identity."""
        self._counts["coverage_interruptions"] += 1
        self._gap_pending = True

    def _compatible(self, row: dict[str, Any]) -> bool:
        product = row["product_type"]
        products = (
            {"blitz-option"}
            if self.asset == "EURUSD_OP"
            else {
                "binary-option",
                "turbo-option",
            }
        )
        return row["symbol"] in {None, self.asset} and product in {None, *products}

    def snapshot(self, response: dict[str, Any], practice_id: str) -> dict[str, Any]:
        # Validate the whole envelope before mutating baseline or retained state.
        rows = normalized_history_records(response)
        practice_id = normalize_ticket_aliases({"id": practice_id}, "id") or ""
        if not practice_id or self._practice_id not in (None, practice_id):
            raise ValueError("History monitor wallet identity unverified")
        self._practice_id = practice_id
        grouped: dict[str, list[tuple[dict[str, Any], bool]]] = {}
        counts: Counter[str] = Counter()
        gaps: list[str] = []
        initial = not self._baseline
        if len(rows) >= self.limit:
            gaps.append("LIMIT_REACHED_POSSIBLE_TRUNCATION")
        if self._gap_pending:
            gaps.append("PRIOR_COVERAGE_INTERRUPTION")
        for row, invalid in rows:
            ticket = row["ticket_id"]
            if not self._compatible(row) and ticket not in self._records:
                continue
            if ticket is None:
                counts["incomplete_records"] += 1
                continue
            grouped.setdefault(ticket, []).append((row, invalid))

        present: set[str] = set()

        def remember(bucket: set[str], ticket: str) -> None:
            if len(bucket) < _CAPACITY:
                bucket.add(ticket)
            elif ticket not in bucket:
                self._identity_capacity_exhausted = True
                gaps.append("MONITOR_CAPACITY_EXCEEDED")

        for ticket, candidates in grouped.items():
            row, invalid = candidates[0]
            record = self._records.get(ticket)
            # Ticket namespace conflicts include different-wallet copies; filter
            # only after comparing duplicates, never attribute by request scope.
            if any(other != row for other, _ in candidates):
                counts["ambiguous_records"] += 1
                remember(self._ambiguous, ticket)
                if record is not None:
                    record.conflict = True
                continue
            counts["duplicate_rows"] += len(candidates) - 1
            invalid |= any(bad for _, bad in candidates)
            if not self._compatible(row):
                counts["ambiguous_records"] += 1
                remember(self._ambiguous, ticket)
                if record is not None:
                    record.conflict = True
                continue
            wallet = row["user_balance_id"]
            wallet = normalize_ticket_aliases({"id": wallet}, "id") if wallet else None
            row["user_balance_id"] = wallet
            if wallet is not None and wallet != practice_id:
                counts["other_wallet_records_excluded"] += 1
                remember(self._excluded, ticket)
                if record is not None:
                    record.conflict = True
                    counts["ambiguous_records"] += 1
                continue
            if ticket in self._excluded or ticket in self._ambiguous:
                counts["ambiguous_records"] += 1
                if record is not None:
                    record.conflict = True
                continue
            if invalid or any(
                row[key] is None
                for key in (
                    "symbol",
                    "product_type",
                    "stake",
                    "opened_at_utc",
                    "expires_at_utc",
                    "status",
                )
            ):
                counts["incomplete_records"] += 1
                remember(self._incomplete, ticket)
                if initial:
                    remember(self._baseline_incomplete, ticket)
                continue
            if len(present) < _CAPACITY:
                present.add(ticket)
            else:
                gaps.append("MONITOR_CAPACITY_EXCEEDED")
            if record is None:
                if len(self._records) >= _CAPACITY or self._identity_capacity_exhausted:
                    counts["capacity_excluded_records"] += 1
                    continue
                record = HistoryRecord(
                    self._next_number,
                    baseline=initial or ticket in self._baseline_incomplete,
                )
                self._next_number += 1
                self._records[ticket] = record
                discovery = (
                    "baseline_records"
                    if initial
                    else "newly_identified_positions"
                    if ticket in self._incomplete
                    else "new_positions"
                )
                counts[discovery] += 1
                if row["status"] in _TERMINAL:
                    counts["terminal_on_first_discovery"] += 1
            else:
                before = dict(record.fields)
                if record.conflict:
                    counts["ambiguous_records"] += 1
                    continue
                # Omitted fields in stale snapshots cannot erase richer evidence.
                # A terminal status cannot be reversed by delayed open evidence.
                for key, value in row.items():
                    if value is None or key == "ticket_id":
                        continue
                    previous = before.get(key)
                    if previous is not None and previous != value:
                        if key == "status" and previous == "open" and value in _TERMINAL:
                            counts["closed_positions"] += 1
                        elif key == "status" and previous in _TERMINAL and value == "open":
                            counts["stale_records"] += 1
                            continue
                        elif key == "status" and previous in _TERMINAL and value in _TERMINAL:
                            # Different terminal labels do not prove another close.
                            continue
                        else:
                            record.conflict = True
                            remember(self._ambiguous, ticket)
                if record.conflict:
                    counts["ambiguous_records"] += 1
                    continue
                changed = any(
                    value is not None
                    and key != "ticket_id"
                    and before.get(key) != value
                    and not (key == "status" and before.get(key) in _TERMINAL)
                    for key, value in row.items()
                )
                counts["updated_positions" if changed else "unchanged_positions"] += 1
            for key, value in row.items():
                if value is not None and key != "ticket_id":
                    if key == "status" and record.fields.get(key) in _TERMINAL:
                        continue
                    record.fields[key] = value

        missing = self._previous - present
        if missing:
            counts["previous_records_missing"] = len(missing)
            gaps.append("PREVIOUS_SNAPSHOT_RECORDS_MISSING")
        if self._previous and present and not self._previous.intersection(present):
            gaps.append("NO_SNAPSHOT_OVERLAP")
        if counts["capacity_excluded_records"]:
            gaps.append("MONITOR_CAPACITY_EXCEEDED")
        status = (
            "HISTORY_AMBIGUOUS"
            if counts["ambiguous_records"]
            else "HISTORY_INCOMPLETE"
            if counts["incomplete_records"]
            else "HISTORY_BASELINE_ESTABLISHED"
            if initial
            else "HISTORY_NEW_POSITION_DETECTED"
            if counts["new_positions"]
            else "HISTORY_POSITION_UPDATED"
            if counts["updated_positions"] or counts["newly_identified_positions"]
            else "HISTORY_NO_CHANGE"
        )
        outcome = {
            "snapshot_number": self._counts["successful_snapshots"] + 1,
            "status": status,
            "counts": dict(counts),
            "coverage": "HISTORY_COVERAGE_UNVERIFIED",
            "capacity_exhausted": self._identity_capacity_exhausted
            or len(self._records) >= _CAPACITY,
            "gap_signals": sorted(set(gaps)),
            "history_may_be_truncated": True,
        }
        self._baseline = True
        self._gap_pending = False
        self._previous = present
        self._counts.update(counts)
        self._counts["successful_snapshots"] += 1
        self._counts["snapshots_with_gap_signals"] += bool(gaps)
        self._recent.append(outcome)
        return outcome

    def report(self) -> dict[str, Any]:
        return {
            "mode": "HISTORY_SNAPSHOTS_ONLY_NOT_LIVE_EVENTS",
            "baseline_status": "ESTABLISHED" if self._baseline else "UNVERIFIED",
            "coverage": "HISTORY_COVERAGE_UNVERIFIED",
            "counts": dict(self._counts),
            "retained_records": len(self._records),
            "conflicting_records": sum(record.conflict for record in self._records.values()),
            "position_samples": [
                {
                    "history_record_number": record.number,
                    "first_seen_in_baseline": record.baseline,
                    "evidence_status": "HISTORY_AMBIGUOUS"
                    if record.conflict
                    else "HISTORY_IDENTIFIED",
                    "wallet_attribution": "ROUTING_UNVERIFIED"
                    if record.conflict or record.fields.get("user_balance_id") is None
                    else "PRACTICE_PER_RECORD",
                    "closed_history_evidence": record.fields.get("status") in _TERMINAL,
                    "settlement_status": "SETTLEMENT_UNCONFIRMED",
                    "opening_time_reported": record.fields.get("opened_at_utc") is not None,
                    "expiry_time_reported": record.fields.get("expires_at_utc") is not None,
                    "close_time_reported": record.fields.get("closed_at_utc") is not None,
                }
                for record in list(self._records.values())[:20]
            ],
            "recent_snapshots": list(self._recent),
        }
