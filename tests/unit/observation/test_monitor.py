"""Snapshot discovery, sparse evidence, bounded coverage, and privacy invariants."""

from __future__ import annotations

import json
from typing import Any

import pytest

import iqoption_research.execution.history_monitor as module
from iqoption_research.execution.blitz_diagnostics import BlitzSchemaDiagnostics
from iqoption_research.execution.history_monitor import HistorySnapshotMonitor

PRACTICE = "710001"
REAL = "710002"
TICKET = 810001


def row(ticket: int = TICKET, **changes: Any) -> dict[str, Any]:
    return {
        "id": [ticket],
        "active": "EURUSD_OP",
        "active_id": 1861,
        "option_type": "blitz-option",
        "amount": 1000,
        "created": 1700000000,
        "expired": 1700000005,
        "win": "win",
        "win_amount": 1850,
        **changes,
    }


def history(*rows: dict[str, Any]) -> dict[str, Any]:
    return {"name": "options", "status": 0, "msg": {"closed_options": list(rows)}}


def monitor(limit: int = 100) -> HistorySnapshotMonitor:
    return HistorySnapshotMonitor("EURUSD_OP", limit)


def test_baseline_repeats_new_discovery_updates_and_stale_sparse_evidence() -> None:
    subject = monitor()
    first = row()
    assert subject.snapshot(history(first), PRACTICE)["status"] == "HISTORY_BASELINE_ESTABLISHED"
    assert subject.snapshot(history(first, first), PRACTICE)["status"] == "HISTORY_NO_CHANGE"
    new = row(TICKET + 1)
    assert (
        subject.snapshot(history(first, new), PRACTICE)["status"] == "HISTORY_NEW_POSITION_DETECTED"
    )
    rich = {**new, "close_time": 1700000006, "user_balance_id": PRACTICE}
    update = subject.snapshot(history(first, rich), PRACTICE)
    assert update["status"] == "HISTORY_POSITION_UPDATED"
    assert subject.snapshot(history(first, new), PRACTICE)["status"] == "HISTORY_NO_CHANGE"
    report = subject.report()
    assert report["counts"]["new_positions"] == 1
    assert report["counts"]["updated_positions"] == 1
    assert report["counts"]["terminal_on_first_discovery"] == 2
    assert report["retained_records"] == 2
    assert report["position_samples"][1]["close_time_reported"]
    assert report["position_samples"][1]["wallet_attribution"] == "PRACTICE_PER_RECORD"
    assert report["position_samples"][1]["settlement_status"] == "SETTLEMENT_UNCONFIRMED"


def test_missing_records_and_no_overlap_are_gaps_never_new_closures() -> None:
    subject = monitor(limit=1)
    first = subject.snapshot(history(row()), PRACTICE)
    assert first["gap_signals"] == ["LIMIT_REACHED_POSSIBLE_TRUNCATION"]
    second = subject.snapshot(history(row(TICKET + 1)), PRACTICE)
    assert second["counts"]["previous_records_missing"] == 1
    assert "NO_SNAPSHOT_OVERLAP" in second["gap_signals"]
    assert "PREVIOUS_SNAPSHOT_RECORDS_MISSING" in second["gap_signals"]
    assert subject.report()["retained_records"] == 2
    assert subject.snapshot(history(row()), PRACTICE)["status"] == "HISTORY_NO_CHANGE"
    assert subject.report()["counts"].get("closed_positions", 0) == 0
    assert second["coverage"] == "HISTORY_COVERAGE_UNVERIFIED"


@pytest.mark.parametrize("missing", ["created", "expired", "amount", "active", "option_type"])
def test_incomplete_then_identified_is_not_counted_as_new_trade(missing: str) -> None:
    subject = monitor()
    incomplete = row()
    incomplete.pop(missing)
    if missing == "active":
        incomplete.pop("active_id")
    result = subject.snapshot(history(incomplete), PRACTICE)
    assert result["status"] == "HISTORY_INCOMPLETE"
    assert subject.snapshot(history(row()), PRACTICE)["status"] == "HISTORY_POSITION_UPDATED"
    assert subject.report()["counts"].get("new_positions", 0) == 0
    assert subject.report()["counts"]["newly_identified_positions"] == 1
    assert subject.report()["position_samples"][0]["first_seen_in_baseline"] is True


@pytest.mark.parametrize(
    "ticket", [[TICKET, TICKET + 1], [True], True, 0, -1, 1.2, "1.2", "private-id"]
)
def test_invalid_or_ambiguous_tickets_never_become_positions(ticket: Any) -> None:
    subject = monitor()
    result = subject.snapshot(history(row(id=ticket)), PRACTICE)
    assert result["status"] == "HISTORY_INCOMPLETE"
    assert subject.report()["retained_records"] == 0


def test_ticket_scalar_and_singleton_normalization_does_not_double_count() -> None:
    subject = monitor()
    subject.snapshot(history(row()), PRACTICE)
    assert subject.snapshot(history(row(id="000810001")), PRACTICE)["status"] == "HISTORY_NO_CHANGE"
    assert subject.report()["retained_records"] == 1


def test_conflicting_duplicates_are_latched_without_record_promotion() -> None:
    subject = monitor()
    result = subject.snapshot(history(row(), row(amount=2000)), PRACTICE)
    assert result["status"] == "HISTORY_AMBIGUOUS"
    assert subject.snapshot(history(row()), PRACTICE)["status"] == "HISTORY_AMBIGUOUS"
    assert subject.report()["retained_records"] == 0


def test_conflict_on_existing_ticket_never_clears() -> None:
    subject = monitor()
    subject.snapshot(history(row()), PRACTICE)
    assert subject.snapshot(history(row(amount=2000)), PRACTICE)["status"] == "HISTORY_AMBIGUOUS"
    assert subject.snapshot(history(row()), PRACTICE)["status"] == "HISTORY_AMBIGUOUS"
    assert subject.report()["conflicting_records"] == 1


def test_real_records_cannot_later_be_relabelled_practice() -> None:
    subject = monitor()
    subject.snapshot(history(row(user_balance_id=REAL)), PRACTICE)
    assert subject.report()["retained_records"] == 0
    assert (
        subject.snapshot(history(row(user_balance_id=PRACTICE)), PRACTICE)["status"]
        == "HISTORY_AMBIGUOUS"
    )
    assert subject.report()["retained_records"] == 0


def test_late_real_evidence_taints_previously_unattributed_record() -> None:
    subject = monitor()
    subject.snapshot(history(row()), PRACTICE)
    subject.snapshot(history(row(user_balance_id=REAL)), PRACTICE)
    assert subject.report()["conflicting_records"] == 1
    assert subject.report()["position_samples"][0]["wallet_attribution"] == "ROUTING_UNVERIFIED"


def test_wallet_change_and_invalid_envelope_do_not_reset_baseline() -> None:
    subject = monitor()
    with pytest.raises(ValueError):
        subject.snapshot({"name": "options", "msg": []}, PRACTICE)
    assert subject.report()["baseline_status"] == "UNVERIFIED"
    subject.gap()
    result = subject.snapshot(history(row()), PRACTICE)
    assert "PRIOR_COVERAGE_INTERRUPTION" in result["gap_signals"]
    with pytest.raises(ValueError, match="wallet identity unverified"):
        subject.snapshot(history(row()), REAL)
    assert subject.report()["counts"]["successful_snapshots"] == 1


def test_capacity_exhaustion_and_recent_diagnostics_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "_CAPACITY", 2)
    subject = monitor()
    subject.snapshot(history(row(), row(TICKET + 1)), PRACTICE)
    result = subject.snapshot(history(row(TICKET + 2)), PRACTICE)
    assert "MONITOR_CAPACITY_EXCEEDED" in result["gap_signals"]
    for _ in range(30):
        subject.snapshot(history(row()), PRACTICE)
    assert subject.report()["retained_records"] == 2
    assert len(subject.report()["recent_snapshots"]) == 20


def test_tombstone_overflow_fails_closed_for_new_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "_CAPACITY", 1)
    subject = monitor()
    subject.snapshot(
        history(row(user_balance_id=REAL), row(TICKET + 1, user_balance_id=REAL)), PRACTICE
    )
    result = subject.snapshot(history(row(TICKET + 1, user_balance_id=PRACTICE)), PRACTICE)
    assert result["counts"]["capacity_excluded_records"] == 1
    assert subject.report()["retained_records"] == 0


def test_unsupported_open_blitz_history_and_other_instruments_are_not_promoted() -> None:
    subject = monitor()
    response = {"name": "options", "msg": {"open_options": [row(id=TICKET)]}}
    assert subject.snapshot(response, PRACTICE)["status"] == "HISTORY_INCOMPLETE"
    assert subject.report()["retained_records"] == 0
    subject.snapshot(
        history(row(active="EURUSD", active_id=1, option_type="turbo-option")), PRACTICE
    )
    assert subject.report()["retained_records"] == 0


def test_classic_out_of_order_open_history_does_not_reopen_closed_record() -> None:
    subject = HistorySnapshotMonitor("EURUSD", 100)
    opened = row(
        id=TICKET, active="EURUSD", active_id=1, option_type="turbo-option", direction="call"
    )
    opened.pop("win")
    opened.pop("win_amount")
    response = {"name": "options", "msg": {"open_options": [opened]}}
    subject.snapshot(response, PRACTICE)
    subject.snapshot(history({**opened, "win": "win"}), PRACTICE)
    assert subject.snapshot(response, PRACTICE)["status"] == "HISTORY_NO_CHANGE"
    assert subject.report()["counts"]["closed_positions"] == 1
    assert subject.report()["position_samples"][0]["closed_history_evidence"]


def test_report_has_no_raw_or_identifying_values() -> None:
    subject = monitor()
    subject.snapshot(
        history(
            row(user_balance_id=PRACTICE, ssid="secret-session", email="private@example.invalid")
        ),
        PRACTICE,
    )
    encoded = json.dumps(subject.report())
    for secret in (
        PRACTICE,
        REAL,
        str(TICKET),
        "1000",
        "1850",
        "1700000000",
        "secret",
        "private@",
        "ssid",
    ):
        assert secret not in encoded


@pytest.mark.parametrize("status", ["open", "closed"])
def test_cached_sdk_reference_schema_remains_metadata_until_gateway_is_verified(
    status: str,
) -> None:
    diagnostics = BlitzSchemaDiagnostics()
    diagnostics.on_frame(
        "position-changed",
        {
            "microserviceName": "portfolio",
            "msg": {
                "id": "private-portfolio-id",
                "external_id": TICKET,
                "user_balance_id": PRACTICE,
                "active_id": 1861,
                "instrument_type": "blitz-option",
                "status": status,
                "open_time": 1700000000000,
                "close_time": 1700000005000,
                "invest": 1000,
                "raw_event": {"binary_options_option_changed1": {"expiration_time": 1700000005}},
            },
        },
    )
    report = diagnostics.report()
    assert report["schema_status"] == "LIVE_SCHEMA_UNVERIFIED"
    assert report["reference_applicability"] == "IQOPTION_GATEWAY_UNVERIFIED"
    assert report["field_presence_counts"]["open_time"] == 1
    assert report["field_presence_counts"]["expiration_time"] == 1
    for secret in (str(TICKET), PRACTICE, "1700000000", "1000", "private-portfolio"):
        assert secret not in json.dumps(report)
