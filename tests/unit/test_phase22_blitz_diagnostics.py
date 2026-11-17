"""Unverified protocol candidates cannot become live trade evidence."""

from __future__ import annotations

import json
from typing import Any

import pytest

from iqoption_research.execution.blitz_diagnostics import BlitzSchemaDiagnostics
from iqoption_research.execution.practice_history import normalize_ticket_aliases
from iqoption_research.execution.practice_observer import PassivePracticeObserver


@pytest.mark.parametrize("value", [1, "1", "0001", " 1 "])
def test_positive_scalar_ticket_normalization(value: Any) -> None:
    assert normalize_ticket_aliases({"id": value, "option_id": 1}, "id", "option_id") == "1"


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, 1.0, "1.0", "0", "\u0661", [], [1], {}, None]
)
def test_unsupported_ticket_is_not_normalized(value: Any) -> None:
    assert normalize_ticket_aliases({"id": value}, "id") is None


def test_conflicting_ticket_aliases() -> None:
    assert normalize_ticket_aliases({"id": 1, "option_id": 2}, "id", "option_id") is None


def test_community_nested_shape_is_metadata_only_and_never_leaks() -> None:
    diagnostics = BlitzSchemaDiagnostics()
    frame = {
        "name": "position-changed",
        "microserviceName": "portfolio",
        "msg": {
            "external_id": "000818144120",
            "instrument_type": "blitz-option",
            "user_balance_id": 764508381,
            "private-key-ssid": "private-value-password",
            "raw_event": {
                "binary_options_option_changed1": {
                    "active_id": 1861,
                    "direction": "call",
                    "result": "opened",
                    "id": [818144120],
                }
            },
        },
    }
    for _ in range(2):
        diagnostics.on_frame("position-changed", frame)
    report = diagnostics.report()
    assert report["schema_status"] == "LIVE_SCHEMA_UNVERIFIED"
    assert report["counts"]["candidate_frames"] == 2
    assert report["counts"]["objects_with_explicit_blitz_label"] == 2
    assert report["identifier_shape_counts"] == {
        "external_id:POSITIVE_SCALAR": 2,
        "id:UNSUPPORTED": 2,
    }
    # Candidate duplicates are counts, never deduplicated or linked as positions.
    assert report["position_ticket_correlation"] == "UNVERIFIED_NO_SUPPORTED_LIVE_SCHEMA"
    encoded = json.dumps(report)
    for secret in ("818144120", "764508381", "private-key", "private-value", "1861"):
        assert secret not in encoded


def test_unknown_topics_and_response_payloads_are_not_inspected() -> None:
    diagnostics = BlitzSchemaDiagnostics()
    diagnostics.on_frame("secret-topic", {"msg": {"option_id": 123}})
    diagnostics.on_frame("position-changed", {"request_id": "x", "msg": {"option_id": 123}})
    diagnostics.on_frame("position-changed", {"msg": []})
    report = diagnostics.report()
    assert report["counts"] == {"correlated_responses_excluded": 1, "non_object_bodies": 1}
    assert report["field_presence_counts"] == {}
    assert "secret" not in json.dumps(report)


def test_metadata_storage_has_fixed_size_vocabulary() -> None:
    diagnostics = BlitzSchemaDiagnostics()
    for i in range(2000):
        diagnostics.on_frame("position-changed", {"msg": {str(i): str(i), "id": i + 1}})
    report = diagnostics.report()
    assert report["field_presence_counts"] == {"id": 2000}
    assert report["identifier_shape_counts"] == {"id:POSITIVE_SCALAR": 2000}


@pytest.mark.parametrize(
    "flags",
    [
        {"is_active": "false"},
        {"active": 1},
        {"is_default": []},
        {"is_active": True, "active": False},
    ],
)
def test_wallet_authority_rejects_coercible_or_conflicting_flags(flags: dict[str, Any]) -> None:
    row = {"id": 101, "type": 4, "amount": 100, **flags}
    assert PassivePracticeObserver._parse_balance_entry(row) is None


def test_wallet_authority_accepts_only_explicit_agreeing_flags() -> None:
    row = {"id": 101, "type": 4, "amount": 100, "active": True, "is_active": True}
    balance = PassivePracticeObserver._parse_balance_entry(row)
    assert balance is not None and balance.raw_is_active is True
