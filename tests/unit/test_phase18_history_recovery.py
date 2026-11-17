"""Regression tests for read-only history and passive observer diagnostics."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.execution.practice_history import (
    extract_rows,
    reconcile_history,
    recover_history,
)
from iqoption_research.execution.practice_observer import (
    PassivePracticeObserver,
    SubscriptionStatus,
)
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.models.account import AccountType, Balance


def row(**updates: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": 42,
        "active_id": 1,
        "instrument_type": "turbo-option",
        "amount": 1000,
        "created_at": 1794904000,
        "expiration_time": 1794904060,
        "direction": "call",
        "status": "closed",
        "result": "win",
        "payout": 1800,
        "net_profit": 800,
        "user_balance_id": 101,
        "ssid": "secret-value",
    }
    result.update(updates)
    return result


def response(*rows: dict[str, Any]) -> dict[str, Any]:
    return {"name": "options", "msg": {"result": {"closed_options": list(rows)}}}


def match(
    message: dict[str, Any],
    *,
    asset: str = "EURUSD",
    ticket_id: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return reconcile_history(
        message,
        asset=asset,
        stake=1000,
        start=1794903900,
        end=1794904100,
        practice_id="101",
        ticket_id=ticket_id,
        diagnostics=diagnostics,
    )


def test_recover_and_sanitize() -> None:
    result = match(response(row()))
    assert result["position_status"] == "POSITION_CONFIRMED"
    assert result["wallet_status"] == "PRACTICE_WALLET_CONFIRMED"
    assert result["settlement_status"] == "SETTLEMENT_CONFIRMED"
    assert result["candidates"][0]["realized_profit"] == 800
    assert "secret-value" not in json.dumps(result)


def test_missing_identity_and_conflicts() -> None:
    missing = match(response(row(user_balance_id=None)))
    assert missing["wallet_status"] == "ROUTING_UNVERIFIED"
    multiple = match(response(row(), row(id=43)))
    assert multiple["position_status"] == "MULTIPLE_OR_CONFLICTING_CANDIDATES"
    assert match(response(row(), row()))["candidate_count"] == 1
    assert match(response(row(), row(direction="put")))["position_status"] == (
        "MULTIPLE_OR_CONFLICTING_CANDIDATES"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"active_id": 76},
        {"instrument_type": "digital-option"},
        {"amount": 999},
        {"amount": 1000.0001},
    ],
)
def test_mismatch_is_not_confirmed(change: dict[str, Any]) -> None:
    assert match(response(row(**change)))["position_status"] == "HISTORY_NOT_FOUND"


@pytest.mark.parametrize("change", [{"instrument_type": None}, {"created_at": None}])
def test_incomplete_is_not_confirmed_or_absent(change: dict[str, Any]) -> None:
    result = match(response(row(**change)))
    assert result["position_status"] == "INCOMPLETE_CANDIDATES"
    assert result["wallet_status"] == "ROUTING_UNVERIFIED"


def test_invalid_history_is_not_not_found() -> None:
    with pytest.raises(ValueError):
        match({"name": "options", "msg": {"error": "failure"}})


@pytest.mark.parametrize(
    ("product_fields", "expected"),
    [
        ({"type_name": "turbo"}, "turbo-option"),
        ({"option_type_id": 1}, "binary-option"),
        ({"type_name": "TURBO", "option_type_id": 3}, "turbo-option"),
    ],
)
def test_classic_closed_option_with_singleton_ticket_and_verified_aliases(
    product_fields: dict[str, Any], expected: str
) -> None:
    item = {
        "id": [42],
        "active": "EUR/USD",
        "act_id": 1,
        "dir": "PUT",
        "invest": "1000",
        "created": 1794904000,
        "expired": 1794904060000,
        "win": "loose",
        "balance_id": "101",
        **product_fields,
    }
    result = match({
        "name": "options",
        "status": 0,
        "msg": {"closed_options": [item], "open_options": []},
    })
    assert result["history_status"] == "HISTORY_CONFIRMED"
    assert result["wallet_status"] == "PRACTICE_WALLET_CONFIRMED"
    assert result["settlement_status"] == "SETTLEMENT_CONFIRMED"
    assert result["candidates"][0] == {
        "ticket_id": "42",
        "symbol": "EURUSD",
        "product_type": expected,
        "direction": "put",
        "stake": 1000,
        "opened_at_utc": 1794904000,
        "expires_at_utc": 1794904060,
        "status": "closed",
        "result": "loose",
        "payout": None,
        "realized_profit": None,
        "user_balance_id": "101",
    }


@pytest.mark.parametrize(
    ("updates", "missing"),
    [
        ({"id": [42, 43], "option_id": 42}, "ticket_id"),
        ({"id": [], "index": 42, "external_id": "42"}, "ticket_id"),
        ({"id": [True]}, "ticket_id"),
        ({"id": [42], "option_id": 43}, "ticket_id"),
        ({"active": "EURUSD-OTC", "active_id": 1}, "symbol"),
        ({"instrument_type": None, "option_type_id": "3"}, "product_type"),
        ({"instrument_type": None, "option_type_id": 6}, "product_type"),
        ({"instrument_type": None, "option_type_id": True}, "product_type"),
        ({"instrument_type": "turbo-option", "option_type_id": 1}, "product_type"),
        ({"direction": "higher"}, "direction"),
        ({"status": "open"}, "status"),
        ({"status": "garbage"}, "status"),
        ({"created_at": "2026-11-17T09:25:30"}, "opened_at_utc"),
        ({"expiration_time": 1794903999}, "expires_at_utc"),
        ({"expiration_time": True}, "expires_at_utc"),
    ],
)
def test_unverified_record_critical_fields(updates: dict[str, Any], missing: str) -> None:
    result = match(response(row(**updates)))
    assert result["history_status"] == "HISTORY_UNVERIFIED"
    assert result["position_status"] == "INCOMPLETE_CANDIDATES"
    assert result["incomplete_candidates"][0][missing] is None


def test_wallet_identity_requires_a_matching_unambiguous_id() -> None:
    for hints in (
        {"user_balance_id": None, "is_demo": True, "balance_type_id": 4},
        {"user_balance_id": 101, "balance_id": 102},
        {"user_balance_id": [101]},
    ):
        result = match(response(row(**hints)))
        assert result["position_status"] == "POSITION_CONFIRMED"
        assert result["wallet_status"] == "ROUTING_UNVERIFIED"
    assert match(response(row(user_balance_id=102)))["wallet_status"] == "WALLET_MISMATCH"


def test_lookup_keeps_eurusd_and_otc_distinct() -> None:
    otc = row(active="EURUSD-OTC", active_id=76)
    assert match(response(otc))["history_status"] == "HISTORY_NOT_FOUND"
    result = reconcile_history(
        response(otc, row(id=43)),
        asset="EURUSD-OTC",
        stake=1000,
        start=1794903900,
        end=1794904100,
        practice_id="101",
    )
    assert result["history_status"] == "HISTORY_CONFIRMED"
    assert result["candidates"][0]["symbol"] == "EURUSD_OTC"


def test_lifecycle_and_explicit_final_result_are_both_needed_for_settlement() -> None:
    for result in (None, "unknown"):
        closed = match(response(row(status=None, result=result)))
        assert closed["candidates"][0]["status"] == "closed"
        assert closed["settlement_status"] == "SETTLEMENT_UNCONFIRMED"
    opened = match({
        "name": "options",
        "msg": {"open_options": [row(status=None, result="equal")]},
    })
    assert opened["candidates"][0]["status"] == "open"
    assert opened["settlement_status"] == "SETTLEMENT_UNCONFIRMED"


@pytest.mark.parametrize(
    ("updates", "expected_payout"),
    [
        ({"win": "win", "win_amount": 1800}, 1800),
        ({"win": "equal", "win_amount": 1800}, None),
        ({"win": "win", "win_amount": True}, None),
        ({"win": "win", "win_amount": -1}, None),
        ({"win": "win", "win_amount": 1800, "payout": 1700}, None),
    ],
)
def test_closed_history_win_amount_uses_only_supported_gross_payout(
    updates: dict[str, Any], expected_payout: float | None
) -> None:
    item = row(result=None, net_profit=None)
    item.pop("payout")
    item.update(updates)
    result = match(response(item))
    assert result["candidates"][0]["payout"] == expected_payout
    assert result["candidates"][0]["realized_profit"] is None


def blitz_row(**updates: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "active": "EURUSD_OP",
        "active_id": 1861,
        "amount": 1000,
        "created": 1794904000,
        "expired": 1794904060,
        "id": [42],
        "option_type": "blitz-option",
        "win": "win",
        "win_amount": 1800,
    }
    item.update(updates)
    return item


def test_observed_blitz_shape_matches_position_only() -> None:
    diagnostics: dict[str, Any] = {}
    item = blitz_row()
    assert match(response(item))["history_status"] == "HISTORY_NOT_FOUND"
    assert match(response(item), asset="EURUSD-OTC")["history_status"] == "HISTORY_NOT_FOUND"
    assert not InstrumentRegistry.is_supported("EURUSD-op")
    result = match(response(item), asset="EURUSD-op", diagnostics=diagnostics)
    assert result["history_status"] == "HISTORY_POSITION_MATCHED"
    assert result["position_status"] == "POSITION_CONFIRMED"
    assert result["wallet_status"] == "ROUTING_UNVERIFIED"
    assert result["settlement_status"] == "SETTLEMENT_UNCONFIRMED"
    assert result["incomplete_candidates"] == []
    assert result["candidates"] == [
        {
            "ticket_id": "42",
            "symbol": "EURUSD_OP",
            "product_type": "blitz-option",
            "direction": None,
            "stake": 1000,
            "opened_at_utc": 1794904000,
            "expires_at_utc": 1794904060,
            "status": "closed",
            "result": "win",
            "payout": None,
            "realized_profit": None,
            "user_balance_id": None,
        }
    ]
    assert diagnostics["record_samples"][0]["identity_field_categories"] == {
        "active": "EURUSD_OP",
        "active_id": "EURUSD_OP",
        "option_type": "blitz-option",
    }
    assert diagnostics["record_samples"][0]["missing_normalized_fields"]["payout"] == (
        "BLITZ_WIN_AMOUNT_SEMANTICS_UNVERIFIED"
    )


@pytest.mark.parametrize("ticket", [42, "42", [42], "0042"])
def test_blitz_single_ticket_formats_agree_with_scalar_alias(ticket: Any) -> None:
    result = match(
        response(blitz_row(id=ticket, option_id=42)), asset="EURUSD_OP", ticket_id="0042"
    )
    assert result["candidates"][0]["ticket_id"] == "42"
    assert result["history_status"] == "HISTORY_POSITION_MATCHED"


@pytest.mark.parametrize("ticket", [[], [42, 43], ["42"], [True], [42.0], [0], {}, True, 42.0, "x"])
def test_blitz_invalid_ticket_is_not_rescued_by_a_scalar_alias(ticket: Any) -> None:
    result = match(response(blitz_row(id=ticket, option_id=42)), asset="EURUSD_OP")
    assert result["position_status"] == "INCOMPLETE_CANDIDATES"
    assert result["incomplete_candidates"][0]["ticket_id"] is None


@pytest.mark.parametrize(
    "updates",
    [
        {"option_id": 43},
        {"active_id": 1},
        {"active_id": 76},
        {"option_type": None},
        {"option_type_id": 3},
        {"stake": 999},
        {"open_time": 1794904090},
        {"expired": None},
        {"expired": 1794904000},
        {"expire_at": 1794904080},
        {"status": "open"},
        {"direction": "put", "dir": "call"},
        {"result": "loss"},
        {"payout": 1800, "payout_amount": 1700},
        {"realized_profit": 800, "pnl": 700},
        {"balance_id": 101, "user_balance_id": 102},
    ],
)
def test_blitz_incomplete_or_conflicting_evidence_blocks_a_position_match(
    updates: dict[str, Any],
) -> None:
    result = match(response(blitz_row(**updates)), asset="EURUSD_OP")
    assert result["position_status"] == "INCOMPLETE_CANDIDATES"
    assert result["candidate_count"] == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"option_type": "turbo-option"},
        {"amount": 999},
        {"created": 1794903899},
        {"active": "EURUSD", "active_id": 1},
        {"active": "EURUSD-OTC", "active_id": 76},
    ],
)
def test_blitz_search_rejects_explicit_criteria_mismatches(updates: dict[str, Any]) -> None:
    result = match(response(blitz_row(**updates)), asset="EURUSD_OP")
    assert result["history_status"] == "HISTORY_NOT_FOUND"


@pytest.mark.parametrize(
    "other",
    [blitz_row(id=[43]), blitz_row(id="0042", amount=999), blitz_row(id=[42, 43])],
)
def test_blitz_ambiguous_or_conflicting_duplicate_candidates_block_match(
    other: dict[str, Any],
) -> None:
    result = match(response(blitz_row(), other), asset="EURUSD_OP")
    assert result["history_status"] == "HISTORY_AMBIGUOUS"


def test_blitz_row_outside_closed_container_cannot_confirm_position() -> None:
    message = {"name": "options", "status": 0, "msg": {"options": [blitz_row(id=42)]}}
    result = match(message, asset="EURUSD_OP")
    assert result["position_status"] == "INCOMPLETE_CANDIDATES"


def test_blitz_wallet_and_settlement_require_independent_row_evidence() -> None:
    item = blitz_row(balance_id=101, payout=1800)
    partial = match(response(item), asset="EURUSD_OP")
    assert partial["wallet_status"] == "PRACTICE_WALLET_CONFIRMED"
    assert partial["settlement_status"] == "SETTLEMENT_UNCONFIRMED"
    explicit = match(response({**item, "realized_profit": 800}), asset="EURUSD_OP")
    assert explicit["settlement_status"] == "SETTLEMENT_CONFIRMED"


def test_bounded_diagnostics_report_nested_structure_without_guessing_aliases() -> None:
    diagnostics: dict[str, Any] = {}
    incomplete = {
        "id": [94194101, 94194102],
        "amount": 1000,
        "open_time": 1794904000,
        "expiration_time": 1794904060,
        "act": 1,
        "active": "private-asset",
        "active_id": 94194104,
        "option_type": "private-product",
        "user_balance_id": None,
        "direction": "private-enum",
        "option": {"index": 94194103, "is_demo": True, "private-key": "private-value"},
        "private-cookie": "secret-token",
    }
    result = match(
        response(*(row(id=53, amount=777) for _ in range(7)), incomplete),
        diagnostics=diagnostics,
    )
    assert result["position_status"] == "INCOMPLETE_CANDIDATES"
    assert len(diagnostics["record_samples"]) == 5
    assert diagnostics["record_samples_omitted"] == 3
    sample = diagnostics["record_samples"][0]
    assert sample["path"] == "msg.result.closed_options[]"
    assert sample["fields"]["id"] == {"type": "ARRAY", "entry_count": 2}
    assert sample["fields"]["user_balance_id"] == {"type": "NULL"}
    assert sample["unknown_field_count"] == 1
    assert sample["objects"]["option"] == {
        "fields": {"index": {"type": "INTEGER"}, "is_demo": {"type": "BOOLEAN"}},
        "unknown_field_count": 1,
    }
    assert sample["missing_normalized_fields"]["ticket_id"] == "MULTIPLE_TICKET_ENTRIES"
    assert (
        sample["missing_normalized_fields"]["symbol"]
        == "UNSUPPORTED_MALFORMED_OR_CONFLICTING_FIELD"
    )
    assert sample["missing_normalized_fields"]["user_balance_id"] == "MISSING_OR_NULL_FIELD"
    assert sample["recognized_enums"] == {"status": "closed"}
    assert sample["identity_field_categories"] == {
        "active": "UNRECOGNIZED",
        "active_id": "UNRECOGNIZED",
        "option_type": "UNRECOGNIZED",
    }
    trace = json.dumps(diagnostics)
    assert all(
        private not in trace
        for private in ("private-", "941941", "secret-token", "1794904000", "1000", "777")
    )


@pytest.mark.parametrize("container", ["closed_options", "open_options", "options"])
@pytest.mark.parametrize("nested", [False, True])
def test_status_zero_validated_array_structure(container: str, nested: bool) -> None:
    position = row(status="open", result=None) if container == "open_options" else row()
    body = {container: [position, {"metadata": "private-value"}], "private-key": ["private-value"]}
    msg = {"result": body} if nested else body
    diagnostics: dict[str, Any] = {}
    result = reconcile_history(
        {"name": "options", "status": 0, "msg": msg},
        asset="EURUSD",
        stake=1000,
        start=1794903900,
        end=1794904100,
        practice_id="101",
        diagnostics=diagnostics,
    )
    assert result["position_status"] == "MULTIPLE_OR_CONFLICTING_CANDIDATES"
    assert diagnostics["parseable_candidate_count"] == 1
    assert diagnostics["matching_candidate_count"] == 1
    assert diagnostics["incomplete_candidate_count"] == 1
    assert diagnostics["unrecognized_array_count"] == 1
    assert {
        "path": f"msg.result.{container}" if nested else f"msg.{container}",
        "is_array": True,
        "entry_count": 2,
        "object_count": 2,
    } in diagnostics["containers"]
    assert "private" not in json.dumps(diagnostics)
    assert "secret-value" not in json.dumps(diagnostics)


@pytest.mark.parametrize(
    ("status", "msg", "reason"),
    [
        (0, [], "OPTIONS_BODY_NOT_OBJECT"),
        (0, {"data": []}, "OPTION_ARRAY_MISSING"),
        (0, {"result": ["private-value"]}, "OPTION_ARRAY_MISSING"),
        (0, {"closed_options": ["private-value"]}, "OPTION_ROW_NOT_OBJECT"),
        (0, {"success": False, "closed_options": []}, "BROKER_REPORTED_FAILURE"),
        (False, {"closed_options": []}, "UNSUPPORTED_OPTIONS_STATUS"),
        ("0", {"closed_options": []}, "UNSUPPORTED_OPTIONS_STATUS"),
        (0.0, {"closed_options": []}, "UNSUPPORTED_OPTIONS_STATUS"),
    ],
)
def test_history_schema_rejection_has_sanitized_reason(status: Any, msg: Any, reason: str) -> None:
    diagnostics: dict[str, Any] = {}
    with pytest.raises(ValueError, match=reason):
        extract_rows({"name": "options", "status": status, "msg": msg}, diagnostics=diagnostics)
    assert diagnostics["schema_rejection_reason"] == reason
    assert "private-value" not in json.dumps(diagnostics)


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"success": False}, "BROKER_REPORTED_FAILURE"),
        (
            {"msg": {"result": {"closed_options": [blitz_row()]}, "closed_options": []}},
            "AMBIGUOUS_OPTIONS_ENVELOPE",
        ),
    ],
)
def test_status_zero_does_not_override_failed_or_ambiguous_history(
    fields: dict[str, Any], reason: str
) -> None:
    reply = {"name": "options", "status": 0, "msg": {"closed_options": [blitz_row()]}, **fields}
    with pytest.raises(ValueError, match=reason):
        match(reply, asset="EURUSD_OP")


@pytest.mark.asyncio
async def test_query_only_get_options_and_snapshots() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    obs._authenticated = True
    obs._practice_identity_established = True
    practice = Balance(id=101, type=AccountType.PRACTICE, amount=9000, currency="USD")
    real = Balance(id=102, type=AccountType.REAL, amount=0, currency="USD")
    obs._practice_balance = practice
    obs._real_balance = real

    class Socket:
        async def send(self, data: str) -> None:
            frame = json.loads(data)
            assert frame["name"] == "sendMessage"
            assert frame["msg"] == {
                "name": "get-options",
                "body": {"limit": 100, "instrument_type": "binary,turbo", "user_balance_id": 101},
            }
            obs._handle_incoming_frame({**response(row()), "request_id": frame["request_id"]})

    obs._ws = Socket()  # type: ignore[assignment]
    obs.async_refresh_account_state = AsyncMock(return_value=True)  # type: ignore[method-assign]
    result = await recover_history(
        obs, asset="EURUSD", stake=1000, start=1794903900, end=1794904100
    )
    assert result["position_status"] == "POSITION_CONFIRMED"
    assert result["account"]["real_wallet_status"] == "UNCHANGED_BETWEEN_SNAPSHOTS"
    obs.async_refresh_account_state = AsyncMock(return_value=False)  # type: ignore[method-assign]
    result = await recover_history(
        obs, asset="EURUSD", stake=1000, start=1794903900, end=1794904100
    )
    assert result["account"]["real_wallet_status"] == "UNVERIFIED"


@pytest.mark.asyncio
async def test_subscriptions_are_product_scoped_and_unacknowledged(monkeypatch: Any) -> None:
    obs = PassivePracticeObserver(ssid="fake")
    obs._practice_balance = Balance(id=101, type=AccountType.PRACTICE, amount=10, currency="USD")
    sent: list[dict[str, Any]] = []

    class Socket:
        async def send(self, data: str) -> None:
            sent.append(json.loads(data))

    async def immediate_wait(futures: Any, timeout: float) -> tuple[set[Any], set[Any]]:
        return set(), set(futures)

    obs._ws = Socket()  # type: ignore[assignment]
    monkeypatch.setattr(asyncio, "wait", immediate_wait)
    assert await obs.async_subscribe_portfolio() == SubscriptionStatus.STATUS_UNKNOWN
    assert len(sent) == 4
    for frame in sent:
        msg = frame["msg"]
        assert msg["params"]["routingFilters"]["instrument_type"] in {
            "binary-option",
            "turbo-option",
        }
        assert msg["version"] == ("2.0" if msg["name"] == "portfolio.position-changed" else "1.0")
    assert all(not item["response_correlated"] for item in obs.diagnostics()["subscriptions"])


def test_nested_events_rejections_and_unknown() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    obs.set_asset_filter("EURUSD")
    obs._handle_incoming_frame({"name": "some-unknown-event", "msg": {"ssid": "secret"}})
    obs._handle_incoming_frame({
        "name": "portfolio.position-changed",
        "msg": {"result": {"id": "a", "status": "open", "active": "EURUSD-OTC"}},
    })
    obs._handle_incoming_frame({
        "name": "portfolio.position-changed",
        "msg": {"id": "b", "status": "open", "instrument_type": "digital-option"},
    })
    obs._handle_incoming_frame({"name": "option-opened", "msg": {"active_id": 1}})
    diag = obs.diagnostics()
    assert diag["unrecognized_event_frames"] == 1
    assert diag["asset_filter_rejected"] == 1
    assert diag["other_product_rejected"] == 1
    assert diag["invalid_schema"] == 1
    assert "secret" not in json.dumps(diag)


def test_cli_history_requires_opt_in() -> None:
    result = CliRunner().invoke(
        app,
        [
            "practice-history",
            "--asset",
            "EURUSD",
            "--stake",
            "1000",
            "--start",
            "2026-11-17T09:25:00+01:00",
            "--end",
            "2026-11-17T09:30:00+01:00",
        ],
    )
    assert result.exit_code != 0
    assert "--live" in result.output


@pytest.mark.asyncio
async def test_malformed_listener_frame_does_not_stop_following_frame() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    obs.set_asset_filter("EURUSD")

    class Socket:
        def __init__(self) -> None:
            self.frames = iter([
                json.dumps({"name": "option-opened", "msg": {"id": "1", "active_id": "bad"}}),
                json.dumps({"name": "option-opened", "msg": {"id": "2", "active_id": 1}}),
            ])

        async def recv(self) -> str:
            try:
                return next(self.frames)
            except StopIteration:
                raise OSError from None

    obs._ws = Socket()  # type: ignore[assignment]
    await obs._listen_loop()
    assert obs.diagnostics()["invalid_schema"] == 1
    assert obs.diagnostics()["connection_error"] == "CONNECTION_ERROR"
    assert obs._observed_position is not None
    assert obs._observed_position.ticket_id == "2"
    assert obs.portfolio_subscription_status != SubscriptionStatus.EVENT_RECEIVED


@pytest.mark.asyncio
async def test_real_mutation_even_when_balance_returns_to_baseline() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    practice = Balance(id="101", type=AccountType.PRACTICE, amount=9000, currency="USD")
    real = Balance(id="102", type=AccountType.REAL, amount=10, currency="USD")
    obs._practice_balance, obs._real_balance = practice, real
    obs._balances = [practice, real]
    obs._handle_incoming_frame({"name": "balance-changed", "msg": {"id": "102", "amount": 0}})
    obs._handle_incoming_frame({"name": "balance-changed", "msg": {"id": "102", "amount": 10}})
    assert obs.real_mutation_observed("102")
    assert obs.real_balance is not None and obs.real_balance.amount == 10


@pytest.mark.asyncio
async def test_history_timeout_is_not_not_found_and_does_not_retry() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    obs._authenticated = True
    obs._practice_identity_established = True
    obs._practice_balance = Balance(id="101", type=AccountType.PRACTICE, amount=10, currency="USD")
    sent: list[dict[str, Any]] = []

    class Socket:
        async def send(self, data: str) -> None:
            sent.append(json.loads(data))

    obs._ws = Socket()  # type: ignore[assignment]
    with pytest.raises(TimeoutError):
        await obs.async_get_options_history(timeout_seconds=0.01)
    assert len(sent) == 1
    assert obs._pending_requests == {}


def test_nested_digital_event_is_not_misclassified_as_binary() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    obs.set_asset_filter("EURUSD")
    obs._handle_incoming_frame({
        "name": "portfolio.position-changed",
        "msg": {
            "id": "10",
            "status": "open",
            "raw_event": {"active_id": 1, "instrument_type": "digital-option"},
        },
    })
    assert obs._observed_position is None
    assert obs.diagnostics()["other_product_rejected"] == 1
