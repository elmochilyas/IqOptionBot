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


def match(message: dict[str, Any]) -> dict[str, Any]:
    return reconcile_history(
        message, asset="EURUSD", stake=1000, start=1794903900, end=1794904100, practice_id="101"
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


@pytest.mark.parametrize("container", ["closed_options", "open_options", "options"])
@pytest.mark.parametrize("nested", [False, True])
def test_status_zero_validated_array_structure(container: str, nested: bool) -> None:
    body = {container: [row(), {"metadata": "private-value"}], "private-key": ["private-value"]}
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
