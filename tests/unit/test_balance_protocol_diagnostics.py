"""Read-only balance request, correlation, and privacy diagnostics regression tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

import iqoption_research.cli.commands.execution_cmd as command_module
from iqoption_research.cli.main import app
from iqoption_research.execution.practice_observer import PassivePracticeObserver
from iqoption_research.models.account import AccountType, Balance

WALLETS: list[dict[str, Any]] = [
    {
        "id": 982345671,
        "type": 4,
        "amount": 8765.43,
        "currency": "USD",
        "email": "private-balance@example.invalid",
        "ssid": "PRIVATE-SSID-IN-PAYLOAD",
    },
    {"id": 982345672, "type": 1, "amount": 25.67, "currency": "USD"},
]


class BalanceSocket:
    def __init__(
        self,
        observer: PassivePracticeObserver,
        response: dict[str, Any] | None,
        *,
        request_id: str = "MATCH",
    ) -> None:
        self.observer = observer
        self.response = response
        self.request_id = request_id
        self.sent: list[dict[str, Any]] = []

    async def send(self, data: str) -> None:
        frame = json.loads(data)
        self.sent.append(frame)
        assert frame["name"] == "sendMessage"
        assert frame["msg"]["name"] in {"get-profile", "get-balances"}
        assert frame["msg"]["version"] == "1.0"
        assert "body" not in frame["msg"]
        if frame["msg"]["name"] == "get-profile":
            self.observer._handle_incoming_frame({"name": "profile", "msg": {"user_id": 1}})
        elif self.response is not None:
            response = dict(self.response)
            if self.request_id == "MATCH":
                response["request_id"] = frame["request_id"]
            elif self.request_id == "DIFFERENT":
                response["request_id"] = "OTHER-REQUEST-ID-PRIVATE"
            self.observer._handle_incoming_frame(response)

    async def close(self) -> None:
        pass


def make_observer(
    response: dict[str, Any] | None, *, request_id: str = "MATCH", enabled: bool = True
) -> tuple[PassivePracticeObserver, BalanceSocket]:
    observer = PassivePracticeObserver(ssid="PRIVATE-SSID", balance_diagnostics=enabled)
    socket = BalanceSocket(observer, response, request_id=request_id)
    observer._ws = socket  # type: ignore[assignment]
    return observer, socket


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "envelope", "shape"),
    [
        ("balances", {"msg": WALLETS}, "MSG_LIST"),
        ("get-balances", {"msg": {"balances": WALLETS}}, "MSG_BALANCES_LIST"),
        ("balance", {"msg": {"result": WALLETS}}, "MSG_RESULT_LIST"),
        ("balances", {"msg": {"msg": WALLETS}}, "MSG_MSG_LIST"),
        ("balances", {"msg": {"result": {"balances": WALLETS}}}, "MSG_RESULT_BALANCES_LIST"),
        ("balances", {"body": WALLETS}, "BODY_LIST"),
        ("balances", {"body": {"balances": WALLETS}}, "BODY_BALANCES_LIST"),
    ],
)
async def test_correlated_envelopes_use_existing_request_schema(
    name: str, envelope: dict[str, Any], shape: str
) -> None:
    obs, socket = make_observer({"name": name, "status": 2000, **envelope})
    assert await obs.async_refresh_account_state(timeout_seconds=0.1)
    assert [frame["msg"]["name"] for frame in socket.sent] == ["get-profile", "get-balances"]
    assert obs.practice_balance is not None and obs.practice_balance.type == AccountType.PRACTICE
    trace = obs.balance_protocol_diagnostics()[0]
    assert trace["dispatch_state"] == "SENT"
    assert trace["initialization_condition"] == "BALANCE_SNAPSHOT_ACCEPTED"
    assert trace["responses"][0] == {
        "message_name": name,
        "request_id": "MATCH",
        "status": "SUCCESS",
        "status_wire_type": "INTEGER",
        "status_wire_code": 2000,
        "status_wire_constant": "2000",
        "envelope": shape,
        "balance_array_detected": True,
        "balance_entry_count": 2,
        "parsed_entry_count": 2,
    }
    serialized = json.dumps(trace)
    for secret in (
        "982345671",
        "982345672",
        "8765.43",
        "25.67",
        "PRIVATE-SSID",
        "private-balance@example.invalid",
    ):
        assert secret not in serialized


@pytest.mark.asyncio
async def test_diagnostic_capture_error_does_not_block_a_valid_response(monkeypatch: Any) -> None:
    obs, _ = make_observer({"name": "balances", "msg": WALLETS})

    def fail(_message: dict[str, Any], _request_id: str) -> None:
        raise ValueError("PRIVATE-LOGGER-ERROR")

    monkeypatch.setattr(obs, "_record_balance_frame", fail)
    assert await obs.async_refresh_account_state(timeout_seconds=0.05)
    trace = obs.balance_protocol_diagnostics()[0]
    assert trace["diagnostic_capture_errors"] > 0
    assert "PRIVATE-LOGGER-ERROR" not in json.dumps(trace)


@pytest.mark.asyncio
async def test_unknown_uncorrelated_frames_are_counted_without_names() -> None:
    obs, _ = make_observer(None)

    class UnknownSocket(BalanceSocket):
        async def send(self, data: str) -> None:
            frame = json.loads(data)
            await super().send(data)
            if frame["msg"]["name"] == "get-balances":
                obs._handle_incoming_frame({"name": "PRIVATE-UNKNOWN-NAME", "msg": WALLETS})

    obs._ws = UnknownSocket(obs, None)  # type: ignore[assignment]
    assert not await obs.async_refresh_account_state(timeout_seconds=0.02)
    trace = obs.balance_protocol_diagnostics()[0]
    assert trace["other_frame_counts"]["unrecognized"] == 1
    assert "PRIVATE-UNKNOWN-NAME" not in json.dumps(trace)


@pytest.mark.asyncio
async def test_cancelled_refresh_is_not_reported_as_accepted() -> None:
    obs, _ = make_observer(None)
    obs._practice_identity_established = True
    task = asyncio.create_task(obs.async_refresh_account_state(timeout_seconds=1))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not obs._practice_identity_established
    assert obs.balance_protocol_diagnostics()[0]["initialization_condition"] == "INTERRUPTED"


@pytest.mark.asyncio
async def test_no_balance_response_is_reported_as_timeout() -> None:
    obs, _ = make_observer(None)
    assert not await obs.async_refresh_account_state(timeout_seconds=0.02)
    assert obs._last_refresh_failure == "BALANCES_RESPONSE_TIMEOUT"
    assert obs.balance_protocol_diagnostics()[0]["responses"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "wire_type"),
    [({}, "MISSING"), ({"status": None}, "NULL"), ({"status": 2000.0}, "NUMBER")],
)
async def test_existing_success_status_representations(
    fields: dict[str, Any], wire_type: str
) -> None:
    obs, _ = make_observer({"name": "balances", "msg": WALLETS, **fields})
    assert await obs.async_refresh_account_state(timeout_seconds=0.05)
    assert obs.practice_balance is not None and obs.practice_balance.type == AccountType.PRACTICE
    assert obs.real_balance is not None and obs.real_balance.type == AccountType.REAL
    assert obs.balance_protocol_diagnostics()[0]["responses"][0]["status_wire_type"] == wire_type
    assert "status_wire_code" not in obs.balance_protocol_diagnostics()[0]["responses"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "classification", "wire_type", "wire_constant", "wire_code"),
    [
        ("2000", "UNRECOGNIZED", "STRING", "2000", None),
        ("success", "UNRECOGNIZED", "STRING", "success", None),
        (2001, "UNRECOGNIZED", "INTEGER", None, 2001),
        (0, "UNRECOGNIZED", "INTEGER", None, 0),
        (-1, "UNRECOGNIZED", "INTEGER", None, None),
        (True, "UNRECOGNIZED", "BOOLEAN", None, None),
        ({"status": 2000}, "UNRECOGNIZED", "OBJECT", None, None),
        (["PRIVATE-STATUS"], "UNRECOGNIZED", "ARRAY", None, None),
        (False, "REJECTED", "BOOLEAN", None, None),
        ("error", "REJECTED", "STRING", "error", None),
        (4000, "REJECTED", "INTEGER", "4000", 4000),
        (9999, "REJECTED", "INTEGER", None, 9999),
        (10000, "REJECTED", "INTEGER", None, None),
        (982345671, "REJECTED", "INTEGER", None, None),
    ],
)
async def test_only_documented_statuses_pass_even_with_valid_wallets(
    status: Any,
    classification: str,
    wire_type: str,
    wire_constant: str | None,
    wire_code: int | None,
) -> None:
    obs, _ = make_observer({"name": "balances", "status": status, "msg": WALLETS})
    assert not await obs.async_refresh_account_state(timeout_seconds=0.05)
    assert obs.practice_balance is None
    assert obs.real_balance is None
    assert not obs.failure_diagnostics()["history_request_dispatched"]
    response = obs.balance_protocol_diagnostics()[0]["responses"][0]
    assert response["status"] == classification
    assert response["status_wire_type"] == wire_type
    assert response.get("status_wire_constant") == wire_constant
    if wire_code is None:
        assert "status_wire_code" not in response
    else:
        assert type(response["status_wire_code"]) is int
        assert response["status_wire_code"] == wire_code
    serialized = json.dumps(response)
    assert "PRIVATE-STATUS" not in serialized
    assert "982345671" not in serialized


@pytest.mark.asyncio
async def test_status_wire_code_ignores_payload_status() -> None:
    obs, _ = make_observer({
        "name": "balances",
        "msg": {"balances": WALLETS, "status": 2001},
    })
    assert await obs.async_refresh_account_state(timeout_seconds=0.05)
    response = obs.balance_protocol_diagnostics()[0]["responses"][0]
    assert response["status_wire_type"] == "MISSING"
    assert "status_wire_code" not in response


@pytest.mark.asyncio
@pytest.mark.parametrize("request_id", ["ABSENT", "DIFFERENT"])
async def test_uncorrelated_balance_does_not_establish_a_wallet(request_id: str) -> None:
    obs, _ = make_observer(
        {"name": "balances", "status": 2000, "msg": WALLETS}, request_id=request_id
    )
    assert not await obs.async_refresh_account_state(timeout_seconds=0.02)
    assert obs._last_refresh_failure == "BALANCES_RESPONSE_TIMEOUT"
    assert obs.practice_balance is None
    assert not obs.failure_diagnostics()["practice_wallet_identity_established"]
    trace = obs.balance_protocol_diagnostics()[0]
    assert trace["responses"][0]["request_id"] == request_id
    assert trace["responses"][0]["parsed_entry_count"] == 2
    assert trace["initialization_condition"] == "BALANCES_RESPONSE_TIMEOUT"
    assert "OTHER-REQUEST-ID-PRIVATE" not in json.dumps(trace)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            {"name": "error", "status": 4000, "msg": {"secret": "DO-NOT-LOG"}},
            "BALANCES_RESPONSE_REJECTED",
        ),
        (
            {"name": "balances", "status": "unexpected-private-status", "msg": WALLETS},
            "BALANCES_RESPONSE_STATUS_UNRECOGNIZED",
        ),
        ({"name": "balances", "msg": {"secret": "DO-NOT-LOG"}}, "BALANCES_RESPONSE_INVALID"),
        ({"name": "private-message-name", "msg": WALLETS}, "BALANCES_RESPONSE_INVALID"),
    ],
)
async def test_classifies_correlated_failures_without_payloads(
    response: dict[str, Any], expected: str
) -> None:
    obs, _ = make_observer(response)
    assert not await obs.async_refresh_account_state(timeout_seconds=0.05)
    assert obs._last_refresh_failure == expected
    trace = json.dumps(obs.balance_protocol_diagnostics())
    for secret in ("DO-NOT-LOG", "unexpected-private-status", "private-message-name", "982345671"):
        assert secret not in trace


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        {"id": 982345672, "type": 1, "currency": "USD"},
        {"id": "invalid-private-id", "type": 1, "amount": 5},
        {"id": 982345672, "type": 17, "amount": 5},
        {"id": 982345671, "type": 1, "amount": 5},
        {"id": "0982345671", "type": 1, "amount": 5},
    ],
)
async def test_invalid_or_duplicate_wallets_do_not_authorize_partial_snapshot(bad: Any) -> None:
    obs, _ = make_observer({"name": "balances", "msg": [WALLETS[0], bad]})
    assert not await obs.async_refresh_account_state(timeout_seconds=0.05)
    assert obs._last_refresh_failure == "BALANCES_RESPONSE_INVALID"
    assert obs.practice_balance is None
    assert not obs.failure_diagnostics()["practice_wallet_identity_established"]


@pytest.mark.asyncio
async def test_multiple_practice_wallets_do_not_choose_one() -> None:
    extra = {"id": 982345673, "type": 4, "amount": 50, "currency": "USD"}
    obs, _ = make_observer({"name": "balances", "status": 2000, "msg": [*WALLETS, extra]})
    assert not await obs.async_refresh_account_state(timeout_seconds=0.05)
    assert obs._last_refresh_failure == "PRACTICE_WALLET_AMBIGUOUS"
    assert obs.practice_balance is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("balances", "error"),
    [([], "BALANCES_RESPONSE_INVALID"), ([WALLETS[1]], "PRACTICE_WALLET_MISSING")],
)
async def test_success_status_does_not_replace_required_wallets(
    balances: list[dict[str, Any]], error: str
) -> None:
    obs, _ = make_observer({"name": "balances", "status": 2000, "msg": balances})
    assert not await obs.async_refresh_account_state(timeout_seconds=0.05)
    assert obs._last_refresh_failure == error
    assert obs.practice_balance is None


@pytest.mark.asyncio
async def test_multiple_real_wallets_remain_unverified() -> None:
    extra = {"id": 982345673, "type": 1, "amount": 50, "currency": "USD"}
    obs, _ = make_observer({"name": "balances", "msg": [*WALLETS, extra]})
    assert await obs.async_refresh_account_state(timeout_seconds=0.05)
    assert obs.practice_balance is not None
    assert obs.real_balance is None
    assert obs.failure_diagnostics()["real_wallet_state"] == "UNVERIFIED"


def test_unsolicited_snapshot_cannot_replace_established_wallet() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    original = Balance(id="123", type=AccountType.PRACTICE, amount=10, currency="USD")
    obs._practice_balance = original
    obs._practice_identity_established = True
    obs._handle_incoming_frame({"name": "balances", "msg": WALLETS})
    assert obs.practice_balance == original


@pytest.mark.asyncio
async def test_no_diagnostic_collection_without_opt_in() -> None:
    obs, _ = make_observer({"name": "balances", "msg": WALLETS}, enabled=False)
    assert await obs.async_refresh_account_state(timeout_seconds=0.05)
    assert obs.balance_protocol_diagnostics() == []
    assert "balance_protocol_diagnostics" not in obs.failure_diagnostics()


@pytest.mark.asyncio
async def test_internal_error_has_separate_classification() -> None:
    obs, _ = make_observer(None)

    class FaultingSocket(BalanceSocket):
        async def send(self, data: str) -> None:
            frame = json.loads(data)
            if frame["msg"]["name"] == "get-balances":
                raise ValueError("PRIVATE INTERNAL DATA")
            await super().send(data)

    obs._ws = FaultingSocket(obs, None)  # type: ignore[assignment]
    assert not await obs.async_refresh_account_state(timeout_seconds=0.05)
    assert obs._last_refresh_failure == "BALANCES_INTERNAL_ERROR"
    assert "PRIVATE INTERNAL DATA" not in json.dumps(obs.balance_protocol_diagnostics())


@pytest.mark.asyncio
async def test_balance_send_timeout_does_not_claim_broker_rejection() -> None:
    obs, _ = make_observer(None)

    class TimedOutSocket(BalanceSocket):
        async def send(self, data: str) -> None:
            frame = json.loads(data)
            if frame["msg"]["name"] == "get-balances":
                raise TimeoutError("PRIVATE SEND DETAILS")
            await super().send(data)

    obs._ws = TimedOutSocket(obs, None)  # type: ignore[assignment]
    assert not await obs.async_refresh_account_state(timeout_seconds=0.05)
    assert obs._last_refresh_failure == "BALANCES_REQUEST_SEND_TIMEOUT"
    trace = obs.balance_protocol_diagnostics()[0]
    assert trace["dispatch_state"] == "DELIVERY_UNKNOWN"
    assert "PRIVATE SEND DETAILS" not in json.dumps(trace)


def test_cli_opt_in_trace_does_not_print_private_balance_data(monkeypatch: Any) -> None:
    obs, _ = make_observer({"name": "balances", "msg": WALLETS}, request_id="ABSENT")

    async def connect() -> None:
        if not await obs.async_refresh_account_state(timeout_seconds=0.02):
            raise obs._fail("BALANCES_FETCH", obs._last_refresh_failure or "UNKNOWN")

    obs.async_connect = connect  # type: ignore[method-assign]
    monkeypatch.setattr(command_module, "PassivePracticeObserver", lambda **_kwargs: obs)
    monkeypatch.setenv("IQOPTION_EMAIL", "private-user@example.invalid")
    monkeypatch.setenv("IQOPTION_PASSWORD", "PRIVATE-PASSWORD")
    res = CliRunner().invoke(
        app,
        [
            "practice-history",
            "--live",
            "--asset",
            "EURUSD",
            "--stake",
            "1000",
            "--start",
            "2026-11-17T09:25:00+01:00",
            "--end",
            "2026-11-17T09:30:00+01:00",
            "--balance-diagnostics",
        ],
    )
    assert res.exit_code == 1
    assert "BALANCES_RESPONSE_TIMEOUT" in res.output
    assert "balance_protocol_diagnostics" in res.output
    assert "ABSENT" in res.output
    for secret in (
        "982345671",
        "982345672",
        "8765.43",
        "25.67",
        "PRIVATE-SSID",
        "PRIVATE-PASSWORD",
        "private-user@example.invalid",
    ):
        assert secret not in res.output


def test_cli_success_prints_only_classifications(monkeypatch: Any) -> None:
    obs, _ = make_observer(None, enabled=True)
    obs.async_connect = AsyncMock()  # type: ignore[method-assign]
    result = {
        "history_status": "HISTORY_CONFIRMED",
        "position_status": "POSITION_CONFIRMED",
        "wallet_status": "PRACTICE_WALLET_CONFIRMED",
        "settlement_status": "SETTLEMENT_CONFIRMED",
        "candidate_count": 1,
        "incomplete_candidates": [],
        "query_limit_may_truncate_history": True,
        "account": {
            "real_wallet_status": "UNVERIFIED",
            "real_amount": 8765.43,
            "practice_balance_id": "982345671",
        },
        "candidates": [{"ticket_id": "PRIVATE-TICKET"}],
    }
    monkeypatch.setattr(command_module, "PassivePracticeObserver", lambda **_kwargs: obs)
    monkeypatch.setattr(command_module, "recover_history", AsyncMock(return_value=result))
    monkeypatch.setenv("IQOPTION_EMAIL", "private-user@example.invalid")
    monkeypatch.setenv("IQOPTION_PASSWORD", "PRIVATE-PASSWORD")
    res = CliRunner().invoke(
        app,
        [
            "practice-history",
            "--live",
            "--asset",
            "EURUSD",
            "--stake",
            "1000",
            "--start",
            "2026-11-17T09:25:00+01:00",
            "--end",
            "2026-11-17T09:30:00+01:00",
            "--balance-diagnostics",
        ],
    )
    assert res.exit_code == 0
    assert "HISTORY_CONFIRMED" in res.output
    assert "balance_protocol_diagnostics" in res.output
    for secret in ("982345671", "8765.43", "PRIVATE-TICKET", "PRIVATE-PASSWORD"):
        assert secret not in res.output
