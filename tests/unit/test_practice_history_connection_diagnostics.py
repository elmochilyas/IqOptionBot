"""Offline protocol-stage and conservative history regressions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

import iqoption_research.cli.commands.execution_cmd as command_module
import iqoption_research.execution.practice_observer as observer_module
from iqoption_research.cli.main import app
from iqoption_research.execution.live_practice_transport import IQOptionLivePracticeTransport
from iqoption_research.execution.modes import PracticeAuthorizationError
from iqoption_research.execution.practice_history import reconcile_history, recover_history
from iqoption_research.execution.practice_observer import (
    ObserverStageError,
    ObserverStageTimeout,
    PassivePracticeObserver,
)
from iqoption_research.models.account import AccountType, Balance


def history(*items: dict[str, Any]) -> dict[str, Any]:
    return {"name": "options", "msg": {"closed_options": list(items)}}


def item(**changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": 42,
        "active_id": 1,
        "instrument_type": "turbo-option",
        "amount": 1000,
        "created_at": 1794904000,
        "user_balance_id": 101,
        "direction": "call",
        "expiration_time": 1794904060,
        "status": "closed",
    }
    value.update(changes)
    return value


def reconcile(*items: dict[str, Any]) -> dict[str, Any]:
    return reconcile_history(
        history(*items),
        asset="EURUSD",
        stake=1000,
        start=1794903900,
        end=1794904100,
        practice_id="101",
    )


def test_history_verdict_requires_one_complete_unique_match() -> None:
    assert reconcile()["history_status"] == "HISTORY_NOT_FOUND"
    assert reconcile(item())["history_status"] == "HISTORY_CONFIRMED"
    assert reconcile(item(id=42), item(id=43))["history_status"] == "HISTORY_AMBIGUOUS"
    assert reconcile(item(id=None))["history_status"] == "HISTORY_UNVERIFIED"
    assert reconcile(item(instrument_type=None))["history_status"] == "HISTORY_UNVERIFIED"
    assert (
        reconcile(item(), item(id=43, instrument_type=None))["history_status"]
        == "HISTORY_AMBIGUOUS"
    )
    assert reconcile(item(user_balance_id=None))["wallet_status"] == "ROUTING_UNVERIFIED"


@pytest.mark.parametrize("missing", ["direction", "expiration_time"])
def test_incomplete_position_fields_do_not_confirm_history(missing: str) -> None:
    assert reconcile(item(**{missing: None}))["history_status"] == "HISTORY_UNVERIFIED"


def test_unclassified_options_array_needs_explicit_lifecycle() -> None:
    reply = {"name": "options", "msg": {"options": [item(status=None)]}}
    result = reconcile_history(
        reply, asset="EURUSD", stake=1000, start=1794903900, end=1794904100, practice_id="101"
    )
    assert result["history_status"] == "HISTORY_UNVERIFIED"


class ScriptedSocket:
    def __init__(
        self,
        obs: PassivePracticeObserver,
        auth: Any = True,
        profile: Any = None,
        balances: Any = None,
        balance_status: int = 2000,
    ) -> None:
        self.obs = obs
        self.auth = auth
        self.profile = {"user_id": 123} if profile is None else profile
        self.balance_status = balance_status
        self.balances = (
            [
                {"id": 101, "type": 4, "amount": 9000, "currency": "USD"},
                {"id": 102, "type": 1, "amount": 0, "currency": "USD"},
            ]
            if balances is None
            else balances
        )
        self.sent: list[dict[str, Any]] = []
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    async def recv(self) -> str:
        return await self.queue.get()

    async def send(self, data: str) -> None:
        frame = json.loads(data)
        self.sent.append(frame)
        if frame["name"] == "authenticate":
            self.obs._handle_incoming_frame({"name": "authenticated", "msg": self.auth})
        elif frame["msg"]["name"] == "get-profile":
            # Real gateways may send this profile without a request id.
            self.obs._handle_incoming_frame({"name": "profile", "msg": self.profile})
        elif frame["msg"]["name"] == "get-balances":
            self.obs._handle_incoming_frame({
                "name": "balances",
                "status": self.balance_status,
                "msg": self.balances,
                "request_id": frame["request_id"],
            })
        elif frame["msg"]["name"] == "get-options":
            assert self.obs.failure_diagnostics()["practice_wallet_identity_established"]
            assert frame["msg"]["body"]["user_balance_id"] == 101
            self.obs._handle_incoming_frame({
                **history(),
                "status": 0,
                "request_id": frame["request_id"],
            })
        else:
            raise AssertionError("Unexpected frame; only read-only requests are permitted")

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("balance_status", [0, 2000])
async def test_unsolicited_session_profile_and_correlated_balance(
    monkeypatch: Any, balance_status: int
) -> None:
    obs = PassivePracticeObserver(ssid="private-ssid", balance_diagnostics=True)
    socket = ScriptedSocket(obs, balance_status=balance_status)
    connector = AsyncMock(return_value=socket)
    monkeypatch.setattr(observer_module, "ws_connect", connector)
    await obs.async_connect()
    assert connector.call_args.kwargs["logger"].disabled is True
    assert obs.practice_balance is not None and str(obs.practice_balance.id) == "101"
    assert obs.real_balance is not None and str(obs.real_balance.id) == "102"
    assert obs.failure_diagnostics()["last_successful_stage"] == "PRACTICE_WALLET_ESTABLISHED"
    assert obs.profile_verification_diagnostics() == {
        "profile_verification": "PROFILE_UNVERIFIED",
        "profile_outcome": "PROFILE_UNCORRELATED",
    }
    assert [f["name"] for f in socket.sent] == ["authenticate", "sendMessage", "sendMessage"]
    assert [f["msg"]["name"] for f in socket.sent[1:]] == ["get-profile", "get-balances"]
    result = await recover_history(obs, asset="EURUSD", stake=1000, start=1, end=2)
    assert result["history_status"] == "HISTORY_NOT_FOUND"
    assert obs.failure_diagnostics()["history_response_correlated"] is True
    trace = obs.account_protocol_diagnostics()["history_protocol_diagnostics"]
    assert trace["status_wire_code"] == 0
    assert trace["structure"]["schema_rejection_reason"] is None
    assert [f["msg"]["name"] for f in socket.sent[1:]] == [
        "get-profile",
        "get-balances",
        "get-options",
        "get-profile",
        "get-balances",
    ]
    for snapshot in obs.balance_protocol_diagnostics():
        assert snapshot["initialization_condition"] == "BALANCE_SNAPSHOT_ACCEPTED"
        response = snapshot["responses"][0]
        assert response["status"] == "SUCCESS"
        assert response["status_wire_code"] == balance_status
    await obs.async_close()


@pytest.mark.asyncio
async def test_late_uncorrelated_profile_after_correlated_balance() -> None:
    obs = PassivePracticeObserver(ssid="fake")

    class Socket(ScriptedSocket):
        async def send(self, data: str) -> None:
            frame = json.loads(data)
            if frame["msg"]["name"] == "get-profile":
                return
            await super().send(data)
            asyncio.get_running_loop().call_soon(
                obs._handle_incoming_frame, {"name": "profile", "msg": {"user_id": 123}}
            )

    obs._ws = Socket(obs)  # type: ignore[assignment]
    assert await obs.async_refresh_account_state(timeout_seconds=0.1)
    assert obs.practice_balance is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("auth", [False, {"authenticated": False}, "true"])
async def test_auth_must_be_explicit_true(monkeypatch: Any, auth: Any) -> None:
    obs = PassivePracticeObserver(ssid="private-ssid")
    socket = ScriptedSocket(obs, auth=auth)
    monkeypatch.setattr(observer_module, "ws_connect", AsyncMock(return_value=socket))
    with pytest.raises(ObserverStageError, match="WS_AUTH_FAILED"):
        await obs.async_connect()
    assert len(socket.sent) == 1
    assert obs.failure_diagnostics()["history_request_dispatched"] is False
    await obs.async_close()


@pytest.mark.asyncio
async def test_handshake_failure_is_sanitized(monkeypatch: Any) -> None:
    obs = PassivePracticeObserver(ssid="private-ssid")
    monkeypatch.setattr(
        observer_module,
        "ws_connect",
        AsyncMock(side_effect=OSError("private-ssid password=very-secret")),
    )
    with pytest.raises(ObserverStageError, match="WS_HANDSHAKE_FAILED"):
        await obs.async_connect()
    report = obs.failure_diagnostics()
    assert report["last_successful_stage"] == "SSID_SUPPLIED"
    assert report["failed_stage"] == "WS_HANDSHAKE"
    assert "private-ssid" not in json.dumps(report)
    assert "very-secret" not in json.dumps(report)
    await obs.async_close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile, balances, expected",
    [
        ({}, [{"id": 101, "type": 4, "amount": 100, "currency": "USD"}], "PROFILE_MALFORMED"),
        ({"user_id": 123}, [], "BALANCES_RESPONSE_INVALID"),
        ({"user_id": 123}, [{"id": 101, "type": 4, "amount": "bad"}], "BALANCES_RESPONSE_INVALID"),
    ],
)
async def test_missing_or_malformed_account_data(
    profile: Any, balances: Any, expected: str
) -> None:
    obs = PassivePracticeObserver(ssid="fake")
    socket = ScriptedSocket(obs, profile=profile, balances=balances)
    obs._ws = socket  # type: ignore[assignment]
    assert not await obs.async_refresh_account_state(timeout_seconds=0.02)
    assert obs._last_refresh_failure == expected
    assert not obs.failure_diagnostics()["practice_wallet_identity_established"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "expected", "correlation"),
    [
        (None, "PROFILE_TIMEOUT", None),
        ({"name": "profile", "msg": {"user_id": 123}}, "PROFILE_UNCORRELATED", "DIFFERENT"),
        (
            {"name": "profile", "status": 0, "msg": {"user_id": 123}},
            "PROFILE_STATUS_UNRECOGNIZED",
            "MATCH",
        ),
        (
            {"name": "profile", "status": 0, "msg": {"result": {"user_id": 123}}},
            "PROFILE_VERIFIED",
            "MATCH",
        ),
        (
            {"name": "profile", "status": 0, "msg": {"result": {"balance_id": 101}}},
            "PROFILE_MALFORMED",
            "MATCH",
        ),
        (
            {"name": "profile", "status": 0, "msg": {"result": {"user_id": 123}}},
            "PROFILE_STATUS_UNRECOGNIZED",
            "DIFFERENT",
        ),
        ({"name": "profile", "msg": {}}, "PROFILE_MALFORMED", "MATCH"),
        (
            {"name": "error", "status": 4000, "msg": {"detail": "PRIVATE-PROFILE-DATA"}},
            "PROFILE_REJECTED",
            "MATCH",
        ),
        (
            {"name": "user-profile", "body": {"profile": {"user_id": 123}}},
            "PROFILE_VERIFIED",
            "MATCH",
        ),
    ],
)
async def test_read_only_wallet_refresh_reports_profile_independently(
    frame: dict[str, Any] | None, expected: str, correlation: str | None
) -> None:
    obs = PassivePracticeObserver(ssid="fake", history_only=True, balance_diagnostics=True)

    class Socket(ScriptedSocket):
        async def send(self, data: str) -> None:
            request = json.loads(data)
            if request["msg"]["name"] != "get-profile":
                await super().send(data)
                return
            if frame is not None:
                req_id = request["request_id"] if correlation == "MATCH" else "PRIVATE-ID"
                obs._handle_incoming_frame({**frame, "request_id": req_id})

    obs._ws = Socket(obs, balance_status=0)  # type: ignore[assignment]
    assert await obs.async_refresh_account_state(timeout_seconds=0.02)
    assert obs.practice_balance is not None and obs.practice_balance.id == "101"
    assert (obs.profile is not None) == (expected == "PROFILE_VERIFIED")
    trace = obs.account_protocol_diagnostics()["profile_protocol_diagnostics"][0]
    assert trace["dispatch_state"] == "SENT"
    assert trace["profile_outcome"] == expected
    if correlation is not None:
        assert trace["responses"][0]["request_id"] == correlation
    else:
        assert trace["responses"] == []
    assert "PRIVATE-PROFILE-DATA" not in json.dumps(trace)
    assert "PRIVATE-ID" not in json.dumps(trace)


@pytest.mark.asyncio
async def test_profile_push_before_query_is_not_claimed_as_correlated(monkeypatch: Any) -> None:
    obs = PassivePracticeObserver(ssid="fake", balance_diagnostics=True)

    class Socket(ScriptedSocket):
        async def send(self, data: str) -> None:
            frame = json.loads(data)
            if frame["name"] == "authenticate":
                obs._handle_incoming_frame({"name": "profile", "msg": {"user_id": 123}})
            elif frame["msg"]["name"] == "get-profile":
                return
            await super().send(data)

    monkeypatch.setattr(observer_module, "ws_connect", AsyncMock(return_value=Socket(obs)))
    try:
        await obs.async_connect()
        trace = obs.account_protocol_diagnostics()["profile_protocol_diagnostics"][0]
        assert trace["profile_outcome"] == "PROFILE_UNCORRELATED"
        assert trace["responses"][0]["before_request"] is True
    finally:
        await obs.async_close()


@pytest.mark.parametrize("diagnostics_enabled", [False, True])
def test_history_cli_proceeds_with_verified_wallet_and_unverified_profile(
    monkeypatch: Any, tmp_path: Path, diagnostics_enabled: bool
) -> None:
    class Socket(ScriptedSocket):
        async def send(self, data: str) -> None:
            request = json.loads(data)
            if request["name"] == "sendMessage" and request["msg"]["name"] == "get-profile":
                self.sent.append(request)
                self.obs._handle_incoming_frame({
                    "name": "profile",
                    "status": 0,
                    "msg": {"user_id": 123},
                    "request_id": request["request_id"],
                })
                return
            if request["name"] == "sendMessage" and request["msg"]["name"] == "get-options":
                self.sent.append(request)
                assert self.obs.failure_diagnostics()["practice_wallet_identity_established"]
                assert request["msg"]["body"]["user_balance_id"] == 101
                for frame in (
                    {"name": "option-opened", "msg": {"id": "private-push-ticket", "active_id": 1}},
                    {"name": "option-closed", "msg": {"id": "private-close-ticket"}},
                    {"name": "balance-changed", "msg": {"id": "837625849", "amount": 71423.25}},
                    {"name": "balance-changed", "msg": {"id": "private-invalid-balance-id"}},
                ):
                    self.obs._handle_incoming_frame(frame)
                self.obs._handle_incoming_frame({
                    "name": "options",
                    "status": 0,
                    "request_id": request["request_id"],
                    "msg": {
                        "closed_options": [
                            {
                                "id": [9944556677],
                                "active": "EURUSD_OP",
                                "active_id": 1861,
                                "option_type": "blitz-option",
                                "amount": 1000,
                                "created": 1794907530,
                                "expired": 1794907560,
                                "win": "win",
                                "win_amount": 198765,
                                "option": {"token": "private-record-token"},
                            }
                        ],
                        "open_options": [],
                    },
                })
                return
            await super().send(data)

    sockets: list[Socket] = []

    def observer_factory(**kwargs: Any) -> PassivePracticeObserver:
        obs = PassivePracticeObserver(**kwargs)
        # The broker response identifies a valid Practice wallet and omits REAL.
        socket = Socket(
            obs,
            balance_status=0,
            balances=[
                {"id": 101, "type": 4, "amount": 9000, "currency": "USD"},
            ],
        )
        sockets.append(socket)
        monkeypatch.setattr(observer_module, "ws_connect", AsyncMock(return_value=socket))
        return obs

    monkeypatch.setattr(command_module, "PassivePracticeObserver", observer_factory)
    monkeypatch.setattr(observer_module, "async_login", AsyncMock(return_value="private-ssid"))
    monkeypatch.setenv("IQOPTION_EMAIL", "private@example.com")
    monkeypatch.setenv("IQOPTION_PASSWORD", "private-password")
    evidence = tmp_path / "evidence.json"
    result = CliRunner().invoke(
        app,
        [
            "--verbose",
            "practice-history",
            "--live",
            "--asset",
            "EURUSD-op",
            "--stake",
            "1000",
            "--start",
            "2026-11-17T09:25:00Z",
            "--end",
            "2026-11-17T09:30:00Z",
            *(["--balance-diagnostics", "--record", str(evidence)] if diagnostics_enabled else []),
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"profile_verification": "PROFILE_UNVERIFIED"' in result.output
    assert '"real_wallet_status": "UNVERIFIED"' in result.output
    assert '"history_status": "HISTORY_POSITION_MATCHED"' in result.output
    assert '"position_status": "POSITION_CONFIRMED"' in result.output
    assert '"wallet_status": "ROUTING_UNVERIFIED"' in result.output
    assert '"settlement_status": "SETTLEMENT_UNCONFIRMED"' in result.output
    assert '"operator_reconciliation_required": true' in result.output
    assert '"start": "2026-11-17T09:25:00+00:00"' in result.output
    assert ('"record_samples"' in result.output) == diagnostics_enabled
    assert ("msg.closed_options[]" in result.output) == diagnostics_enabled
    assert ('"option_type": "blitz-option"' in result.output) == diagnostics_enabled
    private_values = (
        "198765",
        "private-record-token",
        "private-ssid",
        "private@example.com",
        "private-password",
        "private-push-ticket",
        "private-close-ticket",
        "837625849",
        "71423.25",
        "private-invalid-balance-id",
    )
    assert not any(
        value in result.output for value in (*private_values, "9944556677", "1794907530")
    )
    assert evidence.exists() == diagnostics_enabled
    if diagnostics_enabled:
        exported = evidence.read_text(encoding="utf-8")
        assert not any(value in exported for value in private_values)
        assert json.loads(exported)["candidates"][0]["ticket_id"] == "9944556677"
    assert sockets[0].sent[0]["name"] == "authenticate"
    assert [f["msg"]["name"] for f in sockets[0].sent[1:]] == [
        "get-profile",
        "get-balances",
        "get-options",
        "get-profile",
        "get-balances",
    ]


@pytest.mark.asyncio
async def test_history_profile_exception_does_not_apply_to_execution_guard() -> None:
    transport = IQOptionLivePracticeTransport(ssid="fake")
    transport._balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=9000, currency="USD")
    ]
    with pytest.raises(PracticeAuthorizationError, match="Incomplete account data"):
        await transport._verify_practice_account_guard()


@pytest.mark.asyncio
async def test_unrelated_balance_push_does_not_validate_malformed_correlated_response() -> None:
    obs = PassivePracticeObserver(ssid="fake")

    class Socket(ScriptedSocket):
        async def send(self, data: str) -> None:
            frame = json.loads(data)
            if frame["msg"]["name"] == "get-balances":
                self.obs._handle_incoming_frame({"name": "balances", "msg": self.balances})
                self.obs._handle_incoming_frame({
                    "name": "balances",
                    "msg": [],
                    "request_id": frame["request_id"],
                })
            else:
                await super().send(data)

    obs._ws = Socket(obs)  # type: ignore[assignment]
    assert not await obs.async_refresh_account_state(timeout_seconds=0.02)
    assert obs._last_refresh_failure == "BALANCES_RESPONSE_INVALID"


@pytest.mark.asyncio
async def test_close_during_initialization_fails_without_waiting_for_timeout(
    monkeypatch: Any,
) -> None:
    obs = PassivePracticeObserver(ssid="fake")

    class Socket(ScriptedSocket):
        async def recv(self) -> str:
            await self.queue.get()
            raise OSError("sensitive remote error")

        async def send(self, data: str) -> None:
            frame = json.loads(data)
            if frame["name"] == "sendMessage":
                await self.queue.put("close")
            else:
                await super().send(data)

    monkeypatch.setattr(observer_module, "ws_connect", AsyncMock(return_value=Socket(obs)))
    # The close is propagated as soon as the listener observes the dropped socket.
    # Local TLS-context setup can take ~0.2s, so 2s still proves we do not wait
    # for the observer's 5s account-state timeout while avoiding host-speed flakes.
    with pytest.raises(ObserverStageError, match="WS_CONNECTION_CLOSED"):
        await asyncio.wait_for(obs.async_connect(), timeout=2.0)
    assert not obs.failure_diagnostics()["history_request_dispatched"]
    assert not obs.is_connected
    await obs.async_close()


@pytest.mark.asyncio
async def test_history_cannot_dispatch_without_verified_session_and_wallet() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    obs._practice_balance = Balance(id="101", type=AccountType.PRACTICE, amount=100, currency="USD")
    socket = ScriptedSocket(obs)
    obs._ws = socket  # type: ignore[assignment]
    with pytest.raises(ObserverStageError, match="ACCOUNT_STATE_UNVERIFIED"):
        await obs.async_get_options_history()
    assert socket.sent == []
    assert obs.failure_diagnostics()["history_request_dispatched"] is False


@pytest.mark.asyncio
async def test_history_requires_valid_wallet_again_after_failed_refresh(monkeypatch: Any) -> None:
    obs = PassivePracticeObserver(ssid="fake", history_only=True)
    socket = ScriptedSocket(obs, balance_status=0)
    monkeypatch.setattr(observer_module, "ws_connect", AsyncMock(return_value=socket))
    await obs.async_connect()
    try:
        valid = socket.balances
        socket.balances = [{**valid[0], "balance_type": 1}]
        assert not await obs.async_refresh_account_state()
        with pytest.raises(ObserverStageError, match="ACCOUNT_STATE_UNVERIFIED"):
            await obs.async_get_options_history()
        assert [frame["msg"]["name"] for frame in socket.sent[1:]].count("get-options") == 0

        socket.balances = valid
        assert await obs.async_refresh_account_state()
        assert (await obs.async_get_options_history())["name"] == "options"
    finally:
        await obs.async_close()


@pytest.mark.asyncio
async def test_uncorrelated_history_response_is_not_accepted() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    obs._authenticated = True
    obs._practice_identity_established = True
    obs._practice_balance = Balance(id="101", type=AccountType.PRACTICE, amount=100, currency="USD")

    class Socket:
        async def send(self, data: str) -> None:
            obs._handle_incoming_frame({**history(item()), "status": 0, "request_id": "different"})

    obs._ws = Socket()  # type: ignore[assignment]
    with pytest.raises(ObserverStageTimeout):
        await obs.async_get_options_history(timeout_seconds=0.01)
    report = obs.failure_diagnostics()
    assert report["history_request_dispatched"] is True
    assert report["history_response_arrived"] is True
    assert report["history_response_correlated"] is False
    assert report["error_classification"] == "HISTORY_REQUEST_TIMEOUT"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "stage", "code", "reason"),
    [
        (
            {"name": "options", "msg": {"password": "secret"}},
            "HISTORY_PARSING",
            "HISTORY_RESPONSE_INVALID",
            "OPTION_ARRAY_MISSING",
        ),
        (
            {"name": "options", "status": 0, "msg": {"closed_options": {}}},
            "HISTORY_PARSING",
            "HISTORY_RESPONSE_INVALID",
            "OPTION_CONTAINER_NOT_ARRAY",
        ),
        (
            {"name": "options", "status": 2001, "msg": {"closed_options": []}},
            "HISTORY_PARSING",
            "HISTORY_RESPONSE_INVALID",
            "UNSUPPORTED_OPTIONS_STATUS",
        ),
        (
            {"name": "error", "status": 4000, "msg": {"detail": "secret"}},
            "HISTORY_RESPONSE",
            "HISTORY_RESPONSE_REJECTED",
            "BROKER_REPORTED_FAILURE",
        ),
    ],
)
async def test_correlated_history_failure_remains_unverified(
    frame: dict[str, Any],
    stage: str,
    code: str,
    reason: str,
) -> None:
    obs = PassivePracticeObserver(ssid="fake", balance_diagnostics=True)
    obs._authenticated = True
    obs._practice_identity_established = True
    obs._practice_balance = Balance(id="101", type=AccountType.PRACTICE, amount=100, currency="USD")

    class Socket:
        async def send(self, data: str) -> None:
            request = json.loads(data)
            obs._handle_incoming_frame({**frame, "request_id": request["request_id"]})

    obs._ws = Socket()  # type: ignore[assignment]
    with pytest.raises(ObserverStageError, match=code):
        await recover_history(obs, asset="EURUSD", stake=1000, start=1, end=2)
    report = obs.failure_diagnostics()
    assert report["history_response_correlated"] is True
    assert report["history_status"] == "HISTORY_UNVERIFIED"
    assert report["failed_stage"] == stage
    assert report["history_protocol_diagnostics"]["message_name"] == frame["name"]
    assert report["history_protocol_diagnostics"]["request_id"] == "MATCH"
    assert report["history_protocol_diagnostics"]["structure"]["schema_rejection_reason"] == reason
    assert "secret" not in json.dumps(report)


def test_cli_failure_report_does_not_leak_credentials(monkeypatch: Any) -> None:
    monkeypatch.setenv("IQOPTION_EMAIL", "private@example.com")
    monkeypatch.setenv("IQOPTION_PASSWORD", "private-password")
    monkeypatch.setattr(observer_module, "async_login", AsyncMock(return_value="private-ssid"))
    monkeypatch.setattr(
        observer_module,
        "ws_connect",
        AsyncMock(side_effect=OSError("private-ssid private-password private@example.com")),
    )
    result = CliRunner().invoke(
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
        ],
    )
    assert result.exit_code == 1
    assert "WS_HANDSHAKE_FAILED" in result.output
    assert "HISTORY_UNVERIFIED" in result.output
    assert "private-ssid" not in result.output
    assert "private-password" not in result.output
    assert "private@example.com" not in result.output
