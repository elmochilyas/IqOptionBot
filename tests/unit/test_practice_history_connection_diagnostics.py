"""Offline protocol-stage and conservative history regressions."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

import iqoption_research.execution.practice_observer as observer_module
from iqoption_research.cli.main import app
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


@pytest.mark.parametrize("missing", ["direction", "expiration_time", "status"])
def test_incomplete_position_fields_do_not_confirm_history(missing: str) -> None:
    assert reconcile(item(**{missing: None}))["history_status"] == "HISTORY_UNVERIFIED"


class ScriptedSocket:
    def __init__(
        self,
        obs: PassivePracticeObserver,
        auth: Any = True,
        profile: Any = None,
        balances: Any = None,
    ) -> None:
        self.obs = obs
        self.auth = auth
        self.profile = {"user_id": 123} if profile is None else profile
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
                "msg": self.balances,
                "request_id": frame["request_id"],
            })
        else:
            raise AssertionError("Unexpected frame; only read-only requests are permitted")

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_unsolicited_session_profile_and_correlated_balance(monkeypatch: Any) -> None:
    obs = PassivePracticeObserver(ssid="private-ssid")
    socket = ScriptedSocket(obs)
    connector = AsyncMock(return_value=socket)
    monkeypatch.setattr(observer_module, "ws_connect", connector)
    await obs.async_connect()
    assert connector.call_args.kwargs["logger"].disabled is True
    assert obs.practice_balance is not None and str(obs.practice_balance.id) == "101"
    assert obs.failure_diagnostics()["last_successful_stage"] == "WALLET_DISCOVERY"
    assert [f["name"] for f in socket.sent] == ["authenticate", "sendMessage", "sendMessage"]
    assert [f["msg"]["name"] for f in socket.sent[1:]] == ["get-profile", "get-balances"]
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
        ({}, [{"id": 101, "type": 4, "amount": 100, "currency": "USD"}], "PROFILE_FETCH_FAILED"),
        ({"user_id": 123}, [], "BALANCES_FETCH_FAILED"),
        ({"user_id": 123}, [{"id": 101, "type": 4, "amount": "bad"}], "BALANCES_FETCH_FAILED"),
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
    assert obs._last_refresh_failure == "BALANCES_FETCH_FAILED"


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
    with pytest.raises(ObserverStageError, match="WS_CONNECTION_CLOSED"):
        await asyncio.wait_for(obs.async_connect(), timeout=0.5)
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
async def test_uncorrelated_history_response_is_not_accepted() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    obs._authenticated = True
    obs._practice_identity_established = True
    obs._practice_balance = Balance(id="101", type=AccountType.PRACTICE, amount=100, currency="USD")

    class Socket:
        async def send(self, data: str) -> None:
            obs._handle_incoming_frame({**history(item()), "request_id": "different"})

    obs._ws = Socket()  # type: ignore[assignment]
    with pytest.raises(ObserverStageTimeout):
        await obs.async_get_options_history(timeout_seconds=0.01)
    report = obs.failure_diagnostics()
    assert report["history_request_dispatched"] is True
    assert report["history_response_arrived"] is True
    assert report["history_response_correlated"] is False
    assert report["error_classification"] == "HISTORY_REQUEST_TIMEOUT"


@pytest.mark.asyncio
async def test_invalid_correlated_history_reported_as_unverified() -> None:
    obs = PassivePracticeObserver(ssid="fake")
    obs._authenticated = True
    obs._practice_identity_established = True
    obs._practice_balance = Balance(id="101", type=AccountType.PRACTICE, amount=100, currency="USD")

    class Socket:
        async def send(self, data: str) -> None:
            request = json.loads(data)
            obs._handle_incoming_frame({
                "name": "options",
                "msg": {"password": "secret"},
                "request_id": request["request_id"],
            })

    obs._ws = Socket()  # type: ignore[assignment]
    with pytest.raises(ObserverStageError, match="HISTORY_RESPONSE_INVALID"):
        await recover_history(obs, asset="EURUSD", stake=1000, start=1, end=2)
    report = obs.failure_diagnostics()
    assert report["history_response_correlated"] is True
    assert report["history_status"] == "HISTORY_UNVERIFIED"
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
