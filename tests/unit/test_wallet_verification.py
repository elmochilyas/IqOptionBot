"""Offline observations from independent sockets. All payload values are invented."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.models.account import (
    AccountType,
    Balance,
    PracticeStatus,
    WalletIndicator,
    WalletSource,
    wallet_decision,
)

REAL = "private-real-key"
PRACTICE = "private-practice-key"


def socket() -> IQOptionWebSocketClient:
    ws = IQOptionWebSocketClient("private-auth-token")
    ws._dispatch({"name": "authenticated", "msg": True})
    ws._outstanding_balances_request_id = "inventory"
    return ws


def inventory(ws: IQOptionWebSocketClient, active: object = None, default: bool = False) -> None:
    ws._dispatch({
        "name": "result",
        "request_id": "inventory",
        "msg": {
            "balances": [
                {"id": REAL, "type": 1, "default": default, "amount": 4321.99},
                {"id": PRACTICE, "type": 4, "active": active},
            ]
        },
    })


def profile(ws: IQOptionWebSocketClient, **fields: object) -> None:
    ws._dispatch({
        "name": "profile",
        "msg": {"user_id": 45678, "email": "private@invalid.test", **fields},
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["balance_id", "balanceId", "active_balance_id", "current_balance"])
async def test_profile_and_default_agree_without_authority(key: str) -> None:
    ws = socket()
    profile(ws, **{key: {"id": REAL} if key == "current_balance" else REAL})
    inventory(ws, active="false", default=True)
    report = await ws.wallet_diagnostic()
    assert report == {
        "account_types": ["PRACTICE", "REAL"],
        "practice_balance_exists": True,
        "independent_balances_received": True,
        "indicator_sources": ["get-balances.default", "profile"],
        "source_scopes": {"get-balances.default": "default", "profile": "session_scope_unproven"},
        "independent_indicators_agree": True,
        "classification": "ACCOUNT_SELECTION_UNVERIFIED",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("profile_first", [True, False])
async def test_disagreement_is_order_independent(profile_first: bool) -> None:
    ws = socket()
    if profile_first:
        profile(ws, balance_id=REAL)
    inventory(ws, active=True)
    ws._dispatch({"name": "balance-changed", "msg": {"current_balance": {"id": PRACTICE}}})
    if not profile_first:
        profile(ws, balance_id=REAL, balances=[{"id": REAL, "type": 1}])
    assert ws.active_balance_id == PRACTICE
    assert ws.balances[1].raw_is_active
    assert ws.is_practice_verified is False
    report = await ws.wallet_diagnostic()
    assert report["classification"] == "CONFLICTING_ACCOUNT_INDICATORS"
    assert report["independent_indicators_agree"] is False
    assert report["indicator_sources"] == ["balance-changed", "get-balances.active", "profile"]


@pytest.mark.asyncio
async def test_stale_response_and_simultaneous_connections() -> None:
    first, second = socket(), socket()
    inventory(first, active=True)
    inventory(second)
    profile(first, balance_id=PRACTICE)
    profile(second)
    second._dispatch({"name": "profile", "request_id": "expired", "msg": {"balance_id": REAL}})
    a, b = await asyncio.gather(first.wallet_diagnostic(), second.wallet_diagnostic())
    assert a["classification"] == b["classification"] == "ACCOUNT_SELECTION_UNVERIFIED"
    assert a["independent_indicators_agree"] is True
    assert b["indicator_sources"] == []
    assert b["independent_indicators_agree"] is None


@pytest.mark.asyncio
async def test_missing_profile_with_partial_inventory() -> None:
    ws = socket()
    inventory(ws)
    assert (await ws.wallet_diagnostic())["classification"] == "MISSING_ACCOUNT_DATA"


@pytest.mark.parametrize(
    ("wallet", "expected"),
    [
        (REAL, PracticeStatus.VERIFIED_ACTIVE_REAL),
        (PRACTICE, PracticeStatus.VERIFIED_ACTIVE_PRACTICE),
    ],
)
def test_effective_wallet_owned_by_simulated_session(wallet: str, expected: PracticeStatus) -> None:
    balances = (
        Balance(REAL, AccountType.REAL, 1, "USD"),
        Balance(PRACTICE, AccountType.PRACTICE, 2, "USD"),
    )
    result = wallet_decision(
        balances, (WalletIndicator(WalletSource.SIMULATED_SESSION, wallet),), complete=True
    )
    assert result.status is expected
    assert result.is_verified is (wallet == PRACTICE)


def test_diagnostic_cli_emits_only_safe_fields_and_closes() -> None:
    ws = socket()
    profile(ws, balance_id=REAL)
    inventory(ws, default=True)
    client = AsyncMock()
    client.ws = ws

    client.assert_read_only = Mock()
    module = "iqoption_research.cli.commands.wallet_diagnostic"
    with (
        patch(f"{module}.AsyncIQOption", return_value=client),
        patch(
            f"{module}.getpass.getpass", side_effect=["private@invalid.test", "private-password"]
        ),
    ):
        result = CliRunner().invoke(app, ["wallet-diagnostic", "--live"])
    assert result.exit_code == 0
    assert json.loads(result.output)["classification"] == "ACCOUNT_SELECTION_UNVERIFIED"
    for sensitive in (PRACTICE, REAL, "private", "45678", "4321.99"):
        assert sensitive not in result.output
    client.close.assert_awaited_once()
    client.get_candles.assert_not_awaited()
