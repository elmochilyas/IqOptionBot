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
        "selected_local_context": None,
        "selection_scope": "connection_local_inspection",
        "broker_acknowledged_selection": False,
        "broker_verified_effective_wallet": False,
        "wallet_specific_operation_authorization": "DISABLED",
        "inventory_valid_for_selection": True,
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


@pytest.mark.asyncio
async def test_selections_are_inspection_only_and_connection_local() -> None:
    first, second = socket(), socket()
    for ws in (first, second):
        profile(ws, balance_id=REAL)
        inventory(ws, active=True)
        assert ws.wallet_context is None
    first.select_wallet_context(AccountType.PRACTICE)
    second.select_wallet_context(AccountType.REAL)
    assert first.wallet_report()["selected_local_context"] == "PRACTICE"
    report = second.wallet_report()
    assert report["selected_local_context"] == "REAL"
    assert report["classification"] == "CONFLICTING_ACCOUNT_INDICATORS"
    assert not report["broker_acknowledged_selection"]
    assert not report["broker_verified_effective_wallet"]
    assert report["wallet_specific_operation_authorization"] == "DISABLED"
    assert first.messages_tx == second.messages_tx == 0
    # Closing the transport clears the selection even if the same WS object is reused.
    await first.close()
    assert first.wallet_context is None
    assert second.wallet_context is not None


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"id": REAL, "type": 1}],
        [{"id": REAL, "type": 1}, {"id": REAL, "type": 4}],
        [{"id": "one", "type": 4}, {"id": "two", "type": 4}],
        [{"id": PRACTICE, "type": 4}, {"id": REAL, "type": True}],
        [{"id": PRACTICE, "type": 4, "balance_type": 1}],
    ],
)
def test_invalid_inventory_revokes_selection(rows: list[dict[str, object]]) -> None:
    ws = socket()
    inventory(ws)
    ws.select_wallet_context(AccountType.PRACTICE)
    ws._dispatch({"name": "balances", "msg": rows})
    assert ws.wallet_context is None
    with pytest.raises(ValueError):
        ws.select_wallet_context(AccountType.PRACTICE)
    # Stale correlated data cannot reinstate a selection.
    ws._dispatch({
        "name": "balances",
        "request_id": "expired",
        "msg": [{"id": PRACTICE, "type": 4}],
    })
    assert ws.wallet_context is None


def test_typed_profile_contradiction_revokes_selection_until_reselected() -> None:
    ws = socket()
    profile(ws, balance_id=REAL)
    inventory(ws)
    ws.select_wallet_context(AccountType.PRACTICE)
    profile(ws, current_balance={"id": PRACTICE, "type": 1})
    assert ws.wallet_context is None
    assert ws.wallet_classification is PracticeStatus.CONFLICTING_ACCOUNT_INDICATORS
    with pytest.raises(ValueError, match="disagree"):
        ws.select_wallet_context(AccountType.PRACTICE)
    profile(ws, current_balance={"id": PRACTICE, "type": 4})
    assert ws.wallet_context is None
    assert ws.select_wallet_context(AccountType.PRACTICE).balance_id == PRACTICE


def test_interactive_context_commands_share_one_connection() -> None:
    ws = socket()
    profile(ws, balance_id=REAL)
    inventory(ws)
    client = AsyncMock()
    client.ws = ws
    client.assert_read_only = Mock()
    module = "iqoption_research.cli.commands.wallet_diagnostic"
    with (
        patch(f"{module}.AsyncIQOption", return_value=client) as constructor,
        patch(
            f"{module}.getpass.getpass", side_effect=["private@invalid.test", "private-password"]
        ),
        patch("builtins.input", side_effect=["status", "select REAL", "exit"]),
    ):
        result = CliRunner().invoke(
            app, ["wallet", "--live", "--interactive", "--select", "practice"]
        )
    assert result.exit_code == 0
    snapshots = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert [r["selected_local_context"] for r in snapshots] == ["PRACTICE", "PRACTICE", "REAL"]
    constructor.assert_called_once()
    client.close.assert_awaited_once()
    assert ws.messages_tx == 0


@pytest.mark.parametrize(
    ("status", "valid"), [(0, True), (False, False), ("0", False), (4000, False)]
)
def test_wire_balance_status_controls_selection(status: object, valid: bool) -> None:
    ws = socket()
    ws._dispatch({
        "name": "balances",
        "request_id": "inventory",
        "status": status,
        "msg": [{"type": 4, "id": 101, "balance_id": "0101"}],
    })
    if valid:
        assert ws.select_wallet_context(AccountType.PRACTICE).balance_id == "101"
    else:
        with pytest.raises(ValueError):
            ws.select_wallet_context(AccountType.PRACTICE)
