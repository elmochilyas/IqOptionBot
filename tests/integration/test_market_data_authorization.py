"""Exercise the supervised entry point through its real connector and final socket write."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iqoption_research.connectors.iqoption.exceptions import TradingNotAllowedError
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.timing.supervised_session import (
    SupervisedSessionConfig,
    SupervisedSessionRunner,
)


class MarketSocket:
    def __init__(self, default: str) -> None:
        self.default = default
        self.sent: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[str] = asyncio.Queue()

    def __aiter__(self) -> "MarketSocket":
        return self

    async def __anext__(self) -> str:
        return await self.incoming.get()

    async def close(self) -> None:
        pass

    async def send(self, wire: str) -> None:
        frame = json.loads(wire)
        self.sent.append(frame)
        if frame["name"] == "authenticate":
            await self.incoming.put(json.dumps({"name": "authenticated", "msg": True}))
            return
        name = frame["msg"]["name"]
        if name == "get-profile":
            payload: object = {"balance_id": self.default}
        elif name == "get-balances":
            payload = (
                []
                if self.default == "missing"
                else [{"id": "real", "type": 1}, {"id": "practice", "type": 4}]
            )
        elif name == "get-candles":
            name = "candles"
            payload = {
                "active_id": 1,
                "candles": [
                    {
                        "from": 1799999940,
                        "to": 1800000000,
                        "at": 1800000000,
                        "open": 1.05,
                        "close": 1.05,
                        "min": 1.04,
                        "max": 1.06,
                        "volume": 100,
                    }
                ],
            }
        else:
            pytest.fail("Unexpected operation on the wire")
        await self.incoming.put(
            json.dumps({"name": name, "request_id": frame["request_id"], "msg": payload})
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("default", ["practice", "real", "missing"])
async def test_measurement_wire_is_wallet_independent(tmp_path: Path, default: str) -> None:
    wire = MarketSocket(default)
    runner = SupervisedSessionRunner(
        SupervisedSessionConfig(
            live_mode=True,
            simulation_mode=False,
            operator_confirmed=True,
            max_candle_cycles=1,
            project_root=tmp_path,
            output_destination=tmp_path / "measurement.json",
        )
    )
    fixed = datetime.fromtimestamp(1800000002, tz=UTC)
    connector = "iqoption_research.connectors.iqoption.client"
    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="synthetic")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire)),
        patch("asyncio.to_thread", side_effect=["synthetic@example.invalid", "synthetic"]),
        patch("iqoption_research.timing.supervised_session.datetime") as clock,
    ):
        clock.now.return_value = fixed
        evidence = await runner.execute()
    assert [f["name"] for f in wire.sent] == [
        "authenticate",
        "sendMessage",
        "sendMessage",
        "sendMessage",
    ]
    assert [f["msg"]["name"] for f in wire.sent[1:]] == [
        "get-profile",
        "get-balances",
        "get-candles",
    ]
    assert wire.sent[-1]["msg"]["body"] == {
        "active_id": 1,
        "size": 60,
        "count": 22,
        "to": 1800000000,
    }
    assert evidence.summary["authorization"] == "READ_ONLY_MARKET_DATA_AUTHORIZATION"
    expected = "MISSING_ACCOUNT_DATA" if default == "missing" else "ACCOUNT_SELECTION_UNVERIFIED"
    assert evidence.summary["wallet_classification_at_start"] == expected
    assert evidence.summary["wallet_specific_operations"] == "DISABLED"
    assert evidence.verify_integrity()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame",
    [
        {"name": "sendMessage", "msg": {"name": name, "body": {}}}
        for name in (
            "buyV3",
            "place-order",
            "withdraw",
            "transfer",
            "deposit",
            "reset-training-balance",
            "set-user-settings",
            "api_profile_changebalance",
        )
    ]
    + [
        {"name": "get-balances", "body": {}},
        {
            "name": "sendMessage",
            "msg": {"name": "get-profile", "version": "1.0", "body": {"balance_id": "real"}},
        },
        {
            "name": "subscribeMessage",
            "msg": {
                "name": "candle-generated",
                "params": {"routingFilters": {"active_id": 1, "size": 60}},
            },
        },
        {
            "name": "sendMessage",
            "msg": {
                "name": "get-candles",
                "body": {"active_id": 1, "size": 60, "user_balance_id": "real"},
            },
        },
    ],
)
async def test_measurement_rejects_out_of_scope_before_write(frame: dict[str, Any]) -> None:
    ws = IQOptionWebSocketClient("synthetic", config=ConnectionConfig(timing_market_data_only=True))
    sink = AsyncMock()
    ws._ws = sink
    ws._authenticated_ok = True
    with pytest.raises(TradingNotAllowedError):
        await ws._send_raw(frame)
    sink.send.assert_not_awaited()
