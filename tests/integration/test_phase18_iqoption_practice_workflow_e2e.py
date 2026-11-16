"""End-to-End Integration tests for Phase 18 IQ Option Practice Execution Workflow."""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.commands.execution_cmd import execution_app
from iqoption_research.cli.main import app
from iqoption_research.execution.broker_readiness import PracticeExecutionAuthToken
from iqoption_research.execution.execution_engine import (
    ModularExecutionConfig,
    ModularExecutionEngine,
)
from iqoption_research.execution.live_practice_transport import (
    IQOptionLivePracticeTransport,
    IQOptionLivePracticeTransportConfig,
)
from iqoption_research.execution.models import (
    OrderSide,
    OrderState,
    ProductType,
)
from iqoption_research.execution.modes import (
    ExecutionMode,
    ExecutionModeConfig,
)
from iqoption_research.execution.state_machine import ExecutionJournal
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.market import Candle, CandleSize


class MockWebSocketE2E:
    """Mock WebSocket for end-to-end integration testing."""

    def __init__(self, initial_messages: list[str] | None = None) -> None:
        self._initial = list(initial_messages or [])
        self._queue: Any = None
        self._loop: Any = None
        self.sent_messages: list[str] = []
        self.active_balance_id: int = 101
        self.current_balance: float = 10000.0
        self.send = AsyncMock(side_effect=self._record_send)

    def _ensure_queue(self) -> Any:
        import asyncio

        if self._queue is None:
            self._loop = asyncio.get_running_loop()
            self._queue = asyncio.Queue()
            for msg in self._initial:
                self._queue.put_nowait(msg)
        return self._queue

    async def _record_send(self, data: str) -> None:
        self.sent_messages.append(data)
        with contextlib.suppress(Exception):
            parsed = json.loads(data)
            msg_obj = parsed.get("msg")
            msg_name = msg_obj.get("name") if isinstance(msg_obj, dict) else None

            if msg_name == "binary-options.open-option":
                req_id = parsed.get("request_id", "req_e2e")
                self.inject_message({
                    "name": "option-opened",
                    "request_id": req_id,
                    "status": "success",
                    "msg": {
                        "id": "ticket_e2e_7777",
                        "active_id": 1,
                        "price": 1.0,
                        "created": 1700000000,
                    },
                })
            elif msg_name == "get-profile":
                self.inject_message({
                    "name": "profile",
                    "msg": {"user_id": 999123, "balance_id": self.active_balance_id},
                })
            elif msg_name == "get-balances":
                self.inject_message({
                    "name": "balances",
                    "msg": [
                        {
                            "id": self.active_balance_id,
                            "type": 4,
                            "amount": self.current_balance,
                            "currency": "USD",
                            "is_active": True,
                        }
                    ],
                })

    def inject_message(self, message: dict[str, Any] | str) -> None:
        import asyncio

        text = json.dumps(message) if isinstance(message, dict) else message
        q = self._ensure_queue()
        try:
            cur_loop = asyncio.get_running_loop()
        except RuntimeError:
            cur_loop = None

        if self._loop is not None and cur_loop != self._loop:
            self._loop.call_soon_threadsafe(q.put_nowait, text)
        else:
            q.put_nowait(text)

    async def close(self, *args: Any, **kwargs: Any) -> None:
        q = self._ensure_queue()
        q.put_nowait(None)

    def __aiter__(self) -> MockWebSocketE2E:
        return self

    async def __anext__(self) -> str:
        q = self._ensure_queue()
        msg = await q.get()
        if msg is None:
            raise StopAsyncIteration
        return msg


def make_handshake_messages() -> list[str]:
    return [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {
                "user_id": 999123,
                "email": "trader@test.com",
                "first_name": "Demo",
                "last_name": "Trader",
                "balance_id": 101,
            },
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {
                    "id": 101,
                    "type": 4,
                    "amount": 10000.0,
                    "currency": "USD",
                    "is_active": True,
                }
            ],
        }),
    ]


@pytest.mark.asyncio
async def test_live_practice_full_lifecycle_e2e() -> None:
    mock_ws = MockWebSocketE2E(make_handshake_messages())

    async def fake_connect() -> MockWebSocketE2E:
        return mock_ws

    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "execution_practice_e2e.db"
        journal = ExecutionJournal(journal_path)

        tok = PracticeExecutionAuthToken(
            authorization_signature="sig_supervisor_e2e",
            supervisor_id="supervisor_cli_operator",
        )
        mode_cfg = ExecutionModeConfig(
            mode=ExecutionMode.PRACTICE,
            practice_enabled=True,
            practice_auth_token=tok,
        )
        engine = ModularExecutionEngine(
            ModularExecutionConfig(
                symbol="EURUSD",
                mode=ExecutionMode.PRACTICE,
                default_product_type=ProductType.BINARY,
                initial_balance=10000.0,
                mode_config=mode_cfg,
                single_test_order_mode=True,
            ),
            journal=journal,
        )

        transport = IQOptionLivePracticeTransport(
            ssid="test_mock_ssid",
            config=IQOptionLivePracticeTransportConfig(request_timeout=2.0, settlement_timeout=2.0),
            connect_factory=fake_connect,
        )

        try:
            # 1. Connect & Pre-flight verification
            await transport.async_connect()
            assert transport.is_authenticated is True
            assert transport.active_balance_id == "101"

            # 2. Candle for entry
            c1 = Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=1700000000,
                to_ts=1700000060,
                at_ts=1700000060_000_000_000,
                open=1.0850,
                high=1.0855,
                low=1.0848,
                close=1.0852,
                volume=100,
                bid=1.0852,
                ask=1.0852,
            )

            # 3. Create proposal and validate
            prop = engine.create_proposal(
                symbol="EURUSD",
                side=OrderSide.CALL,
                stake=1.0,
                candle_to_ts=float(c1.to_ts),
                signal_id="sig_e2e_live_01",
            )
            order, dec = engine.propose_and_validate(
                prop,
                current_ts=float(c1.to_ts),
                profile=transport.profile,
                balances=transport.balances,
            )
            assert dec.permitted is True
            assert order.state == OrderState.APPROVED

            # 4. Dispatch through transport and register in engine
            adapter = engine.adapters[ProductType.BINARY]
            frame = adapter.practice_transport.build_order_frame(
                order, user_balance_id=101, expiry_ts=int(c1.to_ts + 60)
            )
            resp = await transport.async_send_order_frame(frame)
            assert resp["name"] == "option-opened"
            broker_ticket = transport.active_ticket_id
            assert broker_ticket == "ticket_e2e_7777"

            # Submit through engine to track position
            mock_bridge = type("MockBridge", (), {"send_order_frame": lambda _self, _f: resp})()
            engine.submit_order(order, c1, broker_transport=mock_bridge)
            assert order.state == OrderState.OPEN
            assert order.broker_order_id == "ticket_e2e_7777"
            assert len(engine._open_positions) == 1

            # 5. Settlement injection and parsing
            mock_ws.current_balance = 10000.85
            mock_ws.inject_message({
                "name": "option-closed",
                "msg": {
                    "option_id": broker_ticket,
                    "status": "win",
                    "profit": 0.85,
                    "close_price": 1.0858,
                    "close_time": 1700000120,
                },
            })

            settle_res = await transport.async_wait_for_settlement(
                broker_ticket, timeout_seconds=2.0
            )
            assert settle_res.settled is True
            assert settle_res.outcome == "win"
            assert settle_res.net_pnl == 0.85
            assert settle_res.is_financially_consistent is True

            # 6. Settle in engine with expiry candle
            c2 = Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=1700000060,
                to_ts=1700000120,
                at_ts=1700000120_000_000_000,
                open=1.0852,
                high=1.0862,
                low=1.0850,
                close=1.0858,
                volume=120,
                bid=1.0858,
                ask=1.0858,
            )
            step_res = engine.process_candle(c2)
            assert len(step_res.settled_orders) == 1
            settled_order = step_res.settled_orders[0]
            assert settled_order.state == OrderState.SETTLED
            assert settled_order.simulated_pnl == 0.85

            # 7. Verify persistent journal entries
            saved_orders = engine.journal.load_all_orders()
            assert len(saved_orders) == 1
            saved = next(iter(saved_orders.values()))
            assert saved.broker_order_id == "ticket_e2e_7777"
            assert saved.state == OrderState.SETTLED
        finally:
            await transport.async_close()
            journal.close()


def test_cli_practice_test_offline_mock() -> None:
    runner = CliRunner()
    result = runner.invoke(
        execution_app,
        [
            "practice-test",
            "--offline-mock",
            "--symbol",
            "EURUSD",
            "--side",
            "call",
            "--stake",
            "1.0",
        ],
        input="y\n",
    )
    assert result.exit_code == 0
    assert "Operator Supervised Practice Test" in result.stdout
    assert "Practice Test Order Successfully Placed" in result.stdout
    assert "mock_ticket_cli_supervised" in result.stdout


def test_cli_practice_test_abort_on_no_confirmation() -> None:
    runner = CliRunner()
    result = runner.invoke(
        execution_app,
        ["practice-test", "--offline-mock", "--symbol", "EURUSD"],
        input="n\n",
    )
    assert result.exit_code != 0
    assert "Aborted by operator" in result.stdout


def test_single_test_order_invariant_blocks_second_order() -> None:
    tok = PracticeExecutionAuthToken(
        authorization_signature="sig_invariant_test",
        supervisor_id="supervisor_cli_operator",
    )
    mode_cfg = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=tok,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "execution_inv.db"
        journal = ExecutionJournal(journal_path)
        try:
            engine = ModularExecutionEngine(
                ModularExecutionConfig(
                    symbol="EURUSD",
                    mode=ExecutionMode.PRACTICE,
                    default_product_type=ProductType.BINARY,
                    initial_balance=10000.0,
                    mode_config=mode_cfg,
                    single_test_order_mode=True,
                ),
                journal=journal,
            )

            prof = Profile.from_raw({"user_id": 1, "balance_id": 101})
            bals = [
                Balance(
                    id="101",
                    type=AccountType.PRACTICE,
                    amount=10000.0,
                    currency="USD",
                    is_active=True,
                )
            ]
            c1 = Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=1700000000,
                to_ts=1700000060,
                at_ts=1700000060_000_000_000,
                open=1.0850,
                high=1.0855,
                low=1.0848,
                close=1.0852,
                volume=100,
                bid=1.0852,
                ask=1.0852,
            )

            p1 = engine.create_proposal(
                symbol="EURUSD",
                side=OrderSide.CALL,
                stake=1.0,
                candle_to_ts=float(c1.to_ts),
                signal_id="sig_inv_01",
            )
            o1, dec1 = engine.propose_and_validate(
                p1, current_ts=float(c1.to_ts), profile=prof, balances=bals
            )
            assert dec1.permitted is True

            # Submit first order
            mock_tp = type(
                "MockTP",
                (),
                {
                    "send_order_frame": lambda _self, _f: {
                        "name": "option-opened",
                        "msg": {"id": "tick_1"},
                    }
                },
            )()
            engine.submit_order(o1, c1, broker_transport=mock_tp)
            assert o1.state == OrderState.OPEN

            # Propose second order while first is OPEN
            p2 = engine.create_proposal(
                symbol="EURUSD",
                side=OrderSide.PUT,
                stake=1.0,
                candle_to_ts=float(c1.to_ts),
                signal_id="sig_inv_02",
            )
            _o2, dec2 = engine.propose_and_validate(
                p2, current_ts=float(c1.to_ts), profile=prof, balances=bals
            )
            assert dec2.permitted is False
            assert "Single test order invariant" in dec2.message
        finally:
            journal.close()


def test_cli_practice_diagnostics_offline_mock() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["practice-diagnostics", "--offline-mock", "--asset", "EURUSD"],
    )
    assert result.exit_code == 0
    assert "Read-Only Diagnostics" in result.stdout
    assert "PRACTICE (type 4 - Verified)" in result.stdout
    assert "Zero Orders / Zero State Changes" in result.stdout
    assert "ALL CHECKS PASSED" in result.stdout


def test_cli_practice_test_top_level_command_with_asset_alias() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "practice-test",
            "--offline-mock",
            "--asset",
            "EURUSD",
            "--side",
            "call",
            "--stake",
            "1.0",
        ],
        input="y\n",
    )
    assert result.exit_code == 0
    assert "Operator Supervised Practice Test" in result.stdout
    assert "mock_ticket_cli_supervised" in result.stdout


def test_cli_execution_subcommand_aliases() -> None:
    runner = CliRunner()
    # Test execution practice-diagnostics
    res_diag = runner.invoke(
        app,
        ["execution", "practice-diagnostics", "--offline-mock"],
    )
    assert res_diag.exit_code == 0
    assert "Offline Diagnostic Findings" in res_diag.stdout

    # Test execution practice-test
    res_test = runner.invoke(
        app,
        ["execution", "practice-test", "--offline-mock"],
        input="y\n",
    )
    assert res_test.exit_code == 0
    assert "Practice Test Order Successfully Placed" in res_test.stdout
