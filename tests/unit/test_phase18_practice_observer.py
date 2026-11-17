"""Unit tests for Phase 18 Passive Practice Transaction Observer.

Verifies:
1. Strict read-only invariant (zero order submission methods exist).
2. Wire event parsing for option-opened, portfolio.position-changed, and option-closed.
3. Balance mutation tracking (Practice decrement/increment, Real watchdog).
4. Deterministic offline mock behavior for normal trades, anomalies, and timeouts.
5. CLI command practice-observe offline execution, safety verdicts, and JSON audit recording.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.execution.practice_observer import (
    MockPassivePracticeObserver,
    ObservedPracticeTradeRecord,
    PassivePracticeObserver,
)
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.security.credentials import Credentials

runner = CliRunner()


def test_observer_read_only_invariants() -> None:
    """Observer classes must NOT contain any order submission or trading methods."""
    forbidden_methods = {
        "buy",
        "sell",
        "buy_v2",
        "place_order",
        "submit_order",
        "send_order_frame",
        "open_option",
        "change_balance",
    }
    for method in forbidden_methods:
        assert not hasattr(PassivePracticeObserver, method), (
            f"PassivePracticeObserver must not have '{method}'"
        )
        assert not hasattr(MockPassivePracticeObserver, method), (
            f"MockPassivePracticeObserver must not have '{method}'"
        )


def test_mock_observer_normal_practice_trade() -> None:
    """Mock observer accurately simulates a manual Practice trade routing to Practice wallet."""
    mock_obs = MockPassivePracticeObserver(
        practice_balance_id="101",
        real_balance_id="200",
        initial_practice_amount=10000.0,
        initial_real_amount=50.0,
        ticket_id="ticket_test_999",
        stake=1.00,
        profit=1.85,
    )

    record = mock_obs.observe()
    assert record is not None
    assert isinstance(record, ObservedPracticeTradeRecord)
    assert record.ticket_id == "ticket_test_999"
    assert record.user_balance_id == "101"
    assert record.is_practice_balance is True
    assert record.practice_balance_id == "101"
    assert record.real_balance_id == "200"
    assert record.real_wallet_untouched is True
    assert record.initial_practice_amount == 10000.0
    assert record.post_open_practice_amount == 9999.0
    assert record.final_practice_amount == 10000.85
    assert record.initial_real_amount == 50.0
    assert record.final_real_amount == 50.0
    assert record.settlement_status == "win"
    assert record.settlement_profit == 1.85
    assert "CORRELATED_PRACTICE_CONFIRMED" in record.correlation_verdict

    # Check serialization
    d = record.to_dict()
    assert d["ticket_id"] == "ticket_test_999"
    assert d["user_balance_id"] == "101"
    assert d["is_practice_balance"] is True
    assert d["real_wallet_untouched"] is True
    # Round-trip JSON test
    json_str = json.dumps(d)
    assert "ticket_test_999" in json_str


def test_mock_observer_anomalous_real_trade() -> None:
    """Mock observer detects and flags an anomalous trade routing to the REAL account."""
    mock_obs = MockPassivePracticeObserver(
        practice_balance_id="101",
        real_balance_id="200",
        simulate_real_trade_anomaly=True,
    )

    record = mock_obs.observe()
    assert record is not None
    assert record.user_balance_id == "200"
    assert record.is_practice_balance is False
    assert record.real_wallet_untouched is False
    assert "CRITICAL_SAFETY_BREACH" in record.correlation_verdict


def test_mock_observer_timeout() -> None:
    """Mock observer returns None when simulate_timeout is set."""
    mock_obs = MockPassivePracticeObserver(simulate_timeout=True)
    record = mock_obs.observe()
    assert record is None


def test_passive_practice_observer_frame_parsing() -> None:
    """Verify internal message frame processing in PassivePracticeObserver."""
    creds = Credentials(email="test@example.com", password="dummy_password")
    obs = PassivePracticeObserver(credentials=creds)

    # Mock initial profile & balances
    obs._profile = Profile.from_raw({"user_id": 1, "balance_id": 200, "email": "test@example.com"})
    obs._balances = [
        Balance(
            id="101",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_active=True,
        ),
        Balance(
            id="200",
            type=AccountType.REAL,
            amount=50.0,
            currency="USD",
            is_active=False,
        ),
    ]
    obs._practice_balance = obs._balances[0]
    obs._real_balance = obs._balances[1]

    # 1. Option opened frame
    open_frame: dict[str, Any] = {
        "name": "option-opened",
        "msg": {
            "id": "mock_ticket_888",
            "active_id": 1,
            "dir": "call",
            "amount": 1.0,
            "exp": 1700000060,
            "user_balance_id": 101,
        },
    }
    obs._handle_incoming_frame(open_frame)

    assert obs._observed_position is not None
    assert obs._observed_position.ticket_id == "mock_ticket_888"
    assert obs._observed_position.user_balance_id == "101"
    assert obs._observed_position.active_id == 1
    assert obs._observed_position.direction == "call"
    assert obs._observed_position.stake == 1.0
    assert obs._position_event.is_set()

    # 2. Balance changed frame (Practice stake decrement)
    bal_change_open: dict[str, Any] = {
        "name": "balance-changed",
        "msg": {
            "id": "101",
            "current_balance": 9999.0,
        },
    }
    obs._handle_incoming_frame(bal_change_open)
    assert len(obs._balance_mutations) == 1
    assert obs._balance_mutations[0].balance_id == "101"
    assert obs._balance_mutations[0].amount_before == 10000.0
    assert obs._balance_mutations[0].amount_after == 9999.0
    assert obs._balance_mutations[0].delta == -1.0

    # 3. Option closed frame (Settlement)
    close_frame: dict[str, Any] = {
        "name": "option-closed",
        "msg": {
            "id": "mock_ticket_888",
            "profit": 1.85,
            "status": "win",
            "close_price": 1.0855,
            "close_time": 1700000060,
        },
    }
    obs._handle_incoming_frame(close_frame)
    assert obs._observed_settlement is not None
    assert obs._observed_settlement.ticket_id == "mock_ticket_888"
    assert obs._observed_settlement.profit == 1.85
    assert obs._observed_settlement.status == "win"
    assert obs._settlement_event.is_set()

    # 4. Balance changed frame (Settlement payout)
    bal_change_settle: dict[str, Any] = {
        "name": "balance-changed",
        "msg": {
            "id": "101",
            "current_balance": 10000.85,
        },
    }
    obs._handle_incoming_frame(bal_change_settle)
    assert len(obs._balance_mutations) == 2
    assert obs._balance_mutations[1].delta == 1.85


def test_passive_practice_observer_portfolio_frame_parsing() -> None:
    """Verify parsing of portfolio.position-changed push frames."""
    creds = Credentials(email="test@example.com", password="dummy_password")
    obs = PassivePracticeObserver(credentials=creds)

    port_frame: dict[str, Any] = {
        "name": "portfolio.position-changed",
        "msg": {
            "status": "open",
            "id": "port_ticket_777",
            "user_balance_id": "101",
            "raw_event": {
                "active_id": 1,
                "direction": "put",
                "invest": 2.50,
                "expire_at": 1700000120,
            },
        },
    }
    obs._handle_incoming_frame(port_frame)
    assert obs._observed_position is not None
    assert obs._observed_position.ticket_id == "port_ticket_777"
    assert obs._observed_position.direction == "put"
    assert obs._observed_position.stake == 2.50
    assert obs._observed_position.user_balance_id == "101"


def test_cli_practice_observe_offline_mock() -> None:
    """CLI practice-observe command runs successfully in offline mock mode."""
    res = runner.invoke(app, ["practice-observe", "--offline-mock"])
    assert res.exit_code == 0
    assert "IQ Option Passive Practice Transaction Observer" in res.output
    assert "STRICT READ-ONLY GUARANTEE" in res.output
    assert "MATCHES PRACTICE" in res.output
    assert "VERIFIED UNTOUCHED" in res.output
    assert "PASS (SAFE)" in res.output


def test_cli_practice_observe_mock_anomaly_fails_closed() -> None:
    """CLI practice-observe command fails closed when real account anomaly is simulated."""
    res = runner.invoke(app, ["practice-observe", "--offline-mock", "--mock-anomaly"])
    assert res.exit_code == 1
    assert "ANOMALY: MATCHES REAL" in res.output
    assert "CRITICAL SAFETY BREACH" in res.output
    assert "CRITICAL SAFETY VERDICT" in res.output


def test_cli_practice_observe_mock_timeout() -> None:
    """CLI practice-observe command handles timeout cleanly."""
    res = runner.invoke(app, ["practice-observe", "--offline-mock", "--mock-timeout"])
    assert res.exit_code == 0
    assert "Observation Window Closed" in res.output
    assert "Zero orders were submitted by the bot" in res.output


def test_cli_practice_observe_record_json(tmp_path: Path) -> None:
    """CLI practice-observe saves structured audit record when --record is supplied."""
    record_file = tmp_path / "observed_trade.json"
    res = runner.invoke(
        app,
        ["practice-observe", "--offline-mock", "--record", str(record_file)],
    )
    assert res.exit_code == 0
    assert record_file.exists()

    data = json.loads(record_file.read_text(encoding="utf-8"))
    assert data["ticket_id"] == "mock_manual_ticket_12345"
    assert data["user_balance_id"] == "101"
    assert data["is_practice_balance"] is True
    assert data["real_wallet_untouched"] is True
    assert data["stake"] == 1.0


def test_cli_practice_observe_mutually_exclusive() -> None:
    """CLI practice-observe rejects simultaneous --live and --offline-mock."""
    res = runner.invoke(app, ["practice-observe", "--live", "--offline-mock"])
    assert res.exit_code != 0
    assert "Cannot specify both" in res.output
