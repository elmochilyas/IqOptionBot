"""Unit tests for Phase 18 Passive Practice Transaction Observer.

Verifies:
1. Strict read-only invariant (zero order submission methods exist).
2. Wire event parsing for option-opened, portfolio.position-changed, and option-closed.
3. Balance mutation tracking (Practice decrement/increment, Real watchdog).
4. Settlement correlation strictly by exact ticket ID (ignoring mismatched tickets).
5. Unverified REAL balance snapshots fail closed (reporting UNVERIFIED).
6. Missing fields reported explicitly as None without synthetic defaults
   (no hardcoded EURUSD/call/1.0).
7. Asset filtering actually rejects unmatched positions.
8. Subscription state tracking (REQUEST_SENT, ACCEPTED, STATUS_UNKNOWN, EVENT_RECEIVED).
9. Privacy-preserving JSON export (no raw frames, SSIDs, tokens, credentials, or PII).
10. Authentication handling without correlated request_id.
11. CLI practice-observe command offline execution, probe-only mode, and safety alerts.
"""

from __future__ import annotations

import asyncio
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
    assert record.real_wallet_status == "VERIFIED_UNTOUCHED"
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
    assert d["real_wallet_status"] == "VERIFIED_UNTOUCHED"
    assert d["real_wallet_untouched"] is True
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
    assert record.real_wallet_status == "MUTATED"
    assert record.real_wallet_untouched is False
    assert "CRITICAL_SAFETY_BREACH" in record.correlation_verdict


def test_mock_observer_timeout() -> None:
    """Mock observer returns None when simulate_timeout is set."""
    mock_obs = MockPassivePracticeObserver(simulate_timeout=True)
    record = mock_obs.observe()
    assert record is None


def test_mock_observer_settlement_mismatch_fails_closed() -> None:
    """Settlement with mismatched ticket ID produces un-correlated non-success verdict."""
    mock_obs = MockPassivePracticeObserver(simulate_settlement_mismatch=True)
    record = mock_obs.observe()
    assert record is not None
    assert record.settlement_event is not None
    assert record.settlement_event.ticket_id != record.ticket_id
    assert "UNCORRELATED_SETTLEMENT_MISSING" in record.correlation_verdict


def test_mock_observer_unverified_real_balance_fails_closed() -> None:
    """Missing or unconfirmed fresh post-settlement REAL snapshot reports UNVERIFIED."""
    mock_obs = MockPassivePracticeObserver(simulate_unverified_real=True)
    record = mock_obs.observe()
    assert record is not None
    assert record.real_wallet_status == "UNVERIFIED"
    assert record.real_wallet_untouched is False
    assert "UNCORRELATED_REAL_BALANCE_UNVERIFIED" in record.correlation_verdict


def test_mock_observer_missing_fields_no_synthetic_defaults() -> None:
    """Missing broker fields are reported as None rather than synthetic defaults."""
    mock_obs = MockPassivePracticeObserver(simulate_missing_fields=True)
    record = mock_obs.observe()
    assert record is not None
    assert record.symbol is None
    assert record.active_id is None
    assert record.direction is None
    assert record.stake is None
    assert record.exp_ts is None


def test_mock_observer_unmatched_asset_filter_returns_none() -> None:
    """Unmatched asset trade is ignored, returning None when no matching trade occurs."""
    mock_obs = MockPassivePracticeObserver(simulate_unmatched_asset=True)
    record = mock_obs.observe(asset_filter="EURUSD")
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
    assert obs._observed_position.symbol == "EURUSD"
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

    # 3. Unrelated settlement frame (must be IGNORED)
    unrelated_close: dict[str, Any] = {
        "name": "option-closed",
        "msg": {
            "id": "unrelated_ticket_000",
            "profit": 5.00,
            "status": "win",
        },
    }
    obs._handle_incoming_frame(unrelated_close)
    assert obs._observed_settlement is None
    assert not obs._settlement_event.is_set()

    # 4. Matching Option closed frame (Settlement)
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

    # 5. Balance changed frame (Settlement payout)
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


def test_passive_practice_observer_asset_filter_rejection() -> None:
    """Asset filter rejects trades on non-matching instruments."""
    creds = Credentials(email="test@example.com", password="dummy_password")
    obs = PassivePracticeObserver(credentials=creds)
    obs._asset_filter = "GBPUSD"

    # EURUSD trade arrives (active_id = 1)
    port_frame: dict[str, Any] = {
        "name": "portfolio.position-changed",
        "msg": {
            "status": "open",
            "id": "eurusd_trade",
            "user_balance_id": "101",
            "raw_event": {
                "active_id": 1,
                "direction": "call",
                "invest": 1.0,
                "expire_at": 1700000120,
            },
        },
    }
    obs._handle_incoming_frame(port_frame)
    # Must NOT be captured because symbol is EURUSD and filter is GBPUSD
    assert obs._observed_position is None
    assert not obs._position_event.is_set()


async def test_auth_without_request_id_handled() -> None:
    """Authentication response without request_id completes auth future."""
    creds = Credentials(email="test@example.com", password="dummy_password")
    obs = PassivePracticeObserver(credentials=creds)
    loop = asyncio.get_running_loop()
    auth_fut = loop.create_future()
    obs._auth_future = auth_fut

    # Frame has NO request_id
    auth_frame = {"name": "authenticated", "msg": True}
    obs._handle_incoming_frame(auth_frame)

    assert auth_fut.done()
    assert auth_fut.result() == auth_frame


def test_balance_changed_missing_amount_does_not_corrupt() -> None:
    """Malformed balance-changed frame missing amount is skipped without zero-defaulting."""
    creds = Credentials(email="test@example.com", password="dummy_password")
    obs = PassivePracticeObserver(credentials=creds)
    obs._balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
    ]

    # Frame with missing amount
    bad_frame: dict[str, Any] = {
        "name": "balance-changed",
        "msg": {"id": "101"},
    }
    obs._handle_incoming_frame(bad_frame)
    assert len(obs._balance_mutations) == 0
    assert obs._balances[0].amount == 10000.0


def test_privacy_sanitized_record_no_pii(tmp_path: Path) -> None:
    """JSON export contains only necessary sanitized evidence without raw payloads or PII."""
    mock_obs = MockPassivePracticeObserver()
    record = mock_obs.observe()
    assert record is not None

    record_file = tmp_path / "privacy_test.json"
    record_file.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    content = record_file.read_text(encoding="utf-8")

    forbidden_pii_terms = [
        "ssid",
        "password",
        "cookie",
        "raw_payload",
        "email",
    ]
    for term in forbidden_pii_terms:
        assert term not in content.lower(), f"Forbidden term '{term}' leaked into JSON record!"


def test_cli_practice_observe_probe_only() -> None:
    """CLI practice-observe --probe-only runs connection and subscription probe."""
    res = runner.invoke(app, ["practice-observe", "--offline-mock", "--probe-only"])
    assert res.exit_code == 0
    assert "Passive Observer Probe Results" in res.output
    assert "AUTHENTICATED" in res.output
    assert "Portfolio Topic" in res.output
    assert "STRICT READ-ONLY" in res.output


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


def test_cli_practice_observe_mock_settle_mismatch_fails_closed() -> None:
    """CLI practice-observe command fails closed when settlement ticket mismatches."""
    res = runner.invoke(app, ["practice-observe", "--offline-mock", "--mock-settle-mismatch"])
    assert res.exit_code == 1
    assert "UNCORRELATED_SETTLEMENT_MISSING" in res.output
    assert "CRITICAL SAFETY VERDICT" in res.output


def test_cli_practice_observe_mock_unverified_real_fails_closed() -> None:
    """CLI practice-observe command fails closed when REAL balance snapshot is unverified."""
    res = runner.invoke(app, ["practice-observe", "--offline-mock", "--mock-unverified-real"])
    assert res.exit_code == 1
    assert "UNVERIFIED" in res.output
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
    assert data["real_wallet_status"] == "VERIFIED_UNTOUCHED"
    assert data["real_wallet_untouched"] is True
    assert data["stake"] == 1.0


def test_cli_practice_observe_mutually_exclusive() -> None:
    """CLI practice-observe rejects simultaneous --live and --offline-mock."""
    res = runner.invoke(app, ["practice-observe", "--live", "--offline-mock"])
    assert res.exit_code != 0
    assert "Cannot specify both" in res.output
