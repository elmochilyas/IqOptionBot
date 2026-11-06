"""Tests for post-connection safety improvements.

Verifies:
1. Probe halts before market-data retrieval when active account is REAL or unverified.
2. describe_account removes user IDs, emails, country, balance IDs, and amounts.
3. describe_connection distinguishes readiness from practice authorization and forbids trading.
4. PracticeStatus explicitly distinguishes ACTIVE_REAL from ACTIVE_UNKNOWN.
5. Stale candle assessment includes accurate timestamp, staleness warning, and avoids
   assuming market closure without verified hours.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iqoption_research.cli.commands.probe import run_probe
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.account import (
    AccountStatus,
    AccountType,
    Balance,
    PracticeStatus,
    Profile,
)
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.security.credentials import Credentials


def _sample_credentials() -> Credentials:
    return Credentials(email="trader@example.com", password="SecurePassword123!")


def _sample_profile() -> Profile:
    return Profile(
        user_id=9876543,
        email="personal_user@example.com",
        first_name="Jane",
        last_name="Trader",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="GB",
        currency="USD",
    )


def test_describe_account_contains_no_private_identifiers() -> None:
    """Verifies that describe_account strips all user IDs, emails, country, IDs, and amounts."""
    client = AsyncIQOption(_sample_credentials(), offline=True)
    profile = _sample_profile()
    balances = (
        Balance(
            id="balance_secret_id_1111",
            type=AccountType.REAL,
            amount=54321.0,
            currency="USD",
            is_active=True,
        ),
        Balance(
            id="balance_secret_id_4444",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_active=False,
        ),
    )
    status = AccountStatus(
        profile=profile,
        balances=balances,
        active_balance=balances[0],
        practice_balance=balances[1],
        real_balance=balances[0],
    )

    desc = client.describe_account(status)

    # Personal identifiers removed
    assert "user_id" not in desc
    assert "email" not in desc
    assert "country" not in desc
    assert "first_name" not in desc
    assert "last_name" not in desc

    # Balances output privacy
    balances_desc = desc["balances"]
    assert isinstance(balances_desc, list)
    assert len(balances_desc) == 2
    for b in balances_desc:
        assert isinstance(b, dict)
        assert "id" not in b
        assert "id_last4" not in b
        assert "amount" not in b
        assert "type" in b
        assert "currency" in b
        assert "is_active" in b

    # Practice verification privacy
    pv_desc = desc["practice_verification"]
    assert isinstance(pv_desc, dict)
    assert "active_balance_id" not in pv_desc
    assert "practice_balance_ids" not in pv_desc
    assert pv_desc["status"] == PracticeStatus.ACTIVE_REAL.value
    assert pv_desc["is_verified"] is False


def test_describe_connection_distinguishes_readiness_and_forbids_trading() -> None:
    """Verifies describe_connection states practice authorization and strict read-only nature."""
    client = AsyncIQOption(_sample_credentials(), offline=True)
    desc = client.describe_connection()

    assert "ready" in desc
    assert "practice_verified" in desc
    assert "trade_permission" in desc
    assert "NEVER" in str(desc["trade_permission"])
    assert "Read-Only" in str(desc["trade_permission"])


def test_practice_status_distinguishes_active_real_from_active_unknown() -> None:
    """Verifies ACTIVE_REAL and ACTIVE_UNKNOWN are distinct diagnostic states."""
    assert str(PracticeStatus.ACTIVE_REAL.value) != str(PracticeStatus.ACTIVE_UNKNOWN.value)
    assert PracticeStatus.ACTIVE_REAL.value == "active_real"
    assert PracticeStatus.ACTIVE_UNKNOWN.value == "active_unknown"

    profile_data: dict[str, Any] = {"user_id": 1, "email": "a@b.com"}
    balances_data: list[dict[str, Any]] = [
        {"id": "r1", "type": 1, "amount": 100.0, "currency": "USD"},
        {"id": "p4", "type": 4, "amount": 10000.0, "currency": "USD"},
    ]

    # Active REAL
    status_real = AccountStatus.from_raw(profile_data, balances_data, active_balance_id="r1")
    ver_real = status_real.practice_verification()
    assert ver_real.status == PracticeStatus.ACTIVE_REAL
    assert ver_real.is_verified is False

    # Active UNKNOWN (unmatched ID)
    status_unknown = AccountStatus.from_raw(
        profile_data, balances_data, active_balance_id="missing_id"
    )
    ver_unknown = status_unknown.practice_verification()
    assert ver_unknown.status == PracticeStatus.ACTIVE_UNKNOWN
    assert ver_unknown.is_verified is False


def test_stale_candles_provide_accurate_timestamp_and_warning() -> None:
    """Verifies that stale candles produce exact timestamp warnings without auto-closing market."""
    now = datetime(2026, 11, 6, 23, 0, 0, tzinfo=UTC)
    # Candle that ended 3600s ago
    ended_ts = int((now - timedelta(seconds=3600)).timestamp())
    started_ts = ended_ts - 60
    candle = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=started_ts,
        to_ts=ended_ts,
        at_ts=0,
        open=1.0850,
        high=1.0855,
        low=1.0848,
        close=1.0852,
        volume=100,
        bid=1.0851,
        ask=1.0853,
        is_complete=True,
    )

    report = assess_data_quality([candle], expected_size=60, now_utc=now)
    assert report.stale is True
    assert report.newest_candle_end_ts == ended_ts
    assert report.stale_age_seconds is not None
    assert round(report.stale_age_seconds) == 3600

    report_dict = report.to_dict()
    assert report_dict["stale"] is True
    assert report_dict["newest_candle_utc"] is not None
    assert "2026-11-06T22:00:00" in report_dict["newest_candle_utc"]
    assert report_dict["staleness_warning"] is not None
    assert "trading schedule" in report_dict["staleness_warning"]

    # Issue message gives timestamp details and cautions against assuming closed
    stale_issues = [i for i in report.issues if i.code == "stale_data"]
    assert len(stale_issues) == 1
    assert "Newest candle ended at" in stale_issues[0].message
    assert "trading schedule" in stale_issues[0].message


@pytest.mark.asyncio
async def test_probe_halts_when_active_account_is_real() -> None:
    """Verifies that run_probe halts before candle retrieval when active account is REAL."""
    mock_status = AccountStatus(
        profile=_sample_profile(),
        balances=(
            Balance("r1", AccountType.REAL, 500.0, "USD", is_active=True),
            Balance("p4", AccountType.PRACTICE, 10000.0, "USD", is_active=False),
        ),
        active_balance=Balance("r1", AccountType.REAL, 500.0, "USD", is_active=True),
        practice_balance=Balance("p4", AccountType.PRACTICE, 10000.0, "USD", is_active=False),
        real_balance=Balance("r1", AccountType.REAL, 500.0, "USD", is_active=True),
    )

    mock_client = AsyncMock()
    mock_client.is_ready = True
    mock_client.state = None
    mock_client.connect = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.describe_connection = MagicMock(return_value={"state": "ready", "ready": True})
    mock_client.get_account_status = AsyncMock(return_value=mock_status)
    mock_client.describe_account = MagicMock(
        return_value={
            "balances": [{"type": "REAL", "currency": "USD", "is_active": True}],
            "practice_verification": {"status": "active_real", "is_verified": False},
        }
    )

    with (
        patch(
            "iqoption_research.cli.commands.probe.get_credentials",
            return_value=_sample_credentials(),
        ),
        patch("iqoption_research.cli.commands.probe.AsyncIQOption", return_value=mock_client),
    ):
        summary = await run_probe(
            symbol="EURUSD",
            size=60,
            count=10,
            save=False,
            output_dir="data",
            diagnostics=False,
        )

    # Market-data retrieval MUST have been halted
    mock_client.get_candles.assert_not_called()

    assert summary["practice_verified"] is False
    assert summary["practice_status"] == PracticeStatus.ACTIVE_REAL.value
    assert "candles" not in summary
    assert "PROBE HALTED" in str(summary.get("warning"))
