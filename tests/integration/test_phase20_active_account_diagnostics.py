"""Active-account classification distinguishes genuine REAL reports from ambiguous evidence."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

import iqoption_research.cli.commands.practice_watch as watch_cmd
from iqoption_research.cli.main import app
from iqoption_research.execution.practice_observation import (
    assess_active_account,
)
from iqoption_research.models.account import AccountType, Balance, Profile

PRACTICE = 764508381
REAL = 764508382
OTHER = 764508383


class FakeAccountObserver:
    def __init__(
        self,
        profile_raw: dict[str, Any] | None,
        *,
        outcome: str = "PROFILE_VERIFIED",
        practice_id: int | None = PRACTICE,
        real_id: int | None = REAL,
        balances: list[Balance] | None = None,
    ) -> None:
        self.profile = (
            Profile.from_raw({"user_id": 1, **profile_raw}) if profile_raw is not None else None
        )
        self.practice_balance = (
            Balance(str(practice_id), AccountType.PRACTICE, 100, "USD")
            if practice_id is not None
            else None
        )
        self.real_balance = (
            Balance(str(real_id), AccountType.REAL, 200, "USD") if real_id is not None else None
        )
        self.balances = (
            balances
            if balances is not None
            else [b for b in (self.real_balance, self.practice_balance) if b is not None]
        )
        self._outcome = outcome

    def profile_verification_diagnostics(self) -> dict[str, str]:
        return {"profile_outcome": self._outcome}


def _id_keys(value: int) -> dict[str, int]:
    return {"balance_id": value}


def test_genuine_real_active_is_preserved() -> None:
    result = assess_active_account(FakeAccountObserver(_id_keys(REAL)))
    assert result["practice_account_active"] == "ACTIVE_REAL_OR_OTHER"
    assert result["active_id_relation"] == "MATCHES_REAL"
    assert result["profile_source"] == "CORRELATED"
    assert result["endpoint_consistency"] == "CONSISTENT"
    assert result["classification_reason"] == "CORRELATED_PROFILE_MATCHES_REAL"


def test_genuine_practice_active_is_confirmed() -> None:
    result = assess_active_account(FakeAccountObserver(_id_keys(PRACTICE)))
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_CONFIRMED"
    assert result["active_id_relation"] == "MATCHES_PRACTICE"
    assert result["classification_reason"] == "CORRELATED_PROFILE_MATCHES_PRACTICE"


def test_conflicting_active_id_aliases_are_unverified() -> None:
    result = assess_active_account(
        FakeAccountObserver({"balance_id": PRACTICE, "active_balance_id": REAL})
    )
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert result["active_id_alias"] == "ALIASES_CONFLICT"
    assert result["classification_reason"] == "ACTIVE_ID_ALIASES_CONFLICT"


def test_agreeing_aliases_are_accepted() -> None:
    result = assess_active_account(
        FakeAccountObserver({"balance_id": PRACTICE, "balanceId": str(PRACTICE)})
    )
    assert result["active_id_alias"] == "MULTIPLE_AGREE"
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_CONFIRMED"


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("PROFILE_UNCORRELATED", "PROFILE_UNCORRELATED"),
        ("PROFILE_FROM_EARLIER_FRAME", "PROFILE_EARLIER_FRAME"),
        ("PROFILE_TIMEOUT", "PROFILE_UNKNOWN"),
    ],
)
def test_profile_freshness_is_required(outcome: str, expected: str) -> None:
    result = assess_active_account(FakeAccountObserver(_id_keys(REAL), outcome=outcome))
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert result["classification_reason"] == expected


def test_practice_type_with_unmatched_id_is_namespace_mismatch() -> None:
    result = assess_active_account(FakeAccountObserver({"balance_id": OTHER, "type": 4}))
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert result["profile_balance_type"] == "PRACTICE"
    assert result["active_id_relation"] == "MATCHES_NEITHER"
    assert result["endpoint_consistency"] == "TYPE_ID_NAMESPACE_MISMATCH"
    assert result["classification_reason"] == "PRACTICE_TYPE_ID_NAMESPACE_MISMATCH"


def test_balance_flag_conflict_with_profile_is_unverified() -> None:
    balances = [
        Balance(str(PRACTICE), AccountType.PRACTICE, 100, "USD", raw_is_active=True),
        Balance(str(REAL), AccountType.REAL, 200, "USD"),
    ]
    result = assess_active_account(FakeAccountObserver(_id_keys(REAL), balances=balances))
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert result["explicit_active_attribution"] == "PRACTICE"
    assert result["endpoint_consistency"] == "PROFILE_VS_BALANCE_CONFLICT"
    assert result["classification_reason"] == "PROFILE_BALANCE_EVIDENCE_CONFLICT"


def test_consistent_balance_flag_is_preserved() -> None:
    balances = [
        Balance(str(PRACTICE), AccountType.PRACTICE, 100, "USD"),
        Balance(str(REAL), AccountType.REAL, 200, "USD", raw_is_active=True),
    ]
    result = assess_active_account(FakeAccountObserver(_id_keys(REAL), balances=balances))
    assert result["practice_account_active"] == "ACTIVE_REAL_OR_OTHER"
    assert result["explicit_active_attribution"] == "REAL"
    assert result["endpoint_consistency"] == "CONSISTENT"


def test_missing_profile_or_wallet_is_unverified() -> None:
    missing_profile = assess_active_account(FakeAccountObserver(None))
    assert missing_profile["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert missing_profile["profile_source"] == "ABSENT"
    assert missing_profile["classification_reason"] == "PROFILE_OR_WALLET_MISSING"

    missing_wallet = assess_active_account(FakeAccountObserver(_id_keys(REAL), practice_id=None))
    assert missing_wallet["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"


def test_default_flag_attribution_and_invalid_alias() -> None:
    balances = [Balance(str(PRACTICE), AccountType.PRACTICE, 100, "USD", is_default=True)]
    result = assess_active_account(FakeAccountObserver(_id_keys(PRACTICE), balances=balances))
    assert result["default_attribution"] == "PRACTICE"
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_CONFIRMED"

    invalid = assess_active_account(FakeAccountObserver({"balance_id": "not-a-number"}))
    assert invalid["active_id_alias"] == "INVALID"
    assert invalid["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"


def test_classifications_expose_no_broker_values() -> None:
    result = assess_active_account(
        FakeAccountObserver(
            {"balance_id": PRACTICE, "type": 4},
            balances=[Balance(str(PRACTICE), AccountType.PRACTICE, 123456.78, "USD")],
        )
    )
    serialized = json.dumps(result)
    for private in (str(PRACTICE), str(REAL), "123456", "USD", "user_id"):
        assert private not in serialized


def test_cli_diagnostics_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "event_delivery": "LIVE_EVENT_DELIVERY_UNVERIFIED",
            "websocket_authentication": "VERIFIED",
            "practice_wallet": "VERIFIED",
            "practice_account_active": "ACTIVE_REAL_OR_OTHER",
            "real_wallet_status": "NO_CHANGE_BETWEEN_SNAPSHOTS",
            "position_lifecycle_counts": {},
            "position_samples": [],
        }

    monkeypatch.setattr(watch_cmd, "observe_practice", fake)
    monkeypatch.setenv("IQOPTION_EMAIL", "operator@example.invalid")
    monkeypatch.setenv("IQOPTION_PASSWORD", "synthetic-password")
    runner = CliRunner()

    plain = runner.invoke(app, ["practice-watch", "--live", "--asset", "EURUSD"])
    assert plain.exit_code == 2
    assert captured["active_account_diagnostics"] is False
    assert "active_account_diagnostics" not in plain.output

    detailed = runner.invoke(
        app,
        ["practice-watch", "--live", "--asset", "EURUSD", "--active-account-diagnostics"],
    )
    assert detailed.exit_code == 2
    assert captured["active_account_diagnostics"] is True
