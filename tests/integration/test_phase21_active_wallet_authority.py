"""Profile balance_id reports identity, not authoritative active-wallet selection."""

from __future__ import annotations

import json
from typing import Any

from iqoption_research.execution.practice_observation import assess_active_account
from iqoption_research.models.account import AccountType, Balance, Profile

PRACTICE = 764508381
REAL = 764508382
UNKNOWN = 764508383


class Observer:
    """Minimal observer exposing only the fields the classifier reads."""

    def __init__(
        self,
        profile_raw: dict[str, Any],
        *,
        explicit: str | None = None,
        default: str | None = None,
        outcome: str = "PROFILE_VERIFIED",
    ) -> None:
        self.profile = Profile.from_raw({"user_id": 1, **profile_raw})
        self.practice_balance = Balance(
            str(PRACTICE),
            AccountType.PRACTICE,
            100,
            "USD",
            raw_is_active=explicit == "PRACTICE",
            is_default=default == "PRACTICE",
        )
        self.real_balance = Balance(
            str(REAL),
            AccountType.REAL,
            200,
            "USD",
            raw_is_active=explicit == "REAL",
            is_default=default == "REAL",
        )
        self.balances = [self.real_balance, self.practice_balance]
        self._outcome = outcome

    def profile_verification_diagnostics(self) -> dict[str, str]:
        return {"profile_outcome": self._outcome}


def test_observed_live_evidence_is_unverified() -> None:
    """The four live sessions' exact classification is not proof of REAL."""
    result = assess_active_account(Observer({"balance_id": REAL, "type": 0}))
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert result["profile_source"] == "CORRELATED"
    assert result["active_id_alias"] == "BALANCE_ID"
    assert result["profile_balance_type"] == "OTHER"
    assert result["active_id_relation"] == "MATCHES_REAL"
    assert result["explicit_active_attribution"] == "ABSENT"
    assert result["default_attribution"] == "ABSENT"
    assert result["endpoint_consistency"] == "CONSISTENT"
    assert result["active_wallet_authority"] == "PROFILE_REPORTED_IDENTITY_ONLY"
    assert result["ui_selected_account"] == "NOT_OBSERVED"
    assert result["classification_reason"] == "PROFILE_BALANCE_ID_NOT_AUTHORITATIVE"


def test_profile_practice_identity_without_flag_is_unverified() -> None:
    result = assess_active_account(Observer({"balance_id": PRACTICE, "type": 4}))
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert result["active_id_relation"] == "MATCHES_PRACTICE"
    assert result["active_wallet_authority"] == "PROFILE_REPORTED_IDENTITY_ONLY"
    assert result["classification_reason"] == "PROFILE_BALANCE_ID_NOT_AUTHORITATIVE"


def test_explicit_practice_flag_is_independent_authority() -> None:
    result = assess_active_account(
        Observer({"balance_id": PRACTICE, "type": 4}, explicit="PRACTICE")
    )
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_CONFIRMED"
    assert result["active_wallet_authority"] == "EXPLICIT_BALANCE_ACTIVE_FLAG"
    assert result["classification_reason"] == "EXPLICIT_PRACTICE_ACTIVE_FLAG"


def test_explicit_real_flag_is_independent_authority() -> None:
    result = assess_active_account(Observer({"balance_id": REAL, "type": 1}, explicit="REAL"))
    assert result["practice_account_active"] == "ACTIVE_REAL_OR_OTHER"
    assert result["active_wallet_authority"] == "EXPLICIT_BALANCE_ACTIVE_FLAG"
    assert result["classification_reason"] == "EXPLICIT_REAL_ACTIVE_FLAG"


def test_contradictory_profile_type_and_explicit_flag_are_unverified() -> None:
    real_type_practice_flag = assess_active_account(
        Observer({"balance_id": REAL, "type": 1}, explicit="PRACTICE")
    )
    assert real_type_practice_flag["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert real_type_practice_flag["endpoint_consistency"] == "PROFILE_VS_BALANCE_CONFLICT"
    assert real_type_practice_flag["classification_reason"] == "PROFILE_BALANCE_EVIDENCE_CONFLICT"

    practice_type_real_flag = assess_active_account(
        Observer({"balance_id": PRACTICE, "type": 4}, explicit="REAL")
    )
    assert practice_type_real_flag["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert practice_type_real_flag["endpoint_consistency"] == "PROFILE_VS_BALANCE_CONFLICT"


def test_explicit_flag_does_not_require_a_matching_profile_id() -> None:
    result = assess_active_account(Observer({"balance_id": UNKNOWN, "type": 0}, explicit="REAL"))
    assert result["active_id_relation"] == "MATCHES_NEITHER"
    assert result["active_wallet_authority"] == "EXPLICIT_BALANCE_ACTIVE_FLAG"
    assert result["practice_account_active"] == "ACTIVE_REAL_OR_OTHER"
    assert result["classification_reason"] == "EXPLICIT_REAL_ACTIVE_FLAG"


def test_default_flag_is_not_active_wallet_authority() -> None:
    result = assess_active_account(
        Observer({"balance_id": PRACTICE, "type": 4}, default="PRACTICE")
    )
    assert result["default_attribution"] == "PRACTICE"
    assert result["active_wallet_authority"] == "PROFILE_REPORTED_IDENTITY_ONLY"
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"


def test_uncorrelated_profile_with_explicit_flag_is_unverified() -> None:
    result = assess_active_account(
        Observer(
            {"balance_id": PRACTICE, "type": 4},
            explicit="PRACTICE",
            outcome="PROFILE_UNCORRELATED",
        )
    )
    assert result["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert result["classification_reason"] == "PROFILE_UNCORRELATED"


def test_observer_session_identity_does_not_claim_ui_selection() -> None:
    profile_only = assess_active_account(Observer({"balance_id": REAL, "type": 0}))
    explicit = assess_active_account(Observer({"balance_id": REAL, "type": 0}, explicit="REAL"))
    assert profile_only["ui_selected_account"] == "NOT_OBSERVED"
    assert explicit["ui_selected_account"] == "NOT_OBSERVED"
    assert profile_only["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert explicit["practice_account_active"] == "ACTIVE_REAL_OR_OTHER"


def test_diagnostics_expose_no_broker_values() -> None:
    result = assess_active_account(
        Observer({"balance_id": REAL, "type": 0, "email": "operator@example.invalid"})
    )
    serialized = json.dumps(result)
    for private in (str(PRACTICE), str(REAL), str(UNKNOWN), "123456", "USD", "operator"):
        assert private not in serialized
