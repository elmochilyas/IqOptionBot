"""Account and balance models with strict typing.

These models represent the data returned by IQ Option's profile and
balance endpoints. The practice account verification is explicit —
we NEVER assume a balance is practice without verification.
"""

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, IntEnum
from typing import Any

from iqoption_research.security.credentials import redact_email

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=UTC)


class PracticeStatus(Enum):
    """Outcome of practice-account verification.

    The critical distinction: a PRACTICE balance existing is NOT the same
    as the active trading account being PRACTICE.
    """

    VERIFIED_ACTIVE_PRACTICE = "verified_active_practice"
    VERIFIED_ACTIVE_REAL = "verified_active_real"
    CONFLICTING_ACCOUNT_INDICATORS = "conflicting_account_indicators"
    MISSING_ACCOUNT_DATA = "missing_account_data"
    ACTIVE_REAL = "active_real"
    ACTIVE_UNKNOWN = "active_unknown"
    NO_PRACTICE_BALANCE = "no_practice_balance"
    NO_BALANCES = "no_balances"
    ACCOUNT_SELECTION_UNVERIFIED = "account_selection_unverified"

    # Backward compatibility alias
    PRACTICE_PRESENT_ACTIVE_UNVERIFIED = "practice_present_active_unverified"


@dataclass(frozen=True, slots=True)
class PracticeVerification:
    """Explicit practice-account verification result.

    ``is_verified`` is True ONLY when the transport observed the server's
    active balance id AND that balance is type PRACTICE.
    """

    status: PracticeStatus
    message: str
    active_balance_id: str | None = None
    practice_balance_ids: tuple[str, ...] = ()
    decision_path: tuple[str, ...] = ()

    @property
    def is_verified(self) -> bool:
        return self.status is PracticeStatus.VERIFIED_ACTIVE_PRACTICE

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "is_verified": self.is_verified,
            "message": self.message,
            "active_balance_id": self.active_balance_id,
            "practice_balance_ids": list(self.practice_balance_ids),
            "decision_path": list(self.decision_path),
        }


class AccountType(IntEnum):
    """IQ Option account/balance types.

    Values correspond to IQ Option's internal `type` field.
    Only `PRACTICE` (4) is safe for research. REAL accounts
    require explicit, separate authorization.
    """

    REAL = 1
    PRACTICE = 4
    TOURNAMENT = 5  # Not used in research

    @property
    def is_practice(self) -> bool:
        return self == AccountType.PRACTICE

    @property
    def is_real(self) -> bool:
        return self == AccountType.REAL

    def __str__(self) -> str:
        return self.name.lower()


def wallet_identifier(value: object) -> str:
    """Compare decimal aliases consistently, preserving opaque nonnumeric IDs."""
    if value is None:
        return ""
    if type(value) not in (int, str):
        raise ValueError("Invalid wallet identifier type")
    identifier = str(value).strip()
    if identifier.isascii() and identifier.isdecimal():
        return identifier.lstrip("0")
    return identifier


@dataclass(frozen=True, slots=True)
class Balance:
    """Account balance with explicit type identification.

    IMPORTANT: The existence of a PRACTICE balance does NOT prove
    the active trading account is in PRACTICE mode. IQ Option allows
    multiple balances simultaneously. The `is_active` field reflects
    the server's `balance_id` matching, but this can change.
    """

    id: str
    type: AccountType
    amount: float
    currency: str
    is_active: bool = False
    raw_is_active: bool | None = None
    is_default: bool | None = None

    @property
    def is_practice(self) -> bool:
        return self.type.is_practice

    @property
    def is_real(self) -> bool:
        return self.type.is_real

    @classmethod
    def from_raw(
        cls,
        data: dict[str, Any],
        *,
        active_balance_id: str | None = None,
    ) -> "Balance":
        """Create a Balance from raw server dictionary.

        Explicitly validates account type mapping:
            1 -> AccountType.REAL (real trading account; is_practice is False)
            4 -> AccountType.PRACTICE (demo practice account; is_practice is True)
            5 -> AccountType.TOURNAMENT (tournament account; is_practice is False)

        Accepts only exact integer types (1, 4, 5) or their exact decimal string
        representations ("1", "4", "5"). Rejects booleans, floats, fractional
        coercions, malformed strings, and unknown values.
        Sensitive/raw input values are NEVER included in validation error messages.
        """
        raw_type = data.get("type") if data.get("type") is not None else data.get("balance_type")
        if raw_type is None:
            raise ValueError("Balance payload missing required 'type' field")
        if isinstance(raw_type, bool):
            raise ValueError("Invalid account type: boolean not permitted")

        parsed_type: int
        if isinstance(raw_type, int):
            parsed_type = raw_type
        elif isinstance(raw_type, str) and raw_type.strip() in {"1", "4", "5"}:
            parsed_type = int(raw_type.strip())
        else:
            raise ValueError("Invalid or unknown account type")

        try:
            account_type = AccountType(parsed_type)
        except ValueError as exc:
            raise ValueError("Invalid or unknown account type") from exc

        raw_id = data.get("id") if data.get("id") is not None else data.get("balance_id")
        balance_id = wallet_identifier(raw_id)
        if (
            data.get("id") is not None
            and data.get("balance_id") is not None
            and wallet_identifier(data["balance_id"]) != balance_id
        ):
            raise ValueError("Conflicting wallet identifiers")
        if data.get("balance_type") is not None and str(data["balance_type"]).strip() != str(
            parsed_type
        ):
            raise ValueError("Conflicting wallet types")
        if not balance_id:
            raise ValueError("Balance payload missing or empty 'id' field")

        raw_amount = data.get("amount")
        if raw_amount is None:
            raw_amount = data.get("balance")
        if raw_amount is None:
            raw_amount = 0.0
        if isinstance(raw_amount, bool):
            raise ValueError("Invalid balance amount: boolean not permitted")

        try:
            amount = float(raw_amount)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid balance amount") from exc

        if not math.isfinite(amount):
            raise ValueError("Invalid balance amount: non-finite value")

        currency = str(data.get("currency") or "USD").strip()
        is_active = active_balance_id is not None and wallet_identifier(active_balance_id) == balance_id

        def flag(primary: str, alias: str) -> bool | None:
            value = data.get(primary, data.get(alias))
            return value if isinstance(value, bool) else None

        raw_is_active = flag("is_active", "active")
        is_default = flag("is_default", "default")

        return cls(
            id=balance_id,
            type=account_type,
            amount=amount,
            currency=currency,
            is_active=is_active,
            raw_is_active=raw_is_active,
            is_default=is_default,
        )

    def __repr__(self) -> str:
        active_marker = " (ACTIVE)" if self.is_active else ""
        return (
            f"Balance(id={self.id!r}, type={self.type.name}, "
            f"amount={self.amount} {self.currency}{active_marker})"
        )


@dataclass(frozen=True, slots=True)
class Profile:
    """User profile information from IQ Option."""

    user_id: int
    email: str
    first_name: str
    last_name: str
    created_at: datetime
    is_verified: bool
    country: str
    currency: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (
            f"Profile(user_id={self.user_id}, email={redact_email(self.email)!r}, "
            f"first_name={self.first_name!r}, last_name={self.last_name!r}, "
            f"country={self.country!r}, is_verified={self.is_verified})"
        )

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> "Profile":
        """Create Profile from raw API response."""
        # IQ Option returns timestamps in milliseconds
        created_ts = data.get("created", 0)
        if created_ts > 1e12:  # Milliseconds
            created_ts /= 1000

        created_at = datetime.fromtimestamp(created_ts, tz=UTC) if created_ts else _EPOCH_UTC

        user_id = data.get("user_id") or data.get("id") or 0
        first_name = data.get("first_name") or data.get("name") or ""
        last_name = data.get("last_name") or ""
        country = str(data.get("country") or data.get("country_id") or "")
        currency = str(data.get("currency") or "USD")

        parsed_user_id = (
            int(user_id) if isinstance(user_id, (int, str)) and str(user_id).isdigit() else 0
        )

        return cls(
            user_id=parsed_user_id,
            email=str(data.get("email", "")),
            first_name=str(first_name),
            last_name=str(last_name),
            created_at=created_at,
            is_verified=bool(data.get("is_verified")),
            country=country,
            currency=currency,
            raw=data,
        )


@dataclass(frozen=True, slots=True)
class WalletContext:
    """Operator-selected inspection target in the current connection's inventory."""

    wallet_type: AccountType
    balance_id: str = field(repr=False)

    @classmethod
    def resolve(cls, requested: AccountType, balances: tuple[Balance, ...]) -> "WalletContext":
        if requested not in (AccountType.PRACTICE, AccountType.REAL):
            raise ValueError("Select PRACTICE or REAL for inspection.")
        if len({b.id for b in balances}) != len(balances):
            raise ValueError("Duplicate wallet identifiers in the broker inventory.")
        matches = [b for b in balances if b.type is requested]
        if len(matches) != 1:
            raise ValueError("Requested type has no unique wallet in the current inventory.")
        return cls(requested, matches[0].id)


def wallet_type_conflict(
    inventory: tuple[Balance, ...], profile: dict[str, Any], changed: tuple[str, str] | None
) -> bool:
    """Compare only claims that bind a type to the same wallet identifier."""
    ledger = {b.id: str(b.type.value) for b in inventory}
    claims = [changed] if changed else []
    profile_type = profile.get("balance_type")
    if profile_type is not None:
        for key in ("balance_id", "balanceId", "active_balance_id"):
            if profile.get(key) is not None:
                claims.append((str(profile[key]).strip(), str(profile_type).strip()))
    nested = profile.get("balances", [])
    if isinstance(nested, dict):
        nested = nested.get("balances", [])
    rows = list(nested) if isinstance(nested, list) else []
    rows.append(profile.get("current_balance"))
    for row in rows:
        if isinstance(row, dict):
            identifier = row.get("id", row.get("balance_id"))
            kind = row.get("type", row.get("balance_type"))
            if identifier is not None and kind is not None:
                claims.append((str(identifier).strip(), str(kind).strip()))
    return any(ledger.get(wallet_identifier(key), kind) != kind for key, kind in claims)


class WalletSource(Enum):
    PROFILE = "profile"
    LIST_ACTIVE = "get-balances.active"
    LIST_DEFAULT = "get-balances.default"
    PUSH_ACTIVE = "balances.active"
    PUSH_DEFAULT = "balances.default"
    CHANGE = "balance-changed"
    # Only the offline connector owns and knows its effective session wallet.
    SIMULATED_SESSION = "simulated_session"


@dataclass(frozen=True, slots=True)
class WalletIndicator:
    source: WalletSource
    balance_id: str = field(repr=False)


def wallet_decision(
    balances: tuple[Balance, ...],
    indicators: tuple[WalletIndicator, ...],
    *,
    complete: bool,
    conflicting_types: bool = False,
) -> PracticeVerification:
    """Compare observations from one connection. No live source proves session scope."""
    ids = {item.balance_id for item in indicators}
    ledger = {balance.id: balance.type for balance in balances}
    session = next((i for i in indicators if i.source is WalletSource.SIMULATED_SESSION), None)
    if not complete or not balances:
        status = PracticeStatus.MISSING_ACCOUNT_DATA
    elif conflicting_types or len(ids) > 1 or len(ledger) != len(balances):
        status = PracticeStatus.CONFLICTING_ACCOUNT_INDICATORS
    elif session and ledger.get(session.balance_id) == AccountType.PRACTICE:
        status = PracticeStatus.VERIFIED_ACTIVE_PRACTICE
    elif session and ledger.get(session.balance_id) == AccountType.REAL:
        status = PracticeStatus.VERIFIED_ACTIVE_REAL
    else:
        status = PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED

    return PracticeVerification(
        status=status,
        message={
            PracticeStatus.MISSING_ACCOUNT_DATA: "Account inventory or profile is missing.",
            PracticeStatus.CONFLICTING_ACCOUNT_INDICATORS: (
                "Wallet indicators disagree or inventory has duplicate IDs."
            ),
            PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED: (
                "Wallet selection UNVERIFIED. No supported live message proves session scope."
            ),
            PracticeStatus.VERIFIED_ACTIVE_PRACTICE: "Session selection verified as PRACTICE.",
            PracticeStatus.VERIFIED_ACTIVE_REAL: "Session selection verified as REAL.",
        }[status],
        active_balance_id=session.balance_id if session else None,
        practice_balance_ids=tuple(b.id for b in balances if b.is_practice),
        decision_path=(status.name,),
    )


def balance_wallet_indicators(
    balances: tuple[Balance, ...], *, correlated: bool
) -> tuple[WalletIndicator, ...]:
    active = WalletSource.LIST_ACTIVE if correlated else WalletSource.PUSH_ACTIVE
    default = WalletSource.LIST_DEFAULT if correlated else WalletSource.PUSH_DEFAULT
    indicators = []
    for balance in balances:
        if balance.raw_is_active is True:
            indicators.append(WalletIndicator(active, balance.id))
        if balance.is_default is True:
            indicators.append(WalletIndicator(default, balance.id))
    return tuple(indicators)


def profile_wallet_indicators(raw: dict[str, Any]) -> tuple[WalletIndicator, ...]:
    current = raw.get("current_balance")
    values = [raw.get(k) for k in ("balance_id", "balanceId", "active_balance_id")]
    if isinstance(current, dict):
        values.append(current.get("id"))
    return tuple(
        WalletIndicator(WalletSource.PROFILE, wallet_identifier(value))
        for value in values
        if type(value) in (str, int) and str(value).strip()
    )


@dataclass(frozen=True, slots=True)
class AccountStatus:
    """Complete account status including profile and all balances.

    The legacy active_balance field matches the observed wallet ID. Its
    session scope is unknown. Only practice_verification authorizes research.
    """

    profile: Profile
    balances: tuple[Balance, ...]
    active_balance: Balance | None
    practice_balance: Balance | None
    real_balance: Balance | None
    indicators: tuple[WalletIndicator, ...] = ()
    changed_wallet_type: tuple[str, str] | None = field(default=None, repr=False)

    @classmethod
    def from_raw(
        cls,
        profile_data: dict[str, Any],
        balances_data: list[dict[str, Any]],
        *,
        active_balance_id: str | None = None,
    ) -> "AccountStatus":
        """Create AccountStatus from raw API responses.

        ``active_balance_id`` is the server-reported active balance id, if
        and only if the transport actually observed one. When it is None,
        NO balance is marked active. We deliberately do not guess that a
        practice balance is active merely because it exists.
        """
        profile = Profile.from_raw(profile_data)

        balances = []
        active_balance = None
        practice_balance = None
        real_balance = None

        for b in balances_data:
            if not isinstance(b, dict):
                continue
            try:
                balance = Balance.from_raw(b, active_balance_id=active_balance_id)
            except (ValueError, TypeError, KeyError):
                # Skip malformed balance entries
                continue

            balances.append(balance)

            if balance.is_active:
                active_balance = balance
            if balance.is_practice and practice_balance is None:
                practice_balance = balance
            if balance.is_real and real_balance is None:
                real_balance = balance

        if not balances:
            # A typed profile wallet supplies inventory only.
            prof_bal_id = (
                profile_data.get("balance_id")
                or profile_data.get("balanceId")
                or profile_data.get("active_balance_id")
            )
            prof_bal_type = (
                profile_data.get("balance_type")
                if profile_data.get("balance_type") is not None
                else profile_data.get("type")
            )
            if prof_bal_id is not None and prof_bal_type is not None:
                try:
                    single_bal = Balance.from_raw(
                        {
                            "id": prof_bal_id,
                            "type": prof_bal_type,
                            "amount": profile_data.get("balance", 0.0),
                            "currency": profile_data.get("currency", "USD"),
                        },
                        active_balance_id=active_balance_id,
                    )
                    balances.append(single_bal)
                    if single_bal.is_active:
                        active_balance = single_bal
                    if single_bal.is_practice and practice_balance is None:
                        practice_balance = single_bal
                    if single_bal.is_real and real_balance is None:
                        real_balance = single_bal
                except (ValueError, TypeError, KeyError):
                    pass

        return cls(
            profile=profile,
            balances=tuple(balances),
            active_balance=active_balance,
            practice_balance=practice_balance,
            real_balance=real_balance,
            indicators=profile_wallet_indicators(profile_data)
            + balance_wallet_indicators(tuple(balances), correlated=False),
        )

    @property
    def has_practice(self) -> bool:
        return self.practice_balance is not None

    @property
    def has_real(self) -> bool:
        return self.real_balance is not None

    @property
    def practice_verified(self) -> bool:
        return self.practice_verification().is_verified

    def practice_verification(self) -> PracticeVerification:
        return wallet_decision(
            self.balances,
            self.indicators,
            complete=bool(self.profile.raw or self.profile.user_id),
            conflicting_types=wallet_type_conflict(
                self.balances, self.profile.raw, self.changed_wallet_type
            ),
        )

    def __repr__(self) -> str:
        active = self.active_balance
        return (
            f"AccountStatus(user={redact_email(self.profile.email)}, "
            f"active={active.type.name if active else 'NONE'}, "
            f"practice_verified={self.practice_verified})"
        )
