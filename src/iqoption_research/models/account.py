"""Account and balance models with strict typing.

These models represent the data returned by IQ Option's profile and
balance endpoints. The practice account verification is explicit —
we NEVER assume a balance is practice without verification.
"""

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
    PRACTICE_PRESENT_ACTIVE_UNVERIFIED = "practice_present_active_unverified"
    NO_PRACTICE_BALANCE = "no_practice_balance"
    NO_BALANCES = "no_balances"


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

    @property
    def is_practice(self) -> bool:
        return self.type.is_practice

    @property
    def is_real(self) -> bool:
        return self.type.is_real

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

        created_at = (
            datetime.fromtimestamp(created_ts, tz=UTC)
            if created_ts
            else _EPOCH_UTC
        )

        user_id = data.get("user_id") or data.get("id") or 0
        first_name = data.get("first_name") or data.get("name") or ""
        last_name = data.get("last_name") or ""
        country = str(data.get("country") or data.get("country_id") or "")
        currency = str(data.get("currency") or "USD")

        parsed_user_id = (
            int(user_id)
            if isinstance(user_id, (int, str)) and str(user_id).isdigit()
            else 0
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
class AccountStatus:
    """Complete account status including profile and all balances.

    The `active_balance` is the one matching the server's current
    `balance_id`. This can change if the user switches modes in
    another session. We report it but do not rely on it.
    """
    profile: Profile
    balances: tuple[Balance, ...]
    active_balance: Balance | None
    practice_balance: Balance | None
    real_balance: Balance | None

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
            try:
                balance = Balance(
                    id=str(b.get("id", "")),
                    type=AccountType(int(b.get("type", 0))),
                    amount=float(b.get("amount", 0)),
                    currency=b.get("currency", "USD"),
                    is_active=(
                        active_balance_id is not None
                        and str(b.get("id", "")) == str(active_balance_id)
                    ),
                )
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

        return cls(
            profile=profile,
            balances=tuple(balances),
            active_balance=active_balance,
            practice_balance=practice_balance,
            real_balance=real_balance,
        )

    @property
    def has_practice(self) -> bool:
        return self.practice_balance is not None

    @property
    def has_real(self) -> bool:
        return self.real_balance is not None

    @property
    def practice_verified(self) -> bool:
        """True only if active balance IS the practice balance."""
        return self.active_balance is not None and self.active_balance.is_practice

    def practice_verification(self) -> PracticeVerification:
        """Classify practice status without ever assuming/ switching."""
        practice_ids = tuple(b.id for b in self.balances if b.is_practice)
        if not self.balances:
            return PracticeVerification(
                status=PracticeStatus.NO_BALANCES,
                message="No balances were returned by the server.",
            )
        if not practice_ids:
            return PracticeVerification(
                status=PracticeStatus.NO_PRACTICE_BALANCE,
                message="Account has no PRACTICE balance.",
                active_balance_id=(
                    self.active_balance.id if self.active_balance else None
                ),
            )
        if self.practice_verified:
            active = self.active_balance
            if active is None:  # defensive; practice_verified implies not None
                raise RuntimeError("practice_verified without an active balance")
            return PracticeVerification(
                status=PracticeStatus.VERIFIED_ACTIVE_PRACTICE,
                message=(
                    "Server-reported active balance is PRACTICE. Read-only "
                    "research may proceed."
                ),
                active_balance_id=active.id,
                practice_balance_ids=practice_ids,
            )
        return PracticeVerification(
            status=PracticeStatus.PRACTICE_PRESENT_ACTIVE_UNVERIFIED,
            message=(
                "A PRACTICE balance exists, but the active trading account "
                "could not be established as PRACTICE. Treat as UNVERIFIED; "
                "the active account may be REAL."
            ),
            active_balance_id=(self.active_balance.id if self.active_balance else None),
            practice_balance_ids=practice_ids,
        )

    def __repr__(self) -> str:
        active = self.active_balance
        return (
            f"AccountStatus(user={redact_email(self.profile.email)}, "
            f"active={active.type.name if active else 'NONE'}, "
            f"practice_verified={self.practice_verified})"
        )
