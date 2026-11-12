"""Two-factor authentication (2FA / MFA) challenge models and interactive handlers.

Ensures that 2FA challenges are never bypassed, automated with recovery codes,
or silenced, but instead routed to the owner via secure interactive callbacks.
"""

from __future__ import annotations

import getpass
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final


class MFAMethod(StrEnum):
    """Supported 2FA delivery methods reported by the broker."""

    SMS = "sms"
    EMAIL = "email"
    AUTHENTICATOR = "authenticator"
    UNKNOWN = "unknown"


DEFAULT_MFA_TTL_SECONDS: Final[float] = 120.0


@dataclass(frozen=True, slots=True)
class MFAChallenge:
    """Represents a broker-initiated two-factor authentication challenge."""

    token: str
    method: MFAMethod = MFAMethod.AUTHENTICATOR
    message: str = "Please enter your 2FA verification code."
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = DEFAULT_MFA_TTL_SECONDS

    def __repr__(self) -> str:
        return (
            f"MFAChallenge(method={self.method.value}, "
            f"token=***REDACTED***, ttl={self.ttl_seconds}s)"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def is_expired(self) -> bool:
        """Check whether the MFA challenge window has lapsed."""
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def remaining_seconds(self) -> float:
        """Remaining seconds before the challenge expires."""
        elapsed = time.time() - self.created_at
        return max(0.0, self.ttl_seconds - elapsed)


# Type alias for MFA interactive callback
MFACallback = Callable[[MFAChallenge], str]


def default_interactive_mfa_prompt(challenge: MFAChallenge) -> str:
    """Default secure terminal prompt for two-factor authentication codes.

    Uses getpass or input to capture the verification code interactively without echoing.
    """
    if challenge.is_expired:
        raise TimeoutError("MFA challenge has expired. Please reinitiate authentication.")

    print(f"\n[2FA Required] Method: {challenge.method.value.upper()}")
    if challenge.message:
        print(f"Message: {challenge.message}")

    try:
        # 2FA code is non-secret in transit (one-time), but we avoid logging it
        code = getpass.getpass("Enter 2FA verification code: ").strip()
    except EOFError as err:
        raise RuntimeError("Non-interactive session cannot complete 2FA challenge.") from err

    if not code:
        raise ValueError("Verification code cannot be empty.")
    return code
