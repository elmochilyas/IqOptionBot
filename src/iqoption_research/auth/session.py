"""In-memory authenticated session container.

Holds the temporary SSID session cookie, authentication timestamp,
and account verification state in memory only.

Guarantees:
- In-Memory Only: Never serialized to JSON, SQLite, .env, or disk files.
- Redacted Representation: `__repr__` and `__str__` strictly mask the SSID token.
- TTL Expiration Tracking: Enforces session age limits.
- Memory Minimization: Explicit `invalidate()` method erases memory references.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final

from iqoption_research.security.credentials import redact_email

DEFAULT_SESSION_TTL_SECONDS: Final[float] = 86400.0  # 24 hours


@dataclass(slots=True)
class AuthSession:
    """Active in-memory authenticated session."""

    ssid: str
    email: str
    authenticated_at: float = field(default_factory=time.time)
    ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS
    last_activity_at: float = field(default_factory=time.time)
    reconnect_count: int = 0
    is_practice_verified: bool = False
    server_account_type: str = "UNKNOWN"
    active_balance_id: str | None = None
    _is_valid: bool = True

    def __repr__(self) -> str:
        status_str = "VALID" if self.is_active else "INVALID"
        return (
            f"AuthSession(account={redact_email(self.email)}, "
            f"ssid=***REDACTED***, status={status_str}, "
            f"age={self.age_seconds():.1f}s, practice={self.is_practice_verified})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def is_active(self) -> bool:
        """Whether the session is internally valid and not expired."""
        return self._is_valid and not self.is_expired()

    def is_expired(self, current_time: float | None = None) -> bool:
        """Check whether the session has exceeded its configured time-to-live."""
        now = current_time if current_time is not None else time.time()
        return (now - self.authenticated_at) > self.ttl_seconds

    def age_seconds(self, current_time: float | None = None) -> float:
        """Elapsed seconds since initial authentication."""
        now = current_time if current_time is not None else time.time()
        return max(0.0, now - self.authenticated_at)

    def touch(self, current_time: float | None = None) -> None:
        """Update last activity timestamp."""
        self.last_activity_at = current_time if current_time is not None else time.time()

    def increment_reconnect(self) -> int:
        """Increment and return the reconnect counter."""
        self.reconnect_count += 1
        self.touch()
        return self.reconnect_count

    def invalidate(self) -> None:
        """Erase session validity and clear references."""
        self._is_valid = False
        self.ssid = ""
