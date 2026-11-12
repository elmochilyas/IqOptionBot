"""Authentication state machine and lifecycle event tracking.

Defines the discrete states, allowed transitions, and event history
for secure IQ Option authentication and session management.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final


class AuthStatus(StrEnum):
    """Discrete states in the authentication lifecycle."""

    DISCONNECTED = "DISCONNECTED"
    AUTHENTICATING = "AUTHENTICATING"
    MFA_REQUIRED = "MFA_REQUIRED"
    AUTHENTICATED = "AUTHENTICATED"
    CONNECTED_READ_ONLY = "CONNECTED_READ_ONLY"
    RECONNECTING = "RECONNECTING"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    REAUTH_REQUIRED = "REAUTH_REQUIRED"
    SESSION_INVALID = "SESSION_INVALID"
    ACCOUNT_UNVERIFIED = "ACCOUNT_UNVERIFIED"
    DISCONNECTED_BY_USER = "DISCONNECTED_BY_USER"


# Allowed state transitions: Source -> Set of valid targets
VALID_AUTH_TRANSITIONS: Final[dict[AuthStatus, frozenset[AuthStatus]]] = {
    AuthStatus.DISCONNECTED: frozenset({
        AuthStatus.AUTHENTICATING,
        AuthStatus.DISCONNECTED_BY_USER,
    }),
    AuthStatus.AUTHENTICATING: frozenset({
        AuthStatus.MFA_REQUIRED,
        AuthStatus.AUTHENTICATED,
        AuthStatus.SESSION_INVALID,
        AuthStatus.DISCONNECTED,
        AuthStatus.DISCONNECTED_BY_USER,
    }),
    AuthStatus.MFA_REQUIRED: frozenset({
        AuthStatus.AUTHENTICATING,
        AuthStatus.SESSION_INVALID,
        AuthStatus.DISCONNECTED,
        AuthStatus.DISCONNECTED_BY_USER,
    }),
    AuthStatus.AUTHENTICATED: frozenset({
        AuthStatus.CONNECTED_READ_ONLY,
        AuthStatus.AUTH_EXPIRED,
        AuthStatus.REAUTH_REQUIRED,
        AuthStatus.SESSION_INVALID,
        AuthStatus.ACCOUNT_UNVERIFIED,
        AuthStatus.DISCONNECTED,
        AuthStatus.DISCONNECTED_BY_USER,
    }),
    AuthStatus.CONNECTED_READ_ONLY: frozenset({
        AuthStatus.RECONNECTING,
        AuthStatus.AUTH_EXPIRED,
        AuthStatus.REAUTH_REQUIRED,
        AuthStatus.SESSION_INVALID,
        AuthStatus.ACCOUNT_UNVERIFIED,
        AuthStatus.DISCONNECTED,
        AuthStatus.DISCONNECTED_BY_USER,
    }),
    AuthStatus.RECONNECTING: frozenset({
        AuthStatus.CONNECTED_READ_ONLY,
        AuthStatus.AUTH_EXPIRED,
        AuthStatus.REAUTH_REQUIRED,
        AuthStatus.SESSION_INVALID,
        AuthStatus.DISCONNECTED,
        AuthStatus.DISCONNECTED_BY_USER,
    }),
    AuthStatus.AUTH_EXPIRED: frozenset({
        AuthStatus.REAUTH_REQUIRED,
        AuthStatus.AUTHENTICATING,
        AuthStatus.DISCONNECTED,
        AuthStatus.DISCONNECTED_BY_USER,
    }),
    AuthStatus.REAUTH_REQUIRED: frozenset({
        AuthStatus.AUTHENTICATING,
        AuthStatus.DISCONNECTED,
        AuthStatus.DISCONNECTED_BY_USER,
    }),
    AuthStatus.SESSION_INVALID: frozenset({
        AuthStatus.AUTHENTICATING,
        AuthStatus.DISCONNECTED,
        AuthStatus.DISCONNECTED_BY_USER,
    }),
    AuthStatus.ACCOUNT_UNVERIFIED: frozenset({
        AuthStatus.CONNECTED_READ_ONLY,
        AuthStatus.AUTHENTICATING,
        AuthStatus.DISCONNECTED,
        AuthStatus.DISCONNECTED_BY_USER,
    }),
    AuthStatus.DISCONNECTED_BY_USER: frozenset({
        AuthStatus.AUTHENTICATING,
        AuthStatus.DISCONNECTED,
    }),
}


class AuthStateTransitionError(RuntimeError):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, from_state: AuthStatus, to_state: AuthStatus, reason: str = "") -> None:
        msg = f"Invalid auth transition from {from_state.value} to {to_state.value}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason


@dataclass(frozen=True, slots=True)
class AuthEvent:
    """Immutable audit record of an authentication state transition."""

    from_state: AuthStatus
    to_state: AuthStatus
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    details: str = ""

    def __repr__(self) -> str:
        return (
            f"AuthEvent({self.from_state.value} -> {self.to_state.value}, reason={self.reason!r})"
        )


class AuthStateMachine:
    """Strict state machine enforcing legal authentication transitions."""

    def __init__(self, initial_state: AuthStatus = AuthStatus.DISCONNECTED) -> None:
        self._current_state = initial_state
        self._history: list[AuthEvent] = []

    @property
    def current_state(self) -> AuthStatus:
        """Current authentication state."""
        return self._current_state

    @property
    def history(self) -> tuple[AuthEvent, ...]:
        """Immutable copy of transition history."""
        return tuple(self._history)

    def can_transition(self, target_state: AuthStatus) -> bool:
        """Check whether transition to target state is legally permitted."""
        valid_targets = VALID_AUTH_TRANSITIONS.get(self._current_state, frozenset())
        return target_state in valid_targets

    def transition_to(
        self,
        target_state: AuthStatus,
        reason: str = "",
        details: str = "",
    ) -> AuthEvent:
        """Transition to target state with validation and audit logging.

        Raises:
            AuthStateTransitionError: If the transition is not permitted.
        """
        if target_state == self._current_state:
            # Self-transition is a no-op event if already in that state
            return AuthEvent(
                from_state=self._current_state,
                to_state=target_state,
                reason="no-op",
                details=details,
            )

        if not self.can_transition(target_state):
            raise AuthStateTransitionError(
                from_state=self._current_state,
                to_state=target_state,
                reason=reason,
            )

        event = AuthEvent(
            from_state=self._current_state,
            to_state=target_state,
            reason=reason,
            details=details,
        )
        self._history.append(event)
        self._current_state = target_state
        return event

    def reset(self, reason: str = "reset") -> AuthEvent:
        """Reset state machine back to DISCONNECTED."""
        return self.transition_to(AuthStatus.DISCONNECTED, reason=reason)
