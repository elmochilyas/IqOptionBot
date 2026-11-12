"""Security and runtime configuration for the authentication subsystem.

Provides policy parameters governing credential access, session lifetimes,
reconnection backoff, and environment isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RuntimeSecurityConfig:
    """Security policy configuration for authentication and session management."""

    # Operational Mode: Set to False only in verified owner production environments
    is_development: bool = True

    # Windows Credential Manager target service name
    vault_service_name: str = "IqOptionBot"

    # Whether saved credentials in Windows Credential Manager may be used
    allow_saved_credentials: bool = True

    # Maximum age of an in-memory authenticated session before requiring re-auth
    session_ttl_seconds: float = 86400.0  # 24 hours

    # Maximum transport reconnect attempts before transitioning to REAUTH_REQUIRED
    max_reconnect_attempts: int = 5

    # Exponential backoff parameters for transport reconnects
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0

    # Whether session requires explicit PRACTICE verification before read-only operations
    require_practice_verification: bool = True

    # Automatic reauthentication policy:
    # If True, can re-read saved credentials on session expiry IF 2FA is not required.
    # Default is False: requires explicit user confirmation.
    auto_reauth_with_saved_creds: bool = False

    # Allowed login endpoints
    allowed_auth_hosts: tuple[str, ...] = field(default_factory=lambda: ("auth.iqoption.com",))
