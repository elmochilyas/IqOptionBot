"""Dedicated authentication subsystem for IQ Option trading bot.

Provides secure owner-controlled authentication, Windows Credential Manager
integration, MFA challenge handling, in-memory session tracking, and offline
development isolation.
"""

from __future__ import annotations

from iqoption_research.auth.browser_research import (
    BrowserLoginFeasibilityReport,
    get_browser_login_feasibility,
)
from iqoption_research.auth.config import RuntimeSecurityConfig
from iqoption_research.auth.manager import AuthenticationManager
from iqoption_research.auth.mfa import (
    MFACallback,
    MFAChallenge,
    MFAMethod,
    default_interactive_mfa_prompt,
)
from iqoption_research.auth.provider import (
    AuthProvider,
    AuthResult,
    IQOptionAuthProvider,
    MockAuthProvider,
)
from iqoption_research.auth.session import AuthSession
from iqoption_research.auth.state import (
    VALID_AUTH_TRANSITIONS,
    AuthEvent,
    AuthStateMachine,
    AuthStateTransitionError,
    AuthStatus,
)
from iqoption_research.auth.vault import (
    InsecureKeyringBackendError,
    OptInRequiredError,
    VaultError,
    WindowsCredentialVault,
)

__all__ = [
    "VALID_AUTH_TRANSITIONS",
    "AuthEvent",
    "AuthProvider",
    "AuthResult",
    "AuthSession",
    "AuthStateMachine",
    "AuthStateTransitionError",
    "AuthStatus",
    "AuthenticationManager",
    "BrowserLoginFeasibilityReport",
    "IQOptionAuthProvider",
    "InsecureKeyringBackendError",
    "MFACallback",
    "MFAChallenge",
    "MFAMethod",
    "MockAuthProvider",
    "OptInRequiredError",
    "RuntimeSecurityConfig",
    "VaultError",
    "WindowsCredentialVault",
    "default_interactive_mfa_prompt",
    "get_browser_login_feasibility",
]
