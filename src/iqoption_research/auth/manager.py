"""Authentication manager coordinating credentials, vault, providers, and sessions.

Acts as the central orchestrator for secure owner-controlled authentication.
Enforces:
- Separation from trading and paper execution.
- In-memory session tracking with immediate password disposal.
- Secure fallback and opt-in Windows Credential Vault interaction.
- Explicit state machine transitions and audit logging.
- Read-only broker verification.
"""

from __future__ import annotations

import getpass
import logging
import time
from collections.abc import Callable
from typing import Any

from iqoption_research.auth.config import RuntimeSecurityConfig
from iqoption_research.auth.mfa import (
    MFACallback,
    MFAChallenge,
    default_interactive_mfa_prompt,
)
from iqoption_research.auth.provider import (
    AuthProvider,
    AuthResult,
    MockAuthProvider,
)
from iqoption_research.auth.session import AuthSession
from iqoption_research.auth.state import (
    AuthStateMachine,
    AuthStateTransitionError,
    AuthStatus,
)
from iqoption_research.auth.vault import WindowsCredentialVault
from iqoption_research.connectors.iqoption.exceptions import ConnectionError, LoginError
from iqoption_research.execution.account_diagnostics import (
    BrokerAccountDiagnostic,
    BrokerLedgerStatus,
)
from iqoption_research.security.credentials import Credentials, redact_email

logger = logging.getLogger(__name__)


class AuthenticationManager:
    """Orchestrates authentication lifecycle, credential storage, and session state."""

    def __init__(
        self,
        config: RuntimeSecurityConfig | None = None,
        *,
        provider: AuthProvider | None = None,
        vault: WindowsCredentialVault | None = None,
    ) -> None:
        self.config = config or RuntimeSecurityConfig()
        self.state_machine = AuthStateMachine(initial_state=AuthStatus.DISCONNECTED)
        self.vault = vault or WindowsCredentialVault(service_name=self.config.vault_service_name)
        self.provider = provider or MockAuthProvider()
        self._session: AuthSession | None = None
        self._last_error: str | None = None
        self._account_diagnostic = BrokerAccountDiagnostic()

    @property
    def current_state(self) -> AuthStatus:
        """Current authentication state."""
        return self.state_machine.current_state

    @property
    def session(self) -> AuthSession | None:
        """Active session if authenticated and valid; None otherwise."""
        if self._session is not None and self._session.is_active:
            return self._session
        return None

    @property
    def last_error(self) -> str | None:
        """Most recent authentication or verification error message."""
        return self._last_error

    def is_authenticated(self) -> bool:
        """Check whether an active, valid authenticated session exists."""
        return self._session is not None and self._session.is_active

    async def login_interactive(
        self,
        *,
        email: str | None = None,
        password: str | None = None,
        prompt_fn: Callable[[], tuple[str, str]] | None = None,
        mfa_callback: MFACallback | None = None,
    ) -> bool:
        """Authenticate via local interactive prompt.

        Password is never stored in files, logs, or persistent configuration,
        and is discarded from memory as soon as authentication completes.
        """
        self._last_error = None

        # 1. Obtain credentials
        if email is None or password is None:
            if prompt_fn is not None:
                email, password = prompt_fn()
            else:
                email, password = self._prompt_credentials_from_terminal()

        if not email or not password:
            self._last_error = "Email and password cannot be empty."
            return False

        # 2. Enter AUTHENTICATING state
        try:
            self.state_machine.transition_to(
                AuthStatus.AUTHENTICATING,
                reason="user_interactive_login",
                details=f"account={redact_email(email)}",
            )
        except (AuthStateTransitionError, ValueError) as exc:
            self._last_error = f"Cannot transition to authenticating: {exc}"
            return False

        # 3. Create short-lived Credentials container and dispatch login
        creds = Credentials(email=email, password=password)
        # Immediately forget local password string reference
        del password

        try:
            result = await self.provider.login(creds)
        except (ConnectionError, LoginError, TimeoutError, OSError) as exc:
            logger.error("Authentication provider error: %s", exc)
            self._last_error = f"Connection error: {exc}"
            self.state_machine.transition_to(
                AuthStatus.DISCONNECTED,
                reason="login_exception",
                details=str(exc),
            )
            return False
        finally:
            # Discard credentials container reference
            del creds

        # 4. Handle 2FA challenge if broker requested it
        if result.mfa_challenge is not None:
            result = await self._resolve_mfa_challenge(result.mfa_challenge, mfa_callback)
            if not result.success:
                return False

        # 5. Process result
        return await self._process_auth_result(result, email)

    async def login_saved(self, *, mfa_callback: MFACallback | None = None) -> bool:
        """Authenticate using credentials stored in Windows Credential Manager."""
        self._last_error = None

        if not self.vault.is_secure_backend():
            self._last_error = (
                "Windows Credential Vault is not available or insecure backend detected."
            )
            return False

        if not self.vault.has_saved_credentials():
            self._last_error = (
                "No saved credentials found in Windows Credential Vault. Use interactive login."
            )
            return False

        creds = self.vault.load_saved_credentials()
        if creds is None:
            self._last_error = "Failed to load credentials from Windows Credential Vault."
            return False

        email = creds.email
        try:
            self.state_machine.transition_to(
                AuthStatus.AUTHENTICATING,
                reason="saved_credentials_login",
                details=f"account={redact_email(email)}",
            )
        except (AuthStateTransitionError, ValueError) as exc:
            self._last_error = f"Cannot transition to authenticating: {exc}"
            return False

        try:
            result = await self.provider.login(creds)
        except (ConnectionError, LoginError, TimeoutError, OSError) as exc:
            logger.error("Provider error during saved-credential login: %s", exc)
            self._last_error = f"Connection error: {exc}"
            self.state_machine.transition_to(
                AuthStatus.DISCONNECTED,
                reason="saved_login_exception",
            )
            return False
        finally:
            del creds

        if result.mfa_challenge is not None:
            result = await self._resolve_mfa_challenge(result.mfa_challenge, mfa_callback)
            if not result.success:
                return False

        return await self._process_auth_result(result, email)

    async def _resolve_mfa_challenge(
        self,
        challenge: MFAChallenge,
        callback: MFACallback | None,
    ) -> AuthResult:
        """Prompt user for 2FA code and submit to provider."""
        self.state_machine.transition_to(
            AuthStatus.MFA_REQUIRED,
            reason="broker_mfa_challenge",
            details=f"method={challenge.method.value}",
        )
        cb = callback or default_interactive_mfa_prompt
        try:
            code = cb(challenge)
        except (KeyboardInterrupt, TimeoutError, ValueError, RuntimeError) as exc:
            self._last_error = f"2FA challenge cancelled or timed out: {exc}"
            self.state_machine.transition_to(
                AuthStatus.DISCONNECTED,
                reason="mfa_cancelled",
            )
            return AuthResult(success=False, error_message=str(exc))

        self.state_machine.transition_to(
            AuthStatus.AUTHENTICATING,
            reason="submitting_mfa_code",
        )
        try:
            res = await self.provider.verify_mfa(challenge, code)
            if not res.success:
                self.state_machine.transition_to(
                    AuthStatus.SESSION_INVALID,
                    reason="mfa_verification_rejected",
                    details=res.error_message or "Invalid 2FA code",
                )
                self._last_error = res.error_message or "MFA verification failed"
            return res
        except (ConnectionError, LoginError, TimeoutError, OSError) as exc:
            self._last_error = f"MFA verification failed: {exc}"
            self.state_machine.transition_to(
                AuthStatus.SESSION_INVALID,
                reason="mfa_verification_error",
            )
            return AuthResult(success=False, error_message=str(exc))

    async def _process_auth_result(self, result: AuthResult, email: str) -> bool:
        """Process successful or failed AuthResult and initialize AuthSession."""
        if not result.success or not result.ssid:
            self._last_error = result.error_message or "Authentication failed without message."
            self.state_machine.transition_to(
                AuthStatus.SESSION_INVALID,
                reason="auth_failed",
                details=self._last_error,
            )
            return False

        # Successfully authenticated
        self._session = AuthSession(
            ssid=result.ssid,
            email=email,
            ttl_seconds=self.config.session_ttl_seconds,
        )
        self.state_machine.transition_to(
            AuthStatus.AUTHENTICATED,
            reason="credentials_verified",
            details=f"account={redact_email(email)}",
        )

        # Probe profile and verify PRACTICE status
        await self._discover_account_profile()
        return True

    async def _discover_account_profile(self) -> None:
        """Fetch server profile, assess PRACTICE verification, and transition state."""
        if self._session is None:
            return

        try:
            profile, balances = await self.provider.get_profile_and_balances(self._session.ssid)
        except (ConnectionError, LoginError, TimeoutError, OSError) as exc:
            logger.warning("Failed to discover profile: %s", exc)
            profile, balances = None, []

        if profile is not None:
            active_bal_id = str(profile.raw.get("balance_id", ""))
            diag_report = self._account_diagnostic.evaluate(
                profile=profile,
                balances=balances,
                active_balance_id=active_bal_id,
                last_verification_ts=time.time(),
            )
            self._session.is_practice_verified = diag_report.is_practice_verified
            self._session.active_balance_id = active_bal_id
            matched_b = next((b for b in balances if str(b.id) == active_bal_id), None)
            if matched_b is not None:
                self._session.server_account_type = matched_b.type.name
            else:
                self._session.server_account_type = (
                    "PRACTICE" if diag_report.is_practice_verified else "REAL"
                )

            # If user requires PRACTICE verification and account is not verified practice:
            if self.config.require_practice_verification and not diag_report.is_practice_verified:
                self.state_machine.transition_to(
                    AuthStatus.ACCOUNT_UNVERIFIED,
                    reason="practice_verification_failed",
                    details=diag_report.explanation,
                )
                return

        # Transition to CONNECTED_READ_ONLY
        self.state_machine.transition_to(
            AuthStatus.CONNECTED_READ_ONLY,
            reason="profile_and_read_only_verified",
        )

    async def logout(self) -> bool:
        """Invalidate session on the broker and clear in-memory state."""
        self._last_error = None
        if self._session is not None:
            try:
                await self.provider.logout(self._session.ssid)
            except (ConnectionError, TimeoutError, OSError) as exc:
                logger.warning("Error during provider logout: %s", exc)
            finally:
                self._session.invalidate()
                self._session = None

        self.state_machine.transition_to(
            AuthStatus.DISCONNECTED_BY_USER,
            reason="explicit_user_logout",
        )
        return True

    def handle_transport_disconnect(self) -> None:
        """Called when underlying WebSocket or network connection disconnects."""
        if self.state_machine.current_state in {
            AuthStatus.CONNECTED_READ_ONLY,
            AuthStatus.AUTHENTICATED,
        }:
            self.state_machine.transition_to(
                AuthStatus.RECONNECTING,
                reason="transport_disconnected",
            )
        if self._session is not None:
            reconnects = self._session.increment_reconnect()
            if reconnects > self.config.max_reconnect_attempts:
                self.state_machine.transition_to(
                    AuthStatus.REAUTH_REQUIRED,
                    reason="max_reconnect_attempts_exceeded",
                )

    def handle_reconnect(self, *, success: bool) -> None:
        """Handle transport reconnection result."""
        if not success:
            self.state_machine.transition_to(
                AuthStatus.REAUTH_REQUIRED,
                reason="reconnection_failed",
            )
            return

        if self._session is None or self._session.is_expired():
            self.state_machine.transition_to(
                AuthStatus.AUTH_EXPIRED,
                reason="session_expired_during_reconnect",
            )
            return

        self.state_machine.transition_to(
            AuthStatus.CONNECTED_READ_ONLY,
            reason="reconnection_succeeded",
        )

    def get_status_snapshot(self) -> dict[str, Any]:
        """Generate a complete, non-sensitive dictionary reflecting authentication status."""
        has_vault = self.vault.is_secure_backend()
        saved_id = self.vault.get_saved_identifier(redact=True) if has_vault else None
        storage_mode = "WINDOWS_CREDENTIAL_VAULT" if saved_id else "INTERACTIVE_MEMORY_ONLY"

        session = self.session
        session_age = session.age_seconds() if session else None
        ttl_rem = max(0.0, session.ttl_seconds - session_age) if (session and session_age) else None

        conn_status = "DISCONNECTED"
        if self.current_state == AuthStatus.CONNECTED_READ_ONLY:
            conn_status = "CONNECTED_READ_ONLY"
        elif self.current_state == AuthStatus.RECONNECTING:
            conn_status = "RECONNECTING"
        elif self.current_state == AuthStatus.AUTHENTICATING:
            conn_status = "CONNECTING"

        return {
            "connection_status": conn_status,
            "auth_status": self.current_state.value,
            "session_active": bool(session),
            "session_account": redact_email(session.email) if session else None,
            "session_age_seconds": round(session_age, 1) if session_age is not None else None,
            "session_ttl_remaining_seconds": round(ttl_rem, 1) if ttl_rem is not None else None,
            "credential_storage_mode": storage_mode,
            "saved_account": saved_id,
            "vault_backend_secure": has_vault,
            "server_account_type": session.server_account_type if session else "UNKNOWN",
            "is_practice_verified": session.is_practice_verified if session else False,
            "broker_ledger_status": BrokerLedgerStatus.BLOCKED.value,
            "last_error": self._last_error,
        }

    @staticmethod
    def _prompt_credentials_from_terminal() -> tuple[str, str]:
        """Helper to prompt for credentials securely without echoing password."""
        try:
            email = input("IQ Option email: ").strip()
            password = getpass.getpass("IQ Option password: ")
            return email, password
        except EOFError as err:
            raise RuntimeError("Non-interactive session cannot prompt for credentials.") from err
