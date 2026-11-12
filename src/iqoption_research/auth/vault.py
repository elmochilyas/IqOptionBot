"""Windows Credential Manager integration for secure local credential storage.

Provides encrypted-at-rest credential storage via Windows Credential Manager
(using the native OS DPAPI backend through the Python keyring library).

Security Boundaries & Guarantees:
- Explicit Opt-In: Credentials are never saved automatically.
- Fail-Closed Architecture: Fails immediately if an insecure or fallback
  backend (e.g. plaintext, chainer, fail-keyring) is configured.
- Zero In-File Persistence: Credentials are never stored in Git, .env,
  JSON, SQLite, local files, or logs.
- Redacted Status: Informs callers whether credentials exist without
  exposing passwords.
- Memory Minimization: Passwords are loaded into short-lived Credentials
  objects and discarded once authentication completes.

Host Isolation Limitation Notice:
    Windows Credential Manager encrypts credentials at rest using DPAPI tied to
    the current Windows user account. This protects credentials against offline
    disk extraction and separate Windows users on the machine. However, any
    process running under the same authorized Windows user context can query
    these credentials via CredRead. For enterprise isolation, run the bot
    under a dedicated, unprivileged Windows user account.
"""

from __future__ import annotations

import logging
from typing import Any

from iqoption_research.security.credentials import Credentials, redact_email

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "IqOptionBot"
ACTIVE_ACCOUNT_KEY = "__active_account__"


class VaultError(Exception):
    """Base exception for credential vault operations."""


class OptInRequiredError(VaultError):
    """Raised when an attempt is made to save credentials without explicit opt-in."""


class InsecureKeyringBackendError(VaultError):
    """Raised when the active keyring backend is not a verified OS-backed vault."""


class WindowsCredentialVault:
    """Manages secure credential persistence in Windows Credential Manager."""

    def __init__(
        self,
        service_name: str = DEFAULT_SERVICE_NAME,
        *,
        keyring_backend: Any | None = None,
    ) -> None:
        """Initialize vault with service name and optional custom keyring backend.

        Args:
            service_name: Credential target service name in Windows Credential Vault.
            keyring_backend: Optional backend override (used for testing or dependency injection).
        """
        self._service_name = service_name
        self._custom_backend = keyring_backend

    @property
    def service_name(self) -> str:
        """Target credential service name."""
        return self._service_name

    def _get_backend(self) -> Any:
        """Resolve the active keyring backend."""
        if self._custom_backend is not None:
            return self._custom_backend
        try:
            import keyring

            return keyring.get_keyring()
        except ImportError as err:
            raise InsecureKeyringBackendError(
                "The 'keyring' package is not installed. "
                "Native OS credential storage is unavailable."
            ) from err

    def is_secure_backend(self) -> bool:
        """Verify that the resolved backend is a genuine OS-backed Windows vault.

        Returns:
            True if the backend is verified as native Windows Credential Vault, False otherwise.
        """
        try:
            backend = self._get_backend()
            backend_class_name = backend.__class__.__name__
            backend_module = getattr(backend.__class__, "__module__", "")

            # Approved Windows native backends
            is_win_vault = (
                backend_class_name == "WinVaultKeyring"
                or "keyring.backends.Windows" in backend_module
            )
            # Explicitly reject insecure or null backends
            is_insecure = (
                "fail" in backend_module.lower()
                or "chainer" in backend_module.lower()
                or "plaintext" in backend_class_name.lower()
                or "null" in backend_class_name.lower()
            )
            if not (is_win_vault and not is_insecure):
                return False

            # Verify that the current process context has permission to query CredRead
            try:
                backend.get_password("probe_service", "probe_user")
            except Exception:  # noqa: BLE001
                return False

            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Backend verification check failed: %s", exc)
            return False

    def get_backend_info(self) -> dict[str, str]:
        """Return non-sensitive metadata describing the active keyring backend."""
        try:
            backend = self._get_backend()
            return {
                "name": str(getattr(backend, "name", backend.__class__.__name__)),
                "class": backend.__class__.__name__,
                "module": getattr(backend.__class__, "__module__", "unknown"),
                "is_secure": str(self.is_secure_backend()),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "name": "Unavailable",
                "class": "None",
                "module": "unknown",
                "is_secure": "False",
                "error": str(exc),
            }

    def has_saved_credentials(self) -> bool:
        """Check whether saved credentials exist without returning or logging them.

        Returns:
            True if an active credential set is present in the vault.
        """
        if not self.is_secure_backend():
            return False
        try:
            backend = self._get_backend()
            active_email = backend.get_password(self._service_name, ACTIVE_ACCOUNT_KEY)
            if not active_email:
                return False
            pwd = backend.get_password(self._service_name, active_email)
            return bool(pwd)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to check saved credentials: %s", exc)
            return False

    def get_saved_identifier(self, *, redact: bool = True) -> str | None:
        """Return the account identifier (email) of saved credentials without password.

        Args:
            redact: Whether to mask the returned email (default True for display safety).
        """
        if not self.is_secure_backend():
            return None
        try:
            backend = self._get_backend()
            active_email = backend.get_password(self._service_name, ACTIVE_ACCOUNT_KEY)
            if not active_email:
                return None
            return redact_email(active_email) if redact else active_email
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to get saved identifier: %s", exc)
            return None

    def save_credentials(
        self,
        email: str,
        password: str,
        *,
        explicit_opt_in: bool = False,
    ) -> None:
        """Store credentials in Windows Credential Manager under the service name.

        Args:
            email: Account email.
            password: Plaintext password to encrypt at rest in the OS vault.
            explicit_opt_in: Must be True to proceed.

        Raises:
            OptInRequiredError: If explicit_opt_in is False.
            InsecureKeyringBackendError: If the backend is not a verified OS vault.
            ValueError: If email or password is empty.
        """
        if not explicit_opt_in:
            raise OptInRequiredError(
                "Explicit user opt-in is required before saving "
                "credentials to Windows Credential Vault."
            )
        if not email or not password:
            raise ValueError("Cannot save empty email or password.")
        if not self.is_secure_backend():
            raise InsecureKeyringBackendError(
                "Windows Credential Vault is not available or insecure backend detected. "
                "Credentials were not saved."
            )

        backend = self._get_backend()
        clean_email = email.strip()
        # 1. Store the password for this email identity
        backend.set_password(self._service_name, clean_email, password)
        # 2. Store the active email key so future loads know which identifier to fetch
        backend.set_password(self._service_name, ACTIVE_ACCOUNT_KEY, clean_email)
        logger.info(
            "Credentials successfully saved to Windows Credential Vault for account %s",
            redact_email(clean_email),
        )

    def load_saved_credentials(self) -> Credentials | None:
        """Retrieve stored credentials from Windows Credential Manager.

        Returns:
            Credentials instance if found and valid; None otherwise.

        Raises:
            InsecureKeyringBackendError: If backend is insecure or unavailable.
        """
        if not self.is_secure_backend():
            raise InsecureKeyringBackendError(
                "Cannot load credentials: Insecure or unverified keyring backend."
            )
        try:
            backend = self._get_backend()
            active_email = backend.get_password(self._service_name, ACTIVE_ACCOUNT_KEY)
            if not active_email:
                return None
            pwd = backend.get_password(self._service_name, active_email)
            if not pwd:
                return None
            return Credentials(email=active_email, password=pwd)
        except InsecureKeyringBackendError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error loading credentials from vault: %s", exc)
            return None

    def delete_saved_credentials(self) -> bool:
        """Delete stored credentials and active account pointer from Windows Credential Manager.

        Returns:
            True if credentials were deleted; False if none were present or backend failed.
        """
        if not self.is_secure_backend():
            return False
        try:
            backend = self._get_backend()
            active_email = backend.get_password(self._service_name, ACTIVE_ACCOUNT_KEY)
            deleted_any = False
            if active_email:
                try:
                    backend.delete_password(self._service_name, active_email)
                    deleted_any = True
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to delete password entry for %s: %s", active_email, exc)
            try:
                backend.delete_password(self._service_name, ACTIVE_ACCOUNT_KEY)
                deleted_any = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to delete active account key: %s", exc)

            logger.info("Credentials deleted from Windows Credential Vault.")
            return deleted_any
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete credentials from vault: %s", exc)
            return False
