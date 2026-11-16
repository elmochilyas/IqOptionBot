"""Unit tests for Phase 8 secure authentication, session management, and Windows vault.

Covers:
- Empty and invalid credentials
- 2FA challenge, incorrect code, expiration, and user cancellation
- Session expiration and TTL tracking
- Transport disconnect and bounded backoff
- Reconnect and reauth enforcement
- Insecure keyring backend fail-closed protection
- Windows Credential Vault opt-in and lifecycle
- Zero secret leakage in repr, str, errors, and status
- Clean logout and session invalidation
- Illegal state machine transition rejection
- Browser login feasibility assessment
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iqoption_research.auth.browser_research import get_browser_login_feasibility
from iqoption_research.auth.config import RuntimeSecurityConfig
from iqoption_research.auth.manager import AuthenticationManager
from iqoption_research.auth.mfa import MFAChallenge, MFAMethod
from iqoption_research.auth.provider import AuthResult, IQOptionAuthProvider, MockAuthProvider
from iqoption_research.auth.session import AuthSession
from iqoption_research.auth.state import (
    AuthStateMachine,
    AuthStateTransitionError,
    AuthStatus,
)
from iqoption_research.auth.vault import (
    InsecureKeyringBackendError,
    OptInRequiredError,
    WindowsCredentialVault,
)
from iqoption_research.execution.account_diagnostics import BrokerLedgerStatus
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.security.credentials import (
    Credentials,
    redact_email,
    redact_secrets,
)


class MockKeyringBackend:
    """In-memory mock keyring backend for testing vault logic safely without OS changes."""

    def __init__(self, is_win_vault: bool = True) -> None:
        self._storage: dict[tuple[str, str], str] = {}
        self.__class__.__name__ = "WinVaultKeyring" if is_win_vault else "ChainerBackend"
        self.__class__.__module__ = (
            "keyring.backends.Windows" if is_win_vault else "keyring.backends.chainer"
        )
        self.name = "Mock Windows Vault" if is_win_vault else "Mock Insecure Chainer"

    def get_password(self, service: str, username: str) -> str | None:
        return self._storage.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._storage[service, username] = password

    def delete_password(self, service: str, username: str) -> None:
        self._storage.pop((service, username), None)


# -----------------------------------------------------------------------------
# 1. Credential & Input Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_credentials_rejected() -> None:
    """Empty username or password must fail immediately without network attempts."""
    manager = AuthenticationManager(provider=MockAuthProvider())
    success = await manager.login_interactive(email="", password="")
    assert success is False
    assert manager.current_state == AuthStatus.DISCONNECTED
    assert manager.last_error is not None
    assert "cannot be empty" in manager.last_error.lower()


@pytest.mark.asyncio
async def test_invalid_credentials_rejected_and_session_invalid() -> None:
    """Incorrect password transitions to SESSION_INVALID and halts with zero retries."""
    provider = MockAuthProvider(valid_email="owner@example.com", valid_password="correct_password")
    manager = AuthenticationManager(provider=provider)

    success = await manager.login_interactive(email="owner@example.com", password="wrong_password")
    assert success is False
    assert manager.current_state == AuthStatus.SESSION_INVALID
    assert manager.is_authenticated() is False
    assert manager.session is None


# -----------------------------------------------------------------------------
# 2. MFA Challenge & Resolution Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mfa_challenge_and_successful_resolution() -> None:
    """MFA challenge prompts for code, verifies, and enters CONNECTED_READ_ONLY."""
    provider = MockAuthProvider(
        valid_email="owner@example.com",
        valid_password="correct_password",
        require_2fa=True,
        expected_mfa_code="654321",
        mfa_method=MFAMethod.AUTHENTICATOR,
    )
    manager = AuthenticationManager(provider=provider)

    def _mock_mfa_cb(_challenge: MFAChallenge) -> str:
        assert _challenge.method == MFAMethod.AUTHENTICATOR
        return "654321"

    success = await manager.login_interactive(
        email="owner@example.com",
        password="correct_password",
        mfa_callback=_mock_mfa_cb,
    )
    assert success is True
    assert manager.current_state == AuthStatus.CONNECTED_READ_ONLY
    assert manager.is_authenticated() is True
    assert manager.session is not None
    assert manager.session.email == "owner@example.com"


@pytest.mark.asyncio
async def test_mfa_incorrect_code_rejected() -> None:
    """Submitting an invalid 2FA code fails verification and enters SESSION_INVALID."""
    provider = MockAuthProvider(
        valid_email="owner@example.com",
        valid_password="correct_password",
        require_2fa=True,
        expected_mfa_code="654321",
    )
    manager = AuthenticationManager(provider=provider)

    def _wrong_mfa_cb(_challenge: MFAChallenge) -> str:
        return "000000"

    success = await manager.login_interactive(
        email="owner@example.com",
        password="correct_password",
        mfa_callback=_wrong_mfa_cb,
    )
    assert success is False
    assert manager.current_state == AuthStatus.SESSION_INVALID
    assert manager.is_authenticated() is False


@pytest.mark.asyncio
async def test_mfa_timeout_and_expiration() -> None:
    """Expired MFA challenge is rejected."""
    provider = MockAuthProvider(require_2fa=True)
    manager = AuthenticationManager(provider=provider)

    def _expired_mfa_cb(_challenge: MFAChallenge) -> str:
        # Simulate challenge that was created in the past
        object.__setattr__(_challenge, "created_at", time.time() - 300.0)
        return "123456"

    success = await manager.login_interactive(
        email="trader@example.com",
        password="mock_secret_123",
        mfa_callback=_expired_mfa_cb,
    )
    assert success is False
    assert manager.is_authenticated() is False


@pytest.mark.asyncio
async def test_mfa_user_cancellation() -> None:
    """User cancelling MFA prompt aborts login cleanly."""
    provider = MockAuthProvider(require_2fa=True)
    manager = AuthenticationManager(provider=provider)

    def _cancel_cb(_challenge: MFAChallenge) -> str:
        raise KeyboardInterrupt("User cancelled")

    success = await manager.login_interactive(
        email="trader@example.com",
        password="mock_secret_123",
        mfa_callback=_cancel_cb,
    )
    assert success is False
    assert manager.current_state == AuthStatus.DISCONNECTED


# -----------------------------------------------------------------------------
# 3. Session Lifetime & Reconnection Tests
# -----------------------------------------------------------------------------


def test_session_expiration_tracking() -> None:
    """Session accurately tracks elapsed time and TTL expiration."""
    now = time.time()
    session = AuthSession(
        ssid="test_ssid_12345",
        email="trader@example.com",
        authenticated_at=now,
        ttl_seconds=300.0,
    )

    assert session.is_active is True
    assert session.age_seconds(current_time=now + 100.0) == 100.0
    assert session.is_expired(current_time=now + 200.0) is False

    # After TTL passes
    assert session.is_expired(current_time=now + 301.0) is True


def test_transport_disconnect_and_bounded_reconnect_backoff() -> None:
    """Transport disconnection triggers bounded reconnects and fails to REAUTH_REQUIRED."""
    cfg = RuntimeSecurityConfig(max_reconnect_attempts=3)
    manager = AuthenticationManager(config=cfg)

    # Put state in CONNECTED_READ_ONLY
    manager.state_machine.transition_to(AuthStatus.AUTHENTICATING)
    manager.state_machine.transition_to(AuthStatus.AUTHENTICATED)
    manager.state_machine.transition_to(AuthStatus.CONNECTED_READ_ONLY)
    manager._session = AuthSession(ssid="test_ssid", email="trader@example.com")

    # 1st disconnect
    manager.handle_transport_disconnect()
    assert manager.current_state == AuthStatus.RECONNECTING
    assert manager._session.reconnect_count == 1

    # 2nd disconnect
    manager.handle_transport_disconnect()
    assert manager._session.reconnect_count == 2

    # 3rd disconnect
    manager.handle_transport_disconnect()
    assert manager._session.reconnect_count == 3

    # 4th disconnect exceeds limit -> REAUTH_REQUIRED
    manager.handle_transport_disconnect()
    assert manager.current_state == AuthStatus.REAUTH_REQUIRED


def test_successful_reconnect_restores_read_only() -> None:
    """Successful reconnect with a valid session restores CONNECTED_READ_ONLY."""
    manager = AuthenticationManager()
    manager.state_machine.transition_to(AuthStatus.AUTHENTICATING)
    manager.state_machine.transition_to(AuthStatus.AUTHENTICATED)
    manager.state_machine.transition_to(AuthStatus.CONNECTED_READ_ONLY)
    manager._session = AuthSession(ssid="test_ssid", email="trader@example.com")

    manager.handle_transport_disconnect()
    assert manager.current_state == AuthStatus.RECONNECTING

    manager.handle_reconnect(success=True)
    assert manager.current_state == AuthStatus.CONNECTED_READ_ONLY


# -----------------------------------------------------------------------------
# 4. Account State Change & Discrepancy Diagnostics
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_account_state_change_real_detected() -> None:
    """Active REAL account triggers ACCOUNT_UNVERIFIED when practice verification is required."""
    profile = Profile.from_raw({
        "user_id": 9999,
        "email": "trader@example.com",
        "balance_id": 2002,  # Matches the REAL balance!
    })
    balances = [
        Balance(id="1001", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="2002", type=AccountType.REAL, amount=500.0, currency="USD", is_active=True),
    ]
    provider = MockAuthProvider(
        mock_profile=profile,
        mock_balances=balances,
    )
    cfg = RuntimeSecurityConfig(require_practice_verification=True)
    manager = AuthenticationManager(config=cfg, provider=provider)

    success = await manager.login_interactive(
        email="trader@example.com",
        password="mock_secret_123",
    )
    assert success is True
    # Because REAL is active, PRACTICE verification fails
    assert manager.current_state == AuthStatus.ACCOUNT_UNVERIFIED
    assert manager.session is not None
    assert manager.session.is_practice_verified is False
    assert manager.session.server_account_type == "REAL"

    # Status snapshot still shows broker ledger is BLOCKED
    snap = manager.get_status_snapshot()
    assert snap["broker_ledger_status"] == BrokerLedgerStatus.BLOCKED.value


# -----------------------------------------------------------------------------
# 5. Windows Credential Vault Tests
# -----------------------------------------------------------------------------


def test_vault_explicit_opt_in_enforced() -> None:
    """Saving credentials without explicit opt-in raises OptInRequiredError."""
    backend = MockKeyringBackend(is_win_vault=True)
    vault = WindowsCredentialVault(keyring_backend=backend)

    with pytest.raises(OptInRequiredError):
        vault.save_credentials("user@example.com", "secret123", explicit_opt_in=False)


def test_vault_insecure_backend_fails_closed() -> None:
    """Insecure keyring backend (e.g. chainer / plaintext) is rejected and fails closed."""
    backend = MockKeyringBackend(is_win_vault=False)  # Simulates chainer or insecure backend
    vault = WindowsCredentialVault(keyring_backend=backend)

    assert vault.is_secure_backend() is False
    with pytest.raises(InsecureKeyringBackendError):
        vault.save_credentials("user@example.com", "secret123", explicit_opt_in=True)

    with pytest.raises(InsecureKeyringBackendError):
        vault.load_saved_credentials()


def test_vault_save_load_delete_lifecycle() -> None:
    """Complete save, check, load, and delete lifecycle in Windows vault."""
    backend = MockKeyringBackend(is_win_vault=True)
    vault = WindowsCredentialVault(keyring_backend=backend)

    assert vault.is_secure_backend() is True
    assert vault.has_saved_credentials() is False
    assert vault.get_saved_identifier() is None

    # Save with opt-in
    vault.save_credentials("trader@example.com", "my_vault_password", explicit_opt_in=True)
    assert vault.has_saved_credentials() is True
    assert vault.get_saved_identifier(redact=True) == redact_email("trader@example.com")
    assert vault.get_saved_identifier(redact=False) == "trader@example.com"

    # Load
    creds = vault.load_saved_credentials()
    assert creds is not None
    assert creds.email == "trader@example.com"
    assert creds.password == "my_vault_password"

    # Delete
    deleted = vault.delete_saved_credentials()
    assert deleted is True
    assert vault.has_saved_credentials() is False
    assert vault.load_saved_credentials() is None


# -----------------------------------------------------------------------------
# 6. Secret Redaction & Leakage Prevention Tests
# -----------------------------------------------------------------------------


def test_zero_secret_leakage_in_repr_and_str() -> None:
    """Credentials, AuthSession, and MFAChallenge representations never expose secrets."""
    creds = Credentials(email="trader@example.com", password="SuperSecretPassword!123")
    assert "SuperSecretPassword" not in repr(creds)
    assert "SuperSecretPassword" not in str(creds)
    assert "***REDACTED***" in repr(creds)

    session = AuthSession(ssid="abcdef1234567890abcdef1234567890", email="trader@example.com")
    assert "abcdef1234567890" not in repr(session)
    assert "abcdef1234567890" not in str(session)
    assert "***REDACTED***" in repr(session)

    mfa = MFAChallenge(token="raw_secret_2fa_token_xyz")
    assert "raw_secret_2fa_token_xyz" not in repr(mfa)
    assert "raw_secret_2fa_token_xyz" not in str(mfa)
    assert "***REDACTED***" in repr(mfa)

    res = AuthResult(success=True, ssid="abcdef1234567890")
    assert "abcdef1234567890" not in repr(res)
    assert "***REDACTED***" in repr(res)


def test_zero_secret_leakage_in_error_messages() -> None:
    """Secrets in error strings are automatically redacted."""
    raw_error = (
        "Failed to authenticate with password: MySecretPassword123 and ssid: 1234567890abcdef"
    )
    redacted = redact_secrets(raw_error)
    assert "MySecretPassword123" not in redacted
    assert "***REDACTED***" in redacted


# -----------------------------------------------------------------------------
# 7. Logout and State Machine Transitions
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_logout_invalidates_session() -> None:
    """Logout invalidates in-memory session, erases references, and transitions state."""
    manager = AuthenticationManager(provider=MockAuthProvider())
    await manager.login_interactive(email="trader@example.com", password="mock_secret_123")
    assert manager.is_authenticated() is True

    await manager.logout()
    assert manager.is_authenticated() is False
    assert manager.session is None
    assert manager.current_state == AuthStatus.DISCONNECTED_BY_USER


def test_state_machine_invalid_transitions_raise() -> None:
    """Illegal transitions raise AuthStateTransitionError."""
    sm = AuthStateMachine(initial_state=AuthStatus.DISCONNECTED)

    # Cannot jump straight from DISCONNECTED to CONNECTED_READ_ONLY
    with pytest.raises(AuthStateTransitionError):
        sm.transition_to(AuthStatus.CONNECTED_READ_ONLY)


# -----------------------------------------------------------------------------
# 8. Browser Login Feasibility Assessment
# -----------------------------------------------------------------------------


def test_browser_feasibility_report_concludes_separation() -> None:
    """Verifies that the feasibility report concludes separate authentication is mandated."""
    report = get_browser_login_feasibility()
    assert report.delegated_oauth_supported is False
    assert report.pkce_supported is False
    assert report.cookie_extraction_recommended is False
    assert report.browser_scraping_recommended is False
    assert report.verdict == "SEPARATE_AUTHENTICATION_MANDATED"
    assert len(report.security_risks) >= 5


# -----------------------------------------------------------------------------
# 9. Broker Response Handling & Authentication Failure Telemetry Regressions
# -----------------------------------------------------------------------------


def _build_mock_http_session(
    status: int = 200,
    json_payload: dict | None = None,
    cookies: dict[str, str] | None = None,
) -> MagicMock:
    """Construct mock aiohttp session for testing broker HTTP responses offline."""
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_payload)

    cookie_dict = {}
    if cookies:
        for k, v in cookies.items():
            c_mock = MagicMock()
            c_mock.value = v
            cookie_dict[k] = c_mock
    mock_response.cookies = cookie_dict
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.close = AsyncMock()
    return mock_session


@pytest.mark.asyncio
async def test_iqoption_auth_provider_extracts_ssid_from_cookie() -> None:
    """SSID present in HTTP response cookies is successfully extracted."""
    provider = IQOptionAuthProvider()
    creds = Credentials(email="trader@example.com", password="mock_secret_123")
    mock_session = _build_mock_http_session(
        status=200,
        json_payload={"code": "success"},
        cookies={"ssid": "cookie_ssid_1234567890"},
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await provider.login(creds)
        assert result.success is True
        assert result.ssid == "cookie_ssid_1234567890"
        assert result.status_code == 200


@pytest.mark.asyncio
async def test_iqoption_auth_provider_extracts_ssid_from_json_data() -> None:
    """SSID present in JSON payload under 'data.ssid' is successfully extracted."""
    provider = IQOptionAuthProvider()
    creds = Credentials(email="trader@example.com", password="mock_secret_123")
    mock_session = _build_mock_http_session(
        status=200,
        json_payload={"code": "success", "data": {"ssid": "data_ssid_9876543210"}},
        cookies={},
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await provider.login(creds)
        assert result.success is True
        assert result.ssid == "data_ssid_9876543210"
        assert result.status_code == 200


@pytest.mark.asyncio
async def test_iqoption_auth_provider_extracts_ssid_from_json_root() -> None:
    """SSID present at JSON root payload is successfully extracted."""
    provider = IQOptionAuthProvider()
    creds = Credentials(email="trader@example.com", password="mock_secret_123")
    mock_session = _build_mock_http_session(
        status=200,
        json_payload={"code": "success", "ssid": "root_ssid_5555555555"},
        cookies={},
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await provider.login(creds)
        assert result.success is True
        assert result.ssid == "root_ssid_5555555555"
        assert result.status_code == 200


@pytest.mark.asyncio
async def test_iqoption_auth_provider_rejects_credentials_with_sanitized_error() -> None:
    """HTTP 401 or invalid_credentials returns clear, sanitized error
    without exposing credentials.
    """
    provider = IQOptionAuthProvider()
    creds = Credentials(email="trader@example.com", password="wrong_secret_pass")
    mock_session = _build_mock_http_session(
        status=401,
        json_payload={"code": "invalid_credentials", "message": "Invalid email or password"},
        cookies={},
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await provider.login(creds)
        assert result.success is False
        assert result.ssid is None
        assert result.status_code == 401
        assert result.error_message == "Invalid email or password"
        assert "wrong_secret_pass" not in str(result)


@pytest.mark.asyncio
async def test_iqoption_auth_provider_detects_2fa_challenge_variants() -> None:
    """2FA challenge variants (verify, verify_2fa) return structured MFAChallenge."""
    provider = IQOptionAuthProvider()
    creds = Credentials(email="trader@example.com", password="correct_secret_pass")

    # Variant 1: standard verify
    mock_session1 = _build_mock_http_session(
        status=200,
        json_payload={"code": "verify", "token": "mfa_tok_1", "method": "authenticator"},
        cookies={},
    )
    with patch("aiohttp.ClientSession", return_value=mock_session1):
        res1 = await provider.login(creds)
        assert res1.success is False
        assert res1.mfa_challenge is not None
        assert res1.mfa_challenge.token == "mfa_tok_1"
        assert res1.mfa_challenge.method == MFAMethod.AUTHENTICATOR

    # Variant 2: verify_2fa with SMS
    mock_session2 = _build_mock_http_session(
        status=200,
        json_payload={"code": "verify_2fa", "token": "mfa_tok_2", "method": "sms"},
        cookies={},
    )
    with patch("aiohttp.ClientSession", return_value=mock_session2):
        res2 = await provider.login(creds)
        assert res2.success is False
        assert res2.mfa_challenge is not None
        assert res2.mfa_challenge.token == "mfa_tok_2"
        assert res2.mfa_challenge.method == MFAMethod.SMS


@pytest.mark.asyncio
async def test_iqoption_auth_provider_handles_rate_limiting() -> None:
    """HTTP 429 returns rate limit notification without crashing."""
    provider = IQOptionAuthProvider()
    creds = Credentials(email="trader@example.com", password="mock_secret_123")
    mock_session = _build_mock_http_session(
        status=429,
        json_payload={"message": "Too many requests"},
        cookies={},
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await provider.login(creds)
        assert result.success is False
        assert result.status_code == 429
        assert "rate limit" in str(result.error_message).lower()


@pytest.mark.asyncio
async def test_iqoption_auth_provider_handles_missing_token_or_challenge() -> None:
    """HTTP 200 without SSID or 2FA challenge returns explicit sanitized error."""
    provider = IQOptionAuthProvider()
    creds = Credentials(email="trader@example.com", password="mock_secret_123")
    mock_session = _build_mock_http_session(
        status=200,
        json_payload={"status": "ok"},
        cookies={},
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await provider.login(creds)
        assert result.success is False
        assert result.status_code == 200
        assert "SSID" in str(result.error_message)


def test_authentication_manager_check_session_validity() -> None:
    """check_session_validity diagnoses unauthenticated, invalid, and expired states."""
    manager = AuthenticationManager(provider=MockAuthProvider())
    is_valid, reason = manager.check_session_validity()
    assert is_valid is False
    assert "No active session" in str(reason)

    # Active valid session
    session = AuthSession(ssid="test_ssid_valid", email="trader@example.com", ttl_seconds=3600.0)
    session.is_practice_verified = True
    manager._session = session
    is_valid, reason = manager.check_session_validity()
    assert is_valid is True
    assert reason is None

    # Expired session
    session.authenticated_at = time.time() - 4000.0
    is_valid, reason = manager.check_session_validity()
    assert is_valid is False
    assert "expired" in str(reason).lower()
