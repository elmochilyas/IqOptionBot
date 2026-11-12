"""Integration tests for Phase 8 authentication lifecycle and offline development isolation.

Covers:
- Full interactive login and session usage
- Full saved-credentials workflow with vault backend
- Bounded rate-limit error handling
- Zero network socket assertion during offline development operations
- CLI execution of auth commands
"""

from __future__ import annotations

import socket

import pytest
from typer.testing import CliRunner

from iqoption_research.auth.manager import AuthenticationManager
from iqoption_research.auth.provider import MockAuthProvider
from iqoption_research.auth.state import AuthStatus
from iqoption_research.auth.vault import WindowsCredentialVault
from iqoption_research.cli.main import app
from iqoption_research.security.credentials import redact_email
from tests.unit.test_phase8_secure_auth import MockKeyringBackend


@pytest.mark.asyncio
async def test_end_to_end_interactive_login_flow() -> None:
    """Full lifecycle: Interactive login -> Profile discovery -> Session verification -> Logout."""
    provider = MockAuthProvider(
        valid_email="owner@example.com",
        valid_password="mock_vault_password",
    )
    manager = AuthenticationManager(provider=provider)

    assert manager.is_authenticated() is False
    assert manager.current_state == AuthStatus.DISCONNECTED

    # 1. Login
    success = await manager.login_interactive(
        email="owner@example.com",
        password="mock_vault_password",
    )
    assert success is True
    assert manager.is_authenticated() is True
    assert manager.current_state == AuthStatus.CONNECTED_READ_ONLY

    # 2. Status inspection
    snap = manager.get_status_snapshot()
    assert snap["auth_status"] == "CONNECTED_READ_ONLY"
    assert snap["session_active"] is True
    assert snap["session_account"] == "o***r@example.com"
    assert snap["is_practice_verified"] is True
    assert snap["server_account_type"] == "PRACTICE"
    assert snap["broker_ledger_status"] == "BLOCKED"

    # 3. Touch session
    manager.session.touch()
    assert manager.session.age_seconds() >= 0.0

    # 4. Logout
    logged_out = await manager.logout()
    assert logged_out is True
    assert manager.is_authenticated() is False
    assert manager.session is None
    assert manager.current_state == AuthStatus.DISCONNECTED_BY_USER


@pytest.mark.asyncio
async def test_end_to_end_saved_credentials_login_flow() -> None:
    """Full lifecycle: Save to vault -> Login from vault -> Verify -> Delete from vault."""
    backend = MockKeyringBackend(is_win_vault=True)
    vault = WindowsCredentialVault(keyring_backend=backend)
    provider = MockAuthProvider(
        valid_email="saved_owner@example.com",
        valid_password="saved_password_789",
    )
    manager = AuthenticationManager(provider=provider, vault=vault)

    # 1. Save credentials with opt-in
    vault.save_credentials("saved_owner@example.com", "saved_password_789", explicit_opt_in=True)
    assert vault.has_saved_credentials() is True

    # 2. Login using saved credentials
    success = await manager.login_saved()
    assert success is True
    assert manager.is_authenticated() is True
    assert manager.session is not None
    assert manager.session.email == "saved_owner@example.com"
    assert manager.current_state == AuthStatus.CONNECTED_READ_ONLY

    # 3. Check snapshot reflects storage mode
    snap = manager.get_status_snapshot()
    assert snap["credential_storage_mode"] == "WINDOWS_CREDENTIAL_VAULT"
    assert snap["saved_account"] == redact_email("saved_owner@example.com")

    # 4. Cleanup
    await manager.logout()
    vault.delete_saved_credentials()
    assert vault.has_saved_credentials() is False


@pytest.mark.asyncio
async def test_rate_limiting_and_broker_challenge_halt() -> None:
    """Broker rate-limiting (HTTP 429) stops immediately without endless loops."""
    provider = MockAuthProvider(simulate_rate_limit=True)
    manager = AuthenticationManager(provider=provider)

    success = await manager.login_interactive(
        email="trader@example.com",
        password="mock_secret_123",
    )
    assert success is False
    assert manager.current_state == AuthStatus.SESSION_INVALID
    assert manager.last_error is not None
    assert "rate-limit" in manager.last_error.lower()


def test_no_network_socket_opened_during_mock_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirms zero network sockets are opened during mock authentication operations."""

    def _blocked_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Network socket creation is forbidden during offline mock operations")

    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.setattr(socket, "create_connection", _blocked_socket)

    backend = MockKeyringBackend(is_win_vault=True)
    vault = WindowsCredentialVault(keyring_backend=backend)
    provider = MockAuthProvider()
    manager = AuthenticationManager(provider=provider, vault=vault)

    import asyncio

    success = asyncio.run(
        manager.login_interactive(email="trader@example.com", password="mock_secret_123")
    )
    assert success is True
    assert manager.is_authenticated() is True

    asyncio.run(manager.logout())
    assert manager.is_authenticated() is False


@pytest.fixture(autouse=True)
def _reset_auth_singleton() -> None:
    """Reset module-level singleton _AUTH_MANAGER before each test to isolate CLI tests."""
    from iqoption_research.cli.commands import auth_cmd

    auth_cmd._AUTH_MANAGER = None


def test_cli_auth_status_renders_without_secrets() -> None:
    """CLI 'bot auth status' renders successfully with zero secret leakage."""
    runner = CliRunner()
    result = runner.invoke(app, ["bot", "auth", "status"])
    assert result.exit_code == 0
    assert "Authentication Status" in result.output
    assert "Broker Ledger Routing" in result.output
    assert "BLOCKED" in result.output
    # Secrets must not appear
    assert "password" not in result.output.lower()
    assert "ssid" not in result.output.lower()


def test_cli_auth_login_mock_warning_and_failure_distinction() -> None:
    """CLI 'bot auth login' warns user and clearly reports MOCK AUTHENTICATION FAILED."""
    runner = CliRunner()
    # Provide credentials that do not match the mock fixture
    result = runner.invoke(
        app, ["bot", "auth", "login"], input="trader@realbroker.com\nmypassword\n"
    )
    assert result.exit_code == 1
    assert "OFFLINE MOCK FIXTURE ACTIVE" in result.output
    assert "DO NOT ENTER REAL PASSWORDS" in result.output
    assert "MOCK AUTHENTICATION FAILED" in result.output
    assert "Offline Mock Notice" in result.output
    assert "bot auth login --live" in result.output


def test_cli_auth_login_mock_success() -> None:
    """CLI 'bot auth login' succeeds with valid mock fixture credentials."""
    runner = CliRunner()
    result = runner.invoke(
        app, ["bot", "auth", "login"], input="trader@example.com\nmock_secret_123\n"
    )
    assert result.exit_code == 0
    assert "Authentication Successful!" in result.output
    assert "Offline Mock Fixture" in result.output
    assert "Note on CLI Session Lifetime" in result.output


def test_cli_auth_login_live_routes_to_live_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 'bot auth login --live' correctly routes to IQOptionAuthProvider."""
    from unittest.mock import AsyncMock

    from iqoption_research.auth.provider import AuthResult, IQOptionAuthProvider

    called_live = False

    async def _mock_live_login(self: IQOptionAuthProvider, creds: object) -> AuthResult:
        nonlocal called_live
        called_live = True
        return AuthResult(success=True, ssid="fake_live_ssid_12345")

    monkeypatch.setattr(IQOptionAuthProvider, "login", _mock_live_login)
    monkeypatch.setattr(
        IQOptionAuthProvider,
        "get_profile_and_balances",
        AsyncMock(return_value=(None, [])),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["bot", "auth", "login", "--live"],
        input="user@iqoption.com\nsecretpass123\n",
    )
    assert result.exit_code == 0
    assert called_live is True
    assert "LIVE IQ OPTION BROKER AUTHENTICATION" in result.output
    assert "Live HTTPS Broker" in result.output


def test_cli_auth_check_connection_unauthenticated_message() -> None:
    """CLI 'bot auth check-connection' provides helpful context on memory boundaries."""
    runner = CliRunner()
    result = runner.invoke(app, ["bot", "auth", "check-connection"])
    assert result.exit_code == 1
    assert "No active authenticated session in current process memory" in result.output
    assert "In-memory sessions are bound to a running process" in result.output
