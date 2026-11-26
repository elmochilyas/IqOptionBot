"""Unit tests for Phase 44 saved account selection and fail-closed authentication.

Verifies:
1. Two distinct saved accounts in AccountStore: selecting account 2 selects account 2's context.
2. Requesting a non-existent or unavailable account fails closed (never silently substitutes default).
3. Session state preserves account_id.
4. CLI reject simultaneous --live and --synthetic flags.
5. CLI fails closed when requested --account-id is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from iqoption_research.auth.manager import AuthenticationManager
from iqoption_research.auth.provider import AuthResult
from iqoption_research.auth.vault import WindowsCredentialVault
from iqoption_research.cli.commands import backfill_cmd
from iqoption_research.cli.main import app

runner = CliRunner()


class WinVaultKeyring:
    __module__ = "keyring.backends.Windows"

    def __init__(self):
        self.data: dict[tuple[str, str], str] = {}
        self.priority = 1

    def get_password(self, service: str, key: str) -> str | None:
        return self.data.get((service, key))

    def set_password(self, service: str, key: str, value: str) -> None:
        self.data[service, key] = value

    def delete_password(self, service: str, key: str) -> None:
        self.data.pop((service, key), None)


@pytest.fixture
def mock_vault_with_two_accounts() -> tuple[WindowsCredentialVault, str, str]:
    """Create a vault with two distinct synthetic accounts."""
    backend = WinVaultKeyring()
    vault = WindowsCredentialVault(service_name="test_phase44_vault", keyring_backend=backend)
    store = vault.account_store()

    aid1 = store.save(
        email="trader1@example.com",
        password="Password123!",
        explicit_opt_in=True,
    )
    aid2 = store.save(
        email="trader2@example.com",
        password="Password456!",
        explicit_opt_in=True,
    )

    return vault, aid1, aid2


@pytest.mark.asyncio
async def test_account_selection_authenticates_selected_context(
    mock_vault_with_two_accounts: tuple[WindowsCredentialVault, str, str],
) -> None:
    """Selecting account 2 explicitly authenticates account 2 without substituting default account 1."""
    vault, _aid1, aid2 = mock_vault_with_two_accounts
    mock_provider = MagicMock()
    mock_provider.login = AsyncMock(
        return_value=AuthResult(success=True, ssid="fake_ssid_beta_999")
    )
    mock_provider.get_profile_and_balances = AsyncMock(return_value=(None, []))

    auth = AuthenticationManager(provider=mock_provider, vault=vault)
    with patch.object(vault, "is_secure_backend", return_value=True):
        ok = await auth.login_saved(account_id=aid2)

    assert ok is True
    assert auth.session is not None
    assert auth.session.account_id == aid2
    assert auth.session.email == "trader2@example.com"
    assert auth.session.ssid == "fake_ssid_beta_999"

    # Verify provider received account 2 credentials
    call_creds = mock_provider.login.call_args[0][0]
    assert call_creds.email == "trader2@example.com"


@pytest.mark.asyncio
async def test_account_selection_fails_closed_on_missing_account(
    mock_vault_with_two_accounts: tuple[WindowsCredentialVault, str, str],
) -> None:
    """Requesting an unavailable account fails closed without falling back to default."""
    vault, _, _ = mock_vault_with_two_accounts
    mock_provider = MagicMock()
    auth = AuthenticationManager(provider=mock_provider, vault=vault)

    with patch.object(vault, "is_secure_backend", return_value=True):
        ok = await auth.login_saved(account_id="33333333333333333333333333333333")

    assert ok is False
    assert auth.session is None
    assert auth.last_error is not None
    assert "Selected account '33333333333333333333333333333333' is unavailable" in auth.last_error
    assert mock_provider.login.call_count == 0


def test_cli_rejects_simultaneous_live_and_synthetic() -> None:
    """CLI run and resume fail immediately if both --live and --synthetic are requested."""
    res = runner.invoke(
        app,
        ["data", "collect", "run", "--job-id", "job_dummy", "--live", "--synthetic"],
    )
    assert res.exit_code == 1
    assert "Cannot specify both --live and --synthetic" in res.stdout

    res_resume = runner.invoke(
        app,
        ["data", "collect", "resume", "--job-id", "job_dummy", "--live", "--synthetic"],
    )
    assert res_resume.exit_code == 1
    assert "Cannot specify both --live and --synthetic" in res_resume.stdout


def test_cli_fails_closed_when_live_account_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_vault_with_two_accounts: tuple[WindowsCredentialVault, str, str],
) -> None:
    """CLI fails closed when --live is specified with an unknown --account-id."""
    monkeypatch.chdir(tmp_path)
    vault, _, _ = mock_vault_with_two_accounts
    monkeypatch.setattr("iqoption_research.auth.vault.WindowsCredentialVault", lambda: vault)
    # First plan a dummy job
    res_plan = runner.invoke(
        app,
        [
            "data",
            "collect",
            "plan",
            "--symbol",
            "EURUSD",
            "--from",
            "2026-11-04T10:00:00+00:00",
            "--to",
            "2026-11-04T10:10:00+00:00",
            "--json",
        ],
    )
    assert res_plan.exit_code == 0
    job_id = json.loads(res_plan.stdout)["job_id"]

    res_run = runner.invoke(
        app,
        [
            "data",
            "collect",
            "run",
            "--job-id",
            job_id,
            "--live",
            "--account-id",
            "non_existent_account_id",
            "--json",
        ],
    )
    assert res_run.exit_code == 1
    out_json = json.loads(res_run.stdout)
    assert out_json["status"] == "FAILED"
    assert "is unavailable" in out_json["error"]


@pytest.mark.asyncio
async def test_interactive_login_binds_explicit_account_id() -> None:
    """Explicit interactive selection also binds the account ID into AuthSession."""
    mock_provider = MagicMock()
    mock_provider.login = AsyncMock(
        return_value=AuthResult(success=True, ssid="fake_ssid_interactive_777")
    )
    mock_provider.get_profile_and_balances = AsyncMock(return_value=(None, []))
    auth = AuthenticationManager(provider=mock_provider)

    ok = await auth.login_interactive(
        email="trader2@example.com",
        password="Password456!",
        account_id="22222222222222222222222222222222",
    )

    assert ok is True
    assert auth.session is not None
    assert auth.session.account_id == "22222222222222222222222222222222"


def test_cli_live_collect_requires_explicit_account_id_before_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live collect and resume fail closed before the connection seam is called."""
    monkeypatch.chdir(tmp_path)
    connect_calls: list[object] = []

    async def forbidden_connect(*_args: object, **_kwargs: object) -> None:
        connect_calls.append(True)
        raise AssertionError("no broker must be constructed")

    monkeypatch.setattr(backfill_cmd, "_connect_live_client", forbidden_connect)
    res_plan = runner.invoke(
        app,
        [
            "data",
            "collect",
            "plan",
            "--symbol",
            "EURUSD",
            "--from",
            "2026-11-04T10:00:00+00:00",
            "--to",
            "2026-11-04T10:10:00+00:00",
            "--json",
        ],
    )
    assert res_plan.exit_code == 0
    job_id = json.loads(res_plan.stdout)["job_id"]

    for command in ("run", "resume"):
        res = runner.invoke(
            app,
            ["data", "collect", command, "--job-id", job_id, "--live", "--json"],
        )
        assert res.exit_code == 1
        payload = json.loads(res.stdout)
        assert payload["status"] == "FAILED"
        assert "Explicit --account-id" in payload["error"]

    assert connect_calls == []
