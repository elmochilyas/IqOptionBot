"""Secure multi-account operations, using an in-memory backend only."""

import json
import logging
from unittest.mock import MagicMock

import pytest

from iqoption_research.auth.accounts import INDEX_KEY, AccountStoreError
from iqoption_research.auth.vault import (
    ACTIVE_ACCOUNT_KEY,
    InsecureKeyringBackendError,
    WindowsCredentialVault,
)
from iqoption_research.desktop.workstation import AccountSelection
from iqoption_research.timing import supervised_session


class WinVaultKeyring:
    __module__ = "keyring.backends.Windows"

    def __init__(self):
        self.data = {}
        self.reads = []

    def get_password(self, service, key):
        self.reads.append(key)
        return self.data.get((service, key))

    def set_password(self, service, key, value):
        self.data[service, key] = value

    def delete_password(self, service, key):
        self.data.pop((service, key))


@pytest.fixture
def vault():
    backend = WinVaultKeyring()
    return WindowsCredentialVault(keyring_backend=backend), backend


def test_legacy_migration_reads_no_password_and_is_stable(vault):
    secure, backend = vault
    secure.save_credentials("first@example.test", "fake-secret", explicit_opt_in=True)
    backend.reads.clear()
    store = secure.account_store()
    first = store.list_accounts()[0]
    assert "first@example.test" not in backend.reads
    assert first.account_id == secure.account_store().list_accounts()[0].account_id
    assert "first@example.test" not in first.label
    assert store.remembered_account() is None
    assert secure.load_saved_credentials().password == "fake-secret"
    assert store.load(first.account_id).password == "fake-secret"


def test_two_accounts_selection_remember_update_forget(vault):
    secure, backend = vault
    store = secure.account_store()
    a = store.save("a@example.test", "secret-a", explicit_opt_in=True)
    b = store.save("b@example.test", "secret-b", explicit_opt_in=True)
    assert a != b and len(store.list_accounts()) == 2
    selection = AccountSelection({x.account_id: x.label for x in store.list_accounts()})
    selection.select(a, busy=False)
    selection.select(b, busy=False)
    assert selection.selected == b
    store.remember(b)
    assert secure.account_store().remembered_account() == b
    store.save("b@example.test", "new-secret-b", explicit_opt_in=True, account_id=b)
    assert store.load(a).password == "secret-a"
    assert store.load(b).password == "new-secret-b"
    assert "new-secret-b" not in backend.data[secure.service_name, INDEX_KEY]
    store.forget(b)
    assert store.remembered_account() is None
    assert len(store.list_accounts()) == 1
    with pytest.raises(AccountStoreError):
        store.load(b)
    assert not any(value == "new-secret-b" for value in backend.data.values())


@pytest.mark.parametrize("remember", [True, False])
def test_remember_identity_only_and_each_session_selection(vault, remember):
    secure, _ = vault
    store = secure.account_store()
    aid = store.save("a@example.test", "secret-a", explicit_opt_in=True)
    store.remember(aid if remember else None)
    selection = AccountSelection({aid: "masked"}, aid, remember)
    selection.finished()
    assert selection.selected == (aid if remember else None)
    assert store.remembered_account() == (aid if remember else None)


def test_selection_locked_during_live_or_pending_session():
    selection = AccountSelection({"a": "masked", "b": "masked"}, "a")
    with pytest.raises(ValueError):
        selection.select("b", busy=True)
    assert selection.selected == "a"
    with pytest.raises(ValueError):
        selection.select("missing", busy=False)


def test_save_requires_consent_and_rejects_unknown_update(vault):
    store = vault[0].account_store()
    with pytest.raises(AccountStoreError):
        store.save("a@example.test", "secret", explicit_opt_in=False)
    with pytest.raises(AccountStoreError):
        store.save("a@example.test", "secret", explicit_opt_in=True, account_id="missing")
    assert store.list_accounts() == []


@pytest.mark.parametrize(
    "payload",
    ["not-json", '{"version": 9}', '{"version": 2, "accounts": [], "remembered_id": "missing"}'],
)
def test_corrupt_or_future_index_fails_closed(vault, payload):
    secure, backend = vault
    backend.set_password(secure.service_name, INDEX_KEY, payload)
    with pytest.raises(AccountStoreError):
        secure.account_store().list_accounts()


def test_insecure_backend_refused():
    with pytest.raises(InsecureKeyringBackendError):
        WindowsCredentialVault(keyring_backend=MagicMock()).account_store()


async def test_missing_selected_account_never_falls_back(monkeypatch, vault):
    secure, backend = vault
    aid = secure.account_store().save("a@example.test", "secret", explicit_opt_in=True)
    backend.delete_password(secure.service_name, "account:" + aid)
    monkeypatch.setattr("iqoption_research.auth.vault.WindowsCredentialVault", lambda: secure)
    prompt, client = MagicMock(), MagicMock()
    monkeypatch.setattr(supervised_session.getpass, "getpass", prompt)
    monkeypatch.setattr(supervised_session, "AsyncIQOption", client)
    with pytest.raises(AccountStoreError):
        await supervised_session.supervised_client(MagicMock(), account_id=aid)
    prompt.assert_not_called()
    client.assert_not_called()


def test_backend_exception_cannot_expose_password(vault, monkeypatch, caplog):
    secure, backend = vault
    store = secure.account_store()
    store.list_accounts()
    monkeypatch.setattr(
        backend, "set_password", MagicMock(side_effect=RuntimeError("secret-value"))
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(AccountStoreError) as error:
        store.save("a@example.test", "secret-value", explicit_opt_in=True)
    assert "secret-value" not in str(error.value) + caplog.text


def test_migrated_identity_replace_removes_old_secret(vault):
    secure, backend = vault
    secure.save_credentials("old@example.test", "old-secret", explicit_opt_in=True)
    store = secure.account_store()
    aid = store.list_accounts()[0].account_id
    store.save("new@example.test", "new-secret", explicit_opt_in=True, account_id=aid)
    assert store.load(aid).email == "new@example.test"
    assert (secure.service_name, "old@example.test") not in backend.data
    assert (secure.service_name, ACTIVE_ACCOUNT_KEY) not in backend.data


def test_no_secret_in_index_and_duplicate_identity_retains_id(vault):
    secure, backend = vault
    store = secure.account_store()
    aid = store.save("a@example.test", "do-not-log", explicit_opt_in=True)
    assert store.save("a@example.test", "replacement", explicit_opt_in=True) == aid
    payload = backend.data[secure.service_name, INDEX_KEY]
    assert "replacement" not in payload and "do-not-log" not in payload
    assert json.loads(payload)["version"] == 2
