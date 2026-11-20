"""Versioned multi-account metadata stored inside the verified Windows vault only."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from iqoption_research.security.credentials import Credentials, redact_email

INDEX_KEY = "__accounts_v2__"
LEGACY_KEY = "__active_account__"


@dataclass(frozen=True)
class SavedAccount:
    account_id: str
    label: str


class AccountStoreError(ValueError):
    """Fixed, non-sensitive account-store failure."""


class AccountStore:
    """Created by WindowsCredentialVault only after native backend verification.

    Listing/migration reads identity metadata, never account passwords. Entries are
    written before index publication; failures can leave an inaccessible vault entry,
    but cannot select a half-published identity. No broker login occurs here.
    """

    def __init__(self, backend: Any, service: str) -> None:
        self.backend = backend
        self.service = service

    def _get(self, key: str) -> str | None:
        try:
            value = self.backend.get_password(self.service, key)
        except Exception:  # noqa: BLE001 - backend errors may contain secrets
            raise AccountStoreError("Secure vault read failed") from None
        return str(value) if value is not None else None

    def _set(self, key: str, value: str) -> None:
        try:
            self.backend.set_password(self.service, key, value)
        except Exception:  # noqa: BLE001 - never expose backend exception payloads
            raise AccountStoreError("Secure vault write failed") from None

    def _delete(self, key: str) -> None:
        try:
            self.backend.delete_password(self.service, key)
        except Exception:  # noqa: BLE001
            raise AccountStoreError("Secure vault deletion failed") from None

    def _read(self) -> dict[str, Any]:
        raw = self._get(INDEX_KEY)
        if raw is None:
            index: dict[str, Any] = {"version": 2, "accounts": [], "remembered_id": None}
            legacy = self._get(LEGACY_KEY)
            if legacy:
                index["accounts"].append({
                    "id": uuid.uuid4().hex,
                    "email": legacy,
                    "credential_key": legacy,
                })
            self._write(index)
            return index
        try:
            index = json.loads(raw)
            if index["version"] != 2 or not isinstance(index["accounts"], list):
                raise ValueError
            ids: set[str] = set()
            emails: set[str] = set()
            for entry in index["accounts"]:
                aid, email, key = entry["id"], entry["email"], entry["credential_key"]
                if uuid.UUID(hex=aid).hex != aid or aid in ids:
                    raise ValueError
                if not isinstance(email, str) or not email.strip() or email.casefold() in emails:
                    raise ValueError
                if key not in (email, f"account:{aid}"):
                    raise ValueError
                ids.add(aid)
                emails.add(email.casefold())
            if index["remembered_id"] is not None and index["remembered_id"] not in ids:
                raise ValueError
        except (KeyError, ValueError, TypeError, AttributeError):
            raise AccountStoreError(
                "Account index is invalid or unsupported; access refused"
            ) from None
        return index

    def _write(self, index: dict[str, Any]) -> None:
        self._set(INDEX_KEY, json.dumps(index, sort_keys=True))

    def list_accounts(self) -> list[SavedAccount]:
        return [SavedAccount(e["id"], redact_email(e["email"])) for e in self._read()["accounts"]]

    def remembered_account(self) -> str | None:
        value = self._read()["remembered_id"]
        return str(value) if value is not None else None

    def remember(self, account_id: str | None) -> None:
        index = self._read()
        if account_id is not None and not any(e["id"] == account_id for e in index["accounts"]):
            raise AccountStoreError("Selected account is unavailable")
        index["remembered_id"] = account_id
        self._write(index)

    def save(
        self,
        email: str,
        password: str,
        *,
        explicit_opt_in: bool,
        account_id: str | None = None,
    ) -> str:
        if not explicit_opt_in:
            raise AccountStoreError("Explicit consent is required to save credentials")
        email = email.strip()
        if not email or not password or len(email) > 254 or "@" not in email:
            raise AccountStoreError("A valid account identifier and password are required")
        index = self._read()
        entries = index["accounts"]
        selected = next((e for e in entries if e["id"] == account_id), None)
        duplicate = next((e for e in entries if e["email"].casefold() == email.casefold()), None)
        if account_id is not None and selected is None:
            raise AccountStoreError("Selected account is unavailable")
        if selected is not None and duplicate is not None and selected is not duplicate:
            raise AccountStoreError("That identity already belongs to another saved account")
        entry = selected or duplicate
        if entry is None:
            entry = {"id": uuid.uuid4().hex}
            entries.append(entry)
        key = entry.get("credential_key", f"account:{entry['id']}")
        # Keep legacy keys for the same identity; new identities use opaque keys.
        old_key, old_email = entry.get("credential_key"), entry.get("email")
        if entry.get("email", email) != email:
            key = f"account:{entry['id']}"
        self._set(key, password)
        entry.update(email=email, credential_key=key)
        self._write(index)
        if old_key is not None and old_key != key:
            self._delete(old_key)
            if self._get(LEGACY_KEY) == old_email:
                self._delete(LEGACY_KEY)
        return str(entry["id"])

    def load(self, account_id: str) -> Credentials:
        entry = next((e for e in self._read()["accounts"] if e["id"] == account_id), None)
        if entry is None:
            raise AccountStoreError("Selected account is unavailable")
        password = self._get(entry["credential_key"])
        if not password:
            raise AccountStoreError("Selected account has no saved credentials; update it first")
        return Credentials(email=entry["email"], password=password)

    def forget(self, account_id: str) -> None:
        index = self._read()
        entry = next((e for e in index["accounts"] if e["id"] == account_id), None)
        if entry is None:
            raise AccountStoreError("Selected account is unavailable")
        # Delete secret first; a failed index write leaves an unusable entry, never a usable ghost.
        if self._get(entry["credential_key"]) is not None:
            self._delete(entry["credential_key"])
        if self._get(LEGACY_KEY) == entry["email"]:
            self._delete(LEGACY_KEY)
        index["accounts"].remove(entry)
        if index["remembered_id"] == account_id:
            index["remembered_id"] = None
        self._write(index)
