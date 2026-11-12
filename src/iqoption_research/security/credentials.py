"""Secure credential handling — input, redaction, and non-persistence.

This module provides utilities for securely obtaining credentials from
the user without writing them to disk, committing them to Git, logging them,
or including them in exception messages.

Note on Process Memory:
    While object references are discarded to permit standard garbage collection,
    standard CPython does not guarantee immediate zeroization of immutable string
    memory buffers. Avoid untrusted multi-tenant host processes or unauthorized
    process memory dumps.
"""

import getpass
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Final

# Patterns that match sensitive data for redaction
SECRET_PATTERNS: Final[list[re.Pattern]] = [
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(ssid|session_id|token)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(authorization|bearer)\s+[\w\-._~+/]+=*"),
    re.compile(r"(?i)(cookie|set-cookie)\s*[:=]\s*[^;]+"),
    re.compile(r"(?i)(email|identifier)\s*[:=]\s*\S+@\S+"),
    # SSID cookie value (alphanumeric, typically 32+ chars)
    re.compile(r"\b[a-f0-9]{32,}\b"),
]

REDACTION_REPLACEMENT: Final[str] = "***REDACTED***"


@dataclass(frozen=True, slots=True)
class Credentials:
    """Immutable credential container.

    Instances should be created via `get_credentials()` and discarded
    after use. Never serialize, log, or persist instances.
    """

    email: str
    password: str

    def __repr__(self) -> str:
        return f"Credentials(email={self._redact_email(self.email)!r}, password=***REDACTED***)"

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def _redact_email(email: str) -> str:
        """Partially redact email for display."""
        return redact_email(email)


def redact_email(email: str) -> str:
    """Partially redact an email address for safe display."""
    if "@" not in email:
        return REDACTION_REPLACEMENT
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        return f"{REDACTION_REPLACEMENT}@{domain}"
    return f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}@{domain}"


def get_credentials(
    email: str | None = None,
    password: str | None = None,
    *,
    allow_env_fallback: bool = False,
) -> Credentials:
    """Securely obtain credentials from user.

    Priority:
    1. Explicit arguments (for testing only)
    2. Interactive `getpass` prompts (production)
    3. Environment variables (only if `allow_env_fallback=True`, NOT for production)

    Args:
        email: Pre-supplied email (testing only).
        password: Pre-supplied password (testing only).
        allow_env_fallback: Allow reading from IQ_EMAIL/IQ_PASSWORD env vars.
            MUST be False in production. Only for CI/testing.

    Returns:
        Credentials instance.

    Raises:
        ValueError: If credentials cannot be obtained.
        RuntimeError: If running non-interactively without credentials.
    """
    # 1. Explicit arguments (highest priority, testing only)
    if email is not None and password is not None:
        return Credentials(email=email, password=password)

    # 2. Environment fallback (explicitly opted in, NOT for production)
    if allow_env_fallback:
        env_email = os.environ.get("IQ_EMAIL")
        env_password = os.environ.get("IQ_PASSWORD")
        if env_email and env_password:
            return Credentials(email=env_email, password=env_password)

    # 3. Interactive prompts (production path)
    if email is None:
        try:
            email = input("IQ Option email: ").strip()
        except EOFError as err:
            raise RuntimeError(
                "Non-interactive terminal; provide credentials via arguments or allow_env_fallback"
            ) from err

    if password is None:
        try:
            password = getpass.getpass("IQ Option password: ")
        except EOFError as err:
            raise RuntimeError(
                "Non-interactive terminal; provide credentials via arguments or allow_env_fallback"
            ) from err

    if email and password:
        return Credentials(email=email, password=password)

    raise ValueError("Unable to obtain credentials")


def redact_secrets(text: str) -> str:
    """Redact sensitive patterns from a string.

    Args:
        text: Input text that may contain secrets.

    Returns:
        Text with secrets replaced by REDACTION_REPLACEMENT.
    """
    result = text
    for pattern in SECRET_PATTERNS:
        result = pattern.sub(REDACTION_REPLACEMENT, result)
    return result


def redact_dict(data: dict[str, Any], keys_to_redact: set[str] | None = None) -> dict[str, Any]:
    """Return a copy of dict with sensitive keys redacted.

    Args:
        data: Dictionary to redact.
        keys_to_redact: Additional keys to redact (case-insensitive).

    Returns:
        New dictionary with sensitive values replaced.
    """
    default_keys = {
        "password",
        "passwd",
        "pwd",
        "ssid",
        "session_id",
        "token",
        "authorization",
        "cookie",
        "set-cookie",
        "email",
        "identifier",
        "secret",
        "key",
        "auth",
        "credential",
    }
    if keys_to_redact:
        default_keys.update(k.lower() for k in keys_to_redact)

    result: dict[str, Any] = {}
    for k, v in data.items():
        if k.lower() in default_keys:
            result[k] = REDACTION_REPLACEMENT
        elif isinstance(v, dict):
            result[k] = redact_dict(v, keys_to_redact)
        elif isinstance(v, list):
            new_list: list[Any] = []
            for item in v:
                if isinstance(item, dict):
                    new_list.append(redact_dict(item, keys_to_redact))
                elif isinstance(item, str) and any(p.search(item) for p in SECRET_PATTERNS):
                    new_list.append(REDACTION_REPLACEMENT)
                else:
                    new_list.append(item)
            result[k] = new_list
        elif isinstance(v, str) and any(p.search(v) for p in SECRET_PATTERNS):
            result[k] = redact_secrets(v)
        else:
            result[k] = v
    return result


class SecretRedactingFilter(logging.Filter):
    """Logging filter that redacts secrets from log records."""

    def __init__(self, keys_to_redact: set[str] | None = None):
        super().__init__()
        self._keys = keys_to_redact or set()

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact message
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)

        # Redact args
        if hasattr(record, "args") and record.args:
            new_args: list[Any] = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_args.append(redact_secrets(arg))
                elif isinstance(arg, dict):
                    new_args.append(redact_dict(arg, self._keys))
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)

        return True
