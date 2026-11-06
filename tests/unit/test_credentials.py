"""Tests for credentials handling, redaction, and logging security."""

import logging
from unittest.mock import patch

import pytest

from iqoption_research.security.credentials import (
    Credentials,
    get_credentials,
    redact_dict,
    redact_email,
    redact_secrets,
)
from iqoption_research.security.logging import (
    SanitizedFormatter,
    SanitizedLoggerAdapter,
    configure_logging,
    get_logger,
)


def test_credentials_immutability_and_repr():
    creds = Credentials(email="trader@example.com", password="SuperSecretPassword123!")
    assert creds.email == "trader@example.com"
    assert creds.password == "SuperSecretPassword123!"

    # Secret password must NOT appear in repr or str
    assert "SuperSecretPassword123!" not in repr(creds)
    assert "SuperSecretPassword123!" not in str(creds)
    assert "***REDACTED***" in repr(creds)

    with pytest.raises(Exception):
        creds.password = "new_pass"  # type: ignore[misc]


def test_get_credentials_explicit():
    creds = get_credentials(email="user@test.com", password="pass")
    assert creds.email == "user@test.com"
    assert creds.password == "pass"


def test_get_credentials_prompt():
    with patch("builtins.input", return_value="prompted@test.com"), \
         patch("getpass.getpass", return_value="prompted_pass"):
        creds = get_credentials()
        assert creds.email == "prompted@test.com"
        assert creds.password == "prompted_pass"


def test_get_credentials_non_interactive_raises():
    with patch("builtins.input", side_effect=EOFError), \
         pytest.raises(RuntimeError, match="Non-interactive terminal"):
        get_credentials()


def test_get_credentials_env_fallback():
    # Without allow_env_fallback=True, env vars are ignored and it attempts prompt
    with (
        patch.dict("os.environ", {"IQ_EMAIL": "env@test.com", "IQ_PASSWORD": "env_password"}),
        patch("builtins.input", side_effect=EOFError),
        pytest.raises(RuntimeError, match="Non-interactive terminal"),
    ):
        get_credentials(allow_env_fallback=False)

    with patch.dict("os.environ", {"IQ_EMAIL": "env@test.com", "IQ_PASSWORD": "env_password"}):
        creds = get_credentials(allow_env_fallback=True)
        assert creds.email == "env@test.com"
        assert creds.password == "env_password"


def test_redact_secrets_patterns():
    raw_text = "Logging in with password=MySecretPass and token: 0123456789abcdef0123456789abcdef"
    redacted = redact_secrets(raw_text)
    assert "MySecretPass" not in redacted
    assert "0123456789abcdef0123456789abcdef" not in redacted
    assert "***REDACTED***" in redacted


def test_redact_email():
    assert redact_email("john.doe@example.com") == "j******e@example.com"
    assert redact_email("a@b.com") == "***REDACTED***@b.com"
    assert redact_email("invalid") == "***REDACTED***"


def test_redact_dict_recursive():
    data = {
        "user": {
            "email": "user@example.com",
            "password": "secret_password",
            "nested": {
                "ssid": "abcdef0123456789abcdef0123456789",
                "normal": "safe_value",
            },
        },
        "items": [
            {"token": "token123", "value": 10},
            "regular string",
            "password=inlined_secret",
        ],
    }
    redacted = redact_dict(data)
    assert redacted["user"]["password"] == "***REDACTED***"
    assert redacted["user"]["email"] == "***REDACTED***"
    assert redacted["user"]["nested"]["ssid"] == "***REDACTED***"
    assert redacted["user"]["nested"]["normal"] == "safe_value"
    assert redacted["items"][0]["token"] == "***REDACTED***"
    assert redacted["items"][0]["value"] == 10
    assert redacted["items"][1] == "regular string"
    assert "inlined_secret" not in redacted["items"][2]


def test_sanitized_logging_filter_and_formatter():
    configure_logging(level=logging.DEBUG)
    logger = get_logger("test")

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="User login with password=SuperSecret! payload=%s",
        args=({"ssid": "sensitive_session"},),
        exc_info=None,
    )
    formatter = SanitizedFormatter()
    formatted = formatter.format(record)
    assert "SuperSecret!" not in formatted
    assert "sensitive_session" not in formatted

    adapter = SanitizedLoggerAdapter(logger, {})
    processed_msg, _ = adapter.process("password=my_secret_key", {})
    assert "my_secret_key" not in processed_msg


def test_profile_and_account_status_repr_redacts_sensitive_info():
    from iqoption_research.models.account import AccountStatus, Profile

    raw_payload = {
        "user_id": 9999,
        "email": "confidential_trader@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "secret_profile_token": "token_should_not_leak",
    }
    profile = Profile.from_raw(raw_payload)
    profile_repr = repr(profile)
    # Email must be partially masked
    assert "confidential_trader@example.com" not in profile_repr
    assert "c*****************r@example.com" in profile_repr
    # Raw dictionary must not be dumped in repr
    assert "token_should_not_leak" not in profile_repr

    status = AccountStatus.from_raw(
        profile_data=raw_payload,
        balances_data=[],
    )
    status_repr = repr(status)
    assert "confidential_trader@example.com" not in status_repr
    assert "c*****************r@example.com" in status_repr
