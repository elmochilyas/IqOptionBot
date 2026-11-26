"""Tests for TLS and endpoint policy enforcement."""

import ssl

import pytest

from iqoption_research.cli.commands.security_check import check_no_insecure_tls_patterns
from iqoption_research.security.endpoints import (
    EndpointConfig,
    validate_endpoint_allowlist,
)
from iqoption_research.security.tls import (
    ALLOWED_HTTPS_ENDPOINTS,
    ALLOWED_WSS_ENDPOINTS,
    create_tls_context,
    validate_https_endpoint,
    validate_wss_endpoint,
    verify_certificate_cn,
)


def test_tls_context_strict_configuration():
    ctx = create_tls_context()
    assert ctx.check_hostname is True
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2


def test_endpoint_allowlist_valid():
    for url in ALLOWED_HTTPS_ENDPOINTS:
        validate_https_endpoint(url)  # raises if invalid
    for url in ALLOWED_WSS_ENDPOINTS:
        validate_wss_endpoint(url)  # raises if invalid


def test_endpoint_allowlist_rejects_arbitrary_urls():
    bad_urls = [
        "https://evil-iqoption.com/api/v2/login",
        "https://iqoption.attacker.com/api/v2/login",
        "http://auth.iqoption.com/api/v2/login",
        "https://auth.iqoption.com/evil/path",
        "wss://evil.com/echo/websocket",
        "ws://iqoption.com/echo/websocket",
        "wss://iqoption.com/unknown",
    ]
    for bad in bad_urls:
        if bad.startswith("https") or bad.startswith("http"):
            with pytest.raises(ValueError, match="not allowed"):
                validate_https_endpoint(bad)
        else:
            with pytest.raises(ValueError, match="not allowed"):
                validate_wss_endpoint(bad)


def test_endpoint_config_model():
    endpoints = EndpointConfig.default()
    assert endpoints.login_url in ALLOWED_HTTPS_ENDPOINTS
    assert endpoints.logout_url in ALLOWED_HTTPS_ENDPOINTS
    assert endpoints.wss_url in ALLOWED_WSS_ENDPOINTS
    validate_endpoint_allowlist()

    with pytest.raises(ValueError, match="not allowed"):
        EndpointConfig(
            login_url="https://malicious.org",
            logout_url=endpoints.logout_url,
            wss_url=endpoints.wss_url,
        )


def test_certificate_cn_verification():
    mock_cert_valid = {
        "subject": ((("commonName", "iqoption.com"),),),
        "subjectAltName": (("DNS", "iqoption.com"), ("DNS", "*.iqoption.com")),
    }
    assert verify_certificate_cn(mock_cert_valid, "iqoption.com") is True
    assert verify_certificate_cn(mock_cert_valid, "other.com") is False
    assert verify_certificate_cn(None, "iqoption.com") is False


def test_no_insecure_tls_patterns_in_codebase():
    result = check_no_insecure_tls_patterns()
    assert result.passed is True, result.detail


def test_async_iqoption_rejects_unallowlisted_endpoints_at_init(mock_credentials):
    from iqoption_research.connectors.iqoption.client import AsyncIQOption

    with pytest.raises(ValueError, match="not allowed"):
        AsyncIQOption(mock_credentials, login_url="https://attacker.com/login")

    with pytest.raises(ValueError, match="not allowed"):
        AsyncIQOption(mock_credentials, wss_url="wss://attacker.com/ws")
