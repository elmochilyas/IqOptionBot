"""TLS configuration and certificate verification enforcement.

This module ensures that ALL HTTPS and WSS connections use strict
certificate verification. There is NO way to disable verification
through configuration or code paths.
"""

import ssl
from typing import Final

import certifi

# Allowlisted IQ Option endpoints — hardcoded, not configurable
ALLOWED_HTTPS_ENDPOINTS: Final[frozenset[str]] = frozenset({
    "https://auth.iqoption.com/api/v2/login",
    "https://auth.iqoption.com/api/v1.0/logout",
})

ALLOWED_WSS_ENDPOINTS: Final[frozenset[str]] = frozenset({
    "wss://iqoption.com/echo/websocket",
})

# IQ Option's expected certificate pinning info (subject CN)
EXPECTED_CERT_CN: Final[str] = "iqoption.com"


def create_tls_context() -> ssl.SSLContext:
    """Create a strict TLS context for HTTPS/WSS connections.

    - Uses system trust store + certifi bundle
    - Requires valid certificate chain
    - Verifies hostname (CN/SAN matching)
    - No fallback to insecure verification
    - TLS 1.2 minimum (Python default in 3.12+)

    Returns:
        Configured SSLContext with strict verification.

    Raises:
        ssl.SSLError: If system trust store cannot be loaded.
    """
    # Use certifi's curated CA bundle + system store
    context = ssl.create_default_context(cafile=certifi.where())

    # Enforce verification — these are the Python 3.12+ defaults,
    # but we set them explicitly for clarity and auditability.
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED

    # Minimum TLS 1.2 (Python 3.12 default, but explicit)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_3

    # Disable weak cipher suites
    context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20")

    return context


def validate_https_endpoint(url: str) -> None:
    """Validate that an HTTPS endpoint is in the allowlist.

    Args:
        url: The full HTTPS URL to validate.

    Raises:
        ValueError: If URL is not in the allowlist.
    """
    if url not in ALLOWED_HTTPS_ENDPOINTS:
        allowed = ", ".join(sorted(ALLOWED_HTTPS_ENDPOINTS))
        raise ValueError(
            f"HTTPS endpoint not allowed: {url}. Allowed: {allowed}"
        )


def validate_wss_endpoint(url: str) -> None:
    """Validate that a WSS endpoint is in the allowlist.

    Args:
        url: The full WSS URL to validate.

    Raises:
        ValueError: If URL is not in the allowlist.
    """
    if url not in ALLOWED_WSS_ENDPOINTS:
        allowed = ", ".join(sorted(ALLOWED_WSS_ENDPOINTS))
        raise ValueError(
            f"WSS endpoint not allowed: {url}. Allowed: {allowed}"
        )


def verify_certificate_cn(cert: dict | None, expected_cn: str = EXPECTED_CERT_CN) -> bool:
    """Verify certificate Common Name matches expected.

    Args:
        cert: Peer certificate from SSL socket.
        expected_cn: Expected Common Name.

    Returns:
        True if CN matches, False otherwise.
    """
    if cert is None:
        return False

    # Check Subject CN
    subject = cert.get("subject", ())
    for field in subject:
        if field[0][0] == "commonName" and field[0][1].lower() == expected_cn.lower():
            return True

    # Check Subject Alternative Names
    san = cert.get("subjectAltName", ())
    return any(typ == "DNS" and value.lower() == expected_cn.lower() for typ, value in san)
