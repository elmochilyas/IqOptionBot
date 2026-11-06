"""Secure HTTP authentication against IQ Option's auth endpoint.

Uses aiohttp with strict TLS verification. No credential logging.
Explicitly rejects 2FA challenges (not supported in read-only MVP).
"""

import logging
from typing import Any

import aiohttp
from aiohttp import ClientTimeout

from iqoption_research.connectors.iqoption.exceptions import (
    ConnectionError,
    LoginError,
    RequestTimeoutError,
    TwoFactorRequiredError,
)
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.security import (
    create_tls_context,
    redact_dict,
    validate_https_endpoint,
)
from iqoption_research.security.credentials import Credentials

logger = logging.getLogger(__name__)

# IQ Option login endpoint (allowlisted)
DEFAULT_LOGIN_URL = "https://auth.iqoption.com/api/v2/login"
DEFAULT_LOGOUT_URL = "https://auth.iqoption.com/api/v1.0/logout"

# User-Agent matching IQ Option's web client
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


async def async_login(
    credentials: Credentials,
    *,
    login_url: str = DEFAULT_LOGIN_URL,
    config: ConnectionConfig | None = None,
    session: aiohttp.ClientSession | None = None,
) -> str:
    """Authenticate via HTTPS and return the SSID cookie value.

    Args:
        credentials: Email and password (never logged).
        login_url: Must be allowlisted (validated).
        config: Connection timeouts and settings.
        session: Optional existing session (caller manages lifecycle).

    Returns:
        SSID cookie value for WebSocket authentication.

    Raises:
        LoginError: On authentication failure, missing SSID, or 2FA challenge.
        ConnectionError: On network/transport failure.
        RequestTimeoutError: If request exceeds timeout.
    """
    config = config or ConnectionConfig()
    validate_https_endpoint(login_url)

    # Create TLS context with strict verification
    ssl_context = create_tls_context()

    # Prepare timeout
    timeout = ClientTimeout(total=config.connect_timeout)

    # Use provided session or create temporary one
    own_session = session is None
    if session is None:
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        session = aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=timeout,
        )

    try:
        # POST credentials
        logger.info("Initiating HTTPS login to IQ Option")
        try:
            response = await session.post(
                login_url,
                data={"identifier": credentials.email, "password": credentials.password},
                ssl=ssl_context,
                allow_redirects=False,
            )
        except aiohttp.ClientError as exc:
            raise ConnectionError(f"HTTPS login request failed: {exc}") from exc
        except TimeoutError as exc:
            raise RequestTimeoutError(
                f"Login request timed out after {config.connect_timeout}s"
            ) from exc

        async with response:
            # Reject any HTTP redirect (3xx) immediately without following
            if 300 <= response.status < 400:
                raise LoginError(
                    f"Authentication redirect rejected (HTTP {response.status}). "
                    "HTTP redirects are strictly disabled for credential-bearing authentication."
                )

            # Parse response (may not be JSON)
            payload: dict[str, Any] | None
            try:
                payload = await response.json(content_type=None)
            except (aiohttp.ContentTypeError, ValueError):
                payload = None

            # Log sanitized response only; never log raw auth payloads.
            logger.debug(
                "Login response: status=%s, payload=%s",
                response.status,
                redact_dict(payload or {}),
            )

            if response.status != 200:
                raise LoginError(f"Login failed: HTTP {response.status}")

            # Extract SSID cookie
            ssid_cookie = response.cookies.get("ssid")
            if ssid_cookie is None:
                # Check for 2FA challenge
                if isinstance(payload, dict) and payload.get("code") == "verify":
                    raise TwoFactorRequiredError(
                        "Two-factor authentication (2FA) challenge required by broker. "
                        "Phase 1 read-only connectivity does not implement interactive "
                        "2FA challenge resolution. Authentication halted safely."
                    )
                raise LoginError("Login succeeded but no SSID cookie was returned")

            ssid = ssid_cookie.value
            logger.info("HTTPS login successful, SSID obtained")
            return ssid

    finally:
        if own_session:
            await session.close()


async def async_logout(
    ssid: str,
    *,
    logout_url: str = DEFAULT_LOGOUT_URL,
    config: ConnectionConfig | None = None,
    session: aiohttp.ClientSession | None = None,
) -> bool:
    """Logout and invalidate SSID on server.

    Best-effort cleanup; failures are logged but not raised.

    Args:
        ssid: Session ID to invalidate.
        logout_url: Must be allowlisted.
        config: Connection timeouts.
        session: Optional existing session.

    Returns:
        True if logout request succeeded, False otherwise.
    """
    config = config or ConnectionConfig()
    validate_https_endpoint(logout_url)

    ssl_context = create_tls_context()
    timeout = ClientTimeout(total=config.connect_timeout)

    own_session = session is None
    if session is None:
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        session = aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": DEFAULT_USER_AGENT, "Cookie": f"ssid={ssid}"},
            timeout=timeout,
        )

    try:
        logger.info("Sending logout request")
        try:
            response = await session.post(
                logout_url,
                ssl=ssl_context,
                allow_redirects=False,
            )
        except aiohttp.ClientError as exc:
            logger.warning("Logout request failed: %s", exc)
            return False
        except TimeoutError:
            logger.warning("Logout request timed out")
            return False

        async with response:
            logger.debug("Logout response: status=%s", response.status)
            if 300 <= response.status < 400:
                logger.warning("Logout redirect rejected: HTTP %s", response.status)
                return False
            return response.status == 200

    finally:
        if own_session:
            await session.close()
