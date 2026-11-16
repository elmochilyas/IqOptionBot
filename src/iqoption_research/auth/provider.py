"""Authentication provider interfaces and implementations.

Decouples authentication mechanics from bot execution logic.
Provides:
- AuthProvider: Abstract base interface
- MockAuthProvider: Deterministic offline fixture provider for development isolation
- IQOptionAuthProvider: Production-grade HTTPS broker connector adhering to strict TLS
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import ClientTimeout

from iqoption_research.auth.mfa import MFAChallenge, MFAMethod
from iqoption_research.connectors.iqoption.exceptions import (
    ConnectionError,
    LoginError,
    RequestTimeoutError,
)
from iqoption_research.connectors.iqoption.http_auth import (
    DEFAULT_LOGIN_URL,
    DEFAULT_LOGOUT_URL,
    DEFAULT_USER_AGENT,
    async_logout,
)
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.security import create_tls_context, redact_dict, validate_https_endpoint
from iqoption_research.security.credentials import Credentials

logger = logging.getLogger(__name__)

DEFAULT_2FA_VERIFY_URL = "https://auth.iqoption.com/api/v2/verify/2fa"


@dataclass(frozen=True, slots=True)
class AuthResult:
    """Result of an authentication or verification attempt."""

    success: bool
    ssid: str | None = None
    mfa_challenge: MFAChallenge | None = None
    error_message: str | None = None
    status_code: int | None = None

    def __repr__(self) -> str:
        ssid_display = "***REDACTED***" if self.ssid else "None"
        return (
            f"AuthResult(success={self.success}, ssid={ssid_display}, "
            f"mfa={bool(self.mfa_challenge)}, error={self.error_message!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class AuthProvider(abc.ABC):
    """Abstract authentication provider contract."""

    @abc.abstractmethod
    async def login(self, credentials: Credentials) -> AuthResult:
        """Authenticate with username and password."""
        ...

    @abc.abstractmethod
    async def verify_mfa(self, challenge: MFAChallenge, code: str) -> AuthResult:
        """Resolve an active 2FA challenge with the supplied code."""
        ...

    @abc.abstractmethod
    async def logout(self, ssid: str) -> bool:
        """Invalidate the session on the server."""
        ...

    @abc.abstractmethod
    async def get_profile_and_balances(self, ssid: str) -> tuple[Profile | None, list[Balance]]:
        """Fetch account profile and balances associated with the authenticated session."""
        ...


class MockAuthProvider(AuthProvider):
    """Completely offline authentication provider for development and testing.

    Enables testing the entire authentication and session lifecycle without
    transmitting credentials or opening genuine network sockets.
    """

    def __init__(
        self,
        *,
        valid_email: str = "trader@example.com",
        valid_password: str = "mock_secret_123",  # noqa: S107
        require_2fa: bool = False,
        expected_mfa_code: str = "123456",
        mfa_method: MFAMethod = MFAMethod.AUTHENTICATOR,
        simulate_network_error: bool = False,
        simulate_rate_limit: bool = False,
        mock_ssid: str = "mock_ssid_99999999999999999999999999999999",
        mock_profile: Profile | None = None,
        mock_balances: list[Balance] | None = None,
    ) -> None:
        self.valid_email = valid_email
        self.valid_password = valid_password
        self.require_2fa = require_2fa
        self.expected_mfa_code = expected_mfa_code
        self.mfa_method = mfa_method
        self.simulate_network_error = simulate_network_error
        self.simulate_rate_limit = simulate_rate_limit
        self.mock_ssid = mock_ssid
        self.mock_profile = mock_profile or Profile.from_raw({
            "user_id": 12345,
            "email": valid_email,
            "balance_id": 1001,
            "first_name": "Mock",
            "last_name": "Trader",
        })
        self.mock_balances = mock_balances or [
            Balance(
                id="1001",
                type=AccountType.PRACTICE,
                amount=10000.0,
                currency="USD",
                is_active=True,
            ),
            Balance(id="2002", type=AccountType.REAL, amount=50.0, currency="USD"),
        ]
        self.logged_out_ssids: set[str] = set()

    async def login(self, credentials: Credentials) -> AuthResult:
        if self.simulate_network_error:
            return AuthResult(
                success=False,
                error_message="Simulated mock network connection error",
                status_code=503,
            )
        if self.simulate_rate_limit:
            return AuthResult(
                success=False,
                error_message="Too many login attempts. Broker rate-limit active.",
                status_code=429,
            )

        if not credentials.email or not credentials.password:
            return AuthResult(
                success=False,
                error_message="Credentials cannot be empty",
                status_code=400,
            )

        if credentials.email != self.valid_email or credentials.password != self.valid_password:
            return AuthResult(
                success=False,
                error_message="Invalid email or password",
                status_code=401,
            )

        if self.require_2fa:
            challenge = MFAChallenge(
                token="mock_challenge_token_abc123",  # noqa: S106
                method=self.mfa_method,
                message="Enter your mock 6-digit authenticator code.",
            )
            return AuthResult(
                success=False,
                mfa_challenge=challenge,
                status_code=200,
            )

        return AuthResult(
            success=True,
            ssid=self.mock_ssid,
            status_code=200,
        )

    async def verify_mfa(self, challenge: MFAChallenge, code: str) -> AuthResult:
        if challenge.is_expired:
            return AuthResult(
                success=False,
                error_message="2FA verification challenge has expired",
                status_code=410,
            )
        if code.strip() != self.expected_mfa_code:
            return AuthResult(
                success=False,
                error_message="Invalid 2FA verification code",
                status_code=401,
            )

        return AuthResult(
            success=True,
            ssid=self.mock_ssid,
            status_code=200,
        )

    async def logout(self, ssid: str) -> bool:
        self.logged_out_ssids.add(ssid)
        return True

    async def get_profile_and_balances(self, ssid: str) -> tuple[Profile | None, list[Balance]]:
        if ssid in self.logged_out_ssids:
            return None, []
        return self.mock_profile, list(self.mock_balances)


class IQOptionAuthProvider(AuthProvider):
    """Production HTTPS authentication provider for IQ Option broker endpoints."""

    def __init__(
        self,
        *,
        login_url: str = DEFAULT_LOGIN_URL,
        logout_url: str = DEFAULT_LOGOUT_URL,
        verify_2fa_url: str = DEFAULT_2FA_VERIFY_URL,
        config: ConnectionConfig | None = None,
    ) -> None:
        self.login_url = login_url
        self.logout_url = logout_url
        self.verify_2fa_url = verify_2fa_url
        self.config = config or ConnectionConfig()

    async def login(self, credentials: Credentials) -> AuthResult:
        """Authenticate via HTTPS and obtain the session SSID cookie or 2FA challenge."""
        validate_https_endpoint(self.login_url)
        ssl_context = create_tls_context()
        timeout = ClientTimeout(total=self.config.connect_timeout)

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_context),
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=timeout,
        ) as session:
            try:
                response = await session.post(
                    self.login_url,
                    data={"identifier": credentials.email, "password": credentials.password},
                    ssl=ssl_context,
                    allow_redirects=False,
                )
            except aiohttp.ClientError as exc:
                raise ConnectionError(f"HTTPS login request failed: {exc}") from exc
            except TimeoutError as exc:
                raise RequestTimeoutError(
                    f"Login request timed out after {self.config.connect_timeout}s"
                ) from exc

            async with response:
                if 300 <= response.status < 400:
                    return AuthResult(
                        success=False,
                        error_message=f"Authentication redirect rejected (HTTP {response.status})",
                        status_code=response.status,
                    )

                payload: dict[str, Any] | None
                try:
                    payload = await response.json(content_type=None)
                except (aiohttp.ContentTypeError, ValueError):
                    payload = None

                logger.debug(
                    "Login response: status=%s, payload=%s",
                    response.status,
                    redact_dict(payload or {}),
                )

                # 1. Attempt SSID extraction from cookies or JSON payload
                extracted_ssid: str | None = None
                ssid_cookie = response.cookies.get("ssid")
                if ssid_cookie is not None and ssid_cookie.value:
                    extracted_ssid = ssid_cookie.value
                elif isinstance(payload, dict):
                    data_obj = payload.get("data")
                    if isinstance(data_obj, dict) and data_obj.get("ssid"):
                        extracted_ssid = str(data_obj["ssid"])
                    elif payload.get("ssid"):
                        extracted_ssid = str(payload["ssid"])

                # If successful and SSID obtained, return success immediately
                if response.status == 200 and extracted_ssid:
                    return AuthResult(
                        success=True,
                        ssid=extracted_ssid,
                        status_code=200,
                    )

                # 2. Check for 2FA verification challenge in response payload
                if isinstance(payload, dict):
                    code_val = str(payload.get("code", "")).lower()
                    raw_data = payload.get("data")
                    data_dict: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
                    data_code = str(data_dict.get("code", "")).lower()

                    mfa_codes = {
                        "verify",
                        "verify_2fa",
                        "2fa",
                        "two_factor",
                        "challenge_required",
                    }
                    if (
                        code_val in mfa_codes
                        or data_code in mfa_codes
                        or "2fa" in code_val
                        or code_val == "verify"
                    ):
                        raw_token = payload.get("token") or data_dict.get("token", "")
                        raw_method = str(
                            payload.get("method") or data_dict.get("method", "authenticator")
                        ).lower()
                        method = MFAMethod.AUTHENTICATOR
                        if "sms" in raw_method:
                            method = MFAMethod.SMS
                        elif "email" in raw_method:
                            method = MFAMethod.EMAIL

                        msg = str(
                            payload.get("message")
                            or data_dict.get("message")
                            or (
                                "Two-factor authentication (2FA) verification code "
                                "required by broker."
                            )
                        )
                        challenge = MFAChallenge(
                            token=str(raw_token),
                            method=method,
                            message=msg,
                        )
                        return AuthResult(
                            success=False,
                            mfa_challenge=challenge,
                            status_code=response.status,
                        )

                    # Check for explicit invalid credentials codes
                    cred_err_codes = {
                        "invalid_credentials",
                        "user_not_found",
                        "invalid_password",
                        "bad_credentials",
                    }
                    if code_val in cred_err_codes or data_code in cred_err_codes:
                        err_msg = (
                            payload.get("message")
                            or data_dict.get("message")
                            or "Invalid credentials: email or password rejected by broker."
                        )
                        return AuthResult(
                            success=False,
                            error_message=str(err_msg),
                            status_code=response.status,
                        )

                # 3. Handle HTTP error status codes
                if response.status == 429:
                    return AuthResult(
                        success=False,
                        error_message=(
                            "Broker rate limit exceeded: too many authentication attempts. "
                            "Please wait before retrying."
                        ),
                        status_code=429,
                    )
                if response.status in (401, 403):
                    err_msg = "Authentication rejected by broker. Please verify your credentials."
                    if isinstance(payload, dict) and payload.get("message"):
                        err_msg = str(payload["message"])
                    return AuthResult(
                        success=False,
                        error_message=err_msg,
                        status_code=response.status,
                    )
                if response.status >= 500:
                    return AuthResult(
                        success=False,
                        error_message=(
                            f"Broker authentication service unavailable (HTTP {response.status})."
                        ),
                        status_code=response.status,
                    )
                if response.status != 200:
                    err_msg = f"Authentication failed (HTTP {response.status})."
                    if isinstance(payload, dict) and payload.get("message"):
                        err_msg = str(payload["message"])
                    return AuthResult(
                        success=False,
                        error_message=err_msg,
                        status_code=response.status,
                    )

                # 4. HTTP 200 without SSID or 2FA challenge
                err_msg = (
                    "Broker returned HTTP 200 but did not provide an SSID session "
                    "token or challenge."
                )
                if isinstance(payload, dict) and payload.get("message"):
                    err_msg = str(payload["message"])
                return AuthResult(
                    success=False,
                    error_message=err_msg,
                    status_code=200,
                )

    async def verify_mfa(self, challenge: MFAChallenge, code: str) -> AuthResult:
        """Submit the 2FA verification code to the broker endpoint."""
        validate_https_endpoint(self.verify_2fa_url)
        ssl_context = create_tls_context()
        timeout = ClientTimeout(total=self.config.connect_timeout)

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_context),
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=timeout,
        ) as session:
            try:
                response = await session.post(
                    self.verify_2fa_url,
                    data={"token": challenge.token, "code": code.strip()},
                    ssl=ssl_context,
                    allow_redirects=False,
                )
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                return AuthResult(
                    success=False,
                    error_message=f"2FA verification request failed: {exc}",
                )

            async with response:
                payload: dict[str, Any] | None
                try:
                    payload = await response.json(content_type=None)
                except (aiohttp.ContentTypeError, ValueError):
                    payload = None

                if response.status != 200:
                    err_msg = f"2FA verification rejected (HTTP {response.status})"
                    if isinstance(payload, dict) and payload.get("message"):
                        err_msg = str(payload["message"])
                    return AuthResult(
                        success=False,
                        error_message=err_msg,
                        status_code=response.status,
                    )

                extracted_ssid: str | None = None
                ssid_cookie = response.cookies.get("ssid")
                if ssid_cookie is not None and ssid_cookie.value:
                    extracted_ssid = ssid_cookie.value
                elif isinstance(payload, dict):
                    data_obj = payload.get("data")
                    if isinstance(data_obj, dict) and data_obj.get("ssid"):
                        extracted_ssid = str(data_obj["ssid"])
                    elif payload.get("ssid"):
                        extracted_ssid = str(payload["ssid"])

                if extracted_ssid is not None:
                    return AuthResult(
                        success=True,
                        ssid=extracted_ssid,
                        status_code=200,
                    )

                return AuthResult(
                    success=False,
                    error_message="2FA succeeded but no SSID cookie or token was issued",
                    status_code=response.status,
                )

    async def logout(self, ssid: str) -> bool:
        """Call broker logout endpoint."""
        return await async_logout(ssid, logout_url=self.logout_url, config=self.config)

    async def get_profile_and_balances(self, ssid: str) -> tuple[Profile | None, list[Balance]]:
        """Fetch profile and balances via the read-only WebSocket client facade."""
        from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient

        ws = IQOptionWebSocketClient(ssid=ssid, config=self.config)
        try:
            await ws.connect()
            profile = ws.profile
            balances = list(ws.balances)
            await ws.close()
            return profile, balances
        except (ConnectionError, LoginError, TimeoutError, OSError) as exc:
            logger.warning("Failed to fetch profile and balances: %s", exc)
            return None, []
