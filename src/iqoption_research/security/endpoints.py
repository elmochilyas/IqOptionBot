"""Endpoint allowlisting and validation.

All IQ Option API endpoints are hardcoded and validated. No user-provided
endpoint URLs are accepted without explicit allowlist membership.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Final

from iqoption_research.security.tls import (
    validate_https_endpoint,
    validate_wss_endpoint,
)


class EndpointType(Enum):
    """Type of IQ Option endpoint."""
    HTTPS_LOGIN = "https_login"
    HTTPS_LOGOUT = "https_logout"
    WSS_WEBSOCKET = "wss_websocket"


# Canonical endpoint definitions — single source of truth
ENDPOINTS: Final[dict[EndpointType, str]] = {
    EndpointType.HTTPS_LOGIN: "https://auth.iqoption.com/api/v2/login",
    EndpointType.HTTPS_LOGOUT: "https://auth.iqoption.com/api/v1.0/logout",
    EndpointType.WSS_WEBSOCKET: "wss://iqoption.com/echo/websocket",
}


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    """Immutable endpoint configuration.

    All URLs are validated against the allowlist at construction time.
    """
    login_url: str
    logout_url: str
    wss_url: str

    def __post_init__(self) -> None:
        validate_https_endpoint(self.login_url)
        validate_https_endpoint(self.logout_url)
        validate_wss_endpoint(self.wss_url)

    @classmethod
    def default(cls) -> "EndpointConfig":
        """Create config with default (allowlisted) endpoints."""
        return cls(
            login_url=ENDPOINTS[EndpointType.HTTPS_LOGIN],
            logout_url=ENDPOINTS[EndpointType.HTTPS_LOGOUT],
            wss_url=ENDPOINTS[EndpointType.WSS_WEBSOCKET],
        )

    def get_login_url(self) -> str:
        return self.login_url

    def get_logout_url(self) -> str:
        return self.logout_url

    def get_wss_url(self) -> str:
        return self.wss_url


def get_default_endpoints() -> EndpointConfig:
    """Get the default (and only supported) endpoint configuration."""
    return EndpointConfig.default()


# No custom endpoint configuration allowed — this function exists only
# to make the allowlist explicit and testable.
def validate_endpoint_allowlist() -> None:
    """Verify that all configured endpoints are in the allowlist.

    Raises:
        ValueError: If any endpoint is not allowlisted.
    """
    for ep_type, url in ENDPOINTS.items():
        if ep_type in (EndpointType.HTTPS_LOGIN, EndpointType.HTTPS_LOGOUT):
            validate_https_endpoint(url)
        elif ep_type == EndpointType.WSS_WEBSOCKET:
            validate_wss_endpoint(url)
