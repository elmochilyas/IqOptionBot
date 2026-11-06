"""Exception hierarchy for the IQ Option read-only connector.

All exceptions are designed to be safe to display: they never include
credentials, tokens, cookies, or full authentication payloads.
"""


class IQOptionError(Exception):
    """Base error for the IQ Option connector."""


class ConnectionError(IQOptionError):
    """WebSocket or HTTP transport failure."""


class LoginError(IQOptionError):
    """Authentication failed (HTTP login or WebSocket authenticate)."""


class TwoFactorRequiredError(LoginError):
    """Account requires 2FA, which this read-only client does not bypass."""


class RequestTimeoutError(IQOptionError):
    """A correlated request did not receive a response in time."""


class OfflineModeError(IQOptionError):
    """Network I/O was attempted while offline-only mode is enabled."""


class EndpointNotAllowedError(ValueError, IQOptionError):
    """An endpoint outside the hardcoded allowlist was supplied."""


class UnsupportedInstrumentError(IQOptionError):
    """Requested instrument is not on the supported research allowlist."""


class InvalidCandleError(IQOptionError):
    """Raw candle payload failed structural or consistency validation."""


class ReconnectExhaustedError(ConnectionError):
    """Bounded reconnection attempts were exhausted."""


class TradingNotAllowedError(IQOptionError):
    """Raised by the read-only guard if a trading capability is invoked.

    Phase 1 exposes NO trading methods. This exception exists so that
    accidental future wiring fails closed instead of silently trading.
    """
