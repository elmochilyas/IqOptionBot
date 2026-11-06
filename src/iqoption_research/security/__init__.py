"""Security module — TLS, credentials, endpoints, and sanitized logging.

All security-critical functionality is centralized here. No other module
should handle credentials, TLS contexts, or endpoint configuration directly.
"""

from iqoption_research.security.credentials import (
    Credentials,
    SecretRedactingFilter,
    get_credentials,
    redact_dict,
    redact_email,
    redact_secrets,
)
from iqoption_research.security.endpoints import (
    EndpointConfig,
    EndpointType,
    get_default_endpoints,
    validate_endpoint_allowlist,
)
from iqoption_research.security.logging import (
    SanitizedLoggerAdapter,
    configure_logging,
    get_logger,
)
from iqoption_research.security.tls import (
    ALLOWED_HTTPS_ENDPOINTS,
    ALLOWED_WSS_ENDPOINTS,
    create_tls_context,
    validate_https_endpoint,
    validate_wss_endpoint,
    verify_certificate_cn,
)

__all__ = [
    "ALLOWED_HTTPS_ENDPOINTS",
    "ALLOWED_WSS_ENDPOINTS",
    "Credentials",
    "EndpointConfig",
    "EndpointType",
    "SanitizedLoggerAdapter",
    "SecretRedactingFilter",
    "configure_logging",
    "create_tls_context",
    "get_credentials",
    "get_default_endpoints",
    "get_logger",
    "redact_dict",
    "redact_email",
    "redact_secrets",
    "validate_endpoint_allowlist",
    "validate_https_endpoint",
    "validate_wss_endpoint",
    "verify_certificate_cn",
]
