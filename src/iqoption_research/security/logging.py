"""Sanitized logging configuration with secret redaction.

Provides structured logging that automatically redacts credentials,
tokens, cookies, and other sensitive data from all log output.
"""

import logging
import logging.config
import sys
from collections.abc import MutableMapping
from typing import Any

from iqoption_research.security.credentials import SecretRedactingFilter, redact_secrets


class SanitizedFormatter(logging.Formatter):
    """Formatter that redacts secrets from log messages."""

    def format(self, record: logging.LogRecord) -> str:
        # Redact the main message
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)

        # Redact args
        if record.args:
            new_args: list[Any] = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_args.append(redact_secrets(arg))
                elif isinstance(arg, dict):
                    # Simple dict redaction for common secret keys
                    redacted: dict[str, Any] = {}
                    secret_keys = {"password", "ssid", "token", "cookie", "email", "authorization"}
                    for k, v in arg.items():
                        if k.lower() in secret_keys:
                            redacted[k] = "***REDACTED***"
                        elif isinstance(v, str):
                            redacted[k] = redact_secrets(v)
                        else:
                            redacted[k] = v
                    new_args.append(redacted)
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)

        return super().format(record)


def configure_logging(
    level: int = logging.INFO,
    *,
    json_format: bool = False,
    include_timestamp: bool = True,
) -> None:
    """Configure application-wide sanitized logging.

    Args:
        level: Logging level (default INFO).
        json_format: If True, output JSON lines (for log aggregation).
        include_timestamp: Include ISO timestamp in output.
    """
    fmt_str = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        if include_timestamp
        else "%(levelname)-8s | %(name)s | %(message)s"
    )
    # Base configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "sanitized": {
                "()": SanitizedFormatter,
                "format": fmt_str,
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "timestamp": "true",
            }
            if json_format
            else {
                "()": SanitizedFormatter,
                "format": fmt_str,
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "filters": {
            "secret_redaction": {
                "()": SecretRedactingFilter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "json" if json_format else "sanitized",
                "filters": ["secret_redaction"],
                "level": level,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": level,
        },
        "loggers": {
            "iqoption_research": {
                "level": level,
                "propagate": True,
            },
            "aiohttp": {
                "level": logging.WARNING,  # Suppress aiohttp debug noise
                "propagate": True,
            },
            "websockets": {
                "level": logging.WARNING,
                "propagate": True,
            },
        },
    }

    logging.config.dictConfig(config)

    # Ensure our filter is applied to all handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(SecretRedactingFilter())


def get_logger(name: str) -> logging.Logger:
    """Get a logger with sanitized output."""
    return logging.getLogger(f"iqoption_research.{name}")


class SanitizedLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that ensures all messages are sanitized."""

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        if isinstance(msg, str):
            return redact_secrets(msg), kwargs
        return msg, kwargs
