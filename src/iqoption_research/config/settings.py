"""Application settings loaded from environment variables.

SECURITY NOTE
-------------
This settings object deliberately contains **no credential fields**.
Credentials must be entered interactively via ``getpass`` at runtime and
are never read from files, ``.env``, or environment variables by default.

Environment variables (all optional):
    IQOPTION_OFFLINE       true|false   (default: true — network disabled)
    IQOPTION_LOG_LEVEL     DEBUG|INFO|WARNING|ERROR (default: INFO)
    IQOPTION_DATA_DIR      path         (default: ./data)
    IQOPTION_MAX_CANDLES   int          (default: 500, max: 1000)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration with offline-by-default posture."""

    model_config = SettingsConfigDict(
        env_prefix="IQOPTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    offline: bool = Field(
        default=True,
        description="When true, all network I/O is refused.",
    )
    log_level: str = Field(default="INFO")
    data_dir: Path = Field(default=Path("data"))
    max_candles: int = Field(default=500, ge=1, le=1000)

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, value: Path) -> Path:
        return value.expanduser()
