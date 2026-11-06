"""IQ Option Research — Secure, read-only connectivity for quantitative analysis.

Phase 1: Connection and data only. No trade placement or execution.

Security-first design:
- TLS certificate verification always enabled
- Credentials never logged or persisted
- Endpoint allowlisting enforced
- Offline-only mode supported
"""

__version__ = "0.1.0"
__author__ = "IQ Option Research Project"
__license__ = "MIT"

# Public API - read-only connectors and models only
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.models.account import (
    AccountStatus,
    AccountType,
    Balance,
    PracticeStatus,
    PracticeVerification,
    Profile,
)
from iqoption_research.models.connection import ConnectionState
from iqoption_research.models.market import Candle, CandleSize, PriceTick

__all__ = [
    "AccountStatus",
    "AccountType",
    "AsyncIQOption",
    "Balance",
    "Candle",
    "CandleSize",
    "ConnectionState",
    "PracticeStatus",
    "PracticeVerification",
    "PriceTick",
    "Profile",
]
