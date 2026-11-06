"""Models package — typed data structures for account, market, and connection."""

from iqoption_research.models.account import (
    AccountStatus,
    AccountType,
    Balance,
    PracticeStatus,
    PracticeVerification,
    Profile,
)
from iqoption_research.models.connection import (
    DEFAULT_CONNECTION_CONFIG,
    ConnectionConfig,
    ConnectionHealth,
    ConnectionMetrics,
    ConnectionState,
    DisconnectReason,
)
from iqoption_research.models.market import (
    Candle,
    CandleSize,
    PriceTick,
    Subscription,
)

__all__ = [
    "DEFAULT_CONNECTION_CONFIG",
    "AccountStatus",
    "AccountType",
    "Balance",
    "Candle",
    "CandleSize",
    "ConnectionConfig",
    "ConnectionHealth",
    "ConnectionMetrics",
    "ConnectionState",
    "DisconnectReason",
    "PracticeStatus",
    "PracticeVerification",
    "PriceTick",
    "Profile",
    "Subscription",
]
