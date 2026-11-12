"""Trading bot subsystem for Phase 6.

Provides autonomous trading bot controller, lifecycle management,
session execution tracking, and status monitoring.
"""

from iqoption_research.bot.controller import (
    BotConfig,
    BotController,
    BotMode,
    BotState,
    InvalidBotStateTransitionError,
)
from iqoption_research.bot.monitor import BotStatusMonitor, BotStatusSnapshot
from iqoption_research.bot.session import SessionSummary, TradingSession

__all__ = [
    "BotConfig",
    "BotController",
    "BotMode",
    "BotState",
    "BotStatusMonitor",
    "BotStatusSnapshot",
    "InvalidBotStateTransitionError",
    "SessionSummary",
    "TradingSession",
]
