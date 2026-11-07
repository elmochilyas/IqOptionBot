"""Baseline strategies subpackage."""

from iqoption_research.research.backtest.strategies.base import BaseStrategy
from iqoption_research.research.backtest.strategies.baselines import (
    AlwaysCallStrategy,
    AlwaysPutStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    NoTradeStrategy,
    RandomStrategy,
)

__all__ = [
    "AlwaysCallStrategy",
    "AlwaysPutStrategy",
    "BaseStrategy",
    "MomentumStrategy",
    "MovingAverageCrossoverStrategy",
    "NoTradeStrategy",
    "RandomStrategy",
]
