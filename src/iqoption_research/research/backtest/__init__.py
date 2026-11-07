"""Backtest subpackage — execution engine, payoff models, baselines, and performance metrics."""

from iqoption_research.research.backtest.engine import (
    BacktestEngine,
    BacktestError,
    BacktestRunResult,
)
from iqoption_research.research.backtest.metrics import (
    BacktestMetrics,
    ConfidenceInterval,
    calculate_metrics,
)
from iqoption_research.research.backtest.models import (
    BacktestConfig,
    PortfolioState,
    Position,
    Signal,
    SignalDirection,
    TradeRecord,
)
from iqoption_research.research.backtest.payoffs import (
    EXECUTION_LIMITATION_DISCLAIMER,
    BasePayoffModel,
    BinaryPayoffModel,
    ForexPayoffModel,
    SettlementResult,
    TieAction,
    TradeOutcome,
)
from iqoption_research.research.backtest.reports import (
    ResearchReport,
    generate_research_report,
)
from iqoption_research.research.backtest.strategies import (
    AlwaysCallStrategy,
    AlwaysPutStrategy,
    BaseStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    NoTradeStrategy,
    RandomStrategy,
)

__all__ = [
    "EXECUTION_LIMITATION_DISCLAIMER",
    "AlwaysCallStrategy",
    "AlwaysPutStrategy",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestError",
    "BacktestMetrics",
    "BacktestRunResult",
    "BasePayoffModel",
    "BaseStrategy",
    "BinaryPayoffModel",
    "ConfidenceInterval",
    "ForexPayoffModel",
    "MomentumStrategy",
    "MovingAverageCrossoverStrategy",
    "NoTradeStrategy",
    "PortfolioState",
    "Position",
    "RandomStrategy",
    "ResearchReport",
    "SettlementResult",
    "Signal",
    "SignalDirection",
    "TieAction",
    "TradeOutcome",
    "TradeRecord",
    "calculate_metrics",
    "generate_research_report",
]
