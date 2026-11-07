"""Abstract base strategy interface.

Enables Phase 3 baselines and Phase 4 machine learning models to implement
the exact same prediction and signal interface without modifying the backtesting engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from iqoption_research.research.backtest.models import Signal
from iqoption_research.research.features.pipeline import FeatureRow


class BaseStrategy(ABC):
    """Abstract trading strategy interface."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def on_bar(self, current_row: FeatureRow, history: Sequence[FeatureRow]) -> Signal:
        """Evaluate current point-in-time bar and past history to produce a trading signal.

        Args:
            current_row: The most recently completed bar with causal features.
            history: Sequence of historical completed FeatureRows strictly prior to or
                     including current_row. Future rows are NEVER accessible.

        Returns:
            Signal specifying direction (CALL/PUT/HOLD), optional stake, and horizon.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state between simulation runs."""
