"""Scientific baseline and benchmark strategies.

Provides deterministic baselines for empirical comparison:
- NoTradeStrategy: zero trades, capital preservation control
- AlwaysCallStrategy / AlwaysPutStrategy: directional market exposure benchmarks
- MomentumStrategy: simple 1-bar return sign follower
- MovingAverageCrossoverStrategy: SMA(5) vs SMA(20) crossover
- RandomStrategy: deterministic pseudorandom coin-toss with fixed seed
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from iqoption_research.research.backtest.models import Signal, SignalDirection
from iqoption_research.research.backtest.strategies.base import BaseStrategy
from iqoption_research.research.features.pipeline import FeatureRow


class NoTradeStrategy(BaseStrategy):
    """Control strategy that executes zero trades."""

    def __init__(self) -> None:
        super().__init__(name="no_trade", description="Zero-trade control baseline")

    def on_bar(self, _current_row: FeatureRow, _history: Sequence[FeatureRow]) -> Signal:
        return Signal(direction=SignalDirection.HOLD)


class AlwaysCallStrategy(BaseStrategy):
    """Directional benchmark that continuously buys CALL / Long contracts."""

    def __init__(self, *, horizon_bars: int = 1) -> None:
        super().__init__(name="always_call", description="Always CALL / Buy benchmark")
        self.horizon_bars = horizon_bars

    def on_bar(self, _current_row: FeatureRow, _history: Sequence[FeatureRow]) -> Signal:
        return Signal(direction=SignalDirection.CALL, horizon_bars=self.horizon_bars)


class AlwaysPutStrategy(BaseStrategy):
    """Directional benchmark that continuously buys PUT / Short contracts."""

    def __init__(self, *, horizon_bars: int = 1) -> None:
        super().__init__(name="always_put", description="Always PUT / Sell benchmark")
        self.horizon_bars = horizon_bars

    def on_bar(self, _current_row: FeatureRow, _history: Sequence[FeatureRow]) -> Signal:
        return Signal(direction=SignalDirection.PUT, horizon_bars=self.horizon_bars)


class MomentumStrategy(BaseStrategy):
    """Follows recent directional return sign."""

    def __init__(self, *, threshold: float = 0.0, horizon_bars: int = 1) -> None:
        super().__init__(name="momentum", description="1-bar return momentum follower")
        self.threshold = threshold
        self.horizon_bars = horizon_bars

    def on_bar(self, current_row: FeatureRow, _history: Sequence[FeatureRow]) -> Signal:
        ret = current_row.features.get("return_1", 0.0)
        if ret > self.threshold:
            return Signal(direction=SignalDirection.CALL, horizon_bars=self.horizon_bars)
        if ret < -self.threshold:
            return Signal(direction=SignalDirection.PUT, horizon_bars=self.horizon_bars)
        return Signal(direction=SignalDirection.HOLD)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """SMA(5) vs SMA(20) moving average crossover."""

    def __init__(
        self,
        *,
        fast_sma: str = "sma_5",
        slow_sma: str = "sma_20",
        horizon_bars: int = 1,
    ) -> None:
        super().__init__(
            name="ma_crossover",
            description=f"Moving average crossover ({fast_sma} vs {slow_sma})",
        )
        self.fast_sma = fast_sma
        self.slow_sma = slow_sma
        self.horizon_bars = horizon_bars

    def on_bar(self, current_row: FeatureRow, _history: Sequence[FeatureRow]) -> Signal:
        if current_row.is_warmup:
            return Signal(direction=SignalDirection.HOLD)

        fast_val = current_row.features.get(self.fast_sma)
        slow_val = current_row.features.get(self.slow_sma)

        if fast_val is None or slow_val is None:
            return Signal(direction=SignalDirection.HOLD)

        if fast_val > slow_val:
            return Signal(direction=SignalDirection.CALL, horizon_bars=self.horizon_bars)
        if fast_val < slow_val:
            return Signal(direction=SignalDirection.PUT, horizon_bars=self.horizon_bars)
        return Signal(direction=SignalDirection.HOLD)


class RandomStrategy(BaseStrategy):
    """Deterministic pseudorandom benchmark with explicit seed."""

    def __init__(
        self,
        *,
        seed: int = 42,
        call_prob: float = 0.5,
        horizon_bars: int = 1,
    ) -> None:
        super().__init__(name="random", description=f"Seeded random benchmark (seed={seed})")
        self.seed = seed
        self.call_prob = call_prob
        self.horizon_bars = horizon_bars
        self._rng = random.Random(seed)

    def on_bar(self, _current_row: FeatureRow, _history: Sequence[FeatureRow]) -> Signal:
        roll = self._rng.random()
        direction = SignalDirection.CALL if roll < self.call_prob else SignalDirection.PUT
        return Signal(direction=direction, horizon_bars=self.horizon_bars)

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
