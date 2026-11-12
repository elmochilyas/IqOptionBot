"""Deterministic synthetic market fixtures for Phase 4 engineering demonstrations.

These fixtures exist solely to exercise the machine-learning pipeline end to end.
They are **not** market evidence: any predictive performance observed on synthetic
data says nothing about real IQ Option instruments. Every fixture is labelled with
``is_synthetic=True`` and a human-readable disclaimer so reports cannot mistake it
for authentic data.

The optional planted momentum signal makes the *next* bar return depend on the
*previous* bar return. That relationship is learnable using causal features
(``return_1`` at time ``t`` sees only closes up to ``t``), which lets tests verify
that the pipeline can detect a signal when one genuinely exists, and that it
correctly fails to find one when the signal is removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqoption_research.models.market import Candle, CandleSize

SYNTHETIC_DISCLAIMER = (
    "Synthetic engineering fixture. Not real market data. Results are not evidence "
    "of a tradable edge and must never be presented as expected profitability."
)

DEFAULT_ACTIVE_ID = 1


class SyntheticFixtureError(ValueError):
    """Raised when synthetic fixture parameters are invalid."""


@dataclass(frozen=True, slots=True)
class SyntheticMarketConfig:
    """Parameters controlling a deterministic synthetic candle series.

    Attributes:
        n_bars: Number of complete candles to generate (must be >= 2).
        seed: NumPy seed; identical seeds produce byte-identical candles.
        start_price: Initial mid price (must be > 0).
        start_ts: Epoch second of the first candle window start.
        candle_size_seconds: Candle duration in seconds.
        momentum_strength: Coefficient ``phi`` in ``r_{t+1} = phi * r_t + noise``.
            Zero means no learnable serial dependence. Positive values plant a
            mean-reverting-free momentum relationship.
        volatility: Standard deviation of the per-bar return noise.
        drift: Constant per-bar return drift.
        spread_pips: Base bid/ask spread in pips (1 pip = 0.0001 for EURUSD).
        spread_jitter_pips: Uniform per-bar spread variation in pips.
        volume_base: Baseline integer volume.
        volume_jitter: Maximum additional random integer volume.
        symbol: Instrument symbol attached to every candle.
        candle_size: Candle timeframe enum.
        active_id: Instrument active id.
    """

    n_bars: int = 1200
    seed: int = 20240101
    start_price: float = 1.08500
    start_ts: int = 1_700_000_000
    candle_size_seconds: int = 60
    momentum_strength: float = 0.0
    volatility: float = 0.0006
    drift: float = 0.0
    spread_pips: float = 0.8
    spread_jitter_pips: float = 0.6
    volume_base: int = 50
    volume_jitter: int = 100
    symbol: str = "EURUSD"
    active_id: int = DEFAULT_ACTIVE_ID

    def __post_init__(self) -> None:
        if self.n_bars < 2:
            raise SyntheticFixtureError(f"n_bars must be >= 2, got {self.n_bars}")
        if not math.isfinite(self.start_price) or self.start_price <= 0.0:
            raise SyntheticFixtureError(
                f"start_price must be a positive finite number, got {self.start_price}"
            )
        if self.start_ts < 0:
            raise SyntheticFixtureError(f"start_ts must be >= 0, got {self.start_ts}")
        if self.candle_size_seconds < 1:
            raise SyntheticFixtureError(
                f"candle_size_seconds must be >= 1, got {self.candle_size_seconds}"
            )
        if not (-1.0 < self.momentum_strength < 1.0):
            raise SyntheticFixtureError(
                f"momentum_strength must be in (-1.0, 1.0), got {self.momentum_strength}"
            )
        if not math.isfinite(self.volatility) or self.volatility <= 0.0:
            raise SyntheticFixtureError(
                f"volatility must be positive and finite, got {self.volatility}"
            )
        if not math.isfinite(self.drift):
            raise SyntheticFixtureError(f"drift must be finite, got {self.drift}")
        if self.spread_pips < 0.0 or self.spread_jitter_pips < 0.0:
            raise SyntheticFixtureError("spread values must be non-negative")
        if self.volume_base < 1:
            raise SyntheticFixtureError("volume_base must be >= 1")
        if self.volume_jitter < 0:
            raise SyntheticFixtureError("volume_jitter must be >= 0")
        if not self.symbol:
            raise SyntheticFixtureError("symbol must be a non-empty string")


@dataclass(frozen=True, slots=True)
class SyntheticFixture:
    """Generated synthetic candles plus provenance metadata."""

    candles: tuple[Candle, ...]
    config: SyntheticMarketConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_synthetic(self) -> bool:
        return True

    @property
    def n_bars(self) -> int:
        return len(self.candles)

    @property
    def disclaimer(self) -> str:
        return SYNTHETIC_DISCLAIMER


def generate_synthetic_candles(config: SyntheticMarketConfig | None = None) -> SyntheticFixture:
    """Generate a deterministic, internally consistent synthetic candle series.

    The generated returns follow an AR(1) process ``r_t = drift + phi * r_{t-1} + eps_t``.
    Candles are complete, chronologically ordered, and satisfy OHLC plus positive
    bid/ask invariants. Volume and quotes vary across bars so the feature pipeline
    can exercise its volume and quote branches.
    """
    cfg = config or SyntheticMarketConfig()
    rng = np.random.default_rng(cfg.seed)

    n = cfg.n_bars
    returns = np.zeros(n, dtype=np.float64)
    # First bar has no learnable predecessor.
    returns[0] = cfg.drift + float(rng.normal(0.0, cfg.volatility))
    for i in range(1, n):
        previous = returns[i - 1]
        shock = float(rng.normal(0.0, cfg.volatility))
        returns[i] = cfg.drift + cfg.momentum_strength * previous + shock

    # Keep returns bounded so prices remain positive over long series.
    returns = np.clip(returns, -0.05, 0.05)
    closes = cfg.start_price * np.exp(np.cumsum(returns))

    spreads = np.maximum(
        0.0,
        (cfg.spread_pips + rng.uniform(0.0, cfg.spread_jitter_pips, size=n)) * 1e-4,
    )
    volumes = cfg.volume_base + rng.integers(0, cfg.volume_jitter + 1, size=n, dtype=np.int64)

    candles: list[Candle] = []
    open_price = cfg.start_price
    for i in range(n):
        close_price = float(closes[i])
        # A small deterministic intrabar excursion beyond the open/close body.
        half_range = (
            abs(close_price - open_price) * 0.35
            + float(rng.uniform(0.1, 0.9)) * cfg.volatility * open_price
        )
        high = max(open_price, close_price) + half_range
        low = min(open_price, close_price) - half_range
        low = max(low, 1e-9)

        spread = float(spreads[i])
        bid = close_price - spread / 2.0
        ask = close_price + spread / 2.0
        bid = max(bid, 1e-9)

        from_ts = cfg.start_ts + i * cfg.candle_size_seconds
        to_ts = from_ts + cfg.candle_size_seconds
        candles.append(
            Candle(
                active_id=cfg.active_id,
                symbol=cfg.symbol,
                size=CandleSize(cfg.candle_size_seconds),
                from_ts=from_ts,
                to_ts=to_ts,
                at_ts=to_ts * 1_000_000_000,
                open=float(open_price),
                high=float(high),
                low=float(low),
                close=close_price,
                volume=int(volumes[i]),
                bid=float(bid),
                ask=float(ask),
                is_complete=True,
                provenance="synthetic",
            )
        )
        open_price = close_price

    metadata = {
        "fixture": "synthetic_market",
        "is_synthetic": True,
        "planted_momentum": not math.isclose(cfg.momentum_strength, 0.0, abs_tol=1e-12),
        "momentum_strength": float(cfg.momentum_strength),
        "volatility": float(cfg.volatility),
        "drift": float(cfg.drift),
        "seed": int(cfg.seed),
        "n_bars": n,
        "symbol": cfg.symbol,
        "candle_size_seconds": int(cfg.candle_size_seconds),
        "disclaimer": SYNTHETIC_DISCLAIMER,
    }
    return SyntheticFixture(candles=tuple(candles), config=cfg, metadata=metadata)


def generate_no_edge_fixture(
    n_bars: int = 1200,
    *,
    seed: int = 20240101,
    symbol: str = "EURUSD",
) -> SyntheticFixture:
    """Synthetic series with no planted serial dependence (null-edge fixture)."""
    return generate_synthetic_candles(
        SyntheticMarketConfig(
            n_bars=n_bars,
            seed=seed,
            symbol=symbol,
            momentum_strength=0.0,
        )
    )


def generate_momentum_fixture(
    n_bars: int = 1200,
    *,
    seed: int = 20240101,
    momentum_strength: float = 0.45,
    symbol: str = "EURUSD",
) -> SyntheticFixture:
    """Synthetic series with a planted, causally learnable momentum relationship."""
    return generate_synthetic_candles(
        SyntheticMarketConfig(
            n_bars=n_bars,
            seed=seed,
            symbol=symbol,
            momentum_strength=momentum_strength,
        )
    )
