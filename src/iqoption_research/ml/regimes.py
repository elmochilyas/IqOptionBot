"""Causal Market Regime Research Engine (Phase 43).

Strictly causal, point-in-time market regime classification:
1. Regimes investigated:
   - Volatility Regime: LOW_VOLATILITY, NORMAL_VOLATILITY, HIGH_VOLATILITY.
   - Trend Regime: TRENDING_BULL, TRENDING_BEAR, RANGING.
   - Structural Range Regime: COMPRESSION, EXPANSION, NORMAL_RANGE.
   - Session Context: TOKYO_ASIAN, LONDON_EUROPEAN, NEW_YORK_US, LONDON_NY_OVERLAP, OFF_HOURS.
2. Invariants:
   - Thresholds are fitted EXCLUSIVELY on development data.
   - Classification at timestamp T depends exclusively on information available at or before T.
   - Future target outcomes are never accessed or leaked.
   - Regime classification is descriptive research context, not an objective truth.
3. Stratified Evaluation:
   - Evaluates model accuracy, calibration, coverage, and matched baseline improvement per regime.
   - Preregistration ensures regime rules cannot be cherry-picked post-hoc.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np

from iqoption_research.campaign.analysis import compute_wilson_ci
from iqoption_research.market_data.timeframe_engine import AggregatedCandle
from iqoption_research.models.market import Candle

logger = logging.getLogger(__name__)


class VolatilityRegime(StrEnum):
    LOW = "LOW_VOLATILITY"
    NORMAL = "NORMAL_VOLATILITY"
    HIGH = "HIGH_VOLATILITY"


class TrendRegime(StrEnum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"


class RangeRegime(StrEnum):
    COMPRESSION = "COMPRESSION"
    EXPANSION = "EXPANSION"
    NORMAL = "NORMAL_RANGE"


class MarketSession(StrEnum):
    TOKYO_ASIAN = "TOKYO_ASIAN"
    LONDON_EUROPEAN = "LONDON_EUROPEAN"
    NEW_YORK_US = "NEW_YORK_US"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    OFF_HOURS = "OFF_HOURS"


@dataclass(frozen=True, slots=True)
class RegimeClassification:
    """Point-in-time market regime state at timestamp T."""

    timestamp: int
    volatility: VolatilityRegime
    trend: TrendRegime
    range_state: RangeRegime
    session: MarketSession

    def to_dict(self) -> dict[str, str]:
        return {
            "volatility": self.volatility.value,
            "trend": self.trend.value,
            "range_state": self.range_state.value,
            "session": self.session.value,
        }


@dataclass(frozen=True, slots=True)
class RegimeStratifiedMetrics:
    """Scientific evaluation metrics for a specific market regime subset."""

    regime_family: str  # "volatility" | "trend" | "range" | "session"
    regime_label: str
    total_opportunities: int
    selected_count: int
    wins: int
    losses: int
    distinct_market_days: int
    decisive_accuracy: float | None
    coverage: float
    wilson_ci_95: tuple[float, float] | None
    matched_baseline_accuracy: float | None
    matched_baseline_improvement: float | None
    hypothetical_expectancy_85: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalMarketRegimeClassifier:
    """Fits regime quantile thresholds on development candles and classifies series causally."""

    def __init__(
        self,
        vol_low_threshold: float = 0.0003,
        vol_high_threshold: float = 0.0008,
        compression_threshold: float = 0.70,
        expansion_threshold: float = 1.30,
        trend_slope_threshold: float = 0.00005,
    ) -> None:
        self.vol_low_threshold = vol_low_threshold
        self.vol_high_threshold = vol_high_threshold
        self.compression_threshold = compression_threshold
        self.expansion_threshold = expansion_threshold
        self.trend_slope_threshold = trend_slope_threshold
        self.is_fitted = False

    def fit(self, dev_candles: Sequence[Candle]) -> CausalMarketRegimeClassifier:
        """Fit empirical quantiles exclusively on development training candles."""
        if len(dev_candles) < 100:
            logger.warning("Fewer than 100 development candles; using default regime thresholds.")
            self.is_fitted = True
            return self

        closes = np.array([c.close for c in dev_candles], dtype=np.float64)
        ranges = np.array([c.high - c.low for c in dev_candles], dtype=np.float64)

        # 20-bar rolling standard deviation of percentage returns
        log_rets = np.zeros(len(closes), dtype=np.float64)
        log_rets[1:] = np.log(closes[1:] / closes[:-1])

        rolling_vol: list[float] = []
        for i in range(20, len(closes)):
            w = log_rets[i - 20 : i]
            rolling_vol.append(float(np.std(w, ddof=1)))

        if rolling_vol:
            # 25th and 75th percentiles of rolling volatility
            self.vol_low_threshold = float(np.percentile(rolling_vol, 25))
            self.vol_high_threshold = float(np.percentile(rolling_vol, 75))

        # Range expansion / compression ratio (range / 20-bar SMA range)
        rolling_range_ratio: list[float] = []
        for i in range(20, len(ranges)):
            sma_r = float(np.mean(ranges[i - 20 : i]))
            if sma_r > 1e-8:
                rolling_range_ratio.append(ranges[i] / sma_r)

        if rolling_range_ratio:
            self.compression_threshold = float(np.percentile(rolling_range_ratio, 25))
            self.expansion_threshold = float(np.percentile(rolling_range_ratio, 75))

        self.is_fitted = True
        return self

    @staticmethod
    def get_market_session(ts: int) -> MarketSession:
        """Determine market session context based on UTC hour.

        Approximate standard institutional session hours:
        - Tokyo / Asian: 00:00 - 08:00 UTC
        - London / European: 08:00 - 16:30 UTC
        - New York / US: 13:00 - 21:00 UTC
        - London / NY Overlap: 13:00 - 16:30 UTC
        - Off-hours / Closure: 21:00 - 00:00 UTC or weekend
        """
        dt = datetime.fromtimestamp(ts, tz=UTC)
        weekday = dt.weekday()  # 5 = Sat, 6 = Sun
        if weekday == 5 or (weekday == 4 and dt.hour >= 22) or (weekday == 6 and dt.hour < 21):
            return MarketSession.OFF_HOURS

        hour = dt.hour
        minute = dt.minute
        time_dec = hour + (minute / 60.0)

        if 13.0 <= time_dec < 16.5:
            return MarketSession.LONDON_NY_OVERLAP
        if 8.0 <= time_dec < 16.5:
            return MarketSession.LONDON_EUROPEAN
        if 13.0 <= time_dec < 21.0:
            return MarketSession.NEW_YORK_US
        if 0.0 <= time_dec < 8.0:
            return MarketSession.TOKYO_ASIAN
        return MarketSession.OFF_HOURS

    def classify_series(
        self,
        candles: Sequence[Candle],
        _htf_m15: Sequence[AggregatedCandle] | None = None,
    ) -> list[RegimeClassification]:
        """Classify every candle in sequence using strictly causal lookback."""
        n = len(candles)
        if n == 0:
            return []

        closes = np.array([c.close for c in candles], dtype=np.float64)
        ranges = np.array([c.high - c.low for c in candles], dtype=np.float64)
        from_ts = [c.from_ts for c in candles]

        # Rolling M1 log returns
        log_rets = np.zeros(n, dtype=np.float64)
        log_rets[1:] = np.log(closes[1:] / closes[:-1])

        # Precompute 20-bar rolling vol and 20-bar SMA range
        vols = np.zeros(n, dtype=np.float64)
        range_ratios = np.ones(n, dtype=np.float64)
        slopes = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            lookback_i = max(0, i - 20)
            sub_rets = log_rets[lookback_i : i + 1]
            if len(sub_rets) > 1:
                vols[i] = float(np.std(sub_rets, ddof=1))

            sub_ranges = ranges[lookback_i : i + 1]
            mean_r = float(np.mean(sub_ranges)) if len(sub_ranges) > 0 else 1e-5
            range_ratios[i] = ranges[i] / max(mean_r, 1e-8)

            # Trend slope (20-bar price difference normalized)
            slopes[i] = (closes[i] - closes[lookback_i]) / max(closes[lookback_i], 1e-8)

        classifications: list[RegimeClassification] = []
        for i in range(n):
            ts = from_ts[i]
            v = vols[i]
            rr = range_ratios[i]
            slope = slopes[i]

            # 1. Volatility Regime
            if v < self.vol_low_threshold:
                vol_reg = VolatilityRegime.LOW
            elif v > self.vol_high_threshold:
                vol_reg = VolatilityRegime.HIGH
            else:
                vol_reg = VolatilityRegime.NORMAL

            # 2. Trend Regime
            if slope > self.trend_slope_threshold:
                tr_reg = TrendRegime.TRENDING_BULL
            elif slope < -self.trend_slope_threshold:
                tr_reg = TrendRegime.TRENDING_BEAR
            else:
                tr_reg = TrendRegime.RANGING

            # 3. Range State
            if rr < self.compression_threshold:
                rng_reg = RangeRegime.COMPRESSION
            elif rr > self.expansion_threshold:
                rng_reg = RangeRegime.EXPANSION
            else:
                rng_reg = RangeRegime.NORMAL

            # 4. Session Context
            sess = self.get_market_session(ts)

            classifications.append(
                RegimeClassification(
                    timestamp=ts,
                    volatility=vol_reg,
                    trend=tr_reg,
                    range_state=rng_reg,
                    session=sess,
                )
            )

        return classifications


def evaluate_regimes_stratified(
    regimes: Sequence[RegimeClassification],
    selected_mask: Sequence[bool] | np.ndarray,
    decisions: Sequence[str],
    true_directions: Sequence[str],
    timestamps: Sequence[int],
    majority_direction: str = "UP",
) -> dict[str, list[RegimeStratifiedMetrics]]:
    """Generate comprehensive stratified evaluation metrics across all market regimes."""
    sel = np.asarray(selected_mask, dtype=bool)
    decs = list(decisions)
    tds = list(true_directions)
    ts = list(timestamps)
    n = len(regimes)

    families = {
        "volatility": [r.volatility.value for r in regimes],
        "trend": [r.trend.value for r in regimes],
        "range_state": [r.range_state.value for r in regimes],
        "session": [r.session.value for r in regimes],
    }

    report: dict[str, list[RegimeStratifiedMetrics]] = {}

    for fam_name, labels in families.items():
        unique_labels = sorted(set(labels))
        report[fam_name] = []

        for lbl in unique_labels:
            reg_mask = np.array([lbl == x for x in labels], dtype=bool)
            tot_opps = int(np.sum(reg_mask))

            sel_reg_mask = reg_mask & sel
            tot_sel = int(np.sum(sel_reg_mask))

            wins = 0
            losses = 0
            maj_wins = 0
            sel_ts: list[int] = []

            for i in range(n):
                if sel_reg_mask[i]:
                    sel_ts.append(ts[i])
                    if decs[i] == tds[i]:
                        wins += 1
                    else:
                        losses += 1
                    if majority_direction == tds[i]:
                        maj_wins += 1

            acc = wins / tot_sel if tot_sel > 0 else None
            cov = tot_sel / tot_opps if tot_opps > 0 else 0.0
            distinct_days = len(np.unique(np.array(sel_ts) // 86400)) if sel_ts else 0

            wilson = compute_wilson_ci(wins, tot_sel) if tot_sel > 0 and acc is not None else None
            maj_acc = maj_wins / tot_sel if tot_sel > 0 else None
            maj_imp = (acc - maj_acc) if acc is not None and maj_acc is not None else None
            exp_85 = (wins * 0.85 - losses * 1.0) / tot_sel if tot_sel > 0 else None

            report[fam_name].append(
                RegimeStratifiedMetrics(
                    regime_family=fam_name,
                    regime_label=lbl,
                    total_opportunities=tot_opps,
                    selected_count=tot_sel,
                    wins=wins,
                    losses=losses,
                    distinct_market_days=distinct_days,
                    decisive_accuracy=acc,
                    coverage=cov,
                    wilson_ci_95=wilson,
                    matched_baseline_accuracy=maj_acc,
                    matched_baseline_improvement=maj_imp,
                    hypothetical_expectancy_85=exp_85,
                )
            )

    return report
