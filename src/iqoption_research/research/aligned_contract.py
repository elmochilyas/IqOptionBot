"""Phase30's single mathematical contract for fitting, inference and scoring."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from iqoption_research.models.market import Candle
from iqoption_research.research.features.pipeline import CausalFeaturePipeline

FEATURES = (
    "return_1",
    "log_return_1",
    "intrabar_return",
    "sma_ratio_5",
    "sma_ratio_20",
    "volatility_10",
    "volatility_20",
    "candle_range",
    "body_to_range",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
)


@dataclass(frozen=True)
class AlignedContract:
    version: str = "EURUSD_DELAYED_CONDITIONAL_DIRECTION_V1"
    timeframe: int = 60
    context_bars: int = 21
    max_source_age_seconds: int = 15
    entry_delay_from_feature_close: int = 120
    holding_seconds: int = 60
    confidence_threshold: float = 0.60

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "feature_availability": "CANDLE_CLOSE_F_IS_EARLIEST_NOT_ATTESTED_ARRIVAL",
            "live_gate": "F <= data_observed <= decision <= durable_commit <= F+15",
            "historical_latency": "UNKNOWN; conditional contract, no imputed real arrival times",
            "entry_price": "AUTHENTIC_EURUSD_CANDLE_OPEN_AT_F_PLUS_120",
            "expiry_price": "SAME_CANDLE_CLOSE_AT_F_PLUS_180",
            "labels": {"UP": 1, "DOWN": 0, "TIE": -1},
            "ties": "EXACT_EQUALITY_EXCLUDED_FROM_FITTING_CALIBRATION_DIRECTIONAL_METRICS",
            "probability": "P(UP | DECISIVE); not unconditional probability of winning",
            "missing_policy": "REJECT_NONCONTIGUOUS_CONTEXT_LATENCY_OR_OUTCOME;NO_FILL",
            "blitz": "UNVERIFIED_NOT_SCORED",
        }


CONTRACT = AlignedContract()


def outcome_label(entry: float, exit_price: float) -> int:
    if not np.isfinite([entry, exit_price]).all() or min(entry, exit_price) <= 0:
        raise ValueError("Invalid outcome prices")
    return 1 if exit_price > entry else 0 if exit_price < entry else -1


def target_times(feature_close: int, delay: int = 120) -> tuple[int, int]:
    if feature_close % 60 or delay not in (60, 120, 180):
        raise ValueError("Invalid minute geometry")
    return feature_close + delay, feature_close + delay + 60


def timing_gate(feature_close: int, observed: float, decision: float, committed: float) -> bool:
    return feature_close <= observed <= decision <= committed <= feature_close + 15


def decision(probability: float) -> str:
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid probability")
    return "CALL" if probability >= 0.60 else "PUT" if probability <= 0.40 else "SKIP"


def feature_vector(window: list[Candle]) -> np.ndarray:
    if (
        len(window) != 21
        or any(b.from_ts != a.to_ts for a, b in itertools.pairwise(window))
        or any(not c.is_complete for c in window)
    ):
        raise ValueError("Require exactly 21 contiguous completed feature candles")
    row = (
        CausalFeaturePipeline(include_volume=False, include_quotes=False).transform(window).rows[-1]
    )
    values = np.array([row.features[name] for name in FEATURES], dtype=float)
    if row.is_warmup or not np.isfinite(values).all():
        raise ValueError("Insufficient or invalid causal features")
    return values
