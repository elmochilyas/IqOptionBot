"""Offline Prediction Engine for Shadow Evaluation (Phase 40A).

Executes point-in-time inference on explicitly timestamped candle windows
using frozen, verified candidate models. Enforces strict causal boundaries,
warmup sufficiency, continuity validation, and zero future data leakage.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from iqoption_research.ml.decision.policy import (
    DecisionAction,
    classify_directional_decision,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.shadow.registry import (
    FrozenRegistrationRecord,
    ShadowModelRegistry,
)

logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Base exception for shadow prediction engine errors."""


class PointInTimeViolationError(EngineError):
    """Raised when candles newer than the explicit decision timestamp are accessed."""


class IncompleteCandleError(EngineError):
    """Raised when an unfinalized or incomplete candle is submitted."""


class StaleDataError(EngineError):
    """Raised when the latest candle is older than the configured staleness threshold."""


class WarmupIncompleteError(EngineError):
    """Raised when the input candle window has insufficient warmup history."""


class DiscontinuousDataError(EngineError):
    """Raised when a temporal discontinuity or missing candle occurs in the feature window."""


class FeatureSchemaError(EngineError):
    """Raised when feature schema mismatches the frozen model specification."""


@dataclass(frozen=True, slots=True)
class InferenceDiagnostic:
    """Operational diagnostics for a single inference event."""

    inference_duration_ms: float
    data_age_seconds: float
    candles_in_window: int
    feature_count: int
    model_family: str
    is_baseline: bool
    confidence_threshold: float
    calibrated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "inference_duration_ms": self.inference_duration_ms,
            "data_age_seconds": self.data_age_seconds,
            "candles_in_window": self.candles_in_window,
            "feature_count": self.feature_count,
            "model_family": self.model_family,
            "is_baseline": self.is_baseline,
            "confidence_threshold": self.confidence_threshold,
            "calibrated": self.calibrated,
        }


@dataclass(frozen=True, slots=True)
class PredictionOutput:
    """Strongly typed output of a point-in-time shadow prediction."""

    decision: DecisionAction
    probability_up: float
    probability_down: float
    confidence_threshold: float
    feature_values: dict[str, float]
    feature_fingerprint: str
    feature_cutoff_ts: int
    feature_cutoff_utc: str
    diagnostic: InferenceDiagnostic

    @property
    def is_trade(self) -> bool:
        return self.decision in (DecisionAction.UP, DecisionAction.DOWN)


class ShadowPredictionEngine:
    """Executes point-in-time ML inference for a single registered frozen candidate."""

    def __init__(
        self,
        registry: ShadowModelRegistry,
        model_id: str,
        *,
        max_staleness_seconds: int = 300,
        min_warmup_bars: int = 21,
    ) -> None:
        self.registry = registry
        self.model_id = model_id
        self.max_staleness_seconds = max_staleness_seconds
        self.min_warmup_bars = min_warmup_bars

        # Load and verify frozen registration
        record = self.registry.get_candidate(model_id)
        if record is None:
            raise EngineError(f"Model {model_id} is not registered in the frozen registry")
        self.record: FrozenRegistrationRecord = record

        # Load verified bundle components
        bundle_data = self.registry.load_frozen_bundle(model_id)
        bundle_dict = bundle_data["bundle"]
        self.model = bundle_dict["model"]
        self.scaler = bundle_dict.get("scaler")
        self.calibrator = bundle_dict.get("calibrator")
        self.feature_names = tuple(bundle_dict.get("feature_names", self.record.feature_names))

        # Setup causal feature pipeline with static schema matching model registration
        from iqoption_research.ml.features_v2 import (
            ALL_V2_FEATURE_CONTRACTS,
            CausalFeaturePipelineV2,
        )

        v2_contract_map = {c.name: c for c in ALL_V2_FEATURE_CONTRACTS}
        manifest = bundle_data.get("manifest", {})
        schema_ver = manifest.get("schema_version", manifest.get("feature_schema_version", "1.0"))
        v1_standard_features = {
            "return_1",
            "log_return_1",
            "intrabar_return",
            "candle_range",
            "body_to_range",
            "upper_shadow_ratio",
            "lower_shadow_ratio",
            "log_volume",
            "volume_sma_10",
            "volume_ratio_10",
            "spread_pips",
            "spread_ratio",
            "sin_hour",
            "cos_hour",
            "sin_dow",
            "cos_dow",
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_ratio_5",
            "sma_ratio_20",
            "volatility_10",
            "volatility_20",
        }
        has_v2_exclusive = any(
            f in v2_contract_map and f not in v1_standard_features for f in self.feature_names
        )
        is_v2 = schema_ver == "2.0" or has_v2_exclusive
        self.is_v2 = is_v2

        has_volume = any("volume" in f for f in self.feature_names)
        has_quotes = any("spread" in f for f in self.feature_names)
        has_market_time = any(
            f in ("sin_hour", "cos_hour", "sin_dow", "cos_dow") for f in self.feature_names
        )
        self.feature_pipeline = CausalFeaturePipeline(
            drop_warmup=False,
            include_volume=has_volume,
            include_quotes=has_quotes,
            include_market_time=has_market_time,
            enforce_temporal_continuity=True,
            static_schema=True,
        )

        if is_v2:
            matched_contracts = [
                v2_contract_map[f] for f in self.feature_names if f in v2_contract_map
            ]
            self.v2_pipeline: CausalFeaturePipelineV2 | None = CausalFeaturePipelineV2(
                contracts=matched_contracts
            )
        else:
            self.v2_pipeline = None

    def predict(
        self,
        candles: Sequence[Candle],
        *,
        decision_ts: int | None = None,
    ) -> PredictionOutput:
        """Execute leak-free point-in-time inference on the provided historical candle window.

        Args:
            candles: Chronological sequence of completed Candle objects ending at cutoff.
            decision_ts: Explicit timestamp (epoch seconds) of decision point. Defaults to
                the to_ts of the latest candle. Future candles are strictly prohibited.

        Returns:
            PredictionOutput with decision (UP/DOWN/SKIP), probabilities, and diagnostics.
        """
        t_start = time.perf_counter()

        if len(candles) < self.min_warmup_bars:
            raise WarmupIncompleteError(
                f"Insufficient historical bars: received {len(candles)}, "
                f"minimum required for causal feature warmup is {self.min_warmup_bars}."
            )

        # 1. Temporal sequence and point-in-time validation
        for i in range(len(candles) - 1):
            if candles[i].from_ts >= candles[i + 1].from_ts:
                raise EngineError(
                    f"Non-chronological sequence: {candles[i].from_ts} >= {candles[i + 1].from_ts}"
                )

        latest_candle = candles[-1]
        cutoff_ts = latest_candle.to_ts
        effective_decision_ts = decision_ts if decision_ts is not None else cutoff_ts

        # Point-in-time invariant: no candle may close strictly after decision_ts
        if latest_candle.to_ts > effective_decision_ts:
            raise PointInTimeViolationError(
                f"Future candle access violation: latest candle to_ts {latest_candle.to_ts} "
                f"exceeds decision timestamp {effective_decision_ts}"
            )

        # Check for unfinalized candles
        for c in candles[-5:]:
            if not c.is_complete:
                raise IncompleteCandleError(
                    f"Incomplete candle submitted for inference at ts={c.from_ts}"
                )

        # Staleness check
        data_age = float(max(0, effective_decision_ts - cutoff_ts))
        if data_age > self.max_staleness_seconds:
            raise StaleDataError(
                f"Data is stale: age {data_age:.1f}s > {self.max_staleness_seconds}s"
            )

        # 2. Extract Causal Features
        if self.is_v2 and self.v2_pipeline is not None:
            feat_matrix = self.v2_pipeline.compute_features(candles)
            if feat_matrix.sample_count == 0:
                raise EngineError("Feature extraction produced zero rows")
            latest_idx = feat_matrix.sample_count - 1
            if feat_matrix.warmup_mask[latest_idx]:
                raise DiscontinuousDataError(
                    f"Cutoff candle at ts={cutoff_ts} is marked as warmup. "
                    "Inference cannot be executed across unverified or discontinuous intervals."
                )
            row_features = {
                name: float(feat_matrix.features[latest_idx, col_idx])
                for col_idx, name in enumerate(feat_matrix.feature_names)
            }
        else:
            if self.feature_pipeline is None:
                raise EngineError("No feature pipeline configured")
            feat_dataset = self.feature_pipeline.transform(list(candles))
            if feat_dataset.row_count == 0:
                raise EngineError("Feature extraction produced zero rows")

            latest_row = feat_dataset.rows[-1]

            # Verify latest bar is not marked warmup due to lookback or discontinuity
            if latest_row.is_warmup:
                raise DiscontinuousDataError(
                    f"Cutoff candle at ts={cutoff_ts} is marked as warmup or discontinuous. "
                    "Inference cannot be executed across unverified or discontinuous intervals."
                )
            row_features = latest_row.features
        missing_features = [f for f in self.feature_names if f not in row_features]
        if missing_features:
            raise FeatureSchemaError(
                f"Missing expected features in extracted dataset: {missing_features}"
            )

        feature_vals = {f: float(row_features[f]) for f in self.feature_names}

        # Check for non-finite values (NaN / Inf)
        for name, val in feature_vals.items():
            if np.isnan(val) or np.isinf(val):
                raise DiscontinuousDataError(f"Non-finite feature value detected in {name}: {val}")

        x_raw = np.array([[feature_vals[f] for f in self.feature_names]], dtype=np.float64)

        # Feature fingerprint (deterministic SHA-256 over sorted key-value pairs)
        feat_str = ";".join(f"{k}={v:.8f}" for k, v in sorted(feature_vals.items()))
        feature_fp = hashlib.sha256(feat_str.encode()).hexdigest()[:16]

        # 4. Scale and Predict
        x_input = self.scaler.transform(x_raw) if self.scaler is not None else x_raw
        raw_probs = self.model.predict_proba(x_input)
        cal_probs = (
            self.calibrator.calibrate(raw_probs) if self.calibrator is not None else raw_probs
        )

        # Ensure probabilities are 1D array of length 2
        p_up = float(cal_probs[0, 1]) if cal_probs.shape[1] > 1 else float(cal_probs[0, 0])
        p_down = float(cal_probs[0, 0]) if cal_probs.shape[1] > 1 else 1.0 - p_up

        # Bound probabilities
        p_up = max(0.0, min(1.0, p_up))
        p_down = max(0.0, min(1.0, p_down))

        # 5. Apply Frozen Decision Rule
        tau = self.record.confidence_threshold
        decision = classify_directional_decision(p_up, tau)

        t_end = time.perf_counter()
        duration_ms = (t_end - t_start) * 1000.0

        cutoff_utc = datetime.fromtimestamp(cutoff_ts, UTC).isoformat()

        diag = InferenceDiagnostic(
            inference_duration_ms=round(duration_ms, 3),
            data_age_seconds=round(data_age, 2),
            candles_in_window=len(candles),
            feature_count=len(self.feature_names),
            model_family=self.record.model_family,
            is_baseline=self.record.is_baseline,
            confidence_threshold=tau,
            calibrated=self.calibrator is not None,
        )

        return PredictionOutput(
            decision=decision,
            probability_up=round(p_up, 6),
            probability_down=round(p_down, 6),
            confidence_threshold=tau,
            feature_values=feature_vals,
            feature_fingerprint=feature_fp,
            feature_cutoff_ts=cutoff_ts,
            feature_cutoff_utc=cutoff_utc,
            diagnostic=diag,
        )
