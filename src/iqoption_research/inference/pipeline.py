"""Real-time ML inference pipeline executing saved models on streaming feature rows.

Performs feature schema validation, scaling, inference, probability calibration,
and threshold-based directional decisioning with latency telemetry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Final

import numpy as np

from iqoption_research.inference.validation import ModelIntegrityValidator
from iqoption_research.ml.tracking.artifacts import ModelBundle
from iqoption_research.research.backtest.models import SignalDirection
from iqoption_research.research.features.pipeline import FeatureRow

logger = logging.getLogger(__name__)

DEFAULT_CALL_THRESHOLD: Final[float] = 0.55
DEFAULT_PUT_THRESHOLD: Final[float] = 0.45


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Complete prediction output emitted for an ingested streaming feature row."""

    direction: SignalDirection
    calibrated_probability: float
    raw_probability: float
    confidence: float
    abstention_reason: str | None
    latency_ms: float
    candle_timestamp: int
    model_name: str
    feature_count: int


class ModelInferencePipeline:
    """Inference coordinator executing trained model bundles on streaming feature rows."""

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        call_threshold: float = DEFAULT_CALL_THRESHOLD,
        put_threshold: float = DEFAULT_PUT_THRESHOLD,
        validator: ModelIntegrityValidator | None = None,
    ) -> None:
        if call_threshold <= put_threshold:
            raise ValueError(
                f"call_threshold ({call_threshold}) must be strictly greater than "
                f"put_threshold ({put_threshold})"
            )
        self.bundle = bundle
        self.call_threshold = call_threshold
        self.put_threshold = put_threshold
        self.validator = validator or ModelIntegrityValidator()

        # Validate bundle at initialization
        report = self.validator.validate_bundle(bundle)
        if not report.is_valid:
            raise ValueError(f"Invalid model bundle: {'; '.join(report.errors)}")

        model = bundle.model
        self._model_name = getattr(model, "name", model.__class__.__name__)
        self._feature_names = tuple(bundle.feature_names)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def infer(self, feature_row: FeatureRow) -> InferenceResult:
        """Execute full inference pipeline on an incoming causal FeatureRow."""
        start_time = time.perf_counter()
        candle_ts = feature_row.available_at_ts

        # 1. Schema check
        valid_schema, schema_err = self.validator.validate_feature_row_schema(
            self.bundle, feature_row
        )
        if not valid_schema:
            latency = (time.perf_counter() - start_time) * 1000.0
            return InferenceResult(
                direction=SignalDirection.HOLD,
                calibrated_probability=0.5,
                raw_probability=0.5,
                confidence=0.0,
                abstention_reason=f"SCHEMA_MISMATCH: {schema_err}",
                latency_ms=latency,
                candle_timestamp=candle_ts,
                model_name=self._model_name,
                feature_count=len(feature_row.features),
            )

        try:
            # 2. Extract feature vector in exact expected order
            feat_vector = np.array(
                [[float(feature_row.features[name]) for name in self._feature_names]],
                dtype=np.float64,
            )

            # 3. Apply StandardScaler if present
            if self.bundle.scaler is not None:
                transformed = self.bundle.scaler.transform(feat_vector)
                feat_vector = np.asarray(transformed, dtype=np.float64)

            # 4. Model predict_proba
            raw_probs = self.bundle.model.predict_proba(feat_vector)
            raw_prob_up = float(raw_probs[0, 1])

            # 5. Apply ProbabilityCalibrator if present
            if self.bundle.calibrator is not None:
                cal_dist = self.bundle.calibrator.calibrate(np.array([raw_prob_up]))
                cal_prob_up = float(cal_dist[0, 1])
            else:
                cal_prob_up = raw_prob_up

            # 6. Directional decision & abstention zone
            if cal_prob_up >= self.call_threshold:
                direction = SignalDirection.CALL
                effective_prob = cal_prob_up
                abstention = None
            elif cal_prob_up <= self.put_threshold:
                direction = SignalDirection.PUT
                effective_prob = 1.0 - cal_prob_up
                abstention = None
            else:
                direction = SignalDirection.HOLD
                effective_prob = cal_prob_up
                abstention = (
                    f"UNCERTAIN_ZONE: probability {cal_prob_up:.3f} within "
                    f"[{self.put_threshold:.2f}, {self.call_threshold:.2f}]"
                )

            confidence = abs(cal_prob_up - 0.5)
            latency = (time.perf_counter() - start_time) * 1000.0

            return InferenceResult(
                direction=direction,
                calibrated_probability=effective_prob,
                raw_probability=raw_prob_up,
                confidence=confidence,
                abstention_reason=abstention,
                latency_ms=latency,
                candle_timestamp=candle_ts,
                model_name=self._model_name,
                feature_count=len(self._feature_names),
            )

        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000.0
            logger.exception("Inference failed with unexpected exception")
            return InferenceResult(
                direction=SignalDirection.HOLD,
                calibrated_probability=0.5,
                raw_probability=0.5,
                confidence=0.0,
                abstention_reason=f"INFERENCE_EXCEPTION: {exc}",
                latency_ms=latency,
                candle_timestamp=candle_ts,
                model_name=self._model_name,
                feature_count=len(self._feature_names),
            )
