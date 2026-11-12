"""Machine Learning Trading Strategy Adapter for Phase 3 BacktestEngine.

Bridges a fitted BaseMLModel, optional frozen StandardScaler, ProbabilityCalibrator,
and AIDecisionEngine into BaseStrategy.on_bar(current_row, history).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.decision.engine import AIDecisionEngine
from iqoption_research.ml.models.base import BaseMLModel, ModelValidationError, NotFittedError
from iqoption_research.research.backtest.models import Signal, SignalDirection
from iqoption_research.research.backtest.strategies.base import BaseStrategy
from iqoption_research.research.features.pipeline import FeatureRow
from iqoption_research.research.features.scalers import PreprocessingError, StandardScaler


class MLStrategy(BaseStrategy):
    """Machine learning trading strategy adapter for the Phase 3 offline backtester.

    Encapsulates a fitted BaseMLModel, optional frozen StandardScaler,
    ProbabilityCalibrator, and AIDecisionEngine to evaluate incoming causal
    FeatureRows and synthesize risk-controlled Signal instances.
    """

    def __init__(
        self,
        model: BaseMLModel,
        *,
        decision_engine: AIDecisionEngine | None = None,
        calibrator: ProbabilityCalibrator | None = None,
        feature_names: Sequence[str] | None = None,
        feature_columns: Sequence[str] | None = None,
        scaler: StandardScaler | None = None,
        name: str = "MLStrategy",
        description: str = "ML-driven directional trading strategy",
        default_horizon_bars: int = 1,
        payout_rate: float = 0.85,
    ) -> None:
        super().__init__(name=name, description=description)

        if not isinstance(model, BaseMLModel):
            raise ModelValidationError(
                f"model must be an instance of BaseMLModel, got {type(model).__name__}"
            )
        if not model.is_fitted:
            raise NotFittedError(
                f"Model '{model.name}' must be fitted before initializing MLStrategy."
            )

        self._model = model

        # Harmonize feature_names and feature_columns. Models may expose their
        # fitted schema as either `feature_names` (BaseMLModel property) or the
        # scikit-learn-style `feature_names_` attribute used by adapters/mocks.
        resolved_feats: Sequence[str] | None = (
            feature_names if feature_names is not None else feature_columns
        )

        model_feats: tuple[str, ...] = ()
        for attr in ("feature_names", "feature_names_"):
            candidate = getattr(model, attr, None)
            if candidate:
                model_feats = tuple(candidate)
                break

        if resolved_feats is not None:
            self._feature_names: tuple[str, ...] = tuple(resolved_feats)
            if model_feats and model_feats != self._feature_names:
                raise ModelValidationError(
                    f"Provided feature names {self._feature_names} do not match "
                    f"model's fitted feature names {model_feats}."
                )
        elif model_feats:
            self._feature_names = model_feats
        else:
            raise ModelValidationError(
                "feature_names must be provided explicitly or defined on the fitted model "
                "(model.feature_names / model.feature_names_)."
            )

        if len(self._feature_names) == 0:
            raise ModelValidationError("feature_names cannot be empty.")

        # Scaler validation
        if scaler is not None:
            if not scaler.is_fitted:
                raise PreprocessingError(
                    "StandardScaler must be fitted before initializing MLStrategy."
                )
            scaler_feats = tuple(scaler.params.feature_names)
            if scaler_feats != self._feature_names:
                raise PreprocessingError(
                    f"Scaler feature names {scaler_feats} do not match strategy "
                    f"feature names {self._feature_names}."
                )
        self._scaler = scaler

        # Calibrator validation
        if calibrator is not None and not calibrator.is_fitted:
            raise NotFittedError(
                "ProbabilityCalibrator must be fitted before initializing MLStrategy."
            )
        self._calibrator = calibrator

        # Decision engine resolution
        if decision_engine is not None:
            self._decision_engine = decision_engine
            if self._decision_engine.calibrator is None and calibrator is not None:
                self._decision_engine.calibrator = calibrator
        else:
            self._decision_engine = AIDecisionEngine(
                calibrator=calibrator,
                default_horizon_bars=default_horizon_bars,
                payout_rate=payout_rate,
            )

        self.default_horizon_bars = int(default_horizon_bars)
        self.payout_rate = float(payout_rate)

        # Execution state
        self._bar_count: int = 0
        self._last_signal: Signal | None = None

    @property
    def model(self) -> BaseMLModel:
        return self._model

    @property
    def decision_engine(self) -> AIDecisionEngine:
        return self._decision_engine

    @property
    def calibrator(self) -> ProbabilityCalibrator | None:
        return self._calibrator

    @property
    def scaler(self) -> StandardScaler | None:
        return self._scaler

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Alias for feature_names to support test suite ergonomics."""
        return self._feature_names

    @property
    def bar_count(self) -> int:
        return self._bar_count

    @property
    def last_signal(self) -> Signal | None:
        return self._last_signal

    def reset(self) -> None:
        """Reset internal execution state between simulation runs."""
        super().reset()
        self._bar_count = 0
        self._last_signal = None

    def on_bar(self, current_row: FeatureRow, history: Sequence[FeatureRow]) -> Signal:
        """Evaluate point-in-time completed bar and past history to produce a trading signal.

        Args:
            current_row: Most recently completed bar with causal features.
            history: Sequence of historical completed FeatureRows strictly prior to or
                     including current_row. Future rows are NEVER accessible.

        Returns:
            Signal specifying direction (CALL/PUT/HOLD), optional stake, and horizon.
        """
        _ = history
        self._bar_count += 1

        price = (
            float(current_row.pit.close)
            if hasattr(current_row, "pit") and hasattr(current_row.pit, "close")
            else float(current_row.features.get("close", 0.0))
        )

        # 1. Warmup guard: do not predict during incomplete lookback window
        if current_row.is_warmup:
            sig = Signal(
                direction=SignalDirection.HOLD,
                stake=None,
                horizon_bars=self.default_horizon_bars,
                metadata={
                    "is_warmup": True,
                    "abstention_reasons": ["Warmup bar: causal feature lookback incomplete"],
                    "bar_index": current_row.index,
                    "candle_to_ts": current_row.candle_to_ts,
                    "decision_timestamp": current_row.candle_to_ts,
                    "current_price": price,
                },
            )
            self._last_signal = sig
            return sig

        # 2. Extract and validate required features
        feats = current_row.features
        raw_vals: list[float] = []

        for fn in self._feature_names:
            if fn not in feats:
                raise KeyError(f"FeatureRow is missing required feature '{fn}'")
            val = feats[fn]
            if not math.isfinite(val):
                sig = Signal(
                    direction=SignalDirection.HOLD,
                    stake=None,
                    horizon_bars=self.default_horizon_bars,
                    metadata={
                        "is_warmup": False,
                        "abstention_reasons": [f"Non-finite feature value for '{fn}': {val}"],
                        "bar_index": current_row.index,
                        "candle_to_ts": current_row.candle_to_ts,
                        "decision_timestamp": current_row.candle_to_ts,
                        "current_price": price,
                    },
                )
                self._last_signal = sig
                return sig
            raw_vals.append(float(val))

        # 3. Vectorization and optional scaling
        X_raw = np.array([raw_vals], dtype=np.float64)
        if self._scaler is not None:
            transformed = self._scaler.transform(X_raw)
            X_eval = np.asarray(transformed, dtype=np.float64)
        else:
            X_eval = X_raw

        # 4. Model probability inference
        proba = self._model.predict_proba(X_eval)
        raw_prob_up = float(proba[0, 1])

        # 5. Extract point-in-time market context
        quote_spread = float(feats.get("spread_pips", 0.0))
        candle_to_ts = int(current_row.candle_to_ts)

        # 6. AIDecisionEngine synthesis
        signal = self._decision_engine.decide(
            prob_up=raw_prob_up,
            current_price=price,
            candle_to_ts=candle_to_ts,
            quote_spread_pips=quote_spread,
            horizon_bars=self.default_horizon_bars,
        )

        # 7. Augment telemetry
        augmented_meta = dict(signal.metadata)
        augmented_meta.update({
            "strategy_name": self.name,
            "model_name": self._model.name,
            "bar_index": current_row.index,
            "is_warmup": False,
            "feature_dim": len(self._feature_names),
            "decision_timestamp": candle_to_ts,
            "current_price": price,
        })

        final_signal = Signal(
            direction=signal.direction,
            stake=signal.stake,
            horizon_bars=signal.horizon_bars,
            metadata=augmented_meta,
        )
        self._last_signal = final_signal
        return final_signal
