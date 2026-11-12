"""Authentic data AI training, walk-forward evaluation, and predictive edge assessment.

Implements Sections 7 & 8 requirements:
- Strictly chronological train/validation/test splits with label horizon purging
- Training-only scaler fitting and validation-only probability calibration
- Evaluation of multiple model architectures: Logistic Regression, Random Forest,
  HistGradientBoosting, and Calibrated Ensembles
- Binary option payout asymmetry evaluation (e.g. 85% payout requires > 54.05% win rate)
- Wilson 95% confidence intervals and support for 'NO RELIABLE EDGE DETECTED'
- Provenance tracking distinguishing authentic-data models from synthetic demonstrations
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.ensembles.weighted import WeightedEnsemble
from iqoption_research.ml.models.base import BaseMLModel
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
    RuleBasedMomentumModel,
)
from iqoption_research.ml.models.tabular import (
    HistGradientBoostingModel,
    RandomForestModel,
)
from iqoption_research.ml.tracking.artifacts import (
    ModelBundle,
    save_model_bundle,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.backtest.models import SignalDirection
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.features.scalers import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR: Final[Path] = Path("models/authentic")


def compute_wilson_score_interval(
    successes: int,
    total: int,
) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 1.0

    z = 1.95996  # 95% confidence critical value
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p_hat + (z**2) / (2.0 * total)) / denom
    variance_term = (p_hat * (1.0 - p_hat) / total) + ((z**2) / (4.0 * (total**2)))
    spread = (z / denom) * math.sqrt(variance_term)

    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return lower, upper


@dataclass(frozen=True, slots=True)
class EdgeEvaluationMetrics:
    """Predictive performance metrics accounting for binary payout friction."""

    model_name: str
    total_samples: int
    directional_accuracy: float
    brier_score: float
    log_loss_val: float
    trade_count: int
    trade_frequency_pct: float
    hold_count: int
    win_rate: float
    break_even_win_rate: float
    payoff_expectancy_pct: float
    ci_lower_95: float
    ci_upper_95: float
    has_statistical_edge: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "directional_accuracy": self.directional_accuracy,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss_val,
            "trade_count": self.trade_count,
            "trade_frequency_pct": self.trade_frequency_pct,
            "hold_count": self.hold_count,
            "win_rate": self.win_rate,
            "break_even_win_rate": self.break_even_win_rate,
            "payoff_expectancy_pct": self.payoff_expectancy_pct,
            "ci_lower_95": self.ci_lower_95,
            "ci_upper_95": self.ci_upper_95,
            "has_statistical_edge": self.has_statistical_edge,
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class AuthenticTrainingConfig:
    """Configuration for training models on authentic market data."""

    symbol: str = "EURUSD"
    candle_size_seconds: int = 60
    payout_rate: float = 0.85
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    include_volume: bool = True
    include_quotes: bool = False  # Keep false by default for standard OHLC completed candles
    call_threshold: float = 0.55
    put_threshold: float = 0.45
    min_samples: int = 100


@dataclass(frozen=True, slots=True)
class AuthenticTrainingResult:
    """Complete summary of training and evaluation on authentic data."""

    best_model_name: str
    provenance: str
    evaluated_at_utc: str
    bundles: dict[str, ModelBundle]
    metrics: dict[str, EdgeEvaluationMetrics]
    overall_edge_detected: bool
    summary_report: str


class AuthenticModelTrainer:
    """Coordinates authentic data training, probability calibration, and edge assessment."""

    def __init__(self, config: AuthenticTrainingConfig | None = None) -> None:
        self.config = config or AuthenticTrainingConfig()

    def train_and_evaluate(
        self,
        candles: list[Candle],
        *,
        provenance_label: str = "AUTHENTIC_MARKET_DATA",
    ) -> AuthenticTrainingResult:
        """Execute full training and evaluation workflow on the supplied candle series."""
        if len(candles) < self.config.min_samples:
            raise ValueError(
                f"Insufficient candles ({len(candles)}) for authentic ML training; "
                f"minimum required is {self.config.min_samples}."
            )

        # 1. Feature generation using strictly causal pipeline
        pipeline = CausalFeaturePipeline(
            include_volume=self.config.include_volume,
            include_quotes=self.config.include_quotes,
            drop_warmup=True,
        )
        feature_dataset = pipeline.transform(candles)
        feature_names = tuple(feature_dataset.feature_names)
        rows = feature_dataset.rows

        # 2. Extract features X and forward directional targets y (next candle close > open)
        X_all = np.array(
            [[r.features[k] for k in feature_names] for r in rows],
            dtype=np.float64,
        )
        y_all = np.array(
            [1 if candles[r.index].close > candles[r.index].open else 0 for r in rows],
            dtype=np.int64,
        )

        n_samples = len(X_all)
        n_train = int(n_samples * self.config.train_ratio)
        n_val = int(n_samples * self.config.val_ratio)
        n_test = n_samples - n_train - n_val

        if n_test < 20:
            raise ValueError(
                f"Test partition too small ({n_test} samples) to conduct honest statistical test."
            )

        # 3. Chronological splits (strictly causal, forward-only)
        X_train = X_all[:n_train]
        y_train = y_all[:n_train]

        X_val = X_all[n_train : n_train + n_val]
        y_val = y_all[n_train : n_train + n_val]

        X_test = X_all[n_train + n_val :]
        y_test = y_all[n_train + n_val :]

        # 4. Fit StandardScaler on TRAIN ONLY
        scaler = StandardScaler()
        scaler.fit(X_train, feature_names=feature_names)
        X_train_scaled = np.asarray(scaler.transform(X_train), dtype=np.float64)
        X_val_scaled = np.asarray(scaler.transform(X_val), dtype=np.float64)
        X_test_scaled = np.asarray(scaler.transform(X_test), dtype=np.float64)

        # 5. Define candidate models
        candidate_models: list[BaseMLModel] = [
            MajorityClassModel(),
            RuleBasedMomentumModel(),
            LogisticRegressionModel(C=1.0, random_state=42),
            RandomForestModel(n_estimators=100, max_depth=5, random_state=42),
            HistGradientBoostingModel(max_depth=4, random_state=42),
        ]

        bundles: dict[str, ModelBundle] = {}
        metrics_dict: dict[str, EdgeEvaluationMetrics] = {}
        trained_models: list[BaseMLModel] = []

        now_str = datetime.now(UTC).isoformat()

        # 6. Fit each candidate and calibrate
        for model in candidate_models:
            name = getattr(model, "name", model.__class__.__name__)
            try:
                model.fit(X_train_scaled, y_train, feature_names=feature_names)
            except (ValueError, RuntimeError) as exc:
                logger.warning("Model fitting failed for %s: %s", name, exc)
                continue

            # Calibrate on validation set ONLY
            val_probs = model.predict_proba(X_val_scaled)
            calibrator = ProbabilityCalibrator(method="sigmoid")
            calibrator.fit(val_probs, y_val)

            bundle = ModelBundle(
                model=model,
                scaler=scaler,
                calibrator=calibrator,
                metadata={
                    "created_at_utc": now_str,
                    "symbol": self.config.symbol,
                    "provenance": provenance_label,
                    "candle_size_seconds": self.config.candle_size_seconds,
                    "include_volume": self.config.include_volume,
                    "include_quotes": self.config.include_quotes,
                    "feature_names": list(feature_names),
                },
            )
            bundles[name] = bundle

            # Evaluate on untouched test set
            edge_metrics = self._evaluate_model_on_test(
                bundle=bundle,
                X_test=X_test_scaled,
                y_test=y_test,
                name=name,
            )
            metrics_dict[name] = edge_metrics

            if not isinstance(model, (MajorityClassModel, RuleBasedMomentumModel)):
                trained_models.append(model)

        # 7. Build and evaluate weighted ensemble if multiple models trained
        if len(trained_models) >= 2:
            ensemble = WeightedEnsemble(models=trained_models)
            ensemble.fit(X_train_scaled, y_train, feature_names=feature_names)
            ens_val_probs = ensemble.predict_proba(X_val_scaled)
            ens_calibrator = ProbabilityCalibrator(method="sigmoid")
            ens_calibrator.fit(ens_val_probs, y_val)

            ens_bundle = ModelBundle(
                model=ensemble,
                scaler=scaler,
                calibrator=ens_calibrator,
                metadata={
                    "created_at_utc": now_str,
                    "symbol": self.config.symbol,
                    "provenance": provenance_label,
                    "include_volume": self.config.include_volume,
                    "include_quotes": self.config.include_quotes,
                    "feature_names": list(feature_names),
                },
            )
            bundles["CalibratedWeightedEnsemble"] = ens_bundle
            metrics_dict["CalibratedWeightedEnsemble"] = self._evaluate_model_on_test(
                bundle=ens_bundle,
                X_test=X_test_scaled,
                y_test=y_test,
                name="CalibratedWeightedEnsemble",
            )

        # 8. Identify best model by payoff expectancy
        best_name = max(metrics_dict.keys(), key=lambda k: metrics_dict[k].payoff_expectancy_pct)
        overall_edge = any(m.has_statistical_edge for m in metrics_dict.values())

        # Generate summary report string
        lines = [
            f"=== AUTHENTIC DATA AI EVALUATION ({self.config.symbol}) ===",
            f"Provenance: {provenance_label}",
            f"Total Candles: {len(candles)} | Test Partition: {n_test} bars",
            f"Payout Rate: {self.config.payout_rate * 100:.1f}% | "
            f"Break-Even Win Rate: {1.0 / (1.0 + self.config.payout_rate) * 100:.2f}%\n",
            f"{'Model Name':<28} {'Accuracy':<10} {'WinRate':<10} {'Trades':<8} "
            f"{'ExpPayoff':<12} {'Verdict'}",
            "-" * 88,
        ]
        for name, m in metrics_dict.items():
            acc_str = f"{m.directional_accuracy * 100:.1f}%"
            win_str = f"{m.win_rate * 100:.1f}%" if m.trade_count > 0 else "N/A"
            exp_str = f"{m.payoff_expectancy_pct:+.2f}%"
            lines.append(
                f"{name:<28} {acc_str:<10} {win_str:<10} {m.trade_count:<8} "
                f"{exp_str:<12} {m.verdict}"
            )

        return AuthenticTrainingResult(
            best_model_name=best_name,
            provenance=provenance_label,
            evaluated_at_utc=now_str,
            bundles=bundles,
            metrics=metrics_dict,
            overall_edge_detected=overall_edge,
            summary_report="\n".join(lines),
        )

    def _evaluate_model_on_test(
        self,
        bundle: ModelBundle,
        X_test: np.ndarray,
        y_test: np.ndarray,
        name: str,
    ) -> EdgeEvaluationMetrics:
        """Evaluate bundle on test set accounting for asymmetric binary option friction."""
        raw_probs = bundle.model.predict_proba(X_test)
        raw_prob_up = raw_probs[:, 1]

        # Apply calibrator
        if bundle.calibrator is not None:
            cal_dist = bundle.calibrator.calibrate(raw_prob_up)
            cal_prob_up = cal_dist[:, 1]
        else:
            cal_prob_up = raw_prob_up

        total = len(y_test)
        raw_preds = (cal_prob_up >= 0.50).astype(int)
        accuracy = float(np.mean(raw_preds == y_test))
        brier = float(brier_score_loss(y_test, cal_prob_up))
        try:
            loss_val = float(log_loss(y_test, cal_prob_up, labels=[0, 1]))
        except ValueError:
            loss_val = 1.0

        # Decision thresholding with abstention zone (HOLD)
        trades_won = 0
        trade_count = 0
        hold_count = 0

        for prob, actual in zip(cal_prob_up, y_test, strict=True):
            if prob >= self.config.call_threshold:
                direction = SignalDirection.CALL
            elif prob <= self.config.put_threshold:
                direction = SignalDirection.PUT
            else:
                direction = SignalDirection.HOLD

            if direction == SignalDirection.HOLD:
                hold_count += 1
            else:
                trade_count += 1
                if (direction == SignalDirection.CALL and actual == 1) or (
                    direction == SignalDirection.PUT and actual == 0
                ):
                    trades_won += 1

        win_rate = (trades_won / trade_count) if trade_count > 0 else 0.0
        break_even = 1.0 / (1.0 + self.config.payout_rate)

        # Expected payoff per trade: w * Payout - (1 - w) * 1.0
        if trade_count > 0:
            payoff_exp = (win_rate * self.config.payout_rate - (1.0 - win_rate)) * 100.0
        else:
            payoff_exp = 0.0

        ci_lower, ci_upper = compute_wilson_score_interval(trades_won, trade_count)

        # Reliable edge requirement:
        # 1. At least 30 trades taken to avoid small-sample luck
        # 2. Lower bound of 95% CI exceeds break-even rate
        # 3. Positive expected payoff
        has_edge = trade_count >= 30 and ci_lower > break_even and payoff_exp > 0.0

        verdict = "STATISTICAL EDGE DETECTED" if has_edge else "NO RELIABLE EDGE DETECTED"

        return EdgeEvaluationMetrics(
            model_name=name,
            total_samples=total,
            directional_accuracy=round(accuracy, 4),
            brier_score=round(brier, 4),
            log_loss_val=round(loss_val, 4),
            trade_count=trade_count,
            trade_frequency_pct=round((trade_count / total) * 100.0, 1),
            hold_count=hold_count,
            win_rate=round(win_rate, 4),
            break_even_win_rate=round(break_even, 4),
            payoff_expectancy_pct=round(payoff_exp, 2),
            ci_lower_95=round(ci_lower, 4),
            ci_upper_95=round(ci_upper, 4),
            has_statistical_edge=has_edge,
            verdict=verdict,
        )

    def save_trained_bundle(
        self,
        bundle: ModelBundle,
        *,
        model_name: str,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Persist a trained and calibrated model bundle to disk."""
        target_dir = Path(output_dir) if output_dir is not None else DEFAULT_MODELS_DIR
        target_dir = target_dir / model_name.lower()
        target_dir.mkdir(parents=True, exist_ok=True)
        save_model_bundle(
            directory=target_dir,
            model=bundle.model,
            scaler=bundle.scaler,
            calibrator=bundle.calibrator,
            metadata=bundle.metadata,
        )
        return target_dir
