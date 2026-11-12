"""Unit and Integration Tests for Phase 4 Holdout Evaluation and MLStrategy Backtest Integration.

Exhaustive Category-Partition, Boundary Value Analysis, and Invariant verification
covering 28 test cases across 5 test suites:
- Suite 2.1: HoldoutEvaluator Mechanics & Split Mathematics (Cases 33-38)
- Suite 2.2: MLStrategy Instantiation & Configuration (Cases 39-43)
- Suite 2.3: MLStrategy.on_bar() Signal Generation & Causality (Cases 44-49)
- Suite 2.4: Trade Abstention Verification (Cases 50-54)
- Suite 2.5: End-to-End BacktestEngine Integration & Payoff Evaluations (Cases 55-60)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iqoption_research.ml.calibration.abstention import (
    BinaryPayoffGate,
    UncertaintyGate,
)
from iqoption_research.ml.decision.engine import (
    AIDecisionEngine,
    StakeConfig,
    StakeMode,
)
from iqoption_research.ml.decision.strategy import MLStrategy
from iqoption_research.ml.evaluation.holdout import (
    HoldoutEvaluator,
    HoldoutResult,
    HoldoutSplit,
    HoldoutSplitConfig,
)
from iqoption_research.ml.models.base import (
    BaseMLModel,
    NotFittedError,
)
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
)
from iqoption_research.ml.targets.dataset import MLDataset
from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
)
from iqoption_research.research.backtest.metrics import calculate_metrics
from iqoption_research.research.backtest.models import (
    Signal,
    SignalDirection,
)
from iqoption_research.research.backtest.payoffs import (
    BinaryPayoffModel,
    ForexPayoffModel,
    TieAction,
    TradeOutcome,
)
from iqoption_research.research.datasets.integrity import PointInTimeRecord
from iqoption_research.research.features.pipeline import (
    CausalFeaturePipeline,
    FeatureRow,
)
from iqoption_research.research.features.scalers import StandardScaler

# ==============================================================================
# Test Fixtures and Helpers
# ==============================================================================


def _make_synth_candle(
    from_ts: int,
    open_: float = 1.08500,
    close: float = 1.08520,
    volume: int = 10,
) -> Candle:
    hi = max(open_, close) + 0.00050
    lo = min(open_, close) - 0.00050
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=from_ts + 60,
        at_ts=(from_ts + 60) * 1_000_000_000,
        open=open_,
        high=hi,
        low=lo,
        close=close,
        volume=volume,
        bid=close - 0.00005,
        ask=close + 0.00005,
        is_complete=True,
        provenance="synthetic",
    )


_make_candle = _make_synth_candle


def _make_synth_row(
    index: int = 0,
    features: dict[str, float] | None = None,
    candle_from_ts: int = 1000,
    is_warmup: bool = False,
    close: float = 1.08520,
) -> FeatureRow:
    default_feats = {"return_1": 0.0002, "sma_ratio_5": 1.0001, "close": close}
    feats = features if features is not None else default_feats
    pit = PointInTimeRecord(
        candle_from_ts=candle_from_ts,
        candle_to_ts=candle_from_ts + 60,
        available_at_ts=candle_from_ts + 60,
        decision_ts=candle_from_ts + 60,
        execution_earliest_ts=candle_from_ts + 60,
    )
    return FeatureRow(
        index=index,
        symbol="EURUSD",
        candle_from_ts=candle_from_ts,
        candle_to_ts=candle_from_ts + 60,
        available_at_ts=candle_from_ts + 60,
        pit=pit,
        is_warmup=is_warmup,
        features=feats,
    )


def _make_synth_dataset(n: int = 200, n_features: int = 3, seed: int = 42) -> MLDataset:
    rng = np.random.default_rng(seed)
    feature_names = tuple(f"feat_{j}" for j in range(n_features))
    X = rng.normal(0.0, 1.0, size=(n, n_features)).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-1.2 * X[:, 0]))
    y = rng.binomial(1, p).astype(np.int64)
    y[0] = 0
    y[1] = 1

    base_ts = 1_700_000_000
    timestamps = np.array([base_ts + i * 60 for i in range(n)], dtype=np.int64)
    target_spec = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
    )
    return MLDataset(
        symbol="EURUSD",
        feature_names=feature_names,
        X=X,
        y=y,
        timestamps=timestamps,
        target_spec=target_spec,
    )


class MockPredictorModel(BaseMLModel):
    """Deterministic mock model returning controllable probabilities."""

    def __init__(
        self,
        prob_up: float = 0.50,
        feature_names: Sequence[str] = ("return_1", "sma_ratio_5"),
    ) -> None:
        super().__init__(name="mock_predictor_model", model_type="mock")
        self.prob_up = prob_up
        self._is_fitted = True
        self.feature_names_ = tuple(feature_names)
        self.sample_count_seen_ = 100

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> MockPredictorModel:
        self._is_fitted = True
        if feature_names:
            self.feature_names_ = tuple(feature_names)
        self.sample_count_seen_ = len(X)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        p1 = np.full(n, self.prob_up, dtype=np.float64)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def get_params(self) -> dict[str, Any]:
        return {"prob_up": self.prob_up}

    def save(self, directory: Path | str) -> Path:
        return Path(directory)

    @classmethod
    def load(cls, _directory: Path | str) -> MockPredictorModel:
        return cls()


class DynamicProbModel(BaseMLModel):
    """Mock model whose probabilities vary per call to test Kelly sizing."""

    def __init__(
        self,
        probs: Sequence[float],
        feature_names: Sequence[str] = ("return_1", "sma_ratio_5"),
    ) -> None:
        super().__init__(name="dynamic_prob_model", model_type="mock")
        self.probs = list(probs)
        self.call_idx = 0
        self._is_fitted = True
        self.feature_names_ = tuple(feature_names)
        self.sample_count_seen_ = 100

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> DynamicProbModel:
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_up = self.probs[min(self.call_idx, len(self.probs) - 1)]
        self.call_idx += 1
        return np.array([[1.0 - p_up, p_up]], dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def get_params(self) -> dict[str, Any]:
        return {}

    def save(self, directory: Path | str) -> Path:
        return Path(directory)

    @classmethod
    def load(cls, _directory: Path | str) -> DynamicProbModel:
        return cls([0.5])


# ==============================================================================
# Suite 2.1: HoldoutEvaluator Mechanics & Split Mathematics (6 Cases)
# ==============================================================================


class TestHoldoutSplitAndEvaluation:
    def test_holdout_split_default_proportions(self) -> None:
        """Case 33: Default 70/15/15 ratio correctly partitions 1000 samples."""
        dataset_len = 1000
        config = HoldoutSplitConfig(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=0,
            embargo_bars=0,
        )
        evaluator = HoldoutEvaluator(config)
        split = evaluator.get_indices(dataset_len)

        assert isinstance(split, HoldoutSplit)
        assert len(split.train_indices) == 700
        assert len(split.val_indices) == 150
        assert len(split.test_indices) == 150

        train_s = set(split.train_indices)
        val_s = set(split.val_indices)
        test_s = set(split.test_indices)

        assert train_s & val_s == set()
        assert train_s & test_s == set()
        assert val_s & test_s == set()

    def test_holdout_split_with_purging(self) -> None:
        """Case 34: Trailing bars before validation are omitted for purging."""
        dataset_len = 1000
        config = HoldoutSplitConfig(purge_bars=4)
        evaluator = HoldoutEvaluator(config)
        split = evaluator.get_indices(dataset_len)

        assert len(split.train_indices) == 696
        for dropped in (696, 697, 698, 699):
            assert dropped not in split.train_indices
        gap = int(np.min(split.val_indices)) - int(np.max(split.train_indices))
        assert gap == 5

    def test_holdout_split_with_embargo(self) -> None:
        """Case 35: Leading bars after validation are omitted for embargo."""
        dataset_len = 1000
        config = HoldoutSplitConfig(val_ratio=0.15, embargo_bars=5)
        evaluator = HoldoutEvaluator(config)
        split = evaluator.get_indices(dataset_len)

        assert int(np.max(split.val_indices)) == 849
        assert int(np.min(split.test_indices)) == 855
        gap = int(np.min(split.test_indices)) - int(np.max(split.val_indices))
        assert gap == 6

    def test_holdout_split_error_invalid_ratios(self) -> None:
        """Case 36: Split ratios not summing to 1.0 or non-positive raise ValueError."""
        with pytest.raises(ValueError, match=r"Ratios must sum to 1.0"):
            HoldoutSplitConfig(train_ratio=0.70, val_ratio=0.20, test_ratio=0.20)

        with pytest.raises(ValueError, match=r"Ratios must sum to 1.0"):
            HoldoutSplitConfig(train_ratio=-0.10, val_ratio=0.50, test_ratio=0.60)

    def test_holdout_evaluator_strict_causal_fit_and_isolation(self) -> None:
        """Case 37: Model is trained strictly on train partition without test leakage."""
        ds = _make_synth_dataset(n=200)

        class CausalTrackingModel(LogisticRegressionModel):
            def fit(
                self,
                X: np.ndarray,
                y: np.ndarray,
                feature_names: Sequence[str] | None = None,
            ) -> Any:
                super().fit(X, y, feature_names=feature_names)
                self.sample_count_seen_ = len(X)
                return self

        model = CausalTrackingModel()
        evaluator = HoldoutEvaluator()
        evaluator.evaluate(model, ds)

        split = evaluator.get_indices(ds)
        assert model.sample_count_seen_ == len(split.train_indices)

    def test_holdout_result_metrics_structure(self) -> None:
        """Case 38: HoldoutResult contains strongly typed metrics across partitions."""
        ds = _make_synth_dataset(n=200)
        evaluator = HoldoutEvaluator()
        result = evaluator.evaluate(LogisticRegressionModel(), ds)

        assert isinstance(result, HoldoutResult)
        assert 0.0 <= result.train_metrics.accuracy <= 1.0
        assert 0.0 <= result.val_metrics.brier_score <= 1.0
        assert result.test_metrics.log_loss >= 0.0
        split = evaluator.get_indices(ds)
        assert result.test_metrics.sample_count == len(split.test_indices)


# ==============================================================================
# Suite 2.2: MLStrategy Instantiation & Configuration (5 Cases)
# ==============================================================================


class TestMLStrategyContract:
    def test_ml_strategy_instantiation_defaults(self) -> None:
        """Case 39: Instantiation with fitted model and feature columns sets defaults."""
        model = MajorityClassModel()
        model.fit(
            np.zeros((10, 2)),
            np.array([0, 1] * 5),
            feature_names=("return_1", "sma_ratio_5"),
        )

        feature_cols = ["return_1", "sma_ratio_5"]
        strategy = MLStrategy(model=model, feature_columns=feature_cols)

        assert strategy.name == "MLStrategy"
        assert strategy.feature_columns == tuple(feature_cols)
        assert isinstance(strategy.decision_engine, AIDecisionEngine)
        assert strategy.scaler is None

    def test_ml_strategy_missing_feature_column_raises(self) -> None:
        """Case 40: Missing required feature column in on_bar raises KeyError/ValueError."""
        model = MajorityClassModel()
        model.fit(np.zeros((10, 1)), np.array([0, 1] * 5), feature_names=("non_existent_feature",))

        strategy = MLStrategy(model=model, feature_columns=["non_existent_feature"])
        sample_row = _make_synth_row(features={"return_1": 0.0001})

        with pytest.raises((KeyError, ValueError), match=r"non_existent_feature"):
            strategy.on_bar(sample_row, [])

    def test_ml_strategy_unfitted_model_raises(self) -> None:
        """Case 41: Attempting to instantiate MLStrategy with unfitted model raises."""
        unfitted_model = LogisticRegressionModel()
        with pytest.raises(NotFittedError):
            MLStrategy(model=unfitted_model, feature_columns=["return_1"])

    def test_ml_strategy_with_scaler_transformation(self) -> None:
        """Case 42: StandardScaler transforms feature row before feeding model."""
        X_train = np.array([[10.0, 100.0], [20.0, 200.0]], dtype=np.float64)
        scaler = StandardScaler().fit(X_train, feature_names=["feat_0", "feat_1"])

        seen_X: list[np.ndarray] = []

        class InterceptModel(BaseMLModel):
            def __init__(self) -> None:
                super().__init__(name="intercept", model_type="test")
                self._is_fitted = True
                self.feature_names_ = ("feat_0", "feat_1")

            def fit(
                self,
                X: np.ndarray,
                y: np.ndarray,
                feature_names: Sequence[str] | None = None,
            ) -> Any:
                return self

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                seen_X.append(X.copy())
                return np.array([[0.3, 0.7]], dtype=np.float64)

            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([1])

            def get_params(self) -> dict[str, Any]:
                return {}

            def save(self, directory: Path | str) -> Path:
                return Path(directory)

            @classmethod
            def load(cls, _directory: Path | str) -> InterceptModel:
                return cls()

        model = InterceptModel()
        strategy = MLStrategy(
            model=model,
            feature_columns=["feat_0", "feat_1"],
            scaler=scaler,
        )

        row = _make_synth_row(features={"feat_0": 15.0, "feat_1": 150.0, "close": 1.085})
        strategy.on_bar(row, [])

        expected_scaled = np.asarray(scaler.transform(np.array([[15.0, 150.0]])))
        np.testing.assert_allclose(seen_X[0], expected_scaled)

    def test_ml_strategy_reset_clears_internal_state(self) -> None:
        """Case 43: Reset clears execution counters while preserving model weights."""
        model = MajorityClassModel()
        model.fit(np.zeros((10, 1)), np.array([0, 1] * 5), feature_names=("return_1",))
        strategy = MLStrategy(model=model, feature_columns=["return_1"])

        row = _make_synth_row(features={"return_1": 0.0001, "close": 1.085})
        strategy.on_bar(row, [])
        assert strategy.bar_count == 1
        assert strategy.last_signal is not None

        strategy.reset()
        assert strategy.bar_count == 0
        assert strategy.last_signal is None
        assert strategy.model.is_fitted


# ==============================================================================
# Suite 2.3: MLStrategy.on_bar() Signal Generation & Causality (6 Cases)
# ==============================================================================


class TestMLStrategySignalGeneration:
    def test_ml_strategy_on_bar_returns_valid_signal(self) -> None:
        """Case 44: on_bar returns strongly typed Signal conforming to Phase 3."""
        model = MockPredictorModel(prob_up=0.75)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        row = _make_synth_row()

        sig = strategy.on_bar(row, [row])
        assert isinstance(sig, Signal)
        assert isinstance(sig.direction, SignalDirection)
        assert sig.horizon_bars >= 1
        assert isinstance(sig.metadata, dict)
        assert sig.metadata["decision_timestamp"] == row.candle_to_ts

    def test_ml_strategy_on_bar_determinism(self) -> None:
        """Case 45: Calling on_bar twice with identical inputs produces identical signals."""
        model = MockPredictorModel(prob_up=0.75)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        row = _make_synth_row()

        sig1 = strategy.on_bar(row, [row])
        strategy.reset()
        sig2 = strategy.on_bar(row, [row])

        assert sig1.direction == sig2.direction
        assert sig1.stake == sig2.stake
        assert sig1.metadata["raw_prob_up"] == sig2.metadata["raw_prob_up"]

    def test_ml_strategy_call_signal_on_high_prob_up(self) -> None:
        """Case 46: High P(UP) = 0.75 emits actionable CALL signal."""
        model = MockPredictorModel(prob_up=0.75)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        row = _make_synth_row()

        sig = strategy.on_bar(row, [row])
        assert sig.direction == SignalDirection.CALL
        assert sig.is_actionable is True
        assert sig.metadata["calibrated_prob_up"] == pytest.approx(0.75)

    def test_ml_strategy_put_signal_on_low_prob_up(self) -> None:
        """Case 47: Low P(UP) = 0.20 (P(DOWN) = 0.80) emits actionable PUT signal."""
        model = MockPredictorModel(prob_up=0.20)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        row = _make_synth_row()

        sig = strategy.on_bar(row, [row])
        assert sig.direction == SignalDirection.PUT
        assert sig.is_actionable is True
        assert sig.metadata["calibrated_prob_down"] == pytest.approx(0.80)

    def test_ml_strategy_temporal_causality_timestamp_alignment(self) -> None:
        """Case 48: Signal metadata timestamp strictly equals candle_to_ts."""
        model = MockPredictorModel(prob_up=0.75)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        row = _make_synth_row(candle_from_ts=1000, close=1.0855)

        sig = strategy.on_bar(row, [row])
        assert sig.metadata["decision_timestamp"] == 1060
        assert sig.metadata["current_price"] == pytest.approx(1.0855)

    def test_ml_strategy_does_not_mutate_history(self) -> None:
        """Case 49: Passing history into on_bar does not mutate sequence or elements."""
        model = MockPredictorModel(prob_up=0.75)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        history = [_make_synth_row(index=i, candle_from_ts=1000 + i * 60) for i in range(10)]
        history_copy = list(history)

        strategy.on_bar(history[-1], history)
        assert len(history) == 10
        assert history == history_copy


# ==============================================================================
# Suite 2.4: Trade Abstention Verification (5 Cases)
# ==============================================================================


class TestTradeAbstentionBehavior:
    def test_ml_strategy_abstains_on_50_50_probability(self) -> None:
        """Case 50: When probability is exactly 0.50, strategy emits HOLD."""
        model = MockPredictorModel(prob_up=0.50)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        row = _make_synth_row()

        sig = strategy.on_bar(row, [row])
        assert sig.direction == SignalDirection.HOLD
        assert sig.is_actionable is False
        assert sig.stake is None
        assert sig.metadata["passed_gates"] is False

    def test_ml_strategy_abstains_below_binary_break_even(self) -> None:
        """Case 51: Predicted edge below binary break-even triggers abstention."""
        # For payout = 0.85, break-even is 1 / 1.85 ~ 0.5405. P(UP) = 0.525 fails hurdle.
        model = MockPredictorModel(prob_up=0.525)
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.0)
        engine = AIDecisionEngine(abstention_gate=gate, payout_rate=0.85)
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            decision_engine=engine,
        )
        row = _make_synth_row()

        sig = strategy.on_bar(row, [row])
        assert sig.direction == SignalDirection.HOLD
        assert sig.is_actionable is False
        assert any(
            "break-even" in r.lower() or "sub_breakeven" in r.lower()
            for r in sig.metadata["abstention_reasons"]
        )

    def test_ml_strategy_abstains_on_uncertainty_entropy(self) -> None:
        """Case 52: Prediction entropy exceeding threshold triggers abstention."""
        # P(UP) = 0.51 has entropy ~ 0.999 bits > 0.95 bits max threshold
        model = MockPredictorModel(prob_up=0.51)
        gate = UncertaintyGate(max_entropy=0.95)
        engine = AIDecisionEngine(abstention_gate=gate)
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            decision_engine=engine,
        )
        row = _make_synth_row()

        sig = strategy.on_bar(row, [row])
        assert sig.direction == SignalDirection.HOLD
        assert sig.metadata["entropy"] > 0.95

    def test_ml_strategy_abstains_when_stake_below_minimum(self) -> None:
        """Case 53: Kelly sizing yielding stake below min_stake forces HOLD."""
        # Slight edge: P=0.55 on payout=0.85 gives full Kelly f* ~ 0.0205.
        # With default_capital=100 and min_stake=10, raw stake ~ 2.05 < 10.0 -> HOLD
        model = MockPredictorModel(prob_up=0.55)
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.0)
        stake_cfg = StakeConfig(
            mode=StakeMode.KELLY,
            min_stake=10.0,
            default_capital=100.0,
            payout_rate=0.85,
            kelly_fraction=1.0,
        )
        engine = AIDecisionEngine(
            abstention_gate=gate,
            stake_config=stake_cfg,
            payout_rate=0.85,
        )
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            decision_engine=engine,
        )
        row = _make_synth_row()

        sig = strategy.on_bar(row, [row])
        assert sig.direction == SignalDirection.HOLD
        assert sig.stake is None

    def test_backtest_engine_opens_zero_positions_on_hold(self) -> None:
        """Case 54: When strategy emits HOLD, BacktestEngine executes zero trades."""
        model = MockPredictorModel(prob_up=0.50)  # Always HOLD
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])

        candles = [_make_candle(100 + i * 60, 1.0850, 1.0855) for i in range(25)]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        engine = BacktestEngine(
            strategy=strategy,
            payoff_model=BinaryPayoffModel(),
            config=BacktestConfig(initial_capital=1000.0),
        )
        res = engine.run(feat_ds.rows, candles)

        assert res.trade_count == 0
        assert res.final_capital == res.initial_capital
        assert res.total_net_pnl == 0.0


# ==============================================================================
# Suite 2.5: End-to-End BacktestEngine Integration & Payoff Evaluations (6 Cases)
# ==============================================================================


class TestBacktestEngineIntegration:
    def test_e2e_backtest_ml_strategy_binary_payoff_accounting(self) -> None:
        """Case 55: Binary option payoff simulation and cash accounting."""
        candles = [
            _make_candle(
                100 + i * 60,
                1.0850,
                1.0860 if i % 2 == 0 else 1.0840,
            )
            for i in range(30)
        ]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        # High confidence P=0.75 ensures CALL signals
        model = MockPredictorModel(prob_up=0.75)
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            default_horizon_bars=1,
            payout_rate=0.85,
        )
        payoff = BinaryPayoffModel(payout_rate=0.85, tie_action=TieAction.REFUND)
        cfg = BacktestConfig(initial_capital=1000.0, stake_amount=100.0)

        engine = BacktestEngine(strategy=strategy, payoff_model=payoff, config=cfg)
        res = engine.run(feat_ds.rows, candles)

        assert res.trade_count > 0
        for t in res.trades:
            if t.settlement.outcome == TradeOutcome.WIN:
                assert t.settlement.net_pnl == pytest.approx(85.0)
            elif t.settlement.outcome == TradeOutcome.LOSS:
                assert t.settlement.net_pnl == pytest.approx(-100.0)
            assert t.capital_after == pytest.approx(t.capital_before + t.settlement.net_pnl)
        assert res.final_capital == pytest.approx(res.initial_capital + res.total_net_pnl)

    def test_e2e_backtest_ml_strategy_forex_payoff_costs(self) -> None:
        """Case 56: Forex payoff simulation applies spread and slippage friction."""
        candles = [_make_candle(100 + i * 60, 1.0850, 1.0870) for i in range(30)]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        model = MockPredictorModel(prob_up=0.80)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        payoff = ForexPayoffModel(pip_size=0.0001, leverage=1.0)
        cfg = BacktestConfig(
            spread_pips=1.5,
            slippage_pips=0.5,
            stake_amount=1000.0,
            default_horizon_bars=1,
        )

        engine = BacktestEngine(strategy=strategy, payoff_model=payoff, config=cfg)
        res = engine.run(feat_ds.rows, candles)

        assert res.trade_count > 0
        for t in res.trades:
            assert t.settlement.costs > 0
            assert t.settlement.net_pnl < t.settlement.gross_pnl

    def test_e2e_backtest_ml_strategy_respects_circuit_breaker(self) -> None:
        """Case 57: Drawdown circuit breaker stops execution when max drawdown exceeded."""
        candles = [
            _make_candle(
                100 + i * 60,
                1.0850 - max(0, i - 19) * 0.0010,
                1.0840 - max(0, i - 19) * 0.0010,
            )
            for i in range(30)
        ]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        model = MockPredictorModel(prob_up=0.80)  # Always CALL, will lose continuously
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        payoff = BinaryPayoffModel(payout_rate=0.85)
        cfg = BacktestConfig(
            initial_capital=1000.0,
            stake_amount=100.0,
            max_drawdown_stop_ratio=0.20,
        )

        engine = BacktestEngine(strategy=strategy, payoff_model=payoff, config=cfg)
        res = engine.run(feat_ds.rows, candles)

        assert res.trade_count <= 3
        final_dd = (res.initial_capital - res.final_capital) / res.initial_capital
        assert final_dd >= 0.20

    def test_e2e_backtest_zero_lookahead_execution_timing(self) -> None:
        """Case 58: Trades enter at entry bar and exit at entry + horizon_bars."""
        candles = [_make_candle(100 + i * 60, 1.0850, 1.0860) for i in range(30)]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        model = MockPredictorModel(prob_up=0.80)
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            default_horizon_bars=2,
        )
        payoff = BinaryPayoffModel()
        cfg = BacktestConfig(latency_bars=0, default_horizon_bars=2)

        engine = BacktestEngine(strategy=strategy, payoff_model=payoff, config=cfg)
        res = engine.run(feat_ds.rows, candles)

        assert res.trade_count > 0
        completed_trades = [t for t in res.trades if t.exit_bar_index < len(candles) - 1]
        assert len(completed_trades) > 0
        for t in completed_trades:
            assert t.exit_bar_index == t.entry_bar_index + 2
            assert t.exit_ts > t.entry_ts

    def test_e2e_backtest_kelly_stake_sizing_integration(self) -> None:
        """Case 59: Higher model confidence produces higher Kelly position stake."""
        candles = [_make_candle(100 + i * 60, 1.0850, 1.0860) for i in range(25)]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        model = DynamicProbModel(probs=[0.60, 0.75])
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.0)
        stake_cfg = StakeConfig(
            mode=StakeMode.KELLY,
            payout_rate=0.85,
            default_capital=1000.0,
            kelly_fraction=1.0,
            max_stake_fraction=0.50,
        )
        decision_engine = AIDecisionEngine(
            abstention_gate=gate,
            stake_config=stake_cfg,
            payout_rate=0.85,
        )
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            decision_engine=decision_engine,
        )

        row0 = feat_ds.rows[22]
        row1 = feat_ds.rows[23]

        sig0 = strategy.on_bar(row0, feat_ds.rows[:23])
        sig1 = strategy.on_bar(row1, feat_ds.rows[:24])

        assert sig0.stake is not None and sig1.stake is not None
        assert sig1.stake > sig0.stake

    def test_e2e_backtest_metrics_calculation_parity(self) -> None:
        """Case 60: Backtest run result produces consistent statistical evaluation metrics."""
        candles = [
            _make_candle(100 + i * 60, 1.0850, 1.0850 + (0.0005 if i % 2 == 0 else -0.0005))
            for i in range(25)
        ]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        model = MockPredictorModel(prob_up=0.75)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        payoff = BinaryPayoffModel(payout_rate=0.85)

        res = BacktestEngine(strategy, payoff).run(feat_ds.rows, candles)
        metrics = calculate_metrics(res)

        assert 0.0 <= metrics.win_rate <= 1.0
        assert metrics.win_rate_ci_95.lower <= metrics.win_rate <= metrics.win_rate_ci_95.upper
        assert metrics.total_trades == res.trade_count
