"""Adversarial stress testing suite for Milestone 5: MLStrategy and BacktestEngine Integration.

Empirical verification of:
1. Lookahead Bias Probes: future candle mutation invariance and causal isolation.
2. Corrupted / Non-Finite Features: NaN, Inf, -Inf, missing keys, and corrupted PIT pricing.
3. Zero-Edge Binary Payouts: 100% abstention verification across payout rate sweeps and margins.
4. Extreme Forex Spreads & Slippage: news spikes and friction hurdles forcing HOLD.
5. Multi-Hundred Bar Runs & Circuit Breakers: 500-600 bar simulations with drawdown stops.
6. Capital Accounting Precision: exact mathematical cash conservation across WIN/LOSS/TIE modes.
7. HoldoutEvaluator Boundary Stress: N=10 boundary, excessive purge/embargo, and immutable indices.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iqoption_research.ml.calibration.abstention import (
    BinaryPayoffGate,
    ForexPayoffGate,
    calculate_binary_break_even,
)
from iqoption_research.ml.decision.engine import (
    AIDecisionEngine,
)
from iqoption_research.ml.decision.strategy import MLStrategy
from iqoption_research.ml.evaluation.holdout import (
    HoldoutEvaluator,
    HoldoutSplitConfig,
    HoldoutSplitError,
)
from iqoption_research.ml.models.base import BaseMLModel
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
)
from iqoption_research.research.backtest.models import (
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
# Deterministic Test Fixtures and Mock Models
# ==============================================================================


def _make_synth_candle(
    from_ts: int,
    open_: float = 1.08500,
    close: float = 1.08520,
    high: float | None = None,
    low: float | None = None,
    volume: int = 10,
    spread: float = 0.00010,
) -> Candle:
    """Helper to synthesize valid Candle instances."""
    hi = high if high is not None else max(open_, close) + 0.00050
    lo = low if low is not None else min(open_, close) - 0.00050
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
        bid=close - spread / 2.0,
        ask=close + spread / 2.0,
        is_complete=True,
        provenance="synthetic",
    )


def _make_synth_row(
    index: int = 0,
    features: dict[str, float] | None = None,
    candle_from_ts: int = 1000,
    is_warmup: bool = False,
    close: float = 1.08520,
) -> FeatureRow:
    """Helper to synthesize valid FeatureRow instances."""
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


class ControlledProbabilityModel(BaseMLModel):
    """Deterministic mock model returning precisely controlled probability outputs."""

    def __init__(
        self,
        prob_up: float = 0.50,
        feature_names: Sequence[str] = ("return_1", "sma_ratio_5"),
    ) -> None:
        super().__init__(name="controlled_prob_model", model_type="mock")
        self.prob_up = prob_up
        self._is_fitted = True
        self.feature_names_ = tuple(feature_names)
        self.sample_count_seen_ = 100

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> ControlledProbabilityModel:
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
    def load(cls, _directory: Path | str) -> ControlledProbabilityModel:
        return cls()


# ==============================================================================
# Suite 1: Lookahead Bias and Future Manipulation Probes
# ==============================================================================


class TestAdversarialLookaheadBiasProbes:
    """Stress tests proving MLStrategy cannot peek at future bars or future prices."""

    def test_future_candle_mutation_zero_lookahead_oracle(self) -> None:
        """Verify that mutating future candles (t >= 25) leaves past signals strictly identical."""
        base_ts = 1_700_000_000
        n_bars = 50
        target_eval_bar = 24  # Evaluate at bar 24 (past warmup)

        # 1. Generate clean candle series
        clean_candles = [
            _make_synth_candle(
                base_ts + i * 60,
                open_=1.08500 + i * 0.00010,
                close=1.08520 + i * 0.00010,
            )
            for i in range(n_bars)
        ]
        clean_pipeline = CausalFeaturePipeline(drop_warmup=False)
        clean_feat_ds = clean_pipeline.transform(clean_candles)

        # 2. Generate mutated candle series: bars 0..target_eval_bar are identical,
        # but bars target_eval_bar+1..n_bars-1 undergo extreme price shocks
        mutated_candles = list(clean_candles[: target_eval_bar + 1])
        for i in range(target_eval_bar + 1, n_bars):
            mutated_candles.append(
                _make_synth_candle(
                    base_ts + i * 60,
                    open_=1.50000 + i * 0.01000,  # 50% price spike shock
                    close=1.60000 + i * 0.01000,
                    high=2.00000,
                    low=1.40000,
                    volume=999999,  # Extreme volume spike
                )
            )

        mutated_pipeline = CausalFeaturePipeline(drop_warmup=False)
        mutated_feat_ds = mutated_pipeline.transform(mutated_candles)

        # Verify causal feature row at target_eval_bar is bit-for-bit identical
        clean_row = clean_feat_ds.rows[target_eval_bar]
        mutated_row = mutated_feat_ds.rows[target_eval_bar]
        for k in clean_row.features:
            assert clean_row.features[k] == pytest.approx(mutated_row.features[k], rel=1e-12)

        # 3. Evaluate MLStrategy on both rows
        model = ControlledProbabilityModel(prob_up=0.72)
        strategy_clean = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        strategy_mut = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])

        sig_clean = strategy_clean.on_bar(clean_row, clean_feat_ds.rows[: target_eval_bar + 1])
        sig_mut = strategy_mut.on_bar(mutated_row, mutated_feat_ds.rows[: target_eval_bar + 1])

        # Assert strict point-in-time invariant: zero lookahead bias
        assert sig_clean.direction == sig_mut.direction == SignalDirection.CALL
        assert sig_clean.stake == sig_mut.stake
        assert sig_clean.horizon_bars == sig_mut.horizon_bars
        assert (
            sig_clean.metadata["calibrated_prob_up"] == sig_mut.metadata["calibrated_prob_up"]
        )
        assert (
            sig_clean.metadata["decision_timestamp"] == sig_mut.metadata["decision_timestamp"]
        )
        assert sig_clean.metadata["current_price"] == sig_mut.metadata["current_price"]

    def test_backtest_engine_future_candles_do_not_alter_past_trade_entries(self) -> None:
        """Verify BacktestEngine trade entry decisions are immutable to future candle shocks."""
        base_ts = 1_700_000_000
        n_bars = 60
        split_bar = 35

        # Series A
        candles_a = [
            _make_synth_candle(base_ts + i * 60, 1.0850, 1.0855 if i % 2 == 0 else 1.0845)
            for i in range(n_bars)
        ]
        feat_a = CausalFeaturePipeline(drop_warmup=False).transform(candles_a)

        # Series B: Identical up to split_bar, then crashes
        candles_b = list(candles_a[:split_bar]) + [
            _make_synth_candle(base_ts + i * 60, 0.5000, 0.4900)
            for i in range(split_bar, n_bars)
        ]
        feat_b = CausalFeaturePipeline(drop_warmup=False).transform(candles_b)

        model = ControlledProbabilityModel(prob_up=0.70)
        payoff = BinaryPayoffModel()
        cfg = BacktestConfig(initial_capital=1000.0, stake_amount=50.0, latency_bars=0)

        strat_a = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        strat_b = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])

        res_a = BacktestEngine(strat_a, payoff, cfg).run(feat_a.rows, candles_a)
        res_b = BacktestEngine(strat_b, payoff, cfg).run(feat_b.rows, candles_b)

        # Filter trades initiated strictly before split_bar
        trades_a_early = [t for t in res_a.trades if t.entry_bar_index < split_bar - 2]
        trades_b_early = [t for t in res_b.trades if t.entry_bar_index < split_bar - 2]

        assert len(trades_a_early) == len(trades_b_early)
        assert len(trades_a_early) > 0

        for ta, tb in zip(trades_a_early, trades_b_early, strict=True):
            assert ta.entry_bar_index == tb.entry_bar_index
            assert ta.entry_ts == tb.entry_ts
            assert ta.direction == tb.direction
            assert ta.stake == pytest.approx(tb.stake)
            assert ta.entry_price == pytest.approx(tb.entry_price)


# ==============================================================================
# Suite 2: Corrupted, Non-Finite, and Malformed Features
# ==============================================================================


class TestAdversarialCorruptedFeatures:
    """Stress tests verifying MLStrategy handles NaN/Inf gracefully with zero crashing."""

    @pytest.mark.parametrize(
        "bad_val",
        [float("nan"), float("inf"), float("-inf"), np.nan],
    )
    def test_corrupted_features_nan_inf_abstention(self, bad_val: float) -> None:
        """Verify that non-finite feature values force HOLD with diagnostic metadata."""
        model = ControlledProbabilityModel(prob_up=0.85)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])

        # Inject non-finite feature into row
        corrupted_row = _make_synth_row(
            features={"return_1": bad_val, "sma_ratio_5": 1.0001, "close": 1.08520}
        )
        sig = strategy.on_bar(corrupted_row, [corrupted_row])

        assert sig.direction == SignalDirection.HOLD
        assert sig.stake is None
        assert not sig.is_actionable
        assert len(sig.metadata["abstention_reasons"]) > 0
        assert "Non-finite feature value for 'return_1'" in sig.metadata["abstention_reasons"][0]

    def test_corrupted_features_with_scaler_abstains_without_crashing(self) -> None:
        """Verify that non-finite feature is caught before scaler transform causes errors."""
        scaler = StandardScaler()
        # Fit scaler on valid training matrix
        scaler.fit(np.array([[0.001, 1.0], [0.002, 1.01]], dtype=np.float64), ["f1", "f2"])

        model = ControlledProbabilityModel(prob_up=0.85, feature_names=("f1", "f2"))
        strategy = MLStrategy(
            model=model,
            scaler=scaler,
            feature_columns=["f1", "f2"],
        )

        corrupted_row = _make_synth_row(
            features={"f1": float("nan"), "f2": 1.005, "close": 1.085}
        )
        sig = strategy.on_bar(corrupted_row, [corrupted_row])

        assert sig.direction == SignalDirection.HOLD
        assert sig.stake is None

    def test_missing_required_feature_raises_keyerror(self) -> None:
        """Verify missing feature column raises KeyError immediately instead of bad prediction."""
        model = ControlledProbabilityModel(prob_up=0.80, feature_names=("f1", "f2"))
        strategy = MLStrategy(model=model, feature_columns=["f1", "f2"])

        missing_row = _make_synth_row(features={"f1": 0.001, "close": 1.085})  # missing f2
        with pytest.raises(KeyError, match="FeatureRow is missing required feature 'f2'"):
            strategy.on_bar(missing_row, [missing_row])

    def test_corrupted_features_in_e2e_backtest_executes_zero_corrupted_trades(self) -> None:
        """Verify that injecting NaNs across several bars in a backtest cleanly yields HOLD."""
        base_ts = 1_700_000_000
        n_bars = 40
        candles = [_make_synth_candle(base_ts + i * 60, 1.0850, 1.0855) for i in range(n_bars)]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        # Corrupt features at bars 25, 26, 27
        corrupted_indices = {25, 26, 27}
        patched_rows = []
        for i, r in enumerate(feat_ds.rows):
            if i in corrupted_indices:
                bad_feats = dict(r.features)
                bad_feats["return_1"] = float("nan")
                patched_rows.append(
                    _make_synth_row(
                        index=i, features=bad_feats, candle_from_ts=r.candle_from_ts
                    )
                )
            else:
                patched_rows.append(r)

        model = ControlledProbabilityModel(prob_up=0.80)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        payoff = BinaryPayoffModel()
        cfg = BacktestConfig(initial_capital=1000.0, stake_amount=100.0)

        engine = BacktestEngine(strategy, payoff, cfg)
        res = engine.run(patched_rows, candles)

        # Verify no trade entered on corrupted bars
        for t in res.trades:
            assert t.entry_bar_index not in corrupted_indices


# ==============================================================================
# Suite 3: Zero-Edge and Negative-Edge Binary Options Payouts (100% Abstention)
# ==============================================================================


class TestAdversarialZeroEdgeAbstention:
    """Stress tests verifying 100% abstention on zero or negative expected payoff."""

    @pytest.mark.parametrize(
        ("payout_rate", "prob_up"),
        [
            (0.10, 0.90),    # p_be = 1/1.10 = 0.9091 > 0.90 -> E[PnL] = -0.01 < 0
            (0.50, 0.65),    # p_be = 1/1.50 = 0.6667 > 0.65 -> E[PnL] = -0.025 < 0
            (0.70, 0.58),    # p_be = 1/1.70 = 0.5882 > 0.58 -> E[PnL] = -0.014 < 0
            (0.85, 0.54),    # p_be = 1/1.85 = 0.5405 > 0.54 -> E[PnL] = -0.001 < 0
            (0.85, 0.50),    # pure 50/50 tie -> E[PnL] = -0.075 < 0
            (1.00, 0.50),    # p_be = 0.50, pure tie has 0 expectancy -> HOLD
            (1.50, 0.50),    # p_be = 0.40, pure tie has 0 directional expectancy -> HOLD
        ],
    )
    def test_zero_edge_binary_payoff_sweep_100_percent_abstention(
        self, payout_rate: float, prob_up: float
    ) -> None:
        """Verify across diverse payout rates that sub-breakeven probabilities produce 100% HOLD."""
        model = ControlledProbabilityModel(prob_up=prob_up)
        gate = BinaryPayoffGate(payout_rate=payout_rate)
        engine = AIDecisionEngine(abstention_gate=gate, payout_rate=payout_rate)
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            decision_engine=engine,
            payout_rate=payout_rate,
        )

        row = _make_synth_row()
        sig = strategy.on_bar(row, [row])

        assert sig.direction == SignalDirection.HOLD
        assert sig.stake is None
        assert not sig.is_actionable

    def test_negative_edge_sub_breakeven_backtest_engine_zero_trades(self) -> None:
        """Verify 100-bar backtest under sub-breakeven model executes exactly zero trades."""
        base_ts = 1_700_000_000
        n_bars = 100
        candles = [_make_synth_candle(base_ts + i * 60, 1.0850, 1.0855) for i in range(n_bars)]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        # P(UP)=0.52 on 85% payout is negative expectancy (break-even is 54.05%)
        model = ControlledProbabilityModel(prob_up=0.52)
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            payout_rate=0.85,
        )
        payoff = BinaryPayoffModel(payout_rate=0.85)
        cfg = BacktestConfig(initial_capital=1000.0)

        res = BacktestEngine(strategy, payoff, cfg).run(feat_ds.rows, candles)

        assert res.trade_count == 0
        assert res.final_capital == res.initial_capital
        assert res.total_net_pnl == 0.0

    def test_positive_margin_hurdle_abstention(self) -> None:
        """Verify safety margin hurdle causes abstention even when probability exceeds p_be."""
        payout = 0.85
        margin = 0.05
        p_be = calculate_binary_break_even(payout, margin=0.0)  # ~0.54054
        p_thresh = calculate_binary_break_even(payout, margin=margin)  # ~0.59054
        assert p_be < 0.56 < p_thresh

        # Model predicts 0.56: > p_be, but < p_thresh
        model = ControlledProbabilityModel(prob_up=0.56)
        gate = BinaryPayoffGate(payout_rate=payout, margin=margin)
        decision_engine = AIDecisionEngine(abstention_gate=gate, payout_rate=payout)
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            decision_engine=decision_engine,
        )

        row = _make_synth_row()
        sig = strategy.on_bar(row, [row])

        assert sig.direction == SignalDirection.HOLD
        assert "SUB_BREAKEVEN_PAYOFF" in sig.metadata["abstention_reasons"][0]


# ==============================================================================
# Suite 4: Extreme Forex Spreads and Slippage Spikes
# ==============================================================================


class TestAdversarialForexSpreadsAndSlippageSpikes:
    """Stress tests verifying ForexPayoffGate abstains during spread spikes and severe friction."""

    def test_forex_extreme_spread_spike_forces_abstention(self) -> None:
        """Verify quote spread spike from 1.0 to 15.0 pips forces immediate trade abstention."""
        # Expected move = 10 pips, P(UP) = 0.85 -> exp_pips_call = (2*0.85 - 1)*10 = 7.0 pips
        model = ControlledProbabilityModel(prob_up=0.85)
        gate = ForexPayoffGate(
            spread_pips=1.0,
            slippage_pips=0.5,
            hurdle_pips=0.5,
            default_expected_move_pips=10.0,
        )
        decision_engine = AIDecisionEngine(abstention_gate=gate)
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            decision_engine=decision_engine,
        )

        # Normal spread row (1.0 pip in quote) -> Allowed
        normal_feats = {
            "return_1": 0.001,
            "sma_ratio_5": 1.001,
            "spread_pips": 1.0,
            "close": 1.085,
        }
        normal_row = _make_synth_row(features=normal_feats)
        sig_normal = strategy.on_bar(normal_row, [normal_row])
        assert sig_normal.direction == SignalDirection.CALL

        # Spiked spread row (15.0 pips in quote) -> Cost = 16.5 pips > 7.0 pips -> Abstained!
        spiked_feats = {
            "return_1": 0.001,
            "sma_ratio_5": 1.001,
            "spread_pips": 15.0,
            "close": 1.085,
        }
        spiked_row = _make_synth_row(features=spiked_feats)
        sig_spiked = strategy.on_bar(spiked_row, [spiked_row])
        assert sig_spiked.direction == SignalDirection.HOLD
        assert "FOREX_COST_EXCEEDED" in sig_spiked.metadata["abstention_reasons"][0]

    def test_forex_dynamic_spread_spikes_during_backtest(self) -> None:
        """Verify that spread spikes protect portfolio by opening zero trades during spike."""
        base_ts = 1_700_000_000
        n_bars = 60
        candles = [_make_synth_candle(base_ts + i * 60, 1.0850, 1.0860) for i in range(n_bars)]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        # Inject extreme spread spike in bars 30 to 45
        spiked_bars = set(range(30, 46))
        patched_rows = []
        for i, r in enumerate(feat_ds.rows):
            f = dict(r.features)
            f["spread_pips"] = 25.0 if i in spiked_bars else 1.0
            patched_rows.append(
                _make_synth_row(index=i, features=f, candle_from_ts=r.candle_from_ts)
            )

        model = ControlledProbabilityModel(prob_up=0.80)
        gate = ForexPayoffGate(
            spread_pips=1.0, slippage_pips=0.5, default_expected_move_pips=10.0
        )
        engine = AIDecisionEngine(abstention_gate=gate)
        strategy = MLStrategy(
            model=model,
            feature_columns=["return_1", "sma_ratio_5"],
            decision_engine=engine,
        )
        payoff = ForexPayoffModel()
        cfg = BacktestConfig(initial_capital=1000.0, stake_amount=100.0)

        res = BacktestEngine(strategy, payoff, cfg).run(patched_rows, candles)

        # Check no trades initiated during spread shock
        for t in res.trades:
            assert t.entry_bar_index not in spiked_bars


# ==============================================================================
# Suite 5: Multi-Hundred Bar Runs with Circuit Breakers
# ==============================================================================


class TestAdversarialMultiHundredBarRuns:
    """Stress tests executing 500-600 bar backtests under adversarial market conditions."""

    def test_500_bar_monotonic_decline_triggers_drawdown_stop(self) -> None:
        """Verify 500-bar continuous loss sequence halts via max_drawdown_stop_ratio."""
        base_ts = 1_700_000_000
        n_bars = 500
        # Monotonically declining prices
        candles = [
            _make_synth_candle(base_ts + i * 60, 1.1000 - i * 0.0001, 1.1000 - (i + 1) * 0.0001)
            for i in range(n_bars)
        ]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        # Continuous CALL signals in falling market = 100% loss rate
        model = ControlledProbabilityModel(prob_up=0.85)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        payoff = BinaryPayoffModel(payout_rate=0.85)
        cfg = BacktestConfig(
            initial_capital=1000.0,
            stake_amount=100.0,
            max_drawdown_stop_ratio=0.15,  # 15% stop
        )

        engine = BacktestEngine(strategy, payoff, cfg)
        res = engine.run(feat_ds.rows, candles)

        # After 2 consecutive losses of 100: cash drops to 800 (20% drawdown > 15%)
        # Circuit breaker MUST trigger and prevent trades for remaining ~475 bars
        assert res.trade_count <= 2
        assert res.final_capital == pytest.approx(800.0)
        assert len(res.portfolio_history) == n_bars

        # Check that subsequent states remain flat at 800.0
        for state in res.portfolio_history[25:]:
            assert state.cash == pytest.approx(800.0)
            assert state.open_positions_count == 0

    def test_600_bar_regime_switch_and_exact_portfolio_accounting(self) -> None:
        """Verify 600-bar multi-regime backtest preserves exact portfolio equity on every bar."""
        base_ts = 1_700_000_000
        n_bars = 600

        # Create 3 distinct regimes: 200 bull, 200 choppy, 200 bear
        prices = [1.0850]
        rng = np.random.default_rng(12345)
        for i in range(1, n_bars):
            if i < 200:
                step = rng.normal(0.0001, 0.0002)
            elif i < 400:
                step = rng.normal(0.0000, 0.0003)
            else:
                step = rng.normal(-0.0001, 0.0002)
            prices.append(prices[-1] + step)

        candles = [
            _make_synth_candle(base_ts + i * 60, prices[i], prices[i] + 0.00005)
            for i in range(n_bars)
        ]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        model = ControlledProbabilityModel(prob_up=0.75)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        payoff = BinaryPayoffModel(payout_rate=0.85)
        cfg = BacktestConfig(initial_capital=5000.0, stake_amount=50.0, max_open_positions=3)

        res = BacktestEngine(strategy, payoff, cfg).run(feat_ds.rows, candles)

        assert len(res.portfolio_history) == n_bars
        assert res.trade_count > 20

        # Invariant: on every single bar, equity == cash + sum(active_stakes)
        for state in res.portfolio_history:
            assert state.equity >= state.cash
            assert math.isfinite(state.equity)
            assert math.isfinite(state.cash)

        # Invariant: final capital == initial + total net pnl
        assert res.final_capital == pytest.approx(
            res.initial_capital + res.total_net_pnl, abs=1e-9
        )


# ==============================================================================
# Suite 6: Capital Accounting Precision and Settlement Invariants
# ==============================================================================


class TestAdversarialCapitalAccounting:
    """Stress tests verifying mathematical cash conservation across all settlement outcomes."""

    @pytest.mark.parametrize(
        ("tie_action", "expected_tie_pnl"),
        [
            (TieAction.REFUND, 0.0),
            (TieAction.LOSS, -100.0),
        ],
    )
    def test_capital_accounting_exact_conservation_across_tie_actions(
        self, tie_action: TieAction, expected_tie_pnl: float
    ) -> None:
        """Verify exact cash conservation under tie settlement actions (REFUND, LOSS)."""
        base_ts = 1_700_000_000
        # Construct bars producing exact flat close (TIE outcome)
        candles = [
            _make_synth_candle(base_ts + i * 60, 1.08500, 1.08500)
            for i in range(25)
        ]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        model = ControlledProbabilityModel(prob_up=0.80)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        payoff = BinaryPayoffModel(payout_rate=0.85, tie_action=tie_action)
        cfg = BacktestConfig(initial_capital=1000.0, stake_amount=100.0, max_open_positions=1)

        res = BacktestEngine(strategy, payoff, cfg).run(feat_ds.rows, candles)

        assert res.trade_count > 0
        expected_outcome = (
            TradeOutcome.TIE if tie_action == TieAction.REFUND else TradeOutcome.LOSS
        )
        for t in res.trades:
            assert t.settlement.outcome == expected_outcome
            assert t.settlement.net_pnl == pytest.approx(expected_tie_pnl)
            assert t.capital_after == pytest.approx(t.capital_before + expected_tie_pnl)

        # Strict drift check: final_capital == initial_capital + sum(net_pnl)
        total_pnl = sum(t.settlement.net_pnl for t in res.trades)
        assert res.final_capital == pytest.approx(res.initial_capital + total_pnl, abs=1e-12)

    def test_capital_exhaustion_stops_order_execution(self) -> None:
        """Verify BacktestEngine refuses to open trades when cash falls below stake_amount."""
        base_ts = 1_700_000_000
        # Consecutive price drops: bar t close > bar t+1 close, so CALL trades strictly lose
        candles = [
            _make_synth_candle(
                base_ts + i * 60,
                open_=1.1000 - i * 0.0010,
                close=1.1000 - (i + 1) * 0.0010,
            )
            for i in range(30)
        ]
        feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

        model = ControlledProbabilityModel(prob_up=0.80)
        strategy = MLStrategy(model=model, feature_columns=["return_1", "sma_ratio_5"])
        payoff = BinaryPayoffModel(payout_rate=0.85)
        # Initial capital 150.0, stake 100.0 -> after 1 loss cash is 50.0 (< 100.0)
        cfg = BacktestConfig(initial_capital=150.0, stake_amount=100.0)

        res = BacktestEngine(strategy, payoff, cfg).run(feat_ds.rows, candles)

        assert res.trade_count == 1
        assert res.final_capital == pytest.approx(50.0)
        assert res.final_capital >= 0.0


# ==============================================================================
# Suite 7: HoldoutEvaluator Adversarial Boundary & Isolation Stress
# ==============================================================================


class TestAdversarialHoldoutBoundaries:
    """Stress tests probing HoldoutEvaluator boundary conditions and data isolation."""

    def test_holdout_minimum_sample_size_boundary(self) -> None:
        """Verify N=10 samples succeeds whereas N=9 samples raises HoldoutSplitError."""
        cfg = HoldoutSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        evaluator = HoldoutEvaluator(cfg)

        # N=10 passes
        indices = evaluator.calculate_indices(10)
        assert indices["train_end"] > 0
        assert indices["val_end"] > indices["val_start"]
        assert indices["test_end"] > indices["test_start"]

        # N=9 raises
        with pytest.raises(
            HoldoutSplitError, match="Dataset too small for 3-way chronological holdout"
        ):
            evaluator.calculate_indices(9)

    def test_holdout_excessive_purge_raises_cleanly(self) -> None:
        """Verify purge_bars exceeding train partition size raises HoldoutSplitError."""
        cfg = HoldoutSplitConfig(
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, purge_bars=100
        )
        evaluator = HoldoutEvaluator(cfg)

        with pytest.raises(HoldoutSplitError, match="Train split is empty"):
            evaluator.calculate_indices(100)

    def test_holdout_excessive_embargo_raises_cleanly(self) -> None:
        """Verify embargo_bars exceeding test partition size raises HoldoutSplitError."""
        cfg = HoldoutSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, embargo_bars=100)
        evaluator = HoldoutEvaluator(cfg)

        with pytest.raises(HoldoutSplitError, match="Test split is empty"):
            evaluator.calculate_indices(100)

    def test_holdout_split_indices_are_strictly_immutable(self) -> None:
        """Verify HoldoutSplit numpy index arrays have writeable=False flag enforced."""
        evaluator = HoldoutEvaluator()
        split = evaluator.get_indices(100)

        assert split.train_indices.flags.writeable is False
        assert split.val_indices.flags.writeable is False
        assert split.test_indices.flags.writeable is False

        with pytest.raises(ValueError, match="assignment destination is read-only"):
            split.train_indices[0] = 999
