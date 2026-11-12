"""Empirical Adversarial Stress Test Suite for Phase 4 Milestone 6.

Subsystems Tested:
1. Comparative Benchmarking Subsystem (comparison.py)
   - Statistical Invariants: McNemar's exact vs. continuity-corrected chi-squared,
     boundary at b + c = 25, zero discordance (b=0, c=0), paired loss difference tests.
   - Financial & Payoff Invariants: Binary and Forex payoff simulations, break-even
     hurdle edge cases, extreme abstention, 100% and 0% win rates, zero gross loss (infinite PF),
     negative returns exceeding 100% of initial capital.
   - Ranking Robustness: Leaderboard sorting with inf, -inf, and None values across
     ascending and descending metrics.
2. Hybrid GPT Layer (src/iqoption_research/ml/hybrid/)
   - Offline Safety: Strict zero-socket invariant verified via socket monkeypatching.
   - Fail-Closed Security: OpenAIAssistant dual-gate authorization and immediate LLMSecurityError.
   - Zero Trading Authority: Non-execution invariant, absence of broker APIs, and mandatory
     disclaimers.
"""

from __future__ import annotations

import math
import socket
from typing import Any

import numpy as np
import pytest

from iqoption_research.ml.evaluation.comparison import (
    ComparisonConfig,
    ComparisonError,
    ComparisonResult,
    EvaluationSplitType,
    ModelBenchmarkRow,
    ModelEvaluationMismatchError,
    PayoffAssumptionType,
    RankingMetric,
    compare_models,
    compute_mcnemar_test,
    compute_paired_difference_test,
    generate_comparison_markdown_report,
    simulate_model_payoffs,
)
from iqoption_research.ml.hybrid.assistant import (
    RESEARCH_DISCLAIMER,
    GPTResearchAssistant,
)
from iqoption_research.ml.hybrid.base import (
    LLMProviderUnavailableError,
    LLMSecurityError,
)
from iqoption_research.ml.hybrid.mock import MockGPTProvider
from iqoption_research.ml.hybrid.openai_adapter import OpenAIAssistant
from iqoption_research.ml.models.baselines import (
    MajorityClassModel,
    SeededRandomModel,
)
from iqoption_research.ml.targets.dataset import MLDataset
from iqoption_research.ml.targets.specs import TargetSpec, TargetType

# =====================================================================
# Fixtures and Helpers
# =====================================================================


def _make_adversarial_dataset(n: int = 120, d: int = 3, seed: int = 123) -> MLDataset:
    """Generate deterministic MLDataset for benchmarking tests."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-1.5 * X[:, 0]))
    y = rng.binomial(1, p).astype(np.int64)
    y[0] = 0
    y[1] = 1
    base_ts = 1_700_000_000
    timestamps = np.array([base_ts + i * 60 for i in range(n)], dtype=np.int64)
    spec = TargetSpec(target_type=TargetType.DIRECTION_BINARY, horizon_bars=1)
    return MLDataset(
        symbol="EURUSD",
        feature_names=tuple(f"feat_{j}" for j in range(d)),
        X=X,
        y=y,
        timestamps=timestamps,
        target_spec=spec,
    )


def _make_benchmark_row(
    name: str,
    brier: float = 0.20,
    acc: float = 0.60,
    ret: float | None = 250.0,
    pf: float | None = 1.5,
    max_dd: float | None = 100.0,
    trade_count: int = 50,
    win_rate: float | None = 0.60,
) -> ModelBenchmarkRow:
    """Create a ModelBenchmarkRow with arbitrary metrics for sorting stress tests."""
    n_samples = 50
    return ModelBenchmarkRow(
        model_name=name,
        model_type="baseline",
        accuracy=acc,
        balanced_accuracy=acc,
        precision=acc,
        recall=acc,
        f1_macro=acc,
        brier_score=brier,
        log_loss=0.50,
        roc_auc=0.65,
        ece=0.02,
        trade_count=trade_count,
        win_rate=win_rate,
        total_return=ret,
        profit_factor=pf,
        max_drawdown=max_dd,
        abstention_rate=0.0,
        sample_count=n_samples,
    )


# =====================================================================
# 1. Statistical Invariants & Hypothesis Testing Stress Tests
# =====================================================================


class TestStatisticalInvariants:
    """Adversarial stress testing of McNemar and paired difference statistical tests."""

    @pytest.mark.parametrize("n_samples", [1, 2, 10, 25, 100, 500])
    def test_identical_models_mcnemar_p_value_strictly_one(self, n_samples: int) -> None:
        """Assert identical model predictions always yield p=1.0 and stat=0.0 in McNemar test."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=n_samples)
        y_pred = rng.integers(0, 2, size=n_samples)

        result = compute_mcnemar_test(
            y_true=y_true,
            y_pred_a=y_pred,
            y_pred_b=y_pred,
            model_a_name="Model_1",
            model_b_name="Model_2",
        )

        assert result.p_value == 1.0, f"Expected p-value 1.0, got {result.p_value}"
        assert result.statistic == 0.0, f"Expected statistic 0.0, got {result.statistic}"
        assert not result.is_significant
        assert result.effect_size == 0.0
        assert result.contingency_table is not None
        # Contingency table has b = 0 (n01) and c = 0 (n10)
        assert result.contingency_table[0][1] == 0
        assert result.contingency_table[1][0] == 0
        assert "Zero discordant pairs" in result.notes

    def test_identical_losses_paired_difference_t_test_and_wilcoxon(self) -> None:
        """Assert identical loss vectors yield statistic=0.0 and p=1.0 without variance crash."""
        rng = np.random.default_rng(99)
        loss = rng.uniform(0.01, 0.40, size=100)

        # 1. Paired t-test
        t_res = compute_paired_difference_test(
            loss_a=loss,
            loss_b=loss,
            model_a_name="Model_A",
            model_b_name="Model_B",
            test_type="paired_t",
        )
        assert t_res.p_value == 1.0
        assert t_res.statistic == 0.0
        assert not t_res.is_significant
        assert t_res.effect_size == 0.0
        assert "Identical losses" in t_res.notes

        # 2. Wilcoxon signed-rank test
        w_res = compute_paired_difference_test(
            loss_a=loss,
            loss_b=loss,
            model_a_name="Model_A",
            model_b_name="Model_B",
            test_type="wilcoxon",
        )
        assert w_res.p_value == 1.0
        assert w_res.statistic == 0.0
        assert not w_res.is_significant
        assert w_res.effect_size == 0.0
        assert "Identical losses" in w_res.notes

    def test_paired_losses_near_epsilon_identical(self) -> None:
        """Assert floating point differences < 1e-12 are treated as identical losses."""
        loss_a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        loss_b = loss_a + 1e-13  # within atol=1e-12

        res = compute_paired_difference_test(loss_a, loss_b, test_type="paired_t")
        assert res.p_value == 1.0
        assert res.statistic == 0.0
        assert "Identical losses" in res.notes

    def test_mcnemar_boundary_switch_at_25_discordant_pairs(self) -> None:
        """Verify boundary transition at b + c = 25 between exact and Edwards' tests."""
        # 1. Sub-boundary: b + c = 24 -> MUST use exact binomial test
        # Construct y_true, y_a, y_b such that b=12, c=12 (total 24)
        n = 50
        y_true = np.ones(n, dtype=np.int64)
        y_a = np.ones(n, dtype=np.int64)
        y_b = np.ones(n, dtype=np.int64)
        # 12 times A correct (1), B incorrect (0)
        y_b[:12] = 0
        # 12 times A incorrect (0), B correct (1)
        y_a[12:24] = 0

        res_24 = compute_mcnemar_test(y_true, y_a, y_b)
        assert res_24.contingency_table is not None
        assert res_24.contingency_table[0][1] == 12  # b
        assert res_24.contingency_table[1][0] == 12  # c
        assert "Exact binomial test" in res_24.notes
        assert res_24.p_value == 1.0  # symmetric discordance
        assert res_24.statistic == 0.0

        # Sub-boundary with extreme asymmetry: b=24, c=0
        y_a2 = np.ones(n, dtype=np.int64)
        y_b2 = np.ones(n, dtype=np.int64)
        y_b2[:24] = 0
        res_24_asym = compute_mcnemar_test(y_true, y_a2, y_b2)
        assert "Exact binomial test" in res_24_asym.notes
        # Two-sided p-value for 0 successes out of 24 trials with p=0.5 is 2 * (0.5)^24
        expected_p = 2.0 * (0.5**24)
        assert math.isclose(res_24_asym.p_value, expected_p, rel_tol=1e-6)
        assert res_24_asym.is_significant

        # 2. At boundary: b + c = 25 -> MUST switch to Chi-squared approximation
        y_a3 = np.ones(n, dtype=np.int64)
        y_b3 = np.ones(n, dtype=np.int64)
        # b = 13, c = 12 (total 25)
        y_b3[:13] = 0
        y_a3[13:25] = 0
        res_25 = compute_mcnemar_test(y_true, y_a3, y_b3, continuity_correction=True)
        assert res_25.contingency_table is not None
        assert res_25.contingency_table[0][1] == 13
        assert res_25.contingency_table[1][0] == 12
        assert "Edwards' continuity correction" in res_25.notes
        # (|13 - 12| - 1)^2 / 25 = 0.0 / 25 = 0.0
        assert res_25.statistic == 0.0
        assert res_25.p_value == 1.0

        # 3. At boundary with asymmetry: b = 25, c = 0
        y_a4 = np.ones(n, dtype=np.int64)
        y_b4 = np.ones(n, dtype=np.int64)
        y_b4[:25] = 0
        res_25_asym = compute_mcnemar_test(y_true, y_a4, y_b4, continuity_correction=True)
        assert "Edwards' continuity correction" in res_25_asym.notes
        # (|25 - 0| - 1)^2 / 25 = 24^2 / 25 = 576 / 25 = 23.04
        assert math.isclose(res_25_asym.statistic, 23.04, abs_tol=1e-5)
        assert res_25_asym.is_significant

        # 4. Without continuity correction: b = 25, c = 0
        res_25_nocorr = compute_mcnemar_test(y_true, y_a4, y_b4, continuity_correction=False)
        assert "Standard asymptotic Chi-squared" in res_25_nocorr.notes
        # 25^2 / 25 = 25.0
        assert math.isclose(res_25_nocorr.statistic, 25.0, abs_tol=1e-5)

    def test_mcnemar_error_handling(self) -> None:
        """Assert invalid inputs to McNemar test raise appropriate exceptions."""
        # Empty array
        with pytest.raises(ComparisonError, match="Cannot compute McNemar's test on 0 samples"):
            compute_mcnemar_test(np.array([]), np.array([]), np.array([]))

        # Length mismatch
        with pytest.raises(ModelEvaluationMismatchError, match="Sample length mismatch"):
            compute_mcnemar_test(np.array([1, 0]), np.array([1]), np.array([1, 0]))


# =====================================================================
# 2. Financial & Payoff Invariants
# =====================================================================


class TestFinancialPayoffInvariants:
    """Adversarial stress testing of binary options and Forex payoff simulations."""

    def test_binary_payoff_extreme_abstention_zero_trades(self) -> None:
        """Assert high abstention yields zero trades and valid metrics without division errors."""
        cfg = ComparisonConfig(
            payoff_type=PayoffAssumptionType.BINARY,
            binary_payout_rate=0.85,  # break-even hurdle is 1 / 1.85 ~ 0.5405
            stake_amount=100.0,
            initial_capital=10_000.0,
        )
        n = 100
        y_true = np.random.randint(0, 2, size=n)
        # All probabilities close to 0.50 (below 0.5405 hurdle)
        y_prob = np.full((n, 2), 0.50)

        t_count, win_rate, tot_ret, pf, max_dd, abs_rate, pnls = simulate_model_payoffs(
            y_true, y_prob, cfg
        )

        assert t_count == 0
        assert win_rate is None
        assert tot_ret == 0.0
        assert pf is None
        assert max_dd == 0.0
        assert abs_rate == 1.0
        assert pnls == ()

    def test_binary_payoff_100_percent_win_rate_infinite_profit_factor(self) -> None:
        """Assert 100% win rate produces infinite profit factor without zero-division error."""
        cfg = ComparisonConfig(
            payoff_type=PayoffAssumptionType.BINARY,
            binary_payout_rate=0.85,
            stake_amount=100.0,
            initial_capital=10_000.0,
        )
        n = 20
        y_true = np.ones(n, dtype=np.int64)
        # Confident UP predictions exceeding hurdle: P(UP)=0.90 >= 0.5405
        y_prob = np.column_stack([np.full(n, 0.10), np.full(n, 0.90)])

        t_count, win_rate, tot_ret, pf, max_dd, abs_rate, pnls = simulate_model_payoffs(
            y_true, y_prob, cfg
        )

        assert t_count == n
        assert win_rate == 1.0
        assert math.isclose(tot_ret, n * 100.0 * 0.85, rel_tol=1e-5)
        assert pf is not None and math.isinf(pf) and pf > 0
        assert max_dd == 0.0
        assert abs_rate == 0.0
        assert len(pnls) == n
        assert all(p == 85.0 for p in pnls)

    def test_binary_payoff_0_percent_win_rate_zero_profit_factor(self) -> None:
        """Assert 0% win rate produces profit factor of 0.0 and correct maximum drawdown."""
        cfg = ComparisonConfig(
            payoff_type=PayoffAssumptionType.BINARY,
            binary_payout_rate=0.85,
            stake_amount=100.0,
            initial_capital=10_000.0,
        )
        n = 15
        y_true = np.zeros(n, dtype=np.int64)  # Ground truth is DOWN (0)
        # But model confidently predicts UP (1) with P(UP)=0.85 >= 0.5405
        y_prob = np.column_stack([np.full(n, 0.15), np.full(n, 0.85)])

        t_count, win_rate, tot_ret, pf, max_dd, abs_rate, pnls = simulate_model_payoffs(
            y_true, y_prob, cfg
        )

        assert t_count == n
        assert win_rate == 0.0
        assert math.isclose(tot_ret, -n * 100.0, rel_tol=1e-5)
        assert pf == 0.0
        assert math.isclose(max_dd, n * 100.0, rel_tol=1e-5)
        assert abs_rate == 0.0
        assert all(p == -100.0 for p in pnls)

    def test_binary_payoff_exact_break_even_hurdle_boundary(self) -> None:
        """Assert boundary behavior when probability exactly equals break-even hurdle."""
        payout = 0.85
        hurdle = 1.0 / (1.0 + payout)  # approx 0.54054054054...
        cfg = ComparisonConfig(
            payoff_type=PayoffAssumptionType.BINARY,
            binary_payout_rate=payout,
            stake_amount=100.0,
        )

        # 1. P(UP) = hurdle: condition is p_up >= break_even and p_up > p_down. MUST TRADE.
        y_true = np.array([1, 0], dtype=np.int64)
        y_prob_at_hurdle = np.array([
            [1.0 - hurdle, hurdle],
            [1.0 - hurdle, hurdle],
        ])
        t_count, _, tot_ret, _, _, abs_rate, _ = simulate_model_payoffs(
            y_true, y_prob_at_hurdle, cfg
        )
        assert t_count == 2
        assert abs_rate == 0.0
        # 1 win (+85) and 1 loss (-100) -> total return -15
        assert math.isclose(tot_ret, 85.0 - 100.0, abs_tol=1e-5)

        # 2. P(UP) = hurdle - 1e-6: slightly below hurdle. MUST ABSTAIN.
        sub_hurdle = hurdle - 1e-6
        y_prob_sub = np.array([
            [1.0 - sub_hurdle, sub_hurdle],
            [1.0 - sub_hurdle, sub_hurdle],
        ])
        t_count_sub, _, _, _, _, abs_rate_sub, _ = simulate_model_payoffs(y_true, y_prob_sub, cfg)
        assert t_count_sub == 0
        assert abs_rate_sub == 1.0

        # 3. Payout = 1.0 (hurdle = 0.5). Exact 50/50 probability MUST strictly abstain.
        cfg_even = ComparisonConfig(
            payoff_type=PayoffAssumptionType.BINARY,
            binary_payout_rate=1.0,
        )
        y_prob_half = np.array([[0.5, 0.5], [0.5, 0.5]])
        t_count_half, _, _, _, _, abs_rate_half, _ = simulate_model_payoffs(
            y_true, y_prob_half, cfg_even
        )
        assert t_count_half == 0
        assert abs_rate_half == 1.0

    def test_forex_payoff_extreme_drawdown_exceeding_100_percent(self) -> None:
        """Assert Forex simulation handles extreme losses exceeding 100% of initial capital."""
        cfg = ComparisonConfig(
            payoff_type=PayoffAssumptionType.FOREX,
            stake_amount=100.0,
            initial_capital=500.0,  # low initial capital to trigger > 100% drawdown
            forex_spread_pips=1.0,
            forex_slippage_pips=0.5,
        )
        # 100 consecutive losing trades
        n = 100
        y_true = np.zeros(n, dtype=np.int64)  # Ground truth DOWN
        y_prob = np.column_stack([np.full(n, 0.10), np.full(n, 0.90)])  # Model predicts UP >= 0.52

        t_count, win_rate, tot_ret, pf, max_dd, _abs_rate, _pnls = simulate_model_payoffs(
            y_true, y_prob, cfg
        )

        assert t_count == n
        assert win_rate == 0.0
        # Stake 100 * 0.01 = -1.0 net per trade plus friction
        # Total return will be deeply negative
        assert tot_ret < -100.0
        # Peak equity is 500.0; equity becomes negative; max drawdown > 500.0
        assert max_dd > 100.0
        assert pf == 0.0

    def test_forex_zero_gross_loss_infinite_profit_factor(self) -> None:
        """Assert Forex simulation handles 100% profitable trades with infinite profit factor."""
        cfg = ComparisonConfig(
            payoff_type=PayoffAssumptionType.FOREX,
            stake_amount=1000.0,
            forex_spread_pips=0.0,
            forex_slippage_pips=0.0,
        )
        n = 10
        y_true = np.ones(n, dtype=np.int64)
        y_prob = np.column_stack([np.full(n, 0.10), np.full(n, 0.90)])

        t_count, win_rate, tot_ret, pf, max_dd, _abs_rate, _ = simulate_model_payoffs(
            y_true, y_prob, cfg
        )

        assert t_count == n
        assert win_rate == 1.0
        assert pf is not None and math.isinf(pf) and pf > 0
        assert tot_ret > 0.0
        assert max_dd == 0.0


# =====================================================================
# 3. Leaderboard Ranking & Sorting Robustness
# =====================================================================


class TestRankingRobustness:
    """Stress tests for ComparisonResult.rank_by with edge cases (inf, -inf, None)."""

    def test_rank_by_profit_factor_with_inf_and_none(self) -> None:
        """Assert rank_by orders profit factor correctly: inf > high > low > 0.0 > None."""
        rows = (
            _make_benchmark_row("Model_Inf", pf=float("inf")),
            _make_benchmark_row("Model_High", pf=3.5),
            _make_benchmark_row("Model_Mid", pf=1.2),
            _make_benchmark_row("Model_Zero", pf=0.0),
            _make_benchmark_row("Model_None", pf=None, trade_count=0),
        )
        result = ComparisonResult(
            rows=rows,
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={},
        )

        # Descending (default for profit_factor: higher is better)
        ranked_desc = result.rank_by(RankingMetric.PROFIT_FACTOR)
        ranked_names_desc = [r.model_name for r in ranked_desc]
        assert ranked_names_desc == [
            "Model_Inf",
            "Model_High",
            "Model_Mid",
            "Model_Zero",
            "Model_None",
        ]

        # Ascending (explicitly requested: lower is better)
        ranked_asc = result.rank_by(RankingMetric.PROFIT_FACTOR, ascending=True)
        ranked_names_asc = [r.model_name for r in ranked_asc]
        # In ascending sort, None is mapped to float('inf') so it is placed at the end
        assert ranked_names_asc == [
            "Model_Zero",
            "Model_Mid",
            "Model_High",
            "Model_Inf",
            "Model_None",
        ]

    def test_rank_by_brier_score_with_none(self) -> None:
        """Assert rank_by orders Brier score correctly: lower is better, None ranked last."""
        rows = (
            _make_benchmark_row("Model_Best", brier=0.15),
            _make_benchmark_row("Model_Mid", brier=0.22),
            _make_benchmark_row("Model_Worst", brier=0.35),
        )
        result = ComparisonResult(
            rows=rows,
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={},
        )

        ranked = result.rank_by(RankingMetric.BRIER_SCORE)
        assert [r.model_name for r in ranked] == ["Model_Best", "Model_Mid", "Model_Worst"]
        assert result.best_model(RankingMetric.BRIER_SCORE).model_name == "Model_Best"

    def test_rank_by_total_return_with_negative_and_none(self) -> None:
        """Assert rank_by orders total_return descending: positive > 0 > negative > None."""
        rows = (
            _make_benchmark_row("Model_Pos", ret=500.0),
            _make_benchmark_row("Model_Flat", ret=0.0),
            _make_benchmark_row("Model_Neg", ret=-300.0),
            _make_benchmark_row("Model_None", ret=None),
        )
        result = ComparisonResult(
            rows=rows,
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={},
        )

        ranked = result.rank_by(RankingMetric.TOTAL_RETURN)
        assert [r.model_name for r in ranked] == [
            "Model_Pos",
            "Model_Flat",
            "Model_Neg",
            "Model_None",
        ]

    def test_markdown_report_formatting_handles_inf_and_none(self) -> None:
        """Assert generate_comparison_markdown_report formats inf and None without crashing."""
        rows = (
            _make_benchmark_row("Model_A", pf=float("inf"), brier=0.18, ret=300.0),
            _make_benchmark_row("Model_B", pf=None, brier=0.25, ret=None, win_rate=None),
        )
        result = ComparisonResult(
            rows=rows,
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={},
        )

        md = generate_comparison_markdown_report(result)
        assert "Model_A" in md
        assert "Model_B" in md
        assert "∞" in md  # infinite profit factor rendered as unicode infinity
        assert "—" in md  # None rendered as em-dash
        assert "| Rank | Model Name |" in md


# =====================================================================
# 4. Hybrid GPT Security & Offline Invariant
# =====================================================================


class TestHybridGPTSecurityAndOfflineInvariants:
    """Adversarial security stress testing for MockGPTProvider and OpenAIAssistant."""

    def test_mock_gpt_provider_zero_socket_calls_invariant(self, monkeypatch: Any) -> None:
        """Assert MockGPTProvider complete() makes strictly ZERO socket calls."""
        socket_calls: list[str] = []

        def forbidden_socket(*args: Any, **kwargs: Any) -> Any:
            socket_calls.append(f"socket.socket({args}, {kwargs})")
            raise AssertionError("Socket creation attempted by MockGPTProvider!")

        def forbidden_getaddrinfo(*args: Any, **kwargs: Any) -> Any:
            socket_calls.append(f"socket.getaddrinfo({args})")
            raise AssertionError("DNS resolution attempted by MockGPTProvider!")

        monkeypatch.setattr(socket, "socket", forbidden_socket)
        monkeypatch.setattr(socket, "getaddrinfo", forbidden_getaddrinfo)

        provider = MockGPTProvider()
        assert provider.is_available()

        # Execute diverse completion queries triggering multiple heuristics
        resp1 = provider.complete("Analyze model performance: accuracy=0.85, brier=0.18")
        assert resp1.content
        assert resp1.provider == "mock-gpt"

        resp2 = provider.complete(
            "Critique backtest",
            win_rate=0.45,
            payout_rate=0.85,
            total_trades=50,
            max_drawdown=25.0,
        )
        assert "Negative Expectancy Warning" in resp2.content
        assert "Drawdown Warning" in resp2.content
        assert "Sample Size Warning" in resp2.content

        # Verify zero socket calls occurred
        assert len(socket_calls) == 0, f"Socket calls detected: {socket_calls}"

    def test_gpt_research_assistant_zero_socket_calls_invariant(self, monkeypatch: Any) -> None:
        """Assert GPTResearchAssistant high-level methods make strictly ZERO socket calls."""
        socket_calls: list[str] = []

        def forbidden_socket(*args: Any, **kwargs: Any) -> Any:
            socket_calls.append("socket")
            raise AssertionError("Socket creation attempted by GPTResearchAssistant!")

        monkeypatch.setattr(socket, "socket", forbidden_socket)

        assistant = GPTResearchAssistant()  # defaults to MockGPTProvider
        assert assistant.is_available

        # 1. Performance analysis
        out1 = assistant.analyze_model_performance(
            metrics={"accuracy": 0.65, "brier_score": 0.20, "sample_count": 150},
            model_name="AdversarialTestModel",
            train_metrics={"accuracy": 0.95, "brier_score": 0.05},
        )
        assert RESEARCH_DISCLAIMER in out1
        assert "Overfitting Warning" in out1

        # 2. Strategy backtest critique
        out2 = assistant.critique_strategy_backtest(
            backtest_summary={"win_rate": 0.60, "total_trades": 200, "profit_factor": 1.8},
            strategy_name="AdversarialStrategy",
        )
        assert RESEARCH_DISCLAIMER in out2
        assert "Positive Edge" in out2

        # 3. Feature importance explanation
        out3 = assistant.explain_feature_importance(
            importances={"rsi_14": 0.70, "macd": 0.15, "volatility": 0.15},
            model_name="AdversarialModel",
        )
        assert RESEARCH_DISCLAIMER in out3
        assert "Concentration Risk" in out3

        # 4. General critique
        out4 = assistant.generate_critique("Evaluate offline calibration requirements")
        assert RESEARCH_DISCLAIMER in out4

        assert len(socket_calls) == 0

    def test_openai_assistant_fails_closed_when_disabled(self, monkeypatch: Any) -> None:
        """Assert OpenAIAssistant raises LLMSecurityError immediately when disabled."""
        socket_calls: list[str] = []

        def forbidden_socket(*args: Any, **kwargs: Any) -> Any:
            socket_calls.append("socket")
            raise AssertionError("Socket creation attempted while OpenAI adapter disabled!")

        monkeypatch.setattr(socket, "socket", forbidden_socket)

        # Case 1: Default configuration (enabled=False, env not set)
        assistant = OpenAIAssistant()
        assert not assistant.is_available()
        with pytest.raises(
            LLMSecurityError, match="OpenAIAssistant is disabled for offline safety"
        ):
            assistant.complete("Test prompt")

        # Case 2: Constructor enabled=True, but environment variable missing/zero
        monkeypatch.delenv("IQOPTION_ENABLE_OPENAI", raising=False)
        assistant_half = OpenAIAssistant(enabled=True)
        assert not assistant_half.is_available()
        with pytest.raises(LLMSecurityError):
            assistant_half.complete("Test prompt")

        # Case 3: Constructor enabled=False, but environment variable is 1
        monkeypatch.setenv("IQOPTION_ENABLE_OPENAI", "1")
        assistant_env_only = OpenAIAssistant(enabled=False)
        assert not assistant_env_only.is_available()
        with pytest.raises(LLMSecurityError):
            assistant_env_only.complete("Test prompt")

        # Case 4: Both enabled, but API key missing -> LLMProviderUnavailableError
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assistant_no_key = OpenAIAssistant(enabled=True, api_key="")
        assert not assistant_no_key.is_available()
        with pytest.raises(LLMProviderUnavailableError, match="OpenAI API key missing"):
            assistant_no_key.complete("Test prompt")

        # In all cases, assert ZERO socket activity
        assert len(socket_calls) == 0

    def test_gpt_research_assistant_zero_trading_authority_invariant(self) -> None:
        """Assert GPTResearchAssistant strictly possesses ZERO trading authority."""
        assistant = GPTResearchAssistant()

        # 1. Prohibited trading action keywords that MUST NOT exist as methods or attributes
        prohibited_keywords = [
            "buy",
            "sell",
            "order",
            "trade",
            "execute",
            "submit",
            "cancel",
            "deposit",
            "withdraw",
            "transfer",
            "mutate",
            "balance",
            "account",
            "broker",
            "connect",
            "login",
            "ws",
            "websocket",
        ]

        assistant_attrs = dir(assistant)
        for attr in assistant_attrs:
            attr_lower = attr.lower()
            for kw in prohibited_keywords:
                # Disallow exact matches or direct command prefixes
                assert attr_lower != kw, f"Prohibited trading attribute '{attr}' found on assistant"
                assert not attr_lower.startswith(f"{kw}_"), (
                    f"Prohibited method '{attr}' found on assistant"
                )

        # 2. Verify all outputs contain mandatory research disclaimer
        critique = assistant.generate_critique("Test query")
        assert "Zero trading authority" in critique
        assert "Never use as automated trading instruction" in critique

        # 3. Assert assistant cannot be used as a trading strategy (lacks on_bar)
        assert not hasattr(assistant, "on_bar")
        assert not hasattr(assistant, "generate_signal")


# =====================================================================
# 5. End-to-End Comparative Benchmarking Stress Tests
# =====================================================================


class TestEndToEndComparativeBenchmarkStress:
    """Stress tests for ModelComparisonRunner under WalkForward and Holdout modes."""

    def test_compare_models_identical_pair_has_mcnemar_p_one(self) -> None:
        """Assert compare_models runs cleanly on identical models and yields McNemar p=1.0."""
        ds = _make_adversarial_dataset(n=100)
        # Compare MajorityClassModel against another MajorityClassModel
        m1 = MajorityClassModel()
        m2 = MajorityClassModel()

        res = compare_models(
            dataset=ds,
            models={"Maj_1": m1, "Maj_2": m2},
            split_type=EvaluationSplitType.HOLDOUT,
        )

        assert len(res.rows) == 2
        assert res.rows[0].accuracy == res.rows[1].accuracy
        assert res.rows[0].brier_score == res.rows[1].brier_score

        paired_test = res.get_paired_test("Maj_1", "Maj_2")
        assert paired_test is not None
        assert paired_test.p_value == 1.0
        assert paired_test.statistic == 0.0
        assert not paired_test.is_significant

    def test_compare_models_walk_forward_multi_model(self) -> None:
        """Assert compare_models under Walk-Forward evaluation evaluates all candidate models."""
        ds = _make_adversarial_dataset(n=120)
        m1 = MajorityClassModel()
        m2 = SeededRandomModel(seed=42)

        res = compare_models(
            dataset=ds,
            models={"Majority": m1, "Random": m2},
            split_type=EvaluationSplitType.WALK_FORWARD,
            payoff_type=PayoffAssumptionType.BINARY,
        )

        assert len(res.rows) == 2
        for row in res.rows:
            assert row.sample_count > 0
            assert row.brier_score >= 0.0
            assert row.accuracy >= 0.0
            assert row.abstention_rate >= 0.0

        best = res.best_model(RankingMetric.BRIER_SCORE)
        assert best.model_name in ("Majority", "Random")

        md = res.to_markdown()
        assert "# Model Comparison Benchmark: EURUSD" in md
        assert "Majority" in md
        assert "Random" in md
