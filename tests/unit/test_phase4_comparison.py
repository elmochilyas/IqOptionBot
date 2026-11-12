"""Unit tests for Phase 4 Comparative Benchmarking Subsystem (Feature 37).

Verifies:
- Dataclass invariants, immutability, and serialization.
- Multi-metric collection under identical datasets, splits, and payoffs.
- Directional ranking with automatic polarity inference and edge cases (inf, None).
- Paired statistical hypothesis tests (McNemar with continuity correction/exact binomial,
  paired t, Wilcoxon).
- Payoff simulation under Binary and Forex assumptions with break-even abstention.
- Tabular and GitHub-flavored Markdown report generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from iqoption_research.ml.evaluation.comparison import (
    ComparisonConfig,
    ComparisonError,
    ComparisonResult,
    EvaluationSplitType,
    ModelBenchmarkRow,
    ModelComparisonRunner,
    ModelEvaluationMismatchError,
    PayoffAssumptionType,
    RankingMetric,
    compare_models,
    compute_mcnemar_test,
    compute_paired_difference_test,
    generate_comparison_markdown_report,
    simulate_model_payoffs,
)
from iqoption_research.ml.evaluation.holdout import HoldoutSplitConfig
from iqoption_research.ml.evaluation.walk_forward import WalkForwardConfig
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
    SeededRandomModel,
)
from iqoption_research.ml.targets.dataset import MLDataset
from iqoption_research.ml.targets.specs import TargetSpec, TargetType


def _make_benchmark_dataset(n: int = 150, d: int = 4, seed: int = 42) -> MLDataset:
    """Deterministic synthetic dataset with timestamps and binary targets."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float64)
    # Target correlated with X[:, 0]
    p = 1.0 / (1.0 + np.exp(-1.8 * X[:, 0]))
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
        feature_names=tuple(f"f_{j}" for j in range(d)),
        X=X,
        y=y,
        timestamps=timestamps,
        target_spec=target_spec,
    )


def _make_sample_row(
    name: str = "TestModel",
    brier: float = 0.20,
    acc: float = 0.60,
    ret: float = 250.0,
    pf: float | None = 1.5,
    max_dd: float = 100.0,
    n_samples: int = 50,
) -> ModelBenchmarkRow:
    """Helper creating a valid ModelBenchmarkRow instance."""
    y_true = np.ones(n_samples, dtype=np.int64)
    y_pred = np.ones(n_samples, dtype=np.int64)
    y_prob = np.column_stack([np.zeros(n_samples), np.ones(n_samples)])
    return ModelBenchmarkRow(
        model_name=name,
        model_type="baseline",
        accuracy=acc,
        balanced_accuracy=acc,
        precision=acc,
        recall=acc,
        f1_macro=acc,
        brier_score=brier,
        log_loss=0.55,
        roc_auc=0.65,
        ece=0.03,
        trade_count=n_samples,
        win_rate=acc,
        total_return=ret,
        profit_factor=pf,
        max_drawdown=max_dd,
        abstention_rate=0.10,
        sample_count=n_samples,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        trade_pnls=(10.0,) * n_samples,
    )


class TestComparisonDataStructures:
    """Verifies dataclass immutability, validations, and serialization."""

    def test_model_benchmark_row_immutability_and_slots(self) -> None:
        row = _make_sample_row()
        with pytest.raises(AttributeError):
            row.accuracy = 0.99  # type: ignore[misc]

        # Array immutability
        assert row.y_true is not None
        with pytest.raises(ValueError, match="read-only"):
            row.y_true[0] = 99

    def test_model_benchmark_row_to_dict_roundtrip(self) -> None:
        row = _make_sample_row(pf=float("inf"))
        d = row.to_dict(include_arrays=True)
        assert d["model_name"] == "TestModel"
        assert d["profit_factor"] == "inf"
        assert "y_true" in d
        assert len(d["y_true"]) == 50

        d_no_arrays = row.to_dict(include_arrays=False)
        assert "y_true" not in d_no_arrays

    def test_comparison_config_parameter_validation(self) -> None:
        with pytest.raises(ComparisonError, match="split_type must be"):
            ComparisonConfig(split_type="invalid_split")

        with pytest.raises(ComparisonError, match="payoff_type must be"):
            ComparisonConfig(payoff_type="crypto_futures")

        with pytest.raises(ComparisonError, match="binary_payout_rate must be > 0"):
            ComparisonConfig(binary_payout_rate=-0.1)

        with pytest.raises(ComparisonError, match="stake_amount must be > 0"):
            ComparisonConfig(stake_amount=0.0)

        with pytest.raises(ComparisonError, match="alpha must be in"):
            ComparisonConfig(alpha=1.5)

    def test_comparison_result_get_row_and_model_names(self) -> None:
        r1 = _make_sample_row(name="ModelA")
        r2 = _make_sample_row(name="ModelB")
        result = ComparisonResult(
            rows=(r1, r2),
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={"target_type": "direction_binary"},
        )

        assert result.model_names == ("ModelA", "ModelB")
        assert result.get_row("ModelA").model_name == "ModelA"
        with pytest.raises(KeyError, match="not found"):
            result.get_row("NonExistent")

    def test_comparison_result_to_dataframe(self) -> None:
        r1 = _make_sample_row(name="ModelA")
        result = ComparisonResult(
            rows=(r1,),
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={},
        )
        df = result.to_dataframe()
        assert len(df) == 1
        assert "model_name" in df.columns
        assert "brier_score" in df.columns


class TestRankingAndSorting:
    """Verifies leaderboard ranking logic across different metrics and polarities."""

    def test_rank_by_brier_score_ascending(self) -> None:
        r1 = _make_sample_row(name="M1", brier=0.25)
        r2 = _make_sample_row(name="M2", brier=0.18)
        r3 = _make_sample_row(name="M3", brier=0.22)

        res = ComparisonResult(
            rows=(r1, r2, r3),
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={},
        )
        ranked = res.rank_by(RankingMetric.BRIER_SCORE)
        assert [r.model_name for r in ranked] == ["M2", "M3", "M1"]
        assert res.best_model(RankingMetric.BRIER_SCORE).model_name == "M2"

    def test_rank_by_profit_factor_descending(self) -> None:
        r1 = _make_sample_row(name="M1", pf=1.2)
        r2 = _make_sample_row(name="M2", pf=2.5)
        r3 = _make_sample_row(name="M3", pf=0.8)

        res = ComparisonResult(
            rows=(r1, r2, r3),
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={},
        )
        ranked = res.rank_by(RankingMetric.PROFIT_FACTOR)
        assert [r.model_name for r in ranked] == ["M2", "M1", "M3"]

    def test_ranking_handles_infinite_and_none_profit_factors(self) -> None:
        r_inf = _make_sample_row(name="InfModel", pf=float("inf"))
        r_fin = _make_sample_row(name="FinModel", pf=2.0)
        r_none = _make_sample_row(name="NoneModel", pf=None)

        res = ComparisonResult(
            rows=(r_none, r_fin, r_inf),
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={},
        )
        ranked = res.rank_by(RankingMetric.PROFIT_FACTOR)
        # Infinite is top (best), finite middle, None at bottom (worst)
        assert [r.model_name for r in ranked] == ["InfModel", "FinModel", "NoneModel"]

    def test_ranking_unknown_metric_raises_error(self) -> None:
        res = ComparisonResult(
            rows=(_make_sample_row(),),
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={},
        )
        with pytest.raises(ComparisonError, match="Unknown ranking metric"):
            res.rank_by("non_existent_metric")


class TestPayoffSimulationEngine:
    """Verifies financial payoff logic, break-even hurdles, and drawdown tracking."""

    def test_binary_payoff_win_and_loss(self) -> None:
        config = ComparisonConfig(
            payoff_type=PayoffAssumptionType.BINARY,
            binary_payout_rate=0.85,
            stake_amount=100.0,
            initial_capital=1000.0,
        )
        # Two observations: 1 win, 1 loss
        y_true = np.array([1, 0])
        y_prob = np.array([[0.2, 0.8], [0.2, 0.8]])  # CALL on both

        t_count, win_rate, ret, pf, max_dd, abs_rate, pnls = simulate_model_payoffs(
            y_true, y_prob, config
        )
        assert t_count == 2
        assert win_rate == 0.50
        # 1 win = +85, 1 loss = -100 => net return = -15
        assert ret == pytest.approx(-15.0)
        # pf = 85 / 100 = 0.85
        assert pf == pytest.approx(0.85)
        assert max_dd == pytest.approx(100.0)
        assert abs_rate == 0.0
        assert pnls == (85.0, -100.0)

    def test_binary_break_even_abstention_on_low_confidence(self) -> None:
        config = ComparisonConfig(
            payoff_type=PayoffAssumptionType.BINARY,
            binary_payout_rate=0.85,  # break-even = 54.05%
        )
        y_true = np.array([1, 0])
        # Probabilities 0.52 and 0.48 are within [0.4595, 0.5405] break-even zone => abstain
        y_prob = np.array([[0.48, 0.52], [0.52, 0.48]])

        t_count, win_rate, ret, pf, _max_dd, abs_rate, _ = simulate_model_payoffs(
            y_true, y_prob, config
        )
        assert t_count == 0
        assert win_rate is None
        assert ret == 0.0
        assert pf is None
        assert abs_rate == 1.0

    def test_exact_half_half_abstention(self) -> None:
        config = ComparisonConfig(
            payoff_type=PayoffAssumptionType.BINARY,
            binary_payout_rate=0.85,
        )
        y_true = np.array([1, 0])
        y_prob = np.array([[0.5, 0.5], [0.5, 0.5]])

        t_count, _, _, _, _, abs_rate, _ = simulate_model_payoffs(y_true, y_prob, config)
        assert t_count == 0
        assert abs_rate == 1.0

    def test_forex_payoff_simulation(self) -> None:
        config = ComparisonConfig(
            payoff_type=PayoffAssumptionType.FOREX,
            forex_spread_pips=1.0,
            forex_slippage_pips=0.5,
            stake_amount=100.0,
        )
        y_true = np.array([1, 0])
        # Both confident: CALL on 1, PUT on 0
        y_prob = np.array([[0.2, 0.8], [0.8, 0.2]])

        t_count, win_rate, ret, pf, _, _, _ = simulate_model_payoffs(
            y_true, y_prob, config
        )
        assert t_count == 2
        assert win_rate == 1.0
        assert ret > 0.0
        assert pf == float("inf")


class TestStatisticalHypothesisTests:
    """Verifies McNemar's test and continuous paired loss difference tests."""

    def test_mcnemar_identical_models_yields_p_value_one(self) -> None:
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

        res = compute_mcnemar_test(
            y_true=y_true,
            y_pred_a=y_pred,
            y_pred_b=y_pred,
            model_a_name="A",
            model_b_name="B",
        )
        assert res.statistic == 0.0
        assert res.p_value == 1.0
        assert not res.is_significant
        assert res.contingency_table is not None

    def test_mcnemar_exact_binomial_for_small_discordant_count(self) -> None:
        # Small sample with discordant pairs < 25
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 10/10 correct
        y_b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 0/10 correct (10 discordant pairs)

        res = compute_mcnemar_test(
            y_true=y_true,
            y_pred_a=y_a,
            y_pred_b=y_b,
            alpha=0.05,
        )
        assert "Exact binomial test" in res.notes
        assert res.p_value < 0.01
        assert res.is_significant

    def test_mcnemar_sample_mismatch_raises_error(self) -> None:
        y_t = np.array([1, 0, 1])
        y_a = np.array([1, 0])
        y_b = np.array([1, 0, 1])

        with pytest.raises(ModelEvaluationMismatchError, match="Sample length mismatch"):
            compute_mcnemar_test(y_t, y_a, y_b)

    def test_paired_loss_difference_tests(self) -> None:
        loss_a = np.array([0.1, 0.15, 0.12, 0.08, 0.14, 0.11])
        loss_b = np.array([0.25, 0.28, 0.30, 0.22, 0.26, 0.27])

        res_t = compute_paired_difference_test(loss_a, loss_b, test_type="paired_t")
        assert res_t.p_value < 0.01
        assert res_t.is_significant

        res_w = compute_paired_difference_test(loss_a, loss_b, test_type="wilcoxon")
        assert res_w.is_significant


class TestEndToEndComparativeBenchmark:
    """Verifies complete benchmarking workflows in Walk-Forward and Holdout configurations."""

    def test_walk_forward_benchmark_multiple_models(self) -> None:
        dataset = _make_benchmark_dataset(n=150, d=3)
        wf_cfg = WalkForwardConfig(
            n_splits=3, train_size=60, test_size=20, step_size=20
        )
        comp_cfg = ComparisonConfig(


            split_type=EvaluationSplitType.WALK_FORWARD,
            walk_forward_config=wf_cfg,
        )

        runner = ModelComparisonRunner(config=comp_cfg)
        models = {
            "logistic": LogisticRegressionModel(random_state=42),
            "majority": MajorityClassModel(),
            "random": SeededRandomModel(seed=99),
        }

        result = runner.run(dataset=dataset, models=models)
        assert len(result.rows) == 3
        assert result.symbol == "EURUSD"

        # Check all models evaluated on identical sample counts
        sample_counts = [r.sample_count for r in result.rows]
        assert sample_counts[0] == sample_counts[1] == sample_counts[2]

        # Check pairwise test matrices computed
        assert len(result.paired_tests) == 3  # 3 choose 2 = 3 pairs

    def test_holdout_benchmark_with_calibration(self) -> None:
        dataset = _make_benchmark_dataset(n=120, d=3)
        ho_cfg = HoldoutSplitConfig(train_ratio=0.60, val_ratio=0.20, test_ratio=0.20)

        models = [
            LogisticRegressionModel(random_state=42),
            MajorityClassModel(),
        ]

        result = compare_models(
            dataset=dataset,
            models=models,
            split_type="holdout",
            holdout_config=ho_cfg,
            calibrator_method="sigmoid",
        )
        assert len(result.rows) == 2
        for r in result.rows:
            assert r.brier_score >= 0.0
            assert r.sample_count > 0

    def test_empty_dataset_raises_comparison_error(self) -> None:
        empty_ds = MLDataset(
            symbol="EURUSD",
            feature_names=("f0",),
            X=np.empty((0, 1)),
            y=np.empty(0, dtype=np.int64),
            timestamps=np.empty(0, dtype=np.int64),
            target_spec=TargetSpec(target_type=TargetType.DIRECTION_BINARY, horizon_bars=1),
        )
        runner = ModelComparisonRunner()
        with pytest.raises(ComparisonError, match="empty dataset"):
            runner.run(empty_ds, [MajorityClassModel()])


class TestMarkdownReportGeneration:
    """Verifies GitHub-flavored markdown report output structure and formatting."""

    def test_markdown_report_structure_and_pipes(self) -> None:
        r1 = _make_sample_row(name="ModelA", brier=0.15, acc=0.65, ret=300.0, pf=2.1)
        r2 = _make_sample_row(name="ModelB", brier=0.24, acc=0.52, ret=-50.0, pf=0.8)

        test_res = compute_mcnemar_test(
            y_true=r1.y_true,  # type: ignore[arg-type]
            y_pred_a=r1.y_pred,  # type: ignore[arg-type]
            y_pred_b=r2.y_pred,  # type: ignore[arg-type]
            model_a_name="ModelA",
            model_b_name="ModelB",
        )

        result = ComparisonResult(
            rows=(r1, r2),
            config=ComparisonConfig(),
            symbol="EURUSD",
            target_spec_dict={"target_type": "direction_binary"},
            paired_tests={("ModelA", "ModelB"): test_res},
            metadata={"dataset_rows": 100},
        )

        md = generate_comparison_markdown_report(result)
        assert "# Model Comparison Benchmark: EURUSD" in md
        assert "## Model Performance Leaderboard (Ranked by Brier Score)" in md
        assert "| Rank | Model Name |" in md
        assert "**ModelA**" in md
        assert "**ModelB**" in md
        assert "## Statistical Significance Matrix" in md
        assert "## Key Findings" in md
