"""Adversarial stress testing harness for Phase 4 Walk-Forward CV, Purging, and Embargo.

Milestone 5 Empirical Challenge Suite:
1. Suite 5.1: Extreme Horizons (H = 20, 50, 100) and Purge Invariants
2. Suite 5.2: Boundary Dataset Sizes and Minimal Fold Geometry (N_test in {1, 2, 3}, N = N_min)
3. Suite 5.3: Extreme Class Imbalance, Monocultures, and Regime Shifts
4. Suite 5.4: 1000+ Random Configurations Disjointness and Invariant Fuzzer
5. Suite 5.5: Multi-threaded Concurrency and Repeated Determinism
"""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest

from iqoption_research.ml.evaluation.holdout import (
    HoldoutEvaluator,
    HoldoutSplitConfig,
    HoldoutSplitError,
)
from iqoption_research.ml.evaluation.walk_forward import (
    InsufficientDataError,
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardMode,
    WalkForwardResult,
    evaluate_walk_forward,
)
from iqoption_research.ml.models.baselines import LogisticRegressionModel, MajorityClassModel
from iqoption_research.ml.targets.dataset import MLDataset
from iqoption_research.ml.targets.specs import TargetSpec, TargetType


def _make_synth_dataset(
    n: int = 200,
    n_features: int = 4,
    y_values: np.ndarray | None = None,
    seed: int = 42,
) -> MLDataset:
    """Helper creating deterministic MLDataset with valid timestamps and labels."""
    rng = np.random.default_rng(seed)
    feature_names = tuple(f"feat_{j}" for j in range(n_features))
    X = rng.normal(0.0, 1.0, size=(n, n_features)).astype(np.float64)

    if y_values is not None:
        y = y_values.copy().astype(np.int64)
    else:
        p = 1.0 / (1.0 + np.exp(-1.5 * X[:, 0]))
        y = rng.binomial(1, p).astype(np.int64)
        if len(y) > 1:
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


# ==============================================================================
# Suite 5.1: Extreme Horizons (H = 20, 50, 100) and Purge Invariants
# ==============================================================================


class TestExtremeHorizonsAndPurge:
    @pytest.mark.parametrize("h", [20, 50, 100])
    def test_extreme_horizon_purge_math_expanding(self, h: int) -> None:
        """Purging with extreme horizons drops strictly H - 1 trailing train bars."""
        purge_bars = h - 1
        dataset_len = 1000
        train_size = 300
        test_size = 100
        config = WalkForwardConfig(
            n_splits=3,
            train_size=train_size,
            test_size=test_size,
            purge_bars=purge_bars,
            mode=WalkForwardMode.EXPANDING,
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert len(splits) == 3
        for k, split in enumerate(splits):
            raw_train_end = train_size + k * test_size
            expected_train_end = raw_train_end - purge_bars
            assert split.train_indices[0] == 0
            assert split.train_indices[-1] == expected_train_end - 1
            assert len(split.train_indices) == expected_train_end

            # Verify gap between train end and test start is exactly H
            gap = int(split.test_indices[0]) - int(split.train_indices[-1])
            assert gap == h

            # Verify purged interval is strictly omitted
            purged_set = set(range(expected_train_end, raw_train_end))
            train_set = set(split.train_indices)
            test_set = set(split.test_indices)
            assert len(purged_set & train_set) == 0
            assert len(purged_set & test_set) == 0

    @pytest.mark.parametrize("h", [20, 50, 100])
    def test_extreme_horizon_purge_math_rolling(self, h: int) -> None:
        """Rolling window mode with extreme horizons preserves constant window and H gap."""
        purge_bars = h - 1
        dataset_len = 1000
        train_size = 200
        test_size = 80
        step_size = 50
        config = WalkForwardConfig(
            n_splits=4,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            purge_bars=purge_bars,
            mode=WalkForwardMode.ROLLING,
        )
        splits = list(WalkForwardEngine(config).split(dataset_len))

        assert len(splits) == 4
        for k, split in enumerate(splits):
            raw_train_start = k * step_size
            raw_train_end = raw_train_start + train_size
            expected_train_end = raw_train_end - purge_bars

            assert split.train_indices[0] == raw_train_start
            assert split.train_indices[-1] == expected_train_end - 1
            assert len(split.train_indices) == train_size - purge_bars

            gap = int(split.test_indices[0]) - int(split.train_indices[-1])
            assert gap == h

    @pytest.mark.parametrize("h", [20, 50, 100])
    def test_extreme_horizon_with_validation_and_embargo(self, h: int) -> None:
        """Combined extreme purge (H-1) and embargo (E=15) with inner validation."""
        purge_bars = h - 1
        embargo_bars = 15
        dataset_len = 1200
        config = WalkForwardConfig(
            n_splits=2,
            train_size=300,
            val_size=100,
            test_size=100,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            mode=WalkForwardMode.EXPANDING,
        )
        splits = list(WalkForwardEngine(config).split(dataset_len))

        assert len(splits) == 2
        for split in splits:
            assert split.val_indices is not None
            # Purge gap before val: val_start - train_end == H
            gap_purge = int(split.val_indices[0]) - int(split.train_indices[-1])
            assert gap_purge == h

            # Embargo gap after val: test_start - val_end == E + 1
            gap_embargo = int(split.test_indices[0]) - int(split.val_indices[-1])
            assert gap_embargo == embargo_bars + 1

    @pytest.mark.parametrize("h", [20, 50, 100])
    def test_extreme_horizon_evaluation_execution(self, h: int) -> None:
        """Full evaluate_walk_forward execution with extreme horizon purging."""
        ds = _make_synth_dataset(n=600, seed=42)
        config = WalkForwardConfig(
            n_splits=2,
            train_size=200,
            test_size=60,
            purge_bars=h - 1,
            mode=WalkForwardMode.EXPANDING,
        )
        result = evaluate_walk_forward(
            model_factory=lambda: MajorityClassModel(),
            dataset=ds,
            config=config,
            scale_features=False,
        )
        assert isinstance(result, WalkForwardResult)
        assert result.n_folds == 2
        assert len(result.oof_indices) == 120
        assert 0.0 <= result.mean_accuracy <= 1.0

    def test_extreme_horizon_insufficient_data_rejection(self) -> None:
        """When dataset size is smaller than extreme purge requirement, raise error."""
        h = 100
        purge_bars = h - 1  # 99
        config = WalkForwardConfig(
            n_splits=1,
            train_size=20,
            test_size=10,
            purge_bars=purge_bars,
        )
        engine = WalkForwardEngine(config)
        with pytest.raises(InsufficientDataError):
            list(engine.split(100))  # 100 < 20 + 99 + 10 = 129


# ==============================================================================
# Suite 5.2: Boundary Dataset Sizes and Minimal Fold Geometry
# ==============================================================================


class TestBoundarySizesAndMinimalFoldGeometry:
    @pytest.mark.parametrize("n_test", [1, 2, 3])
    def test_minimal_test_fold_sizes(self, n_test: int) -> None:
        """Test folds of size 1, 2, 3 are properly partitioned and evaluated."""
        dataset_len = 100
        config = WalkForwardConfig(
            n_splits=2,
            train_size=40,
            test_size=n_test,
            step_size=n_test,
            min_test_size=1,
            purge_bars=2,
            mode=WalkForwardMode.EXPANDING,
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert len(splits) == 2
        for s in splits:
            assert len(s.test_indices) == n_test

    @pytest.mark.parametrize("n_test", [1, 2, 3])
    def test_minimal_test_fold_evaluate_walk_forward(self, n_test: int) -> None:
        """evaluate_walk_forward executes successfully with minimal test fold sizes 1, 2, 3."""
        ds = _make_synth_dataset(n=100, seed=123)
        config = WalkForwardConfig(
            n_splits=2,
            train_size=40,
            test_size=n_test,
            step_size=n_test,
            min_test_size=1,
            purge_bars=0,
            mode=WalkForwardMode.EXPANDING,
        )
        result = evaluate_walk_forward(
            model_factory=lambda: MajorityClassModel(),
            dataset=ds,
            config=config,
            scale_features=False,
        )
        assert len(result.oof_y_true) == 2 * n_test
        assert 0.0 <= result.mean_accuracy <= 1.0

    def test_exact_minimum_dataset_boundary(self) -> None:
        """Exact minimum dataset boundary: N = train + val + test + purge + embargo.

        Reveals UnboundLocalError in WalkForwardEngine when exact_fit is True and val_bars > 0.
        """
        train_bars = 25
        val_bars = 10
        test_bars = 10
        p = 5
        e = 3
        # Total exact = 25 + 10 + 10 + 5 + 3 = 53
        exact_n = train_bars + val_bars + test_bars + p + e
        config = WalkForwardConfig(
            n_splits=1,
            train_size=train_bars,
            val_size=val_bars,
            test_size=test_bars,
            purge_bars=p,
            embargo_bars=e,
        )
        splits = list(WalkForwardEngine(config).split(exact_n))
        assert len(splits) == 1
        s = splits[0]
        assert len(s.train_indices) == train_bars - p
        assert len(s.val_indices) == val_bars  # type: ignore[arg-type]
        assert len(s.test_indices) == test_bars
        assert int(s.test_indices[-1]) == exact_n - 1

        # N - 1 must raise InsufficientDataError
        with pytest.raises(InsufficientDataError):
            list(WalkForwardEngine(config).split(exact_n - 1))

    def test_holdout_evaluator_exact_boundary_rejection(self) -> None:
        """HoldoutEvaluator rejects datasets with fewer than 10 samples."""
        evaluator = HoldoutEvaluator(HoldoutSplitConfig())
        with pytest.raises(HoldoutSplitError, match=r"Dataset too small"):
            evaluator.calculate_indices(9)

        # 10 samples with default 70/15/15: train_end=7, val_end=8, test_end=10
        idx = evaluator.calculate_indices(10)
        assert idx["train_end"] == 7
        assert idx["val_end"] == 8
        assert idx["test_end"] == 10


# ==============================================================================
# Suite 5.3: Extreme Class Imbalance, Monocultures, and Regime Shifts
# ==============================================================================


class TestClassImbalanceAndRegimeShifts:
    def test_regime_shift_all_zeros_train_all_ones_test(self) -> None:
        """Extreme regime shift: train set is 100% class 0, test set is 100% class 1."""
        n = 100
        y = np.zeros(n, dtype=np.int64)
        # Folds: train is 0..40, test is 40..60. Set test to 1
        y[40:60] = 1
        ds = _make_synth_dataset(n=n, y_values=y)

        config = WalkForwardConfig(
            n_splits=1,
            train_size=40,
            test_size=20,
            purge_bars=0,
            mode=WalkForwardMode.EXPANDING,
        )
        result = evaluate_walk_forward(
            model_factory=lambda: MajorityClassModel(),
            dataset=ds,
            config=config,
            scale_features=False,
        )

        fold = result.folds[0]
        # Majority class trained on all 0s predicts 0, test is all 1s -> accuracy must be 0.0
        assert fold.metrics.accuracy == 0.0
        assert fold.metrics.balanced_accuracy == 0.0
        assert fold.metrics.roc_auc is None  # Single class in test set

    def test_regime_shift_all_ones_train_all_zeros_test(self) -> None:
        """Extreme regime shift: train set is 100% class 1, test set is 100% class 0."""
        n = 100
        y = np.ones(n, dtype=np.int64)
        y[40:60] = 0
        ds = _make_synth_dataset(n=n, y_values=y)

        config = WalkForwardConfig(
            n_splits=1,
            train_size=40,
            test_size=20,
            purge_bars=0,
        )
        result = evaluate_walk_forward(
            model_factory=lambda: MajorityClassModel(),
            dataset=ds,
            config=config,
            scale_features=False,
        )
        fold = result.folds[0]
        assert fold.metrics.accuracy == 0.0
        assert fold.metrics.roc_auc is None

    def test_monoculture_dataset_all_zeros(self) -> None:
        """Complete monoculture: entire dataset is 100% class 0 across all partitions."""
        n = 120
        y = np.zeros(n, dtype=np.int64)
        ds = _make_synth_dataset(n=n, y_values=y)

        config = WalkForwardConfig(
            n_splits=2,
            train_size=50,
            test_size=25,
            purge_bars=2,
        )
        result = evaluate_walk_forward(
            model_factory=lambda: MajorityClassModel(),
            dataset=ds,
            config=config,
            scale_features=True,
        )
        # All 0s correctly predicted by majority model
        assert result.mean_accuracy == 1.0
        assert result.pooled_metrics.accuracy == 1.0
        assert result.mean_roc_auc is None

    def test_extreme_imbalance_logistic_regression(self) -> None:
        """Logistic regression evaluated on highly imbalanced dataset (98% class 0, 2% class 1)."""
        n = 200
        y = np.zeros(n, dtype=np.int64)
        y[[10, 50, 110, 180]] = 1  # Only 4 positive samples
        ds = _make_synth_dataset(n=n, y_values=y, seed=777)

        config = WalkForwardConfig(
            n_splits=2,
            train_size=80,
            test_size=40,
            purge_bars=1,
        )
        result = evaluate_walk_forward(
            model_factory=lambda: LogisticRegressionModel(),
            dataset=ds,
            config=config,
            scale_features=True,
        )
        assert isinstance(result, WalkForwardResult)
        assert 0.0 <= result.mean_accuracy <= 1.0
        assert result.mean_brier_score >= 0.0


# ==============================================================================
# Suite 5.4: 1000+ Random Configurations Disjointness and Invariant Fuzzer
# ==============================================================================


class TestFuzzRandomConfigurations:
    def test_1000_random_walk_forward_configurations_invariants(self) -> None:
        """Fuzz testing 1000+ randomized configurations for strict mathematical invariants."""
        rng = np.random.default_rng(20261107)
        tested_configurations = 0
        valid_split_count = 0
        insufficient_data_count = 0

        for _ in range(1200):
            tested_configurations += 1
            dataset_len = int(rng.integers(30, 800))
            n_splits = int(rng.integers(1, 8))
            mode = rng.choice([WalkForwardMode.EXPANDING, WalkForwardMode.ROLLING])
            purge_bars = int(rng.integers(0, 40))
            embargo_bars = int(rng.integers(0, 20))

            has_val = bool(rng.choice([True, False]))
            val_size = int(rng.integers(5, 30)) if has_val else 0

            test_size = int(rng.integers(5, 40))
            train_size = int(rng.integers(15, 150))
            step_size = (
                int(rng.integers(test_size, test_size + 20)) if rng.choice([True, False]) else None
            )
            anchor = str(rng.choice(["start", "end"]))

            config = WalkForwardConfig(
                n_splits=n_splits,
                train_size=train_size,
                val_size=val_size if val_size > 0 else None,
                test_size=test_size,
                step_size=step_size,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
                mode=mode,
                min_train_size=10,
                min_test_size=5,
                anchor=anchor,
            )

            engine = WalkForwardEngine(config)
            try:
                splits = list(engine.split(dataset_len))
            except InsufficientDataError:
                insufficient_data_count += 1
                continue

            valid_split_count += 1
            assert len(splits) == n_splits

            for split in splits:
                # 1. Read-only arrays invariant
                assert not split.train_indices.flags.writeable
                assert not split.test_indices.flags.writeable
                if split.val_indices is not None:
                    assert not split.val_indices.flags.writeable

                # 2. Bound constraints: all indices within [0, dataset_len - 1]
                assert int(split.train_indices[0]) >= 0
                assert int(split.train_indices[-1]) < dataset_len
                assert int(split.test_indices[0]) >= 0
                assert int(split.test_indices[-1]) < dataset_len
                if split.val_indices is not None and len(split.val_indices) > 0:
                    assert int(split.val_indices[0]) >= 0
                    assert int(split.val_indices[-1]) < dataset_len

                # 3. Disjointness invariants
                train_set = set(split.train_indices)
                test_set = set(split.test_indices)
                assert len(train_set & test_set) == 0, (
                    f"Train and test overlap: {train_set & test_set}"
                )

                if split.val_indices is not None and len(split.val_indices) > 0:
                    val_set = set(split.val_indices)
                    assert len(train_set & val_set) == 0, (
                        f"Train and val overlap: {train_set & val_set}"
                    )
                    assert len(val_set & test_set) == 0, (
                        f"Val and test overlap: {val_set & test_set}"
                    )

                    # 4. Temporal ordering with validation
                    assert split.train_indices[-1] < split.val_indices[0]
                    assert split.val_indices[-1] < split.test_indices[0]

                    # 5. Purge and embargo gap invariants
                    gap_purge = int(split.val_indices[0]) - int(split.train_indices[-1])
                    assert gap_purge >= purge_bars + 1

                    gap_embargo = int(split.test_indices[0]) - int(split.val_indices[-1])
                    assert gap_embargo >= embargo_bars + 1
                else:
                    # Temporal ordering without validation
                    assert split.train_indices[-1] < split.test_indices[0]
                    gap_purge = int(split.test_indices[0]) - int(split.train_indices[-1])
                    assert gap_purge >= purge_bars + 1

            # 6. Test partitions pairwise disjointness when step_size >= test_bars
            actual_step = step_size if step_size is not None else test_size
            if actual_step >= test_size:
                test_sets = [set(s.test_indices) for s in splits]
                for i in range(len(test_sets)):
                    for j in range(i + 1, len(test_sets)):
                        assert len(test_sets[i] & test_sets[j]) == 0

        # Verify we executed over 1000 configurations
        assert tested_configurations >= 1000
        assert valid_split_count > 300
        assert insufficient_data_count > 100


# ==============================================================================
# Suite 5.5: Multi-threaded Concurrency and Determinism
# ==============================================================================


class TestConcurrencyAndDeterminism:
    def test_concurrent_split_generation_thread_safety(self) -> None:
        """WalkForwardEngine instances generate identical splits concurrently across threads."""
        config = WalkForwardConfig(
            n_splits=4,
            train_size=60,
            val_size=20,
            test_size=30,
            purge_bars=5,
            embargo_bars=3,
            mode=WalkForwardMode.EXPANDING,
        )
        dataset_len = 300

        def run_split() -> list[tuple[int, int, int]]:
            engine = WalkForwardEngine(config)
            splits = list(engine.split(dataset_len))
            return [
                (
                    int(s.train_indices[-1]),
                    int(s.val_indices[-1]) if s.val_indices is not None else -1,
                    int(s.test_indices[-1]),
                )
                for s in splits
            ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_split) for _ in range(32)]
            results = [f.result() for f in futures]

        # All thread results must be identical
        baseline = results[0]
        for res in results[1:]:
            assert res == baseline

    def test_concurrent_evaluate_walk_forward_thread_safety(self) -> None:
        """evaluate_walk_forward runs thread-safely across concurrent workers."""
        ds = _make_synth_dataset(n=180, seed=999)
        config = WalkForwardConfig(
            n_splits=3,
            train_size=60,
            test_size=30,
            purge_bars=2,
            mode=WalkForwardMode.EXPANDING,
        )

        def run_eval() -> float:
            res = evaluate_walk_forward(
                model_factory=lambda: MajorityClassModel(),
                dataset=ds,
                config=config,
                scale_features=True,
            )
            return float(res.mean_accuracy)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_eval) for _ in range(8)]
            accs = [f.result() for f in futures]

        for acc in accs:
            assert acc == pytest.approx(accs[0])
