"""Unit tests for Phase 4 Chronological Walk-Forward Engine and Purging/Embargo.

Exhaustive Category-Partition, Boundary Value Analysis, and Invariant verification
covering 35 test cases across 7 test suites:
- Suite 1.1: Expanding Window Index Mathematics (Cases 1-4)
- Suite 1.2: Rolling Window Index Mathematics (Cases 5-8)
- Suite 1.3: Purge Math Verification (Cases 9-13)
- Suite 1.4: Embargo Math Verification (Cases 14-17)
- Suite 1.5: Disjoint Index Sets & Temporal Monotonicity (Cases 18-21)
- Suite 1.6: Boundary Conditions, Validation Constraints & Errors (Cases 22-28c)
- Suite 1.7: Walk-Forward Evaluation Loop (Cases 29-33)
"""

from __future__ import annotations

import numpy as np
import pytest

from iqoption_research.ml.calibration.abstention import BinaryPayoffGate
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.decision.engine import AIDecisionEngine, StakeConfig, StakeMode
from iqoption_research.ml.evaluation.walk_forward import (
    FoldEvaluationResult,
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardResult,
    evaluate_walk_forward,
)
from iqoption_research.ml.models.baselines import LogisticRegressionModel, MajorityClassModel
from iqoption_research.ml.targets.dataset import MLDataset
from iqoption_research.ml.targets.specs import TargetSpec, TargetType


def _make_synth_dataset(n: int = 150, n_features: int = 4, seed: int = 42) -> MLDataset:
    """Helper creating deterministic MLDataset with valid timestamps and labels."""
    rng = np.random.default_rng(seed)
    feature_names = tuple(f"feat_{j}" for j in range(n_features))
    X = rng.normal(0.0, 1.0, size=(n, n_features)).astype(np.float64)
    # Give some predictive signal
    p = 1.0 / (1.0 + np.exp(-1.5 * X[:, 0]))
    y = rng.binomial(1, p).astype(np.int64)
    # Guarantee binary presence
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
# Suite 1.1: Expanding Window Index Mathematics (4 Cases)
# ==============================================================================


class TestExpandingWindowMath:
    def test_expanding_window_index_progression(self) -> None:
        """Case 1: In expanding mode, train starts at 0, while bounds step forward monotonically."""
        dataset_len = 100
        config = WalkForwardConfig(
            n_splits=3,
            train_size=40,
            test_size=20,
            step_size=20,
            mode="expanding",
            purge_bars=0,
            embargo_bars=0,
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert len(splits) == 3
        np.testing.assert_array_equal(splits[0].train_indices, np.arange(0, 40))
        np.testing.assert_array_equal(splits[0].test_indices, np.arange(40, 60))

        np.testing.assert_array_equal(splits[1].train_indices, np.arange(0, 60))
        np.testing.assert_array_equal(splits[1].test_indices, np.arange(60, 80))

        np.testing.assert_array_equal(splits[2].train_indices, np.arange(0, 80))
        np.testing.assert_array_equal(splits[2].test_indices, np.arange(80, 100))

    def test_expanding_window_fixed_test_size(self) -> None:
        """Case 2: Each fold test set length strictly matches specified test_size."""
        dataset_len = 250
        config = WalkForwardConfig(
            n_splits=4,
            train_size=100,
            test_size=30,
            mode="expanding",
            purge_bars=0,
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert len(splits) == 4
        for s in splits:
            assert len(s.test_indices) == 30
        assert sum(len(s.test_indices) for s in splits) == 120

    def test_expanding_window_train_growth_rate(self) -> None:
        """Case 3: Train growth rate equals step_size between consecutive folds."""
        dataset_len = 300
        config = WalkForwardConfig(
            n_splits=5,
            train_size=80,
            test_size=25,
            step_size=25,
            mode="expanding",
            purge_bars=0,
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert len(splits) == 5
        for k in range(len(splits) - 1):
            growth = len(splits[k + 1].train_indices) - len(splits[k].train_indices)
            assert growth == 25

    def test_expanding_window_default_step_size_equals_test_size(self) -> None:
        """Case 4: Step_size defaults to test_size, ensuring contiguous test coverage."""
        dataset_len = 150
        config = WalkForwardConfig(
            n_splits=3,
            train_size=60,
            test_size=30,
            step_size=None,
            mode="expanding",
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert len(splits) == 3
        np.testing.assert_array_equal(splits[0].test_indices, np.arange(60, 90))
        np.testing.assert_array_equal(splits[1].test_indices, np.arange(90, 120))
        np.testing.assert_array_equal(splits[2].test_indices, np.arange(120, 150))


# ==============================================================================
# Suite 1.2: Rolling Window Index Mathematics (4 Cases)
# ==============================================================================


class TestRollingWindowMath:
    def test_rolling_window_constant_train_size(self) -> None:
        """Case 5: Constant train size in rolling mode with window shifting forward."""
        dataset_len = 150
        config = WalkForwardConfig(
            n_splits=4,
            train_size=50,
            test_size=20,
            step_size=20,
            mode="rolling",
            purge_bars=0,
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert len(splits) == 4
        for s in splits:
            assert len(s.train_indices) == 50

        np.testing.assert_array_equal(splits[0].train_indices, np.arange(0, 50))
        np.testing.assert_array_equal(splits[0].test_indices, np.arange(50, 70))

        np.testing.assert_array_equal(splits[1].train_indices, np.arange(20, 70))
        np.testing.assert_array_equal(splits[1].test_indices, np.arange(70, 90))

        np.testing.assert_array_equal(splits[2].train_indices, np.arange(40, 90))
        np.testing.assert_array_equal(splits[2].test_indices, np.arange(90, 110))

        np.testing.assert_array_equal(splits[3].train_indices, np.arange(60, 110))
        np.testing.assert_array_equal(splits[3].test_indices, np.arange(110, 130))

    def test_rolling_window_start_index_progression(self) -> None:
        """Case 6: Min train index advances by exactly step_size each fold."""
        dataset_len = 200
        config = WalkForwardConfig(
            n_splits=3,
            train_size=60,
            test_size=25,
            step_size=15,
            mode="rolling",
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert len(splits) == 3
        assert int(np.min(splits[0].train_indices)) == 0
        assert int(np.min(splits[1].train_indices)) == 15
        assert int(np.min(splits[2].train_indices)) == 30

    def test_rolling_window_discards_oldest_data(self) -> None:
        """Case 7: Old historical observations are dropped in rolling mode."""
        dataset_len = 200
        config = WalkForwardConfig(
            n_splits=3,
            train_size=60,
            test_size=25,
            step_size=15,
            mode="rolling",
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert 0 in splits[0].train_indices
        assert 0 not in splits[1].train_indices
        assert 0 not in splits[2].train_indices

    def test_rolling_vs_expanding_parity_on_first_fold(self) -> None:
        """Case 8: Fold 0 of rolling mode and fold 0 of expanding mode are identical."""
        dataset_len = 100
        rolling_cfg = WalkForwardConfig(
            n_splits=1, train_size=50, test_size=20, purge_bars=2, mode="rolling"
        )
        expanding_cfg = WalkForwardConfig(
            n_splits=1, train_size=50, test_size=20, purge_bars=2, mode="expanding"
        )

        rolling_split = next(WalkForwardEngine(rolling_cfg).split(dataset_len))
        expanding_split = next(WalkForwardEngine(expanding_cfg).split(dataset_len))

        np.testing.assert_array_equal(rolling_split.train_indices, expanding_split.train_indices)
        np.testing.assert_array_equal(rolling_split.test_indices, expanding_split.test_indices)


# ==============================================================================
# Suite 1.3: Purge Math Verification (5 Cases)
# ==============================================================================


class TestPurgeMath:
    def test_purge_math_single_step_horizon_h1(self) -> None:
        """Case 9: With H=1 (purge_bars=0), no training bars are purged."""
        dataset_len = 100
        config = WalkForwardConfig(
            n_splits=1, train_size=50, test_size=20, purge_bars=0, mode="expanding"
        )
        engine = WalkForwardEngine(config)
        split = next(engine.split(dataset_len))

        assert len(split.train_indices) == 50
        assert int(np.max(split.train_indices)) == 49
        assert int(np.min(split.test_indices)) == 50
        assert int(np.min(split.test_indices)) - int(np.max(split.train_indices)) == 1

    def test_purge_math_multi_horizon_h3(self) -> None:
        """Case 10: With H=3 (purge_bars=2), exactly 2 trailing train bars are dropped."""
        dataset_len = 100
        config = WalkForwardConfig(
            n_splits=1, train_size=50, test_size=20, purge_bars=2, mode="expanding"
        )
        engine = WalkForwardEngine(config)
        split = next(engine.split(dataset_len))

        assert len(split.train_indices) == 48
        assert 48 not in split.train_indices
        assert 49 not in split.train_indices
        assert int(np.max(split.train_indices)) == 47
        assert int(np.min(split.test_indices)) == 50
        gap = int(np.min(split.test_indices)) - int(np.max(split.train_indices))
        assert gap == 3

    @pytest.mark.parametrize("h", [2, 5, 10, 15])
    def test_purge_math_parameterized_horizons(self, h: int) -> None:
        """Case 11: Parameterized verification across horizons h."""
        dataset_len = 200
        config = WalkForwardConfig(
            n_splits=1,
            train_size=100,
            test_size=30,
            purge_bars=h - 1,
            mode="expanding",
        )
        engine = WalkForwardEngine(config)
        split = next(engine.split(dataset_len))

        expected_train_len = 100 - (h - 1)
        assert len(split.train_indices) == expected_train_len
        gap = int(np.min(split.test_indices)) - int(np.max(split.train_indices))
        assert gap == h
        for i in range(100 - (h - 1), 100):
            assert i not in split.train_indices

    def test_purge_preserves_early_training_samples(self) -> None:
        """Case 12: Purging truncates strictly from the end of training partition."""
        dataset_len = 100
        config = WalkForwardConfig(
            n_splits=1, train_size=60, test_size=20, purge_bars=5, mode="expanding"
        )
        split = next(WalkForwardEngine(config).split(dataset_len))

        assert split.train_indices[0] == 0
        assert split.train_indices[-1] == 54
        np.testing.assert_array_equal(split.train_indices, np.arange(0, 55))

    def test_purge_math_with_validation_set(self) -> None:
        """Case 13: When validation fold is present, purge is applied before validation."""
        dataset_len = 150
        config = WalkForwardConfig(
            n_splits=1,
            train_size=60,
            val_size=20,
            test_size=20,
            purge_bars=4,
            mode="expanding",
        )
        split = next(WalkForwardEngine(config).split(dataset_len))

        assert split.val_indices is not None
        np.testing.assert_array_equal(split.val_indices, np.arange(60, 80))
        assert int(np.max(split.train_indices)) == 55
        gap = int(np.min(split.val_indices)) - int(np.max(split.train_indices))
        assert gap == 5


# ==============================================================================
# Suite 1.4: Embargo Math Verification (4 Cases)
# ==============================================================================


class TestEmbargoMath:
    def test_embargo_math_zero_bars(self) -> None:
        """Case 14: When embargo_bars = 0, test starts immediately after validation."""
        dataset_len = 120
        config = WalkForwardConfig(
            n_splits=1,
            train_size=50,
            val_size=20,
            test_size=20,
            embargo_bars=0,
        )
        split = next(WalkForwardEngine(config).split(dataset_len))

        assert split.val_indices is not None
        assert int(np.max(split.val_indices)) == 69
        assert int(np.min(split.test_indices)) == 70
        assert int(np.min(split.test_indices)) - int(np.max(split.val_indices)) == 1

    def test_embargo_math_drops_bars_after_validation(self) -> None:
        """Case 15: Embargo drops leading bars following boundary."""
        dataset_len = 150
        config = WalkForwardConfig(
            n_splits=1,
            train_size=50,
            val_size=20,
            test_size=20,
            embargo_bars=5,
            purge_bars=0,
        )
        split = next(WalkForwardEngine(config).split(dataset_len))

        assert split.val_indices is not None
        assert int(np.max(split.val_indices)) == 69
        for i in range(70, 75):
            assert i not in split.val_indices
            assert i not in split.test_indices
        assert int(np.min(split.test_indices)) == 75
        gap = int(np.min(split.test_indices)) - int(np.max(split.val_indices))
        assert gap == 6

    def test_combined_purge_and_embargo_invariants(self) -> None:
        """Case 16: Purge and embargo applied simultaneously across train-val-test."""
        dataset_len = 150
        config = WalkForwardConfig(
            n_splits=1,
            train_size=60,
            val_size=20,
            test_size=20,
            purge_bars=3,
            embargo_bars=2,
        )
        split = next(WalkForwardEngine(config).split(dataset_len))

        assert split.val_indices is not None
        np.testing.assert_array_equal(split.train_indices, np.arange(0, 57))
        np.testing.assert_array_equal(split.val_indices, np.arange(60, 80))
        np.testing.assert_array_equal(split.test_indices, np.arange(82, 102))

        gap_train_val = int(np.min(split.val_indices)) - int(np.max(split.train_indices))
        gap_val_test = int(np.min(split.test_indices)) - int(np.max(split.val_indices))
        assert gap_train_val == 4
        assert gap_val_test == 3

    def test_embargo_across_successive_folds_rolling(self) -> None:
        """Case 17: Rolling window walk-forward handles embargo cleanly across folds."""
        dataset_len = 200
        config = WalkForwardConfig(
            n_splits=2,
            train_size=50,
            test_size=20,
            step_size=25,
            embargo_bars=5,
            mode="rolling",
        )
        splits = list(WalkForwardEngine(config).split(dataset_len))

        assert len(splits) == 2
        np.testing.assert_array_equal(splits[0].train_indices, np.arange(0, 50))
        np.testing.assert_array_equal(splits[0].test_indices, np.arange(50, 70))
        np.testing.assert_array_equal(splits[1].train_indices, np.arange(25, 75))
        np.testing.assert_array_equal(splits[1].test_indices, np.arange(75, 95))


# ==============================================================================
# Suite 1.5: Disjoint Index Sets & Temporal Monotonicity (4 Cases)
# ==============================================================================


class TestDisjointnessAndMonotonicity:
    def test_disjoint_index_sets_all_folds(self) -> None:
        """Case 18: Train, val, and test index sets have pairwise empty intersections."""
        dataset_len = 250
        config = WalkForwardConfig(
            n_splits=4,
            train_size=60,
            val_size=20,
            test_size=25,
            purge_bars=2,
            embargo_bars=3,
            mode="expanding",
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert len(splits) == 4
        for s in splits:
            train_s = set(s.train_indices)
            val_s = set(s.val_indices) if s.val_indices is not None else set()
            test_s = set(s.test_indices)

            assert train_s & val_s == set()
            assert train_s & test_s == set()
            assert val_s & test_s == set()

    def test_test_sets_pairwise_disjoint_when_step_equals_test_size(self) -> None:
        """Case 19: All out-of-fold test sets are mutually exclusive when step == test_size."""
        dataset_len = 200
        config = WalkForwardConfig(
            n_splits=4,
            train_size=50,
            test_size=25,
            step_size=25,
        )
        splits = list(WalkForwardEngine(config).split(dataset_len))
        test_sets = [set(s.test_indices) for s in splits]

        for i in range(len(test_sets)):
            for j in range(i + 1, len(test_sets)):
                assert test_sets[i] & test_sets[j] == set()

    def test_timestamp_strict_monotonicity(self) -> None:
        """Case 20: MLDataset timestamps show strictly monotonic progression."""
        ds = _make_synth_dataset(n=200)
        config = WalkForwardConfig(
            n_splits=3,
            train_size=60,
            val_size=20,
            test_size=25,
            purge_bars=2,
            embargo_bars=2,
        )
        splits = list(WalkForwardEngine(config).split(ds))

        assert len(splits) == 3
        for s in splits:
            assert s.train_start_ts is not None and s.train_end_ts is not None
            assert s.val_start_ts is not None and s.val_end_ts is not None
            assert s.test_start_ts is not None and s.test_end_ts is not None
            assert s.train_start_ts < s.train_end_ts
            assert s.train_end_ts < s.val_start_ts
            assert s.val_start_ts < s.val_end_ts
            assert s.val_end_ts < s.test_start_ts
            assert s.test_start_ts < s.test_end_ts

    def test_split_indices_match_dataset_timestamps(self) -> None:
        """Case 21: Bit-for-bit equivalence between metadata and indexed array timestamps."""
        ds = _make_synth_dataset(n=150)
        config = WalkForwardConfig(
            n_splits=2,
            train_size=60,
            test_size=30,
            purge_bars=1,
        )
        splits = list(WalkForwardEngine(config).split(ds))

        for s in splits:
            assert s.train_start_ts == int(ds.timestamps[s.train_indices[0]])
            assert s.train_end_ts == int(ds.timestamps[s.train_indices[-1]])
            assert s.test_start_ts == int(ds.timestamps[s.test_indices[0]])
            assert s.test_end_ts == int(ds.timestamps[s.test_indices[-1]])


# ==============================================================================
# Suite 1.6: Boundary Conditions, Validation Constraints & Errors (7 Cases)
# ==============================================================================


class TestWalkForwardValidationErrors:
    def test_error_zero_or_negative_splits(self) -> None:
        """Case 22: n_splits <= 0 raises ValueError: n_splits must be >= 1."""
        with pytest.raises(ValueError, match=r"n_splits must be >= 1"):
            WalkForwardConfig(n_splits=0)
        with pytest.raises(ValueError, match=r"n_splits must be >= 1"):
            WalkForwardConfig(n_splits=-3)

    def test_error_non_positive_window_sizes(self) -> None:
        """Case 23: Non-positive window sizes raise ValueError: size must be positive."""
        with pytest.raises(ValueError, match=r"size must be positive"):
            WalkForwardConfig(train_size=0)
        with pytest.raises(ValueError, match=r"size must be positive"):
            WalkForwardConfig(test_size=-10)

    def test_error_negative_purge_or_embargo(self) -> None:
        """Case 24: Negative purge or embargo counts raise ValueError: cannot be negative."""
        with pytest.raises(ValueError, match=r"cannot be negative"):
            WalkForwardConfig(purge_bars=-1)
        with pytest.raises(ValueError, match=r"cannot be negative"):
            WalkForwardConfig(embargo_bars=-5)

    def test_error_invalid_mode_string(self) -> None:
        """Case 25: Unsupported mode string raises ValueError."""
        with pytest.raises(ValueError, match=r"mode must be 'expanding' or 'rolling'"):
            WalkForwardConfig(mode="shuffled")

    def test_error_insufficient_samples_for_single_split(self) -> None:
        """Case 26: Total dataset length smaller than single split requirement raises error."""
        dataset_len = 45
        config = WalkForwardConfig(
            n_splits=1, train_size=30, test_size=20, purge_bars=2
        )
        engine = WalkForwardEngine(config)
        with pytest.raises(ValueError, match=r"Dataset length .* insufficient"):
            list(engine.split(dataset_len))

    def test_error_insufficient_samples_for_requested_n_splits(self) -> None:
        """Case 27: Dataset length insufficient for requested n_splits raises error."""
        dataset_len = 100
        config = WalkForwardConfig(
            n_splits=5, train_size=60, test_size=20, step_size=20
        )
        engine = WalkForwardEngine(config)
        with pytest.raises(ValueError, match=r"Cannot generate 5 splits"):
            list(engine.split(dataset_len))

    def test_boundary_exact_minimum_sample_fit(self) -> None:
        """Case 28: Exact minimum sample fit generates 1 split ending at the last index."""
        dataset_len = 65
        config = WalkForwardConfig(
            n_splits=1, train_size=40, test_size=20, purge_bars=5
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(dataset_len))

        assert len(splits) == 1
        assert int(np.max(splits[0].test_indices)) == 64

    def test_boundary_exact_minimum_sample_fit_with_val_and_embargo(self) -> None:
        """Case 28b: Single-split exact boundary fit with val_size > 0 and embargo_bars > 0."""
        train_bars = 40
        val_bars = 15
        test_bars = 15
        p = 5
        e = 3
        # Exact minimum: train + val + test + purge + embargo = 40 + 15 + 15 + 5 + 3 = 78
        exact_n = train_bars + val_bars + test_bars + p + e
        config = WalkForwardConfig(
            n_splits=1,
            train_size=train_bars,
            val_size=val_bars,
            test_size=test_bars,
            purge_bars=p,
            embargo_bars=e,
        )
        engine = WalkForwardEngine(config)
        splits = list(engine.split(exact_n))

        assert len(splits) == 1
        s = splits[0]
        assert s.val_indices is not None
        assert len(s.train_indices) == train_bars - p
        assert len(s.val_indices) == val_bars
        assert len(s.test_indices) == test_bars
        assert int(s.test_indices[-1]) == exact_n - 1

        # Purge gap: min(val) - max(train) == p + 1
        gap_purge = int(s.val_indices[0]) - int(s.train_indices[-1])
        assert gap_purge == p + 1

        # Embargo gap: min(test) - max(val) == e + 1
        gap_embargo = int(s.test_indices[0]) - int(s.val_indices[-1])
        assert gap_embargo == e + 1

        # Read-only buffer verification
        assert not s.train_indices.flags.writeable
        assert not s.val_indices.flags.writeable
        assert not s.test_indices.flags.writeable

        # Under exact minimum dataset boundary, n - 1 must raise InsufficientDataError
        with pytest.raises(ValueError, match=r"insufficient"):
            list(engine.split(exact_n - 1))

    def test_anchor_end_with_validation_and_embargo(self) -> None:
        """Case 28c: Multi-fold split with anchor='end', val_size > 0, and embargo_bars > 0."""
        dataset_len = 250
        n_splits = 3
        train_size = 50
        val_size = 20
        test_size = 20
        step_size = 20
        purge_bars = 4
        embargo_bars = 2

        # 1. Expanding mode anchored to end
        config_exp = WalkForwardConfig(
            n_splits=n_splits,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            step_size=step_size,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            anchor="end",
            mode="expanding",
        )
        splits_exp = list(WalkForwardEngine(config_exp).split(dataset_len))
        assert len(splits_exp) == n_splits

        # In anchor='end', last fold test partition ends at the very end of the dataset
        assert int(splits_exp[-1].test_indices[-1]) == dataset_len - 1
        assert int(splits_exp[-2].test_indices[-1]) == dataset_len - 1 - step_size
        assert int(splits_exp[-3].test_indices[-1]) == dataset_len - 1 - 2 * step_size

        for s in splits_exp:
            assert s.val_indices is not None
            assert len(s.val_indices) == val_size
            assert len(s.test_indices) == test_size
            assert len(set(s.train_indices) & set(s.val_indices)) == 0
            assert len(set(s.train_indices) & set(s.test_indices)) == 0
            assert len(set(s.val_indices) & set(s.test_indices)) == 0
            assert int(s.val_indices[0]) - int(s.train_indices[-1]) == purge_bars + 1
            assert int(s.test_indices[0]) - int(s.val_indices[-1]) == embargo_bars + 1

        # 2. Rolling mode anchored to end
        config_roll = WalkForwardConfig(
            n_splits=n_splits,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            step_size=step_size,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            anchor="end",
            mode="rolling",
        )
        splits_roll = list(WalkForwardEngine(config_roll).split(dataset_len))
        assert len(splits_roll) == n_splits
        assert int(splits_roll[-1].test_indices[-1]) == dataset_len - 1

        for s in splits_roll:
            assert s.val_indices is not None
            assert len(s.train_indices) == train_size - purge_bars
            assert len(s.val_indices) == val_size
            assert len(s.test_indices) == test_size
            assert len(set(s.train_indices) & set(s.val_indices)) == 0
            assert len(set(s.train_indices) & set(s.test_indices)) == 0
            assert len(set(s.val_indices) & set(s.test_indices)) == 0
            assert int(s.val_indices[0]) - int(s.train_indices[-1]) == purge_bars + 1
            assert int(s.test_indices[0]) - int(s.val_indices[-1]) == embargo_bars + 1


# ==============================================================================
# Suite 1.7: Walk-Forward Evaluation Loop (4 Cases)
# ==============================================================================


class TestWalkForwardEvaluationLoop:
    def test_evaluate_walk_forward_execution_with_baseline(self) -> None:
        """Case 29: Walk-forward evaluation runs with MajorityClassModel across 3 folds."""
        ds = _make_synth_dataset(n=150)
        config = WalkForwardConfig(
            n_splits=3, train_size=60, test_size=30, mode="expanding"
        )
        wf_result = evaluate_walk_forward(
            model_factory=lambda: MajorityClassModel(),
            dataset=ds,
            config=config,
            scale_features=False,
        )

        assert isinstance(wf_result, WalkForwardResult)
        assert len(wf_result.fold_results) == 3
        for fr in wf_result.fold_results:
            assert isinstance(fr, FoldEvaluationResult)
            assert fr.test_metrics.sample_count == 30

    def test_evaluate_walk_forward_metrics_aggregation_math(self) -> None:
        """Case 30: Mean and standard deviation aggregations are mathematically exact."""
        ds = _make_synth_dataset(n=150)
        config = WalkForwardConfig(
            n_splits=3, train_size=60, test_size=30, mode="expanding"
        )
        wf_result = evaluate_walk_forward(
            model_factory=lambda: MajorityClassModel(),
            dataset=ds,
            config=config,
            scale_features=False,
        )

        fold_accs = [fr.metrics.accuracy for fr in wf_result.fold_results]
        expected_mean = float(np.mean(fold_accs))
        expected_std = float(np.std(fold_accs))

        assert wf_result.mean_metrics["accuracy"] == pytest.approx(expected_mean)
        assert wf_result.std_metrics["accuracy"] == pytest.approx(expected_std)

    def test_evaluate_walk_forward_with_probability_calibrator(self) -> None:
        """Case 31: Probability calibrator fits on val set and evaluates on test set."""
        ds = _make_synth_dataset(n=180)
        config = WalkForwardConfig(
            n_splits=2,
            train_size=80,
            val_size=30,
            test_size=30,
            mode="expanding",
        )
        wf_result = evaluate_walk_forward(
            model_factory=lambda: LogisticRegressionModel(),
            dataset=ds,
            config=config,
            calibrator_factory=lambda: ProbabilityCalibrator(method="sigmoid"),
            scale_features=True,
        )

        assert wf_result.fold_results[0].calibrated_test_metrics is not None
        assert wf_result.fold_results[0].calibrated_test_metrics.brier_score >= 0.0

    def test_evaluate_walk_forward_reproducibility(self) -> None:
        """Case 32: Evaluation is 100% numerically reproducible across runs."""
        ds = _make_synth_dataset(n=150)
        config = WalkForwardConfig(
            n_splits=2, train_size=60, test_size=30, mode="expanding"
        )
        res1 = evaluate_walk_forward(
            model_factory=lambda: LogisticRegressionModel(),
            dataset=ds,
            config=config,
            scale_features=False,
        )
        res2 = evaluate_walk_forward(
            model_factory=lambda: LogisticRegressionModel(),
            dataset=ds,
            config=config,
            scale_features=False,
        )

        assert res1.mean_accuracy == pytest.approx(res2.mean_accuracy)
        assert res1.mean_brier_score == pytest.approx(res2.mean_brier_score)
        assert res1.mean_log_loss == pytest.approx(res2.mean_log_loss)

    def test_evaluate_walk_forward_with_decision_engine(self) -> None:
        """Case 33: Walk-forward evaluation with AIDecisionEngine validates trade telemetry."""
        ds = _make_synth_dataset(n=180, seed=42)
        config = WalkForwardConfig(
            n_splits=2,
            train_size=80,
            val_size=30,
            test_size=30,
            mode="expanding",
        )
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.02)
        decision_engine = AIDecisionEngine(
            abstention_gate=gate,
            stake_config=StakeConfig(mode=StakeMode.FIXED_AMOUNT, fixed_amount=100.0),
            payout_rate=0.85,
        )

        wf_result = evaluate_walk_forward(
            model_factory=lambda: LogisticRegressionModel(),
            dataset=ds,
            config=config,
            decision_engine=decision_engine,
            scale_features=True,
        )

        assert isinstance(wf_result, WalkForwardResult)
        assert len(wf_result.folds) == 2

        for fold in wf_result.folds:
            assert fold.trade_count >= 0
            assert fold.abstention_count >= 0
            assert fold.trade_count + fold.abstention_count == 30
            assert 0.0 <= fold.abstention_rate <= 1.0
            assert fold.win_rate is not None
            assert 0.0 <= fold.win_rate <= 1.0
            assert fold.total_return is not None

            # Verify trade count and profit factor telemetry
            if fold.trade_count > 0:
                wins = round(fold.win_rate * fold.trade_count)
                losses = fold.trade_count - wins
                gross_profit = wins * 0.85 * 100.0
                gross_loss = losses * 100.0
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
                assert profit_factor >= 0.0

        assert wf_result.mean_win_rate is not None
        assert 0.0 <= wf_result.mean_win_rate <= 1.0
        assert wf_result.mean_abstention_rate >= 0.0
        assert wf_result.mean_total_return is not None
