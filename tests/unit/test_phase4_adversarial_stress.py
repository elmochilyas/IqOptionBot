"""Adversarial stress testing harness for Phase 4 targets, datasets, and scaling.

Empirical verification of:
1. Floating-point precision stability ($10^{-8}$ fractional price movements and
   sub-epsilon IEEE-754 noise rejection).
2. Zero-variance and near-zero-variance feature scaling with epsilon guards (NaN-free).
3. Slicing, indexing, chronological ordering, and metadata integrity in MLDataset.
4. Embargoed and purged train/val/test splitting invariants.
5. Immutability guards on sliced and scaled dataset partitions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from iqoption_research.ml.targets.dataset import (
    DatasetAlignmentError,
    MLDataset,
    build_ml_dataset,
    scale_dataset_partitions,
)
from iqoption_research.ml.targets.generator import (
    generate_targets,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.features.pipeline import (
    DEFAULT_FEATURE_SPECS,
    CausalFeaturePipeline,
)
from iqoption_research.research.features.scalers import StandardScaler


def _create_candle(
    from_ts: int,
    close: float,
    open_: float = 1.08500,
    size: int = 60,
    symbol: str = "EURUSD",
    active_id: int = 1,
) -> Candle:
    """Helper to create minimal valid synthetic candles."""
    hi = max(open_, close) + 0.00010
    lo = min(open_, close) - 0.00010
    return Candle(
        active_id=active_id,
        symbol=symbol,
        size=CandleSize(size),
        from_ts=from_ts,
        to_ts=from_ts + size,
        at_ts=(from_ts + size) * 1_000_000_000,
        open=open_,
        high=hi,
        low=lo,
        close=close,
        volume=10,
        bid=close - 0.00005,
        ask=close + 0.00005,
        is_complete=True,
        provenance="synthetic",
    )


# ============================================================================
# 1. Floating-Point Precision Stress Tests ($10^{-8}$ and IEEE-754 noise)
# ============================================================================


def test_adversarial_floating_point_tiny_fractional_diff_1e8() -> None:
    """Stress-test prices differing by tiny fractional amounts ($10^{-8}$).

    Verifies:
    - Positive diff of 1e-8 registers as UP (1).
    - Negative diff of -1e-8 registers as FLAT/DOWN (0 for binary, -1 for ternary).
    - IEEE-754 machine epsilon jitter (~1e-16) is rounded to 0.0 and maps to flat (0).
    """
    base_price = 1.08500000
    spec_bin = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        flat_threshold_pips=0.0,
    )
    spec_ter = TargetSpec(
        target_type=TargetType.DIRECTION_TERNARY,
        horizon_bars=1,
        flat_threshold_pips=0.0,
    )

    # 1. Up by exactly 1e-8
    c1 = _create_candle(from_ts=1000, close=base_price)
    c2_up = _create_candle(from_ts=1060, close=base_price + 1e-8)
    candles_up = [c1, c2_up]

    y_bin_up, mask_up = generate_targets(candles_up, spec_bin)
    assert mask_up[0] is np.True_
    assert y_bin_up[0] == 1, "1e-8 upward difference must produce binary 1 (UP)"

    y_ter_up, _ = generate_targets(candles_up, spec_ter)
    assert y_ter_up[0] == 1, "1e-8 upward difference must produce ternary +1 (UP)"

    # 2. Down by exactly 1e-8
    c2_down = _create_candle(from_ts=1060, close=base_price - 1e-8)
    candles_down = [c1, c2_down]

    y_bin_down, mask_down = generate_targets(candles_down, spec_bin)
    assert mask_down[0] is np.True_
    assert y_bin_down[0] == 0, "-1e-8 downward difference must produce binary 0"

    y_ter_down, _ = generate_targets(candles_down, spec_ter)
    assert y_ter_down[0] == -1, "-1e-8 downward difference must produce ternary -1 (DOWN)"

    # 3. Micro-jitter: 1e-15 to 1e-16 (IEEE-754 floating point subtraction noise)
    # 1.0850000000000002 - 1.085 = 2.220446049250313e-16
    c2_noise_pos = _create_candle(from_ts=1060, close=1.0850000000000002)
    c1_exact = _create_candle(from_ts=1000, close=1.0850000000000000)
    candles_noise = [c1_exact, c2_noise_pos]

    y_bin_noise, _ = generate_targets(candles_noise, spec_bin)
    # Without 9-decimal rounding, 2.22e-16 > 0 would spurious-trigger 1 (UP).
    # With 9-decimal rounding, round(2.22e-16, 9) == 0.0, which maps to flat (0).
    assert y_bin_noise[0] == 0, "Machine epsilon noise must be rounded to zero and map to 0"

    y_ter_noise, _ = generate_targets(candles_noise, spec_ter)
    assert y_ter_noise[0] == 0, "Machine epsilon noise must map to ternary 0 (FLAT)"


def test_adversarial_floating_point_dense_boundary_grid() -> None:
    """Evaluate an adversarial dense grid of price perturbations around zero and thresholds.

    Generates 200 pairs of candles spanning increments from -1e-7 to +1e-7 in steps of 1e-9.
    Verifies that binary direction is monotonic across zero and strictly causal.
    """
    base_ts = 1_700_000_000
    base_price = 1.250000000
    spec = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        flat_threshold_pips=0.0,
    )

    steps = np.linspace(-1e-7, 1e-7, 201)
    candles: list[Candle] = []

    # Construct paired sequence of (C_t, C_{t+1})
    for i, delta in enumerate(steps):
        t_cur = base_ts + i * 120
        t_fut = t_cur + 60
        c_cur = _create_candle(from_ts=t_cur, close=base_price)
        c_fut = _create_candle(from_ts=t_fut, close=base_price + delta)
        candles.extend([c_cur, c_fut])

    # Generator will process the sequence
    targets, _ = generate_targets(candles, spec)
    assert len(targets) == len(candles)

    # For each pair at index 2*k:
    for k, delta in enumerate(steps):
        idx = 2 * k
        diff = round(candles[idx + 1].close - candles[idx].close, 9)
        expected = 1 if diff > 0.0 else 0
        assert targets[idx] == expected, f"Failed at delta={delta}, diff={diff}"


def test_adversarial_threshold_deadband_sub_pip_stability() -> None:
    """Verify deadband behavior around threshold boundary under fractional pips."""
    spec_pip = TargetSpec.binary(horizon_bars=1, flat_threshold_pips=0.5, pip_size=1e-4)
    # threshold_price = 0.5 * 1e-4 = 0.00005 (5e-5)
    assert math.isclose(spec_pip.threshold_price, 5e-5)

    base = 1.100000000
    # Exactly on threshold: 5e-5 -> diff > threshold is False -> 0
    c1 = _create_candle(1000, base)
    c2_exact = _create_candle(1060, base + 0.00005)
    y, _ = generate_targets([c1, c2_exact], spec_pip)
    assert y[0] == 0, "Exact threshold difference must not trigger strictly greater test"

    # Epsilon above threshold: 5e-5 + 1e-8 -> diff > threshold is True -> 1
    c2_above = _create_candle(1060, base + 0.00005001)
    y_above, _ = generate_targets([c1, c2_above], spec_pip)
    assert y_above[0] == 1, "Exceeding threshold by 1e-8 must trigger UP (1)"


# ============================================================================
# 2. Zero-Variance & Constant Feature Scaling Stress Tests
# ============================================================================


def test_adversarial_zero_variance_constant_features_epsilon_guard() -> None:
    """Stress-test StandardScaler on datasets containing completely constant features.

    Verifies:
    - Constant feature variance is 0.0 (or sub-epsilon floating representation < 1e-20).
    - Scale parameter falls back safely to 1.0 via epsilon guard (epsilon=1e-8).
    - Division by zero is prevented; transformed values are strictly 0.0 without NaNs or Infs.
    """
    n_rows = 150
    # 5 features:
    # 0: all zeros
    # 1: all constant 42.0
    # 2: all constant -999.99
    # 3: tiny sub-epsilon perturbation (std ~ 1e-10 < 1e-8)
    # 4: normal variance feature
    col0 = np.zeros(n_rows, dtype=np.float64)
    col1 = np.full(n_rows, 42.0, dtype=np.float64)
    col2 = np.full(n_rows, -999.99, dtype=np.float64)
    col3 = 10.0 + 1e-10 * np.sin(np.linspace(0, 10, n_rows))  # std < 1e-9 < epsilon
    col4 = np.linspace(1.0, 50.0, n_rows, dtype=np.float64)

    X = np.column_stack([col0, col1, col2, col3, col4])
    y = np.ones(n_rows, dtype=np.int64)
    timestamps = np.arange(1000, 1000 + n_rows * 60, 60, dtype=np.int64)
    feature_names = ("feat_zero", "feat_42", "feat_neg", "feat_subeps", "feat_norm")
    spec = TargetSpec.binary(horizon_bars=1)

    ds = MLDataset(
        symbol="EURUSD",
        feature_names=feature_names,
        X=X,
        y=y,
        timestamps=timestamps,
        target_spec=spec,
    )

    scaler = StandardScaler(epsilon=1e-8)
    fitted_scaler = ds.fit_scaler(scaler)

    params = fitted_scaler.params
    assert params.var["feat_zero"] == 0.0
    assert params.var["feat_42"] == 0.0
    assert math.isclose(params.var["feat_neg"], 0.0, abs_tol=1e-15)
    assert params.scale["feat_zero"] == 1.0
    assert params.scale["feat_42"] == 1.0
    assert params.scale["feat_neg"] == 1.0

    # feat_subeps std is < 1e-8, so it must also be guarded to 1.0
    assert params.scale["feat_subeps"] == 1.0

    # feat_norm std is normal
    assert params.scale["feat_norm"] > 1.0

    # Apply transform
    scaled_ds = ds.transform_with(fitted_scaler)

    # Assert no NaNs or Infs
    assert np.all(np.isfinite(scaled_ds.X)), "Transformed feature matrix must be finite"
    assert not np.any(np.isnan(scaled_ds.X)), "Transformed feature matrix must not contain NaNs"

    # Constant features must transform to exact 0.0
    assert np.allclose(scaled_ds.X[:, 0], 0.0, atol=1e-12)
    assert np.allclose(scaled_ds.X[:, 1], 0.0, atol=1e-12)
    assert np.allclose(scaled_ds.X[:, 2], 0.0, atol=1e-12)
    # Sub-epsilon feature must not explode into huge values
    assert np.all(np.abs(scaled_ds.X[:, 3]) < 1e-6)


def test_adversarial_single_sample_dataset_scaling() -> None:
    """Stress-test StandardScaler on an edge-case single sample dataset (row_count=1).

    In sample variance with ddof=1, N=1 has undefined variance (0/0).
    The implementation must fallback to var=0.0 and scale=1.0 without emitting NaNs.
    """
    X = np.array([[123.45, 678.90]], dtype=np.float64)
    y = np.array([1], dtype=np.int64)
    ts = np.array([1000], dtype=np.int64)
    names = ("f1", "f2")

    ds = MLDataset(
        symbol="EURUSD",
        feature_names=names,
        X=X,
        y=y,
        timestamps=ts,
        target_spec=TargetSpec.binary(),
    )

    scaler = StandardScaler()
    scaler.fit(ds)

    assert scaler.params.n_samples_seen == 1
    assert scaler.params.var["f1"] == 0.0
    assert scaler.params.var["f2"] == 0.0
    assert scaler.params.scale["f1"] == 1.0
    assert scaler.params.scale["f2"] == 1.0

    scaled = ds.transform_with(scaler)
    assert scaled.X[0, 0] == 0.0
    assert scaled.X[0, 1] == 0.0
    assert not np.isnan(scaled.X).any()


# ============================================================================
# 3. Slicing, Splitting, and Chronological Ordering Stress Tests
# ============================================================================


def test_adversarial_dataset_slicing_and_chronology_preservation() -> None:
    """Stress-test slicing operations to ensure timestamps and ordering are immutable."""
    n = 200
    timestamps = np.arange(1_000_000, 1_000_000 + n * 60, 60, dtype=np.int64)
    X = np.tile(np.arange(n, dtype=np.float64).reshape(-1, 1), (1, 3))
    y = (np.arange(n) % 2).astype(np.int64)
    names = ("a", "b", "c")

    ds = MLDataset(
        symbol="EURUSD",
        feature_names=names,
        X=X,
        y=y,
        timestamps=timestamps,
        target_spec=TargetSpec.binary(),
    )

    # 1. Slice [50:150]
    sub1 = ds[50:150]
    assert len(sub1) == 100
    assert sub1.timestamps[0] == timestamps[50]
    assert sub1.timestamps[-1] == timestamps[149]
    assert np.all(
        sub1.timestamps[1:] > sub1.timestamps[:-1]
    ), "Timestamps must be strictly increasing"
    assert np.array_equal(sub1.X, X[50:150])
    assert np.array_equal(sub1.y, y[50:150])

    # 2. Slice method .slice(20, 80)
    sub2 = ds.slice(20, 80)
    assert len(sub2) == 60
    assert sub2.row_count == 60

    # 3. Non-contiguous boolean masking (every second row)
    mask = (np.arange(n) % 2) == 0
    sub3 = ds[mask]
    assert len(sub3) == 100
    assert np.all(sub3.timestamps[1:] > sub3.timestamps[:-1])

    # 4. Immutability: verifying writeable flags
    assert not sub1.X.flags.writeable
    assert not sub1.y.flags.writeable
    assert not sub1.timestamps.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        sub1.X[0, 0] = 999.0

    with pytest.raises(ValueError, match="read-only"):
        sub1.y[0] = 999


def test_adversarial_train_val_test_split_embargo_and_purging() -> None:
    """Stress-test chronological train/val/test splitting with embargo and horizon purging.

    Verifies:
    - Split boundaries are strictly chronological: max(Train) < min(Val) < min(Test).
    - When label_horizon_bars > 0 and embargo_bars > 0, the purge gap is strictly observed.
    """
    n = 100
    timestamps = np.arange(1000, 1000 + n * 60, 60, dtype=np.int64)
    X = np.ones((n, 2), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)

    ds = MLDataset(
        symbol="EURUSD",
        feature_names=("f1", "f2"),
        X=X,
        y=y,
        timestamps=timestamps,
        target_spec=TargetSpec.binary(horizon_bars=3),
    )

    # Standard split without embargo
    train, val, test = ds.train_val_test_split(0.60, 0.20, 0.20)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    assert train.timestamps[-1] < val.timestamps[0]
    assert val.timestamps[-1] < test.timestamps[0]

    # Split with horizon purging and embargo
    # label_horizon=3, embargo=2 -> train loses 3 bars at tail, val loses 3 bars at tail,
    # and embargo removes 2 bars at start of val and test.
    train_p, val_p, test_p = ds.train_val_test_split(
        0.60,
        0.20,
        0.20,
        label_horizon_bars=3,
        embargo_bars=2,
    )

    assert len(train_p) < 60
    assert train_p.timestamps[-1] < val_p.timestamps[0]
    assert val_p.timestamps[-1] < test_p.timestamps[0]

    # Check temporal gap between train end and val start
    train_end_ts = train_p.timestamps[-1]
    val_start_ts = val_p.timestamps[0]
    # Gap in bars must be at least label_horizon_bars + embargo_bars
    gap_bars = (val_start_ts - train_end_ts) // 60
    assert gap_bars >= (3 + 2), f"Expected gap >= 5 bars, got {gap_bars}"


def test_adversarial_scale_dataset_partitions_isolation() -> None:
    """Stress-test scale_dataset_partitions to ensure zero data leakage across folds."""
    n = 150
    timestamps = np.arange(1000, 1000 + n * 60, 60, dtype=np.int64)
    # Train features around 10.0, Val around 500.0, Test around 1,000,000.0
    x_train = np.full((100, 2), 10.0, dtype=np.float64)
    x_train += np.random.RandomState(42).normal(0, 1.0, (100, 2))

    x_val = np.full((25, 2), 500.0, dtype=np.float64)
    x_test = np.full((25, 2), 1_000_000.0, dtype=np.float64)

    X_all = np.vstack([x_train, x_val, x_test])
    y_all = np.zeros(n, dtype=np.int64)

    ds = MLDataset(
        symbol="EURUSD",
        feature_names=("feat1", "feat2"),
        X=X_all,
        y=y_all,
        timestamps=timestamps,
        target_spec=TargetSpec.binary(),
    )

    train_ds = ds[:100]
    val_ds = ds[100:125]
    test_ds = ds[125:150]

    _train_s, val_s, test_s, sc = scale_dataset_partitions(train_ds, val_ds, test_ds)
    assert val_s is not None
    assert test_s is not None

    # Scaler mean must be strictly ~10.0 (from train), unaffected by 500 or 1,000,000
    for name in ds.feature_names:
        assert math.isclose(sc.params.mean[name], 10.0, abs_tol=0.5)

    # Val transformed must have large positive z-scores: (500 - 10) / 1.0 ~ 490
    assert np.all(val_s.X > 400.0)
    # Test transformed must have huge positive z-scores: ~ 999,990
    assert np.all(test_s.X > 900_000.0)


# ============================================================================
# 4. Alignment & Anti-Leakage Invariance Verification
# ============================================================================


def test_adversarial_build_ml_dataset_rejects_desynchronized_timestamps() -> None:
    """Stress-test build_ml_dataset when FeatureDataset timestamps disagree with candles."""
    candles = [_create_candle(1000 + i * 60, 1.08500 + i * 0.00010) for i in range(10)]
    pipe = CausalFeaturePipeline(
        custom_specs=list(DEFAULT_FEATURE_SPECS[:1]),
        include_volume=False,
    )
    feat_ds = pipe.transform(candles)

    # Deliberately desynchronize candle timestamp at index 5
    corrupted_candles = list(candles)
    c5 = candles[5]
    corrupted_candles[5] = Candle(
        active_id=c5.active_id,
        symbol=c5.symbol,
        size=c5.size,
        from_ts=c5.from_ts,
        to_ts=c5.to_ts + 9999,  # Desynchronized!
        at_ts=c5.at_ts,
        open=c5.open,
        high=c5.high,
        low=c5.low,
        close=c5.close,
        volume=c5.volume,
        bid=c5.bid,
        ask=c5.ask,
        is_complete=c5.is_complete,
        provenance=c5.provenance,
    )

    with pytest.raises(DatasetAlignmentError, match="Timestamp mismatch"):
        build_ml_dataset(corrupted_candles, feat_ds, TargetSpec.binary())
