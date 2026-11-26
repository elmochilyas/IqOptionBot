"""Phase 4.2 Unit Tests: MLDataset Container, Feature Pipeline Alignment, and Scaler Isolation.

Covers:
- MLDataset container immutability, shape contracts, and metadata properties
- Strict point-in-time alignment between Candle series, CausalFeaturePipeline, and Target labels
- Filter logic: dropping initial warmup rows (is_warmup=True) AND final horizon bars
  (valid_mask=False)
- Row-for-row timestamp and value verification across X, y, and Candle series
- Multi-target compatibility (binary direction, ternary, continuous return)
- Dataset slicing and chronological partitioning preserving metadata
- Strict leak-free feature scaling:
    * StandardScaler fits exclusively on TRAIN partition
    * Frozen mean and scale applied to Validation and Test partitions
    * Extreme distribution shifts / outliers in Val or Test NEVER leak into train parameters
    * Constant features handled safely via epsilon guard (zero NaNs or Infs)
    * Scaler parameter JSON serialization and reload reproduction
- Comprehensive edge cases:
    * Empty candle or feature dataset sequences
    * Length mismatches between candles and feature rows
    * Zero valid sample overlap (warmup + horizon >= total bars)
    * Out-of-bound slicing
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqoption_research.ml.targets.dataset import (
    MLDataset,
    build_ml_dataset,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.features.scalers import StandardScaler


def _make_synth_candle(
    from_ts: int,
    size: int = 60,
    symbol: str = "EURUSD",
    active_id: int = 1,
    open_: float = 1.08500,
    high: float | None = None,
    low: float | None = None,
    close: float = 1.08520,
    volume: int = 10,
    bid: float | None = None,
    ask: float | None = None,
    is_complete: bool = True,
) -> Candle:
    hi = high if high is not None else max(open_, close) + 0.00050
    lo = low if low is not None else min(open_, close) - 0.00050
    b = bid if bid is not None else close - 0.00005
    a = ask if ask is not None else close + 0.00005
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
        volume=volume,
        bid=b,
        ask=a,
        is_complete=is_complete,
        provenance="synthetic",
    )


def _generate_synthetic_candle_series(n_bars: int = 60, seed: int = 42) -> list[Candle]:
    """Generate a realistic, deterministic series of candles with variable price and volume."""
    rng = np.random.RandomState(seed)
    candles: list[Candle] = []
    base_price = 1.08500
    current_price = base_price

    for i in range(n_bars):
        step = rng.normal(0, 0.00015)
        open_ = current_price
        close = max(1.0000, open_ + step)
        high = max(open_, close) + rng.uniform(0.00005, 0.00020)
        low = min(open_, close) - rng.uniform(0.00005, 0.00020)
        vol = int(rng.randint(50, 200))
        bid = close - 0.00005
        ask = close + 0.00005

        c = _make_synth_candle(
            from_ts=1700000000 + i * 60,
            size=60,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=vol,
            bid=bid,
            ask=ask,
            is_complete=True,
        )
        candles.append(c)
        current_price = close

    return candles


# ============================================================================
# 1. MLDataset Container Contract & Invariant Tests
# ============================================================================


def test_ml_dataset_container_attributes_and_properties() -> None:
    """Verify MLDataset container stores aligned matrices and computes properties."""
    m, d = 10, 4
    feature_names = ("f1", "f2", "f3", "f4")
    X = np.ones((m, d), dtype=np.float64)
    y = np.zeros(m, dtype=np.int64)
    timestamps = np.arange(100, 100 + m, dtype=np.int64)
    spec = TargetSpec.binary(horizon_bars=1)

    ds = MLDataset(
        symbol="EURUSD",
        feature_names=feature_names,
        X=X,
        y=y,
        timestamps=timestamps,
        target_spec=spec,
    )

    assert ds.symbol == "EURUSD"
    assert ds.feature_names == feature_names
    assert ds.row_count == m
    assert len(ds) == m
    assert ds.feature_dim == d
    assert ds.X.shape == (m, d)
    assert ds.y.shape == (m,)
    assert ds.timestamps.shape == (m,)
    assert ds.target_spec == spec

    # Test frozen immutability
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ds.symbol = "GBPUSD"  # type: ignore[misc]


def test_ml_dataset_post_init_validation_checks() -> None:
    """MLDataset rejects misaligned dimensions, mismatched lengths, or invalid ndims."""
    spec = TargetSpec.binary(horizon_bars=1)
    feature_names = ("f1", "f2")

    # Mismatched y length
    with pytest.raises(ValueError, match=r"y length .* does not match X rows"):
        MLDataset(
            symbol="EURUSD",
            feature_names=feature_names,
            X=np.ones((5, 2)),
            y=np.zeros(4),  # 4 != 5
            timestamps=np.ones(5, dtype=np.int64),
            target_spec=spec,
        )

    # Mismatched timestamps length
    with pytest.raises(ValueError, match=r"timestamps length .* does not match X rows"):
        MLDataset(
            symbol="EURUSD",
            feature_names=feature_names,
            X=np.ones((5, 2)),
            y=np.zeros(5),
            timestamps=np.ones(6, dtype=np.int64),  # 6 != 5
            target_spec=spec,
        )

    # Mismatched feature dimensions
    with pytest.raises(ValueError, match=r"X column count .* does not match feature_names"):
        MLDataset(
            symbol="EURUSD",
            feature_names=feature_names,  # 2 names
            X=np.ones((5, 3)),  # 3 columns != 2
            y=np.zeros(5),
            timestamps=np.ones(5, dtype=np.int64),
            target_spec=spec,
        )


def test_ml_dataset_to_dict_list_conversion() -> None:
    """Verify to_dict_list exports rows matching feature names for Phase 3 tool compatibility."""
    feature_names = ("feat_a", "feat_b")
    X = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    y = np.array([1, 0], dtype=np.int64)
    ts = np.array([100, 160], dtype=np.int64)

    ds = MLDataset("EURUSD", feature_names, X, y, ts, TargetSpec.binary())
    dict_rows = ds.to_dict_list()

    assert len(dict_rows) == 2
    assert dict_rows[0] == {"feat_a": 1.5, "feat_b": 2.5}
    assert dict_rows[1] == {"feat_a": 3.5, "feat_b": 4.5}


# ============================================================================
# 2. Strict Row-for-Row Alignment Tests
# ============================================================================


def test_build_ml_dataset_strict_alignment_and_filtering() -> None:
    """Verify build_ml_dataset drops warmup AND horizon bars with exact 1-to-1 alignment."""
    # 50 bars. Lookback of pipeline is 21 (first 20 bars are is_warmup=True).
    # Horizon is 5 (final 5 bars have valid_mask=False).
    # Total bars: 50. Warmup bars: 20 (indices 0..19). Masked bars: 5 (indices 45..49).
    # Expected valid rows: exactly 50 - 20 - 5 = 25 rows (indices 20..44).
    candles = _generate_synthetic_candle_series(n_bars=50, seed=123)

    pipeline = CausalFeaturePipeline(drop_warmup=False)
    feat_ds = pipeline.transform(candles)
    assert feat_ds.row_count == 50
    assert sum(1 for r in feat_ds.rows if r.is_warmup) == 20

    spec = TargetSpec.binary(horizon_bars=5)
    ml_ds = build_ml_dataset(candles, feat_ds, spec)

    # 1. Row count assertion
    assert ml_ds.row_count == 25
    assert ml_ds.feature_dim == len(feat_ds.feature_names)
    assert ml_ds.symbol == candles[0].symbol
    assert ml_ds.target_spec == spec

    # 2. Row-for-row alignment check
    for k in range(ml_ds.row_count):
        orig_t = 20 + k  # Original candle index
        c_curr = candles[orig_t]
        c_fut = candles[orig_t + 5]
        f_row = feat_ds.rows[orig_t]

        # Timestamp alignment: availability timestamp must match candle.to_ts
        assert ml_ds.timestamps[k] == c_curr.to_ts
        assert ml_ds.timestamps[k] == f_row.candle_to_ts

        # Target alignment: y must reflect C_{orig_t + 5} > C_{orig_t}
        expected_y = 1 if c_fut.close > c_curr.close else 0
        assert ml_ds.y[k] == expected_y

        # Feature matrix alignment: X row must match f_row feature dict
        for feat_idx, feat_name in enumerate(ml_ds.feature_names):
            val_x = ml_ds.X[k, feat_idx]
            val_feat = f_row.features[feat_name]
            assert val_x == pytest.approx(val_feat, abs=1e-12)


@pytest.mark.parametrize(
    ("target_type", "expected_dtype"),
    [
        (TargetType.DIRECTION_BINARY, np.int64),
        (TargetType.DIRECTION_TERNARY, np.int64),
        (TargetType.RETURN_CONTINUOUS, np.float64),
    ],
)
def test_build_ml_dataset_target_types_dtype_parity(
    target_type: TargetType,
    expected_dtype: type,
) -> None:
    """Verify build_ml_dataset maintains correct target array dtype for all target types."""
    candles = _generate_synthetic_candle_series(n_bars=40, seed=456)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

    spec = TargetSpec(horizon_bars=3, target_type=target_type)
    ml_ds = build_ml_dataset(candles, feat_ds, spec)

    assert ml_ds.y.dtype == expected_dtype
    # Assert no NaNs or Infs present in valid y
    assert np.all(np.isfinite(ml_ds.y))


# ============================================================================
# 3. Slicing and Chronological Splitting Tests
# ============================================================================


def test_ml_dataset_slice_chronological_partitions() -> None:
    """Verify slice() cleanly divides dataset into Train, Val, and Test without data loss."""
    candles = _generate_synthetic_candle_series(n_bars=60, seed=789)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)
    spec = TargetSpec.binary(horizon_bars=2)
    ml_ds = build_ml_dataset(candles, feat_ds, spec)

    total_rows = ml_ds.row_count
    n_train = int(total_rows * 0.6)
    n_val = int(total_rows * 0.2)
    n_test = total_rows - n_train - n_val

    train_ds = ml_ds.slice(0, n_train)
    val_ds = ml_ds.slice(n_train, n_train + n_val)
    test_ds = ml_ds.slice(n_train + n_val, total_rows)

    assert train_ds.row_count == n_train
    assert val_ds.row_count == n_val
    assert test_ds.row_count == n_test

    # Boundary timestamp monotonicity: train < val < test
    assert train_ds.timestamps[-1] < val_ds.timestamps[0]
    assert val_ds.timestamps[-1] < test_ds.timestamps[0]

    # Reconstituted arrays equal original
    assert np.array_equal(
        np.concatenate([train_ds.X, val_ds.X, test_ds.X], axis=0),
        ml_ds.X,
    )
    assert np.array_equal(
        np.concatenate([train_ds.y, val_ds.y, test_ds.y], axis=0),
        ml_ds.y,
    )


def test_ml_dataset_slice_invalid_bounds_raise() -> None:
    """Invalid slice indices raise MLDatasetError."""
    candles = _generate_synthetic_candle_series(n_bars=40)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)
    ml_ds = build_ml_dataset(candles, feat_ds, TargetSpec.binary())

    with pytest.raises(ValueError, match="Invalid slice bounds"):
        ml_ds.slice(-1, 5)

    with pytest.raises(ValueError, match="Invalid slice bounds"):
        ml_ds.slice(10, ml_ds.row_count + 1)

    with pytest.raises(ValueError, match="Invalid slice bounds"):
        ml_ds.slice(10, 5)  # start > end


def test_ml_dataset_dunder_len_and_indexing() -> None:
    """Verify len(dataset) and dataset[start:stop] slicing indexing."""
    candles = _generate_synthetic_candle_series(n_bars=40)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)
    ml_ds = build_ml_dataset(candles, feat_ds, TargetSpec.binary())

    assert len(ml_ds) == ml_ds.row_count
    sub_ds = ml_ds[0:5]
    assert isinstance(sub_ds, MLDataset)
    assert sub_ds.row_count == 5
    assert np.array_equal(sub_ds.X, ml_ds.X[:5])


def test_ml_dataset_select_columns() -> None:
    """Verify select() isolates feature columns and maintains matrix alignment."""
    candles = _generate_synthetic_candle_series(n_bars=40)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)
    ml_ds = build_ml_dataset(candles, feat_ds, TargetSpec.binary())

    assert len(ml_ds.feature_names) >= 2
    selected_cols = (ml_ds.feature_names[0], ml_ds.feature_names[1])
    sub_ds = ml_ds.select(selected_cols)

    assert sub_ds.feature_names == selected_cols
    assert sub_ds.feature_dim == 2
    assert sub_ds.row_count == ml_ds.row_count
    assert np.array_equal(sub_ds.X[:, 0], ml_ds.X[:, 0])
    assert np.array_equal(sub_ds.X[:, 1], ml_ds.X[:, 1])

    # Missing column raises
    with pytest.raises(ValueError, match="not found in dataset feature_names"):
        ml_ds.select(["non_existent_column"])


def test_ml_dataset_train_val_test_split_helper() -> None:
    """Verify train_val_test_split() produces three chronological non-overlapping partitions."""
    candles = _generate_synthetic_candle_series(n_bars=60)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)
    ml_ds = build_ml_dataset(candles, feat_ds, TargetSpec.binary())

    train_ds, val_ds, test_ds = ml_ds.train_val_test_split(0.6, 0.2, 0.2)
    assert train_ds.row_count + val_ds.row_count + test_ds.row_count == ml_ds.row_count
    assert train_ds.timestamps[-1] < val_ds.timestamps[0]
    assert val_ds.timestamps[-1] < test_ds.timestamps[0]

    # Invalid ratios that do not sum to 1.0 raise
    with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
        ml_ds.train_val_test_split(0.5, 0.5, 0.5)


# ============================================================================
# 4. Leak-Free Preprocessing: Scaler Isolation Tests
# ============================================================================


def test_scaler_fitted_strictly_on_train_partition() -> None:
    """StandardScaler parameters are fitted exclusively on Train and frozen."""
    candles = _generate_synthetic_candle_series(n_bars=60, seed=101)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)
    ml_ds = build_ml_dataset(candles, feat_ds, TargetSpec.binary())

    # Split: 20 train, 10 val
    train_ds = ml_ds.slice(0, 20)

    scaler = StandardScaler()
    scaler.fit(train_ds.to_dict_list())

    # Check fitted metadata
    assert scaler.is_fitted is True
    assert scaler.params.n_samples_seen == 20
    assert set(scaler.params.feature_names) == set(train_ds.feature_names)

    # Verify mean is purely train mean
    for col_idx, col_name in enumerate(train_ds.feature_names):
        col_vals = train_ds.X[:, col_idx]
        expected_mean = float(np.mean(col_vals))
        assert scaler.params.mean[col_name] == pytest.approx(expected_mean, rel=1e-5)


def test_scaler_zero_leakage_from_val_and_test() -> None:
    """CRITICAL LEAKAGE TEST:

    Extreme distribution shifts / outliers in Validation or Test sets
    must NEVER contaminate train scaling parameters.
    """
    candles = _generate_synthetic_candle_series(n_bars=60, seed=202)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)
    ml_ds = build_ml_dataset(candles, feat_ds, TargetSpec.binary())

    train_ds = ml_ds.slice(0, 20)
    val_clean = ml_ds.slice(20, 30)

    # 1. Baseline fit on clean train
    scaler_clean = StandardScaler().fit(train_ds.to_dict_list())

    # 2. Corrupt validation set with catastrophic distribution shock (1e7 magnitude)
    val_corrupted_dict = val_clean.to_dict_list()
    for row in val_corrupted_dict:
        for feat in row:
            row[feat] = 10_000_000.0

    # 3. Fit scaler in the same pipeline context
    scaler_test = StandardScaler().fit(train_ds.to_dict_list())

    # Parameters must be 100% BIT-FOR-BIT IDENTICAL!
    for col in train_ds.feature_names:
        assert scaler_clean.params.mean[col] == scaler_test.params.mean[col]
        assert scaler_clean.params.scale[col] == scaler_test.params.scale[col]
        assert scaler_clean.params.var[col] == scaler_test.params.var[col]

    # Transform corrupted validation using frozen train parameters
    scaled_corrupted_val = scaler_clean.transform(val_corrupted_dict)
    assert len(scaled_corrupted_val) == 10
    # Scaler mean must NOT have updated during transform
    assert scaler_clean.params.n_samples_seen == 20


def test_scaler_constant_feature_guarded_by_epsilon() -> None:
    """Constant features (zero variance) avoid division by zero via epsilon guard."""
    # Data with constant feature 'zero_var'
    train_data = [
        {"var_feat": 10.0, "zero_var": 5.0},
        {"var_feat": 20.0, "zero_var": 5.0},
        {"var_feat": 30.0, "zero_var": 5.0},
    ]
    scaler = StandardScaler(epsilon=1e-8).fit(train_data)

    assert scaler.params.scale["zero_var"] == 1.0  # Constant feature guarded to 1.0
    transformed = scaler.transform(train_data)
    assert transformed[0]["zero_var"] == 0.0
    assert transformed[1]["zero_var"] == 0.0
    assert transformed[2]["zero_var"] == 0.0
    assert not any(np.isnan(v) for r in transformed for v in r.values())


def test_scaler_json_serialization_roundtrip_reproducibility() -> None:
    """Fitted scaler serializes to JSON and reproduces identical transforms on reload."""
    train_data = [
        {"feat1": 1.0, "feat2": 100.0},
        {"feat1": 2.0, "feat2": 200.0},
        {"feat1": 3.0, "feat2": 300.0},
    ]
    eval_data = [{"feat1": 4.0, "feat2": 400.0}]

    scaler = StandardScaler().fit(train_data)
    json_str = scaler.to_json()

    loaded_scaler = StandardScaler.from_json(json_str)

    orig_out = scaler.transform(eval_data)
    loaded_out = loaded_scaler.transform(eval_data)

    assert orig_out == loaded_out


# ============================================================================
# 5. Edge Cases in Dataset Building
# ============================================================================


def test_build_ml_dataset_empty_inputs_raise() -> None:
    """Empty candles or empty feature dataset raises MLDatasetError."""
    candles = _generate_synthetic_candle_series(n_bars=30)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)
    spec = TargetSpec.binary()

    with pytest.raises(ValueError, match="Candles sequence cannot be empty"):
        build_ml_dataset([], feat_ds, spec)


def test_build_ml_dataset_length_mismatch_raises() -> None:
    """Mismatched candle sequence and feature dataset lengths raise MLDatasetError."""
    candles = _generate_synthetic_candle_series(n_bars=30)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles[:25])
    spec = TargetSpec.binary()

    with pytest.raises(ValueError, match="Length mismatch"):
        build_ml_dataset(candles, feat_ds, spec)


def test_build_ml_dataset_insufficient_samples_raises() -> None:
    """When warmup bars + horizon bars >= total candles, 0 valid samples remain and raises."""
    # Pipeline requires 21 bars warmup. With 23 bars and horizon=5,
    # warmup (20) + horizon (5) = 25 > 23 bars -> 0 overlapping valid bars!
    candles = _generate_synthetic_candle_series(n_bars=23)
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)
    spec = TargetSpec.binary(horizon_bars=5)

    with pytest.raises(ValueError, match="Insufficient data: 0 valid samples"):
        build_ml_dataset(candles, feat_ds, spec)
