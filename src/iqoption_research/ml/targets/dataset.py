"""Machine Learning Dataset Container, Pipeline Alignment, and Leak-Free Scaling.

Provides:
1. `MLDataset`: Immutable, frozen dataclass container pairing causal feature matrices X,
   forward prediction targets y, availability timestamps, and target metadata.
2. `build_ml_dataset`: Causal builder aligning historical candles, Phase 3 FeatureDataset,
   and prediction targets. Drops warmup bars and invalid horizon tail bars.
3. Leak-free feature scaling helpers: `fit_scaler_on_dataset`, `transform_dataset`, and
   `scale_dataset_partitions`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.models.market import Candle
from iqoption_research.research.datasets.splitter import ChronologicalSplitter, DatasetSplits
from iqoption_research.research.features.pipeline import FeatureDataset
from iqoption_research.research.features.scalers import (
    PreprocessingError,
    StandardScaler,
)


class MLDatasetError(ValueError):
    """Base exception for ML dataset container, alignment, and operations errors."""


class DatasetAlignmentError(MLDatasetError):
    """Raised when candles, features, and targets fail causal alignment or verification."""


@dataclass(frozen=True, slots=True)
class MLDataset:
    """Immutable point-in-time machine learning dataset container.

    Attributes:
        symbol: Market instrument symbol (e.g. "EURUSD").
        feature_names: Ordered tuple of feature column names corresponding to columns of X.
        X: 2D feature matrix of shape (M, D), dtype float64. Read-only array.
        y: 1D target vector of shape (M,), dtype int64 (for classification) or float64. Read-only.
        timestamps: 1D array of shape (M,), epoch seconds of feature availability. Read-only.
        target_spec: Specification of the forward target formulation.
        row_count: Number of aligned samples M.
        feature_dim: Number of feature columns D.
    """

    symbol: str
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    target_spec: TargetSpec
    row_count: int = -1
    feature_dim: int = -1

    def __post_init__(self) -> None:
        """Validate shapes, types, finite constraints, and enforce array immutability."""
        if not self.symbol or not isinstance(self.symbol, str):
            raise DatasetAlignmentError("MLDataset symbol must be a non-empty string")

        if not isinstance(self.feature_names, tuple):
            object.__setattr__(self, "feature_names", tuple(self.feature_names))
        if len(self.feature_names) == 0:
            raise DatasetAlignmentError("MLDataset feature_names cannot be empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise DatasetAlignmentError("MLDataset feature_names must not contain duplicates")

        if not isinstance(self.X, np.ndarray) or self.X.ndim != 2:
            shape_str = getattr(self.X, "shape", None)
            raise DatasetAlignmentError(
                f"MLDataset X must be a 2D numpy ndarray, got shape {shape_str}"
            )
        if self.X.dtype != np.float64:
            object.__setattr__(self, "X", np.ascontiguousarray(self.X, dtype=np.float64))

        if not isinstance(self.y, np.ndarray) or self.y.ndim != 1:
            shape_str = getattr(self.y, "shape", None)
            raise DatasetAlignmentError(
                f"MLDataset y must be a 1D numpy ndarray, got shape {shape_str}"
            )

        if not isinstance(self.timestamps, np.ndarray) or self.timestamps.ndim != 1:
            shape_str = getattr(self.timestamps, "shape", None)
            raise DatasetAlignmentError(
                f"MLDataset timestamps must be a 1D numpy ndarray, got shape {shape_str}"
            )
        if self.timestamps.dtype != np.int64:
            object.__setattr__(
                self, "timestamps", np.ascontiguousarray(self.timestamps, dtype=np.int64)
            )

        m = int(self.X.shape[0])
        d = int(self.X.shape[1])

        if self.y.shape[0] != m:
            raise DatasetAlignmentError(
                f"Row count mismatch: y length ({self.y.shape[0]}) does not match X rows ({m})"
            )
        if self.timestamps.shape[0] != m:
            raise DatasetAlignmentError(
                f"Row count mismatch: timestamps length ({self.timestamps.shape[0]}) "
                f"does not match X rows ({m})"
            )
        if d != len(self.feature_names):
            raise DatasetAlignmentError(
                f"Dimension mismatch: X column count ({d}) "
                f"does not match feature_names ({len(self.feature_names)})"
            )

        # Finite number validation
        if not np.all(np.isfinite(self.X)):
            raise DatasetAlignmentError("MLDataset X contains NaN or infinite values")
        if not np.all(np.isfinite(self.y)):
            raise DatasetAlignmentError("MLDataset y contains NaN or infinite values")

        # Chronological non-decreasing timestamp verification
        if m > 1 and np.any(self.timestamps[1:] < self.timestamps[:-1]):
            raise DatasetAlignmentError(
                "MLDataset timestamps must be chronologically non-decreasing"
            )

        # Lock NumPy memory buffers against accidental in-place mutation
        self.X.flags.writeable = False
        self.y.flags.writeable = False
        self.timestamps.flags.writeable = False

        object.__setattr__(self, "row_count", m)
        object.__setattr__(self, "feature_dim", d)

    def __len__(self) -> int:
        """Return the number of aligned samples."""
        return self.row_count

    def to_dict_list(self) -> list[dict[str, float]]:
        """Export rows as list of feature dictionaries for Phase 3 tool compatibility."""
        names = self.feature_names
        return [
            {names[j]: float(self.X[i, j]) for j in range(self.feature_dim)}
            for i in range(self.row_count)
        ]

    def __getitem__(self, item: slice | np.ndarray | Sequence[int]) -> MLDataset:
        """Slice or filter dataset rows, returning a new immutable MLDataset instance."""
        if isinstance(item, slice):
            sub_x = self.X[item].copy()
            sub_y = self.y[item].copy()
            sub_ts = self.timestamps[item].copy()
            return MLDataset(
                symbol=self.symbol,
                feature_names=self.feature_names,
                X=sub_x,
                y=sub_y,
                timestamps=sub_ts,
                target_spec=self.target_spec,
            )

        if isinstance(item, (np.ndarray, list, tuple)):
            arr = np.asarray(item)
            if arr.dtype == bool:
                if len(arr) != self.row_count:
                    raise IndexError(
                        f"Boolean mask length ({len(arr)}) must match row count ({self.row_count})"
                    )
            elif not np.issubdtype(arr.dtype, np.integer):
                raise TypeError(f"Invalid index array dtype: {arr.dtype}")

            sub_x = self.X[arr].copy()
            sub_y = self.y[arr].copy()
            sub_ts = self.timestamps[arr].copy()
            return MLDataset(
                symbol=self.symbol,
                feature_names=self.feature_names,
                X=sub_x,
                y=sub_y,
                timestamps=sub_ts,
                target_spec=self.target_spec,
            )

        type_name = type(item).__name__
        raise TypeError(
            "MLDataset indexing requires a slice, boolean mask, or integer index sequence; "
            f"single integer indexing ({type_name}) is not supported to preserve matrix shape"
        )

    def slice(self, start: int, end: int) -> MLDataset:
        """Return a slice of the dataset [start:end]."""
        if start < 0 or end > self.row_count or start > end:
            raise MLDatasetError(
                f"Invalid slice bounds: start={start}, end={end}, row_count={self.row_count}"
            )
        return self[start:end]

    def select(
        self,
        columns_or_indices: Sequence[str] | Sequence[int] | np.ndarray,
    ) -> MLDataset:
        """Filter dataset columns by feature names or rows by index/mask."""
        if isinstance(columns_or_indices, (list, tuple)) and (
            len(columns_or_indices) == 0 or isinstance(columns_or_indices[0], str)
        ):
            selected_names = tuple(columns_or_indices)
            for name in selected_names:
                if name not in self.feature_names:
                    raise MLDatasetError(
                        f"Feature '{name}' not found in dataset feature_names: {self.feature_names}"
                    )
            col_indices = [self.feature_names.index(name) for name in selected_names]
            new_x = self.X[:, col_indices].copy()
            return MLDataset(
                symbol=self.symbol,
                feature_names=selected_names,
                X=new_x,
                y=self.y.copy(),
                timestamps=self.timestamps.copy(),
                target_spec=self.target_spec,
            )

        row_indices: Sequence[int] | np.ndarray = columns_or_indices  # type: ignore[assignment]
        return self[row_indices]

    def train_val_test_split(
        self,
        train_ratio_or_splits: float | DatasetSplits[Any] | None = None,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        *,
        train_ratio: float | None = None,
        label_horizon_bars: int = 0,
        embargo_bars: int = 0,
    ) -> tuple[MLDataset, MLDataset, MLDataset]:
        """Split dataset chronologically into Train, Validation, and Test partitions.

        Supports:
        - Passing an external `DatasetSplits` (slices by timestamp bounds if present)
        - Passing positional float ratios: `train_val_test_split(0.6, 0.2, 0.2)`
        - Passing keyword float ratios with optional embargo
        """
        if isinstance(train_ratio_or_splits, DatasetSplits):
            splits = train_ratio_or_splits
            tb = splits.train_boundary
            vb = splits.val_boundary
            teb = splits.test_boundary

            # Check if timestamp boundaries are valid and non-zero
            if tb.start_ts > 0 and tb.end_ts >= tb.start_ts:
                t_mask = (self.timestamps >= tb.start_ts) & (self.timestamps <= tb.end_ts)
                v_mask = (self.timestamps >= vb.start_ts) & (self.timestamps <= vb.end_ts)
                te_mask = (self.timestamps >= teb.start_ts) & (self.timestamps <= teb.end_ts)

                if not np.any(t_mask):
                    raise DatasetAlignmentError(
                        "Train partition is empty under provided timestamp bounds"
                    )
                if not np.any(v_mask):
                    raise DatasetAlignmentError(
                        "Validation partition is empty under provided timestamp bounds"
                    )
                if not np.any(te_mask):
                    raise DatasetAlignmentError(
                        "Test partition is empty under provided timestamp bounds"
                    )

                return self[t_mask], self[v_mask], self[te_mask]

            # Index-based boundaries
            return (
                self[tb.start_index : tb.end_index],
                self[vb.start_index : vb.end_index],
                self[teb.start_index : teb.end_index],
            )

        # Ratio resolution
        if isinstance(train_ratio_or_splits, (int, float)):
            actual_train_ratio = float(train_ratio_or_splits)
            actual_val_ratio = float(val_ratio)
            actual_test_ratio = float(test_ratio)
        elif train_ratio is not None:
            actual_train_ratio = float(train_ratio)
            actual_val_ratio = float(val_ratio)
            actual_test_ratio = float(test_ratio)
        else:
            actual_train_ratio = 0.60
            actual_val_ratio = float(val_ratio)
            actual_test_ratio = float(test_ratio)

        total_ratio = actual_train_ratio + actual_val_ratio + actual_test_ratio
        if not math.isclose(total_ratio, 1.0, abs_tol=1e-5):
            raise MLDatasetError(
                f"Split ratios must sum to 1.0, got sum={total_ratio} "
                f"({actual_train_ratio} + {actual_val_ratio} + {actual_test_ratio})"
            )

        splitter = ChronologicalSplitter(
            train_ratio=actual_train_ratio,
            val_ratio=actual_val_ratio,
            test_ratio=actual_test_ratio,
            label_horizon_bars=label_horizon_bars,
            embargo_bars=embargo_bars,
        )
        int_splits = splitter.split(list(range(self.row_count)))
        tb = int_splits.train_boundary
        vb = int_splits.val_boundary
        teb = int_splits.test_boundary
        return (
            self[tb.start_index : tb.end_index],
            self[vb.start_index : vb.end_index],
            self[teb.start_index : teb.end_index],
        )

    def fit_scaler(self, scaler: StandardScaler | None = None) -> StandardScaler:
        """Fit a StandardScaler strictly on this dataset partition (train)."""
        return fit_scaler_on_dataset(self, scaler)

    def transform_with(self, scaler: StandardScaler) -> MLDataset:
        """Return a new MLDataset with X normalized by frozen StandardScaler parameters."""
        return transform_dataset(self, scaler)


def build_ml_dataset(
    candles: Sequence[Candle],
    feature_dataset: FeatureDataset,
    target_spec: TargetSpec,
    *,
    targets: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> MLDataset:
    """Align historical candles, causal FeatureDataset, and target labels into an MLDataset.

    Parameters:
        candles: Chronological sequence of complete historical candles.
        feature_dataset: Output of CausalFeaturePipeline.transform(candles).
        target_spec: Target formulation specification.
        targets: Optional precomputed target array of shape (N,). If None, generated automatically.
        valid_mask: Optional precomputed mask of shape (N,). If None, generated automatically.

    Returns:
        MLDataset with aligned X, y, timestamps, and metadata. Drops warmup bars (is_warmup=True)
        and final H bars lacking future outcomes (valid_mask=False).

    Raises:
        DatasetAlignmentError: On mismatched lengths, invalid candle chronology, empty rows,
                               or non-finite values.
    """
    if not candles:
        raise DatasetAlignmentError("Candles sequence cannot be empty")
    if not feature_dataset.rows:
        raise DatasetAlignmentError("Cannot build MLDataset from an empty FeatureDataset")

    n = len(candles)
    symbol = feature_dataset.symbol

    if len(candles) != feature_dataset.row_count:
        raise DatasetAlignmentError(
            f"Length mismatch: candles ({len(candles)}) != "
            f"feature_dataset ({feature_dataset.row_count})"
        )

    if candles[0].symbol != symbol:
        raise DatasetAlignmentError(
            f"Candle symbol '{candles[0].symbol}' does not match FeatureDataset symbol '{symbol}'"
        )

    # 1. Validate candle chronology and completeness
    prev_from = -1
    for i, c in enumerate(candles):
        if not c.is_complete:
            raise DatasetAlignmentError(
                f"Incomplete candle detected at index {i} (from_ts={c.from_ts})"
            )
        if i > 0 and c.from_ts <= prev_from:
            raise DatasetAlignmentError(
                f"Non-chronological candles at index {i}: from_ts={c.from_ts} <= prev={prev_from}"
            )
        prev_from = c.from_ts

    # 2. Compute targets if not provided
    if targets is None or valid_mask is None:
        from iqoption_research.ml.targets.generator import generate_targets

        targets, valid_mask = generate_targets(candles, target_spec)

    if len(targets) != n:
        raise DatasetAlignmentError(
            f"Targets length ({len(targets)}) does not match candles length ({n})"
        )
    if len(valid_mask) != n:
        raise DatasetAlignmentError(
            f"Valid mask length ({len(valid_mask)}) does not match candles length ({n})"
        )

    # 3. Canonical feature names
    feat_names = tuple(f.name for f in feature_dataset.feature_specs)
    if not feat_names:
        raise DatasetAlignmentError("FeatureDataset has no active feature specifications")

    x_rows: list[list[float]] = []
    y_vals: list[int | float] = []
    ts_vals: list[int] = []

    # 4. Row-for-row alignment
    for row in feature_dataset.rows:
        t = row.index

        if t < 0 or t >= n:
            raise DatasetAlignmentError(
                f"FeatureRow index {t} is outside candle series bounds [0, {n})"
            )

        c = candles[t]
        if row.candle_to_ts != c.to_ts:
            raise DatasetAlignmentError(
                f"Timestamp mismatch at index {t}: "
                f"row to_ts={row.candle_to_ts} != candle to_ts={c.to_ts}"
            )
        if row.symbol != c.symbol:
            raise DatasetAlignmentError(
                f"Symbol mismatch at index {t}: row symbol={row.symbol} != candle symbol={c.symbol}"
            )

        # Drop warmup rows (insufficient historical lookback)
        if row.is_warmup:
            continue

        # Drop invalid target horizon rows (tail H bars)
        if not valid_mask[t]:
            continue

        # Extract feature vector in deterministic order
        row_feats: list[float] = []
        for fn in feat_names:
            if fn not in row.features:
                raise DatasetAlignmentError(f"Feature '{fn}' missing in FeatureRow at index {t}")
            v = row.features[fn]
            if not math.isfinite(v):
                raise DatasetAlignmentError(
                    f"Non-finite value ({v}) in feature '{fn}' at index {t}"
                )
            row_feats.append(float(v))

        target_val = targets[t]
        if not math.isfinite(target_val):
            raise DatasetAlignmentError(f"Non-finite target value ({target_val}) at index {t}")

        x_rows.append(row_feats)
        y_vals.append(target_val)
        ts_vals.append(c.to_ts)

    if not x_rows:
        raise DatasetAlignmentError(
            "Insufficient data: 0 valid samples remaining after filtering warmup "
            f"(max_lookback={feature_dataset.max_lookback}) and target horizon "
            f"(H={target_spec.horizon_bars})"
        )

    # 5. Build NumPy arrays
    x_arr = np.array(x_rows, dtype=np.float64)

    # Determine y dtype
    if target_spec.target_type in (TargetType.DIRECTION_BINARY, TargetType.DIRECTION_TERNARY):
        y_arr = np.array(y_vals, dtype=np.int64)
    else:
        y_arr = np.array(y_vals, dtype=np.float64)

    ts_arr = np.array(ts_vals, dtype=np.int64)

    return MLDataset(
        symbol=symbol,
        feature_names=feat_names,
        X=x_arr,
        y=y_arr,
        timestamps=ts_arr,
        target_spec=target_spec,
    )


def fit_scaler_on_dataset(
    dataset: MLDataset,
    scaler: StandardScaler | None = None,
) -> StandardScaler:
    """Fit a StandardScaler strictly on an MLDataset's training X matrix.

    Computes sample mean and sample variance (ddof=1) across dataset.X.
    Freezes parameters into ScalerParameters.
    """
    if dataset.row_count == 0:
        raise PreprocessingError("Cannot fit StandardScaler on empty MLDataset")

    sc = scaler if scaler is not None else StandardScaler()
    return sc.fit(dataset)


def transform_dataset(
    dataset: MLDataset,
    scaler: StandardScaler,
) -> MLDataset:
    """Transform MLDataset.X using previously frozen StandardScaler parameters."""
    if not scaler.is_fitted:
        raise PreprocessingError("Cannot transform MLDataset: StandardScaler is not fitted")

    params = scaler.params
    if tuple(dataset.feature_names) != tuple(params.feature_names):
        raise PreprocessingError(
            f"Feature names mismatch: scaler was fitted on {params.feature_names}, "
            f"but dataset contains {dataset.feature_names}"
        )

    mean_vec = np.array([params.mean[name] for name in dataset.feature_names], dtype=np.float64)
    scale_vec = np.array([params.scale[name] for name in dataset.feature_names], dtype=np.float64)

    scaled_x = (dataset.X - mean_vec) / scale_vec

    return MLDataset(
        symbol=dataset.symbol,
        feature_names=dataset.feature_names,
        X=scaled_x,
        y=dataset.y.copy(),
        timestamps=dataset.timestamps.copy(),
        target_spec=dataset.target_spec,
    )


def scale_dataset_partitions(
    train: MLDataset,
    val: MLDataset | None = None,
    test: MLDataset | None = None,
    *,
    scaler: StandardScaler | None = None,
) -> tuple[MLDataset, MLDataset | None, MLDataset | None, StandardScaler]:
    """Fit a scaler strictly on train, then apply frozen parameters to train, val, and test.

    Guarantees strict zero-leakage isolation: neither validation nor test features
    can influence the scaler's mean, variance, or scale parameters.
    """
    sc = fit_scaler_on_dataset(train, scaler)
    train_scaled = transform_dataset(train, sc)
    val_scaled = transform_dataset(val, sc) if val is not None else None
    test_scaled = transform_dataset(test, sc) if test is not None else None
    return train_scaled, val_scaled, test_scaled, sc
