"""Leak-free feature preprocessing and parameter persistence.

Ensures scalers fit strictly on TRAIN data and freeze their parameters
to prevent data leakage into validation and test sets.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from iqoption_research.research.features.pipeline import FeatureRow


class PreprocessingError(ValueError):
    """Raised when scaling inputs are invalid or violate finite number constraints."""


@dataclass(frozen=True, slots=True)
class ScalerParameters:
    """Frozen parameters of a fitted StandardScaler."""

    feature_names: tuple[str, ...]
    mean: dict[str, float]
    scale: dict[str, float]
    var: dict[str, float]
    n_samples_seen: int
    fitted_at_utc: str
    epsilon: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean,
            "scale": self.scale,
            "var": self.var,
            "n_samples_seen": self.n_samples_seen,
            "fitted_at_utc": self.fitted_at_utc,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScalerParameters:
        return cls(
            feature_names=tuple(data["feature_names"]),
            mean={k: float(v) for k, v in data["mean"].items()},
            scale={k: float(v) for k, v in data["scale"].items()},
            var={k: float(v) for k, v in data["var"].items()},
            n_samples_seen=int(data["n_samples_seen"]),
            fitted_at_utc=str(data["fitted_at_utc"]),
            epsilon=float(data["epsilon"]),
        )


class StandardScaler:
    """Deterministic z-score standard scaler with parameter freezing and serialization."""

    def __init__(
        self,
        *,
        epsilon: float = 1e-8,
        on_nan: str = "raise",  # "raise" | "drop"
        on_inf: str = "raise",  # "raise"
    ) -> None:
        self.epsilon = epsilon
        self.on_nan = on_nan
        self.on_inf = on_inf
        self._params: ScalerParameters | None = None

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    @property
    def params(self) -> ScalerParameters:
        if self._params is None:
            raise PreprocessingError("StandardScaler is not fitted yet")
        return self._params

    def _extract_dict(self, item: Any) -> dict[str, float]:
        if isinstance(item, FeatureRow):
            return item.features
        if isinstance(item, dict):
            return item
        raise PreprocessingError(f"Unsupported item type for scaling: {type(item)}")

    def fit(
        self,
        data: Sequence[Any] | np.ndarray | Any,
        feature_names: Sequence[str] | None = None,
    ) -> StandardScaler:
        """Compute sample mean and standard deviation strictly on TRAIN observations."""
        if not isinstance(data, np.ndarray) and not data:
            raise PreprocessingError("Cannot fit StandardScaler on empty training data")

        # Native MLDataset support
        if hasattr(data, "X") and hasattr(data, "feature_names") and hasattr(data, "row_count"):
            if data.row_count == 0:
                raise PreprocessingError("Cannot fit StandardScaler on empty MLDataset")
            return self._fit_array(data.X, data.feature_names)

        # 2D NumPy array support
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise PreprocessingError("NumPy array data must be 2D (N, D)")
            if data.shape[0] == 0:
                raise PreprocessingError("Cannot fit StandardScaler on empty array")
            names = (
                tuple(feature_names)
                if feature_names is not None
                else tuple(f"feat_{j}" for j in range(data.shape[1]))
            )
            return self._fit_array(data, names)

        rows = [self._extract_dict(item) for item in data]
        feature_names_list = sorted(rows[0].keys())
        n = len(rows)

        # Validate values across rows
        means: dict[str, float] = {}
        vars_: dict[str, float] = {}
        scales: dict[str, float] = {}

        for feat in feature_names_list:
            vals: list[float] = []
            for r_idx, r in enumerate(rows):
                if feat not in r:
                    raise PreprocessingError(f"Feature '{feat}' missing in row index {r_idx}")
                v = r[feat]
                if math.isnan(v):
                    if self.on_nan == "raise":
                        raise PreprocessingError(f"NaN detected in feature '{feat}' at row {r_idx}")
                    continue
                if math.isinf(v):
                    raise PreprocessingError(f"Inf detected in feature '{feat}' at row {r_idx}")
                vals.append(v)

            if not vals:
                raise PreprocessingError(f"Feature '{feat}' has no valid numeric samples")

            count = len(vals)
            mean_val = sum(vals) / count
            var_val = (
                sum((x - mean_val) ** 2 for x in vals) / (count - 1)
                if count > 1
                else 0.0
            )

            std_val = math.sqrt(var_val)
            # Avoid division by zero on constant features
            scale_val = std_val if std_val > self.epsilon else 1.0

            means[feat] = mean_val
            vars_[feat] = var_val
            scales[feat] = scale_val

        self._params = ScalerParameters(
            feature_names=tuple(feature_names_list),
            mean=means,
            scale=scales,
            var=vars_,
            n_samples_seen=n,
            fitted_at_utc=datetime.now(UTC).isoformat(),
            epsilon=self.epsilon,
        )
        return self

    def _fit_array(self, X: np.ndarray, feature_names: Sequence[str]) -> StandardScaler:
        """Internal helper for high-performance vectorized array fitting."""
        if X.shape[1] != len(feature_names):
            raise PreprocessingError("Feature names count does not match array columns")
        if not np.all(np.isfinite(X)):
            if self.on_nan == "raise" and np.any(np.isnan(X)):
                raise PreprocessingError("NaN detected in feature array")
            if np.any(np.isinf(X)):
                raise PreprocessingError("Inf detected in feature array")

        count = int(X.shape[0])
        means = np.mean(X, axis=0)
        vars_ = np.var(X, axis=0, ddof=1) if count > 1 else np.zeros(X.shape[1])
        stds = np.sqrt(np.maximum(vars_, 0.0))
        scales = np.where(stds > self.epsilon, stds, 1.0)

        names_tuple = tuple(feature_names)
        self._params = ScalerParameters(
            feature_names=names_tuple,
            mean={name: float(means[j]) for j, name in enumerate(names_tuple)},
            scale={name: float(scales[j]) for j, name in enumerate(names_tuple)},
            var={name: float(vars_[j]) for j, name in enumerate(names_tuple)},
            n_samples_seen=count,
            fitted_at_utc=datetime.now(UTC).isoformat(),
            epsilon=self.epsilon,
        )
        return self

    def transform(
        self,
        data: Sequence[Any] | np.ndarray | Any,
    ) -> list[dict[str, float]] | np.ndarray | Any:
        """Standardize observations using previously frozen training parameters."""
        if not self.is_fitted:
            raise PreprocessingError("Cannot transform data: StandardScaler is not fitted")

        # Native MLDataset transform
        if hasattr(data, "X") and hasattr(data, "feature_names") and hasattr(data, "symbol"):
            from iqoption_research.ml.targets.dataset import MLDataset, transform_dataset

            if isinstance(data, MLDataset):
                return transform_dataset(data, self)

        # 2D NumPy array transform
        if isinstance(data, np.ndarray):
            return self._transform_array(data, self.params.feature_names)

        params = self.params
        scaled_rows: list[dict[str, float]] = []

        for r_idx, item in enumerate(data):
            r = self._extract_dict(item)
            scaled: dict[str, float] = {}
            for feat in params.feature_names:
                if feat not in r:
                    raise PreprocessingError(f"Feature '{feat}' missing in row index {r_idx}")
                v = r[feat]
                if math.isnan(v):
                    if self.on_nan == "raise":
                        raise PreprocessingError(f"NaN detected in feature '{feat}' at row {r_idx}")
                    scaled[feat] = float("nan")
                    continue
                if math.isinf(v):
                    raise PreprocessingError(f"Inf detected in feature '{feat}' at row {r_idx}")

                scaled[feat] = (v - params.mean[feat]) / params.scale[feat]
            scaled_rows.append(scaled)

        return scaled_rows

    def _transform_array(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        """Internal helper for high-performance vectorized array transformation."""
        params = self.params
        if tuple(feature_names) != tuple(params.feature_names):
            raise PreprocessingError(
                f"Feature names mismatch: scaler expects {params.feature_names}, "
                f"got {tuple(feature_names)}"
            )
        if not np.all(np.isfinite(X)):
            if self.on_nan == "raise" and np.any(np.isnan(X)):
                raise PreprocessingError("NaN detected in transform data")
            if np.any(np.isinf(X)):
                raise PreprocessingError("Inf detected in transform data")

        mean_vec = np.array(
            [params.mean[name] for name in params.feature_names], dtype=np.float64
        )
        scale_vec = np.array(
            [params.scale[name] for name in params.feature_names], dtype=np.float64
        )
        return (X - mean_vec) / scale_vec

    def fit_transform(
        self,
        data: Sequence[Any] | np.ndarray | Any,
        feature_names: Sequence[str] | None = None,
    ) -> list[dict[str, float]] | np.ndarray | Any:
        """Fit strictly on the provided training set and transform it."""
        return self.fit(data, feature_names=feature_names).transform(data)

    def to_json(self) -> str:
        """Serialize fitted parameters to JSON string for persistent reproducibility."""
        return json.dumps(self.params.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> StandardScaler:
        """Construct a frozen StandardScaler from serialized JSON parameters."""
        data = json.loads(json_str)
        scaler = cls(epsilon=float(data.get("epsilon", 1e-8)))
        scaler._params = ScalerParameters.from_dict(data)
        return scaler
