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

    def fit(self, data: Sequence[Any]) -> StandardScaler:
        """Compute sample mean and standard deviation strictly on TRAIN observations."""
        if not data:
            raise PreprocessingError("Cannot fit StandardScaler on empty training data")

        rows = [self._extract_dict(item) for item in data]
        feature_names = sorted(rows[0].keys())
        n = len(rows)

        # Validate values across rows
        means: dict[str, float] = {}
        vars_: dict[str, float] = {}
        scales: dict[str, float] = {}

        for feat in feature_names:
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
            feature_names=tuple(feature_names),
            mean=means,
            scale=scales,
            var=vars_,
            n_samples_seen=n,
            fitted_at_utc=datetime.now(UTC).isoformat(),
            epsilon=self.epsilon,
        )
        return self

    def transform(self, data: Sequence[Any]) -> list[dict[str, float]]:
        """Standardize observations using previously frozen training parameters."""
        if not self.is_fitted:
            raise PreprocessingError("Cannot transform data: StandardScaler is not fitted")

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

    def fit_transform(self, data: Sequence[Any]) -> list[dict[str, float]]:
        """Fit strictly on the provided training set and transform it."""
        return self.fit(data).transform(data)

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
