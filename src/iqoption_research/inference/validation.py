"""Model integrity, schema compatibility, and artifact validation for streaming inference.

Ensures that loaded model artifacts strictly match the incoming streaming feature
schema, verify that transformers are properly fitted, and detect stale model weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from iqoption_research.ml.tracking.artifacts import ModelBundle
from iqoption_research.research.features.pipeline import FeatureRow

logger = logging.getLogger(__name__)

DEFAULT_MAX_MODEL_AGE_DAYS: Final[int] = 30


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Detailed audit report of model artifact and schema validation."""

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    feature_count: int
    model_name: str
    training_timestamp_utc: str | None


class ModelIntegrityValidator:
    """Validates model bundles against runtime schema and training recency."""

    def __init__(self, max_model_age_days: int = DEFAULT_MAX_MODEL_AGE_DAYS) -> None:
        self.max_model_age_days = max_model_age_days

    def validate_bundle(self, bundle: ModelBundle) -> ValidationReport:
        """Perform comprehensive integrity checks on a loaded ModelBundle."""
        errors: list[str] = []
        warnings: list[str] = []

        model = bundle.model
        model_name = getattr(model, "name", model.__class__.__name__)

        # 1. Check if model is fitted
        if not getattr(model, "is_fitted", False):
            errors.append(f"Model '{model_name}' is not marked as fitted.")

        # 2. Check feature names
        feature_names = bundle.feature_names
        if not feature_names:
            errors.append(f"Model '{model_name}' has an empty feature schema.")

        # 3. Check scaler dimensions
        if bundle.scaler is not None:
            if not getattr(bundle.scaler, "is_fitted", False):
                errors.append("StandardScaler in bundle is not marked as fitted.")
            elif len(bundle.scaler.params.feature_names) != len(feature_names):
                errors.append(
                    f"Scaler feature count ({len(bundle.scaler.params.feature_names)}) does not "
                    f"match model feature count ({len(feature_names)})."
                )

        # 4. Check calibrator
        if (
            bundle.calibrator is not None
            and not getattr(bundle.calibrator, "is_fitted", False)
        ):
            errors.append("ProbabilityCalibrator in bundle is not marked as fitted.")

        # 5. Check model age from metadata
        created_str = bundle.metadata.get("created_at_utc")
        if created_str:
            try:
                created_dt = datetime.fromisoformat(str(created_str))
                now_dt = datetime.now(UTC)
                age_days = (now_dt - created_dt).total_seconds() / 86400.0
                if age_days > self.max_model_age_days:
                    warnings.append(
                        f"Model is {age_days:.1f} days old (exceeds {self.max_model_age_days}d)."
                    )
            except (ValueError, TypeError):
                warnings.append(f"Could not parse artifact creation timestamp: {created_str}")
        else:
            warnings.append("Artifact metadata lacks 'created_at_utc' timestamp.")

        return ValidationReport(
            is_valid=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            feature_count=len(feature_names),
            model_name=model_name,
            training_timestamp_utc=str(created_str) if created_str else None,
        )

    def validate_feature_row_schema(
        self,
        bundle: ModelBundle,
        feature_row: FeatureRow,
    ) -> tuple[bool, str | None]:
        """Validate that an incoming FeatureRow contains all required model features."""
        expected_features = bundle.feature_names
        row_keys = set(feature_row.features.keys())

        missing = [f for f in expected_features if f not in row_keys]
        if missing:
            return False, f"Feature row missing {len(missing)} required features: {missing[:3]}..."

        return True, None
