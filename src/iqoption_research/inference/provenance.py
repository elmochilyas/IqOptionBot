"""Explicit model provenance classification for inference authority.

A model bundle must state where its training data came from before the runtime may
turn its predictions into paper-trade proposals. Synthetic engineering fixtures are
useful for wiring checks, but they must never be presented as evidence of real-market
predictive performance.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from iqoption_research.ml.tracking.artifacts import ModelBundle

PROVENANCE_METADATA_KEY: Final[str] = "model_provenance"
LEGACY_PROVENANCE_METADATA_KEY: Final[str] = "provenance"


class ModelProvenance(StrEnum):
    """Where a model bundle's training data came from."""

    DEMO_SYNTHETIC = "demo_synthetic"
    SYNTHETIC_TRAINED = "synthetic_trained"
    AUTHENTIC_TRAINED = "authentic_trained"
    PRODUCTION_CANDIDATE = "production_candidate"
    UNKNOWN = "unknown"

    @property
    def permits_actionable_signals(self) -> bool:
        """Only authentic or validated production models may propose paper trades."""
        return self in {
            ModelProvenance.AUTHENTIC_TRAINED,
            ModelProvenance.PRODUCTION_CANDIDATE,
        }


def classify_bundle_provenance(bundle: ModelBundle) -> ModelProvenance:
    """Resolve a bundle's provenance from metadata, failing closed to UNKNOWN."""
    raw = bundle.metadata.get(PROVENANCE_METADATA_KEY) or bundle.metadata.get(
        LEGACY_PROVENANCE_METADATA_KEY
    )
    if raw is None:
        return ModelProvenance.UNKNOWN
    normalized = str(raw).strip().lower()
    aliases = {
        "demo": ModelProvenance.DEMO_SYNTHETIC,
        "synthetic": ModelProvenance.SYNTHETIC_TRAINED,
        "synthetic_engineering_only": ModelProvenance.SYNTHETIC_TRAINED,
        "authentic": ModelProvenance.AUTHENTIC_TRAINED,
        "authentic_market_data": ModelProvenance.AUTHENTIC_TRAINED,
        "broker_authentic_collection": ModelProvenance.AUTHENTIC_TRAINED,
        "production": ModelProvenance.PRODUCTION_CANDIDATE,
    }
    try:
        return ModelProvenance(normalized)
    except ValueError:
        return aliases.get(normalized, ModelProvenance.UNKNOWN)
