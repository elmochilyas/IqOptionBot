"""Safe model artifact loader verifying SHA-256 manifests and schema consistency.

Wraps model bundle loading with cryptographic integrity checks and cached access.
"""

from __future__ import annotations

import logging
from pathlib import Path

from iqoption_research.inference.validation import ModelIntegrityValidator, ValidationReport
from iqoption_research.ml.tracking.artifacts import (
    ArtifactError,
    ModelBundle,
    load_model_bundle,
)

logger = logging.getLogger(__name__)


class SafeModelArtifactLoader:
    """Loads and validates model bundles from trusted local storage with caching."""

    def __init__(
        self,
        validator: ModelIntegrityValidator | None = None,
    ) -> None:
        self.validator = validator or ModelIntegrityValidator()
        self._cache: dict[str, ModelBundle] = {}

    def load(
        self,
        bundle_path: Path | str,
        *,
        use_cache: bool = True,
    ) -> tuple[ModelBundle, ValidationReport]:
        """Load a ModelBundle from disk and verify its integrity and schema."""
        resolved = Path(bundle_path).resolve()
        cache_key = str(resolved)

        if use_cache and cache_key in self._cache:
            bundle = self._cache[cache_key]
            report = self.validator.validate_bundle(bundle)
            return bundle, report

        if not resolved.exists() or not resolved.is_dir():
            raise FileNotFoundError(f"Model bundle directory not found: {resolved}")

        # Cryptographically verified load via manifest
        try:
            bundle = load_model_bundle(resolved)
        except ArtifactError as exc:
            raise ValueError(f"Model artifact integrity check failed: {exc}") from exc

        # Deep schema & recency validation
        report = self.validator.validate_bundle(bundle)
        if not report.is_valid:
            error_msg = "; ".join(report.errors)
            raise ValueError(f"Model bundle validation failed: {error_msg}")

        if use_cache:
            self._cache[cache_key] = bundle

        return bundle, report
