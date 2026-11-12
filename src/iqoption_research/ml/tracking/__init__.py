"""Tamper-evident artifact persistence and experiment tracking for Phase 4.

This package intentionally contains no broker, network, or execution code. It only
persists local research artifacts with SHA-256 manifests and indexes experiment
metadata in a local SQLite database.
"""

from iqoption_research.ml.tracking.artifacts import (
    ArtifactError,
    ArtifactFile,
    ArtifactIntegrityError,
    ArtifactManifest,
    ArtifactSecurityError,
    ModelBundle,
    create_manifest,
    load_manifest,
    load_model_bundle,
    save_model_bundle,
    sha256_file,
    verify_manifest,
)
from iqoption_research.ml.tracking.tracker import (
    ExperimentRecord,
    ExperimentTracker,
    TrackerError,
)

__all__ = [
    "ArtifactError",
    "ArtifactFile",
    "ArtifactIntegrityError",
    "ArtifactManifest",
    "ArtifactSecurityError",
    "ExperimentRecord",
    "ExperimentTracker",
    "ModelBundle",
    "TrackerError",
    "create_manifest",
    "load_manifest",
    "load_model_bundle",
    "save_model_bundle",
    "sha256_file",
    "verify_manifest",
]
