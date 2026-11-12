"""Tamper-evident model artifact bundles with SHA-256 manifests.

Artifacts contain a serialized model, optional frozen StandardScaler, optional
probability calibrator, and JSON metadata. Every file is hashed into a manifest
written last. Loading verifies the complete manifest and rejects path traversal,
missing files, modified files, and unexpected extra files.

Security notes:
- Artifacts must only be loaded from trusted local directories produced by this
  project. Joblib/pickle payloads are not safe to deserialize from untrusted
  sources.
- No network access, broker connection, or execution authority exists here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelPersistenceError,
    load_model,
)
from iqoption_research.research.features.scalers import StandardScaler


class ArtifactError(Exception):
    """Base exception for artifact persistence and verification errors."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when hashes, sizes, or the file set do not match the manifest."""


class ArtifactSecurityError(ArtifactError):
    """Raised when an artifact path attempts traversal or escapes its root."""


MANIFEST_FILENAME = "artifact_manifest.json"
BUNDLE_METADATA_FILENAME = "bundle_metadata.json"
SCALER_FILENAME = "scaler.json"
CALIBRATOR_FILENAME = "calibrator.json"
MANIFEST_VERSION = "1.0"
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """One hashed file entry inside an artifact manifest."""

    path: str
    sha256: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": int(self.size)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactFile:
        path = str(data["path"])
        digest = str(data["sha256"])
        size = data["size"]
        if not path or not digest:
            raise ArtifactIntegrityError("Manifest file entries require path and sha256.")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArtifactIntegrityError(f"Invalid file size in manifest for {path!r}: {size!r}")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ArtifactIntegrityError(f"Invalid SHA-256 digest for {path!r}: {digest!r}")
        return cls(path=path, sha256=digest.lower(), size=size)


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Deterministic manifest of every file in an artifact directory."""

    files: tuple[ArtifactFile, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest_version: str = MANIFEST_VERSION
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "created_at_utc": self.created_at_utc,
            "files": [f.to_dict() for f in self.files],
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        if str(data.get("manifest_version", "")) != MANIFEST_VERSION:
            raise ArtifactIntegrityError(
                f"Unsupported artifact manifest version: {data.get('manifest_version')!r}"
            )
        raw_files = data.get("files")
        if not isinstance(raw_files, list):
            raise ArtifactIntegrityError("Manifest 'files' must be a list.")
        files = tuple(ArtifactFile.from_dict(item) for item in raw_files)
        paths = [f.path for f in files]
        if len(paths) != len(set(paths)):
            raise ArtifactIntegrityError("Manifest contains duplicate file paths.")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ArtifactIntegrityError("Manifest metadata must be a dictionary.")
        return cls(
            files=files,
            metadata=dict(metadata),
            manifest_version=str(data.get("manifest_version", MANIFEST_VERSION)),
            created_at_utc=str(data.get("created_at_utc", datetime.now(UTC).isoformat())),
        )

    def digest(self) -> str:
        """SHA-256 digest of the canonical manifest payload."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """A verified model bundle loaded from disk."""

    model: BaseMLModel
    scaler: StandardScaler | None
    calibrator: ProbabilityCalibrator | None
    metadata: dict[str, Any]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.model.feature_names)


def sha256_file(path: Path | str) -> str:
    """Compute the SHA-256 hex digest of a file without loading it into memory."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert common scientific Python values into JSON-serializable objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _safe_path(root: Path, relative: str) -> Path:
    """Resolve a manifest-relative path while enforcing containment under root."""
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts:
        raise ArtifactSecurityError(f"Absolute or empty artifact path rejected: {relative!r}")
    for part in pure.parts:
        if part in ("", ".", ".."):
            raise ArtifactSecurityError(f"Path traversal component rejected: {relative!r}")
        if ":" in part or "\\" in part:
            raise ArtifactSecurityError(f"Unsafe artifact path component rejected: {relative!r}")

    candidate = (root / Path(*pure.parts)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and not candidate.is_relative_to(resolved_root):
        raise ArtifactSecurityError(f"Artifact path escapes root: {relative!r}")
    if candidate.is_symlink():
        raise ArtifactSecurityError(f"Symbolic link artifacts are not supported: {relative!r}")
    return candidate


def _iter_artifact_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_FILENAME:
            continue
        if path.is_symlink():
            raise ArtifactSecurityError(f"Symbolic link artifacts are not supported: {path}")
        files.append(path)
    return files


def create_manifest(
    root_dir: Path | str,
    *,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Hash every regular file under ``root_dir`` into a manifest (in memory)."""
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise ArtifactError(f"Artifact root is not a directory: {root}")

    entries = [
        ArtifactFile(
            path=path.relative_to(root).as_posix(),
            sha256=sha256_file(path),
            size=int(path.stat().st_size),
        )
        for path in _iter_artifact_files(root)
    ]
    return ArtifactManifest(files=tuple(entries), metadata=dict(metadata or {}))


def write_manifest(
    root_dir: Path | str,
    *,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Compute and atomically write the artifact manifest last."""
    root = Path(root_dir).resolve()
    manifest = create_manifest(root, metadata=metadata)
    manifest_path = root / MANIFEST_FILENAME
    temporary = root / f".{MANIFEST_FILENAME}.tmp"
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def load_manifest(root_dir: Path | str) -> ArtifactManifest:
    """Load and parse the manifest file from an artifact directory."""
    manifest_path = Path(root_dir).resolve() / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(f"Artifact manifest missing: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(f"Corrupted artifact manifest: {manifest_path}") from exc
    if not isinstance(raw, dict):
        raise ArtifactIntegrityError("Artifact manifest must be a JSON object.")
    return ArtifactManifest.from_dict(raw)


def verify_manifest(
    root_dir: Path | str,
    *,
    manifest: ArtifactManifest | None = None,
    allow_extra_files: bool = False,
) -> ArtifactManifest:
    """Verify all manifest file hashes/sizes and reject unexpected files."""
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise ArtifactIntegrityError(f"Artifact directory does not exist: {root}")

    expected = manifest if manifest is not None else load_manifest(root)
    expected_paths: set[str] = set()

    for entry in expected.files:
        target = _safe_path(root, entry.path)
        if not target.is_file():
            raise ArtifactIntegrityError(f"Manifest file is missing: {entry.path}")
        actual_size = int(target.stat().st_size)
        if actual_size != entry.size:
            raise ArtifactIntegrityError(
                f"Size mismatch for {entry.path}: expected {entry.size}, got {actual_size}"
            )
        actual_digest = sha256_file(target)
        if actual_digest != entry.sha256:
            raise ArtifactIntegrityError(
                f"Checksum mismatch for {entry.path}: expected {entry.sha256}, "
                f"got {actual_digest}"
            )
        expected_paths.add(entry.path)

    if not allow_extra_files:
        actual_paths = {
            path.relative_to(root).as_posix() for path in _iter_artifact_files(root)
        }
        extras = sorted(actual_paths - expected_paths)
        if extras:
            raise ArtifactIntegrityError(
                f"Unexpected files present in artifact: {', '.join(extras)}"
            )

    return expected


def _ensure_empty_artifact_dir(root: Path, *, overwrite: bool) -> None:
    if root.exists() and not root.is_dir():
        raise ArtifactError(f"Artifact path exists and is not a directory: {root}")
    if root.is_dir() and any(root.iterdir()):
        if not overwrite:
            raise ArtifactError(
                f"Artifact directory is not empty: {root}. Use a fresh directory or "
                "explicitly request overwrite."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def save_model_bundle(
    directory: Path | str,
    model: BaseMLModel,
    *,
    scaler: StandardScaler | None = None,
    calibrator: ProbabilityCalibrator | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> ArtifactManifest:
    """Persist a model and optional preprocessing/calibration artifacts."""
    root = Path(directory).resolve()
    if not model.is_fitted:
        raise ModelPersistenceError(f"Cannot persist unfitted model '{model.name}'.")
    if scaler is not None and not scaler.is_fitted:
        raise ModelPersistenceError("Cannot persist an unfitted StandardScaler.")
    if calibrator is not None and not calibrator.is_fitted:
        raise ModelPersistenceError("Cannot persist an unfitted ProbabilityCalibrator.")

    _ensure_empty_artifact_dir(root, overwrite=overwrite)

    model.save(root)
    if scaler is not None:
        (root / SCALER_FILENAME).write_text(scaler.to_json(), encoding="utf-8")
    if calibrator is not None:
        calibrator.save(root / CALIBRATOR_FILENAME)

    bundle_meta: dict[str, Any] = {
        "bundle_format_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": model.get_metadata(),
        "has_scaler": scaler is not None,
        "has_calibrator": calibrator is not None,
        "calibrator_method": calibrator.method if calibrator is not None else None,
        "feature_names": list(model.feature_names),
        "metadata": _jsonable(metadata or {}),
        "security_note": (
            "Trusted local artifact only. Do not load joblib/pickle artifacts from "
            "untrusted sources."
        ),
    }
    (root / BUNDLE_METADATA_FILENAME).write_text(
        json.dumps(bundle_meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return write_manifest(root, metadata={"kind": "model_bundle", **bundle_meta})


def load_model_bundle(
    directory: Path | str,
    *,
    verify: bool = True,
) -> ModelBundle:
    """Verify and load a model bundle, including optional scaler and calibrator."""
    root = Path(directory).resolve()
    if verify:
        verify_manifest(root)

    meta_path = root / BUNDLE_METADATA_FILENAME
    if not meta_path.is_file():
        raise ArtifactIntegrityError(f"Bundle metadata missing: {meta_path}")
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(f"Corrupted bundle metadata: {meta_path}") from exc
    if not isinstance(metadata, dict):
        raise ArtifactIntegrityError("Bundle metadata must be a JSON object.")

    model = load_model(root)

    scaler_path = _safe_path(root, SCALER_FILENAME)
    scaler = (
        StandardScaler.from_json(scaler_path.read_text(encoding="utf-8"))
        if scaler_path.is_file()
        else None
    )

    calibrator_path = _safe_path(root, CALIBRATOR_FILENAME)
    calibrator = (
        ProbabilityCalibrator.load(calibrator_path) if calibrator_path.is_file() else None
    )

    if (
        scaler is not None
        and model.feature_names
        and tuple(scaler.params.feature_names) != tuple(model.feature_names)
    ):
        raise ArtifactIntegrityError(
            "Scaler feature names do not match the persisted model feature names."
        )
    if calibrator is not None and not calibrator.is_fitted:
        raise ArtifactIntegrityError("Persisted calibrator is not fitted.")

    return ModelBundle(
        model=model,
        scaler=scaler,
        calibrator=calibrator,
        metadata=metadata,
    )
