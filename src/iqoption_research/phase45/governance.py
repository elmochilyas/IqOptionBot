"""Phase 45 derived development-population publishing and governance.

Derived populations are published only under ``data/phase45/snapshots``. Every
published parquet and manifest is immutable; all parent evidence remains
read-only. Manifests are self-digested and bind exact source/file hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Final

from iqoption_research.market_data.parquet_storage import (
    read_candles_parquet,
    write_candles_parquet,
)
from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_END_TS,
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
    AuthorizedDevelopmentPopulation,
    compute_candle_content_digest,
)
from iqoption_research.models.market import Candle
from iqoption_research.phase45.evidence import (
    AUTHORITATIVE_M1_SNAPSHOT_ID,
    DISPOSITION,
    EXECUTION_AUTHORITY,
    PROTECTED_EXPOSURE_CLASS,
    RESEARCH_PLAN_ID,
    SUPPORTED_EVIDENCE_TIMEFRAMES,
    build_evidence_inventory,
    load_population_candles,
    verify_all_receipts,
)
from iqoption_research.research.prospective import digest

MANIFEST_SCHEMA_VERSION: Final[str] = "phase45.1"
EXPOSURE_HISTORY: Final[tuple[str, ...]] = (
    "PHASE42_EURUSD_M1_RESEARCH_HISTORICAL_REUSE",
    "PHASE43_MULTI_HORIZON_RESEARCH_HISTORICAL_REUSE",
    "PHASE44_PROTECTED_BOUNDARY_AUDIT_HISTORICAL_REUSE",
)
FORBIDDEN_SNAPSHOT_MARKERS: Final[tuple[str, ...]] = (
    "SYNTHETIC",
    "TEST_ONLY",
    "TEST_FIXTURE",
    "FAKE",
    "FABRICAT",
)
BOUNDARY_POLICY: Final[dict[str, Any]] = {
    "development_membership": "from_ts < boundary_ts",
    "last_allowed_decision_start": "from_ts",
    "to_ts_equals_boundary_allowed": True,
    "to_ts_strictly_pre_boundary_required": False,
    "label_maturity_invalidation": "max_label_maturity_ts",
    "note": (
        "Boundary-touching candles are retained when from_ts < boundary_ts; to_ts may equal "
        "boundary_ts. Label maturity at or after the boundary is invalidated downstream by "
        "max_label_maturity_ts, not by dropping the decision-start candle."
    ),
}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _repo_relative(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


@dataclass(frozen=True, slots=True)
class ResearchDatasetGovernanceManifest:
    """Tamper-evident governance record for one derived development population.

    ``boundary_policy`` records the exact development-boundary semantics: a
    candle is retained when its ``from_ts`` is strictly before ``boundary_ts``
    and its ``to_ts`` may equal that boundary. It is not a protected-data
    violation; label maturity is invalidated downstream by
    ``max_label_maturity_ts``.
    """

    manifest_id: str
    symbol: str
    timeframe_seconds: int
    population_id: str
    source_snapshot_id: str
    source_content_digest: str
    snapshot_id: str
    parquet_path: str
    parquet_sha256: str
    content_digest: str
    from_ts: int
    to_ts: int
    candle_count: int
    boundary_ts: int
    provenance_status: str
    exposure_history: tuple[str, ...]
    plan_id: str
    parent_hashes: dict[str, str]
    source_file_hashes: tuple[dict[str, Any], ...]
    schema_version: str = MANIFEST_SCHEMA_VERSION
    disposition: str = DISPOSITION
    execution_authority: str = EXECUTION_AUTHORITY
    manifest_sha256: str = ""
    boundary_policy: dict[str, Any] = field(default_factory=lambda: dict(BOUNDARY_POLICY))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete serialized manifest, including its computed digest."""
        return asdict(self)

    def _digest_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("manifest_sha256", None)
        return payload

    def compute_digest(self) -> str:
        """Return SHA-256 of the canonical manifest payload excluding its own digest."""
        return digest(self._digest_payload())


def _source_snapshot_id(symbol: str, timeframe_seconds: int) -> str:
    if timeframe_seconds == 60:
        return AUTHORITATIVE_M1_SNAPSHOT_ID
    return f"receipt_bound_{symbol.lower()}_{timeframe_seconds}s"


def _collect_source_file_hashes(
    repo_root: Path,
    authoritative_parquet_path: Path,
    authoritative_parquet_sha256: str,
) -> tuple[dict[str, Any], ...]:
    hashes: list[dict[str, Any]] = [
        {
            "path": _repo_relative(repo_root, authoritative_parquet_path),
            "sha256": authoritative_parquet_sha256,
            "role": "AUTHORITATIVE_SNAPSHOT_PARQUET",
        }
    ]
    for _timeframe, summary in sorted(verify_all_receipts(repo_root).items()):
        receipt_ids = tuple(summary["receipt_ids"])
        receipt_hashes = tuple(summary["receipt_sha256"])
        parquet_paths = tuple(summary["parquet_paths"])
        parquet_hashes = tuple(summary["parquet_sha256"])
        for index, receipt_id in enumerate(receipt_ids):
            hashes.append({
                "path": f"data/collection_receipts/{receipt_id}.json",
                "sha256": receipt_hashes[index],
                "role": "RECEIPT_JSON",
                "receipt_id": receipt_id,
            })
            hashes.append({
                "path": parquet_paths[index],
                "sha256": parquet_hashes[index],
                "role": "RECEIPT_PARQUET",
                "receipt_id": receipt_id,
            })
    return tuple(hashes)


def publish_development_population(
    repo_root: Path,
    timeframe_seconds: int,
    *,
    boundary_ts: int = DEFAULT_PROTECTED_HOLDOUT_START_TS,
    destination_root: Path | None = None,
) -> ResearchDatasetGovernanceManifest:
    """Publish an immutable derived development parquet plus its governance manifest.

    All published candles have ``from_ts < boundary_ts``. A final candle with
    ``to_ts == boundary_ts`` is deliberately retained; maturity-based exclusion
    is enforced downstream by ``max_label_maturity_ts``.
    """
    if timeframe_seconds not in SUPPORTED_EVIDENCE_TIMEFRAMES:
        raise ValueError(
            f"Unsupported Phase 45 timeframe {timeframe_seconds}s; "
            f"expected one of {SUPPORTED_EVIDENCE_TIMEFRAMES}"
        )
    source_root = repo_root.resolve()
    destination = destination_root.resolve() if destination_root is not None else source_root

    candles = load_population_candles(
        source_root,
        timeframe_seconds,
        development_only=True,
        boundary_ts=boundary_ts,
    )
    if not candles:
        raise ValueError(f"No development candles available for {timeframe_seconds}s population")
    assert not any(candle.from_ts >= boundary_ts for candle in candles), (  # noqa: S101
        "protected candle rejected from development population"
    )
    if any(
        candles[index].from_ts >= candles[index + 1].from_ts for index in range(len(candles) - 1)
    ):
        raise ValueError("Development population timestamps are not strictly increasing")

    inventory = build_evidence_inventory(source_root, boundary_ts=boundary_ts)
    population = inventory.populations.get(timeframe_seconds)
    if population is None:
        raise ValueError(f"No verified evidence population for {timeframe_seconds}s")
    content_digest = compute_candle_content_digest(candles)
    if content_digest != population.development_content_digest:
        raise ValueError("Development candle digest does not match verified evidence inventory")
    if len(candles) != population.development_unique_candles:
        raise ValueError("Development candle count does not match verified evidence inventory")

    target_dir = destination / "data/phase45/snapshots" / str(timeframe_seconds)
    target_dir.mkdir(parents=True, exist_ok=True)
    parquet_target = target_dir / "candles.parquet"
    manifest_target = target_dir / "manifest.json"
    if parquet_target.exists() or manifest_target.exists():
        raise FileExistsError(
            f"Immutable Phase 45 snapshot already exists at {target_dir}; refusing to overwrite"
        )

    parquet_created = False
    manifest_created = False
    try:
        with tempfile.TemporaryDirectory(prefix=".phase45_stage_", dir=target_dir) as stage:
            staged = write_candles_parquet(
                candles,
                root_dir=Path(stage),
                source="phase45",
                is_synthetic=False,
                catalog=None,
                on_duplicate="raise",
            )
            with (
                staged.parquet_path.open("rb") as source_handle,
                parquet_target.open("xb") as target_handle,
            ):
                shutil.copyfileobj(source_handle, target_handle)
                target_handle.flush()
                os.fsync(target_handle.fileno())
        parquet_created = True

        written_sha256 = _sha256_file(parquet_target)
        if written_sha256 != staged.sha256_checksum:
            raise ValueError("Published parquet checksum does not match staged write result")
        written_candles = read_candles_parquet(
            parquet_target,
            expected_symbol=inventory.symbol,
            verify_checksum=True,
            expected_checksum=written_sha256,
        )
        if len(written_candles) != len(candles):
            raise ValueError("Published parquet candle count mismatch")
        if compute_candle_content_digest(written_candles) != content_digest:
            raise ValueError("Published parquet canonical content digest mismatch")
        if any(candle.from_ts >= boundary_ts for candle in written_candles):
            raise ValueError("Published parquet contains protected-boundary candles")

        source_hashes = _collect_source_file_hashes(
            source_root,
            source_root / _authoritative_parquet_relative(inventory),
            str(inventory.integrity["authoritative_parquet_sha256"]),
        )
        source_content_digest = str(inventory.integrity["authoritative_content_digest"])
        parent_hashes = {
            "authoritative_parquet_sha256": str(
                inventory.integrity["authoritative_parquet_sha256"]
            ),
            "authoritative_content_digest": source_content_digest,
            "authoritative_manifest_sha256": str(
                inventory.integrity["authoritative_manifest_sha256"]
            ),
            "evidence_full_content_digest": population.content_digest,
            "evidence_development_content_digest": population.development_content_digest,
        }
        manifest = ResearchDatasetGovernanceManifest(
            manifest_id=f"gov_{inventory.symbol.lower()}_{timeframe_seconds}s_{content_digest[:12]}",
            symbol=inventory.symbol,
            timeframe_seconds=timeframe_seconds,
            population_id=(
                f"dev_{inventory.symbol.lower()}_{timeframe_seconds}s_phase45_{content_digest[:12]}"
            ),
            source_snapshot_id=_source_snapshot_id(inventory.symbol, timeframe_seconds),
            source_content_digest=source_content_digest,
            snapshot_id=f"phase45_{inventory.symbol.lower()}_{timeframe_seconds}s_{content_digest[:12]}",
            parquet_path=_repo_relative(destination, parquet_target),
            parquet_sha256=written_sha256,
            content_digest=content_digest,
            from_ts=candles[0].from_ts,
            to_ts=candles[-1].to_ts,
            candle_count=len(candles),
            boundary_ts=boundary_ts,
            provenance_status=PROTECTED_EXPOSURE_CLASS,
            exposure_history=EXPOSURE_HISTORY,
            plan_id=RESEARCH_PLAN_ID,
            parent_hashes=parent_hashes,
            source_file_hashes=source_hashes,
            schema_version=MANIFEST_SCHEMA_VERSION,
            disposition=DISPOSITION,
            execution_authority=EXECUTION_AUTHORITY,
            boundary_policy=dict(BOUNDARY_POLICY),
        )
        manifest = replace(manifest, manifest_sha256=manifest.compute_digest())
        with manifest_target.open("x", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_created = True
        return manifest
    except Exception:
        if manifest_created:
            manifest_target.unlink(missing_ok=True)
        if parquet_created:
            parquet_target.unlink(missing_ok=True)
        raise


def _authoritative_parquet_relative(inventory: Any) -> str:
    """Return the authoritative parquet path recorded in the verified inventory."""
    m1_population = inventory.populations.get(60)
    if m1_population is None:
        raise ValueError("Evidence inventory has no authoritative M1 population")
    for source in m1_population.source_hashes:
        if source.get("role") == "AUTHORITATIVE_SNAPSHOT_PARQUET":
            return str(source["path"])
    raise ValueError("Evidence inventory is missing the authoritative parquet hash binding")


def load_governance_manifest(path: Path) -> ResearchDatasetGovernanceManifest:
    """Load a governance manifest without implicitly trusting its self-digest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Governance manifest must be a JSON object: {path}")
    allowed = {field.name for field in fields(ResearchDatasetGovernanceManifest)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown governance manifest fields: {unknown}")
    values = dict(payload)
    values["exposure_history"] = tuple(values.get("exposure_history", ()))
    values["source_file_hashes"] = tuple(values.get("source_file_hashes", ()))
    values["parent_hashes"] = dict(values.get("parent_hashes", {}))
    values["boundary_policy"] = dict(values.get("boundary_policy", BOUNDARY_POLICY))
    return ResearchDatasetGovernanceManifest(**values)


def _is_within(repo_root: Path, relative: str) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    candidate = (repo_root / relative).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError:
        return None
    return candidate


def verify_governance_manifest(
    manifest: ResearchDatasetGovernanceManifest,
    repo_root: Path,
) -> bool:
    """Fail-closed verification of manifest self-digest, bindings, and published candles.

    Verification checks that every candle ``from_ts`` is strictly before the
    boundary. It deliberately permits ``to_ts == boundary_ts`` and does not
    impose a strict pre-boundary ``to_ts`` rule.
    """
    try:
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            return False
        if (
            manifest.disposition != DISPOSITION
            or manifest.execution_authority != EXECUTION_AUTHORITY
        ):
            return False
        if manifest.manifest_sha256 != manifest.compute_digest():
            return False
        if manifest.timeframe_seconds not in SUPPORTED_EVIDENCE_TIMEFRAMES:
            return False
        if manifest.provenance_status != PROTECTED_EXPOSURE_CLASS:
            return False
        if manifest.plan_id != RESEARCH_PLAN_ID:
            return False
        if manifest.boundary_policy != BOUNDARY_POLICY:
            return False
        joined_history = " ".join(manifest.exposure_history).upper()
        if "PHASE42" not in joined_history or "PHASE43" not in joined_history:
            return False
        for marker in FORBIDDEN_SNAPSHOT_MARKERS:
            if (
                marker in manifest.snapshot_id.upper()
                or marker in manifest.provenance_status.upper()
            ):
                return False
        if manifest.source_snapshot_id != _source_snapshot_id(
            manifest.symbol, manifest.timeframe_seconds
        ):
            return False

        parquet_path = _is_within(repo_root.resolve(), manifest.parquet_path)
        if parquet_path is None or not parquet_path.is_file():
            return False
        if _sha256_file(parquet_path) != manifest.parquet_sha256:
            return False
        candles: list[Candle] = read_candles_parquet(
            parquet_path,
            expected_symbol=manifest.symbol,
            verify_checksum=True,
            expected_checksum=manifest.parquet_sha256,
        )
        if len(candles) != manifest.candle_count:
            return False
        if compute_candle_content_digest(candles) != manifest.content_digest:
            return False
        if not candles or candles[0].from_ts != manifest.from_ts:
            return False
        if candles[-1].to_ts != manifest.to_ts:
            return False
        if any(candle.from_ts >= manifest.boundary_ts for candle in candles):
            return False
        if any(candle.size.value != manifest.timeframe_seconds for candle in candles):
            return False

        for item in manifest.source_file_hashes:
            source_path = _is_within(repo_root.resolve(), str(item.get("path", "")))
            if source_path is None or not source_path.is_file():
                return False
            if _sha256_file(source_path) != str(item.get("sha256", "")).lower():
                return False
            if item.get("role") == "RECEIPT_JSON":
                receipt_id = str(item.get("receipt_id", ""))
                if not receipt_id or not source_path.name.startswith(f"{receipt_id}.json"):
                    return False

        parent = manifest.parent_hashes
        if parent.get("authoritative_content_digest") != manifest.source_content_digest:
            return False
        if parent.get("evidence_development_content_digest") != manifest.content_digest:
            return False
        for key in (
            "authoritative_parquet_sha256",
            "authoritative_content_digest",
            "authoritative_manifest_sha256",
        ):
            value = parent.get(key, "")
            if not isinstance(value, str) or len(value) != 64:
                return False
        return True
    except Exception:  # noqa: BLE001 - verification must fail closed on any error
        return False


def build_preflight_contract(
    manifest: ResearchDatasetGovernanceManifest,
) -> AuthorizedDevelopmentPopulation:
    """Build the complete GENUINE preflight contract bound to a published manifest.

    ``max_allowed_ts`` is the protected boundary for candle ``from_ts`` values.
    The contract intentionally allows a final candle whose ``to_ts`` equals the
    boundary; label maturity is governed by ``max_label_maturity_ts`` downstream.
    """
    return AuthorizedDevelopmentPopulation(
        population_id=manifest.population_id,
        symbol=manifest.symbol,
        expected_content_digest=manifest.content_digest,
        from_ts=manifest.from_ts,
        to_ts=manifest.to_ts,
        timeframe_seconds=manifest.timeframe_seconds,
        snapshot_id=manifest.snapshot_id,
        snapshot_verified=True,
        provenance=manifest.provenance_status,
        plan_id=manifest.plan_id,
        max_allowed_ts=manifest.boundary_ts,
        excluded_protected_intervals=((manifest.boundary_ts, DEFAULT_PROTECTED_HOLDOUT_END_TS),),
        exposure_history=manifest.exposure_history,
        min_candle_count=manifest.candle_count,
        max_candle_count=manifest.candle_count,
        is_development_authorized=True,
        notes=(
            "Phase 45 derived development population. All candle from_ts values are strictly "
            "before boundary_ts; to_ts may equal boundary_ts and maturity invalidation is "
            "enforced by max_label_maturity_ts. Protected historically reused evidence only; "
            "zero execution authority; no demonstrated edge."
        ),
    )
