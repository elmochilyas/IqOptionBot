"""Offline unit tests for Phase 45 derived-population governance.

All evidence is generated as small synthetic fixtures under ``tmp_path``. No
repo data directory, broker, or network resource is consulted.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from iqoption_research.market_data.parquet_storage import read_candles_parquet
from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_END_TS,
    PreflightEnforcement,
    compute_candle_content_digest,
    validate_development_population_preflight,
)
from iqoption_research.phase45.governance import (
    ResearchDatasetGovernanceManifest,
    build_preflight_contract,
    load_governance_manifest,
    publish_development_population,
    verify_governance_manifest,
)
from tests.unit.market_data.test_evidence import (
    BOUNDARY_TS,
    _build_boundary_repo,
)


@pytest.fixture
def published_repo(
    tmp_path: Path,
) -> tuple[Path, ResearchDatasetGovernanceManifest]:
    repo_root = _build_boundary_repo(tmp_path)
    manifest = publish_development_population(repo_root, 300)
    return repo_root, manifest


def test_publish_writes_verified_immutable_snapshot(
    published_repo: tuple[Path, ResearchDatasetGovernanceManifest],
) -> None:
    repo_root, manifest = published_repo

    parquet_path = repo_root / "data/phase45/snapshots/300/candles.parquet"
    manifest_path = repo_root / "data/phase45/snapshots/300/manifest.json"
    assert parquet_path.is_file()
    assert manifest_path.is_file()
    assert manifest.parquet_path == "data/phase45/snapshots/300/candles.parquet"
    assert manifest.manifest_sha256 == manifest.compute_digest()
    assert manifest.to_ts == BOUNDARY_TS

    assert manifest.boundary_policy["development_membership"] == "from_ts < boundary_ts"
    assert manifest.boundary_policy["last_allowed_decision_start"] == "from_ts"
    assert manifest.boundary_policy["to_ts_equals_boundary_allowed"] is True
    assert manifest.boundary_policy["to_ts_strictly_pre_boundary_required"] is False
    assert manifest.boundary_policy["label_maturity_invalidation"] == "max_label_maturity_ts"

    loaded = load_governance_manifest(manifest_path)
    assert loaded == manifest
    assert loaded.compute_digest() == manifest.manifest_sha256
    assert verify_governance_manifest(loaded, repo_root) is True

    roles = {item["role"] for item in manifest.source_file_hashes}
    assert {"AUTHORITATIVE_SNAPSHOT_PARQUET", "RECEIPT_JSON", "RECEIPT_PARQUET"} <= roles
    assert manifest.parent_hashes["evidence_development_content_digest"] == manifest.content_digest

    candles = read_candles_parquet(
        parquet_path,
        expected_symbol=manifest.symbol,
        verify_checksum=True,
        expected_checksum=manifest.parquet_sha256,
    )
    assert len(candles) == manifest.candle_count == 2
    assert all(candle.from_ts < BOUNDARY_TS for candle in candles)
    assert candles[-1].to_ts == BOUNDARY_TS
    assert compute_candle_content_digest(candles) == manifest.content_digest

    contract = build_preflight_contract(manifest)
    assert contract.snapshot_verified is True
    assert contract.is_development_authorized is True
    assert contract.max_allowed_ts == BOUNDARY_TS
    assert contract.excluded_protected_intervals == (
        (BOUNDARY_TS, DEFAULT_PROTECTED_HOLDOUT_END_TS),
    )
    assert contract.provenance == "PROTECTED_HISTORICALLY_REUSED"
    assert contract.plan_id == "phase45_multi_timeframe_research"
    assert contract.min_candle_count == manifest.candle_count
    assert contract.max_candle_count == manifest.candle_count
    assert "PHASE42" in " ".join(contract.exposure_history)
    assert "PHASE43" in " ".join(contract.exposure_history)
    assert "TEST_ONLY" not in contract.notes.upper()
    assert "max_label_maturity_ts" in contract.notes

    report = validate_development_population_preflight(
        candles,
        contract=contract,
        verified_snapshot_id=manifest.snapshot_id,
        enforcement=PreflightEnforcement.GENUINE,
    )
    assert report.is_valid is True
    assert report.status == "AUTHORIZED_DEVELOPMENT_POPULATION"
    assert report.snapshot_verified is True
    assert report.computed_content_digest == manifest.content_digest

    with pytest.raises(FileExistsError):
        publish_development_population(repo_root, 300)


def test_manifest_self_digest_and_tamper_detection(
    published_repo: tuple[Path, ResearchDatasetGovernanceManifest],
) -> None:
    repo_root, _published = published_repo
    manifest_path = repo_root / "data/phase45/snapshots/300/manifest.json"
    manifest = load_governance_manifest(manifest_path)

    changed_sha = replace(manifest, manifest_sha256="0" * 64)
    assert changed_sha.compute_digest() == manifest.compute_digest()
    assert verify_governance_manifest(changed_sha, repo_root) is False

    changed_count = replace(manifest, candle_count=manifest.candle_count + 1)
    assert verify_governance_manifest(changed_count, repo_root) is False

    changed_policy = replace(
        manifest,
        boundary_policy={
            **manifest.boundary_policy,
            "to_ts_strictly_pre_boundary_required": True,
        },
    )
    assert verify_governance_manifest(changed_policy, repo_root) is False

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["content_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tampered = load_governance_manifest(manifest_path)
    assert verify_governance_manifest(tampered, repo_root) is False


def test_manifest_verification_detects_modified_parquet(
    published_repo: tuple[Path, ResearchDatasetGovernanceManifest],
) -> None:
    repo_root, _manifest = published_repo
    manifest = load_governance_manifest(repo_root / "data/phase45/snapshots/300/manifest.json")
    parquet_path = repo_root / manifest.parquet_path
    payload = bytearray(parquet_path.read_bytes())
    payload.append(0)
    parquet_path.write_bytes(bytes(payload))

    assert verify_governance_manifest(manifest, repo_root) is False


def test_load_manifest_rejects_unknown_fields(
    published_repo: tuple[Path, ResearchDatasetGovernanceManifest],
) -> None:
    repo_root, _manifest = published_repo
    manifest_path = repo_root / "data/phase45/snapshots/300/manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["unauthorized_field"] = True
    manifest_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown governance manifest fields"):
        load_governance_manifest(manifest_path)


def test_unsupported_timeframe_is_rejected(tmp_path: Path) -> None:
    repo_root = _build_boundary_repo(tmp_path)
    with pytest.raises(ValueError, match="Unsupported Phase 45 timeframe"):
        publish_development_population(repo_root, 3600)
