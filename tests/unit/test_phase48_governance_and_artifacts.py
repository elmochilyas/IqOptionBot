"""Phase 48 WS1, WS2, WS3 & WS10 — Governance, Campaign Contract & Artifact Integrity Tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pytest

from iqoption_research.ml.calibration.calibrator import CalibrationMethod, ProbabilityCalibrator
from iqoption_research.ml.walk_forward import create_model_instance
from iqoption_research.phase47.features import Phase47FeaturePipeline
from iqoption_research.phase48.artifacts import (
    COMPONENT_FILES,
    ArtifactIntegrityError,
    FrozenExploratoryManifest,
    compute_bundle_digest,
    load_verified_exploratory_bundle,
    sha256_file,
    verify_exploratory_bundle,
)
from iqoption_research.phase48.constants import CANDIDATE_SPECS, DISPOSITION, EXECUTION_AUTHORITY
from iqoption_research.phase48.governance import (
    CampaignAuthorizationStatus,
    CampaignUsageStore,
    ResearchPathwayTier,
    authorize_campaign_contract,
    build_default_proposed_campaign_contract,
    load_campaign_contract,
    revoke_campaign_contract,
    save_campaign_contract,
    validate_campaign_authorization,
)
from iqoption_research.research.features.scalers import StandardScaler


def create_synthetic_frozen_bundle(
    models_dir: Path, slot_idx: int = 0
) -> FrozenExploratoryManifest:
    """Create a real, cryptographically valid exploratory bundle in tmp_path for CI/unit tests."""
    spec = CANDIDATE_SPECS[slot_idx]
    bdir = models_dir / spec.frozen_candidate_id
    bdir.mkdir(parents=True, exist_ok=True)

    pipe = Phase47FeaturePipeline(
        timeframe_seconds=spec.timeframe_seconds,
        preset=spec.feature_preset,
    )
    rng = np.random.default_rng(spec.trial_seed)
    X_tr = rng.normal(size=(200, len(pipe.feature_names)))
    y_tr = (rng.random(200) > 0.5).astype(np.int64)
    X_cal = rng.normal(size=(100, len(pipe.feature_names)))
    y_cal = (rng.random(100) > 0.5).astype(np.int64)

    scaler = StandardScaler().fit(X_tr)
    model = create_model_instance(
        spec.model_family,
        {**spec.hyperparameters, "random_state": spec.model_random_state},
    )
    model.fit(
        np.asarray(scaler.transform(X_tr), dtype=np.float64),
        y_tr,
        feature_names=pipe.feature_names,
    )
    p_cal_raw = np.asarray(
        model.predict_proba(np.asarray(scaler.transform(X_cal), dtype=np.float64)),
        dtype=np.float64,
    )
    cal = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
    cal.fit(p_cal_raw, y_cal)

    joblib.dump(model, bdir / "model.joblib")
    joblib.dump(scaler, bdir / "scaler.joblib")
    joblib.dump(cal, bdir / "calibrator.joblib")

    comp_hashes = {name: sha256_file(bdir / name) for name in COMPONENT_FILES}
    bundle_hash = compute_bundle_digest(bdir)
    manifest = FrozenExploratoryManifest(
        frozen_candidate_id=spec.frozen_candidate_id,
        candidate_slot=spec.candidate_slot,
        trial_id=spec.trial_id,
        candidate_classification=ResearchPathwayTier.EXPLORATORY_FROZEN_RESEARCH_CANDIDATE.value,
        research_disposition=DISPOSITION,
        execution_authority=EXECUTION_AUTHORITY,
        native_timeframe_seconds=spec.timeframe_seconds,
        timeframe_label=spec.timeframe_label,
        horizon_bars=spec.horizon_bars,
        prediction_minutes=spec.prediction_minutes,
        model_family=spec.model_family,
        feature_preset=spec.feature_preset,
        feature_implementation_identity=f"Phase47FeaturePipeline:{spec.feature_preset}",
        ordered_feature_names=pipe.feature_names,
        hyperparameters=dict(spec.hyperparameters),
        calibration_method=spec.calibration_method,
        calibration_status="CALIBRATED_ISOTONIC",
        frozen_confidence_threshold=0.52,
        operational_target_coverage=0.10,
        training_fitted_majority_dir="DOWN",
        training_fitted_up_rate=0.49,
        population_manifest={"manifest_id": "synth_test", "total_candles": 300},
        split_manifest={"strategy": "CHRONOLOGICAL_PURGED_TRAIN_CAL"},
        training_metrics={"train_unconditional_accuracy": 0.55},
        calibration_metrics={"cal_selective_accuracy": 0.53},
        reproducibility_check={
            "verified": True,
            "max_probability_diff": 0.0,
            "threshold_diff": 0.0,
        },
        dependency_versions={"python": "3.12"},
        git_commit="test",
        trial_seed=spec.trial_seed,
        model_random_state=spec.model_random_state,
        component_sha256=comp_hashes,
        bundle_sha256=bundle_hash,
        created_at_utc=datetime.now(UTC).isoformat(),
        research_limitations=spec.research_limitations,
    )
    sealed = FrozenExploratoryManifest(**{
        **asdict(manifest),
        "manifest_sha256": manifest.compute_manifest_sha256(),
    })
    (bdir / "manifest.json").write_text(
        json.dumps(sealed.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sealed


def test_proposed_campaign_contract_is_inactive_and_fail_closed(tmp_path: Path) -> None:
    m1 = create_synthetic_frozen_bundle(tmp_path, 0)
    contract = build_default_proposed_campaign_contract({m1.frozen_candidate_id: m1.bundle_sha256})
    assert (
        contract.authorization_status == CampaignAuthorizationStatus.PROPOSED_NOT_AUTHORIZED.value
    )
    assert contract.execution_authority == "NONE"
    assert contract.research_disposition == "NO_DEMONSTRATED_EDGE"
    assert contract.allow_otc is False

    # Genuine mode must be blocked when contract is not authorized
    res = validate_campaign_authorization(
        contract,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=15,
        loaded_artifact_digests={m1.frozen_candidate_id: m1.bundle_sha256},
        account_ref="vault_acc_1",
        is_genuine_mode=True,
    )
    assert res.authorized is False
    assert res.status == "BLOCKED_UNAUTHORIZED_GENUINE_CAMPAIGN"

    # Synthetic mode is permitted without genuine campaign authorization
    synth_res = validate_campaign_authorization(
        None,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=15,
        loaded_artifact_digests={m1.frozen_candidate_id: m1.bundle_sha256},
        account_ref="",
        is_genuine_mode=False,
    )
    assert synth_res.authorized is True


def test_explicit_owner_authorization_and_tamper_rejection(tmp_path: Path) -> None:
    m1 = create_synthetic_frozen_bundle(tmp_path, 0)
    proposed = build_default_proposed_campaign_contract({m1.frozen_candidate_id: m1.bundle_sha256})
    authorized = authorize_campaign_contract(
        proposed,
        authorized_by="owner@example.local",
        allowed_account_ref="vault_practice_eur",
        authorized_stage="STAGE_A_15M",
        authorization_note="Owner sign-off for Stage A pilot",
    )
    cpath = tmp_path / "contract.json"
    save_campaign_contract(authorized, cpath)
    loaded = load_campaign_contract(cpath)

    ok_res = validate_campaign_authorization(
        loaded,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=15,
        loaded_artifact_digests={m1.frozen_candidate_id: m1.bundle_sha256},
        account_ref="vault_practice_eur",
        is_genuine_mode=True,
    )
    assert ok_res.authorized is True

    # Wrong account must fail closed
    bad_acct = validate_campaign_authorization(
        loaded,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=15,
        loaded_artifact_digests={m1.frozen_candidate_id: m1.bundle_sha256},
        account_ref="wrong_account",
        is_genuine_mode=True,
    )
    assert bad_acct.authorized is False

    # Digest mismatch must fail closed
    bad_digest = validate_campaign_authorization(
        loaded,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=15,
        loaded_artifact_digests={m1.frozen_candidate_id: "0" * 64},
        account_ref="vault_practice_eur",
        is_genuine_mode=True,
    )
    assert bad_digest.authorized is False


def test_artifact_integrity_verification_and_tamper_rejection(tmp_path: Path) -> None:
    m1 = create_synthetic_frozen_bundle(tmp_path, 0)
    bdir = tmp_path / m1.frozen_candidate_id
    verified = verify_exploratory_bundle(bdir)
    assert verified.bundle_sha256 == m1.bundle_sha256
    loaded = load_verified_exploratory_bundle(bdir)
    assert loaded.manifest.frozen_candidate_id == m1.frozen_candidate_id

    # Tamper with model bytes -> must reject before loading
    (bdir / "model.joblib").write_bytes(b"corrupted_bytes")
    with pytest.raises(ArtifactIntegrityError):
        load_verified_exploratory_bundle(bdir)


def test_stage_a_enforces_duration_drain_expiration_revocation_and_usage(tmp_path: Path) -> None:
    m1 = create_synthetic_frozen_bundle(tmp_path, 0)
    proposed = build_default_proposed_campaign_contract({m1.frozen_candidate_id: m1.bundle_sha256})
    authorized = authorize_campaign_contract(
        proposed,
        authorized_by="owner@example.local",
        allowed_account_ref="vault_practice_eur",
        authorized_stage="STAGE_A_15M",
        hours_valid=2,
        hmac_key=b"test-owner-secret",
    )

    # Stage A rejects longer durations (120m)
    bad_dur = validate_campaign_authorization(
        authorized,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=120,
        loaded_artifact_digests={m1.frozen_candidate_id: m1.bundle_sha256},
        account_ref="vault_practice_eur",
        is_genuine_mode=True,
        hmac_key=b"test-owner-secret",
    )
    assert bad_dur.authorized is False

    # Stage A rejects outcome drain
    bad_drain = validate_campaign_authorization(
        authorized,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=15,
        loaded_artifact_digests={m1.frozen_candidate_id: m1.bundle_sha256},
        account_ref="vault_practice_eur",
        is_genuine_mode=True,
        requested_allow_outcome_drain=True,
        requested_drain_minutes=15,
        hmac_key=b"test-owner-secret",
    )
    assert bad_drain.authorized is False

    # Wrong HMAC key rejected
    bad_key = validate_campaign_authorization(
        authorized,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=15,
        loaded_artifact_digests={m1.frozen_candidate_id: m1.bundle_sha256},
        account_ref="vault_practice_eur",
        is_genuine_mode=True,
        hmac_key=b"wrong-key",
    )
    assert bad_key.authorized is False

    # Expired authorization rejected
    expired = validate_campaign_authorization(
        authorized,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=15,
        loaded_artifact_digests={m1.frozen_candidate_id: m1.bundle_sha256},
        account_ref="vault_practice_eur",
        is_genuine_mode=True,
        now_utc=datetime.now(UTC) + timedelta(hours=5),
        hmac_key=b"test-owner-secret",
    )
    assert expired.authorized is False

    # Durable atomic session consumption accounting
    usage = CampaignUsageStore(tmp_path / "usage.json")
    usage.reserve_session(authorized, session_id="ses_1", stage_id="STAGE_A_15M", max_sessions=1)
    exhausted = validate_campaign_authorization(
        authorized,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=15,
        loaded_artifact_digests={m1.frozen_candidate_id: m1.bundle_sha256},
        account_ref="vault_practice_eur",
        is_genuine_mode=True,
        usage_store=usage,
        session_id="ses_2",
        hmac_key=b"test-owner-secret",
    )
    assert exhausted.authorized is False

    # Revocation rejected
    revoked = revoke_campaign_contract(authorized, revoked_by="owner", reason="test")
    rev_res = validate_campaign_authorization(
        revoked,
        requested_candidate_ids=(m1.frozen_candidate_id,),
        requested_duration_minutes=15,
        loaded_artifact_digests={m1.frozen_candidate_id: m1.bundle_sha256},
        account_ref="vault_practice_eur",
        is_genuine_mode=True,
        hmac_key=b"test-owner-secret",
    )
    assert rev_res.authorized is False
