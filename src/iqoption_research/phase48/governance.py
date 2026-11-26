"""Phase 48 WS1 & WS10 — Exploratory Research Governance & Campaign Contracts.

Implements the four-tier candidate classification pathway, fail-closed owner
authorization gates, and the proposed bounded exploratory campaign contract.

Preserves all Stage C, Phase 40C, confirmatory, and execution_authority=NONE gates.
A dashboard button can never override governance restrictions or forge an
authorized campaign contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from iqoption_research.phase48.constants import (
    ALLOW_OTC_ASSETS,
    ALLOWED_ASSET,
    CANDIDATE_SPECS,
    DISPOSITION,
    EXECUTION_AUTHORITY,
    EXPLORATORY_EVIDENCE_CLASSIFICATION,
    PHASE47_RECONCILED_COMMIT,
    PHASE48_CAMPAIGNS_DIR,
    PROTECTED_BOUNDARY_TS,
    SUPPORTED_SESSION_DURATIONS_MINUTES,
)


class ResearchPathwayTier(StrEnum):
    """Four-tier research candidate classification (Phase 47 WS9 / Phase 48 WS1)."""

    DEVELOPMENT_RESEARCH_CANDIDATE = "DEVELOPMENT_RESEARCH_CANDIDATE"
    EXPLORATORY_FROZEN_RESEARCH_CANDIDATE = "EXPLORATORY_FROZEN_RESEARCH_CANDIDATE"
    SCIENTIFICALLY_QUALIFIED_FROZEN_RESEARCH_CANDIDATE = (
        "SCIENTIFICALLY_QUALIFIED_FROZEN_RESEARCH_CANDIDATE"
    )
    EXECUTION_AUTHORIZED_CANDIDATE = "EXECUTION_AUTHORIZED_CANDIDATE"


class CampaignAuthorizationStatus(StrEnum):
    """Authorization lifecycle for an exploratory prospective campaign."""

    PROPOSED_NOT_AUTHORIZED = "PROPOSED_NOT_AUTHORIZED"
    AUTHORIZED_ACTIVE = "AUTHORIZED_ACTIVE"
    SUSPENDED = "SUSPENDED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class ExploratoryCampaignStage(StrEnum):
    """Fixed staged progression for supervised exploratory pilots."""

    STAGE_A_15M = "STAGE_A_15M"
    STAGE_B_120M = "STAGE_B_120M"
    STAGE_C_240M = "STAGE_C_240M"
    STAGE_D_480M = "STAGE_D_480M"


@dataclass(frozen=True, slots=True)
class ExploratoryStageSpec:
    """Definition of one bounded stage in the exploratory campaign proposal."""

    stage_id: str
    title: str
    session_duration_minutes: int
    target_sessions: int
    purpose: str
    planned_non_overlapping_yield_note: str
    allow_outcome_drain: bool
    max_outcome_drain_minutes: int


@dataclass(frozen=True, slots=True)
class ExploratoryCampaignContract:
    """Owner-reviewable, fail-closed exploratory campaign contract."""

    contract_id: str
    protocol_version: str
    research_pathway_tier: str
    evidence_classification: str
    research_disposition: str
    execution_authority: str
    asset: str
    allow_otc: bool
    candidate_ids: tuple[str, ...]
    candidate_artifact_digests: dict[str, str]
    allowed_account_ref: str
    native_timeframes: tuple[int, ...]
    prediction_horizons: dict[str, int]
    supported_durations_minutes: tuple[int, ...]
    max_session_duration_minutes: int
    allow_outcome_drain: bool
    max_outcome_drain_minutes: int
    stage_plan: tuple[ExploratoryStageSpec, ...]
    fixed_exploratory_sample_goals: dict[str, int]
    stop_conditions: tuple[str, ...]
    provenance_requirements: tuple[str, ...]
    analysis_plan: dict[str, Any]
    operational_and_scientific_limitations: tuple[str, ...]
    source_phase47_commit: str
    protected_boundary_ts: int
    created_at_utc: str
    authorization_status: str = CampaignAuthorizationStatus.PROPOSED_NOT_AUTHORIZED.value
    authorized_stage: str | None = None
    authorized_by: str | None = None
    authorized_at_utc: str | None = None
    authorization_note: str = ""
    authorization_signature_sha256: str = ""
    contract_sha256: str = ""

    def canonical_payload(self) -> dict[str, Any]:
        """Return canonical dictionary of immutable contract terms (excluding auth sig)."""
        return {
            "contract_id": self.contract_id,
            "protocol_version": self.protocol_version,
            "research_pathway_tier": self.research_pathway_tier,
            "evidence_classification": self.evidence_classification,
            "research_disposition": self.research_disposition,
            "execution_authority": self.execution_authority,
            "asset": self.asset,
            "allow_otc": self.allow_otc,
            "candidate_ids": list(self.candidate_ids),
            "candidate_artifact_digests": dict(sorted(self.candidate_artifact_digests.items())),
            "native_timeframes": list(self.native_timeframes),
            "prediction_horizons": dict(sorted(self.prediction_horizons.items())),
            "supported_durations_minutes": list(self.supported_durations_minutes),
            "max_session_duration_minutes": self.max_session_duration_minutes,
            "allow_outcome_drain": self.allow_outcome_drain,
            "max_outcome_drain_minutes": self.max_outcome_drain_minutes,
            "stage_plan": [asdict(s) for s in self.stage_plan],
            "fixed_exploratory_sample_goals": dict(
                sorted(self.fixed_exploratory_sample_goals.items())
            ),
            "stop_conditions": list(self.stop_conditions),
            "provenance_requirements": list(self.provenance_requirements),
            "analysis_plan": self.analysis_plan,
            "operational_and_scientific_limitations": list(
                self.operational_and_scientific_limitations
            ),
            "source_phase47_commit": self.source_phase47_commit,
            "protected_boundary_ts": self.protected_boundary_ts,
            "created_at_utc": self.created_at_utc,
        }

    def compute_contract_sha256(self) -> str:
        """Compute canonical SHA-256 of the contract specification."""
        raw = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def compute_authorization_digest(self) -> str:
        """Compute canonical SHA-256 binding owner authorization to exact contract terms."""
        payload = {
            "contract_sha256": self.contract_sha256 or self.compute_contract_sha256(),
            "authorization_status": self.authorization_status,
            "authorized_stage": self.authorized_stage,
            "authorized_by": self.authorized_by,
            "authorized_at_utc": self.authorized_at_utc,
            "allowed_account_ref": self.allowed_account_ref,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize contract to JSON-compatible dictionary."""
        d = self.canonical_payload()
        d.update({
            "allowed_account_ref": self.allowed_account_ref,
            "authorization_status": self.authorization_status,
            "authorized_stage": self.authorized_stage,
            "authorized_by": self.authorized_by,
            "authorized_at_utc": self.authorized_at_utc,
            "authorization_note": self.authorization_note,
            "authorization_signature_sha256": self.authorization_signature_sha256,
            "contract_sha256": self.contract_sha256 or self.compute_contract_sha256(),
        })
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExploratoryCampaignContract:
        """Deserialize contract from dictionary."""
        stages = tuple(ExploratoryStageSpec(**s) for s in data.get("stage_plan", ()))
        return cls(
            contract_id=str(data["contract_id"]),
            protocol_version=str(data["protocol_version"]),
            research_pathway_tier=str(data["research_pathway_tier"]),
            evidence_classification=str(data["evidence_classification"]),
            research_disposition=str(data["research_disposition"]),
            execution_authority=str(data["execution_authority"]),
            asset=str(data["asset"]),
            allow_otc=bool(data["allow_otc"]),
            candidate_ids=tuple(str(x) for x in data["candidate_ids"]),
            candidate_artifact_digests={
                str(k): str(v) for k, v in data["candidate_artifact_digests"].items()
            },
            allowed_account_ref=str(data["allowed_account_ref"]),
            native_timeframes=tuple(int(x) for x in data["native_timeframes"]),
            prediction_horizons={str(k): int(v) for k, v in data["prediction_horizons"].items()},
            supported_durations_minutes=tuple(int(x) for x in data["supported_durations_minutes"]),
            max_session_duration_minutes=int(data["max_session_duration_minutes"]),
            allow_outcome_drain=bool(data["allow_outcome_drain"]),
            max_outcome_drain_minutes=int(data["max_outcome_drain_minutes"]),
            stage_plan=stages,
            fixed_exploratory_sample_goals={
                str(k): int(v) for k, v in data["fixed_exploratory_sample_goals"].items()
            },
            stop_conditions=tuple(str(x) for x in data["stop_conditions"]),
            provenance_requirements=tuple(str(x) for x in data["provenance_requirements"]),
            analysis_plan=dict(data["analysis_plan"]),
            operational_and_scientific_limitations=tuple(
                str(x) for x in data["operational_and_scientific_limitations"]
            ),
            source_phase47_commit=str(data["source_phase47_commit"]),
            protected_boundary_ts=int(data["protected_boundary_ts"]),
            created_at_utc=str(data["created_at_utc"]),
            authorization_status=str(
                data.get(
                    "authorization_status",
                    CampaignAuthorizationStatus.PROPOSED_NOT_AUTHORIZED.value,
                )
            ),
            authorized_stage=data.get("authorized_stage"),
            authorized_by=data.get("authorized_by"),
            authorized_at_utc=data.get("authorized_at_utc"),
            authorization_note=str(data.get("authorization_note", "")),
            authorization_signature_sha256=str(data.get("authorization_signature_sha256", "")),
            contract_sha256=str(data.get("contract_sha256", "")),
        )


@dataclass(frozen=True, slots=True)
class AuthorizationCheckResult:
    """Result of fail-closed exploratory campaign authorization check."""

    authorized: bool
    status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    contract_id: str | None = None
    contract_sha256: str | None = None


def build_default_proposed_campaign_contract(
    candidate_artifact_digests: dict[str, str] | None = None,
    *,
    allowed_account_ref: str = "OWNER_VAULT_PRACTICE_ACCOUNT_UNASSIGNED",
    created_at_utc: str | None = None,
) -> ExploratoryCampaignContract:
    """Construct the fixed, owner-reviewable Phase 48 exploratory campaign proposal.

    Always returns in PROPOSED_NOT_AUTHORIZED state (inactive, fail-closed).
    """
    digests = dict(candidate_artifact_digests or {})
    for spec in CANDIDATE_SPECS:
        digests.setdefault(spec.frozen_candidate_id, "PENDING_LOCAL_FREEZE_DIGEST")

    stages = (
        ExploratoryStageSpec(
            stage_id=ExploratoryCampaignStage.STAGE_A_15M.value,
            title="Stage A — Supervised Connectivity & Lifecycle Pilot",
            session_duration_minutes=15,
            target_sessions=1,
            purpose=(
                "Verify end-to-end authenticated read-only multi-timeframe candle ingestion, "
                "warmup, commitment hashing, and clean shutdown."
            ),
            planned_non_overlapping_yield_note=(
                "Planning yield only: ~0.35 M1, ~0.15 M5, ~0.12 M15 non-overlapping signals; "
                "most predictions remain pending without outcome drain."
            ),
            allow_outcome_drain=False,
            max_outcome_drain_minutes=0,
        ),
        ExploratoryStageSpec(
            stage_id=ExploratoryCampaignStage.STAGE_B_120M.value,
            title="Stage B — Supervised Multi-Model Maturity Pilot",
            session_duration_minutes=120,
            target_sessions=1,
            purpose=(
                "Observe genuine M1/M5/M15 predictions and full 15m/30m/60m outcome maturity "
                "within a single bounded 2-hour supervised window."
            ),
            planned_non_overlapping_yield_note=(
                "Planning yield only (Phase 47 0.75x): ~2.83 M1, ~1.16 M5, ~0.97 M15 "
                "non-overlapping signals."
            ),
            allow_outcome_drain=True,
            max_outcome_drain_minutes=60,
        ),
        ExploratoryStageSpec(
            stage_id=ExploratoryCampaignStage.STAGE_C_240M.value,
            title="Stage C — Supervised Operational Reliability Session",
            session_duration_minutes=240,
            target_sessions=1,
            purpose=(
                "Validate 4-hour continuous multi-model observation stability, memory bounds, "
                "heartbeat/stale-stream monitoring, and sealed evidence export."
            ),
            planned_non_overlapping_yield_note=(
                "Planning yield only (Phase 47 0.75x): ~5.66 M1, ~2.32 M5, ~1.95 M15 "
                "non-overlapping signals."
            ),
            allow_outcome_drain=True,
            max_outcome_drain_minutes=60,
        ),
        ExploratoryStageSpec(
            stage_id=ExploratoryCampaignStage.STAGE_D_480M.value,
            title="Stage D — Supervised Exploratory Evidence Accumulation",
            session_duration_minutes=480,
            target_sessions=18,
            purpose=(
                "Accumulate preregistered exploratory prospective evidence across supervised 8h "
                "sessions without optional stopping."
            ),
            planned_non_overlapping_yield_note=(
                "Planning yield only (Phase 47 0.75x): ~11.32 M1, ~4.65 M5, ~3.90 M15 "
                "non-overlapping signals per 8h session; actual live rates are unknown."
            ),
            allow_outcome_drain=True,
            max_outcome_drain_minutes=60,
        ),
    )

    contract = ExploratoryCampaignContract(
        contract_id="phase48_exploratory_eurusd_parallel_v1",
        protocol_version="48.1.0",
        research_pathway_tier=ResearchPathwayTier.EXPLORATORY_FROZEN_RESEARCH_CANDIDATE.value,
        evidence_classification=EXPLORATORY_EVIDENCE_CLASSIFICATION,
        research_disposition=DISPOSITION,
        execution_authority=EXECUTION_AUTHORITY,
        asset=ALLOWED_ASSET,
        allow_otc=ALLOW_OTC_ASSETS,
        candidate_ids=tuple(c.frozen_candidate_id for c in CANDIDATE_SPECS),
        candidate_artifact_digests=digests,
        allowed_account_ref=allowed_account_ref,
        native_timeframes=(60, 300, 900),
        prediction_horizons={c.frozen_candidate_id: c.horizon_bars for c in CANDIDATE_SPECS},
        supported_durations_minutes=SUPPORTED_SESSION_DURATIONS_MINUTES,
        max_session_duration_minutes=480,
        allow_outcome_drain=True,
        max_outcome_drain_minutes=60,
        stage_plan=stages,
        fixed_exploratory_sample_goals={
            "p48_exp_cand_a_m1_h15_hgb_v1": 200,
            "p48_exp_cand_b_m5_h6_hgb_v1": 200,
            "p48_exp_cand_c_m15_h4_hgb_v1": 200,
        },
        stop_conditions=(
            "Bounded session timer expiration (plus optional authorized outcome-drain tail).",
            "Operator manual Stop/Cancel or application shutdown.",
            "Shared transport disconnect or stale-stream timeout (no automatic reconnect).",
            "Shared provenance or artifact integrity failure (fail-closed).",
            "Optional stopping based on favorable interim accuracy is strictly prohibited.",
        ),
        provenance_requirements=(
            "Closed native EURUSD regular-market candles only (no OTC, no resampling).",
            "Server-time freshness and timestamp alignment validated on every fetch/stream.",
            "Pre-maturity prediction commitment and SHA-256 digest persisted before maturity.",
            "Append-only session ledger with independent verify_session_integrity() check.",
        ),
        analysis_plan={
            "evidence_classification": EXPLORATORY_EVIDENCE_CLASSIFICATION,
            "dependence_aware_inference": "UTC_DAY_BLOCK_BOOTSTRAP_10000",
            "multiple_comparison_control": "HOLM_BONFERRONI",
            "non_overlapping_reporting": "MANDATORY_SEPARATE_FROM_OVERLAPPING",
            "cross_horizon_aggregation": "PROHIBITED",
            "auto_promotion_allowed": False,
        },
        operational_and_scientific_limitations=(
            "NO_DEMONSTRATED_EDGE; execution_authority=NONE.",
            "Exploratory evidence is EXPLORATORY_PROSPECTIVE_ONLY and cannot auto-qualify.",
            "Historical effective yield estimates (M1=1.415/hr, M5=0.581/hr, M15=0.487/hr) "
            "are planning estimates only; actual prospective rates are unknown.",
            "Small operational pilots carry zero statistical significance.",
            "Candle direction does not equal verified broker trade settlement or payout.",
        ),
        source_phase47_commit=PHASE47_RECONCILED_COMMIT,
        protected_boundary_ts=PROTECTED_BOUNDARY_TS,
        created_at_utc=created_at_utc or datetime.now(UTC).isoformat(),
        authorization_status=CampaignAuthorizationStatus.PROPOSED_NOT_AUTHORIZED.value,
    )
    return replace(contract, contract_sha256=contract.compute_contract_sha256())


def authorize_campaign_contract(
    contract: ExploratoryCampaignContract,
    *,
    authorized_by: str,
    allowed_account_ref: str,
    authorized_stage: str,
    authorization_note: str = "",
    authorized_at_utc: str | None = None,
) -> ExploratoryCampaignContract:
    """Explicitly authorize an exploratory campaign contract on behalf of the owner.

    Intended for explicit CLI/owner review only — never called automatically by UI buttons.
    """
    owner = authorized_by.strip()
    acct = allowed_account_ref.strip()
    if not owner:
        raise ValueError("authorized_by must be non-empty")
    if not acct or acct == "OWNER_VAULT_PRACTICE_ACCOUNT_UNASSIGNED":
        raise ValueError("allowed_account_ref must identify an explicit vault account reference")
    valid_stages = {s.stage_id for s in contract.stage_plan}
    if authorized_stage not in valid_stages:
        raise ValueError(f"authorized_stage must be one of {sorted(valid_stages)}")

    contract_hash = contract.compute_contract_sha256()
    ts = authorized_at_utc or datetime.now(UTC).isoformat()
    updated = replace(
        contract,
        allowed_account_ref=acct,
        authorization_status=CampaignAuthorizationStatus.AUTHORIZED_ACTIVE.value,
        authorized_stage=authorized_stage,
        authorized_by=owner,
        authorized_at_utc=ts,
        authorization_note=authorization_note.strip(),
        contract_sha256=contract_hash,
    )
    sig = updated.compute_authorization_digest()
    return replace(updated, authorization_signature_sha256=sig)


def validate_campaign_authorization(
    contract: ExploratoryCampaignContract | None,
    *,
    requested_candidate_ids: tuple[str, ...],
    requested_duration_minutes: int,
    loaded_artifact_digests: dict[str, str],
    account_ref: str,
    is_genuine_mode: bool,
) -> AuthorizationCheckResult:
    """Fail-closed authorization gate for exploratory observation sessions."""
    if not is_genuine_mode:
        return AuthorizationCheckResult(
            authorized=True,
            status="SYNTHETIC_MODE_PERMITTED",
            reasons=("Synthetic demonstration/acceptance mode; zero broker connection.",),
        )

    reasons: list[str] = []
    if contract is None:
        return AuthorizationCheckResult(
            authorized=False,
            status="BLOCKED_NO_CAMPAIGN_CONTRACT",
            reasons=("No exploratory campaign contract is loaded or authorized.",),
        )

    if contract.execution_authority != EXECUTION_AUTHORITY:
        reasons.append("execution_authority must remain NONE.")
    if contract.evidence_classification != EXPLORATORY_EVIDENCE_CLASSIFICATION:
        reasons.append("evidence_classification must remain EXPLORATORY_PROSPECTIVE_ONLY.")
    if contract.asset != ALLOWED_ASSET or contract.allow_otc:
        reasons.append("Only EURUSD regular market (allow_otc=False) is permitted.")

    expected_contract_hash = contract.compute_contract_sha256()
    if contract.contract_sha256 != expected_contract_hash:
        reasons.append("Campaign contract SHA-256 mismatch (tampered or unsealed contract).")

    if contract.authorization_status != CampaignAuthorizationStatus.AUTHORIZED_ACTIVE.value:
        reasons.append(
            f"Campaign authorization_status is {contract.authorization_status} "
            "(owner authorization required)."
        )
    if not contract.authorized_by or not contract.authorized_at_utc:
        reasons.append("Missing explicit authorized_by or authorized_at_utc owner sign-off.")
    if (
        not contract.authorization_signature_sha256
        or contract.authorization_signature_sha256 != contract.compute_authorization_digest()
    ):
        reasons.append("Missing or invalid owner authorization_signature_sha256.")

    if not requested_candidate_ids:
        reasons.append("At least one exploratory candidate must be selected.")
    for cid in requested_candidate_ids:
        if cid not in contract.candidate_ids:
            reasons.append(f"Candidate {cid} is not in contract.candidate_ids.")
        expected_digest = contract.candidate_artifact_digests.get(cid)
        actual_digest = loaded_artifact_digests.get(cid)
        if not expected_digest or not actual_digest or expected_digest != actual_digest:
            reasons.append(f"Artifact digest mismatch or missing digest for {cid}.")

    if requested_duration_minutes not in SUPPORTED_SESSION_DURATIONS_MINUTES:
        reasons.append(f"Unsupported session duration: {requested_duration_minutes}m.")
    if requested_duration_minutes > contract.max_session_duration_minutes:
        reasons.append(
            f"Requested duration {requested_duration_minutes}m exceeds contract max "
            f"{contract.max_session_duration_minutes}m."
        )

    if not account_ref or account_ref != contract.allowed_account_ref:
        reasons.append("Selected vault account does not match contract allowed_account_ref.")

    if reasons:
        return AuthorizationCheckResult(
            authorized=False,
            status="BLOCKED_UNAUTHORIZED_GENUINE_CAMPAIGN",
            reasons=tuple(reasons),
            contract_id=contract.contract_id,
            contract_sha256=contract.contract_sha256,
        )

    return AuthorizationCheckResult(
        authorized=True,
        status="AUTHORIZED_FOR_SUPERVISED_EXPLORATORY_OBSERVATION",
        reasons=(),
        contract_id=contract.contract_id,
        contract_sha256=contract.contract_sha256,
    )


def save_campaign_contract(contract: ExploratoryCampaignContract, path: Path) -> Path:
    """Persist campaign contract JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_campaign_contract(path: Path) -> ExploratoryCampaignContract:
    """Load campaign contract JSON from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExploratoryCampaignContract.from_dict(data)


def default_campaign_contract_path(repo_root: Path) -> Path:
    """Return default local path for the Phase 48 exploratory campaign contract."""
    return Path(repo_root).resolve() / PHASE48_CAMPAIGNS_DIR / "phase48_exploratory_contract.json"
