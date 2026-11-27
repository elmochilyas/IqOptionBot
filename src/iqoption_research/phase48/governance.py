"""Phase 48 WS1 & WS10 — Exploratory Research Governance & Campaign Contracts.

Implements the four-tier candidate classification pathway, fail-closed owner
authorization gates, and the proposed bounded exploratory campaign contract.

Preserves all Stage C, Phase 40C, confirmatory, and execution_authority=NONE gates.
A dashboard button can never override governance restrictions or forge an
authorized campaign contract.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
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


class ObservationAuthorizationMode(StrEnum):
    """Distinct market-data vs practice-account governance modes."""

    PRACTICE_VERIFIED_REQUIRED = "PRACTICE_VERIFIED_REQUIRED"
    WALLET_INDEPENDENT_SPOT_READ_ONLY = "WALLET_INDEPENDENT_SPOT_READ_ONLY"


class SessionLifecycleState(StrEnum):
    """Durable lifecycle states for supervised Phase 48 observation sessions."""

    RESERVED = "RESERVED"
    CONNECTING = "CONNECTING"
    FAILED_STARTUP = "FAILED_STARTUP"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


_VALID_LIFECYCLE_TRANSITIONS: dict[SessionLifecycleState, frozenset[SessionLifecycleState]] = {
    SessionLifecycleState.RESERVED: frozenset({
        SessionLifecycleState.CONNECTING,
        SessionLifecycleState.FAILED_STARTUP,
        SessionLifecycleState.CANCELLED,
    }),
    SessionLifecycleState.CONNECTING: frozenset({
        SessionLifecycleState.FAILED_STARTUP,
        SessionLifecycleState.RUNNING,
        SessionLifecycleState.INTERRUPTED,
        SessionLifecycleState.CANCELLED,
    }),
    SessionLifecycleState.RUNNING: frozenset({
        SessionLifecycleState.INTERRUPTED,
        SessionLifecycleState.CANCELLED,
        SessionLifecycleState.COMPLETED,
    }),
    SessionLifecycleState.FAILED_STARTUP: frozenset(),
    SessionLifecycleState.INTERRUPTED: frozenset(),
    SessionLifecycleState.CANCELLED: frozenset(),
    SessionLifecycleState.COMPLETED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class StageReplacementAuthorization:
    """Explicit owner sign-off for one bounded Stage A replacement attempt."""

    replacement_grant_id: str
    stage_id: str
    previous_failed_session_id: str
    previous_failed_status: str
    previous_failed_reason_summary: str
    authorized_by: str
    authorized_at_utc: str
    expires_at_utc: str
    max_attempts: int = 1
    session_duration_minutes: int = 15
    allow_outcome_drain: bool = False
    max_outcome_drain_minutes: int = 0
    review_note: str = ""
    authorization_mode: str = ""


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
    expires_at_utc: str | None = None
    require_practice_account: bool = True
    authorization_mode: str = ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED.value
    authorization_mechanism: str = "LOCAL_UNKEYED_INTEGRITY_DIGEST_V1"
    authorization_note: str = ""
    authorization_signature_sha256: str = ""
    contract_sha256: str = ""
    replacement_authorization: StageReplacementAuthorization | None = None

    def canonical_payload(self) -> dict[str, Any]:
        """Return canonical dictionary of immutable contract terms (excluding auth sig)."""
        payload: dict[str, Any] = {
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
        if (
            self.authorization_mode != ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED.value
            or self.protocol_version != "48.1.0"
        ):
            payload["authorization_mode"] = self.authorization_mode
        return payload

    def compute_contract_sha256(self) -> str:
        """Compute canonical SHA-256 of the contract specification."""
        raw = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def compute_authorization_digest(self, *, hmac_key: bytes | None = None) -> str:
        """Compute canonical integrity digest (or HMAC-SHA256 when keyed) binding owner sign-off.

        Trust boundary: When hmac_key is None, this is an unkeyed SHA-256 integrity
        digest that detects accidental corruption or UI tampering, not an external
        cryptographic proof of identity. When hmac_key is provided, it computes
        HMAC-SHA256 using that local owner secret.
        """
        payload: dict[str, Any] = {
            "contract_sha256": self.contract_sha256 or self.compute_contract_sha256(),
            "authorization_status": self.authorization_status,
            "authorized_stage": self.authorized_stage,
            "authorized_by": self.authorized_by,
            "authorized_at_utc": self.authorized_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "require_practice_account": self.require_practice_account,
            "authorization_mechanism": self.authorization_mechanism,
            "allowed_account_ref": self.allowed_account_ref,
        }
        if (
            self.authorization_mode != ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED.value
            or self.protocol_version != "48.1.0"
        ):
            payload["authorization_mode"] = self.authorization_mode
        if self.replacement_authorization is not None:
            payload["replacement_authorization"] = asdict(self.replacement_authorization)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hmac_key is not None:
            return hmac.new(hmac_key, raw, hashlib.sha256).hexdigest()
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize contract to JSON-compatible dictionary."""
        d = self.canonical_payload()
        d.update({
            "allowed_account_ref": self.allowed_account_ref,
            "authorization_mode": self.authorization_mode,
            "authorization_status": self.authorization_status,
            "authorized_stage": self.authorized_stage,
            "authorized_by": self.authorized_by,
            "authorized_at_utc": self.authorized_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "require_practice_account": self.require_practice_account,
            "authorization_mechanism": self.authorization_mechanism,
            "authorization_note": self.authorization_note,
            "authorization_signature_sha256": self.authorization_signature_sha256,
            "contract_sha256": self.contract_sha256 or self.compute_contract_sha256(),
        })
        if self.replacement_authorization is not None:
            d["replacement_authorization"] = asdict(self.replacement_authorization)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExploratoryCampaignContract:
        """Deserialize contract from dictionary."""
        stages = tuple(ExploratoryStageSpec(**s) for s in data.get("stage_plan", ()))
        repl_raw = data.get("replacement_authorization")
        repl = (
            StageReplacementAuthorization(
                replacement_grant_id=str(repl_raw["replacement_grant_id"]),
                stage_id=str(repl_raw["stage_id"]),
                previous_failed_session_id=str(repl_raw["previous_failed_session_id"]),
                previous_failed_status=str(repl_raw["previous_failed_status"]),
                previous_failed_reason_summary=str(
                    repl_raw.get("previous_failed_reason_summary", "")
                ),
                authorized_by=str(repl_raw["authorized_by"]),
                authorized_at_utc=str(repl_raw["authorized_at_utc"]),
                expires_at_utc=str(repl_raw["expires_at_utc"]),
                max_attempts=int(repl_raw.get("max_attempts", 1)),
                session_duration_minutes=int(repl_raw.get("session_duration_minutes", 15)),
                allow_outcome_drain=bool(repl_raw.get("allow_outcome_drain", False)),
                max_outcome_drain_minutes=int(repl_raw.get("max_outcome_drain_minutes", 0)),
                review_note=str(repl_raw.get("review_note", "")),
                authorization_mode=str(repl_raw.get("authorization_mode", "")),
            )
            if isinstance(repl_raw, dict)
            else None
        )
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
            expires_at_utc=data.get("expires_at_utc"),
            require_practice_account=bool(data.get("require_practice_account", True)),
            authorization_mode=str(
                data.get(
                    "authorization_mode",
                    ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED.value,
                )
            ),
            authorization_mechanism=str(
                data.get("authorization_mechanism", "LOCAL_UNKEYED_INTEGRITY_DIGEST_V1")
            ),
            authorization_note=str(data.get("authorization_note", "")),
            authorization_signature_sha256=str(data.get("authorization_signature_sha256", "")),
            contract_sha256=str(data.get("contract_sha256", "")),
            replacement_authorization=repl,
        )


@dataclass(frozen=True, slots=True)
class AuthorizationCheckResult:
    """Result of fail-closed exploratory campaign authorization check."""

    authorized: bool
    status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    contract_id: str | None = None
    contract_sha256: str | None = None
    exhausted_stage_budget: bool = False
    authorization_mode: str = ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED.value


def build_default_proposed_campaign_contract(
    candidate_artifact_digests: dict[str, str] | None = None,
    *,
    allowed_account_ref: str = "OWNER_VAULT_PRACTICE_ACCOUNT_UNASSIGNED",
    created_at_utc: str | None = None,
    authorization_mode: ObservationAuthorizationMode
    | str = ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED,
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

    mode = ObservationAuthorizationMode(str(authorization_mode))
    is_wallet_indep = mode == ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY
    provenance_reqs = [
        "Closed native EURUSD regular-market candles only (no OTC, no resampling).",
        "Server-time freshness and timestamp alignment validated on every fetch/stream.",
        "Pre-maturity prediction commitment and SHA-256 digest persisted before maturity.",
        "Append-only session ledger with independent verify_session_integrity() check.",
    ]
    limitations = [
        "NO_DEMONSTRATED_EDGE; execution_authority=NONE.",
        "Exploratory evidence is EXPLORATORY_PROSPECTIVE_ONLY and cannot auto-qualify.",
        "Historical effective yield estimates (M1=1.415/hr, M5=0.581/hr, M15=0.487/hr) "
        "are planning estimates only; actual prospective rates are unknown.",
        "Small operational pilots carry zero statistical significance.",
        "Candle direction does not equal verified broker trade settlement or payout.",
    ]
    if is_wallet_indep:
        provenance_reqs.append(
            "WALLET_INDEPENDENT_SPOT_READ_ONLY outbound wire allowlist enforced "
            "(EURUSD M1/M5/M15 market data only; zero order or wallet-mutation frames)."
        )
        limitations.append(
            "Live IQ Option WebSocket protocol cannot prove active practice wallet selection; "
            "observed wallet status is recorded honestly without practice verification bypass."
        )

    contract = ExploratoryCampaignContract(
        contract_id="phase48_exploratory_eurusd_parallel_v1",
        protocol_version="49.2.0" if is_wallet_indep else "48.1.0",
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
        provenance_requirements=tuple(provenance_reqs),
        analysis_plan={
            "evidence_classification": EXPLORATORY_EVIDENCE_CLASSIFICATION,
            "dependence_aware_inference": "UTC_DAY_BLOCK_BOOTSTRAP_10000",
            "multiple_comparison_control": "HOLM_BONFERRONI",
            "non_overlapping_reporting": "MANDATORY_SEPARATE_FROM_OVERLAPPING",
            "cross_horizon_aggregation": "PROHIBITED",
            "auto_promotion_allowed": False,
        },
        operational_and_scientific_limitations=tuple(limitations),
        source_phase47_commit=PHASE47_RECONCILED_COMMIT,
        protected_boundary_ts=PROTECTED_BOUNDARY_TS,
        created_at_utc=created_at_utc or datetime.now(UTC).isoformat(),
        authorization_status=CampaignAuthorizationStatus.PROPOSED_NOT_AUTHORIZED.value,
        require_practice_account=not is_wallet_indep,
        authorization_mode=mode.value,
    )
    return replace(contract, contract_sha256=contract.compute_contract_sha256())


_USAGE_LOCKS_GUARD = threading.Lock()
_USAGE_LOCKS: dict[str, threading.RLock] = {}


def _get_usage_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _USAGE_LOCKS_GUARD:
        lock = _USAGE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _USAGE_LOCKS[key] = lock
        return lock


class CampaignUsageStore:
    """Durable, atomic session consumption ledger preventing duplicate/excess runs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()

    @contextmanager
    def _locked(self, timeout_seconds: float = 10.0) -> Generator[None, None, None]:
        """Acquire both in-process thread lock and cross-process exclusive lockfile."""
        tlock = _get_usage_lock(self.path)
        with tlock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_name(f"{self.path.name}.lock")
            deadline = time.monotonic() + timeout_seconds
            fd: int | None = None
            while fd is None:
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                except FileExistsError:
                    # Recover from crashed lock holders older than 60s
                    try:
                        age = time.time() - lock_path.stat().st_mtime
                        if age > 60.0:
                            lock_path.unlink(missing_ok=True)
                            continue
                    except OSError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for usage store lock: {lock_path}"
                        ) from None
                    time.sleep(0.01)
            try:
                yield
            finally:
                try:
                    os.close(fd)
                finally:
                    lock_path.unlink(missing_ok=True)

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"consumed_sessions": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"consumed_sessions": []}
        return data

    def _write_atomic(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def list_consumed(self, contract_id: str, stage_id: str) -> list[dict[str, Any]]:
        with self._locked():
            data = self._read()
            return [
                dict(s)
                for s in data.get("consumed_sessions", [])
                if s.get("contract_id") == contract_id and s.get("stage_id") == stage_id
            ]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._locked():
            data = self._read()
            for s in data.get("consumed_sessions", []):
                if s.get("session_id") == session_id:
                    return dict(s)
            return None

    def has_session(self, session_id: str) -> bool:
        return self.get_session(session_id) is not None

    def has_replacement_for(self, failed_session_id: str) -> bool:
        with self._locked():
            data = self._read()
            return any(
                s.get("replacement_for_session_id") == failed_session_id
                for s in data.get("consumed_sessions", [])
            )

    def has_used_replacement_grant(self, replacement_grant_id: str) -> bool:
        with self._locked():
            data = self._read()
            return any(
                s.get("replacement_grant_id") == replacement_grant_id
                for s in data.get("consumed_sessions", [])
            )

    def reserve_session(
        self,
        contract: ExploratoryCampaignContract,
        *,
        session_id: str,
        stage_id: str,
        max_sessions: int,
        replacement_grant_id: str | None = None,
        replacement_for_session_id: str | None = None,
    ) -> None:
        """Atomically reserve one authorized session slot before broker connection."""
        with self._locked():
            data = self._read()
            sessions = list(data.get("consumed_sessions", []))
            if any(s.get("session_id") == session_id for s in sessions):
                raise ValueError(f"Duplicate session_id {session_id} already consumed.")
            if replacement_for_session_id and session_id == replacement_for_session_id:
                raise ValueError("Replacement session must use a new session_id.")

            stage_sessions = [
                s
                for s in sessions
                if s.get("contract_id") == contract.contract_id and s.get("stage_id") == stage_id
            ]
            stage_count = len(stage_sessions)

            if replacement_grant_id is not None or replacement_for_session_id is not None:
                if not replacement_grant_id or not replacement_for_session_id:
                    raise ValueError(
                        "Both replacement_grant_id and replacement_for_session_id are required."
                    )
                if not any(
                    s.get("session_id") == replacement_for_session_id for s in stage_sessions
                ):
                    raise ValueError(
                        f"Referenced failed session {replacement_for_session_id!r} not found in {stage_id}."
                    )
                if any(s.get("replacement_grant_id") == replacement_grant_id for s in sessions):
                    raise ValueError(
                        f"Replacement grant {replacement_grant_id!r} already consumed."
                    )
                if any(
                    s.get("replacement_for_session_id") == replacement_for_session_id
                    for s in sessions
                ):
                    raise ValueError(
                        f"Failed session {replacement_for_session_id!r} already has a replacement."
                    )
                effective_cap = max_sessions + 1
                if stage_count >= effective_cap:
                    raise ValueError(
                        f"Authorized session budget (with replacement) exhausted for {stage_id}: "
                        f"{stage_count}/{effective_cap} consumed."
                    )
            elif stage_count >= max_sessions:
                raise ValueError(
                    f"Authorized session budget exhausted for {stage_id}: "
                    f"{stage_count}/{max_sessions} consumed."
                )

            now_utc = datetime.now(UTC).isoformat()
            entry: dict[str, Any] = {
                "contract_id": contract.contract_id,
                "stage_id": stage_id,
                "session_id": session_id,
                "reserved_at_utc": now_utc,
                "updated_at_utc": now_utc,
                "status": SessionLifecycleState.RESERVED.value,
                "lifecycle_state": SessionLifecycleState.RESERVED.value,
                "state_history": [
                    {"state": SessionLifecycleState.RESERVED.value, "at_utc": now_utc}
                ],
            }
            if replacement_grant_id and replacement_for_session_id:
                entry["replacement_grant_id"] = replacement_grant_id
                entry["replacement_for_session_id"] = replacement_for_session_id
            sessions.append(entry)
            data["consumed_sessions"] = sessions
            self._write_atomic(data)

    def transition_session(
        self,
        session_id: str,
        new_state: SessionLifecycleState | str,
        *,
        reason: str = "",
        error_type: str = "",
        runtime_status: str = "",
    ) -> dict[str, Any]:
        """Atomically transition an existing session usage record to a durable lifecycle state."""
        target = SessionLifecycleState(new_state)
        with self._locked():
            data = self._read()
            sessions = list(data.get("consumed_sessions", []))
            idx = next(
                (i for i, s in enumerate(sessions) if s.get("session_id") == session_id),
                None,
            )
            if idx is None:
                raise KeyError(f"Session {session_id!r} not found in usage store.")
            entry = dict(sessions[idx])
            raw_cur = str(
                entry.get("lifecycle_state")
                or entry.get("status")
                or SessionLifecycleState.RESERVED.value
            )
            current = SessionLifecycleState(raw_cur)
            if target != current and target not in _VALID_LIFECYCLE_TRANSITIONS[current]:
                raise ValueError(
                    f"Invalid lifecycle transition for {session_id}: {current.value} -> {target.value}"
                )
            now_utc = datetime.now(UTC).isoformat()
            history = list(entry.get("state_history") or [])
            if not history:
                history.append({
                    "state": current.value,
                    "at_utc": entry.get("reserved_at_utc") or now_utc,
                })
            if target != current:
                hist_item: dict[str, Any] = {"state": target.value, "at_utc": now_utc}
                if reason:
                    hist_item["reason"] = reason[:240]
                history.append(hist_item)

            entry["status"] = target.value
            entry["lifecycle_state"] = target.value
            entry["updated_at_utc"] = now_utc
            entry["state_history"] = history
            if reason:
                entry["stop_reason"] = reason[:500]
            if error_type:
                entry["error_type"] = error_type
            if runtime_status:
                entry["runtime_status"] = runtime_status

            sessions[idx] = entry
            data["consumed_sessions"] = sessions
            self._write_atomic(data)
            return dict(entry)


def default_campaign_usage_path(repo_root: Path, contract_id: str) -> Path:
    """Return default durable session consumption path for a campaign contract."""
    return Path(repo_root).resolve() / PHASE48_CAMPAIGNS_DIR / f"{contract_id}.usage.json"


def verify_failed_startup_session(
    *,
    usage_store: CampaignUsageStore,
    sessions_dir: Path,
    contract_id: str,
    stage_id: str,
    failed_session_id: str,
) -> tuple[str, str]:
    """Verify a previous Stage A session failed at startup with zero predictions/inputs."""
    if stage_id != ExploratoryCampaignStage.STAGE_A_15M.value:
        raise ValueError("Replacement authorization is only supported for STAGE_A_15M.")
    rec = usage_store.get_session(failed_session_id)
    if rec is None:
        raise ValueError(f"Failed session {failed_session_id!r} not found in CampaignUsageStore.")
    if rec.get("contract_id") != contract_id or rec.get("stage_id") != stage_id:
        raise ValueError(
            f"Session {failed_session_id!r} belongs to {rec.get('contract_id')}/{rec.get('stage_id')}, "
            f"not {contract_id}/{stage_id}."
        )
    if usage_store.has_replacement_for(failed_session_id):
        raise ValueError(
            f"Failed session {failed_session_id!r} already has an authorized/consumed replacement."
        )

    rec_state = str(rec.get("lifecycle_state") or rec.get("status") or "")
    sdir = Path(sessions_dir) / failed_session_id
    meta_path = sdir / "session.json"
    if not meta_path.is_file():
        raise ValueError(f"Missing session metadata for failed session {failed_session_id!r}.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("session_id") != failed_session_id or not bool(meta.get("is_genuine_mode")):
        raise ValueError(f"Session {failed_session_id!r} is not a genuine Phase 48 session.")
    if meta.get("campaign_contract_id") != contract_id:
        raise ValueError(f"Session {failed_session_id!r} contract_id mismatch.")

    per_diag = meta.get("per_candidate_diagnostics", {})
    if any(
        int(d.get("total_opportunities", 0)) > 0 or int(d.get("selected_predictions", 0)) > 0
        for d in per_diag.values()
        if isinstance(d, dict)
    ):
        raise ValueError(
            f"Session {failed_session_id!r} recorded predictions/opportunities; not a startup failure."
        )

    for jsonl_name in ("inputs.jsonl", "predictions.jsonl"):
        jpath = sdir / jsonl_name
        if jpath.is_file() and jpath.read_text(encoding="utf-8").strip():
            raise ValueError(
                f"Session {failed_session_id!r} has non-empty {jsonl_name}; not a startup failure."
            )

    sess_status = str(meta.get("status", ""))
    is_new_failed_startup = (
        rec_state == SessionLifecycleState.FAILED_STARTUP.value
        and sess_status == SessionLifecycleState.FAILED_STARTUP.value
    )
    # Pre-Phase-49.1 startup failure left usage record RESERVED and session.json INTERRUPTED_DISCONNECT/FAILED_PROVENANCE.
    is_legacy_failed_startup = (
        rec_state == SessionLifecycleState.RESERVED.value
        and sess_status in ("FAILED_STARTUP", "INTERRUPTED_DISCONNECT", "FAILED_PROVENANCE")
    )
    if not (is_new_failed_startup or is_legacy_failed_startup):
        raise ValueError(
            f"Session {failed_session_id!r} is not an eligible FAILED_STARTUP session "
            f"(usage={rec_state}, session={sess_status})."
        )

    return sess_status, str(meta.get("stop_reason", ""))


def authorize_stage_a_replacement(
    contract: ExploratoryCampaignContract,
    *,
    failed_session_id: str,
    authorized_by: str,
    allowed_account_ref: str,
    review_note: str,
    usage_store: CampaignUsageStore,
    sessions_dir: Path,
    authorized_at_utc: str | None = None,
    hours_valid: int = 24,
    expires_at_utc: str | None = None,
    authorization_mode: ObservationAuthorizationMode | str | None = None,
    hmac_key: bytes | None = None,
) -> ExploratoryCampaignContract:
    """Explicitly authorize ONE bounded 15m Stage A replacement attempt after a failed startup.

    Preserves the original failed session and usage history untouched; never resets or deletes
    usage records; prohibits outcome drain; prevents duplicate or unattended retries.
    """
    owner = authorized_by.strip()
    note = review_note.strip()
    acct = allowed_account_ref.strip()
    if not owner:
        raise ValueError("authorized_by must be non-empty")
    if not note:
        raise ValueError("review_note must be non-empty when authorizing a replacement")
    if not acct or acct == "OWNER_VAULT_PRACTICE_ACCOUNT_UNASSIGNED":
        raise ValueError("allowed_account_ref must identify an explicit vault account reference")
    if hours_valid <= 0:
        raise ValueError("hours_valid must be positive")
    if contract.authorization_status != CampaignAuthorizationStatus.AUTHORIZED_ACTIVE.value:
        raise ValueError(
            f"Contract authorization_status must be AUTHORIZED_ACTIVE (got {contract.authorization_status})."
        )
    if contract.authorized_stage != ExploratoryCampaignStage.STAGE_A_15M.value:
        raise ValueError(
            f"Replacement authorization only applies to STAGE_A_15M (got {contract.authorized_stage})."
        )
    if contract.allowed_account_ref != acct:
        raise ValueError("allowed_account_ref does not match authorized contract account.")
    if contract.contract_sha256 != contract.compute_contract_sha256():
        raise ValueError("Campaign contract SHA-256 mismatch; cannot authorize replacement.")

    target_mode = (
        ObservationAuthorizationMode(str(authorization_mode))
        if authorization_mode is not None
        else ObservationAuthorizationMode(contract.authorization_mode)
    )
    if (
        contract.authorization_mode
        == ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY.value
        and target_mode != ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY
    ):
        raise ValueError(
            "Cannot switch a WALLET_INDEPENDENT_SPOT_READ_ONLY contract back to "
            "PRACTICE_VERIFIED_REQUIRED during replacement authorization."
        )

    stage_consumed = usage_store.list_consumed(
        contract.contract_id, ExploratoryCampaignStage.STAGE_A_15M.value
    )
    if len(stage_consumed) >= 2:
        raise ValueError(
            "Stage A replacement budget already consumed (maximum 1 original + 1 replacement)."
        )

    prev_status, prev_reason = verify_failed_startup_session(
        usage_store=usage_store,
        sessions_dir=sessions_dir,
        contract_id=contract.contract_id,
        stage_id=ExploratoryCampaignStage.STAGE_A_15M.value,
        failed_session_id=failed_session_id,
    )

    is_wallet_indep = target_mode == ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY
    eff_protocol = (
        "49.2.0"
        if (is_wallet_indep and contract.protocol_version == "48.1.0")
        else contract.protocol_version
    )
    eff_require_practice = not is_wallet_indep

    now_dt = datetime.fromisoformat(authorized_at_utc) if authorized_at_utc else datetime.now(UTC)
    exp_str = expires_at_utc or (now_dt + timedelta(hours=hours_valid)).isoformat()
    mech = "LOCAL_HMAC_SHA256_V1" if hmac_key is not None else "LOCAL_UNKEYED_INTEGRITY_DIGEST_V1"
    repl = StageReplacementAuthorization(
        replacement_grant_id=f"p48_repl_{uuid.uuid4().hex[:12]}",
        stage_id=ExploratoryCampaignStage.STAGE_A_15M.value,
        previous_failed_session_id=failed_session_id,
        previous_failed_status=prev_status,
        previous_failed_reason_summary=prev_reason[:240],
        authorized_by=owner,
        authorized_at_utc=now_dt.isoformat(),
        expires_at_utc=exp_str,
        max_attempts=1,
        session_duration_minutes=15,
        allow_outcome_drain=False,
        max_outcome_drain_minutes=0,
        review_note=note,
        authorization_mode=target_mode.value,
    )
    updated = replace(
        contract,
        protocol_version=eff_protocol,
        authorization_mode=target_mode.value,
        require_practice_account=eff_require_practice,
        authorized_by=owner,
        authorized_at_utc=now_dt.isoformat(),
        expires_at_utc=exp_str,
        authorization_mechanism=mech,
        authorization_note=f"Stage A replacement for {failed_session_id}: {note}",
        replacement_authorization=repl,
    )
    updated = replace(updated, contract_sha256=updated.compute_contract_sha256())
    sig = updated.compute_authorization_digest(hmac_key=hmac_key)
    return replace(updated, authorization_signature_sha256=sig)


def authorize_campaign_contract(
    contract: ExploratoryCampaignContract,
    *,
    authorized_by: str,
    allowed_account_ref: str,
    authorized_stage: str,
    authorization_note: str = "",
    authorized_at_utc: str | None = None,
    hours_valid: int = 24,
    expires_at_utc: str | None = None,
    require_practice_account: bool = True,
    authorization_mode: ObservationAuthorizationMode
    | str = ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED,
    hmac_key: bytes | None = None,
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
    if hours_valid <= 0:
        raise ValueError("hours_valid must be positive")
    valid_stages = {s.stage_id for s in contract.stage_plan}
    if authorized_stage not in valid_stages:
        raise ValueError(f"authorized_stage must be one of {sorted(valid_stages)}")

    mode = ObservationAuthorizationMode(str(authorization_mode))
    is_wallet_indep = mode == ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY
    if not is_wallet_indep and not require_practice_account:
        raise ValueError(
            "require_practice_account cannot be False under PRACTICE_VERIFIED_REQUIRED mode"
        )
    eff_protocol = (
        "49.2.0"
        if (is_wallet_indep and contract.protocol_version == "48.1.0")
        else contract.protocol_version
    )
    eff_require_practice = not is_wallet_indep

    now_dt = datetime.fromisoformat(authorized_at_utc) if authorized_at_utc else datetime.now(UTC)
    exp_str = expires_at_utc or (now_dt + timedelta(hours=hours_valid)).isoformat()
    mech = "LOCAL_HMAC_SHA256_V1" if hmac_key is not None else "LOCAL_UNKEYED_INTEGRITY_DIGEST_V1"
    updated = replace(
        contract,
        protocol_version=eff_protocol,
        allowed_account_ref=acct,
        authorization_status=CampaignAuthorizationStatus.AUTHORIZED_ACTIVE.value,
        authorized_stage=authorized_stage,
        authorized_by=owner,
        authorized_at_utc=now_dt.isoformat(),
        expires_at_utc=exp_str,
        require_practice_account=eff_require_practice,
        authorization_mode=mode.value,
        authorization_mechanism=mech,
        authorization_note=authorization_note.strip(),
        replacement_authorization=None,
    )
    updated = replace(updated, contract_sha256=updated.compute_contract_sha256())
    sig = updated.compute_authorization_digest(hmac_key=hmac_key)
    return replace(updated, authorization_signature_sha256=sig)


def revoke_campaign_contract(
    contract: ExploratoryCampaignContract,
    *,
    revoked_by: str,
    reason: str = "Operator revoked authorization",
) -> ExploratoryCampaignContract:
    """Revoke an active exploratory campaign contract."""
    return replace(
        contract,
        authorization_status=CampaignAuthorizationStatus.REVOKED.value,
        authorization_note=f"REVOKED by {revoked_by.strip()}: {reason.strip()}",
    )


def get_stage_spec(
    contract: ExploratoryCampaignContract, stage_id: str | None
) -> ExploratoryStageSpec | None:
    """Lookup stage specification by stage_id."""
    if not stage_id:
        return None
    for s in contract.stage_plan:
        if s.stage_id == stage_id:
            return s
    return None


def validate_campaign_authorization(
    contract: ExploratoryCampaignContract | None,
    *,
    requested_candidate_ids: tuple[str, ...],
    requested_duration_minutes: int,
    loaded_artifact_digests: dict[str, str],
    account_ref: str,
    is_genuine_mode: bool,
    requested_allow_outcome_drain: bool = False,
    requested_drain_minutes: int = 0,
    usage_store: CampaignUsageStore | None = None,
    session_id: str | None = None,
    now_utc: datetime | None = None,
    hmac_key: bytes | None = None,
) -> AuthorizationCheckResult:
    """Fail-closed authorization gate for exploratory observation sessions."""
    if not is_genuine_mode:
        return AuthorizationCheckResult(
            authorized=True,
            status="SYNTHETIC_MODE_PERMITTED",
            reasons=("Synthetic demonstration/acceptance mode; zero broker connection.",),
        )

    reasons: list[str] = []
    exhausted_stage_budget = False
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

    valid_modes = {m.value for m in ObservationAuthorizationMode}
    if contract.authorization_mode not in valid_modes:
        reasons.append(
            f"Unknown or unsupported authorization_mode: {contract.authorization_mode!r}."
        )
    elif (
        contract.authorization_mode == ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED.value
    ):
        if not contract.require_practice_account:
            reasons.append(
                "PRACTICE_VERIFIED_REQUIRED mode requires require_practice_account=True."
            )
    elif (
        contract.authorization_mode
        == ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY.value
    ):
        if contract.protocol_version == "48.1.0":
            reasons.append(
                "WALLET_INDEPENDENT_SPOT_READ_ONLY requires reviewed contract protocol_version >= 49.2.0."
            )
        if contract.require_practice_account:
            reasons.append(
                "WALLET_INDEPENDENT_SPOT_READ_ONLY contract cannot set require_practice_account=True."
            )
        if tuple(contract.native_timeframes) != (60, 300, 900):
            reasons.append(
                "WALLET_INDEPENDENT_SPOT_READ_ONLY only permits native timeframes (60, 300, 900)."
            )
        if contract.allowed_account_ref == "OWNER_VAULT_PRACTICE_ACCOUNT_UNASSIGNED":
            reasons.append("WALLET_INDEPENDENT_SPOT_READ_ONLY requires a bound vault account_ref.")

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

    check_now = now_utc or datetime.now(UTC)
    if contract.expires_at_utc:
        try:
            exp_dt = datetime.fromisoformat(contract.expires_at_utc)
            if check_now >= exp_dt:
                reasons.append(f"Campaign authorization expired at {contract.expires_at_utc}.")
        except ValueError:
            reasons.append("Invalid expires_at_utc timestamp in contract.")

    if contract.authorization_mechanism == "LOCAL_HMAC_SHA256_V1":
        if hmac_key is None:
            reasons.append("HMAC-signed contract requires owner HMAC key for verification.")
        elif not contract.authorization_signature_sha256 or not hmac.compare_digest(
            contract.authorization_signature_sha256,
            contract.compute_authorization_digest(hmac_key=hmac_key),
        ):
            reasons.append("Invalid HMAC owner authorization signature.")
    elif (
        not contract.authorization_signature_sha256
        or contract.authorization_signature_sha256 != contract.compute_authorization_digest()
    ):
        reasons.append("Missing or invalid owner authorization_signature_sha256.")

    stage_spec = get_stage_spec(contract, contract.authorized_stage)
    if stage_spec is None:
        reasons.append(f"Invalid or missing authorized_stage: {contract.authorized_stage}.")
    else:
        if requested_duration_minutes != stage_spec.session_duration_minutes:
            reasons.append(
                f"Stage {stage_spec.stage_id} permits only {stage_spec.session_duration_minutes}m "
                f"sessions (requested {requested_duration_minutes}m)."
            )
        if requested_allow_outcome_drain or requested_drain_minutes > 0:
            if not stage_spec.allow_outcome_drain or not contract.allow_outcome_drain:
                reasons.append(f"Outcome drain is prohibited under {stage_spec.stage_id}.")
            elif requested_drain_minutes > stage_spec.max_outcome_drain_minutes:
                reasons.append(
                    f"Requested drain {requested_drain_minutes}m exceeds {stage_spec.stage_id} "
                    f"max {stage_spec.max_outcome_drain_minutes}m."
                )
        if usage_store is not None:
            if session_id and usage_store.has_session(session_id):
                reasons.append(f"Session {session_id} was already consumed (no restarts).")
            repl = contract.replacement_authorization
            if session_id and repl is not None and session_id == repl.previous_failed_session_id:
                reasons.append(f"Replacement attempt cannot reuse failed session_id {session_id}.")
            consumed = len(usage_store.list_consumed(contract.contract_id, stage_spec.stage_id))
            if consumed >= stage_spec.target_sessions:
                if (
                    repl is not None
                    and stage_spec.stage_id == ExploratoryCampaignStage.STAGE_A_15M.value
                ):
                    repl_errors: list[str] = []
                    if repl.stage_id != ExploratoryCampaignStage.STAGE_A_15M.value:
                        repl_errors.append("Replacement authorization must target STAGE_A_15M.")
                    if repl.max_attempts != 1:
                        repl_errors.append("Replacement authorization max_attempts must be 1.")
                    if repl.session_duration_minutes != 15 or requested_duration_minutes != 15:
                        repl_errors.append("Replacement attempt is strictly bounded to 15 minutes.")
                    if (
                        repl.allow_outcome_drain
                        or repl.max_outcome_drain_minutes != 0
                        or requested_allow_outcome_drain
                        or requested_drain_minutes > 0
                    ):
                        repl_errors.append("Replacement Stage A attempt prohibits outcome drain.")
                    if not repl.authorized_by.strip() or not repl.review_note.strip():
                        repl_errors.append(
                            "Replacement authorization missing authorized_by or review_note."
                        )
                    try:
                        repl_exp_dt = datetime.fromisoformat(repl.expires_at_utc)
                        if check_now >= repl_exp_dt:
                            repl_errors.append(
                                f"Replacement authorization expired at {repl.expires_at_utc}."
                            )
                    except ValueError:
                        repl_errors.append("Invalid replacement expires_at_utc timestamp.")
                    if (
                        repl.authorization_mode
                        and repl.authorization_mode != contract.authorization_mode
                    ):
                        repl_errors.append(
                            f"Replacement authorization_mode {repl.authorization_mode!r} "
                            f"conflicts with contract mode {contract.authorization_mode!r}."
                        )
                    prev_rec = usage_store.get_session(repl.previous_failed_session_id)
                    if (
                        prev_rec is None
                        or prev_rec.get("contract_id") != contract.contract_id
                        or prev_rec.get("stage_id") != ExploratoryCampaignStage.STAGE_A_15M.value
                    ):
                        repl_errors.append(
                            f"Referenced failed session {repl.previous_failed_session_id} "
                            "not found in Stage A usage history."
                        )
                    if usage_store.has_replacement_for(repl.previous_failed_session_id):
                        exhausted_stage_budget = True
                        repl_errors.append(
                            f"Replacement for {repl.previous_failed_session_id} was already consumed."
                        )
                    if usage_store.has_used_replacement_grant(repl.replacement_grant_id):
                        exhausted_stage_budget = True
                        repl_errors.append(
                            f"Replacement grant {repl.replacement_grant_id} was already consumed."
                        )
                    if consumed >= stage_spec.target_sessions + 1:
                        exhausted_stage_budget = True
                        repl_errors.append(
                            f"Stage A replacement budget exhausted ({consumed}/{stage_spec.target_sessions + 1})."
                        )
                    reasons.extend(repl_errors)
                else:
                    exhausted_stage_budget = True
                    reasons.append(
                        f"Authorized session count exhausted for {stage_spec.stage_id} "
                        f"({consumed}/{stage_spec.target_sessions})."
                    )

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
            exhausted_stage_budget=exhausted_stage_budget,
            authorization_mode=contract.authorization_mode,
        )

    return AuthorizationCheckResult(
        authorized=True,
        status="AUTHORIZED_FOR_SUPERVISED_EXPLORATORY_OBSERVATION",
        reasons=(),
        contract_id=contract.contract_id,
        contract_sha256=contract.contract_sha256,
        authorization_mode=contract.authorization_mode,
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
