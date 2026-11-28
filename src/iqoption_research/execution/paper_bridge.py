"""Strictly isolated Phase 48 Frozen AI -> PAPER Execution Bridge (Phase 50).

Connects the three frozen Phase 48 exploratory candidates
(``p48_exp_cand_a_m1_h15_hgb_v1``, ``p48_exp_cand_b_m5_h6_hgb_v1``,
``p48_exp_cand_c_m15_h4_hgb_v1``) to independent ``ExecutionMode.PAPER``
execution engines via immutable ``PredictionCommitmentRecord`` inputs.

Scientific & safety guarantees:
- Strictly ``PAPER/SIMULATED`` only; ``PRACTICE`` and ``REAL`` are refused.
- Never retrains, calibrates, modifies thresholds, or alters frozen bundles.
- Never touches ``data/phase48/sessions/*`` genuine prospective ledgers or
  consumes Stage C campaign authorization.
- ``ABSTAIN`` commitments never generate orders.
- Rejects stale, duplicate, tampered, threshold-drifted, invalid, or
  simultaneously conflicting cross-candidate directional signals.
- Explicitly distinguishes candle-direction results from live broker settlement
  and states fixed payout simulation assumptions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from iqoption_research.execution.execution_engine import (
    ModularExecutionConfig,
    ModularExecutionEngine,
)
from iqoption_research.execution.models import (
    ExecutionMode,
    OrderSide,
    PaperOrder,
    ProductType,
)
from iqoption_research.execution.modes import PaperAirGapViolationError
from iqoption_research.execution.risk import RiskConfig
from iqoption_research.execution.state_machine import ExecutionJournal
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.phase48.artifacts import (
    LoadedExploratoryBundle,
    default_models_dir,
    load_verified_exploratory_bundle,
)
from iqoption_research.phase48.constants import CANDIDATE_SPECS
from iqoption_research.phase48.evidence import PredictionCommitmentRecord

if TYPE_CHECKING:
    from iqoption_research.phase47.features import Phase47FeaturePipeline

SIMULATED_PAYOUT_ASSUMPTION: Final[str] = (
    "SIMULATED_FIXED_BINARY_PAYOUT_85PCT_NO_BROKER_FEED: "
    "Candle-close direction WIN/LOSS/TIE is NOT equivalent to live broker contract settlement "
    "(spread, strike timing, broker rejection, and dynamic payout are unavailable offline)."
)


def make_close_price_only_candle(
    *,
    symbol: str = "EURUSD",
    timeframe_seconds: int = 60,
    from_ts: int,
    to_ts: int,
    close: float,
    candle_digest: str = "",
    provenance: str = "CLOSE_PRICE_ONLY_SIMULATION",
) -> Candle:
    """Construct an explicitly labeled close-price-only Candle for historical replay.

    Never silently fabricates synthetic high, low, volume, or bid/ask spread.
    Sets open=close, high=close, low=close, volume=0.0, bid=close, ask=close,
    and explicitly marks the simulation mode in candle provenance and metadata.
    """
    size_map = {
        60: CandleSize.MIN_1,
        300: CandleSize.MIN_5,
        900: CandleSize.MIN_15,
    }
    size = size_map.get(timeframe_seconds, CandleSize.MIN_1)
    return Candle(
        active_id=1,
        symbol=symbol,
        size=size,
        from_ts=from_ts,
        to_ts=to_ts,
        at_ts=to_ts * 1_000_000_000,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=0.0,
        bid=close,
        ask=close,
        is_complete=True,
        provenance=provenance,
        identity_provenance=candle_digest or "CLOSE_PRICE_ONLY",
        raw={
            "simulation_model": "CLOSE_PRICE_ONLY_SIMULATION",
            "evidence_note": "No synthetic OHLC, volume, or bid/ask spread added.",
            "candle_digest": candle_digest,
        },
    )


class AISignalRejectionReason(StrEnum):
    """Deterministic rejection codes for AI-to-PAPER commitment evaluation."""

    ABSTAIN_SIGNAL = "abstain_signal"
    UNRECOGNIZED_CANDIDATE = "unrecognized_candidate"
    MISSING_FROZEN_MODEL_BUNDLE = "missing_frozen_model_bundle"
    MISMATCHED_ARTIFACT_DIGEST = "mismatched_artifact_digest"
    THRESHOLD_TAMPERED = "threshold_tampered"
    INVALID_COMMITMENT_DIGEST = "invalid_commitment_digest"
    INVALID_PROBABILITY = "invalid_probability"
    INCONSISTENT_PROBABILITY_DECISION = "inconsistent_probability_decision"
    INVALID_EXPIRATION = "invalid_expiration"
    INVALID_TIMEFRAME_OR_HORIZON = "invalid_timeframe_or_horizon"
    FUTURE_DATED_SIGNAL = "future_dated_signal"
    EXPIRED_SIGNAL = "expired_signal"
    STALE_SIGNAL = "stale_signal"
    REFERENCE_CANDLE_MISMATCH = "reference_candle_mismatch"
    DUPLICATE_SIGNAL = "duplicate_signal"
    CONFLICTING_CANDIDATE_SIGNAL = "conflicting_candidate_signal"
    RISK_REJECTED = "risk_rejected"


@dataclass(frozen=True, slots=True)
class ValidatedAICommitmentSignal:
    """Validated AI execution input derived from an immutable PredictionCommitmentRecord."""

    candidate_id: str
    candidate_artifact_digest: str
    native_timeframe_seconds: int
    prediction_horizon_bars: int
    decision_timestamp: int
    raw_probability: float
    calibrated_probability: float
    prob_up: float
    prob_down: float
    confidence_threshold: float
    confidence_margin: float
    decision: str
    expiration_ts: int
    provenance: dict[str, Any]
    order_intent_id: str

    def to_metadata(self) -> dict[str, Any]:
        """Serialize complete contract metadata into PaperOrder.metadata."""
        return {
            "candidate_id": self.candidate_id,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "native_timeframe_seconds": self.native_timeframe_seconds,
            "prediction_horizon_bars": self.prediction_horizon_bars,
            "decision_timestamp": self.decision_timestamp,
            "raw_prob_up": self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "prob_up": self.prob_up,
            "prob_down": self.prob_down,
            "confidence_threshold": self.confidence_threshold,
            "confidence_margin": self.confidence_margin,
            "decision": self.decision,
            "expiration_ts": self.expiration_ts,
            "expiry_ts": float(self.expiration_ts),
            "expiry_horizon_bars": self.prediction_horizon_bars,
            "provenance": dict(self.provenance),
            "order_intent_id": self.order_intent_id,
            "execution_label": "PAPER/SIMULATED",
            "contract_terms_assumption": SIMULATED_PAYOUT_ASSUMPTION,
            "signal_source": f"Phase48Frozen:{self.candidate_id}",
        }


@dataclass(frozen=True, slots=True)
class AISignalEvaluationResult:
    """Outcome of validating and routing a single PredictionCommitmentRecord."""

    record_id: str
    candidate_id: str
    decision: str
    accepted: bool
    order_intent_id: str
    rejection_reason: AISignalRejectionReason | None
    message: str
    order: PaperOrder | None = None


def compute_order_intent_id(record: PredictionCommitmentRecord) -> str:
    """Derive a deterministic unique order intent identity from commitment provenance."""
    canonical = (
        f"{record.candidate_id}|{record.candidate_artifact_digest}|"
        f"{record.timeframe_seconds}|{record.horizon_bars}|"
        f"{record.decision_timestamp}|{record.decision}|{record.commitment_digest}"
    )
    return "intent_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def verify_commitment_integrity(record: PredictionCommitmentRecord) -> bool:
    """Verify commitment SHA-256 integrity when commitment_digest is populated."""
    if not record.commitment_digest:
        return False
    encoded = json.dumps(record.commitment_payload(), sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return expected == record.commitment_digest


def load_readonly_commitments(
    predictions_jsonl_path: Path,
) -> list[PredictionCommitmentRecord]:
    """Read PredictionCommitmentRecords strictly read-only without altering files."""
    if not predictions_jsonl_path.exists():
        raise FileNotFoundError(f"Commitment ledger not found: {predictions_jsonl_path}")
    records: list[PredictionCommitmentRecord] = []
    for line in predictions_jsonl_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        data = json.loads(raw)
        records.append(PredictionCommitmentRecord(**data))
    return records


class FrozenPhase48Scorer:
    """Deterministic offline scorer using frozen verified Phase 48 artifacts.

    Strictly read-only on frozen models and manifests; emits sealed
    ``PredictionCommitmentRecord`` instances for permitted offline candle streams.
    """

    def __init__(
        self,
        *,
        models_dir: Path | None = None,
        candidate_ids: Sequence[str] | None = None,
    ) -> None:
        from iqoption_research.phase47.features import Phase47FeaturePipeline

        self.models_dir = models_dir or default_models_dir(Path(__file__).resolve().parents[3])
        target_ids = (
            tuple(candidate_ids)
            if candidate_ids
            else tuple(spec.frozen_candidate_id for spec in CANDIDATE_SPECS)
        )
        self.bundles: dict[str, LoadedExploratoryBundle] = {}
        self.pipelines: dict[str, Phase47FeaturePipeline] = {}

        for cid in target_ids:
            bundle = load_verified_exploratory_bundle(self.models_dir / cid)
            self.bundles[cid] = bundle
            self.pipelines[cid] = Phase47FeaturePipeline(
                timeframe_seconds=bundle.manifest.native_timeframe_seconds,
                preset=bundle.manifest.feature_preset,
            )

    def score_candles_to_commitments(
        self,
        candidate_id: str,
        candles: Sequence[Candle],
        *,
        session_id: str = "offline_paper_replay",
        evidence_classification: str = "OFFLINE_PAPER_REPLAY_ONLY",
    ) -> list[PredictionCommitmentRecord]:
        """Causally score a candle history slice and yield sealed commitment records."""
        bundle = self.bundles.get(candidate_id)
        pipeline = self.pipelines.get(candidate_id)
        if bundle is None or pipeline is None:
            raise KeyError(f"Unknown or unloaded frozen candidate '{candidate_id}'")

        tf = bundle.manifest.native_timeframe_seconds
        horizon = bundle.manifest.horizon_bars
        tau = float(bundle.manifest.frozen_confidence_threshold)

        feat_mat = pipeline.compute_features(candles)
        out: list[PredictionCommitmentRecord] = []

        for idx, candle in enumerate(candles):
            if feat_mat.warmup_mask[idx]:
                continue
            x_row = np.asarray(feat_mat.features[idx : idx + 1], dtype=np.float64)
            feat_digest = hashlib.sha256(x_row.tobytes()).hexdigest()
            x_scaled = np.asarray(bundle.scaler.transform(x_row), dtype=np.float64)
            p_raw = np.asarray(bundle.model.predict_proba(x_scaled), dtype=np.float64)
            raw_p_up = float(p_raw[0, 1])
            p_cal = np.asarray(bundle.calibrator.calibrate(p_raw), dtype=np.float64)
            p_up = float(p_cal[0, 1])
            p_down = float(1.0 - p_up)
            conf = max(p_up, p_down)
            margin = conf - tau
            if conf >= tau:
                decision = "UP" if p_up >= 0.5 else "DOWN"
                reason = (
                    f"Selected {decision}: conf={conf:.6f} >= tau={tau:.6f} (margin={margin:+.6f})"
                )
                abstain_reason = ""
            else:
                decision = "ABSTAIN"
                reason = f"Abstain: conf={conf:.6f} < tau={tau:.6f} (margin={margin:+.6f})"
                abstain_reason = reason

            close_ts = int(candle.to_ts)
            maturity_ts = close_ts + (tf * horizon)
            candle_payload = json.dumps(
                {
                    "symbol": candle.symbol,
                    "from_ts": int(candle.from_ts),
                    "to_ts": close_ts,
                    "close": candle.close,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            candle_digest = hashlib.sha256(candle_payload.encode("utf-8")).hexdigest()
            iso_now = datetime.fromtimestamp(close_ts, tz=UTC).isoformat()
            rec = PredictionCommitmentRecord(
                record_id=f"{candidate_id}_{close_ts}_{feat_digest[:8]}",
                session_id=session_id,
                candidate_id=candidate_id,
                candidate_artifact_digest=bundle.manifest.bundle_sha256,
                campaign_contract_sha256="OFFLINE_PAPER_REPLAY_NO_CAMPAIGN",
                timeframe_seconds=tf,
                horizon_bars=horizon,
                candle_open_ts=int(candle.from_ts),
                candle_close_ts=close_ts,
                decision_timestamp=close_ts,
                feature_cutoff_ts=close_ts,
                feature_digest=feat_digest,
                reference_candle_price=float(candle.close),
                reference_candle_digest=candle_digest,
                prob_up=p_up,
                prob_down=p_down,
                confidence_threshold=tau,
                decision=decision,
                abstention_reason=abstain_reason,
                maturity_timestamp=maturity_ts,
                evidence_classification=evidence_classification,
                predicted_at_utc=iso_now,
                committed_at_ts=close_ts,
                committed_at_utc=iso_now,
                raw_prob_up=raw_p_up,
                confidence=conf,
                confidence_margin=margin,
                decision_reason=reason,
            ).sealed()
            out.append(rec)
        return out


@dataclass(slots=True)
class CandidatePaperMetrics:
    """Telemetry summary for one independent frozen candidate portfolio."""

    candidate_id: str
    timeframe_seconds: int
    horizon_bars: int
    frozen_confidence_threshold: float
    total_commitments: int = 0
    abstained_count: int = 0
    rejected_count: int = 0
    accepted_orders_count: int = 0
    settled_orders_count: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    open_orders_count: int = 0
    initial_balance: float = 10000.0
    final_balance: float = 10000.0
    simulated_pnl: float = 0.0
    execution_label: str = "PAPER/SIMULATED"
    contract_terms_assumption: str = SIMULATED_PAYOUT_ASSUMPTION


class PaperBridgePortfolioCoordinator:
    """Routes Phase 48 PredictionCommitmentRecords into isolated PAPER candidate engines.

    Maintains separate independent portfolios for each frozen candidate before any
    ensemble consideration, AND an explicit cross-candidate conflict-guarded book that
    prevents simultaneous opposite directional signals from opening conflicting trades.
    """

    def __init__(
        self,
        *,
        models_dir: Path | None = None,
        output_dir: Path | None = None,
        execution_mode: ExecutionMode = ExecutionMode.PAPER,
        initial_balance: float = 10000.0,
        fixed_stake: float = 10.0,
        payout_rate: float = 0.85,
        max_signal_age_seconds: float = 120.0,
        risk_config: RiskConfig | None = None,
    ) -> None:
        if execution_mode != ExecutionMode.PAPER:
            raise PaperAirGapViolationError(
                "Frozen Phase 48 AI-to-execution bridge is strictly restricted to PAPER mode. "
                f"Attempted mode: {execution_mode.value}."
            )
        self.models_dir = models_dir or default_models_dir(Path(__file__).resolve().parents[3])
        self.output_dir = output_dir
        self.fixed_stake = float(fixed_stake)
        self.payout_rate = float(payout_rate)
        self.max_signal_age_seconds = float(max_signal_age_seconds)
        self.initial_balance = float(initial_balance)
        self._risk_config = risk_config or RiskConfig(
            max_stake_per_trade=max(200.0, self.fixed_stake),
            max_stake_pct_balance=0.10,
            max_concurrent_positions=50,
            max_simultaneous_exposure=1000.0,
            max_trades_per_session=500,
            max_daily_loss=2000.0,
            max_consecutive_losses=100,
            min_confidence_prob=0.50,
            min_payout_rate=0.70,
            loss_cooldown_seconds=0.0,
        )

        self.bundles: dict[str, LoadedExploratoryBundle] = {}
        self.engines: dict[str, ModularExecutionEngine] = {}
        self.candidate_stats: dict[str, CandidatePaperMetrics] = {}

        for spec in CANDIDATE_SPECS:
            cid = spec.frozen_candidate_id
            bdir = self.models_dir / cid
            if bdir.exists():
                bundle = load_verified_exploratory_bundle(bdir)
                self.bundles[cid] = bundle

            db_path: Path | str | None = None
            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                db_path = self.output_dir / f"paper_{cid}.db"

            journal = ExecutionJournal(db_path)
            cfg = ModularExecutionConfig(
                symbol="EURUSD",
                mode=ExecutionMode.PAPER,
                default_product_type=ProductType.BINARY,
                initial_balance=self.initial_balance,
                bar_duration_seconds=spec.timeframe_seconds,
                default_payout_rate=self.payout_rate,
                default_horizon_bars=spec.horizon_bars,
                risk_config=self._risk_config,
            )
            self.engines[cid] = ModularExecutionEngine(cfg, journal=journal)
            binary_adapter = self.engines[cid].adapters.get(ProductType.BINARY)
            if binary_adapter is not None:
                binary_adapter.exact_maturity_only = True
            tau = (
                float(self.bundles[cid].manifest.frozen_confidence_threshold)
                if cid in self.bundles
                else 0.50
            )
            self.candidate_stats[cid] = CandidatePaperMetrics(
                candidate_id=cid,
                timeframe_seconds=spec.timeframe_seconds,
                horizon_bars=spec.horizon_bars,
                frozen_confidence_threshold=tau,
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
            )

        # Shared cross-candidate conflict-guarded book (prevents simultaneous opposing trades)
        shared_db = (self.output_dir / "paper_conflict_guarded.db") if self.output_dir else None
        self.conflict_guarded_engine = ModularExecutionEngine(
            ModularExecutionConfig(
                symbol="EURUSD",
                mode=ExecutionMode.PAPER,
                default_product_type=ProductType.BINARY,
                initial_balance=self.initial_balance,
                bar_duration_seconds=60,
                default_payout_rate=self.payout_rate,
                default_horizon_bars=1,
                risk_config=self._risk_config,
            ),
            journal=ExecutionJournal(shared_db),
        )
        shared_adapter = self.conflict_guarded_engine.adapters.get(ProductType.BINARY)
        if shared_adapter is not None:
            shared_adapter.exact_maturity_only = True
        self._seen_intents: set[str] = set()
        self._seen_record_ids: set[str] = set()
        self._evaluations: list[AISignalEvaluationResult] = []
        self._conflict_rejections: list[AISignalEvaluationResult] = []

        # Populate persistent duplicate sets from existing journals across restarts
        for eng in (*self.engines.values(), self.conflict_guarded_engine):
            for stored_order in eng.journal.load_all_orders().values():
                if stored_order.signal_id:
                    self._seen_intents.add(stored_order.signal_id)
                if stored_order.metadata and "order_intent_id" in stored_order.metadata:
                    self._seen_intents.add(str(stored_order.metadata["order_intent_id"]))
                if (
                    stored_order.metadata
                    and "provenance" in stored_order.metadata
                    and isinstance(stored_order.metadata["provenance"], dict)
                ):
                    rec_id = stored_order.metadata["provenance"].get("record_id")
                    if rec_id:
                        self._seen_record_ids.add(str(rec_id))

    def validate_commitment(
        self,
        record: PredictionCommitmentRecord,
        *,
        current_ts: float,
        entry_candle: Candle | None = None,
        require_sealed_digest: bool = True,
    ) -> tuple[ValidatedAICommitmentSignal | None, AISignalRejectionReason | None, str]:
        """Validate commitment integrity, staleness, bounds, and candidate binding."""
        if record.candidate_id not in self.engines:
            return (
                None,
                AISignalRejectionReason.UNRECOGNIZED_CANDIDATE,
                f"Unknown candidate_id '{record.candidate_id}'.",
            )

        bundle = self.bundles.get(record.candidate_id)
        if bundle is None:
            return (
                None,
                AISignalRejectionReason.MISSING_FROZEN_MODEL_BUNDLE,
                (
                    f"Verified frozen model bundle for candidate '{record.candidate_id}' is unavailable. "
                    "Commitments cannot be evaluated without verified immutable frozen artifacts."
                ),
            )

        if require_sealed_digest and not verify_commitment_integrity(record):
            return (
                None,
                AISignalRejectionReason.INVALID_COMMITMENT_DIGEST,
                f"Commitment integrity digest mismatch for record '{record.record_id}'.",
            )

        if record.candidate_artifact_digest != bundle.manifest.bundle_sha256:
            return (
                None,
                AISignalRejectionReason.MISMATCHED_ARTIFACT_DIGEST,
                (
                    f"Artifact digest mismatch: got {record.candidate_artifact_digest[:12]}, "
                    f"expected {bundle.manifest.bundle_sha256[:12]}."
                ),
            )
        if abs(record.confidence_threshold - bundle.manifest.frozen_confidence_threshold) > 1e-9:
            return (
                None,
                AISignalRejectionReason.THRESHOLD_TAMPERED,
                "Confidence threshold diverges from immutable frozen manifest.",
            )
        if (
            record.timeframe_seconds != bundle.manifest.native_timeframe_seconds
            or record.horizon_bars != bundle.manifest.horizon_bars
        ):
            return (
                None,
                AISignalRejectionReason.INVALID_TIMEFRAME_OR_HORIZON,
                "Timeframe or prediction horizon diverges from frozen model spec.",
            )

        if (
            not math.isfinite(record.prob_up)
            or not math.isfinite(record.prob_down)
            or not (0.0 <= record.prob_up <= 1.0)
            or not (0.0 <= record.prob_down <= 1.0)
            or abs((record.prob_up + record.prob_down) - 1.0) > 1e-4
        ):
            return (
                None,
                AISignalRejectionReason.INVALID_PROBABILITY,
                f"Invalid probabilities prob_up={record.prob_up}, prob_down={record.prob_down}.",
            )

        if record.raw_prob_up is not None and (
            not math.isfinite(record.raw_prob_up) or not (0.0 <= record.raw_prob_up <= 1.0)
        ):
            return (
                None,
                AISignalRejectionReason.INVALID_PROBABILITY,
                f"Invalid raw_prob_up={record.raw_prob_up}.",
            )

        if record.confidence is not None and (
            not math.isfinite(record.confidence) or not (0.0 <= record.confidence <= 1.0)
        ):
            return (
                None,
                AISignalRejectionReason.INVALID_PROBABILITY,
                f"Invalid confidence={record.confidence}.",
            )

        if record.confidence_margin is not None and not math.isfinite(record.confidence_margin):
            return (
                None,
                AISignalRejectionReason.INVALID_PROBABILITY,
                f"Invalid confidence_margin={record.confidence_margin}.",
            )

        # Independently recompute expected decision using frozen threshold and full-precision probabilities
        tau = float(bundle.manifest.frozen_confidence_threshold)
        recomputed_conf = max(float(record.prob_up), float(record.prob_down))
        if recomputed_conf >= tau:
            expected_decision = "UP" if record.prob_up >= 0.5 else "DOWN"
        else:
            expected_decision = "ABSTAIN"

        if record.decision != expected_decision:
            return (
                None,
                AISignalRejectionReason.INCONSISTENT_PROBABILITY_DECISION,
                (
                    f"Decision '{record.decision}' is inconsistent with recomputed decision "
                    f"'{expected_decision}' (prob_up={record.prob_up:.6f}, prob_down={record.prob_down:.6f}, "
                    f"conf={recomputed_conf:.6f}, tau={tau:.6f})."
                ),
            )

        # Validate exact maturity timestamp = decision_timestamp + (timeframe_seconds * horizon_bars)
        expected_maturity = int(record.decision_timestamp) + (
            int(record.timeframe_seconds) * int(record.horizon_bars)
        )
        if int(record.maturity_timestamp) != expected_maturity:
            return (
                None,
                AISignalRejectionReason.INVALID_EXPIRATION,
                (
                    f"Invalid maturity_timestamp ({record.maturity_timestamp}) != expected "
                    f"({expected_maturity}) for decision_ts={record.decision_timestamp}, "
                    f"tf={record.timeframe_seconds}, horizon={record.horizon_bars}."
                ),
            )

        # Reject future-dated, stale and expired signals
        if float(record.decision_timestamp) > current_ts + 1.0:
            return (
                None,
                AISignalRejectionReason.FUTURE_DATED_SIGNAL,
                (
                    f"Future-dated commitment signal: decision_timestamp ({record.decision_timestamp}) "
                    f"> current_ts ({current_ts:.1f})."
                ),
            )

        if (current_ts - float(record.decision_timestamp)) > self.max_signal_age_seconds:
            return (
                None,
                AISignalRejectionReason.STALE_SIGNAL,
                (
                    f"Stale commitment signal: age "
                    f"{current_ts - record.decision_timestamp:.1f}s > "
                    f"{self.max_signal_age_seconds:.1f}s."
                ),
            )

        if float(record.maturity_timestamp) <= current_ts:
            return (
                None,
                AISignalRejectionReason.EXPIRED_SIGNAL,
                (
                    f"Already expired commitment signal: maturity_timestamp ({record.maturity_timestamp}) "
                    f"<= current_ts ({current_ts:.1f})."
                ),
            )

        if record.decision == "ABSTAIN":
            return (
                None,
                AISignalRejectionReason.ABSTAIN_SIGNAL,
                "ABSTAIN commitments never produce execution orders.",
            )

        if record.decision not in {"UP", "DOWN"}:
            return (
                None,
                AISignalRejectionReason.INVALID_PROBABILITY,
                f"Invalid decision value '{record.decision}'.",
            )

        # Validate reference candle identity, symbol, timeframe, boundary and price against commitment
        if entry_candle is not None:
            cand_sym = getattr(entry_candle, "symbol", "EURUSD")
            if cand_sym != "EURUSD":
                return (
                    None,
                    AISignalRejectionReason.REFERENCE_CANDLE_MISMATCH,
                    f"Reference candle symbol '{cand_sym}' does not match expected 'EURUSD'.",
                )
            if int(entry_candle.to_ts) != int(record.decision_timestamp):
                return (
                    None,
                    AISignalRejectionReason.REFERENCE_CANDLE_MISMATCH,
                    (
                        f"Reference candle close ts {entry_candle.to_ts} "
                        f"!= decision_timestamp {record.decision_timestamp}."
                    ),
                )
            if int(entry_candle.to_ts) != int(record.candle_close_ts):
                return (
                    None,
                    AISignalRejectionReason.REFERENCE_CANDLE_MISMATCH,
                    (
                        f"Reference candle to_ts {entry_candle.to_ts} "
                        f"!= commitment candle_close_ts {record.candle_close_ts}."
                    ),
                )
            if int(entry_candle.from_ts) != int(record.candle_open_ts):
                return (
                    None,
                    AISignalRejectionReason.REFERENCE_CANDLE_MISMATCH,
                    (
                        f"Reference candle from_ts {entry_candle.from_ts} "
                        f"!= commitment candle_open_ts {record.candle_open_ts}."
                    ),
                )
            if (int(entry_candle.to_ts) - int(entry_candle.from_ts)) != int(
                record.timeframe_seconds
            ):
                return (
                    None,
                    AISignalRejectionReason.REFERENCE_CANDLE_MISMATCH,
                    (
                        f"Reference candle duration {int(entry_candle.to_ts) - int(entry_candle.from_ts)}s "
                        f"!= commitment timeframe {record.timeframe_seconds}s."
                    ),
                )
            if abs(float(entry_candle.close) - float(record.reference_candle_price)) > 1e-4:
                return (
                    None,
                    AISignalRejectionReason.REFERENCE_CANDLE_MISMATCH,
                    (
                        f"Reference candle close {entry_candle.close} diverges from commitment "
                        f"reference_candle_price {record.reference_candle_price}."
                    ),
                )
            if (
                record.reference_candle_digest
                and getattr(entry_candle, "raw", None)
                and isinstance(entry_candle.raw, dict)
            ):
                raw_digest = entry_candle.raw.get("candle_digest")
                if raw_digest and raw_digest != record.reference_candle_digest:
                    return (
                        None,
                        AISignalRejectionReason.REFERENCE_CANDLE_MISMATCH,
                        (
                            f"Reference candle digest '{raw_digest}' != commitment digest "
                            f"'{record.reference_candle_digest}'."
                        ),
                    )

        # Persistent duplicate check across memory and SQLite journals
        intent_id = compute_order_intent_id(record)
        if intent_id in self._seen_intents or record.record_id in self._seen_record_ids:
            return (
                None,
                AISignalRejectionReason.DUPLICATE_SIGNAL,
                f"Duplicate commitment intent '{intent_id}' / record '{record.record_id}'.",
            )
        for eng in self.engines.values():
            if eng.state_machine.get_order_by_signal(intent_id) is not None:
                return (
                    None,
                    AISignalRejectionReason.DUPLICATE_SIGNAL,
                    f"Duplicate commitment intent '{intent_id}' already registered in persistent journal.",
                )

        cal_prob = max(float(record.prob_up), float(record.prob_down))
        raw_prob = (
            float(record.raw_prob_up) if record.raw_prob_up is not None else float(record.prob_up)
        )
        margin = (
            float(record.confidence_margin)
            if record.confidence_margin is not None
            else (cal_prob - float(record.confidence_threshold))
        )
        provenance = {
            "evidence_classification": record.evidence_classification,
            "session_id": record.session_id,
            "campaign_contract_sha256": record.campaign_contract_sha256,
            "feature_digest": record.feature_digest,
            "reference_candle_digest": record.reference_candle_digest,
            "commitment_digest": record.commitment_digest,
            "record_id": record.record_id,
            "predicted_at_utc": record.predicted_at_utc,
            "research_disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
        }
        validated = ValidatedAICommitmentSignal(
            candidate_id=record.candidate_id,
            candidate_artifact_digest=record.candidate_artifact_digest,
            native_timeframe_seconds=int(record.timeframe_seconds),
            prediction_horizon_bars=int(record.horizon_bars),
            decision_timestamp=int(record.decision_timestamp),
            raw_probability=raw_prob,
            calibrated_probability=cal_prob,
            prob_up=float(record.prob_up),
            prob_down=float(record.prob_down),
            confidence_threshold=float(record.confidence_threshold),
            confidence_margin=margin,
            decision=record.decision,
            expiration_ts=int(record.maturity_timestamp),
            provenance=provenance,
            order_intent_id=intent_id,
        )
        return validated, None, "Commitment validated."

    def submit_commitment(
        self,
        record: PredictionCommitmentRecord,
        *,
        entry_candle: Candle,
        current_ts: float | None = None,
        require_sealed_digest: bool = True,
        route_conflict_guarded_book: bool = True,
    ) -> AISignalEvaluationResult:
        """Evaluate a PredictionCommitmentRecord and submit PAPER order if permitted."""
        now = float(current_ts) if current_ts is not None else float(entry_candle.to_ts)
        stats = self.candidate_stats.get(record.candidate_id)
        if stats is not None:
            stats.total_commitments += 1

        validated, rej_reason, msg = self.validate_commitment(
            record,
            current_ts=now,
            entry_candle=entry_candle,
            require_sealed_digest=require_sealed_digest,
        )
        intent_id = compute_order_intent_id(record)

        if validated is None:
            if stats is not None:
                if rej_reason == AISignalRejectionReason.ABSTAIN_SIGNAL:
                    stats.abstained_count += 1
                else:
                    stats.rejected_count += 1
            res = AISignalEvaluationResult(
                record_id=record.record_id,
                candidate_id=record.candidate_id,
                decision=record.decision,
                accepted=False,
                order_intent_id=intent_id,
                rejection_reason=rej_reason,
                message=msg,
                order=None,
            )
            self._evaluations.append(res)
            return res

        self._seen_intents.add(intent_id)
        self._seen_record_ids.add(record.record_id)

        engine = self.engines[validated.candidate_id]
        side = OrderSide.CALL if validated.decision == "UP" else OrderSide.PUT
        proposal = engine.create_proposal(
            symbol=entry_candle.symbol,
            side=side,
            stake=self.fixed_stake,
            confidence_probability=validated.calibrated_probability,
            payout_rate=self.payout_rate,
            product_type=ProductType.BINARY,
            candle_to_ts=float(entry_candle.to_ts),
            signal_id=validated.order_intent_id,
            metadata=validated.to_metadata(),
        )
        order, decision = engine.propose_and_validate(
            proposal,
            current_ts=now,
            is_replay=True,
        )
        if not decision.permitted:
            if stats is not None:
                stats.rejected_count += 1
            res = AISignalEvaluationResult(
                record_id=record.record_id,
                candidate_id=validated.candidate_id,
                decision=validated.decision,
                accepted=False,
                order_intent_id=validated.order_intent_id,
                rejection_reason=AISignalRejectionReason.RISK_REJECTED,
                message=decision.message,
                order=order,
            )
            self._evaluations.append(res)
            return res

        engine.submit_order(order, entry_candle)
        order.expiry_ts = float(validated.expiration_ts)
        engine.journal.upsert_order(order)
        if stats is not None:
            stats.accepted_orders_count += 1
            stats.open_orders_count = len(engine.open_orders)

        # Cross-candidate simultaneous conflict guard on shared multi-candidate book
        if route_conflict_guarded_book:
            self._route_to_conflict_guarded_book(validated, entry_candle, side, now)

        res = AISignalEvaluationResult(
            record_id=record.record_id,
            candidate_id=validated.candidate_id,
            decision=validated.decision,
            accepted=True,
            order_intent_id=validated.order_intent_id,
            rejection_reason=None,
            message="Order submitted in isolated PAPER mode.",
            order=order,
        )
        self._evaluations.append(res)
        return res

    def _route_to_conflict_guarded_book(
        self,
        sig: ValidatedAICommitmentSignal,
        entry_candle: Candle,
        side: OrderSide,
        now: float,
    ) -> None:
        """Reject simultaneous opposing directional signals from becoming opposing positions."""
        opposing_open = [
            o
            for o in self.conflict_guarded_engine.open_orders
            if o.side != side
            and (o.expiry_ts is None or o.expiry_ts > float(sig.decision_timestamp))
        ]
        if opposing_open:
            conflict_ids = [o.order_id for o in opposing_open]
            rej = AISignalEvaluationResult(
                record_id=str(sig.provenance.get("record_id", sig.order_intent_id)),
                candidate_id=sig.candidate_id,
                decision=sig.decision,
                accepted=False,
                order_intent_id=f"{sig.order_intent_id}_book",
                rejection_reason=AISignalRejectionReason.CONFLICTING_CANDIDATE_SIGNAL,
                message=(
                    f"Blocked conflicting simultaneous signal ({sig.decision} from "
                    f"{sig.candidate_id}) while opposite position(s) {conflict_ids} remain active."
                ),
            )
            self._conflict_rejections.append(rej)
            return

        book_prop = self.conflict_guarded_engine.create_proposal(
            symbol=entry_candle.symbol,
            side=side,
            stake=self.fixed_stake,
            confidence_probability=sig.calibrated_probability,
            payout_rate=self.payout_rate,
            product_type=ProductType.BINARY,
            candle_to_ts=float(entry_candle.to_ts),
            signal_id=f"book_{sig.order_intent_id}",
            metadata=sig.to_metadata(),
        )
        b_order, b_dec = self.conflict_guarded_engine.propose_and_validate(
            book_prop, current_ts=now, is_replay=True
        )
        if b_dec.permitted:
            self.conflict_guarded_engine.submit_order(b_order, entry_candle)
            b_order.expiry_ts = float(sig.expiration_ts)
            self.conflict_guarded_engine.journal.upsert_order(b_order)

    def process_candle(self, candle: Candle) -> None:
        """Advance market candle across matching native timeframe candidate engines & shared book."""
        candle_tf = int(candle.to_ts - candle.from_ts)
        for cid, engine in self.engines.items():
            st = self.candidate_stats[cid]
            if st.timeframe_seconds == candle_tf:
                engine.process_candle(candle)
                st.open_orders_count = len(engine.open_orders)
                st.settled_orders_count = engine.portfolio.total_trades
                st.wins = engine.portfolio.winning_trades
                st.losses = engine.portfolio.losing_trades
                st.ties = engine.portfolio.flat_trades
                st.final_balance = engine.portfolio.current_balance
                st.simulated_pnl = engine.portfolio.total_pnl

        self.conflict_guarded_engine.process_candle(candle)

    @property
    def evaluations(self) -> list[AISignalEvaluationResult]:
        return list(self._evaluations)

    @property
    def conflict_rejections(self) -> list[AISignalEvaluationResult]:
        return list(self._conflict_rejections)

    def summary_report(self) -> dict[str, Any]:
        """Build complete scientific/execution report distinguishing candle outcomes vs settlement."""
        return {
            "schema_version": "phase50.ai_paper_bridge.v1",
            "execution_mode": ExecutionMode.PAPER.value,
            "execution_label": "PAPER/SIMULATED",
            "research_disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
            "stage_c_untouched": True,
            "contract_terms_assumption": SIMULATED_PAYOUT_ASSUMPTION,
            "settlement_distinction": (
                "Simulated Binary outcomes evaluate candle close vs entry under a fixed "
                "85% assumed payout. They are NOT broker-verified contract settlements."
            ),
            "candidates": {
                cid: {
                    "candidate_id": s.candidate_id,
                    "timeframe_seconds": s.timeframe_seconds,
                    "horizon_bars": s.horizon_bars,
                    "frozen_confidence_threshold": s.frozen_confidence_threshold,
                    "total_commitments": s.total_commitments,
                    "abstained_count": s.abstained_count,
                    "rejected_count": s.rejected_count,
                    "accepted_orders_count": s.accepted_orders_count,
                    "settled_orders_count": s.settled_orders_count,
                    "open_orders_count": s.open_orders_count,
                    "wins": s.wins,
                    "losses": s.losses,
                    "ties": s.ties,
                    "initial_balance": s.initial_balance,
                    "final_balance": s.final_balance,
                    "simulated_pnl": s.simulated_pnl,
                }
                for cid, s in self.candidate_stats.items()
            },
            "conflict_guarded_book": {
                "total_trades": self.conflict_guarded_engine.portfolio.total_trades,
                "accepted_orders": self.conflict_guarded_engine.portfolio.total_trades
                + len(self.conflict_guarded_engine.open_orders),
                "settled_orders": self.conflict_guarded_engine.portfolio.total_trades,
                "open_orders": len(self.conflict_guarded_engine.open_orders),
                "wins": self.conflict_guarded_engine.portfolio.winning_trades,
                "losses": self.conflict_guarded_engine.portfolio.losing_trades,
                "ties": self.conflict_guarded_engine.portfolio.flat_trades,
                "simulated_pnl": self.conflict_guarded_engine.portfolio.total_pnl,
                "total_pnl": self.conflict_guarded_engine.portfolio.total_pnl,
                "conflicts_blocked": len(self._conflict_rejections),
            },
        }

    def close(self) -> None:
        """Close all SQLite journals owned by candidate engines cleanly."""
        for engine in self.engines.values():
            engine.close()
        self.conflict_guarded_engine.close()
