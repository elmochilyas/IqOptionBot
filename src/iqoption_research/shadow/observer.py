"""Prospective AI Shadow Observer Engine (Phase 40B).

Executes supervised read-only prospective shadow observation over real-time or
synthetic streaming market data. Enforces:
1. Frozen candidate model integrity (hashes, schemas, execution authority NONE).
2. Point-in-time prediction commitment before future outcomes exist.
3. Delayed outcome evaluation upon maturity candle receipt.
4. Full eligible opportunity tracking and matched baseline comparisons.
5. Contract-specific simulated economics (Generic vs Binary ATM refund vs Digital ATM loss).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.models.market import Candle
from iqoption_research.shadow.engine import EngineError, ShadowPredictionEngine
from iqoption_research.shadow.evaluator import (
    EvaluationSummary,
    ShadowOutcomeEvaluator,
)
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    OutcomeStatus,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
    validate_storage_identifier,
)
from iqoption_research.shadow.registry import (
    FrozenRegistrationRecord,
    ShadowModelRegistry,
)
from iqoption_research.shadow.streaming import (
    MarketStreamProtocol,
    StreamingCandleEvent,
)

logger = logging.getLogger(__name__)

DISCLAIMER: Final[str] = (
    "OFFLINE RESEARCH SHADOW EVALUATION. NO LIVE EXECUTION AUTHORITY. "
    "NO DEMONSTRATED EDGE. CANDLE-CLOSE COMPARISONS ARE HYPOTHETICAL "
    "RESEARCH BENCHMARKS AND NEVER OBSERVED BROKER ORDER SETTLEMENTS. "
    "ZERO LIVE ORDERS. STRICTLY READ-ONLY SHADOW OBSERVATION."
)


@dataclass(frozen=True, slots=True)
class ObservationSessionConfig:
    """Immutable configuration for a prospective shadow observation session."""

    session_id: str
    model_ids: tuple[str, ...]
    duration_minutes: int
    account_id: str | None = None
    assumed_payout: float = 0.85
    is_synthetic: bool = False
    max_data_age_seconds: int = 300
    min_warmup_bars: int = 21
    max_history_bars: int = 120
    allow_shortened_session: bool = False
    status_path: Path | None = None
    cancel_file: Path | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.status_path is not None:
            d["status_path"] = str(self.status_path)
        if self.cancel_file is not None:
            d["cancel_file"] = str(self.cancel_file)
        return d


@dataclass(slots=True)
class ModelObservationState:
    """Real-time operational metrics for a single model under prospective observation."""

    model_id: str
    model_family: str
    threshold: float
    bundle_sha256: str
    is_baseline: bool
    total_predictions: int = 0
    pending_predictions: int = 0
    resolved_outcomes: int = 0
    trades_taken: int = 0
    skips_count: int = 0
    wins_count: int = 0
    losses_count: int = 0
    ties_count: int = 0
    invalid_count: int = 0
    win_rate: float = 0.0
    win_rate_decisive: float = 0.0
    brier_score: float = 0.25
    directional_accuracy: float = 0.0
    coverage: float = 0.0
    abstention_rate: float = 0.0
    matched_baseline_accuracy: float | None = None
    wilson_ci_decisive: tuple[float, float] = (0.0, 1.0)
    expectancy_binary_pct: float = 0.0
    expectancy_digital_pct: float = 0.0
    break_even_gap_binary: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ObservationProgress:
    """Operational progress and diagnostics snapshot for the active observation session."""

    session_id: str
    connection_status: str
    elapsed_seconds: float
    remaining_seconds: float
    valid_candles_count: int
    duplicate_boundaries_count: int
    continuity_gaps_count: int
    stale_candles_count: int
    last_verified_candle_ts: int | None
    last_candle_received_utc: str | None
    models: dict[str, ModelObservationState] = field(default_factory=dict)
    termination_reason: str | None = None
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["models"] = {m: state.to_dict() for m, state in self.models.items()}
        return d


@dataclass(frozen=True, slots=True)
class ObservationSessionResult:
    """Comprehensive final summary of a completed prospective shadow observation session."""

    session_id: str
    started_at_utc: str
    ended_at_utc: str
    actual_duration_seconds: float
    requested_minutes: int
    termination_reason: str
    is_synthetic: bool
    total_candles_processed: int
    duplicate_boundaries_count: int
    continuity_gaps_count: int
    stale_candles_count: int
    summaries: dict[str, EvaluationSummary]
    manifest_sha256: str
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "actual_duration_seconds": round(self.actual_duration_seconds, 2),
            "requested_minutes": self.requested_minutes,
            "termination_reason": self.termination_reason,
            "is_synthetic": self.is_synthetic,
            "total_candles_processed": self.total_candles_processed,
            "duplicate_boundaries_count": self.duplicate_boundaries_count,
            "continuity_gaps_count": self.continuity_gaps_count,
            "stale_candles_count": self.stale_candles_count,
            "summaries": {m: s.to_dict() for m, s in self.summaries.items()},
            "manifest_sha256": self.manifest_sha256,
            "disclaimer": self.disclaimer,
        }


class ProspectiveShadowObserver:
    """Supervised prospective shadow observer for EURUSD M1 candidate models."""

    def __init__(
        self,
        config: ObservationSessionConfig,
        registry: ShadowModelRegistry,
        *,
        root: Path | None = None,
        journal: ShadowPredictionJournal | None = None,
        initial_history: Sequence[Candle] | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.root = root.resolve() if root is not None else Path.cwd().resolve()

        validate_storage_identifier(config.session_id)
        for m in config.model_ids:
            validate_storage_identifier(m)

        self.session_dir = self.root / DEFAULT_SHADOW_SESSIONS_DIR / config.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.journal = (
            journal
            if journal is not None
            else ShadowPredictionJournal(
                config.session_id, root=self.root, session_dir=self.session_dir
            )
        )

        # Pre-flight candidate model verification
        self.engines: dict[str, ShadowPredictionEngine] = {}
        self.records: dict[str, FrozenRegistrationRecord] = {}
        self.states: dict[str, ModelObservationState] = {}

        for model_id in config.model_ids:
            rec = self.registry.get_candidate(model_id)
            if rec is None:
                raise ValueError(
                    f"Candidate model '{model_id}' is not registered in frozen registry"
                )
            if rec.execution_authority != "NONE":
                raise ValueError(
                    f"Candidate model '{model_id}' must have execution_authority='NONE', "
                    f"got {rec.execution_authority!r}"
                )
            # Re-verify bundle and manifest integrity
            bundle_data = self.registry.load_frozen_bundle(model_id)
            if bundle_data is None:
                raise ValueError(f"Failed to load verified bundle for '{model_id}'")

            self.records[model_id] = rec
            self.engines[model_id] = ShadowPredictionEngine(
                self.registry,
                model_id,
                min_warmup_bars=config.min_warmup_bars,
                max_staleness_seconds=config.max_data_age_seconds,
            )
            self.states[model_id] = ModelObservationState(
                model_id=model_id,
                model_family=rec.model_family,
                threshold=rec.confidence_threshold,
                bundle_sha256=rec.bundle_sha256,
                is_baseline=rec.is_baseline,
            )

        self.evaluator = ShadowOutcomeEvaluator(
            journal=self.journal,
            assumed_payout=config.assumed_payout,
        )

        # Operational tracking
        self._candle_buffer: list[Candle] = list(initial_history) if initial_history else []
        self._seen_boundaries: set[int] = {c.to_ts for c in self._candle_buffer}
        self._valid_candles_count = 0
        self._duplicate_boundaries_count = 0
        self._continuity_gaps_count = 0
        self._stale_candles_count = 0
        self._last_verified_candle_ts: int | None = (
            self._candle_buffer[-1].to_ts if self._candle_buffer else None
        )
        self._last_received_utc: str | None = None
        self._started_at_wall: float = 0.0
        self._started_at_mono: float = 0.0
        self._deadline_mono: float = 0.0
        self._is_cancelled = False

        self._init_session_metadata()

    def _init_session_metadata(self) -> None:
        """Write initial tamper-evident session.json file."""
        now_utc = datetime.now(UTC).isoformat()
        session_meta = {
            "schema_version": 2,
            "session_id": self.config.session_id,
            "evidence_class": (
                "SYNTHETIC_STREAMING_TEST" if self.config.is_synthetic else "PROSPECTIVE_SHADOW"
            ),
            "created_at_utc": now_utc,
            "authorized_duration_minutes": self.config.duration_minutes,
            "account_id": self.config.account_id,
            "model_ids": list(self.config.model_ids),
            "assumed_payout": self.config.assumed_payout,
            "is_synthetic": self.config.is_synthetic,
            "execution_authority": "NONE",
            "disposition": "NO_DEMONSTRATED_EDGE",
            "disclaimer": DISCLAIMER,
            "models_metadata": {
                m: {
                    "family": r.model_family,
                    "threshold": r.confidence_threshold,
                    "bundle_sha256": r.bundle_sha256,
                    "is_baseline": r.is_baseline,
                    "data_cutoff_ts": r.data_cutoff_ts,
                }
                for m, r in self.records.items()
            },
        }
        meta_path = self.session_dir / "session.json"
        meta_path.write_text(json.dumps(session_meta, indent=2, sort_keys=True), encoding="utf-8")

    def cancel(self) -> None:
        """Signal cooperative cancellation."""
        self._is_cancelled = True

    def process_candle_event(self, event: StreamingCandleEvent) -> None:
        """Process a single incoming market observation event.

        Execution stages:
        1. Evidence validation and continuity checking.
        2. Delayed outcome resolution of any matured predictions.
        3. Point-in-time prediction generation and immediate flush before future exists.
        4. Operational progress update.
        """
        c = event.candle
        self._last_received_utc = event.received_at_utc

        # 1. Evidence validation
        if c.symbol != "EURUSD" or c.size.value != 60:
            logger.warning("Rejected candle with unexpected instrument or size: %s", c)
            return

        if not c.is_complete:
            logger.warning("Rejected incomplete forming candle at ts=%d", c.from_ts)
            return

        if getattr(c, "is_otc", False):
            logger.warning("Rejected prohibited OTC candle at ts=%d", c.from_ts)
            return

        # Staleness check
        age = event.received_at_utc_ts - c.to_ts
        if age > self.config.max_data_age_seconds:
            self._stale_candles_count += 1
            logger.warning(
                "Rejected stale candle (age %.1fs > %ds)",
                age,
                self.config.max_data_age_seconds,
            )
            return

        # Deduplication check
        if c.to_ts in self._seen_boundaries:
            self._duplicate_boundaries_count += 1
            return

        # Continuity check against buffer tail
        if self._candle_buffer:
            prev_to = self._candle_buffer[-1].to_ts
            if c.from_ts != prev_to:
                self._continuity_gaps_count += 1
                logger.warning(
                    "Temporal continuity gap detected: prev_to=%d, current_from=%d",
                    prev_to,
                    c.from_ts,
                )
                return

        self._seen_boundaries.add(c.to_ts)
        self._candle_buffer.append(c)
        if len(self._candle_buffer) > self.config.max_history_bars:
            self._candle_buffer.pop(0)

        self._valid_candles_count += 1
        self._last_verified_candle_ts = c.to_ts

        # 2. Delayed outcome resolution
        self.evaluator.resolve_pending(self._candle_buffer, current_time_ts=c.to_ts)

        # 3. Point-in-time prediction generation (if warmup satisfied)
        if len(self._candle_buffer) >= self.config.min_warmup_bars:
            cutoff_ts = c.to_ts
            maturity_ts = cutoff_ts + 60
            maturity_iso = datetime.fromtimestamp(maturity_ts, UTC).isoformat()
            now_iso = datetime.now(UTC).isoformat()

            for model_id, engine in self.engines.items():
                rec = self.records[model_id]
                try:
                    pred_out = engine.predict(self._candle_buffer, decision_ts=cutoff_ts)
                    prob_dict = {"UP": pred_out.probability_up, "DOWN": pred_out.probability_down}

                    pred_rec = ShadowPredictionRecord(
                        session_id=self.config.session_id,
                        prediction_id=f"pred_{model_id}_{cutoff_ts}_{uuid.uuid4().hex[:8]}",
                        model_id=model_id,
                        artifact_hash=rec.bundle_sha256,
                        symbol="EURUSD",
                        timeframe=60,
                        feature_cutoff_ts=cutoff_ts,
                        feature_cutoff_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
                        data_availability_ts=int(event.received_at_utc_ts),
                        data_availability_utc=event.received_at_utc,
                        provenance_class=event.source_provenance,
                        prediction_created_ts=now_iso,
                        probabilities=prob_dict,
                        decision=pred_out.decision.value,
                        confidence_threshold=rec.confidence_threshold,
                        prediction_horizon_bars=1,
                        expected_maturity_ts=maturity_ts,
                        expected_maturity_utc=maturity_iso,
                        feature_fingerprint=pred_out.feature_fingerprint,
                        prediction_fingerprint="",
                        runtime_diagnostics={
                            "latency_ms": event.latency_ms,
                            "inference_duration_ms": pred_out.diagnostic.inference_duration_ms,
                            "is_synthetic": event.is_synthetic,
                            "request_id": event.request_id,
                            "feature_count": pred_out.diagnostic.feature_count,
                        },
                        evidence_class=(
                            "SYNTHETIC_STREAMING_TEST"
                            if self.config.is_synthetic
                            else "PROSPECTIVE_SHADOW"
                        ),
                        simulated_decision_ts=cutoff_ts,
                        executed_wall_clock_utc=now_iso,
                        broker_availability_type="OBSERVED_FEED_RECEIPT",
                        outcome_status=OutcomeStatus.PENDING.value,
                    )
                    # Commit prediction durably before future outcome exists
                    self.journal.record_prediction(pred_rec)

                except EngineError as exc:
                    logger.error(
                        "Prediction engine error for model %s at ts=%d: %s",
                        model_id,
                        cutoff_ts,
                        exc,
                    )

        # 4. Update model metrics
        self._update_model_states()
        self.publish_progress()

    def _update_model_states(self) -> None:
        """Update metrics and counters for each model under observation."""
        for model_id, state in self.states.items():
            summary = self.evaluator.compute_summary(
                model_id=model_id,
                is_functional_test=self.config.is_synthetic,
            )
            state.total_predictions = summary.total_predictions
            state.pending_predictions = summary.pending_count
            state.resolved_outcomes = summary.resolved_count
            state.trades_taken = summary.trade_count
            state.skips_count = summary.skipped_abstain_count
            state.wins_count = summary.win_count
            state.losses_count = summary.loss_count
            state.ties_count = summary.tie_count
            state.invalid_count = summary.invalid_discontinuity_count
            state.win_rate = summary.win_rate
            state.win_rate_decisive = summary.win_rate_decisive
            state.brier_score = summary.brier_score
            state.directional_accuracy = summary.directional_accuracy
            state.coverage = summary.coverage
            state.abstention_rate = summary.abstention_rate
            state.wilson_ci_decisive = (
                summary.wilson_ci_lower_95_decisive,
                summary.wilson_ci_upper_95_decisive,
            )
            state.expectancy_binary_pct = summary.expectancy_binary_pct
            state.expectancy_digital_pct = summary.expectancy_digital_pct
            state.break_even_gap_binary = round(
                summary.win_rate_decisive - summary.break_even_win_rate, 4
            )

            # Matched baseline comparison
            maj_comp = next(
                (b for b in summary.baseline_comparisons if b.baseline_name == "majority_class"),
                None,
            )
            state.matched_baseline_accuracy = (
                maj_comp.decisive_directional_accuracy if maj_comp is not None else None
            )

    def get_progress(self, *, connection_status: str = "OBSERVING") -> ObservationProgress:
        """Generate current operational progress snapshot."""
        now_mono = time.monotonic()
        elapsed = min(
            self.config.duration_minutes * 60.0,
            max(0.0, now_mono - self._started_at_mono) if self._started_at_mono > 0 else 0.0,
        )
        remaining = max(0.0, self._deadline_mono - now_mono) if self._deadline_mono > 0 else 0.0

        return ObservationProgress(
            session_id=self.config.session_id,
            connection_status=connection_status,
            elapsed_seconds=round(elapsed, 1),
            remaining_seconds=round(remaining, 1),
            valid_candles_count=self._valid_candles_count,
            duplicate_boundaries_count=self._duplicate_boundaries_count,
            continuity_gaps_count=self._continuity_gaps_count,
            stale_candles_count=self._stale_candles_count,
            last_verified_candle_ts=self._last_verified_candle_ts,
            last_candle_received_utc=self._last_received_utc,
            models=self.states,
            termination_reason=None,
            disclaimer=DISCLAIMER,
        )

    def publish_progress(
        self,
        *,
        is_final: bool = False,
        termination_reason: str | None = None,
    ) -> None:
        """Write atomic progress update to state.json and optional status_path."""
        conn_status = "COMPLETED" if is_final else "OBSERVING"
        prog = self.get_progress(connection_status=conn_status)
        prog.termination_reason = termination_reason

        state_dict = prog.to_dict()
        state_file = self.session_dir / "state.json"
        state_file.write_text(json.dumps(state_dict, indent=2, sort_keys=True), encoding="utf-8")

        if self.config.status_path is not None:
            self.config.status_path.write_text(
                json.dumps(state_dict, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    async def run(
        self,
        stream: MarketStreamProtocol,
        *,
        progress_callback: Any = None,
    ) -> ObservationSessionResult:
        """Execute the supervised observation session over the provided market stream."""
        self._started_at_wall = time.time()
        self._started_at_mono = time.monotonic()
        self._deadline_mono = self._started_at_mono + (self.config.duration_minutes * 60.0)

        started_iso = datetime.fromtimestamp(self._started_at_wall, UTC).isoformat()
        termination_reason = "COMPLETED_DURATION_LIMIT"

        await stream.connect()
        try:
            while time.monotonic() < self._deadline_mono:
                # Check for cancellation
                if self._is_cancelled:
                    termination_reason = "OPERATOR_CANCELLED"
                    break
                if self.config.cancel_file is not None and self.config.cancel_file.exists():
                    termination_reason = "OPERATOR_CANCELLED"
                    break

                event = await stream.get_next_event()
                if event is None:
                    # Stream exhausted (EOF or duration reached)
                    if time.monotonic() >= self._deadline_mono:
                        termination_reason = "COMPLETED_DURATION_LIMIT"
                    else:
                        termination_reason = "STREAM_COMPLETED_EOF"
                    break

                self.process_candle_event(event)
                if progress_callback is not None:
                    progress_callback(self.get_progress())

        except Exception as exc:
            termination_reason = f"FAILED_{type(exc).__name__}"
            logger.error(
                "Observation session encountered unexpected exception: %s",
                exc,
                exc_info=True,
            )
            raise
        finally:
            await stream.disconnect()

        # Final outcome resolution pass
        self.evaluator.resolve_pending(self._candle_buffer)
        self._update_model_states()

        ended_wall = time.time()
        ended_iso = datetime.fromtimestamp(ended_wall, UTC).isoformat()
        actual_duration = max(0.0, ended_wall - self._started_at_wall)

        # Build final evaluation summaries per model
        summaries: dict[str, EvaluationSummary] = {}
        for m in self.config.model_ids:
            summaries[m] = self.evaluator.compute_summary(
                model_id=m,
                is_functional_test=self.config.is_synthetic,
            )

        # Generate cryptographic manifest
        manifest_data = {
            "session_id": self.config.session_id,
            "started_at_utc": started_iso,
            "ended_at_utc": ended_iso,
            "actual_duration_seconds": round(actual_duration, 2),
            "requested_minutes": self.config.duration_minutes,
            "termination_reason": termination_reason,
            "is_synthetic": self.config.is_synthetic,
            "total_candles_processed": self._valid_candles_count,
            "duplicate_boundaries_count": self._duplicate_boundaries_count,
            "continuity_gaps_count": self._continuity_gaps_count,
            "stale_candles_count": self._stale_candles_count,
            "summaries": {m: s.to_dict() for m, s in summaries.items()},
            "disclaimer": DISCLAIMER,
        }
        manifest_json_bytes = json.dumps(manifest_data, indent=2, sort_keys=True).encode("utf-8")
        manifest_sha = hashlib.sha256(manifest_json_bytes).hexdigest()

        manifest_file = self.session_dir / "manifest.json"
        manifest_file.write_bytes(manifest_json_bytes)

        # Finalize session.json
        meta_file = self.session_dir / "session.json"
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            meta["ended_at_utc"] = ended_iso
            meta["actual_duration_seconds"] = round(actual_duration, 2)
            meta["termination_reason"] = termination_reason
            meta["manifest_sha256"] = manifest_sha
            meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        self.publish_progress(is_final=True, termination_reason=termination_reason)

        return ObservationSessionResult(
            session_id=self.config.session_id,
            started_at_utc=started_iso,
            ended_at_utc=ended_iso,
            actual_duration_seconds=actual_duration,
            requested_minutes=self.config.duration_minutes,
            termination_reason=termination_reason,
            is_synthetic=self.config.is_synthetic,
            total_candles_processed=self._valid_candles_count,
            duplicate_boundaries_count=self._duplicate_boundaries_count,
            continuity_gaps_count=self._continuity_gaps_count,
            stale_candles_count=self._stale_candles_count,
            summaries=summaries,
            manifest_sha256=manifest_sha,
            disclaimer=DISCLAIMER,
        )
