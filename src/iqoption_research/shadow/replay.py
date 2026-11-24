"""Deterministic Historical Replay Engine for Shadow Evaluation (Phase 40A).

Feeds historical candles chronologically through a sliding causal window to simulate
point-in-time prospective prediction and delayed outcome resolution without
future information leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.models.market import Candle
from iqoption_research.research.prospective import digest
from iqoption_research.shadow.engine import (
    PointInTimeViolationError,
    ShadowPredictionEngine,
    WarmupIncompleteError,
)
from iqoption_research.shadow.evaluator import (
    EvaluationSummary,
    ShadowOutcomeEvaluator,
)
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
    validate_storage_identifier,
)
from iqoption_research.shadow.registry import ShadowModelRegistry

logger = logging.getLogger(__name__)


class ReplayError(ValueError):
    """Base exception for replay errors."""


class SessionAlreadyCompletedError(ReplayError):
    """Raised when attempting to reuse an already completed session ID."""


class SessionIntegrityError(ReplayError):
    """Raised when session metadata or journal digests fail verification."""


class ReplayCancelledError(Exception):
    """Raised when historical replay is cancelled by operator."""


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Configuration for an offline historical replay evaluation run."""

    model_id: str
    session_id: str = ""
    snapshot_id: str = ""
    max_candles: int | None = None
    window_lookback_bars: int = 60
    assumed_payout: float = 0.85
    min_qualifying_trades: int = 100
    evidence_class: str = "OFFLINE_REPLAY"
    notes: str = ""

    def __post_init__(self) -> None:
        validate_storage_identifier(self.model_id)
        if not self.session_id:
            now_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            safe_model = self.model_id.replace("candidate_", "").split("_")[0]
            object.__setattr__(self, "session_id", f"sreplay_{safe_model}_{now_str}")
        validate_storage_identifier(self.session_id)
        if self.evidence_class != "OFFLINE_REPLAY":
            raise ReplayError(
                f"Historical replay cannot be labeled {self.evidence_class!r}. "
                "Only OFFLINE_REPLAY is authorized for historical evaluations."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Completed result of an offline historical replay run."""

    session_id: str
    model_id: str
    session_dir: Path
    total_candles_processed: int
    predictions_recorded: int
    outcomes_resolved: int
    summary: EvaluationSummary
    is_functional_test: bool
    data_cutoff_overlap_warning: str | None
    completed_at_utc: str
    disclaimer: str
    prediction_failures_count: int = 0
    exposure_classification: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "session_dir": str(self.session_dir),
            "total_candles_processed": self.total_candles_processed,
            "predictions_recorded": self.predictions_recorded,
            "outcomes_resolved": self.outcomes_resolved,
            "prediction_failures_count": self.prediction_failures_count,
            "summary": self.summary.to_dict(),
            "is_functional_test": self.is_functional_test,
            "exposure_classification": self.exposure_classification,
            "data_cutoff_overlap_warning": self.data_cutoff_overlap_warning,
            "completed_at_utc": self.completed_at_utc,
            "disclaimer": self.disclaimer,
        }


class HistoricalReplayEngine:
    """Orchestrates deterministic chronological replay for shadow model evaluation."""

    def __init__(
        self,
        registry: ShadowModelRegistry,
        *,
        root: Path | None = None,
        sessions_base_dir: Path | None = None,
    ) -> None:
        self.registry = registry
        self.root = root.resolve() if root is not None else Path().resolve()
        self.sessions_base_dir = (
            sessions_base_dir.resolve()
            if sessions_base_dir is not None
            else (self.root / DEFAULT_SHADOW_SESSIONS_DIR).resolve()
        )
        self.sessions_base_dir.mkdir(parents=True, exist_ok=True)

    def run_replay(
        self,
        config: ReplayConfig,
        candles: Sequence[Candle],
        *,
        progress_cb: Callable[[int, str], None] | None = None,
        progress_callback: Callable[..., None] | None = None,
        cancel_token: Callable[[], bool] | None = None,
    ) -> ReplayResult:
        """Run leak-free chronological historical replay on the provided candles.

        Feeds candles strictly one-by-one, keeping memory bounded by retaining only
        a rolling lookback buffer for feature computation. Resolves delayed outcomes
        as future candles become available.
        """
        if not candles:
            raise ValueError("No candles provided for replay")

        # 1. Initialize prediction engine with verified frozen model
        engine = ShadowPredictionEngine(self.registry, config.model_id)
        frozen_record = engine.record

        # Check multi-tier model exposure boundaries
        first_candle_ts = candles[0].from_ts
        is_functional_test = False
        exposure_classification = "OUT_OF_DATASET_RETROSPECTIVE"
        overlap_warning = None

        t_cutoff = frozen_record.training_cutoff_ts or frozen_record.data_cutoff_ts
        val_cutoff = frozen_record.validation_cutoff_ts
        holdout_cutoff = frozen_record.holdout_cutoff_ts
        snap_end = frozen_record.source_snapshot_end_ts

        if first_candle_ts <= t_cutoff:
            is_functional_test = True
            exposure_classification = "OVERLAPS_TRAINING_EXPOSURE"
            overlap_warning = (
                f"REPLAY DATA OVERLAPS TRAINING EXPOSURE (data starts at {first_candle_ts}, "
                f"training cutoff was {t_cutoff}). "
                "Designated as FUNCTIONAL_REPLAY_TEST_ONLY; cannot be cited as out-of-sample edge."
            )
        elif val_cutoff and first_candle_ts <= val_cutoff:
            is_functional_test = True
            exposure_classification = "OVERLAPS_VALIDATION_EXPOSURE"
            overlap_warning = (
                f"REPLAY DATA OVERLAPS VALIDATION/CALIBRATION EXPOSURE "
                f"(data starts at {first_candle_ts}, validation cutoff was {val_cutoff}). "
                "Designated as FUNCTIONAL_REPLAY_TEST_ONLY; model calibration used this window."
            )
        elif holdout_cutoff and first_candle_ts <= holdout_cutoff:
            is_functional_test = True
            exposure_classification = "OVERLAPS_HOLDOUT_EXPOSURE"
            overlap_warning = (
                f"REPLAY DATA OVERLAPS HOLDOUT TEST EXPOSURE (data starts at {first_candle_ts}, "
                f"holdout cutoff was {holdout_cutoff}). "
                "Designated as FUNCTIONAL_REPLAY_TEST_ONLY; holdout evaluation used this window."
            )
        elif snap_end and first_candle_ts <= snap_end:
            is_functional_test = True
            exposure_classification = "OVERLAPS_SOURCE_SNAPSHOT_EXPOSURE"
            overlap_warning = (
                f"REPLAY DATA OVERLAPS SOURCE DATASET SNAPSHOT (data starts at {first_candle_ts}, "
                f"snapshot end was {snap_end}). "
                "Designated as FUNCTIONAL_REPLAY_TEST_ONLY; data is within research archive."
            )
        else:
            # Replay data occurs after source dataset snapshot
            # All historical replays remain retrospective and must NEVER claim prospective evidence!
            is_functional_test = False
            exposure_classification = "POST_SNAPSHOT_HISTORICAL_REPLAY"
            overlap_warning = (
                f"HISTORICAL RETROSPECTIVE REPLAY ONLY (data starts at {first_candle_ts}). "
                "Replay data occurs after source dataset snapshot, but historical replay "
                "DOES NOT CONSTITUTE PROSPECTIVE OBSERVATION. Zero execution authority."
            )

        logger.info(
            "Replay exposure classification: %s (is_functional_test=%s)",
            exposure_classification,
            is_functional_test,
        )

        # 2. Setup session directory and journal
        session_dir = self.sessions_base_dir / config.session_id
        session_file = session_dir / "session.json"
        if session_file.is_file():
            try:
                existing_data = json.loads(session_file.read_text(encoding="utf-8"))
                if "summary" in existing_data:
                    raise SessionAlreadyCompletedError(
                        f"Replay session '{config.session_id}' already completed. "
                        "Unsafe reuse of completed session IDs is prohibited."
                    )
            except (json.JSONDecodeError, OSError):
                pass

        session_dir.mkdir(parents=True, exist_ok=True)
        journal = ShadowPredictionJournal(config.session_id, session_dir=session_dir)
        evaluator = ShadowOutcomeEvaluator(
            journal,
            assumed_payout=config.assumed_payout,
            min_qualifying_trades=config.min_qualifying_trades,
            frozen_majority_direction=frozen_record.training_majority_direction,
            frozen_majority_prob_up=frozen_record.training_majority_prob_up,
            baseline_comparison_available=frozen_record.baseline_comparison_available,
            baseline_source_id=frozen_record.baseline_source_model_id,
            baseline_source_sha256=frozen_record.baseline_source_sha256,
            exposure_overlap_classification=exposure_classification,
        )

        total_candles = len(candles)
        if config.max_candles is not None and config.max_candles > 0:
            total_candles = min(total_candles, config.max_candles)

        lookback_bars = max(
            config.window_lookback_bars,
            engine.min_warmup_bars + 5 + int(frozen_record.horizon_bars),
        )
        rolling_buffer: list[Candle] = []
        prediction_failures_count = 0
        prediction_failures: list[dict[str, Any]] = []

        logger.info(
            "Starting historical replay session %s with %d candles for model %s",
            config.session_id,
            total_candles,
            config.model_id,
        )

        # 3. Chronological Replay Loop
        for i in range(total_candles):
            if cancel_token is not None and cancel_token():
                logger.info("Replay cancelled by token at candle index %d", i)
                raise ReplayCancelledError("Replay cancelled by operator")

            curr_candle = candles[i]
            rolling_buffer.append(curr_candle)
            if len(rolling_buffer) > lookback_bars:
                rolling_buffer.pop(0)

            # A. Attempt delayed outcome resolution for any previously committed predictions
            evaluator.resolve_pending(rolling_buffer, current_time_ts=curr_candle.to_ts)

            # B. Check if we have enough warmup bars to generate a prediction
            if len(rolling_buffer) >= engine.min_warmup_bars:
                try:
                    pred_output = engine.predict(
                        rolling_buffer,
                        decision_ts=curr_candle.to_ts,
                    )

                    pred_id = (
                        f"pred_{config.model_id[:16]}_{curr_candle.to_ts}_"
                        f"{hashlib.sha256(f'{config.session_id}_{i}'.encode()).hexdigest()[:8]}"
                    )
                    pred_created_utc = datetime.now(UTC).isoformat()
                    expected_mat_ts = curr_candle.to_ts + (frozen_record.horizon_bars * 60)
                    expected_mat_utc = datetime.fromtimestamp(expected_mat_ts, UTC).isoformat()
                    data_avail_utc = datetime.fromtimestamp(curr_candle.to_ts, UTC).isoformat()

                    rec = ShadowPredictionRecord(
                        session_id=config.session_id,
                        prediction_id=pred_id,
                        model_id=config.model_id,
                        artifact_hash=frozen_record.bundle_sha256,
                        symbol="EURUSD",
                        timeframe=60,
                        feature_cutoff_ts=curr_candle.to_ts,
                        feature_cutoff_utc=datetime.fromtimestamp(
                            curr_candle.to_ts, UTC
                        ).isoformat(),
                        data_availability_ts=curr_candle.to_ts,
                        data_availability_utc=data_avail_utc,
                        provenance_class=config.snapshot_id or "HISTORICAL_REPLAY",
                        prediction_created_ts=pred_created_utc,
                        simulated_decision_ts=curr_candle.to_ts,
                        executed_wall_clock_utc=pred_created_utc,
                        broker_availability_type="HISTORICAL_THEORETICAL_CLOSE",
                        probabilities={
                            "UP": pred_output.probability_up,
                            "DOWN": pred_output.probability_down,
                        },
                        decision=pred_output.decision.value,
                        confidence_threshold=pred_output.confidence_threshold,
                        prediction_horizon_bars=frozen_record.horizon_bars,
                        expected_maturity_ts=expected_mat_ts,
                        expected_maturity_utc=expected_mat_utc,
                        feature_fingerprint=pred_output.feature_fingerprint,
                        prediction_fingerprint="",
                        runtime_diagnostics=pred_output.diagnostic.to_dict(),
                        evidence_class=config.evidence_class,
                    )
                    journal.record_prediction(rec)
                except (WarmupIncompleteError, PointInTimeViolationError) as exc:
                    logger.debug("Skipped prediction at index %d: %s", i, exc)
                except Exception as exc:  # noqa: BLE001
                    prediction_failures_count += 1
                    failure_info = {
                        "candle_index": i,
                        "candle_cutoff_ts": curr_candle.to_ts,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    prediction_failures.append(failure_info)
                    logger.warning("Prediction failure at index %d: %s", i, exc)

            if progress_callback is not None and (i % 25 == 0 or i == total_candles - 1):
                try:
                    progress_callback(i + 1, total_candles)
                except TypeError:
                    progress_callback(
                        int((i + 1) / total_candles * 100),
                        f"Replaying candle {i + 1}/{total_candles}",
                    )
            elif progress_cb is not None and (i % 25 == 0 or i == total_candles - 1):
                pct = int((i + 1) / total_candles * 100)
                msg = (
                    f"Replaying candle {i + 1}/{total_candles} "
                    f"(predictions={journal.total_count}, resolved={journal.resolved_count})"
                )
                progress_cb(pct, msg)

        # 4. Final outcome resolution sweep on remaining buffered candles
        evaluator.resolve_pending(rolling_buffer)

        # 5. Compute summary and persist session.json
        summary = evaluator.compute_summary(is_functional_test=is_functional_test)
        now_utc = datetime.now(UTC).isoformat()

        # Compute SHA-256 hashes of predictions and outcomes files for evidence integrity
        pred_bytes = (
            journal.predictions_file.read_bytes() if journal.predictions_file.is_file() else b""
        )
        out_bytes = journal.outcomes_file.read_bytes() if journal.outcomes_file.is_file() else b""
        predictions_sha256 = hashlib.sha256(pred_bytes).hexdigest()
        outcomes_sha256 = hashlib.sha256(out_bytes).hexdigest()

        session_meta = {
            "session_id": config.session_id,
            "created_at_utc": now_utc,
            "evidence_class": config.evidence_class,
            "execution_authority": "NONE",
            "config": config.to_dict(),
            "frozen_model": frozen_record.to_dict(),
            "total_candles_processed": total_candles,
            "predictions_recorded": journal.total_count,
            "outcomes_resolved": journal.resolved_count,
            "outcomes_pending": journal.pending_count,
            "predictions_sha256": predictions_sha256,
            "outcomes_sha256": outcomes_sha256,
            "prediction_failures_count": prediction_failures_count,
            "prediction_failures": prediction_failures,
            "is_functional_test": is_functional_test,
            "exposure_classification": exposure_classification,
            "data_cutoff_overlap_warning": overlap_warning,
            "summary": summary.to_dict(),
            "disclaimer": summary.scientific_disclaimer,
        }
        clean_meta = {k: v for k, v in session_meta.items() if k != "session_sha256"}
        session_meta["session_sha256"] = digest(clean_meta)

        (session_dir / "session.json").write_text(
            json.dumps(session_meta, indent=2, sort_keys=True), encoding="utf-8"
        )

        return ReplayResult(
            session_id=config.session_id,
            model_id=config.model_id,
            session_dir=session_dir,
            total_candles_processed=total_candles,
            predictions_recorded=journal.total_count,
            outcomes_resolved=journal.resolved_count,
            prediction_failures_count=prediction_failures_count,
            summary=summary,
            is_functional_test=is_functional_test,
            exposure_classification=exposure_classification,
            data_cutoff_overlap_warning=overlap_warning,
            completed_at_utc=now_utc,
            disclaimer=summary.scientific_disclaimer,
        )

    def verify_session(self, session_id: str) -> dict[str, Any]:
        """Verify session metadata against journal files and compute integrity status.

        Does not modify or delete any files. Classifies older artifacts lacking new
        digests as LEGACY_UNVERIFIED_EVIDENCE.
        """
        validate_storage_identifier(session_id)
        session_dir = self.sessions_base_dir / session_id
        session_file = session_dir / "session.json"
        if not session_file.is_file():
            return {
                "session_id": session_id,
                "status": "MISSING_SESSION_METADATA",
                "verified": False,
                "error": f"Session file not found: {session_file}",
            }

        try:
            meta = json.loads(session_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return {
                "session_id": session_id,
                "status": "CORRUPT_SESSION_METADATA",
                "verified": False,
                "error": f"Failed to parse session metadata: {exc}",
            }

        recorded_session_sha = meta.get("session_sha256")
        clean_meta = {k: v for k, v in meta.items() if k != "session_sha256"}
        computed_session_sha = digest(clean_meta)
        if recorded_session_sha != computed_session_sha:
            return {
                "session_id": session_id,
                "status": "SESSION_METADATA_TAMPERED",
                "verified": False,
                "error": "Session metadata SHA-256 mismatch",
            }

        predictions_file = session_dir / "predictions.jsonl"
        outcomes_file = session_dir / "outcomes.jsonl"

        # Check if legacy session without journal sha256 fields
        if "predictions_sha256" not in meta:
            return {
                "session_id": session_id,
                "status": "LEGACY_UNVERIFIED_EVIDENCE",
                "verified": False,
                "note": (
                    "Older session artifact lacks cryptographic journal checksums. "
                    "Preserved without modification."
                ),
                "session_meta": meta,
            }

        # Verify predictions.jsonl checksum
        if not predictions_file.is_file():
            return {
                "session_id": session_id,
                "status": "MISSING_PREDICTIONS_JOURNAL",
                "verified": False,
                "error": f"Missing {predictions_file}",
            }

        actual_pred_sha = hashlib.sha256(predictions_file.read_bytes()).hexdigest()
        if actual_pred_sha != meta.get("predictions_sha256"):
            return {
                "session_id": session_id,
                "status": "PREDICTIONS_JOURNAL_TAMPERED",
                "verified": False,
                "error": (
                    f"Predictions checksum mismatch: expected {meta.get('predictions_sha256')}, "
                    f"got {actual_pred_sha}"
                ),
            }

        # Verify outcomes.jsonl checksum and presence
        has_outcomes_digest = "outcomes_sha256" in meta
        expected_out_sha = meta.get("outcomes_sha256")
        outcomes_resolved = meta.get("outcomes_resolved", 0)

        if has_outcomes_digest or outcomes_resolved > 0:
            if not outcomes_file.is_file():
                return {
                    "session_id": session_id,
                    "status": "MISSING_OUTCOMES_JOURNAL",
                    "verified": False,
                    "error": f"Missing outcomes journal at {outcomes_file}",
                }
            actual_out_sha = hashlib.sha256(outcomes_file.read_bytes()).hexdigest()
            if has_outcomes_digest and actual_out_sha != expected_out_sha:
                return {
                    "session_id": session_id,
                    "status": "OUTCOMES_JOURNAL_TAMPERED",
                    "verified": False,
                    "error": (
                        f"Outcomes checksum mismatch: expected {expected_out_sha}, "
                        f"got {actual_out_sha}"
                    ),
                }
            if outcomes_resolved > 0 and outcomes_file.stat().st_size == 0:
                return {
                    "session_id": session_id,
                    "status": "OUTCOMES_JOURNAL_EMPTY_INCONSISTENT",
                    "verified": False,
                    "error": "Outcomes journal is empty but metadata claims resolved outcomes",
                }
            if outcomes_resolved == 0 and outcomes_file.stat().st_size > 0:
                return {
                    "session_id": session_id,
                    "status": "OUTCOMES_JOURNAL_UNEXPECTED_CONTENT",
                    "verified": False,
                    "error": (
                        "Outcomes journal contains records but metadata claims 0 resolved outcomes"
                    ),
                }
        else:
            # Genuinely empty outcomes journal (0 outcomes resolved, no outcomes digest recorded)
            if outcomes_file.is_file() and outcomes_file.stat().st_size > 0:
                return {
                    "session_id": session_id,
                    "status": "OUTCOMES_JOURNAL_UNEXPECTED_CONTENT",
                    "verified": False,
                    "error": (
                        "Outcomes journal contains records but metadata claims 0 resolved outcomes"
                    ),
                }

        return {
            "session_id": session_id,
            "status": "VERIFIED",
            "verified": True,
            "evidence_class": meta.get("evidence_class", "UNKNOWN"),
            "model_id": meta.get("config", {}).get("model_id", ""),
            "predictions_recorded": meta.get("predictions_recorded", 0),
            "outcomes_resolved": meta.get("outcomes_resolved", 0),
            "is_functional_test": meta.get("is_functional_test", False),
            "session_meta": meta,
        }
