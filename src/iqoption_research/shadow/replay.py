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
)
from iqoption_research.shadow.registry import ShadowModelRegistry

logger = logging.getLogger(__name__)


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
        if not self.session_id:
            now_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            safe_model = self.model_id.replace("candidate_", "").split("_")[0]
            object.__setattr__(self, "session_id", f"sreplay_{safe_model}_{now_str}")

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "session_dir": str(self.session_dir),
            "total_candles_processed": self.total_candles_processed,
            "predictions_recorded": self.predictions_recorded,
            "outcomes_resolved": self.outcomes_resolved,
            "summary": self.summary.to_dict(),
            "is_functional_test": self.is_functional_test,
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

        # Check training cutoff overlap
        first_candle_ts = candles[0].from_ts
        is_functional_test = False
        overlap_warning = None
        if first_candle_ts <= frozen_record.data_cutoff_ts:
            is_functional_test = True
            overlap_warning = (
                f"REPLAY DATA OVERLAPS TRAINING EXPOSURE (data starts at {first_candle_ts}, "
                f"training cutoff was {frozen_record.data_cutoff_ts}). "
                "Designated as FUNCTIONAL_REPLAY_TEST_ONLY; cannot be cited as out-of-sample edge."
            )
            logger.info("Replay designated as functional test: %s", overlap_warning)

        # 2. Setup session directory and journal
        session_dir = self.sessions_base_dir / config.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        journal = ShadowPredictionJournal(config.session_id, session_dir=session_dir)
        evaluator = ShadowOutcomeEvaluator(
            journal,
            assumed_payout=config.assumed_payout,
            min_qualifying_trades=config.min_qualifying_trades,
        )

        total_candles = len(candles)
        if config.max_candles is not None and config.max_candles > 0:
            total_candles = min(total_candles, config.max_candles)

        lookback_bars = max(config.window_lookback_bars, engine.min_warmup_bars + 5)
        rolling_buffer: list[Candle] = []

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
                    logger.warning("Unexpected prediction error at index %d: %s", i, exc)

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
        summary = evaluator.compute_summary()
        now_utc = datetime.now(UTC).isoformat()

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
            "is_functional_test": is_functional_test,
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
            summary=summary,
            is_functional_test=is_functional_test,
            data_cutoff_overlap_warning=overlap_warning,
            completed_at_utc=now_utc,
            disclaimer=summary.scientific_disclaimer,
        )
