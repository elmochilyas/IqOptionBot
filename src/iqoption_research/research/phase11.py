"""Phase 11 historical expansion planning and confirmation-period research.

All planning and evaluation functions are offline.  The only object capable of
fetching broker data is the returned Phase 9 ``BackfillJob``; callers must still
provide an explicitly authorized read-only broker client.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from iqoption_research.market_data.collector import (
    BACKFILL_SOURCE,
    JOB_ID_PATTERN,
    BackfillCheckpoint,
    BackfillJob,
    DataAcquisitionError,
    HistoricalBroker,
    HistoricalDataCollector,
)
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.market_data.sessions import GapClassification, get_market_schedule
from iqoption_research.market_data.validation import find_gaps
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.models.base import BaseMLModel
from iqoption_research.ml.phase10 import (
    AuthenticDataAudit,
    CleanResearchDataset,
    ModelPerformance,
    Phase10Config,
    Phase10Error,
    _simulate_trades,
    audit_authentic_datasets,
    evaluate_clean_dataset,
    prepare_clean_dataset,
)
from iqoption_research.ml.pipeline import create_model
from iqoption_research.models.market import Candle
from iqoption_research.research.features.scalers import StandardScaler

DEFAULT_TARGET_DAYS: Final[int] = 90
DEFAULT_CONFIRMATION_DAYS: Final[int] = 14
PHASE10_EVALUATION_CUTOFF_TS: Final[int] = 1_794_607_200
PHASE10_EVALUATED_DATASET_FINGERPRINT: Final[str] = (
    "70d8b1e906d5b76f97b93346ad4399e23abe1630dca66e14b1a1828a617e6632"
)
CONFIRMATION_STATE_VERSION: Final[int] = 2
DEFAULT_COST_SCENARIOS: Final[tuple[tuple[float, float], ...]] = (
    (0.5, 0.0),
    (1.0, 0.1),
    (1.5, 0.25),
    (2.0, 0.5),
)


@dataclass(frozen=True, slots=True)
class CoverageGapAudit:
    """Closure-aware description of one missing interval."""

    expected_start_ts: int
    actual_start_ts: int
    expected_start_utc: str
    actual_start_utc: str
    missing_candles: int
    duration_seconds: int
    classification: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExpansionCoverageAudit:
    """True unique coverage with market-closure classifications."""

    base: AuthenticDataAudit
    oldest_utc: str
    newest_utc: str
    target_calendar_days: int
    meets_target: bool
    scheduled_closure_count: int
    unexpected_gap_count: int
    unverified_gap_count: int
    gaps: tuple[CoverageGapAudit, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "oldest_utc": self.oldest_utc,
            "newest_utc": self.newest_utc,
            "target_calendar_days": self.target_calendar_days,
            "meets_target": self.meets_target,
            "scheduled_closure_count": self.scheduled_closure_count,
            "unexpected_gap_count": self.unexpected_gap_count,
            "unverified_gap_count": self.unverified_gap_count,
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


@dataclass(frozen=True, slots=True)
class BackwardExpansionPlan:
    """Immutable plan for extending strictly before current verified history."""

    symbol: str
    candle_size_seconds: int
    source: str
    current_oldest_ts: int
    current_newest_ts: int
    target_oldest_ts: int
    target_calendar_days: int
    current_unique_candles: int
    max_new_candles: int
    page_size: int
    max_broker_requests: int
    existing_dataset_ids: tuple[str, ...]
    source_fingerprint: str
    needs_collection: bool

    @property
    def current_oldest_utc(self) -> str:
        return datetime.fromtimestamp(self.current_oldest_ts, tz=UTC).isoformat()

    @property
    def target_oldest_utc(self) -> str:
        return datetime.fromtimestamp(self.target_oldest_ts, tz=UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["existing_dataset_ids"] = list(self.existing_dataset_ids)
        payload["current_oldest_utc"] = self.current_oldest_utc
        payload["target_oldest_utc"] = self.target_oldest_utc
        return payload


@dataclass(frozen=True, slots=True)
class CostSensitivityResult:
    spread_pips: float
    slippage_pips: float
    trade_count: int
    execution_costs: float
    net_pnl: float
    return_on_capital: float
    max_drawdown_percent: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfirmationProtocolError(Phase10Error):
    """Raised when an action would invalidate independent confirmation evidence."""


@dataclass(frozen=True, slots=True)
class ConfirmationProtocolState:
    """Durable, append-only-in-meaning record of independent confirmation use."""

    schema_version: int
    prior_evaluation_cutoff_ts: int
    prior_evaluation_cutoff_utc: str
    prior_evaluated_dataset_fingerprint: str
    confirmation_days: int
    protocol_status: str
    development_freeze_ts: int | None = None
    development_freeze_utc: str | None = None
    development_dataset_fingerprint: str | None = None
    frozen_model_name: str | None = None
    model_frozen_at_utc: str | None = None
    confirmation_start_ts: int | None = None
    confirmation_end_ts: int | None = None
    confirmation_dataset_fingerprint: str | None = None
    full_dataset_fingerprint_at_evaluation: str | None = None
    evaluation_started_at_utc: str | None = None
    evaluated_at_utc: str | None = None
    audit_history: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConfirmationProtocolState:
        state = cls(**payload)
        if state.schema_version != CONFIRMATION_STATE_VERSION:
            raise ConfirmationProtocolError(
                f"Unsupported confirmation state version {state.schema_version}."
            )
        if state.protocol_status not in {
            "DEVELOPMENT_OPEN",
            "MODEL_FROZEN",
            "CONFIRMATION_STARTED",
            "CONFIRMATION_EVALUATED",
        }:
            raise ConfirmationProtocolError(
                f"Invalid research protocol status {state.protocol_status!r}."
            )
        return state


@dataclass(frozen=True, slots=True)
class Phase11EvaluationResult:
    """Development/confirmation evidence or an explicit insufficiency result."""

    coverage: ExpansionCoverageAudit
    ready_for_confirmation: bool
    confirmation_evaluated: bool
    confirmation_evaluation_status: str
    development_freeze_utc: str | None
    development_dataset_fingerprint: str | None
    frozen_model_name: str | None
    prior_evaluation_cutoff_utc: str
    prior_evaluated_dataset_fingerprint: str
    confirmation_dataset_fingerprint: str | None
    confirmation_days: int
    confirmation_start_utc: str | None
    development_rows: int
    confirmation_rows: int
    development_metrics: dict[str, ModelPerformance]
    selected_model: str | None
    selected_baseline: str | None
    confirmation_metrics: dict[str, ModelPerformance]
    cost_sensitivity: tuple[CostSensitivityResult, ...]
    limitations: tuple[str, ...]
    evaluated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage.to_dict(),
            "ready_for_confirmation": self.ready_for_confirmation,
            "confirmation_evaluated": self.confirmation_evaluated,
            "confirmation_evaluation_status": self.confirmation_evaluation_status,
            "development_freeze_utc": self.development_freeze_utc,
            "development_dataset_fingerprint": self.development_dataset_fingerprint,
            "frozen_model_name": self.frozen_model_name,
            "prior_evaluation_cutoff_utc": self.prior_evaluation_cutoff_utc,
            "prior_evaluated_dataset_fingerprint": (self.prior_evaluated_dataset_fingerprint),
            "confirmation_dataset_fingerprint": self.confirmation_dataset_fingerprint,
            "confirmation_days": self.confirmation_days,
            "confirmation_start_utc": self.confirmation_start_utc,
            "development_rows": self.development_rows,
            "confirmation_rows": self.confirmation_rows,
            "development_metrics": {
                name: metric.to_dict() for name, metric in self.development_metrics.items()
            },
            "selected_model": self.selected_model,
            "selected_baseline": self.selected_baseline,
            "confirmation_metrics": {
                name: metric.to_dict() for name, metric in self.confirmation_metrics.items()
            },
            "cost_sensitivity": [row.to_dict() for row in self.cost_sensitivity],
            "limitations": list(self.limitations),
            "evaluated_at_utc": self.evaluated_at_utc,
        }


def audit_expansion_coverage(
    storage_dir: Path | str,
    *,
    symbol: str = "EURUSD",
    candle_size_seconds: int = 60,
    source: str = BACKFILL_SOURCE,
    target_calendar_days: int = DEFAULT_TARGET_DAYS,
) -> tuple[ExpansionCoverageAudit, list[Candle]]:
    """Verify every source file and classify expected versus unexpected gaps."""
    if target_calendar_days <= 0:
        raise ValueError("target_calendar_days must be positive")
    base, candles = audit_authentic_datasets(
        storage_dir,
        symbol=symbol,
        candle_size_seconds=candle_size_seconds,
        source=source,
    )
    raw_gaps = find_gaps(
        candles,
        candle_size_seconds,
        market_schedule=get_market_schedule(symbol),
    )
    gaps = tuple(
        CoverageGapAudit(
            expected_start_ts=gap.expected_start,
            actual_start_ts=gap.actual_start,
            expected_start_utc=datetime.fromtimestamp(gap.expected_start, tz=UTC).isoformat(),
            actual_start_utc=datetime.fromtimestamp(gap.actual_start, tz=UTC).isoformat(),
            missing_candles=gap.missing_count,
            duration_seconds=gap.duration_seconds,
            classification=gap.classification,
        )
        for gap in raw_gaps
    )
    scheduled = sum(
        gap.classification == GapClassification.SCHEDULED_WEEKEND_CLOSURE for gap in gaps
    )
    unexpected = sum(
        gap.classification == GapClassification.UNSCHEDULED_INTRADAY_GAP for gap in gaps
    )
    unverified = sum(gap.classification == GapClassification.UNVERIFIED_GAP for gap in gaps)
    audit = ExpansionCoverageAudit(
        base=base,
        oldest_utc=datetime.fromtimestamp(candles[0].from_ts, tz=UTC).isoformat(),
        newest_utc=datetime.fromtimestamp(candles[-1].to_ts, tz=UTC).isoformat(),
        target_calendar_days=target_calendar_days,
        meets_target=base.calendar_span_days >= target_calendar_days,
        scheduled_closure_count=scheduled,
        unexpected_gap_count=unexpected,
        unverified_gap_count=unverified,
        gaps=gaps,
    )
    return audit, candles


def build_backward_expansion_plan(
    storage_dir: Path | str,
    *,
    symbol: str = "EURUSD",
    candle_size_seconds: int = 60,
    source: str = BACKFILL_SOURCE,
    target_calendar_days: int = DEFAULT_TARGET_DAYS,
    max_new_candles: int = 150_000,
    page_size: int = 500,
    max_broker_requests: int | None = None,
) -> BackwardExpansionPlan:
    """Plan a bounded job ending exactly at the oldest verified candle."""
    if max_new_candles <= 0:
        raise ValueError("max_new_candles must be positive")
    if not 1 <= page_size <= 1000:
        raise ValueError("page_size must be within 1..1000")
    coverage, _ = audit_expansion_coverage(
        storage_dir,
        symbol=symbol,
        candle_size_seconds=candle_size_seconds,
        source=source,
        target_calendar_days=target_calendar_days,
    )
    base = coverage.base
    target_oldest = base.end_ts - target_calendar_days * 86400
    default_requests = math.ceil(max_new_candles / max(1, page_size - 1)) + 10
    request_budget = max_broker_requests if max_broker_requests is not None else default_requests
    if request_budget <= 0:
        raise ValueError("max_broker_requests must be positive")
    return BackwardExpansionPlan(
        symbol=base.symbol,
        candle_size_seconds=candle_size_seconds,
        source=source,
        current_oldest_ts=base.start_ts,
        current_newest_ts=base.end_ts,
        target_oldest_ts=target_oldest,
        target_calendar_days=target_calendar_days,
        current_unique_candles=base.unique_candles,
        max_new_candles=max_new_candles,
        page_size=page_size,
        max_broker_requests=request_budget,
        existing_dataset_ids=base.dataset_ids,
        source_fingerprint=base.aggregate_fingerprint,
        needs_collection=base.start_ts > target_oldest,
    )


def launch_backward_expansion(
    *,
    collector: HistoricalDataCollector,
    plan: BackwardExpansionPlan,
    job_id: str,
    client: HistoricalBroker,
    max_retries: int = 5,
    backoff_base_seconds: float = 1.0,
) -> BackfillJob:
    """Create a Phase 9 job seeded with all verified catalog candles.

    Seeding the checkpoint with existing immutable dataset IDs makes boundary and
    cross-job overlaps duplicates instead of new records.  The job persists only
    genuinely older candles and can resume through the standard Phase 9 command.
    """
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise DataAcquisitionError("Invalid backfill job ID.")
    if not plan.needs_collection:
        raise DataAcquisitionError("Historical coverage already satisfies the target.")
    checkpoint_path = collector.storage_dir / f"backfill_checkpoint_{job_id}.json"
    if checkpoint_path.exists():
        raise DataAcquisitionError(
            f"Checkpoint already exists for job {job_id}; use backfill-resume."
        )
    current, _ = audit_authentic_datasets(
        collector.storage_dir,
        symbol=plan.symbol,
        candle_size_seconds=plan.candle_size_seconds,
        source=plan.source,
    )
    if (
        current.aggregate_fingerprint != plan.source_fingerprint
        or current.start_ts != plan.current_oldest_ts
        or current.dataset_ids != plan.existing_dataset_ids
    ):
        raise DataAcquisitionError(
            "Catalog changed after expansion planning; generate a fresh coverage plan."
        )
    instrument = InstrumentRegistry.get(plan.symbol)
    checkpoint = BackfillCheckpoint(
        job_id=job_id,
        instrument=plan.symbol,
        active_id=instrument.active_id,
        timeframe_seconds=plan.candle_size_seconds,
        target_count=plan.current_unique_candles + plan.max_new_candles,
        page_size=plan.page_size,
        start_ts=plan.target_oldest_ts,
        end_ts=plan.current_oldest_ts,
        last_cursor_ts=plan.current_oldest_ts,
        unique_candles_persisted=plan.current_unique_candles,
        last_successful_batch=0,
        committed_dataset_ids=plan.existing_dataset_ids,
        max_broker_requests=plan.max_broker_requests,
        complete_on_start_boundary=True,
    )
    return BackfillJob(
        collector=collector,
        checkpoint=checkpoint,
        client=client,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
    )


def _research_candle_fingerprint(candles: list[Candle]) -> str:
    """Hash stable persisted market fields, excluding non-persisted broker metadata."""
    digest = hashlib.sha256()
    for candle in sorted(candles, key=lambda item: item.from_ts):
        values = (
            candle.active_id,
            candle.symbol.upper(),
            int(candle.size),
            candle.from_ts,
            candle.to_ts,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.bid,
            candle.ask,
        )
        digest.update(repr(values).encode("utf-8"))
    return digest.hexdigest()


def _write_confirmation_state(
    state_path: Path,
    state: ConfirmationProtocolState,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(state_path)


def load_or_create_confirmation_state(
    state_path: Path | str,
    *,
    candles: list[Candle],
    confirmation_days: int,
    prior_evaluation_cutoff_ts: int = PHASE10_EVALUATION_CUTOFF_TS,
) -> ConfirmationProtocolState:
    """Load the immutable protocol anchor or establish it from Phase 10 evidence."""
    path = Path(state_path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 1)) == 1:
            legacy_status = str(payload.get("evaluation_status", "RESERVED"))
            reserved_fields = (
                payload.get("confirmation_start_ts"),
                payload.get("confirmation_end_ts"),
                payload.get("confirmation_dataset_fingerprint"),
                payload.get("evaluation_started_at_utc"),
                payload.get("evaluated_at_utc"),
            )
            if legacy_status != "RESERVED" or any(value is not None for value in reserved_fields):
                raise ConfirmationProtocolError(
                    "Legacy protocol contains reserved or consumed confirmation evidence and "
                    "cannot be silently reclassified."
                )
            migrated_at = datetime.now(UTC).isoformat()
            state = ConfirmationProtocolState(
                schema_version=CONFIRMATION_STATE_VERSION,
                prior_evaluation_cutoff_ts=int(payload["prior_evaluation_cutoff_ts"]),
                prior_evaluation_cutoff_utc=str(payload["prior_evaluation_cutoff_utc"]),
                prior_evaluated_dataset_fingerprint=str(
                    payload["prior_evaluated_dataset_fingerprint"]
                ),
                confirmation_days=int(payload["confirmation_days"]),
                protocol_status="DEVELOPMENT_OPEN",
                audit_history=(
                    {
                        "event": "MIGRATED_V1_RESERVED",
                        "at_utc": migrated_at,
                        "legacy_status": legacy_status,
                        "reason": (
                            "The legacy fixed cutoff was previously evaluated Phase 10 history, "
                            "not independent confirmation."
                        ),
                    },
                ),
            )
            _write_confirmation_state(path, state)
        else:
            state = ConfirmationProtocolState.from_dict(payload)
        if state.prior_evaluation_cutoff_ts != prior_evaluation_cutoff_ts:
            raise ConfirmationProtocolError(
                "The persisted Phase 10 cutoff cannot be changed. Use the recorded cutoff."
            )
        if state.confirmation_days != confirmation_days:
            raise ConfirmationProtocolError(
                "The persisted confirmation duration cannot be changed after reservation."
            )
        return state

    prior_evaluated = [candle for candle in candles if candle.from_ts <= prior_evaluation_cutoff_ts]
    if not prior_evaluated:
        raise ConfirmationProtocolError(
            "No verified candles exist at or before the Phase 10 evaluation cutoff."
        )
    recorded_fingerprint = (
        PHASE10_EVALUATED_DATASET_FINGERPRINT
        if prior_evaluation_cutoff_ts == PHASE10_EVALUATION_CUTOFF_TS
        else _research_candle_fingerprint(prior_evaluated)
    )
    state = ConfirmationProtocolState(
        schema_version=CONFIRMATION_STATE_VERSION,
        prior_evaluation_cutoff_ts=prior_evaluation_cutoff_ts,
        prior_evaluation_cutoff_utc=datetime.fromtimestamp(
            prior_evaluation_cutoff_ts, tz=UTC
        ).isoformat(),
        prior_evaluated_dataset_fingerprint=recorded_fingerprint,
        confirmation_days=confirmation_days,
        protocol_status="DEVELOPMENT_OPEN",
        audit_history=(
            {
                "event": "PROTOCOL_CREATED",
                "at_utc": datetime.now(UTC).isoformat(),
                "phase10_cutoff_ts": prior_evaluation_cutoff_ts,
            },
        ),
    )
    _write_confirmation_state(path, state)
    return state


def freeze_development_protocol(
    storage_dir: Path | str,
    *,
    state_path: Path | str,
    selected_model: str,
    target_calendar_days: int = DEFAULT_TARGET_DAYS,
    confirmation_days: int = DEFAULT_CONFIRMATION_DAYS,
    symbol: str = "EURUSD",
    candle_size_seconds: int = 60,
    prior_evaluation_cutoff_ts: int = PHASE10_EVALUATION_CUTOFF_TS,
) -> ConfirmationProtocolState:
    """Irreversibly freeze the current verified development dataset and model name."""
    if not selected_model.strip():
        raise ValueError("selected_model cannot be empty")
    coverage, candles = audit_expansion_coverage(
        storage_dir,
        symbol=symbol,
        candle_size_seconds=candle_size_seconds,
        target_calendar_days=target_calendar_days,
    )
    state = load_or_create_confirmation_state(
        state_path,
        candles=candles,
        confirmation_days=confirmation_days,
        prior_evaluation_cutoff_ts=prior_evaluation_cutoff_ts,
    )
    if state.protocol_status != "DEVELOPMENT_OPEN":
        raise ConfirmationProtocolError("Development has already been frozen.")
    if not coverage.meets_target:
        raise ConfirmationProtocolError(
            f"Development coverage is {coverage.base.calendar_span_days:.2f} days; "
            f"{target_calendar_days} days are required before model freeze."
        )
    frozen_at = datetime.now(UTC).isoformat()
    development_fingerprint = _research_candle_fingerprint(candles)
    event = {
        "event": "MODEL_FROZEN",
        "at_utc": frozen_at,
        "model_name": selected_model.strip(),
        "development_freeze_ts": candles[-1].from_ts,
        "development_dataset_fingerprint": development_fingerprint,
        "unique_candles": coverage.base.unique_candles,
    }
    frozen = replace(
        state,
        protocol_status="MODEL_FROZEN",
        development_freeze_ts=candles[-1].from_ts,
        development_freeze_utc=datetime.fromtimestamp(candles[-1].from_ts, tz=UTC).isoformat(),
        development_dataset_fingerprint=development_fingerprint,
        frozen_model_name=selected_model.strip(),
        model_frozen_at_utc=frozen_at,
        audit_history=(*state.audit_history, event),
    )
    _write_confirmation_state(Path(state_path), frozen)
    return frozen


def partition_confirmation_candles(
    candles: list[Candle],
    *,
    prior_evaluation_cutoff_ts: int,
) -> tuple[list[Candle], list[Candle]]:
    """Separate previously seen development history from strictly later data."""
    if not candles:
        raise Phase10Error("Cannot partition an empty candle series.")
    development = [candle for candle in candles if candle.from_ts <= prior_evaluation_cutoff_ts]
    confirmation = [candle for candle in candles if candle.from_ts > prior_evaluation_cutoff_ts]
    if not development:
        raise Phase10Error("No development candles exist at the recorded Phase 10 cutoff.")
    return development, confirmation


def _predict_confirmation(
    name: str,
    development: CleanResearchDataset,
    confirmation: CleanResearchDataset,
    config: Phase10Config,
) -> np.ndarray:
    if development.feature_names != confirmation.feature_names:
        raise Phase10Error("Development and confirmation feature schemas differ.")
    validation_size = max(100, int(development.row_count * config.validation_ratio))
    validation_start = development.row_count - validation_size
    train_end = validation_start - config.horizon_bars
    if train_end < config.minimum_rows:
        raise Phase10Error("Development period is too small after validation purging.")
    train_X = development.X[:train_end]
    train_y = development.y[:train_end]
    validation_X = development.X[validation_start:]
    validation_y = development.y[validation_start:]
    model: BaseMLModel = create_model(name, seed=config.seed, quick=True)
    if name in {"majority_class", "rule_based_momentum"}:
        model.fit(train_X, train_y, feature_names=development.feature_names)
        return np.asarray(model.predict_proba(confirmation.X)[:, 1], dtype=np.float64)

    scaler = StandardScaler()
    scaler.fit(train_X, feature_names=development.feature_names)
    scaled_train = np.asarray(scaler.transform(train_X), dtype=np.float64)
    scaled_validation = np.asarray(scaler.transform(validation_X), dtype=np.float64)
    scaled_confirmation = np.asarray(scaler.transform(confirmation.X), dtype=np.float64)
    model.fit(scaled_train, train_y, feature_names=development.feature_names)
    probabilities = model.predict_proba(scaled_confirmation)
    if len(np.unique(validation_y)) == 2:
        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(model.predict_proba(scaled_validation), validation_y)
        probabilities = calibrator.calibrate(probabilities)
    return np.asarray(probabilities[:, 1], dtype=np.float64)


def _confirmation_metrics(
    name: str,
    probabilities: np.ndarray,
    confirmation: CleanResearchDataset,
    config: Phase10Config,
) -> ModelPerformance:
    predictions = (probabilities >= 0.5).astype(np.int64)
    trading = _simulate_trades(
        probabilities,
        confirmation.timestamps,
        confirmation.entry_timestamps,
        confirmation.exit_timestamps,
        confirmation.entry_prices,
        confirmation.exit_prices,
        config,
    )
    return ModelPerformance(
        model_name=name,
        model_class=(
            "baseline" if name in {"majority_class", "rule_based_momentum"} else "machine_learning"
        ),
        sample_count=confirmation.row_count,
        accuracy=float(accuracy_score(confirmation.y, predictions)),
        precision=float(precision_score(confirmation.y, predictions, zero_division=0.0)),
        recall=float(recall_score(confirmation.y, predictions, zero_division=0.0)),
        trade_count=trading.trade_count,
        win_count=trading.wins,
        loss_count=trading.losses,
        win_rate=(trading.wins / trading.trade_count if trading.trade_count else 0.0),
        gross_pnl=float(trading.gross_pnl),
        execution_costs=float(trading.costs),
        net_pnl=float(trading.net_pnl),
        return_on_capital=float(trading.net_pnl / config.initial_capital),
        max_drawdown_amount=float(trading.max_drawdown),
        max_drawdown_percent=float(trading.max_drawdown_pct),
        profitable=bool(trading.net_pnl > 0.0),
    )


def run_phase11_evaluation(
    storage_dir: Path | str,
    *,
    target_calendar_days: int = DEFAULT_TARGET_DAYS,
    confirmation_days: int = DEFAULT_CONFIRMATION_DAYS,
    config: Phase10Config | None = None,
    cost_scenarios: tuple[tuple[float, float], ...] = DEFAULT_COST_SCENARIOS,
    confirmation_state_path: Path | str | None = None,
    prior_evaluation_cutoff_ts: int = PHASE10_EVALUATION_CUTOFF_TS,
    evaluate_confirmation: bool = False,
) -> Phase11EvaluationResult:
    """Run development research and, only when explicit, one confirmation evaluation."""
    if confirmation_days <= 0:
        raise ValueError("confirmation_days must be positive")
    cfg = config or Phase10Config(minimum_calendar_days=float(target_calendar_days))
    coverage, candles = audit_expansion_coverage(
        storage_dir,
        symbol=cfg.symbol,
        candle_size_seconds=cfg.candle_size_seconds,
        source=cfg.source,
        target_calendar_days=target_calendar_days,
    )
    state_path = (
        Path(confirmation_state_path)
        if confirmation_state_path is not None
        else Path(storage_dir).parent / "phase11" / "confirmation_state.json"
    )
    protocol = load_or_create_confirmation_state(
        state_path,
        candles=candles,
        confirmation_days=confirmation_days,
        prior_evaluation_cutoff_ts=prior_evaluation_cutoff_ts,
    )
    if evaluate_confirmation and protocol.protocol_status != "MODEL_FROZEN":
        raise ConfirmationProtocolError(
            "Independent confirmation requires a frozen model and cannot be rerun."
        )
    if protocol.protocol_status == "DEVELOPMENT_OPEN":
        development_candles = candles
        confirmation_candles: list[Candle] = []
    else:
        if protocol.development_freeze_ts is None:
            raise ConfirmationProtocolError("Frozen protocol is missing its development cutoff.")
        development_candles, confirmation_candles = partition_confirmation_candles(
            candles,
            prior_evaluation_cutoff_ts=protocol.development_freeze_ts,
        )
        if (
            protocol.development_dataset_fingerprint is None
            or _research_candle_fingerprint(development_candles)
            != protocol.development_dataset_fingerprint
        ):
            raise ConfirmationProtocolError(
                "Frozen development dataset fingerprint no longer matches verified storage."
            )
    evaluated_at = datetime.now(UTC).isoformat()
    base_limitations = [
        "The October-November Phase 10 history remains permanently recorded as previously "
        "evaluated development data, not independent confirmation.",
        "Stored candles lack verified bid/ask quotes; every spread and slippage value is a "
        "modeled sensitivity assumption, not an observed historical quote.",
    ]
    if coverage.unexpected_gap_count:
        base_limitations.append(
            f"Coverage contains {coverage.unexpected_gap_count} unexpected open-market gaps."
        )
    development_span_days = (
        development_candles[-1].from_ts - development_candles[0].from_ts
    ) / 86400.0
    development_ready = development_span_days >= target_calendar_days
    confirmation_ready = bool(
        confirmation_candles
        and (confirmation_candles[-1].from_ts - confirmation_candles[0].from_ts)
        >= confirmation_days * 86400
    )
    if not development_ready:
        base_limitations.insert(
            0,
            f"Development coverage at the Phase 10 cutoff is {development_span_days:.2f} "
            f"days; {target_calendar_days} days are required for development research.",
        )
        return Phase11EvaluationResult(
            coverage=coverage,
            ready_for_confirmation=confirmation_ready,
            confirmation_evaluated=protocol.protocol_status == "CONFIRMATION_EVALUATED",
            confirmation_evaluation_status=protocol.protocol_status,
            development_freeze_utc=protocol.development_freeze_utc,
            development_dataset_fingerprint=protocol.development_dataset_fingerprint,
            frozen_model_name=protocol.frozen_model_name,
            prior_evaluation_cutoff_utc=protocol.prior_evaluation_cutoff_utc,
            prior_evaluated_dataset_fingerprint=(protocol.prior_evaluated_dataset_fingerprint),
            confirmation_dataset_fingerprint=(protocol.confirmation_dataset_fingerprint),
            confirmation_days=confirmation_days,
            confirmation_start_utc=(
                datetime.fromtimestamp(confirmation_candles[0].from_ts, tz=UTC).isoformat()
                if confirmation_candles
                else None
            ),
            development_rows=0,
            confirmation_rows=len(confirmation_candles),
            development_metrics={},
            selected_model=None,
            selected_baseline=None,
            confirmation_metrics={},
            cost_sensitivity=(),
            limitations=tuple(base_limitations),
            evaluated_at_utc=evaluated_at,
        )

    development = prepare_clean_dataset(
        development_candles,
        candle_size_seconds=cfg.candle_size_seconds,
        horizon_bars=cfg.horizon_bars,
    )
    if development.row_count < cfg.minimum_rows:
        raise Phase10Error("Development partition is statistically insufficient.")
    development_evaluation = evaluate_clean_dataset(development, cfg)
    if not confirmation_ready:
        base_limitations.insert(
            0,
            f"No genuinely unseen {confirmation_days}-day post-Phase-10 period is available; "
            "confirmation was not evaluated.",
        )
    elif not evaluate_confirmation:
        base_limitations.insert(
            0,
            "Genuinely unseen confirmation data is reserved. Use the explicit one-time "
            "confirmation option only after model development is frozen.",
        )

    if not confirmation_ready or not evaluate_confirmation:
        return Phase11EvaluationResult(
            coverage=coverage,
            ready_for_confirmation=confirmation_ready,
            confirmation_evaluated=protocol.protocol_status == "CONFIRMATION_EVALUATED",
            confirmation_evaluation_status=protocol.protocol_status,
            development_freeze_utc=protocol.development_freeze_utc,
            development_dataset_fingerprint=protocol.development_dataset_fingerprint,
            frozen_model_name=protocol.frozen_model_name,
            prior_evaluation_cutoff_utc=protocol.prior_evaluation_cutoff_utc,
            prior_evaluated_dataset_fingerprint=(protocol.prior_evaluated_dataset_fingerprint),
            confirmation_dataset_fingerprint=(protocol.confirmation_dataset_fingerprint),
            confirmation_days=confirmation_days,
            confirmation_start_utc=(
                datetime.fromtimestamp(confirmation_candles[0].from_ts, tz=UTC).isoformat()
                if confirmation_candles
                else None
            ),
            development_rows=development.row_count,
            confirmation_rows=len(confirmation_candles),
            development_metrics=development_evaluation.metrics,
            selected_model=protocol.frozen_model_name or development_evaluation.best_model,
            selected_baseline=development_evaluation.best_baseline,
            confirmation_metrics={},
            cost_sensitivity=(),
            limitations=tuple(base_limitations),
            evaluated_at_utc=evaluated_at,
        )

    confirmation = prepare_clean_dataset(
        confirmation_candles,
        candle_size_seconds=cfg.candle_size_seconds,
        horizon_bars=cfg.horizon_bars,
    )
    if confirmation.row_count < 100:
        raise Phase10Error("Confirmation partition is statistically insufficient.")
    frozen_model = protocol.frozen_model_name
    if frozen_model is None:
        raise ConfirmationProtocolError("Frozen protocol is missing its selected model name.")
    confirmation_fingerprint = _research_candle_fingerprint(confirmation_candles)
    started_at = datetime.now(UTC).isoformat()
    protocol = replace(
        protocol,
        protocol_status="CONFIRMATION_STARTED",
        confirmation_start_ts=confirmation_candles[0].from_ts,
        confirmation_end_ts=confirmation_candles[-1].from_ts,
        confirmation_dataset_fingerprint=confirmation_fingerprint,
        full_dataset_fingerprint_at_evaluation=coverage.base.aggregate_fingerprint,
        evaluation_started_at_utc=evaluated_at,
        audit_history=(
            *protocol.audit_history,
            {
                "event": "CONFIRMATION_STARTED",
                "at_utc": started_at,
                "confirmation_dataset_fingerprint": confirmation_fingerprint,
            },
        ),
    )
    _write_confirmation_state(state_path, protocol)
    selected_names = (
        development_evaluation.best_baseline,
        frozen_model,
    )
    confirmation_metrics: dict[str, ModelPerformance] = {}
    selected_probabilities: np.ndarray | None = None
    for name in selected_names:
        probabilities = _predict_confirmation(name, development, confirmation, cfg)
        confirmation_metrics[name] = _confirmation_metrics(
            name,
            probabilities,
            confirmation,
            cfg,
        )
        if name == frozen_model:
            selected_probabilities = probabilities
    if selected_probabilities is None:
        raise Phase10Error("Selected model produced no confirmation probabilities.")

    sensitivity: list[CostSensitivityResult] = []
    for spread_pips, slippage_pips in cost_scenarios:
        if spread_pips < 0 or slippage_pips < 0:
            raise ValueError("Cost sensitivity values cannot be negative")
        scenario = replace(cfg, spread_pips=spread_pips, slippage_pips=slippage_pips)
        trading = _simulate_trades(
            selected_probabilities,
            confirmation.timestamps,
            confirmation.entry_timestamps,
            confirmation.exit_timestamps,
            confirmation.entry_prices,
            confirmation.exit_prices,
            scenario,
        )
        sensitivity.append(
            CostSensitivityResult(
                spread_pips=spread_pips,
                slippage_pips=slippage_pips,
                trade_count=trading.trade_count,
                execution_costs=float(trading.costs),
                net_pnl=float(trading.net_pnl),
                return_on_capital=float(trading.net_pnl / scenario.initial_capital),
                max_drawdown_percent=float(trading.max_drawdown_pct),
            )
        )

    base_limitations.append(
        "This confirmation period has now been consumed and cannot be presented as independent "
        "again; additional later data is required for replication."
    )
    completed_at = datetime.now(UTC).isoformat()
    protocol = replace(
        protocol,
        protocol_status="CONFIRMATION_EVALUATED",
        evaluated_at_utc=completed_at,
        audit_history=(
            *protocol.audit_history,
            {
                "event": "CONFIRMATION_EVALUATED",
                "at_utc": completed_at,
                "confirmation_dataset_fingerprint": confirmation_fingerprint,
            },
        ),
    )
    _write_confirmation_state(state_path, protocol)
    return Phase11EvaluationResult(
        coverage=coverage,
        ready_for_confirmation=True,
        confirmation_evaluated=True,
        confirmation_evaluation_status=protocol.protocol_status,
        development_freeze_utc=protocol.development_freeze_utc,
        development_dataset_fingerprint=protocol.development_dataset_fingerprint,
        frozen_model_name=protocol.frozen_model_name,
        prior_evaluation_cutoff_utc=protocol.prior_evaluation_cutoff_utc,
        prior_evaluated_dataset_fingerprint=protocol.prior_evaluated_dataset_fingerprint,
        confirmation_dataset_fingerprint=confirmation_fingerprint,
        confirmation_days=confirmation_days,
        confirmation_start_utc=datetime.fromtimestamp(
            confirmation_candles[0].from_ts, tz=UTC
        ).isoformat(),
        development_rows=development.row_count,
        confirmation_rows=confirmation.row_count,
        development_metrics=development_evaluation.metrics,
        selected_model=protocol.frozen_model_name,
        selected_baseline=development_evaluation.best_baseline,
        confirmation_metrics=confirmation_metrics,
        cost_sensitivity=tuple(sensitivity),
        limitations=tuple(base_limitations),
        evaluated_at_utc=completed_at,
    )


def render_phase11_markdown(result: Phase11EvaluationResult) -> str:
    """Render coverage, confirmation isolation, sensitivity, and limitations."""
    coverage = result.coverage
    lines = [
        "# Phase 11 Historical Expansion and Robustness",
        "",
        f"Generated: `{result.evaluated_at_utc}`",
        "",
        "## Verified coverage",
        "",
        f"- Unique candles: **{coverage.base.unique_candles:,}**",
        f"- Oldest candle: `{coverage.oldest_utc}`",
        f"- Newest candle end: `{coverage.newest_utc}`",
        f"- Calendar span: **{coverage.base.calendar_span_days:.2f} days**",
        f"- Target: **{coverage.target_calendar_days} days**",
        f"- Scheduled closures: **{coverage.scheduled_closure_count}**",
        f"- Unexpected gaps: **{coverage.unexpected_gap_count}**",
        f"- Unverified gaps: **{coverage.unverified_gap_count}**",
        "",
        "## Confirmation protocol",
        "",
        f"- Phase 10 evaluation cutoff: `{result.prior_evaluation_cutoff_utc}`",
        "- Pre-cutoff history classification: **previously evaluated development data**",
        f"- Pre-cutoff dataset fingerprint: `{result.prior_evaluated_dataset_fingerprint}`",
        f"- Protocol status: **{result.confirmation_evaluation_status}**",
        f"- Development freeze: `{result.development_freeze_utc or 'not frozen'}`",
        f"- Frozen development fingerprint: "
        f"`{result.development_dataset_fingerprint or 'not frozen'}`",
        f"- Frozen model: **{result.frozen_model_name or 'not frozen'}**",
        f"- Confirmation dataset fingerprint: "
        f"`{result.confirmation_dataset_fingerprint or 'not assigned'}`",
        "",
    ]
    if result.development_metrics:
        lines.extend([
            "## Development research",
            "",
            f"- Development rows: **{result.development_rows:,}**",
            f"- Selected ML model: **{result.selected_model}**",
            f"- Selected baseline: **{result.selected_baseline}**",
            "",
        ])
    if not result.confirmation_evaluated:
        readiness = (
            "READY BUT RESERVED — confirmation was not run."
            if result.ready_for_confirmation
            else "NOT YET AVAILABLE — confirmation was not run."
        )
        lines.extend([
            "## Status",
            "",
            f"**{readiness}**",
            "",
        ])
    else:
        lines.extend([
            "## Chronological isolation",
            "",
            f"- One-time confirmation rows: **{result.confirmation_rows:,}**",
            f"- Confirmation begins: `{result.confirmation_start_utc}`",
            f"- Selected ML model from development only: **{result.selected_model}**",
            f"- Selected baseline from development only: **{result.selected_baseline}**",
            "",
            "## One-time independent confirmation results",
            "",
            "| Model | Accuracy | Precision | Recall | Trades | Net PnL | Max DD |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for metric in result.confirmation_metrics.values():
            lines.append(
                f"| {metric.model_name} | {metric.accuracy:.2%} | {metric.precision:.2%} | "
                f"{metric.recall:.2%} | {metric.trade_count:,} | {metric.net_pnl:+.2f} | "
                f"{metric.max_drawdown_percent:.2%} |"
            )
        lines.extend([
            "",
            "## Selected-model execution-cost sensitivity",
            "",
            "| Spread (pips) | Slippage/side (pips) | Trades | Costs | Net PnL | Max DD |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for row in result.cost_sensitivity:
            lines.append(
                f"| {row.spread_pips:.2f} | {row.slippage_pips:.2f} | "
                f"{row.trade_count:,} | {row.execution_costs:.2f} | {row.net_pnl:+.2f} | "
                f"{row.max_drawdown_percent:.2%} |"
            )
        lines.append("")
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in result.limitations)
    lines.extend([
        "",
        "> Offline research only. No broker connection or order placement occurs during audit "
        "or evaluation.",
        "",
    ])
    return "\n".join(lines)


def write_phase11_reports(
    result: Phase11EvaluationResult,
    *,
    markdown_path: Path | str,
    json_path: Path | str,
) -> tuple[Path, Path]:
    md_target = Path(markdown_path)
    json_target = Path(json_path)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.write_text(render_phase11_markdown(result), encoding="utf-8")
    json_target.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return md_target, json_target
