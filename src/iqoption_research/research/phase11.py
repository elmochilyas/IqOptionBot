"""Phase 11 historical expansion planning and confirmation-period research.

All planning and evaluation functions are offline.  The only object capable of
fetching broker data is the returned Phase 9 ``BackfillJob``; callers must still
provide an explicitly authorized read-only broker client.
"""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class Phase11EvaluationResult:
    """Development/confirmation evidence or an explicit insufficiency result."""

    coverage: ExpansionCoverageAudit
    ready_for_confirmation: bool
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


def partition_confirmation_candles(
    candles: list[Candle],
    *,
    confirmation_days: int,
) -> tuple[list[Candle], list[Candle], int]:
    """Reserve the latest calendar period before any model development begins."""
    if confirmation_days <= 0:
        raise ValueError("confirmation_days must be positive")
    if not candles:
        raise Phase10Error("Cannot partition an empty candle series.")
    boundary = candles[-1].from_ts - confirmation_days * 86400
    development = [candle for candle in candles if candle.from_ts < boundary]
    confirmation = [candle for candle in candles if candle.from_ts >= boundary]
    if not development or not confirmation:
        raise Phase10Error("Coverage is too short for the requested confirmation period.")
    return development, confirmation, boundary


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
) -> Phase11EvaluationResult:
    """Run development walk-forward and one untouched final confirmation period."""
    cfg = config or Phase10Config(minimum_calendar_days=float(target_calendar_days))
    coverage, candles = audit_expansion_coverage(
        storage_dir,
        symbol=cfg.symbol,
        candle_size_seconds=cfg.candle_size_seconds,
        source=cfg.source,
        target_calendar_days=target_calendar_days,
    )
    evaluated_at = datetime.now(UTC).isoformat()
    base_limitations = [
        "Stored candles lack verified bid/ask quotes; every spread and slippage value is a "
        "modeled sensitivity assumption, not an observed historical quote."
    ]
    if coverage.unexpected_gap_count:
        base_limitations.append(
            f"Coverage contains {coverage.unexpected_gap_count} unexpected open-market gaps."
        )
    if not coverage.meets_target:
        base_limitations.insert(
            0,
            f"Coverage is {coverage.base.calendar_span_days:.2f} days; "
            f"{target_calendar_days} days are required before confirmation evaluation.",
        )
        return Phase11EvaluationResult(
            coverage=coverage,
            ready_for_confirmation=False,
            confirmation_days=confirmation_days,
            confirmation_start_utc=None,
            development_rows=0,
            confirmation_rows=0,
            development_metrics={},
            selected_model=None,
            selected_baseline=None,
            confirmation_metrics={},
            cost_sensitivity=(),
            limitations=tuple(base_limitations),
            evaluated_at_utc=evaluated_at,
        )

    development_candles, confirmation_candles, boundary = partition_confirmation_candles(
        candles,
        confirmation_days=confirmation_days,
    )
    development = prepare_clean_dataset(
        development_candles,
        candle_size_seconds=cfg.candle_size_seconds,
        horizon_bars=cfg.horizon_bars,
    )
    confirmation = prepare_clean_dataset(
        confirmation_candles,
        candle_size_seconds=cfg.candle_size_seconds,
        horizon_bars=cfg.horizon_bars,
    )
    if development.row_count < cfg.minimum_rows or confirmation.row_count < 100:
        raise Phase10Error("Development or confirmation partition is statistically insufficient.")
    development_evaluation = evaluate_clean_dataset(development, cfg)
    selected_names = (
        development_evaluation.best_baseline,
        development_evaluation.best_model,
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
        if name == development_evaluation.best_model:
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
        "The confirmation period is evaluated once after model selection; additional later data "
        "is required for replication."
    )
    return Phase11EvaluationResult(
        coverage=coverage,
        ready_for_confirmation=True,
        confirmation_days=confirmation_days,
        confirmation_start_utc=datetime.fromtimestamp(boundary, tz=UTC).isoformat(),
        development_rows=development.row_count,
        confirmation_rows=confirmation.row_count,
        development_metrics=development_evaluation.metrics,
        selected_model=development_evaluation.best_model,
        selected_baseline=development_evaluation.best_baseline,
        confirmation_metrics=confirmation_metrics,
        cost_sensitivity=tuple(sensitivity),
        limitations=tuple(base_limitations),
        evaluated_at_utc=evaluated_at,
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
    ]
    if not result.ready_for_confirmation:
        lines.extend([
            "## Status",
            "",
            "**INSUFFICIENT HISTORY — confirmation evaluation was not run.**",
            "",
        ])
    else:
        lines.extend([
            "## Chronological isolation",
            "",
            f"- Development rows: **{result.development_rows:,}**",
            f"- Untouched confirmation rows: **{result.confirmation_rows:,}**",
            f"- Confirmation begins: `{result.confirmation_start_utc}`",
            f"- Selected ML model from development only: **{result.selected_model}**",
            f"- Selected baseline from development only: **{result.selected_baseline}**",
            "",
            "## Untouched confirmation results",
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
