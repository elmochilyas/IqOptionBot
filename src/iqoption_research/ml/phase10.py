"""Phase 10 authentic-data audit, training, and walk-forward evaluation.

This module is deliberately offline.  It reads immutable Phase 9 catalog entries,
verifies every Parquet and metadata checksum, deduplicates overlapping collections,
builds causal features independently inside contiguous market-data segments, and
evaluates deterministic baseline and ML models on forward-only folds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.parquet_storage import read_candles_parquet
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.models.base import BaseMLModel
from iqoption_research.ml.pipeline import create_model
from iqoption_research.models.market import Candle
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.features.scalers import StandardScaler

AUTHENTIC_SOURCE: Final[str] = "broker_authentic_collection"
PHASE10_DISCLAIMER: Final[str] = (
    "Strictly offline historical simulation. No broker connection or order API is used. "
    "Past simulated performance is not evidence of future profitability."
)
_STABLE_CANDLE_FIELDS: Final[tuple[str, ...]] = (
    "active_id",
    "symbol",
    "size",
    "from_ts",
    "to_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid",
    "ask",
    "is_complete",
)


class Phase10Error(RuntimeError):
    """Raised when authentic data cannot be audited or evaluated safely."""


def _stable_values(candle: Candle) -> tuple[object, ...]:
    return tuple(getattr(candle, name) for name in _STABLE_CANDLE_FIELDS)


@dataclass(frozen=True, slots=True)
class CoverageGap:
    """A discontinuity between two otherwise chronological candles."""

    previous_from_ts: int
    next_from_ts: int
    duration_seconds: int
    missing_bars: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AuthenticDataAudit:
    """Checksum, overlap, conflict, and coverage evidence for catalog data."""

    symbol: str
    candle_size_seconds: int
    source: str
    catalog_entries: int
    checksum_verified_entries: int
    catalog_rows: int
    unique_candles: int
    duplicate_rows: int
    metadata_duplicate_variants: int
    conflicting_duplicates: int
    start_ts: int
    end_ts: int
    trading_coverage_hours: float
    calendar_span_days: float
    gaps: tuple[CoverageGap, ...]
    dataset_ids: tuple[str, ...]
    aggregate_fingerprint: str

    @property
    def gap_count(self) -> int:
        return len(self.gaps)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gaps"] = [gap.to_dict() for gap in self.gaps]
        return payload


@dataclass(frozen=True, slots=True)
class CleanResearchDataset:
    """Causal features, forward labels, and execution prices."""

    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    entry_timestamps: np.ndarray
    exit_timestamps: np.ndarray
    entry_prices: np.ndarray
    exit_prices: np.ndarray
    feature_names: tuple[str, ...]
    segment_count: int
    dropped_warmup_rows: int
    dropped_gap_crossing_labels: int
    fingerprint: str

    @property
    def row_count(self) -> int:
        return len(self.y)


@dataclass(frozen=True, slots=True)
class Phase10Config:
    """Reproducible walk-forward and cost simulation parameters."""

    symbol: str = "EURUSD"
    candle_size_seconds: int = 60
    source: str = AUTHENTIC_SOURCE
    horizon_bars: int = 1
    n_folds: int = 4
    initial_train_ratio: float = 0.50
    validation_ratio: float = 0.10
    test_ratio: float = 0.10
    confidence_threshold: float = 0.55
    seed: int = 42
    spread_pips: float = 1.0
    slippage_pips: float = 0.10
    commission_per_trade: float = 0.0
    pip_size: float = 0.0001
    notional_per_trade: float = 10_000.0
    initial_capital: float = 10_000.0
    minimum_rows: int = 2_000
    minimum_calendar_days: float = 90.0
    model_names: tuple[str, ...] = (
        "majority_class",
        "rule_based_momentum",
        "logistic_regression",
        "random_forest",
    )

    def validate(self) -> None:
        if self.horizon_bars < 1 or self.n_folds < 1:
            raise ValueError("horizon_bars and n_folds must be positive")
        ratios = self.initial_train_ratio + self.validation_ratio + self.test_ratio * self.n_folds
        if ratios > 1.0000001:
            raise ValueError("Walk-forward ratios exceed the available dataset")
        if not 0.5 <= self.confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be in [0.5, 1.0)")
        if min(self.spread_pips, self.slippage_pips, self.commission_per_trade) < 0:
            raise ValueError("Execution costs cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_names"] = list(self.model_names)
        return payload


@dataclass(frozen=True, slots=True)
class FoldDefinition:
    fold: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int
    train_end_ts: int
    validation_start_ts: int
    validation_end_ts: int
    test_start_ts: int
    test_end_ts: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelPerformance:
    model_name: str
    model_class: str
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    gross_pnl: float
    execution_costs: float
    net_pnl: float
    return_on_capital: float
    max_drawdown_amount: float
    max_drawdown_percent: float
    profitable: bool
    profit_factor: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Phase10Result:
    audit: AuthenticDataAudit
    config: Phase10Config
    dataset_rows: int
    feature_names: tuple[str, ...]
    dataset_fingerprint: str
    folds: tuple[FoldDefinition, ...]
    metrics: dict[str, ModelPerformance]
    best_model: str
    best_baseline: str
    needs_more_data: bool
    limitations: tuple[str, ...]
    evaluated_at_utc: str
    disclaimer: str = PHASE10_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit": self.audit.to_dict(),
            "config": self.config.to_dict(),
            "dataset_rows": self.dataset_rows,
            "feature_names": list(self.feature_names),
            "dataset_fingerprint": self.dataset_fingerprint,
            "folds": [fold.to_dict() for fold in self.folds],
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "best_model": self.best_model,
            "best_baseline": self.best_baseline,
            "needs_more_data": self.needs_more_data,
            "limitations": list(self.limitations),
            "evaluated_at_utc": self.evaluated_at_utc,
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardDatasetEvaluation:
    """Reusable walk-forward output for an already prepared causal dataset."""

    folds: tuple[FoldDefinition, ...]
    metrics: dict[str, ModelPerformance]
    best_model: str
    best_baseline: str


def _safe_catalog_path(storage_dir: Path, relative_path: str) -> Path:
    root = storage_dir.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise Phase10Error(f"Catalog path escapes storage root: {relative_path}")
    return candidate


def audit_authentic_datasets(
    storage_dir: Path | str,
    *,
    symbol: str = "EURUSD",
    candle_size_seconds: int = 60,
    source: str = AUTHENTIC_SOURCE,
) -> tuple[AuthenticDataAudit, list[Candle]]:
    """Verify catalog files and return conflict-checked unique candles."""
    root = Path(storage_dir)
    catalog = DatasetCatalog(root / "catalog.db")
    entries = catalog.find_datasets(
        symbol=symbol,
        candle_size_seconds=candle_size_seconds,
        source=source,
    )
    if not entries:
        raise Phase10Error(
            f"No authentic catalog datasets for {symbol} {candle_size_seconds}s in {root}."
        )

    unique: dict[int, Candle] = {}
    duplicate_rows = 0
    metadata_duplicate_variants = 0
    conflicts: list[str] = []
    digest = hashlib.sha256()
    catalog_rows = 0

    for entry in entries:
        if entry.is_synthetic:
            raise Phase10Error(f"Authentic source contains synthetic dataset {entry.dataset_id}.")
        path = _safe_catalog_path(root, entry.file_path)
        candles = read_candles_parquet(path, catalog_entry=entry)
        if len(candles) != entry.candle_count:
            raise Phase10Error(
                f"Catalog row count mismatch for {entry.dataset_id}: "
                f"expected {entry.candle_count}, loaded {len(candles)}."
            )
        catalog_rows += len(candles)
        digest.update(entry.dataset_id.encode("utf-8"))
        digest.update(entry.sha256_checksum.encode("ascii"))
        digest.update(entry.metadata_sha256.encode("ascii"))
        for candle in candles:
            previous = unique.get(candle.from_ts)
            if previous is None:
                unique[candle.from_ts] = candle
                continue
            duplicate_rows += 1
            if previous.at_ts != candle.at_ts:
                metadata_duplicate_variants += 1
            differing = tuple(
                name
                for name, old, new in zip(
                    _STABLE_CANDLE_FIELDS,
                    _stable_values(previous),
                    _stable_values(candle),
                    strict=True,
                )
                if old != new
            )
            if differing:
                conflicts.append(f"{candle.from_ts}: {', '.join(differing)}")

    if conflicts:
        preview = "; ".join(conflicts[:5])
        raise Phase10Error(
            f"Found {len(conflicts)} conflicting duplicate candles; first conflicts: {preview}"
        )

    candles = sorted(unique.values(), key=lambda item: item.from_ts)
    if not candles:
        raise Phase10Error("Authentic catalog contained no candles after verification.")
    gaps: list[CoverageGap] = []
    for previous, current in pairwise(candles):
        delta = current.from_ts - previous.from_ts
        if delta != candle_size_seconds:
            gaps.append(
                CoverageGap(
                    previous_from_ts=previous.from_ts,
                    next_from_ts=current.from_ts,
                    duration_seconds=delta,
                    missing_bars=max(0, delta // candle_size_seconds - 1),
                )
            )

    for candle in candles:
        digest.update(repr(_stable_values(candle)).encode("utf-8"))
    audit = AuthenticDataAudit(
        symbol=symbol.upper(),
        candle_size_seconds=candle_size_seconds,
        source=source,
        catalog_entries=len(entries),
        checksum_verified_entries=len(entries),
        catalog_rows=catalog_rows,
        unique_candles=len(candles),
        duplicate_rows=duplicate_rows,
        metadata_duplicate_variants=metadata_duplicate_variants,
        conflicting_duplicates=0,
        start_ts=candles[0].from_ts,
        end_ts=candles[-1].from_ts,
        trading_coverage_hours=len(candles) * candle_size_seconds / 3600.0,
        calendar_span_days=(candles[-1].from_ts - candles[0].from_ts) / 86400.0,
        gaps=tuple(gaps),
        dataset_ids=tuple(entry.dataset_id for entry in entries),
        aggregate_fingerprint=digest.hexdigest(),
    )
    return audit, candles


def _contiguous_segments(candles: list[Candle], size_seconds: int) -> list[list[Candle]]:
    segments: list[list[Candle]] = []
    current: list[Candle] = []
    for candle in candles:
        if current and candle.from_ts - current[-1].from_ts != size_seconds:
            segments.append(current)
            current = []
        current.append(candle)
    if current:
        segments.append(current)
    return segments


def prepare_clean_dataset(
    candles: list[Candle],
    *,
    candle_size_seconds: int,
    horizon_bars: int = 1,
) -> CleanResearchDataset:
    """Create causal rows without allowing features or labels to cross gaps."""
    segments = _contiguous_segments(candles, candle_size_seconds)
    feature_pipeline = CausalFeaturePipeline(
        include_volume=True,
        include_quotes=False,
        drop_warmup=False,
    )
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    entry_timestamps: list[np.ndarray] = []
    exit_timestamps: list[np.ndarray] = []
    entries: list[np.ndarray] = []
    exits: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    dropped_warmup = 0

    for segment in segments:
        if len(segment) <= feature_pipeline.max_lookback + horizon_bars:
            continue
        features = feature_pipeline.transform(segment)
        names = tuple(features.feature_names)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise Phase10Error("Feature schema changed between contiguous data segments.")
        valid_rows = [
            row
            for row in features.rows
            if not row.is_warmup and row.index + horizon_bars < len(segment)
        ]
        dropped_warmup += sum(1 for row in features.rows if row.is_warmup)
        matrices.append(
            np.asarray(
                [[row.features[name] for name in names] for row in valid_rows],
                dtype=np.float64,
            )
        )
        labels.append(
            np.asarray(
                [
                    int(
                        round(
                            segment[row.index + horizon_bars].close - segment[row.index + 1].open,
                            9,
                        )
                        > 0.0
                    )
                    for row in valid_rows
                ],
                dtype=np.int64,
            )
        )
        timestamps.append(np.asarray([row.available_at_ts for row in valid_rows], dtype=np.int64))
        entry_timestamps.append(
            np.asarray([segment[row.index + 1].from_ts for row in valid_rows], dtype=np.int64)
        )
        exit_timestamps.append(
            np.asarray(
                [segment[row.index + horizon_bars].to_ts for row in valid_rows],
                dtype=np.int64,
            )
        )
        entries.append(np.asarray([segment[row.index + 1].open for row in valid_rows], dtype=float))
        exits.append(
            np.asarray([segment[row.index + horizon_bars].close for row in valid_rows], dtype=float)
        )

    if not matrices or feature_names is None:
        raise Phase10Error("No contiguous segment is long enough for causal feature preparation.")
    X = np.concatenate(matrices)
    y = np.concatenate(labels)
    ts = np.concatenate(timestamps)
    entry_ts = np.concatenate(entry_timestamps)
    exit_ts = np.concatenate(exit_timestamps)
    entry = np.concatenate(entries)
    exit_prices = np.concatenate(exits)
    if np.any(np.diff(ts) <= 0):
        raise Phase10Error("Prepared timestamps are not strictly chronological.")
    if np.any(entry_ts < ts):
        raise Phase10Error("An execution entry precedes its signal availability timestamp.")
    if np.any(exit_ts <= entry_ts):
        raise Phase10Error("An execution exit is not later than its entry timestamp.")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(X).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    digest.update(np.ascontiguousarray(ts).tobytes())
    digest.update(np.ascontiguousarray(entry_ts).tobytes())
    digest.update(np.ascontiguousarray(exit_ts).tobytes())
    digest.update(np.ascontiguousarray(entry).tobytes())
    digest.update(np.ascontiguousarray(exit_prices).tobytes())
    return CleanResearchDataset(
        X=X,
        y=y,
        timestamps=ts,
        entry_timestamps=entry_ts,
        exit_timestamps=exit_ts,
        entry_prices=entry,
        exit_prices=exit_prices,
        feature_names=feature_names,
        segment_count=len(segments),
        dropped_warmup_rows=dropped_warmup,
        dropped_gap_crossing_labels=max(0, len(segments) - 1) * horizon_bars,
        fingerprint=digest.hexdigest(),
    )


def build_walk_forward_folds(
    dataset: CleanResearchDataset, config: Phase10Config
) -> tuple[FoldDefinition, ...]:
    """Build expanding, purged train/validation/test folds."""
    config.validate()
    n = dataset.row_count
    initial_train = int(n * config.initial_train_ratio)
    validation_size = int(n * config.validation_ratio)
    test_size = int(n * config.test_ratio)
    if min(initial_train, validation_size, test_size) < 1:
        raise Phase10Error("Dataset is too small for configured walk-forward partitions.")
    folds: list[FoldDefinition] = []
    for fold_idx in range(config.n_folds):
        validation_start = initial_train + fold_idx * test_size
        test_start = validation_start + validation_size
        test_end = n if fold_idx == config.n_folds - 1 else test_start + test_size
        train_end = validation_start - config.horizon_bars
        validation_end = test_start - config.horizon_bars
        if train_end <= 0 or validation_end <= validation_start or test_end <= test_start:
            raise Phase10Error("Invalid purged walk-forward fold boundaries.")
        if test_end > n:
            raise Phase10Error("Configured walk-forward folds exceed the dataset.")
        folds.append(
            FoldDefinition(
                fold=fold_idx,
                train_start=0,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
                train_end_ts=int(dataset.timestamps[train_end - 1]),
                validation_start_ts=int(dataset.timestamps[validation_start]),
                validation_end_ts=int(dataset.timestamps[validation_end - 1]),
                test_start_ts=int(dataset.timestamps[test_start]),
                test_end_ts=int(dataset.timestamps[test_end - 1]),
            )
        )
    return tuple(folds)


def _model_is_baseline(name: str) -> bool:
    return name in {"majority_class", "rule_based_momentum"}


def _fit_fold_model(
    name: str,
    dataset: CleanResearchDataset,
    fold: FoldDefinition,
    config: Phase10Config,
) -> tuple[np.ndarray, np.ndarray]:
    model: BaseMLModel = create_model(name, seed=config.seed, quick=True)
    train_X = dataset.X[fold.train_start : fold.train_end]
    train_y = dataset.y[fold.train_start : fold.train_end]
    validation_X = dataset.X[fold.validation_start : fold.validation_end]
    validation_y = dataset.y[fold.validation_start : fold.validation_end]
    test_X = dataset.X[fold.test_start : fold.test_end]

    calibrator: ProbabilityCalibrator | None = None
    if _model_is_baseline(name):
        model.fit(train_X, train_y, feature_names=dataset.feature_names)
        test_probabilities = model.predict_proba(test_X)
    else:
        scaler = StandardScaler()
        scaler.fit(train_X, feature_names=dataset.feature_names)
        scaled_train = np.asarray(scaler.transform(train_X), dtype=np.float64)
        scaled_validation = np.asarray(scaler.transform(validation_X), dtype=np.float64)
        scaled_test = np.asarray(scaler.transform(test_X), dtype=np.float64)
        model.fit(scaled_train, train_y, feature_names=dataset.feature_names)
        validation_probabilities = model.predict_proba(scaled_validation)
        if len(np.unique(validation_y)) == 2:
            calibrator = ProbabilityCalibrator(method="platt")
            calibrator.fit(validation_probabilities, validation_y)
        test_probabilities = model.predict_proba(scaled_test)
        if calibrator is not None:
            test_probabilities = calibrator.calibrate(test_probabilities)

    probabilities_up = np.asarray(test_probabilities[:, 1], dtype=np.float64)
    predictions = (probabilities_up >= 0.5).astype(np.int64)
    return predictions, probabilities_up


@dataclass(slots=True)
class _TradingAccumulator:
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    equity: float = 0.0
    peak: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss > 0.0:
            return self.gross_profit / self.gross_loss
        if self.gross_profit > 0.0:
            return float("inf")
        return 0.0


def _simulate_trades(
    probabilities_up: np.ndarray,
    signal_timestamps: np.ndarray,
    entry_timestamps: np.ndarray,
    exit_timestamps: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    config: Phase10Config,
) -> _TradingAccumulator:
    lengths = {
        len(probabilities_up),
        len(signal_timestamps),
        len(entry_timestamps),
        len(exit_timestamps),
        len(entry_prices),
        len(exit_prices),
    }
    if len(lengths) != 1:
        raise Phase10Error("Signal, execution timestamp, and price arrays must align.")
    if np.any(entry_timestamps < signal_timestamps):
        raise Phase10Error("Cannot execute a trade before its signal is available.")
    if np.any(exit_timestamps <= entry_timestamps):
        raise Phase10Error("Trade exits must occur after their entries.")
    if np.any(entry_prices <= 0.0) or np.any(exit_prices <= 0.0):
        raise Phase10Error("Execution prices must be positive.")
    state = _TradingAccumulator(equity=config.initial_capital, peak=config.initial_capital)
    upper = config.confidence_threshold
    lower = 1.0 - upper
    spread = config.spread_pips * config.pip_size
    slippage = config.slippage_pips * config.pip_size
    for probability, entry, exit_price in zip(
        probabilities_up, entry_prices, exit_prices, strict=True
    ):
        if probability >= upper:
            direction = 1.0
        elif probability <= lower:
            direction = -1.0
        else:
            continue
        gross = direction * (exit_price - entry) / entry * config.notional_per_trade
        price_cost = (spread + 2.0 * slippage) / entry * config.notional_per_trade
        costs = price_cost + config.commission_per_trade
        net = gross - costs
        state.trade_count += 1
        state.gross_pnl += gross
        state.costs += costs
        state.net_pnl += net
        state.equity += net
        if net > 0:
            state.wins += 1
            state.gross_profit += net
        else:
            state.losses += 1
            state.gross_loss += abs(net)
        state.peak = max(state.peak, state.equity)
        drawdown = state.peak - state.equity
        state.max_drawdown = max(state.max_drawdown, drawdown)
        if state.peak > 0:
            state.max_drawdown_pct = max(state.max_drawdown_pct, drawdown / state.peak)
    return state


def evaluate_clean_dataset(
    clean: CleanResearchDataset,
    config: Phase10Config,
) -> WalkForwardDatasetEvaluation:
    """Evaluate a prepared dataset without reading storage or changing its time range."""
    folds = build_walk_forward_folds(clean, config)
    metrics: dict[str, ModelPerformance] = {}

    for name in config.model_names:
        true_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        probability_parts: list[np.ndarray] = []
        signal_timestamp_parts: list[np.ndarray] = []
        entry_timestamp_parts: list[np.ndarray] = []
        exit_timestamp_parts: list[np.ndarray] = []
        entry_parts: list[np.ndarray] = []
        exit_parts: list[np.ndarray] = []
        for fold in folds:
            predictions, probabilities = _fit_fold_model(name, clean, fold, config)
            true_parts.append(clean.y[fold.test_start : fold.test_end])
            prediction_parts.append(predictions)
            probability_parts.append(probabilities)
            signal_timestamp_parts.append(clean.timestamps[fold.test_start : fold.test_end])
            entry_timestamp_parts.append(clean.entry_timestamps[fold.test_start : fold.test_end])
            exit_timestamp_parts.append(clean.exit_timestamps[fold.test_start : fold.test_end])
            entry_parts.append(clean.entry_prices[fold.test_start : fold.test_end])
            exit_parts.append(clean.exit_prices[fold.test_start : fold.test_end])

        y_true = np.concatenate(true_parts)
        y_pred = np.concatenate(prediction_parts)
        probabilities_up = np.concatenate(probability_parts)
        trading = _simulate_trades(
            probabilities_up,
            np.concatenate(signal_timestamp_parts),
            np.concatenate(entry_timestamp_parts),
            np.concatenate(exit_timestamp_parts),
            np.concatenate(entry_parts),
            np.concatenate(exit_parts),
            config,
        )
        metrics[name] = ModelPerformance(
            model_name=name,
            model_class="baseline" if _model_is_baseline(name) else "machine_learning",
            sample_count=len(y_true),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0.0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0.0)),
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
            profit_factor=float(trading.profit_factor),
        )

    baseline_names = [name for name in config.model_names if _model_is_baseline(name)]
    ml_names = [name for name in config.model_names if not _model_is_baseline(name)]
    if not baseline_names or not ml_names:
        raise Phase10Error("At least one baseline and one ML model are required.")
    return WalkForwardDatasetEvaluation(
        folds=folds,
        metrics=metrics,
        best_baseline=max(baseline_names, key=lambda name: metrics[name].net_pnl),
        best_model=max(ml_names, key=lambda name: metrics[name].net_pnl),
    )


def run_phase10_experiment(
    storage_dir: Path | str,
    *,
    config: Phase10Config | None = None,
) -> Phase10Result:
    """Run the complete offline authentic-data experiment."""
    cfg = config or Phase10Config()
    cfg.validate()
    audit, candles = audit_authentic_datasets(
        storage_dir,
        symbol=cfg.symbol,
        candle_size_seconds=cfg.candle_size_seconds,
        source=cfg.source,
    )
    clean = prepare_clean_dataset(
        candles,
        candle_size_seconds=cfg.candle_size_seconds,
        horizon_bars=cfg.horizon_bars,
    )
    if clean.row_count < cfg.minimum_rows:
        raise Phase10Error(
            f"Insufficient authentic rows ({clean.row_count}); minimum is {cfg.minimum_rows}."
        )
    evaluation = evaluate_clean_dataset(clean, cfg)
    folds = evaluation.folds
    metrics = evaluation.metrics
    best_baseline = evaluation.best_baseline
    best_model = evaluation.best_model
    limitations: list[str] = []
    if audit.calendar_span_days < cfg.minimum_calendar_days:
        limitations.append(
            f"Only {audit.calendar_span_days:.1f} calendar days are covered; at least "
            f"{cfg.minimum_calendar_days:.0f} days are required for a less fragile regime sample."
        )
    if audit.gap_count:
        limitations.append(
            f"The source contains {audit.gap_count} coverage gaps; features and labels were "
            "segmented so no example crosses a gap."
        )
    limitations.append(
        "Stored candles contain no verified bid/ask quotes; spread and slippage are configurable "
        "assumptions rather than observed execution data."
    )
    limitations.append(
        "The reported best model is a descriptive comparison of predeclared candidates, not an "
        "independent confirmation set; a later untouched dataset is required before deployment."
    )
    if metrics[best_model].net_pnl <= metrics[best_baseline].net_pnl:
        limitations.append(
            "The best ML model did not outperform the best simple baseline after costs."
        )
    needs_more_data = audit.calendar_span_days < cfg.minimum_calendar_days
    return Phase10Result(
        audit=audit,
        config=cfg,
        dataset_rows=clean.row_count,
        feature_names=clean.feature_names,
        dataset_fingerprint=clean.fingerprint,
        folds=folds,
        metrics=metrics,
        best_model=best_model,
        best_baseline=best_baseline,
        needs_more_data=needs_more_data,
        limitations=tuple(limitations),
        evaluated_at_utc=datetime.now(UTC).isoformat(),
    )


def render_phase10_markdown(result: Phase10Result) -> str:
    """Render a concise, auditable experiment report."""
    audit = result.audit
    lines = [
        "# Phase 10 Authentic Market Data Evaluation",
        "",
        f"Generated: `{result.evaluated_at_utc}`",
        "",
        "## Authentic data audit",
        "",
        f"- Catalog partitions verified: **{audit.checksum_verified_entries}**",
        f"- Catalog rows: **{audit.catalog_rows:,}**",
        f"- Unique candles after overlap deduplication: **{audit.unique_candles:,}**",
        f"- Duplicate overlap rows: **{audit.duplicate_rows:,}**",
        f"- Metadata-only duplicate variants: **{audit.metadata_duplicate_variants:,}**",
        f"- Conflicting duplicates: **{audit.conflicting_duplicates}**",
        f"- Trading coverage: **{audit.trading_coverage_hours:.2f} hours**",
        f"- Calendar span: **{audit.calendar_span_days:.2f} days**",
        f"- Coverage gaps: **{audit.gap_count}**",
        f"- Aggregate fingerprint: `{audit.aggregate_fingerprint}`",
        "",
        "## Leakage controls",
        "",
        f"- Prepared causal rows: **{result.dataset_rows:,}**",
        f"- Walk-forward folds: **{len(result.folds)}**",
        f"- Label horizon and purge: **{result.config.horizon_bars} bar(s)**",
        "- Features are backward-looking; scalers fit on each training fold only.",
        "- A signal is available only after candle t closes; execution enters at candle t+1 "
        "open and exits at candle t+horizon close.",
        "- Directional labels use that same executable next-open-to-exit return.",
        "- Probability calibration uses only the immediately preceding validation fold.",
        "- Test folds are strictly later and are never used for fitting or calibration.",
        "- Gap boundaries are hard segment boundaries for both features and labels.",
        "",
        "## Strict out-of-sample performance",
        "",
        "| Model | Class | Accuracy | Precision | Recall | Trades | Win rate | Costs | "
        "Net PnL | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in result.metrics.values():
        lines.append(
            f"| {metric.model_name} | {metric.model_class} | {metric.accuracy:.2%} | "
            f"{metric.precision:.2%} | {metric.recall:.2%} | {metric.trade_count:,} | "
            f"{metric.win_rate:.2%} | {metric.execution_costs:.2f} | "
            f"{metric.net_pnl:+.2f} | {metric.max_drawdown_percent:.2%} |"
        )
    lines.extend([
        "",
        "## Assessment",
        "",
        f"- Best ML model after costs: **{result.best_model}**",
        f"- Best simple baseline after costs: **{result.best_baseline}**",
        f"- More historical data needed: **{'yes' if result.needs_more_data else 'no'}**",
        "",
        "### Limitations",
        "",
    ])
    lines.extend(f"- {limitation}" for limitation in result.limitations)
    lines.extend(["", f"> {result.disclaimer}", ""])
    return "\n".join(lines)


def write_phase10_reports(
    result: Phase10Result,
    *,
    markdown_path: Path | str,
    json_path: Path | str,
) -> tuple[Path, Path]:
    """Persist human- and machine-readable offline experiment evidence."""
    md_target = Path(markdown_path)
    json_target = Path(json_path)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.write_text(render_phase10_markdown(result), encoding="utf-8")
    json_target.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return md_target, json_target
