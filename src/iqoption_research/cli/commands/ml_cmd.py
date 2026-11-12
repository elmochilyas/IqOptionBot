"""CLI commands for offline Phase 4 machine-learning research.

Every command is offline by default. Synthetic fixtures are labelled as such and
never produce an edge verdict. Authentic CSV data can be supplied explicitly with
``--authentic`` for genuine out-of-sample assessment.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.market_data.importer import import_candles_from_csv
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.evaluation.comparison import (
    ComparisonConfig,
    EvaluationSplitType,
    PayoffAssumptionType,
)
from iqoption_research.ml.evaluation.holdout import HoldoutSplitConfig
from iqoption_research.ml.evaluation.walk_forward import WalkForwardConfig
from iqoption_research.ml.fixtures import (
    SyntheticMarketConfig,
    generate_synthetic_candles,
)
from iqoption_research.ml.pipeline import (
    AVAILABLE_MODEL_NAMES,
    PHASE4_DISCLAIMER,
    BacktestOutcome,
    PreparedDataset,
    create_model,
    load_and_evaluate_bundle,
    load_prepared_dataset,
    prepare_dataset,
    prepare_from_fixture,
    run_bundle_backtest,
    run_comparison,
    run_walk_forward,
    save_prepared_dataset,
    save_training_bundle,
    train_holdout,
)
from iqoption_research.ml.reporting import (
    EdgeAssessment,
    ReportContext,
    assess_bundle_evaluation,
    assess_evidence,
    render_phase4_markdown,
    write_report,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.ml.tracking.artifacts import load_model_bundle
from iqoption_research.ml.tracking.tracker import ExperimentRecord, ExperimentTracker

console = Console()
ml_app = typer.Typer(
    name="ml",
    help="Offline Phase 4 machine learning: prepare, train, evaluate, compare, backtest.",
    no_args_is_help=True,
)

DEFAULT_WORKSPACE = Path("data/ml_workspace")
DEFAULT_DB = Path("data/ml_experiments.db")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _target_spec(
    target_type: str,
    horizon: int,
    flat_threshold_pips: float,
) -> TargetSpec:
    try:
        normalised = TargetType(target_type.strip().lower())
    except ValueError as exc:
        raise typer.BadParameter(
            f"Unknown target type '{target_type}'. Use direction_binary, "
            "direction_ternary, or return_continuous."
        ) from exc
    return TargetSpec(
        target_type=normalised,
        horizon_bars=horizon,
        flat_threshold_pips=flat_threshold_pips,
    )


def _make_calibrator(method: str | None) -> ProbabilityCalibrator | None:
    if method is None or method.strip().lower() in ("", "none", "off"):
        return None
    return ProbabilityCalibrator(method=method.strip().lower())


def _holdout_config(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    purge_bars: int,
    embargo_bars: int,
) -> HoldoutSplitConfig:
    return HoldoutSplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )


def _build_prepared(
    *,
    csv_path: Path | None,
    symbol: str,
    timeframe: int,
    synthetic: bool,
    n_bars: int,
    seed: int,
    momentum_strength: float,
    target_spec: TargetSpec,
    include_quotes: bool,
) -> PreparedDataset:
    if csv_path is not None:
        candles = import_candles_from_csv(
            csv_path,
            symbol=symbol,
            candle_size_seconds=timeframe,
            is_synthetic=synthetic,
        )
        if not candles:
            raise typer.BadParameter(f"CSV contains no candles: {csv_path}")
        return prepare_dataset(
            candles,
            target_spec=target_spec,
            include_quotes=include_quotes,
            is_synthetic=synthetic,
            metadata={
                "source": "csv",
                "csv_path": str(csv_path.resolve()),
                "authentic": not synthetic,
            },
        )

    if not synthetic:
        raise typer.BadParameter(
            "Authentic evaluation requires --csv with real historical candles; "
            "the synthetic generator cannot produce authentic data."
        )
    fixture = generate_synthetic_candles(
        SyntheticMarketConfig(
            n_bars=n_bars,
            seed=seed,
            symbol=symbol,
            candle_size_seconds=timeframe,
            momentum_strength=momentum_strength,
        )
    )
    return prepare_from_fixture(
        fixture,
        target_spec=target_spec,
        include_quotes=include_quotes,
    )


def _context(
    prepared: PreparedDataset,
    *,
    title: str,
    model_name: str,
    split_type: str,
    config: dict[str, object],
) -> ReportContext:
    return ReportContext(
        title=title,
        symbol=prepared.symbol,
        model_name=model_name,
        dataset_rows=prepared.row_count,
        dataset_fingerprint=prepared.fingerprint,
        target_spec=prepared.dataset.target_spec.to_dict(),
        is_synthetic=prepared.is_synthetic,
        split_type=split_type,
        config=config,
        metadata={
            "generated_at_utc": _timestamp(),
            "prepared_at_utc": prepared.metadata.get("prepared_at_utc", ""),
            "fixture": prepared.metadata.get("fixture", ""),
        },
    )


def _print_assessment(assessment: EdgeAssessment) -> None:
    style = "bold yellow" if assessment.is_synthetic else "bold green"
    if assessment.verdict.value in (
        "NO_RELIABLE_EDGE_DETECTED",
        "INSUFFICIENT_OUT_OF_SAMPLE_DATA",
    ):
        style = "bold red"
    console.print(
        Panel(
            f"[{style}]{assessment.verdict.value}[/{style}]\n"
            f"Evidence class: {assessment.evidence_class}\n"
            f"Accuracy: {assessment.accuracy:.4f} | ROC-AUC: {assessment.roc_auc}\n"
            f"Samples: {assessment.sample_count} | p={assessment.binomial_p_value:.4g}",
            title="Evidence Assessment",
        )
    )


def _log_run(
    db_path: Path,
    *,
    experiment_name: str,
    prepared: PreparedDataset,
    model_name: str,
    split_type: str,
    calibration_method: str,
    assessment: EdgeAssessment,
    metrics: dict[str, Any],
    config: dict[str, Any],
    artifact_path: Path | None = None,
    artifact_manifest_sha256: str = "",
    notes: str = "",
    tags: list[str] | None = None,
) -> ExperimentRecord:
    tracker = ExperimentTracker(db_path)
    return tracker.log_experiment(
        experiment_name,
        model_name=model_name,
        dataset_symbol=prepared.symbol,
        dataset_rows=prepared.row_count,
        dataset_fingerprint=prepared.fingerprint,
        is_synthetic=prepared.is_synthetic,
        target_type=prepared.dataset.target_spec.target_type.value,
        horizon_bars=prepared.dataset.target_spec.horizon_bars,
        split_type=split_type,
        calibration_method=calibration_method,
        evidence_class=assessment.evidence_class,
        verdict=assessment.verdict.value,
        metrics=metrics,
        config=config,
        artifact_path=artifact_path,
        artifact_manifest_sha256=artifact_manifest_sha256,
        notes=notes,
        tags=tags or ["phase4"],
    )


def _write_and_report(
    report_path: Path | None,
    markdown: str,
) -> None:
    if report_path is None:
        return
    destination = write_report(report_path, markdown)
    console.print(f"[green]Report written to {destination}[/green]")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@ml_app.command("prepare", help="Prepare a causal ML dataset from synthetic or CSV data.")
def prepare_command(
    output: Path = typer.Option(
        DEFAULT_WORKSPACE / "prepared", "--output", "-o", help="Prepared dataset directory"
    ),
    csv_path: Path | None = typer.Option(
        None, "--csv", help="Local historical CSV (never fetched from a network)."
    ),
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    timeframe: int = typer.Option(60, "--timeframe", "-t", help="Candle size in seconds"),
    authentic: bool = typer.Option(
        False, "--authentic", help="Mark CSV data as authentic (requires --csv)."
    ),
    n_bars: int = typer.Option(1400, "--n-bars", help="Synthetic candle count"),
    seed: int = typer.Option(20240101, "--seed", help="Synthetic generator seed"),
    momentum_strength: float = typer.Option(
        0.0, "--momentum-strength", help="Planted synthetic momentum coefficient"
    ),
    target_type: str = typer.Option(
        "direction_binary",
        "--target-type",
        help="direction_binary | direction_ternary | return_continuous",
    ),
    horizon: int = typer.Option(1, "--horizon", help="Forward label horizon in bars"),
    flat_threshold_pips: float = typer.Option(
        0.0, "--flat-threshold-pips", help="Flat deadband for directional labels"
    ),
    include_quotes: bool = typer.Option(
        True, "--quotes/--no-quotes", help="Include bid/ask spread features"
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing directory"),
) -> None:
    """Align causal features and forward labels, then persist the prepared dataset."""
    spec = _target_spec(target_type, horizon, flat_threshold_pips)
    prepared = _build_prepared(
        csv_path=csv_path,
        symbol=symbol,
        timeframe=timeframe,
        synthetic=not authentic,
        n_bars=n_bars,
        seed=seed,
        momentum_strength=momentum_strength,
        target_spec=spec,
        include_quotes=include_quotes,
    )
    save_prepared_dataset(output, prepared, overwrite=overwrite)

    table = Table(title="Prepared Dataset")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold green")
    table.add_row("Path", str(output.resolve()))
    table.add_row("Symbol", prepared.symbol)
    table.add_row("Rows", str(prepared.row_count))
    table.add_row("Features", str(prepared.dataset.feature_dim))
    table.add_row("Target", spec.target_type.value)
    table.add_row("Horizon", str(spec.horizon_bars))
    table.add_row("Synthetic", str(prepared.is_synthetic))
    table.add_row("Fingerprint", prepared.fingerprint[:16] + "...")
    console.print(table)
    if prepared.is_synthetic:
        console.print(
            Panel(
                "[bold yellow]Synthetic engineering fixture only.\n"
                "No profitability claim is possible from this data.[/bold yellow]"
            )
        )


@ml_app.command("train", help="Train a calibrated holdout bundle and save it.")
def train_command(
    prepared_dir: Path = typer.Option(
        DEFAULT_WORKSPACE / "prepared", "--prepared", help="Prepared dataset directory"
    ),
    model_name: str = typer.Option(
        "logistic_regression", "--model", "-m", help=", ".join(AVAILABLE_MODEL_NAMES)
    ),
    bundle_dir: Path = typer.Option(
        DEFAULT_WORKSPACE / "bundle", "--bundle", "-b", help="Output bundle directory"
    ),
    report_path: Path | None = typer.Option(
        DEFAULT_WORKSPACE / "reports", "--report", help="Output Markdown report path"
    ),
    db_path: Path = typer.Option(DEFAULT_DB, "--db", help="Experiment SQLite database"),
    calibrator_method: str = typer.Option("platt", "--calibrator", help="platt | isotonic | none"),
    train_ratio: float = typer.Option(0.70, "--train-ratio"),
    val_ratio: float = typer.Option(0.15, "--val-ratio"),
    test_ratio: float = typer.Option(0.15, "--test-ratio"),
    purge_bars: int = typer.Option(0, "--purge-bars"),
    embargo_bars: int = typer.Option(0, "--embargo-bars"),
    seed: int = typer.Option(42, "--seed"),
    quick: bool = typer.Option(True, "--quick/--full", help="Smaller model configurations"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Fit on the training split, calibrate on validation, test once on holdout."""
    prepared = load_prepared_dataset(prepared_dir)
    config = _holdout_config(train_ratio, val_ratio, test_ratio, purge_bars, embargo_bars)
    model = create_model(model_name, seed=seed, quick=quick)
    calibrator = _make_calibrator(calibrator_method)

    training = train_holdout(prepared, model, config=config, calibrator=calibrator)
    manifest = save_training_bundle(bundle_dir, training, prepared, overwrite=overwrite)

    _bundle, evaluation = load_and_evaluate_bundle(bundle_dir, prepared, config=config)
    assessment = assess_bundle_evaluation(
        evaluation,
        is_synthetic=prepared.is_synthetic,
        split_type="holdout",
    )

    metrics_payload = {
        "holdout": evaluation.metrics.to_dict(),
        "assessment": assessment.to_dict(),
    }
    _log_run(
        db_path,
        experiment_name="holdout_train",
        prepared=prepared,
        model_name=evaluation.model_name,
        split_type="holdout",
        calibration_method=calibrator_method or "none",
        assessment=assessment,
        metrics=metrics_payload,
        config={
            **config.to_dict(),
            "model": model_name,
            "seed": seed,
            "calibrator": calibrator_method,
            "quick": quick,
        },
        artifact_path=bundle_dir,
        artifact_manifest_sha256=manifest.digest(),
        notes="Holdout-only run. Holdout was not used for fitting or calibration.",
        tags=["phase4", "holdout", str(assessment.verdict)],
    )

    context = _context(
        prepared,
        title="Phase 4 Holdout Training Report",
        model_name=evaluation.model_name,
        split_type="holdout",
        config={
            **config.to_dict(),
            "calibrator": calibrator_method,
            "seed": seed,
            "quick": quick,
        },
    )
    markdown = render_phase4_markdown(
        context,
        assessment,
        holdout=training.result,
        bundle_evaluation=evaluation,
    )
    _write_and_report(report_path, markdown)
    console.print(training.result.summary())
    _print_assessment(assessment)
    console.print(f"Bundle: [green]{bundle_dir.resolve()}[/green]")


@ml_app.command("evaluate", help="Verify and evaluate a saved bundle on the holdout.")
def evaluate_command(
    prepared_dir: Path = typer.Option(
        DEFAULT_WORKSPACE / "prepared", "--prepared", help="Prepared dataset directory"
    ),
    bundle_dir: Path = typer.Option(
        DEFAULT_WORKSPACE / "bundle", "--bundle", "-b", help="Bundle directory"
    ),
    report_path: Path | None = typer.Option(
        DEFAULT_WORKSPACE / "reports" / "evaluation.md", "--report"
    ),
    db_path: Path | None = typer.Option(None, "--db", help="Optional tracker database"),
    calibrator_method: str = typer.Option("platt", "--calibrator"),
) -> None:
    """Reload a verified bundle and evaluate it on the untouched holdout split."""
    prepared = load_prepared_dataset(prepared_dir)
    _bundle, evaluation = load_and_evaluate_bundle(bundle_dir, prepared)
    assessment = assess_bundle_evaluation(
        evaluation,
        is_synthetic=prepared.is_synthetic,
        split_type="holdout",
    )
    if db_path is not None:
        _log_run(
            db_path,
            experiment_name="holdout_evaluate",
            prepared=prepared,
            model_name=evaluation.model_name,
            split_type="holdout",
            calibration_method=calibrator_method,
            assessment=assessment,
            metrics={
                "holdout": evaluation.metrics.to_dict(),
                "assessment": assessment.to_dict(),
            },
            config=evaluation.config.to_dict(),
            artifact_path=bundle_dir,
            notes="Saved-bundle re-evaluation.",
            tags=["phase4", "evaluate", str(assessment.verdict)],
        )
    context = _context(
        prepared,
        title="Phase 4 Saved-Bundle Evaluation Report",
        model_name=evaluation.model_name,
        split_type="holdout",
        config=evaluation.config.to_dict(),
    )
    markdown = render_phase4_markdown(
        context,
        assessment,
        bundle_evaluation=evaluation,
    )
    _write_and_report(report_path, markdown)
    _print_assessment(assessment)
    console.print(f"Bundle verified: [green]{bundle_dir.resolve()}[/green]")


@ml_app.command("walk-forward", help="Run chronological walk-forward validation.")
def walk_forward_command(
    prepared_dir: Path = typer.Option(
        DEFAULT_WORKSPACE / "prepared", "--prepared", help="Prepared dataset directory"
    ),
    model_name: str = typer.Option("logistic_regression", "--model", "-m"),
    folds: int = typer.Option(4, "--folds", "-k"),
    calibrator_method: str = typer.Option("platt", "--calibrator", help="platt | isotonic | none"),
    purge_bars: int = typer.Option(1, "--purge-bars"),
    embargo_bars: int = typer.Option(0, "--embargo-bars"),
    report_path: Path | None = typer.Option(
        DEFAULT_WORKSPACE / "reports" / "walk_forward.md", "--report"
    ),
    db_path: Path | None = typer.Option(DEFAULT_DB, "--db"),
    seed: int = typer.Option(42, "--seed"),
    quick: bool = typer.Option(True, "--quick/--full"),
) -> None:
    """Evaluate expanding-window folds with purge, embargo, and pooled OOF metrics."""
    prepared = load_prepared_dataset(prepared_dir)
    calibrator_factory = (
        (lambda: ProbabilityCalibrator(method=calibrator_method))
        if _make_calibrator(calibrator_method) is not None
        else None
    )
    config = WalkForwardConfig(
        n_splits=folds,
        train_size=None,
        val_size=0.15 if calibrator_factory is not None else None,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    result = run_walk_forward(
        prepared,
        lambda: create_model(model_name, seed=seed, quick=quick),
        config=config,
        calibrator_factory=calibrator_factory,
    )
    pooled = result.pooled_metrics
    base_rate = float(np.mean(result.oof_y_true)) if len(result.oof_y_true) else 0.5
    ece = result.pooled_calibration_metrics.ece if result.pooled_calibration_metrics else None
    assessment = assess_evidence(
        is_synthetic=prepared.is_synthetic,
        split_type="walk_forward",
        sample_count=pooled.sample_count,
        accuracy=pooled.accuracy,
        base_rate=base_rate,
        roc_auc=pooled.roc_auc,
        brier_score=pooled.brier_score,
        ece=ece,
        n_folds=result.n_folds,
    )
    if db_path is not None:
        _log_run(
            db_path,
            experiment_name="walk_forward",
            prepared=prepared,
            model_name=model_name,
            split_type="walk_forward",
            calibration_method=calibrator_method,
            assessment=assessment,
            metrics={
                "pooled": pooled.to_dict(),
                "assessment": assessment.to_dict(),
            },
            config=result.to_dict()["aggregate_metrics"],
            notes="Chronological walk-forward with purge/embargo.",
            tags=["phase4", "walk_forward", str(assessment.verdict)],
        )
    context = _context(
        prepared,
        title="Phase 4 Walk-Forward Evaluation Report",
        model_name=model_name,
        split_type="walk_forward",
        config={
            "n_splits": folds,
            "purge_bars": purge_bars,
            "embargo_bars": embargo_bars,
            "calibrator": calibrator_method,
            "seed": seed,
        },
    )
    markdown = render_phase4_markdown(
        context,
        assessment,
        walk_forward=result,
    )
    _write_and_report(report_path, markdown)
    console.print(result.summary())
    _print_assessment(assessment)


@ml_app.command("compare", help="Benchmark multiple models under identical settings.")
def compare_command(
    prepared_dir: Path = typer.Option(
        DEFAULT_WORKSPACE / "prepared", "--prepared", help="Prepared dataset directory"
    ),
    models: str = typer.Option(
        "majority_class,seeded_random,logistic_regression,random_forest,hist_gradient_boosting",
        "--models",
        help="Comma-separated model names",
    ),
    split: str = typer.Option("holdout", "--split", help="holdout | walk_forward"),
    calibrator_method: str = typer.Option("platt", "--calibrator"),
    payoff: str = typer.Option("binary", "--payoff", help="binary | forex"),
    report_path: Path | None = typer.Option(
        DEFAULT_WORKSPACE / "reports" / "comparison.md", "--report"
    ),
    db_path: Path | None = typer.Option(DEFAULT_DB, "--db"),
    seed: int = typer.Option(42, "--seed"),
    quick: bool = typer.Option(True, "--quick/--full"),
) -> None:
    """Run a leaderboard comparison with paired McNemar tests."""
    prepared = load_prepared_dataset(prepared_dir)
    requested = [item.strip() for item in models.split(",") if item.strip()]
    known_aliases = {
        "majority",
        "random",
        "logistic",
        "logreg",
        "momentum",
        "rf",
        "hgb",
        "voting",
        "weighted",
        "stacking",
    }
    unknown = [
        item
        for item in requested
        if item.strip().lower() not in set(AVAILABLE_MODEL_NAMES)
        and item.strip().lower() not in known_aliases
    ]
    if unknown:
        raise typer.BadParameter(f"Unknown models: {', '.join(unknown)}")

    split_norm = split.strip().lower()
    if split_norm not in ("holdout", "walk_forward"):
        raise typer.BadParameter("--split must be 'holdout' or 'walk_forward'.")
    payoff_norm = payoff.strip().lower()
    if payoff_norm not in ("binary", "forex"):
        raise typer.BadParameter("--payoff must be 'binary' or 'forex'.")

    comparison_config = ComparisonConfig(
        split_type=(
            EvaluationSplitType.HOLDOUT
            if split_norm == "holdout"
            else EvaluationSplitType.WALK_FORWARD
        ),
        walk_forward_config=WalkForwardConfig(
            n_splits=4,
            val_size=0.15 if _make_calibrator(calibrator_method) else None,
            purge_bars=1,
        ),
        calibrator_method=calibrator_method,
        payoff_type=(
            PayoffAssumptionType.BINARY if payoff_norm == "binary" else PayoffAssumptionType.FOREX
        ),
    )
    factories = {
        name: (lambda n=name: create_model(n, seed=seed, quick=quick)) for name in requested
    }
    result = run_comparison(prepared, factories, config=comparison_config)
    best = result.best_model("brier_score")
    base_rate = float(np.mean(best.y_true)) if best.y_true is not None else 0.5
    assessment = assess_evidence(
        is_synthetic=prepared.is_synthetic,
        split_type=split_norm,
        sample_count=best.sample_count,
        accuracy=best.accuracy,
        base_rate=base_rate,
        roc_auc=best.roc_auc,
        brier_score=best.brier_score,
        ece=best.ece,
        n_folds=result.metadata.get("folds", 1)
        if isinstance(result.metadata.get("folds", 1), int)
        else 1,
        trade_count=best.trade_count,
        win_rate=best.win_rate,
        payout_rate=comparison_config.binary_payout_rate,
    )
    if db_path is not None:
        _log_run(
            db_path,
            experiment_name="model_comparison",
            prepared=prepared,
            model_name=best.model_name,
            split_type=split_norm,
            calibration_method=calibrator_method,
            assessment=assessment,
            metrics={
                "leaderboard": [row.to_dict() for row in result.rows],
                "assessment": assessment.to_dict(),
            },
            config={
                "models": requested,
                "split": split_norm,
                "payoff": payoff_norm,
                "seed": seed,
                "quick": quick,
            },
            notes="Leaderboard selection is exploratory; significance is not corrected.",
            tags=["phase4", "comparison", str(assessment.verdict)],
        )
    context = _context(
        prepared,
        title="Phase 4 Model Comparison Report",
        model_name=best.model_name,
        split_type=split_norm,
        config={
            "models": requested,
            "payoff": payoff_norm,
            "calibrator": calibrator_method,
            "seed": seed,
            "quick": quick,
        },
    )
    markdown = render_phase4_markdown(
        context,
        assessment,
        comparison=result,
    )
    _write_and_report(report_path, markdown)
    console.print(result.summary())
    _print_assessment(assessment)


@ml_app.command("backtest", help="Backtest a saved bundle over the holdout window, offline.")
def backtest_command(
    prepared_dir: Path = typer.Option(
        DEFAULT_WORKSPACE / "prepared", "--prepared", help="Prepared dataset directory"
    ),
    bundle_dir: Path = typer.Option(
        DEFAULT_WORKSPACE / "bundle", "--bundle", "-b", help="Bundle directory"
    ),
    payoff: str = typer.Option("binary", "--payoff", help="binary | forex"),
    payout_rate: float = typer.Option(0.85, "--payout-rate"),
    stake: float = typer.Option(100.0, "--stake"),
    capital: float = typer.Option(10000.0, "--capital"),
    report_path: Path | None = typer.Option(
        DEFAULT_WORKSPACE / "reports" / "backtest.md", "--report"
    ),
    db_path: Path | None = typer.Option(DEFAULT_DB, "--db"),
) -> None:
    """Simulate payoff settlements on the untouched holdout using the Phase 3 engine."""
    prepared = load_prepared_dataset(prepared_dir)
    bundle, evaluation = load_and_evaluate_bundle(bundle_dir, prepared)
    outcome = run_bundle_backtest(
        bundle,
        prepared,
        payoff=payoff,
        payout_rate=payout_rate,
        stake=stake,
        capital=capital,
    )
    assessment = assess_bundle_evaluation(
        evaluation,
        is_synthetic=prepared.is_synthetic,
        split_type="holdout",
        payout_rate=payout_rate,
        trade_count=outcome.metrics.total_trades,
        win_rate=outcome.metrics.win_rate,
    )
    if db_path is not None:
        _log_run(
            db_path,
            experiment_name="offline_backtest",
            prepared=prepared,
            model_name=evaluation.model_name,
            split_type="holdout",
            calibration_method="bundle",
            assessment=assessment,
            metrics={
                "holdout": evaluation.metrics.to_dict(),
                "backtest": outcome.to_dict(),
                "assessment": assessment.to_dict(),
            },
            config={
                "payoff": payoff,
                "payout_rate": payout_rate,
                "stake": stake,
                "capital": capital,
            },
            artifact_path=bundle_dir,
            notes="Offline simulated settlements only; no orders were placed.",
            tags=["phase4", "backtest", str(assessment.verdict)],
        )
    context = _context(
        prepared,
        title="Phase 4 Offline Backtest Report",
        model_name=evaluation.model_name,
        split_type="holdout",
        config={
            "payoff": payoff,
            "payout_rate": payout_rate,
            "stake": stake,
            "capital": capital,
        },
    )
    markdown = render_phase4_markdown(
        context,
        assessment,
        bundle_evaluation=evaluation,
        backtest=outcome,
    )
    _write_and_report(report_path, markdown)
    console.print(
        f"Trades: {outcome.metrics.total_trades} | Net PnL: "
        f"{outcome.run.total_net_pnl:+.2f} | Win rate: {outcome.metrics.win_rate:.2%}"
    )
    _print_assessment(assessment)


@ml_app.command("list", help="List tracked experiments.")
def list_command(
    db_path: Path = typer.Option(DEFAULT_DB, "--db"),
    limit: int = typer.Option(20, "--limit"),
    name: str | None = typer.Option(None, "--name", help="Filter by experiment name"),
    verdict: str | None = typer.Option(None, "--verdict", help="Filter by verdict"),
) -> None:
    """Show recent experiment records from the local SQLite tracker."""
    if not db_path.is_file():
        console.print(f"[yellow]No experiment database at {db_path}[/yellow]")
        return
    tracker = ExperimentTracker(db_path)
    records = tracker.list_experiments(
        limit=limit,
        experiment_name=name,
        verdict=verdict,
    )
    if not records:
        console.print("[yellow]No matching experiments.[/yellow]")
        return
    table = Table(title=f"Experiments ({tracker.count()} total)")
    table.add_column("Run ID", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("Split")
    table.add_column("Verdict", style="magenta")
    for record in records:
        table.add_row(
            record.run_id[:12],
            record.created_at_utc[:19],
            record.experiment_name,
            record.model_name or "n/a",
            record.split_type or "n/a",
            record.verdict or "n/a",
        )
    console.print(table)


@ml_app.command("inspect", help="Inspect an experiment record or a saved bundle.")
def inspect_command(
    run_id: str | None = typer.Option(None, "--run-id", help="Experiment run ID"),
    bundle_dir: Path | None = typer.Option(None, "--bundle", help="Bundle directory"),
    db_path: Path = typer.Option(DEFAULT_DB, "--db"),
) -> None:
    """Print full metadata for a tracked run and/or a verified bundle."""
    if run_id is None and bundle_dir is None:
        raise typer.BadParameter("Provide --run-id and/or --bundle.")
    if run_id is not None:
        tracker = ExperimentTracker(db_path)
        record = tracker.get_experiment(run_id)
        console.print(
            Panel(
                json.dumps(record.to_dict(), indent=2, sort_keys=True),
                title=f"Experiment {record.run_id}",
            )
        )
    if bundle_dir is not None:
        bundle = load_model_bundle(bundle_dir)
        summary = {
            "model": bundle.model.get_metadata(),
            "feature_names": list(bundle.feature_names),
            "has_scaler": bundle.scaler is not None,
            "has_calibrator": bundle.calibrator is not None,
            "metadata": bundle.metadata,
        }
        console.print(
            Panel(
                json.dumps(summary, indent=2, sort_keys=True, default=str),
                title=f"Verified bundle {bundle_dir.resolve()}",
            )
        )


@ml_app.command("report", help="Render a Markdown summary for a tracked experiment.")
def report_command(
    run_id: str = typer.Option(..., "--run-id", help="Experiment run ID"),
    db_path: Path = typer.Option(DEFAULT_DB, "--db"),
    output: Path = typer.Option(DEFAULT_WORKSPACE / "reports" / "experiment.md", "--output", "-o"),
) -> None:
    """Write a compact, honest report from stored experiment metadata."""
    tracker = ExperimentTracker(db_path)
    record = tracker.get_experiment(run_id)
    lines = [
        f"# Tracked Experiment: {record.experiment_name}",
        "",
        f"- Run ID: `{record.run_id}`",
        f"- Created: `{record.created_at_utc}`",
        f"- Status: `{record.status}`",
        f"- Verdict: `{record.verdict or 'N/A'}`",
        f"- Evidence class: `{record.evidence_class or 'N/A'}`",
        f"- Model: `{record.model_name}`",
        f"- Split: `{record.split_type}`",
        f"- Symbol: `{record.dataset_symbol}`",
        f"- Rows: `{record.dataset_rows}`",
        f"- Dataset fingerprint: `{record.dataset_fingerprint}`",
        f"- Synthetic: `{record.is_synthetic}`",
        f"- Calibration: `{record.calibration_method}`",
        f"- Artifact: `{record.artifact_path}`",
        f"- Manifest SHA-256: `{record.artifact_manifest_sha256}`",
        f"- Notes: {record.notes}",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(record.metrics, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(record.config, indent=2, sort_keys=True, default=str),
        "```",
        "",
        f"> {PHASE4_DISCLAIMER}",
        "",
    ]
    destination = write_report(output, "\n".join(lines))
    console.print(f"[green]Report written to {destination}[/green]")


@ml_app.command("demo", help="Run the complete offline Phase 4 engineering demonstration.")
def demo_command(
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", help="Demo workspace"),
    n_bars: int = typer.Option(1400, "--n-bars"),
    seed: int = typer.Option(20240101, "--seed"),
    db_path: Path = typer.Option(DEFAULT_DB, "--db"),
) -> None:
    """Prepare, train, evaluate, walk-forward, compare, and backtest, end to end."""
    console.print(
        Panel(
            "[bold green]IQ Option Research Phase 4 - Offline Engineering "
            "Demonstration[/bold green]\n"
            "Causal features -> forward labels -> train-only scaling -> validation calibration\n"
            "-> untouched holdout -> walk-forward -> comparison -> payoff backtest.\n"
            "All data is deterministic synthetic. No broker or network access.",
        )
    )
    workspace.mkdir(parents=True, exist_ok=True)
    prepared_dir = workspace / "prepared"
    bundle_dir = workspace / "bundle"
    reports_dir = workspace / "reports"

    prepared = _build_prepared(
        csv_path=None,
        symbol="EURUSD",
        timeframe=60,
        synthetic=True,
        n_bars=n_bars,
        seed=seed,
        momentum_strength=0.45,
        target_spec=TargetSpec.binary(horizon_bars=1),
        include_quotes=True,
    )
    save_prepared_dataset(prepared_dir, prepared, overwrite=True)
    console.print(
        f"[green]1. Prepared {prepared.row_count} causal rows "
        f"({prepared.dataset.feature_dim} features).[/green]"
    )

    config = HoldoutSplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    training = train_holdout(
        prepared,
        create_model("logistic_regression", seed=seed, quick=True),
        config=config,
        calibrator=ProbabilityCalibrator(method="platt"),
    )
    manifest = save_training_bundle(bundle_dir, training, prepared, overwrite=True)
    bundle, evaluation = load_and_evaluate_bundle(bundle_dir, prepared, config=config)
    holdout_assessment = assess_bundle_evaluation(
        evaluation, is_synthetic=True, split_type="holdout"
    )
    console.print(f"[green]2. Holdout: {training.result.summary()}[/green]")

    walk_forward = run_walk_forward(
        prepared,
        lambda: create_model("logistic_regression", seed=seed, quick=True),
        config=WalkForwardConfig(
            n_splits=4,
            val_size=0.15,
            purge_bars=1,
        ),
        calibrator_factory=lambda: ProbabilityCalibrator(method="platt"),
    )
    console.print(f"[green]3. Walk-forward: {walk_forward.summary()}[/green]")

    comparison_models = {
        name: (lambda n=name: create_model(n, seed=seed, quick=True))
        for name in (
            "majority_class",
            "seeded_random",
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
        )
    }
    comparison = run_comparison(
        prepared,
        comparison_models,
        config=ComparisonConfig(
            split_type=EvaluationSplitType.HOLDOUT,
            calibrator_method="platt",
            payoff_type=PayoffAssumptionType.BINARY,
        ),
    )
    console.print("[green]4. Comparison leaderboard computed.[/green]")

    outcome: BacktestOutcome = run_bundle_backtest(
        bundle,
        prepared,
        config=config,
        payoff="binary",
        payout_rate=0.85,
        stake=100.0,
        capital=10000.0,
    )
    console.print(
        f"[green]5. Backtest: {outcome.metrics.total_trades} trades, "
        f"Net PnL {outcome.run.total_net_pnl:+.2f}.[/green]"
    )

    assessment = assess_bundle_evaluation(
        evaluation,
        is_synthetic=True,
        split_type="holdout",
        trade_count=outcome.metrics.total_trades,
        win_rate=outcome.metrics.win_rate,
    )
    context = _context(
        prepared,
        title="Phase 4 End-to-End Engineering Demonstration",
        model_name=evaluation.model_name,
        split_type="holdout+walk_forward",
        config={
            "holdout": config.to_dict(),
            "walk_forward": {
                "n_splits": 4,
                "val_size": 0.15,
                "purge_bars": 1,
            },
            "calibrator": "platt",
            "seed": seed,
        },
    )
    markdown = render_phase4_markdown(
        context,
        assessment,
        holdout=training.result,
        bundle_evaluation=evaluation,
        walk_forward=walk_forward,
        comparison=comparison,
        backtest=outcome,
    )
    report_path = write_report(reports_dir / "phase4_demo.md", markdown)
    console.print(f"[green]6. Report written to {report_path}[/green]")

    tracker = ExperimentTracker(db_path)
    tracker.log_experiment(
        "phase4_demo",
        model_name=evaluation.model_name,
        dataset_symbol=prepared.symbol,
        dataset_rows=prepared.row_count,
        dataset_fingerprint=prepared.fingerprint,
        is_synthetic=True,
        target_type=prepared.dataset.target_spec.target_type.value,
        horizon_bars=prepared.dataset.target_spec.horizon_bars,
        split_type="holdout+walk_forward",
        calibration_method="platt",
        evidence_class=assessment.evidence_class,
        verdict=assessment.verdict.value,
        metrics={
            "holdout": evaluation.metrics.to_dict(),
            "walk_forward": walk_forward.to_dict(),
            "backtest": outcome.to_dict(),
            "assessment": holdout_assessment.to_dict(),
        },
        config={
            "n_bars": n_bars,
            "seed": seed,
            "models": list(comparison_models),
        },
        artifact_path=bundle_dir,
        artifact_manifest_sha256=manifest.digest(),
        notes="Full engineering demonstration on synthetic fixtures.",
        tags=["phase4", "demo", assessment.verdict.value],
    )

    _print_assessment(assessment)
    console.print(
        Panel(
            "[bold yellow]DEMONSTRATION COMPLETE - SYNTHETIC ENGINEERING ONLY\n"
            "The pipeline works end to end. This does not establish a market edge.[/bold yellow]"
        )
    )
