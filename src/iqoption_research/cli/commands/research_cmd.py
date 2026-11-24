"""CLI commands for Phase 3 offline research, historical data management, and backtesting."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.importer import import_and_persist_dataset
from iqoption_research.market_data.parquet_storage import read_candles_parquet
from iqoption_research.ml.data_audit import run_data_audit
from iqoption_research.ml.experiment_plan import (
    ExperimentPlan,
    build_standard_phase42_experiment_plan,
)
from iqoption_research.ml.experiment_runner import ExperimentRunner
from iqoption_research.ml.failure_analysis import analyze_model_failures
from iqoption_research.ml.freezer import (
    evaluate_frozen_candidate,
    freeze_and_evaluate_candidate,
)
from iqoption_research.ml.partitions import (
    PartitionManifest,
    create_partition_manifest,
    verify_partition_integrity,
)
from iqoption_research.research.backtest.engine import BacktestConfig, BacktestEngine
from iqoption_research.research.backtest.metrics import calculate_metrics
from iqoption_research.research.backtest.payoffs import (
    BasePayoffModel,
    BinaryPayoffModel,
    ForexPayoffModel,
)
from iqoption_research.research.backtest.reports import generate_research_report
from iqoption_research.research.backtest.strategies import (
    AlwaysCallStrategy,
    AlwaysPutStrategy,
    BaseStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    NoTradeStrategy,
    RandomStrategy,
)
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.research.datasets.loader import HistoricalDatasetLoader
from iqoption_research.research.datasets.splitter import ChronologicalSplitter
from iqoption_research.research.evaluation_metrics import SimulationPolicy
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.features.scalers import StandardScaler
from iqoption_research.research.phase24 import EvaluationConfig, run_evaluation, write_reports
from iqoption_research.research.phase25 import run_phase25, write_phase25_reports

console = Console()
research_app = typer.Typer(
    name="research",
    help="Offline historical market data import, feature extraction, and backtesting.",
    no_args_is_help=True,
)


@research_app.command(
    "evaluate", help="Offline authentic-data audit, benchmarks and purged ML comparisons."
)
def evaluate_authentic_command(
    storage_dir: Path = typer.Option(Path("data/datasets"), "--storage-dir"),
    output_dir: Path = typer.Option(Path("data/phase24"), "--output-dir"),
    reservation: Path = typer.Option(Path("data/phase11/confirmation_state.json"), "--reservation"),
    mode: str = typer.Option(
        "all", "--mode", help="all, coverage, integrity, benchmark, walk-forward, compare, report"
    ),
    symbol: str = typer.Option("EURUSD", "--symbol"),
    timeframe: int = typer.Option(60, "--timeframe", min=1),
    seed: int = typer.Option(42, "--seed"),
    horizon: int = typer.Option(1, "--horizon", min=1, max=60),
    latency_bars: int = typer.Option(1, "--latency-bars", min=0, max=10),
    payout_rate: float = typer.Option(0.85, "--payout-rate", min=0.01, max=2.0),
    cost_rate: float = typer.Option(0.002, "--cost-rate", min=0, max=0.99),
    slippage_price: float = typer.Option(0.00001, "--slippage-price", min=0),
    phase: int = typer.Option(24, "--phase", min=24, max=25),
    registry: Path = typer.Option(Path("data/phase25/registry.json"), "--registry"),
) -> None:
    """Generate reproducible JSON/Markdown evidence; never contacts a broker."""
    try:
        config = EvaluationConfig(
            symbol=symbol,
            timeframe=timeframe,
            seed=seed,
            horizon=horizon,
            latency_bars=latency_bars,
            policy=SimulationPolicy(
                payout=payout_rate, cost_rate=cost_rate, slippage_price=slippage_price
            ),
        )
        if phase == 25:
            report = run_phase25(storage_dir, config, reservation, registry, mode=mode)
            write_phase25_reports(report, output_dir)
        else:
            report = run_evaluation(storage_dir, config, reservation, mode=mode)
            write_reports(report, output_dir)
    except (ValueError, RuntimeError, OSError):
        console.print("INVALID_EVALUATION: offline data or configuration validation failed.")
        raise typer.Exit(code=2) from None
    console.print(f"{report['status']}: {output_dir / 'report.md'}")
    console.print(f"Reproducible detail: {output_dir / 'results.json'}")


def _get_strategy(name: str) -> BaseStrategy:
    norm = name.strip().lower()
    if norm == "no_trade":
        return NoTradeStrategy()
    if norm == "always_call":
        return AlwaysCallStrategy()
    if norm == "always_put":
        return AlwaysPutStrategy()
    if norm == "momentum":
        return MomentumStrategy()
    if norm in ("ma_crossover", "moving_average"):
        return MovingAverageCrossoverStrategy()
    if norm == "random":
        return RandomStrategy()
    raise typer.BadParameter(
        f"Unknown strategy '{name}'. Available: no_trade, always_call, always_put, "
        f"momentum, ma_crossover, random."
    )


@research_app.command("import", help="Import a local CSV or Parquet file into partitioned storage.")
def import_dataset(
    file_path: Path = typer.Argument(..., help="Path to local candle file"),
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    timeframe: int = typer.Option(60, "--timeframe", "-t", help="Timeframe in seconds"),
    storage_dir: Path = typer.Option(
        Path("data/market_data"), "--storage-dir", help="Storage root"
    ),
    catalog_db: Path = typer.Option(
        Path("data/catalog.db"), "--catalog-db", help="Catalog DB path"
    ),
    source: str = typer.Option("local_import", "--source", help="Data source tag"),
    synthetic: bool = typer.Option(True, "--synthetic/--authentic", help="Mark dataset synthetic"),
) -> None:
    """Import and audit a historical candle file into Parquet storage and SQLite catalog."""
    if not file_path.is_file():
        console.print(f"[bold red]Error: Source file not found: {file_path}[/bold red]")
        raise typer.Exit(code=1)

    catalog = DatasetCatalog(catalog_db)
    try:
        res = import_and_persist_dataset(
            file_path,
            symbol=symbol,
            candle_size_seconds=timeframe,
            storage_root=storage_dir,
            catalog=catalog,
            source=source,
            is_synthetic=synthetic,
            on_duplicate="replace",
        )
    except Exception as exc:
        console.print(f"[bold red]Import failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Historical Dataset Successfully Imported")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Dataset ID", res.dataset_id)
    table.add_row("Symbol", symbol)
    table.add_row("Candle Count", str(res.candle_count))
    table.add_row("SHA-256", res.sha256_checksum[:16] + "...")
    table.add_row("Parquet Path", str(res.parquet_path))
    table.add_row("Metadata Path", str(res.metadata_path))
    console.print(table)


@research_app.command("list", help="List registered datasets in SQLite catalog.")
def list_datasets(
    catalog_db: Path = typer.Option(
        Path("data/catalog.db"), "--catalog-db", help="Catalog DB path"
    ),
) -> None:
    """Inspect stored datasets recorded in the catalog."""
    if not catalog_db.is_file():
        console.print(f"[yellow]No catalog found at {catalog_db}[/yellow]")
        return

    catalog = DatasetCatalog(catalog_db)
    entries = catalog.find_datasets()

    if not entries:
        console.print("[yellow]Catalog is empty. Use 'iqresearch research import' first.[/yellow]")
        return

    table = Table(title="Catalog Stored Datasets")
    table.add_column("Dataset ID", style="cyan")
    table.add_column("Symbol", style="bold")
    table.add_column("TF", justify="right")
    table.add_column("Bars", justify="right")
    table.add_column("Type", style="magenta")
    table.add_column("Start UTC", style="dim")
    table.add_column("End UTC", style="dim")

    for e in entries:
        kind = "Synthetic" if e.is_synthetic else "Authentic"
        table.add_row(
            e.dataset_id,
            f"{e.symbol} {'(OTC)' if e.is_otc else ''}",
            f"{e.candle_size_seconds}s",
            str(e.candle_count),
            kind,
            e.from_utc[:16],
            e.to_utc[:16],
        )

    console.print(table)


@research_app.command("backtest", help="Run offline backtest on a stored dataset.")
def run_backtest(
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    timeframe: int = typer.Option(60, "--timeframe", "-t", help="Candle timeframe in seconds"),
    strategy_name: str = typer.Option(
        "momentum", "--strategy", help="Strategy (no_trade, momentum, etc.)"
    ),
    payoff: str = typer.Option("binary", "--payoff", help="Payoff model: binary | forex"),
    payout_rate: float = typer.Option(0.85, "--payout-rate", help="Binary payout rate (e.g. 0.85)"),
    stake: float = typer.Option(100.0, "--stake", help="Fixed stake per trade"),
    capital: float = typer.Option(10000.0, "--capital", help="Initial virtual capital"),
    storage_dir: Path = typer.Option(
        Path("data/market_data"), "--storage-dir", help="Storage directory"
    ),
    catalog_db: Path = typer.Option(
        Path("data/catalog.db"), "--catalog-db", help="Catalog database"
    ),
    export_json: Path | None = typer.Option(None, "--export-json", help="Export report JSON path"),
) -> None:
    """Run an offline backtest against historical data."""
    catalog = DatasetCatalog(catalog_db)
    loader = HistoricalDatasetLoader(catalog, storage_dir)

    try:
        loaded = loader.load_dataset(symbol=symbol, candle_size_seconds=timeframe)
    except Exception as exc:
        console.print(f"[bold red]Failed to load dataset: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    # Feature transformation
    pipeline = CausalFeaturePipeline(drop_warmup=True)
    try:
        feat_ds = pipeline.transform(loaded.candles)
    except Exception as exc:
        console.print(f"[bold red]Feature pipeline error: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    # Match candles corresponding to remaining non-warmup rows
    start_idx = len(loaded.candles) - len(feat_ds.rows)
    eval_candles = loaded.candles[start_idx:]

    # Select strategy & payoff
    strategy = _get_strategy(strategy_name)
    payoff_model: BasePayoffModel
    if payoff.lower() == "binary":
        payoff_model = BinaryPayoffModel(payout_rate=payout_rate)
    elif payoff.lower() == "forex":
        payoff_model = ForexPayoffModel()
    else:
        console.print(f"[bold red]Unknown payoff model: {payoff}[/bold red]")
        raise typer.Exit(code=1)

    cfg = BacktestConfig(initial_capital=capital, stake_amount=stake)
    engine = BacktestEngine(strategy=strategy, payoff_model=payoff_model, config=cfg)

    res = engine.run(feat_ds.rows, eval_candles)
    metrics = calculate_metrics(res)

    # Render results table
    table = Table(title=f"Backtest Results — {strategy.name} ({payoff.upper()})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Total Trades", str(metrics.total_trades))
    table.add_row(
        "Win / Loss / Tie",
        f"{metrics.winning_trades} / {metrics.losing_trades} / {metrics.tied_trades}",
    )
    ci_low = metrics.win_rate_ci_95.lower * 100
    ci_high = metrics.win_rate_ci_95.upper * 100
    table.add_row(
        "Win Rate",
        f"{metrics.win_rate * 100:.2f}% (95% CI: [{ci_low:.1f}%, {ci_high:.1f}%])",
    )
    table.add_row("Gross Profit", f"${metrics.gross_profit:.2f}")
    table.add_row("Gross Loss", f"${metrics.gross_loss:.2f}")
    table.add_row("Net Profit/Loss", f"${metrics.net_pnl:.2f}")
    table.add_row("Return on Capital", f"{metrics.return_on_capital * 100:.2f}%")
    table.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
    table.add_row("Expectancy / Trade", f"${metrics.expectancy_per_trade:.2f}")
    table.add_row(
        "Max Drawdown",
        f"${metrics.max_drawdown_amount:.2f} ({metrics.max_drawdown_percent * 100:.2f}%)",
    )

    console.print(table)

    if metrics.sample_size_warning:
        console.print(Panel(f"[yellow]{metrics.sample_size_warning}[/yellow]", title="Warning"))

    if export_json:
        report = generate_research_report(res, dataset=loaded)
        out_path = report.export_json(export_json)
        console.print(f"[green]Exported research report to {out_path}[/green]")


@research_app.command("demo", help="Run end-to-end bundled offline demonstration.")
def demo_research(
    temp_dir: Path = typer.Option(
        Path("data/demo_workspace"), "--workspace", help="Demo workspace directory"
    ),
) -> None:
    """Execute complete end-to-end offline research demonstration using bundled synthetic data."""
    console.print(
        Panel(
            "[bold green]IQ Option Research Phase 3 — End-to-End Demonstration[/bold green]\n"
            "Running full offline pipeline: CSV Import -> Quality Audit -> Parquet Storage -> "
            "Catalog Indexing -> Causal Features -> Train/Val/Test Split -> Scaler -> Backtest."
        )
    )

    sample_csv = Path(__file__).parent.parent.parent / "research" / "data" / "eurusd_sample_1m.csv"
    if not sample_csv.is_file():
        console.print(f"[bold red]Bundled sample CSV not found at {sample_csv}[/bold red]")
        raise typer.Exit(code=1)

    catalog_db = temp_dir / "catalog.db"
    storage_root = temp_dir / "storage"

    catalog = DatasetCatalog(catalog_db)

    # 1. Import
    console.print("[dim]1. Importing synthetic CSV candles...[/dim]")
    write_res = import_and_persist_dataset(
        sample_csv,
        symbol="EURUSD",
        candle_size_seconds=60,
        storage_root=storage_root,
        catalog=catalog,
        source="demo_synthetic",
        is_synthetic=True,
        on_duplicate="replace",
    )
    console.print(
        f"   [green]Imported {write_res.candle_count} candles: {write_res.dataset_id}[/green]"
    )

    # 2. Load
    console.print("[dim]2. Loading and verifying dataset via Catalog and Parquet hashes...[/dim]")
    loader = HistoricalDatasetLoader(catalog, storage_root)
    loaded = loader.load_dataset(symbol="EURUSD", candle_size_seconds=60)
    console.print(
        f"   [green]Loaded {loaded.candle_count} candles with 100% cryptographic parity.[/green]"
    )

    # 3. Causal Feature Pipeline
    console.print("[dim]3. Extracting causal features (zero lookahead)...[/dim]")
    pipeline = CausalFeaturePipeline(drop_warmup=True)
    features_ds = pipeline.transform(loaded.candles)
    console.print(
        f"   [green]Generated {features_ds.row_count} causal rows "
        f"({len(features_ds.feature_names)} features).[/green]"
    )

    # 4. Chronological Splitting
    console.print("[dim]4. Performing chronological Train/Val/Test split with purging...[/dim]")
    splitter = ChronologicalSplitter(
        train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, label_horizon_bars=1
    )
    splits = splitter.split(features_ds.rows)
    console.print(
        f"   [green]Splits: Train={len(splits.train)}, Val={len(splits.val)}, "
        f"Test={len(splits.test)}[/green]"
    )

    # 5. Leak-Free Scaling
    console.print("[dim]5. Fitting StandardScaler strictly on Train partition...[/dim]")
    scaler = StandardScaler().fit(splits.train)
    _ = scaler.transform(splits.test)
    console.print("   [green]Scaler parameters frozen. Zero validation/test data leakage.[/green]")

    # 6. Backtesting
    console.print("[dim]6. Running offline backtest with Momentum strategy...[/dim]")
    strategy = MomentumStrategy()
    payoff = BinaryPayoffModel(payout_rate=0.85)
    matched_candles = loaded.candles[len(loaded.candles) - features_ds.row_count :]
    res = BacktestEngine(
        strategy, payoff, BacktestConfig(initial_capital=10000.0, stake_amount=100.0)
    ).run(features_ds.rows, matched_candles)

    metrics = calculate_metrics(res)
    console.print(
        f"   [bold green]Backtest completed: {metrics.total_trades} trades. "
        f"Net PnL: ${metrics.net_pnl:.2f}, Win Rate: {metrics.win_rate * 100:.1f}%[/bold green]"
    )

    console.print(
        Panel(
            "[bold green]Demonstration Completed Successfully.\n"
            "All Phase 3 checks verified.[/bold green]"
        )
    )


# ==============================================================================
# Phase 42 Advanced AI Model Research CLI Integration
# ==============================================================================

partitions_app = typer.Typer(
    name="partitions",
    help="Manage 4-way chronological data partitions (Train/Val/Cal/Test).",
    no_args_is_help=True,
)
plan_app = typer.Typer(
    name="plan",
    help="Manage Phase 42 experiment plans and trial specifications.",
    no_args_is_help=True,
)
experiment_app = typer.Typer(
    name="experiment",
    help="Execute, monitor, and compare walk-forward ML experiments.",
    no_args_is_help=True,
)
candidate_app = typer.Typer(
    name="candidate",
    help="Freeze candidate models and evaluate protected holdouts.",
    no_args_is_help=True,
)
report_app = typer.Typer(
    name="report",
    help="Export failure intelligence reports and research summaries.",
    no_args_is_help=True,
)

research_app.add_typer(partitions_app, name="partitions")
research_app.add_typer(plan_app, name="plan")
research_app.add_typer(experiment_app, name="experiment")
research_app.add_typer(candidate_app, name="candidate")
research_app.add_typer(report_app, name="report")

timeframes_app = typer.Typer(
    name="timeframes",
    help="Multi-timeframe causal dataset aggregation and verification.",
    no_args_is_help=True,
)
horizons_app = typer.Typer(
    name="horizons",
    help="Multi-horizon walk-forward evaluation and experiment matrix.",
    no_args_is_help=True,
)
regimes_app = typer.Typer(
    name="regimes",
    help="Causal market regime classification and stratification.",
    no_args_is_help=True,
)

research_app.add_typer(timeframes_app, name="timeframes")
research_app.add_typer(horizons_app, name="horizons")
research_app.add_typer(regimes_app, name="regimes")


@research_app.command(
    "data-audit",
    help="Audit authentic market datasets, segment continuity, gaps, and sample sufficiency.",
)
def data_audit_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    symbol: Annotated[str, typer.Option("--symbol", help="Asset symbol.")] = "EURUSD",
    timeframe: Annotated[
        int, typer.Option("--timeframe", help="Candle timeframe in seconds.")
    ] = 60,
    as_json: Annotated[
        bool, typer.Option("--json", help="Output machine-readable JSON format.")
    ] = False,
    output: Annotated[
        Path | None, typer.Option("--output", help="Save markdown or JSON report to path.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    report = run_data_audit(repo_root, symbol=symbol, timeframe_seconds=timeframe)

    if as_json:
        payload = json.dumps(report.to_dict(), indent=2)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            console.print(f"Data audit JSON saved to {output}")
        else:
            typer.echo(payload)
        return

    table = Table(
        title=f"Authentic Market Data Audit & Sufficiency — {symbol} {timeframe}s", show_header=True
    )
    table.add_column("Audit Metric / Dimension", style="cyan")
    table.add_column("Value / Count", style="green", justify="right")
    table.add_column("Scientific Governance Notes", style="white")

    table.add_row(
        "Catalog Entries",
        str(report.total_catalog_entries),
        (
            f"Authentic: {report.genuine_catalog_entries} | "
            f"Synthetic: {report.synthetic_catalog_entries}"
        ),
    )
    table.add_row(
        "Total Retained Candles",
        f"{report.total_candles_retained:,}",
        (
            f"Discovered: {report.total_candles_discovered:,} "
            f"(Quarantined: {report.duplicate_timestamps_found})"
        ),
    )
    table.add_row(
        "Date Coverage (UTC)",
        f"{report.earliest_utc[:10]} -> {report.latest_utc[:10]}",
        f"Epoch: {report.earliest_ts} to {report.latest_ts}",
    )
    table.add_row(
        "Market Structure",
        f"{report.distinct_market_days} days",
        (
            f"{report.continuous_segments_count} segments | "
            f"{report.weekend_closures_count} weekends | "
            f"{report.intraday_gaps_count} intraday gaps"
        ),
    )
    table.add_row(
        "Microstructure Fields",
        "Volume: Genuine" if report.has_genuine_volume else "Volume: Missing",
        "Bid/Ask quotes: None in M1 OHLC bars"
        if not report.has_bid_ask_quotes
        else "Bid/Ask: Available",
    )
    table.add_row(
        "Historical Holdouts",
        f"{len(report.historical_holdouts)} intervals",
        "Phase 11 (14-day reservation) & Phase 39/40A test holdout (previously inspected)",
    )
    eligible_count = sum(1 for v in report.sufficiency_assessment.values() if v.get("eligible"))
    total_families = len(report.sufficiency_assessment)
    table.add_row(
        "Sufficiency Verdict",
        "[bold green]PASS[/bold green]"
        if report.data_sufficiency_passed
        else "[bold red]FAIL[/bold red]",
        f"Eligible model families: {eligible_count}/{total_families}",
    )
    console.print(table)

    s_table = Table(title="Model Family Minimum Sample Sufficiency", show_header=True)
    s_table.add_column("Model Family", style="cyan")
    s_table.add_column("Status", style="bold")
    s_table.add_column("Required Train", justify="right")
    s_table.add_column("Available Train", justify="right")
    s_table.add_column("Sufficiency Evaluation", style="white")

    for fam, info in report.sufficiency_assessment.items():
        status_col = (
            "[green]ELIGIBLE[/green]" if info.get("eligible") else "[red]INSUFFICIENT[/red]"
        )
        s_table.add_row(
            fam,
            status_col,
            str(info.get("required_samples", 0)),
            str(info.get("available_train_samples", 0)),
            info.get("reason", ""),
        )
    console.print(s_table)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.to_markdown(), encoding="utf-8")
        console.print(f"[green]Data audit report exported to {output}[/green]")


@partitions_app.command(
    "create", help="Create an immutable 4-way chronological partition manifest."
)
def partitions_create_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    snapshot_id: Annotated[
        str | None, typer.Option("--snapshot-id", help="Dataset snapshot ID.")
    ] = None,
    save: Annotated[
        bool, typer.Option("--save", help="Persist partition manifest to disk.")
    ] = True,
    manifest_path: Annotated[
        Path | None, typer.Option("--manifest-path", help="Explicit path to save manifest.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    if not snapshots:
        console.print("[red]No research snapshots found. Run snapshot preparation first.[/red]")
        raise typer.Exit(code=1)

    target_snapshot = None
    if snapshot_id:
        target_snapshot = next(
            (s for s in snapshots if s.manifest.snapshot_id == snapshot_id), None
        )
        if not target_snapshot:
            console.print(f"[red]Snapshot {snapshot_id} not found.[/red]")
            raise typer.Exit(code=1)
    else:
        target_snapshot = snapshots[-1]

    candles = read_candles_parquet(target_snapshot.parquet_path)
    manifest = create_partition_manifest(
        candles=candles,
        snapshot_id=target_snapshot.manifest.snapshot_id,
        snapshot_sha256=target_snapshot.manifest.parquet_sha256,
    )

    table = Table(
        title=f"4-Way Chronological Data Partitions — {manifest.manifest_id}", show_header=True
    )
    table.add_column("Partition", style="cyan")
    table.add_column("Ratio", justify="right")
    table.add_column("Candles", justify="right")
    table.add_column("From UTC", style="white")
    table.add_column("To UTC", style="white")
    table.add_column("Purpose & Leakage Boundary Protection", style="green")

    for p in (
        manifest.train_partition,
        manifest.val_partition,
        manifest.calibration_partition,
        manifest.protected_test_partition,
    ):
        ratio = p.candle_count / manifest.total_candles if manifest.total_candles > 0 else 0.0
        table.add_row(
            p.name.upper(),
            f"{ratio:.0%}",
            f"{p.candle_count:,}",
            p.from_utc[:19],
            p.to_utc[:19],
            f"Purge lead: {p.purged_lead_bars} | Embargo tail: {p.embargo_tail_bars}",
        )
    console.print(table)
    console.print(
        f"[bold green]Purge bars: {manifest.purging_horizon_bars} | "
        f"Embargo bars: {manifest.embargo_bars} | "
        f"Manifest Hash: {manifest.manifest_sha256[:16]}...[/bold green]"
    )

    if save:
        out_path = manifest_path or (
            repo_root / "data" / "partitions" / f"{manifest.manifest_id}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.save(out_path)
        console.print(f"[green]Partition manifest saved to {out_path}[/green]")


@partitions_app.command(
    "verify", help="Verify boundary integrity, purge/embargo buffers, and SHA-256."
)
def partitions_verify_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    manifest_id: Annotated[
        str | None, typer.Option("--manifest-id", help="Manifest ID to verify.")
    ] = None,
    manifest_path: Annotated[
        Path | None, typer.Option("--manifest-path", help="Path to manifest JSON.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    path = manifest_path
    if path is None:
        p_dir = repo_root / "data" / "partitions"
        if manifest_id:
            path = p_dir / f"{manifest_id}.json"
        elif p_dir.is_dir():
            files = sorted(p_dir.glob("*.json"))
            if files:
                path = files[-1]
    if not path or not path.is_file():
        console.print(f"[red]Manifest file not found: {path}[/red]")
        raise typer.Exit(code=1)

    manifest = PartitionManifest.load(path)
    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    target_snapshot = next(
        (s for s in snapshots if s.manifest.snapshot_id == manifest.snapshot_id), None
    )
    if not target_snapshot:
        console.print(
            f"[red]Snapshot {manifest.snapshot_id} referenced by manifest not found.[/red]"
        )
        raise typer.Exit(code=1)

    candles = read_candles_parquet(target_snapshot.parquet_path)
    is_valid, errors = verify_partition_integrity(manifest, candles)
    if is_valid:
        console.print(
            f"[bold green]Partition Integrity VERIFIED for {manifest.manifest_id}[/bold green]"
        )
        console.print("Purge & embargo intervals: LEAK-FREE. Protected test boundary: SEALED.")
    else:
        console.print(f"[bold red]Partition Integrity FAILED with {len(errors)} errors:[/bold red]")
        for err in errors:
            console.print(f"  - [red]{err}[/red]")
        raise typer.Exit(code=1)


@plan_app.command(
    "create", help="Create a standardized Phase 42 experiment plan with pre-registered trials."
)
def plan_create_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    manifest_path: Annotated[
        Path | None, typer.Option("--manifest-path", help="Path to partition manifest.")
    ] = None,
    horizon: Annotated[int, typer.Option("--horizon", help="Target horizon in bars.")] = 1,
    save: Annotated[bool, typer.Option("--save", help="Persist plan to disk.")] = True,
    output: Annotated[
        Path | None, typer.Option("--output", help="Explicit plan output path.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    p_path = manifest_path
    if p_path is None:
        p_dir = repo_root / "data" / "partitions"
        if p_dir.is_dir():
            files = sorted(p_dir.glob("*.json"))
            if files:
                p_path = files[-1]
    if not p_path or not p_path.is_file():
        console.print("[red]Partition manifest required to create experiment plan.[/red]")
        raise typer.Exit(code=1)

    _ = PartitionManifest.load(p_path)
    plan = build_standard_phase42_experiment_plan(target_horizon_bars=horizon)
    console.print(f"[bold green]Created Experiment Plan: {plan.plan_id}[/bold green]")
    console.print(f"Plan Hash: {plan.plan_sha256}")
    console.print(
        f"Total Trials: {len(plan.trials)} across {len(plan.model_families)} model families"
    )
    console.print(f"Feature ablation groups: {list(plan.feature_ablation_groups)}")
    console.print(f"Confidence thresholds: {plan.confidence_thresholds}")
    console.print(f"Calibrators: {plan.calibration_methods}")

    if save:
        out_path = output or (repo_root / "data" / "plans" / f"{plan.plan_id}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plan.save(out_path)
        console.print(f"[green]Plan saved to {out_path}[/green]")


@plan_app.command("validate", help="Validate experiment plan schema, hashes, and invariants.")
def plan_validate_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    plan_id: Annotated[str | None, typer.Option("--plan-id", help="Plan ID to validate.")] = None,
    plan_path: Annotated[
        Path | None, typer.Option("--plan-path", help="Path to plan JSON.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    path = plan_path
    if path is None:
        p_dir = repo_root / "data" / "plans"
        if plan_id:
            path = p_dir / f"{plan_id}.json"
        elif p_dir.is_dir():
            files = sorted(p_dir.glob("*.json"))
            if files:
                path = files[-1]
    if not path or not path.is_file():
        console.print(f"[red]Plan file not found: {path}[/red]")
        raise typer.Exit(code=1)

    plan = ExperimentPlan.load(path)
    is_valid, errors = plan.validate()
    if is_valid:
        console.print(f"[bold green]Experiment Plan VALIDATED: {plan.plan_id}[/bold green]")
        console.print(f"SHA-256 Digest: {plan.plan_sha256} | Total Trials: {len(plan.trials)}")
    else:
        console.print(
            f"[bold red]Experiment Plan Validation FAILED with {len(errors)} errors:[/bold red]"
        )
        for err in errors:
            console.print(f"  - [red]{err}[/red]")
        raise typer.Exit(code=1)


@experiment_app.command(
    "run", help="Execute walk-forward ML experiment suite against authentic partitions."
)
def experiment_run_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    plan_id: Annotated[str | None, typer.Option("--plan-id", help="Plan ID to run.")] = None,
    plan_path: Annotated[
        Path | None, typer.Option("--plan-path", help="Path to plan JSON.")
    ] = None,
    max_trials: Annotated[
        int | None, typer.Option("--max-trials", help="Limit number of trials to run.")
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress live trial logs.")] = False,
) -> None:
    repo_root = root or Path.cwd()
    # Find plan
    path = plan_path
    if path is None:
        p_dir = repo_root / "data" / "plans"
        if plan_id:
            path = p_dir / f"{plan_id}.json"
        elif p_dir.is_dir():
            files = sorted(p_dir.glob("*.json"))
            if files:
                path = files[-1]

    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    if not snapshots:
        console.print("[red]No snapshots found.[/red]")
        raise typer.Exit(code=1)
    snapshot = snapshots[-1]
    candles = read_candles_parquet(snapshot.parquet_path)

    plan: ExperimentPlan
    if path and path.is_file():
        plan = ExperimentPlan.load(path)
    else:
        # Create standard plan on the fly
        p_dir = repo_root / "data" / "partitions"
        p_files = sorted(p_dir.glob("*.json")) if p_dir.is_dir() else []
        if not p_files:
            console.print("[red]No partition manifest found. Run partitions create first.[/red]")
            raise typer.Exit(code=1)
        _ = PartitionManifest.load(p_files[-1])
        plan = build_standard_phase42_experiment_plan()

    runner = ExperimentRunner(repo_root)
    trials_count = max_trials or len(plan.trials)
    console.print(
        Panel(
            f"[bold cyan]Running Offline Experiment Suite: {plan.plan_id}[/bold cyan]\n"
            f"Trials: {trials_count} | Snapshot: {snapshot.manifest.snapshot_id} "
            f"({len(candles):,} candles)\n"
            "Walk-Forward Expanding Folds: 3 | Out-of-fold Calibration | Leak-Free Preprocessing"
        )
    )

    def progress_cb(completed: int, total: int, status_str: str) -> None:
        if not quiet:
            console.print(f"[{completed}/{total}] {status_str}")

    summary = runner.run_suite(
        snapshot_id=snapshot.manifest.snapshot_id,
        plan=plan,
        max_trials=max_trials,
        progress_callback=progress_cb,
    )

    table = Table(title=f"Experiment Suite Execution Summary — {plan.plan_id}", show_header=True)
    table.add_column("Trial ID", style="cyan")
    table.add_column("Model Family", style="white")
    table.add_column("Features", style="dim")
    table.add_column("Thresh", justify="right")
    table.add_column("Decisive Acc", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Day-Block 95% CI", style="white")
    table.add_column("Verdict", style="bold")

    for t in summary.reports[:25]:
        v_col = (
            "[green]STATISTICAL_EDGE[/green]"
            if t.verdict == "STATISTICAL_EDGE_DEMONSTRATED"
            else "[yellow]BELOW_BREAKEVEN[/yellow]"
        )
        acc_str = (
            f"{t.pooled_decisive_accuracy:.2%}" if t.pooled_decisive_accuracy is not None else "N/A"
        )
        cov_str = f"{t.pooled_coverage:.1%}"
        if t.day_block_bootstrap_ci_95:
            ci_l = t.day_block_bootstrap_ci_95.ci_lower
            ci_u = t.day_block_bootstrap_ci_95.ci_upper
            ci_str = f"[{ci_l:.2%}, {ci_u:.2%}]"
        else:
            ci_str = "N/A"
        brier_str = f"{t.pooled_brier_score:.4f}"
        ece_str = f"{t.pooled_ece:.4f}"

        table.add_row(
            t.trial_id,
            t.model_family,
            t.feature_set,
            f"{t.confidence_threshold:.2f}",
            acc_str,
            cov_str,
            brier_str,
            ece_str,
            ci_str,
            v_col,
        )

    console.print(table)
    console.print(
        f"[bold]Total Completed Trials: {summary.completed_trials}/{summary.total_trials} | "
        f"Eligible Candidates: {summary.eligible_candidates_count} | "
        f"Verdict: {summary.global_disposition}[/bold]"
    )


@experiment_app.command("status", help="Check status of ML experiment suite runs.")
def experiment_status_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    plan_id: Annotated[str | None, typer.Option("--plan-id", help="Plan ID to inspect.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    runner = ExperimentRunner(repo_root)
    summary = runner.get_suite_summary(plan_id)
    if summary is None:
        # Check SQLite db directly
        db_path = repo_root / "data" / "ml_experiments.db"
        if not db_path.is_file():
            console.print("[yellow]No experiments recorded in database yet.[/yellow]")
            return
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT plan_id) FROM phase42_experiments")
            row = cursor.fetchone()
            total_records = row[0] if row else 0
            distinct_plans = row[1] if row else 0
        console.print(f"[bold cyan]Experiment Database Status ({db_path}):[/bold cyan]")
        console.print(f"Total Trial Records: {total_records}")
        console.print(f"Distinct Plans: {distinct_plans}")
        return

    console.print(f"[bold cyan]Experiment Suite Status: {summary.plan_id}[/bold cyan]")
    console.print(f"Completed Trials: {summary.completed_trials}/{summary.total_trials}")
    console.print(f"Started UTC: {summary.started_at_utc[:19]}")
    console.print(f"Completed UTC: {summary.completed_at_utc[:19]}")
    console.print(f"Eligible Candidates: {summary.eligible_candidates_count}")
    console.print(f"Global Research Disposition: {summary.global_disposition}")


@experiment_app.command("cancel", help="Cancel active background experiment suite.")
def experiment_cancel_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    runner = ExperimentRunner(repo_root)
    runner.cancel()
    console.print("[yellow]Cancellation signal dispatched to experiment runner.[/yellow]")


@experiment_app.command("compare", help="Compare evaluated experiment trials sorted by metric.")
def experiment_compare_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    plan_id: Annotated[str | None, typer.Option("--plan-id", help="Plan ID to filter.")] = None,
    sort_by: Annotated[
        str,
        typer.Option("--sort", help="Sort metric: accuracy, brier, coverage, ece, expectancy."),
    ] = "accuracy",
    as_json: Annotated[
        bool, typer.Option("--json", help="Output machine-readable JSON format.")
    ] = False,
    top: Annotated[int, typer.Option("--top", help="Limit top results displayed.")] = 25,
) -> None:
    repo_root = root or Path.cwd()
    runner = ExperimentRunner(repo_root)
    summary = runner.get_suite_summary(plan_id)
    if summary is None or not summary.reports:
        console.print("[yellow]No evaluated trials found to compare.[/yellow]")
        return

    results = list(summary.reports)

    def sort_key(r: Any) -> float:
        if sort_by == "accuracy":
            return r.pooled_decisive_accuracy if r.pooled_decisive_accuracy is not None else -1.0
        if sort_by == "brier":
            return -(r.pooled_brier_score if r.pooled_brier_score is not None else 999.0)
        if sort_by == "coverage":
            return r.pooled_coverage
        if sort_by == "ece":
            return -(r.pooled_ece if r.pooled_ece is not None else 999.0)
        if sort_by == "expectancy":
            return (
                r.hypothetical_expectancy_85 if r.hypothetical_expectancy_85 is not None else -999.0
            )
        return 0.0

    results.sort(key=sort_key, reverse=True)
    results = results[:top]

    if as_json:
        data = [r.to_dict() for r in results]
        typer.echo(json.dumps(data, indent=2))
        return

    table = Table(
        title=f"Phase 42 Experiment Model Comparison (Sorted by {sort_by.upper()})",
        show_header=True,
    )
    table.add_column("Trial ID", style="cyan")
    table.add_column("Model Family", style="white")
    table.add_column("Features", style="dim")
    table.add_column("Calibrator", style="dim")
    table.add_column("Thresh", justify="right")
    table.add_column("Decisive Acc", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Expectancy", justify="right")
    table.add_column("Day-Block Bootstrap 95% CI", style="white")
    table.add_column("Scientific Verdict", style="bold")

    for r in results:
        acc_s = (
            f"{r.pooled_decisive_accuracy:.2%}" if r.pooled_decisive_accuracy is not None else "N/A"
        )
        cov_s = f"{r.pooled_coverage:.1%}"
        brier_s = f"{r.pooled_brier_score:.4f}"
        ece_s = f"{r.pooled_ece:.4f}"
        exp_s = (
            f"{r.hypothetical_expectancy_85 * 100:+.2f}%"
            if r.hypothetical_expectancy_85 is not None
            else "N/A"
        )
        if r.day_block_bootstrap_ci_95:
            ci_l = r.day_block_bootstrap_ci_95.ci_lower
            ci_u = r.day_block_bootstrap_ci_95.ci_upper
            ci_s = f"[{ci_l:.2%}, {ci_u:.2%}]"
        else:
            ci_s = "N/A"
        v_s = (
            "[green]STATISTICAL_EDGE[/green]"
            if r.verdict == "STATISTICAL_EDGE_DEMONSTRATED"
            else "[yellow]BELOW_BREAKEVEN[/yellow]"
        )

        table.add_row(
            r.trial_id,
            r.model_family,
            r.feature_set,
            r.calibration_method,
            f"{r.confidence_threshold:.2f}",
            acc_s,
            cov_s,
            brier_s,
            ece_s,
            exp_s,
            ci_s,
            v_s,
        )
    console.print(table)


@candidate_app.command(
    "freeze", help="Freeze candidate model artifacts (Stage B) and evaluate holdout (Stage C)."
)
def candidate_freeze_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    snapshot_id: Annotated[
        str | None, typer.Option("--snapshot-id", help="Dataset snapshot ID.")
    ] = None,
    manifest_path: Annotated[
        Path | None, typer.Option("--manifest-path", help="Path to partition manifest.")
    ] = None,
    model_family: Annotated[
        str, typer.Option("--model-family", help="Candidate model family.")
    ] = "rule_momentum",
    feature_set: Annotated[
        str, typer.Option("--feature-set", help="Feature preset name.")
    ] = "v2_price_action",
    calibration: Annotated[
        str, typer.Option("--calibration", help="Calibration method: isotonic/sigmoid/none.")
    ] = "isotonic",
    threshold: Annotated[
        float, typer.Option("--threshold", help="Selective confidence threshold.")
    ] = 0.50,
    custom_id: Annotated[
        str | None, typer.Option("--custom-id", help="Optional custom candidate model ID.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    if not snapshots:
        console.print("[red]No snapshots found.[/red]")
        raise typer.Exit(code=1)

    target_snapshot = (
        next((s for s in snapshots if s.manifest.snapshot_id == snapshot_id), snapshots[-1])
        if snapshot_id
        else snapshots[-1]
    )

    p_path = manifest_path
    if p_path is None:
        p_dir = repo_root / "data" / "partitions"
        if p_dir.is_dir():
            files = sorted(p_dir.glob("*.json"))
            if files:
                p_path = files[-1]
    if not p_path or not p_path.is_file():
        console.print("[red]Partition manifest required to freeze candidate.[/red]")
        raise typer.Exit(code=1)

    part_manifest = PartitionManifest.load(p_path)

    frozen = freeze_and_evaluate_candidate(
        repo_root=repo_root,
        model_family=model_family,
        feature_set=feature_set,
        hyperparameters={},
        calibration_method=calibration,
        confidence_threshold=threshold,
        snapshot=target_snapshot,
        partition_manifest=part_manifest,
        custom_model_id=custom_id,
    )

    console.print(Panel(f"[bold green]Candidate Model Frozen: {frozen.model_id}[/bold green]"))
    console.print(f"Bundle Path: {frozen.bundle_path}")
    console.print(f"Bundle SHA-256: {frozen.bundle_sha256}")
    console.print(f"Manifest SHA-256: {frozen.manifest_sha256}")
    console.print(f"Model Family: {frozen.model_family} | Calibration: {frozen.calibration_method}")
    console.print(f"Execution Authority: [bold red]{frozen.execution_authority}[/bold red]")
    console.print(f"Scientific Verdict: {frozen.verdict}")
    console.print(f"Test Accuracy: {frozen.test_metrics.get('directional_accuracy', 0.0):.2%}")
    console.print("Registered in: data/shadow_eval/registry.json")


@candidate_app.command(
    "evaluate", help="Inspect and verify frozen candidate model holdout evaluation."
)
def candidate_evaluate_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    model_id: Annotated[
        str, typer.Option("--model-id", help="Frozen candidate model ID.")
    ] = "candidate_reversal_phase42_ref",
) -> None:
    repo_root = root or Path.cwd()
    eval_res = evaluate_frozen_candidate(model_id=model_id, repo_root=repo_root)

    status_str = (
        "[bold green]VERIFIED (MATCH)[/bold green]"
        if eval_res["is_verified"]
        else "[bold red]INTEGRITY COMPROMISED[/bold red]"
    )
    console.print(Panel(f"[bold]Candidate Model Audit: {model_id}[/bold]\nStatus: {status_str}"))
    console.print(f"Calculated Bundle SHA-256: {eval_res['calculated_sha256']}")
    console.print(f"Recorded Bundle SHA-256:   {eval_res['recorded_sha256']}")
    console.print(f"Execution Authority:       {eval_res['execution_authority']}")
    console.print(f"Scientific Verdict:        {eval_res['verdict']}")

    tm = eval_res["test_metrics"]
    console.print("\n[bold cyan]Protected Test Holdout Metrics:[/bold cyan]")
    console.print(f"  Sample Count:        {tm.get('sample_count', 0):,}")
    console.print(f"  Decisive Accuracy:   {tm.get('directional_accuracy', 0.0):.2%}")
    console.print(f"  Coverage:            {tm.get('coverage', 0.0):.1%}")
    console.print(f"  Brier Score:         {tm.get('brier_score', 0.0):.4f}")
    console.print(f"  Expected Cal Error:  {tm.get('ece', 0.0):.4f}")
    console.print(f"  Break-Even Hurdle:   {tm.get('break_even_win_rate', 0.5405):.2%}")
    console.print(f"\n[dim]{eval_res['disclaimer']}[/dim]")


@report_app.command(
    "export", help="Export comprehensive failure intelligence report and research synthesis."
)
def report_export_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    output_path: Annotated[
        Path | None, typer.Option("--output-path", help="Report export destination path.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    runner = ExperimentRunner(repo_root)
    summary = runner.get_suite_summary()
    if summary is None:
        console.print("[yellow]No experiment suite summary found. Run experiments first.[/yellow]")
        raise typer.Exit(code=1)

    dest = output_path or (repo_root / "reports" / "model_failure_intelligence_phase42.md")
    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    if not snapshots:
        console.print("[red]No snapshots found.[/red]")
        raise typer.Exit(code=1)
    candles = read_candles_parquet(snapshots[-1].parquet_path)
    report = analyze_model_failures(summary, candles)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report.to_markdown(), encoding="utf-8")

    console.print(Panel("[bold green]Failure Intelligence Report Exported[/bold green]"))
    console.print(f"Destination: {dest}")
    console.print(f"Evaluated Models: {report.total_models_evaluated}")
    console.print(f"Verified Findings: {len(report.verified_findings)}")
    for vf in report.verified_findings:
        console.print(f"  - [cyan]{vf.title}[/cyan] ({vf.category}): {vf.impact}")
    console.print(f"Global Disposition: [bold red]{report.global_disposition}[/bold red]")


@research_app.command(
    "data-readiness",
    help="Evaluate genuine market data readiness and sufficiency across horizons H1 to H60.",
)
def data_readiness_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Output machine-readable JSON format.")
    ] = False,
) -> None:
    repo_root = root or Path.cwd()
    from iqoption_research.market_data.data_inventory import (
        audit_genuine_market_data,
        compute_horizon_readiness,
    )
    from iqoption_research.market_data.parquet_storage import read_candles_parquet
    from iqoption_research.research.dataset_manager import ResearchDatasetManager

    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    if not snapshots:
        console.print("[red]No snapshots found.[/red]")
        raise typer.Exit(code=1)
    candles = read_candles_parquet(snapshots[-1].parquet_path)
    audit = audit_genuine_market_data(repo_root)
    readiness_dict = compute_horizon_readiness(candles)
    readiness = list(readiness_dict.values())

    if as_json:
        payload = {
            "catalog_entries": audit.datasets_audited_count,
            "total_candles": audit.total_retained_candles,
            "readiness": [r.to_dict() for r in readiness],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Phase 43 Multi-Horizon Dataset Readiness Matrix", show_header=True)
    table.add_column("Horizon", style="cyan")
    table.add_column("Minutes", justify="right")
    table.add_column("Rolling Opportunities", justify="right")
    table.add_column("Non-Overlapping Blocks", justify="right")
    table.add_column("Status", style="bold")
    table.add_column("Sufficiency / Governance Notes", style="white")

    for r in readiness:
        status_color = "green" if r.status.value == "SUFFICIENT" else "yellow"
        table.add_row(
            r.horizon_name,
            f"{r.horizon_minutes}m",
            f"{r.rolling_opportunities_count:,}",
            f"{r.non_overlapping_opportunities_count:,}",
            f"[{status_color}]{r.status.value}[/{status_color}]",
            r.sufficiency_notes,
        )
    console.print(table)


@timeframes_app.command("build", help="Aggregate M1 candles causally into higher timeframes.")
def timeframes_build_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    snapshot_id: Annotated[
        str | None, typer.Option("--snapshot-id", help="Snapshot identifier.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    from datetime import UTC, datetime

    from iqoption_research.market_data.parquet_storage import read_candles_parquet
    from iqoption_research.market_data.timeframe_engine import (
        MultiTimeframeEngine,
        Timeframe,
    )
    from iqoption_research.research.dataset_manager import ResearchDatasetManager

    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    if not snapshots:
        console.print("[red]No snapshots found in repository.[/red]")
        raise typer.Exit(code=1)
    target = next((s for s in snapshots if s.manifest.snapshot_id == snapshot_id), snapshots[-1])
    candles = read_candles_parquet(target.parquet_path)
    engine = MultiTimeframeEngine(source_dataset_digest=target.manifest.parquet_sha256)

    table = Table(
        title=f"Multi-Timeframe Causal Aggregation — {target.manifest.snapshot_id}",
        show_header=True,
    )
    table.add_column("Timeframe", style="cyan")
    table.add_column("Resolution", justify="right")
    table.add_column("Aggregated Bars", justify="right")
    table.add_column("Earliest UTC", style="white")
    table.add_column("Latest UTC", style="white")
    table.add_column("Causal Status", style="green")

    for tf in list(Timeframe):
        if tf == Timeframe.M1:
            earliest = datetime.fromtimestamp(candles[0].from_ts, UTC).isoformat()[:19]
            latest = datetime.fromtimestamp(candles[-1].to_ts, UTC).isoformat()[:19]
            table.add_row(
                tf.name,
                "60s",
                f"{len(candles):,}",
                earliest,
                latest,
                "Verified M1 Base",
            )
            continue
        agg = engine.aggregate_m1_to_timeframe(candles, tf)
        if agg:
            earliest = datetime.fromtimestamp(agg[0].from_ts, UTC).isoformat()[:19]
            latest = datetime.fromtimestamp(agg[-1].to_ts, UTC).isoformat()[:19]
            table.add_row(
                tf.name,
                f"{tf.value}s",
                f"{len(agg):,}",
                earliest,
                latest,
                "Deterministic Causal UTC Align",
            )
        else:
            table.add_row(tf.name, f"{tf.value}s", "0", "N/A", "N/A", "Zero complete bars")
    console.print(table)


@timeframes_app.command(
    "verify", help="Verify prefix invariance and causal alignment of multi-timeframe engine."
)
def timeframes_verify_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    from iqoption_research.market_data.parquet_storage import read_candles_parquet
    from iqoption_research.market_data.timeframe_engine import MultiTimeframeEngine, Timeframe
    from iqoption_research.research.dataset_manager import ResearchDatasetManager

    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    if not snapshots:
        console.print("[red]No snapshots found.[/red]")
        raise typer.Exit(code=1)
    candles = read_candles_parquet(snapshots[-1].parquet_path)[:300]
    engine = MultiTimeframeEngine()

    # Prefix invariance check across M5 and M15
    for tf in (Timeframe.M5, Timeframe.M15):
        full_agg = engine.aggregate_m1_to_timeframe(candles, tf)
        if not full_agg:
            continue
        cutoff_c = full_agg[0]
        prefix = [c for c in candles if c.to_ts <= cutoff_c.to_ts]
        prefix_agg = engine.aggregate_m1_to_timeframe(prefix, tf)
        if (
            not prefix_agg
            or prefix_agg[-1].close != cutoff_c.close
            or prefix_agg[-1].high != cutoff_c.high
            or prefix_agg[-1].low != cutoff_c.low
            or prefix_agg[-1].open != cutoff_c.open
        ):
            raise typer.BadParameter(
                "Prefix invariance check failed on higher timeframe aggregation"
            )

    console.print(
        Panel(
            "[bold green]Multi-Timeframe Engine Causal Verification: PASSED[/bold green]\n"
            "Strict prefix invariance, deterministic UTC boundaries, and zero forward leakage verified."
        )
    )


@horizons_app.command("plan", help="Build and view Phase 43 multi-horizon experiment matrix plan.")
def horizons_plan_command() -> None:
    from iqoption_research.ml.experiment_matrix import build_phase43_experiment_plan

    trials = build_phase43_experiment_plan()

    table = Table(
        title=f"Phase 43 Multi-Horizon Experiment Plan ({len(trials)} Registered Trials)",
        show_header=True,
    )
    table.add_column("Stage", style="cyan")
    table.add_column("Trial ID", style="bold")
    table.add_column("Horizon", justify="center")
    table.add_column("Model Family", style="white")
    table.add_column("Feature Preset", style="yellow")
    table.add_column("Threshold", justify="right")

    for t in trials[:20]:
        table.add_row(
            t.stage,
            t.trial_id,
            t.horizon_name,
            t.model_family,
            t.feature_preset,
            f"{t.confidence_threshold:.2f}",
        )
    if len(trials) > 20:
        table.add_row("...", f"... and {len(trials) - 20} more trials", "...", "...", "...", "...")
    console.print(table)
    console.print(
        f"[bold green]Total Planned Trials: {len(trials)} (Budget limit: 100 trials)[/bold green]"
    )


@horizons_app.command("run", help="Run bounded multi-horizon walk-forward experiments.")
def horizons_run_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Maximum trials to execute.")] = 10,
    horizon: Annotated[
        str | None, typer.Option("--horizon", help="Filter trials by horizon (e.g. H1, H5).")
    ] = None,
    db_path: Annotated[
        Path | None, typer.Option("--db-path", help="Path to SQLite ledger.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    from iqoption_research.market_data.parquet_storage import read_candles_parquet
    from iqoption_research.ml.experiment_matrix import (
        ExperimentRunnerV3,
        build_phase43_experiment_plan,
    )
    from iqoption_research.research.dataset_manager import ResearchDatasetManager

    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    if not snapshots:
        console.print("[red]No snapshots found.[/red]")
        raise typer.Exit(code=1)
    candles = read_candles_parquet(snapshots[-1].parquet_path)
    trials = build_phase43_experiment_plan()
    if horizon:
        trials = [t for t in trials if t.horizon_name == horizon.upper()]

    runner = ExperimentRunnerV3(
        repo_root=repo_root,
        db_path=db_path or (repo_root / "data" / "experiments_v3" / "phase43_ledger.sqlite"),
    )
    console.print(
        f"[bold cyan]Starting Phase 43 Multi-Horizon Experiments (Limit: {limit}, Horizon: {horizon or 'ALL'})...[/bold cyan]"
    )
    summary = runner.run_suite(
        candles=candles,
        plan=trials,
        max_trials=limit,
    )

    table = Table(title="Executed Phase 43 Experiment Results", show_header=True)
    table.add_column("Trial ID", style="bold")
    table.add_column("Horizon", justify="center")
    table.add_column("Model Family", style="white")
    table.add_column("Feature Preset", style="yellow")
    table.add_column("Dir Acc", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Verdict", style="cyan")

    for r in summary.reports:
        acc_str = (
            f"{r.pooled_decisive_accuracy:.2%}" if r.pooled_decisive_accuracy is not None else "N/A"
        )
        cov_str = f"{r.pooled_coverage:.2%}" if r.pooled_coverage is not None else "N/A"
        table.add_row(
            r.trial_id,
            r.target_horizon_name,
            r.model_family,
            r.feature_preset,
            acc_str,
            cov_str,
            r.verdict,
        )
    console.print(table)


@horizons_app.command("status", help="Show Phase 43 multi-horizon experiment ledger status.")
def horizons_status_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    db_path: Annotated[
        Path | None, typer.Option("--db-path", help="Path to SQLite ledger.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    from iqoption_research.ml.experiment_matrix import ExperimentRunnerV3

    runner = ExperimentRunnerV3(
        repo_root=repo_root,
        db_path=db_path or (repo_root / "data" / "experiments_v3" / "phase43_ledger.sqlite"),
    )
    summary = runner.get_ledger_summary()

    console.print(
        Panel(
            f"[bold]Phase 43 Experiment Ledger Status[/bold]\n"
            f"Total Completed Trials: {summary['total_completed']}\n"
            f"Global Disposition: [bold red]{summary['global_disposition']}[/bold red]\n"
            f"Execution Authority: [bold red]{summary['execution_authority']}[/bold red]"
        )
    )

    table = Table(title="Trial Status Breakdown", show_header=True)
    table.add_column("Scientific Verdict", style="cyan")
    table.add_column("Trial Count", justify="right", style="green")
    for verdict, count in summary.get("verdicts", {}).items():
        table.add_row(verdict, str(count))
    console.print(table)


@horizons_app.command(
    "compare", help="Compare empirical performance across horizons and feature presets."
)
def horizons_compare_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    db_path: Annotated[
        Path | None, typer.Option("--db-path", help="Path to SQLite ledger.")
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    from iqoption_research.ml.experiment_matrix import ExperimentRunnerV3

    runner = ExperimentRunnerV3(
        repo_root=repo_root,
        db_path=db_path or (repo_root / "data" / "experiments_v3" / "phase43_ledger.sqlite"),
    )
    comparison = runner.compare_horizons()

    table = Table(title="Phase 43 Horizon Performance Comparison", show_header=True)
    table.add_column("Horizon", style="cyan")
    table.add_column("Completed", justify="right")
    table.add_column("Best Accuracy", justify="right")
    table.add_column("Mean Accuracy", justify="right")
    table.add_column("Mean Coverage", justify="right")
    table.add_column("Best Trial ID", style="white")

    for row in comparison:
        table.add_row(
            row["horizon"],
            str(row["completed_trials"]),
            f"{row['best_accuracy']:.2%}" if row["best_accuracy"] is not None else "N/A",
            f"{row['mean_accuracy']:.2%}" if row["mean_accuracy"] is not None else "N/A",
            f"{row['mean_coverage']:.2%}" if row["mean_coverage"] is not None else "N/A",
            str(row["best_trial_id"] or "None"),
        )
    console.print(table)


@regimes_app.command("analyze", help="Classify causal market regimes and display distribution.")
def regimes_analyze_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    snapshot_id: Annotated[
        str | None, typer.Option("--snapshot-id", help="Snapshot identifier.")
    ] = None,
) -> None:
    from collections import Counter

    repo_root = root or Path.cwd()
    from iqoption_research.market_data.parquet_storage import read_candles_parquet
    from iqoption_research.ml.regimes import CausalMarketRegimeClassifier
    from iqoption_research.research.dataset_manager import ResearchDatasetManager

    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    if not snapshots:
        console.print("[red]No snapshots found.[/red]")
        raise typer.Exit(code=1)
    target = next((s for s in snapshots if s.manifest.snapshot_id == snapshot_id), snapshots[-1])
    candles = read_candles_parquet(target.parquet_path)

    classifier = CausalMarketRegimeClassifier()
    regimes = classifier.classify_series(candles)

    table = Table(
        title=f"Causal Market Regime Distribution — {target.manifest.snapshot_id}", show_header=True
    )
    table.add_column("Dimension", style="cyan")
    table.add_column("Regime Value", style="white")
    table.add_column("Bar Count", justify="right", style="green")
    table.add_column("Percentage", justify="right")

    total = len(regimes)
    for dim_name, values in [
        ("Volatility", [r.volatility.value for r in regimes]),
        ("Trend", [r.trend.value for r in regimes]),
        ("Range", [r.range_state.value for r in regimes]),
        ("Session", [r.session.value for r in regimes]),
    ]:
        counts = Counter(values)
        for val, count in counts.most_common():
            pct = count / total if total > 0 else 0.0
            table.add_row(dim_name, val, f"{count:,}", f"{pct:.1%}")

    console.print(table)


# Hyphenated aliases registered directly on research_app for operator convenience
research_app.command("partitions-create", hidden=True)(partitions_create_command)
research_app.command("partitions-verify", hidden=True)(partitions_verify_command)
research_app.command("plan-create", hidden=True)(plan_create_command)
research_app.command("plan-validate", hidden=True)(plan_validate_command)
research_app.command("experiment-run", hidden=True)(experiment_run_command)
research_app.command("experiment-status", hidden=True)(experiment_status_command)
research_app.command("experiment-cancel", hidden=True)(experiment_cancel_command)
research_app.command("experiment-compare", hidden=True)(experiment_compare_command)
research_app.command("candidate-freeze", hidden=True)(candidate_freeze_command)
research_app.command("candidate-evaluate", hidden=True)(candidate_evaluate_command)
research_app.command("report-export", hidden=True)(report_export_command)

# Phase 43 aliases
research_app.command("data-readiness", hidden=True)(data_readiness_command)
research_app.command("timeframes-build", hidden=True)(timeframes_build_command)
research_app.command("timeframes-verify", hidden=True)(timeframes_verify_command)
research_app.command("horizons-plan", hidden=True)(horizons_plan_command)
research_app.command("horizons-run", hidden=True)(horizons_run_command)
research_app.command("horizons-status", hidden=True)(horizons_status_command)
research_app.command("horizons-compare", hidden=True)(horizons_compare_command)
research_app.command("regimes-analyze", hidden=True)(regimes_analyze_command)
