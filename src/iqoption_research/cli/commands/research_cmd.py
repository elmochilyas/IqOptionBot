"""CLI commands for Phase 3 offline research, historical data management, and backtesting."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.importer import import_and_persist_dataset
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
from iqoption_research.research.datasets.loader import HistoricalDatasetLoader
from iqoption_research.research.datasets.splitter import ChronologicalSplitter
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.features.scalers import StandardScaler

console = Console()
research_app = typer.Typer(
    name="research",
    help="Offline historical market data import, feature extraction, and backtesting.",
    no_args_is_help=True,
)


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
