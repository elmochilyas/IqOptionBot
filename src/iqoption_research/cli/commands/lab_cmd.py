"""CLI commands for Phase 39 Offline Model Laboratory.

All commands operate strictly offline: zero network, zero broker credentials,
immutable source preservation, leak-free evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.ml.laboratory import (
    ExperimentConfig,
    ModelLaboratory,
)
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
)

console = Console()
lab_app = typer.Typer(
    name="laboratory",
    help="Offline Model Laboratory: dataset preparation, training, and evaluation.",
    no_args_is_help=True,
)


@lab_app.command(
    "discover-datasets",
    help="Discover eligible authentic EURUSD M1 datasets and V2 collection sessions.",
)
def discover_datasets_command(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Repository root path (defaults to current dir)."),
    ] = None,
    show_excluded: Annotated[
        bool,
        typer.Option("--show-excluded", help="Display details of excluded sources."),
    ] = False,
) -> None:
    repo_root = root or Path.cwd()
    manager = ResearchDatasetManager(repo_root)
    sources = manager.discover_sources()

    eligible = [s for s in sources if s.is_eligible]
    excluded = [s for s in sources if not s.is_eligible]

    table = Table(title="Offline Market Data Discovery", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Notes", style="white")

    catalog_eligible = len([s for s in eligible if s.source_type == "CATALOG_PARQUET"])
    live_eligible = len([s for s in eligible if s.source_type == "EXTENDED_SESSION"])
    empty_excluded = len([
        s
        for s in excluded
        if s.exclusion_reason
        and ("zero candles" in s.exclusion_reason.lower() or "empty" in s.exclusion_reason.lower())
    ])
    other_excluded = len(excluded) - empty_excluded

    table.add_row("Eligible Catalog Parquet", str(catalog_eligible), "Authentic EURUSD M1")
    table.add_row("Eligible Live V2 Sessions", str(live_eligible), "Verified with receipts")
    table.add_row("Excluded: Empty Sessions", str(empty_excluded), "Zero candles (fail-closed)")
    table.add_row("Excluded: Other", str(other_excluded), "Non-EURUSD / synthetic / unverified")
    table.add_row("TOTAL DISCOVERED", str(len(sources)), f"{len(eligible)} eligible for snapshots")

    console.print(table)

    if show_excluded and excluded:
        exc_table = Table(title="Excluded Sources Audit", show_header=True)
        exc_table.add_column("Source ID", style="yellow")
        exc_table.add_column("Type", style="cyan")
        exc_table.add_column("Reason", style="red")
        for s in excluded:
            exc_table.add_row(s.source_id, s.source_type, s.exclusion_reason)
        console.print(exc_table)


@lab_app.command(
    "prepare-dataset",
    help="Build an immutable, versioned, checksum-verified training dataset snapshot.",
)
def prepare_dataset_command(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Repository root path (defaults to current dir)."),
    ] = None,
    max_gap_minutes: Annotated[
        int,
        typer.Option("--max-gap-minutes", help="Tolerance for normal intraday gaps."),
    ] = 10,
) -> None:
    repo_root = root or Path.cwd()
    manager = ResearchDatasetManager(repo_root)

    console.print("[bold blue]Starting offline dataset preparation...[/bold blue]")
    try:
        snapshot = manager.build_snapshot(max_gap_minutes=max_gap_minutes)
    except Exception as exc:
        console.print(f"[bold red]Failed to build dataset snapshot: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    m = snapshot.manifest
    panel_text = (
        f"[bold green]Snapshot ID:[/bold green] {m.snapshot_id}\n"
        f"[bold]Created UTC:[/bold] {m.created_at_utc}\n"
        f"[bold]Total Retained Candles:[/bold] {m.total_candles:,}\n"
        f"[bold]Date Range (UTC):[/bold] {m.date_range_start_utc} -> {m.date_range_end_utc}\n"
        f"[bold]Deduplicated Boundaries:[/bold] {m.deduplicated_boundaries_count}\n"
        f"[bold]Quarantined Conflicts:[/bold] {m.quarantined_conflicts_count}\n"
        f"[bold]Weekend Closures:[/bold] {len(m.weekend_closures)}\n"
        f"[bold]Intraday Gaps:[/bold] {len(m.intraday_gaps)}\n"
        f"[bold]Parquet SHA-256:[/bold] {m.parquet_sha256}\n"
        f"[bold]Storage Directory:[/bold] {snapshot.path}"
    )
    console.print(
        Panel(
            panel_text,
            title="[bold green]Research Dataset Snapshot Created[/bold green]",
            expand=False,
        )
    )


@lab_app.command(
    "list-snapshots",
    help="List all immutable research dataset snapshots.",
)
def list_snapshots_command(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Repository root path (defaults to current dir)."),
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()

    if not snapshots:
        console.print(
            "[yellow]No research dataset snapshots found in data/research_datasets/.[/yellow]"
        )
        return

    table = Table(title="Research Dataset Snapshots", show_header=True)
    table.add_column("Snapshot ID", style="cyan")
    table.add_column("Created UTC", style="white")
    table.add_column("Candles", style="green", justify="right")
    table.add_column("Date Range", style="white")
    table.add_column("Sources", style="yellow", justify="right")

    for s in snapshots:
        m = s.manifest
        date_str = f"{m.date_range_start_utc[:10]} .. {m.date_range_end_utc[:10]}"
        table.add_row(
            m.snapshot_id,
            m.created_at_utc[:19],
            f"{m.total_candles:,}",
            date_str,
            str(len(m.source_files)),
        )

    console.print(table)


@lab_app.command(
    "train",
    help="Train an independent candidate AI model on a prepared dataset snapshot.",
)
def train_command(
    snapshot_id: Annotated[
        str | None,
        typer.Option("--snapshot-id", help="Target snapshot ID (defaults to newest)."),
    ] = None,
    model_family: Annotated[
        str,
        typer.Option(
            "--family",
            "-f",
            help="Model family: random_forest, logistic_regression, hist_gradient_boosting, etc.",
        ),
    ] = "random_forest",
    horizon_bars: Annotated[
        int,
        typer.Option("--horizon-bars", "-h", help="Forward prediction horizon in bars (minutes)."),
    ] = 1,
    payout_rate: Annotated[
        float,
        typer.Option("--payout-rate", help="Simulated binary option payout rate (e.g. 0.85)."),
    ] = 0.85,
    min_confidence: Annotated[
        float,
        typer.Option(
            "--min-confidence", help="Minimum calibrated probability threshold for trading."
        ),
    ] = 0.55,
    calibrate: Annotated[
        bool,
        typer.Option("--calibrate/--no-calibrate", help="Calibrate model probabilities."),
    ] = True,
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Repository root path (defaults to current dir)."),
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    lab = ModelLaboratory(repo_root)

    target_snapshot = snapshot_id
    if not target_snapshot:
        manager = ResearchDatasetManager(repo_root)
        snapshots = manager.list_snapshots()
        if not snapshots:
            console.print("[bold red]No snapshots found. Run prepare-dataset first.[/bold red]")
            raise typer.Exit(code=1)
        target_snapshot = snapshots[0].manifest.snapshot_id

    config = ExperimentConfig(
        model_family=model_family,
        snapshot_id=target_snapshot,
        horizon_bars=horizon_bars,
        payout_rate=payout_rate,
        min_confidence=min_confidence,
        calibrate=calibrate,
    )

    console.print(
        f"[bold blue]Training model '{model_family}' on '{target_snapshot}'...[/bold blue]"
    )

    try:
        result = lab.run_experiment(config)
    except Exception as exc:
        console.print(f"[bold red]Training failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    m_test = result.metrics.get("test", {})
    m_val = result.metrics.get("val", {})

    report_table = Table(
        title=f"Candidate Model Evaluation: {result.model_id} ({model_family})", show_header=True
    )
    report_table.add_column("Metric", style="cyan")
    report_table.add_column("Validation", style="white")
    report_table.add_column("Test (Holdout)", style="green")

    report_table.add_row(
        "Directional Accuracy",
        f"{m_val.get('directional_accuracy', 0.0):.2%}",
        f"{m_test.get('directional_accuracy', 0.0):.2%}",
    )
    report_table.add_row(
        "Balanced Accuracy",
        f"{m_val.get('balanced_accuracy', 0.0):.2%}",
        f"{m_test.get('balanced_accuracy', 0.0):.2%}",
    )
    report_table.add_row(
        "Brier Score",
        f"{m_val.get('brier_score', 0.0):.4f}",
        f"{m_test.get('brier_score', 0.0):.4f}",
    )
    report_table.add_row(
        "Coverage / Abstention",
        f"{m_val.get('coverage', 0.0):.1%} / {m_val.get('abstention_rate', 0.0):.1%}",
        f"{m_test.get('coverage', 0.0):.1%} / {m_test.get('abstention_rate', 0.0):.1%}",
    )
    report_table.add_row(
        "Selective Win Rate",
        f"{m_val.get('win_rate', 0.0):.2%}",
        f"{m_test.get('win_rate', 0.0):.2%}",
    )
    report_table.add_row(
        "Wilson 95% CI (Test)",
        "-",
        f"[{m_test.get('ci_lower_95', 0.0):.2%}, {m_test.get('ci_upper_95', 0.0):.2%}]",
    )
    report_table.add_row(
        "Simulated Expectancy",
        "-",
        f"{m_test.get('expectancy_pct', 0.0):+.2f}% per trade",
    )

    console.print(report_table)

    verdict_color = "green" if result.has_statistical_edge else "yellow"
    console.print(
        Panel(
            f"[{verdict_color}]Scientific Verdict: {result.verdict}[/{verdict_color}]\n"
            f"Statistical Edge Detected: {result.has_statistical_edge}\n"
            f"Model Bundle Path: {result.bundle_path}\n\n"
            f"[dim]{result.disclaimer}[/dim]",
            title="[bold]Evaluation Assessment[/bold]",
            expand=False,
        )
    )


@lab_app.command(
    "list-models",
    help="List all candidate models saved in data/candidate_models/.",
)
def list_models_command(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Repository root path (defaults to current dir)."),
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    lab = ModelLaboratory(repo_root)
    models = lab.list_candidate_models()

    if not models:
        console.print("[yellow]No candidate models found in data/candidate_models/.[/yellow]")
        return

    table = Table(title="Candidate AI Models", show_header=True)
    table.add_column("Model ID", style="cyan")
    table.add_column("Family", style="white")
    table.add_column("Created UTC", style="white")
    table.add_column("Test Acc", style="green", justify="right")
    table.add_column("Brier", style="white", justify="right")
    table.add_column("Verdict", style="yellow")

    for m in models:
        table.add_row(
            m["model_id"],
            m["model_family"],
            m["created_at_utc"][:19],
            f"{m['test_accuracy']:.2%}",
            f"{m.get('test_brier', m.get('brier_score', 0.0)):.4f}",
            m["verdict"],
        )

    console.print(table)


@lab_app.command(
    "evaluate-model",
    help="Display detailed evaluation manifest and metrics for a candidate model.",
)
def evaluate_model_command(
    model_id: Annotated[str, typer.Argument(help="Candidate model ID.")],
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Repository root path (defaults to current dir)."),
    ] = None,
) -> None:
    repo_root = root or Path.cwd()
    manifest_path = repo_root / "data" / "candidate_models" / model_id / "manifest.json"
    if not manifest_path.is_file():
        console.print(f"[bold red]Model manifest not found: {manifest_path}[/bold red]")
        raise typer.Exit(code=1)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    console.print(
        Panel(
            json.dumps(data, indent=2),
            title=f"Candidate Model Manifest: {model_id}",
            expand=False,
        )
    )
