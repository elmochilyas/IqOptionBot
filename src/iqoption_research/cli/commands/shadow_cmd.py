"""CLI commands for Phase 40A Offline Shadow Evaluation Foundation.

All commands operate strictly offline: zero network, zero broker credentials,
immutable prediction logs, delayed outcome resolution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
)
from iqoption_research.shadow.registry import ShadowModelRegistry
from iqoption_research.shadow.replay import HistoricalReplayEngine, ReplayConfig

console = Console()
shadow_app = typer.Typer(
    name="shadow",
    help="Offline Shadow Evaluation: candidate registration, replay, and evaluation.",
    no_args_is_help=True,
)


@shadow_app.command("register-candidate", help="Register a verified candidate model as frozen.")
def register_candidate_command(
    model_id: Annotated[str, typer.Argument(help="Candidate model ID or path.")],
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
    threshold: Annotated[
        float | None, typer.Option("--threshold", help="Override confidence threshold.")
    ] = None,
    notes: Annotated[str, typer.Option("--notes", help="Registration notes.")] = "",
) -> None:
    repo_root = root or Path.cwd()
    registry = ShadowModelRegistry(repo_root)

    try:
        rec = registry.register_candidate(model_id, confidence_threshold=threshold, notes=notes)
        console.print(
            Panel(
                f"[bold green]Successfully registered frozen candidate:[/bold green]\n\n"
                f"Registration ID: [cyan]{rec.registration_id}[/cyan]\n"
                f"Model ID: [bold]{rec.model_id}[/bold]\n"
                f"Family: {rec.model_family} (baseline={rec.is_baseline})\n"
                f"Confidence Threshold: {rec.confidence_threshold:.2f}\n"
                f"Data Cutoff (ts): {rec.data_cutoff_ts}\n"
                f"Bundle Hash: {rec.bundle_sha256[:16]}...\n"
                f"Lab Verdict: [yellow]{rec.verdict}[/yellow]\n\n"
                f"[dim]{rec.disclaimer}[/dim]",
                title="Frozen Candidate Registered",
                border_style="green",
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Registration failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@shadow_app.command("list-candidates", help="List all registered frozen candidate models.")
def list_candidates_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    registry = ShadowModelRegistry(repo_root)
    candidates = registry.list_candidates()

    if not candidates:
        console.print("[yellow]No frozen candidate models currently registered.[/yellow]")
        return

    table = Table(title="Registered Frozen Candidate Models", show_header=True)
    table.add_column("Model ID", style="cyan")
    table.add_column("Family", style="green")
    table.add_column("Threshold", style="white", justify="right")
    table.add_column("Data Cutoff (ts)", style="white", justify="right")
    table.add_column("Bundle SHA-256", style="white")
    table.add_column("Verdict", style="yellow")

    for c in candidates:
        table.add_row(
            c.model_id,
            c.model_family,
            f"{c.confidence_threshold:.2f}",
            str(c.data_cutoff_ts),
            c.bundle_sha256[:12] + "...",
            c.verdict,
        )

    console.print(table)


@shadow_app.command("replay", help="Run offline historical replay with a frozen candidate.")
def replay_command(
    model_id: Annotated[str, typer.Option("--model-id", "-m", help="Frozen model ID.")],
    snapshot_id: Annotated[
        str | None, typer.Option("--snapshot-id", "-s", help="Target dataset snapshot ID.")
    ] = None,
    max_candles: Annotated[
        int | None, typer.Option("--max-candles", help="Max candles to replay.")
    ] = None,
    session_id: Annotated[
        str | None, typer.Option("--session-id", help="Explicit session ID.")
    ] = None,
    assumed_payout: Annotated[
        float, typer.Option("--payout", help="Simulated binary option payout rate.")
    ] = 0.85,
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    registry = ShadowModelRegistry(repo_root)
    mgr = ResearchDatasetManager(repo_root)
    engine = HistoricalReplayEngine(registry, root=repo_root)

    # 1. Resolve snapshot
    snapshots = mgr.list_snapshots()
    if not snapshots:
        console.print("[bold red]No research dataset snapshots found.[/bold red]")
        raise typer.Exit(code=1)

    target_snap = None
    if snapshot_id is not None:
        target_snap = next((s for s in snapshots if s.manifest.snapshot_id == snapshot_id), None)
        if target_snap is None:
            console.print(f"[bold red]Snapshot '{snapshot_id}' not found.[/bold red]")
            raise typer.Exit(code=1)
    else:
        target_snap = snapshots[0]

    console.print(
        f"Loading [cyan]{target_snap.manifest.snapshot_id}[/cyan] "
        f"({target_snap.manifest.total_candles:,} candles)..."
    )
    candles = target_snap.load_candles()

    cfg = ReplayConfig(
        model_id=model_id,
        session_id=session_id or "",
        snapshot_id=target_snap.manifest.snapshot_id,
        max_candles=max_candles,
        assumed_payout=assumed_payout,
    )

    console.print(
        f"Starting historical replay for model [bold]{model_id}[/bold] "
        f"in session [yellow]{cfg.session_id}[/yellow]..."
    )

    def on_progress(pct: int, msg: str) -> None:
        if pct % 25 == 0 or pct == 100:
            console.print(f"[{pct:3d}%] {msg}")

    try:
        res = engine.run_replay(cfg, candles, progress_cb=on_progress)
    except Exception as exc:
        console.print(f"[bold red]Replay execution error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    s = res.summary
    table = Table(title=f"Replay Evaluation Summary: {res.session_id}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total Predictions", f"{s.total_predictions:,}")
    table.add_row("Resolved Outcomes", f"{s.resolved_count:,}")
    table.add_row("Pending Outcomes", f"{s.pending_count:,}")
    table.add_row("Skipped (Abstained)", f"{s.skipped_abstain_count:,}")
    table.add_row("Invalid (Discontinuity)", f"{s.invalid_discontinuity_count:,}")
    table.add_row("Trades Taken", f"{s.trade_count:,}")
    table.add_row("Wins / Losses / Ties", f"{s.win_count} / {s.loss_count} / {s.tie_count}")
    table.add_row("Selective Win Rate", f"{s.win_rate * 100:.2f}%")
    ci_str = f"[{s.wilson_ci_lower_95 * 100:.2f}%, {s.wilson_ci_upper_95 * 100:.2f}%]"
    table.add_row("Wilson 95% CI", ci_str)
    table.add_row("Break-Even Threshold", f"{s.break_even_win_rate * 100:.2f}%")
    table.add_row("Full Directional Accuracy", f"{s.directional_accuracy * 100:.2f}%")
    table.add_row("Brier Score", f"{s.brier_score:.4f}")
    cov_str = f"{s.coverage * 100:.1f}% / {s.abstention_rate * 100:.1f}%"
    table.add_row("Coverage / Abstention", cov_str)
    table.add_row("Statistical Edge Proven", str(s.has_statistical_edge))

    console.print(table)
    if res.data_cutoff_overlap_warning:
        console.print(f"[bold yellow]Warning:[/bold yellow] {res.data_cutoff_overlap_warning}")
    console.print(f"[dim]{res.disclaimer}[/dim]")


@shadow_app.command("show-session", help="Display details of a completed replay session.")
def show_session_command(
    session_id: Annotated[str, typer.Argument(help="Replay session ID.")],
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    s_dir = repo_root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    meta_file = s_dir / "session.json"
    if not meta_file.is_file():
        console.print(f"[bold red]Session metadata missing: {meta_file}[/bold red]")
        raise typer.Exit(code=1)

    data = json.loads(meta_file.read_text(encoding="utf-8"))
    console.print(
        Panel(
            json.dumps(data, indent=2, sort_keys=True),
            title=f"Shadow Evaluation Session: {session_id}",
            border_style="cyan",
        )
    )
