"""CLI commands for Phase 40A Offline Shadow Evaluation Foundation.

All commands operate strictly offline: zero network, zero broker credentials,
immutable prediction logs, delayed outcome resolution.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    validate_storage_identifier,
)
from iqoption_research.shadow.observer import (
    ObservationSessionConfig,
    ProspectiveShadowObserver,
    verify_session_integrity,
)
from iqoption_research.shadow.registry import ShadowModelRegistry
from iqoption_research.shadow.replay import HistoricalReplayEngine, ReplayConfig
from iqoption_research.shadow.streaming import (
    LiveCollectorStream,
    MarketStreamProtocol,
    SyntheticMarketStream,
)

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
    baseline_model_id: Annotated[
        str | None,
        typer.Option(
            "--baseline-model-id",
            "-b",
            help="Optional registered authentic baseline model ID for benchmark comparison.",
        ),
    ] = None,
    notes: Annotated[str, typer.Option("--notes", help="Registration notes.")] = "",
) -> None:
    repo_root = root or Path.cwd()
    try:
        validate_storage_identifier(model_id)
        if baseline_model_id is not None:
            validate_storage_identifier(baseline_model_id)
    except Exception as exc:
        console.print(f"[bold red]Invalid model ID:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    registry = ShadowModelRegistry(repo_root)

    try:
        rec = registry.register_candidate(
            model_id,
            confidence_threshold=threshold,
            baseline_model_id=baseline_model_id,
            notes=notes,
        )
        console.print(
            Panel(
                f"[bold green]Successfully registered frozen candidate:[/bold green]\n\n"
                f"Registration ID: [cyan]{rec.registration_id}[/cyan]\n"
                f"Model ID: [bold]{rec.model_id}[/bold]\n"
                f"Family: {rec.model_family} (baseline={rec.is_baseline})\n"
                f"Confidence Threshold: {rec.confidence_threshold:.2f}\n"
                f"Data Cutoff (ts): {rec.data_cutoff_ts}\n"
                f"Exposure Boundaries: train={rec.training_cutoff_ts}, "
                f"val={rec.validation_cutoff_ts}, holdout={rec.holdout_cutoff_ts}, "
                f"snap_end={rec.source_snapshot_end_ts}\n"
                f"Baseline Status: [bold]{rec.baseline_status}[/bold] "
                f"(source={rec.baseline_source_model_id or 'none'}, "
                f"maj_dir={rec.training_majority_direction})\n"
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
    try:
        validate_storage_identifier(model_id)
        if session_id:
            validate_storage_identifier(session_id)
    except Exception as exc:
        console.print(f"[bold red]Identifier validation failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

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
    table.add_row("Selective Win Rate (All)", f"{s.win_rate * 100:.2f}%")
    table.add_row("Selective Win Rate (Decisive)", f"{s.win_rate_decisive * 100:.2f}%")
    table.add_row("Tie Rate", f"{s.tie_rate * 100:.2f}%")
    ci_str = f"[{s.wilson_ci_lower_95 * 100:.2f}%, {s.wilson_ci_upper_95 * 100:.2f}%]"
    table.add_row("Wilson 95% CI (All)", ci_str)
    be_str = f"{s.break_even_win_rate * 100:.2f}% / {s.break_even_win_rate_all * 100:.2f}%"
    table.add_row("Break-Even (Decisive / All)", be_str)
    table.add_row("Simulated Expectancy", f"{s.simulated_expectancy_pct:+.2f}%")
    table.add_row("Full Directional Accuracy", f"{s.directional_accuracy * 100:.2f}%")
    table.add_row("Brier Score", f"{s.brier_score:.4f}")
    cov_str = f"{s.coverage * 100:.1f}% / {s.abstention_rate * 100:.1f}%"
    table.add_row("Coverage / Abstention", cov_str)
    table.add_row("Prediction Failures", f"{res.prediction_failures_count}")
    table.add_row("Scientific Verdict", s.scientific_verdict)
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
    try:
        validate_storage_identifier(session_id)
    except Exception as exc:
        console.print(f"[bold red]Invalid session identifier:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    registry = ShadowModelRegistry(repo_root)
    engine = HistoricalReplayEngine(registry, root=repo_root)
    verif = engine.verify_session(session_id)

    status = verif.get("status")
    if status == "VERIFIED":
        console.print(
            "[bold green]Cryptographic Evidence Status: VERIFIED "
            "(Journal and session hashes verified)[/bold green]"
        )
    elif status == "LEGACY_UNVERIFIED_EVIDENCE":
        console.print(
            "[bold yellow]Cryptographic Evidence Status: LEGACY_UNVERIFIED_EVIDENCE "
            "(Older evidence preserved without modification)[/bold yellow]"
        )
    else:
        console.print(
            f"[bold red]Cryptographic Evidence Status: {status} "
            f"({verif.get('error', 'Integrity issue')})[/bold red]"
        )

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


def _generate_synthetic_candles(n_candles: int = 60) -> list[Candle]:
    """Generate deterministic synthetic EURUSD M1 candles for offline observation."""
    base_ts = 1700000000
    base_price = 1.08500
    candles: list[Candle] = []
    for i in range(n_candles):
        ts = base_ts + (i * 60)
        step = 0.00030 if i % 2 == 0 else -0.00030
        if i % 7 == 0:
            step = 0.0
        close_p = round(base_price + step, 5)
        high_p = max(base_price, close_p) + 0.00020
        low_p = min(base_price, close_p) - 0.00020
        c = Candle(
            active_id=1,
            symbol="EURUSD",
            size=CandleSize.MIN_1,
            from_ts=ts,
            to_ts=ts + 60,
            at_ts=ts + 60,
            open=base_price,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=100,
            bid=close_p,
            ask=close_p + 0.00005,
            is_complete=True,
        )
        candles.append(c)
        base_price = close_p
    return candles


@shadow_app.command(
    "observe",
    help="Run supervised prospective shadow observation on live or synthetic stream.",
)
def observe_command(
    model_ids: Annotated[
        list[str],
        typer.Option("--model-id", "-m", help="One or more frozen model IDs to observe."),
    ],
    minutes: Annotated[
        int,
        typer.Option("--minutes", help="Observation duration in minutes (15, 30, 60, 120)."),
    ] = 15,
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Explicit session ID."),
    ] = None,
    account_id: Annotated[
        str | None,
        typer.Option("--account-id", help="Windows Vault account ID (required for live)."),
    ] = None,
    synthetic: Annotated[
        bool,
        typer.Option(
            "--synthetic",
            "--offline",
            help="Run deterministic synthetic offline streaming harness (zero broker connection).",
        ),
    ] = False,
    live: Annotated[
        bool,
        typer.Option("--live", help="Opt-in to live broker connection."),
    ] = False,
    authorize: Annotated[
        bool,
        typer.Option(
            "--yes-i-understand",
            help="Confirm authorization for supervised live observation.",
        ),
    ] = False,
    allow_shortened_session: Annotated[
        bool,
        typer.Option(
            "--allow-shortened-session",
            help="Authorize session cut short by scheduled market close.",
        ),
    ] = False,
    payout: Annotated[
        float,
        typer.Option("--payout", help="Simulated binary option payout rate."),
    ] = 0.85,
    cancel_file: Annotated[
        Path | None,
        typer.Option("--cancel-file", hidden=True),
    ] = None,
    status_file: Annotated[
        Path | None,
        typer.Option("--status-file", hidden=True),
    ] = None,
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Repository root path."),
    ] = None,
) -> None:
    repo_root = root or Path.cwd()

    if not model_ids:
        console.print("[bold red]At least one --model-id is required.[/bold red]")
        raise typer.Exit(code=1)

    for mid in model_ids:
        try:
            validate_storage_identifier(mid)
        except Exception as exc:
            console.print(f"[bold red]Invalid model ID '{mid}':[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    eff_session_id = session_id or f"pobs_{uuid.uuid4().hex[:12]}"
    try:
        validate_storage_identifier(eff_session_id)
    except Exception as exc:
        console.print(f"[bold red]Invalid session identifier '{eff_session_id}':[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    registry = ShadowModelRegistry(repo_root)

    # Pre-flight candidate registration check
    for mid in model_ids:
        rec = registry.get_candidate(mid)
        if rec is None:
            console.print(
                f"[bold red]Model '{mid}' is not registered in frozen registry.[/bold red]"
            )
            raise typer.Exit(code=1)
        if rec.execution_authority != "NONE":
            console.print(f"[bold red]Model '{mid}' execution authority must be NONE.[/bold red]")
            raise typer.Exit(code=1)

    stream: MarketStreamProtocol
    if synthetic:
        console.print(
            "[bold yellow]Mode: SYNTHETIC STREAMING HARNESS "
            "(Deterministic Offline Test)[/bold yellow]"
        )
        candles = _generate_synthetic_candles(n_candles=30 + minutes)
        stream = SyntheticMarketStream(candles, latency_simulation_ms=250.0)
    else:
        if not live or not authorize:
            console.print(
                "[bold red]Live observation requires both --live and "
                "--yes-i-understand flags.[/bold red]"
            )
            raise typer.Exit(code=1)
        if not account_id:
            console.print(
                "[bold red]Live observation requires a valid --account-id "
                "from Windows Vault.[/bold red]"
            )
            raise typer.Exit(code=1)
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            console.print(
                "[bold red]An operator-owned interactive terminal is required "
                "for live observation.[/bold red]"
            )
            raise typer.Exit(code=1)

        console.print(
            f"[bold green]Mode: SUPERVISED LIVE OBSERVATION (Account: {account_id})[/bold green]"
        )
        stream = LiveCollectorStream(
            root=repo_root,
            minutes=minutes,
            account_id=account_id,
            allow_shortened_session=allow_shortened_session,
            status_path=status_file,
        )

    cfg = ObservationSessionConfig(
        session_id=eff_session_id,
        model_ids=tuple(model_ids),
        duration_minutes=minutes,
        account_id=account_id if not synthetic else None,
        assumed_payout=payout,
        is_synthetic=synthetic,
        allow_shortened_session=allow_shortened_session,
        status_path=status_file,
        cancel_file=cancel_file,
    )

    observer = ProspectiveShadowObserver(cfg, registry, root=repo_root)

    console.print(
        Panel(
            f"Session ID: [cyan]{eff_session_id}[/cyan]\n"
            f"Models Under Observation: [bold]{', '.join(model_ids)}[/bold]\n"
            f"Authorized Duration: {minutes} minutes\n"
            f"Assumed Payout: {payout * 100:.1f}%\n"
            f"Execution Authority: [yellow]NONE[/yellow] (Orders strictly disabled)\n\n"
            f"[dim]CANDLE-CLOSE COMPARISONS ARE HYPOTHETICAL RESEARCH BENCHMARKS "
            f"AND NEVER OBSERVED BROKER ORDER SETTLEMENTS. NO_DEMONSTRATED_EDGE.[/dim]",
            title="Prospective AI Shadow Observation Initialized",
            border_style="cyan",
        )
    )

    try:
        result = asyncio.run(observer.run(stream))
    except Exception as exc:
        console.print(f"[bold red]Observation session terminated with error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"Session ID: [cyan]{result.session_id}[/cyan]\n"
            f"Termination Reason: [bold]{result.termination_reason}[/bold]\n"
            f"Warmup State: {result.warmup_state} ({result.consecutive_valid_candles} valid bars)\n"
            f"Candles Processed: {result.total_candles_processed:,} "
            f"(Gaps: {result.continuity_gaps_count}, Stale: {result.stale_candles_count})\n"
            f"Opportunities: {result.accepted_opportunities_count} accepted / "
            f"{result.late_opportunities_rejected_count} late rejected\n"
            f"Clock Diagnostics: {result.clock_uncertainty_rejections_count} uncertainty, "
            f"{result.clock_disagreements_count} disagreements\n"
            f"Evidence Provenance: {result.genuine_observations_count} genuine / "
            f"{result.synthetic_observations_count} synthetic\n"
            f"Manifest SHA-256: [cyan]{result.manifest_sha256}[/cyan]",
            title="Session Integrity & Progress Summary",
            border_style="cyan",
        )
    )

    table = Table(
        title=f"Prospective Observation Summary: {result.session_id} ({result.termination_reason})",
        show_header=True,
    )
    table.add_column("Model ID", style="cyan")
    table.add_column("Predictions", justify="right")
    table.add_column("Trades (W/L/T)", justify="center")
    table.add_column("Decisive Win%", justify="right", style="green")
    table.add_column("Matched Maj%", justify="right")
    table.add_column("Wilson 95% CI", justify="center")
    table.add_column("Exp Binary", justify="right")
    table.add_column("Exp Digital", justify="right")
    table.add_column("BE Hurdle (Gap)", justify="right")
    table.add_column("Verdict", style="yellow")

    for mid, s in result.summaries.items():
        wlt = f"{s.win_count}/{s.loss_count}/{s.tie_count}"

        if s.trade_count > 0:
            win_str = f"{s.win_rate_decisive * 100:.2f}%"
            ci_str = (
                f"[{s.wilson_ci_lower_95_decisive * 100:.1f}%, "
                f"{s.wilson_ci_upper_95_decisive * 100:.1f}%]"
            )
            exp_bin_str = f"{s.expectancy_binary_pct:+.2f}%"
            exp_dig_str = f"{s.expectancy_digital_pct:+.2f}%"
            be_hurdle = s.break_even_win_rate * 100
            be_gap = (s.win_rate_decisive - s.break_even_win_rate) * 100
            be_str = f"{be_hurdle:.1f}% ({be_gap:+.2f}%)"
        else:
            win_str = "N/A"
            ci_str = "N/A"
            exp_bin_str = "N/A"
            exp_dig_str = "N/A"
            be_str = "N/A"

        maj_str = "N/A"
        maj_comp = next(
            (b for b in s.baseline_comparisons if b.baseline_name == "majority_class"),
            None,
        )
        if maj_comp is not None and maj_comp.is_verified and s.trade_count > 0:
            maj_str = f"{maj_comp.decisive_directional_accuracy * 100:.2f}%"

        table.add_row(
            mid,
            f"{s.total_predictions:,} (pend: {s.pending_count})",
            wlt,
            win_str,
            maj_str,
            ci_str,
            exp_bin_str,
            exp_dig_str,
            be_str,
            s.scientific_verdict,
        )

    console.print(table)
    console.print(f"[dim]{result.disclaimer}[/dim]")


@shadow_app.command(
    "verify-session",
    help="Verify cryptographic sealing and internal consistency of a shadow session.",
)
def verify_session_command(
    session_target: Annotated[
        str,
        typer.Argument(help="Session ID or direct path to session directory."),
    ],
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    candidate_path = Path(session_target)
    if candidate_path.is_dir():
        session_dir = candidate_path.resolve()
    else:
        try:
            validate_storage_identifier(session_target)
        except Exception as exc:
            console.print(f"[bold red]Invalid session identifier:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
        session_dir = (repo_root / DEFAULT_SHADOW_SESSIONS_DIR / session_target).resolve()

    if not session_dir.is_dir():
        console.print(f"[bold red]Session directory does not exist:[/bold red] {session_dir}")
        raise typer.Exit(code=1)

    result = verify_session_integrity(session_dir)
    if result["is_valid"]:
        console.print(
            Panel(
                f"[bold green]Session integrity verified successfully.[/bold green]\n\n"
                f"Session ID: [cyan]{result['session_id']}[/cyan]\n"
                f"Evidence Class: [bold]{result.get('evidence_classification', 'N/A')}[/bold] "
                f"(Prospective Verified: {result.get('is_prospective_verified')})\n"
                f"Manifest SHA-256: {result['checksums'].get('manifest_sha256')}\n"
                f"Predictions SHA-256: {result['checksums'].get('predictions_sha256')}\n"
                f"Outcomes SHA-256: {result['checksums'].get('outcomes_sha256')}\n"
                f"Commitments SHA-256: {result['checksums'].get('commitments_sha256', 'N/A')}\n"
                f"Session JSON SHA-256: {result['checksums'].get('session_json_sha256')}\n\n"
                f"[dim]All digests match; journal logs are contiguous and valid.[/dim]",
                title="Cryptographic Sealing Verified",
                border_style="green",
            )
        )
    else:
        err_lines = "\n".join(f"- {e}" for e in result["errors"])
        console.print(
            Panel(
                f"[bold red]Session integrity verification FAILED:[/bold red]\n\n"
                f"{err_lines}\n\n"
                f"Session Directory: {session_dir}",
                title="Integrity Verification Failure",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)
