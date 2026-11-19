"""CLI commands for market data acquisition timing diagnostics and feasibility audits."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.timing.blitz_audit import audit_blitz_market_evidence
from iqoption_research.timing.evidence import TimingEvidenceRegistry
from iqoption_research.timing.feasibility import run_feasibility_assessment
from iqoption_research.timing.pipeline_audit import audit_full_collection_pipeline
from iqoption_research.timing.protocol import get_supervised_protocol_spec
from iqoption_research.timing.reconstruction import reconstruct_all_timing_evidence
from iqoption_research.timing.replay import OfflineTimingReplayHarness
from iqoption_research.timing.supervised_session import (
    HARD_MAX_CANDLE_CYCLES,
    HARD_MAX_DURATION_SECONDS,
    SupervisedSessionConfig,
    SupervisedSessionRunner,
)

app = typer.Typer(
    help="Offline timing diagnostics, timestamp reconstruction, and acquisition feasibility audits."
)
console = Console()


def _project_root() -> Path:
    # src/iqoption_research/cli/commands/timing_cmd.py -> root is parents[4]
    return Path(__file__).resolve().parents[4]


@app.command(
    "audit",
    help="Audit all historical checkpoints and prospective journals for timing evidence.",
)
def audit_command(
    project_dir: Path = typer.Option(None, "--project-dir", help="Project root directory"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only"),
) -> None:
    root = project_dir or _project_root()
    evidence = reconstruct_all_timing_evidence(root)

    if json_output:
        typer.echo(json.dumps(evidence, indent=2))
        return

    console.print(
        Panel(
            "[bold cyan]Market Data Acquisition Timing Evidence Audit[/bold cyan]\n"
            "Forensic reconstruction of forward checkpoints and prospective observation journals.",
            title="Timing Audit",
            border_style="cyan",
        )
    )

    # Checkpoints Table
    table = Table(
        title="Historical Forward Sync Checkpoints Timing Decomposition",
        border_style="blue",
    )
    table.add_column("Job ID", style="bold yellow")
    table.add_column("Target End (F)", justify="center")
    table.add_column("Invocation Offset", justify="right")
    table.add_column("Collection Duration", justify="right")
    table.add_column("Total Offset", justify="right")
    table.add_column("15s Gate Compliant?", justify="center")

    for cp in evidence.get("checkpoints", []):
        if "error" in cp:
            continue
        valid_str = (
            "[bold green]YES[/bold green]"
            if cp["satisfies_phase30_15s_gate"]
            else "[bold red]NO[/bold red]"
        )
        table.add_row(
            cp["job_id"],
            str(cp["target_completed_end_ts"]),
            f"{cp['invocation_delay_seconds']:.2f} s",
            f"{cp['collection_duration_seconds']:.2f} s",
            f"{cp['total_offset_seconds']:.2f} s",
            valid_str,
        )
    console.print(table)

    # Observations Table
    obs_table = Table(title="Prospective Prediction Timing Gates", border_style="magenta")
    obs_table.add_column("Phase", style="bold")
    obs_table.add_column("Prediction Key", style="yellow")
    obs_table.add_column("Feature Close F", justify="center")
    obs_table.add_column("Offset to Decision", justify="right")
    obs_table.add_column("Phase 29 Gate", justify="center")
    obs_table.add_column("Phase 30 Gate (15s)", justify="center")

    for phase, preds in evidence.get("prospective_journals", {}).items():
        if not preds:
            obs_table.add_row(phase, "(Zero recorded predictions)", "-", "-", "-", "-")
            continue
        for p in preds:
            p29_str = "[green]PASS[/green]" if p["meets_phase29_gate"] else "[red]FAIL[/red]"
            p30_str = (
                "[green]PASS[/green]" if p["meets_phase30_gate"] else "[bold red]FAIL[/bold red]"
            )
            obs_table.add_row(
                phase,
                p["prediction_key"],
                str(p["feature_close_F"]),
                f"{p['offset_decided_from_F']:.2f} s",
                p29_str,
                p30_str,
            )
    console.print(obs_table)

    # Findings Panel
    findings = evidence.get("epistemic_findings", {})
    body_text = (
        "[bold]Epistemic Status of Existing Timestamps:[/bold]\n"
        "• [green]Genuinely Measured:[/green] Client process startup, parquet write, "
        "decision clock, SQLite commit.\n"
        "• [yellow]Proxy / Discrete Labels:[/yellow] Candle from_ts/to_ts, pipeline "
        "available_at_ts (hardcoded to to_ts).\n"
        "• [red]Unmeasured Physical Events:[/red] Broker candle frame emit, socket wire transit, "
        "physical socket read.\n\n"
        f"[bold yellow]Conclusion:[/bold yellow] {findings.get('critical_conclusion')}"
    )
    console.print(
        Panel(
            body_text,
            title="Epistemic Findings",
            border_style="yellow",
        )
    )


@app.command(
    "feasibility",
    help="Evaluate mathematical latency budgets and 15-second gate feasibility.",
)
def feasibility_command(
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only"),
) -> None:
    assessment = run_feasibility_assessment()

    if json_output:
        typer.echo(json.dumps(assessment, indent=2))
        return

    console.print(
        Panel(
            "[bold cyan]Acquisition Latency Budget & 15-Second Gate Feasibility[/bold cyan]\n"
            "Evaluation across cold-start CLI, warm background polling, and streaming websocket.",
            title="Feasibility Assessment",
            border_style="cyan",
        )
    )

    table = Table(title="Architecture Latency Comparison", border_style="blue")
    table.add_column("Architecture", style="bold yellow")
    table.add_column("Best Case", justify="right")
    table.add_column("Typical", justify="right")
    table.add_column("Worst Case", justify="right")
    table.add_column("Model 15s Ratio", justify="center")
    table.add_column("Safety Compliant", justify="center")
    table.add_column("Verified in Repo", justify="center")

    for _key, arch in assessment.get("architectures", {}).items():
        prob_str = f"{arch['probability_meeting_15s_gate'] * 100:.1f}%"
        prob_style = "bold green" if arch["probability_meeting_15s_gate"] > 0.5 else "bold red"
        safe_str = (
            "[green]YES[/green]"
            if arch["read_only_safety_compliant"]
            else "[red]NO (Requires Scope Change)[/red]"
        )
        ver_str = (
            "[green]YES[/green]"
            if arch["verified_in_codebase"]
            else "[yellow]NO (Hypothetical)[/yellow]"
        )

        table.add_row(
            arch["architecture_name"],
            f"{arch['total_best_case_seconds']:.2f} s",
            f"{arch['total_typical_seconds']:.2f} s",
            f"{arch['total_worst_case_seconds']:.2f} s",
            f"[{prob_style}]{prob_str}[/{prob_style}]",
            safe_str,
            ver_str,
        )
    console.print(table)

    verdict = assessment.get("phase30_gate_verdict", {})
    status_label = verdict.get("current_verified_workflow_status")
    console.print(
        Panel(
            f"[bold red]Verdict for Current Verified Workflow: {status_label}[/bold red]\n\n"
            f"{verdict.get('reasoning')}\n\n"
            f"[bold]Policy:[/bold] {verdict.get('gate_relaxation_policy')}",
            title="Phase 30 Gate Decision",
            border_style="red",
        )
    )


@app.command(
    "horizons",
    help="Evaluate alternative prediction horizons and timeframe latency scaling.",
)
def horizons_command(
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only"),
) -> None:
    assessment = run_feasibility_assessment()
    horizons = assessment.get("horizon_scaling_evaluations", [])

    if json_output:
        typer.echo(json.dumps(horizons, indent=2))
        return

    console.print(
        Panel(
            "[bold cyan]Alternative Prediction Horizons & Timeframe Latency Scaling[/bold cyan]\n"
            "Conceptual scaling of acquisition latency relative to candle holding periods.",
            title="Horizon Scaling",
            border_style="cyan",
        )
    )

    table = Table(title="Timeframe Scaling Metrics", border_style="green")
    table.add_column("Timeframe", style="bold yellow")
    table.add_column("Cold Latency %", justify="right")
    table.add_column("Warm Latency %", justify="right")
    table.add_column("Pre-Entry Window", justify="right")
    table.add_column("Cold Margin", justify="right")
    table.add_column("Signal Quality", justify="center")
    table.add_column("Feasibility Assessment", style="cyan")

    for h in horizons:
        table.add_row(
            h["timeframe_label"],
            f"{h['cold_latency_fraction'] * 100:.1f}%",
            f"{h['warm_latency_fraction'] * 100:.2f}%",
            f"{h['available_pre_entry_seconds']} s",
            f"{h['cold_pre_entry_margin_seconds']} s",
            h["statistical_signal_quality"],
            h["feasibility_verdict"],
        )
    console.print(table)


@app.command(
    "blitz",
    help="Audit authentic market evidence for EURUSD Spot vs. Blitz Options.",
)
def blitz_command(
    project_dir: Path = typer.Option(None, "--project-dir", help="Project root directory"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only"),
) -> None:
    root = project_dir or _project_root()
    audit = audit_blitz_market_evidence(root)

    if json_output:
        typer.echo(json.dumps(audit, indent=2))
        return

    console.print(
        Panel(
            "[bold cyan]EURUSD Spot Data vs. IQ Option Blitz Contract Mechanics[/bold cyan]\n"
            "Ground truth audit separating verified empirical facts from unverified unknowns.",
            title="Market Evidence Audit",
            border_style="cyan",
        )
    )

    table = Table(title="Evidence Matrix", border_style="yellow")
    table.add_column("Dimension", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Source / Proof", style="dim")
    table.add_column("Details & Research Impact", style="cyan")

    for item in audit.get("items", []):
        status = item["empirical_status"]
        if status == "VERIFIED_FACT":
            status_str = "[bold green]VERIFIED FACT[/bold green]"
        elif status == "UNVERIFIED_UNKNOWN":
            status_str = "[bold red]UNVERIFIED UNKNOWN[/bold red]"
        else:
            status_str = "[bold yellow]ASSUMPTION[/bold yellow]"

        table.add_row(
            item["dimension"],
            status_str,
            item["evidence_source"],
            f"{item['details']}\n-> {item['research_impact']}",
        )
    console.print(table)

    synthesis = audit.get("synthesis", {})
    console.print(
        Panel(
            f"[bold]Core Finding:[/bold] {synthesis.get('core_finding')}\n\n"
            f"[bold red]Research Conclusion: {synthesis.get('research_conclusion')}[/bold red]",
            title="Synthesis",
            border_style="red",
        )
    )


@app.command(
    "protocol",
    help="Display the operator-supervised measurement protocol specification.",
)
def protocol_command(
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only"),
) -> None:
    spec = get_supervised_protocol_spec()

    if json_output:
        typer.echo(json.dumps(spec, indent=2))
        return

    console.print(
        Panel(
            "[bold cyan]Operator-Supervised Acquisition Timing Protocol Specification[/bold cyan]\n"
            "Formal specification for future supervised measurement session "
            "(SPECIFICATION ONLY - UNEXECUTED).",
            title="Supervised Protocol",
            border_style="cyan",
        )
    )
    console.print(json.dumps(spec, indent=2))


@app.command(
    "pipeline-audit",
    help="Audit all pipeline stages from broker response to durable journal commit.",
)
def pipeline_audit_command(
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only"),
) -> None:
    audit_data = audit_full_collection_pipeline()

    if json_output:
        typer.echo(json.dumps(audit_data, indent=2))
        return

    console.print(
        Panel(
            "[bold cyan]Full Data Pipeline Timing & Timestamp Audit[/bold cyan]\n"
            "Decomposition of lifecycle stages from broker emission to SQLite journal commitment.",
            title="Pipeline Audit",
            border_style="cyan",
        )
    )

    table = Table(title="Pipeline Stages Decomposition", border_style="blue")
    table.add_column("Stage ID", style="bold yellow")
    table.add_column("Stage Name", style="bold")
    table.add_column("Code Location", style="dim")
    table.add_column("Typical Latency", justify="right")
    table.add_column("Existing Timestamps", style="green")
    table.add_column("Missing / Uninstrumented", style="red")

    for s in audit_data["stages"]:
        existing = ", ".join(t["name"] for t in s["existing_timestamps"]) or "(none)"
        missing = ", ".join(t["name"] for t in s["missing_timestamps"]) or "(none)"
        table.add_row(
            s["stage_id"],
            s["stage_name"],
            s["code_location"],
            f"{s['typical_latency_ms']:.1f} ms",
            existing,
            missing,
        )
    console.print(table)

    summary_table = Table(title="Epistemic Classification Breakdown", border_style="magenta")
    summary_table.add_column("Classification", style="bold")
    summary_table.add_column("Count", justify="right")
    for k, v in audit_data["epistemic_breakdown"].items():
        summary_table.add_row(k.replace("_", " ").title(), str(v))
    console.print(summary_table)


@app.command(
    "replay",
    help="Run deterministic offline timing replay simulations across 10 failure scenarios.",
)
def replay_command(
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only"),
) -> None:
    harness = OfflineTimingReplayHarness()
    results = harness.run_all_scenarios()

    if json_output:
        typer.echo(json.dumps([r.to_dict() for r in results], indent=2))
        return

    console.print(
        Panel(
            "[bold cyan]Offline Timing Simulation & Replay Harness[/bold cyan]\n"
            "Evaluating fail-closed invariants across 10 edge, delay, and corruption scenarios.",
            title="Timing Replay",
            border_style="cyan",
        )
    )

    table = Table(title="Scenario Verification Matrix", border_style="green")
    table.add_column("Scenario", style="bold yellow")
    table.add_column("Expected", justify="center")
    table.add_column("Actual Verdict", justify="center")
    table.add_column("Gate Satisfied?", justify="center")
    table.add_column("Delivery Offset", justify="right")
    table.add_column("Pipeline Latency", justify="right")
    table.add_column("Safety Passed?", justify="center")

    all_passed = True
    for r in results:
        gate_str = "[green]YES[/green]" if r.phase30_gate_satisfied else "[red]NO[/red]"
        safety_str = (
            "[bold green]PASS[/bold green]"
            if r.passed_safety_invariants
            else "[bold red]FAIL[/bold red]"
        )
        if not r.passed_safety_invariants:
            all_passed = False

        off_str = (
            f"{r.broker_delivery_offset_ms:.1f} ms"
            if r.broker_delivery_offset_ms is not None
            else "-"
        )
        pipe_str = (
            f"{r.local_pipeline_total_ms:.1f} ms" if r.local_pipeline_total_ms is not None else "-"
        )

        table.add_row(
            r.scenario_name,
            r.expected_verdict,
            r.actual_verdict,
            gate_str,
            off_str,
            pipe_str,
            safety_str,
        )
    console.print(table)

    if all_passed:
        console.print(
            Panel(
                "[bold green]All 10 replay scenarios passed safety invariants.[/bold green]\n"
                "System verified to fail closed on late candles, incomplete bars, duplicate "
                "frames, clock jumps, and causality violations.",
                title="Simulation Status",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]One or more simulation scenarios failed invariants.[/bold red]",
                title="Simulation Failure",
                border_style="red",
            )
        )


@app.command(
    "measure",
    help="Bounded operator-supervised measurement command (max 300s, max 5 cycles).",
)
def measure_command(
    live: bool = typer.Option(
        False,
        "--live",
        help="Enable live operator-supervised mode (interactive prompt; zero orders)",
    ),
    yes_i_understand: bool = typer.Option(
        False,
        "--yes-i-understand",
        help="Explicit confirmation that operator is present and understands measurement bounds",
    ),
    simulation: bool = typer.Option(
        False,
        "--simulation",
        help="Run bounded measurement in safe offline simulation mode",
    ),
    duration: int = typer.Option(
        300,
        "--duration",
        help=f"Maximum duration in seconds (hard ceiling {HARD_MAX_DURATION_SECONDS}s)",
    ),
    max_candles: int = typer.Option(
        5,
        "--max-candles",
        help=f"Maximum candle cycles to observe (hard ceiling {HARD_MAX_CANDLE_CYCLES})",
    ),
    output: Path = typer.Option(
        Path("data/phase33/supervised_timing_evidence.json"),
        "--output",
        help="Path to persist the immutable timing evidence registry",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Allow overwriting an existing evidence file (default: create-only)",
    ),
    project_dir: Path = typer.Option(None, "--project-dir", help="Project root directory"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only"),
) -> None:
    root = project_dir or _project_root()

    if duration <= 0:
        console.print("[bold red]Error:[/bold red] duration must be positive (>= 1).")
        raise typer.Exit(code=1)
    if max_candles <= 0:
        console.print("[bold red]Error:[/bold red] max-candles must be positive (>= 1).")
        raise typer.Exit(code=1)

    # Determine execution mode
    if live:
        if not yes_i_understand:
            console.print(
                "[bold red]Error:[/bold red] Live measurement requires explicit operator "
                "confirmation flag: [yellow]--yes-i-understand[/yellow]."
            )
            raise typer.Exit(code=1)
        if overwrite:
            console.print(
                "[bold red]Error:[/bold red] Overwriting evidence (--overwrite) is strictly "
                "forbidden in live authenticated measurement mode to protect broker evidence."
            )
            raise typer.Exit(code=1)
        sim_mode = False
    else:
        # Default to safe offline simulation mode
        sim_mode = True
        _ = simulation

    config = SupervisedSessionConfig(
        symbol="EURUSD",
        candle_size_seconds=60,
        max_duration_seconds=min(duration, HARD_MAX_DURATION_SECONDS),
        max_candle_cycles=min(max_candles, HARD_MAX_CANDLE_CYCLES),
        live_mode=live,
        simulation_mode=sim_mode,
        operator_confirmed=yes_i_understand if live else True,
        allow_overwrite=overwrite,
        output_destination=output,
        project_root=root,
    )

    runner = SupervisedSessionRunner(config)
    try:
        registry = asyncio.run(runner.execute())
    except FileExistsError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(registry.to_dict(), indent=2))
        return

    console.print(
        Panel(
            f"[bold cyan]Timing Measurement Evidence Registry Preserved[/bold cyan]\n"
            f"Session ID: [yellow]{registry.session_id}[/yellow]\n"
            f"Mode: [bold]{registry.operational_mode}[/bold]\n"
            f"Destination: [dim]{runner.output_destination}[/dim]\n"
            f"Integrity Hash: [green]{registry.integrity_hash}[/green]",
            title="Measurement Summary",
            border_style="cyan",
        )
    )

    table = Table(title="Observed Candle Cycles", border_style="blue")
    table.add_column("Cycle", justify="center")
    table.add_column("Nominal Close (F)", justify="center")
    table.add_column("Receipt <=15s")
    table.add_column("Validation <=15s")
    table.add_column("Acquisition")
    table.add_column("Full contract", no_wrap=True)
    for obs in registry.observations:
        receipt = obs["candle_received_within_15s"]
        validation = obs["candle_validated_within_15s"]
        full = "PASS" if obs["phase30_gate_satisfied"] else "FAIL"
        lats = obs["latencies_ms"]
        receipt_text = "PASS" if receipt else "NO"
        if "broker_delivery_offset_ms" in lats:
            receipt_text += f"\n{lats['broker_delivery_offset_ms']:.1f} ms"
        latency = lats.get("close_to_commit_total_ms")
        if latency is not None:
            full += f"\n{latency:.1f} ms"
        table.add_row(
            str(obs["observation_index"]),
            str(obs["feature_close_F"]),
            receipt_text,
            "PASS" if validation else "NO",
            "PASS" if receipt and validation and not obs["rejection_reason"] else "INCOMPLETE/FAIL",
            full if obs["prediction_stages_measured"] else "NOT_MEASURED",
        )
    console.print(table)


@app.command(
    "verify-evidence",
    help="Verify internal consistency and SHA-256 integrity hashes of an evidence registry file.",
)
def verify_evidence_command(
    path: Path = typer.Option(
        ...,
        "--path",
        "-p",
        help="Path to the evidence registry JSON file to verify",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only"),
) -> None:
    path = path.resolve()
    if not path.is_file():
        console.print(f"[bold red]Error:[/bold red] Evidence file not found: {path}")
        raise typer.Exit(code=1)

    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        registry = TimingEvidenceRegistry.from_dict(content)
        is_valid, errors = registry.verify_integrity()
        if is_valid and registry.instrumentation_version == "PHASE_34_SHADOW_V1":
            from iqoption_research.timing.shadow import verify_shadow_receipts

            verify_shadow_receipts(registry, path.parent)
    except (ValueError, OSError, KeyError, TypeError, AttributeError, sqlite3.Error) as exc:
        is_valid = False
        errors = [f"Failed to load or parse evidence file: {exc}"]
        registry = None

    if json_output:
        res = {
            "path": str(path),
            "is_valid": is_valid,
            "errors": errors,
            "session_id": registry.session_id if registry else None,
            "operational_mode": registry.operational_mode if registry else None,
            "integrity_hash": registry.integrity_hash if registry else None,
        }
        typer.echo(json.dumps(res, indent=2))
        if not is_valid:
            raise typer.Exit(code=1)
        return

    if is_valid and registry is not None:
        recv_cnt = registry.summary.get("candle_received_within_15s_count", 0)
        val_cnt = registry.summary.get("candle_validated_within_15s_count", 0)
        gate_cnt = registry.summary.get("full_phase30_gate_compliant_count", 0)
        measured = registry.summary["prediction_stages_measured_count"]
        full = f"{gate_cnt}/{measured} measured" if measured else "NOT_MEASURED"
        console.print(
            Panel(
                f"[bold green]Evidence Registry Verification PASSED[/bold green]\n\n"
                f"Session ID:         [yellow]{registry.session_id}[/yellow]\n"
                f"Created UTC:        {registry.session_created_at_utc}\n"
                f"Operational Mode:   [bold]{registry.operational_mode}[/bold]\n"
                f"Observations:       {len(registry.observations)}\n"
                f"Receipt <= 15s:     {recv_cnt}\n"
                f"Validated <= 15s:   {val_cnt}\n"
                f"Full Phase 30 Gate: {full}\n"
                f"Integrity Hash:     [green]{registry.integrity_hash}[/green]\n\n"
                "[dim]Epistemic Notice: Unkeyed SHA-256 integrity hash provides tamper-evidence "
                "against accidental mutation, not cryptographic non-repudiation.[/dim]",
                title="Evidence Verification Result",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]Evidence Registry Verification FAILED[/bold red]\n\n"
                f"Path: {path}\n"
                f"Detected {len(errors)} error(s):",
                title="Evidence Verification Failure",
                border_style="red",
            )
        )
        for err in errors:
            console.print(f"  [red]•[/red] {err}")
        raise typer.Exit(code=1)


@app.command("reliability")
def reliability_command(paths: list[Path] = typer.Argument(...)) -> None:
    """Evaluate immutable registries or directories offline."""
    from iqoption_research.timing.reliability import evaluate

    try:
        report = evaluate(paths)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(report, indent=2))
    if report["invalid_registries"]:
        raise typer.Exit(1)


@app.command("shadow-trial")
def shadow_trial_command(
    live: bool = typer.Option(False, "--live"),
    yes_i_understand: bool = typer.Option(False, "--yes-i-understand"),
    duration: int = typer.Option(300, min=1, max=300),
    max_candles: int = typer.Option(5, min=1, max=5),
) -> None:
    """Preflight offline or run one foreground read-only EURUSD shadow trial."""
    import sys

    from iqoption_research.timing.shadow import MODEL_PATH, ShadowTrial, eligible_model

    if not live:
        typer.echo(
            json.dumps({
                "mode": "OFFLINE_PREFLIGHT",
                "model_eligible": eligible_model(MODEL_PATH) is not None,
                "orders_sent": 0,
            })
        )
        return
    if not yes_i_understand or not sys.stdin.isatty() or not sys.stdout.isatty():
        raise typer.BadParameter(
            "Live trials require an interactive terminal and --yes-i-understand"
        )
    config = SupervisedSessionConfig(
        live_mode=True,
        simulation_mode=False,
        operator_confirmed=True,
        max_duration_seconds=duration,
        max_candle_cycles=max_candles,
        project_root=_project_root(),
    )
    trial = ShadowTrial(config, Path("data/phase34"), MODEL_PATH)
    try:
        registry = asyncio.run(trial.run_live())
    finally:
        typer.echo(f"Shadow evidence destination {trial.output}")
    typer.echo(json.dumps(registry.summary, indent=2))


@app.command(
    "evaluate-predictions",
    help="Offline prospective evaluation pairing predictions with authentic candle outcomes.",
)
def evaluate_predictions_command(
    evidence: list[Path] = typer.Option(
        None,
        "--evidence",
        "-e",
        help="Path(s) to shadow evidence JSON file(s) or directory.",
    ),
    journal: Path | None = typer.Option(
        None,
        "--journal",
        "-j",
        help="Path to prospective SQLite journal database (observations.db).",
    ),
    storage: Path = typer.Option(
        Path("data/datasets"),
        "--storage",
        "-s",
        help="Path to historical authentic catalog storage.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Filter to a specific session ID to prevent cross-session combination.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write the immutable prospective evaluation report JSON.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only to stdout."),
) -> None:
    from iqoption_research.research.prospective_eval import (
        CONTRACT_PROFITABILITY_DISCLAIMER,
        ProspectiveEvaluationError,
        evaluate_prospective_shadow_offline,
    )

    paths = evidence or [Path("data/phase34/operator_supervised_live")]
    try:
        report = evaluate_prospective_shadow_offline(
            paths,
            journal_path=journal,
            storage_dir=storage if storage.is_dir() else None,
            session_filter=session_id,
        )
    except ProspectiveEvaluationError as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc), "status": "FAILED"}, indent=2))
        else:
            console.print(f"[bold red]Prospective Evaluation Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
        return

    # Rich formatted presentation
    console.print(
        Panel(
            "[bold cyan]Authentic Prospective Model Evaluation (Phase 36)[/bold cyan]\n"
            "Causally aligned offline evaluation against authentic outcomes.\n"
            "[yellow]Authority: NONE | Disposition: NO_DEMONSTRATED_EDGE | Orders: 0[/yellow]",
            title="Model Evaluation Pipeline",
            border_style="cyan",
        )
    )

    # Cohort & Overview Table
    meta_table = Table(title="Evaluation Cohort & Summary", border_style="blue")
    meta_table.add_column("Property", style="bold yellow")
    meta_table.add_column("Value", style="white")

    meta_table.add_row("Sessions Evaluated", ", ".join(report.cohort.sessions))
    meta_table.add_row("Model Name", str(report.cohort.model_name))
    meta_table.add_row(
        "Model Manifest SHA-256", str(report.cohort.model_manifest_sha256)[:16] + "..."
    )
    meta_table.add_row("Total Committed Predictions", str(report.total_predictions))
    meta_table.add_row("Causally Aligned Pairs", f"[green]{report.aligned_pairs_count}[/green]")
    meta_table.add_row(
        "Missing/Pending Outcomes",
        f"[yellow]{report.missing_outcomes_count}[/yellow]"
        if report.missing_outcomes_count > 0
        else "0",
    )
    meta_table.add_row("Signal Coverage Rate", f"{report.signal_coverage_rate * 100:.1f}%")
    meta_table.add_row("Abstention Rate", f"{report.abstention_rate * 100:.1f}%")
    meta_table.add_row("Evidence Provenance", f"[bold]{report.provenance_level}[/bold]")
    meta_table.add_row(
        "Authentic Outcomes Verified",
        f"[bold]{report.all_outcomes_authentic}[/bold]",
    )
    meta_table.add_row("Report Evidence SHA-256", f"[dim]{report.evidence_sha256[:16]}...[/dim]")
    console.print(meta_table)

    # Decisions breakdown
    dec_table = Table(title="Decisions & Policy Abstention", border_style="magenta")
    dec_table.add_column("Decision", justify="center", style="bold")
    dec_table.add_column("Count", justify="right")
    dec_table.add_column("Notes", style="dim")
    for dec, cnt in report.decision_distribution.items():
        dec_table.add_row(dec, str(cnt), "Frozen threshold: 0.60")
    console.print(dec_table)

    if report.skip_reasons_breakdown:
        skip_table = Table(title="Analytical Skip Reasons Breakdown", border_style="dim")
        skip_table.add_column("Reason", style="yellow")
        skip_table.add_column("Count", justify="right")
        for reason, count in report.skip_reasons_breakdown.items():
            skip_table.add_row(reason, str(count))
        console.print(skip_table)

    # Missing Labels Notice
    if report.missing_outcomes_count > 0:
        missing_str = ", ".join(str(b) for b in report.missing_boundaries[:10])
        if len(report.missing_boundaries) > 10:
            missing_str += f" (+{len(report.missing_boundaries) - 10} more)"
        console.print(
            Panel(
                f"[bold yellow]Missing Outcome Labels:[/bold yellow] "
                f"{report.missing_outcomes_count} boundary(ies) await backfill.\n"
                f"Missing boundaries: [dim]{missing_str}[/dim]\n"
                "Predictions remain preserved. Forecast scoring will complete once candles arrive.",
                title="Label Completeness Notice",
                border_style="yellow",
            )
        )

    # Probabilistic & Directional Metrics (if aligned)
    prob = report.probabilistic_metrics
    if prob.get("status") == "COMPLETED":
        metrics_table = Table(title="Probabilistic Forecasting Quality", border_style="green")
        metrics_table.add_column("Metric", style="bold green")
        metrics_table.add_column("Score", justify="right")
        metrics_table.add_column("Description", style="dim")

        metrics_table.add_row(
            "Brier Score",
            f"{prob.get('brier_score', 0.0):.4f}",
            "Mean squared probability error (lower is better)",
        )
        metrics_table.add_row(
            "Log Loss",
            f"{prob.get('log_loss', 0.0):.4f}",
            "Binary cross-entropy loss (lower is better)",
        )
        metrics_table.add_row(
            "Expected Calibration Error (ECE)",
            f"{prob.get('expected_calibration_error', 0.0):.4f}",
            "Reliability bin divergence",
        )
        metrics_table.add_row(
            "Mean Calibration Bias",
            f"{prob.get('mean_calibration_bias', 0.0):+.4f}",
            "Predicted mean - Empirical base rate",
        )
        metrics_table.add_row(
            "Spot Directional Accuracy",
            f"{prob.get('directional_accuracy_overall', 0.0) * 100:.1f}%",
            "Overall directional forecast accuracy",
        )
        if prob.get("directional_accuracy_non_neutral") is not None:
            metrics_table.add_row(
                "Non-Neutral Directional Accuracy",
                f"{prob.get('directional_accuracy_non_neutral', 0.0) * 100:.1f}%",
                "Excluding flat candles (C == O)",
            )
        metrics_table.add_row(
            "Binary Target Accuracy",
            f"{prob.get('binary_target_accuracy', 0.0) * 100:.1f}%",
            "Accuracy on training target 1(C > O)",
        )
        console.print(metrics_table)

        # Benchmarks Table
        if report.benchmarks:
            bench_table = Table(title="Benchmark Model Comparisons", border_style="cyan")
            bench_table.add_column("Benchmark", style="bold")
            bench_table.add_column("Causal / Type")
            bench_table.add_column("Brier Score", justify="right")
            bench_table.add_column("Brier Skill Score", justify="right")
            bench_table.add_column("Log Loss", justify="right")
            bench_table.add_column("Accuracy", justify="right")

            for b in report.benchmarks:
                bss_str = (
                    f"[bold green]{b.brier_skill_score:+.4f}[/bold green]"
                    if b.brier_skill_score > 0
                    else f"[red]{b.brier_skill_score:+.4f}[/red]"
                )
                causal_tag = (
                    "[green]CAUSAL[/green]"
                    if b.is_causal
                    else "[yellow]ORACLE (NON-CAUSAL)[/yellow]"
                )
                bench_table.add_row(
                    b.name,
                    causal_tag,
                    f"{b.brier_score:.4f}",
                    bss_str,
                    f"{b.log_loss:.4f}",
                    f"{b.directional_accuracy * 100:.1f}%",
                )
            console.print(bench_table)

    # Uncertainty & Epistemic Panel
    u = report.uncertainty
    console.print(
        Panel(
            f"[bold]Epistemic Status:[/bold] [yellow]{u.epistemic_status}[/yellow]\n"
            f"[bold]Statistical Interpretation:[/bold] {u.interpretation}\n\n"
            f"[dim]{CONTRACT_PROFITABILITY_DISCLAIMER}[/dim]",
            title="Statistical Uncertainty & Safety Notice",
            border_style="yellow"
            if u.epistemic_status == "EXPLORATORY_DESCRIPTIVE_ONLY"
            else "green",
        )
    )

    if output is not None:
        console.print(f"[bold green]Evaluation report saved to:[/bold green] {output}")
