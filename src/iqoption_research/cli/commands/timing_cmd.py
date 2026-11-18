"""CLI commands for market data acquisition timing diagnostics and feasibility audits."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.timing.blitz_audit import audit_blitz_market_evidence
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
        Path("data/phase32/supervised_timing_evidence.json"),
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
            f"Destination: [dim]{output}[/dim]\n"
            f"Integrity Hash: [green]{registry.integrity_hash}[/green]",
            title="Measurement Summary",
            border_style="cyan",
        )
    )

    table = Table(title="Observed Candle Cycles", border_style="blue")
    table.add_column("Cycle", justify="center")
    table.add_column("Nominal Close (F)", justify="center")
    table.add_column("Gate Verdict", justify="center")
    table.add_column("15s Gate?", justify="center")
    table.add_column("Delivery Offset", justify="right")
    table.add_column("Pipeline Latency", justify="right")
    table.add_column("Total Latency", justify="right")

    for obs in registry.observations:
        g_str = (
            "[green]PASS[/green]" if obs["phase30_gate_satisfied"] else "[bold red]FAIL[/bold red]"
        )
        lats = obs.get("latencies_ms", {})
        table.add_row(
            str(obs["observation_index"]),
            str(obs["feature_close_F"]),
            obs["gate_verdict"],
            g_str,
            f"{lats.get('broker_delivery_offset_ms', 0):.1f} ms"
            if "broker_delivery_offset_ms" in lats
            else "-",
            f"{lats.get('local_pipeline_total_ms', 0):.1f} ms"
            if "local_pipeline_total_ms" in lats
            else "-",
            f"{lats.get('close_to_commit_total_ms', 0):.1f} ms"
            if "close_to_commit_total_ms" in lats
            else "-",
        )
    console.print(table)
