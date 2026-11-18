"""CLI commands for market data acquisition timing diagnostics and feasibility audits."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.timing.blitz_audit import audit_blitz_market_evidence
from iqoption_research.timing.feasibility import run_feasibility_assessment
from iqoption_research.timing.protocol import get_supervised_protocol_spec
from iqoption_research.timing.reconstruction import reconstruct_all_timing_evidence

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
    table.add_column("P(<= 15s Gate)", justify="center")
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
