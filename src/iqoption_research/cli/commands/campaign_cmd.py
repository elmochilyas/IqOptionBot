"""CLI commands for Research Campaign Management & Analysis (Phase 40C).

Provides pre-registration, calendar scheduling, slot lifecycle management,
verification tracking, and unblinded day-block bootstrap statistical evaluation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.campaign.analysis import CampaignEvaluator
from iqoption_research.campaign.manager import CampaignManager
from iqoption_research.campaign.registry import CampaignRegistry
from iqoption_research.shadow.observer import verify_session_integrity

console = Console()
campaign_app = typer.Typer(
    name="campaign",
    help="Research Campaign Lifecycle & Day-Block Bootstrap Statistical Analysis (Phase 40C).",
    no_args_is_help=True,
)


@campaign_app.command("create", help="Create a new research campaign in DRAFT state.")
def create_campaign_command(
    campaign_id: Annotated[str, typer.Argument(help="Unique campaign identifier.")],
    start_date: Annotated[
        str,
        typer.Option("--start-date", "-d", help="First scheduled market date (YYYY-MM-DD)."),
    ],
    candidate_model_id: Annotated[
        str,
        typer.Option("--candidate-model-id", "-c", help="Registered candidate model ID."),
    ],
    baseline_model_id: Annotated[
        str,
        typer.Option(
            "--baseline-model-id",
            "-b",
            help="Baseline comparator model ID (e.g. baseline_majority_class).",
        ),
    ] = "baseline_majority_class",
    notes: Annotated[str, typer.Option("--notes", help="Campaign design notes.")] = "",
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    manager = CampaignManager(repo_root)

    try:
        record = manager.create_campaign(
            campaign_id=campaign_id,
            start_date=start_date,
            candidate_model_id=candidate_model_id,
            baseline_model_id=baseline_model_id,
            notes=notes,
        )
    except Exception as exc:
        console.print(f"[bold red]Failed to create campaign:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"[bold green]Research Campaign Created in DRAFT state.[/bold green]\n\n"
            f"Campaign ID: [cyan]{record.campaign_id}[/cyan]\n"
            f"Candidate Model: [bold]{record.protocol.candidate_model_id}[/bold] "
            f"(Threshold {record.protocol.candidate_threshold})\n"
            f"Baseline Model: [bold]{record.protocol.baseline_model_id}[/bold] "
            f"(Threshold {record.protocol.baseline_threshold})\n"
            f"Target Population: 40 eligible market days / 80 windows (08:00 & 14:00 UTC)\n"
            f"Local Timezone: [cyan]{record.protocol.local_timezone_name}[/cyan]\n"
            f"Minimum Target: {record.protocol.min_decisive_outcomes:,} decisive outcomes "
            f"/ {record.protocol.min_completed_days} days\n\n"
            f"[dim]Run 'register' to formalize pre-registration before collection begins.[/dim]",
            title="Research Campaign Draft Initialized",
            border_style="cyan",
        )
    )


@campaign_app.command("register", help="Formalize and lock pre-registration for a campaign.")
def register_campaign_command(
    campaign_id: Annotated[str, typer.Argument(help="Campaign identifier to pre-register.")],
    operator: Annotated[
        str,
        typer.Option("--operator", help="Supervised operator identity."),
    ] = "SUPERVISED_OPERATOR",
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    manager = CampaignManager(repo_root)

    try:
        record = manager.register_campaign(campaign_id, operator=operator)
    except Exception as exc:
        console.print(f"[bold red]Failed to register campaign:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"[bold green]Campaign Pre-registration Formalized & Locked.[/bold green]\n\n"
            f"Campaign ID: [cyan]{record.campaign_id}[/cyan]\n"
            f"Protocol Version: [bold]{record.protocol.protocol_version}[/bold]\n"
            f"Protocol SHA-256: [cyan]{record.protocol.protocol_sha256}[/cyan]\n"
            f"Registered Operator: [bold]{record.registered_operator}[/bold]\n"
            f"Registered At UTC: {record.registered_at_utc}\n"
            f"State: [bold green]{record.state}[/bold green]\n\n"
            f"[dim]Protocol is cryptographically sealed; "
            f"future observations must match protocol_sha256.[/dim]",
            title="Campaign Preregistration Sealed",
            border_style="green",
        )
    )


@campaign_app.command("list", help="List all research campaigns in the registry.")
def list_campaigns_command(
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    reg = CampaignRegistry(repo_root)
    campaigns = reg.list_all()

    if not campaigns:
        console.print("[dim]No research campaigns found in registry.[/dim]")
        return

    table = Table(title="Preregistered Research Campaigns", border_style="cyan")
    table.add_column("Campaign ID", style="cyan")
    table.add_column("State", style="bold")
    table.add_column("Protocol Version")
    table.add_column("Candidate Model")
    table.add_column("Completed / Slots")
    table.add_column("Verified")
    table.add_column("Protocol SHA-256", style="dim")

    for c in campaigns:
        proto_hash = (
            c.protocol.protocol_sha256[:16] + "..." if c.protocol.protocol_sha256 else "UNLOCKED"
        )
        table.add_row(
            c.campaign_id,
            c.state,
            c.protocol.protocol_version,
            c.protocol.candidate_model_id,
            f"{c.completed_sessions_count} / {len(c.slots)}",
            str(c.verified_sessions_count),
            proto_hash,
        )

    console.print(table)


@campaign_app.command("status", help="Show detailed execution status and schedule for a campaign.")
def status_campaign_command(
    campaign_id: Annotated[str, typer.Argument(help="Campaign identifier.")],
    limit_slots: Annotated[
        int, typer.Option("--limit", "-n", help="Number of slots to show.")
    ] = 10,
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    reg = CampaignRegistry(repo_root)
    camp = reg.get(campaign_id)
    if camp is None:
        console.print(f"[bold red]Campaign '{campaign_id}' not found.[/bold red]")
        raise typer.Exit(code=1)

    evaluator = CampaignEvaluator(campaign_id, root=repo_root)
    rep = evaluator.evaluate(unblind=False)

    console.print(
        Panel(
            f"Campaign ID: [cyan]{camp.campaign_id}[/cyan]\n"
            f"State: [bold]{camp.state}[/bold]\n"
            f"Protocol Version: {camp.protocol.protocol_version}\n"
            f"Protocol SHA-256: [cyan]{camp.protocol.protocol_sha256 or 'UNSEALED'}[/cyan]\n"
            f"Candidate: [bold]{camp.protocol.candidate_model_id}[/bold] "
            f"(Threshold {camp.protocol.candidate_threshold})\n"
            f"Baseline: [bold]{camp.protocol.baseline_model_id}[/bold] "
            f"(Threshold {camp.protocol.baseline_threshold})\n"
            f"Progress: {camp.completed_sessions_count} sessions completed, "
            f"{camp.verified_sessions_count} verified\n"
            f"Hurdle: {camp.protocol.break_even_hurdle:.4f} (85% payout assumption)\n"
            + (f"\n[bold yellow]{rep.blinding_notice}[/bold yellow]\n" if rep.is_blinded else ""),
            title=f"Campaign Status: {camp.campaign_id}",
            border_style="cyan",
        )
    )

    table = Table(title=f"Scheduled Observation Slots (Next {limit_slots})", border_style="dim")
    table.add_column("Slot ID", style="cyan")
    table.add_column("Day / Win")
    table.add_column("Date (UTC)")
    table.add_column("Start (UTC)")
    table.add_column("Start (Casablanca)")
    table.add_column("Status", style="bold")
    table.add_column("Session ID")

    for s in camp.slots[:limit_slots]:
        table.add_row(
            s.slot_id,
            f"Day {s.day_index} / W{s.window_index}",
            s.scheduled_date,
            s.scheduled_start_utc.split("T")[1][:5],
            s.scheduled_start_local.split("T")[1][:5],
            s.slot_state,
            s.session_id or "-",
        )

    console.print(table)


@campaign_app.command("verify", help="Verify cryptographic integrity of all campaign sessions.")
def verify_campaign_command(
    campaign_id: Annotated[str, typer.Argument(help="Campaign identifier to verify.")],
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    reg = CampaignRegistry(repo_root)
    camp = reg.get(campaign_id)
    if camp is None:
        console.print(f"[bold red]Campaign '{campaign_id}' not found.[/bold red]")
        raise typer.Exit(code=1)

    sessions_dir = repo_root / "data" / "shadow_eval" / "sessions"
    completed_slots = [s for s in camp.slots if s.session_id]

    if not completed_slots:
        console.print("[dim]No completed sessions linked to this campaign yet.[/dim]")
        return

    table = Table(title=f"Campaign Sessions Verification: {campaign_id}", border_style="cyan")
    table.add_column("Slot ID", style="cyan")
    table.add_column("Session ID")
    table.add_column("Valid", style="bold")
    table.add_column("Prospective")
    table.add_column("Schema 3")
    table.add_column("Errors", style="red")

    all_passed = True
    for s in completed_slots:
        if s.session_id is None:
            continue
        sess_dir = sessions_dir / s.session_id
        res = verify_session_integrity(sess_dir)
        is_val = res["is_valid"]
        is_pros = res.get("is_prospective_verified", False)
        is_s3 = res.get("is_schema3_verified", False)
        errs = "; ".join(res["errors"]) if res["errors"] else "None"
        if not is_val:
            all_passed = False

        table.add_row(
            s.slot_id,
            s.session_id,
            "[green]PASS[/green]" if is_val else "[red]FAIL[/red]",
            "[green]YES[/green]" if is_pros else "[yellow]NO[/yellow]",
            "[green]YES[/green]" if is_s3 else "[yellow]NO[/yellow]",
            errs,
        )

    console.print(table)
    if all_passed:
        console.print(
            "[bold green]All campaign sessions passed cryptographic verification.[/bold green]"
        )
    else:
        console.print("[bold red]One or more sessions failed verification.[/bold red]")
        raise typer.Exit(code=1)


@campaign_app.command(
    "analyze",
    help="Run Day-Block Bootstrap (20,000 replicates) and statistical hypothesis testing.",
)
def analyze_campaign_command(
    campaign_id: Annotated[str, typer.Argument(help="Campaign identifier to analyze.")],
    unblind: Annotated[
        bool,
        typer.Option(
            "--unblind",
            help="Explicit operator override to unblind confirmatory win-rate outcomes.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output results as JSON.")] = False,
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    try:
        evaluator = CampaignEvaluator(campaign_id, root=repo_root)
        rep = evaluator.evaluate(unblind=unblind)
    except Exception as exc:
        console.print(f"[bold red]Analysis failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        console.print(json.dumps(rep.to_dict(), indent=2))
        return

    if rep.is_blinded:
        console.print(
            Panel(
                f"[bold yellow]{rep.blinding_notice}[/bold yellow]\n\n"
                f"Campaign: [cyan]{rep.campaign_id}[/cyan]\n"
                f"Completed Days: {rep.completed_days}\n"
                f"Completed Sessions: {rep.completed_sessions}\n"
                f"Verified Sessions: {rep.verified_sessions}\n\n"
                f"[dim]Confirmatory blinding is active until study completion. "
                f"To irreversibly break blinding for exploratory inspection, run:\n"
                f"agy campaign break-blind {rep.campaign_id} --reason <audited_reason>[/dim]",
                title="Confirmatory Blinding Active",
                border_style="yellow",
            )
        )
        return

    # Unblinded statistical report
    b_ci = rep.day_block_bootstrap_ci_95
    boot_ci_str = f"[{b_ci.ci_lower:.4f}, {b_ci.ci_upper:.4f}]" if b_ci else "N/A"

    t_ci = rep.two_day_block_sensitivity_ci_95
    two_day_ci_str = f"[{t_ci.ci_lower:.4f}, {t_ci.ci_upper:.4f}]" if t_ci else "N/A"

    p_ci = rep.paired_diff_bootstrap_ci_95
    paired_ci_str = f"[{p_ci.ci_lower:.4f}, {p_ci.ci_upper:.4f}]" if p_ci else "N/A"

    decisive_summary = (
        f"Decisive Observations: {rep.decisive_observations_count:,} "
        f"(W={rep.total_wins}, L={rep.total_losses}, T={rep.total_ties}, Skips={rep.total_skips})"
    )
    hurdle_str = (
        f"Economic Break-Even Hurdle (1/1.85): {rep.break_even_hurdle:.4f} "
        f"(Clears Hurdle: [bold]{rep.clears_economic_hurdle}[/bold])"
    )
    brier_str = (
        f"Candidate Brier Score: {rep.candidate_brier_score} vs "
        f"Baseline: {rep.baseline_brier_score}"
    )
    exp_bin_str = (
        f"{rep.expectancy_binary_pct:+.2f}%" if rep.expectancy_binary_pct is not None else "N/A"
    )
    sens_dig_str = (
        f"{rep.sensitivity_digital_pct:+.2f}%" if rep.sensitivity_digital_pct is not None else "N/A"
    )
    expectancy_str = f"Binary Expectancy: {exp_bin_str} | Digital Sensitivity: {sens_dig_str}"

    cov_pct_str = f"{rep.coverage_rate * 100:.1f}%" if rep.coverage_rate is not None else "N/A"
    abs_pct_str = f"{rep.abstention_rate * 100:.1f}%" if rep.abstention_rate is not None else "N/A"
    coverage_str = f"Coverage: {cov_pct_str} | Abstention: {abs_pct_str}"

    cand_acc_str = f"{rep.decisive_accuracy:.4f}" if rep.decisive_accuracy is not None else "N/A"
    wilson_str = (
        f"[{rep.wilson_ci_95[0]:.4f}, {rep.wilson_ci_95[1]:.4f}]" if rep.wilson_ci_95 else "N/A"
    )
    base_acc_str = (
        f"{rep.baseline_decisive_accuracy:.4f}"
        if rep.baseline_decisive_accuracy is not None
        else "N/A"
    )
    paired_diff_str = (
        f"{rep.paired_accuracy_diff:+.4f}" if rep.paired_accuracy_diff is not None else "N/A"
    )

    console.print(
        Panel(
            f"[bold]Scientific Disposition Verdict:[/bold] [bold cyan]{rep.verdict}[/bold cyan]\n"
            f"Rationale: {rep.verdict_rationale}\n\n"
            f"Candidate Model: [bold]{rep.candidate_model_id}[/bold]\n"
            f"{decisive_summary}\n"
            f"Candidate Decisive Accuracy: [bold]{cand_acc_str}[/bold]\n"
            f"95% Wilson Score CI: {wilson_str}\n"
            f"95% Day-Block Bootstrap CI: {boot_ci_str} (20,000 replicates, seed 20261123)\n"
            f"95% Two-Day Block Sensitivity CI: {two_day_ci_str}\n"
            f"{hurdle_str}\n\n"
            f"Matched Baseline ({rep.baseline_model_id}) "
            f"Accuracy: {base_acc_str}\n"
            f"Paired Accuracy Difference: [bold]{paired_diff_str}[/bold] "
            f"(95% CI: {paired_ci_str})\n"
            f"Beats Paired Baseline: [bold]{rep.beats_paired_baseline}[/bold]\n\n"
            f"{brier_str}\n"
            f"{expectancy_str}\n"
            f"{coverage_str}\n\n"
            f"[dim]{rep.disclaimer}[/dim]",
            title="Scientific Campaign Evaluation (Day-Block Bootstrap Analysis)",
            border_style="cyan",
        )
    )


@campaign_app.command(
    "export", help="Export campaign summary and scheduled slots to Markdown, JSON, or CSV."
)
def export_campaign_command(
    campaign_id: Annotated[str, typer.Argument(help="Campaign identifier to export.")],
    format_type: Annotated[
        str,
        typer.Option("--format", "-f", help="Export format: markdown, json, or csv."),
    ] = "markdown",
    output_file: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination file path (prints to stdout if omitted)."),
    ] = None,
    unblind: Annotated[bool, typer.Option("--unblind", help="Include unblinded outcomes.")] = False,
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    reg = CampaignRegistry(repo_root)
    camp = reg.get(campaign_id)
    if camp is None:
        console.print(f"[bold red]Campaign '{campaign_id}' not found.[/bold red]")
        raise typer.Exit(code=1)

    evaluator = CampaignEvaluator(campaign_id, root=repo_root)
    rep = evaluator.evaluate(unblind=unblind)

    fmt = format_type.lower()
    if fmt == "json":
        out_content = json.dumps(
            {"campaign": camp.to_dict(), "analysis": rep.to_dict()},
            indent=2,
            sort_keys=True,
        )
    elif fmt == "csv":
        import io

        sio = io.StringIO()
        writer = csv.writer(sio)
        writer.writerow([
            "slot_id",
            "day_index",
            "window_index",
            "scheduled_date",
            "scheduled_start_utc",
            "scheduled_start_local",
            "slot_state",
            "session_id",
            "verified_integrity",
        ])
        for s in camp.slots:
            writer.writerow([
                s.slot_id,
                s.day_index,
                s.window_index,
                s.scheduled_date,
                s.scheduled_start_utc,
                s.scheduled_start_local,
                s.slot_state,
                s.session_id or "",
                s.verified_integrity,
            ])
        out_content = sio.getvalue()
    else:  # Markdown
        cand_line = (
            f"- **Candidate Model:** `{camp.protocol.candidate_model_id}` "
            f"(Threshold: {camp.protocol.candidate_threshold})"
        )
        base_line = (
            f"- **Baseline Model:** `{camp.protocol.baseline_model_id}` "
            f"(Threshold: {camp.protocol.baseline_threshold})"
        )
        b_ci_md = (
            f"`{rep.day_block_bootstrap_ci_95.ci_lower}` to "
            f"`{rep.day_block_bootstrap_ci_95.ci_upper}`"
            if rep.day_block_bootstrap_ci_95
            else "`N/A`"
        )
        paired_diff_str = (
            f"{rep.paired_accuracy_diff}" if rep.paired_accuracy_diff is not None else "N/A"
        )
        cand_acc_md = f"{rep.decisive_accuracy:.4f}" if rep.decisive_accuracy is not None else "N/A"
        out_content = f"""# Research Campaign Report: {camp.campaign_id}

- **Protocol Version:** {camp.protocol.protocol_version}
- **Protocol SHA-256:** `{camp.protocol.protocol_sha256 or "UNSEALED"}`
- **State:** {camp.state}
{cand_line}
{base_line}
- **Completed Sessions:** {camp.completed_sessions_count} / {len(camp.slots)}
- **Verified Sessions:** {camp.verified_sessions_count}

## Scientific Analysis Summary
- **Verdict:** `{rep.verdict}`
- **Verdict Rationale:** {rep.verdict_rationale}
- **Decisive Observations:** {rep.decisive_observations_count}
- **Candidate Decisive Accuracy:** {cand_acc_md}
- **Day-Block Bootstrap 95% CI:** {b_ci_md}
- **Economic Hurdle (1/1.85):** {rep.break_even_hurdle:.4f} (Clears: {rep.clears_economic_hurdle})
- **Paired Baseline Difference:** {paired_diff_str} (Beats Baseline: {rep.beats_paired_baseline})

> {rep.disclaimer}
"""

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(out_content, encoding="utf-8")
        console.print(f"[bold green]Report exported successfully to {output_file}[/bold green]")
    else:
        console.print(out_content)


@campaign_app.command(
    "break-blind",
    help="Irreversibly break confirmatory blinding on an active research campaign.",
)
def break_blind_command(
    campaign_id: Annotated[str, typer.Argument(help="Campaign identifier to break blinding for.")],
    reason: Annotated[
        str,
        typer.Option("--reason", "-r", help="Audited scientific rationale for breaking blinding."),
    ],
    operator: Annotated[
        str, typer.Option("--operator", help="Supervised operator identity.")
    ] = "SUPERVISED_OPERATOR",
    root: Annotated[Path | None, typer.Option("--root", help="Repository root path.")] = None,
) -> None:
    repo_root = root or Path.cwd()
    mgr = CampaignManager(root=repo_root)
    try:
        camp = mgr.break_blind(campaign_id, reason=reason, operator=operator)
        console.print(
            Panel(
                f"[bold red]CONFIRMATORY BLINDING PERMANENTLY BROKEN[/bold red]\n\n"
                f"Campaign ID: [cyan]{camp.campaign_id}[/cyan]\n"
                f"New Scientific State: [bold red]{camp.state}[/bold red]\n"
                f"Operator: {operator}\n"
                f"Reason: {reason}\n\n"
                "[yellow]This campaign has been permanently reclassified as exploratory/"
                "non-confirmatory. It cannot be returned to confirmatory status.[/yellow]",
                title="Blinding Broken (Audited)",
                border_style="red",
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Failed to break blind:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
