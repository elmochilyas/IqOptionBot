"""Manual offline prospective paper observation; no credentials or network options."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from iqoption_research.market_data.history_expansion import atomic_json
from iqoption_research.research.phase28_audit import audit_phase28, product_history_evidence
from iqoption_research.research.prospective import (
    ProspectivePolicy,
    initialize_run,
    record_prediction,
    resolve_predictions,
)
from iqoption_research.research.prospective_report import prospective_report

app = typer.Typer(help="Offline prospective paper-only observations, frozen before outcomes")


@app.command("init")
def init_command(
    run: Path = typer.Option(Path("data/phase28/prospective-001")),
    storage: Path = typer.Option(Path("data/datasets")),
    model: Path = typer.Option(Path("data/phase13/candidate_models/candidate_h1_rf")),
) -> None:
    result = initialize_run(run, model, storage, ProspectivePolicy())
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("record")
def record_command(
    run: Path = typer.Option(Path("data/phase28/prospective-001")),
    storage: Path = typer.Option(Path("data/datasets")),
) -> None:
    typer.echo(json.dumps(record_prediction(run, storage), sort_keys=True))


@app.command("resolve")
def resolve_command(
    run: Path = typer.Option(Path("data/phase28/prospective-001")),
    storage: Path = typer.Option(Path("data/datasets")),
) -> None:
    typer.echo(json.dumps(resolve_predictions(run, storage), sort_keys=True))


@app.command("report")
def report_command(
    run: Path = typer.Option(Path("data/phase28/prospective-001")),
    report: Path = typer.Option(Path("data/phase28/prospective-report.json")),
) -> None:
    if report.resolve().is_relative_to(run.resolve()) or report.suffix != ".json":
        raise typer.BadParameter("Export a JSON report outside the frozen run directory")
    result = prospective_report(run)
    report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report, result)
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("audit")
def audit_command(
    storage: Path = typer.Option(Path("data/datasets")),
    project: Path = typer.Option(Path()),
    output: Path = typer.Option(Path("data/phase28/audit")),
) -> None:
    if output.resolve().is_relative_to(storage.resolve()):
        raise typer.BadParameter("Audit output must be outside immutable source storage")
    dataset = audit_phase28(storage, project)
    products = product_history_evidence(project)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "dataset.json", dataset)
    atomic_json(output / "product_evidence.json", products)
    typer.echo(
        json.dumps({
            "verified_candles": dataset["coverage"]["unique_candles"],
            "product_classification": products["classification"],
        })
    )


@app.command("evaluate-predictions")
def evaluate_predictions_cmd(
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
    catalog: Path | None = typer.Option(
        None,
        "--catalog",
        "-c",
        help="Path to SQLite dataset catalog (catalog.db).",
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
    """Offline prospective evaluation pairing predictions with authentic candle outcomes."""
    from iqoption_research.cli.commands.timing_cmd import evaluate_predictions_command

    evaluate_predictions_command(
        evidence=evidence,
        journal=journal,
        storage=storage,
        catalog=catalog,
        session_id=session_id,
        output=output,
        json_output=json_output,
    )


@app.command("evaluate-cohort")
def evaluate_cohort_cmd(
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
    catalog: Path | None = typer.Option(
        None,
        "--catalog",
        "-c",
        help="Optional path to SQLite dataset catalog (catalog.db).",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Filter to a specific session ID.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to write research dataset, evaluation report, and manifest.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Permit overwriting existing published research artifacts.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only to stdout."),
) -> None:
    """Offline prospective cohort research dataset builder and evaluator (Phase 37)."""
    from iqoption_research.cli.commands.timing_cmd import evaluate_cohort_command

    evaluate_cohort_command(
        evidence=evidence,
        journal=journal,
        storage=storage,
        catalog=catalog,
        session_id=session_id,
        output_dir=output_dir,
        overwrite=overwrite,
        json_output=json_output,
    )


@app.command("plan-missing-labels")
def plan_missing_labels_cmd(
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
    catalog: Path | None = typer.Option(
        None,
        "--catalog",
        "-c",
        help="Optional path to SQLite dataset catalog (catalog.db).",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Filter to a specific session ID.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to write the JSON recovery plan.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Permit overwriting existing output recovery plan file.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only to stdout."),
) -> None:
    """Offline bounded recovery planner grouping truly contiguous missing candle windows."""
    from iqoption_research.cli.commands.timing_cmd import plan_missing_labels_command

    plan_missing_labels_command(
        evidence=evidence,
        journal=journal,
        storage=storage,
        catalog=catalog,
        session_id=session_id,
        output=output,
        overwrite=overwrite,
        json_output=json_output,
    )


@app.command("pilot-init")
def pilot_init_cmd(
    output: Path = typer.Option(
        Path("data/phase38/pilot_study_spec.json"),
        "--output",
        "-o",
        help="Path to write the immutable pilot study specification JSON.",
    ),
    study_id: str = typer.Option(
        "pilot-eurusd-m1-rf-001",
        "--study-id",
        help="Authoritative study identifier.",
    ),
    target_total: int = typer.Option(
        100,
        "--target-total",
        help="Total pilot target observations.",
    ),
    reference_count: int = typer.Option(
        5,
        "--reference-count",
        help="Count of historical pilot-reference observations.",
    ),
    model: Path = typer.Option(
        Path("data/phase13/candidate_models/candidate_h1_rf"),
        "--model",
        help="Path to frozen candidate model bundle.",
    ),
    sampling_rule: str = typer.Option(
        "daily_regular_market_open_1300_to_1500_utc_5_candles_per_session",
        "--sampling-rule",
        help="Pre-determined sampling protocol rule description.",
    ),
    schedule_file: Path | None = typer.Option(
        None,
        "--schedule-file",
        help="Optional path to machine-readable sampling schedule JSON.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only to stdout."),
) -> None:
    """Initialize an immutable prospective pilot study specification (Phase 38)."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from iqoption_research.research.prospective_pilot import (
        ProspectivePilotError,
        initialize_pilot_study,
    )

    console = Console()
    try:
        if schedule_file is not None and schedule_file.is_file():
            sched = json.loads(schedule_file.read_text(encoding="utf-8"))
        else:
            sched = {
                "schedule_type": "RECURRING_WEEKDAY_TIME",
                "days_of_week": [0, 1, 2, 3, 4],
                "start_time_utc": "13:00:00",
                "end_time_utc": "15:00:00",
            }
        spec = initialize_pilot_study(
            output,
            study_id=study_id,
            target_total=target_total,
            baseline_reference_count=reference_count,
            model_path=model,
            sampling_rule_description=sampling_rule,
            sampling_schedule=sched,
        )
    except ProspectivePilotError as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc), "status": "FAILED"}, indent=2))
        else:
            console.print(f"[bold red]Pilot Initialization Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(spec.to_dict(), indent=2))
        return

    console.print(
        Panel(
            "[bold cyan]Prospective Evidence Pilot Study Initialized (Phase 38)[/bold cyan]\n"
            "Immutable study specification registered offline before new observations occur.\n"
            "[yellow]Authority: NONE | Disposition: NO_DEMONSTRATED_EDGE | Target: 100[/yellow]",
            title="Pilot Study Registration",
            border_style="cyan",
        )
    )

    t = Table(title="Study Specification Invariants", border_style="blue")
    t.add_column("Property", style="bold yellow")
    t.add_column("Value", style="white")

    t.add_row("Study ID", spec.study_id)
    t.add_row(
        "Specification Digest",
        f"[bold green]{spec.study_specification_sha256[:16]}...[/bold green]",
    )
    t.add_row("Symbol & Instrument", f"{spec.symbol} ({spec.instrument_type})")
    t.add_row("Timeframe", f"{spec.candle_size_seconds}s")
    t.add_row("Model Manifest Digest", f"{spec.model_manifest_sha256[:16]}...")
    t.add_row("Analytical Threshold", f"{spec.threshold:.2f}")
    t.add_row("Pilot Target Total", str(spec.pilot_target_total))
    t.add_row("Baseline Reference Count", str(spec.baseline_reference_count))
    t.add_row("Additional Required", str(spec.pilot_target_total - spec.baseline_reference_count))
    t.add_row("Output Location", str(output.resolve()))
    console.print(t)


@app.command("pilot-status")
def pilot_status_cmd(
    study: Path = typer.Option(
        Path("data/phase38/pilot_study_spec.json"),
        "--study",
        help="Path to pilot study specification JSON.",
    ),
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
        help="Optional path to prospective SQLite journal database (observations.db).",
    ),
    storage: Path = typer.Option(
        Path("data/datasets"),
        "--storage",
        "-s",
        help="Path to historical authentic catalog storage.",
    ),
    catalog: Path | None = typer.Option(
        None,
        "--catalog",
        "-c",
        help="Optional path to SQLite dataset catalog (catalog.db).",
    ),
    stop_reason: str | None = typer.Option(
        None,
        "--stop-reason",
        help="Record explicit operator cancellation or stop reason.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only to stdout."),
) -> None:
    """Audit operator-supervised prospective pilot progress against target (Phase 38)."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from iqoption_research.research.prospective_pilot import (
        PILOT_TARGET_NOTICE,
        ProspectivePilotError,
        load_and_verify_study_specification,
        reconcile_pilot_sessions,
    )

    console = Console()
    study_path = study.resolve()
    if not study_path.is_file():
        msg = f"Study specification file not found: {study_path}. Run 'pilot-init' first."
        if json_output:
            typer.echo(json.dumps({"error": msg, "status": "FAILED"}, indent=2))
        else:
            console.print(f"[bold red]Error:[/bold red] {msg}")
        raise typer.Exit(code=1)

    try:
        spec = load_and_verify_study_specification(study_path)
        paths = evidence or [Path("data/phase34/operator_supervised_live")]
        storage_dir = (
            storage
            if (isinstance(storage, Path) and storage.is_dir())
            else (Path("data/datasets") if Path("data/datasets").is_dir() else None)
        )
        cat_path = catalog if isinstance(catalog, Path) else None
        j_path = journal if isinstance(journal, Path) else None

        status_rep, _, _ = reconcile_pilot_sessions(
            spec,
            paths,
            journal_path=j_path,
            storage_dir=storage_dir,
            catalog_path=cat_path,
            stop_reason=stop_reason,
        )
    except ProspectivePilotError as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc), "status": "FAILED"}, indent=2))
        else:
            console.print(f"[bold red]Pilot Status Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(status_rep.to_dict(), indent=2))
        return

    console.print(
        Panel(
            "[bold cyan]Prospective Evidence Pilot Status (Phase 38)[/bold cyan]\n"
            "Operator-supervised offline study tracking frozen model prospective forecasts.\n"
            "[yellow]Authority: NONE | Disposition: NO_DEMONSTRATED_EDGE | Network: ZERO[/yellow]",
            title="Pilot Progress Audit",
            border_style="cyan",
        )
    )

    overview_table = Table(title="Pilot Progress Overview", border_style="blue")
    overview_table.add_column("Metric", style="bold yellow")
    overview_table.add_column("Value", style="white")

    overview_table.add_row("Study ID", status_rep.study_id)
    overview_table.add_row(
        "Study Specification SHA-256", f"[dim]{status_rep.study_specification_sha256[:16]}...[/dim]"
    )
    st_color = (
        "green"
        if status_rep.pilot_status == "TARGET_REACHED"
        else ("red" if "STOPPED" in status_rep.pilot_status else "yellow")
    )
    overview_table.add_row("Pilot Status", f"[{st_color}]{status_rep.pilot_status}[/{st_color}]")
    overview_table.add_row("Pilot Target Total", str(status_rep.pilot_target_total))
    overview_table.add_row(
        "Total Committed Predictions",
        f"[bold]{status_rep.total_unique_committed_predictions}[/bold]",
    )
    overview_table.add_row(
        "  • Historical Reference Observations", str(status_rep.reference_observations)
    )
    overview_table.add_row(
        "  • New Unique Prospective Predictions", str(status_rep.new_unique_committed_predictions)
    )
    overview_table.add_row(
        "Remaining Observations to Target",
        f"[bold yellow]{status_rep.remaining_to_target}[/bold yellow]"
        if status_rep.remaining_to_target > 0
        else "[bold green]TARGET REACHED[/bold green]",
    )
    console.print(overview_table)

    recon_table = Table(title="Label Coverage & Session Reconciliation", border_style="green")
    recon_table.add_column("Property", style="bold green")
    recon_table.add_column("Count / Rate", justify="right")

    recon_table.add_row(
        "Verified Labeled Predictions", f"[green]{status_rep.verified_labeled_predictions}[/green]"
    )
    recon_table.add_row(
        "Unlabeled (Missing) Predictions",
        f"[bold red]{status_rep.unlabeled_predictions}[/bold red]"
        if status_rep.unlabeled_predictions > 0
        else "[green]0[/green]",
    )
    recon_table.add_row("Signal Coverage Rate", f"{status_rep.coverage_rate * 100:.1f}%")
    recon_table.add_row("Abstention Rate", f"{status_rep.abstention_rate * 100:.1f}%")
    recon_table.add_row("Rejected Observations Count", str(status_rep.rejected_observations))
    recon_table.add_row("Verified Sessions Count", str(len(status_rep.verified_sessions)))
    recon_table.add_row("Incomplete Sessions Count", str(len(status_rep.incomplete_sessions)))
    recon_table.add_row("Rejected Sessions Count", str(len(status_rep.rejected_sessions)))
    console.print(recon_table)

    prob = status_rep.probabilistic_metrics
    if prob.get("status") != "NO_LABELED_PREDICTIONS":
        metrics_table = Table(
            title="Accumulated Probabilistic Metrics (Labeled Sample)", border_style="magenta"
        )
        metrics_table.add_column("Metric", style="bold magenta")
        metrics_table.add_column("Value", justify="right")

        metrics_table.add_row("Brier Score", f"{prob.get('brier_score', 0.0):.6f}")
        bss = prob.get("brier_skill_score_against_0_50_baseline")
        bss_str = f"{bss:.4f}" if bss is not None else "N/A"
        metrics_table.add_row("Brier Skill Score (vs 0.50 baseline)", bss_str)
        metrics_table.add_row("Log Loss", f"{prob.get('log_loss', 0.0):.6f}")
        metrics_table.add_row(
            "Binary Target Accuracy", f"{prob.get('binary_target_accuracy', 0.0) * 100:.1f}%"
        )
        metrics_table.add_row(
            "Spot Directional Accuracy",
            f"{prob.get('directional_accuracy_overall', 0.0) * 100:.1f}%",
        )
        console.print(metrics_table)

    if status_rep.missing_label_windows:
        win_table = Table(
            title="Contiguous Missing Windows Requiring Historical Recovery", border_style="yellow"
        )
        win_table.add_column("#", justify="center", style="bold")
        win_table.add_column("Opening UTC", style="cyan")
        win_table.add_column("Closing UTC", style="cyan")
        win_table.add_column("Candles", justify="right", style="bold red")
        win_table.add_column("Suggested Operator Command", style="dim")

        for idx, w in enumerate(status_rep.missing_label_windows, 1):
            win_table.add_row(
                str(idx),
                w["from_utc"],
                w["to_utc"],
                str(w["missing_candle_count"]),
                w["suggested_command"],
            )
        console.print(win_table)

    rd = status_rep.readiness_assessment
    console.print(
        Panel(
            f"[bold]Readiness Assessment:[/bold] {rd.get('summary', '')}\n"
            f"[bold]Operational Recommendation:[/bold] {rd.get('recommendation', '')}\n\n"
            f"[dim]{PILOT_TARGET_NOTICE}[/dim]",
            title="Readiness & Scientific Limitations",
            border_style="yellow",
        )
    )


@app.command("pilot-checkpoint")
def pilot_checkpoint_cmd(
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory to write the immutable pilot checkpoint snapshot.",
    ),
    study: Path = typer.Option(
        Path("data/phase38/pilot_study_spec.json"),
        "--study",
        help="Path to pilot study specification JSON.",
    ),
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
        help="Optional path to prospective SQLite journal database (observations.db).",
    ),
    storage: Path = typer.Option(
        Path("data/datasets"),
        "--storage",
        "-s",
        help="Path to historical authentic catalog storage.",
    ),
    catalog: Path | None = typer.Option(
        None,
        "--catalog",
        "-c",
        help="Optional path to SQLite dataset catalog (catalog.db).",
    ),
    checkpoint_id: str | None = typer.Option(
        None,
        "--checkpoint-id",
        help="Optional custom checkpoint identifier.",
    ),
    stop_reason: str | None = typer.Option(
        None,
        "--stop-reason",
        help="Record explicit operator cancellation or stop reason.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only to stdout."),
) -> None:
    """Create an immutable, reproducible pilot checkpoint snapshot (Phase 38)."""
    from rich.console import Console
    from rich.panel import Panel

    from iqoption_research.research.prospective_cohort import (
        build_integrity_manifest,
        build_research_dataset,
    )
    from iqoption_research.research.prospective_eval import (
        CohortIdentity,
        compute_prospective_metrics,
    )
    from iqoption_research.research.prospective_pilot import (
        ProspectivePilotError,
        load_and_verify_study_specification,
        publish_pilot_checkpoint,
        reconcile_pilot_sessions,
    )

    console = Console()
    study_path = study.resolve()
    if not study_path.is_file():
        msg = f"Study specification file not found: {study_path}. Run 'pilot-init' first."
        if json_output:
            typer.echo(json.dumps({"error": msg, "status": "FAILED"}, indent=2))
        else:
            console.print(f"[bold red]Error:[/bold red] {msg}")
        raise typer.Exit(code=1)

    try:
        spec = load_and_verify_study_specification(study_path)
        paths = evidence or [Path("data/phase34/operator_supervised_live")]
        storage_dir = (
            storage
            if (isinstance(storage, Path) and storage.is_dir())
            else (Path("data/datasets") if Path("data/datasets").is_dir() else None)
        )
        cat_path = catalog if isinstance(catalog, Path) else None
        j_path = journal if isinstance(journal, Path) else None

        recon_res = reconcile_pilot_sessions(
            spec,
            paths,
            journal_path=j_path,
            storage_dir=storage_dir,
            catalog_path=cat_path,
            stop_reason=stop_reason,
        )

        status_rep = recon_res.status
        pairs = recon_res.pairs
        source_fingerprints = recon_res.source_fingerprints
        audit = recon_res.evidence_audit

        cohort = CohortIdentity(
            sessions=list(status_rep.verified_sessions),
            model_manifest_sha256=spec.model_manifest_sha256,
            policy_version=spec.policy_version,
            symbol=spec.symbol,
            candle_size_seconds=spec.candle_size_seconds,
        )

        report = compute_prospective_metrics(
            pairs,
            cohort,
            audit_summary=audit,
            source_file_fingerprints=source_fingerprints,
        )

        dataset = build_research_dataset(
            pairs,
            cohort,
            source_file_fingerprints=source_fingerprints,
        )

        manifest = build_integrity_manifest(
            dataset,
            report,
            source_fingerprints,
            model_manifest_sha256=spec.model_manifest_sha256,
        )

        saved = publish_pilot_checkpoint(
            spec,
            status_rep,
            dataset,
            report,
            manifest,
            output_dir,
            checkpoint_id=checkpoint_id,
        )
    except (ProspectivePilotError, FileExistsError) as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc), "status": "FAILED"}, indent=2))
        else:
            console.print(f"[bold red]Pilot Checkpoint Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        res = {
            "status": "PUBLISHED",
            "checkpoint_dir": str(output_dir.resolve()),
            "artifacts": {k: str(p) for k, p in saved.items()},
            "study_id": spec.study_id,
            "pilot_status": status_rep.pilot_status,
            "total_unique_committed_predictions": status_rep.total_unique_committed_predictions,
            "verified_labeled_predictions": status_rep.verified_labeled_predictions,
            "unlabeled_predictions": status_rep.unlabeled_predictions,
        }
        typer.echo(json.dumps(res, indent=2))
        return

    console.print(
        Panel(
            f"[bold green]Pilot Checkpoint Published Successfully[/bold green]\n"
            f"Location: {output_dir.resolve()}\n"
            f"Study ID: {spec.study_id}\n"
            f"Status: {status_rep.pilot_status} "
            f"({status_rep.total_unique_committed_predictions}/{spec.pilot_target_total} "
            f"observations)\n"
            f"[yellow]Authority: NONE | Disposition: NO_DEMONSTRATED_EDGE[/yellow]",
            title="Checkpoint Publication",
            border_style="green",
        )
    )
    for k, p in saved.items():
        console.print(f"  • {k}: {p.name}")


@app.command("pilot-verify")
def pilot_verify_cmd(
    checkpoint_dir: Path = typer.Option(
        ...,
        "--checkpoint-dir",
        "-c",
        help="Directory of published pilot checkpoint to verify.",
    ),
    study: Path | None = typer.Option(
        None,
        "--study",
        help="Optional path to expected study specification JSON to verify bindings.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON only to stdout."),
) -> None:
    """Offline cryptographic verification of a published pilot checkpoint (Phase 38)."""
    from rich.console import Console
    from rich.panel import Panel

    from iqoption_research.research.prospective_pilot import (
        load_and_verify_study_specification,
        verify_pilot_checkpoint,
    )

    console = Console()
    expected_spec = None
    if study is not None:
        sp = study.resolve()
        if not sp.is_file():
            msg = f"Expected study specification not found: {sp}"
            if json_output:
                typer.echo(json.dumps({"valid": False, "error": msg}, indent=2))
            else:
                console.print(f"[bold red]Error:[/bold red] {msg}")
            raise typer.Exit(code=1)
        expected_spec = load_and_verify_study_specification(sp)

    is_valid, msg = verify_pilot_checkpoint(
        checkpoint_dir,
        expected_study_spec=expected_spec,
    )

    if not is_valid:
        if json_output:
            typer.echo(
                json.dumps(
                    {"valid": False, "error": msg, "checkpoint_dir": str(checkpoint_dir.resolve())},
                    indent=2,
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]Verification FAILED[/bold red]\n{msg}\n"
                    f"Checkpoint: {checkpoint_dir.resolve()}",
                    title="Pilot Checkpoint Verification",
                    border_style="red",
                )
            )
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(
            json.dumps(
                {"valid": True, "message": msg, "checkpoint_dir": str(checkpoint_dir.resolve())},
                indent=2,
            )
        )
        return

    console.print(
        Panel(
            f"[bold green]Verification PASSED[/bold green]\n{msg}\n"
            f"Checkpoint: {checkpoint_dir.resolve()}",
            title="Pilot Checkpoint Verification",
            border_style="green",
        )
    )
