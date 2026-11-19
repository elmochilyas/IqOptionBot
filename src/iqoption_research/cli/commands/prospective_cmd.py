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
        session_id=session_id,
        output=output,
        json_output=json_output,
    )
