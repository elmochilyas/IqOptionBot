"""Phase29 operator-initiated local observation and target assessment."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from iqoption_research.research.observation_report import observation_report
from iqoption_research.research.observation_session import (
    export_new_json,
    observe_session,
    register_observation_run,
)
from iqoption_research.research.target_alignment import target_audit

app = typer.Typer(help="Bounded local-catalog observation only; zero broker requests")


@app.command("audit-target")
def audit_command(
    model: Path = typer.Option(Path("data/phase13/candidate_models/candidate_h1_rf")),
    output: Path = typer.Option(...),
) -> None:
    result = target_audit(model)
    export_new_json(output, result, (model, Path("data"), Path("docs/phase28")))
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("init")
def init_command(
    run: Path = typer.Option(...),
    storage: Path = typer.Option(Path("data/datasets")),
    model: Path = typer.Option(Path("data/phase13/candidate_models/candidate_h1_rf")),
) -> None:
    typer.echo(json.dumps(register_observation_run(run, model, storage), sort_keys=True))


@app.command("session")
def session_command(
    run: Path = typer.Option(...),
    duration_seconds: int = typer.Option(..., min=1, max=3600),
    operator_supervised: bool = typer.Option(False, "--operator-supervised"),
    storage: Path = typer.Option(Path("data/datasets")),
) -> None:
    if not operator_supervised:
        raise typer.BadParameter("Explicit --operator-supervised initiation required")
    typer.echo(json.dumps(observe_session(run, storage, duration_seconds), sort_keys=True))


@app.command("report")
def report_command(
    run: Path = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    result = observation_report(run)
    export_new_json(output, result, (run, Path("data"), Path("docs/phase28")))
    typer.echo(json.dumps(result, sort_keys=True))
