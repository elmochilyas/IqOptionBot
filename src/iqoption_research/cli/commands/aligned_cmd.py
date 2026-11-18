"""Separately registered Phase30 research and offline prospective observations."""

from pathlib import Path

import typer

from iqoption_research.research.aligned_acquisition import seed_collection
from iqoption_research.research.aligned_observer import (
    aligned_report,
    record_aligned,
    register_prospective,
    resolve_aligned,
)
from iqoption_research.research.aligned_study import register_study, train_study, write_new_json

app = typer.Typer(help="Phase30 target-aligned DEVELOPMENT_ONLY research; no broker access")


@app.command("register")
def register_command(
    run: Path = typer.Option(...),
    storage: Path = typer.Option(Path("data/datasets")),
    project: Path = typer.Option(Path()),
) -> None:
    result = register_study(run, storage, project)
    typer.echo(f"Registered before loading prices: {result['created_at_ts']}")


@app.command("train")
def train_command(run: Path = typer.Option(...)) -> None:
    result = train_study(run)
    typer.echo(f"DEVELOPMENT_ONLY: {result['selected_development_model']}; NO_DEMONSTRATED_EDGE")


@app.command("init")
def init_command(
    run: Path = typer.Option(...),
    study: Path = typer.Option(...),
    storage: Path | None = typer.Option(None),
) -> None:
    register_prospective(run, study, storage)
    typer.echo("New immutable target-aligned prospective registration; no observations backfilled")


@app.command("seed-collection")
def seed_command(
    destination: Path = typer.Option(...),
    storage: Path = typer.Option(Path("data/datasets")),
    project: Path = typer.Option(Path()),
) -> None:
    typer.echo(str(seed_collection(project, storage, destination)))


@app.command("record")
def record_command(run: Path = typer.Option(...)) -> None:
    typer.echo(str(record_aligned(run)))


@app.command("resolve")
def resolve_command(run: Path = typer.Option(...)) -> None:
    typer.echo(str(resolve_aligned(run)))


@app.command("report")
def report_command(run: Path = typer.Option(...), output: Path = typer.Option(...)) -> None:
    if output.resolve().is_relative_to(run.resolve()) or output.resolve().is_relative_to(
        Path("data").resolve()
    ):
        raise typer.BadParameter("Export a NEW report outside frozen runs/source data")
    write_new_json(output, aligned_report(run))
    typer.echo(str(output))
