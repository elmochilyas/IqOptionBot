"""Phase 26 interactive-only historical collection and offline validation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from iqoption_research.cli.commands.backfill_cmd import _connect_live_client, app
from iqoption_research.market_data.history_expansion import (
    ExpansionLimits,
    atomic_json,
    collect_history,
    validate_history,
)


@app.command("expand-history")
def expand_history_command(
    storage: Path = typer.Option(Path("data/datasets")),
    job_id: str = typer.Option("phase26-year"),
    target_days: int = typer.Option(365, min=1, max=3660),
    max_requests: int = typer.Option(1000, min=1, max=10_000),
    max_new_candles: int = typer.Option(400_000, min=1, max=2_000_000),
    page_size: int = typer.Option(500, min=1, max=500),
    resume: bool = typer.Option(False),
    live: bool = typer.Option(False, help="Operator opt-in to interactive read-only collection"),
) -> None:
    """Inspect coverage offline, or collect with local interactive login and MFA."""
    limits = ExpansionLimits(target_days, max_requests, max_new_candles, page_size)
    limits.validate()
    coverage = validate_history(storage)
    if not live:
        typer.echo(
            json.dumps({"mode": "OFFLINE_PLAN", "limits": vars(limits), "coverage": coverage})
        )
        return
    if not coverage["unique_candles"]:
        raise typer.BadParameter("Existing verified authentic coverage is required")

    async def run() -> None:
        # False selects the existing interactive workflow; vault login is unreachable.
        broker, cleanup = await _connect_live_client(False)
        try:
            report = await collect_history(
                storage, broker, job_id=job_id, limits=limits, resume=resume
            )
            typer.echo(json.dumps(report, sort_keys=True))
        finally:
            await cleanup()

    try:
        asyncio.run(run())
    except Exception as exc:  # noqa: BLE001 -- fail closed without private transport details
        # Authentication/transport exceptions may contain private response details.
        typer.echo(f"Collection stopped ({type(exc).__name__}); inspect the local checkpoint.")
        raise typer.Exit(1) from None


@app.command("validate-history")
def validate_history_command(
    storage: Path = typer.Option(Path("data/datasets")),
    report: Path = typer.Option(Path("data/phase26/coverage.json")),
) -> None:
    """Verify immutable source checksums, conflicts, UTC windows, and session gaps offline."""
    report.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = validate_history(storage)
    except Exception as exc:  # noqa: BLE001 -- machine-readable failure, no private values
        result = {"quality_status": "REJECTED", "error_type": type(exc).__name__}
        atomic_json(report, result)
        typer.echo(json.dumps(result))
        raise typer.Exit(1) from None
    atomic_json(report, result)
    typer.echo(json.dumps(result, sort_keys=True))
