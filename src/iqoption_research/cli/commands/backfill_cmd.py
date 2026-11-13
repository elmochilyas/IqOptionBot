"""CLI command for Phase 9 authentic historical data backfilling.

Exposes ``bot data backfill`` with intuitive options for symbol, size, count,
date ranges, pagination, resume, and status reporting.  Offline by default;
``--live`` requires explicit opt-in for read-only broker access.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from iqoption_research.cli.output import (
    print_error,
    print_header,
    print_success,
    print_warning,
)
from iqoption_research.market_data import DataAcquisitionError
from iqoption_research.market_data.collector import (
    BackfillCheckpoint,
    launch_backfill,
)

app = typer.Typer(help="Historical market-data backfill and management")


@app.command(name="backfill")
def backfill_command(
    instrument: str = typer.Option(
        "EURUSD", "--symbol", help="Currency pair symbol (e.g. EURUSD, EURUSD-OTC)"
    ),
    count: int = typer.Option(
        1000, "--count", help="Number of candles to collect"
    ),
    size: int = typer.Option(
        60, "--size", help="Candle size in seconds (60, 300, 900, etc.)"
    ),
    from_date: str | None = typer.Option(
        None, "--from", help="Start date (ISO 8601 UTC), e.g. 2026-01-15T10:30:00Z"
    ),
    to_date: str | None = typer.Option(
        None, "--to", help="End date (ISO 8601 UTC), e.g. 2026-01-20T10:30:00Z"
    ),
    page_size: int = typer.Option(
        500, "--page-size", help="Number of candles per paginated fetch"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from last checkpoint if available"
    ),
    live: bool = typer.Option(
        False, "--live", help="Connect to broker for read-only data retrieval"
    ),
    yes_i_understand: bool = typer.Option(
        False, "--yes-i-understand", help="Explicit opt-in for --live mode"
    ),
    output_dir: str = typer.Option(
        "data/datasets", "--output-dir", help="Directory for checkpoint and storage"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit with code 1 if dataset is not research-grade"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON-only output"
    ),
) -> None:
    """Start an authentic historical data backfill session."""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    # Offline by default; --live requires explicit confirmation
    if live and not yes_i_understand:
        raise typer.BadParameter(
            "The --live flag requires explicit --yes-i-understand confirmation."
        )

    # Normalize instrument symbol
    from iqoption_research.market_data.instruments import normalize_symbol

    canonical = normalize_symbol(instrument)

    # Launch the backfill job
    try:
        launch_backfill(
            instrument=canonical,
            count=count,
            size=size,
            from_date=from_date,
            to_date=to_date,
            page_size=page_size,
            resume=resume,
            live=live,
            yes_i_understand=yes_i_understand,
            output_dir=output_dir,
            strict=strict,
        )
    except DataAcquisitionError as exc:
        print_error(f"Data acquisition error: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Backfill error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="backfill-status")
def backfill_status_command(
    job_id: str = typer.Option(
        ..., "--job-id", help="The job ID to check status for"
    ),
    output_dir: str = typer.Option(
        "data/datasets", "--output-dir", help="Directory where checkpoints are stored"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON-only output"
    ),
) -> None:
    """Check the status of a backfill job."""
    checkpoint_path = Path(output_dir) / f"backfill_checkpoint_{job_id}.json"

    if not checkpoint_path.exists():
        print_error(f"No checkpoint found for job ID: {job_id}")
        raise typer.Exit(code=1)

    try:
        data = checkpoint_path.read_text(encoding="utf-8")
        checkpoint = BackfillCheckpoint.from_dict(json.loads(data))

        if not json_output:
            print_header(f"Backfill Job Status: {job_id}")
            print_success(f"  State: {checkpoint.state}")
            print_success(f"  Instrument: {checkpoint.instrument}")
            print_success(f"  Active ID: {checkpoint.active_id}")
            print_success(f"  Timeframe: {checkpoint.timeframe_seconds}s")
            print_success(f"  Target count: {checkpoint.target_count}")
            print_success(f"  Unique candles persisted: {checkpoint.unique_candles_persisted}")
            print_success(f"  Retry count: {checkpoint.retry_count}")
            print_success(f"  Date range: {checkpoint.date_range}")

        if checkpoint.state == "completed":
            print_success("  Job completed successfully.")
        elif checkpoint.state == "partial":
            print_warning("  Job completed partially. History may be exhausted.")
        elif checkpoint.state == "failed":
            print_error("  Job failed.")
        elif checkpoint.state == "running":
            print_warning("  Job is still running.")
        elif checkpoint.state == "not_started":
            print_warning("  Job has not started yet.")

    except Exception as exc:
        print_error(f"Resume backfill error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="backfill-resume")
def backfill_resume_command(
    job_id: str = typer.Option(
        ..., "--job-id", help="The job ID to resume"
    ),
    output_dir: str = typer.Option(
        "data/datasets", "--output-dir", help="Directory where checkpoints are stored"
    ),
    live: bool = typer.Option(
        False, "--live", help="Continue with read-only broker connection"
    ),
    yes_i_understand: bool = typer.Option(
        False, "--yes-i-understand", help="Explicit opt-in for --live mode"
    ),
) -> None:
    """Resume a backfill job from its last checkpoint."""
    from iqoption_research.market_data.collector import BackfillCheckpoint

    checkpoint_path = Path(output_dir) / f"backfill_checkpoint_{job_id}.json"

    if not checkpoint_path.exists():
        print_error(f"No checkpoint found for job ID: {job_id}")
        raise typer.Exit(code=1)

    # Offline by default; --live requires explicit confirmation
    if live and not yes_i_understand:
        raise typer.BadParameter(
            "The --live flag requires explicit --yes-i-understand confirmation."
        )

    try:
        existing_checkpoint = BackfillCheckpoint.from_dict(
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
        )
    except Exception:
        print_error("Could not read checkpoint, starting fresh.")
        existing_checkpoint = None

    if existing_checkpoint is None:
        print_error(f"Could not read checkpoint for job {job_id}, starting fresh.")
        raise typer.Exit(code=1)

    # Launch resumption
    try:
        launch_backfill(
            instrument=existing_checkpoint.instrument,
            count=existing_checkpoint.target_count,
            size=existing_checkpoint.timeframe_seconds,
            from_date=existing_checkpoint.date_range[0]
            if existing_checkpoint.date_range
            else None,
            to_date=existing_checkpoint.date_range[1]
            if existing_checkpoint.date_range
            else None,
            page_size=existing_checkpoint.page_size
            if hasattr(existing_checkpoint, "page_size")
            else 500,
            resume=True,
            live=live,
            yes_i_understand=yes_i_understand,
            output_dir=output_dir,
        )
    except DataAcquisitionError as exc:
        print_error(f"Data acquisition error: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Resume backfill error: {exc}")
        raise typer.Exit(code=1) from exc
