"""CLI commands for authentic, resumable historical data backfilling."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import typer

from iqoption_research.cli.output import print_error, print_header, print_success, print_warning
from iqoption_research.market_data.collector import (
    BackfillCheckpoint,
    BackfillResult,
    DataAcquisitionError,
    HistoricalBroker,
    HistoricalDataCollector,
    JobState,
    launch_backfill,
    load_checkpoint,
)

app = typer.Typer(help="Authentic historical market-data backfill and recovery")

_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
Cleanup = Callable[[], Awaitable[None]]


def _validate_job_id(job_id: str) -> str:
    if not _JOB_ID_PATTERN.fullmatch(job_id):
        raise typer.BadParameter(
            "job ID must contain only letters, digits, underscores, or hyphens (max 64)"
        )
    return job_id


async def _connect_live_client(saved_credentials: bool) -> tuple[HistoricalBroker, Cleanup]:
    """Open an explicitly authorized live read-only session.

    Kept behind one seam so automated tests can inject a fully offline broker while
    exercising the real CLI, job, validation, storage, catalog, and checkpoint path.
    """
    from iqoption_research.auth.manager import AuthenticationManager
    from iqoption_research.auth.provider import IQOptionAuthProvider
    from iqoption_research.connectors.iqoption.client import AsyncIQOption

    auth = AuthenticationManager(provider=IQOptionAuthProvider())
    logged_in = await (auth.login_saved() if saved_credentials else auth.login_interactive())
    if not logged_in:
        raise DataAcquisitionError(auth.last_error or "Broker authentication failed.")
    session = auth.session
    if session is None or not session.is_active:
        await auth.logout()
        raise DataAcquisitionError("Broker authentication produced no active session.")

    client = AsyncIQOption(ssid=session.ssid, offline=False)
    client.assert_read_only()
    try:
        await client.connect()
    except Exception:
        await auth.logout()
        raise

    async def cleanup() -> None:
        try:
            await client.close()
        finally:
            await auth.logout()

    return client, cleanup


async def _run_authorized_job(
    *,
    job_id: str,
    instrument: str,
    count: int,
    size: int,
    from_date: str | None,
    to_date: str | None,
    page_size: int,
    output_dir: Path,
    saved_credentials: bool,
    resume: bool,
) -> BackfillResult:
    client, cleanup = await _connect_live_client(saved_credentials)
    try:
        collector = HistoricalDataCollector(storage_dir=output_dir)
        job = launch_backfill(
            collector=collector,
            job_id=job_id,
            instrument=instrument,
            timeframe_seconds=size,
            target_count=count,
            from_date=from_date,
            to_date=to_date,
            page_size=page_size,
            client=client,
            resume=resume,
        )
        return await job.run()
    finally:
        await cleanup()


def _emit_result(result: BackfillResult, *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(result.to_dict(), sort_keys=True))
        return
    print_header(f"Backfill Job: {result.job_id}")
    if result.state == JobState.COMPLETED:
        print_success(f"State: {result.state}")
    else:
        print_warning(f"State: {result.state}")
    print_success(f"Unique candles persisted: {result.unique_candles}")
    print_success(f"New candles this run: {result.newly_persisted}")
    print_success(f"Pages fetched: {result.pages_fetched}")
    print_success(f"Duplicates skipped: {result.duplicates_skipped}")
    print_success(f"Rejected broker records: {result.rejected_records}")


@app.command("backfill")
def backfill_command(
    instrument: str = typer.Option("EURUSD", "--symbol", help="Instrument (EURUSD or EURUSD-OTC)"),
    count: int = typer.Option(1000, "--count", min=1, help="Unique candle target"),
    size: int = typer.Option(60, "--size", help="Candle size in seconds"),
    from_date: str | None = typer.Option(None, "--from", help="Inclusive ISO-8601 start"),
    to_date: str | None = typer.Option(None, "--to", help="Exclusive ISO-8601 end"),
    page_size: int = typer.Option(500, "--page-size", min=1, max=1000),
    job_id: str | None = typer.Option(None, "--job-id", help="Stable recovery identifier"),
    live: bool = typer.Option(False, "--live", help="Authorize a real read-only connection"),
    yes_i_understand: bool = typer.Option(
        False,
        "--yes-i-understand",
        help="Required confirmation for broker network access",
    ),
    saved_credentials: bool = typer.Option(False, "--saved", help="Use saved credentials"),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output-dir"),
    strict: bool = typer.Option(False, "--strict", help="Fail if the target is only partial"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only"),
) -> None:
    """Start a new authentic historical backfill.

    This command is offline by default. It never substitutes simulated candles for
    authentic data; an actual connection requires both explicit live flags.
    """
    if not live:
        print_error(
            "Offline safety mode is active. Authentic backfill requires "
            "--live --yes-i-understand; no broker connection was attempted."
        )
        raise typer.Exit(code=2)
    if not yes_i_understand:
        raise typer.BadParameter("--live requires --yes-i-understand")
    resolved_job_id = _validate_job_id(job_id or uuid.uuid4().hex)
    try:
        result = asyncio.run(
            _run_authorized_job(
                job_id=resolved_job_id,
                instrument=instrument,
                count=count,
                size=size,
                from_date=from_date,
                to_date=to_date,
                page_size=page_size,
                output_dir=output_dir,
                saved_credentials=saved_credentials,
                resume=False,
            )
        )
        _emit_result(result, json_output=json_output)
        if strict and result.state != JobState.COMPLETED:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (DataAcquisitionError, ValueError) as exc:
        print_error(f"Backfill failed: {exc}")
        raise typer.Exit(code=1) from exc


def _checkpoint_payload(checkpoint: BackfillCheckpoint) -> dict[str, Any]:
    payload = checkpoint.to_dict()
    payload["remaining"] = max(0, checkpoint.target_count - checkpoint.unique_candles_persisted)
    return payload


@app.command("backfill-status")
def backfill_status_command(
    job_id: str = typer.Option(..., "--job-id"),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output-dir"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only"),
) -> None:
    """Inspect a durable backfill checkpoint without connecting to the broker."""
    resolved_job_id = _validate_job_id(job_id)
    try:
        checkpoint = load_checkpoint(output_dir, resolved_job_id)
    except DataAcquisitionError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    payload = _checkpoint_payload(checkpoint)
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    print_header(f"Backfill Job Status: {resolved_job_id}")
    print_success(f"State: {checkpoint.state}")
    print_success(f"Instrument: {checkpoint.instrument}")
    print_success(f"Timeframe: {checkpoint.timeframe_seconds}s")
    print_success(f"Target count: {checkpoint.target_count}")
    print_success(f"Unique candles persisted: {checkpoint.unique_candles_persisted}")
    print_success(f"Remaining: {payload['remaining']}")
    print_success(f"Committed partitions: {len(checkpoint.committed_dataset_ids)}")
    if checkpoint.last_error:
        print_warning(f"Last error: {checkpoint.last_error}")


@app.command("backfill-resume")
def backfill_resume_command(
    job_id: str = typer.Option(..., "--job-id"),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output-dir"),
    live: bool = typer.Option(False, "--live", help="Authorize a real read-only connection"),
    yes_i_understand: bool = typer.Option(False, "--yes-i-understand"),
    saved_credentials: bool = typer.Option(False, "--saved", help="Use saved credentials"),
    strict: bool = typer.Option(False, "--strict"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only"),
) -> None:
    """Resume an existing job from its last committed page."""
    if not live:
        print_error(
            "Offline safety mode is active. Resume requires --live --yes-i-understand; "
            "no broker connection was attempted."
        )
        raise typer.Exit(code=2)
    if not yes_i_understand:
        raise typer.BadParameter("--live requires --yes-i-understand")
    resolved_job_id = _validate_job_id(job_id)
    try:
        checkpoint = load_checkpoint(output_dir, resolved_job_id)
        result = asyncio.run(
            _run_authorized_job(
                job_id=resolved_job_id,
                instrument=checkpoint.instrument,
                count=checkpoint.target_count,
                size=checkpoint.timeframe_seconds,
                from_date=None,
                to_date=None,
                page_size=checkpoint.page_size,
                output_dir=output_dir,
                saved_credentials=saved_credentials,
                resume=True,
            )
        )
        _emit_result(result, json_output=json_output)
        if strict and result.state != JobState.COMPLETED:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (DataAcquisitionError, ValueError) as exc:
        print_error(f"Resume failed: {exc}")
        raise typer.Exit(code=1) from exc
