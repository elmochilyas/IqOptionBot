"""CLI commands for authentic, resumable historical data backfilling."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from iqoption_research.cli.output import print_error, print_header, print_success, print_warning
from iqoption_research.market_data.collector import (
    BackfillCheckpoint,
    BackfillResult,
    BoundaryDiagnosticResult,
    DataAcquisitionError,
    HistoricalBroker,
    HistoricalDataCollector,
    JobState,
    diagnose_exclusive_boundary,
    launch_backfill,
    load_checkpoint,
)
from iqoption_research.market_data.forward_sync import (
    ForwardState,
    ForwardSyncPlan,
    ForwardSyncResult,
    build_forward_sync_plan,
    create_forward_sync_job,
    forward_checkpoint_path,
    load_forward_checkpoint,
)
from iqoption_research.ml.phase10 import Phase10Error
from iqoption_research.research.phase11 import (
    BackwardExpansionPlan,
    audit_expansion_coverage,
    build_backward_expansion_plan,
    launch_backward_expansion,
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
    additional_broker_requests: int | None = None,
    request_before_cursor: bool = False,
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
            additional_broker_requests=additional_broker_requests,
            request_before_cursor=request_before_cursor,
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


def _emit_forward_result(result: ForwardSyncResult, *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(result.to_dict(), sort_keys=True))
        return
    print_header(f"Forward Sync: {result.job_id}")
    print_success(f"State: {result.state}")
    print_success(f"Unique candles: {result.unique_candles}")
    print_success(f"Calendar coverage: {result.calendar_coverage_days:.2f} days")
    print_success(f"Newest completed candle: {result.newest_timestamp_utc}")
    print_success(f"New records saved: {result.new_records_saved}")
    print_success(f"Duplicates skipped: {result.duplicates_skipped}")
    if result.unresolved_gap_count:
        print_warning(f"Unresolved open-market gaps: {result.unresolved_gap_count}")
    else:
        print_success("Unresolved open-market gaps: 0")


async def _run_authorized_extension(
    *,
    plan: BackwardExpansionPlan,
    job_id: str,
    output_dir: Path,
    saved_credentials: bool,
) -> BackfillResult:
    client, cleanup = await _connect_live_client(saved_credentials)
    try:
        collector = HistoricalDataCollector(storage_dir=output_dir)
        job = launch_backward_expansion(
            collector=collector,
            plan=plan,
            job_id=job_id,
            client=client,
        )
        return await job.run()
    finally:
        await cleanup()


async def _run_boundary_diagnostic(
    *,
    checkpoint: BackfillCheckpoint,
    output_dir: Path,
    probe_count: int,
    saved_credentials: bool,
) -> BoundaryDiagnosticResult:
    client, cleanup = await _connect_live_client(saved_credentials)
    try:
        collector = HistoricalDataCollector(storage_dir=output_dir)
        return await diagnose_exclusive_boundary(
            collector=collector,
            checkpoint=checkpoint,
            client=client,
            probe_count=probe_count,
        )
    finally:
        await cleanup()


async def _run_forward_sync(
    *,
    plan: ForwardSyncPlan,
    job_id: str,
    output_dir: Path,
    saved_credentials: bool,
    additional_requests: int | None,
) -> ForwardSyncResult:
    client, cleanup = await _connect_live_client(saved_credentials)
    try:
        collector = HistoricalDataCollector(storage_dir=output_dir)
        job = create_forward_sync_job(
            collector=collector,
            plan=plan,
            job_id=job_id,
            client=client,
            additional_requests=additional_requests,
        )
        return await job.run()
    finally:
        await cleanup()


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


@app.command("coverage-audit")
def coverage_audit_command(
    instrument: str = typer.Option("EURUSD", "--symbol"),
    size: int = typer.Option(60, "--size"),
    target_days: int = typer.Option(90, "--target-days", min=1),
    max_new_candles: int = typer.Option(150_000, "--max-new-candles", min=1),
    page_size: int = typer.Option(500, "--page-size", min=1, max=1000),
    max_requests: int | None = typer.Option(None, "--max-requests", min=1),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output-dir"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Audit verified coverage and print a bounded backward-extension plan offline."""
    try:
        coverage, _ = audit_expansion_coverage(
            output_dir,
            symbol=instrument,
            candle_size_seconds=size,
            target_calendar_days=target_days,
        )
        plan = build_backward_expansion_plan(
            output_dir,
            symbol=instrument,
            candle_size_seconds=size,
            target_calendar_days=target_days,
            max_new_candles=max_new_candles,
            page_size=page_size,
            max_broker_requests=max_requests,
        )
    except (DataAcquisitionError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    payload = {"coverage": coverage.to_dict(), "plan": plan.to_dict()}
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    print_header("Authentic Historical Coverage")
    print_success(f"Unique candles: {coverage.base.unique_candles}")
    print_success(f"Oldest: {coverage.oldest_utc}")
    print_success(f"Newest: {coverage.newest_utc}")
    print_success(f"Calendar span: {coverage.base.calendar_span_days:.2f} days")
    print_success(f"Scheduled closures: {coverage.scheduled_closure_count}")
    print_warning(f"Unexpected gaps: {coverage.unexpected_gap_count}")
    print_success(f"Target: {target_days} days | satisfied: {coverage.meets_target}")
    if plan.needs_collection:
        print_success(f"Collect backward to: {plan.target_oldest_utc}")
        print_success(f"New-candle cap: {plan.max_new_candles}")
        print_success(f"Broker-request cap: {plan.max_broker_requests}")


@app.command("backfill-extend")
def backfill_extend_command(
    instrument: str = typer.Option("EURUSD", "--symbol"),
    size: int = typer.Option(60, "--size"),
    target_days: int = typer.Option(120, "--target-days", min=1),
    max_new_candles: int = typer.Option(150_000, "--max-new-candles", min=1),
    page_size: int = typer.Option(500, "--page-size", min=1, max=1000),
    max_requests: int = typer.Option(400, "--max-requests", min=1),
    job_id: str = typer.Option(..., "--job-id"),
    live: bool = typer.Option(False, "--live"),
    yes_i_understand: bool = typer.Option(False, "--yes-i-understand"),
    saved_credentials: bool = typer.Option(False, "--saved"),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output-dir"),
    strict: bool = typer.Option(False, "--strict"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Extend strictly before the oldest verified candle using a bounded live job."""
    if not live:
        print_error(
            "Offline safety mode is active. Historical extension requires "
            "--live --yes-i-understand; no broker connection was attempted."
        )
        raise typer.Exit(code=2)
    if not yes_i_understand:
        raise typer.BadParameter("--live requires --yes-i-understand")
    resolved_job_id = _validate_job_id(job_id)
    try:
        plan = build_backward_expansion_plan(
            output_dir,
            symbol=instrument,
            candle_size_seconds=size,
            target_calendar_days=target_days,
            max_new_candles=max_new_candles,
            page_size=page_size,
            max_broker_requests=max_requests,
        )
        if not plan.needs_collection:
            print_success("Historical coverage already satisfies the requested target.")
            return
        result = asyncio.run(
            _run_authorized_extension(
                plan=plan,
                job_id=resolved_job_id,
                output_dir=output_dir,
                saved_credentials=saved_credentials,
            )
        )
        _emit_result(result, json_output=json_output)
        if strict and result.state != JobState.COMPLETED:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (DataAcquisitionError, ValueError) as exc:
        print_error(f"Historical extension failed: {exc}")
        raise typer.Exit(code=1) from exc


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
    if checkpoint.max_broker_requests is not None:
        print_success(
            f"Broker requests: {checkpoint.broker_requests_made}/{checkpoint.max_broker_requests}"
        )
    if checkpoint.last_error:
        print_warning(f"Last error: {checkpoint.last_error}")


@app.command("forward-plan")
def forward_plan_command(
    instrument: str = typer.Option("EURUSD", "--symbol"),
    size: int = typer.Option(60, "--size"),
    page_size: int = typer.Option(500, "--page-size", min=1, max=1000),
    max_requests: int = typer.Option(20, "--max-requests", min=1),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output-dir"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Plan an incremental forward sync entirely offline."""
    try:
        plan = build_forward_sync_plan(
            output_dir,
            symbol=instrument,
            candle_size_seconds=size,
            page_size=page_size,
            max_requests=max_requests,
        )
    except (DataAcquisitionError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(plan.to_dict(), sort_keys=True))
        return
    print_header("Incremental Forward Collection Plan")
    print_success(f"Unique candles: {plan.current_unique_candles}")
    print_success(
        "Newest verified candle: "
        f"{datetime.fromtimestamp(plan.latest_stored_to_ts, tz=UTC).isoformat()}"
    )
    print_success(
        "Latest completed boundary: "
        f"{datetime.fromtimestamp(plan.target_completed_end_ts, tz=UTC).isoformat()}"
    )
    print_success(f"Expected open-market candles: {plan.expected_open_candles}")
    print_success(f"Estimated broker requests: {plan.estimated_requests}")
    print_success(f"Request budget: {plan.max_requests}")
    if not plan.needs_broker_request:
        print_success("No broker request is needed; the market window contains no missing opens.")


@app.command("forward-sync")
def forward_sync_command(
    job_id: str = typer.Option("eurusd-forward-1m", "--job-id"),
    instrument: str = typer.Option("EURUSD", "--symbol"),
    size: int = typer.Option(60, "--size"),
    page_size: int = typer.Option(500, "--page-size", min=1, max=1000),
    max_requests: int = typer.Option(20, "--max-requests", min=1),
    additional_requests: int | None = typer.Option(
        None,
        "--additional-requests",
        min=1,
        help="Explicitly extend an exhausted in-progress cycle budget",
    ),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output-dir"),
    live: bool = typer.Option(False, "--live", help="Authorize read-only broker access"),
    yes_i_understand: bool = typer.Option(False, "--yes-i-understand"),
    saved_credentials: bool = typer.Option(False, "--saved", help="Use saved credentials"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Manually run one bounded incremental forward-collection cycle."""
    if not live:
        print_error(
            "Offline safety mode is active. Forward sync requires "
            "--live --yes-i-understand; no broker connection was attempted."
        )
        raise typer.Exit(code=2)
    if not yes_i_understand:
        raise typer.BadParameter("--live requires --yes-i-understand")
    resolved_job_id = _validate_job_id(job_id)
    try:
        plan = build_forward_sync_plan(
            output_dir,
            symbol=instrument,
            candle_size_seconds=size,
            page_size=page_size,
            max_requests=max_requests,
        )
        try:
            existing_forward = load_forward_checkpoint(output_dir, resolved_job_id)
        except DataAcquisitionError:
            existing_forward = None
        resume_pending = bool(
            existing_forward is not None
            and existing_forward.state
            in {ForwardState.RUNNING, ForwardState.INTERRUPTED, ForwardState.PARTIAL}
        )
        if not plan.needs_broker_request and not resume_pending:
            coverage, _ = audit_expansion_coverage(
                output_dir,
                symbol=instrument,
                candle_size_seconds=size,
            )
            result = ForwardSyncResult(
                job_id=resolved_job_id,
                state="completed",
                unique_candles=coverage.base.unique_candles,
                calendar_coverage_days=coverage.base.calendar_span_days,
                newest_timestamp_utc=coverage.newest_utc,
                new_records_saved=0,
                duplicates_skipped=0,
                requests_made=0,
                unresolved_gap_count=coverage.unexpected_gap_count,
                scheduled_closure_count=coverage.scheduled_closure_count,
            )
        else:
            result = asyncio.run(
                _run_forward_sync(
                    plan=plan,
                    job_id=resolved_job_id,
                    output_dir=output_dir,
                    saved_credentials=saved_credentials,
                    additional_requests=additional_requests,
                )
            )
        _emit_forward_result(result, json_output=json_output)
    except typer.Exit:
        raise
    except (DataAcquisitionError, ValueError) as exc:
        print_error(f"Forward sync failed: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("forward-status")
def forward_status_command(
    job_id: str = typer.Option("eurusd-forward-1m", "--job-id"),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output-dir"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show the durable forward checkpoint and current verified coverage offline."""
    resolved_job_id = _validate_job_id(job_id)
    try:
        checkpoint_path = forward_checkpoint_path(output_dir, resolved_job_id)
        checkpoint = (
            load_forward_checkpoint(output_dir, resolved_job_id)
            if checkpoint_path.is_file()
            else None
        )
        symbol = checkpoint.symbol if checkpoint is not None else "EURUSD"
        size = checkpoint.candle_size_seconds if checkpoint is not None else 60
        coverage, _ = audit_expansion_coverage(
            output_dir,
            symbol=symbol,
            candle_size_seconds=size,
            target_calendar_days=90,
        )
    except (DataAcquisitionError, ValueError, Phase10Error) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    payload: dict[str, Any] = {
        "checkpoint": checkpoint.to_dict() if checkpoint is not None else None,
        "coverage": coverage.to_dict(),
    }
    if checkpoint is None:
        payload["message"] = "No forward collection has been required yet"

    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    print_header(f"Forward Sync Status: {resolved_job_id}")
    if checkpoint is not None:
        print_success(f"State: {checkpoint.state}")
        print_success(f"Cycle: {checkpoint.cycle_number}")
        print_success(f"Requests: {checkpoint.requests_made}/{checkpoint.max_requests}")
        print_success(f"New records this cycle: {checkpoint.new_records_saved}")
        print_success(f"Duplicates this cycle: {checkpoint.duplicates_skipped}")
    else:
        print_success("No forward collection has been required yet")
    print_success(f"Unique candles: {coverage.base.unique_candles}")
    print_success(f"Calendar coverage: {coverage.base.calendar_span_days:.2f} days")
    print_success(f"Newest timestamp: {coverage.newest_utc}")
    if coverage.unexpected_gap_count:
        print_warning(f"Unresolved open-market gaps: {coverage.unexpected_gap_count}")
    else:
        print_success("Unresolved open-market gaps: 0")


@app.command("backfill-boundary-diagnostic")
def backfill_boundary_diagnostic_command(
    job_id: str = typer.Option(..., "--job-id"),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output-dir"),
    probe_count: int = typer.Option(500, "--probe-count", min=1, max=1000),
    live: bool = typer.Option(False, "--live", help="Authorize one read-only broker request"),
    yes_i_understand: bool = typer.Option(False, "--yes-i-understand"),
    saved_credentials: bool = typer.Option(False, "--saved", help="Use saved credentials"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only"),
) -> None:
    """Probe cursor-1 once without modifying data, catalog, or checkpoint."""
    if not live:
        print_error(
            "Offline safety mode is active. The boundary diagnostic requires "
            "--live --yes-i-understand; no broker connection was attempted."
        )
        raise typer.Exit(code=2)
    if not yes_i_understand:
        raise typer.BadParameter("--live requires --yes-i-understand")
    resolved_job_id = _validate_job_id(job_id)
    try:
        checkpoint = load_checkpoint(output_dir, resolved_job_id)
        result = asyncio.run(
            _run_boundary_diagnostic(
                checkpoint=checkpoint,
                output_dir=output_dir,
                probe_count=probe_count,
                saved_credentials=saved_credentials,
            )
        )
    except (DataAcquisitionError, ValueError) as exc:
        print_error(f"Boundary diagnostic failed: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(result.to_dict(), sort_keys=True))
        return
    print_header(f"Exclusive Boundary Diagnostic: {resolved_job_id}")
    print_success(
        "Requested endtime: "
        f"{datetime.fromtimestamp(result.requested_endtime, tz=UTC).isoformat()} "
        f"({result.requested_endtime})"
    )
    print_success(f"Broker records returned: {result.returned_count}")
    print_success(f"Validated older records: {result.valid_older_count}")
    if result.older_history_available:
        print_success("Older history is available when the inclusive boundary is excluded.")
        print_success(
            "Resume with --exclusive-cursor; this diagnostic did not persist broker data."
        )
    else:
        print_warning(
            "No older candles were returned by the exclusive probe. This supports a broker "
            "endpoint availability limit at the current boundary."
        )


@app.command("backfill-resume")
def backfill_resume_command(
    job_id: str = typer.Option(..., "--job-id"),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output-dir"),
    live: bool = typer.Option(False, "--live", help="Authorize a real read-only connection"),
    yes_i_understand: bool = typer.Option(False, "--yes-i-understand"),
    saved_credentials: bool = typer.Option(False, "--saved", help="Use saved credentials"),
    strict: bool = typer.Option(False, "--strict"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only"),
    additional_requests: int | None = typer.Option(
        None,
        "--additional-requests",
        min=1,
        help="Explicitly add a bounded request budget to this resume",
    ),
    exclusive_cursor: bool = typer.Option(
        False,
        "--exclusive-cursor",
        help="Request each next page at checkpoint cursor minus one second",
    ),
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
                additional_broker_requests=additional_requests,
                request_before_cursor=exclusive_cursor,
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
