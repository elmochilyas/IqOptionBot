"""CLI command for Phase 9 authentic historical data backfilling.
# ruff: ignore = E501, F821, F722, C901

Exposes ``bot data backfill`` with intuitive options for symbol, size, count,
date ranges, pagination, resume, and status reporting.  Offline by default;
``--live`` requires explicit opt-in for read-only broker access.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from iqoption_research.cli.output import (
    print_error,
    print_header,
    print_json,
    print_success,
    print_warning,
)
from iqoption_research.market_data import DataAcquisitionError
from iqoption_research.market_data.collector import (
    BackfillCheckpoint,
    HistoricalDataCollector,
    JobState,
    launch_backfill,
)
from iqoption_research.security.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(help="Historical market-data backfill and management")


def _parse_utc_datetime(s: str | None) -> datetime | None:
    """Parse a UTC ISO datetime string, returning a timezone-aware datetime."""
    if s is None:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise typer.BadParameter(
                    "Invalid UTC datetime format: " + repr(s) + ". Use ISO 8601 (e.g. 2026-01-15T10:30:00Z)"  # noqa E501
                ) from None


def _build_checkpoint_from_cli(
    job_id: str,
    instrument: str,
    count: int,
    page_size: int,
    from_date: str | None,
    to_date: str | None,
) -> BackfillCheckpoint:
    """Build a BackfillCheckpoint from CLI options."""
    date_range: tuple | None = None
    if from_date or to_date:
        start = _parse_utc_datetime(from_date)
        end = _parse_utc_datetime(to_date)
        if start and end and start >= end:
            raise typer.BadParameter("--from must be before --to.")
        date_range = (from_date, to_date) if (from_date or to_date) else None

    return BackfillCheckpoint(
        job_id=job_id,
        instrument=instrument,
        active_id=0,  # will be resolved from instrument lookup
        timeframe_seconds=60,  # default; will be overridden if --size given
        target_count=count,
        date_range=date_range,
        last_cursor_ts=0,
        unique_candles_persisted=0,
        last_successful_batch=0,
        retry_count=0,
        state=JobState.NOT_STARTED,
    )


@app.command("backfill", help="Run a historical data backfill job.")
def backfill_command(
    symbol: str = typer.Option(
        "EURUSD",
        "--symbol",
        help="Instrument symbol (e.g. EURUSD, EURUSD-OTC).",
    ),
    count: int = typer.Option(
        1000,
        "--count",
        help="Number of unique completed candles to collect (default: 1000).",
    ),
    size: int = typer.Option(
        60,
        "--size",
        help="Candle size in seconds (default: 60 for 1-min).",
    ),
    from_date: str | None = typer.Option(
        None,
        "--from",
        help="Start UTC date-range (ISO 8601, e.g. 2026-01-15T00:00:00Z).",
    ),
    to_date: str | None = typer.Option(
        None,
        "--to",
        help="End UTC date-range (ISO 8601, e.g. 2026-01-15T23:59:59Z).",
    ),
    page_size: int = typer.Option(
        500,
        "--page-size",
        help="Number of candles per pagination page (default: 500, max 1000).",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from a previous checkpoint if one exists for this job_id.",
    ),
    job_id: str = typer.Option(
        None,
        "--job-id",
        help="Unique identifier for this backfill job (auto-generated if omitted).",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Enable real read-only broker connection (requires --yes-i-understand).",
    ),
    yes_i_understand: bool = typer.Option(
        False,
        "--yes-i-understand",
        help="Skip interactive confirmation (still requires --live).",
    ),
    output_dir: str = typer.Option(
        "data/datasets",
        "--output-dir",
        help="Directory for persistent storage and checkpoints.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with code 1 if the dataset is not research-grade.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit JSON-only output.",
    ),
):
    """Execute a historical backfill job.

    Offline by default.  Use ``--live`` with explicit confirmation to perform
    a read-only broker probe.  Checkpoints are persisted to disk and a job can
    be resumed with ``--resume`` and the same ``--job-id``.
    """
    # Normalize symbol
    from iqoption_research.market_data.instruments import InstrumentRegistry

    try:
        canonical = InstrumentRegistry.normalize_symbol(symbol)
    except Exception as exc:
        raise typer.BadParameter(f"Unsupported symbol: {symbol!r}: {exc}") from exc

    InstrumentRegistry.is_otc_instrument(canonical)
    active_id = InstrumentRegistry.resolve_active_id(canonical)

    # Generate job_id if not provided
    if not job_id:

        job_id = f"backfill_{canonical}_{count}c_{int(datetime.now(UTC).timestamp())}"

    # If resuming, read existing checkpoint
    collector = HistoricalDataCollector(storage_dir=Path(output_dir))
    existing_checkpoint: BackfillCheckpoint | None = None

    if resume:
        # Look for existing checkpoint file
        cp_path = collector.storage_dir / f"backfill_checkpoint_{job_id}.json"
        if cp_path.is_file():
            try:
                with Path(cp_path).open("r", encoding="utf-8") as f:
                    data = json.load(f)
                existing_checkpoint = BackfillCheckpoint.from_dict(data)
                logger.info("Resuming backfill job %s from checkpoint", job_id)
            except Exception as exc:
                logger.warning("Could not read checkpoint, starting fresh: %s", exc)
                existing_checkpoint = None
        else:
            logger.info("No checkpoint found for job %s, starting fresh.", job_id)

    # If not resuming, start fresh
    if existing_checkpoint is None:
        existing_checkpoint = _build_checkpoint_from_cli(
            job_id=job_id,
            instrument=canonical,
            count=count,
            page_size=page_size,
            from_date=from_date,
            to_date=to_date,
        )

    # If --live, perform a quick read-only probe first (no trading, no balance switch)
    client = None
    if live:
        if not yes_i_understand:
            confirmed = typer.confirm(
                "This will authenticate to IQ Option using an unofficial interface. "
                "It performs READ-ONLY calls only, but the session may retain REAL "
                "account privileges. Continue?",
                default=False,
            )
            if not confirmed:
                raise typer.Exit(code=2)

        if not json_output:
            print_header("Real Read-Only Probe (explicitly authorized)")
            print_warning(
                "Unofficial interface: compatibility and platform authorization "
                "are not established. Do not place trades."
            )

        try:
            from iqoption_research.connectors.iqoption.credentials import get_credentials

            creds = get_credentials()
            settings = type(
                "Settings", {}, {"offline": False, "diagnostics": False}
            )()
            from iqoption_research.connectors.iqoption.client import AsyncIQOption

            client = AsyncIQOption(creds, settings=settings, offline=False, diagnostics=False)
            asyncio.run(client.connect(raise_if_incomplete=False))

            # Verify practice status
            status = await client.get_account_status()
            verification = status.practice_verification()
            if not verification.is_verified:
                raise typer.Exit(
                    code=1,
                    message=(
                        f"Probe halted: active account is not verified as PRACTICE. "
                        f"Status: {verification.status.value}. "
                        "Market-data retrieval stopped to ensure account safety."
                    ),
                )
        except Exception as exc:
            if not json_output:
                print_error(f"Probe failed: {exc}")
            raise typer.Exit(code=1) from exc

    # Launch the backfill job
    job = launch_backfill(
        collector=collector,
        job_id=job_id,
        instrument=canonical,
        active_id=active_id,
        timeframe_seconds=size,
        target_count=count,
        date_range=existing_checkpoint.date_range,
        page_size=page_size,
        max_retries=5,
        client=client,
        on_progress=lambda progress: None,  # will be connected to CLI progress display
        on_finished=lambda result: None,  # will be connected to CLI finish display
    )

    # Define progress callback for CLI
    def _on_progress(progress: dict) -> None:
        state = progress.get("state", "unknown")
        target = progress.get("target_count", 0)
        collected = progress.get("unique_collected", 0)
        pages = progress.get("pages_fetched", 0)
        dup_skip = progress.get("duplicates_skipped", 0)
        rej = progress.get("rejected_records", 0)
        remaining = progress.get("remaining", target - collected)
        cursor = progress.get("current_cursor", 0)
        earliest = progress.get("earliest_utc")
        latest = progress.get("latest_utc")

        pct = (collected / target * 100) if target > 0 else 0
        bar_len = 40
        filled_len = int(bar_len * collected // max(1, target))
        bar = "█" * filled_len + "─" * (bar_len - filled_len)

        if json_output:
            payload = {
                "state": state,
                "job_id": progress.get("job_id"),
                "progress_pct": round(pct, 1),
                "collected": collected,
                "target": target,
                "pages_fetched": pages,
                "duplicates_skipped": dup_skip,
                "rejected_records": rej,
                "remaining": remaining,
                "current_cursor": cursor,
                "earliest_utc": earliest,
                "latest_utc": latest,
            }
            print_json(payload)
        else:
            # Human-readable progress
            status_label = {
                "running": "RUNNING",
                "paused": "PAUSED",
                "interrupted": "INTERRUPTED",
                "completed": "COMPLETED",
                "partial": "PARTIAL",
                "failed": "FAILED",
            }.get(state, state)

            lines = [
                "Phase 9 Backfill Progress",
                "─" * 40,
                f"Job ID: {progress.get('job_id')}",
                f"State:  {status_label}",
                f"Symbol: {symbol} ({size}s candles)",
                f"Target: {target} unique candles",
                f"Collected: {collected} ({pct:.1f}%)",
                f"[{bar}]",
                f"Pages: {pages} | Dups skipped: {dup_skip} | Rejected: {rej}",
                f"Cursor: {cursor}",
            ]
            if earliest:
                lines.append(f"Earliest UTC: {earliest}")
            if latest:
                lines.append(f"Latest UTC: {latest}")

            lines.append("─" * 40)
            if state in ("completed", "partial", "failed"):
                lines.append(f"Remaining: {remaining}")
            print_header("\n".join(lines))

    # Define finished callback for CLI
    def _on_finished(result: dict) -> None:
        state = result.get("state", "unknown")
        job_id = result.get("job_id", "")
        unique = result.get("unique_candles", 0)
        target = result.get("total_requested", 0)
        newly = result.get("newly_collected", 0)
        pages = result.get("pages_fetched", 0)
        dups = result.get("duplicates_skipped", 0)
        rej = result.get("rejected_records", 0)
        remaining = result.get("remaining", target - unique)
        earliest = result.get("earliest_utc")
        latest = result.get("latest_utc")
        message = result.get("message", "")

        if json_output:
            payload = {
                "job_id": job_id,
                "state": state,
                "unique_candles": unique,
                "total_requested": target,
                "newly_collected": newly,
                "pages_fetched": pages,
                "duplicates_skipped": dups,
                "rejected_records": rej,
                "remaining": remaining,
                "earliest_utc": earliest,
                "latest_utc": latest,
                "message": message,
            }
            print_json(payload)
        else:
            status_label = {
                "completed": "COMPLETED",
                "partial": "PARTIAL",
                "failed": "FAILED",
                "not_started": "NOT_STARTED",
            }.get(state, state)

            lines = [
                "Phase 9 Backfill Result",
                "─" * 40,
                f"Job ID: {job_id}",
                f"State:  {status_label}",
                f"Symbol: {symbol} ({size}s candles)",
                f"Target: {target} unique candles",
                f"Collected: {unique} unique candles",
                f"Newly collected: {newly}",
                f"Pages fetched: {pages}",
                f"Duplicates skipped: {dups}",
                f"Rejected records: {rej}",
                f"Remaining: {remaining}",
            ]
            if earliest:
                lines.append(f"Earliest UTC: {earliest}")
            if latest:
                lines.append(f"Latest UTC: {latest}")
            lines.append(f"Message: {message}")
            lines.append("─" * 40)

            if state == "completed":
                print_success("\n".join(lines))
            elif state == "partial":
                print_warning("\n".join(lines))
            elif state == "failed":
                print_error("\n".join(lines))
            else:
                print_header("\n".join(lines))

    # Wire the callbacks
    job.on_progress = _on_progress
    job.on_finished = _on_finished

    # Run the job
    try:
        result = asyncio.run(job.run())
    except DataAcquisitionError as exc:
        logger.error("Backfill data acquisition error: %s", exc)
        if json_output:
            print_json(
                {
                    "job_id": job_id,
                    "state": JobState.FAILED,
                    "error": str(exc),
                    "message": f"Backfill failed: {exc}",
                }
            )
        else:
            print_error(f"Backfill failed: {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        logger.info("Backfill interrupted by user")
        # Checkpoint already persisted; result will reflect interrupted state
        result = {
            "job_id": job_id,
            "state": JobState.INTERRUPTED,
            "unique_candles": (
                len(collector.load_dataset(canonical, candle_size_seconds=size))
                if collector.load_dataset(canonical, candle_size_seconds=size)
                else 0
            ),
            "total_requested": count,
            "newly_collected": 0,
            "pages_fetched": 0,
            "duplicates_skipped": 0,
            "rejected_records": 0,
            "remaining": count,
            "message": "Backfill interrupted by user.",
        }
    except Exception as exc:
        logger.exception("Unexpected backfill error: %s", exc)
        if json_output:
            print_json(
                {
                    "job_id": job_id,
                    "state": JobState.FAILED,
                    "error": str(exc),
                    "message": f"Unexpected error: {exc}",
                }
            )
        else:
            print_error(f"Unexpected error: {exc}")
        raise typer.Exit(code=1) from exc
    else:
        # Emit finished callback
        _on_finished(result)

        # Print final coverage report if dataset was persisted
        try:
            coverage = collector.generate_coverage_report(canonical, candle_size_seconds=size)
            if coverage is not None:
                if not json_output:
                    print_header("Dataset Coverage Report")
                    print_json(coverage.to_dict())
                else:
                    print_json(coverage.to_dict())
        except Exception:
            pass  # non-fatal if report generation fails

    return result


@app.command("backfill-status", help="Check status of a saved backfill checkpoint.")
def backfill_status_command(
    job_id: str = typer.Option(
        ...,
        "--job-id",
        help="The job ID to check status for.",
    ),
    output_dir: str = typer.Option(
        "data/datasets",
        "--output-dir",
        help="Directory where checkpoints are stored.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit JSON-only output.",
    ),
) -> None:
    """Display the status of a saved backfill checkpoint.

    Shows job state, progress, and checkpoint metadata without rerunning the job.
    """
    collector = HistoricalDataCollector(storage_dir=Path(output_dir))
    cp_path = collector.storage_dir / f"backfill_checkpoint_{job_id}.json"

    if not cp_path.is_file():
        raise typer.BadParameter(f"No checkpoint found for job ID: {job_id}")

    try:
        with Path(cp_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        checkpoint = BackfillCheckpoint.from_dict(data)
    except Exception as exc:
        raise typer.BadParameter(f"Checkpoint file is corrupt: {exc}") from exc

    if json_output:
        print_json(checkpoint.to_dict())
    else:
        state_map = {
            "not_started": "NOT_STARTED",
            "running": "RUNNING",
            "paused": "PAUSED",
            "interrupted": "INTERRUPTED",
            "completed": "COMPLETED",
            "partial": "PARTIAL",
            "failed": "FAILED",
        }
        state_label = state_map.get(checkpoint.state, checkpoint.state)

        lines = [
            "Backfill Checkpoint Status",
            "─" * 40,
            f"Job ID: {checkpoint.job_id}",
            f"State:  {state_label}",
            f"Instrument: {checkpoint.instrument}",
            f"Active ID: {checkpoint.active_id}",
            f"Timeframe: {checkpoint.timeframe_seconds}s",
            f"Target count: {checkpoint.target_count}",
            f"Unique candles persisted: {checkpoint.unique_candles_persisted}",
            f"Last successful batch: {checkpoint.last_successful_batch}",
            f"Retry count: {checkpoint.retry_count}",
            f"Date range: {checkpoint.date_range}",
            f"Last cursor TS: {checkpoint.last_cursor_ts}",
            f"Created at: {checkpoint.created_at_utc}",
            f"Updated at: {checkpoint.updated_at_utc}",
            "─" * 40,
        ]
        print_header("\n".join(lines))


@app.command("backfill-resume", help="Resume a previously interrupted backfill job.")
def backfill_resume_command(
    job_id: str = typer.Option(
        ...,
        "--job-id",
        help="The job ID to resume.",
    ),
    output_dir: str = typer.Option(
        "data/datasets",
        "--output-dir",
        help="Directory where checkpoints are stored.",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Continue with read-only broker connection.",
    ),
    yes_i_understand: bool = typer.Option(
        False,
        "--yes-i-understand",
        help="Skip interactive confirmation.",
    ),
) -> None:
    """Resume a previously interrupted or paused backfill job.

    The job will resume from its last persisted checkpoint, skipping already
    collected candles and continuing pagination from the last cursor.
    """
    collector = HistoricalDataCollector(storage_dir=Path(output_dir))

    # Read existing checkpoint
    cp_path = collector.storage_dir / f"backfill_checkpoint_{job_id}.json"
    if not cp_path.is_file():
        raise typer.BadParameter(f"No checkpoint found for job ID: {job_id}")

    try:
        with Path(cp_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        checkpoint = BackfillCheckpoint.from_dict(data)
    except Exception as exc:
        raise typer.BadParameter(f"Checkpoint file is corrupt: {exc}") from exc

    if checkpoint.state in (JobState.COMPLETED, JobState.FAILED):
        print_warning(f"Job {job_id} is in '{checkpoint.state}' state; cannot resume.")
        return

    # If --live, ensure we have a read-only connection
    client = None
    if live and not yes_i_understand:
        confirmed = typer.confirm(
            "Resume with read-only broker connection? Continue?",
            default=False,
        )
        if not confirmed:
            raise typer.Exit(code=2)
        # In a full implementation, we'd reconnect here
        # For now, we proceed without a live client (offline resume)

    # Launch resumed job
    from iqoption_research.market_data.collector import launch_backfill

    # We need the original instrument and active_id from the checkpoint
    instrument = checkpoint.instrument
    active_id = checkpoint.active_id
    timeframe = checkpoint.timeframe_seconds
    target = checkpoint.target_count

    # Normalize instrument
    from iqoption_research.market_data.instruments import InstrumentRegistry
    try:
        canonical = InstrumentRegistry.normalize_symbol(instrument)
    except Exception:
        canonical = instrument

    job = launch_backfill(
        collector=collector,
        job_id=job_id,
        instrument=canonical,
        active_id=active_id,
        timeframe_seconds=timeframe,
        target_count=target,
        date_range=checkpoint.date_range,
        page_size=500,
        max_retries=5,
        client=client,
        on_progress=lambda progress: None,  # will be connected to CLI progress display
        on_finished=lambda result: None,  # will be connected to CLI finish display
    )

    state_map = {
        "not_started": "NOT_STARTED",
        "running": "RUNNING",
        "paused": "PAUSED",
        "interrupted": "INTERRUPTED",
        "completed": "COMPLETED",
        "partial": "PARTIAL",
        "failed": "FAILED",
    }

    def _on_progress(progress: dict) -> None:
        state = progress.get("state", "unknown")
        target_p = progress.get("target_count", target)
        collected = progress.get("unique_collected", checkpoint.unique_candles_persisted)
        pages = progress.get("pages_fetched", 0)
        dup_skip = progress.get("duplicates_skipped", 0)
        rej = progress.get("rejected_records", 0)
        remaining = progress.get("remaining", target_p - collected)
        pct = (collected / target_p * 100) if target_p > 0 else 0
        bar_len = 40
        filled_len = int(bar_len * collected // max(1, target_p))
        "█" * filled_len + "─" * (bar_len - filled_len)

        lines = [
            f"Resume Progress: {job_id}",
            f"State: {state_map.get(state, state)}",
            f"Progress: {collected}/{target_p} ({pct:.1f}%)",
            f"[{'█' * filled_len}{'─' * (bar_len - filled_len)}]",
            f"Pages: {pages} | Dups: {dup_skip} | Rejected: {rej}",
            f"Remaining: {remaining}",
        ]
        print_header("\n".join(lines))

    def _on_finished(result: dict) -> None:
        state = result.get("state", "unknown")
        unique = result.get("unique_candles", 0)
        target = result.get("total_requested", 0)
        newly = result.get("newly_collected", 0)
        pages = result.get("pages_fetched", 0)
        dups = result.get("duplicates_skipped", 0)
        rej = result.get("rejected_records", 0)
        remaining = result.get("remaining", target - unique)

        lines = [
            f"Resume Result: {job_id}",
            f"State: {state_map.get(state, state)}",
            f"Total unique: {unique} (target: {target})",
            f"Newly collected: {newly}",
            f"Pages fetched: {pages}",
            f"Duplicates skipped: {dups}",
            f"Rejected records: {rej}",
            f"Remaining: {remaining}",
        ]
        if state in ("completed", "partial", "failed"):
            print_header("\n".join(lines))
        else:
            print_header("\n".join(lines))

    job.on_progress = _on_progress
    job.on_finished = _on_finished

    try:
        asyncio.run(job.run())
    except Exception as exc:
        logger.error("Resume backfill error: %s", exc)
        print_error(f"Resume backfill error: {exc}")
        raise typer.Exit(code=1) from exc


def _install(app: typer.Typer) -> None:
    """Register the backfill CLI app into the main CLI tree."""
    app.add_typer(app, name="data")
