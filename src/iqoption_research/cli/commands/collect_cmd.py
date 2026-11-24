"""CLI workflow for operator-governed historical data collection (Phase 44).

Commands:
    agy data collect plan --symbol EURUSD --from <ISO> --to <ISO> [--timeframe 60]
    agy data collect run --job-id <ID> [--chunk-size 500] [--live]
    agy data collect status --job-id <ID> [--json]
    agy data collect cancel --job-id <ID> [--json]
    agy data collect resume --job-id <ID> [--chunk-size 500] [--live]
    agy data collect verify --job-id <ID> [--json]

Exit codes:
    0: Success / Verified / Planned
    1: Failure / Unverified / Source Unavailable
    2: Cancelled / Interrupted / Partially Collected
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from iqoption_research.market_data.collection_engine import (
    CollectionProgress,
    CollectionStatus,
    HistoricalCollectionCoordinator,
)

logger = logging.getLogger(__name__)

collect_app = typer.Typer(
    name="collect",
    help="Operator-governed historical market data collection center.",
    no_args_is_help=True,
)


def _resolve_exit_code(status: CollectionStatus | str) -> int:
    """Map collection status to explicit CLI exit codes.

    0: Success / Verified / Pending
    1: Failure / Unverified / Source Unavailable / Not Found
    2: Cancelled / Interrupted / Partially Collected
    """
    s = status.value if isinstance(status, CollectionStatus) else str(status)
    if s in (CollectionStatus.COMPLETED_VERIFIED.value, CollectionStatus.PENDING.value):
        return 0
    if s in (
        CollectionStatus.CANCELLED.value,
        CollectionStatus.INTERRUPTED.value,
        CollectionStatus.PARTIALLY_COLLECTED.value,
    ):
        return 2
    return 1


def _emit_output(data: dict[str, Any], json_output: bool) -> None:
    """Emit either machine-readable JSON or formatted human-readable text."""
    if json_output:
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
    else:
        for k, v in data.items():
            typer.echo(f"{k}: {v}")


@collect_app.command("plan")
def plan_collection_command(
    symbol: str = typer.Option("EURUSD", "--symbol", help="Asset symbol (e.g. EURUSD)."),
    from_utc: str = typer.Option(
        ...,
        "--from",
        "--from-utc",
        "--start",
        help="UTC start timestamp in ISO-8601 format.",
    ),
    to_utc: str = typer.Option(
        ...,
        "--to",
        "--to-utc",
        "--end",
        help="UTC end timestamp in ISO-8601 format.",
    ),
    timeframe: int = typer.Option(60, "--timeframe", help="Candle timeframe in seconds."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Inspect local coverage and generate a planned collection job."""
    coord = HistoricalCollectionCoordinator(Path())
    try:
        job_id, missing, progress = coord.plan_collection(
            symbol=symbol,
            start_utc=from_utc,
            end_utc=to_utc,
            timeframe_seconds=timeframe,
        )
    except Exception as exc:  # noqa: BLE001
        err_data = {"error": str(exc), "status": "FAILED"}
        _emit_output(err_data, json_output)
        raise typer.Exit(1) from None

    result = {
        "job_id": job_id,
        "symbol": progress.symbol,
        "status": progress.status.value,
        "requested_from_utc": progress.requested_from_utc,
        "requested_to_utc": progress.requested_to_utc,
        "timeframe_seconds": progress.timeframe_seconds,
        "missing_intervals_count": len(missing),
        "missing_intervals": [m.to_dict() for m in missing],
    }
    _emit_output(result, json_output)
    raise typer.Exit(0)


@collect_app.command("status")
def status_collection_command(
    job_id: str = typer.Option(..., "--job-id", help="Collection job ID."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Inspect durable collection checkpoint progress and status."""
    coord = HistoricalCollectionCoordinator(Path())
    try:
        progress = coord.get_progress(job_id)
    except FileNotFoundError:
        err_data = {"error": f"Job {job_id} not found.", "status": "NOT_FOUND"}
        _emit_output(err_data, json_output)
        raise typer.Exit(1) from None

    data = progress.to_dict()
    _emit_output(data, json_output)
    code = _resolve_exit_code(progress.status)
    raise typer.Exit(code)


@collect_app.command("cancel")
def cancel_collection_command(
    job_id: str = typer.Option(..., "--job-id", help="Collection job ID."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Cancel an active or pending collection job safely at page boundary."""
    coord = HistoricalCollectionCoordinator(Path())
    try:
        progress = coord.cancel_collection(job_id)
    except FileNotFoundError:
        err_data = {"error": f"Job {job_id} not found.", "status": "NOT_FOUND"}
        _emit_output(err_data, json_output)
        raise typer.Exit(1) from None

    data = progress.to_dict()
    _emit_output(data, json_output)
    raise typer.Exit(2)  # 2: cancelled


@collect_app.command("verify")
def verify_collection_command(
    job_id: str = typer.Option(..., "--job-id", help="Collection job ID."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Verify cryptographic receipts, parquet integrity, and coverage offline."""
    coord = HistoricalCollectionCoordinator(Path())
    report = coord.verify_job_integrity(job_id)
    _emit_output(report, json_output)
    if report.get("verified", False):
        raise typer.Exit(0)
    raise typer.Exit(1)


@collect_app.command("run")
def run_collection_command(
    job_id: str = typer.Option(..., "--job-id", help="Collection job ID."),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Candles per batch request."),
    live: bool = typer.Option(
        False, "--live", help="Operator opt-in to authorized live read-only broker connection."
    ),
    saved_credentials: bool = typer.Option(
        True, "--saved-credentials", help="Use secure Windows credential vault."
    ),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Offline synthetic test execution fixture."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Execute an authorized historical collection job."""
    coord = HistoricalCollectionCoordinator(Path())

    if not live and not synthetic:
        data = {
            "mode": "OFFLINE_INSPECTION",
            "message": "Actual collection requires explicit --live flag with operator authorization, or --synthetic for offline testing.",
            "job_id": job_id,
        }
        _emit_output(data, json_output)
        raise typer.Exit(0)

    async def _run() -> CollectionProgress:
        if synthetic:
            # Synthetic mock client for testing
            from unittest.mock import AsyncMock, MagicMock

            from iqoption_research.market_data.fetch import CandleFetchResult
            from iqoption_research.market_data.validation import assess_data_quality
            from iqoption_research.models.market import CandleSize
            from tests.unit.test_phase43_data_inventory import (
                create_deterministic_synthetic_candles,
            )

            mock_client = MagicMock()
            mock_client.assert_read_only = MagicMock()
            ckpt = coord.checkpoints_dir / f"{job_id}.json"
            c_data = json.loads(ckpt.read_text(encoding="utf-8"))
            req_from = int(c_data["requested_from_ts"])
            candles = create_deterministic_synthetic_candles(count=chunk_size, start_ts=req_from)
            quality = assess_data_quality(candles, expected_size=CandleSize(60))
            fetch_res = CandleFetchResult(
                symbol="EURUSD",
                size=CandleSize(60),
                candles=tuple(candles),
                errors=(),
                quality=quality,
                fetched_at=datetime.now(UTC),
                raw_count=len(candles),
                source="synthetic",
            )
            mock_client.get_candles = AsyncMock(return_value=fetch_res)
            return await coord.execute_collection(
                job_id=job_id,
                client=mock_client,
                chunk_size=chunk_size,
                is_synthetic=True,
                operator_authorized=True,
            )

        from iqoption_research.cli.commands.backfill_cmd import _connect_live_client

        client, cleanup = await _connect_live_client(saved_credentials)
        try:
            return await coord.execute_collection(
                job_id=job_id,
                client=client,
                chunk_size=chunk_size,
                is_synthetic=False,
                operator_authorized=True,
            )
        finally:
            await cleanup()

    try:
        progress = asyncio.run(_run())
        data = progress.to_dict()
        _emit_output(data, json_output)
        raise typer.Exit(_resolve_exit_code(progress.status))
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        err_data = {"error": str(exc), "status": "FAILED"}
        _emit_output(err_data, json_output)
        raise typer.Exit(1) from None


@collect_app.command("resume")
def resume_collection_command(
    job_id: str = typer.Option(..., "--job-id", help="Collection job ID to resume."),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Candles per batch request."),
    live: bool = typer.Option(
        False, "--live", help="Operator opt-in to authorized live read-only broker connection."
    ),
    saved_credentials: bool = typer.Option(
        True, "--saved-credentials", help="Use secure Windows credential vault."
    ),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Offline synthetic test execution fixture."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Reconcile storage and durable receipts, then resume collection."""
    coord = HistoricalCollectionCoordinator(Path())

    try:
        # Reconcile progress before resuming
        reconciled_ckpt = coord.reconcile_job_state(job_id)
    except FileNotFoundError:
        err_data = {"error": f"Job {job_id} not found.", "status": "NOT_FOUND"}
        _emit_output(err_data, json_output)
        raise typer.Exit(1) from None

    if not live and not synthetic:
        data = {
            "mode": "OFFLINE_RECONCILED",
            "message": "Storage and receipts reconciled. Resuming requires explicit --live or --synthetic.",
            "job_id": job_id,
            "reconciled_status": reconciled_ckpt.status,
            "completed_intervals": len(reconciled_ckpt.completed_interval_indices),
            "total_intervals": len(reconciled_ckpt.missing_intervals),
            "candles_collected": reconciled_ckpt.total_candles_collected,
        }
        _emit_output(data, json_output)
        raise typer.Exit(0)

    async def _resume() -> CollectionProgress:
        if synthetic:
            from unittest.mock import AsyncMock, MagicMock

            from iqoption_research.market_data.fetch import CandleFetchResult
            from iqoption_research.market_data.validation import assess_data_quality
            from iqoption_research.models.market import CandleSize
            from tests.unit.test_phase43_data_inventory import (
                create_deterministic_synthetic_candles,
            )

            mock_client = MagicMock()
            mock_client.assert_read_only = MagicMock()
            ckpt = coord.checkpoints_dir / f"{job_id}.json"
            c_data = json.loads(ckpt.read_text(encoding="utf-8"))
            req_from = int(c_data["requested_from_ts"])
            candles = create_deterministic_synthetic_candles(count=chunk_size, start_ts=req_from)
            quality = assess_data_quality(candles, expected_size=CandleSize(60))
            fetch_res = CandleFetchResult(
                symbol="EURUSD",
                size=CandleSize(60),
                candles=tuple(candles),
                errors=(),
                quality=quality,
                fetched_at=datetime.now(UTC),
                raw_count=len(candles),
                source="synthetic",
            )
            mock_client.get_candles = AsyncMock(return_value=fetch_res)
            return await coord.resume_collection(
                job_id=job_id,
                client=mock_client,
                chunk_size=chunk_size,
                is_synthetic=True,
                operator_authorized=True,
            )

        from iqoption_research.cli.commands.backfill_cmd import _connect_live_client

        client, cleanup = await _connect_live_client(saved_credentials)
        try:
            return await coord.resume_collection(
                job_id=job_id,
                client=client,
                chunk_size=chunk_size,
                is_synthetic=False,
                operator_authorized=True,
            )
        finally:
            await cleanup()

    try:
        progress = asyncio.run(_resume())
        data = progress.to_dict()
        _emit_output(data, json_output)
        raise typer.Exit(_resolve_exit_code(progress.status))
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        err_data = {"error": str(exc), "status": "FAILED"}
        _emit_output(err_data, json_output)
        raise typer.Exit(1) from None
