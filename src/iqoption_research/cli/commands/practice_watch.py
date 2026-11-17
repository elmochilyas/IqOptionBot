"""Operator entry point for read-only, bounded Practice observation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from typing import Any

import typer
from rich.console import Console

from iqoption_research.connectors.iqoption.exceptions import UnsupportedInstrumentError
from iqoption_research.execution.practice_history import normalize_history_asset
from iqoption_research.execution.practice_observation import observe_practice
from iqoption_research.execution.practice_observer import PassivePracticeObserver
from iqoption_research.security.credentials import Credentials

_OBSERVABLE_ASSETS = {"EURUSD", "EURUSD-OTC", "EURUSD_OP"}
_EXIT_UNVERIFIED = 2
_EXIT_SAFETY_CONFLICT = 3


def practice_watch_command(
    live: bool = typer.Option(False, "--live", help="Opt into read-only broker access."),
    asset: str = typer.Option(
        "EURUSD", "--asset", help="Exact instrument to observe and reconcile."
    ),
    seconds: int = typer.Option(60, "--seconds", min=10, max=3600, help="Observation budget."),
    history_limit: int = typer.Option(100, "--history-limit", min=1, max=1000),
    reconnect_limit: int = typer.Option(
        3, "--reconnect-limit", min=0, max=5, help="Bounded reconnection attempts."
    ),
    active_account_diagnostics: bool = typer.Option(
        False,
        "--active-account-diagnostics",
        help="Add sanitized active-account evidence classifications (no ids or balances).",
    ),
) -> None:
    """Observe existing Practice activity read-only and reconcile broker history.

    Credentials are read from IQOPTION_EMAIL / IQOPTION_PASSWORD or prompted for.
    No order, account-selection or other state-changing broker message is sent.
    EURUSD_OP (Blitz) collects passive schema metadata and bounded history only.
    No Blitz subscription or live payload schema has been verified.
    """
    if not live:
        raise typer.BadParameter("Explicit --live is required for a broker observation.")
    try:
        canonical = normalize_history_asset(asset)
    except UnsupportedInstrumentError:
        canonical = ""
    if canonical not in _OBSERVABLE_ASSETS:
        raise typer.BadParameter(
            "Choose EURUSD, EURUSD-OTC or EURUSD_OP.", param_hint="--asset"
        ) from None
    email = os.environ.get("IQOPTION_EMAIL") or typer.prompt("IQ Option Account Email")
    password = os.environ.get("IQOPTION_PASSWORD") or typer.prompt(
        "IQ Option Password", hide_input=True
    )
    credentials = Credentials(email=email, password=password)
    console = Console()

    def factory(handler: Callable[[str, dict[str, Any]], str | None]) -> PassivePracticeObserver:
        return PassivePracticeObserver(credentials=credentials, event_handler=handler)

    try:
        report = asyncio.run(
            observe_practice(
                factory,
                asset=canonical,
                seconds=seconds,
                history_limit=history_limit,
                reconnect_limit=reconnect_limit,
                active_account_diagnostics=active_account_diagnostics,
            )
        )
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except Exception:  # noqa: BLE001 - never expose broker or transport errors
        console.print("[red]Practice observation failed. Live delivery remains unverified.[/red]")
        raise typer.Exit(code=1) from None
    console.print_json(json.dumps(report))
    conflicts = report.get("position_lifecycle_counts", {}).get("CONFLICT", 0) > 0
    if (
        report["real_wallet_status"] == "MUTATION_OBSERVED"
        or conflicts
        or report.get("account_evidence_conflict_observed")
        or report.get("active_real_or_other_observed")
    ):
        raise typer.Exit(code=_EXIT_SAFETY_CONFLICT)
    if (
        report.get("websocket_authentication") != "VERIFIED"
        or report.get("practice_wallet") != "VERIFIED"
        or report.get("practice_account_active") != "ACTIVE_PRACTICE_CONFIRMED"
        or report.get("event_delivery") != "LIVE_POSITION_EVENT_OBSERVED"
    ):
        raise typer.Exit(code=_EXIT_UNVERIFIED)
