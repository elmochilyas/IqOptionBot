"""Explicit local account entry and bounded collection; no network on import or help."""

from __future__ import annotations

import asyncio
import getpass
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import typer

from iqoption_research.auth.vault import WindowsCredentialVault
from iqoption_research.market_data.extended import (
    ExtendedCollection,
    check_duration_vs_market_close,
    validate_duration,
)

app = typer.Typer(help="Local account entry and separate extended market collection.")


def require_terminal() -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise typer.BadParameter("An operator-owned interactive terminal is required")


@app.command("account-entry")
def account_entry(account_id: str | None = typer.Option(None, "--account-id")) -> None:
    """Add/update a saved identity; the operator enters credentials only in this terminal."""
    require_terminal()
    store = WindowsCredentialVault().account_store()
    if account_id is not None and account_id not in {a.account_id for a in store.list_accounts()}:
        raise typer.BadParameter("Selected account is unavailable")
    if not typer.confirm(
        "Save these credentials in this Windows user's secure vault?", default=False
    ):
        raise typer.Exit(130)
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        email = getpass.getpass("Account email (hidden): ")
        password = getpass.getpass("Account password (hidden): ")
    store.save(email, password, explicit_opt_in=True, account_id=account_id)
    del password
    typer.echo("Account saved. No broker login was performed.")


@app.command("collect")
def collect(
    account_id: str = typer.Option(..., "--account-id"),
    minutes: int = typer.Option(..., "--minutes"),
    live: bool = typer.Option(False, "--live"),
    authorize: bool = typer.Option(False, "--yes-i-understand"),
    cancel_file: Path | None = typer.Option(None, "--cancel-file", hidden=True),
    status_file: Path | None = typer.Option(None, "--status-file", hidden=True),
) -> None:
    """One read-only EURUSD collection, completely separate from Phase 38."""
    validate_duration(minutes)
    require_terminal()
    if not live or not authorize:
        raise typer.BadParameter("Collection requires --live and --yes-i-understand")

    # Pre-flight: warn if the requested duration extends past the next scheduled market close.
    # This is advisory only; it does not block the session.
    now_utc = datetime.now(UTC)
    duration_warning = check_duration_vs_market_close(now_utc, minutes)
    if duration_warning:
        typer.echo(f"\n{duration_warning}\n", err=True)

    root = Path(__file__).resolve().parents[4]
    session = ExtendedCollection(root, minutes, account_id, status_path=status_file)
    typer.echo(f"Extended collection destination: {session.directory}")
    asyncio.run(session.run(cancel_file))
    reason = session.state["termination_reason"]
    typer.echo(f"Collection ended: {reason}. Saved data is not prospective predictions.")
    if reason == "SCHEDULED_MARKET_CLOSURE":
        ended_at = session.state.get(
            "scheduled_closure_at_utc", session.state.get("ended_at_utc", "")
        )
        typer.echo(
            f"Market closed on schedule at {ended_at}. "
            "Committed candles and final manifest have been preserved. "
            "This is not a failure; the duration was shorter than requested due to market hours."
        )
    elif reason == "OPERATOR_CANCELLED":
        raise typer.Exit(130)
