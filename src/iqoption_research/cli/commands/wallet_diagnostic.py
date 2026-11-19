"""Operator-invoked, account-only wallet diagnostics."""

import asyncio
import getpass
import json
import logging
import warnings
from enum import StrEnum

import typer

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.models.account import AccountType
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.security.credentials import Credentials


class InspectionMode(StrEnum):
    PRACTICE = "PRACTICE"
    REAL = "REAL"


async def inspect_wallet(
    credentials: Credentials, *, select: InspectionMode | None = None, interactive: bool = False
) -> dict[str, object]:
    client = AsyncIQOption(
        credentials,
        offline=False,
        diagnostics=False,
        config=ConnectionConfig(max_reconnect_attempts=0),
    )
    try:
        client.assert_read_only()
        await client.connect(raise_if_incomplete=False)
        if client.ws is None:
            raise RuntimeError("Connection unavailable")
        ws = client.ws
        report = await ws.wallet_diagnostic()
        if select:
            try:
                ws.select_wallet_context(AccountType[select.name])
            except ValueError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(2) from None
            report = ws.wallet_report()
        if interactive:
            typer.echo("This connection supports status, select PRACTICE, select REAL, and exit.")
            while True:
                typer.echo(json.dumps(report, sort_keys=True))
                try:
                    command = (await asyncio.to_thread(input, "wallet> ")).strip().upper()
                except EOFError:
                    break
                if command == "EXIT":
                    break
                if command in ("SELECT PRACTICE", "SELECT REAL"):
                    try:
                        ws.select_wallet_context(AccountType[command.split()[1]])
                    except ValueError as exc:
                        typer.echo(str(exc), err=True)
                elif command != "STATUS":
                    typer.echo("Enter status, select PRACTICE, select REAL, or exit.", err=True)
                report = ws.wallet_report()
        return report
    finally:
        await client.close()


def wallet_diagnostic_command(
    live: bool = typer.Option(False, "--live"),
    select: InspectionMode | None = typer.Option(
        None, "--select", case_sensitive=False, help="Select a connection-local inspection context."
    ),
    interactive: bool = typer.Option(
        False, "--interactive", help="Keep this inspection connection open."
    ),
) -> None:
    """Inspect profile and balance metadata once, with hidden interactive credentials."""
    if not live:
        typer.echo("Supply --live to authenticate for an account-only diagnostic.")
        return
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        # Abort if the terminal would fall back to echoing either prompt.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            credentials = Credentials(
                getpass.getpass("Account email (hidden): "),
                getpass.getpass("Password (hidden): "),
            )
        report = asyncio.run(inspect_wallet(credentials, select=select, interactive=interactive))
        if not interactive:
            typer.echo(json.dumps(report, sort_keys=True))
    except typer.Exit:
        raise
    except Exception:  # noqa: BLE001 - never expose transport errors or credential input
        typer.echo("Wallet diagnostic unavailable. Session selection remains unverified.", err=True)
        raise typer.Exit(1) from None
    finally:
        logging.disable(previous)
