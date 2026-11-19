"""Operator-invoked, account-only wallet diagnostics."""

import asyncio
import getpass
import json
import logging
import warnings

import typer

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.security.credentials import Credentials


async def inspect_wallet(credentials: Credentials) -> dict[str, object]:
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
        return await client.ws.wallet_diagnostic()
    finally:
        await client.close()


def wallet_diagnostic_command(live: bool = typer.Option(False, "--live")) -> None:
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
        report = asyncio.run(inspect_wallet(credentials))
        typer.echo(json.dumps(report, sort_keys=True))
    except Exception:  # noqa: BLE001 - never expose transport errors or credential input
        typer.echo("Wallet diagnostic unavailable. Session selection remains unverified.", err=True)
        raise typer.Exit(1) from None
    finally:
        logging.disable(previous)
