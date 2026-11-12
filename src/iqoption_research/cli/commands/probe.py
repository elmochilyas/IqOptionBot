"""Optional real read-only connection probe.

SAFETY GATES (all must hold):
1. The user explicitly runs this command.
2. ``--live`` is passed.
3. The user confirms interactively (or passes ``--yes-i-understand``).
4. Credentials are entered via getpass, never via CLI flags.

The probe only reads profile/balances/candles. It cannot place or close
orders and it never switches account balances, because no such methods
exist on the read-only client.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from iqoption_research.cli.output import (
    print_error,
    print_header,
    print_json,
    print_success,
    print_warning,
)
from iqoption_research.config.settings import Settings
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import IQOptionError
from iqoption_research.market_data.storage import save_candles
from iqoption_research.models.account import PracticeStatus
from iqoption_research.security.credentials import get_credentials, redact_secrets


async def run_probe(
    *,
    symbol: str,
    size: int,
    count: int,
    save: bool,
    output_dir: str | Path,
    diagnostics: bool = False,
) -> dict[str, object]:
    """Perform one explicit, read-only live probe. Caller supplies consent."""
    credentials = get_credentials()
    settings = Settings(offline=False, diagnostics=diagnostics)
    client = AsyncIQOption(credentials, settings=settings, offline=False, diagnostics=diagnostics)

    try:
        await client.connect(raise_if_incomplete=False)

        if not client.is_ready:
            # Authenticated over WS, but profile or balances were missing/timed out
            summary: dict[str, object] = {
                "mode": "real_read_only_probe",
                "connection": client.describe_connection(),
                "account_data_missing": True,
                "practice_verified": False,
                "note": (
                    "WebSocket authenticated successfully, but required account data "
                    "(profile/balances) was not received within timeout. "
                    "PRACTICE status remains explicitly UNVERIFIED."
                ),
            }
            if diagnostics:
                summary["diagnostics"] = client.diagnostics.summary(connection_state=client.state)
            return summary

        status = await client.get_account_status()
        verification = status.practice_verification()

        # Gate: Require fresh server-confirmed active PRACTICE verification
        # before proceeding beyond account discovery. If REAL is active or unverified,
        # halt before market-data retrieval to protect real account.
        if not verification.is_verified:
            summary = {
                "mode": "real_read_only_probe",
                "connection": client.describe_connection(),
                "account": client.describe_account(status),
                "practice_verified": False,
                "practice_status": verification.status.value,
                "warning": (
                    "PROBE HALTED: Server-reported active balance is not verified as PRACTICE. "
                    f"Status: {verification.status.value.upper()}. "
                    "Market-data retrieval stopped to ensure account safety."
                ),
                "note": (
                    "Read-only probe halted before market-data retrieval. "
                    "No orders placed, no balances switched."
                ),
            }
            if diagnostics:
                summary["diagnostics"] = client.diagnostics.summary(connection_state=client.state)
            return summary

        result = await client.get_candles(symbol, size, count)

        summary = {
            "mode": "real_read_only_probe",
            "connection": client.describe_connection(),
            "account": client.describe_account(status),
            "practice_verified": True,
            "candles": result.to_dict(),
            "note": (
                "Read-only probe. No orders were placed, no balances were "
                "switched, and no credentials were displayed."
            ),
        }
        if diagnostics:
            summary["diagnostics"] = client.diagnostics.summary(connection_state=client.state)

        if save:
            saved = save_candles(
                result.candle_list,
                output_dir,
                extra={
                    "source": "iqoption_unofficial_websocket",
                    "probe": "explicit_read_only",
                },
            )
            summary["saved"] = {
                "data": str(saved.data_path),
                "metadata": str(saved.metadata_path),
                "count": saved.candle_count,
            }

        return summary
    finally:
        await client.close()


def probe_command(
    live: bool = typer.Option(
        False, "--live", help="Enable the real read-only connection (required)."
    ),
    yes_i_understand: bool = typer.Option(
        False,
        "--yes-i-understand",
        help="Skip the interactive confirmation (still requires --live).",
    ),
    diagnostics: bool = typer.Option(
        False,
        "--diagnostics",
        help="Enable safe metadata-only protocol diagnostics (no secrets/personal data logged).",
    ),
    symbol: str = typer.Option("EURUSD", help="Instrument (Phase 1: EURUSD)."),
    size: int = typer.Option(60, help="Candle size in seconds."),
    count: int = typer.Option(20, help="Historical candles to read."),
    save: bool = typer.Option(False, "--save", help="Write sanitized candles to disk."),
    output_dir: str = typer.Option("data/live_probe", help="Output directory."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only."),
) -> None:
    """Explicit real read-only probe against IQ Option PRACTICE.

    This command performs no trading and does not switch account mode.
    It is never run automatically by install, tests, or startup.
    """
    if not live:
        print_warning(
            "Real probe disabled. This is the safe default. "
            "Re-run with --live (plus confirmation) to perform a read-only "
            "connection probe. No network I/O was attempted."
        )
        return

    if not yes_i_understand:
        confirmed = typer.confirm(
            "This will authenticate to IQ Option using an unofficial interface. "
            "It performs READ-ONLY calls only, but the session may retain REAL "
            "account privileges. Continue?",
            default=False,
        )
        if not confirmed:
            print_warning("Probe cancelled by user.")
            raise typer.Exit(code=2)

    if not json_output:
        print_header("Real Read-Only Probe (explicitly authorized)")
        print_warning(
            "Unofficial interface: compatibility and platform authorization "
            "are not established. Do not place trades."
        )

    try:
        summary = asyncio.run(
            run_probe(
                symbol=symbol,
                size=size,
                count=count,
                save=save,
                output_dir=output_dir,
                diagnostics=diagnostics,
            )
        )
    except IQOptionError as exc:
        print_error(f"{type(exc).__name__}: {redact_secrets(str(exc))}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"{type(exc).__name__}: {redact_secrets(str(exc))}")
        raise typer.Exit(code=1) from exc

    if json_output:
        print_json(summary)
    else:
        print_json(summary, title="Probe summary")
        if summary.get("account_data_missing"):
            print_warning(
                "WebSocket authenticated, but required account profile/balances "
                "were missing or timed out."
            )
            print_warning("PRACTICE status remains explicitly UNVERIFIED.")
            if diagnostics and "diagnostics" in summary:
                print_json(summary["diagnostics"], title="Safe Protocol Diagnostics")
        elif not summary.get("practice_verified", False):
            practice_status = summary.get("practice_status")
            if practice_status == PracticeStatus.ACTIVE_REAL.value:
                print_warning(
                    "CRITICAL WARNING: Server-reported active account is REAL. "
                    "Probe halted before market-data retrieval to protect real account. "
                    "No account switching was performed."
                )
            else:
                print_warning(
                    f"PRACTICE status UNVERIFIED ({practice_status}): active account is not "
                    "verified as PRACTICE. Probe halted before market-data retrieval. "
                    "No account switching was performed."
                )
            if diagnostics and "diagnostics" in summary:
                print_json(summary["diagnostics"], title="Safe Protocol Diagnostics")
        else:
            candles_info = summary.get("candles")
            if isinstance(candles_info, dict):
                quality = candles_info.get("quality", {})
                if isinstance(quality, dict) and quality.get("stale"):
                    print_warning(
                        f"DATA QUALITY WARNING: Retrieved candles are stale. "
                        f"Newest candle ended at {quality.get('newest_candle_utc')} "
                        f"({quality.get('stale_age_seconds', 0):.0f}s ago). "
                        "Do not assume market is closed without verified trading schedule."
                    )
            print_success("Read-only probe completed. No orders placed, no balances switched.")
