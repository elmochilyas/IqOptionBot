"""Offline connection simulation command.

Runs the full read-only flow against a deterministic synthetic connector.
No network, no credentials, no real market data.
"""

from __future__ import annotations

from pathlib import Path

import typer

from iqoption_research.cli.output import (
    print_header,
    print_json,
    print_success,
    print_warning,
)
from iqoption_research.connectors.simulation import (
    SIMULATION_DISCLAIMER,
    SimulatedReadOnlyConnector,
)
from iqoption_research.market_data.storage import save_candles


async def run_simulation(
    *,
    symbol: str = "EURUSD",
    size: int = 60,
    count: int = 50,
    save: bool = False,
    output_dir: str | Path = "data/simulated",
) -> dict[str, object]:
    """Execute the offline simulation and return a summary payload."""
    connector = SimulatedReadOnlyConnector()
    await connector.connect()
    try:
        status = await connector.get_account_status()
        result = await connector.get_candles(symbol, size, count)
    finally:
        await connector.close()

    verification = status.practice_verification()
    summary: dict[str, object] = {
        "mode": "offline_simulation",
        "disclaimer": SIMULATION_DISCLAIMER,
        "account": {
            "practice_verified": verification.is_verified,
            "practice_status": verification.status.value,
            "balances": len(status.balances),
        },
        "candles": result.to_dict(),
    }

    if save:
        saved = save_candles(
            result.candle_list,
            output_dir,
            extra={"source": "simulated", "disclaimer": SIMULATION_DISCLAIMER},
        )
        summary["saved"] = {
            "data": str(saved.data_path),
            "metadata": str(saved.metadata_path),
            "count": saved.candle_count,
        }

    return summary


def simulate_command(
    symbol: str = typer.Option("EURUSD", help="Instrument (Phase 1: EURUSD)."),
    size: int = typer.Option(60, help="Candle size in seconds."),
    count: int = typer.Option(50, help="Number of synthetic candles."),
    save: bool = typer.Option(False, "--save", help="Write sanitized candles to disk."),
    output_dir: str = typer.Option("data/simulated", help="Output directory."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only."),
) -> None:
    """Simulate the read-only connection offline (no network I/O)."""
    import asyncio

    if not json_output:
        print_header("Offline Read-Only Simulation")
        print_warning(SIMULATION_DISCLAIMER)

    summary = asyncio.run(
        run_simulation(
            symbol=symbol,
            size=size,
            count=count,
            save=save,
            output_dir=output_dir,
        )
    )

    if json_output:
        print_json(summary)
        return

    print_json(summary, title="Simulation summary")
    print_success("Offline simulation completed; no network I/O was performed.")
