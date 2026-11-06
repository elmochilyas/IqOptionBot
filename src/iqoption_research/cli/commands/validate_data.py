"""Offline market-data validation command."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer

from iqoption_research.cli.output import print_header, print_json, print_warning
from iqoption_research.connectors.simulation import (
    SIMULATION_DISCLAIMER,
    SimulatedReadOnlyConnector,
)
from iqoption_research.market_data.storage import load_candles_jsonl
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.market import Candle


async def _simulated_candles(symbol: str, size: int, count: int) -> list[Candle]:
    connector = SimulatedReadOnlyConnector()
    await connector.connect()
    try:
        now_epoch = int(time.time())
        result = await connector.get_candles(symbol, size, count, endtime=now_epoch)
        return result.candle_list
    finally:
        await connector.close()


def validate_command(
    file: Path | None = typer.Option(
        None, "--file", exists=True, dir_okay=False, help="Normalized JSONL dataset."
    ),
    simulate: bool = typer.Option(False, "--simulate", help="Validate simulated candles."),
    symbol: str = typer.Option("EURUSD", help="Symbol when using simulated data."),
    size: int = typer.Option(60, help="Candle size in seconds when using simulated data."),
    count: int = typer.Option(50, help="Number of simulated candles."),
    strict: bool = typer.Option(False, "--strict", help="Exit 1 if data is not research grade."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only."),
) -> None:
    """Validate candle structure, timestamps, duplicates, and gaps offline."""
    if file is not None and not simulate:
        candles = load_candles_jsonl(file)
        source = str(file)
        if not candles:
            raise typer.BadParameter("Dataset contains no candles")
        size_seconds = candles[0].size.value
    else:
        candles = asyncio.run(_simulated_candles(symbol, size, count))
        source = "simulated"
        size_seconds = size

    report = assess_data_quality(candles, expected_size=size_seconds)
    payload = {
        "source": source,
        "strict": strict,
        "quality": report.to_dict(),
    }

    if json_output:
        print_json(payload)
    else:
        print_header("Offline Market-Data Validation")
        if file is None:
            print_warning(SIMULATION_DISCLAIMER)
        print_json(payload)

    if strict and not report.is_research_grade:
        raise typer.Exit(code=1)
