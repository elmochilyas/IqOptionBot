"""CLI command group for Phase 9 authentic historical data backfitting.

Registers the ``backfill`` typer group under the ``data`` namespace,
making commands available as ``bot data backfill``, ``bot data backfill-status``,
and ``bot data backfill-resume``.
"""

from __future__ import annotations

import typer

from iqoption_research.cli.commands.backfill_cmd import app as backfill_app

app = typer.Typer(help="Historical market-data backfill and management")

app.add_typer(backfill_app, name="backfill")
