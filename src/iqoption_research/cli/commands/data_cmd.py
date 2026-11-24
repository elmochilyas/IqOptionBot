"""Top-level historical data commands."""

from __future__ import annotations

from iqoption_research.cli.commands.backfill_cmd import app
from iqoption_research.cli.commands.collect_cmd import collect_app
from iqoption_research.cli.commands.history_cmd import (
    diagnose_history_boundary_command,
    expand_history_command,
    validate_history_command,
)

app.add_typer(collect_app, name="collect")

__all__ = [
    "app",
    "collect_app",
    "diagnose_history_boundary_command",
    "expand_history_command",
    "validate_history_command",
]
