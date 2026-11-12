"""Command-line interface for Phase 1 research tooling.

All commands default to offline behaviour. The only network-using
command is ``probe``, which requires an explicit ``--live`` flag and
interactive confirmation.
"""

from __future__ import annotations

import typer

from iqoption_research.cli.commands.ml_cmd import ml_app
from iqoption_research.cli.commands.probe import probe_command
from iqoption_research.cli.commands.research_cmd import research_app
from iqoption_research.cli.commands.security_check_cmd import security_check_command
from iqoption_research.cli.commands.simulate import simulate_command
from iqoption_research.cli.commands.validate_data import validate_command
from iqoption_research.security.logging import configure_logging

app = typer.Typer(
    name="iqresearch",
    help=(
        "Secure, read-only IQ Option research tooling (Phases 1-3). "
        "Offline by default; the live probe requires explicit opt-in."
    ),
    no_args_is_help=True,
    add_completion=False,
)

app.command("simulate", help="Offline connection simulation (no network).")(simulate_command)
app.command("security-check", help="Offline TLS/endpoint/read-only checks.")(
    security_check_command
)
app.command("validate-data", help="Offline candle validation.")(validate_command)
app.command("probe", help="Explicit read-only live probe (requires --live).")(
    probe_command
)
app.add_typer(research_app, name="research")
app.add_typer(ml_app, name="ml")


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """Configure sanitized logging only. Never touches the network."""
    configure_logging(level=10 if verbose else 20)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
