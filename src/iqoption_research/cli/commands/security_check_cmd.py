"""Offline security-check command."""

from __future__ import annotations

import typer

from iqoption_research.cli.commands.security_check import run_security_checks
from iqoption_research.cli.output import print_header, print_json


def security_check_command(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only."),
) -> None:
    """Validate TLS, endpoint, read-only, and secret-handling posture offline."""
    results = run_security_checks()
    passed = all(r.passed for r in results)
    payload = {
        "passed": passed,
        "checks": [r.to_dict() for r in results],
    }

    if json_output:
        print_json(payload)
    else:
        print_header("Offline Security Checks")
        print_json(payload)

    if not passed:
        raise typer.Exit(code=1)
