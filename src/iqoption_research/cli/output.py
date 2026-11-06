"""Sanitized console output helpers (rich-based)."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel

from iqoption_research.security.credentials import redact_secrets

console = Console()


def print_header(title: str) -> None:
    console.print(Panel.fit(f"[bold]{title}[/bold]", border_style="cyan"))


def print_json(payload: Any, *, title: str | None = None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    # Defense in depth: never print redactable material to the console.
    text = redact_secrets(text)
    if title:
        console.print(f"[bold]{title}[/bold]")
    console.print(text)


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]WARNING:[/bold yellow] {redact_secrets(message)}")


def print_error(message: str) -> None:
    console.print(f"[bold red]ERROR:[/bold red] {redact_secrets(message)}")


def print_success(message: str) -> None:
    console.print(f"[bold green]OK:[/bold green] {redact_secrets(message)}")
