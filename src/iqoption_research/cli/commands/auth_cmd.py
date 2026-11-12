"""CLI commands for owner-controlled authentication and session management.

Provides local interactive login, Windows Credential Vault management,
and zero-leakage session status monitoring.
"""

from __future__ import annotations

import getpass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.auth.config import RuntimeSecurityConfig
from iqoption_research.auth.manager import AuthenticationManager
from iqoption_research.auth.provider import IQOptionAuthProvider, MockAuthProvider
from iqoption_research.auth.vault import InsecureKeyringBackendError, WindowsCredentialVault
from iqoption_research.security.credentials import redact_email

console = Console()
auth_app = typer.Typer(
    name="auth",
    help="Secure authentication, session management, and Windows vault controls.",
    no_args_is_help=True,
)

# Global or module-level session manager reference
_AUTH_MANAGER: AuthenticationManager | None = None


def get_auth_manager(*, force_mock: bool | None = None) -> AuthenticationManager:
    """Resolve the singleton AuthenticationManager instance."""
    global _AUTH_MANAGER
    if _AUTH_MANAGER is None:
        cfg = RuntimeSecurityConfig()
        # Default to MockAuthProvider in dev unless explicitly configured otherwise
        use_mock = cfg.is_development if force_mock is None else force_mock
        provider = MockAuthProvider() if use_mock else IQOptionAuthProvider()
        _AUTH_MANAGER = AuthenticationManager(config=cfg, provider=provider)
    return _AUTH_MANAGER


@auth_app.command("status", help="Display current authentication and session status.")
def status_cmd() -> None:
    """Display active authentication state, session lifetime, and vault status."""
    manager = get_auth_manager()
    snapshot = manager.get_status_snapshot()

    table = Table(
        title="IQ Option Bot — Authentication & Session Status",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Property", style="bold white", width=32)
    table.add_column("Value", style="bold", width=48)

    # Colorize status
    auth_color = "green" if snapshot["auth_status"] == "CONNECTED_READ_ONLY" else "yellow"
    if "INVALID" in snapshot["auth_status"] or "EXPIRED" in snapshot["auth_status"]:
        auth_color = "red"

    table.add_row(
        "Authentication Status", f"[{auth_color}]{snapshot['auth_status']}[/{auth_color}]"
    )
    table.add_row("Connection Status", snapshot["connection_status"])
    table.add_row("Active Session", str(snapshot["session_active"]))

    if snapshot["session_account"]:
        table.add_row("Authenticated Account", snapshot["session_account"])

    if snapshot["session_age_seconds"] is not None:
        table.add_row("Session Age", f"{snapshot['session_age_seconds']:.1f} seconds")
    if snapshot["session_ttl_remaining_seconds"] is not None:
        table.add_row(
            "Session TTL Remaining", f"{snapshot['session_ttl_remaining_seconds']:.1f} seconds"
        )

    table.add_row("Credential Storage Mode", snapshot["credential_storage_mode"])
    table.add_row("Saved Account in Vault", snapshot["saved_account"] or "None")
    table.add_row(
        "Windows Vault Backend",
        "Verified Secure" if snapshot["vault_backend_secure"] else "Unavailable / Insecure",
    )
    table.add_row("Server Account Type", snapshot["server_account_type"])

    practice_color = "green" if snapshot["is_practice_verified"] else "yellow"
    table.add_row(
        "PRACTICE Verified",
        f"[{practice_color}]{snapshot['is_practice_verified']}[/{practice_color}]",
    )
    table.add_row(
        "Broker Ledger Routing", f"[bold red]{snapshot['broker_ledger_status']}[/bold red]"
    )

    if snapshot["last_error"]:
        table.add_row("Last Error", f"[red]{snapshot['last_error']}[/red]")

    console.print(table)


@auth_app.command("login", help="Authenticate interactively or via saved Windows vault.")
def login_cmd(
    saved: bool = typer.Option(
        False,
        "--saved",
        "-s",
        help="Authenticate using credentials saved in Windows Credential Vault.",
    ),
    dev_mock: bool = typer.Option(
        True,
        "--mock/--live",
        help="Use mock offline provider (default) or live broker endpoints.",
    ),
) -> None:
    """Execute owner authentication."""
    import asyncio

    manager = get_auth_manager(force_mock=dev_mock)

    console.print(
        Panel(
            f"[bold cyan]IQ Option Secure Authentication[/bold cyan]\n"
            f"Mode: {'Windows Credential Vault' if saved else 'Interactive Terminal Prompt'}\n"
            f"Provider: {'Offline Mock Fixture' if dev_mock else 'Live HTTPS Broker'}",
            border_style="cyan",
        )
    )

    if saved:
        if not manager.vault.has_saved_credentials():
            console.print(
                "[yellow]No credentials saved in Windows Credential Vault.[/yellow]\n"
                "Run [bold green]bot auth save-credentials[/bold green] first, "
                "or log in interactively."
            )
            raise typer.Exit(code=1)
        console.print("[cyan]Loading credentials from Windows Credential Vault...[/cyan]")
        success = asyncio.run(manager.login_saved())
    else:
        # Prompt credentials
        console.print("Please enter your IQ Option account credentials.")
        try:
            email = input("IQ Option email: ").strip()
            password = getpass.getpass("IQ Option password: ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Authentication cancelled by user.[/yellow]")
            raise typer.Exit(code=1) from None

        success = asyncio.run(manager.login_interactive(email=email, password=password))

    if success:
        session = manager.session
        acct_display = redact_email(session.email) if session else "Authenticated"
        console.print(
            Panel(
                f"[bold green]Authentication Successful![/bold green]\n"
                f"Account: {acct_display}\n"
                f"State: {manager.current_state.value}\n"
                f"PRACTICE Verified: {session.is_practice_verified if session else False}\n"
                f"Session held in memory only. Password discarded.",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]Authentication Failed[/bold red]\n"
                f"Error: {manager.last_error or 'Unknown error'}",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@auth_app.command("logout", help="Invalidate current session and clear in-memory state.")
def logout_cmd() -> None:
    """Log out and invalidate active session."""
    import asyncio

    manager = get_auth_manager()
    asyncio.run(manager.logout())
    console.print(
        Panel(
            "[bold green]Successfully logged out.[/bold green]\n"
            "Active session invalidated and erased from process memory.",
            border_style="green",
        )
    )


@auth_app.command(
    "save-credentials",
    help="Explicitly store credentials in Windows Credential Manager.",
)
def save_credentials_cmd(
    opt_in: bool = typer.Option(
        False,
        "--opt-in",
        help="Explicit confirmation to store credentials in Windows vault.",
    ),
) -> None:
    """Store credentials encrypted-at-rest in the Windows Credential Vault."""
    vault = WindowsCredentialVault()

    if not vault.is_secure_backend():
        console.print(
            "[bold red]Error: Windows Credential Vault is unavailable "
            "or insecure backend configured.[/bold red]"
        )
        raise typer.Exit(code=1)

    if not opt_in:
        confirm = typer.confirm(
            "Do you explicitly authorize saving your credentials to the Windows Credential Vault?",
            default=False,
        )
        if not confirm:
            console.print("[yellow]Operation aborted. Credentials were not saved.[/yellow]")
            raise typer.Exit(code=0)
        opt_in = True

    try:
        email = input("IQ Option email to save: ").strip()
        password = getpass.getpass("IQ Option password to save: ")
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Operation aborted.[/yellow]")
        raise typer.Exit(code=1) from None

    if not email or not password:
        console.print("[red]Email and password cannot be empty.[/red]")
        raise typer.Exit(code=1)

    try:
        vault.save_credentials(email, password, explicit_opt_in=opt_in)
        console.print(
            Panel(
                "[bold green]Credentials stored in Windows Credential Vault![/bold green]\n"
                f"Account: {redact_email(email)}\n"
                f"Target Service: {vault.service_name}\n\n"
                "[dim]Credentials are encrypted with Windows DPAPI tied to your user account.\n"
                "You can now log in using: [cyan]bot auth login --saved[/cyan][/dim]",
                border_style="green",
            )
        )
    except InsecureKeyringBackendError as err:
        console.print(f"[bold red]Storage rejected:[/bold red] {err}")
        raise typer.Exit(code=1) from err
    except Exception as exc:
        console.print(f"[bold red]Failed to save credentials:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@auth_app.command(
    "forget-credentials",
    help="Delete saved credentials from Windows Credential Manager.",
)
def forget_credentials_cmd() -> None:
    """Delete credentials from Windows Credential Vault."""
    vault = WindowsCredentialVault()
    if not vault.has_saved_credentials():
        console.print("[yellow]No saved credentials found in Windows Credential Vault.[/yellow]")
        return

    saved_email = vault.get_saved_identifier(redact=True)
    confirm = typer.confirm(
        f"Are you sure you want to delete saved credentials for {saved_email}?",
        default=True,
    )
    if not confirm:
        console.print("[yellow]Deletion cancelled.[/yellow]")
        return

    deleted = vault.delete_saved_credentials()
    if deleted:
        console.print(
            Panel(
                "[bold green]Saved credentials successfully removed from "
                "Windows Credential Vault.[/bold green]",
                border_style="green",
            )
        )
    else:
        console.print("[yellow]No credentials were removed.[/yellow]")


@auth_app.command(
    "check-connection", help="Validate active session against read-only broker probe."
)
def check_connection_cmd() -> None:
    """Verify active session validity and read-only broker connection."""
    manager = get_auth_manager()
    if not manager.is_authenticated():
        console.print(
            "[yellow]No active authenticated session.[/yellow]\n"
            "Please run [bold green]bot auth login[/bold green] first."
        )
        raise typer.Exit(code=1)

    session = manager.session
    if session is None:
        console.print("[red]Session expired or invalid.[/red]")
        raise typer.Exit(code=1)

    table = Table(
        title="Session Connection Validation",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Check", style="bold white", width=32)
    table.add_column("Result", style="bold", width=48)

    table.add_row("Session Account", redact_email(session.email))
    table.add_row("Session Validity", "Valid" if session.is_active else "Expired")
    table.add_row("Session Age", f"{session.age_seconds():.1f}s")
    table.add_row("Server Account Type", session.server_account_type)
    table.add_row("PRACTICE Verified", str(session.is_practice_verified))
    table.add_row("Broker Order Ledger", "[bold red]BLOCKED[/bold red]")

    console.print(table)
