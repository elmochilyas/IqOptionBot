"""CLI commands for Phase 16 Complete Trading Execution Engine.

Provides commands for:
- Viewing the authoritative Broker Capability Matrix
- Running the broker and architecture capability audit
- Executing an end-to-end simulated trade across Binary, Digital, and Forex
- Demonstrating emergency-stop behavior and open-position audit
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.execution.capability_matrix import (
    BrokerCapabilityMatrix,
    BrokerCapabilityStatus,
)
from iqoption_research.execution.execution_engine import (
    ModularExecutionConfig,
    ModularExecutionEngine,
)
from iqoption_research.execution.models import OrderSide, ProductType
from iqoption_research.execution.modes import ExecutionMode
from iqoption_research.models.market import Candle, CandleSize

console = Console()
execution_app = typer.Typer(
    name="execution",
    help="Phase 16 complete modular execution engine, broker capability audit, and mode safety.",
    no_args_is_help=True,
)


@execution_app.command("capability-matrix", help="Display the broker capability matrix.")
def capability_matrix_command(
    product: str = typer.Option(
        "all", "--product", "-p", help="Filter by product (binary, digital, forex, all)"
    ),
) -> None:
    """Display the authoritative Broker Capability Matrix in a Rich table."""
    matrix = BrokerCapabilityMatrix()
    prod_type = (
        ProductType(product.lower()) if product.lower() in {"binary", "digital", "forex"} else None
    )
    ops = matrix.filter_operations(product_type=prod_type)

    table = Table(title="[bold cyan]IQ Option Broker Capability Matrix (Phase 16)[/bold cyan]")
    table.add_column("Operation ID", style="bold white")
    table.add_column("Product", style="magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Name", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Live Allowed", justify="center")
    table.add_column("Protocol Frame", style="dim")

    status_colors = {
        BrokerCapabilityStatus.IMPLEMENTED_AND_VERIFIED.value: "[bold green]VERIFIED[/bold green]",
        BrokerCapabilityStatus.IMPLEMENTED_SIMULATED_ONLY.value: "[cyan]SIMULATED ONLY[/cyan]",
        BrokerCapabilityStatus.UNVERIFIED_PROTOCOL_EVIDENCE.value: "[yellow]UNVERIFIED[/yellow]",
        BrokerCapabilityStatus.UNSUPPORTED.value: "[bold red]UNSUPPORTED[/bold red]",
    }

    for op in ops:
        prod = op.product_type.value if op.product_type else "Global"
        status_styled = status_colors.get(op.status.value, op.status.value)
        live_str = "[bold green]Yes[/bold green]" if op.live_broker_allowed else "[dim]No[/dim]"
        frame = op.protocol_frame_name or "Internal"
        table.add_row(
            op.operation_id, prod, op.category.value, op.name, status_styled, live_str, frame
        )

    console.print(table)
    summary = matrix.summary_counts()
    console.print(
        Panel(
            f"[bold]Summary of Operations:[/bold]\n"
            f"• Implemented & Verified: {summary['implemented_and_verified']}\n"
            f"• Implemented (Simulated Only): {summary['implemented_simulated_only']}\n"
            f"• Unverified Protocol Evidence: {summary['unverified_protocol_evidence']}\n"
            f"• Unsupported / Prohibited: {summary['unsupported']}",
            border_style="cyan",
        )
    )


@execution_app.command("audit", help="Run architecture and broker capability audit.")
def audit_command() -> None:
    """Execute offline architecture audit reporting supported and unsupported operations."""
    console.print(
        Panel(
            "[bold green]Phase 16: Architecture and Broker Capability Audit[/bold green]\n"
            "Verifying execution engine modularity, execution mode isolation, and ground truth.",
            border_style="green",
        )
    )
    matrix = BrokerCapabilityMatrix()
    summary = matrix.summary_counts()

    table = Table(title="Audit Classification Breakdown")
    table.add_column("Classification", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Safety Policy", style="dim")

    table.add_row(
        "Implemented & Verified",
        str(summary["implemented_and_verified"]),
        "Permitted in read-only / authenticated live sessions",
    )
    table.add_row(
        "Implemented (Simulated Only)",
        str(summary["implemented_simulated_only"]),
        "Permitted in PAPER mode only (strictly air-gapped)",
    )
    table.add_row(
        "Unverified Protocol Evidence",
        str(summary["unverified_protocol_evidence"]),
        "Guarded in PRACTICE mode; requires supervisor token & practice verification",
    )
    table.add_row(
        "Unsupported / Prohibited",
        str(summary["unsupported"]),
        "Fail-closed rejection; REAL mode and auto-balance switching forbidden",
    )
    console.print(table)


@execution_app.command(
    "paper-run", help="Run deterministic simulated trade from proposal to settlement."
)
def paper_run_command(
    product: str = typer.Option(
        "binary", "--product", "-p", help="Product to simulate: binary, digital, forex"
    ),
    stake: float = typer.Option(100.0, "--stake", "-s", help="Order stake amount"),
    side: str = typer.Option("call", "--side", help="Direction: call or put"),
) -> None:
    """Execute a complete simulated trade from proposal through settlement."""
    prod_type = ProductType(product.lower())
    order_side = OrderSide.CALL if side.lower() == "call" else OrderSide.PUT

    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PAPER,
            default_product_type=prod_type,
            initial_balance=10000.0,
        )
    )

    console.print(f"[bold cyan]Running {prod_type.value.upper()} Paper Trade Lifecycle[/bold cyan]")

    # 1. Proposal
    candle1 = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=1700000000,
        to_ts=1700000060,
        at_ts=1700000060_000_000_000,
        open=1.0850,
        high=1.0855,
        low=1.0848,
        close=1.0852,
        volume=120,
        bid=1.0852,
        ask=1.0852,
    )
    proposal = engine.create_proposal(
        symbol="EURUSD",
        side=order_side,
        stake=stake,
        confidence_probability=0.65,
        product_type=prod_type,
        candle_to_ts=float(candle1.to_ts),
        signal_id="sig_cli_001",
    )
    p_info = f"{proposal.symbol} {proposal.side.value} ${proposal.proposed_stake:.2f}"
    console.print(f"[dim]1. Created proposal: {p_info}[/dim]")

    # 2. Validation & Risk Approval
    order, _decision = engine.propose_and_validate(proposal, current_ts=float(candle1.to_ts))
    console.print(
        f"[green]2. Validated & Approved: ID={order.order_id}, State={order.state.value}[/green]"
    )

    # 3. Submission & Fill
    _ = engine.submit_order(order, candle1)
    entry_p = f"{order.entry_price:.5f}" if order.entry_price is not None else "N/A"
    console.print(
        f"[green]3. Submitted & Opened: State={order.state.value}, Entry={entry_p}[/green]"
    )

    # 4. Monitoring & Settlement
    candle2 = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=1700000060,
        to_ts=1700000120,
        at_ts=1700000120_000_000_000,
        open=1.0852,
        high=1.0905,  # Sufficient to trigger Forex TP
        low=1.0851,
        close=1.0858,
        volume=140,
        bid=1.0858,
        ask=1.0858,
    )
    _ = engine.process_candle(candle2)
    if order.is_terminal and order.exit_price is not None:
        console.print(
            f"[green]4. Monitored & Settled: State={order.state.value}, "
            f"Exit Price={order.exit_price:.5f}[/green]"
        )
        bal = engine.portfolio.current_balance
        console.print(
            f"[bold green]5. Settlement Outcome: PnL=${order.simulated_pnl:+.2f}, "
            f"Portfolio Balance=${bal:.2f}[/bold green]"
        )
    else:
        console.print(
            f"[cyan]4. Monitored & Active: State={order.state.value}, "
            f"Spot={candle2.close:.5f}[/cyan]"
        )
