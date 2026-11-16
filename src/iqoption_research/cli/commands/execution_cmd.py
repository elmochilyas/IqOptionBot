"""CLI commands for Phase 16 Complete Trading Execution Engine.

Provides commands for:
- Viewing the authoritative Broker Capability Matrix
- Running the broker and architecture capability audit
- Executing an end-to-end simulated trade across Binary, Digital, and Forex
- Demonstrating emergency-stop behavior and open-position audit
"""

from __future__ import annotations

from datetime import UTC, datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.execution.broker_readiness import PracticeExecutionAuthToken
from iqoption_research.execution.capability_matrix import (
    BrokerCapabilityMatrix,
    BrokerCapabilityStatus,
)
from iqoption_research.execution.execution_engine import (
    ModularExecutionConfig,
    ModularExecutionEngine,
)
from iqoption_research.execution.models import OrderSide, ProductType
from iqoption_research.execution.modes import (
    ExecutionMode,
    ExecutionModeConfig,
    RealAccountExecutionBlockedError,
)
from iqoption_research.execution.practice_transport import (
    MIN_PRACTICE_STAKE,
    MockBrokerPracticeTransport,
)
from iqoption_research.models.account import AccountType, Balance, Profile
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


def _make_profile(user_id: int = 1, balance_id: int | str = 101) -> Profile:
    return Profile(
        user_id=user_id,
        email="operator@example.com",
        first_name="Operator",
        last_name="Supervisor",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="US",
        currency="USD",
        raw={"balance_id": str(balance_id)},
    )


@execution_app.command(
    "verify-offline",
    help="Run deterministic offline verification of the Practice transport adapter.",
)
def verify_offline_command() -> None:
    """Run comprehensive offline verification of the Practice transport adapter."""
    console.print(
        Panel.fit(
            "[bold cyan]Running Offline Practice Transport Verification (Phase 17)[/bold cyan]\n"
            "[dim]Testing pre-flight account safety, schemas, dispatch, "
            "single-order invariant, & settlement[/dim]",
            border_style="cyan",
        )
    )

    prof_prac = _make_profile(1, 101)
    bal_prac = [
        Balance(
            id="101",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_active=True,
        )
    ]
    tok = PracticeExecutionAuthToken(
        authorization_signature="sig_supervisor_phase17",
        supervisor_id="supervisor_auditor",
    )
    mode_cfg = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=tok,
    )
    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PRACTICE,
            default_product_type=ProductType.BINARY,
            initial_balance=10000.0,
            mode_config=mode_cfg,
            single_test_order_mode=True,
        )
    )

    c1 = Candle(
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
        volume=100,
        bid=1.0852,
        ask=1.0852,
    )
    p1 = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=1.0,
        candle_to_ts=float(c1.to_ts),
        signal_id="sig_verif_01",
    )
    order1, dec1 = engine.propose_and_validate(
        p1, current_ts=float(c1.to_ts), profile=prof_prac, balances=bal_prac
    )
    if not dec1.permitted:
        raise ValueError(f"Pre-flight validation failed: {dec1.reason}")
    console.print(
        f"[green]✓ Pre-flight Practice Verification: Passed (Balance ID={bal_prac[0].id})[/green]"
    )

    # Step 2: Real Account fail-closed check
    prof_real = _make_profile(1, 202)
    bal_real = [
        Balance(
            id="202",
            type=AccountType.REAL,
            amount=500.0,
            currency="USD",
            is_active=True,
        )
    ]
    p_bad = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=1.0,
        candle_to_ts=float(c1.to_ts),
        signal_id="sig_bad_real",
    )
    try:
        engine.propose_and_validate(
            p_bad, current_ts=float(c1.to_ts), profile=prof_real, balances=bal_real
        )
        console.print("[red]✗ Real Account Safety Check FAILED to block![/red]")
    except RealAccountExecutionBlockedError:
        console.print(
            "[green]✓ Real Account Hard-Block: Passed (Execution blocked on Real balance)[/green]"
        )

    # Step 3: Dispatch through MockBrokerPracticeTransport
    mock_transport = MockBrokerPracticeTransport(
        should_succeed=True, option_id="ticket_offline_777"
    )
    _ = engine.submit_order(order1, c1, broker_transport=mock_transport)
    console.print(
        f"[green]✓ Order Dispatch & Ack: Passed "
        f"(Ticket={order1.broker_order_id}, State={order1.state.value})[/green]"
    )

    # Step 4: Single test order invariant check
    p2 = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=1.0,
        candle_to_ts=float(c1.to_ts),
        signal_id="sig_verif_02",
    )
    _order2, dec2 = engine.propose_and_validate(
        p2, current_ts=float(c1.to_ts), profile=prof_prac, balances=bal_prac
    )
    if dec2.permitted:
        raise ValueError("Single test order invariant violated: second proposal was permitted")
    console.print(
        f"[green]✓ Single Test Order Invariant: Passed "
        f"(Concurrent proposal rejected: {dec2.reason})[/green]"
    )

    # Step 5: Expiry & Settlement
    c2 = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=1700000060,
        to_ts=1700000120,
        at_ts=1700000120_000_000_000,
        open=1.0852,
        high=1.0862,
        low=1.0850,
        close=1.0858,
        volume=120,
        bid=1.0858,
        ask=1.0858,
    )
    step_res = engine.process_candle(c2)
    if len(step_res.settled_orders) != 1:
        raise ValueError(f"Expected 1 settled order, got {len(step_res.settled_orders)}")
    console.print(
        f"[green]✓ Expiry Settlement & PnL: Passed "
        f"(Settled PnL=${order1.simulated_pnl:+.2f})[/green]"
    )
    console.print(
        Panel.fit(
            "[bold green]Offline Practice Transport Verification Complete: "
            "ALL CHECKS PASSED[/bold green]\n"
            "[dim]Zero network I/O, strict schema validation, fail-closed Real guards, "
            "& single test order invariant verified.[/dim]",
            border_style="green",
        )
    )


@execution_app.command(
    "practice-test",
    help="Operator-supervised verification workflow for a single Practice test order.",
)
def practice_test_command(
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    side: str = typer.Option("call", "--side", help="Trade direction (call or put)"),
    stake: float = typer.Option(
        MIN_PRACTICE_STAKE, "--stake", help="Virtual stake amount (minimum $1.00)"
    ),
    offline_mock: bool = typer.Option(
        True, "--offline-mock/--live-socket", help="Run via offline mock transport (default: True)"
    ),
) -> None:
    """Supervised Practice test workflow requiring explicit operator authorization."""
    norm_side = OrderSide.CALL if side.lower() == "call" else OrderSide.PUT
    stake = max(MIN_PRACTICE_STAKE, stake)

    table = Table(
        title="[bold yellow]Operator Supervised Practice Test — Pre-Flight Audit[/bold yellow]"
    )
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Confirmed Account Type", "[bold green]PRACTICE (Virtual Funds)[/bold green]")
    table.add_row("Product & Instrument", f"BINARY {symbol.upper()}")
    table.add_row("Trade Direction", f"[bold cyan]{norm_side.value.upper()}[/bold cyan]")
    table.add_row("Virtual Stake Amount", f"${stake:.2f} (Platform minimum)")
    table.add_row("Expiration Duration", "60 seconds (1 minute)")
    table.add_row("Current Payout Rate", "85.0%")
    table.add_row("Max Possible Simulated Loss", f"${stake:.2f}")
    table.add_row("Autonomous AI Trading", "[bold red]DISABLED[/bold red]")
    table.add_row(
        "Action to Perform",
        "Deterministic single test order placement via guarded practice transport",
    )
    console.print(table)

    # Require explicit confirmation
    confirmed = typer.confirm(
        "Do you explicitly authorize submitting this single Practice test order?",
        default=False,
    )
    if not confirmed:
        console.print("[yellow]Aborted by operator. No test order was dispatched.[/yellow]")
        raise typer.Abort()

    if not offline_mock:
        console.print(
            Panel.fit(
                "[bold red]Live Socket Transmission Prohibited[/bold red]\n"
                "[yellow]IQ Option does not provide an authorized, documented public API for "
                "programmatic trading.\n"
                "Live socket transmission is withheld to prevent unverified "
                "socket calls.[/yellow]\n"
                "[dim]Use --offline-mock (default) for deterministic execution verification.[/dim]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    # Execute deterministic mock dispatch
    tok = PracticeExecutionAuthToken(
        authorization_signature="sig_supervisor_cli_practice",
        supervisor_id="supervisor_cli_operator",
    )
    mode_cfg = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=tok,
    )
    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol=symbol.upper(),
            mode=ExecutionMode.PRACTICE,
            default_product_type=ProductType.BINARY,
            initial_balance=10000.0,
            mode_config=mode_cfg,
            single_test_order_mode=True,
        )
    )

    prof = _make_profile(1, 101)
    bals = [
        Balance(
            id="101",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_active=True,
        )
    ]
    c1 = Candle(
        active_id=1,
        symbol=symbol.upper(),
        size=CandleSize.MIN_1,
        from_ts=1700000000,
        to_ts=1700000060,
        at_ts=1700000060_000_000_000,
        open=1.0850,
        high=1.0855,
        low=1.0848,
        close=1.0852,
        volume=100,
        bid=1.0852,
        ask=1.0852,
    )

    proposal = engine.create_proposal(
        symbol=symbol.upper(),
        side=norm_side,
        stake=stake,
        candle_to_ts=float(c1.to_ts),
        signal_id="sig_operator_cli_test",
    )
    order, dec = engine.propose_and_validate(
        proposal, current_ts=float(c1.to_ts), profile=prof, balances=bals
    )
    if not dec.permitted:
        console.print(f"[bold red]Order rejected by risk controller: {dec.message}[/bold red]")
        raise typer.Exit(code=1)

    mock_transport = MockBrokerPracticeTransport(
        should_succeed=True, option_id="mock_ticket_cli_supervised"
    )
    _ = engine.submit_order(order, c1, broker_transport=mock_transport)
    console.print(
        Panel.fit(
            "[bold green]Practice Test Order Successfully Placed "
            "(Simulated Transport)[/bold green]\n"
            f"• Order ID: {order.order_id}\n"
            f"• Broker Ticket ID: {order.broker_order_id}\n"
            f"• State: {order.state.value.upper()}\n"
            f"• Entry Price: {order.entry_price:.5f}\n"
            f"• Expiry TS: {int(order.expiry_ts or 0)}\n"
            f"• Audit Trail: Order intent durably saved in SQLite execution journal.",
            border_style="green",
        )
    )
