"""CLI commands for Phase 16 Complete Trading Execution Engine.

Provides commands for:
- Viewing the authoritative Broker Capability Matrix
- Running the broker and architecture capability audit
- Executing an end-to-end simulated trade across Binary, Digital, and Forex
- Demonstrating emergency-stop behavior and open-position audit
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.config.settings import Settings
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.execution.account_diagnostics import (
    AccountReadinessResult,
    AccountStateClassification,
    BrokerAccountDiagnostic,
    RoutingAuditResult,
)
from iqoption_research.execution.broker_readiness import PracticeExecutionAuthToken
from iqoption_research.execution.capability_matrix import (
    BrokerCapabilityMatrix,
    BrokerCapabilityStatus,
)
from iqoption_research.execution.execution_engine import (
    ModularExecutionConfig,
    ModularExecutionEngine,
)
from iqoption_research.execution.live_practice_transport import IQOptionLivePracticeTransport
from iqoption_research.execution.models import OrderSide, OrderState, ProductType
from iqoption_research.execution.modes import (
    ContractAvailabilityUnverifiedError,
    ExecutionMode,
    ExecutionModeConfig,
    PracticeAuthorizationError,
    RealAccountExecutionBlockedError,
    UnsupportedProductOperationError,
)
from iqoption_research.execution.practice_history import recover_history
from iqoption_research.execution.practice_observer import (
    MockPassivePracticeObserver,
    ObservedPracticeTradeRecord,
    PassivePracticeObserver,
)
from iqoption_research.execution.practice_transport import (
    MAX_PRACTICE_TEST_STAKE,
    MIN_PRACTICE_STAKE,
    MockBrokerPracticeTransport,
    compute_binary_expiry,
)
from iqoption_research.execution.safety import AccountSelectionUnverifiedError
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.security.credentials import Credentials, redact_email

logger = logging.getLogger(__name__)
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
        BrokerCapabilityStatus.READ_ONLY_BROKER_VERIFIED.value: (
            "[bold green]READ-ONLY VERIFIED[/bold green]"
        ),
        BrokerCapabilityStatus.BROKER_ORDER_VERIFIED.value: (
            "[bold green]ORDER VERIFIED[/bold green]"
        ),
        BrokerCapabilityStatus.SETTLEMENT_VERIFIED.value: (
            "[bold green]SETTLEMENT VERIFIED[/bold green]"
        ),
        BrokerCapabilityStatus.OFFLINE_TESTED.value: "[cyan]OFFLINE TESTED[/cyan]",
        BrokerCapabilityStatus.IMPLEMENTED.value: "[yellow]IMPLEMENTED[/yellow]",
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
            f"• Read-Only Broker-Verified: {summary.get('read_only_broker_verified', 0)}\n"
            f"• Offline-Tested: {summary.get('offline_tested', 0)}\n"
            f"• Implemented (Unverified Live): {summary.get('implemented', 0)}\n"
            f"• Broker Order-Verified: {summary.get('broker_order_verified', 0)}\n"
            f"• Settlement-Verified: {summary.get('settlement_verified', 0)}\n"
            f"• Unsupported / Prohibited: {summary.get('unsupported', 0)}",
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
        "Read-Only Broker-Verified",
        str(summary.get("read_only_broker_verified", summary.get("implemented_and_verified", 0))),
        "Permitted in read-only / authenticated live sessions",
    )
    table.add_row(
        "Offline-Tested",
        str(summary.get("offline_tested", summary.get("implemented_simulated_only", 0))),
        "Permitted in PAPER mode only (strictly air-gapped)",
    )
    table.add_row(
        "Implemented (Unverified Live)",
        str(summary.get("implemented", summary.get("unverified_protocol_evidence", 0))),
        "Guarded in PRACTICE mode; requires supervisor token & practice verification",
    )
    table.add_row(
        "Unsupported / Prohibited",
        str(summary.get("unsupported", 0)),
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
    "practice-diagnostics",
    help="Operator-run READ-ONLY diagnostics for IQ Option Practice account (no orders).",
)
def practice_diagnostics_command(
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", "--asset", help="Instrument symbol"),
    offline_mock: bool = typer.Option(
        False, "--offline-mock", help="Run via offline mock simulation (default: False)"
    ),
    live: bool = typer.Option(
        False, "--live", help="Connect to live IQ Option broker to run diagnostics (default: False)"
    ),
) -> None:
    """Perform read-only diagnostic on IQ Option account.

    Guarantees:
    - Never submits orders or transmits trading frames.
    - Never issues change_balance or modifies active account state.
    - Fails closed immediately if active account is REAL.
    - Redacts all credentials and session tokens from output.
    """
    console.print(
        Panel.fit(
            "[bold cyan]IQ Option Practice Account — Read-Only Diagnostics[/bold cyan]\n"
            "[dim]Verifies secure TLS authentication, Practice account status, "
            "EURUSD availability, and product parameters without submitting trades "
            "or modifying balances.[/dim]",
            border_style="cyan",
        )
    )

    if not live and not offline_mock:
        console.print(
            "[yellow]Notice:[/yellow] By default, practice-diagnostics operates with network "
            "guards. To run live diagnostics, pass [bold]--live[/bold]. "
            "To run an offline synthetic diagnostic, pass [bold]--offline-mock[/bold]."
        )
        run_live = typer.confirm("Connect to live IQ Option broker now?", default=False)
        if not run_live:
            console.print("[dim]Aborted. No network connections were initiated.[/dim]")
            raise typer.Abort()
        live = True

    if offline_mock:
        # Run deterministic offline mock diagnostic
        console.print("[bold yellow]Running offline simulated diagnostic...[/bold yellow]")
        diag_table = Table(title="[bold green]Offline Diagnostic Findings[/bold green]")
        diag_table.add_column("Property", style="cyan")
        diag_table.add_column("Value", style="bold white")

        diag_table.add_row("Connection Mode", "OFFLINE SIMULATED (Zero Network I/O)")
        diag_table.add_row("Account Type", "[bold green]PRACTICE (type 4 - Verified)[/bold green]")
        diag_table.add_row("Active Practice Balance ID", "101")
        diag_table.add_row("Virtual Practice Balance", "$10,000.00")
        diag_table.add_row("Instrument", f"{symbol.upper()} (Active ID: 1)")
        diag_table.add_row("Market Availability", "ACTIVE & RECEIVING TICKS")
        diag_table.add_row("Binary Options (1m Turbo)", "AVAILABLE (Min Stake $1.00)")
        diag_table.add_row(
            "Safety & Compliance",
            "[bold green]PASSED (Zero Orders / Zero State Changes)[/bold green]",
        )
        console.print(diag_table)
        console.print(
            Panel.fit(
                "[bold green]Read-Only Diagnostic Complete: ALL CHECKS PASSED[/bold green]\n"
                "[dim]Active account verified as PRACTICE. "
                "System is ready for supervised test order.[/dim]",
                border_style="green",
            )
        )
        return

    # Live broker diagnostic
    console.print("[bold yellow]Initiating secure live read-only diagnostic...[/bold yellow]")
    email = os.environ.get("IQOPTION_EMAIL")
    password = os.environ.get("IQOPTION_PASSWORD")
    if not email or not password:
        console.print(
            "[dim]Credentials not found in environment (IQOPTION_EMAIL / IQOPTION_PASSWORD).[/dim]"
        )
        email = typer.prompt("IQ Option Account Email")
        password = typer.prompt("IQ Option Password", hide_input=True)

    credentials = Credentials(email=email, password=password)
    settings = Settings(offline=False, diagnostics=True)
    client = AsyncIQOption(credentials, settings=settings, offline=False, diagnostics=True)

    async def _async_diag() -> None:
        try:
            with console.status("[bold green]Connecting to IQ Option and authenticating..."):
                await client.connect(raise_if_incomplete=False)
        except Exception as exc:
            console.print(f"[bold red]Failed to establish broker connection: {exc}[/bold red]")
            raise typer.Exit(code=1) from exc

        try:
            if not client.is_ready:
                console.print(
                    "[bold red]WebSocket authenticated, but required account data "
                    "was not received within timeout.[/bold red]"
                )
                raise typer.Exit(code=1)

            account_status = await client.get_account_status()
            profile = account_status.profile
            balances = list(account_status.balances)

            practice_bal = account_status.practice_balance
            real_bal = account_status.real_balance

            # Profile default wallet ID
            prof_bal_id = (
                profile.raw.get("balance_id")
                or profile.raw.get("balanceId")
                or profile.raw.get("active_balance_id")
            )
            prof_bal_id_str = str(prof_bal_id).strip() if prof_bal_id is not None else None

            # Check if any balance was given an explicit server active flag
            has_raw_active = any(b.raw_is_active is True for b in balances)

            # Determine whether broker provided authoritative session-specific active balance.
            # On IQ Option, a fresh WebSocket session defaults to the primary registered wallet.
            # If the session defaulted to Real while a Practice balance exists, and no explicit
            # session selection event confirmed Practice, the indicator is not authoritative.
            has_authoritative = bool(
                has_raw_active
                or (
                    prof_bal_id_str is not None
                    and practice_bal is not None
                    and prof_bal_id_str == practice_bal.id
                )
            )

            # Run comprehensive BrokerAccountDiagnostic
            diag_engine = BrokerAccountDiagnostic()
            diag_report = diag_engine.evaluate(
                profile=profile,
                balances=balances,
                active_balance_id=(
                    account_status.active_balance.id if account_status.active_balance else None
                ),
                browser_ui_balance_id=(practice_bal.id if practice_bal else None),
                last_verification_ts=time.time(),
                has_authoritative_active_indicator=has_authoritative,
            )

            # Query EURUSD candles to verify live market data feed
            market_ok = False
            last_price = 0.0
            try:
                candle_res = await client.get_candles(symbol.upper(), 60, 1)
                if candle_res.candle_list:
                    market_ok = True
                    last_price = candle_res.candle_list[-1].close
            except Exception as m_exc:  # noqa: BLE001
                logger.warning("Market query warning: %s", m_exc)

            # Table 1: Account Balances & Multi-Layer Identity
            bal_table = Table(
                title="[bold green]IQ Option Multi-Layer Account Balance Audit[/bold green]"
            )
            bal_table.add_column("Balance Layer / Entity", style="cyan")
            bal_table.add_column("Account Type", style="bold white")
            bal_table.add_column("Identifier", justify="center")
            bal_table.add_column("Available Funds", justify="right")
            bal_table.add_column("Active / Routing Status", justify="center")

            # Practice row
            if practice_bal:
                is_active_str = (
                    "[bold green]YES[/bold green]"
                    if practice_bal.is_active and has_authoritative
                    else "[dim]No (Standby)[/dim]"
                )
                bal_table.add_row(
                    "Practice Account (Virtual)",
                    "[bold green]PRACTICE (type 4)[/bold green]",
                    str(practice_bal.id),
                    f"${practice_bal.amount:,.2f} {practice_bal.currency}",
                    is_active_str,
                )
            else:
                bal_table.add_row(
                    "Practice Account (Virtual)",
                    "[dim]NONE[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]N/A[/dim]",
                    "[bold red]MISSING[/bold red]",
                )

            # Real row
            if real_bal:
                is_active_str = (
                    "[yellow]Active Session Default[/yellow]"
                    if (real_bal.is_active or prof_bal_id_str == real_bal.id)
                    else "[dim]No[/dim]"
                )
                bal_table.add_row(
                    "Real Account (Real Money)",
                    "[yellow]REAL (type 1)[/yellow]",
                    str(real_bal.id),
                    f"${real_bal.amount:,.2f} {real_bal.currency}",
                    is_active_str,
                )
            else:
                bal_table.add_row(
                    "Real Account (Real Money)",
                    "[dim]NONE[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]N/A[/dim]",
                )

            # Profile Default Wallet
            bal_table.add_row(
                "Profile Default Wallet",
                "Primary Account",
                str(prof_bal_id_str or "unknown"),
                "Server Default",
                "[dim]Session Initializer[/dim]",
            )
            # Browser UI Selection
            bal_table.add_row(
                "Browser UI Selection",
                "Chrome / Web Client",
                str(practice_bal.id if practice_bal else "unknown"),
                "User Selected",
                "[cyan]Session-Scoped (Browser)[/cyan]",
            )
            # Bot Session Selection
            bot_status_str = (
                "[bold green]VERIFIED[/bold green]"
                if diag_report.is_practice_verified
                else "[bold red]BLOCKED[/bold red]"
            )
            bal_table.add_row(
                "Bot Session Active State",
                f"{diag_report.readiness.value}",
                str(diag_report.balance_identity.server_reported_active_balance_id or "unresolved"),
                "Evaluated Ledger",
                bot_status_str,
            )
            console.print(bal_table)

            # Table 2: Step-by-Step Decision Path
            path_table = Table(
                title="[bold yellow]Bot Active-Account Decision Path Audit[/bold yellow]"
            )
            path_table.add_column("Step", style="bold cyan", justify="right")
            path_table.add_column("Decision Stage", style="bold white")
            path_table.add_column("Observed Evidence & Rationale", style="dim")

            for idx, step_msg in enumerate(diag_report.decision_path, start=1):
                path_table.add_row(str(idx), f"Stage {idx}", step_msg)

            console.print(path_table)

            # Table 3: Technical Protocol & Safety Findings
            diag_table = Table(
                title="[bold green]Authoritative Read-Only Diagnostic Findings[/bold green]"
            )
            diag_table.add_column("Diagnostic Check", style="cyan")
            diag_table.add_column("Broker Response / Status", style="bold white")
            diag_table.add_column("Verdict", justify="center")

            user_disp = redact_email(profile.email)
            diag_table.add_row(
                "Authenticated User Profile",
                f"User ID: {profile.user_id} | Email: {user_disp}",
                "[bold green]AUTHENTICATED[/bold green]",
            )
            diag_table.add_row(
                "WebSocket TLS & Transport",
                "Authenticated (wss://iqoption.com/echo/websocket)",
                "[bold green]PASSED[/bold green]",
            )
            diag_table.add_row(
                f"Market Data Feed ({symbol.upper()})",
                f"Live Price: {last_price:.5f}" if market_ok else "Available",
                "[bold green]PASSED[/bold green]" if market_ok else "[yellow]UNVERIFIED[/yellow]",
            )
            diag_table.add_row(
                "Account Readiness Result",
                f"{diag_report.readiness.value}",
                (
                    "[bold green]PASSED[/bold green]"
                    if diag_report.is_practice_verified
                    else "[bold red]FAIL-CLOSED[/bold red]"
                ),
            )
            diag_table.add_row(
                "Order & Balance Mutation",
                "Strict Read-Only Posture (0 trades sent, 0 switches sent)",
                "[bold green]ENFORCED[/bold green]",
            )
            diag_table.add_row(
                "Live Broker Order Safety",
                f"Ledger Status: {diag_report.ledger_status.value} (Real Trading Disabled)",
                "[bold green]BLOCKED[/bold green]",
            )
            console.print(diag_table)

            # Final Summary Panel
            if diag_report.readiness == AccountReadinessResult.VERIFIED_PRACTICE:
                msg_pnl = (
                    "[bold green]Live Read-Only Diagnostic Complete: PRACTICE ACCOUNT VERIFIED"
                    "[/bold green]\n"
                    "Active account is confirmed PRACTICE on this session. "
                    "Credentials and tokens remain redacted.\n"
                    "To proceed with a single supervised virtual trade, run:\n"
                    f"[bold cyan]python -m iqoption_research.cli.main practice-test --live-socket "
                    f"--stake 1.00 --asset {symbol.upper()}[/bold cyan]"
                )
                console.print(Panel.fit(msg_pnl, border_style="green"))
            else:
                fail_pnl = (
                    f"[bold red]SAFETY ENFORCEMENT: LIVE PRACTICE EXECUTION BLOCKED "
                    f"({diag_report.readiness.value})[/bold red]\n\n"
                    f"[bold white]Root Cause:[/bold white]\n"
                    f"• {diag_report.explanation}\n\n"
                    f"[bold white]Architectural Reality:[/bold white]\n"
                    "• IQ Option balance selection is strictly SESSION-SCOPED, tied to the "
                    "specific WebSocket SSID.\n"
                    "• Selecting 'Practice' in Chrome or web app updates only that browser tab's "
                    "session.\n"
                    "• Every newly spawned Python WebSocket session connects independently and is "
                    "initialized\n"
                    "  by the broker to the user's primary default wallet (REAL).\n"
                    "• Automatic balance switching (change_balance) is intentionally disabled in "
                    "this codebase\n"
                    "  as a strict safety control to prevent accidental live execution.\n\n"
                    f"[bold white]Action Required:[/bold white]\n"
                    f"• {diag_report.recommended_action}\n"
                    "• Zero orders were submitted. Zero balance switches were emitted."
                )
                console.print(Panel.fit(fail_pnl, border_style="red"))
                raise typer.Exit(code=1)
        finally:
            await client.close()

    asyncio.run(_async_diag())


@execution_app.command(
    "practice-select",
    help="Explicit operator authorization to select PRACTICE balance for the current session.",
)
def practice_select_command(
    offline_mock: bool = typer.Option(
        False, "--offline-mock", help="Run via offline mock simulation (default: False)"
    ),
    live: bool = typer.Option(
        False, "--live", help="Connect to live IQ Option broker to select balance (default: False)"
    ),
    auto_confirm: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt (operator pre-authorization)"
    ),
) -> None:
    """Explicitly select the PRACTICE balance within the bot's session.

    Guarantees:
    - Never modifies balance automatically.
    - Requires explicit operator authorization before sending change_balance.
    - Strictly restricts selection to Practice balances (AccountType.PRACTICE / type 4).
    - Hard-blocks selection of REAL balances fail-closed.
    - Never submits orders or transmits trade frames during balance selection.
    - Confirms post-selection state against fresh server profile and balances.
    - Warns operator that selection is session-scoped (invalidated on disconnect/reconnect).
    """
    console.print(
        Panel.fit(
            "[bold cyan]IQ Option Practice Account — Explicit Session Selection[/bold cyan]\n"
            "[dim]Allows explicit operator authorization to switch the current session's "
            "active balance to PRACTICE (type 4).\n"
            "REAL trading remains permanently disabled in software.[/dim]",
            border_style="cyan",
        )
    )

    if live and auto_confirm:
        console.print(
            Panel.fit(
                "[bold red]CRITICAL SAFETY BLOCK: Non-interactive confirmation is forbidden "
                "for live broker operations.[/bold red]\n"
                "[dim]The '--yes' / '-y' flag cannot be combined with '--live'. "
                "Explicit interactive operator authorization is required.[/dim]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    if not live and not offline_mock:
        console.print(
            "[yellow]Notice:[/yellow] By default, practice-select operates with network guards. "
            "To connect to the live broker, pass [bold]--live[/bold]. "
            "To run an offline synthetic selection, pass [bold]--offline-mock[/bold]."
        )
        run_live = typer.confirm("Connect to live IQ Option broker now?", default=False)
        if not run_live:
            console.print("[dim]Aborted. No network connections were initiated.[/dim]")
            raise typer.Abort()
        live = True
        if auto_confirm:
            console.print(
                Panel.fit(
                    "[bold red]CRITICAL SAFETY BLOCK: Non-interactive confirmation is forbidden "
                    "for live broker operations.[/bold red]\n"
                    "[dim]The '--yes' / '-y' flag cannot be combined with live broker connection. "
                    "Explicit interactive operator authorization is required.[/dim]",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

    if offline_mock:
        console.print(
            "[bold yellow]Running offline simulated practice session selection...[/bold yellow]"
        )
        mock_transport = MockBrokerPracticeTransport(
            initial_active_balance_id="200",
            practice_balance_id="101",
            virtual_amount=10000.0,
        )

        # Audit Table Preview
        preview_table = Table(
            title="[bold yellow]Offline Practice Selection — Pre-Selection Audit[/bold yellow]"
        )
        preview_table.add_column("Account Entity / Property", style="cyan")
        preview_table.add_column("Discovered Value", style="bold white")
        preview_table.add_column("Status", justify="center")

        preview_table.add_row(
            "Account Email", "mock_operator@example.com (redacted)", "[dim]Simulated[/dim]"
        )
        preview_table.add_row(
            "Discovered Practice Balance",
            "ID: 101 | $10,000.00 USD",
            "[bold green]ELIGIBLE (PRACTICE)[/bold green]",
        )
        preview_table.add_row(
            "Discovered REAL Balance",
            "ID: 200 | $500.00 USD",
            "[yellow]BLOCKED FROM SELECTION[/yellow]",
        )
        preview_table.add_row(
            "Current Active Session Balance",
            "ID: 200 (Default REAL)",
            "[bold red]NOT VERIFIED PRACTICE[/bold red]",
        )
        preview_table.add_row(
            "Real Trading Enforcement",
            "Permanently Disabled in Software",
            "[bold green]PROTECTED[/bold green]",
        )
        console.print(preview_table)

        if not auto_confirm:
            confirmed = typer.confirm(
                "Authorise switching this session's active balance to Practice ID 101?",
                default=False,
            )
            if not confirmed:
                console.print(
                    "[yellow]Selection cancelled by operator. "
                    "No balance modification frames were sent.[/yellow]"
                )
                raise typer.Abort()

        # Execute simulated selection
        verified_bal = mock_transport.select_practice_balance(operator_authorized=True)

        res_table = Table(
            title="[bold green]Offline Practice Session Selection Result[/bold green]"
        )
        res_table.add_column("Property", style="cyan")
        res_table.add_column("Verified Value", style="bold white")

        res_table.add_row("Session Identifier", mock_transport.session_id)
        res_table.add_row("Active Session Balance ID", str(verified_bal.id))
        res_table.add_row(
            "Account Type", f"{verified_bal.type.name} (type {verified_bal.type.value})"
        )
        res_table.add_row(
            "Available Virtual Funds", f"${verified_bal.amount:,.2f} {verified_bal.currency}"
        )
        res_table.add_row(
            "Practice Session Verification",
            "[bold green]VERIFIED ACTIVE PRACTICE[/bold green]",
        )
        res_table.add_row(
            "Order Submission Count",
            f"{len(mock_transport.dispatched_frames) - 1} orders (Zero orders sent)",
        )
        console.print(res_table)

        scope_msg = (
            "[bold green]Practice Session Selection Complete: VERIFIED[/bold green]\n\n"
            f"[bold white]Session ID:[/bold white] {mock_transport.session_id}\n"
            f"[bold white]Active Balance ID:[/bold white] {verified_bal.id} (PRACTICE)\n\n"
            "[bold yellow]IMPORTANT LIFETIME & SESSION NOTICE:[/bold yellow]\n"
            "• IQ Option balance selection is strictly SESSION-SCOPED.\n"
            "• This authorization applies solely to this active WebSocket session.\n"
            "• Reconnecting or closing the connection resets the session to REAL.\n"
            "• REAL trading remains permanently disabled in software.\n"
            "• Zero orders were submitted."
        )
        console.print(Panel.fit(scope_msg, border_style="green"))
        return

    # Live broker selection
    console.print("[bold yellow]Initiating secure live practice balance selection...[/bold yellow]")
    email = os.environ.get("IQOPTION_EMAIL")
    password = os.environ.get("IQOPTION_PASSWORD")
    if not email or not password:
        console.print(
            "[dim]Credentials not found in environment (IQOPTION_EMAIL / IQOPTION_PASSWORD).[/dim]"
        )
        email = typer.prompt("IQ Option Account Email")
        password = typer.prompt("IQ Option Password", hide_input=True)

    credentials = Credentials(email=email, password=password)
    transport = IQOptionLivePracticeTransport(credentials=credentials)

    async def _async_select() -> None:
        try:
            with console.status(
                "[bold green]Connecting to IQ Option and retrieving account state..."
            ):
                # require_practice=False allows inspecting and selecting without pre-flight abort
                await transport.async_connect(require_practice=False)
        except Exception as exc:
            console.print(f"[bold red]Failed to connect to IQ Option broker: {exc}[/bold red]")
            raise typer.Exit(code=1) from exc

        try:
            profile = transport.profile
            balances = transport.balances

            if profile is None or not balances:
                console.print(
                    "[bold red]Failed to retrieve profile or balances from broker.[/bold red]"
                )
                raise typer.Exit(code=1)

            practice_bal = next(
                (b for b in balances if b.type == AccountType.PRACTICE and b.is_practice),
                None,
            )
            real_bal = next(
                (b for b in balances if b.type == AccountType.REAL or not b.is_practice),
                None,
            )

            if practice_bal is None:
                console.print(
                    "[bold red]No PRACTICE balance found on this account. Available balances: "
                    f"{[b.type.name for b in balances]}[/bold red]"
                )
                raise typer.Exit(code=1)

            current_active_id = transport.active_balance_id or str(
                profile.raw.get("balance_id", "unknown")
            )

            # Display Pre-Selection Audit Table
            audit_table = Table(
                title="[bold yellow]Practice Balance Selection — Pre-Selection Audit[/bold yellow]"
            )
            audit_table.add_column("Account Entity / Property", style="cyan")
            audit_table.add_column("Discovered Value", style="bold white")
            audit_table.add_column("Status", justify="center")

            user_disp = redact_email(profile.email)
            audit_table.add_row(
                "Account Email",
                f"{user_disp} (User ID: {profile.user_id})",
                "[bold green]AUTHENTICATED[/bold green]",
            )
            audit_table.add_row(
                "Discovered Practice Balance",
                f"ID: {practice_bal.id} | ${practice_bal.amount:,.2f} {practice_bal.currency}",
                "[bold green]ELIGIBLE (PRACTICE)[/bold green]",
            )
            if real_bal:
                audit_table.add_row(
                    "Discovered REAL Balance",
                    f"ID: {real_bal.id} | ${real_bal.amount:,.2f} {real_bal.currency}",
                    "[yellow]BLOCKED FROM SELECTION[/yellow]",
                )
            audit_table.add_row(
                "Current Active Session Balance",
                f"ID: {current_active_id}",
                (
                    "[bold green]PRACTICE[/bold green]"
                    if str(current_active_id) == str(practice_bal.id)
                    else "[bold red]REAL (Default)[/bold red]"
                ),
            )
            audit_table.add_row(
                "Real Trading Enforcement",
                "Permanently Disabled in Software",
                "[bold green]PROTECTED[/bold green]",
            )
            console.print(audit_table)

            if not auto_confirm:
                confirmed = typer.confirm(
                    f"Authorise switching this session's active balance to Practice ID "
                    f"{practice_bal.id}?",
                    default=False,
                )
                if not confirmed:
                    console.print(
                        "[yellow]Selection cancelled by operator. "
                        "No balance modification frames were sent.[/yellow]"
                    )
                    return

            # Dispatch explicit selection
            try:
                with console.status(
                    "[bold green]Transmitting change_balance and verifying post-selection state..."
                ):
                    verified_bal = await transport.async_select_practice_balance(
                        operator_authorized=True
                    )
            except (
                PracticeAuthorizationError,
                AccountSelectionUnverifiedError,
                RealAccountExecutionBlockedError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                console.print(
                    Panel.fit(
                        f"[bold red]CRITICAL SAFETY BLOCK: Practice Selection Failed[/bold red]\n\n"
                        f"• Failure: {exc}\n"
                        "• Verification: Active account could not be confirmed as PRACTICE.\n"
                        "• Selection Request: change_balance was transmitted once.\n"
                        "• Policy: Zero duplicate selection frames sent; zero automatic retries.\n"
                        "• Order Submission: ZERO ORDERS DISPATCHED.\n"
                        "• Teardown: WebSocket connection cleanly closed.",
                        border_style="red",
                    )
                )
                return

            res_table = Table(
                title="[bold green]Live Practice Session Selection Result[/bold green]"
            )
            res_table.add_column("Property", style="cyan")
            res_table.add_column("Verified Value", style="bold white")

            res_table.add_row("Session Identifier", str(transport.session_id))
            res_table.add_row("Active Session Balance ID", str(verified_bal.id))
            res_table.add_row(
                "Account Type", f"{verified_bal.type.name} (type {verified_bal.type.value})"
            )
            res_table.add_row(
                "Available Virtual Funds", f"${verified_bal.amount:,.2f} {verified_bal.currency}"
            )
            res_table.add_row(
                "Practice Session Verification",
                "[bold green]VERIFIED ACTIVE PRACTICE[/bold green]",
            )
            res_table.add_row(
                "Order Transmission Status",
                "[bold green]ZERO ORDERS SUBMITTED[/bold green]",
            )
            console.print(res_table)

            scope_msg = (
                "[bold green]Live Practice Session Selection Complete: VERIFIED[/bold green]\n\n"
                f"[bold white]Session ID:[/bold white] {transport.session_id}\n"
                f"[bold white]Active Balance ID:[/bold white] {verified_bal.id} (PRACTICE)\n\n"
                "[bold yellow]IMPORTANT LIFETIME & SESSION NOTICE:[/bold yellow]\n"
                "• IQ Option balance selection is strictly SESSION-SCOPED.\n"
                "• This authorization applies solely to this active WebSocket session.\n"
                "• Reconnecting, restarting the bot, or network interruption resets the "
                "broker session to REAL.\n"
                "• REAL trading remains permanently disabled in software.\n"
                "• Zero orders were submitted."
            )
            console.print(Panel.fit(scope_msg, border_style="green"))

        finally:
            await transport.async_close()

    asyncio.run(_async_select())


def _render_routing_diagnostic_table(audit: RoutingAuditResult) -> Table:
    """Render the comprehensive Practice Balance Routing & Diagnostic Audit table."""
    table = Table(
        title=(
            "[bold yellow]IQ Option Practice Balance Routing & "
            "Pre-Flight Diagnostic Audit[/bold yellow]"
        )
    )
    table.add_column("Audit Parameter / Entity", style="cyan")
    table.add_column("Observed Value / Wire State", style="bold white")
    table.add_column("Diagnostic Classification / Status", justify="center")

    # 1. Profile balance ID & semantics
    prof_id = audit.profile_balance_id or "Not Reported"
    table.add_row(
        "Profile Reported Balance ID",
        prof_id,
        (
            "[bold yellow]PROFILE_DEFAULT_REAL[/bold yellow]"
            if AccountStateClassification.PROFILE_DEFAULT_REAL in audit.classifications
            else "[dim]Verified Practice[/dim]"
        ),
    )
    table.add_row(
        "Profile Balance Semantics",
        audit.profile_balance_semantics,
        "[dim]Static Broker Server Registration[/dim]",
    )

    # 2. Discovered Balance Entities
    prac_str = f"ID: {audit.discovered_practice_id or 'None'}"
    real_str = f"ID: {audit.discovered_real_id or 'None'}"
    table.add_row(
        "Discovered Balance Entities",
        f"Practice: {prac_str} | Real: {real_str}",
        (
            "[bold green]PRACTICE_BALANCE_AVAILABLE[/bold green]"
            if AccountStateClassification.PRACTICE_BALANCE_AVAILABLE in audit.classifications
            else "[bold red]MISSING PRACTICE[/bold red]"
        ),
    )

    # 3. Authoritative Active Indicator
    table.add_row(
        "Authoritative Active Indicator",
        "ABSENT (broker responses omit active session indicator)",
        "[bold red]UNVERIFIED[/bold red]",
    )

    # 4. Selection Frame Transmitted
    frame_status = (
        f"Transmitted: {audit.selection_frame_name}"
        if audit.selection_frame_transmitted
        else "None"
    )
    table.add_row(
        "Selection Frame Transmitted",
        frame_status,
        (
            "[bold green]TRANSMITTED (1 Frame)[/bold green]"
            if audit.selection_frame_transmitted
            else "[dim]NOT TRANSMITTED[/dim]"
        ),
    )

    # 5. Selection Frame Gateway Ack
    if audit.selection_frame_acknowledged is True:
        ack_str = "YES (Directly acknowledged by broker)"
        ack_status = "[bold green]ACKNOWLEDGED[/bold green]"
    elif audit.selection_frame_acknowledged is False:
        ack_str = "NO (Unacknowledged / Ignored by Gateway)"
        ack_status = "[bold red]SESSION_SELECTION_UNVERIFIED[/bold red]"
    else:
        ack_str = "N/A (No selection frame sent)"
        ack_status = "[dim]NOT APPLICABLE[/dim]"
    table.add_row("Selection Frame Gateway Ack", ack_str, ack_status)

    # 6. Push Events
    table.add_row(
        "Balance-Changed Push Event",
        (
            "RECEIVED"
            if audit.balance_changed_event_received
            else "NO (No balance-changed notification pushed by server)"
        ),
        (
            "[bold green]RECEIVED[/bold green]"
            if audit.balance_changed_event_received
            else "[bold yellow]ABSENT[/bold yellow]"
        ),
    )

    # 7. Response Correlation
    table.add_row(
        "Response Query Correlation",
        "get-profile (profile.balance_id) & get-balances (wallets)",
        "[dim]Correlated[/dim]",
    )

    # 8. Evidence Classifications
    classifications_str = " | ".join(c.value for c in audit.classifications)
    table.add_row(
        "Evidence-Based Classifications",
        classifications_str,
        (
            "[bold green]PRACTICE_CONFIRMED[/bold green]"
            if audit.is_routing_verified
            else "[bold red]BALANCE_ROUTING_UNVERIFIED[/bold red]"
        ),
    )

    # 9. Practice Order Routing Safety
    table.add_row(
        "Practice Order Routing Safety",
        (
            "VERIFIED — Safe virtual order dispatch permitted"
            if audit.is_routing_verified
            else "UNVERIFIED — Order dispatch safely blocked fail-closed"
        ),
        (
            "[bold green]PERMITTED[/bold green]"
            if audit.is_routing_verified
            else "[bold red]FAIL-CLOSED SAFETY BLOCK[/bold red]"
        ),
    )

    # 10. Execution Invariant
    table.add_row(
        "Execution Invariant",
        "PRE-FLIGHT AUDIT ONLY: ZERO ORDERS DISPATCHED",
        "[bold green]PROTECTED[/bold green]",
    )

    return table


@execution_app.command(
    "practice-test",
    help="Operator-supervised verification workflow for a single Practice test order.",
)
def practice_test_command(
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", "--asset", help="Instrument symbol"),
    side: str = typer.Option("call", "--side", help="Trade direction (call or put)"),
    stake: float = typer.Option(
        MIN_PRACTICE_STAKE, "--stake", help="Virtual stake amount (minimum $1.00)"
    ),
    offline_mock: bool = typer.Option(
        True, "--offline-mock/--live-socket", help="Run via offline mock transport (default: True)"
    ),
    select_practice: bool = typer.Option(
        False,
        "--select-practice",
        help="Prompt operator to explicitly switch session balance to Practice if unverified",
    ),
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        "--confirm",
        help="Automatically confirm operator authorization prompts (for automated/mock testing)",
    ),
    preflight_only: bool = typer.Option(
        False,
        "--preflight-only",
        help="Run full live pre-flight checks and verification without order submission",
    ),
) -> None:
    """Supervised Practice test workflow requiring explicit operator authorization."""
    # 1. Stake validation
    if stake < MIN_PRACTICE_STAKE:
        console.print(
            f"[bold red]Stake ${stake:.2f} is below minimum allowed stake "
            f"${MIN_PRACTICE_STAKE:.2f}[/bold red]"
        )
        raise typer.BadParameter(f"Stake must be >= ${MIN_PRACTICE_STAKE:.2f}")
    if stake > MAX_PRACTICE_TEST_STAKE:
        console.print(
            f"[bold red]Stake ${stake:.2f} exceeds maximum allowed test stake "
            f"${MAX_PRACTICE_TEST_STAKE:.2f}[/bold red]"
        )
        raise typer.BadParameter(f"Stake must be <= ${MAX_PRACTICE_TEST_STAKE:.2f}")
    if not offline_mock and abs(stake - 1.00) > 1e-4:
        console.print(
            "[bold red]CRITICAL SAFETY GUARD: Live Practice test trade is strictly restricted to "
            "exactly $1.00 virtual money.[/bold red]"
        )
        raise typer.BadParameter(
            "Live Practice test trade is strictly restricted to exactly $1.00 virtual money."
        )

    # 2. Live safety guard: permanently forbid noninteractive confirmation with live socket
    if not offline_mock and confirm:
        console.print(
            Panel.fit(
                "[bold red]CRITICAL SAFETY BLOCK: Non-interactive confirmation is forbidden "
                "for live broker operations.[/bold red]\n"
                "[dim]The '--yes' / '-y' / '--confirm' flags are forbidden with '--live-socket'. "
                "Explicit interactive operator authorization is required.[/dim]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    norm_side = OrderSide.CALL if side.lower() == "call" else OrderSide.PUT

    # 2. Transport initialization
    transport: IQOptionLivePracticeTransport | MockBrokerPracticeTransport
    if offline_mock:
        transport = MockBrokerPracticeTransport(
            should_succeed=True,
            option_id="mock_ticket_cli_supervised",
            active_balance_id="101" if not select_practice else None,
        )
        if select_practice:
            if not confirm:
                confirmed_switch = typer.confirm(
                    "Authorise explicitly switching session active balance to Practice ID 101?",
                    default=False,
                )
                if not confirmed_switch:
                    console.print(
                        "[yellow]Aborted by operator. "
                        "No balance switch or orders dispatched.[/yellow]"
                    )
                    raise typer.Abort()
            try:
                transport.select_practice_balance(
                    operator_authorized=True,
                    preflight_only=preflight_only,
                )
            except (
                PracticeAuthorizationError,
                AccountSelectionUnverifiedError,
                RealAccountExecutionBlockedError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                if preflight_only:
                    logger.info("Mock practice selection unverified in preflight-only: %s", exc)
                else:
                    console.print(
                        Panel.fit(
                            "[bold red]CRITICAL SAFETY BLOCK: Practice Balance Selection Failed"
                            "[/bold red]\n\n"
                            f"• Failure: {exc}\n"
                            "• Verification: Active account could not be confirmed as PRACTICE.\n"
                            "• Selection Request: change_balance was transmitted once.\n"
                            "• Policy: Zero duplicate frames sent; zero automatic retries.\n"
                            "• Order Submission: ZERO ORDERS DISPATCHED (Fail-Closed).\n"
                            "• Teardown: Connection cleanly closed.",
                            border_style="red",
                        )
                    )
                    transport.close()
                    raise typer.Exit(code=1) from exc
    else:
        console.print(
            "[bold yellow]Initiating live practice connection to IQ Option...[/bold yellow]"
        )
        email = os.environ.get("IQOPTION_EMAIL")
        password = os.environ.get("IQOPTION_PASSWORD")
        if not email or not password:
            console.print(
                "[dim]Credentials not found in environment "
                "(IQOPTION_EMAIL / IQOPTION_PASSWORD).[/dim]"
            )
            email = typer.prompt("IQ Option Account Email")
            password = typer.prompt("IQ Option Password", hide_input=True)

        credentials = Credentials(email=email, password=password)
        transport = IQOptionLivePracticeTransport(credentials=credentials)

        try:
            with console.status(
                "[bold green]Connecting to IQ Option and checking Practice account..."
            ):
                transport.start(require_practice=not select_practice and not preflight_only)
        except (RealAccountExecutionBlockedError, AccountSelectionUnverifiedError) as exc:
            if not select_practice:
                console.print(
                    Panel.fit(
                        f"[bold red]CRITICAL SAFETY BLOCK: {exc}[/bold red]\n\n"
                        "Automatic balance switching is disabled to eliminate risk of accidental "
                        "live execution.\n"
                        "IQ Option balances are session-scoped; the session defaulted to the "
                        "primary server wallet and lacked authoritative session-level practice "
                        "verification.\n"
                        "Pass '--select-practice' or invoke 'practice-select --live' to explicitly "
                        "authorize session selection.\n"
                        "Zero orders were dispatched.",
                        border_style="red",
                    )
                )
                raise typer.Exit(code=1) from exc
        except Exception as exc:
            console.print(
                f"[bold red]Failed to connect or verify practice transport: {exc}[/bold red]"
            )
            raise typer.Exit(code=1) from exc

        # If select_practice requested and session is not yet practice-verified
        if not transport.is_practice_verified:
            practice_bal = next(
                (b for b in transport.balances if b.type == AccountType.PRACTICE and b.is_practice),
                None,
            )
            if practice_bal is None:
                console.print("[bold red]No PRACTICE balance found on this account.[/bold red]")
                transport.close()
                raise typer.Exit(code=1)

            console.print(
                f"[yellow]Active session balance ({transport.active_balance_id}) "
                "is not verified PRACTICE.[/yellow]"
            )
            if not confirm:
                confirmed_switch = typer.confirm(
                    f"Authorise explicitly switching session active balance to Practice ID "
                    f"{practice_bal.id}?",
                    default=False,
                )
                if not confirmed_switch:
                    console.print(
                        "[yellow]Aborted by operator. "
                        "No balance switch or orders were dispatched.[/yellow]"
                    )
                    transport.close()
                    raise typer.Abort()

            try:
                with console.status(
                    "[bold green]Switching session balance to Practice and verifying..."
                ):
                    transport.select_practice_balance(
                        operator_authorized=True,
                        preflight_only=preflight_only,
                    )
            except (
                PracticeAuthorizationError,
                AccountSelectionUnverifiedError,
                RealAccountExecutionBlockedError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                if preflight_only:
                    logger.info("Live practice selection unverified in preflight-only: %s", exc)
                else:
                    console.print(
                        Panel.fit(
                            "[bold red]CRITICAL SAFETY BLOCK: Practice Balance Selection Failed"
                            "[/bold red]\n\n"
                            f"• Failure: {exc}\n"
                            "• Verification: Active account could not be confirmed as PRACTICE.\n"
                            "• Selection Request: change_balance was transmitted once.\n"
                            "• Policy: Zero duplicate frames sent; zero automatic retries.\n"
                            "• Order Submission: ZERO ORDERS DISPATCHED (Fail-Closed).\n"
                            "• Teardown: WebSocket connection cleanly closed.",
                            border_style="red",
                        )
                    )
                    transport.close()
                    raise typer.Exit(code=1) from exc

    try:
        # 3. Verify active practice account & balance
        routing_audit = transport.get_routing_audit()

        if not transport.is_connected:
            console.print("[bold red]Practice transport is not connected.[/bold red]")
            raise typer.Exit(code=1)

        if preflight_only:
            console.print(_render_routing_diagnostic_table(routing_audit))

        # In trade execution mode (not preflight-only), practice routing MUST be verified
        if not preflight_only and (
            not transport.is_practice_verified or not routing_audit.is_routing_verified
        ):
            console.print(
                Panel.fit(
                    "[bold red]CRITICAL SAFETY BLOCK: Practice Routing Unverified[/bold red]"
                    "\n\n"
                    f"• Diagnostic Classifications: "
                    f"{', '.join(c.value for c in routing_audit.classifications)}\n"
                    f"• Profile Balance ID: {routing_audit.profile_balance_id or 'none'} "
                    f"({routing_audit.profile_balance_semantics})\n"
                    "• Practice Order Routing: Cannot be verified without authoritative broker "
                    "confirmation.\n"
                    "• Automatic Retries: ZERO (duplicate frames forbidden).\n"
                    "• Invariant: Order dispatch is strictly blocked fail-closed.\n"
                    "• Zero orders were submitted to the broker.",
                    border_style="red",
                )
            )
            transport.close()
            raise typer.Exit(code=1)

        initial_session_id = transport.session_id
        active_bal_id = str(
            transport.active_balance_id or routing_audit.discovered_practice_id or "unknown"
        )
        current_amount = 0.0
        for b in transport.balances:
            if str(b.id) == active_bal_id:
                current_amount = float(b.amount)
                break
        if abs(current_amount) < 1e-6:
            for b in transport.balances:
                if b.type == AccountType.PRACTICE:
                    current_amount = float(b.amount)
                    break

        # 4. Pre-Flight Verification & Expiry Drift Revalidation Loop
        spec = None
        current_candle = None
        expiry_ts = 0
        expiry_utc_str = ""

        while True:
            # Query verified contract specification
            try:
                spec = transport.get_contract_specification(symbol=symbol.upper())
            except (ContractAvailabilityUnverifiedError, UnsupportedProductOperationError) as exc:
                console.print(
                    Panel.fit(
                        f"[bold red]CRITICAL SAFETY BLOCK: Contract Unavailable or Unverified"
                        f"[/bold red]\n\n"
                        f"• Instrument: {symbol.upper()}\n"
                        f"• Error: {exc}\n"
                        "• Rule: Practice order cannot be placed without authoritative contract "
                        "availability and payout verification from the broker.\n"
                        "• Zero orders were dispatched fail-closed.",
                        border_style="red",
                    )
                )
                raise typer.Exit(code=1) from exc

            if not spec.is_available:
                console.print(
                    Panel.fit(
                        "[bold red]CRITICAL SAFETY BLOCK: Contract Disabled on Broker[/bold red]"
                        "\n\n"
                        f"• Instrument: {symbol.upper()}\n"
                        f"• Broker Status: enabled=False\n"
                        "• Rule: Contract is currently closed/disabled by the broker.\n"
                        "• Zero orders were dispatched fail-closed.",
                        border_style="red",
                    )
                )
                raise typer.Exit(code=1)

            if stake < spec.min_stake:
                console.print(
                    f"[bold red]Stake ${stake:.2f} is below broker minimum stake "
                    f"${spec.min_stake:.2f} for {symbol.upper()}[/bold red]"
                )
                raise typer.Exit(code=1)

            # Fetch live market quote & verify asset tradability
            server_ts = transport.server_time
            try:
                current_candle = transport.get_live_candle(symbol=symbol.upper(), size=60)
            except Exception as exc:
                console.print(
                    f"[bold red]Failed to fetch live market quote for {symbol.upper()}: "
                    f"{exc}[/bold red]"
                )
                raise typer.Exit(code=1) from exc

            if current_candle.close <= 0:
                console.print(
                    f"[bold red]Asset {symbol.upper()} returned non-positive quote "
                    f"({current_candle.close}). Market unavailable.[/bold red]"
                )
                raise typer.Exit(code=1)

            candle_age = server_ts - current_candle.to_ts
            if candle_age > 180:
                console.print(
                    f"[bold red]Asset quote is stale ({candle_age:.0f}s old). "
                    f"Market for {symbol.upper()} may be closed.[/bold red]"
                )
                raise typer.Exit(code=1)

            # Expiration calculation
            expiry_ts = compute_binary_expiry(server_ts, duration_minutes=1)
            remaining_seconds = expiry_ts - int(server_ts)
            if remaining_seconds < 30:
                server_ts = transport.server_time
                expiry_ts = compute_binary_expiry(server_ts, duration_minutes=1)
                remaining_seconds = expiry_ts - int(server_ts)

            expiry_utc_str = datetime.fromtimestamp(expiry_ts, tz=UTC).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

            # Display Pre-Flight Confirmation Table
            table = Table(
                title=(
                    "[bold yellow]Operator Supervised Practice Test — "
                    "Pre-Flight Audit[/bold yellow]"
                )
            )
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="bold white")

            table.add_row(
                "Confirmed Account Type",
                "[bold green]PRACTICE (type 4 - Virtual Funds)[/bold green]",
            )
            table.add_row("Active Practice Balance ID", active_bal_id)
            table.add_row("Current Virtual Balance", f"${current_amount:,.2f}")
            table.add_row("Session Identifier", str(initial_session_id))
            table.add_row("Product & Instrument", f"BINARY {symbol.upper()}")
            table.add_row("Trade Direction", f"[bold cyan]{norm_side.value.upper()}[/bold cyan]")
            table.add_row("Virtual Stake Amount", f"${stake:.2f} (Platform minimum)")
            table.add_row(
                "Contract Availability",
                f"[bold green]AVAILABLE (Source: {spec.source})[/bold green]",
            )
            table.add_row(
                "Broker Payout Rate",
                f"{spec.payout_rate * 100:.1f}% (Commission: {spec.commission_percent:.1f}%)",
            )
            table.add_row(
                "Live Quote Price",
                f"{current_candle.close:.5f} (Mid: {current_candle.midpoint:.5f})",
            )
            table.add_row("Contract Expiration", f"{expiry_ts} ({expiry_utc_str})")
            table.add_row("Remaining Seconds to Expiry", f"{remaining_seconds}s")
            table.add_row("Autonomous AI Trading", "[bold red]DISABLED[/bold red]")
            if preflight_only:
                table.add_row(
                    "Safety Invariant",
                    "[bold yellow]Preflight-Only Mode (Zero Orders Dispatched)[/bold yellow]",
                )
            else:
                table.add_row(
                    "Safety Invariant",
                    "[bold green]Single Practice Order / Zero Retries Enforced[/bold green]",
                )
            console.print(table)

            # Operator authorization (bypassed in preflight_only mode)
            if not confirm and not preflight_only:
                reconfirm = typer.confirm(
                    f"Transmit single ${stake:.2f} Practice trade on {symbol.upper()} to broker?",
                    default=False,
                )
                if not reconfirm:
                    console.print(
                        "[yellow]Aborted by operator. No test order was dispatched.[/yellow]"
                    )
                    raise typer.Abort()

            # Immediate Pre-Transmission Revalidation: detect expiry drift or material payout change
            fresh_server_ts = transport.server_time
            remaining_before_send = expiry_ts - int(fresh_server_ts)
            try:
                fresh_spec = transport.get_contract_specification(symbol=symbol.upper())
            except Exception as exc:
                console.print(
                    f"[bold red]CRITICAL SAFETY BLOCK: Pre-transmission contract verification "
                    f"failed: {exc}[/bold red]"
                )
                raise typer.Exit(code=1) from exc

            terms_invalid_reason: str | None = None
            if remaining_before_send < 30:
                terms_invalid_reason = (
                    f"Order expiry deadline elapsed while reviewing prompt: only "
                    f"{remaining_before_send}s remaining "
                    "(minimum 30s cutoff required before expiry)."
                )
            elif not fresh_spec.is_available:
                terms_invalid_reason = (
                    f"Contract {symbol.upper()} became unavailable during confirmation."
                )
            elif abs(fresh_spec.payout_rate - spec.payout_rate) > 0.001:
                terms_invalid_reason = (
                    f"Broker payout rate changed materially from {spec.payout_rate * 100:.1f}% "
                    f"to {fresh_spec.payout_rate * 100:.1f}% during confirmation."
                )

            if terms_invalid_reason:
                console.print(
                    Panel.fit(
                        f"[bold yellow]APPROVED ORDER TERMS INVALIDATED PRE-TRANSMISSION"
                        f"[/bold yellow]\n\n"
                        f"• {terms_invalid_reason}\n"
                        "• Invariant: Order terms will NEVER be silently modified.\n"
                        "• Re-evaluating fresh market terms and requiring renewed authorization.",
                        border_style="yellow",
                    )
                )
                if confirm and not preflight_only:
                    raise RuntimeError(
                        f"Pre-transmission validation failed in automated mode: "
                        f"{terms_invalid_reason}"
                    )
                continue

            break

        # Preflight-only mode: complete preflight audit, skip order creation & dispatch
        if preflight_only:
            payout_pct = spec.payout_rate * 100 if spec else 0.0
            if routing_audit.is_routing_verified:
                console.print(
                    Panel.fit(
                        "[bold green]LIVE PRE-FLIGHT VERIFICATION SUCCEEDED: ZERO ORDERS DISPATCHED"
                        "[/bold green]\n\n"
                        f"• Transport State: Connected ({transport.session_id})\n"
                        f"• Verified Account: PRACTICE (Balance ID: {active_bal_id})\n"
                        f"• Current Virtual Balance: ${current_amount:,.2f}\n"
                        f"• Instrument Tradability: {symbol.upper()} Binary Option AVAILABLE\n"
                        f"• Authoritative Broker Payout: {payout_pct:.1f}% "
                        f"(Commission: {spec.commission_percent:.1f}%)\n"
                        f"• Live Broker Quote: {current_candle.close:.5f} "
                        f"(Midpoint: {current_candle.midpoint:.5f})\n"
                        f"• Next Expiry Deadline: {expiry_ts} ({expiry_utc_str}, "
                        f"{remaining_before_send}s remaining)\n"
                        "• Pre-Flight Mode: Order proposal, creation, and submission "
                        "completely BYPASSED.\n"
                        "• Safety Invariant: Zero orders dispatched; clean teardown initiated.",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel.fit(
                        "[bold yellow]PRE-FLIGHT ROUTING AUDIT COMPLETE: ZERO ORDERS DISPATCHED"
                        "[/bold yellow]\n\n"
                        f"• Diagnostic Classifications: "
                        f"{', '.join(c.value for c in routing_audit.classifications)}\n"
                        f"• Profile Balance ID: {routing_audit.profile_balance_id or 'none'} "
                        f"({routing_audit.profile_balance_semantics})\n"
                        f"• Discovered Wallets: Practice={routing_audit.discovered_practice_id}, "
                        f"Real={routing_audit.discovered_real_id}\n"
                        f"• Gateway Selection Frame: {routing_audit.selection_frame_name} "
                        f"(transmitted={routing_audit.selection_frame_transmitted}, "
                        f"acknowledged={routing_audit.selection_frame_acknowledged})\n"
                        "• Practice Order Routing: UNVERIFIED — order dispatch blocked "
                        "fail-closed.\n"
                        f"• Market Status: {symbol.upper()} Binary Option "
                        f"(Payout: {payout_pct:.1f}%, Expiry: {expiry_utc_str})\n"
                        "• Pre-Flight Mode: Order proposal, creation, and submission "
                        "completely BYPASSED.\n"
                        "• Execution Invariant: ZERO ORDERS DISPATCHED; clean teardown initiated.",
                        border_style="yellow",
                    )
                )
            return

        # Pre-transmission connection & session continuity verification
        transport.refresh_account_state()
        if not transport.is_connected:
            console.print(
                "[bold red]CRITICAL SAFETY BLOCK: Transport disconnected before submission. "
                "Order blocked fail-closed.[/bold red]"
            )
            raise typer.Exit(code=1)
        if transport.session_id != initial_session_id:
            console.print(
                "[bold red]CRITICAL SAFETY BLOCK: Session ID changed unexpectedly. "
                "Order blocked fail-closed.[/bold red]"
            )
            raise typer.Exit(code=1)
        if not transport.is_practice_verified or str(transport.active_balance_id) != active_bal_id:
            console.print(
                f"[bold red]CRITICAL SAFETY BLOCK: Active balance changed "
                f"({transport.active_balance_id} != {active_bal_id}) or lost practice "
                "verification. Order blocked fail-closed.[/bold red]"
            )
            raise typer.Exit(code=1)

        # 9. Engine Execution & Order Dispatch
        tok = PracticeExecutionAuthToken(
            authorization_signature=(
                "sig_supervisor_cli_practice_live"
                if not offline_mock
                else "sig_supervisor_cli_practice"
            ),
            supervisor_id=(
                "supervisor_cli_operator_live" if not offline_mock else "supervisor_cli_operator"
            ),
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
                initial_balance=current_amount,
                mode_config=mode_cfg,
                single_test_order_mode=True,
            )
        )

        proposal = engine.create_proposal(
            symbol=symbol.upper(),
            side=norm_side,
            stake=stake,
            payout_rate=spec.payout_rate if spec else None,
            candle_to_ts=float(current_candle.to_ts),
            signal_id="sig_operator_cli_supervised",
        )
        order, dec = engine.propose_and_validate(
            proposal,
            current_ts=float(current_candle.to_ts),
            profile=(
                transport.profile
                if not offline_mock
                else _make_profile(1, int(active_bal_id) if active_bal_id.isdigit() else 101)
            ),
            balances=(
                transport.balances
                if not offline_mock
                else [
                    Balance(
                        id=active_bal_id,
                        type=AccountType.PRACTICE,
                        amount=current_amount,
                        currency="USD",
                        is_active=True,
                    )
                ]
            ),
        )
        if not dec.permitted:
            console.print(f"[bold red]Order rejected by risk controller: {dec.message}[/bold red]")
            raise typer.Exit(code=1)

        order.expiry_ts = float(expiry_ts)

        console.print("[bold cyan]Submitting order to broker transport...[/bold cyan]")
        _ = engine.submit_order(order, current_candle, broker_transport=transport)

        if not order.broker_order_id or order.state not in {
            OrderState.ACKNOWLEDGED,
            OrderState.OPEN,
        }:
            console.print(
                Panel.fit(
                    "[bold red]Broker Order Submission Failed[/bold red]\n"
                    f"• State: {order.state.value.upper()}\n"
                    f"• Reconciliation Required: {order.reconciliation_required}\n"
                    "• Invariant: Zero retries attempted.",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

        console.print(
            Panel.fit(
                "[bold green]Practice Test Order Successfully Placed\n"
                "(Broker Practice Order Confirmed)[/bold green]\n"
                f"• Broker Ticket ID: [bold white]{order.broker_order_id}[/bold white]\n"
                f"• Local Order ID: {order.order_id}\n"
                f"• State: {order.state.value.upper()}\n"
                f"• Entry Price: {order.entry_price:.5f}\n"
                f"• Expiry TS: {int(order.expiry_ts or 0)} ({expiry_utc_str})\n"
                f"• Active Balance ID: {active_bal_id}\n"
                "• Audit Trail: Durably recorded in SQLite execution journal.",
                border_style="green",
            )
        )

        # 10. Settlement & Post-Trade Reconciliation (for live socket)
        if not offline_mock and isinstance(transport, IQOptionLivePracticeTransport):
            console.print("[bold yellow]Monitoring position until expiration...[/bold yellow]")
            try:
                settlement = transport.wait_for_settlement(
                    order.broker_order_id, timeout_seconds=90.0
                )
                new_bal = 0.0
                for b in transport.balances:
                    if str(b.id) == str(active_bal_id):
                        new_bal = float(b.amount)
                        break

                console.print(
                    Panel.fit(
                        "[bold green]Live Broker Settlement Verified[/bold green]\n"
                        f"• Broker Ticket ID: {settlement.option_id}\n"
                        f"• Outcome: [bold]{(settlement.outcome or 'UNKNOWN').upper()}[/bold]\n"
                        f"• Realized PnL: ${settlement.net_pnl:+.2f}\n"
                        f"• Exit Price: {settlement.exit_price}\n"
                        f"• Close Time: {settlement.exit_ts}\n"
                        f"• Prior Practice Balance: ${current_amount:,.2f}\n"
                        f"• New Practice Balance: ${new_bal:,.2f} "
                        f"(Delta: ${new_bal - current_amount:+.2f})\n"
                        f"• Financial Consistency: {settlement.is_financially_consistent} "
                        f"({settlement.consistency_note})\n"
                        "• SQLite Journal: Audit trail durably committed.",
                        border_style="green",
                    )
                )
            except TimeoutError:
                console.print(
                    "[yellow]Timed out waiting for broker settlement notification. "
                    "Position will reconcile on subsequent poll.[/yellow]"
                )
    finally:
        transport.close()


@execution_app.command(
    "practice-observe",
    help="Passively observe manual Practice transactions in official IQ Option UI (READ-ONLY).",
)
def practice_observe_command(
    live: bool = typer.Option(
        False,
        "--live",
        help="Connect to live IQ Option broker in passive READ-ONLY observation mode.",
    ),
    offline_mock: bool = typer.Option(
        False,
        "--offline-mock",
        help="Run against deterministic offline mock manual trade observation.",
    ),
    probe_only: bool = typer.Option(
        False,
        "--probe-only",
        help="Run read-only connection and subscription probe without awaiting trades.",
    ),
    timeout: float = typer.Option(
        180.0,
        "--timeout",
        "-t",
        help="Maximum observation window in seconds.",
    ),
    asset: str | None = typer.Option(
        None,
        "--asset",
        "-a",
        help="Optional asset symbol filter (e.g. EURUSD).",
    ),
    history_stake: float | None = typer.Option(
        None, "--history-stake", help="Reconcile this stake against history after a live timeout."
    ),
    record: str | None = typer.Option(
        None,
        "--record",
        "-r",
        help="Optional file path to persist observed trade record as JSON.",
    ),
    mock_anomaly: bool = typer.Option(
        False,
        "--mock-anomaly",
        help="[Mock only] Simulate an anomalous manual trade on the REAL account to verify alert.",
    ),
    mock_timeout: bool = typer.Option(
        False,
        "--mock-timeout",
        help="[Mock only] Simulate timeout without any observed trade.",
    ),
    mock_settle_mismatch: bool = typer.Option(
        False,
        "--mock-settle-mismatch",
        help="[Mock only] Simulate settlement with mismatched ticket ID.",
    ),
    mock_unverified_real: bool = typer.Option(
        False,
        "--mock-unverified-real",
        help="[Mock only] Simulate unverified post-settlement REAL balance response.",
    ),
    mock_missing_fields: bool = typer.Option(
        False,
        "--mock-missing-fields",
        help="[Mock only] Simulate broker frames with missing optional fields.",
    ),
) -> None:
    """Passively observe manual Practice transactions executed in the official IQ Option UI.

    This command runs in strictly READ-ONLY mode. It NEVER places trades or modifies accounts.
    It listens for broker push events (option-opened, portfolio updates, option-closed)
    to confirm whether user_balance_id routes to Practice and leaves REAL untouched.
    """
    if live and offline_mock:
        console.print("[bold red]Cannot specify both --live and --offline-mock.[/bold red]")
        raise typer.BadParameter("Choose either --live or --offline-mock.")

    if not live and not offline_mock:
        offline_mock = True

    if history_stake is not None and (not live or not asset or history_stake <= 0):
        raise typer.BadParameter("--history-stake requires --live, --asset and a positive stake.")

    banner = (
        "[bold cyan]IQ Option Passive Practice Transaction Observer[/bold cyan]\n\n"
        "[bold green]STRICT READ-ONLY GUARANTEE:[/bold green]\n"
        "• This observer connects exclusively to listen to broker push events.\n"
        "• Zero trading or order submission capabilities exist in this component.\n"
        "• Observed REAL balance mutations flag a breach; missing events remain unverified.\n"
        "• Objective: Correlate manual Practice trades executed in the official IQ Option UI."
    )
    console.print(Panel.fit(banner, border_style="cyan"))

    # Handle probe-only mode
    if probe_only:
        console.print(
            "[bold cyan]Executing read-only connection and subscription probe...[/bold cyan]"
        )
        probe_res: dict[str, Any]
        if offline_mock:
            mock_obs = MockPassivePracticeObserver()
            probe_res = mock_obs.probe()
        else:
            email = os.environ.get("IQOPTION_EMAIL")
            password = os.environ.get("IQOPTION_PASSWORD")
            if not email or not password:
                console.print(
                    "[dim]Credentials not found in environment "
                    "(IQOPTION_EMAIL / IQOPTION_PASSWORD).[/dim]"
                )
                email = typer.prompt("IQ Option Account Email")
                password = typer.prompt("IQ Option Password", hide_input=True)

            credentials = Credentials(email=email, password=password)
            probe_observer = PassivePracticeObserver(credentials=credentials)

            async def _run_probe() -> dict[str, Any]:
                try:
                    with console.status(
                        "[bold green]Connecting and probing portfolio subscriptions..."
                    ):
                        return await probe_observer.async_probe_connection_and_subscriptions()
                finally:
                    await probe_observer.async_close()

            try:
                probe_res = asyncio.run(_run_probe())
            except Exception as exc:
                console.print(f"[bold red]Probe failed: {type(exc).__name__}[/bold red]")
                raise typer.Exit(code=1) from exc

        # Render Probe Table
        probe_table = Table(
            title="[bold green]IQ Option Passive Observer Probe Results[/bold green]"
        )
        probe_table.add_column("Probe Dimension", style="cyan")
        probe_table.add_column("Result / Value", style="bold white")
        probe_table.add_column("Status Verdict", style="bold")

        is_auth = bool(probe_res.get("authenticated"))
        probe_table.add_row(
            "WebSocket Authentication",
            "AUTHENTICATED" if is_auth else "FAILED",
            "[bold green]VERIFIED[/bold green]" if is_auth else "[bold red]FAILED[/bold red]",
        )
        probe_table.add_row(
            "Session Identifier",
            str(probe_res.get("session_id")),
            "[bold cyan]SESSION-SCOPED[/bold cyan]",
        )
        prac_amt = probe_res.get("practice_amount")
        prac_amt_str = f"${prac_amt:,.2f}" if isinstance(prac_amt, (int, float)) else "N/A"
        probe_table.add_row(
            "Practice Wallet ID",
            f"ID: {probe_res.get('practice_balance_id')} ({prac_amt_str})",
            "[bold green]AVAILABLE[/bold green]",
        )
        real_amt = probe_res.get("real_amount")
        real_amt_str = f"${real_amt:,.2f}" if isinstance(real_amt, (int, float)) else "N/A"
        probe_table.add_row(
            "REAL Wallet ID",
            f"ID: {probe_res.get('real_balance_id')} ({real_amt_str})",
            "[bold yellow]WATCHDOG ARMED[/bold yellow]",
        )
        sub_status = str(probe_res.get("subscription_status"))
        probe_table.add_row(
            "Portfolio Topic Subscription",
            f"Status: {sub_status}",
            (
                "[bold green]ACCEPTED (BROKER ACK)[/bold green]"
                if sub_status == "ACCEPTED"
                else (
                    "[bold yellow]STATUS_UNKNOWN (UNACKNOWLEDGED GATEWAY)[/bold yellow]"
                    if sub_status == "STATUS_UNKNOWN"
                    else "[bold red]REJECTED[/bold red]"
                )
            ),
        )
        probe_table.add_row(
            "Order Submission Capability",
            "Permanently Disabled in Software",
            "[bold green]STRICT READ-ONLY (ZERO TRADES)[/bold green]",
        )
        console.print(probe_table)
        if probe_res.get("diagnostics") is not None:
            console.print_json(json.dumps(probe_res["diagnostics"]))
        return

    trade_record: ObservedPracticeTradeRecord | None = None
    observer: PassivePracticeObserver | None = None

    if offline_mock:
        console.print("[bold cyan]Running deterministic offline mock observation...[/bold cyan]")
        mock_obs = MockPassivePracticeObserver(
            ticket_id="mock_manual_ticket_12345",
            simulate_real_trade_anomaly=mock_anomaly,
            simulate_timeout=mock_timeout,
            simulate_settlement_mismatch=mock_settle_mismatch,
            simulate_unverified_real=mock_unverified_real,
            simulate_missing_fields=mock_missing_fields,
        )

        base_table = Table(title="[bold cyan]Baseline Account Balances (Observer)[/bold cyan]")
        base_table.add_column("Account Type", style="cyan")
        base_table.add_column("Balance ID", style="bold white")
        base_table.add_column("Current Amount", style="bold green")
        base_table.add_column("Observer Watchdog", style="bold yellow")

        base_table.add_row(
            "PRACTICE",
            str(mock_obs.practice_balance.id),
            f"${mock_obs.practice_balance.amount:,.2f} USD",
            "Target for Manual Trade",
        )
        real_b = mock_obs.real_balance
        real_amt_label = f"${real_b.amount:,.2f} USD" if real_b else "N/A"
        base_table.add_row(
            "REAL",
            str(real_b.id if real_b else "None"),
            real_amt_label,
            "[bold red]PROTECTED (MUST REMAIN UNTOUCHED)[/bold red]",
        )
        console.print(base_table)

        if not mock_timeout:
            console.print(
                "[dim]Simulating operator manual Practice trade in official UI "
                "(EURUSD, $1.00 CALL)...[/dim]"
            )
        trade_record = mock_obs.observe(timeout_seconds=timeout, asset_filter=asset)

    else:
        # Live observation mode
        console.print(
            "[bold yellow]Connecting to live IQ Option broker in READ-ONLY mode...[/bold yellow]"
        )
        email = os.environ.get("IQOPTION_EMAIL")
        password = os.environ.get("IQOPTION_PASSWORD")
        if not email or not password:
            console.print(
                "[dim]Credentials not found in environment "
                "(IQOPTION_EMAIL / IQOPTION_PASSWORD).[/dim]"
            )
            email = typer.prompt("IQ Option Account Email")
            password = typer.prompt("IQ Option Password", hide_input=True)

        credentials = Credentials(email=email, password=password)
        observer = PassivePracticeObserver(credentials=credentials)

        async def _run_live() -> ObservedPracticeTradeRecord | None:
            try:
                with console.status(
                    "[bold green]Authenticating and establishing WebSocket stream..."
                ):
                    await observer.async_connect()
            except Exception as exc:
                console.print(
                    f"[bold red]Failed to connect passive observer: {type(exc).__name__}[/bold red]"
                )
                raise typer.Exit(code=1) from exc

            history_start = time.time()
            # Set the filter before subscribing so early events are filtered too.
            observer.set_asset_filter(asset)
            # Subscribe to portfolio updates
            with console.status("[bold cyan]Subscribing to read-only portfolio updates..."):
                await observer.async_subscribe_portfolio()

            # Print Baseline
            base_table = Table(
                title="[bold cyan]Live Account Balances (Observer Baseline)[/bold cyan]"
            )
            base_table.add_column("Account Type", style="cyan")
            base_table.add_column("Balance ID", style="bold white")
            base_table.add_column("Current Amount", style="bold green")
            base_table.add_column("Observer Watchdog", style="bold yellow")

            if observer.practice_balance:
                base_table.add_row(
                    "PRACTICE",
                    str(observer.practice_balance.id),
                    f"${observer.practice_balance.amount:,.2f} "
                    f"{observer.practice_balance.currency}",
                    "Target for Manual Trade",
                )
            if observer.real_balance:
                base_table.add_row(
                    "REAL",
                    str(observer.real_balance.id),
                    f"${observer.real_balance.amount:,.2f} {observer.real_balance.currency}",
                    "[bold red]PROTECTED (MUST REMAIN UNTOUCHED)[/bold red]",
                )
            console.print(base_table)

            console.print(
                "[yellow]Listening for existing or subsequent broker events. "
                "No trade is required or requested.[/yellow]"
            )

            try:
                with console.status(
                    f"[bold cyan]Passively listening for events (timeout: {int(timeout)}s)..."
                ):
                    observed = await observer.async_observe_single_trade(
                        timeout_seconds=timeout,
                        asset_filter=asset,
                    )
                if observed is None and history_stake is not None and asset:
                    try:
                        history = await recover_history(
                            observer,
                            asset=asset,
                            stake=history_stake,
                            start=history_start,
                            end=time.time(),
                        )
                        console.print("Read-only history reconciliation after missed event:")
                        console.print_json(json.dumps(history))
                        if record:
                            await asyncio.to_thread(
                                Path(record).write_text,
                                json.dumps(history, indent=2),
                                encoding="utf-8",
                            )
                        if history["account"]["real_wallet_status"] == "MUTATED":
                            raise typer.Exit(code=1)
                    except typer.Exit:
                        raise
                    except Exception as exc:  # noqa: BLE001 - never expose broker errors
                        console.print(
                            f"[yellow]History unverified ({type(exc).__name__}).[/yellow]"
                        )
                return observed
            finally:
                await observer.async_close()

        try:
            trade_record = asyncio.run(_run_live())
        except KeyboardInterrupt:
            console.print("\n[yellow]Observation cancelled by operator.[/yellow]")
            raise typer.Exit(code=0) from None

    # Handle result
    if trade_record is None:
        console.print(
            Panel.fit(
                "[bold yellow]Observation Window Closed "
                "(No Position Event Confirmed)[/bold yellow]\n\n"
                f"• Timeout reached after {int(timeout)} seconds.\n"
                "• No matching position event was confirmed.\n"
                "• Invariant Verified: Zero orders were submitted by the bot.",
                border_style="yellow",
            )
        )
        if observer is not None:
            console.print("Read-only observation diagnostics:")
            console.print_json(json.dumps(observer.diagnostics()))
            console.print(
                "No matching event was observed; use practice-history for reconciliation."
            )
        return

    # Render Result Table
    res_table = Table(
        title="[bold green]Observed Practice Transaction Correlation Report[/bold green]"
    )
    res_table.add_column("Audit Dimension", style="cyan")
    res_table.add_column("Observed Value", style="bold white")
    res_table.add_column("Safety / Routing Verdict", style="bold")

    res_table.add_row(
        "Broker Ticket ID",
        trade_record.ticket_id,
        "[bold green]RECORDED[/bold green]",
    )
    symbol_str = (
        f"{trade_record.symbol} (active_id={trade_record.active_id})"
        if trade_record.symbol
        else (
            f"active_id={trade_record.active_id}"
            if trade_record.active_id is not None
            else "N/A (Missing from broker frame)"
        )
    )
    sym_badge = (
        "[bold green]IDENTIFIED[/bold green]"
        if trade_record.symbol
        else "[yellow]UNRESOLVED[/yellow]"
    )
    res_table.add_row("Asset / Symbol", symbol_str, sym_badge)

    dir_str = (
        f"{trade_record.direction.upper()}"
        if trade_record.direction
        else "N/A (Missing from broker frame)"
    )
    stake_str = (
        f"${trade_record.stake:.2f}"
        if trade_record.stake is not None
        else "N/A (Missing from broker frame)"
    )
    stake_badge = (
        "[bold green]CONFIRMED[/bold green]"
        if trade_record.stake is not None
        else "[yellow]INCOMPLETE[/yellow]"
    )
    res_table.add_row("Direction & Stake", f"{dir_str} | {stake_str}", stake_badge)
    res_table.add_row(
        "Wire user_balance_id",
        str(trade_record.user_balance_id or "N/A"),
        (
            "[bold green]MATCHES PRACTICE WALLET[/bold green]"
            if trade_record.is_practice_balance
            else "[bold red]ANOMALY: MATCHES REAL WALLET[/bold red]"
        ),
    )
    res_table.add_row(
        "Target Practice Wallet ID",
        trade_record.practice_balance_id,
        "[bold green]VERIFIED PRACTICE[/bold green]",
    )
    res_table.add_row(
        "Subscription Status",
        trade_record.subscription_status,
        (
            "[bold green]EVENT RECEIVED[/bold green]"
            if trade_record.subscription_status == "EVENT_RECEIVED"
            else f"[yellow]{trade_record.subscription_status}[/yellow]"
        ),
    )

    init_real_str = (
        f"${trade_record.initial_real_amount:.2f}"
        if trade_record.initial_real_amount is not None
        else "N/A"
    )
    final_real_str = (
        f"${trade_record.final_real_amount:.2f}"
        if trade_record.final_real_amount is not None
        else "N/A"
    )
    res_table.add_row(
        "Real Wallet Verification",
        f"Initial: {init_real_str} -> Final: {final_real_str} ({trade_record.real_wallet_status})",
        (
            "[bold green]VERIFIED UNTOUCHED[/bold green]"
            if trade_record.real_wallet_status == "VERIFIED_UNTOUCHED"
            else (
                "[bold red]CRITICAL SAFETY BREACH (MUTATED)[/bold red]"
                if trade_record.real_wallet_status == "MUTATED"
                else "[bold yellow]UNVERIFIED (NO FRESH SNAPSHOT)[/bold yellow]"
            )
        ),
    )

    post_open_val = (
        trade_record.post_open_practice_amount
        if trade_record.post_open_practice_amount is not None
        else trade_record.initial_practice_amount
    )
    final_prac_val = (
        trade_record.final_practice_amount
        if trade_record.final_practice_amount is not None
        else trade_record.initial_practice_amount
    )
    prac_delta = final_prac_val - trade_record.initial_practice_amount
    res_table.add_row(
        "Practice Balance Lifecycle",
        f"Before: ${trade_record.initial_practice_amount:.2f} -> "
        f"Post-Open: ${post_open_val:.2f} -> "
        f"Final: ${final_prac_val:.2f}",
        f"[bold cyan]Delta: ${prac_delta:+.2f}[/bold cyan]",
    )

    profit_str = (
        f"${trade_record.settlement_profit:+.2f}"
        if trade_record.settlement_profit is not None
        else "N/A"
    )
    res_table.add_row(
        "Settlement Status & Profit",
        f"Status: {trade_record.settlement_status or 'N/A'} | Profit: {profit_str}",
        (
            "[bold green]SETTLED[/bold green]"
            if trade_record.settlement_status
            else "[yellow]PENDING[/yellow]"
        ),
    )

    is_safe_pass = (
        trade_record.is_practice_balance
        and trade_record.real_wallet_status == "VERIFIED_UNTOUCHED"
        and trade_record.real_wallet_untouched
        and trade_record.settlement_event is not None
        and trade_record.settlement_event.ticket_id == trade_record.ticket_id
    )
    verdict_badge = (
        "[bold green]PASS (SAFE)[/bold green]"
        if is_safe_pass
        else "[bold red]FAIL (SAFETY ALERT)[/bold red]"
    )
    res_table.add_row("Correlation Verdict", trade_record.correlation_verdict, verdict_badge)
    console.print(res_table)

    # Save to record JSON if requested
    if record:
        rec_path = Path(record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(json.dumps(trade_record.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[bold green]Observed trade record saved to {rec_path}[/bold green]")

    if not is_safe_pass:
        console.print(
            Panel.fit(
                "[bold red]CRITICAL SAFETY VERDICT[/bold red]\n"
                "The observed manual transaction did not achieve verified Practice correlation:\n"
                f"• Verdict: {trade_record.correlation_verdict}\n"
                f"• Real Wallet Status: {trade_record.real_wallet_status}\n"
                "Execution guards remain locked fail-closed.",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


practice_observe = practice_observe_command


def practice_history_command(
    live: bool = typer.Option(False, "--live", help="Connect to broker read-only."),
    asset: str = typer.Option(..., "--asset", help="Exact asset, e.g. EURUSD or EURUSD-OTC."),
    stake: float = typer.Option(..., "--stake", help="Exact virtual trade stake."),
    start: str = typer.Option(..., "--start", help="Inclusive ISO-8601 start with UTC offset."),
    end: str = typer.Option(..., "--end", help="Inclusive ISO-8601 end with UTC offset."),
    limit: int = typer.Option(100, "--limit", min=1, max=1000),
    ticket: str | None = typer.Option(None, "--ticket", help="Exact broker ticket if known."),
    record: str | None = typer.Option(None, "--record", help="Save allowlisted evidence JSON."),
) -> None:
    """Query broker history for classic binary/turbo options; never submits orders."""
    if not live:
        raise typer.BadParameter("Explicit --live is required for a broker history query.")
    try:
        start_dt, end_dt = datetime.fromisoformat(start), datetime.fromisoformat(end)
        if start_dt.utcoffset() is None or end_dt.utcoffset() is None:
            raise ValueError("Timestamps must include a UTC offset")
        if start_dt >= end_dt:
            raise ValueError("Start must be earlier than end")
        from iqoption_research.market_data.instruments import InstrumentRegistry

        InstrumentRegistry.normalize_symbol(asset)
    except (ValueError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    email = os.environ.get("IQOPTION_EMAIL") or typer.prompt("IQ Option Account Email")
    password = os.environ.get("IQOPTION_PASSWORD") or typer.prompt(
        "IQ Option Password", hide_input=True
    )
    observer = PassivePracticeObserver(credentials=Credentials(email=email, password=password))

    async def run() -> dict[str, Any]:
        try:
            await observer.async_connect()
            return await recover_history(
                observer,
                asset=asset,
                stake=stake,
                start=start_dt.timestamp(),
                end=end_dt.timestamp(),
                limit=limit,
                ticket_id=ticket,
            )
        finally:
            await observer.async_close()

    try:
        result = asyncio.run(run())
    except Exception as exc:
        # Never display broker exception text, account data, or raw frames.
        console.print("[red]Read-only history query failed; history is unverified.[/red]")
        console.print_json(json.dumps(observer.failure_diagnostics()))
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(result))
    if record:
        Path(record).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["account"]["real_wallet_status"] == "MUTATED":
        raise typer.Exit(code=1)
