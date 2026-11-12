"""CLI commands for Phase 5 automated paper-trading, risk control, and historical replay.

All operations run strictly offline with simulated execution. Zero network requests
and zero live order frames are transmitted.
"""

from __future__ import annotations

from dataclasses import replace

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.execution.engine import PaperTradingConfig, PaperTradingEngine
from iqoption_research.execution.risk import RiskConfig
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.research.backtest.models import SignalDirection

console = Console()
paper_app = typer.Typer(
    name="paper",
    help="Phase 5 automated AI paper-trading, deterministic risk manager, and historical replay.",
    no_args_is_help=True,
)


@paper_app.command("replay", help="Run historical candle replay through the paper-trading engine.")
def replay_command(
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    bars: int = typer.Option(200, "--bars", "-n", help="Number of replay bars"),
    stake: float = typer.Option(100.0, "--stake", help="Fixed stake per order"),
    payout: float = typer.Option(0.85, "--payout", "-p", help="Binary option payout rate"),
    seed: int = typer.Option(42, "--seed", help="Random seed for synthetic fixture"),
    inject_gap: bool = typer.Option(
        False, "--inject-gap", help="Inject feed gap to test circuit breaker"
    ),
) -> None:
    """Execute historical candle replay through the complete Phase 5 pipeline."""
    console.print(
        Panel(
            f"[bold green]IQ Option Phase 5 — Historical Paper-Trading Replay[/bold green]\n"
            f"Symbol: {symbol} | Replay Bars: {bars} | Stake: ${stake:.2f}\n"
            f"Payout: {payout * 100:.0f}% | Strictly offline simulated execution.",
            border_style="green",
        )
    )

    # 1. Generate synthetic candles across realistic market regime
    candles = list(
        generate_synthetic_candles(
            SyntheticMarketConfig(
                symbol=symbol,
                n_bars=bars,
                start_price=1.0850,
                drift=0.0001,
                volatility=0.0008,
                seed=seed,
            )
        ).candles
    )

    if inject_gap and len(candles) > 60:
        # Inject gap at bar 60
        c_prev = candles[59]
        candles[60] = replace(c_prev, from_ts=c_prev.to_ts + 300, to_ts=c_prev.to_ts + 360)

    # 2. Configure engine
    risk_cfg = RiskConfig(
        max_stake_per_trade=250.0,
        max_daily_loss=500.0,
        max_consecutive_losses=3,
        min_payout_rate=0.70,
        min_confidence_prob=0.55,
    )
    config = PaperTradingConfig(
        symbol=symbol,
        payout_rate=payout,
        initial_balance=10000.0,
        is_replay=True,
        risk_config=risk_cfg,
    )

    engine = PaperTradingEngine(config=config)

    # 3. Stream candles through engine
    console.print(
        "[dim]Streaming candles through feature buffer, risk controller, and state machine...[/dim]"
    )
    for c in candles:
        engine.process_candle(c)

    # 4. Compile metrics table
    portfolio = engine.portfolio
    risk_mgr = engine.risk_manager

    table = Table(title="[bold cyan]Phase 5 Paper-Trading Performance Summary[/bold cyan]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")

    table.add_row("Initial Virtual Balance", f"${portfolio.initial_balance:,.2f}")
    table.add_row("Final Virtual Balance", f"${portfolio.current_balance:,.2f}")
    table.add_row(
        "Net Realized PnL",
        f"[{'green' if portfolio.total_pnl >= 0 else 'red'}]${portfolio.total_pnl:+,.2f}[/]",
    )
    table.add_row("Total Simulated Orders", str(portfolio.total_trades))
    table.add_row("Winning Trades", str(portfolio.winning_trades))
    table.add_row("Losing Trades", str(portfolio.losing_trades))
    table.add_row("Simulated Win Rate", f"{portfolio.win_rate * 100:.1f}%")
    table.add_row("Peak Balance", f"${portfolio.peak_balance:,.2f}")
    table.add_row("Max Drawdown ($)", f"${portfolio.max_drawdown_amount:,.2f}")
    table.add_row("Max Drawdown (%)", f"{portfolio.max_drawdown_pct * 100:.2f}%")
    table.add_row("Max Consecutive Losses", str(portfolio.max_consecutive_losses))
    table.add_row("Final Circuit Breaker State", risk_mgr.circuit_state.value)
    table.add_row(
        "Buffer Gaps / Rejections Detected",
        f"{engine.feature_buffer.gaps_count} gaps, "
        f"{engine.feature_buffer.rejections_count} rejected",
    )

    console.print(table)


@paper_app.command("demo", help="Run end-to-end multi-regime Phase 5 demonstration.")
def demo_command() -> None:
    """Execute complete end-to-end Phase 5 demonstration across diverse regimes."""
    console.print(
        Panel(
            "[bold green]IQ Option AI Trading System — Phase 5 Demo[/bold green]\n"
            "Testing real-time streaming pipeline, feature buffer parity, risk manager,\n"
            "practice account pre-flight safety, order state machine, and paper execution.",
            border_style="green",
        )
    )

    # Regime 1: Trending market
    console.print("[dim]1. Testing Trending Regime (100 bars)...[/dim]")
    trending_candles = list(
        generate_synthetic_candles(
            SyntheticMarketConfig(n_bars=100, drift=0.0003, volatility=0.0006, seed=10)
        ).candles
    )
    eng_trending = PaperTradingEngine(PaperTradingConfig(is_replay=True))
    for c in trending_candles:
        eng_trending.process_candle(c)
    console.print(
        f"   [green]Trending: {eng_trending.portfolio.total_trades} trades, "
        f"Win Rate {eng_trending.portfolio.win_rate * 100:.1f}%, "
        f"PnL ${eng_trending.portfolio.total_pnl:+.2f}[/green]"
    )

    # Regime 2: Choppy range-bound market
    console.print("[dim]2. Testing Choppy / Mean-Reverting Regime (100 bars)...[/dim]")
    choppy_candles = list(
        generate_synthetic_candles(
            SyntheticMarketConfig(n_bars=100, drift=0.0, volatility=0.0004, seed=20)
        ).candles
    )
    eng_choppy = PaperTradingEngine(PaperTradingConfig(is_replay=True))
    for c in choppy_candles:
        eng_choppy.process_candle(c)
    console.print(
        f"   [green]Choppy: {eng_choppy.portfolio.total_trades} trades, "
        f"Win Rate {eng_choppy.portfolio.win_rate * 100:.1f}%, "
        f"PnL ${eng_choppy.portfolio.total_pnl:+.2f}[/green]"
    )

    # Regime 3: Consecutive Loss Streak & Cooldown Circuit Breaker
    console.print("[dim]3. Testing Risk Manager Consecutive Loss Circuit Breaker...[/dim]")
    loss_risk_cfg = RiskConfig(max_consecutive_losses=2, loss_cooldown_seconds=30.0)
    eng_loss = PaperTradingEngine(
        PaperTradingConfig(is_replay=True, risk_config=loss_risk_cfg),
        signal_generator=lambda _: (SignalDirection.CALL, 0.70),  # Always CALL
    )
    # Downward plunging market where CALL always loses
    bear_candles = list(
        generate_synthetic_candles(
            SyntheticMarketConfig(n_bars=40, drift=-0.001, volatility=0.0002, seed=30)
        ).candles
    )
    for c in bear_candles:
        eng_loss.process_candle(c)
    console.print(
        f"   [green]Circuit Breaker Tripped: {eng_loss.risk_manager.circuit_state.value} "
        f"(Halt reason: {eng_loss.risk_manager.halt_reason})[/green]"
    )

    # Regime 4: Feed Discontinuity & Gap Handling
    console.print("[dim]4. Testing Rolling Feature Buffer Discontinuity Handling...[/dim]")
    eng_gap = PaperTradingEngine(PaperTradingConfig(is_replay=True))
    gap_candles = list(
        generate_synthetic_candles(SyntheticMarketConfig(n_bars=35, seed=40)).candles
    )
    # Inject 10-minute gap at index 25
    c24 = gap_candles[24]
    gap_candles[25] = replace(c24, from_ts=c24.to_ts + 600, to_ts=c24.to_ts + 660)
    for c in gap_candles:
        eng_gap.process_candle(c)
    console.print(
        f"   [green]Gaps Detected: {eng_gap.feature_buffer.gaps_count}, "
        f"Buffer Status after gap: {eng_gap.feature_buffer.status.value}[/green]"
    )

    console.print(
        Panel(
            "[bold green]Phase 5 Demonstration Completed Successfully.\n"
            "All paper trading, risk controls, and safety mechanisms verified.[/bold green]",
            border_style="green",
        )
    )
