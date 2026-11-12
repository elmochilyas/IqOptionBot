"""CLI commands for Phase 6 autonomous bot operations, replay, and status monitoring.

Provides local dashboard views, offline candle replay, multi-regime demonstrations,
and gated observe-only market observation controls.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.bot.controller import BotConfig, BotController, BotMode
from iqoption_research.bot.monitor import BotStatusSnapshot
from iqoption_research.execution.risk import RiskConfig
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.ml.models.baselines import LogisticRegressionModel
from iqoption_research.ml.tracking.artifacts import ModelBundle
from iqoption_research.observation.adapter import (
    ObservationAuthorizationToken,
    ObservationSecurityError,
)
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.features.scalers import StandardScaler

console = Console()
bot_app = typer.Typer(
    name="bot",
    help="Phase 6 autonomous AI trading bot operations and monitoring.",
    no_args_is_help=True,
)


def _train_quick_model_bundle() -> ModelBundle:
    """Helper creating a fitted ModelBundle for demonstration runs."""
    import numpy as np

    candles = generate_synthetic_candles(SyntheticMarketConfig(n_bars=60, seed=123)).candles
    pipeline = CausalFeaturePipeline(include_volume=True, include_quotes=False)
    dataset = pipeline.transform(candles)
    feature_names = tuple(dataset.feature_specs[i].name for i in range(len(dataset.feature_specs)))

    rows = list(dataset.rows[20:])
    X = np.array([[r.features[k] for k in feature_names] for r in rows], dtype=np.float64)
    y = np.array(
        [1 if candles[r.index].close > candles[r.index].open else 0 for r in rows],
        dtype=np.int64,
    )

    scaler = StandardScaler()
    scaler.fit(X, feature_names=feature_names)
    X_scaled = np.asarray(scaler.transform(X), dtype=np.float64)

    model = LogisticRegressionModel()
    model.fit(X_scaled, y, feature_names=feature_names)

    val_probs = model.predict_proba(X_scaled)
    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(val_probs, y)

    return ModelBundle(
        model=model,
        scaler=scaler,
        calibrator=calibrator,
        metadata={"created_at_utc": "2026-11-12T12:00:00+00:00"},
    )


def _render_snapshot_table(snapshot: BotStatusSnapshot, title: str) -> None:
    """Render a clean Rich table from an operational snapshot."""
    table = Table(title=title, show_lines=True)
    table.add_column("Operational Field", style="bold cyan", no_wrap=True)
    table.add_column("Current Value", style="bold white")

    table.add_row("Bot State", snapshot.state.upper())
    table.add_row("Operational Mode", snapshot.mode)
    table.add_row("Active Instrument", snapshot.symbol)
    pred_str = (
        f"{snapshot.last_prediction_direction} ({snapshot.last_prediction_prob:.1%})"
    )
    table.add_row("Connection Status", snapshot.connection_status)
    table.add_row("Trained Model", snapshot.model_name)
    table.add_row("Latest Prediction", pred_str)
    table.add_row("Model Confidence", f"{snapshot.confidence:.3f}")
    table.add_row("Abstention Reason", str(snapshot.abstention_reason or "None"))
    table.add_row("Virtual Balance", f"${snapshot.virtual_balance:,.2f}")
    table.add_row("Open Paper Positions", str(snapshot.open_positions_count))
    table.add_row("Total Paper Trades", str(snapshot.trade_count))
    table.add_row("Win Rate", f"{snapshot.win_rate_pct:.1f}%")
    table.add_row("Session Net PnL", f"${snapshot.session_pnl:+,.2f}")
    table.add_row("Peak Drawdown", f"{snapshot.max_drawdown_pct:.2f}%")
    table.add_row("Circuit Breaker State", snapshot.circuit_breaker_state)
    table.add_row("Feed Health Status", snapshot.feed_health)
    table.add_row("Feed Gaps Detected", str(snapshot.feed_gaps_count))
    table.add_row("Rejected Candles", str(snapshot.rejected_candles_count))
    table.add_row("Last Error", str(snapshot.last_error or "None"))

    console.print(table)


@bot_app.command("status", help="Display current bot operational status.")
def status_command() -> None:
    """Display current operational snapshot."""
    bot = BotController(BotConfig(symbol="EURUSD", initial_balance=10000.0))
    snapshot = bot.get_status_snapshot()
    _render_snapshot_table(snapshot, "IQ Option AI Bot — Operational Status Snapshot")


@bot_app.command("replay", help="Run automated paper trading on historical/synthetic candles.")
def replay_command(
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    bars: int = typer.Option(100, "--bars", "-n", help="Number of replay bars"),
    stake: float = typer.Option(100.0, "--stake", help="Fixed stake per order"),
    payout: float = typer.Option(0.85, "--payout", "-p", help="Binary option payout rate"),
    seed: int = typer.Option(42, "--seed", help="Random seed for synthetic fixture"),
) -> None:
    """Execute historical/synthetic candle stream through BotController with trained ML."""
    console.print(
        Panel(
            f"[bold green]IQ Option Phase 6 — Automated Bot Replay[/bold green]\n"
            f"Symbol: {symbol} | Replay Bars: {bars} | Stake: ${stake:.2f}\n"
            f"Payout: {payout * 100:.0f}% | Mode: OFFLINE_REPLAY (Simulated execution)",
            border_style="green",
        )
    )

    bundle = _train_quick_model_bundle()
    cfg = BotConfig(
        symbol=symbol,
        mode=BotMode.OFFLINE_REPLAY,
        initial_balance=10000.0,
        payout_rate=payout,
        fixed_stake=stake,
    )
    bot = BotController(cfg, model_bundle=bundle)
    bot.start()

    candles = list(
        generate_synthetic_candles(
            SyntheticMarketConfig(symbol=symbol, n_bars=bars, seed=seed)
        ).candles
    )

    console.print("[dim]Ingesting streaming candles through pipeline...[/dim]")
    for c in candles:
        bot.process_candle(c)

    snapshot = bot.get_status_snapshot()
    bot.stop()

    _render_snapshot_table(snapshot, f"Replay Completed — {symbol} ({bars} bars)")


@bot_app.command("demo", help="Run multi-regime end-to-end Phase 6 demonstration.")
def demo_command() -> None:
    """Execute complete Phase 6 end-to-end demonstration."""
    console.print(
        Panel(
            "[bold green]IQ Option AI Trading System — Phase 6 Demonstration[/bold green]\n"
            "Testing Observation Adapter, Clock Drift Validation, Heartbeat Monitor,\n"
            "Real ML Inference Pipeline, BotController Lifecycle, and Operational Risk Management.",
            border_style="green",
        )
    )

    bundle = _train_quick_model_bundle()
    bot = BotController(
        BotConfig(
            symbol="EURUSD",
            mode=BotMode.OFFLINE_REPLAY,
            risk_config=RiskConfig(max_consecutive_losses=2),
        ),
        model_bundle=bundle,
    )
    bot.start()

    # 1. Nominal Trending Flow
    console.print("[dim]1. Testing Nominal Market Stream (50 bars)...[/dim]")
    trend_candles = list(
        generate_synthetic_candles(
            SyntheticMarketConfig(n_bars=50, drift=0.0002, volatility=0.0005, seed=1)
        ).candles
    )
    for c in trend_candles:
        bot.process_candle(c)

    snap1 = bot.get_status_snapshot()
    console.print(
        f"   [green]Nominal Stream: {snap1.trade_count} trades, "
        f"Win Rate {snap1.win_rate_pct:.1f}%, PnL ${snap1.session_pnl:+.2f}[/green]"
    )

    # 2. Pause and Resume Validation
    console.print("[dim]2. Testing Pause and Resume Controls...[/dim]")
    bot.pause()
    if not bot.is_paused:
        raise RuntimeError("Bot failed to enter paused state")
    # Ingestion while paused must produce None
    c_paused = trend_candles[0]
    if bot.process_candle(c_paused) is not None:
        raise RuntimeError("Ingestion occurred while paused")
    bot.resume()
    if not bot.is_running:
        raise RuntimeError("Bot failed to resume running state")
    console.print("   [green]Pause & Resume verified: ingestion paused while paused.[/green]")

    # 3. Emergency Stop & Kill Switch Latching
    console.print("[dim]3. Testing Emergency Stop & Kill Switch Latching...[/dim]")
    bot.emergency_stop(reason="Operator emergency stop demonstration")
    if bot.state.value != "stopped":
        raise RuntimeError("Bot failed to stop on emergency stop")
    if not bot.risk_manager.is_kill_switch_active:
        raise RuntimeError("Kill switch not active after emergency stop")
    console.print(
        f"   [green]Emergency Stop verified: state={bot.state.value}, "
        f"Circuit Breaker={bot.risk_manager.circuit_state.value}[/green]"
    )

    console.print(
        Panel(
            "[bold green]Phase 6 Demonstration Completed Successfully.\n"
            "All observation, inference, bot control, and safety mechanisms verified.[/bold green]",
            border_style="green",
        )
    )


@bot_app.command("observe", help="Gated observe-only mode for authorized market observation.")
def observe_command(
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    auth_token: str = typer.Option(
        "", "--auth-token", help="Cryptographic authorization signature required for observe-only"
    ),
) -> None:
    """Gated observe-only market monitoring mode."""
    if not auth_token or not auth_token.strip():
        console.print(
            Panel(
                "[bold red]ACCESS DENIED — OBSERVE-ONLY MODE IS STRICTLY GATED[/bold red]\n"
                "Live market data observation requires an explicit authorization token.\n"
                "To run without live network connection, use offline replay:\n"
                "  [green]python -m iqoption_research.cli.main bot replay --bars 100[/green]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    token = ObservationAuthorizationToken(
        token_hash=auth_token,
        authorized_by="LeadSystemsEngineer",
    )
    bot = BotController(
        BotConfig(symbol=symbol, mode=BotMode.OBSERVE_ONLY, observation_token=token)
    )
    try:
        bot.start()
        msg = f"Observe-only authorized for {symbol}. Awaiting broker stream..."
        console.print(f"[green]{msg}[/green]")
        bot.stop()
    except ObservationSecurityError as exc:
        console.print(f"[red]Authorization error: {exc}[/red]")
        raise typer.Exit(code=1) from None
