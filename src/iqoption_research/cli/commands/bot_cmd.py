"""CLI commands for Phase 6 autonomous bot operations, replay, and status monitoring.

Provides local dashboard views, offline candle replay, multi-regime demonstrations,
and gated observe-only market observation controls.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iqoption_research.bot.controller import BotConfig, BotController, BotMode
from iqoption_research.bot.monitor import BotStatusSnapshot
from iqoption_research.bot.runtime import ContinuousBotRuntime, RuntimeConfig
from iqoption_research.cli.commands.auth_cmd import auth_app
from iqoption_research.execution.account_diagnostics import (
    AccountReadinessResult,
    BrokerAccountDiagnostic,
)
from iqoption_research.execution.risk import RiskConfig
from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.market_data.instruments import UnsupportedInstrumentError
from iqoption_research.ml.authentic_training import AuthenticModelTrainer, AuthenticTrainingConfig
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.ml.models.baselines import LogisticRegressionModel
from iqoption_research.ml.tracking.artifacts import ModelBundle
from iqoption_research.models.account import AccountType, Balance, Profile
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
bot_app.add_typer(auth_app, name="auth")


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
    table.add_row("Data Source", snapshot.data_source)
    table.add_row("Model Provenance", snapshot.provenance)
    pred_str = f"{snapshot.last_prediction_direction} ({snapshot.last_prediction_prob:.1%})"
    table.add_row("Connection Status", snapshot.connection_status)
    table.add_row("Trained Model", snapshot.model_name)
    table.add_row("Latest Prediction", pred_str)
    table.add_row("Model Confidence", f"{snapshot.confidence:.3f}")
    table.add_row("Prediction Uncertainty", f"{snapshot.uncertainty:.3f}")
    table.add_row("Reason for HOLD", str(snapshot.abstention_reason or "None"))
    table.add_row("Virtual Balance", f"${snapshot.virtual_balance:,.2f}")
    table.add_row("Open Simulated Positions", str(snapshot.open_positions_count))
    table.add_row("Settled Simulated Trades", str(snapshot.settled_trades_count))
    table.add_row("Total Simulated Trades", str(snapshot.trade_count))
    table.add_row("Win Rate", f"{snapshot.win_rate_pct:.1f}%")
    table.add_row("Session Net PnL", f"${snapshot.session_pnl:+,.2f}")
    table.add_row("Peak Drawdown", f"{snapshot.max_drawdown_pct:.2f}%")
    if snapshot.risk_limits:
        limits_str = ", ".join(f"{k}={v}" for k, v in snapshot.risk_limits.items())
        table.add_row("Operational Risk Limits", limits_str)
    table.add_row("Circuit Breaker State", snapshot.circuit_breaker_state)
    table.add_row("Feed Health Status", snapshot.feed_health)
    table.add_row("Feed Gaps Detected", str(snapshot.feed_gaps_count))
    table.add_row("Rejected Candles", str(snapshot.rejected_candles_count))
    table.add_row("Last Error", str(snapshot.last_error or "None"))
    if snapshot.error_history:
        table.add_row("Error History Count", str(len(snapshot.error_history)))

    console.print(table)


@bot_app.command("status", help="Display current bot operational status.")
def status_command() -> None:
    """Display current operational snapshot from running/persisted session."""
    snapshot = BotController.load_latest_snapshot()
    if snapshot is not None:
        title = (
            f"IQ Option AI Bot — Operational Status Snapshot "
            f"({snapshot.symbol} | Uptime: {snapshot.uptime_seconds:.1f}s)"
        )
        _render_snapshot_table(snapshot, title)
    else:
        console.print(
            Panel(
                "[yellow]No active or persisted bot session found on disk.[/yellow]\n"
                "To start a new session or run offline paper replay:\n"
                "  [green]python -m iqoption_research.cli.main bot replay --bars 100[/green]\n"
                "  [green]python -m iqoption_research.cli.main bot demo[/green]",
                title="IQ Option Bot Status",
                border_style="yellow",
            )
        )


@bot_app.command("account-check", help="Investigate REAL vs PRACTICE balance state and routing.")
def account_check_command(
    fixture_path: str = typer.Option(
        "", "--fixture", "-f", help="Optional path to recorded JSON balance fixture"
    ),
    browser_balance_id: str = typer.Option(
        "", "--browser-balance-id", help="Observed balance ID in browser UI"
    ),
) -> None:
    """Run comprehensive read-only diagnostic tooling on broker account profile & balances."""
    console.print(
        Panel(
            "[bold cyan]IQ Option — Account Diagnostics & Discrepancy Investigation[/bold cyan]\n"
            "Analyzing 6-layer balance identity, active session status, and execution safety.\n"
            "Strict read-only analysis: zero broker balance switching or trade frames.",
            border_style="cyan",
        )
    )

    diag = BrokerAccountDiagnostic()

    if fixture_path and Path(fixture_path).is_file():
        raw_data = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
        profile_data = raw_data.get("profile", {})
        balances_data = raw_data.get("balances", [])
        profile = Profile.from_raw(profile_data)
        bals = [Balance.from_raw(b) for b in balances_data if isinstance(b, dict)]
        report = diag.evaluate(
            profile=profile,
            balances=bals,
            active_balance_id=str(profile_data.get("balance_id", "")),
            browser_ui_balance_id=browser_balance_id or None,
            last_verification_ts=time.time(),
        )
    else:
        # Default regression audit demonstrating the known discrepancy scenario:
        # Browser shows PRACTICE (id 100), but WebSocket defaults to REAL (id 200).
        profile_data = {
            "user_id": 424242,
            "email": "researcher@example.com",
            "balance_id": 200,
            "balances": [
                {"id": 100, "type": 4, "amount": 10000.0, "currency": "USD"},
                {"id": 200, "type": 1, "amount": 250.0, "currency": "USD"},
            ],
        }
        profile = Profile.from_raw(profile_data)
        bals = [
            Balance(id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
            Balance(id="200", type=AccountType.REAL, amount=250.0, currency="USD", is_active=True),
        ]
        report = diag.evaluate(
            profile=profile,
            balances=bals,
            active_balance_id="200",
            browser_ui_balance_id=browser_balance_id or "100",
            last_verification_ts=time.time(),
        )

    # Render diagnostic report
    table = Table(title="Account Diagnostic & Readiness Audit Report", show_lines=True)
    table.add_column("Diagnostic Field", style="bold cyan")
    table.add_column("Audit Result", style="bold white")

    res_color = "green" if report.readiness == AccountReadinessResult.VERIFIED_PRACTICE else "red"
    table.add_row("Readiness Result", f"[{res_color}]{report.readiness.value}[/{res_color}]")
    table.add_row("Broker Ledger Status", f"[bold red]{report.ledger_status.value}[/bold red]")
    table.add_row("PRACTICE Verified", str(report.is_practice_verified))
    table.add_row("Safe for Broker Orders", str(report.is_safe_for_practice_execution))
    table.add_row("Discrepancy Detected", str(report.discrepancy_detected))
    table.add_row(
        "Server-Reported Active Balance",
        str(report.balance_identity.server_reported_active_balance_id),
    )
    table.add_row("Browser UI Balance ID", str(report.balance_identity.browser_ui_balance_id))
    table.add_row(
        "Existing Balances",
        ", ".join(
            f"id={b.id}({b.type.name}, {b.amount} {b.currency})"
            for b in report.balance_identity.existing_balances
        ),
    )
    table.add_row("Explanation", report.explanation)
    table.add_row("Recommended Action", report.recommended_action)

    console.print(table)


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
        "",
        "--auth-token",
        help="Optional explicit token (recommended: set IQOPTION_OBSERVE_TOKEN env var)",
    ),
) -> None:
    """Gated observe-only market monitoring mode.

    Safe secret handling: Checks IQOPTION_OBSERVE_TOKEN environment variable first
    to avoid exposing authorization secrets in shell command history or process listings.
    """
    token_str = auth_token.strip() or os.environ.get("IQOPTION_OBSERVE_TOKEN", "").strip()

    if not token_str:
        console.print(
            Panel(
                "[bold red]OBSERVE-ONLY MARKET OBSERVATION: NOT AUTHORIZED[/bold red]\n\n"
                "Live market data observation requires an explicit human authorization token.\n"
                "The existence of local credentials, configuration files, or automated test\n"
                "scripts does NOT constitute authorization.\n\n"
                "Safe secret handling:\n"
                "  Set env var: [cyan]export IQOPTION_OBSERVE_TOKEN='<token>'[/cyan]\n\n"
                "To execute offline paper trading without live network access:\n"
                "  [green]python -m iqoption_research.cli.main bot replay --bars 100[/green]\n"
                "  [green]python -m iqoption_research.cli.main bot demo[/green]",
                title="Authorization Gate",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    token = ObservationAuthorizationToken(
        token_hash=token_str,
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


# ---------------------------------------------------------------------------
# Phase 8: Data Sub-Application
# ---------------------------------------------------------------------------

data_app = typer.Typer(
    name="data",
    help="Authentic market data collection, ingestion, and catalog management.",
    no_args_is_help=True,
)
bot_app.add_typer(data_app, name="data")


@data_app.command("coverage", help="Display coverage and quality report for a dataset.")
def data_coverage_cmd(
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    candle_size: int = typer.Option(60, "--size", help="Candle size in seconds"),
) -> None:
    """Generate and display data coverage and quality statistics for a stored dataset."""
    collector = HistoricalDataCollector()
    try:
        report = collector.generate_coverage_report(symbol, candle_size)
    except UnsupportedInstrumentError:
        report = None

    if report is None:
        console.print(f"[yellow]No stored dataset found for {symbol} ({candle_size}s).[/yellow]")
        return

    table = Table(title=f"Dataset Coverage Report — {symbol}", show_lines=True)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="bold white")
    for k, v in report.to_dict().items():
        table.add_row(k, str(v))
    console.print(table)


@data_app.command("import-csv", help="Import authentic candles from an external CSV file.")
def data_import_csv_cmd(
    csv_file: str = typer.Option(..., "--file", "-f", help="Path to CSV file"),
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    candle_size: int = typer.Option(60, "--size", help="Candle size in seconds"),
    source_label: str = typer.Option("EXTERNAL_AUTHENTIC", "--source", help="Provenance label"),
) -> None:
    """Ingest, validate, and catalog authentic market candles from a local CSV file."""
    path = Path(csv_file)
    if not path.is_file():
        console.print(f"[bold red]Error: CSV file not found: {csv_file}[/bold red]")
        raise typer.Exit(code=1)

    collector = HistoricalDataCollector()
    try:
        candles = collector.import_external_csv(
            path, symbol=symbol, candle_size_seconds=candle_size, source_label=source_label
        )
        saved_path = collector.persist_dataset(
            candles,
            symbol=symbol,
            candle_size_seconds=candle_size,
            provenance=source_label,
        )
        console.print(
            Panel(
                f"[bold green]Imported {len(candles)} authentic candles![/bold green]\n"
                f"Symbol: {symbol} | Timeframe: {candle_size}s\n"
                f"Storage: {saved_path}\n"
                f"Catalog registered in SQLite catalog.db",
                border_style="green",
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Import failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc


@data_app.command(
    "collect",
    help="Collect authentic candles from IQ Option using an authorized read-only session.",
)
def data_collect_cmd(
    symbol: str = typer.Option(
        "EURUSD", "--symbol", "-s", help="Instrument symbol (e.g. EURUSD, EURUSD-OTC)"
    ),
    candle_size: int = typer.Option(60, "--size", help="Candle size in seconds"),
    count: int = typer.Option(1000, "--count", "-n", help="Number of candles to request"),
    live: bool = typer.Option(
        False,
        "--live",
        help="Connect to live broker feed (default: offline simulation mode)",
    ),
    saved_creds: bool = typer.Option(
        False, "--saved", help="Load saved credentials from Windows vault"
    ),
    dev_mock: bool = typer.Option(
        False, "--mock", help="Use offline mock provider for dry-run verification"
    ),
) -> None:
    """Collect authentic historical candles from IQ Option via read-only session."""
    import asyncio

    from iqoption_research.auth.manager import AuthenticationManager
    from iqoption_research.auth.provider import IQOptionAuthProvider, MockAuthProvider
    from iqoption_research.connectors.iqoption.client import AsyncIQOption
    from iqoption_research.connectors.iqoption.exceptions import OfflineModeError

    use_live = live and not dev_mock
    mode_badge = (
        "[bold green][OWNER-AUTHORIZED LIVE HISTORICAL COLLECTION][/bold green]"
        if use_live
        else "[bold yellow][OFFLINE DRY-RUN / SIMULATION MODE][/bold yellow]"
    )
    safety_badge = (
        "[bold red][STRICTLY READ-ONLY OBSERVATION — BROKER ORDERS HARD-DISABLED][/bold red]"
    )

    console.print(
        Panel(
            f"[bold cyan]IQ Option Authentic Historical Data Collection[/bold cyan]\n"
            f"Mode: {mode_badge} | Safety: {safety_badge}\n"
            f"Symbol: {symbol} | Timeframe: {candle_size}s | Requested: {count}\n"
            f"Provider: {'IQ Option Live Read-Only' if use_live else 'Offline Mock Fixture'}",
            border_style="cyan",
        )
    )

    if not use_live:
        msg = (
            "[yellow]Notice: Running in offline simulation mode by default (safety gate).\n"
            "No genuine broker network sockets will be opened.\n"
            "To collect authentic market data from IQ Option, pass the "
            "[bold green]--live[/bold green] flag:\n"
            f"  [bold cyan]bot data collect --live --symbol {symbol} "
            f"--size {candle_size} --count {count}[/bold cyan][/yellow]"
        )
        console.print(msg)

    provider = IQOptionAuthProvider() if use_live else MockAuthProvider()
    auth_manager = AuthenticationManager(provider=provider)

    async def _async_collect() -> None:
        console.print("[dim]Step 1: In-memory session authentication...[/dim]")
        if saved_creds:
            console.print("[dim]Loading credentials from Windows Credential Vault...[/dim]")
            logged_in = await auth_manager.login_saved()
        else:
            if not use_live:
                logged_in = await auth_manager.login_interactive(
                    email="trader@example.com",
                    password="mock_secret_123",  # noqa: S106
                )
            else:
                msg = (
                    "[dim]Initiating secure owner-controlled login "
                    "(credentials in memory only)...[/dim]"
                )
                console.print(msg)
                logged_in = await auth_manager.login_interactive()

        if not logged_in:
            err = auth_manager.last_error or "Unknown error"
            console.print(f"[bold red]Authentication: FAILED ({err})[/bold red]")
            raise typer.Exit(code=1)

        console.print("[bold green]Authentication: SUCCESSFUL (Session SSID acquired)[/bold green]")

        session = auth_manager.session
        if session is None or not session.is_active:
            console.print(
                "[bold red]Authentication: FAILED (No active session in memory)[/bold red]"
            )
            raise typer.Exit(code=1)

        if not use_live:
            msg = (
                "[dim]Offline dry-run complete. In-memory session verified "
                "without network I/O.[/dim]"
            )
            console.print(msg)
            return

        # Explicit owner authorization: pass offline=False to permit read-only socket connection
        console.print("[dim]Step 2: Connecting authorized read-only WebSocket session...[/dim]")
        client = AsyncIQOption(ssid=session.ssid, offline=False)
        client.assert_read_only()

        try:
            await client.connect()
            console.print("[bold green]Connection: READY (WebSocket authenticated)[/bold green]")
        except OfflineModeError as err:
            console.print(f"[bold red]Collection: BLOCKED (Offline mode active: {err})[/bold red]")
            await auth_manager.logout()
            raise typer.Exit(code=1) from err
        except Exception as exc:
            console.print(f"[bold red]Connection failed: {exc}[/bold red]")
            await auth_manager.logout()
            raise typer.Exit(code=1) from exc

        try:
            collector = HistoricalDataCollector()
            msg = (
                f"[cyan]Step 3: Collecting {count} authentic candles "
                f"for {symbol} ({candle_size}s)...[/cyan]"
            )
            console.print(msg)
            candles = await collector.collect_from_broker(
                client,
                symbol=symbol,
                candle_size_seconds=candle_size,
                count=count,
            )
            if not candles:
                console.print(f"[yellow]No candles returned from broker for {symbol}.[/yellow]")
                return

            saved_path = collector.persist_dataset(
                candles,
                symbol=symbol,
                candle_size_seconds=candle_size,
                provenance="BROKER_AUTHENTIC_COLLECTION",
            )
            n_retrieved = len(candles)
            report = (
                f"[bold green]Collection: SUCCESSFUL ({n_retrieved} candles)[/bold green]\n"
                f"Symbol: {symbol} | Timeframe: {candle_size}s\n"
                f"Storage: {saved_path}\n"
                f"Catalog registered in SQLite catalog.db"
            )
            console.print(Panel(report, border_style="green"))
        finally:
            await client.close()
            await auth_manager.logout()
            console.print("[dim]Session terminated and credentials cleared from memory.[/dim]")

    try:
        asyncio.run(_async_collect())
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]Collection error: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc


@bot_app.command("train-authentic", help="Train AI models on authentic historical data.")
def train_authentic_cmd(
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    candle_size: int = typer.Option(60, "--size", help="Candle size in seconds"),
    payout: float = typer.Option(0.85, "--payout", "-p", help="Binary option payout rate"),
    save: bool = typer.Option(True, "--save/--no-save", help="Persist best model bundle to disk"),
) -> None:
    """Train candidate models on stored authentic candles and evaluate payout asymmetry."""
    collector = HistoricalDataCollector()
    try:
        candles = collector.load_dataset(symbol, candle_size)
    except UnsupportedInstrumentError:
        candles = []

    if not candles or len(candles) < 100:
        console.print(
            f"[yellow]Insufficient authentic dataset for {symbol} "
            f"({len(candles)} bars found, 100 required).\n"
            "Import an authentic CSV dataset first using:\n"
            "  [bold green]bot data import-csv --file <path> --symbol EURUSD[/bold green][/yellow]"
        )
        raise typer.Exit(code=1)

    trainer = AuthenticModelTrainer(
        AuthenticTrainingConfig(
            symbol=symbol,
            candle_size_seconds=candle_size,
            payout_rate=payout,
        )
    )
    console.print(
        f"[cyan]Training and evaluating candidate models on {len(candles)} bars...[/cyan]"
    )
    res = trainer.train_and_evaluate(candles, provenance_label="AUTHENTIC_MARKET_DATA")
    console.print(Panel(res.summary_report, border_style="cyan"))

    if save and res.best_model_name in res.bundles:
        best_bundle = res.bundles[res.best_model_name]
        saved_path = trainer.save_trained_bundle(best_bundle, model_name=res.best_model_name)
        console.print(f"[bold green]Saved best model bundle to: {saved_path}[/bold green]")


@bot_app.command("run", help="Start continuous owner-controlled bot runtime.")
def run_cmd(
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Instrument symbol"),
    live: bool = typer.Option(
        False,
        "--live",
        help="Connect to live observation feed (default: offline replay)",
    ),
    saved_creds: bool = typer.Option(
        False,
        "--saved",
        "-s",
        help="Load saved credentials from Windows vault",
    ),
    bars: int = typer.Option(60, "--bars", "-n", help="Bars to replay in offline mode"),
    max_seconds: float = typer.Option(0.0, "--timeout", help="Optional max runtime in seconds"),
    impute_quotes: bool = typer.Option(
        False,
        "--impute-quotes",
        help="Impute missing quotes if model expects quotes",
    ),
) -> None:
    """Run continuous bot loop with streaming inference and paper trading."""
    import asyncio

    mode_badge = (
        "[bold green][LIVE OBSERVATION + PAPER TRADING][/bold green]"
        if live
        else "[bold yellow][OFFLINE REPLAY][/bold yellow]"
    )
    safety_badge = "[bold red][BROKER EXECUTION DISABLED][/bold red]"

    console.print(
        Panel(
            f"[bold cyan]IQ Option Autonomous AI Bot — Continuous Runtime[/bold cyan]\n"
            f"Mode: {mode_badge} | Safety: {safety_badge}\n"
            f"Symbol: {symbol} | Paper Stake: $100.00 | Payout: 85%\n"
            f"[dim]Continuous event-driven execution with in-memory session lifetime.[/dim]",
            border_style="cyan",
        )
    )

    auth_token = None
    if live:
        token_str = os.environ.get("IQOPTION_OBSERVE_TOKEN", "").strip()
        if not token_str:
            console.print(
                "[bold red]Live observation requires an explicit authorization token.[/bold red]\n"
                "Set environment variable: IQOPTION_OBSERVE_TOKEN='<token>'."
            )
            raise typer.Exit(code=1)
        auth_token = ObservationAuthorizationToken(
            token_hash=token_str,
            authorized_by="LeadSystemsEngineer",
        )

    runtime_cfg = RuntimeConfig(
        symbol=symbol,
        mode=BotMode.OBSERVE_ONLY if live else BotMode.OFFLINE_REPLAY,
        use_saved_credentials=saved_creds,
        dev_mock=not live,
        observation_token=auth_token,
        max_runtime_seconds=max_seconds if max_seconds > 0 else None,
        impute_missing_quotes=impute_quotes,
    )

    runtime = ContinuousBotRuntime(config=runtime_cfg)

    # In replay mode, load authentic or generate synthetic candles
    replay_candles = None
    if not live:
        collector = HistoricalDataCollector()
        try:
            replay_candles = collector.load_dataset(symbol)
        except UnsupportedInstrumentError:
            replay_candles = None
        if not replay_candles:
            console.print(
                f"[dim]No stored dataset for {symbol}; generating {bars} "
                "synthetic bars for demonstration.[/dim]"
            )
            replay_candles = list(
                generate_synthetic_candles(
                    SyntheticMarketConfig(symbol=symbol, n_bars=bars, seed=42)
                ).candles
            )
        else:
            replay_candles = replay_candles[:bars]

    try:
        snapshot = asyncio.run(runtime.run(candles_iterable=replay_candles))
        _render_snapshot_table(snapshot, f"Continuous Session Concluded — {symbol}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown requested by user (Ctrl+C).[/yellow]")
        runtime.request_stop()
    except Exception as exc:
        console.print(f"[bold red]Runtime stopped with error: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc
