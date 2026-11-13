"""Continuous bot runtime coordinating authentication, live observation, and paper trading.

Implements Sections 3, 4, 9, 11:
- Long-running, owner-controlled async runtime without multi-process restarts
- In-memory authenticated session lifetime with zero disk-persistence of passwords or SSID tokens
- Live observation stream integration with strictly read-only broker connectors
- Dynamic streaming inference, probability calibration, risk management, and forward paper trading
- Interactive start, pause, resume, emergency stop, and graceful shutdown controls
- Complete telemetry logging and session persistence
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from iqoption_research.auth.manager import AuthenticationManager
from iqoption_research.auth.provider import IQOptionAuthProvider, MockAuthProvider
from iqoption_research.bot.controller import BotConfig, BotController, BotMode
from iqoption_research.bot.monitor import BotStatusSnapshot
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.execution.risk import RiskConfig
from iqoption_research.inference.loader import SafeModelArtifactLoader
from iqoption_research.inference.pipeline import ModelInferencePipeline
from iqoption_research.inference.validation import ModelIntegrityValidator
from iqoption_research.ml.tracking.artifacts import ModelBundle
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.observation.adapter import (
    MarketObservationAdapter,
    ObservationAuthorizationToken,
    ObservationMode,
)
from iqoption_research.observation.heartbeat import FeedHealthStatus

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_DIR: Final[Path] = Path("data/bot_sessions")


class RuntimeState(StrEnum):
    """Lifecycle states of the continuous bot runtime."""

    IDLE = "IDLE"
    AUTHENTICATING = "AUTHENTICATING"
    AUTHENTICATED = "AUTHENTICATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Consolidated configuration for continuous bot runtime execution."""

    symbol: str = "EURUSD"
    candle_size_seconds: int = 60
    mode: BotMode = BotMode.OFFLINE_REPLAY
    initial_balance: float = 10000.0
    payout_rate: float = 0.85
    fixed_stake: float = 100.0
    model_path: Path | str | None = None
    use_saved_credentials: bool = False
    dev_mock: bool = True
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    observation_token: ObservationAuthorizationToken | None = None
    max_runtime_seconds: float | None = None
    startup_grace_seconds: float = 30.0
    impute_missing_quotes: bool = False


class ContinuousBotRuntime:
    """Long-running async runtime orchestrating broker observation, ML, and paper trading."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        auth_manager: AuthenticationManager | None = None,
        client: AsyncIQOption | None = None,
        model_bundle: ModelBundle | None = None,
        sessions_dir: Path | str | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.state = RuntimeState.IDLE
        self.auth_manager = auth_manager or self._init_auth_manager()
        self.client = client
        self.model_bundle = model_bundle
        self.sessions_dir = Path(sessions_dir) if sessions_dir is not None else DEFAULT_SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self._controller: BotController | None = None
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default
        self._last_error: str | None = None
        self._started_at: float | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None

    def _init_auth_manager(self) -> AuthenticationManager:
        provider = MockAuthProvider() if self.config.dev_mock else IQOptionAuthProvider()
        return AuthenticationManager(provider=provider)

    @property
    def is_running(self) -> bool:
        return self.state == RuntimeState.RUNNING

    @property
    def is_paused(self) -> bool:
        return self.state == RuntimeState.PAUSED

    @property
    def controller(self) -> BotController | None:
        return self._controller

    def pause(self) -> None:
        """Pause trade execution while retaining market observation."""
        if self.state != RuntimeState.RUNNING:
            raise RuntimeError(f"Cannot pause runtime from state '{self.state.value}'")
        self._pause_event.clear()
        if self._controller is not None:
            self._controller.pause()
        self.state = RuntimeState.PAUSED
        logger.info("Continuous bot runtime PAUSED by operator.")

    def resume(self) -> None:
        """Resume trade execution from paused state."""
        if self.state != RuntimeState.PAUSED:
            raise RuntimeError(f"Cannot resume runtime from state '{self.state.value}'")
        self._pause_event.set()
        if self._controller is not None:
            self._controller.resume()
        self.state = RuntimeState.RUNNING
        logger.info("Continuous bot runtime RESUMED by operator.")

    def request_stop(self) -> None:
        """Signal the continuous runtime loop to shut down cleanly."""
        logger.info("Stop requested by operator.")
        self.state = RuntimeState.STOPPING
        self._shutdown_event.set()

    def emergency_stop(self, reason: str = "Operator triggered emergency kill switch") -> None:
        """Immediately trip risk kill switch and halt runtime."""
        logger.critical("EMERGENCY STOP TRIGGERED: %s", reason)
        if self._controller is not None:
            self._controller.emergency_stop(reason=reason)
        self.state = RuntimeState.STOPPED
        self._shutdown_event.set()

    async def start_interactive_session(
        self,
        *,
        email: str | None = None,
        password: str | None = None,
    ) -> bool:
        """Authenticate in-memory session interactively or via saved vault."""
        self.state = RuntimeState.AUTHENTICATING
        if self.config.use_saved_credentials:
            logger.info("Loading credentials from Windows Credential Vault...")
            success = await self.auth_manager.login_saved()
        else:
            logger.info("Authenticating interactively...")
            success = await self.auth_manager.login_interactive(email=email, password=password)

        if not success:
            self.state = RuntimeState.ERROR
            self._last_error = self.auth_manager.last_error or "Authentication failed"
            return False

        self.state = RuntimeState.AUTHENTICATED
        return True

    def _prepare_model_bundle(self) -> ModelBundle:
        """Load and validate model bundle from configured path or fallback."""
        if self.model_bundle is not None:
            return self.model_bundle

        if self.config.model_path is not None:
            path = Path(self.config.model_path)
            if not path.exists():
                raise FileNotFoundError(f"Configured model artifact not found: {path}")
            loader = SafeModelArtifactLoader()
            bundle, _report = loader.load(path)
            return bundle

        # Build quick baseline bundle if none provided
        from iqoption_research.cli.commands.bot_cmd import _train_quick_model_bundle

        logger.info("No external model path supplied; initializing baseline bundle.")
        return _train_quick_model_bundle()

    async def run(
        self,
        *,
        candles_iterable: list[Candle] | None = None,
    ) -> BotStatusSnapshot:
        """Execute the continuous runtime loop.

        Supports:
        - OFFLINE_REPLAY with deterministic historical or synthetic candles.
        - OBSERVE_ONLY with live authenticated WebSocket completed-candle feed.
        """
        self.state = RuntimeState.STARTING
        self._started_at = time.time()
        self._shutdown_event.clear()
        self._pause_event.set()

        # 1. Prepare and validate model bundle
        bundle = self._prepare_model_bundle()
        validator = ModelIntegrityValidator()
        report = validator.validate_bundle(bundle)
        if not report.is_valid:
            self.state = RuntimeState.ERROR
            self._last_error = f"Model validation failed: {'; '.join(report.errors)}"
            raise ValueError(self._last_error)

        # Check stream compatibility
        has_quotes = self.config.impute_missing_quotes
        has_vol = getattr(self.config, "include_volume", True)
        compat, compat_err = validator.check_stream_compatibility(
            bundle, has_quotes=has_quotes, has_volume=has_vol
        )
        if not compat:
            logger.warning("Stream compatibility notice: %s", compat_err)

        inference = ModelInferencePipeline(bundle)

        # 2. Setup observation adapter
        obs_mode = (
            ObservationMode.OBSERVE_ONLY
            if self.config.mode == BotMode.OBSERVE_ONLY
            else ObservationMode.OFFLINE_REPLAY
        )
        adapter = MarketObservationAdapter(
            mode=obs_mode,
            symbol=self.config.symbol,
            candle_size_seconds=self.config.candle_size_seconds,
            authorization_token=self.config.observation_token,
        )

        # 3. Setup bot controller
        bot_cfg = BotConfig(
            symbol=self.config.symbol,
            mode=self.config.mode,
            candle_size_seconds=self.config.candle_size_seconds,
            initial_balance=self.config.initial_balance,
            payout_rate=self.config.payout_rate,
            fixed_stake=self.config.fixed_stake,
            risk_config=self.config.risk_config,
            observation_token=self.config.observation_token,
        )
        self._controller = BotController(
            config=bot_cfg,
            model_bundle=bundle,
            inference_pipeline=inference,
            observation_adapter=adapter,
        )

        # 4. Start controller
        self._controller.start()
        self.state = RuntimeState.RUNNING

        try:
            # 5. Ingestion loop
            if self.config.mode == BotMode.OFFLINE_REPLAY:
                await self._run_replay_loop(candles_iterable or [])
            else:
                await self._run_live_observation_loop()
        except asyncio.CancelledError:
            logger.info("Runtime execution cancelled.")
        except Exception as exc:
            logger.error("Error in continuous runtime loop: %s", exc)
            self.state = RuntimeState.ERROR
            self._last_error = str(exc)
            raise
        finally:
            await self._cleanup()

        return self._controller.get_status_snapshot()

    async def _run_replay_loop(self, candles: list[Candle]) -> None:
        """Ingest historical/synthetic candle series in replay mode."""
        for candle in candles:
            if self._shutdown_event.is_set():
                break

            await self._pause_event.wait()

            if (
                self.config.max_runtime_seconds is not None
                and self._started_at is not None
                and (time.time() - self._started_at) >= self.config.max_runtime_seconds
            ):
                logger.info("Max runtime seconds reached. Halting replay.")
                break

            if self._controller is not None:
                self._controller.process_candle(candle, now_ts=float(candle.to_ts))

            # Small yield to permit event handling
            await asyncio.sleep(0.001)

    async def _watchdog_loop(self) -> None:
        """Independent timer-driven feed health watchdog."""
        from iqoption_research.observation.watchdog import FeedHealthWatchdog
        if self._controller is None:
            return
        watchdog = FeedHealthWatchdog(self._controller.observation_adapter)
        while not self._shutdown_event.is_set():
            try:
                watchdog.poll(now_ts=time.time())
                # If feed latched as stale/disconnected, block ingestion
                if watchdog.is_latched and self._controller is not None:
                    ls = watchdog.latched_status
                    # is_latched + controller not None means ingestion blocking is valid
                    if ls is not None and ls in {
                        FeedHealthStatus.STALE,
                        FeedHealthStatus.DISCONNECTED,
                    }:
                        self._controller.block_ingestion(
                            f"Feed health {ls.value}: {watchdog.latched_reason}"
                        )
                    logger.warning("Feed health issue latched: %s", watchdog.latched_reason)
            except Exception as exc:  # noqa BL001
                logger.error("Watchdog poll error: %s", exc)
                if self._controller is not None:
                    self._controller.block_ingestion(f"Watchdog error: {exc}")
                self.state = RuntimeState.ERROR
                self._last_error = f"Watchdog poll error: {exc}"
            # Poll every 5 seconds, respecting shutdown
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=5.0)
            except TimeoutError:
                continue

    async def _run_live_observation_loop(self) -> None:
        """Ingest live completed market candles from authorized read-only broker feed."""
        if self._controller is None:
            return

        adapter = self._controller.observation_adapter
        adapter.assert_mode_permission()

        # Start interactive session if not already authenticated
        if self.auth_manager.session is None or not self.auth_manager.session.is_active:
            await self.start_interactive_session()

        if self.client is None:
            # Initialize read-only client using active authenticated session
            session = self.auth_manager.session
            if session is None or not session.is_active:
                raise RuntimeError("No active session available for live observation.")
            self.client = AsyncIQOption(
                ssid=session.ssid,
                offline=self.config.dev_mock,
            )

        self.client.assert_read_only()
        await self.client.connect()

        # Schedule the independent watchdog task for feed health monitoring
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

        # Bounded startup grace period: fail closed if no valid candle arrives
        self._startup_deadline = time.monotonic() + self.config.startup_grace_seconds

        try:
            async for raw_candle in self.client.stream_candles(
                self.config.symbol,
                size=CandleSize(self.config.candle_size_seconds),
            ):
                if self._shutdown_event.is_set():
                    break

                # Enforce startup grace period - fail closed if no candle before deadline
                if time.monotonic() > self._startup_deadline:
                    logger.error(
                        "Startup grace period exceeded; no valid candle received. "
                        "Halting."
                    )
                    self.state = RuntimeState.ERROR
                    self._last_error = "Startup grace period exceeded; no valid candle received"
                    self._controller.emergency_stop(reason="Startup grace period exceeded")
                    break

                await self._pause_event.wait()

                # Enforce max_runtime_seconds independently of candle arrival
                if (
                    self.config.max_runtime_seconds is not None
                    and self._started_at is not None
                    and (time.time() - self._started_at) >= self.config.max_runtime_seconds
                ):
                    logger.info("Max observation runtime reached. Halting live stream.")
                    break

                self._controller.process_candle(raw_candle)
        finally:
            # Cancel the watchdog task first, then close the client
            self._watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog_task
            await self.client.close()

    async def _cleanup(self) -> None:
        """Orderly shutdown of controller, sessions, and persistence."""
        logger.info("Commencing clean runtime shutdown...")

        # Preserve ERROR state through cleanup - do not overwrite to STOPPED
        if self.state == RuntimeState.ERROR:
            logger.error(
                "Preserving ERROR state through cleanup: %s",
                self._last_error or "unknown",
            )
        else:
            self.state = RuntimeState.STOPPING

        if self._controller is not None:
            self._controller.stop()
            snapshot = self._controller.get_status_snapshot()

            # Save post-session report
            session_id = f"session_{int(time.time())}"
            summary_path = self.sessions_dir / f"{session_id}_summary.json"
            all_session_orders = (
                self._controller.session.settled_orders + self._controller.session.open_orders
            )
            summary_data = {
                "session_id": session_id,
                "closed_at_utc": datetime.now(UTC).isoformat(),
                "mode": self.config.mode.value,
                "symbol": self.config.symbol,
                "snapshot": snapshot.to_dict(),
                "trades": [
                    {
                        "order_id": o.order_id,
                        "side": o.side.value,
                        "stake": o.stake,
                        "entry_price": o.entry_price,
                        "entry_ts": o.entry_ts,
                        "expiry_ts": o.expiry_ts,
                        "state": o.state.value,
                        "simulated_pnl": o.simulated_pnl,
                    }
                    for o in all_session_orders
                ],
            }
            summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
            logger.info("Session telemetry persisted to %s", summary_path)

        # Cancel the watchdog task first
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog_task

        # Clear in-memory session on full stop
        await self.auth_manager.logout()

        # Preserve ERROR state; only transition to STOPPED if not already in ERROR
        if self.state != RuntimeState.ERROR:
            self.state = RuntimeState.STOPPED
        logger.info("Runtime shutdown complete. Active session cleared.")

    def get_status_snapshot(self) -> BotStatusSnapshot | None:
        if self._controller is not None:
            return self._controller.get_status_snapshot()
        return None
