"""Bot controller managing automated execution, lifecycle state transitions, and safety controls.

Orchestrates the market observation adapter, ML inference pipeline, operational risk controller,
and paper-trading engine into a unified event-driven trading system.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

from iqoption_research.bot.monitor import BotStatusMonitor, BotStatusSnapshot
from iqoption_research.bot.session import TradingSession
from iqoption_research.execution.engine import (
    EngineStepResult,
    PaperTradingConfig,
    PaperTradingEngine,
)
from iqoption_research.execution.risk import (
    OperationalRiskManager,
    RiskConfig,
)
from iqoption_research.inference.pipeline import ModelInferencePipeline
from iqoption_research.ml.tracking.artifacts import ModelBundle
from iqoption_research.models.market import Candle
from iqoption_research.observation.adapter import (
    MarketObservationAdapter,
    ObservationAuthorizationToken,
    ObservationMode,
    ObservationSecurityError,
)
from iqoption_research.research.backtest.models import SignalDirection
from iqoption_research.research.features.pipeline import FeatureRow

logger = logging.getLogger(__name__)

DEFAULT_BOT_SNAPSHOT_PATH: Final[Path] = Path("data/bot_sessions/latest_snapshot.json")


class BotState(StrEnum):
    """Lifecycle states of the AI trading bot."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BotMode(StrEnum):
    """Operational mode of the trading bot."""

    OFFLINE_REPLAY = "offline_replay"
    OBSERVE_ONLY = "observe_only"


class InvalidBotStateTransitionError(RuntimeError):
    """Raised when an illegal lifecycle transition is attempted."""


@dataclass(frozen=True, slots=True)
class BotConfig:
    """Consolidated configuration for the autonomous trading bot."""

    symbol: str = "EURUSD"
    mode: BotMode = BotMode.OFFLINE_REPLAY
    candle_size_seconds: int = 60
    initial_balance: float = 10000.0
    payout_rate: float = 0.85
    fixed_stake: float = 100.0
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    observation_token: ObservationAuthorizationToken | None = None


class BotController:
    """Autonomous trading bot controller coordinating observation, inference, and execution."""

    def __init__(
        self,
        config: BotConfig | None = None,
        *,
        model_bundle: ModelBundle | None = None,
        inference_pipeline: ModelInferencePipeline | None = None,
        paper_engine: PaperTradingEngine | None = None,
        observation_adapter: MarketObservationAdapter | None = None,
    ) -> None:
        self.config = config or BotConfig()
        self._state = BotState.IDLE
        self._started_at: float | None = None

        # Setup monitoring & session
        self.session = TradingSession(initial_balance=self.config.initial_balance)
        self.monitor = BotStatusMonitor()

        # Setup observation adapter
        obs_mode = (
            ObservationMode.OBSERVE_ONLY
            if self.config.mode == BotMode.OBSERVE_ONLY
            else ObservationMode.OFFLINE_REPLAY
        )
        self.observation_adapter = observation_adapter or MarketObservationAdapter(
            mode=obs_mode,
            symbol=self.config.symbol,
            candle_size_seconds=self.config.candle_size_seconds,
            authorization_token=self.config.observation_token,
        )

        # Setup inference pipeline
        self.inference_pipeline = inference_pipeline
        if self.inference_pipeline is None and model_bundle is not None:
            self.inference_pipeline = ModelInferencePipeline(model_bundle)

        # Signal generator bridge connecting inference pipeline to PaperTradingEngine
        def _signal_bridge(
            feature_row: FeatureRow,
        ) -> tuple[SignalDirection, float]:
            if self.inference_pipeline is not None:
                res = self.inference_pipeline.infer(feature_row)
                self.session.record_prediction(res)
                return res.direction, res.calibrated_probability
            return SignalDirection.HOLD, 0.5

        # Setup paper trading engine
        engine_cfg = PaperTradingConfig(
            symbol=self.config.symbol,
            payout_rate=self.config.payout_rate,
            candle_size_seconds=self.config.candle_size_seconds,
            initial_balance=self.config.initial_balance,
            is_replay=(self.config.mode == BotMode.OFFLINE_REPLAY),
            risk_config=self.config.risk_config,
        )
        self.risk_manager = OperationalRiskManager(self.config.risk_config)
        self.engine = paper_engine or PaperTradingEngine(
            config=engine_cfg,
            model_bundle=model_bundle,
            risk_manager=self.risk_manager,
            signal_generator=_signal_bridge,
        )

    @property
    def state(self) -> BotState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == BotState.RUNNING

    @property
    def is_paused(self) -> bool:
        return self._state == BotState.PAUSED

    @property
    def uptime_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, time.time() - self._started_at)

    @property
    def portfolio_balance(self) -> float:
        return self.engine.portfolio.current_balance

    def start(self) -> None:
        """Start the bot and transition from IDLE/STOPPED -> RUNNING."""
        if self._state not in {BotState.IDLE, BotState.STOPPED}:
            raise InvalidBotStateTransitionError(
                f"Cannot start bot from current state '{self._state.value}'."
            )

        self._state = BotState.STARTING

        # Gated check for OBSERVE_ONLY mode
        if self.config.mode == BotMode.OBSERVE_ONLY:
            try:
                self.observation_adapter.assert_mode_permission()
            except ObservationSecurityError as exc:
                self._state = BotState.ERROR
                self.monitor.record_error(str(exc))
                raise

        self._state = BotState.RUNNING
        self._started_at = time.time()
        self.monitor.clear_error()

    def pause(self) -> None:
        """Pause trading and feature ingestion."""
        if self._state != BotState.RUNNING:
            raise InvalidBotStateTransitionError(
                f"Cannot pause bot from current state '{self._state.value}'."
            )
        self._state = BotState.PAUSED

    def resume(self) -> None:
        """Resume trading from PAUSED state."""
        if self._state != BotState.PAUSED:
            raise InvalidBotStateTransitionError(
                f"Cannot resume bot from current state '{self._state.value}'."
            )
        self._state = BotState.RUNNING

    def stop(self) -> None:
        """Gracefully stop the bot and persist state snapshot."""
        self._state = BotState.STOPPING
        self._state = BotState.STOPPED
        self.persist_snapshot()
        self.session.save_to_disk()

    def emergency_stop(self, reason: str = "Emergency kill switch activated by operator") -> None:
        """Immediately trigger emergency kill switch, latch circuit breaker, and halt."""
        self.risk_manager.trigger_kill_switch(reason=reason)
        self._state = BotState.STOPPED
        self.monitor.record_error(f"EMERGENCY_STOP: {reason}")
        self.persist_snapshot()
        self.session.save_to_disk()

    def process_candle(
        self,
        candle: Candle,
        *,
        now_ts: float | None = None,
    ) -> EngineStepResult | None:
        """Ingest candle through observation validation, ML inference, and execution."""
        if self._state != BotState.RUNNING:
            return None

        current_now = now_ts if now_ts is not None else float(candle.to_ts)
        self.session.record_candle()

        # 1. Observation layer validation
        obs_event = self.observation_adapter.process_incoming_candle(
            candle, current_now_ts=current_now
        )
        if not obs_event.is_valid:
            return None

        # 2. Feed completed candle into paper-trading engine
        step_result = self.engine.process_candle(candle, now_ts=current_now)

        # 3. Track orders in session
        if step_result.new_order is not None:
            self.session.record_new_order(step_result.new_order)
        for settled in step_result.settled_orders:
            self.session.record_settled_order(settled)

        # Periodically persist snapshot to maintain live telemetry
        self.persist_snapshot()
        return step_result

    def get_status_snapshot(self) -> BotStatusSnapshot:
        """Compile a comprehensive operational snapshot of the bot."""
        latest_pred = self.session.latest_prediction
        model_name = (
            self.inference_pipeline.model_name
            if self.inference_pipeline is not None
            else "BuiltinDecisionEngine"
        )
        conf = latest_pred.confidence if latest_pred is not None else 0.0

        risk_limits: dict[str, float] = {
            "max_stake_per_trade": self.config.risk_config.max_stake_per_trade,
            "max_daily_loss": self.config.risk_config.max_daily_loss,
            "max_daily_drawdown_pct": self.config.risk_config.max_daily_drawdown_pct * 100.0,
            "max_consecutive_losses": float(self.config.risk_config.max_consecutive_losses),
        }

        return self.monitor.build_snapshot(
            state=self._state.value,
            mode=self.config.mode.value,
            symbol=self.config.symbol,
            connection_status="connected" if self._state == BotState.RUNNING else "idle",
            model_name=model_name,
            last_prediction_direction=(
                latest_pred.direction.value if latest_pred is not None else "N/A"
            ),
            last_prediction_prob=(
                latest_pred.calibrated_probability if latest_pred is not None else 0.5
            ),
            confidence=conf,
            abstention_reason=latest_pred.abstention_reason if latest_pred is not None else None,
            virtual_balance=self.engine.portfolio.current_balance,
            open_positions_count=len(self.engine.open_orders),
            trade_count=self.engine.portfolio.total_trades,
            win_rate_pct=self.engine.portfolio.win_rate * 100.0,
            session_pnl=self.engine.portfolio.total_pnl,
            max_drawdown_pct=self.engine.portfolio.max_drawdown_pct * 100.0,
            circuit_breaker_state=self.risk_manager.circuit_state.value,
            feed_health=self.observation_adapter.heartbeat_monitor.evaluate_health(
                is_replay=(self.config.mode == BotMode.OFFLINE_REPLAY)
            ).status.value,
            feed_gaps_count=self.observation_adapter.gaps_count,
            rejected_candles_count=self.observation_adapter.rejected_count,
            data_source=(
                "offline_replay"
                if self.config.mode == BotMode.OFFLINE_REPLAY
                else "live_observation"
            ),
            provenance="SYNTHETIC_ENGINEERING_ONLY",
            uncertainty=max(0.0, min(1.0, 1.0 - (conf * 2.0))),
            settled_trades_count=len(self.session.settled_orders),
            risk_limits=risk_limits,
        )

    def persist_snapshot(self, path: Path | str | None = None) -> Path:
        """Write current operational snapshot to disk."""
        target = Path(path) if path is not None else DEFAULT_BOT_SNAPSHOT_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.get_status_snapshot()
        target.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_latest_snapshot(cls, path: Path | str | None = None) -> BotStatusSnapshot | None:
        """Load the most recent persisted snapshot from disk."""
        target = Path(path) if path is not None else DEFAULT_BOT_SNAPSHOT_PATH
        if not target.is_file():
            return None
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            return BotStatusSnapshot.from_dict(data)
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to load snapshot from %s: %s", target, exc)
            return None
