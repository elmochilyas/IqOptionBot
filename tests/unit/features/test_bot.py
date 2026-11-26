"""Unit tests for Phase 6 autonomous bot controller, session tracking, and monitoring.

Covers state transitions, emergency stops, observe-only gating, streaming candle processing,
and automated paper-trading execution with trained ML model pipelines.
"""

from __future__ import annotations

import pytest

from iqoption_research.bot.controller import (
    BotConfig,
    BotController,
    BotMode,
    BotState,
    InvalidBotStateTransitionError,
)
from iqoption_research.execution.risk import CircuitBreakerState
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.ml.models.baselines import LogisticRegressionModel
from iqoption_research.ml.tracking.artifacts import ModelBundle
from iqoption_research.observation.adapter import (
    ObservationAuthorizationToken,
    ObservationSecurityError,
)
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.features.scalers import StandardScaler


@pytest.fixture
def model_bundle_fixture() -> ModelBundle:
    """Fixture providing a fitted model bundle."""
    candles = generate_synthetic_candles(SyntheticMarketConfig(n_bars=50, seed=123)).candles
    pipeline = CausalFeaturePipeline(include_volume=True, include_quotes=True)
    dataset = pipeline.transform(candles)
    feature_names = tuple(dataset.feature_specs[i].name for i in range(len(dataset.feature_specs)))

    import numpy as np

    rows = list(dataset.rows[20:])
    X = np.array([[r.features[k] for k in feature_names] for r in rows], dtype=np.float64)
    y = np.array(
        [1 if candles[r.index].close > candles[r.index].open else 0 for r in rows], dtype=np.int64
    )

    scaler = StandardScaler()
    scaler.fit(X, feature_names=feature_names)
    X_scaled = scaler.transform(X)

    model = LogisticRegressionModel()
    model.fit(X_scaled, y, feature_names=feature_names)

    return ModelBundle(
        model=model,
        scaler=scaler,
        calibrator=None,
        metadata={
            "created_at_utc": "2026-11-12T12:00:00+00:00",
            "model_provenance": "demo_synthetic",
        },
    )


def test_bot_controller_lifecycle_state_machine() -> None:
    bot = BotController(BotConfig(symbol="EURUSD"))
    assert bot.state == BotState.IDLE
    assert bot.is_running is False

    # Start
    bot.start()
    assert bot.state == BotState.RUNNING
    assert bot.is_running is True

    # Pause
    bot.pause()
    assert bot.state == BotState.PAUSED
    assert bot.is_paused is True
    assert bot.is_running is False

    # Resume
    bot.resume()
    assert bot.state == BotState.RUNNING

    # Stop
    bot.stop()
    assert bot.state == BotState.STOPPED
    assert bot.is_running is False

    # Cannot pause when stopped
    with pytest.raises(InvalidBotStateTransitionError):
        bot.pause()


def test_bot_controller_emergency_stop() -> None:
    bot = BotController(BotConfig(symbol="EURUSD"))
    bot.start()

    bot.emergency_stop(reason="Operator detected anomalous risk")
    assert bot.state == BotState.STOPPED
    assert bot.risk_manager.circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH
    assert bot.risk_manager.is_kill_switch_active is True
    assert "Operator detected anomalous risk" in bot.risk_manager.halt_reason


def test_bot_controller_observe_only_requires_authorization() -> None:
    # Unauthorized observe-only mode
    bot_unauth = BotController(BotConfig(mode=BotMode.OBSERVE_ONLY))
    with pytest.raises(ObservationSecurityError):
        bot_unauth.start()
    assert bot_unauth.state == BotState.ERROR

    # Authorized observe-only mode
    token = ObservationAuthorizationToken(token_hash="auth_tok_441", authorized_by="Supervisor")
    bot_auth = BotController(BotConfig(mode=BotMode.OBSERVE_ONLY, observation_token=token))
    bot_auth.start()
    assert bot_auth.state == BotState.RUNNING


def test_bot_controller_automated_paper_trading(model_bundle_fixture: ModelBundle) -> None:
    bot = BotController(
        BotConfig(
            symbol="EURUSD",
            initial_balance=10000.0,
            payout_rate=0.85,
            fixed_stake=100.0,
            impute_missing_quotes=True,
        ),
        model_bundle=model_bundle_fixture,
    )
    bot.start()

    candles = list(
        generate_synthetic_candles(
            SyntheticMarketConfig(symbol="EURUSD", n_bars=40, seed=99)
        ).candles
    )

    # Process candles through running bot
    results = []
    for c in candles:
        res = bot.process_candle(c)
        if res is not None:
            results.append(res)

    assert len(results) > 0
    assert bot.session.candles_processed == len(candles)
    assert bot.session.prediction_count > 0

    # Verify snapshot telemetry
    snapshot = bot.get_status_snapshot()
    assert snapshot.state == "running"
    assert snapshot.symbol == "EURUSD"
    assert snapshot.virtual_balance > 0.0
    assert snapshot.last_error is None
    assert snapshot.model_name == "logistic_regression"

    # Stop bot
    bot.stop()
    assert bot.state == BotState.STOPPED
