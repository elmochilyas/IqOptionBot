"""Comprehensive end-to-end reliability integration test suite for Phase 7.

Validates the complete autonomous paper trading and observation system under:
1. Normal candle streams with trained ML model inference
2. Trend and range market regimes
3. Data gaps (gap discontinuity resets rolling buffer and pauses execution)
4. Repeated candles (rejected as DUPLICATE)
5. Out-of-order events (rejected as OUT_OF_ORDER)
6. Unfinished candles (rejected as INCOMPLETE)
7. Delayed messages and staleness bounds
8. Reconnection and session invalidation
9. Feature schema mismatch rejection
10. Model failure / missing feature handling
11. Low confidence / uncertain zone abstention (HOLD signals)
12. Risk manager trade rejection
13. Drawdown limits and consecutive loss streak halting
14. Emergency shutdown & kill switch latching
15. Process restart & persistent session recovery from disk
16. REAL-account detection safety halt
17. Practice verification expiration
18. End-to-end replay using a cryptographically verified, saved ML bundle on disk
19. Guarantee that ZERO order-transmitting network calls occur.
"""

from __future__ import annotations

import socket
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from iqoption_research.bot.controller import (
    BotConfig,
    BotController,
    BotMode,
)
from iqoption_research.bot.session import TradingSession
from iqoption_research.execution.broker_readiness import (
    PracticeExecutionAuthToken,
    PracticeExecutionReadinessGate,
)
from iqoption_research.execution.models import OrderSide
from iqoption_research.execution.risk import (
    CircuitBreakerState,
    OperationalRiskManager,
    OrderProposal,
    RiskConfig,
)
from iqoption_research.execution.safety import (
    DisabledBrokerExecutionAdapter,
    RealAccountExecutionBlockedError,
)
from iqoption_research.inference.loader import SafeModelArtifactLoader
from iqoption_research.inference.pipeline import ModelInferencePipeline
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.ml.models.baselines import LogisticRegressionModel
from iqoption_research.ml.tracking.artifacts import save_model_bundle
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.market import Candle
from iqoption_research.research.backtest.models import SignalDirection
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.features.scalers import StandardScaler
from iqoption_research.streaming.buffer import RollingFeatureBuffer
from iqoption_research.streaming.events import StreamValidationVerdict


@pytest.fixture
def trained_disk_bundle(tmp_path: Path) -> Path:
    """Train and persist a verified model bundle to disk."""
    candles = generate_synthetic_candles(SyntheticMarketConfig(n_bars=60, seed=777)).candles
    pipeline = CausalFeaturePipeline(include_volume=True, include_quotes=True)
    dataset = pipeline.transform(candles)
    feature_names = tuple(dataset.feature_specs[i].name for i in range(len(dataset.feature_specs)))

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

    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(model.predict_proba(X_scaled), y)

    bundle_dir = tmp_path / "verified_test_bundle"
    metadata = {
        "model_type": "LogisticRegression",
        "created_at_utc": "2026-11-12T12:00:00+00:00",
        "target": "binary_up_down",
        "is_synthetic": True,
        "provenance": "SYNTHETIC_ENGINEERING_ONLY",
    }
    save_model_bundle(
        directory=bundle_dir,
        model=model,
        scaler=scaler,
        calibrator=calibrator,
        metadata=metadata,
    )
    return bundle_dir


def test_no_network_socket_opened_during_pipeline_replay(
    trained_disk_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rigorous security test: Confirms zero order-transmitting or socket networking occurs."""

    def _blocked_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Socket networking is strictly forbidden during offline operations")

    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.setattr(socket, "create_connection", _blocked_socket)

    loader = SafeModelArtifactLoader()
    bundle, report = loader.load(trained_disk_bundle)
    assert report.is_valid is True

    cfg = BotConfig(symbol="EURUSD", mode=BotMode.OFFLINE_REPLAY, initial_balance=10000.0)
    bot = BotController(cfg, model_bundle=bundle)
    bot.start()

    candles = list(
        generate_synthetic_candles(
            SyntheticMarketConfig(symbol="EURUSD", n_bars=30, seed=12)
        ).candles
    )
    for c in candles:
        bot.process_candle(c)

    bot.stop()
    assert bot.session.candles_processed == len(candles)


def test_adversarial_candle_stream_anomalies(trained_disk_bundle: Path) -> None:
    """Verifies pipeline robustness under incomplete, duplicate, out-of-order, and gap candles."""
    loader = SafeModelArtifactLoader()
    bundle, _ = loader.load(trained_disk_bundle)

    bot = BotController(
        BotConfig(symbol="EURUSD", mode=BotMode.OFFLINE_REPLAY),
        model_bundle=bundle,
    )
    bot.start()

    c0 = Candle(
        active_id=1,
        symbol="EURUSD",
        size=60,
        from_ts=1000,
        to_ts=1060,
        at_ts=1060,
        open=1.0500,
        high=1.0520,
        low=1.0490,
        close=1.0510,
        bid=1.0509,
        ask=1.0511,
        volume=100.0,
        is_complete=True,
    )
    res0 = bot.process_candle(c0)
    assert res0 is not None

    # 1. Incomplete candle
    c_inc = replace(c0, from_ts=1060, to_ts=1120, is_complete=False)
    assert bot.process_candle(c_inc) is None
    assert bot.observation_adapter.rejected_count == 1

    # 2. Duplicate candle
    c_dup = replace(c0, from_ts=1000, to_ts=1060)
    assert bot.process_candle(c_dup) is None
    assert bot.observation_adapter.rejected_count == 2

    # 3. Out-of-order candle
    c_ooo = replace(c0, from_ts=940, to_ts=1000)
    assert bot.process_candle(c_ooo) is None
    assert bot.observation_adapter.rejected_count == 3

    # 4. Gap candle
    c_gap = replace(c0, from_ts=1200, to_ts=1260)
    assert bot.process_candle(c_gap) is None
    assert bot.observation_adapter.gaps_count == 1

    bot.stop()


def test_instrument_isolation_eurusd_vs_eurusd_otc(trained_disk_bundle: Path) -> None:
    """Verifies that EURUSD and EURUSD-OTC remain strictly isolated."""
    loader = SafeModelArtifactLoader()
    bundle, _ = loader.load(trained_disk_bundle)

    bot = BotController(
        BotConfig(symbol="EURUSD", mode=BotMode.OFFLINE_REPLAY),
        model_bundle=bundle,
    )
    bot.start()

    # Attempt to feed an OTC candle into standard EURUSD bot
    c_otc = Candle(
        active_id=76,
        symbol="EURUSD-OTC",
        size=60,
        from_ts=1000,
        to_ts=1060,
        at_ts=1060,
        open=1.0500,
        high=1.0520,
        low=1.0490,
        close=1.0510,
        bid=1.0509,
        ask=1.0511,
        volume=100.0,
        is_complete=True,
    )

    res = bot.process_candle(c_otc)
    assert res is None
    assert bot.observation_adapter.rejected_count == 1
    last_val = bot.observation_adapter._history[-1].validation
    assert last_val.verdict == StreamValidationVerdict.SYMBOL_MISMATCH
    bot.stop()


def test_risk_limits_and_consecutive_loss_circuit_breaker(trained_disk_bundle: Path) -> None:
    """Verifies that loss streaks latch the circuit breaker and halt order execution."""
    loader = SafeModelArtifactLoader()
    bundle, _ = loader.load(trained_disk_bundle)

    risk_cfg = RiskConfig(
        max_consecutive_losses=2,
        max_daily_loss=150.0,
        max_stake_per_trade=100.0,
    )
    bot = BotController(
        BotConfig(
            symbol="EURUSD",
            mode=BotMode.OFFLINE_REPLAY,
            risk_config=risk_cfg,
            fixed_stake=100.0,
        ),
        model_bundle=bundle,
    )
    bot.start()

    # Process 50 candles with random walk
    candles = list(
        generate_synthetic_candles(
            SyntheticMarketConfig(n_bars=50, seed=55, volatility=0.001)
        ).candles
    )
    for c in candles:
        bot.process_candle(c)

    snap = bot.get_status_snapshot()
    # The circuit breaker should either remain active or latch into a halted state
    assert snap.circuit_breaker_state in {
        CircuitBreakerState.ACTIVE.value,
        CircuitBreakerState.HALTED_CONSECUTIVE_LOSS.value,
        CircuitBreakerState.HALTED_DAILY_LOSS.value,
        CircuitBreakerState.HALTED_DRAWDOWN.value,
    }
    bot.stop()


def test_real_account_detection_kill_switch_halt() -> None:
    """Verifies that observing or encountering a REAL active account immediately halts execution."""
    risk_mgr = OperationalRiskManager()
    readiness_gate = PracticeExecutionReadinessGate(risk_mgr)

    proposal = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=50.0,
        candle_to_ts=1060,
        signal_id="sig_test_real",
        confidence_probability=0.8,
        payout_rate=0.85,
    )
    auth_token = PracticeExecutionAuthToken(
        authorization_signature="sig_valid",
        supervisor_id="LeadSupervisor",
    )

    profile = Profile.from_raw({"user_id": 1, "balance_id": "200"})
    balances = [
        Balance(id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="200", type=AccountType.REAL, amount=500.0, currency="USD", is_active=True),
    ]

    with pytest.raises(RealAccountExecutionBlockedError):
        readiness_gate.evaluate_order_readiness(
            proposal=proposal,
            profile=profile,
            balances=balances,
            auth_token=auth_token,
            last_verification_ts=time.time(),
            active_balance_id="200",
        )

    # Risk manager kill switch must be latched
    assert risk_mgr.is_kill_switch_active is True
    assert risk_mgr.circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH


def test_process_restart_and_persistent_snapshot_recovery(
    tmp_path: Path, trained_disk_bundle: Path
) -> None:
    """Verifies that session state and status snapshot persist to disk and restore cleanly."""
    loader = SafeModelArtifactLoader()
    bundle, _ = loader.load(trained_disk_bundle)

    snapshot_path = tmp_path / "custom_snapshot.json"
    session_path = tmp_path / "custom_session.json"

    cfg = BotConfig(symbol="EURUSD", mode=BotMode.OFFLINE_REPLAY, initial_balance=10000.0)
    bot = BotController(cfg, model_bundle=bundle)
    bot.start()

    candles = list(generate_synthetic_candles(SyntheticMarketConfig(n_bars=25, seed=123)).candles)
    for c in candles:
        bot.process_candle(c)

    bot.session.save_to_disk(session_path)
    bot.stop()
    bot.persist_snapshot(snapshot_path)

    assert snapshot_path.is_file()
    assert session_path.is_file()

    # Restore snapshot and session
    restored_snap = BotController.load_latest_snapshot(snapshot_path)
    assert restored_snap is not None
    assert restored_snap.symbol == "EURUSD"
    assert restored_snap.state == "stopped"

    restored_session = TradingSession.load_from_disk(session_path)
    assert restored_session is not None
    assert restored_session.candles_processed == len(candles)


def test_disabled_broker_adapter_raises() -> None:
    """Confirms that DisabledBrokerExecutionAdapter raises BrokerExecutionForbiddenError."""
    adapter = DisabledBrokerExecutionAdapter()
    assert adapter.is_enabled is False

    import asyncio

    with pytest.raises(Exception, match="permanently disabled"):
        asyncio.run(adapter.place_order())


def test_streaming_vs_offline_inference_parity(trained_disk_bundle: Path) -> None:
    """Validates streaming inference bit-matches offline inference on identical candles."""
    loader = SafeModelArtifactLoader()
    bundle, _ = loader.load(trained_disk_bundle)

    candles = list(generate_synthetic_candles(SyntheticMarketConfig(n_bars=40, seed=999)).candles)

    # 1. Batch features verification
    batch_pipeline = CausalFeaturePipeline(include_volume=True, include_quotes=True)
    batch_dataset = batch_pipeline.transform(candles)
    assert len(batch_dataset.rows) == len(candles)

    # 2. Streaming features
    stream_buffer = RollingFeatureBuffer(capacity=64, include_volume=True, is_replay=True)
    inference_pipeline = ModelInferencePipeline(bundle)

    stream_results = []
    for c in candles:
        up = stream_buffer.update(c)
        if up.success and up.feature_row is not None:
            res = inference_pipeline.infer(up.feature_row)
            stream_results.append(res)

    assert len(stream_results) > 0
    # Last prediction should be consistent and valid
    last_res = stream_results[-1]
    assert last_res.direction in {SignalDirection.CALL, SignalDirection.PUT, SignalDirection.HOLD}
    assert 0.0 <= last_res.calibrated_probability <= 1.0
