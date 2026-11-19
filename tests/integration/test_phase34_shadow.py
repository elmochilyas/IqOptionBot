"""Shadow trials over actual connector frames, pinned model and SQLite journal."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import TradingNotAllowedError
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.inference.pipeline import InferenceResult
from iqoption_research.market_data.candles import CandleError
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.validation import DataQualityReport
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.backtest.models import SignalDirection
from iqoption_research.timing.reliability import evaluate
from iqoption_research.timing.shadow import (
    MODEL_PATH,
    ShadowTrial,
    diagnose_candle_window,
    eligible_model,
    is_temporary_readiness_condition,
)
from iqoption_research.timing.supervised_session import SupervisedSessionConfig
from tests.integration.test_market_data_authorization import MarketSocket

F = 1795077300


class CandleSocket(MarketSocket):
    def __init__(self, scenario: str) -> None:
        super().__init__("missing" if scenario == "missing_wallet" else "real")
        self.scenario = scenario

    async def send(self, wire: str) -> None:
        frame = json.loads(wire)
        if frame["name"] != "sendMessage" or frame["msg"]["name"] != "get-candles":
            await super().send(wire)
            return
        self.sent.append(frame)
        if self.scenario == "disconnect":
            from iqoption_research.connectors.iqoption.exceptions import ConnectionError

            raise ConnectionError("fixture disconnected")
        bars = [
            {
                "from": F - 60 * (22 - i),
                "to": F - 60 * (21 - i),
                "open": 1.08,
                "close": 1.0801 + i * 0.00001,
                "min": 1.07,
                "max": 1.09,
                "volume": 1 if self.scenario == "features" else i + 1,
            }
            for i in range(22)
        ]
        if self.scenario == "invalid":
            bars[-1]["min"] = 100
        if self.scenario == "partial":
            bars[-1]["to"] += 60
        msg = {"name": "candles", "msg": {"active_id": 1, "candles": bars}}
        await self.incoming.put(json.dumps({**msg, "request_id": "unrelated"}))
        await self.incoming.put(json.dumps({**msg, "request_id": frame["request_id"]}))


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    # Keep the bundle age and wall/monotonic relationship deterministic in future CI runs.
    base = time.perf_counter_ns()
    shift = [0]

    def wall() -> int:
        return (F + 2) * 10**9 + time.perf_counter_ns() - base + shift[0]

    monkeypatch.setattr(time, "time_ns", wall)
    monkeypatch.setattr(time, "time", lambda: wall() / 1e9)

    class FixtureDate(datetime):
        @classmethod
        def now(cls, tz=UTC):
            return datetime.fromtimestamp(wall() / 1e9, tz)

    monkeypatch.setattr("iqoption_research.inference.validation.datetime", FixtureDate)
    monkeypatch.setattr("iqoption_research.timing.shadow.datetime", FixtureDate)
    return shift


@pytest.mark.parametrize(
    "scenario",
    [
        "normal",
        "missing_wallet",
        "unavailable",
        "features",
        "invalid",
        "partial",
        "disconnect",
        "journal_failure",
        "clock_step",
        "late",
        "commit_overrun",
        "restart",
    ],
)
async def test_shadow(scenario: str, tmp_path: Path, clock: list[int]) -> None:
    for attempt in range(2 if scenario == "restart" else 1):
        trial = ShadowTrial(
            SupervisedSessionConfig(max_candle_cycles=1),
            tmp_path,
            tmp_path / "missing" if scenario == "unavailable" else MODEL_PATH,
        )
        socket = CandleSocket(scenario)
        client = AsyncIQOption(
            ssid="synthetic",
            config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
            offline=False,
            timing_recorder=trial.recorder,
        )
        append = trial.journal.append

        def commit(key, payload, append=append, trial=trial):
            if scenario == "journal_failure":
                raise sqlite3.OperationalError("no space")
            event = append(key, payload)
            if scenario == "commit_overrun":
                # Shift both fixture clocks at commit.
                trial.recorder.clock.anchor_utc_ns += 16 * 10**9
                clock[0] += 16 * 10**9
            return event

        with (
            patch.object(
                IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)
            ),
            patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
            patch.object(trial.journal, "append", side_effect=commit),
        ):
            async with client:
                if scenario == "clock_step":
                    clock[0] += 10**9
                if scenario == "late":
                    clock[0] += 16 * 10**9
                    trial.recorder.clock.anchor_utc_ns += 16 * 10**9
                    await trial.cycle(client, F, time.perf_counter() + 1)
                else:
                    await trial.observe(client)
        if scenario == "restart" and not attempt:
            # Process loss after the durable write, before the registry export.
            assert not trial.output.exists() and len(trial.journal.events()) == 2
            continue
        reg = trial.save()
        assert reg.verify_integrity() == (True, [])
        assert (
            CliRunner()
            .invoke(app, ["timing", "verify-evidence", "--path", str(trial.output)])
            .exit_code
            == 0
        )
        obs = reg.observations[0]
        assert [f["msg"]["name"] for f in socket.sent[1:]] == [
            "get-profile",
            "get-balances",
            "get-candles",
        ]
        assert reg.summary["orders_sent"] == 0
        if scenario in ("invalid", "partial", "disconnect", "journal_failure") or attempt:
            assert obs["rejection_reason"]
            assert "STAGE_9_JOURNAL_COMMIT" not in obs["stages"]
            assert not obs["phase30_gate_satisfied"]
            assert len(trial.journal.events()) == (2 if attempt else 1)
        else:
            event = trial.journal.events()[-1]
            assert event["session"] == reg.session_id
            assert obs["stages"]["STAGE_9_JOURNAL_COMMIT"]["details"]["event_key"] == event["key"]
            if scenario in ("unavailable", "features"):
                assert event["skip_reason"] == (
                    "MODEL_UNAVAILABLE" if scenario == "unavailable" else "MISSING_FEATURES"
                )
                assert event["decision"] == "SKIP" and event["raw_probability"] is None
                assert not obs["prediction_stages_measured"]
            else:
                assert 0 <= event["calibrated_probability"] <= 1
                assert obs["phase30_gate_satisfied"] == (
                    scenario not in ("clock_step", "late", "commit_overrun")
                )
                assert event["decision"] == "SKIP"
                if scenario in ("normal", "missing_wallet"):
                    assert event["skip_reason"] == "MISSING_TRADABLE_PAYOUT"
                    assert "analytical_direction" in event
                    assert obs["stages"]["STAGE_POLICY_DECISION"]["details"]["decision"] == "SKIP"
                    assert (
                        obs["stages"]["STAGE_POLICY_DECISION"]["details"]["skip_reason"]
                        == "MISSING_TRADABLE_PAYOUT"
                    )
        assert evaluate([trial.output])["unique_live_boundaries"] == 0


def test_untrusted_and_stale_bundle(tmp_path: Path, clock: list[int]) -> None:
    copy = tmp_path / "bundle"
    shutil.copytree(MODEL_PATH, copy)
    (copy / "model.joblib").write_bytes(b"invalid")
    assert eligible_model(copy) is None
    clock[0] += 40 * 86400 * 10**9
    assert eligible_model(MODEL_PATH) is None


def test_offline_and_confirmation() -> None:
    runner = CliRunner()
    assert "OFFLINE_PREFLIGHT" in runner.invoke(app, ["timing", "shadow-trial"]).output
    assert runner.invoke(app, ["timing", "shadow-trial", "--live"]).exit_code != 0


async def test_shadow_runner_forbids_trading_and_account_mutations(tmp_path: Path) -> None:
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    socket = CandleSocket("normal")
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
    ):
        async with client:
            client.assert_read_only()
            forbidden_payloads = [
                {"name": "sendMessage", "msg": {"name": "binary-options.open-option", "body": {}}},
                {"name": "sendMessage", "msg": {"name": "digital-options.open-option", "body": {}}},
                {
                    "name": "sendMessage",
                    "msg": {"name": "change-balance", "body": {"balance_id": 123}},
                },
                {"name": "sendMessage", "msg": {"name": "deposit", "body": {}}},
                {"name": "sendMessage", "msg": {"name": "withdraw", "body": {}}},
                {"name": "buyV2", "msg": {}},
            ]
            for payload in forbidden_payloads:
                with pytest.raises(TradingNotAllowedError):
                    await client.ws._send_raw(payload)

    assert trial.metadata["orders_sent"] == 0
    assert trial.metadata["execution_authority"] == "NONE"
    assert trial.metadata["trading_eligibility"] == "UNPROVEN"


async def test_positive_signal_cannot_become_trade_eligible_without_payout(
    tmp_path: Path, clock: list[int]
) -> None:
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    assert trial.model is not None
    assert trial.metadata["assumed_payout_for_gate_only"] == 0.85
    assert trial.metadata["actual_broker_payout_verified"] is False

    # Force model prediction to a strong CALL signal (0.80 > 0.60 threshold)
    strong_call = InferenceResult(
        direction=SignalDirection.CALL,
        calibrated_probability=0.80,
        raw_probability=0.80,
        confidence=0.80,
        abstention_reason=None,
        latency_ms=1.0,
        candle_timestamp=F,
        model_name="random_forest",
        feature_count=10,
        calibrated_prob_up=0.80,
        side_probability=0.80,
    )
    socket = CandleSocket("normal")
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
        patch.object(trial.model, "infer", return_value=strong_call),
    ):
        async with client:
            await trial.observe(client)

    reg = trial.save()
    obs = reg.observations[0]
    policy_stage = obs["stages"]["STAGE_POLICY_DECISION"]["details"]

    # Analytical signal is CALL
    assert policy_stage["analytical_direction"] == "CALL"
    assert policy_stage["analytical_reason"] is None
    assert policy_stage["eligibility_block"] == "missing_payout_data"

    # But eligible shadow action is strictly SKIP because actual tradable payout is missing
    assert policy_stage["decision"] == "SKIP"
    assert policy_stage["skip_reason"] == "MISSING_TRADABLE_PAYOUT"
    assert obs["decision"] == "SKIP"

    # Full 15-second pipeline timing is satisfied (engineering compliance)
    assert obs["prediction_stages_measured"] is True
    assert obs["phase30_gate_satisfied"] is True
    assert obs["gate_verdict"] == "PASS"

    # Journal records the distinction between analytical direction and trade eligibility
    event = trial.journal.events()[-1]
    assert event["decision"] == "SKIP"
    assert event["skip_reason"] == "MISSING_TRADABLE_PAYOUT"
    assert event["analytical_direction"] == "CALL"
    assert event["analytical_reason"] is None
    assert event["calibrated_probability"] == 0.80

    # Strict zero-orders guarantee maintained
    assert reg.summary["orders_sent"] == 0
    assert trial.metadata["orders_sent"] == 0
    assert trial.metadata["execution_authority"] == "NONE"


async def test_low_payout_and_contract_risk_rejections(tmp_path: Path, clock: list[int]) -> None:
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    assert trial.model is not None
    strong_call = InferenceResult(
        direction=SignalDirection.CALL,
        calibrated_probability=0.80,
        raw_probability=0.80,
        confidence=0.80,
        abstention_reason=None,
        latency_ms=1.0,
        candle_timestamp=F,
        model_name="random_forest",
        feature_count=10,
        calibrated_prob_up=0.80,
        side_probability=0.80,
    )
    socket = CandleSocket("normal")
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
        patch.object(trial.model, "infer", return_value=strong_call),
    ):
        async with client:
            await trial.cycle(client, F, time.perf_counter() + 10, tradable_payout=0.50)

    reg = trial.save()
    obs = reg.observations[0]
    policy_stage = obs["stages"]["STAGE_POLICY_DECISION"]["details"]
    assert policy_stage["analytical_direction"] == "CALL"
    assert policy_stage["decision"] == "SKIP"
    assert policy_stage["skip_reason"] == "LOW_PAYOUT_RATE"
    assert policy_stage["eligibility_block"] == "low_payout_rate"


def test_diagnostics_classification_precision() -> None:
    now = datetime.now(UTC)
    good_quality = DataQualityReport(
        symbol="EURUSD",
        size_seconds=60,
        candle_count=22,
        score=100.0,
    )
    bad_quality = DataQualityReport(
        symbol="EURUSD",
        size_seconds=60,
        candle_count=22,
        score=50.0,
    )

    def make_candle(to_ts: int, is_complete: bool = True, symbol: str = "EURUSD") -> Candle:
        return Candle(
            active_id=1,
            symbol=symbol,
            size=CandleSize.MIN_1,
            from_ts=to_ts - 60,
            to_ts=to_ts,
            at_ts=to_ts,
            open=1.08,
            high=1.09,
            low=1.07,
            close=1.085,
            volume=10,
            bid=1.0849,
            ask=1.0851,
            raw={},
            is_complete=is_complete,
            provenance="historical",
            identity_provenance="historical",
        )

    valid_candles = tuple(make_candle(F - 60 * (21 - i)) for i in range(22))

    def make_res(
        candles: tuple[Candle, ...] = valid_candles,
        errors: tuple[CandleError, ...] = (),
        quality: DataQualityReport = good_quality,
        raw_count: int = 22,
        source: str = "iqoption",
    ) -> CandleFetchResult:
        return CandleFetchResult(
            symbol="EURUSD",
            size=CandleSize.MIN_1,
            candles=candles,
            errors=errors,
            quality=quality,
            fetched_at=now,
            raw_count=raw_count,
            source=source,
        )

    # 1. Stale or unverified broker clock reference
    assert (
        diagnose_candle_window(make_res(), F, is_server_time_fresh=False)
        == "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED"
    )

    # 2. No valid candles
    assert diagnose_candle_window(make_res(candles=()), F) == "NO_VALID_CANDLES"

    # 3. Normalization errors or dropped raw candles
    err = CandleError(index=0, reason="invalid", detail="bad OHLC")
    assert diagnose_candle_window(make_res(errors=(err,)), F) == "NORMALIZATION_ERROR"
    assert diagnose_candle_window(make_res(raw_count=23), F) == "NORMALIZATION_ERROR"

    # 4. Requested boundary mismatch
    assert diagnose_candle_window(make_res(), F + 60) == "REQUESTED_BOUNDARY_MISMATCH"
    delayed_candles = tuple(make_candle((F - 60) - 60 * (21 - i)) for i in range(22))
    assert (
        diagnose_candle_window(make_res(candles=delayed_candles), F)
        == "REQUESTED_BOUNDARY_MISMATCH"
    )

    # 5. Missing, gapped, or duplicate candles
    short_candles = valid_candles[2:]
    assert (
        diagnose_candle_window(make_res(candles=short_candles, raw_count=20), F)
        == "MISSING_GAPPED_OR_DUPLICATE_CANDLES"
    )
    gapped_candles = list(valid_candles)
    gapped_candles[10] = make_candle(valid_candles[10].to_ts + 60)
    assert (
        diagnose_candle_window(make_res(candles=tuple(gapped_candles)), F)
        == "MISSING_GAPPED_OR_DUPLICATE_CANDLES"
    )

    # 6. Data quality rejection
    assert diagnose_candle_window(make_res(quality=bad_quality), F) == "DATA_QUALITY_REJECTION"
    non_eurusd = tuple(make_candle(F - 60 * (21 - i), symbol="GBPUSD") for i in range(22))
    assert diagnose_candle_window(make_res(candles=non_eurusd), F) == "DATA_QUALITY_REJECTION"

    # 7. Last candle not yet complete
    incomplete_candles = list(valid_candles)
    incomplete_candles[-1] = make_candle(F, is_complete=False)
    assert (
        diagnose_candle_window(make_res(candles=tuple(incomplete_candles)), F)
        == "LAST_CANDLE_NOT_YET_COMPLETE"
    )

    # 8. All valid
    assert diagnose_candle_window(make_res(), F) is None

    # Eligibility for temporary retry
    assert is_temporary_readiness_condition("LAST_CANDLE_NOT_YET_COMPLETE", None, F) is True
    assert (
        is_temporary_readiness_condition(
            "REQUESTED_BOUNDARY_MISMATCH", make_res(candles=delayed_candles), F
        )
        is True
    )
    assert is_temporary_readiness_condition("REQUESTED_BOUNDARY_MISMATCH", make_res(), F) is False
    assert is_temporary_readiness_condition("DATA_QUALITY_REJECTION", None, F) is False
    assert is_temporary_readiness_condition("NORMALIZATION_ERROR", None, F) is False
    assert (
        is_temporary_readiness_condition("BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED", None, F)
        is False
    )


class MultiAttemptCandleSocket(MarketSocket):
    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        super().__init__("real")
        self.responses = list(responses)
        self.call_count = 0
        self.sent: list[dict[str, Any]] = []

    async def send(self, wire: str) -> None:
        frame = json.loads(wire)
        if frame["name"] != "sendMessage" or frame["msg"]["name"] != "get-candles":
            await super().send(wire)
            return
        self.sent.append(frame)
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        bars = self.responses[idx]
        msg = {"name": "candles", "msg": {"active_id": 1, "candles": bars}}
        await self.incoming.put(json.dumps({**msg, "request_id": frame["request_id"]}))


async def test_startup_boundary_and_incomplete_candle_retry_success(
    tmp_path: Path, clock: list[int]
) -> None:
    """Trial starting at F+2.886s with an incomplete initial candle retries to PASS."""
    # Startup at F + 2.886s
    clock[0] = int(0.886 * 10**9)
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    bars = [
        {
            "from": F - 60 * (22 - i),
            "to": F - 60 * (21 - i),
            "open": 1.08,
            "close": 1.0801 + i * 0.00001,
            "min": 1.07,
            "max": 1.09,
            "volume": i + 1,
        }
        for i in range(22)
    ]
    socket = MultiAttemptCandleSocket([bars, bars])
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )

    # Attempt 1: server_time_ns is F - 1 (candle is incomplete)
    # Attempt 2: server_time_ns is F + 3 (candle is complete)
    server_time_sequence = [(F - 1) * 10**9, (F + 3) * 10**9, (F + 3) * 10**9]
    st_idx = [0]

    def mock_server_time_ns(self_ws: Any) -> int:
        val = server_time_sequence[min(st_idx[0], len(server_time_sequence) - 1)]
        st_idx[0] += 1
        return val

    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
        patch.object(
            IQOptionWebSocketClient,
            "server_time_ns",
            new_callable=lambda: property(mock_server_time_ns),
        ),
    ):
        async with client:
            ok = await trial.cycle(
                client,
                F,
                time.perf_counter() + 10,
                retry_delay_seconds=0.01,
            )
            assert ok is True

    reg = trial.save()
    assert reg.verify_integrity() == (True, [])
    obs = reg.observations[0]
    assert obs["phase30_gate_satisfied"] is True
    assert obs["gate_verdict"] == "PASS"
    stg4 = obs["stages"]["STAGE_4_CANDLE_VALIDATE"]["details"]
    assert stg4.get("attempts") == 2
    assert obs["request_id"] == "candles-2"
    assert socket.call_count == 2


async def test_delayed_boundary_retry_success(tmp_path: Path, clock: list[int]) -> None:
    """Initial broker response lagging by 1 minute (to_ts = F - 60) retries and completes."""
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    lagged_bars = [
        {
            "from": (F - 60) - 60 * (22 - i),
            "to": (F - 60) - 60 * (21 - i),
            "open": 1.08,
            "close": 1.0801 + i * 0.00001,
            "min": 1.07,
            "max": 1.09,
            "volume": i + 1,
        }
        for i in range(22)
    ]
    current_bars = [
        {
            "from": F - 60 * (22 - i),
            "to": F - 60 * (21 - i),
            "open": 1.08,
            "close": 1.0801 + i * 0.00001,
            "min": 1.07,
            "max": 1.09,
            "volume": i + 1,
        }
        for i in range(22)
    ]
    socket = MultiAttemptCandleSocket([lagged_bars, current_bars])
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
    ):
        async with client:
            ok = await trial.cycle(
                client,
                F,
                time.perf_counter() + 10,
                retry_delay_seconds=0.01,
            )
            assert ok is True

    reg = trial.save()
    assert reg.verify_integrity() == (True, [])
    obs = reg.observations[0]
    assert obs["phase30_gate_satisfied"] is True
    assert obs["gate_verdict"] == "PASS"
    stg4 = obs["stages"]["STAGE_4_CANDLE_VALIDATE"]["details"]
    assert stg4.get("attempts") == 2
    assert socket.call_count == 2


async def test_stale_or_unverified_broker_clock_reference(tmp_path: Path, clock: list[int]) -> None:
    """In live mode, unverified or stale server clock rejects immediately without retry."""
    trial = ShadowTrial(
        SupervisedSessionConfig(
            max_candle_cycles=1, live_mode=True, simulation_mode=False, operator_confirmed=True
        ),
        tmp_path,
        MODEL_PATH,
    )
    socket = CandleSocket("normal")
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
        # is_server_time_fresh is False because no timeSync has been received
        patch.object(
            IQOptionWebSocketClient,
            "is_server_time_fresh",
            new_callable=lambda: property(lambda _self: False),
        ),
    ):
        async with client:
            await trial.cycle(client, F, time.perf_counter() + 10, retry_delay_seconds=0.01)

    reg = trial.save()
    obs = reg.observations[0]
    assert obs["rejection_reason"] == "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED"
    assert not obs["phase30_gate_satisfied"]
    assert obs["gate_verdict"] == "FAIL_REJECTED"
    # Zero retries attempted
    assert len(socket.sent) <= 4


async def test_unretryable_quality_failures(tmp_path: Path, clock: list[int]) -> None:
    """Structural/quality defects (e.g. invalid OHLC, gaps) fail immediately with zero retries."""
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    socket = CandleSocket("invalid")
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
    ):
        async with client:
            await trial.cycle(client, F, time.perf_counter() + 10, retry_delay_seconds=0.01)

    reg = trial.save()
    obs = reg.observations[0]
    assert obs["rejection_reason"] == "NORMALIZATION_ERROR"
    assert not obs["phase30_gate_satisfied"]
    assert obs["gate_verdict"] == "FAIL_REJECTED"


async def test_retry_deadline_expiration_never_compliant(tmp_path: Path, clock: list[int]) -> None:
    """When decision completes after F + 15s deadline, it is NEVER classified as compliant."""
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    socket = CandleSocket("normal")
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
    ):
        async with client:
            # Advance clock past 15-second deadline before cycle completes
            clock[0] += 16 * 10**9
            trial.recorder.clock.anchor_utc_ns += 16 * 10**9
            await trial.cycle(client, F, time.perf_counter() + 10)

    reg = trial.save()
    obs = reg.observations[0]
    assert obs["phase30_gate_satisfied"] is False
    assert obs["gate_verdict"] == "FAIL_GATE_OVERRUN"
    assert obs["rejection_reason"] == "OUTSIDE_15_SECOND_ACQUISITION_GATE"
    assert obs["prediction_stages_measured"] is True


async def test_successful_retry_preserves_attempt_history(tmp_path: Path, clock: list[int]) -> None:
    """Successful retry preserves attempt 1 (rejected) and attempt 2 (passed) in audit trail."""
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    lagged_bars = [
        {
            "from": (F - 60) - 60 * (22 - i),
            "to": (F - 60) - 60 * (21 - i),
            "open": 1.08,
            "close": 1.0801 + i * 0.00001,
            "min": 1.07,
            "max": 1.09,
            "volume": i + 1,
        }
        for i in range(22)
    ]
    current_bars = [
        {
            "from": F - 60 * (22 - i),
            "to": F - 60 * (21 - i),
            "open": 1.08,
            "close": 1.0801 + i * 0.00001,
            "min": 1.07,
            "max": 1.09,
            "volume": i + 1,
        }
        for i in range(22)
    ]
    socket = MultiAttemptCandleSocket([lagged_bars, current_bars])
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
    ):
        async with client:
            ok = await trial.cycle(client, F, time.perf_counter() + 10, retry_delay_seconds=0.01)
            assert ok is True

    reg = trial.save()
    is_valid, errors = reg.verify_integrity()
    assert is_valid is True, f"Integrity verification failed: {errors}"
    obs = reg.observations[0]
    assert obs["phase30_gate_satisfied"] is True
    assert obs["gate_verdict"] == "PASS"

    attempts = obs.get("acquisition_attempts", [])
    assert len(attempts) == 2

    # Attempt 1 forensic assertions
    att1 = attempts[0]
    assert att1["attempt_number"] == 1
    assert att1["request_id"] == "candles-1"
    assert att1["retry_attempted"] is True
    assert att1["rejection_classification"] == "REQUESTED_BOUNDARY_MISMATCH"
    assert att1["is_complete"] is False
    assert att1["candle_count"] == 22
    assert att1["request_mono_ns"] is not None
    assert att1["receipt_mono_ns"] is not None
    assert att1["validation_mono_ns"] is not None
    assert att1["request_mono_ns"] <= att1["receipt_mono_ns"] <= att1["validation_mono_ns"]

    # Attempt 2 forensic assertions
    att2 = attempts[1]
    assert att2["attempt_number"] == 2
    assert att2["request_id"] == "candles-2"
    assert att2["retry_attempted"] is False
    assert att2["rejection_classification"] is None
    assert att2["is_complete"] is True
    assert att2["candle_count"] == 22
    assert att2["request_mono_ns"] >= att1["validation_mono_ns"]

    # Denominator check: single final observation for candle boundary F
    assert len(reg.observations) == 1
    assert obs["request_id"] == "candles-2"


async def test_unsuccessful_retry_exhaustion_preserves_attempt_history(
    tmp_path: Path, clock: list[int]
) -> None:
    """Exhausting retry attempts preserves all attempt records and marks final rejection."""
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    lagged_bars = [
        {
            "from": (F - 60) - 60 * (22 - i),
            "to": (F - 60) - 60 * (21 - i),
            "open": 1.08,
            "close": 1.0801 + i * 0.00001,
            "min": 1.07,
            "max": 1.09,
            "volume": i + 1,
        }
        for i in range(22)
    ]
    # Always return lagged bars across 3 attempts
    socket = MultiAttemptCandleSocket([lagged_bars, lagged_bars, lagged_bars])
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
    ):
        async with client:
            ok = await trial.cycle(
                client,
                F,
                time.perf_counter() + 10,
                max_readiness_retries=2,
                retry_delay_seconds=0.01,
            )
            assert ok is True  # No clock step

    reg = trial.save()
    is_valid, errors = reg.verify_integrity()
    assert is_valid is True, f"Integrity verification failed: {errors}"
    obs = reg.observations[0]
    assert obs["phase30_gate_satisfied"] is False
    assert obs["gate_verdict"] == "FAIL_REJECTED"
    assert obs["rejection_reason"] == "REQUESTED_BOUNDARY_MISMATCH"

    attempts = obs.get("acquisition_attempts", [])
    assert len(attempts) == 3
    for i in range(2):
        assert attempts[i]["attempt_number"] == i + 1
        assert attempts[i]["retry_attempted"] is True
        assert attempts[i]["rejection_classification"] == "REQUESTED_BOUNDARY_MISMATCH"

    assert attempts[2]["attempt_number"] == 3
    assert attempts[2]["retry_attempted"] is False
    assert attempts[2]["rejection_classification"] == "REQUESTED_BOUNDARY_MISMATCH"
    assert obs["request_id"] == attempts[2]["request_id"]


async def test_candle_deadline_timeout_prevents_retry(tmp_path: Path, clock: list[int]) -> None:
    """When time before F + 15s is insufficient, retry is prohibited and timeout recorded."""
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
        pipeline_reserve_seconds=1.0,
    )
    socket = CandleSocket("normal")
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(
            timing_market_data_only=True, max_reconnect_attempts=0, request_timeout=0.05
        ),
        offline=False,
        timing_recorder=trial.recorder,
    )

    async def hanging_get_candles(*args: Any, **kwargs: Any) -> CandleFetchResult:
        await asyncio.sleep(1.0)
        raise TimeoutError("Simulated socket timeout")

    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
        patch.object(client, "get_candles", side_effect=hanging_get_candles),
    ):
        async with client:
            ok = await trial.cycle(client, F, time.perf_counter() + 0.05, retry_delay_seconds=0.01)
            assert ok is True

    reg = trial.save()
    is_valid, errors = reg.verify_integrity()
    assert is_valid is True, f"Integrity verification failed: {errors}"
    obs = reg.observations[0]
    assert obs["phase30_gate_satisfied"] is False
    assert obs["rejection_reason"] in ("SESSION_DEADLINE", "OUTSIDE_15_SECOND_ACQUISITION_GATE")
    attempts = obs.get("acquisition_attempts", [])
    assert len(attempts) == 1
    assert attempts[0]["rejection_classification"] == "TIMEOUT_EXCEEDED_DEADLINE"
    assert attempts[0]["retry_attempted"] is False


async def test_operator_cancellation_preserves_interrupted_attempt(
    tmp_path: Path, clock: list[int]
) -> None:
    """Operator cancellation during in-flight retry preserves interrupted attempt in audit trail."""
    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    lagged_bars = [
        {
            "from": (F - 60) - 60 * (22 - i),
            "to": (F - 60) - 60 * (21 - i),
            "open": 1.08,
            "close": 1.0801 + i * 0.00001,
            "min": 1.07,
            "max": 1.09,
            "volume": i + 1,
        }
        for i in range(22)
    ]
    socket = MultiAttemptCandleSocket([lagged_bars, lagged_bars])
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )

    call_count = [0]
    orig_get_candles = client.get_candles

    async def cancel_on_retry(*args: Any, **kwargs: Any) -> CandleFetchResult:
        call_count[0] += 1
        if call_count[0] == 2:
            raise asyncio.CancelledError()
        return await orig_get_candles(*args, **kwargs)

    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
        patch.object(client, "get_candles", side_effect=cancel_on_retry),
    ):
        async with client:
            with pytest.raises(asyncio.CancelledError):
                await trial.cycle(client, F, time.perf_counter() + 10, retry_delay_seconds=0.01)

    reg = trial.save()
    is_valid, errors = reg.verify_integrity()
    assert is_valid is True, f"Integrity verification failed: {errors}"
    obs = reg.observations[0]
    assert obs["rejection_reason"] == "OPERATOR_CANCELLED"

    attempts = obs.get("acquisition_attempts", [])
    assert len(attempts) == 2
    assert attempts[0]["attempt_number"] == 1
    assert attempts[0]["retry_attempted"] is True
    assert attempts[0]["rejection_classification"] == "REQUESTED_BOUNDARY_MISMATCH"

    assert attempts[1]["attempt_number"] == 2
    assert attempts[1]["retry_attempted"] is False
    assert attempts[1]["rejection_classification"] == "OPERATOR_CANCELLED"


async def test_attempt_history_integrity_verification_and_tampering(
    tmp_path: Path, clock: list[int]
) -> None:
    """Integrity verification catches attempt ordering, duplicate ID, and timestamp tampering."""
    from iqoption_research.timing.evidence import TimingEvidenceRegistry

    trial = ShadowTrial(
        SupervisedSessionConfig(max_candle_cycles=1),
        tmp_path,
        MODEL_PATH,
    )
    lagged_bars = [
        {
            "from": (F - 60) - 60 * (22 - i),
            "to": (F - 60) - 60 * (21 - i),
            "open": 1.08,
            "close": 1.0801 + i * 0.00001,
            "min": 1.07,
            "max": 1.09,
            "volume": i + 1,
        }
        for i in range(22)
    ]
    current_bars = [
        {
            "from": F - 60 * (22 - i),
            "to": F - 60 * (21 - i),
            "open": 1.08,
            "close": 1.0801 + i * 0.00001,
            "min": 1.07,
            "max": 1.09,
            "volume": i + 1,
        }
        for i in range(22)
    ]
    socket = MultiAttemptCandleSocket([lagged_bars, current_bars])
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
    ):
        async with client:
            ok = await trial.cycle(client, F, time.perf_counter() + 10, retry_delay_seconds=0.01)
            assert ok is True

    reg = trial.save()
    assert reg.verify_integrity() == (True, [])

    # Tamper 1: non-sequential attempt number
    d1 = json.loads(json.dumps(reg.to_dict()))
    d1["observations"][0]["acquisition_attempts"][1]["attempt_number"] = 5
    reg_t1 = TimingEvidenceRegistry.from_dict(d1)
    is_valid, errors = reg_t1.verify_integrity()
    assert is_valid is False
    assert any("attempt 1 number mismatch" in e for e in errors)

    # Tamper 2: duplicate request ID across attempts
    d2 = json.loads(json.dumps(reg.to_dict()))
    d2["observations"][0]["acquisition_attempts"][1]["request_id"] = d2["observations"][0][
        "acquisition_attempts"
    ][0]["request_id"]
    reg_t2 = TimingEvidenceRegistry.from_dict(d2)
    is_valid, errors = reg_t2.verify_integrity()
    assert is_valid is False
    assert any("duplicate request_id" in e for e in errors)

    # Tamper 3: final attempt marked with retry_attempted=True
    d3 = json.loads(json.dumps(reg.to_dict()))
    d3["observations"][0]["acquisition_attempts"][1]["retry_attempted"] = True
    reg_t3 = TimingEvidenceRegistry.from_dict(d3)
    is_valid, errors = reg_t3.verify_integrity()
    assert is_valid is False
    assert any("final attempt 2 cannot have retry_attempted=True" in e for e in errors)

    # Tamper 4: inverted monotonic timestamps (request > receipt)
    d4 = json.loads(json.dumps(reg.to_dict()))
    att0 = d4["observations"][0]["acquisition_attempts"][0]
    att0["request_mono_ns"] = att0["receipt_mono_ns"] + 1000000
    reg_t4 = TimingEvidenceRegistry.from_dict(d4)
    is_valid, errors = reg_t4.verify_integrity()
    assert is_valid is False
    assert any("timestamp order violation: request" in e for e in errors)

    # Tamper 5: cross-attempt monotonic inversion (attempt 2 started before attempt 1 finished)
    d5 = json.loads(json.dumps(reg.to_dict()))
    d5["observations"][0]["acquisition_attempts"][1]["request_mono_ns"] = (
        d5["observations"][0]["acquisition_attempts"][0]["validation_mono_ns"] - 1000
    )
    reg_t5 = TimingEvidenceRegistry.from_dict(d5)
    is_valid, errors = reg_t5.verify_integrity()
    assert is_valid is False
    assert any("started before previous attempt finished" in e for e in errors)

    # Backward compatibility: verify historical Phase 33 & Phase 34 baseline evidence files
    p33_path = Path("data/phase33/supervised_timing_evidence_meas_705e2ab066f0_1795077245.json")
    if p33_path.exists():
        p33_reg = TimingEvidenceRegistry.load_from_file(p33_path)
        p33_valid, p33_errs = p33_reg.verify_integrity()
        assert p33_valid is True, f"Phase 33 baseline failed integrity: {p33_errs}"
        assert p33_reg.integrity_hash == p33_reg.compute_hash()

    p34_path = (
        Path("data/phase34/operator_supervised_live")
        / "shadow_d0f7098c87644fcfabe1f8b12dfbaf84.json"
    )
    if p34_path.exists():
        p34_reg = TimingEvidenceRegistry.load_from_file(p34_path)
        p34_valid, p34_errs = p34_reg.verify_integrity()
        assert p34_valid is True, f"Phase 34 baseline failed integrity: {p34_errs}"
        assert p34_reg.integrity_hash == p34_reg.compute_hash()
