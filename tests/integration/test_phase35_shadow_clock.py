"""Integration tests for Phase 35: Broker Clock Freshness in Shadow Trials."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.timing.shadow import (
    MODEL_PATH,
    ShadowTrial,
)
from iqoption_research.timing.supervised_session import SupervisedSessionConfig
from tests.integration.test_phase34_shadow import CandleSocket, F


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Align wall clock to F + 2 seconds for valid 15-second arrival gate."""
    base = time.perf_counter_ns()
    shift = [0]

    def wall() -> int:
        return (F + 2) * 10**9 + time.perf_counter_ns() - base + shift[0]

    monkeypatch.setattr(time, "time_ns", wall)
    monkeypatch.setattr(time, "time", lambda: wall() / 1e9)

    class FixtureDate(datetime):
        @classmethod
        def now(cls, tz: Any = UTC) -> datetime:
            return datetime.fromtimestamp(wall() / 1e9, tz)

    monkeypatch.setattr("iqoption_research.inference.validation.datetime", FixtureDate)
    monkeypatch.setattr("iqoption_research.timing.shadow.datetime", FixtureDate)
    return shift


class ClockScenarioSocket(CandleSocket):
    """Socket simulator that controls timeSync delivery."""

    def __init__(self, *, push_timesync: bool = True) -> None:
        super().__init__("normal")
        self.push_timesync = push_timesync

    async def send(self, wire: str) -> None:
        frame = json.loads(wire)
        if frame["name"] == "authenticate":
            await self.incoming.put(json.dumps({"name": "authenticated", "msg": True}))
            if self.push_timesync:
                sync_payload = {"name": "timeSync", "msg": int((F + 2) * 1000)}
                await self.incoming.put(json.dumps(sync_payload))
            return
        if frame["name"] == "sendMessage" and frame["msg"]["name"] == "get-candles":
            self.sent.append(frame)
            endtime = int(frame["msg"].get("end") or frame["msg"].get("endtime") or F)
            bars = [
                {
                    "from": endtime - 60 * (22 - i),
                    "to": endtime - 60 * (21 - i),
                    "open": 1.08,
                    "close": 1.0801 + i * 0.00001,
                    "min": 1.07,
                    "max": 1.09,
                    "volume": i + 1,
                }
                for i in range(22)
            ]
            msg = {"name": "candles", "msg": {"active_id": 1, "candles": bars}}
            await self.incoming.put(json.dumps({**msg, "request_id": frame["request_id"]}))
            return
        await super().send(wire)

    def push_msg(self, data: dict[str, Any]) -> None:
        self.incoming.put_nowait(json.dumps(data))


@pytest.mark.asyncio
async def test_missing_timesync_fails_closed_never_synchronized(
    tmp_path: Path, clock: list[int]
) -> None:
    """If broker never sends timeSync, observation fails closed with NEVER_SYNCHRONIZED."""
    trial = ShadowTrial(
        SupervisedSessionConfig(
            live_mode=True,
            simulation_mode=False,
            operator_confirmed=True,
            max_candle_cycles=1,
            max_duration_seconds=10,
        ),
        tmp_path,
        MODEL_PATH,
    )
    socket = ClockScenarioSocket(push_timesync=False)
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
    assert obs["rejection_reason"] == "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED"
    assert not obs["phase30_gate_satisfied"]
    assert obs["gate_verdict"] == "FAIL_REJECTED"

    # Audit trail must show never synchronized
    attempts = obs["acquisition_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["broker_clock_fresh"] is False
    assert attempts[0]["broker_clock_sync_status"] == "NEVER_SYNCHRONIZED"
    assert attempts[0]["broker_clock_age_seconds"] is None
    assert attempts[0]["rejection_classification"] == "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED"


@pytest.mark.asyncio
async def test_timesync_expiry_during_session_fails_closed(
    tmp_path: Path, clock: list[int]
) -> None:
    """If timeSync expires (> 120s) between observations, subsequent observation fails closed."""
    trial = ShadowTrial(
        SupervisedSessionConfig(
            live_mode=True,
            simulation_mode=False,
            operator_confirmed=True,
            max_candle_cycles=2,
            max_duration_seconds=300,
        ),
        tmp_path,
        MODEL_PATH,
    )
    socket = ClockScenarioSocket(push_timesync=True)
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )

    mono_val = [1000.0]

    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
        patch("time.monotonic", side_effect=lambda: mono_val[0]),
    ):
        async with client:
            # Observation 1 at virtual t=1000.0 (fresh)
            await trial.cycle(client, F, time.perf_counter() + 10, retry_delay_seconds=0.01)

            # Observation 2 at virtual t=1150.0 (150s after initial sync -> stale)
            mono_val[0] = 1150.0
            clock[0] += 60 * 10**9
            await trial.cycle(client, F + 60, time.perf_counter() + 10, retry_delay_seconds=0.01)

    reg = trial.save()
    assert len(reg.observations) == 2

    obs1 = reg.observations[0]
    assert obs1["rejection_reason"] is None
    assert obs1["candle_validated_within_15s"] is True
    assert obs1["prediction_stages_measured"] is True
    assert obs1["acquisition_attempts"][0]["broker_clock_fresh"] is True
    assert obs1["acquisition_attempts"][0]["broker_clock_sync_status"] == "SYNCHRONIZED_AND_FRESH"

    obs2 = reg.observations[1]
    assert obs2["rejection_reason"] == "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED"
    assert obs2["candle_validated_within_15s"] is False
    assert obs2["acquisition_attempts"][0]["broker_clock_fresh"] is False
    assert obs2["acquisition_attempts"][0]["broker_clock_sync_status"] == "STALE_SYNCHRONIZATION"
    assert obs2["acquisition_attempts"][0]["broker_clock_age_seconds"] is not None
    assert obs2["acquisition_attempts"][0]["broker_clock_age_seconds"] > 120.0


@pytest.mark.asyncio
async def test_initial_clock_wait_recovers_before_boundary(
    tmp_path: Path, clock: list[int]
) -> None:
    """observe() startup wait catches delayed initial timeSync before boundary observation."""
    trial = ShadowTrial(
        SupervisedSessionConfig(
            live_mode=True,
            simulation_mode=False,
            operator_confirmed=True,
            max_candle_cycles=1,
            max_duration_seconds=5,
        ),
        tmp_path,
        MODEL_PATH,
    )
    socket = ClockScenarioSocket(push_timesync=False)
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )

    # Simulate broker pushing timeSync 50ms after connection
    async def push_timesync_after_delay() -> None:
        await asyncio.sleep(0.05)
        socket.push_msg({"name": "timeSync", "msg": int(F * 1000)})

    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
    ):
        async with client:
            bg_sync = asyncio.create_task(push_timesync_after_delay())
            # Run trial.observe with boundary matching F
            with patch("time.time", side_effect=[float(F + 1), float(F + 2), float(F + 3)]):
                await trial.observe(client)
            await bg_sync

    reg = trial.save()
    assert len(reg.observations) == 1
    obs = reg.observations[0]
    assert obs["rejection_reason"] is None
    assert obs["candle_validated_within_15s"] is True
    assert obs["acquisition_attempts"][0]["broker_clock_fresh"] is True
    assert obs["acquisition_attempts"][0]["broker_clock_sync_status"] == "SYNCHRONIZED_AND_FRESH"


@pytest.mark.asyncio
async def test_deadline_enforcement_terminates_fail_closed(
    tmp_path: Path, clock: list[int]
) -> None:
    """When deadline expires during acquisition attempt, terminates immediately fail closed."""
    trial = ShadowTrial(
        SupervisedSessionConfig(
            live_mode=True,
            simulation_mode=False,
            operator_confirmed=True,
            max_candle_cycles=1,
            max_duration_seconds=1,
        ),
        tmp_path,
        MODEL_PATH,
    )
    socket = ClockScenarioSocket(push_timesync=True)
    client = AsyncIQOption(
        ssid="synthetic",
        config=ConnectionConfig(timing_market_data_only=True, max_reconnect_attempts=0),
        offline=False,
        timing_recorder=trial.recorder,
    )

    # Fast-forward deadline so session time left is <= 0
    with (
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=socket)),
        patch("iqoption_research.connectors.iqoption.client.async_logout", new=AsyncMock()),
    ):
        async with client:
            past_deadline = time.perf_counter() - 1.0
            await trial.cycle(client, F, past_deadline)

    reg = trial.save()
    obs = reg.observations[0]
    assert obs["rejection_reason"] == "SESSION_DEADLINE"
    assert obs["gate_verdict"] == "FAIL_REJECTED"
    assert trial.metadata["orders_sent"] == 0
