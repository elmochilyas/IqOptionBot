"""Shadow trials over actual connector frames, pinned model and SQLite journal."""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import TradingNotAllowedError
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.timing.reliability import evaluate
from iqoption_research.timing.shadow import MODEL_PATH, ShadowTrial, eligible_model
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
