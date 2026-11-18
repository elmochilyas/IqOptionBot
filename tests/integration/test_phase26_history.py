"""Offline fixtures verify acquisition mechanics; never authentic acquisition evidence."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.market_data.collector import DataAcquisitionError, HistoricalDataCollector
from iqoption_research.market_data.history_expansion import (
    ExpansionLimits,
    collect_history,
    validate_history,
)
from tests.unit.test_phase24_evaluation import fixture_candles


class Broker:
    """The entire available surface is read-only; unexpected access fails."""

    def __init__(self, candles: list[Any]) -> None:
        self.candles = candles
        self.calls: list[tuple[int, int]] = []
        self.failures = 0
        self.defect: str | None = None

    def assert_read_only(self) -> None:
        pass

    async def get_candles(self, symbol: str, size: int, count: int, *, endtime: int) -> Any:
        assert symbol == "EURUSD" and size == 60
        self.calls.append((count, endtime))
        if self.failures:
            self.failures -= 1
            raise TimeoutError("private-auth-response-must-not-be-saved")
        batch = [c for c in self.candles if c.from_ts <= endtime][-count:]
        if self.defect == "future":
            batch = self.candles[-count:]
        elif self.defect == "oversized":
            batch = self.candles
        elif self.defect == "duplicate" and batch:
            batch = [batch[0]] * min(count, 2)
        elif self.defect == "conflict" and batch:
            batch = [batch[0], replace(batch[0], volume=batch[0].volume + 1)]
        elif self.defect == "timestamp" and batch:
            batch = [replace(batch[0], from_ts=batch[0].from_ts + 1)]
        elif self.defect == "identity" and batch:
            batch = [replace(batch[0], symbol="EURUSD_OP")]
        elif self.defect == "synthetic" and batch:
            batch = [replace(batch[0], provenance="synthetic")]
        return SimpleNamespace(
            candle_list=list(reversed(batch)),
            source="unknown" if self.defect == "source" else "iqoption",
            errors=["rejected"] if self.defect == "errors" else [],
        )


def seed(root: Path) -> list[Any]:
    candles = fixture_candles(100)
    HistoricalDataCollector(storage_dir=root).persist_batch(
        candles[80:], symbol="EURUSD", candle_size_seconds=60
    )
    return candles


def limits(**kwargs: Any) -> ExpansionLimits:
    return ExpansionLimits(**{"max_requests": 3, "page_size": 10, "interval_seconds": 0, **kwargs})


@pytest.mark.asyncio
async def test_backward_limits_ordering_and_resume(tmp_path: Path) -> None:
    candles = seed(tmp_path)
    broker = Broker(candles)
    cfg = limits()
    result = await collect_history(tmp_path, broker, job_id="bounded", limits=cfg, resume=False)
    assert result["collection"]["status"] == "REQUEST_LIMIT"
    assert result["new_candles_this_job"] == 30
    assert len(broker.calls) == 3
    assert all(end < candles[80].from_ts and count <= 10 for count, end in broker.calls)
    assert broker.calls[1][1] < broker.calls[0][1]
    resumed = await collect_history(tmp_path, broker, job_id="bounded", limits=cfg, resume=True)
    assert len(broker.calls) == 3  # process restart does not replenish budget
    assert resumed["fingerprint_sha256"] == result["fingerprint_sha256"]
    validated = validate_history(tmp_path)
    assert validated["unique_candles"] == 50
    assert validated["source_archive_pages"] == 3


@pytest.mark.asyncio
async def test_retry_and_recovery_without_secret_persistence(tmp_path: Path) -> None:
    broker = Broker(seed(tmp_path))
    broker.failures = 1
    result = await collect_history(tmp_path, broker, job_id="retry", limits=limits(), resume=False)
    assert result["collection"]["requests"] == 3
    assert result["new_candles_this_job"] == 20
    assert "private-auth-response" not in (tmp_path / "phase26_retry.json").read_text()


@pytest.mark.asyncio
async def test_retry_exhaustion_then_resume(tmp_path: Path) -> None:
    broker = Broker(seed(tmp_path))
    broker.failures = 1
    cfg = limits(retries=0)
    result = await collect_history(tmp_path, broker, job_id="recover", limits=cfg, resume=False)
    assert result["collection"]["status"] == "RETRY_EXHAUSTED"
    resumed = await collect_history(tmp_path, broker, job_id="recover", limits=cfg, resume=True)
    assert resumed["new_candles_this_job"] == 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    ["future", "oversized", "conflict", "timestamp", "identity", "source", "errors", "synthetic"],
)
async def test_bad_pages_never_enter_catalog(tmp_path: Path, defect: str) -> None:
    broker = Broker(seed(tmp_path))
    broker.defect = defect
    before = validate_history(tmp_path)
    with pytest.raises(DataAcquisitionError):
        await collect_history(tmp_path, broker, job_id="invalid", limits=limits(), resume=False)
    assert validate_history(tmp_path)["fingerprint_sha256"] == before["fingerprint_sha256"]
    assert not (tmp_path / "phase26.lock").exists()
    state = json.loads((tmp_path / "phase26_invalid.json").read_text())
    assert state["status"] == "FAILED" and state["requests"] == 1


@pytest.mark.asyncio
async def test_identical_duplicates_and_candle_cap(tmp_path: Path) -> None:
    broker = Broker(seed(tmp_path))
    broker.defect = "duplicate"
    result = await collect_history(
        tmp_path, broker, job_id="duplicates", limits=limits(max_new_candles=2), resume=False
    )
    assert result["new_candles_this_job"] == 2
    assert result["collection"]["status"] == "CANDLE_LIMIT"


@pytest.mark.asyncio
async def test_crash_after_publish_before_cursor_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = Broker(seed(tmp_path))
    original = HistoricalDataCollector.persist_batch
    crashed = False

    def publish_then_crash(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal crashed
        result = original(self, *args, **kwargs)
        if not crashed:
            crashed = True
            raise OSError("simulated interruption after catalog commit")
        return result

    monkeypatch.setattr(HistoricalDataCollector, "persist_batch", publish_then_crash)
    cfg = limits()
    with pytest.raises(OSError):
        await collect_history(tmp_path, broker, job_id="crash", limits=cfg, resume=False)
    assert validate_history(tmp_path)["unique_candles"] == 30
    result = await collect_history(tmp_path, broker, job_id="crash", limits=cfg, resume=True)
    assert result["new_candles_this_job"] == 20
    assert result["unique_candles"] == 40
    assert broker.calls[0] == broker.calls[1]


@pytest.mark.asyncio
async def test_empty_response_is_not_retention_guarantee_and_archive_tamper_fails(
    tmp_path: Path,
) -> None:
    seed(tmp_path)
    result = await collect_history(
        tmp_path, Broker([]), job_id="empty", limits=limits(), resume=False
    )
    assert result["collection"]["status"] == "EMPTY_RESPONSE_LOOKBACK_UNPROVEN"
    assert result["new_candles_this_job"] == 0
    source = next((tmp_path / "phase26_sources").glob("*/*.json"))
    source.write_text("{}")
    with pytest.raises(DataAcquisitionError, match="checksum"):
        validate_history(tmp_path)


def test_catalog_conflicts_are_not_silently_deduplicated(tmp_path: Path) -> None:
    candles = seed(tmp_path)
    HistoricalDataCollector(storage_dir=tmp_path).persist_batch(
        [replace(candles[85], volume=candles[85].volume + 1)],
        symbol="EURUSD",
        candle_size_seconds=60,
    )
    with pytest.raises(DataAcquisitionError, match="Conflicting"):
        validate_history(tmp_path)


def test_offline_cli_never_authenticates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed(tmp_path)

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Authentication forbidden offline")

    monkeypatch.setattr("iqoption_research.cli.commands.history_cmd._connect_live_client", blocked)
    result = CliRunner().invoke(app, ["data", "expand-history", "--storage", str(tmp_path)])
    assert result.exit_code == 0
    assert "OFFLINE_PLAN" in result.output


def test_live_cli_uses_only_interactive_connection_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = Broker(seed(tmp_path))
    cleaned = []

    async def connect(saved_credentials: bool) -> Any:
        assert saved_credentials is False

        async def cleanup() -> None:
            cleaned.append(True)

        return broker, cleanup

    monkeypatch.setattr("iqoption_research.cli.commands.history_cmd._connect_live_client", connect)
    result = CliRunner().invoke(
        app, ["data", "expand-history", "--storage", str(tmp_path), "--live", "--max-requests", "1"]
    )
    assert result.exit_code == 0, result.output
    assert len(broker.calls) == 1 and cleaned == [True]


@pytest.mark.asyncio
async def test_cancellation_charges_request_and_releases_lock(tmp_path: Path) -> None:
    candles = seed(tmp_path)

    class CancelledBroker(Broker):
        async def get_candles(self, *args: Any, **kwargs: Any) -> Any:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await collect_history(
            tmp_path, CancelledBroker(candles), job_id="cancel", limits=limits(), resume=False
        )
    state = json.loads((tmp_path / "phase26_cancel.json").read_text())
    assert state["requests"] == 1 and state["status"] == "INTERRUPTED"
    assert not (tmp_path / "phase26.lock").exists()


@pytest.mark.asyncio
async def test_source_market_payload_preserved_without_unknown_auth_fields(tmp_path: Path) -> None:
    candles = seed(tmp_path)
    candles = [
        replace(c, raw={"open": c.open, "from": c.from_ts, "token": "private-token-value"})
        for c in candles
    ]
    await collect_history(
        tmp_path, Broker(candles), job_id="raw", limits=limits(max_requests=1), resume=False
    )
    path = next((tmp_path / "phase26_sources").glob("*/*.json"))
    payload = path.read_text()
    assert "private-token-value" not in payload
    source = json.loads(payload)
    assert source["records"][0]["raw"]["open"] == source["records"][0]["open"]


@pytest.mark.asyncio
async def test_real_connector_emits_only_get_candles_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candles = seed(tmp_path)
    sent = []

    class Transport:
        server_time_ns = candles[-1].to_ts * 1_000_000_000

        async def send_and_wait(self, name: str, frame: dict[str, Any], **kwargs: Any) -> Any:
            sent.append((name, frame))
            assert name == "sendMessage" and frame["name"] == "get-candles"
            body = frame["body"]
            assert body["active_id"] == 1 and body["size"] == 60
            selected = [c for c in candles if c.from_ts <= body["to"]][-body["count"] :]
            return {
                "name": "candles",
                "request_id": kwargs["request_id"],
                "msg": {
                    "candles": [
                        {
                            "active_id": 1,
                            "size": 60,
                            "from": c.from_ts,
                            "to": c.to_ts,
                            "at": c.to_ts * 1_000_000_000,
                            "open": c.open,
                            "close": c.close,
                            "min": c.low,
                            "max": c.high,
                            "volume": c.volume,
                            "bid": c.bid,
                            "ask": c.ask,
                        }
                        for c in selected
                    ]
                },
            }

    client = AsyncIQOption(offline=True)
    monkeypatch.setattr(client, "_require_ready", lambda: Transport())
    result = await collect_history(
        tmp_path, client, job_id="wire", limits=limits(max_requests=1), resume=False
    )
    assert result["new_candles_this_job"] == 10
    assert len(sent) == 1  # zero order submissions, wallet changes, or subscriptions


@pytest.mark.asyncio
async def test_target_boundary_is_not_persisted_beyond(tmp_path: Path) -> None:
    # Two days of offline fixture history, with only the newest ten minutes seeded.
    candles = fixture_candles(3000)
    HistoricalDataCollector(storage_dir=tmp_path).persist_batch(
        candles[-10:], symbol="EURUSD", candle_size_seconds=60
    )
    cfg = limits(target_days=1, max_requests=4, page_size=500)
    result = await collect_history(
        tmp_path, Broker(candles), job_id="target", limits=cfg, resume=False
    )
    assert result["collection"]["status"] == "TARGET_REACHED"
    assert result["oldest_ts"] >= result["collection"]["target_ts"]


@pytest.mark.asyncio
async def test_lock_resume_limits_and_invalid_limits_fail_before_requests(tmp_path: Path) -> None:
    broker = Broker(seed(tmp_path))
    with pytest.raises(ValueError):
        await collect_history(
            tmp_path, broker, job_id="bad", limits=limits(page_size=501), resume=False
        )
    (tmp_path / "phase26.lock").write_text("operator-owned lock")
    with pytest.raises(FileExistsError):
        await collect_history(tmp_path, broker, job_id="lock", limits=limits(), resume=False)
    assert not broker.calls
    (tmp_path / "phase26.lock").unlink()
    await collect_history(tmp_path, broker, job_id="limits", limits=limits(), resume=False)
    calls = len(broker.calls)
    checkpoint = tmp_path / "phase26_limits.json"
    original = checkpoint.read_bytes()
    with pytest.raises(DataAcquisitionError, match="limits"):
        await collect_history(
            tmp_path, broker, job_id="limits", limits=limits(max_requests=4), resume=True
        )
    assert len(broker.calls) == calls
    assert checkpoint.read_bytes() == original
    state = json.loads(original)
    state["initial_fingerprint"] = "0" * 64
    checkpoint.write_text(json.dumps(state))
    with pytest.raises(DataAcquisitionError, match="Original coverage changed"):
        await collect_history(tmp_path, broker, job_id="limits", limits=limits(), resume=True)
    assert len(broker.calls) == calls
