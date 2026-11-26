"""Offline diagnostic fixtures are engineering checks, not historical availability evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.market_data.fetch import summarize_candle_response
from iqoption_research.market_data.history_boundary import diagnose_boundary, plan_boundary
from tests.unit.connectors.test_evaluation import fixture_candles


def seed(root: Path) -> list[Any]:
    candles = fixture_candles(100)
    HistoricalDataCollector(storage_dir=root).persist_batch(
        candles[60:], symbol="EURUSD", candle_size_seconds=60
    )
    return candles


def immutable_files(root: Path) -> dict[str, str]:
    state = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and not p.name.startswith("catalog.db")
    }
    # SQLite may checkpoint its WAL while closing unrelated existing handles;
    # compare catalog contents, not the transient physical page layout.
    with closing(
        sqlite3.connect((root / "catalog.db").resolve().as_uri() + "?mode=ro", uri=True)
    ) as connection:
        state["catalog_logical_state"] = hashlib.sha256(
            "\n".join(connection.iterdump()).encode()
        ).hexdigest()
    return state


class Broker:
    def __init__(self, candles: list[Any], defect: str | None = None) -> None:
        self.candles = candles
        self.defect = defect
        self.calls: list[tuple[str, int, int, int]] = []

    def assert_read_only(self) -> None:
        pass

    async def get_candles(self, symbol: str, size: int, count: int, *, endtime: int) -> Any:
        self.calls.append((symbol, size, count, endtime))
        if self.defect == "timeout":
            raise TimeoutError("private-token-not-for-the-report")
        if self.defect == "cancel":
            raise asyncio.CancelledError
        records = [c for c in self.candles if c.from_ts <= endtime][-count:]
        response = {"name": "candles", "request_id": "correlated", "msg": {"candles": []}}
        evidence = summarize_candle_response(response, "correlated", 1)
        if self.defect == "empty_control":
            records = []
        elif self.defect == "conflict" and records:
            records = [replace(c, volume=c.volume + 1) for c in records]
        elif self.defect == "after_end":
            records = self.candles[-count:]
        elif self.defect == "rejected":
            return SimpleNamespace(
                candle_list=records,
                errors=[object()],
                source="iqoption",
                response_evidence=evidence,
            )
        elif self.defect == "bad_source":
            return SimpleNamespace(
                candle_list=records, errors=[], source="simulated", response_evidence=evidence
            )
        elif self.defect == "rate_limit":
            evidence["payload_status"] = "RATE_LIMITED"
            records = []
        elif self.defect == "unknown_status" and not records:
            evidence["payload_code"] = 4000
        elif self.defect == "missing_correlation":
            evidence["correlation"] = "ABSENT"
        elif self.defect == "error_response":
            error = InvalidCandleError("private broker error text")
            error.response_evidence = {**evidence, "frame_type": "ERROR"}
            raise error
        return SimpleNamespace(
            candle_list=records,
            errors=[],
            source="iqoption",
            response_evidence=evidence,
            raw_count=len(records),
        )


@pytest.fixture(autouse=True)
def instant_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_delay(delay: float) -> None:
        pass

    monkeypatch.setattr("iqoption_research.market_data.history_boundary.asyncio.sleep", no_delay)


def test_plan_has_exact_documented_boundaries_and_utc_seconds(tmp_path: Path) -> None:
    storage = tmp_path / "datasets"
    candles = seed(storage)
    oldest = candles[60].from_ts
    plan = plan_boundary(storage)
    assert plan["maximum_candle_requests"] == 5
    assert [p["endtime"] for p in plan["requests"]] == [
        oldest + 1199,
        oldest,
        oldest - 1,
        oldest - 7 * 86400 - 1,
        oldest - 28 * 86400 - 1,
    ]
    assert all(
        p["count"] == 20 and 1_000_000_000 < p["endtime"] < 10_000_000_000 for p in plan["requests"]
    )
    assert plan_boundary(storage, earlier_windows=1)["maximum_candle_requests"] == 4
    with pytest.raises(ValueError):
        plan_boundary(storage, count=501)
    with pytest.raises(ValueError):
        plan_boundary(storage, earlier_windows=3)
    # A later known block is a better control than repeatedly querying the old boundary.
    later = [
        replace(c, from_ts=c.from_ts + 14 * 86400, to_ts=c.to_ts + 14 * 86400) for c in candles[80:]
    ]
    HistoricalDataCollector(storage_dir=storage).persist_batch(
        later, symbol="EURUSD", candle_size_seconds=60
    )
    recent_plan = plan_boundary(storage)
    assert recent_plan["known_control_start_ts"] == later[0].from_ts
    assert recent_plan["oldest_catalog_ts"] == oldest


@pytest.mark.asyncio
async def test_known_control_and_empty_older_windows_preserve_every_source(tmp_path: Path) -> None:
    storage = tmp_path / "datasets"
    candles = seed(storage)
    (storage / "phase26_sources").mkdir()
    (storage / "phase26_sources" / "original.json").write_text("immutable source evidence")
    (storage / "phase26_job.json").write_text("original checkpoint")
    before = immutable_files(storage)
    broker = Broker(candles[60:])
    report = await diagnose_boundary(storage, broker, tmp_path / "report.json")
    assert len(broker.calls) == report["candle_requests_attempted"] == 5
    assert report["historical_availability"] == "UNAVAILABLE_IN_TESTED_WINDOWS"
    assert report["retention_ceiling_confirmed"] is False
    assert report["results"][0]["matched_catalog_candles"] == 20
    assert report["results"][1]["matched_catalog_candles"] == 1
    assert report["boundary_observation"] == "OLDEST_OPEN_RETURNED_AT_EQUAL_ENDTIME"
    assert all(r["returned_candle_count"] == 0 for r in report["results"][2:])
    assert immutable_files(storage) == before


@pytest.mark.asyncio
async def test_older_records_demonstrate_retrieval_without_catalog_publication(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "datasets"
    broker = Broker(seed(storage))
    before = immutable_files(storage)
    report = await diagnose_boundary(storage, broker, tmp_path / "report.json")
    assert report["historical_availability"] == "SUCCESSFULLY_RETRIEVABLE"
    assert report["results"][2]["older_than_catalog_count"] == 20
    assert report["catalog_publications"] == 0
    assert before == immutable_files(storage)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("defect", "outcome"),
    [
        ("empty_control", "STILL_UNVERIFIED"),
        ("conflict", "PROTOCOL_ANOMALY_OBSERVED_CAUSE_UNVERIFIED"),
        ("after_end", "PROTOCOL_ANOMALY_OBSERVED_CAUSE_UNVERIFIED"),
        ("rejected", "PROTOCOL_ANOMALY_OBSERVED_CAUSE_UNVERIFIED"),
        ("bad_source", "PROTOCOL_ANOMALY_OBSERVED_CAUSE_UNVERIFIED"),
        ("missing_correlation", "STILL_UNVERIFIED"),
        ("unknown_status", "STILL_UNVERIFIED"),
        ("error_response", "PROTOCOL_ANOMALY_OBSERVED_CAUSE_UNVERIFIED"),
    ],
)
async def test_anomalies_do_not_masquerade_as_retention(
    tmp_path: Path, defect: str, outcome: str
) -> None:
    storage = tmp_path / "datasets"
    candles = seed(storage)
    broker = Broker(candles[60:], defect)
    report = await diagnose_boundary(storage, broker, tmp_path / "report.json")
    assert report["historical_availability"] == outcome
    assert len(broker.calls) <= 5
    assert "private" not in json.dumps(report)


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["timeout", "rate_limit", "cancel"])
async def test_failure_is_charged_durable_and_not_retried(tmp_path: Path, defect: str) -> None:
    storage = tmp_path / "datasets"
    broker = Broker(seed(storage), defect)
    path = tmp_path / "report.json"
    if defect == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await diagnose_boundary(storage, broker, path)
    else:
        await diagnose_boundary(storage, broker, path)
    saved = json.loads(path.read_text())
    assert saved["candle_requests_attempted"] == len(broker.calls) == 1
    assert saved["state"] != "RUNNING"
    assert "private-token" not in path.read_text()
    with pytest.raises(FileExistsError):
        await diagnose_boundary(storage, broker, path)
    assert len(broker.calls) == 1


def test_offline_cli_never_authenticates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = tmp_path / "datasets"
    seed(storage)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No authentication allowed")

    monkeypatch.setattr(
        "iqoption_research.cli.commands.history_cmd._connect_live_client", forbidden
    )
    result = CliRunner().invoke(
        app, ["data", "diagnose-history-boundary", "--storage", str(storage)]
    )
    assert result.exit_code == 0 and "OFFLINE_PLAN" in result.output


def test_cli_live_is_interactive_only_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "datasets"
    broker = Broker(seed(storage)[60:])
    cleaned = []

    async def connect(saved: bool) -> Any:
        assert saved is False

        async def cleanup() -> None:
            cleaned.append(True)

        return broker, cleanup

    monkeypatch.setattr("iqoption_research.cli.commands.history_cmd._connect_live_client", connect)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "diagnose-history-boundary",
            "--storage",
            str(storage),
            "--report",
            str(tmp_path / "report.json"),
            "--live",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(broker.calls) == 5 and cleaned == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect", [None, "correlation", "identity", "shape", "error", "empty_error_field"]
)
async def test_real_connector_frames_correlation_and_sanitized_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str | None
) -> None:
    storage = tmp_path / "datasets"
    candles = seed(storage)
    sent = []

    class Transport:
        server_time_ns = candles[-1].to_ts * 1_000_000_000

        async def send_and_wait(self, name: str, frame: dict[str, Any], **kwargs: Any) -> Any:
            sent.append(frame)
            assert name == "sendMessage" and frame["name"] == "get-candles"
            body = frame["body"]
            assert body["size"] == 60 and body["active_id"] == 1 and body["count"] == 20
            assert type(body["to"]) is int and body["to"] < 10_000_000_000
            selected = [c for c in candles[60:] if c.from_ts <= body["to"]][-20:]
            response = {
                "name": "candles",
                "request_id": kwargs["request_id"],
                "status": "private-response-token",
                "msg": {
                    "active_id": 1,
                    "candles": [
                        {
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
                    ],
                },
            }
            if defect == "correlation":
                response["request_id"] = "private-wrong-request"
            elif defect == "identity":
                response["msg"]["active_id"] = 76
            elif defect == "shape":
                response["msg"].pop("candles")
            elif defect == "error":
                response["name"] = "error"
            elif defect == "empty_error_field":
                response["status"] = "ok"
                response["msg"]["errors"] = []
            return response

    client = AsyncIQOption(offline=True)
    monkeypatch.setattr(client, "_require_ready", lambda: Transport())
    report = await diagnose_boundary(storage, client, tmp_path / "report.json")
    assert len(sent) == 5
    assert "private-" not in json.dumps(report)
    if defect == "empty_error_field":
        assert report["historical_availability"] == "UNAVAILABLE_IN_TESTED_WINDOWS"
        assert all(r["response"]["error_field_present"] for r in report["results"])
        assert not any(r["response"]["error_signal_present"] for r in report["results"])
    elif defect is None:
        assert report["results"][0]["matched_catalog_candles"] == 20
        assert report["historical_availability"] == "STILL_UNVERIFIED"  # unknown status on empties
    else:
        assert all(r["status"] == "REQUEST_FAILED" for r in report["results"])
        assert report["historical_availability"] == "PROTOCOL_ANOMALY_OBSERVED_CAUSE_UNVERIFIED"


@pytest.mark.asyncio
async def test_cannot_write_report_into_immutable_storage(tmp_path: Path) -> None:
    storage = tmp_path / "datasets"
    broker = Broker(seed(storage))
    with pytest.raises(ValueError):
        await diagnose_boundary(storage, broker, storage / "report.json")
    assert not broker.calls
