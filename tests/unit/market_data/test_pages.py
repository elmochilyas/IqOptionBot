"""Phase 44 CLI synthetic fixture regressions: timeframes and real pagination.

All candles here are fabricated by the production-side synthetic fixture module.
These tests exercise engineering behavior only; synthetic output is never genuine
market evidence.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.commands import collect_cmd
from iqoption_research.cli.main import app
from iqoption_research.market_data import synthetic_collection_fixture
from iqoption_research.market_data.collection_engine import CollectionStatus
from iqoption_research.market_data.synthetic_collection_fixture import (
    create_synthetic_collection_client,
)
from iqoption_research.models.market import CandleSize

runner = CliRunner()


def _parse_json(stdout: str) -> dict[str, Any]:
    return json.loads(stdout)


def _plan_job(*, timeframe: int, start: str, end: str) -> str:
    result = runner.invoke(
        app,
        [
            "data",
            "collect",
            "plan",
            "--symbol",
            "EURUSD",
            "--from",
            start,
            "--to",
            end,
            "--timeframe",
            str(timeframe),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    return str(_parse_json(result.stdout)["job_id"])


def _verify_job(job_id: str) -> dict[str, Any]:
    result = runner.invoke(
        app,
        ["data", "collect", "verify", "--job-id", job_id, "--json"],
    )
    assert result.exit_code == 0, result.output
    return _parse_json(result.stdout)


def test_synthetic_non_m1_timeframe_completes_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 300s synthetic job produces M5 pages and verifies as SYNTHETIC_TEST_FIXTURE."""
    monkeypatch.chdir(tmp_path)
    job_id = _plan_job(
        timeframe=300,
        start="2026-11-05T10:00:00+00:00",
        end="2026-11-05T11:00:00+00:00",
    )

    run = runner.invoke(
        app,
        [
            "data",
            "collect",
            "run",
            "--job-id",
            job_id,
            "--synthetic",
            "--json",
            "--chunk-size",
            "12",
        ],
    )
    assert run.exit_code == 0, run.output
    progress = _parse_json(run.stdout)
    assert progress["status"] == CollectionStatus.COMPLETED_VERIFIED.value
    assert progress["timeframe_seconds"] == 300
    assert progress["candles_collected_this_session"] == 12

    report = _verify_job(job_id)
    assert report["verified"] is True
    assert report["unique_verified_candles"] == 12
    assert report["coverage_gaps_count"] == 0
    assert report["provenance_status"] == ["SYNTHETIC_TEST_FIXTURE"]

    assert list((tmp_path / "data" / "synthetic_datasets").rglob("*.parquet"))
    assert not list((tmp_path / "data" / "datasets").rglob("*.parquet"))


def test_synthetic_job_advances_cursor_across_multiple_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every page is generated from the actual call endtime/count/size arguments."""
    monkeypatch.chdir(tmp_path)

    real_builder = collect_cmd.create_synthetic_collection_client
    calls: list[dict[str, Any]] = []

    def tracking_builder(*, symbol: str, timeframe_seconds: int) -> Any:
        client = real_builder(symbol=symbol, timeframe_seconds=timeframe_seconds)
        original_get_candles = client.get_candles

        async def tracked_get_candles(*args: Any, **kwargs: Any) -> Any:
            calls.append({
                "symbol": args[0] if args else kwargs.get("symbol"),
                "size": kwargs.get("size"),
                "count": kwargs.get("count"),
                "endtime": kwargs.get("endtime"),
            })
            return await original_get_candles(*args, **kwargs)

        client.get_candles = AsyncMock(side_effect=tracked_get_candles)
        return client

    monkeypatch.setattr(collect_cmd, "create_synthetic_collection_client", tracking_builder)

    job_id = _plan_job(
        timeframe=60,
        start="2026-11-05T10:00:00+00:00",
        end="2026-11-05T10:35:00+00:00",
    )
    run = runner.invoke(
        app,
        [
            "data",
            "collect",
            "run",
            "--job-id",
            job_id,
            "--synthetic",
            "--json",
            "--chunk-size",
            "10",
        ],
    )
    assert run.exit_code == 0, run.output
    progress = _parse_json(run.stdout)
    assert progress["status"] == CollectionStatus.COMPLETED_VERIFIED.value
    assert progress["candles_collected_this_session"] == 35
    assert progress["receipt_count"] == 4

    from_ts = int(datetime(2026, 11, 5, 10, 0, tzinfo=UTC).timestamp())
    assert [call["endtime"] for call in calls] == [
        from_ts + 35 * 60,
        from_ts + 25 * 60,
        from_ts + 15 * 60,
        from_ts + 5 * 60,
    ]
    assert [call["count"] for call in calls] == [10, 10, 10, 6]
    assert all(call["symbol"] == "EURUSD" for call in calls)
    assert all(call["size"] == CandleSize.MIN_1 for call in calls)

    report = _verify_job(job_id)
    assert report["verified"] is True
    assert report["receipt_count"] == 4
    assert report["unique_verified_candles"] == 35
    assert report["coverage_gaps_count"] == 0
    assert report["provenance_status"] == ["SYNTHETIC_TEST_FIXTURE"]


def test_production_cli_has_no_test_tree_imports() -> None:
    """Installed packages contain only iqoption_research*, never tests.*."""
    for module in (collect_cmd, synthetic_collection_fixture):
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        assert "from tests." not in source
        assert "import tests" not in source


def test_synthetic_page_is_derived_from_call_arguments() -> None:
    """One mocked fetch returns exactly count fabricated bars ending at endtime."""
    client = create_synthetic_collection_client(symbol="EURUSD", timeframe_seconds=300)
    endtime = int(datetime(2026, 11, 5, 10, 0, tzinfo=UTC).timestamp())

    result = asyncio.run(
        client.get_candles(
            "EURUSD",
            size=CandleSize.MIN_5,
            count=4,
            endtime=endtime,
        )
    )

    assert result.source == "synthetic"
    assert result.raw_count == 4
    assert [c.from_ts for c in result.candles] == [
        endtime - 1200,
        endtime - 900,
        endtime - 600,
        endtime - 300,
    ]
    assert all(c.to_ts - c.from_ts == 300 for c in result.candles)
    assert all(c.size == CandleSize.MIN_5 and c.symbol == "EURUSD" for c in result.candles)

    with pytest.raises(ValueError, match="checkpoint timeframe"):
        asyncio.run(
            client.get_candles(
                "EURUSD",
                size=CandleSize.MIN_1,
                count=2,
                endtime=endtime,
            )
        )
