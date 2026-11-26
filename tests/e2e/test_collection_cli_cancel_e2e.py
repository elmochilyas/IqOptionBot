"""End-to-end Phase 44 CLI cancellation and explicit live-account governance tests.

The CLI handlers and production coordinator are exercised with an injected offline
broker. No genuine broker connection, order, or wallet mutation is attempted.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from iqoption_research.auth.vault import WindowsCredentialVault
from iqoption_research.cli.commands import backfill_cmd
from iqoption_research.cli.main import app
from iqoption_research.market_data.collection_engine import (
    CollectionCheckpoint,
    CollectionStatus,
)
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.market import CandleSize
from tests.desktop.collection_center.test_account_selection import WinVaultKeyring
from tests.unit.market_data.test_inventory import create_deterministic_synthetic_candles

runner = CliRunner()


def _auth_candles(count: int, start_ts: int) -> list[Any]:
    """Offline fixture candles carrying authentic-broker provenance metadata."""
    return [
        replace(
            candle,
            provenance="historical",
            identity_provenance="explicit_broker_metadata",
        )
        for candle in create_deterministic_synthetic_candles(count=count, start_ts=start_ts)
    ]


def _two_account_vault() -> tuple[WindowsCredentialVault, str, str]:
    backend = WinVaultKeyring()
    vault = WindowsCredentialVault(service_name="phase44_cli_e2e_vault", keyring_backend=backend)
    store = vault.account_store()
    aid1 = store.save(
        email="cli-alpha@example.com",
        password="AlphaPassword!",
        explicit_opt_in=True,
    )
    aid2 = store.save(
        email="cli-beta@example.com",
        password="BetaPassword!",
        explicit_opt_in=True,
    )
    return vault, aid1, aid2


def _plan_job() -> str:
    result = runner.invoke(
        app,
        [
            "data",
            "collect",
            "plan",
            "--symbol",
            "EURUSD",
            "--from",
            "2026-11-04T10:00:00+00:00",
            "--to",
            "2026-11-04T10:10:00+00:00",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    return str(json.loads(result.stdout)["job_id"])


def test_cli_cancel_stops_active_worker_without_post_cancel_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel issued during an in-flight fetch leaves no receipt or parquet behind."""
    monkeypatch.chdir(tmp_path)
    vault, aid1, aid2 = _two_account_vault()
    monkeypatch.setattr("iqoption_research.auth.vault.WindowsCredentialVault", lambda: vault)

    job_id = _plan_job()
    start_ts = int(datetime(2026, 11, 4, 10, 0, tzinfo=UTC).timestamp())
    candles = _auth_candles(10, start_ts)
    quality = assess_data_quality(candles, expected_size=CandleSize(60))
    fetch_result = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(candles),
        errors=(),
        quality=quality,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles),
        source="iqoption",
    )

    fetch_calls = 0

    async def cancel_during_fetch(*_args: Any, **_kwargs: Any) -> CandleFetchResult:
        nonlocal fetch_calls
        fetch_calls += 1
        cancel_result = runner.invoke(
            app,
            ["data", "collect", "cancel", "--job-id", job_id, "--json"],
        )
        assert cancel_result.exit_code == 2, cancel_result.output
        # The fetch remains in flight after cancellation; its result must be discarded.
        await asyncio.sleep(0)
        return fetch_result

    broker = MagicMock()
    broker.assert_read_only = MagicMock()
    broker.get_candles = AsyncMock(side_effect=cancel_during_fetch)
    cleanup_calls = 0

    async def connect_offline(*_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        async def cleanup() -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

        return broker, cleanup

    monkeypatch.setattr(backfill_cmd, "_connect_live_client", connect_offline)

    result = runner.invoke(
        app,
        [
            "data",
            "collect",
            "run",
            "--job-id",
            job_id,
            "--live",
            "--account-id",
            aid2,
            "--json",
        ],
    )

    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == CollectionStatus.CANCELLED.value
    assert fetch_calls == 1
    assert cleanup_calls == 1

    checkpoint = CollectionCheckpoint.load(
        tmp_path / "data" / "collection_checkpoints" / f"{job_id}.json"
    )
    assert checkpoint.status == CollectionStatus.CANCELLED.value
    assert checkpoint.account_id == aid2
    assert checkpoint.account_label == aid2
    assert checkpoint.receipt_ids == []
    assert aid1 not in result.stdout
    assert "cli-alpha@example.com" not in result.stdout
    assert "cli-beta@example.com" not in result.stdout

    assert list((tmp_path / "data" / "collection_receipts").glob("rcpt_*.json")) == []
    assert list((tmp_path / "data" / "datasets").rglob("*.parquet")) == []
    assert list((tmp_path / "data" / "synthetic_datasets").rglob("*.parquet")) == []
    assert not (tmp_path / "data" / "collection_checkpoints" / f"{job_id}.cancel").exists()


def test_cli_resume_cancel_during_live_authentication_is_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel during resume authentication is not mistaken for a stale signal."""
    monkeypatch.chdir(tmp_path)
    vault, aid1, _ = _two_account_vault()
    monkeypatch.setattr("iqoption_research.auth.vault.WindowsCredentialVault", lambda: vault)

    job_id = _plan_job()
    idle_cancel = runner.invoke(app, ["data", "collect", "cancel", "--job-id", job_id, "--json"])
    assert idle_cancel.exit_code == 2, idle_cancel.output
    cancel_file = tmp_path / "data" / "collection_checkpoints" / f"{job_id}.cancel"
    original_token = cancel_file.read_text(encoding="utf-8")

    start_ts = int(datetime(2026, 11, 4, 10, 0, tzinfo=UTC).timestamp())
    candles = _auth_candles(10, start_ts)
    quality = assess_data_quality(candles, expected_size=CandleSize(60))
    fetch_result = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(candles),
        errors=(),
        quality=quality,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles),
        source="iqoption",
    )
    broker = MagicMock()
    broker.assert_read_only = MagicMock()
    broker.get_candles = AsyncMock(return_value=fetch_result)

    async def connect_offline(*_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        # This is the real auth gap: CLI has captured the stale token, then the
        # operator cancels while the broker client is still being constructed.
        new_cancel = runner.invoke(app, ["data", "collect", "cancel", "--job-id", job_id, "--json"])
        assert new_cancel.exit_code == 2, new_cancel.output
        assert cancel_file.read_text(encoding="utf-8") != original_token

        async def cleanup() -> None:
            return None

        return broker, cleanup

    monkeypatch.setattr(backfill_cmd, "_connect_live_client", connect_offline)

    result = runner.invoke(
        app,
        [
            "data",
            "collect",
            "resume",
            "--job-id",
            job_id,
            "--live",
            "--account-id",
            aid1,
            "--json",
        ],
    )

    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == CollectionStatus.CANCELLED.value
    assert broker.get_candles.await_count == 0

    checkpoint = CollectionCheckpoint.load(
        tmp_path / "data" / "collection_checkpoints" / f"{job_id}.json"
    )
    assert checkpoint.status == CollectionStatus.CANCELLED.value
    assert checkpoint.receipt_ids == []
    assert not cancel_file.exists()


def test_cli_live_backfill_fails_closed_without_explicit_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved-credential live backfill refuses to construct provider or broker without --account-id."""
    monkeypatch.chdir(tmp_path)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("no broker or provider must be constructed")

    monkeypatch.setattr("iqoption_research.auth.manager.AuthenticationManager", forbidden)
    monkeypatch.setattr("iqoption_research.connectors.iqoption.client.AsyncIQOption", forbidden)

    result = runner.invoke(
        app,
        [
            "data",
            "backfill",
            "--live",
            "--yes-i-understand",
            "--saved",
            "--json",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Explicit --account-id" in result.output


def test_cli_live_backfill_unavailable_account_fails_before_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable selected account fails before any auth provider or broker construction."""
    monkeypatch.chdir(tmp_path)
    vault, _, _ = _two_account_vault()
    monkeypatch.setattr("iqoption_research.auth.vault.WindowsCredentialVault", lambda: vault)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("no broker or provider must be constructed")

    monkeypatch.setattr("iqoption_research.auth.manager.AuthenticationManager", forbidden)
    monkeypatch.setattr("iqoption_research.connectors.iqoption.client.AsyncIQOption", forbidden)

    result = runner.invoke(
        app,
        [
            "data",
            "backfill",
            "--live",
            "--yes-i-understand",
            "--saved",
            "--account-id",
            "99999999999999999999999999999999",
            "--json",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "unavailable" in result.output
