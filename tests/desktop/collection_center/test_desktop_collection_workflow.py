"""End-to-end Phase 44 desktop Collection Center workflow test (offline only).

Launches the production dashboard and panel, selects one of two synthetic vault
accounts, confirms, starts, cancels, reopens, resumes, verifies and snapshots with
an injected offline read-only broker. No genuine broker connection is attempted.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iqoption_research.auth.vault import WindowsCredentialVault
from iqoption_research.cli.commands import backfill_cmd
from iqoption_research.desktop import collection_center_panel
from iqoption_research.desktop.app import Dashboard
from iqoption_research.market_data.collection_engine import CollectionStatus
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.market import CandleSize
from tests.desktop.collection_center.test_account_selection import WinVaultKeyring
from tests.unit.market_data.test_inventory import create_deterministic_synthetic_candles

START_UTC = "2026-11-04T10:00:00+00:00"
END_UTC = "2026-11-04T10:10:00+00:00"


def _auth_candles(count: int, start_ts: int) -> list[Any]:
    return [
        replace(
            candle,
            provenance="historical",
            identity_provenance="explicit_broker_metadata",
        )
        for candle in create_deterministic_synthetic_candles(count=count, start_ts=start_ts)
    ]


def _drain_after_queue(root: tk.Tk, queue: list[Callable[[], None]]) -> None:
    """Run currently queued Tk callbacks without chasing self-rescheduling ones."""
    pending = list(queue)
    queue.clear()
    for callback in pending:
        callback()
        root.update()


def _pump_until(
    root: tk.Tk,
    predicate: Callable[[], bool],
    *,
    after_queue: list[Callable[[], None]] | None = None,
    timeout: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if after_queue is not None:
            _drain_after_queue(root, after_queue)
        root.update()
        if time.monotonic() > deadline:
            pytest.fail("Timed out waiting for desktop worker to finish")
        time.sleep(0.01)
    if after_queue is not None:
        _drain_after_queue(root, after_queue)
    root.update()


def test_desktop_two_account_workflow_start_cancel_reopen_resume_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected second account drives the whole production desktop workflow."""
    monkeypatch.chdir(tmp_path)
    real_vault_cls = WindowsCredentialVault
    backend = WinVaultKeyring()
    vault = real_vault_cls(service_name="phase44_desktop_e2e_vault", keyring_backend=backend)
    store = vault.account_store()
    aid1 = store.save(
        email="desktop-alpha@example.com",
        password="AlphaDesktop!",
        explicit_opt_in=True,
    )
    aid2 = store.save(
        email="desktop-beta@example.com",
        password="BetaDesktop!",
        explicit_opt_in=True,
    )

    def vault_factory(*_args: Any, **_kwargs: Any) -> WindowsCredentialVault:
        return real_vault_cls(service_name="phase44_desktop_e2e_vault", keyring_backend=backend)

    monkeypatch.setattr("iqoption_research.auth.vault.WindowsCredentialVault", vault_factory)

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

    broker_state: dict[str, Any] = {
        "block_first_fetch": True,
        "fetch_started": threading.Event(),
        "loop": None,
        "release": None,
        "calls": 0,
    }

    class OfflineReadOnlyClient:
        def assert_read_only(self) -> None:
            return None

        async def get_candles(self, *_args: Any, **_kwargs: Any) -> CandleFetchResult:
            broker_state["calls"] += 1
            if broker_state["block_first_fetch"]:
                broker_state["loop"] = asyncio.get_running_loop()
                broker_state["release"] = asyncio.Event()
                broker_state["fetch_started"].set()
                await broker_state["release"].wait()
            return fetch_result

    broker = OfflineReadOnlyClient()
    connection_accounts: list[str | None] = []
    cleanup_calls = 0

    async def connect_offline(
        saved_credentials: bool, account_id: str | None = None
    ) -> tuple[Any, Any]:
        assert saved_credentials is True
        connection_accounts.append(account_id)

        async def cleanup() -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

        return broker, cleanup

    monkeypatch.setattr(backfill_cmd, "_connect_live_client", connect_offline)

    confirmations: list[tuple[str, str]] = []
    dialogs: list[tuple[str, str]] = []

    def fake_askyesno(title: str, message: str) -> bool:
        confirmations.append((title, message))
        return True

    def fake_dialog(title: str, message: str) -> None:
        dialogs.append((title, message))

    monkeypatch.setattr(collection_center_panel.messagebox, "askyesno", fake_askyesno)
    monkeypatch.setattr(collection_center_panel.messagebox, "showerror", fake_dialog)
    monkeypatch.setattr(collection_center_panel.messagebox, "showwarning", fake_dialog)
    monkeypatch.setattr(collection_center_panel.messagebox, "showinfo", fake_dialog)

    roots: list[tk.Tk] = []

    def open_dashboard() -> tuple[tk.Tk, Dashboard, Any, list[Callable[[], None]]]:
        root = tk.Tk()
        root.withdraw()
        roots.append(root)
        queue: list[Callable[[], None]] = []

        def queued_after(_ms: int, callback: Callable[..., None], *args: Any) -> str:
            queue.append(lambda: callback(*args))
            return "test-after"

        root.after = queued_after  # type: ignore[method-assign]
        dashboard = Dashboard(root, tmp_path)
        root.update()
        return root, dashboard, dashboard.collection_center, queue

    root, dashboard, panel, after_queue = open_dashboard()
    try:
        assert len(panel._accounts_map) == 2
        panel.entry_start_utc.delete(0, "end")
        panel.entry_start_utc.insert(0, START_UTC)
        panel.entry_end_utc.delete(0, "end")
        panel.entry_end_utc.insert(0, END_UTC)

        second_label = next(
            label for label, account_id in panel._accounts_map.items() if account_id == aid2
        )
        panel.combo_account.set(second_label)

        panel._plan_collection()
        _pump_until(
            root,
            lambda: panel.current_job_id is not None and panel._active_operation is None,
            after_queue=after_queue,
        )
        job_id = panel.current_job_id
        assert job_id is not None

        panel._start_collection()
        _pump_until(root, broker_state["fetch_started"].is_set, after_queue=after_queue)
        panel._stop_collection()
        broker_state["loop"].call_soon_threadsafe(broker_state["release"].set)
        _pump_until(root, lambda: not panel._worker_thread.is_alive(), after_queue=after_queue)
        root.update()

        assert panel.lbl_status_badge.cget("text") == CollectionStatus.CANCELLED.value
        assert connection_accounts == [aid2]
        assert aid1 not in str(connection_accounts)
        assert "desktop-alpha@example.com" not in str(connection_accounts)
        assert "desktop-beta@example.com" not in str(connection_accounts)
        assert list((tmp_path / "data" / "collection_receipts").glob("rcpt_*.json")) == []
        assert list((tmp_path / "data" / "datasets").rglob("*.parquet")) == []

        assert confirmations and confirmations[0][0] == "Confirm Live Read-Only Collection"
        first_confirmation = confirmations[0][1]
        for expected in (job_id, aid2, START_UTC, END_UTC, "EURUSD", "60 seconds"):
            assert expected in first_confirmation

        # Reopen the dashboard and select the existing cancelled job.
        dashboard.close()
        root, dashboard, panel, after_queue = open_dashboard()
        job_labels = [label for label, value in panel._jobs_map.items() if value == job_id]
        assert job_labels
        panel.combo_jobs.set(job_labels[0])
        panel._on_job_selected()
        root.update()
        assert panel.current_job_id == job_id
        panel.combo_account.set(second_label)

        broker_state["block_first_fetch"] = False
        panel._resume_collection()
        _pump_until(root, lambda: not panel._worker_thread.is_alive(), after_queue=after_queue)
        root.update()

        assert panel.lbl_status_badge.cget("text") == CollectionStatus.COMPLETED_VERIFIED.value
        assert connection_accounts == [aid2, aid2]
        assert len(confirmations) >= 2
        assert "resume" in confirmations[1][1].lower()
        assert aid2 in confirmations[1][1]

        panel._verify_and_snapshot()
        _pump_until(root, lambda: panel._active_operation is None, after_queue=after_queue)
        root.update()
        assert any(title == "Snapshot Registered" for title, _ in dialogs)

        integrity_dialog = next(
            (msg for title, msg in confirmations if title == "Integrity Verified"), None
        )
        assert integrity_dialog is not None
        report = panel.coordinator.verify_job_integrity(job_id)
        assert report["unique_verified_candles"] == 10
        assert f"Unique Verified Candles: {report['unique_verified_candles']:,}" in integrity_dialog
        assert f"Receipt Candles: {report['receipt_candles_sum']:,}" in integrity_dialog
        assert f"Receipts: {report['receipt_count']:,}" in integrity_dialog
        assert "Total Candles: 0" not in integrity_dialog

        # OTC instruments stay visible for inspection/planning only.
        assert "EURUSD-OTC" in panel.combo_symbol.cget("values")
        assert "GBPUSD-OTC" in panel.combo_symbol.cget("values")
        panel.entry_start_utc.delete(0, "end")
        panel.entry_start_utc.insert(0, START_UTC)
        panel.entry_end_utc.delete(0, "end")
        panel.entry_end_utc.insert(0, END_UTC)
        panel.combo_symbol.set("EURUSD-OTC")
        panel._on_symbol_changed()
        assert "Unverified" in panel.lbl_instrument_type.cget("text")

        previous_job = panel.current_job_id
        panel._plan_collection()
        _pump_until(
            root,
            lambda: panel.current_job_id != previous_job and panel._active_operation is None,
            after_queue=after_queue,
        )
        otc_job = panel.current_job_id
        assert otc_job is not None
        assert panel.coordinator.get_progress(otc_job).symbol == "EURUSD_OTC"

        # Start must be blocked before any live-connection confirmation or broker access.
        connection_count = len(connection_accounts)
        confirmation_count = len(confirmations)
        dialog_count = len(dialogs)
        panel._start_collection()
        root.update()
        assert len(connection_accounts) == connection_count
        assert len(confirmations) == confirmation_count
        assert any(
            collection_center_panel.UNVERIFIED_OTC_MESSAGE in msg
            for _, msg in dialogs[dialog_count:]
        )

        # Resume must derive the symbol from the job checkpoint, not the visible selector.
        panel.coordinator.cancel_collection(otc_job)
        panel.combo_symbol.set("EURUSD")
        panel._on_symbol_changed()
        panel._refresh_jobs()
        otc_job_labels = [label for label, value in panel._jobs_map.items() if value == otc_job]
        assert otc_job_labels
        panel.combo_jobs.set(otc_job_labels[0])
        panel._on_job_selected()
        assert str(panel.btn_resume.cget("state")) == "normal"
        dialog_count = len(dialogs)
        panel._resume_collection()
        root.update()
        assert len(connection_accounts) == connection_count
        assert len(confirmations) == confirmation_count
        assert any(
            collection_center_panel.UNVERIFIED_OTC_MESSAGE in msg
            for _, msg in dialogs[dialog_count:]
        )

        # Even a receipt-verified OTC job may not produce a receipt-bound snapshot.
        snapshot_calls: list[str] = []
        verified_report = {
            "job_id": otc_job,
            "symbol": "EURUSD_OTC",
            "verified": True,
            "status": "COMPLETED_VERIFIED",
            "unique_verified_candles": 10,
            "receipt_candles_sum": 10,
            "receipt_count": 1,
        }
        monkeypatch.setattr(
            panel.coordinator, "verify_job_integrity", lambda _job_id: verified_report
        )
        monkeypatch.setattr(
            panel.coordinator,
            "create_snapshot_for_job",
            lambda job_id: snapshot_calls.append(job_id),
        )
        dialog_count = len(dialogs)
        panel._verify_and_snapshot()
        root.update()
        assert snapshot_calls == []
        assert len(confirmations) == confirmation_count
        assert any(
            collection_center_panel.UNVERIFIED_OTC_MESSAGE in msg
            for _, msg in dialogs[dialog_count:]
        )
    finally:
        for opened_root in roots:
            with contextlib.suppress(tk.TclError):
                opened_root.destroy()
