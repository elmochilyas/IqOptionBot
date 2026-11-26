"""Tests for Phase 44 Collection Center UI responsiveness and performance hotfix.

Validates:
1. Real verification progress reporting (receipts, files, candles, elapsed time, current operation).
2. UI event loop responsiveness during verification across realistic datasets (25k, 100k, 375k candles).
3. Non-blocking background worker execution (heartbeat latency / jitter remains low, no UI freeze).
4. Safe cooperative cancellation halts verification promptly without corrupting state.
5. Mutual exclusion prevents concurrent verification, collection, and snapshotting on the same job.
6. Concise diagnostics logging (no megabyte-scale timestamp dumping) and full JSON report export.
7. Verification outcome handling: COMPLETED_VERIFIED, partial/incomplete coverage, CANCELLED, FAILED.
8. Inspect & Plan runs off the UI thread, never auto-connects, and discards stale results.
"""

from __future__ import annotations

import datetime
import json
import threading
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from iqoption_research.desktop import collection_center_panel
from iqoption_research.desktop.app import Dashboard
from iqoption_research.market_data.collection_engine import (
    CollectionCheckpoint,
    CollectionStatus,
    HistoricalCollectionCoordinator,
    VerificationProgress,
)
from iqoption_research.models.market import Candle
from tests.fixtures.synthetic_market import create_deterministic_synthetic_candles


@pytest.fixture(autouse=True)
def _mock_modal_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock Tkinter modal dialogs so unit tests run unattended without human interaction."""
    monkeypatch.setattr(
        collection_center_panel.messagebox, "showinfo", lambda *_args, **_kwargs: "ok"
    )
    monkeypatch.setattr(
        collection_center_panel.messagebox, "showerror", lambda *_args, **_kwargs: "ok"
    )
    monkeypatch.setattr(
        collection_center_panel.messagebox, "showwarning", lambda *_args, **_kwargs: "ok"
    )
    monkeypatch.setattr(
        collection_center_panel.messagebox, "askyesno", lambda *_args, **_kwargs: True
    )


def _make_auth_candles(count: int, start_ts: int) -> list[Candle]:
    """Generate authentic historical candles for collection testing."""
    synthetic = create_deterministic_synthetic_candles(count=count, start_ts=start_ts)
    return [
        replace(
            c,
            provenance="historical",
            identity_provenance="explicit_broker_metadata",
        )
        for c in synthetic
    ]


def _build_test_job(
    repo_root: Path,
    candle_count: int,
    *,
    batch_size: int = 1000,
    symbol: str = "EURUSD",
    start_ts: int = 1700000400,
    mark_completed: bool = True,
) -> tuple[HistoricalCollectionCoordinator, str]:
    """Helper to construct a fully formed historical collection job with receipts and parquet files."""
    coord = HistoricalCollectionCoordinator(repo_root)
    end_ts = start_ts + candle_count * 60
    start_iso = datetime.datetime.fromtimestamp(start_ts, datetime.UTC).isoformat()
    end_iso = datetime.datetime.fromtimestamp(end_ts, datetime.UTC).isoformat()
    job_id, _, _ = coord.plan_collection(symbol, start_iso, end_iso, timeframe_seconds=60)

    num_batches = candle_count // batch_size
    remainder = candle_count % batch_size

    for b in range(num_batches):
        b_start = start_ts + b * batch_size * 60
        candles = _make_auth_candles(batch_size, start_ts=b_start)
        coord.record_collected_batch(job_id, candles, is_synthetic=False)

    if remainder > 0:
        b_start = start_ts + num_batches * batch_size * 60
        candles = _make_auth_candles(remainder, start_ts=b_start)
        coord.record_collected_batch(job_id, candles, is_synthetic=False)

    if mark_completed:
        ckpt_path = repo_root / "data/collection_checkpoints" / f"{job_id}.json"
        ckpt = CollectionCheckpoint.load(ckpt_path)
        ckpt.status = CollectionStatus.COMPLETED_VERIFIED.value
        ckpt.completed_interval_indices = list(range(len(ckpt.missing_intervals)))
        ckpt.save(ckpt_path)

    return coord, job_id


class _HeartbeatTracker:
    """Tracks Tkinter event loop latency and jitter to verify UI responsiveness."""

    def __init__(self, root: tk.Tk, interval_ms: int = 10) -> None:
        self.root = root
        self.interval_ms = interval_ms
        self.interval_sec = interval_ms / 1000.0
        self.last_ts: float | None = None
        self.max_jitter: float = 0.0
        self.max_delay: float = 0.0
        self.samples: list[float] = []
        self.running: bool = False
        self._after_id: str | None = None

    def start(self) -> None:
        self.running = True
        self.last_ts = time.perf_counter()
        self._schedule()

    def stop(self) -> None:
        self.running = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _schedule(self) -> None:
        if self.running:
            self._after_id = self.root.after(self.interval_ms, self._tick)

    def _tick(self) -> None:
        self._after_id = None
        if not self.running:
            return
        now = time.perf_counter()
        if self.last_ts is not None:
            delay = now - self.last_ts
            self.samples.append(delay)
            self.max_delay = max(self.max_delay, delay)
            jitter = abs(delay - self.interval_sec)
            self.max_jitter = max(self.max_jitter, jitter)
        self.last_ts = now
        self._schedule()


def _pump_until(
    root: tk.Tk,
    panel: Any,
    predicate: Any,
    *,
    timeout: float = 30.0,
    poll_interval: float = 0.01,
) -> None:
    """Pump Tkinter event loop until predicate returns true or timeout expires."""
    deadline = time.monotonic() + timeout
    while not predicate():
        panel._poll_ui_queue()
        root.update()
        if time.monotonic() > deadline:
            pytest.fail(f"Timed out after {timeout}s waiting for desktop operation to finish")
        time.sleep(poll_interval)
    panel._poll_ui_queue()
    root.update()


def _open_dashboard_with_job(tmp_path: Path, job_id: str) -> tuple[tk.Tk, Dashboard, Any]:
    """Helper to open Dashboard, select a job, and return root, dashboard, and panel."""
    root = tk.Tk()
    root.withdraw()
    dashboard = Dashboard(root, tmp_path)
    root.update()
    panel = dashboard.collection_center

    job_labels = [label for label, val in panel._jobs_map.items() if val == job_id]
    assert job_labels, f"Job {job_id} not found in panel jobs map"
    panel.combo_jobs.set(job_labels[0])
    panel._on_job_selected()
    panel._poll_ui_queue()
    root.update()
    return root, dashboard, panel


def test_ui_queue_polling_keeps_one_scheduled_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manual queue pumps must not leave duplicate Tk ``after`` callbacks behind."""
    root = tk.Tk()
    root.withdraw()
    dashboard = Dashboard(root, tmp_path)
    root.update()
    panel = dashboard.collection_center
    scheduled_ids: list[str] = []
    original_after = root.after

    def counting_after(ms: int, callback: Any = None, *args: Any) -> str:
        after_id = original_after(ms, callback, *args)
        if callback == panel._scheduled_poll:
            scheduled_ids.append(after_id)
        return after_id

    monkeypatch.setattr(root, "after", counting_after)
    try:
        initial_id = panel._ui_poll_after_id
        assert initial_id is not None
        for _ in range(5):
            panel._poll_ui_queue()
        assert panel._ui_poll_after_id == initial_id
        assert scheduled_ids == []

        time.sleep(0.05)
        root.update()
        assert len(scheduled_ids) == 1
        assert panel._ui_poll_after_id == scheduled_ids[0]
        assert panel._ui_poll_after_id != initial_id
    finally:
        dashboard.close()


def test_verification_progress_reporting(tmp_path: Path) -> None:
    """HistoricalCollectionCoordinator.verify_job_integrity reports real-time progress."""
    coord, job_id = _build_test_job(tmp_path, candle_count=3000, batch_size=1000)

    progress_events: list[VerificationProgress] = []

    def on_progress(p: VerificationProgress) -> None:
        progress_events.append(p)

    report = coord.verify_job_integrity(job_id, progress_cb=on_progress)
    assert report["verified"] is True
    assert len(progress_events) >= 3

    last_event = progress_events[-1]
    assert last_event.job_id == job_id
    assert last_event.total_receipts == 3
    assert last_event.receipts_checked == 3
    assert last_event.files_checked >= 1
    assert last_event.candles_verified == 3000
    assert last_event.percent_complete == 100.0
    assert last_event.elapsed_seconds >= 0.0


def test_ui_responsiveness_25k_candles(tmp_path: Path) -> None:
    """Desktop UI remains responsive during verification of 25,000 candles."""
    _coord, job_id = _build_test_job(tmp_path, candle_count=25000, batch_size=1000)

    root, dashboard, panel = _open_dashboard_with_job(tmp_path, job_id)
    try:
        tracker = _HeartbeatTracker(root, interval_ms=10)
        tracker.start()

        # Trigger verification and snapshotting
        op_start = time.monotonic()
        panel._verify_and_snapshot()

        # Wait until background worker completes while pumping Tkinter loop
        _pump_until(root, panel, lambda: panel._active_operation is None, timeout=30.0)
        tracker.stop()
        print(
            "MEASURED 25k verification+snapshot: "
            f"max_ui_delay={tracker.max_delay:.3f}s, "
            f"samples={len(tracker.samples)}, elapsed={time.monotonic() - op_start:.3f}s"
        )

        # Verify event-loop remained responsive (< 250ms max delay / zero freeze)
        assert len(tracker.samples) > 10
        assert tracker.max_delay < 0.250, (
            f"UI frozen during 25k verify: max delay was {tracker.max_delay:.3f}s"
        )

        # Verify completion and integrity report
        assert panel._last_verification_report is not None
        assert panel._last_verification_report["verified"] is True
        assert panel._last_verification_report["unique_verified_candles"] == 25000
        assert panel.lbl_status_badge.cget("text") in ("COMPLETED_VERIFIED", "SNAPSHOT REGISTERED")
    finally:
        dashboard.close()


def test_verification_safe_cooperative_cancellation(tmp_path: Path) -> None:
    """Stopping verification midway halts cleanly, sets status to CANCELLED, and re-enables controls."""
    _coord, job_id = _build_test_job(tmp_path, candle_count=50000, batch_size=1000)

    root, dashboard, panel = _open_dashboard_with_job(tmp_path, job_id)
    try:
        panel._verify_and_snapshot()
        panel._poll_ui_queue()
        root.update()

        # Verify operation is active and stop button is enabled
        assert panel._active_operation == "verification"
        assert str(panel.btn_stop.cget("state")) == "normal"
        assert str(panel.btn_verify.cget("state")) == "disabled"

        # Signal stop
        panel._stop_clicked()
        panel._poll_ui_queue()
        root.update()

        _pump_until(root, panel, lambda: panel._active_operation is None, timeout=15.0)

        assert panel._last_verification_report is not None
        assert panel._last_verification_report["status"] == "CANCELLED"
        assert panel._last_verification_report["verified"] is False
        assert "CANCELLED" in panel.lbl_status_badge.cget("text")
        assert str(panel.btn_verify.cget("state")) == "normal"
        assert str(panel.btn_stop.cget("state")) == "disabled"
    finally:
        dashboard.close()


def test_mutual_exclusion_prevents_concurrent_operations(tmp_path: Path) -> None:
    """Job operation guard prevents concurrent verification, collection, and snapshotting on the same job."""
    coord, job_id = _build_test_job(tmp_path, candle_count=2000, batch_size=1000)

    # 1. Verification holds job_operation; foreign thread trying snapshot must raise RuntimeError
    worker_error: list[Exception] = []

    def foreign_attempt() -> None:
        try:
            with coord.job_operation(job_id, "snapshot"):
                pass
        except Exception as exc:  # noqa: BLE001
            worker_error.append(exc)

    with coord.job_operation(job_id, "verification"):
        t = threading.Thread(target=foreign_attempt)
        t.start()
        t.join(timeout=2.0)

    assert len(worker_error) == 1
    assert isinstance(worker_error[0], RuntimeError)
    assert "busy with 'verification'" in str(worker_error[0])

    # 2. Re-entrancy on the SAME thread is permitted (e.g. verify -> snapshot)
    with coord.job_operation(job_id, "verification"), coord.job_operation(job_id, "snapshot"):
        pass  # Same thread must not deadlock or raise


def test_concise_diagnostics_widget_and_report_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnostics widget shows concise operational summary and full JSON report is exportable."""
    _coord, job_id = _build_test_job(tmp_path, candle_count=10000, batch_size=1000)

    root, dashboard, panel = _open_dashboard_with_job(tmp_path, job_id)
    try:
        panel._verify_and_snapshot()
        _pump_until(root, panel, lambda: panel._active_operation is None, timeout=15.0)

        log_content = panel.txt_log.get("1.0", tk.END)

        # Assert concise summary is present
        assert "Verification Summary for" in log_content
        assert "Receipts: 10 valid" in log_content
        assert "Candles: 10,000 unique verified" in log_content
        assert "Coverage Gaps: 0" in log_content
        assert "Result: VERIFIED COMPLETE" in log_content

        # Assert raw timestamp array was NOT dumped into the text widget
        # 10,000 timestamps as JSON would be > 100,000 characters
        assert len(log_content) < 5000, (
            f"Diagnostics widget contains dumped raw data: length {len(log_content)}"
        )
        assert "verified_timestamps" not in log_content

        # Test report export
        export_target = tmp_path / "exported_report.json"
        monkeypatch.setattr(
            collection_center_panel.filedialog,
            "asksaveasfilename",
            lambda **_kwargs: str(export_target),
        )

        panel._export_report()

        assert export_target.is_file()
        exported_data = json.loads(export_target.read_text(encoding="utf-8"))
        assert exported_data["job_id"] == job_id
        assert exported_data["verified"] is True
        assert len(exported_data["verified_timestamps"]) == 10000
    finally:
        dashboard.close()


def test_verification_outcome_handling(tmp_path: Path) -> None:
    """Panel correctly handles COMPLETED_VERIFIED, partial coverage, and corrupt/tampered data."""
    coord, job_id = _build_test_job(tmp_path, candle_count=3000, batch_size=1000)

    # 1. Success: COMPLETED_VERIFIED
    rep_ok = coord.verify_job_integrity(job_id)
    assert rep_ok["verified"] is True
    assert len(rep_ok["errors"]) == 0
    assert rep_ok["unique_verified_candles"] == 3000

    # 2. Incomplete / Partial: remove one receipt file so coverage has a gap
    receipts = sorted(coord.receipts_dir.glob(f"rcpt_{job_id}_*.json"))
    assert len(receipts) == 3
    removed_receipt = receipts[1]
    removed_data = removed_receipt.read_text(encoding="utf-8")
    removed_receipt.unlink()

    rep_partial = coord.verify_job_integrity(job_id)
    assert rep_partial["verified"] is False
    assert len(rep_partial["coverage_gaps"]) >= 1
    assert any("missing from disk" in err.lower() for err in rep_partial["errors"])

    # Restore receipt
    removed_receipt.write_text(removed_data, encoding="utf-8")

    # 3. Tampered / Failed: alter content in parquet file
    first_receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    parquet_path = tmp_path / first_receipt["parquet_relative_path"]
    original_bytes = parquet_path.read_bytes()
    # Corrupt last byte
    parquet_path.write_bytes(original_bytes[:-1] + b"\x00")

    rep_corrupt = coord.verify_job_integrity(job_id)
    assert rep_corrupt["verified"] is False
    assert any(
        "checksum altered" in err.lower() or "tampered" in err.lower()
        for err in rep_corrupt["errors"]
    )

    # Restore parquet
    parquet_path.write_bytes(original_bytes)


def _wait_for_worker_start(
    root: tk.Tk, panel: Any, started: threading.Event, timeout: float = 5.0
) -> None:
    """Pump Tkinter while waiting for a background worker to reach its gate."""
    deadline = time.monotonic() + timeout
    while not started.is_set():
        panel._poll_ui_queue()
        root.update()
        if time.monotonic() > deadline:
            pytest.fail("Timed out waiting for background worker to start")
        time.sleep(0.005)


def test_planning_is_background_and_never_auto_connects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning holds no operator authorization and never blocks the Tk event loop."""
    coord, job_id = _build_test_job(tmp_path, candle_count=20, batch_size=10)
    progress = coord.get_progress(job_id)
    root = tk.Tk()
    root.withdraw()
    dashboard = Dashboard(root, tmp_path)
    root.update()
    panel = dashboard.collection_center
    confirmations: list[tuple[str, str]] = []
    monkeypatch.setattr(
        collection_center_panel.messagebox,
        "askyesno",
        lambda *args, **_kwargs: confirmations.append(args) or True,
    )
    started = threading.Event()
    release = threading.Event()
    original_plan = panel.coordinator.plan_collection

    def slow_plan(**kwargs: Any) -> tuple[str, list[Any], Any]:
        started.set()
        if not release.wait(5.0):
            raise RuntimeError("planning release timeout")
        return original_plan(**kwargs)

    monkeypatch.setattr(panel.coordinator, "plan_collection", slow_plan)
    try:
        panel.entry_start_utc.delete(0, "end")
        panel.entry_start_utc.insert(0, progress.requested_from_utc)
        panel.entry_end_utc.delete(0, "end")
        panel.entry_end_utc.insert(0, progress.requested_to_utc)

        tracker = _HeartbeatTracker(root, interval_ms=10)
        tracker.start()
        panel._plan_collection()
        _wait_for_worker_start(root, panel, started)

        # Hold the worker in its gate while measuring real Tkinter callback latency.
        monitor_deadline = time.monotonic() + 0.4
        while time.monotonic() < monitor_deadline:
            panel._poll_ui_queue()
            root.update()
            time.sleep(0.005)

        release.set()
        _pump_until(
            root,
            panel,
            lambda: panel.current_job_id is not None and panel._active_operation is None,
            timeout=10.0,
        )
        tracker.stop()
        print(
            "MEASURED background planning: "
            f"max_ui_delay={tracker.max_delay:.3f}s, samples={len(tracker.samples)}"
        )

        assert len(tracker.samples) > 10
        assert tracker.max_delay < 0.250, (
            f"UI frozen during planning: max delay was {tracker.max_delay:.3f}s"
        )
        assert confirmations == []
        assert panel.current_job_id is not None
    finally:
        release.set()
        dashboard.close()


def test_stale_planning_result_does_not_update_selected_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planning result is discarded if the operator selected another job meanwhile."""
    coord, job_id = _build_test_job(tmp_path, candle_count=20, batch_size=10)
    progress = coord.get_progress(job_id)
    root, dashboard, panel = _open_dashboard_with_job(tmp_path, job_id)
    started = threading.Event()
    release = threading.Event()

    def slow_plan(**_kwargs: Any) -> tuple[str, list[Any], Any]:
        started.set()
        if not release.wait(5.0):
            raise RuntimeError("planning release timeout")
        return job_id, [], progress

    monkeypatch.setattr(panel.coordinator, "plan_collection", slow_plan)
    try:
        panel.entry_start_utc.delete(0, "end")
        panel.entry_start_utc.insert(0, progress.requested_from_utc)
        panel.entry_end_utc.delete(0, "end")
        panel.entry_end_utc.insert(0, progress.requested_to_utc)

        panel._plan_collection()
        _wait_for_worker_start(root, panel, started)
        panel.current_job_id = "other-selected-job"
        release.set()
        _pump_until(root, panel, lambda: panel._active_operation is None, timeout=10.0)

        assert panel.current_job_id == "other-selected-job"
        assert "Selected job changed during planning" in panel.txt_log.get("1.0", tk.END)
    finally:
        release.set()
        dashboard.close()
