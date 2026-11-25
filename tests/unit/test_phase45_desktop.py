"""Unit tests for the Phase 45 Research desktop subtab."""

from __future__ import annotations

import contextlib
import json
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

import pytest

from iqoption_research.desktop import phase45_panel
from iqoption_research.desktop.laboratory_panel import ModelLaboratoryPanel
from iqoption_research.desktop.phase45_panel import Phase45ResearchPanel


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
        root.withdraw()
    except (tk.TclError, RuntimeError, OSError, Exception) as exc:  # noqa: BLE001
        pytest.skip(f"Tkinter/Tcl not available in environment: {exc}")
    yield root
    with contextlib.suppress(Exception):
        root.destroy()


@pytest.fixture
def dashboard(tmp_path: Path, tk_root: tk.Tk) -> SimpleNamespace:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phase45-worker")
    try:
        yield SimpleNamespace(root=tmp_path, window=tk_root, executor=executor)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _make_panel(
    dashboard: SimpleNamespace, tk_root: tk.Tk
) -> tuple[ttk.Notebook, Phase45ResearchPanel]:
    notebook = ttk.Notebook(tk_root)
    panel = Phase45ResearchPanel(dashboard, notebook)
    return notebook, panel


def _close_panel(notebook: ttk.Notebook, panel: Phase45ResearchPanel) -> None:
    if panel.active_future is not None:
        with contextlib.suppress(Exception):
            panel.active_future.result(timeout=5)
    panel.shutdown()
    with contextlib.suppress(Exception):
        notebook.destroy()


def _write_manifest(
    root: Path,
    *,
    manifest_id: str = "dev_m1_native_001",
    timeframe: str = "M1",
    status: str = "READY",
    **extra: object,
) -> None:
    directory = root / "reports" / "phase45" / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_id": manifest_id,
        "timeframe": timeframe,
        "status": status,
        "provenance_class": "development",
        **extra,
    }
    (directory / f"{manifest_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_ledger(root: Path, records: list[dict[str, object]]) -> None:
    directory = root / "reports" / "phase45"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment_ledger.json").write_text(
        json.dumps({"trial_records": records, "execution_authority": "NONE"}),
        encoding="utf-8",
    )


def _noop(*_args: object, **_kwargs: object) -> None:
    return None


def _progress(
    *,
    completed: int = 1,
    total: int = 2,
    model: str = "random_forest",
    records: tuple[dict[str, object], ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        completed_trials=completed,
        total_trials=total,
        active_model=model,
        elapsed_seconds=1.5,
        cancelled=False,
        message="trial completed",
        trial_records=records,
    )


def _fake_runner_factory(threads: list[str]) -> type:
    class FakeRunner:
        def __init__(self, repo_root: Path) -> None:
            self.repo_root = repo_root

        def run_suite(self, manifest_ids: list[str], **kwargs: object) -> SimpleNamespace:
            threads.append(threading.current_thread().name)
            progress_callback = kwargs["progress_callback"]
            progress_callback(_progress())
            return SimpleNamespace(
                to_dict=lambda: {
                    "disposition": "NO_DEMONSTRATED_EDGE",
                    "execution_authority": "NONE",
                    "trial_count": 1,
                    "resource_usage": {"elapsed_seconds": 1.5},
                    "message": "research finished",
                }
            )

    return FakeRunner


def test_laboratory_has_phase45_subtab(
    dashboard: SimpleNamespace, tk_root: tk.Tk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(phase45_panel.messagebox, "showinfo", _noop)
    notebook = ttk.Notebook(tk_root)
    lab = ModelLaboratoryPanel(dashboard, notebook)
    try:
        if lab.active_future is not None:
            lab.active_future.result(timeout=5)
        lab.tick()

        subtab_names = [
            lab.sub_notebook.tab(index, "text") for index in range(lab.sub_notebook.index("end"))
        ]
        assert "Phase 45 Research" in subtab_names

        panel = lab.phase45_panel
        assert notebook.tab(lab.frame, "text") == "Model Laboratory"
        assert lab.sub_notebook.tab(panel.frame, "text") == "Phase 45 Research"
        assert phase45_panel.SAFETY_LABEL in panel.lbl_safety["text"]
        assert "disabled" in panel.btn_start.state()
        assert "Refresh Artifacts" in panel.btn_refresh["text"]
        assert lab.busy() is False
    finally:
        lab.phase45_panel.shutdown()
        notebook.destroy()


def test_worker_progress_only_dispatches_to_ui_queue(
    dashboard: SimpleNamespace,
    tk_root: tk.Tk,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_manifest(dashboard.root)
    notebook, panel = _make_panel(dashboard, tk_root)
    after_threads: list[str] = []
    callback_threads: list[str] = []
    original_after = tk_root.after

    def spy_after(ms: int, callback: object = None) -> object:
        after_threads.append(threading.current_thread().name)
        return original_after(ms, callback)

    try:
        tk_root.after = spy_after  # type: ignore[method-assign]
        monkeypatch.setattr(
            phase45_panel,
            "_load_runner_class",
            lambda: _fake_runner_factory(callback_threads),
        )
        panel.refresh_artifacts()
        assert "disabled" not in panel.btn_start.state()

        panel.start_research()
        assert panel.active_future is not None
        panel.active_future.result(timeout=5)

        assert callback_threads
        assert all(name.startswith("phase45-worker") for name in callback_threads)
        assert not any(name.startswith("phase45-worker") for name in after_threads)
        assert panel._ui_queue.qsize() >= 2

        panel.tick()
        assert "Trials 1/2" in panel.lbl_progress["text"]
        assert "COMPLETE" in panel.lbl_progress["text"]
        assert "NO_DEMONSTRATED_EDGE" in panel.summary_text.get("1.0", "end")
        assert not panel.busy()
    finally:
        tk_root.after = original_after  # type: ignore[method-assign]
        _close_panel(notebook, panel)


def test_stale_generation_results_are_discarded(dashboard: SimpleNamespace, tk_root: tk.Tk) -> None:
    notebook, panel = _make_panel(dashboard, tk_root)
    try:
        before_progress = panel.lbl_progress["text"]
        before_summary = panel.summary_text.get("1.0", "end")

        panel._dispatch_to_ui(
            lambda: panel._on_progress(_progress(completed=99, total=99), generation=1)
        )
        panel._dispatch_to_ui(
            lambda: panel._on_complete(
                SimpleNamespace(to_dict=lambda: {"disposition": "STALE"}), generation=1
            )
        )
        panel.tick()

        assert panel.lbl_progress["text"] == before_progress
        assert panel.summary_text.get("1.0", "end") == before_summary
        assert "STALE" not in panel.summary_text.get("1.0", "end")
    finally:
        _close_panel(notebook, panel)


def test_cancel_preserves_loaded_trial_rows(dashboard: SimpleNamespace, tk_root: tk.Tk) -> None:
    _write_ledger(
        dashboard.root,
        [
            {"completed_at_utc": "2026-11-25T09:00:00Z", "active_model": "rf", "status": "done"},
            {"completed_at_utc": "2026-11-25T09:01:00Z", "active_model": "et", "status": "done"},
        ],
    )
    notebook, panel = _make_panel(dashboard, tk_root)
    try:
        assert len(panel.trials_tree.get_children()) == 2
        panel._operation_generation = 1
        panel._active_generation = 1

        panel.cancel_research()
        assert panel._cancel_requested is True
        assert "disabled" in panel.btn_cancel.state()

        panel.tick()
        assert "CANCELL" in panel.lbl_progress["text"]
        assert len(panel.trials_tree.get_children()) == 2
        assert "preserved" in panel.lbl_message["text"].lower()
    finally:
        _close_panel(notebook, panel)


def test_export_writes_only_sanitized_report(
    dashboard: SimpleNamespace,
    tk_root: tk.Tk,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_dir = dashboard.root / "reports" / "phase45"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "final_report.json").write_text(
        json.dumps({
            "disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
            "trial_count": 1,
            "resource_usage": {"elapsed_seconds": 1.5},
            "account_id": "ACC-123456",
            "wallet_id": "WALLET-9",
            "credentials": {"password": "do-not-export"},
            "raw_candles": [{"open": 1.0, "close": 1.1}],
            "predictions": [{"p": 0.51}],
        }),
        encoding="utf-8",
    )
    export_path = tmp_path / "sanitized_report.json"
    monkeypatch.setattr(
        phase45_panel.filedialog,
        "asksaveasfilename",
        lambda **_kwargs: str(export_path),
    )
    monkeypatch.setattr(phase45_panel.messagebox, "showinfo", _noop)
    monkeypatch.setattr(phase45_panel.messagebox, "showerror", _noop)

    notebook, panel = _make_panel(dashboard, tk_root)
    try:
        panel.export_report()
        exported = export_path.read_text(encoding="utf-8")

        assert "NO_DEMONSTRATED_EDGE" in exported
        assert '"execution_authority": "NONE"' in exported
        assert "trial_count" in exported
        for forbidden in (
            "account_id",
            "ACC-123456",
            "wallet_id",
            "credentials",
            "password",
            "raw_candles",
            "predictions",
            "do-not-export",
        ):
            assert forbidden not in exported
    finally:
        _close_panel(notebook, panel)


def test_export_markdown_is_sanitized(
    dashboard: SimpleNamespace,
    tk_root: tk.Tk,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_manifest(dashboard.root)
    notebook, panel = _make_panel(dashboard, tk_root)
    export_path = tmp_path / "sanitized_report.md"
    monkeypatch.setattr(
        phase45_panel.filedialog,
        "asksaveasfilename",
        lambda **_kwargs: str(export_path),
    )
    monkeypatch.setattr(phase45_panel.messagebox, "showinfo", _noop)
    try:
        panel._last_result = {
            "disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "LIVE",
            "account_number": "999",
            "raw_ticks": [1, 2, 3],
        }
        panel.export_report()
        exported = export_path.read_text(encoding="utf-8")
        assert "NO_DEMONSTRATED_EDGE" in exported
        assert "execution_authority**: NONE" in exported
        assert "account_number" not in exported
        assert "raw_ticks" not in exported
    finally:
        _close_panel(notebook, panel)


def test_no_manifests_keeps_start_disabled(dashboard: SimpleNamespace, tk_root: tk.Tk) -> None:
    notebook, panel = _make_panel(dashboard, tk_root)
    try:
        assert "disabled" in panel.btn_start.state()
        assert "INSPECTION ONLY" in panel.lbl_artifacts["text"]
        assert "disabled" not in panel.btn_refresh.state()

        panel.refresh_artifacts()
        assert "disabled" in panel.btn_start.state()
        assert "disabled" not in panel.btn_refresh.state()
    finally:
        _close_panel(notebook, panel)


def test_generated_manifest_list_file_is_loaded(dashboard: SimpleNamespace, tk_root: tk.Tk) -> None:
    directory = dashboard.root / "reports" / "phase45" / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "development_manifests.json").write_text(
        json.dumps([
            {
                "manifest_id": "gov_eurusd_60s_abc123def456",
                "timeframe_seconds": 60,
                "provenance_status": "PROTECTED_HISTORICALLY_REUSED",
                "execution_authority": "NONE",
            },
            {
                "manifest_id": "gov_eurusd_900s_xyz789abc123",
                "timeframe_seconds": 900,
                "provenance_status": "PROTECTED_HISTORICALLY_REUSED",
                "execution_authority": "NONE",
            },
        ]),
        encoding="utf-8",
    )
    notebook, panel = _make_panel(dashboard, tk_root)
    try:
        assert [manifest["manifest_id"] for manifest in panel._manifests] == [
            "gov_eurusd_60s_abc123def456",
            "gov_eurusd_900s_xyz789abc123",
        ]
        assert [manifest["timeframe"] for manifest in panel._manifests] == ["M1", "M15"]
        assert "PROTECTED_HISTORICALLY_REUSED" in [
            manifest["status"] for manifest in panel._manifests
        ]
        assert "disabled" not in panel.btn_start.state()
    finally:
        _close_panel(notebook, panel)


def test_protected_snapshot_manifest_is_never_listed(
    dashboard: SimpleNamespace, tk_root: tk.Tk
) -> None:
    _write_manifest(
        dashboard.root,
        manifest_id="protected_authoritative_snapshot_m1",
        timeframe="M1",
        protected_snapshot=True,
    )
    _write_manifest(dashboard.root, manifest_id="dev_m5_native_002", timeframe="M5")
    notebook, panel = _make_panel(dashboard, tk_root)
    try:
        ids = [manifest["manifest_id"] for manifest in panel._manifests]
        assert ids == ["dev_m5_native_002"]
        assert "disabled" not in panel.btn_start.state()
    finally:
        _close_panel(notebook, panel)
