"""Phase 46 WS5 — Desktop Observation Center lifecycle, responsiveness & governance tests."""

from __future__ import annotations

import contextlib
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from unittest.mock import MagicMock

import pytest

from iqoption_research.desktop.shadow_panel import ShadowEvaluationPanel


def _make_mock_dashboard(root: Path, tk_root: tk.Tk) -> MagicMock:
    dash = MagicMock()
    dash.root = root
    dash.window = tk_root
    dash.process = MagicMock()
    dash.process.child = None
    dash.executor = MagicMock()
    return dash


def test_observation_center_genuine_mode_blocked_by_default(tmp_path: Path) -> None:
    tk_root = tk.Tk()
    try:
        dash = _make_mock_dashboard(tmp_path, tk_root)
        nb = ttk.Notebook(tk_root)
        panel = ShadowEvaluationPanel(dash, nb)
        oc = panel.observation_center

        # Default mode is synthetic and Start is enabled
        assert panel.var_obs_mode.get() == "synthetic"
        assert str(panel.btn_start_obs["state"]) == "normal"
        assert "SYNTHETIC_FIXTURE" in panel.lbl_obs_eligibility.cget("text")

        # Switching to genuine/live mode blocks activation
        panel.var_obs_mode.set("live")
        oc.on_mode_changed()
        assert str(panel.btn_start_obs["state"]) == "disabled"
        assert "BLOCKED" in panel.lbl_obs_eligibility.cget("text")
    finally:
        with contextlib.suppress(Exception):
            panel.shutdown()
            tk_root.destroy()


def test_observation_center_timeframe_sync_and_duration(tmp_path: Path) -> None:
    tk_root = tk.Tk()
    try:
        dash = _make_mock_dashboard(tmp_path, tk_root)
        nb = ttk.Notebook(tk_root)
        panel = ShadowEvaluationPanel(dash, nb)
        oc = panel.observation_center

        panel.combo_obs_timeframe.set("M5 (300s)")
        oc.on_timeframe_selected()
        assert "15 min (H3)" in panel.lbl_obs_horizon.cget("text")
        assert "M5 / H3" in panel.combo_obs_model.get()

        panel.combo_obs_timeframe.set("M15 (900s)")
        oc.on_timeframe_selected()
        assert "30 min (H2)" in panel.lbl_obs_horizon.cget("text")
        assert "M15 / H2" in panel.combo_obs_model.get()
    finally:
        with contextlib.suppress(Exception):
            panel.shutdown()
            tk_root.destroy()


def test_observation_center_full_synthetic_lifecycle_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *_a, **_kw: "ok")
    monkeypatch.setattr("tkinter.messagebox.showerror", lambda *_a, **_kw: "ok")
    tk_root = tk.Tk()
    try:
        dash = _make_mock_dashboard(tmp_path, tk_root)
        nb = ttk.Notebook(tk_root)
        panel = ShadowEvaluationPanel(dash, nb)
        oc = panel.observation_center

        panel.combo_obs_pace.set("Instant (0ms/bar)")
        panel.combo_obs_timeframe.set("M1 (60s)")
        oc.on_timeframe_selected()

        panel.start_observation()
        assert oc.is_running is True
        assert panel.busy() is True

        t0 = time.monotonic()
        while oc.is_running and (time.monotonic() - t0) < 20.0:
            tk_root.update()
            time.sleep(0.02)
        for _ in range(10):
            tk_root.update()
            time.sleep(0.01)

        assert oc.is_running is False
        status_text = panel.lbl_obs_status.cget("text")
        assert len(panel.obs_history_tree.get_children()) >= 1, f"status={status_text}"
        assert "Resolved:" in panel.lbl_obs_kpis.cget("text")

        # Stale generation rejection check
        prev_gen = oc._generation_id
        oc._generation_id += 1
        oc._on_worker_error(prev_gen, "stale error should be ignored")
        assert "stale error" not in panel.lbl_obs_status.cget("text")

        # Verify & Export sealed evidence
        oc.open_selected_diagnostics()
        oc.verify_selected_session()
        export_out = oc.export_selected_evidence(export_dir=tmp_path / "exports")
        assert export_out is not None
        assert (export_out / "ws6_synthetic_session_diagnostics.json").exists()
        assert (export_out / "ws6_synthetic_session_diagnostics.md").exists()
        assert (export_out / "verification_result.json").exists()
    finally:
        with contextlib.suppress(Exception):
            panel.shutdown()
            tk_root.destroy()
