"""Phase 48 WS8 — Desktop Observation Center Multi-Model UI Tests."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from iqoption_research.desktop.shadow_panel import ShadowEvaluationPanel
from iqoption_research.phase48.constants import CANDIDATE_SPECS, SUPPORTED_SESSION_DURATIONS_MINUTES


class _DummyDashboard:
    def __init__(self, root_tk: tk.Tk, repo_root: Path) -> None:
        self.window = root_tk
        self.root = repo_root
        self.events: list[str] = []

    def log(self, msg: str) -> None:
        self.events.append(msg)


def _PumpTk(root_tk: tk.Tk, steps: int = 10) -> None:
    for _ in range(steps):
        root_tk.update_idletasks()
        root_tk.update()


def test_phase48_observation_center_controls_and_genuine_block(tmp_path: Path) -> None:
    root_tk = tk.Tk()
    root_tk.geometry("1280x720")
    try:
        nb = ttk.Notebook(root_tk)
        nb.pack(fill="both", expand=True)
        dash = _DummyDashboard(root_tk, tmp_path)
        panel = ShadowEvaluationPanel(dash, nb)  # type: ignore[arg-type]
        _PumpTk(root_tk)

        ctrl = panel.observation_center
        assert list(panel.combo_obs_duration["values"]) == [
            str(d) for d in SUPPORTED_SESSION_DURATIONS_MINUTES
        ]
        assert len(ctrl._p48_candidate_vars) == len(CANDIDATE_SPECS)
        assert ctrl.p48_pred_tree is not None
        assert len(ctrl.p48_pred_tree["columns"]) == 10

        # Switch to Genuine Exploratory (Phase 48) -> must be fail-closed blocked
        panel.var_obs_mode.set("exploratory_genuine")
        ctrl.on_mode_changed()
        _PumpTk(root_tk)
        assert str(panel.btn_start_obs.cget("state")) == "disabled"
        assert "BLOCKED" in str(panel.lbl_obs_eligibility.cget("text"))
    finally:
        panel.observation_center.shutdown()
        root_tk.destroy()
