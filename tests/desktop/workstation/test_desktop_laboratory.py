"""Unit tests for Desktop Model Laboratory Panel."""

from __future__ import annotations

import contextlib
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import ttk
from unittest.mock import MagicMock

import pytest

from iqoption_research.desktop import laboratory_panel
from iqoption_research.desktop.laboratory_panel import ModelLaboratoryPanel


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
def mock_dashboard(tmp_path: Path, tk_root: tk.Tk) -> MagicMock:
    dashboard = MagicMock()
    dashboard.root = tmp_path
    dashboard.window = tk_root
    dashboard.executor = ThreadPoolExecutor(max_workers=1)
    return dashboard


def test_panel_initialization_and_tabs(mock_dashboard: MagicMock, tk_root: tk.Tk) -> None:
    notebook = ttk.Notebook(tk_root)
    panel = ModelLaboratoryPanel(mock_dashboard, notebook)

    # Frame added to notebook
    assert notebook.tab(panel.frame, "text") == "Model Laboratory"

    # Check subtabs
    subtab_names = [
        panel.sub_notebook.tab(i, "text") for i in range(panel.sub_notebook.index("end"))
    ]
    assert "Research Datasets" in subtab_names
    assert "Model Training" in subtab_names
    assert "Candidate Models & Evaluation" in subtab_names

    # Check buttons and states
    assert "Prepare Research Snapshot" in panel.btn_prepare_snapshot["text"]
    assert panel.btn_start_train["text"] == "Run Training Experiment"
    assert panel.btn_cancel_train["text"] == "Cancel"
    assert "disabled" in panel.btn_cancel_train.state()

    if panel.active_future is not None:
        panel.active_future.result(timeout=5)
    panel.tick()
    assert not panel.busy()


def test_panel_refresh_and_tick(mock_dashboard: MagicMock, tk_root: tk.Tk) -> None:
    notebook = ttk.Notebook(tk_root)
    panel = ModelLaboratoryPanel(mock_dashboard, notebook)

    # Trigger refresh
    panel.refresh()
    assert panel.busy()

    # Wait for background thread
    assert panel.active_future is not None
    panel.active_future.result(timeout=5)

    # Tick should process the completed future
    panel.tick()
    assert not panel.busy()
    assert "Discovered Sources" in panel.dataset_status_label["text"]


def test_model_selection_and_report_rendering(mock_dashboard: MagicMock, tk_root: tk.Tk) -> None:
    notebook = ttk.Notebook(tk_root)
    panel = ModelLaboratoryPanel(mock_dashboard, notebook)

    # Simulate having candidate models
    panel.candidate_models = [
        {
            "model_id": "candidate_rf_test",
            "model_family": "random_forest",
            "created_at_utc": "2026-11-21T10:00:00+00:00",
            "test_accuracy": 0.582,
            "brier_score": 0.241,
            "verdict": "NO_RELIABLE_EDGE_DETECTED",
            "path": str(mock_dashboard.root),
        }
    ]

    panel._update_models_tree(panel.candidate_models)
    items = panel.model_tree.get_children()
    assert len(items) == 1
    vals = panel.model_tree.item(items[0])["values"]
    assert vals[0] == "candidate_rf_test"
    assert vals[1] == "random_forest"
    assert "58.20%" in vals[4]


def test_multi_horizon_matrix_action_is_inspection_only(
    mock_dashboard: MagicMock,
    tk_root: tk.Tk,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dashboard button cannot run Phase 43 trials outside the dev-only script."""
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        laboratory_panel.messagebox,
        "showwarning",
        lambda title, message: warnings.append((title, message)),
    )

    notebook = ttk.Notebook(tk_root)
    panel = ModelLaboratoryPanel(mock_dashboard, notebook)
    panel.run_multi_horizon_matrix_action()

    assert len(warnings) == 1
    title, message = warnings[0]
    assert title == "Phase 43 Trials Inspection-Only"
    assert "run_phase43_experiments.py" in message
    assert "TEST_ONLY" in message
    assert "run_phase43_experiments.py" in panel.mh_log_text.get("1.0", "end")
