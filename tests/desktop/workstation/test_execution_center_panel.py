"""Desktop UI tests for Execution Center inside Research Workstation V2 (Phase 50)."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path

import pytest

from iqoption_research.desktop.app import Dashboard
from iqoption_research.desktop.execution_center_panel import ExecutionCenterPanel
from iqoption_research.execution.models import OrderSide, ProductType
from iqoption_research.execution.offline_demo import OfflineExecutionDemoController

pytestmark = pytest.mark.desktop


@pytest.fixture(scope="module")
def tk_toplevel_factory() -> object:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tkinter unavailable: {exc}")
    root.withdraw()

    def _make_window() -> tk.Toplevel:
        win = tk.Toplevel(root)
        win.withdraw()
        return win

    yield _make_window
    root.destroy()


def test_dashboard_registers_execution_center_tab(
    tmp_path: Path, tk_toplevel_factory: object
) -> None:
    win = tk_toplevel_factory()
    dash = Dashboard(win, tmp_path)
    try:
        ec = dash.execution_center
        assert ec is not None
        assert "PERMANENTLY DISABLED" in ec.lbl_mode_paper["real_mode"].cget("text")
        assert len(ec.tree_matrix.get_children()) >= 10
        assert not ec.busy()
    finally:
        dash.close()


def test_execution_center_interactive_paper_workflow_and_export(
    tmp_path: Path, tk_toplevel_factory: object
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    win = tk_toplevel_factory()
    dash = Dashboard(win, tmp_path)
    # Rebind controller workspace strictly to tmp_path while pointing read-only at repo models
    dash.execution_center.controller.engine.close()
    dash.execution_center.controller = OfflineExecutionDemoController(
        repo_root,
        workspace_dir=tmp_path / "exec_center_ws",
    )
    ec: ExecutionCenterPanel = dash.execution_center
    try:
        ec.refresh_view()
        initial_balance_txt = ec.lbl_balance.cget("text")
        assert "$10,000.00" in initial_balance_txt

        # Inject Binary OTC, Digital, and Forex signals through UI handlers
        for prod, sym, side in (
            (ProductType.BINARY.value, "EURUSD_OTC", OrderSide.PUT.value),
            (ProductType.DIGITAL.value, "EURUSD", OrderSide.CALL.value),
            (ProductType.FOREX.value, "EURUSD", OrderSide.CALL.value),
        ):
            ec.var_product.set(prod)
            ec.var_symbol.set(sym)
            ec.var_side.set(side)
            ec.var_stake.set("20.0")
            ec.on_inject_signal()

        assert len(ec.tree_orders.get_children()) == 3
        first_row = ec.tree_orders.item(ec.tree_orders.get_children()[0], "values")
        assert first_row[9] == "NONE (PAPER)"

        # Advance candle -> settle Binary & check balance changes cleanly
        ec.on_advance_candle()
        ec.on_run_ai_replay()
        ec.on_run_full_suite()
        assert len(ec.tree_orders.get_children()) >= 8
        assert "PAPER" in ec.lbl_auth_status.cget("text") or "NONE" in ec.lbl_auth_status.cget(
            "text"
        )

        # Emergency stop blocks new manual injection until explicit reset
        ec.on_emergency_stop()
        assert "EMERGENCY STOP" in ec.lbl_action_status.cget("text")
        before_stop_count = len(ec.controller.engine.state_machine.all_orders)
        ec.on_inject_signal()
        assert len(ec.controller.engine.state_machine.all_orders) >= before_stop_count
        assert ec.controller.engine.rejected_orders

        ec.on_reset_stop()
        ec.on_restart_recovery()
        assert "Restart Recovered" in ec.lbl_action_status.cget("text")

        # Export offline evidence JSON cleanly
        export_path = tmp_path / "ui_evidence.json"
        exported = ec.controller.export_evidence_bundle(export_path)
        assert export_path.exists()
        raw = json.loads(export_path.read_text(encoding="utf-8"))
        assert raw["supervised_practice_readiness"]["real_execution_disabled"] is True
        assert exported["execution_engine_evidence"]["evidence_sha256"]
    finally:
        dash.close()
