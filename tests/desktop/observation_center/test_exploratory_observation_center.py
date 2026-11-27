"""Phase 48 WS8 — Desktop Observation Center Multi-Model UI Tests."""

from __future__ import annotations

import time
import tkinter as tk
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tkinter import ttk
from unittest.mock import patch

from typer.testing import CliRunner

from iqoption_research.cli.main import app as cli_app
from iqoption_research.desktop.shadow_panel import ShadowEvaluationPanel
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.phase48.constants import CANDIDATE_SPECS, SUPPORTED_SESSION_DURATIONS_MINUTES
from iqoption_research.phase48.governance import (
    authorize_campaign_contract,
    build_default_proposed_campaign_contract,
    default_campaign_contract_path,
    save_campaign_contract,
)
from iqoption_research.phase48.transport import SyntheticMultiTimeframeTransport
from iqoption_research.security.credentials import Credentials
from tests.unit.governance.test_exploratory_governance_and_artifacts import (
    create_synthetic_frozen_bundle,
)


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


def test_phase48_observation_center_controls_and_genuine_block(
    tmp_path: Path, obs_tk_root: tk.Tk
) -> None:
    root_tk = obs_tk_root
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

        # Even if start_observation() is invoked directly while unauthorized, it never connects
        with (
            patch("iqoption_research.desktop.observation_center.messagebox.showwarning") as m_warn,
            patch("iqoption_research.desktop.observation_center.messagebox.showerror") as m_err,
        ):
            panel.start_observation()
            assert m_warn.called or m_err.called
            assert ctrl.is_running is False
    finally:
        panel.observation_center.shutdown()


def _make_ui_candles(tf_sec: int, count: int, start_dt: datetime) -> list[Candle]:
    out: list[Candle] = []
    price = 1.0900
    for i in range(count):
        o = price
        c = o + (0.0002 if i % 2 == 0 else -0.00015)
        ts = int((start_dt + timedelta(seconds=i * tf_sec)).timestamp())
        out.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize(tf_sec),
                from_ts=ts,
                to_ts=ts + tf_sec,
                at_ts=ts + tf_sec,
                open=round(o, 5),
                high=round(max(o, c) + 0.0001, 5),
                low=round(min(o, c) - 0.0001, 5),
                close=round(c, 5),
                bid=round(c, 5),
                ask=round(c + 0.0001, 5),
                volume=100,
                is_complete=True,
                provenance="streaming",
            )
        )
        price = c
    return out


def test_authorized_dashboard_genuine_dispatch_end_to_end(
    tmp_path: Path, obs_tk_root: tk.Tk
) -> None:
    models_dir = tmp_path / "data" / "phase48" / "models"
    m1 = create_synthetic_frozen_bundle(models_dir, 0)
    proposed = build_default_proposed_campaign_contract({m1.frozen_candidate_id: m1.bundle_sha256})
    authorized = authorize_campaign_contract(
        proposed,
        authorized_by="owner@local",
        allowed_account_ref="acct_practice_123",
        authorized_stage="STAGE_A_15M",
    )
    save_campaign_contract(authorized, default_campaign_contract_path(tmp_path))

    base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    warmup = {60: _make_ui_candles(60, 70, base_dt - timedelta(minutes=70))}
    events = [(60, c) for c in _make_ui_candles(60, 20, base_dt)]
    mock_transport = SyntheticMultiTimeframeTransport(
        warmup_by_tf=warmup,
        events=events,
        simulate_genuine_flag_for_test=True,
    )

    class _MockStore:
        def load(self, account_id: str) -> Credentials:
            assert account_id == "acct_practice_123"
            return Credentials(email="owner@example.com", password="secret")

    class _MockVault:
        def account_store(self) -> _MockStore:
            return _MockStore()

    root_tk = obs_tk_root
    root_tk.geometry("1280x720")
    try:
        nb = ttk.Notebook(root_tk)
        nb.pack(fill="both", expand=True)
        dash = _DummyDashboard(root_tk, tmp_path)
        panel = ShadowEvaluationPanel(dash, nb)  # type: ignore[arg-type]
        ctrl = panel.observation_center

        # Select only Candidate A, set vault account map, and switch to exploratory_genuine
        for cid, var in ctrl._p48_candidate_vars.items():
            var.set(cid == m1.frozen_candidate_id)
        panel._obs_accounts_map["Practice [acct...123]"] = "acct_practice_123"
        panel.combo_obs_account["values"] = ["Practice [acct...123]"]
        panel.combo_obs_account.set("Practice [acct...123]")
        panel.combo_obs_duration.set("15")
        panel.var_obs_mode.set("exploratory_genuine")
        ctrl.on_mode_changed()
        _PumpTk(root_tk)

        assert str(panel.btn_start_obs.cget("state")) == "normal"

        started = ctrl.start_genuine_exploratory_observation(
            transport_factory=lambda _creds, _tfs: mock_transport,
            vault_factory=_MockVault,
            confirm_override=True,
        )
        assert started is True

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _PumpTk(root_tk, steps=3)
            if ctrl.p48_pred_tree is not None and len(ctrl.p48_pred_tree.get_children()) >= 15:
                break
            time.sleep(0.02)
        ctrl.cancel_synthetic_observation()
        end_wait = time.monotonic() + 3.0
        while ctrl.is_running and time.monotonic() < end_wait:
            _PumpTk(root_tk, steps=3)
            time.sleep(0.02)
        _PumpTk(root_tk, steps=5)

        assert ctrl.is_running is False
        assert ctrl.p48_pred_tree is not None
        rows = [ctrl.p48_pred_tree.item(iid)["values"] for iid in ctrl.p48_pred_tree.get_children()]
        assert len(rows) > 0
        assert any(str(r[6]) in ("RESOLVED", "ABSTAIN") for r in rows)

        # Session history & export (patch modal showinfo)
        with patch("iqoption_research.desktop.observation_center.messagebox.showinfo"):
            assert len(panel.obs_history_tree.get_children()) >= 1
            sid = panel.obs_history_tree.get_children()[0]
            panel.obs_history_tree.selection_set(sid)
            ctrl.open_selected_diagnostics()
            exported = ctrl.export_selected_evidence(export_dir=tmp_path / "exports")
            assert exported is not None and exported.is_dir()
    finally:
        panel.observation_center.shutdown()


def test_phase48_cli_authorize_status_and_revoke(tmp_path: Path) -> None:
    models_dir = tmp_path / "data" / "phase48" / "models"
    for idx in range(3):
        create_synthetic_frozen_bundle(models_dir, idx)

    runner = CliRunner()
    res_auth = runner.invoke(
        cli_app,
        [
            "phase48",
            "authorize",
            "--account-id",
            "acct_cli_1",
            "--stage",
            "STAGE_A_15M",
            "--owner",
            "owner@cli",
            "--allow-unverified-vault",
            "--repo-root",
            str(tmp_path),
        ],
    )
    assert res_auth.exit_code == 0, res_auth.output
    assert "AUTHORIZED_ACTIVE" in res_auth.output

    res_stat = runner.invoke(cli_app, ["phase48", "status", "--repo-root", str(tmp_path)])
    assert res_stat.exit_code == 0
    assert "STAGE_A_15M" in res_stat.output

    res_rev = runner.invoke(
        cli_app,
        ["phase48", "revoke", "--owner", "owner@cli", "--repo-root", str(tmp_path)],
    )
    assert res_rev.exit_code == 0
    assert "REVOKED" in res_rev.output


def test_observation_center_layout_visibility_wrapping_and_failed_startup_display(
    tmp_path: Path, obs_tk_root: tk.Tk
) -> None:
    """Verify Refresh button stays visible, long text wraps, legacy controls hide, and errors show clearly."""
    models_dir = tmp_path / "data" / "phase48" / "models"
    digests = {}
    for idx in range(3):
        m = create_synthetic_frozen_bundle(models_dir, idx)
        digests[m.frozen_candidate_id] = m.bundle_sha256

    proposed = build_default_proposed_campaign_contract(digests)
    authorized = authorize_campaign_contract(
        proposed,
        authorized_by="owner@local",
        allowed_account_ref="acct_practice_123",
        authorized_stage="STAGE_A_15M",
    )
    save_campaign_contract(authorized, default_campaign_contract_path(tmp_path))

    class _FailingStartupTransport:
        is_connected = False
        is_synthetic = False
        last_heartbeat_mono = 0.0

        def last_heartbeat_for_tf(self, _tf: int) -> float:
            return 0.0

        async def connect(self) -> None:
            from iqoption_research.phase48.transport import ProvenanceError

            raise ProvenanceError(
                "Genuine exploratory pilot requires a verified PRACTICE account "
                "(status=account_selection_unverified)."
            )

        async def disconnect(self) -> None:
            self.is_connected = False

        async def get_warmup_candles(self, _tf: int, _count: int) -> tuple[list[Candle], dict]:
            return [], {}

        async def next_event(self, _timeout: float) -> None:
            return None

    class _MockStore:
        def load(self, _account_id: str) -> Credentials:
            return Credentials(email="owner@example.com", password="secret")

    class _MockVault:
        def account_store(self) -> _MockStore:
            return _MockStore()

    root_tk = obs_tk_root
    root_tk.geometry("1280x720")
    try:
        nb = ttk.Notebook(root_tk)
        nb.pack(fill="both", expand=True)
        dash = _DummyDashboard(root_tk, tmp_path)
        panel = ShadowEvaluationPanel(dash, nb)  # type: ignore[arg-type]
        ctrl = panel.observation_center

        panel._obs_accounts_map["Practice [acct...123]"] = "acct_practice_123"
        panel.combo_obs_account["values"] = ["Practice [acct...123]"]
        panel.combo_obs_account.set("Practice [acct...123]")
        panel.combo_obs_duration.set("15")
        _PumpTk(root_tk)

        # In synthetic mode, legacy Model / Fixture row and Sim Pace are visible
        assert ctrl._legacy_model_row is not None and ctrl._legacy_model_row.winfo_ismapped()
        assert ctrl._sim_pace_lbl is not None and ctrl._sim_pace_lbl.winfo_ismapped()
        assert panel.combo_obs_pace.winfo_ismapped()

        # Switch to genuine exploratory mode -> legacy Model / Fixture and Sim Pace are hidden
        panel.var_obs_mode.set("exploratory_genuine")
        ctrl.on_mode_changed()
        _PumpTk(root_tk)
        assert not ctrl._legacy_model_row.winfo_ismapped()
        assert not ctrl._sim_pace_lbl.winfo_ismapped()
        assert not panel.combo_obs_pace.winfo_ismapped()

        # Refresh Models & Contract button remains visible within auth_row bounds
        assert ctrl.btn_p48_refresh is not None and ctrl._auth_row is not None
        assert ctrl.btn_p48_refresh.winfo_ismapped()
        assert (
            ctrl.btn_p48_refresh.winfo_x() + ctrl.btn_p48_refresh.winfo_width()
            <= ctrl._auth_row.winfo_width() + 2
        )
        assert ctrl.lbl_p48_auth_status is not None
        assert int(ctrl.lbl_p48_auth_status.cget("wraplength")) > 0
        assert int(panel.lbl_obs_eligibility.cget("wraplength")) > 0

        # Execute startup failure -> status displays FAILED_STARTUP + error, auth shows EXHAUSTED (not model failure)
        started = ctrl.start_genuine_exploratory_observation(
            transport_factory=lambda _creds, _tfs: _FailingStartupTransport(),
            vault_factory=_MockVault,
            confirm_override=True,
        )
        assert started is True
        deadline = time.monotonic() + 5.0
        while ctrl.is_running and time.monotonic() < deadline:
            _PumpTk(root_tk, steps=3)
            time.sleep(0.02)
        _PumpTk(root_tk, steps=5)

        status_txt = str(panel.lbl_obs_status.cget("text"))
        assert "FAILED_STARTUP" in status_txt
        assert "PRACTICE account" in status_txt
        auth_txt = str(ctrl.lbl_p48_auth_status.cget("text"))
        assert "AUTHORIZATION EXHAUSTED" in auth_txt
        assert "0 Model Failures" in auth_txt
        assert ctrl.btn_p48_refresh.winfo_ismapped()

        # Tamper a model bundle -> UI clearly reports MODEL FAILURE rather than exhausted authorization
        first_cid = next(iter(digests))
        (models_dir / first_cid / "model.joblib").write_bytes(b"corrupted")
        ctrl._refresh_p48_governance_status()
        _PumpTk(root_tk)
        assert "MODEL FAILURE" in str(ctrl.lbl_p48_auth_status.cget("text"))
    finally:
        panel.observation_center.shutdown()
