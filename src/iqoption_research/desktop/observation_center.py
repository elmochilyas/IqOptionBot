"""Phase 46 WS5 — Functional Desktop Observation Center.

Provides an offline synthetic observation demonstration & genuine-blocked governance UI
integrated into ShadowEvaluationPanel, using a background worker thread and a
thread-safe UI queue (matching CollectionCenterPanel).

DEVELOPMENT_RESEARCH only. execution_authority=NONE. NO_DEMONSTRATED_EDGE.
"""

from __future__ import annotations

import contextlib
import json
import queue
import shutil
import threading
import time
import tkinter as tk
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Literal, cast

from iqoption_research.phase46.analysis import (
    compute_phase46_session_diagnostics,
    write_phase46_synthetic_diagnostics_reports,
)
from iqoption_research.phase46.observation.runtime import (
    Phase46SyntheticRunSpec,
    run_phase46_synthetic_observation,
)
from iqoption_research.phase46.observation.synthetic_fixtures import (
    PHASE46_SYNTHETIC_FIXTURES,
    get_synthetic_fixture,
    provision_synthetic_fixture_workspace,
)
from iqoption_research.phase48.artifacts import default_models_dir, verify_exploratory_bundle
from iqoption_research.phase48.constants import (
    CANDIDATE_SPECS,
    PHASE48_SESSIONS_DIR,
    SUPPORTED_SESSION_DURATIONS_MINUTES,
)
from iqoption_research.phase48.evidence import verify_phase48_session_integrity
from iqoption_research.phase48.governance import (
    CampaignUsageStore,
    default_campaign_contract_path,
    default_campaign_usage_path,
    get_stage_spec,
    load_campaign_contract,
    validate_campaign_authorization,
)
from iqoption_research.phase48.runtime import (
    ParallelObservationConfig,
    ParallelObservationProgress,
    ParallelObservationRuntime,
)
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    OutcomeStatus,
    ShadowPredictionJournal,
)
from iqoption_research.shadow.observer import (
    ObservationProgress,
    ProspectiveShadowObserver,
    verify_session_integrity,
)

if TYPE_CHECKING:
    from iqoption_research.desktop.shadow_panel import ShadowEvaluationPanel

PHASE46_SYNTH_SESSIONS_REL = "data/phase46/synthetic_sessions"
PHASE46_SYNTH_WORKSPACE_REL = "data/phase46/synthetic_workspace"
PHASE46_EXPORTS_REL = "data/phase46/exports"

_PACE_MAP: dict[str, float] = {
    "Interactive (120ms/bar)": 0.12,
    "Fast (25ms/bar)": 0.025,
    "Instant (0ms/bar)": 0.0,
}

_TF_DISPLAY_TO_SEC: dict[str, int] = {
    "M1 (60s)": 60,
    "M5 (300s)": 300,
    "M15 (900s)": 900,
}
_TF_SEC_TO_DISPLAY: dict[int, str] = {v: k for k, v in _TF_DISPLAY_TO_SEC.items()}


class ObservationCenterMixin:
    """Mixin/helper methods attached to ShadowEvaluationPanel for the Phase 46 Observation Center."""

    pass


class ObservationCenterController:
    """Manages the Phase 46 Observation Center UI, background worker, and session history."""

    def __init__(self, panel: ShadowEvaluationPanel) -> None:
        self.panel = panel
        self.root: Path = panel.root
        self.synth_root: Path = self.root / PHASE46_SYNTH_WORKSPACE_REL
        self._ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._poll_after_id: str | None = None
        self._worker_thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._observer_ref: list[ProspectiveShadowObserver] = []
        self._generation_id: int = 0
        self._is_running: bool = False
        self._is_destroyed: bool = False
        self._active_session_id: str | None = None
        self._active_session_dir: Path | None = None
        self._accuracy_history: list[float] = []
        self._sessions_index: dict[str, Path] = {}
        self._last_ui_push_mono: float = 0.0
        self._is_config_collapsed: bool = False
        self.obs_scroll_canvas: tk.Canvas | None = None
        self.obs_scroll_bar: ttk.Scrollbar | None = None
        self.obs_scroll_inner: ttk.Frame | None = None
        self._scroll_window_id: int | None = None
        self.btn_toggle_config: ttk.Button | None = None
        self.obs_config_body: ttk.Frame | None = None
        self.lbl_kpi_opps: ttk.Label | None = None
        self.lbl_kpi_preds: ttk.Label | None = None
        self.lbl_kpi_resolved: ttk.Label | None = None
        self.lbl_kpi_baseline: ttk.Label | None = None
        self._fixture_label_to_tf: dict[str, int] = {
            f"{fx.fixture_id} ({fx.timeframe_label} / H{fx.horizon_bars})": fx.timeframe_seconds
            for fx in PHASE46_SYNTHETIC_FIXTURES
        }
        # Phase 48 multi-model exploratory state
        self._p48_candidate_vars: dict[str, tk.BooleanVar] = {}
        self._p48_candidate_status_lbls: dict[str, ttk.Label] = {}
        self._p48_kpi_labels: dict[str, ttk.Label] = {}
        self._p48_multi_series: dict[str, list[float]] = {}
        self._p48_recent_rows: list[dict[str, Any]] = []
        self._p48_recent_outcomes: dict[str, dict[str, Any]] = {}
        self._p48_runtime: ParallelObservationRuntime | None = None
        self.p48_pred_tree: ttk.Treeview | None = None
        self.p48_filter_cand: ttk.Combobox | None = None
        self.p48_filter_status: ttk.Combobox | None = None
        self.lbl_p48_auth_status: ttk.Label | None = None
        self.rb_obs_p48_genuine: ttk.Radiobutton | None = None
        self.btn_p48_synth_demo: ttk.Button | None = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    def is_synthetic_fixture(self, label: str) -> bool:
        return label in self._fixture_label_to_tf

    # ------------------------------------------------------------------
    # Thread-safe UI queue
    # ------------------------------------------------------------------
    def _post_ui(self, fn: Callable[[], None]) -> None:
        if not self._is_destroyed:
            self._ui_queue.put(fn)

    def start_polling(self) -> None:
        if self._is_destroyed or self._poll_after_id is not None:
            return
        with contextlib.suppress(tk.TclError):
            self._poll_after_id = self.panel.dashboard.window.after(30, self._poll_queue)

    def _poll_queue(self) -> None:
        self._poll_after_id = None
        if self._is_destroyed:
            return
        processed = 0
        while processed < 50:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            with contextlib.suppress(Exception):
                fn()
            processed += 1
        if not self._is_destroyed:
            with contextlib.suppress(tk.TclError):
                self._poll_after_id = self.panel.dashboard.window.after(30, self._poll_queue)

    def shutdown(self) -> None:
        """Cooperatively cancel any running worker and stop polling timers."""
        self._is_destroyed = True
        self._cancel_event.set()
        if self._observer_ref:
            with contextlib.suppress(Exception):
                self._observer_ref[0].cancel()
        if self._p48_runtime is not None:
            with contextlib.suppress(Exception):
                self._p48_runtime.cancel()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if self._poll_after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.panel.dashboard.window.after_cancel(self._poll_after_id)
            self._poll_after_id = None

    # ------------------------------------------------------------------
    # Scroll & Collapsible Section Helpers
    # ------------------------------------------------------------------
    def _on_inner_configure(self, _event: Any = None) -> None:
        if self.obs_scroll_canvas is not None:
            self.obs_scroll_canvas.configure(scrollregion=self.obs_scroll_canvas.bbox("all"))

    def _on_canvas_configure(self, event: Any) -> None:
        if self.obs_scroll_canvas is not None and self._scroll_window_id is not None:
            self.obs_scroll_canvas.itemconfigure(self._scroll_window_id, width=max(1, event.width))

    def _on_mousewheel(self, event: Any) -> None:
        if self.obs_scroll_canvas is not None and event.delta:
            self.obs_scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def toggle_config_section(self) -> None:
        if self.obs_config_body is None or self.btn_toggle_config is None:
            return
        self._is_config_collapsed = not self._is_config_collapsed
        if self._is_config_collapsed:
            self.obs_config_body.pack_forget()
            self.btn_toggle_config.config(text="▸ Expand Session Configuration")
        else:
            self.obs_config_body.pack(fill="x", pady=(4, 0))
            self.btn_toggle_config.config(text="▾ Collapse Session Configuration")
        self._on_inner_configure()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def build_ui(self, parent: ttk.Frame) -> None:
        p = self.panel

        # Scrollable outer container so tables, chart & history never collapse at 1280x720
        scroll_host = ttk.Frame(parent)
        scroll_host.pack(fill="both", expand=True)
        scroll_host.rowconfigure(0, weight=1)
        scroll_host.columnconfigure(0, weight=1)

        self.obs_scroll_canvas = tk.Canvas(
            scroll_host, highlightthickness=0, borderwidth=0, background="#eef2f6"
        )
        self.obs_scroll_bar = ttk.Scrollbar(
            scroll_host, orient="vertical", command=self.obs_scroll_canvas.yview
        )
        self.obs_scroll_canvas.configure(yscrollcommand=self.obs_scroll_bar.set)
        self.obs_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self.obs_scroll_bar.grid(row=0, column=1, sticky="ns")

        self.obs_scroll_inner = ttk.Frame(self.obs_scroll_canvas, padding=(2, 2, 6, 6))
        self._scroll_window_id = self.obs_scroll_canvas.create_window(
            (0, 0), window=self.obs_scroll_inner, anchor="nw"
        )
        self.obs_scroll_inner.bind("<Configure>", self._on_inner_configure)
        self.obs_scroll_canvas.bind("<Configure>", self._on_canvas_configure)
        self.obs_scroll_canvas.bind(
            "<Enter>",
            lambda _e: (
                self.obs_scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
                if self.obs_scroll_canvas
                else None
            ),
        )
        self.obs_scroll_canvas.bind(
            "<Leave>",
            lambda _e: (
                self.obs_scroll_canvas.unbind_all("<MouseWheel>")
                if self.obs_scroll_canvas
                else None
            ),
        )

        container = self.obs_scroll_inner

        # ==================================================================
        # 1) Session Configuration (Collapsible)
        # ==================================================================
        controls_frame = ttk.LabelFrame(
            container, text="1. Observation Session Configuration", padding=6
        )
        controls_frame.pack(fill="x", pady=(0, 8))

        cfg_header = ttk.Frame(controls_frame)
        cfg_header.pack(fill="x")
        p.obs_safety_label = ttk.Label(
            cfg_header,
            text=(
                "Phase 46 Observation Center: EXECUTION_AUTHORITY=NONE | NO_DEMONSTRATED_EDGE. "
                "Offline Synthetic Mode validates lifecycle plumbing only; Genuine Mode is BLOCKED."
            ),
            font=("Segoe UI", 8, "bold"),
            foreground="#9c3414",
            wraplength=820,
            justify="left",
        )
        p.obs_safety_label.pack(side="left", fill="x", expand=True)

        self.btn_toggle_config = ttk.Button(
            cfg_header,
            text="▾ Collapse Session Configuration",
            command=self.toggle_config_section,
        )
        self.btn_toggle_config.pack(side="right", padx=(8, 0))

        self.obs_config_body = ttk.Frame(controls_frame)
        self.obs_config_body.pack(fill="x", pady=(4, 0))

        # Row 1: Mode & Eligibility
        r1 = ttk.Frame(self.obs_config_body)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Observation Mode:", width=18).pack(side="left")
        p.var_obs_mode = tk.StringVar(value="synthetic")
        p.rb_obs_synth = ttk.Radiobutton(
            r1,
            text="Offline Synthetic Demonstration",
            variable=p.var_obs_mode,
            value="synthetic",
            command=self.on_mode_changed,
        )
        p.rb_obs_synth.pack(side="left", padx=4)
        self.rb_obs_p48_genuine = ttk.Radiobutton(
            r1,
            text="Genuine Exploratory (Phase 48)",
            variable=p.var_obs_mode,
            value="exploratory_genuine",
            command=self.on_mode_changed,
        )
        self.rb_obs_p48_genuine.pack(side="left", padx=6)
        p.rb_obs_live = ttk.Radiobutton(
            r1,
            text="Confirmatory (Blocked)",
            variable=p.var_obs_mode,
            value="live",
            command=self.on_mode_changed,
        )
        p.rb_obs_live.pack(side="left", padx=6)

        p.lbl_obs_eligibility = ttk.Label(
            r1,
            text="SYNTHETIC_FIXTURE (Lifecycle Demo Only)",
            font=("Segoe UI", 9, "bold"),
            foreground="#1b4d3e",
        )
        p.lbl_obs_eligibility.pack(side="right", padx=4)

        # Row 2: Model, Timeframe, Prediction Duration
        r2 = ttk.Frame(self.obs_config_body)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Model / Fixture:", width=18).pack(side="left")
        p.combo_obs_model = ttk.Combobox(r2, state="readonly", width=40)
        p.combo_obs_model.pack(side="left", padx=4)
        p.combo_obs_model.bind("<<ComboboxSelected>>", self.on_model_selected)

        ttk.Label(r2, text="Timeframe:", width=10).pack(side="left", padx=(8, 2))
        p.combo_obs_timeframe = ttk.Combobox(
            r2, state="readonly", width=11, values=list(_TF_DISPLAY_TO_SEC.keys())
        )
        p.combo_obs_timeframe.set("M1 (60s)")
        p.combo_obs_timeframe.pack(side="left", padx=4)
        p.combo_obs_timeframe.bind("<<ComboboxSelected>>", self.on_timeframe_selected)

        p.lbl_obs_horizon = ttk.Label(
            r2,
            text="Prediction Duration: 1 min (H1)",
            font=("Segoe UI", 9, "bold"),
            foreground="#20364a",
        )
        p.lbl_obs_horizon.pack(side="left", padx=(10, 4))

        # Row 3: Account (Genuine), Duration, Pace
        r3 = ttk.Frame(self.obs_config_body)
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Vault Account:", width=18).pack(side="left")
        p.combo_obs_account = ttk.Combobox(r3, state="disabled", width=26)
        p.combo_obs_account.pack(side="left", padx=4)

        ttk.Label(r3, text="Duration (min):", width=13).pack(side="left", padx=(8, 2))
        p.combo_obs_duration = ttk.Combobox(
            r3,
            state="readonly",
            width=6,
            values=[str(d) for d in SUPPORTED_SESSION_DURATIONS_MINUTES],
        )
        p.combo_obs_duration.set("15")
        p.combo_obs_duration.pack(side="left", padx=4)

        ttk.Label(r3, text="Sim Pace:", width=9).pack(side="left", padx=(8, 2))
        p.combo_obs_pace = ttk.Combobox(
            r3, state="readonly", width=21, values=list(_PACE_MAP.keys())
        )
        p.combo_obs_pace.set("Interactive (120ms/bar)")
        p.combo_obs_pace.pack(side="left", padx=4)

        # Row 3b: Campaign & Slot (Phase 40B/40C compatibility)
        r3b = ttk.Frame(self.obs_config_body)
        r3b.pack(fill="x", pady=2)
        ttk.Label(r3b, text="Research Campaign:", width=18).pack(side="left")
        p.combo_obs_campaign = ttk.Combobox(r3b, state="readonly", width=22)
        p.combo_obs_campaign.pack(side="left", padx=4)
        p.combo_obs_campaign.bind(
            "<<ComboboxSelected>>",
            lambda e: p._on_campaign_selected(e),  # noqa: SLF001
        )
        ttk.Label(r3b, text="Slot:", width=5).pack(side="left", padx=(8, 2))
        p.combo_obs_slot = ttk.Combobox(r3b, state="readonly", width=16)
        p.combo_obs_slot.pack(side="left", padx=4)
        p.lbl_campaign_blinding = ttk.Label(
            r3b, text="Ad-hoc (Unblinded)", font=("Segoe UI", 9, "italic"), foreground="#476582"
        )
        p.lbl_campaign_blinding.pack(side="left", padx=(10, 4))

        # Row 3c: Phase 48 Exploratory Multi-Model Candidates & Authorization Gate
        p48_cfg = ttk.LabelFrame(
            self.obs_config_body,
            text="Phase 48 Exploratory Candidates (EXPLORATORY_FROZEN_RESEARCH_CANDIDATE | NO_DEMONSTRATED_EDGE)",
            padding=4,
        )
        p48_cfg.pack(fill="x", pady=(4, 2))
        for spec in CANDIDATE_SPECS:
            row_c = ttk.Frame(p48_cfg)
            row_c.pack(fill="x", pady=1)
            var = tk.BooleanVar(value=True)
            self._p48_candidate_vars[spec.frozen_candidate_id] = var
            cb = ttk.Checkbutton(
                row_c,
                text=(
                    f"[{spec.candidate_slot}] {spec.frozen_candidate_id} "
                    f"({spec.timeframe_label} / H{spec.horizon_bars}={spec.prediction_minutes}m)"
                ),
                variable=var,
                command=self._refresh_p48_governance_status,
            )
            cb.pack(side="left")
            st_lbl = ttk.Label(
                row_c,
                text="Checking artifact...",
                font=("Consolas", 8),
                foreground="#476582",
            )
            st_lbl.pack(side="right", padx=4)
            self._p48_candidate_status_lbls[spec.frozen_candidate_id] = st_lbl

        auth_row = ttk.Frame(p48_cfg)
        auth_row.pack(fill="x", pady=(3, 1))
        self.lbl_p48_auth_status = ttk.Label(
            auth_row,
            text="Campaign Gate: Checking Phase 48 authorization...",
            font=("Segoe UI", 8, "bold"),
            foreground="#9c3414",
        )
        self.lbl_p48_auth_status.pack(side="left")
        ttk.Button(
            auth_row,
            text="Refresh Models & Contract",
            command=self._refresh_p48_governance_status,
        ).pack(side="right", padx=4)

        # Row 4: Action Buttons (always visible even if config body collapsed)
        r4 = ttk.Frame(controls_frame)
        r4.pack(fill="x", pady=(6, 2))
        p.btn_start_obs = ttk.Button(r4, text="Start Observation", command=p.start_observation)
        p.btn_start_obs.pack(side="left", padx=(0, 6))

        p.btn_cancel_obs = ttk.Button(
            r4, text="Stop / Cancel", command=p.cancel_observation, state="disabled"
        )
        p.btn_cancel_obs.pack(side="left", padx=4)

        p.btn_replay_evidence = ttk.Button(
            r4,
            text="Replay Session Evidence",
            command=lambda: p._prompt_replay_evidence(),  # noqa: SLF001
        )
        p.btn_replay_evidence.pack(side="left", padx=(12, 4))

        p.btn_verify_inputs = ttk.Button(
            r4,
            text="Verify Session Inputs",
            command=lambda: p._prompt_verify_inputs(),  # noqa: SLF001
        )
        p.btn_verify_inputs.pack(side="left", padx=4)

        self.btn_p48_synth_demo = ttk.Button(
            r4,
            text="Run Phase 48 Multi-Model Demo (Offline)",
            command=self.start_phase48_synthetic_demo,
        )
        self.btn_p48_synth_demo.pack(side="left", padx=(8, 4))

        # ==================================================================
        # 2) Live Monitoring & Performance
        # ==================================================================
        live_frame = ttk.LabelFrame(container, text="2. Live Monitoring & Performance", padding=6)
        live_frame.pack(fill="x", pady=(0, 8))

        p.obs_progress_bar = ttk.Progressbar(live_frame, mode="determinate")
        p.obs_progress_bar.pack(fill="x", pady=(0, 4))

        p.lbl_obs_status = ttk.Label(
            live_frame,
            text="State: IDLE | Source: Disconnected | Elapsed: 0.0s | Latest Candle: N/A | Latest Pred: N/A",
            font=("Segoe UI", 9, "bold"),
            foreground="#20364a",
            wraplength=980,
        )
        p.lbl_obs_status.pack(anchor="w", pady=(0, 4))

        # Compact 4-card KPI grid
        kpi_grid = ttk.Frame(live_frame)
        kpi_grid.pack(fill="x", pady=(0, 4))
        for ci in range(4):
            kpi_grid.columnconfigure(ci, weight=1, uniform="kpi")

        def _make_kpi_card(col_idx: int, title: str, initial: str) -> ttk.Label:
            card = ttk.LabelFrame(kpi_grid, text=title, padding=(6, 4))
            card.grid(row=0, column=col_idx, sticky="nsew", padx=(0 if col_idx == 0 else 4, 0))
            lbl = ttk.Label(card, text=initial, font=("Consolas", 9), foreground="#17324d")
            lbl.pack(anchor="w")
            return lbl

        self.lbl_kpi_opps = _make_kpi_card(
            0, "Opportunities", "Total: 0 | Sel: 0\nCov: N/A | Abst: 0"
        )
        self.lbl_kpi_preds = _make_kpi_card(1, "Pending & Resolved", "Pending: 0\nResolved: 0")
        self.lbl_kpi_resolved = _make_kpi_card(2, "Accuracy (W/L/T)", "W:0 / L:0 / T:0\nAcc: N/A")
        self.lbl_kpi_baseline = _make_kpi_card(
            3, "Baseline & 85% Hyp.", "Baseline: N/A\nExp/BE: N/A"
        )

        p.lbl_obs_kpis = ttk.Label(
            live_frame,
            text=(
                "Opportunities: 0 | Selected: 0 | Pending: 0 | Resolved: 0 (W:0 / L:0 / T:0) | "
                "Abstentions: 0 | Accuracy: N/A | Coverage: N/A | Baseline: N/A"
            ),
            font=("Consolas", 8),
            foreground="#476582",
        )
        p.lbl_obs_kpis.pack(anchor="w", pady=(2, 1))

        p.lbl_obs_payout = ttk.Label(
            live_frame,
            text="Hypothetical 85% Payout Sensitivity (NOT actual broker settlement): N/A",
            font=("Segoe UI", 8, "italic"),
            foreground="#5a6872",
        )
        p.lbl_obs_payout.pack(anchor="w", pady=(0, 6))

        # Prediction Table (left, with V+H scrollbars) + Cumulative Accuracy Chart (right, min height 170px)
        mid_frame = ttk.Frame(live_frame)
        mid_frame.pack(fill="both", expand=True)
        mid_frame.columnconfigure(0, weight=3)
        mid_frame.columnconfigure(1, weight=2)
        mid_frame.rowconfigure(0, weight=1, minsize=175)

        tree_box = ttk.Frame(mid_frame)
        tree_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tree_box.rowconfigure(0, weight=1)
        tree_box.columnconfigure(0, weight=1)

        cols = (
            "model_id",
            "predictions",
            "pending",
            "trades",
            "win_rate",
            "matched_maj",
            "ci_95",
            "exp_bin",
            "exp_dig",
            "be_gap",
        )
        p.obs_tree = ttk.Treeview(tree_box, columns=cols, show="headings", height=5)
        headers = {
            "model_id": ("Model ID", 190, "w"),
            "predictions": ("Total", 55, "e"),
            "pending": ("Pending", 60, "e"),
            "trades": ("W/L/T", 80, "center"),
            "win_rate": ("Win%", 70, "e"),
            "matched_maj": ("Baseline%", 75, "e"),
            "ci_95": ("Wilson 95%", 120, "center"),
            "exp_bin": ("Exp Bin", 75, "e"),
            "exp_dig": ("Exp Dig", 75, "e"),
            "be_gap": ("BE Gap", 70, "e"),
        }
        for col, (title, w, anc) in headers.items():
            p.obs_tree.heading(col, text=title)
            p.obs_tree.column(
                col,
                width=w,
                minwidth=max(50, w - 20),
                stretch=False,
                anchor=cast(Literal["w", "e", "center"], anc),
            )

        obs_vsb = ttk.Scrollbar(tree_box, orient="vertical", command=p.obs_tree.yview)
        obs_hsb = ttk.Scrollbar(tree_box, orient="horizontal", command=p.obs_tree.xview)
        p.obs_tree.configure(yscrollcommand=obs_vsb.set, xscrollcommand=obs_hsb.set)
        p.obs_tree.grid(row=0, column=0, sticky="nsew")
        obs_vsb.grid(row=0, column=1, sticky="ns")
        obs_hsb.grid(row=1, column=0, sticky="ew")

        chart_box = ttk.Frame(mid_frame)
        chart_box.grid(row=0, column=1, sticky="nsew")
        chart_box.rowconfigure(0, weight=1, minsize=175)
        chart_box.columnconfigure(0, weight=1)
        p.obs_chart_canvas = tk.Canvas(
            chart_box,
            width=360,
            height=175,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#cfd8e3",
        )
        p.obs_chart_canvas.grid(row=0, column=0, sticky="nsew")
        p.obs_chart_canvas.bind(
            "<Configure>", lambda _e: self._draw_accuracy_chart(self._accuracy_history)
        )
        self._draw_accuracy_chart([])

        # ==================================================================
        # 2b) Phase 48 Parallel Multi-Model Exploratory Evaluation
        # ==================================================================
        p48_live = ttk.LabelFrame(
            container,
            text="2b. Phase 48 Parallel Multi-Model Exploratory Cards & Live Prediction Stream",
            padding=6,
        )
        p48_live.pack(fill="x", pady=(0, 8))

        p48_cards = ttk.Frame(p48_live)
        p48_cards.pack(fill="x", pady=(0, 4))
        for idx_c, spec in enumerate(CANDIDATE_SPECS):
            p48_cards.columnconfigure(idx_c, weight=1, uniform="p48kpi")
            cframe = ttk.LabelFrame(
                p48_cards,
                text=f"[{spec.candidate_slot}] {spec.timeframe_label} H{spec.horizon_bars} ({spec.prediction_minutes}m)",
                padding=(6, 4),
            )
            cframe.grid(row=0, column=idx_c, sticky="nsew", padx=(0 if idx_c == 0 else 4, 0))
            klbl = ttk.Label(
                cframe,
                text="Warmup: 0/60 | Opps: 0 | Sel: 0\nPending: 0 | Res: 0 (W:0/L:0) | Acc: N/A",
                font=("Consolas", 8),
                foreground="#17324d",
            )
            klbl.pack(anchor="w")
            self._p48_kpi_labels[spec.frozen_candidate_id] = klbl

        flt_row = ttk.Frame(p48_live)
        flt_row.pack(fill="x", pady=(2, 2))
        ttk.Label(flt_row, text="Filter Candidate:").pack(side="left")
        self.p48_filter_cand = ttk.Combobox(
            flt_row,
            state="readonly",
            width=32,
            values=["ALL"] + [s.frozen_candidate_id for s in CANDIDATE_SPECS],
        )
        self.p48_filter_cand.set("ALL")
        self.p48_filter_cand.pack(side="left", padx=4)
        self.p48_filter_cand.bind("<<ComboboxSelected>>", lambda _e: self._render_p48_predictions())

        ttk.Label(flt_row, text="Filter Status:").pack(side="left", padx=(8, 2))
        self.p48_filter_status = ttk.Combobox(
            flt_row,
            state="readonly",
            width=14,
            values=["ALL", "PENDING", "RESOLVED", "ABSTAIN"],
        )
        self.p48_filter_status.set("ALL")
        self.p48_filter_status.pack(side="left", padx=4)
        self.p48_filter_status.bind(
            "<<ComboboxSelected>>", lambda _e: self._render_p48_predictions()
        )

        p48_tree_box = ttk.Frame(p48_live)
        p48_tree_box.pack(fill="x", pady=(2, 0))
        p48_tree_box.rowconfigure(0, weight=1)
        p48_tree_box.columnconfigure(0, weight=1)
        p48_cols = (
            "decision_time",
            "candidate",
            "timeframe",
            "horizon",
            "direction",
            "cal_prob",
            "status",
            "actual_dir",
            "outcome",
            "maturity_time",
        )
        self.p48_pred_tree = ttk.Treeview(p48_tree_box, columns=p48_cols, show="headings", height=5)
        p48_headers = {
            "decision_time": ("Decision Time", 145, "w"),
            "candidate": ("Candidate", 200, "w"),
            "timeframe": ("TF", 50, "center"),
            "horizon": ("Horizon", 70, "center"),
            "direction": ("Direction", 75, "center"),
            "cal_prob": ("Cal Prob", 75, "e"),
            "status": ("Status", 90, "center"),
            "actual_dir": ("Actual Dir", 80, "center"),
            "outcome": ("Outcome", 85, "center"),
            "maturity_time": ("Maturity Time", 145, "w"),
        }
        for col, (title, w, anc) in p48_headers.items():
            self.p48_pred_tree.heading(col, text=title)
            self.p48_pred_tree.column(
                col,
                width=w,
                minwidth=max(45, w - 15),
                stretch=False,
                anchor=cast(Literal["w", "e", "center"], anc),
            )
        p48_vsb = ttk.Scrollbar(p48_tree_box, orient="vertical", command=self.p48_pred_tree.yview)
        p48_hsb = ttk.Scrollbar(p48_tree_box, orient="horizontal", command=self.p48_pred_tree.xview)
        self.p48_pred_tree.configure(yscrollcommand=p48_vsb.set, xscrollcommand=p48_hsb.set)
        self.p48_pred_tree.grid(row=0, column=0, sticky="nsew")
        p48_vsb.grid(row=0, column=1, sticky="ns")
        p48_hsb.grid(row=1, column=0, sticky="ew")

        # ==================================================================
        # 3) Session History & Sealed Evidence
        # ==================================================================
        hist_frame = ttk.LabelFrame(
            container, text="3. Observation Session History & Sealed Evidence", padding=6
        )
        hist_frame.pack(fill="x", pady=(0, 8))

        hist_btns = ttk.Frame(hist_frame)
        hist_btns.pack(fill="x", pady=(0, 4))
        p.btn_refresh_obs_history = ttk.Button(
            hist_btns, text="Refresh History", command=self.refresh_session_history
        )
        p.btn_refresh_obs_history.pack(side="left", padx=(0, 6))
        p.btn_open_obs_diag = ttk.Button(
            hist_btns, text="Open Diagnostics Report", command=self.open_selected_diagnostics
        )
        p.btn_open_obs_diag.pack(side="left", padx=4)
        p.btn_verify_obs_session = ttk.Button(
            hist_btns, text="Verify Session Integrity", command=self.verify_selected_session
        )
        p.btn_verify_obs_session.pack(side="left", padx=4)
        p.btn_export_obs_evidence = ttk.Button(
            hist_btns, text="Export Sealed Evidence", command=self.export_selected_evidence
        )
        p.btn_export_obs_evidence.pack(side="left", padx=4)

        hist_tree_box = ttk.Frame(hist_frame)
        hist_tree_box.pack(fill="both", expand=True)
        hist_tree_box.rowconfigure(0, weight=1)
        hist_tree_box.columnconfigure(0, weight=1)

        hist_cols = ("session_id", "provenance", "tf", "model_id", "state", "resolved", "accuracy")
        p.obs_history_tree = ttk.Treeview(
            hist_tree_box, columns=hist_cols, show="headings", height=5
        )
        for col, title, w, anc in [
            ("session_id", "Session ID", 220, "w"),
            ("provenance", "Provenance", 100, "center"),
            ("tf", "TF", 55, "center"),
            ("model_id", "Candidate / Fixture", 230, "w"),
            ("state", "State", 145, "center"),
            ("resolved", "Resolved", 75, "e"),
            ("accuracy", "Accuracy", 85, "e"),
        ]:
            p.obs_history_tree.heading(col, text=title)
            p.obs_history_tree.column(
                col,
                width=w,
                minwidth=max(50, w - 20),
                stretch=False,
                anchor=cast(Literal["w", "e", "center"], anc),
            )

        hist_vsb = ttk.Scrollbar(hist_tree_box, orient="vertical", command=p.obs_history_tree.yview)
        hist_hsb = ttk.Scrollbar(
            hist_tree_box, orient="horizontal", command=p.obs_history_tree.xview
        )
        p.obs_history_tree.configure(yscrollcommand=hist_vsb.set, xscrollcommand=hist_hsb.set)
        p.obs_history_tree.grid(row=0, column=0, sticky="nsew")
        hist_vsb.grid(row=0, column=1, sticky="ns")
        hist_hsb.grid(row=1, column=0, sticky="ew")

        # ==================================================================
        # 4) Diagnostics & Event Log
        # ==================================================================
        log_frame = ttk.LabelFrame(container, text="4. Diagnostics & Event Log", padding=6)
        log_frame.pack(fill="x", pady=(0, 4))
        log_frame.columnconfigure(0, weight=1)

        p.obs_log_text = tk.Text(
            log_frame, height=5, wrap="word", font=("Consolas", 9), state="disabled"
        )
        log_vsb = ttk.Scrollbar(log_frame, orient="vertical", command=p.obs_log_text.yview)
        p.obs_log_text.configure(yscrollcommand=log_vsb.set)
        p.obs_log_text.grid(row=0, column=0, sticky="nsew")
        log_vsb.grid(row=0, column=1, sticky="ns")

        self.start_polling()
        self.refresh_models()
        self.refresh_session_history()
        self._refresh_p48_governance_status()
        self.on_mode_changed()
        self._on_inner_configure()

    # ------------------------------------------------------------------
    # Configuration state handlers
    # ------------------------------------------------------------------
    def refresh_models(self) -> None:
        p = self.panel
        mode = p.var_obs_mode.get()
        genuine_ids = [c.model_id for c in p.candidates]
        if mode == "synthetic":
            values = list(self._fixture_label_to_tf.keys()) + genuine_ids
            p.combo_obs_model["values"] = values
            if values and p.combo_obs_model.get() not in values:
                p.combo_obs_model.set(values[0])
                self.on_model_selected()
        else:
            p.combo_obs_model["values"] = genuine_ids
            if genuine_ids:
                if p.combo_obs_model.get() not in genuine_ids:
                    p.combo_obs_model.set(genuine_ids[0])
            else:
                p.combo_obs_model.set("")

    def on_mode_changed(self) -> None:
        p = self.panel
        mode = p.var_obs_mode.get()
        self.refresh_models()
        if mode == "exploratory_genuine":
            p.combo_obs_account.config(state="readonly")
            auth_ok, msg = self._refresh_p48_governance_status()
            p.lbl_obs_eligibility.config(
                text=msg,
                foreground="#1b4d3e" if auth_ok else "#9c3414",
            )
            if not self._is_running:
                p.btn_start_obs.config(state="normal" if auth_ok else "disabled")
        elif mode == "live":
            p.combo_obs_account.config(state="readonly")
            p.lbl_obs_eligibility.config(
                text="BLOCKED: Phase 46 confirmatory contract UNAUTHORIZED (0 qualified candidates)",
                foreground="#9c3414",
            )
            if not self._is_running:
                p.btn_start_obs.config(state="disabled")
        else:
            p.combo_obs_account.config(state="disabled")
            p.lbl_obs_eligibility.config(
                text="SYNTHETIC_FIXTURE (Offline Lifecycle Demo Only)",
                foreground="#1b4d3e",
            )
            if not self._is_running:
                p.btn_start_obs.config(state="normal")

    def _selected_p48_candidate_ids(self) -> tuple[str, ...]:
        return tuple(cid for cid, var in self._p48_candidate_vars.items() if var.get())

    def _refresh_p48_governance_status(self) -> tuple[bool, str]:
        models_dir = default_models_dir(self.root)
        loaded_digests: dict[str, str] = {}
        for spec in CANDIDATE_SPECS:
            cid = spec.frozen_candidate_id
            lbl = self._p48_candidate_status_lbls.get(cid)
            bdir = models_dir / cid
            if not bdir.is_dir():
                if lbl is not None:
                    lbl.config(text="MISSING BUNDLE", foreground="#9c3414")
                continue
            try:
                man = verify_exploratory_bundle(bdir)
                loaded_digests[cid] = man.bundle_sha256
                if lbl is not None:
                    lbl.config(
                        text=(
                            f"EXPLORATORY_FROZEN | tau={man.frozen_confidence_threshold:.4f} | "
                            f"sha={man.bundle_sha256[:10]}"
                        ),
                        foreground="#1b4d3e",
                    )
            except Exception as exc:  # noqa: BLE001
                if lbl is not None:
                    lbl.config(text=f"TAMPERED/INVALID: {exc}", foreground="#9c3414")

        cpath = default_campaign_contract_path(self.root)
        contract = load_campaign_contract(cpath) if cpath.is_file() else None
        acct = ""
        if hasattr(self.panel, "_resolve_selected_account_id"):
            acct = str(self.panel._resolve_selected_account_id() or "")  # noqa: SLF001
        if not acct and hasattr(self.panel, "combo_obs_account"):
            acct = str(self.panel.combo_obs_account.get() or "")
        dur = 15
        if hasattr(self.panel, "combo_obs_duration"):
            with contextlib.suppress(ValueError):
                dur = int(self.panel.combo_obs_duration.get() or 15)

        usage_store = (
            CampaignUsageStore(default_campaign_usage_path(self.root, contract.contract_id))
            if contract is not None
            else None
        )
        stage_spec = get_stage_spec(contract, contract.authorized_stage) if contract else None
        res = validate_campaign_authorization(
            contract,
            requested_candidate_ids=self._selected_p48_candidate_ids(),
            requested_duration_minutes=dur,
            loaded_artifact_digests=loaded_digests,
            account_ref=acct,
            is_genuine_mode=True,
            requested_allow_outcome_drain=bool(stage_spec and stage_spec.allow_outcome_drain),
            requested_drain_minutes=(stage_spec.max_outcome_drain_minutes if stage_spec else 0),
            usage_store=usage_store,
        )
        msg = (
            "AUTHORIZED: Phase 48 Supervised Exploratory Campaign"
            if res.authorized
            else f"BLOCKED: {res.status} ({res.reasons[0] if res.reasons else 'Unauthorized'})"
        )
        if self.lbl_p48_auth_status is not None:
            self.lbl_p48_auth_status.config(
                text=msg,
                foreground="#1b4d3e" if res.authorized else "#9c3414",
            )
        return res.authorized, msg

    def on_model_selected(self, _event: Any = None) -> None:
        p = self.panel
        sel = p.combo_obs_model.get()
        if sel in self._fixture_label_to_tf:
            tf = self._fixture_label_to_tf[sel]
            p.combo_obs_timeframe.set(_TF_SEC_TO_DISPLAY.get(tf, "M1 (60s)"))
            fx = get_synthetic_fixture(tf)
            p.lbl_obs_horizon.config(
                text=f"Prediction Duration: {fx.prediction_duration_minutes} min (H{fx.horizon_bars})"
            )

    def on_timeframe_selected(self, _event: Any = None) -> None:
        p = self.panel
        tf_label = p.combo_obs_timeframe.get()
        tf = _TF_DISPLAY_TO_SEC.get(tf_label, 60)
        fx = get_synthetic_fixture(tf)
        p.lbl_obs_horizon.config(
            text=f"Prediction Duration: {fx.prediction_duration_minutes} min (H{fx.horizon_bars})"
        )
        # Sync fixture model if currently using a synthetic fixture
        if p.var_obs_mode.get() == "synthetic":
            for label, l_tf in self._fixture_label_to_tf.items():
                if l_tf == tf:
                    p.combo_obs_model.set(label)
                    break

    # ------------------------------------------------------------------
    # Start / Stop Offline Synthetic Observation
    # ------------------------------------------------------------------
    def start_synthetic_observation(self) -> bool:
        """Start an offline Phase 46 synthetic observation in a background thread."""
        p = self.panel
        if self._is_running:
            return False

        sel_model = p.combo_obs_model.get().strip()
        tf_sec = _TF_DISPLAY_TO_SEC.get(p.combo_obs_timeframe.get(), 60)
        if sel_model in self._fixture_label_to_tf:
            tf_sec = self._fixture_label_to_tf[sel_model]

        minutes = int(p.combo_obs_duration.get() or "15")
        pace_sec = _PACE_MAP.get(p.combo_obs_pace.get(), 0.05)

        self._generation_id += 1
        gen_id = self._generation_id
        self._cancel_event.clear()
        self._is_running = True
        self._accuracy_history = []
        self._draw_accuracy_chart([])

        p.btn_start_obs.config(state="disabled")
        p.btn_cancel_obs.config(state="normal")
        p.obs_progress_bar["value"] = 0
        p.lbl_obs_status.config(
            text=f"State: INITIALIZING | Source: SYNTHETIC_STREAM ({tf_sec}s) | Provisioning fixture..."
        )
        self._append_log(f"[Phase 46] Starting offline synthetic demonstration ({tf_sec}s)...")

        def worker() -> None:
            try:
                reg, model_id, warmup, stream_candles, fx = provision_synthetic_fixture_workspace(
                    self.synth_root,
                    timeframe_seconds=tf_sec,
                    n_train_candles=360,
                    n_warmup_candles=35,
                    n_stream_candles=28,
                )
                session_id = f"p46_synth_{fx.timeframe_label.lower()}_{int(time.time())}"
                sessions_root = self.root / "data" / "phase46"
                sessions_root.mkdir(parents=True, exist_ok=True)
                self._active_session_id = session_id
                self._active_session_dir = sessions_root / DEFAULT_SHADOW_SESSIONS_DIR / session_id

                total_bars = len(stream_candles)
                spec = Phase46SyntheticRunSpec(
                    session_id=session_id,
                    symbol=fx.symbol,
                    timeframe_seconds=tf_sec,
                    model_ids=[model_id],
                    candles=stream_candles,
                    warmup_candles=warmup,
                    duration_minutes=minutes,
                    max_data_age_seconds=3600 * 24 * 365 * 20,  # deterministic historical epoch
                    latency_simulation_ms=25.0,
                    step_delay_seconds=pace_sec,
                )

                def on_progress(prog: ObservationProgress) -> None:
                    if gen_id != self._generation_id or self._is_destroyed:
                        return
                    if self._cancel_event.is_set() and self._observer_ref:
                        self._observer_ref[0].cancel()
                    now_m = time.monotonic()
                    if (
                        now_m - self._last_ui_push_mono < 0.04
                        and prog.valid_candles_count < total_bars
                    ):
                        return
                    self._last_ui_push_mono = now_m
                    snapshot = self._build_progress_snapshot(
                        prog,
                        session_id=session_id,
                        sessions_root=sessions_root,
                        total_bars=total_bars,
                        tf_sec=tf_sec,
                    )
                    self._post_ui(
                        lambda s=snapshot: self._apply_progress_snapshot(gen_id, s)  # type: ignore[misc]
                    )

                result = run_phase46_synthetic_observation(
                    spec,
                    registry=reg,
                    root=sessions_root,
                    progress_callback=on_progress,
                    observer_ref=self._observer_ref,
                )
                diag = compute_phase46_session_diagnostics(
                    session_id=session_id,
                    root=sessions_root,
                    model_id=model_id,
                    symbol=fx.symbol,
                    timeframe_seconds=tf_sec,
                    is_functional_test=True,
                )
                self._post_ui(lambda: self._on_worker_finished(gen_id, result, diag))
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{type(exc).__name__}: {exc}"
                self._post_ui(lambda: self._on_worker_error(gen_id, err_msg))

        self._worker_thread = threading.Thread(target=worker, name="phase46-synth-obs", daemon=True)
        self._worker_thread.start()
        return True

    def start_phase48_synthetic_demo(self) -> bool:
        """Run an offline Phase 48 parallel multi-model synthetic demonstration."""
        import asyncio
        import math
        from datetime import timedelta

        from iqoption_research.models.market import Candle, CandleSize
        from iqoption_research.phase48.transport import SyntheticMultiTimeframeTransport

        if self._is_running:
            return False
        cids = self._selected_p48_candidate_ids()
        if not cids:
            messagebox.showwarning("Phase 48", "Select at least one exploratory candidate.")
            return False

        p = self.panel
        self._cancel_event.clear()
        self._generation_id += 1
        gen_id = self._generation_id
        self._is_running = True
        self._p48_recent_rows = []
        self._p48_multi_series = {}
        p.btn_start_obs.config(state="disabled")
        p.btn_cancel_obs.config(state="normal")
        if self.btn_p48_synth_demo is not None:
            self.btn_p48_synth_demo.config(state="disabled")

        session_id = f"p48_synth_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        self._active_session_id = session_id
        dur = int(p.combo_obs_duration.get() or 15)
        if dur not in SUPPORTED_SESSION_DURATIONS_MINUTES:
            dur = 15

        self._append_log(
            f"[Phase 48] Starting offline multi-model synthetic demo ({len(cids)} candidates, {dur}m window)..."
        )

        def _build_synth_candles(tf_sec: int, count: int, start_dt: datetime) -> list[Candle]:
            out: list[Candle] = []
            price = 1.0850
            for i in range(count):
                delta = 0.00025 * math.sin(i * 0.35 + tf_sec) + 0.00008 * math.cos(i * 0.13)
                o = price
                c = o + delta
                h = max(o, c) + 0.00018
                lo = min(o, c) - 0.00018
                ts_dt = start_dt + timedelta(seconds=i * tf_sec)
                from_ts = int(ts_dt.timestamp())
                out.append(
                    Candle(
                        active_id=1,
                        symbol="EURUSD",
                        size=CandleSize(tf_sec),
                        from_ts=from_ts,
                        to_ts=from_ts + tf_sec,
                        at_ts=from_ts + tf_sec,
                        open=round(o, 5),
                        high=round(h, 5),
                        low=round(lo, 5),
                        close=round(c, 5),
                        bid=round(c, 5),
                        ask=round(c + 0.0001, 5),
                        volume=100 + (i % 25),
                        is_complete=True,
                        provenance="SYNTHETIC_ACCEPTANCE_FEED",
                    )
                )
                price = c
            return out

        def worker() -> None:
            try:
                base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
                warmup_by_tf = {
                    60: _build_synth_candles(60, 80, base_dt - timedelta(minutes=80)),
                    300: _build_synth_candles(300, 80, base_dt - timedelta(minutes=400)),
                    900: _build_synth_candles(900, 80, base_dt - timedelta(minutes=1200)),
                }
                events: list[tuple[int, Candle]] = []
                m1_stream = _build_synth_candles(60, 90, base_dt)
                m5_stream = _build_synth_candles(300, 18, base_dt)
                m15_stream = _build_synth_candles(900, 6, base_dt)
                for c in m1_stream:
                    events.append((60, c))
                for c in m5_stream:
                    events.append((300, c))
                for c in m15_stream:
                    events.append((900, c))
                events.sort(key=lambda item: (item[1].timestamp_utc, item[0]))

                transport = SyntheticMultiTimeframeTransport(
                    warmup_by_tf=warmup_by_tf,
                    events=events,
                    step_delay_seconds=0.01,
                )
                cfg = ParallelObservationConfig(
                    session_id=session_id,
                    candidate_ids=cids,
                    duration_minutes=dur,
                    is_genuine_mode=False,
                    simulated_duration_seconds=30.0,
                )
                rt = ParallelObservationRuntime(cfg, repo_root=self.root)
                self._p48_runtime = rt

                def _on_prog(prog: ParallelObservationProgress) -> None:
                    self._post_ui(lambda: self._apply_p48_progress(gen_id, prog))

                res = asyncio.run(rt.run(transport, progress_callback=_on_prog))
                self._post_ui(lambda: self._on_p48_finished(gen_id, res))
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{type(exc).__name__}: {exc}"
                self._post_ui(lambda: self._on_worker_error(gen_id, err_msg))

        self._worker_thread = threading.Thread(target=worker, name="phase48-synth-obs", daemon=True)
        self._worker_thread.start()
        return True

    def start_genuine_exploratory_observation(
        self,
        *,
        transport_factory: Callable[[Any, tuple[int, ...]], Any] | None = None,
        vault_factory: Callable[[], Any] | None = None,
        confirm_override: bool | None = None,
    ) -> bool:
        """Start an explicitly authorized, supervised genuine Phase 48 exploratory session."""
        import asyncio

        from iqoption_research.phase48.transport import IQOptionReadOnlyMultiTransport

        if self._is_running:
            return False

        p = self.panel
        cids = self._selected_p48_candidate_ids()
        if not cids:
            messagebox.showwarning("Phase 48", "Select at least one exploratory candidate.")
            return False

        account_id = str(p._resolve_selected_account_id() or "")  # noqa: SLF001
        if not account_id:
            messagebox.showwarning(
                "Account Required",
                "Select a saved Windows Credential Vault account for genuine observation.",
            )
            return False

        models_dir = default_models_dir(self.root)
        loaded_digests: dict[str, str] = {}
        selected_tfs: list[int] = []
        for cid in cids:
            try:
                man = verify_exploratory_bundle(models_dir / cid)
                loaded_digests[cid] = man.bundle_sha256
                selected_tfs.append(man.native_timeframe_seconds)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror(
                    "Artifact Integrity Error",
                    f"Frozen bundle verification failed for {cid}: {exc}",
                )
                return False

        cpath = default_campaign_contract_path(self.root)
        if not cpath.is_file():
            messagebox.showwarning(
                "Genuine Exploratory Blocked",
                "No Phase 48 campaign contract found. Authorize via CLI first.",
            )
            return False

        contract = load_campaign_contract(cpath)
        dur = int(p.combo_obs_duration.get() or 15)
        stage_spec = get_stage_spec(contract, contract.authorized_stage)
        allow_drain = bool(stage_spec and stage_spec.allow_outcome_drain)
        drain_min = stage_spec.max_outcome_drain_minutes if stage_spec else 0
        usage_store = CampaignUsageStore(
            default_campaign_usage_path(self.root, contract.contract_id)
        )
        session_id = f"p48_gen_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        auth_res = validate_campaign_authorization(
            contract,
            requested_candidate_ids=cids,
            requested_duration_minutes=dur,
            loaded_artifact_digests=loaded_digests,
            account_ref=account_id,
            is_genuine_mode=True,
            requested_allow_outcome_drain=allow_drain,
            requested_drain_minutes=drain_min,
            usage_store=usage_store,
            session_id=session_id,
        )
        if not auth_res.authorized:
            messagebox.showwarning(
                "Genuine Exploratory Blocked",
                f"{auth_res.status}:\n" + "\n".join(auth_res.reasons),
            )
            return False

        display_acc = p.combo_obs_account.get() or account_id
        confirmed = (
            confirm_override
            if confirm_override is not None
            else messagebox.askyesno(
                "Start Genuine Exploratory Observation?",
                f"Campaign: {contract.contract_id} ({contract.authorized_stage})\n"
                f"Account: {display_acc}\n"
                f"Candidates: {', '.join(cids)}\n"
                f"Duration: {dur} min (Drain: {drain_min} min)\n\n"
                "Read-only market data only. execution_authority=NONE.\n"
                "Proceed with supervised observation?",
                parent=p.dashboard.window,
            )
        )
        if not confirmed:
            return False

        try:
            if vault_factory is not None:
                vault = vault_factory()
            else:
                from iqoption_research.auth.vault import WindowsCredentialVault

                vault = WindowsCredentialVault()
            creds = vault.account_store().load(account_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "Vault Error",
                f"Failed to load credentials from Windows Vault: {type(exc).__name__}",
            )
            return False

        self._cancel_event.clear()
        self._generation_id += 1
        gen_id = self._generation_id
        self._is_running = True
        self._active_session_id = session_id
        self._p48_recent_rows = []
        self._p48_recent_outcomes = {}
        self._p48_multi_series = {}
        p.btn_start_obs.config(state="disabled")
        p.btn_cancel_obs.config(state="normal")
        if self.btn_p48_synth_demo is not None:
            self.btn_p48_synth_demo.config(state="disabled")

        self._append_log(
            f"[Phase 48 Genuine] Starting supervised session {session_id} "
            f"({contract.authorized_stage}, {dur}m, {len(cids)} models)..."
        )

        tfs_tuple = tuple(sorted(set(selected_tfs)))

        def worker() -> None:
            try:
                transport = (
                    transport_factory(creds, tfs_tuple)
                    if transport_factory is not None
                    else IQOptionReadOnlyMultiTransport(
                        creds,
                        timeframes=tfs_tuple,
                        require_practice_account=contract.require_practice_account,
                    )
                )
                cfg = ParallelObservationConfig(
                    session_id=session_id,
                    candidate_ids=cids,
                    duration_minutes=dur,
                    is_genuine_mode=True,
                    account_ref=account_id,
                    campaign_contract=contract,
                    models_dir=models_dir,
                    allow_outcome_drain=allow_drain,
                    max_outcome_drain_minutes=drain_min,
                )
                rt = ParallelObservationRuntime(cfg, repo_root=self.root)
                self._p48_runtime = rt

                def _on_prog(prog: ParallelObservationProgress) -> None:
                    self._post_ui(lambda: self._apply_p48_progress(gen_id, prog))

                res = asyncio.run(rt.run(transport, progress_callback=_on_prog))
                self._post_ui(lambda: self._on_p48_finished(gen_id, res))
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{type(exc).__name__}: {exc}"
                self._post_ui(lambda: self._on_worker_error(gen_id, err_msg))

        self._worker_thread = threading.Thread(
            target=worker, name="phase48-genuine-obs", daemon=True
        )
        self._worker_thread.start()
        return True

    def _apply_p48_progress(self, gen_id: int, prog: ParallelObservationProgress) -> None:
        if gen_id != self._generation_id or self._is_destroyed:
            return
        p = self.panel
        total = max(1.0, prog.elapsed_seconds + prog.remaining_seconds)
        p.obs_progress_bar["value"] = min(100, int((prog.elapsed_seconds / total) * 100))
        p.lbl_obs_status.config(
            text=(
                f"State: {prog.status} | Classification: {prog.evidence_classification} | "
                f"Elapsed: {prog.elapsed_seconds:.1f}s | Remaining: {prog.remaining_seconds:.1f}s"
            )
        )
        series_map: dict[str, list[float]] = {}
        for cid, st in prog.per_candidate.items():
            lbl = self._p48_kpi_labels.get(cid)
            acc = st.get("accuracy")
            acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
            if lbl is not None:
                lbl.config(
                    text=(
                        f"Warmup: {st['warmup_bars_loaded']}/{st['warmup_required']} | "
                        f"Opps: {st['opportunities']} | Sel: {st['selected']}\n"
                        f"Pending: {st['pending']} | Res: {st['resolved']} "
                        f"(W:{st['correct']}/L:{st['incorrect']}) | Acc: {acc_str}"
                    )
                )
            series_map[cid] = list(st.get("accuracy_series", []))
        self._p48_multi_series = series_map
        self._draw_multi_accuracy_chart(series_map)
        self._p48_recent_rows = list(prog.recent_predictions)
        self._p48_recent_outcomes = dict(prog.recent_outcomes)
        self._render_p48_predictions()

    def _on_p48_finished(self, gen_id: int, res: Any) -> None:
        if gen_id != self._generation_id or self._is_destroyed:
            return
        p = self.panel
        self._is_running = False
        self._p48_runtime = None
        p.btn_cancel_obs.config(state="disabled")
        if self.btn_p48_synth_demo is not None:
            self.btn_p48_synth_demo.config(state="normal")
        self.on_mode_changed()
        p.obs_progress_bar["value"] = 100
        p.lbl_obs_status.config(
            text=(
                f"State: {res.status} | {res.evidence_classification} | "
                f"Integrity: {'VERIFIED' if res.integrity_verified else 'FAILED'} | "
                f"Elapsed: {res.elapsed_seconds:.1f}s"
            )
        )
        self._append_log(
            f"[Phase 48] Session {res.session_id} finished ({res.status}): {res.stop_reason}"
        )
        self.refresh_session_history()

    def _render_p48_predictions(self) -> None:
        tree = self.p48_pred_tree
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        f_cand = self.p48_filter_cand.get() if self.p48_filter_cand else "ALL"
        f_stat = self.p48_filter_status.get() if self.p48_filter_status else "ALL"
        for row in self._p48_recent_rows:
            cid = str(row.get("candidate_id", ""))
            dec = str(row.get("decision", ""))
            rid = str(row.get("record_id", ""))
            if f_cand != "ALL" and cid != f_cand:
                continue
            out = self._p48_recent_outcomes.get(rid)
            if out is not None and str(out.get("resolution_status", "")) == "RESOLVED":
                status = "RESOLVED"
                actual_dir = str(out.get("actual_direction", "N/A"))
                outcome_lbl = str(out.get("classification", "N/A"))
            elif dec == "ABSTAIN":
                status = "ABSTAIN"
                actual_dir = str(out.get("actual_direction", "N/A")) if out else "PENDING"
                outcome_lbl = "ABSTAIN"
            else:
                status = "PENDING"
                actual_dir = "PENDING"
                outcome_lbl = "PENDING"

            if f_stat != "ALL" and status != f_stat:
                continue
            dt_str = datetime.fromtimestamp(int(row["decision_timestamp"]), UTC).strftime(
                "%Y-%m-%d %H:%M"
            )
            mat_str = datetime.fromtimestamp(int(row["maturity_timestamp"]), UTC).strftime(
                "%Y-%m-%d %H:%M"
            )
            tf_lbl = {60: "M1", 300: "M5", 900: "M15"}.get(int(row["timeframe_seconds"]), "?")
            prob = max(float(row["prob_up"]), float(row["prob_down"]))
            tree.insert(
                "",
                "end",
                values=(
                    dt_str,
                    cid,
                    tf_lbl,
                    f"H{row['horizon_bars']}",
                    dec,
                    f"{prob * 100:.2f}%",
                    status,
                    actual_dir,
                    outcome_lbl,
                    mat_str,
                ),
            )

    def _draw_multi_accuracy_chart(self, series_map: dict[str, list[float]]) -> None:
        canvas = getattr(self.panel, "obs_chart_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        w = max(
            260, canvas.winfo_width() if canvas.winfo_width() > 20 else int(canvas.cget("width"))
        )
        h = max(
            170, canvas.winfo_height() if canvas.winfo_height() > 20 else int(canvas.cget("height"))
        )
        pad_l, pad_r, pad_t, pad_b = 34, 10, 18, 20
        pw = w - pad_l - pad_r
        ph = h - pad_t - pad_b
        canvas.create_text(
            w // 2,
            8,
            text="Phase 48 Multi-Model Cumulative Accuracy (Separate Series)",
            font=("Segoe UI", 8, "bold"),
            fill="#17324d",
        )
        canvas.create_rectangle(pad_l, pad_t, w - pad_r, h - pad_b, outline="#b8c4d0")
        y50 = pad_t + ph * 0.50
        canvas.create_line(pad_l, y50, w - pad_r, y50, fill="#9aa5b1", dash=(3, 3))
        canvas.create_text(pad_l - 16, y50, text="50%", font=("Segoe UI", 7), fill="#52606d")

        colors = ["#1d4ed8", "#0f766e", "#b45309"]
        any_pts = False
        for idx, (cid, vals) in enumerate(sorted(series_map.items())):
            color = colors[idx % len(colors)]
            canvas.create_text(
                pad_l + 6 + idx * 95,
                h - 8,
                anchor="w",
                text=f"● {cid.split('_')[3].upper()}",
                font=("Segoe UI", 7, "bold"),
                fill=color,
            )
            if not vals:
                continue
            any_pts = True
            pts: list[float] = []
            n = len(vals)
            for i, v in enumerate(vals):
                x = pad_l + (pw * (i / max(1, n - 1))) if n > 1 else pad_l + pw / 2
                y = pad_t + ph * (1.0 - max(0.0, min(1.0, float(v))))
                pts.extend([x, y])
            if len(pts) >= 4:
                canvas.create_line(*pts, fill=color, width=2)
            elif len(pts) == 2:
                canvas.create_oval(pts[0] - 3, pts[1] - 3, pts[0] + 3, pts[1] + 3, fill=color)
        if not any_pts:
            canvas.create_text(
                pad_l + pw // 2,
                pad_t + ph // 2,
                text="No resolved Phase 48 predictions yet",
                font=("Segoe UI", 8, "italic"),
                fill="#7b8794",
            )

    def cancel_synthetic_observation(self) -> None:
        self._cancel_event.set()
        if self._observer_ref:
            with contextlib.suppress(Exception):
                self._observer_ref[0].cancel()
        if self._p48_runtime is not None:
            with contextlib.suppress(Exception):
                self._p48_runtime.cancel()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if self._p48_runtime is not None:
            with contextlib.suppress(Exception):
                self._p48_runtime.cancel()
        self.panel.btn_cancel_obs.config(state="disabled")
        self._append_log("[Observation] Cancellation requested; sealing partial evidence...")

    def _build_progress_snapshot(
        self,
        prog: ObservationProgress,
        *,
        session_id: str,
        sessions_root: Path,
        total_bars: int,
        tf_sec: int,
    ) -> dict[str, Any]:
        journal = ShadowPredictionJournal(session_id, root=sessions_root)
        preds = journal.list_predictions()
        latest_pred_str = "N/A"
        if preds:
            lp = preds[-1]
            probs = getattr(lp, "probabilities", None) or {}
            conf_pct = max(float(probs.get("UP", 0.5)), float(probs.get("DOWN", 0.5))) * 100.0
            latest_pred_str = f"{lp.decision} ({conf_pct:.1f}%)"

        acc_series: list[float] = []
        wins = 0
        decisive_res = 0
        for pr in preds:
            if pr.outcome_status == OutcomeStatus.RESOLVED_WIN.value:
                wins += 1
                decisive_res += 1
                acc_series.append(wins / decisive_res)
            elif pr.outcome_status == OutcomeStatus.RESOLVED_LOSS.value:
                decisive_res += 1
                acc_series.append(wins / decisive_res)

        return {
            "prog": prog,
            "tf_sec": tf_sec,
            "total_bars": total_bars,
            "latest_pred_str": latest_pred_str,
            "acc_series": acc_series,
        }

    def _apply_progress_snapshot(self, gen_id: int, snap: dict[str, Any]) -> None:
        if gen_id != self._generation_id or self._is_destroyed:
            return
        p = self.panel
        prog: ObservationProgress = snap["prog"]
        total_bars: int = snap["total_bars"]
        pct = min(100, int((prog.valid_candles_count / max(1, total_bars)) * 100))
        p.obs_progress_bar["value"] = pct

        candle_utc = prog.last_candle_received_utc or "N/A"
        p.lbl_obs_status.config(
            text=(
                f"State: {prog.fsm_state} | Source: SYNTHETIC_STREAM ({snap['tf_sec']}s) | "
                f"Elapsed: {prog.elapsed_seconds:.1f}s | Latest Candle: {candle_utc} | "
                f"Latest Pred: {snap['latest_pred_str']}"
            )
        )

        # Aggregate KPIs across models
        total_preds = sum(m.total_predictions for m in prog.models.values())
        pending = sum(m.pending_predictions for m in prog.models.values())
        wins = sum(m.wins_count for m in prog.models.values())
        losses = sum(m.losses_count for m in prog.models.values())
        ties = sum(m.ties_count for m in prog.models.values())
        abstains = sum(
            max(
                0,
                m.total_predictions
                - m.trades_taken
                - m.pending_predictions
                - m.invalid_count
                - m.quarantined_late_count,
            )
            for m in prog.models.values()
        )
        resolved = wins + losses + ties
        selected = total_preds - abstains
        cov_str = f"{(selected / total_preds) * 100:.1f}%" if total_preds > 0 else "N/A"
        acc_str = f"{(wins / (wins + losses)) * 100:.2f}%" if (wins + losses) > 0 else "N/A"

        first_m = next(iter(prog.models.values()), None)
        base_str = (
            f"{first_m.matched_baseline_accuracy * 100:.2f}%"
            if (first_m and first_m.matched_baseline_accuracy is not None and (wins + losses) > 0)
            else "N/A"
        )
        p.lbl_obs_kpis.config(
            text=(
                f"Opportunities: {prog.accepted_opportunities_count} | Selected: {selected} | "
                f"Pending: {pending} | Resolved: {resolved} (W:{wins} / L:{losses} / T:{ties}) | "
                f"Abstentions: {abstains} | Accuracy: {acc_str} | Coverage: {cov_str} | Baseline: {base_str}"
            )
        )
        payout_summary = (
            f"{first_m.expectancy_binary_pct:+.2f}% / {first_m.break_even_gap_binary * 100:+.2f}%"
            if (first_m and (wins + losses) > 0)
            else "N/A"
        )
        self._update_kpi_cards(
            opps=prog.accepted_opportunities_count,
            selected=selected,
            cov_str=cov_str,
            abstains=abstains,
            pending=pending,
            resolved=resolved,
            wins=wins,
            losses=losses,
            ties=ties,
            acc_str=acc_str,
            base_str=base_str,
            payout_str=payout_summary,
        )
        if first_m and (wins + losses) > 0:
            p.lbl_obs_payout.config(
                text=(
                    f"Hypothetical 85% Payout Sensitivity (NOT actual broker settlement): "
                    f"Exp Binary {first_m.expectancy_binary_pct:+.2f}% | BE Gap {first_m.break_even_gap_binary * 100:+.2f}%"
                )
            )
        self._render_obs_tree(prog.models)
        self._accuracy_history = snap["acc_series"]
        self._draw_accuracy_chart(self._accuracy_history)

    def _render_obs_tree(self, models_data: dict[str, Any]) -> None:
        p = self.panel
        for item in p.obs_tree.get_children():
            p.obs_tree.delete(item)
        for mid, mstate in models_data.items():
            w = mstate.wins_count
            loss = mstate.losses_count
            tie = mstate.ties_count
            trades = mstate.trades_taken
            if trades > 0:
                win_str = f"{mstate.win_rate_decisive * 100:.2f}%"
                ci = mstate.wilson_ci_decisive
                ci_str = f"[{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]"
                exp_b_str = f"{mstate.expectancy_binary_pct:+.2f}%"
                exp_d_str = f"{mstate.expectancy_digital_pct:+.2f}%"
                be_str = f"{mstate.break_even_gap_binary * 100:+.2f}%"
            else:
                win_str = ci_str = exp_b_str = exp_d_str = be_str = "N/A"
            maj = mstate.matched_baseline_accuracy
            maj_str = f"{maj * 100:.2f}%" if (maj is not None and trades > 0) else "N/A"
            p.obs_tree.insert(
                "",
                "end",
                values=(
                    mid,
                    mstate.total_predictions,
                    mstate.pending_predictions,
                    f"{w}/{loss}/{tie}",
                    win_str,
                    maj_str,
                    ci_str,
                    exp_b_str,
                    exp_d_str,
                    be_str,
                ),
            )

    def _on_worker_finished(self, gen_id: int, result: Any, diag: Any) -> None:
        if gen_id != self._generation_id or self._is_destroyed:
            return
        self._is_running = False
        p = self.panel
        p.obs_progress_bar["value"] = 100
        self.on_mode_changed()
        p.btn_cancel_obs.config(state="disabled")

        sr = result.session_result
        p.lbl_obs_status.config(
            text=(
                f"State: COMPLETED ({sr.termination_reason}) | "
                f"Source: SYNTHETIC_STREAM | Session: {sr.session_id} | Sealed: True"
            )
        )
        # Update KPIs from final diagnostics
        selected_final = diag.total_predictions - diag.skipped_abstain_count
        acc_final = f"{diag.win_rate_decisive * 100:.2f}%" if diag.resolved_count > 0 else "N/A"
        cov_final = f"{diag.coverage * 100:.1f}%" if diag.total_predictions > 0 else "N/A"
        p.lbl_obs_kpis.config(
            text=(
                f"Opportunities: {diag.total_predictions} | Selected: {selected_final} | "
                f"Pending: {diag.pending_count} | Resolved: {diag.resolved_count} "
                f"(W:{diag.win_count} / L:{diag.loss_count} / T:{diag.tie_count}) | "
                f"Abstentions: {diag.skipped_abstain_count} | "
                f"Accuracy: {acc_final} | Coverage: {cov_final}"
            )
        )
        self._update_kpi_cards(
            opps=diag.total_predictions,
            selected=selected_final,
            cov_str=cov_final,
            abstains=diag.skipped_abstain_count,
            pending=diag.pending_count,
            resolved=diag.resolved_count,
            wins=diag.win_count,
            losses=diag.loss_count,
            ties=diag.tie_count,
            acc_str=acc_final,
            base_str="N/A",
            payout_str="See table",
        )
        self._append_log(
            f"[Phase 46] Session {sr.session_id} finished: {sr.termination_reason} "
            f"(resolved={diag.resolved_count}, W={diag.win_count}, L={diag.loss_count}, "
            f"pending={diag.pending_count})."
        )
        self.refresh_session_history()

    def _on_worker_error(self, gen_id: int, err_msg: str) -> None:
        if gen_id != self._generation_id or self._is_destroyed:
            return
        self._is_running = False
        self.on_mode_changed()
        self.panel.btn_cancel_obs.config(state="disabled")
        self.panel.lbl_obs_status.config(text=f"State: FAILED | Error: {err_msg}")
        self._append_log(f"[Phase 46] ERROR: {err_msg}")

    def _update_kpi_cards(
        self,
        *,
        opps: int,
        selected: int,
        cov_str: str,
        abstains: int,
        pending: int,
        resolved: int,
        wins: int,
        losses: int,
        ties: int,
        acc_str: str,
        base_str: str,
        payout_str: str,
    ) -> None:
        if self.lbl_kpi_opps is not None:
            self.lbl_kpi_opps.config(
                text=f"Total: {opps} | Sel: {selected}\nCov: {cov_str} | Abst: {abstains}"
            )
        if self.lbl_kpi_preds is not None:
            self.lbl_kpi_preds.config(text=f"Pending: {pending}\nResolved: {resolved}")
        if self.lbl_kpi_resolved is not None:
            self.lbl_kpi_resolved.config(text=f"W:{wins} / L:{losses} / T:{ties}\nAcc: {acc_str}")
        if self.lbl_kpi_baseline is not None:
            self.lbl_kpi_baseline.config(text=f"Baseline: {base_str}\nExp/BE: {payout_str}")

    # ------------------------------------------------------------------
    # Cumulative Accuracy Chart (Tkinter Canvas)
    # ------------------------------------------------------------------
    def _draw_accuracy_chart(self, series: list[float]) -> None:
        canvas = getattr(self.panel, "obs_chart_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        w = max(
            260, canvas.winfo_width() if canvas.winfo_width() > 20 else int(canvas.cget("width"))
        )
        h = max(
            170, canvas.winfo_height() if canvas.winfo_height() > 20 else int(canvas.cget("height"))
        )
        pad_l, pad_r, pad_t, pad_b = 34, 10, 14, 18
        pw = w - pad_l - pad_r
        ph = h - pad_t - pad_b

        canvas.create_text(
            w // 2,
            7,
            text="Cumulative Accuracy (Synthetic)",
            font=("Segoe UI", 8, "bold"),
            fill="#17324d",
        )
        canvas.create_rectangle(pad_l, pad_t, w - pad_r, h - pad_b, outline="#b8c4d0")

        # 50% and 54.05% reference lines
        y50 = pad_t + ph * (1.0 - 0.50)
        y54 = pad_t + ph * (1.0 - 0.5405)
        canvas.create_line(pad_l, y50, w - pad_r, y50, fill="#9aa5b1", dash=(3, 3))
        canvas.create_text(pad_l - 16, y50, text="50%", font=("Segoe UI", 7), fill="#52606d")
        canvas.create_line(pad_l, y54, w - pad_r, y54, fill="#d97706", dash=(2, 2))

        if not series:
            canvas.create_text(
                pad_l + pw // 2,
                pad_t + ph // 2,
                text="No resolved predictions yet",
                font=("Segoe UI", 8, "italic"),
                fill="#7b8794",
            )
            return

        pts: list[float] = []
        n = len(series)
        for i, val in enumerate(series):
            x = pad_l + (pw * (i / max(1, n - 1))) if n > 1 else pad_l + pw / 2
            clamped = max(0.0, min(1.0, float(val)))
            y = pad_t + ph * (1.0 - clamped)
            pts.extend([x, y])
        if len(pts) >= 4:
            canvas.create_line(*pts, fill="#1b4d3e", width=2)
        elif len(pts) == 2:
            canvas.create_oval(pts[0] - 3, pts[1] - 3, pts[0] + 3, pts[1] + 3, fill="#1b4d3e")
        canvas.create_text(
            w - pad_r - 24,
            h - 8,
            text=f"n={n}",
            font=("Segoe UI", 7),
            fill="#52606d",
        )

    # ------------------------------------------------------------------
    # Session History & Sealed Evidence Export
    # ------------------------------------------------------------------
    def refresh_session_history(self) -> None:
        tree = getattr(self.panel, "obs_history_tree", None)
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        self._sessions_index.clear()

        candidates_dirs: list[Path] = []
        for rel in (
            PHASE48_SESSIONS_DIR,
            f"data/phase46/{DEFAULT_SHADOW_SESSIONS_DIR}",
            DEFAULT_SHADOW_SESSIONS_DIR,
        ):
            base = self.root / rel
            if base.exists():
                for child in sorted(base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    if child.is_dir() and (child / "session.json").exists():
                        candidates_dirs.append(child)

        for sdir in candidates_dirs[:40]:
            meta: dict[str, Any] | None = None
            with contextlib.suppress(Exception):
                meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                continue
            sid = str(meta.get("session_id", sdir.name))
            if sid in self._sessions_index:
                continue
            self._sessions_index[sid] = sdir
            if str(meta.get("schema_version", "")).startswith("phase48") or sid.startswith("p48_"):
                is_gen = bool(meta.get("is_genuine_mode", False))
                prov = "GENUINE_EXPLORATORY" if is_gen else "SYNTHETIC_P48"
                cids = meta.get("candidate_ids", [])
                mid_str = (
                    f"{len(cids)} models" if len(cids) > 1 else (cids[0] if cids else "unknown")
                )
                tf_lbl = "MULTI" if len(cids) > 1 else "NATIVE"
                state = str(meta.get("status", "UNKNOWN"))
                per_diag = meta.get("per_candidate_diagnostics", {})
                resolved = sum(
                    int(d.get("correct_outcomes", 0))
                    + int(d.get("incorrect_outcomes", 0))
                    + int(d.get("ties", 0))
                    for d in per_diag.values()
                )
                acc_str = "Per-Model"
                tree.insert(
                    "",
                    "end",
                    iid=sid,
                    values=(sid, prov, tf_lbl, mid_str, state, resolved, acc_str),
                )
                continue

            is_synth = bool(meta.get("is_synthetic", False))
            prov = "SYNTHETIC" if is_synth else "GENUINE"
            tf_sec = int(meta.get("timeframe", 60))
            tf_lbl = {60: "M1", 300: "M5", 900: "M15"}.get(tf_sec, f"{tf_sec}s")
            mids = meta.get("model_ids", [])
            mid_str = mids[0] if mids else "unknown"
            state = str(meta.get("terminal_status", "UNKNOWN"))

            journal = ShadowPredictionJournal(sid, session_dir=sdir)
            preds = journal.list_predictions()
            wins = sum(1 for pr in preds if pr.outcome_status == OutcomeStatus.RESOLVED_WIN.value)
            losses = sum(
                1 for pr in preds if pr.outcome_status == OutcomeStatus.RESOLVED_LOSS.value
            )
            ties = sum(1 for pr in preds if pr.outcome_status == OutcomeStatus.RESOLVED_TIE.value)
            resolved = wins + losses + ties
            acc_str = f"{(wins / (wins + losses)) * 100:.1f}%" if (wins + losses) > 0 else "N/A"
            tree.insert(
                "",
                "end",
                iid=sid,
                values=(sid, prov, tf_lbl, mid_str, state, resolved, acc_str),
            )

    def _selected_session_id(self) -> str | None:
        tree = getattr(self.panel, "obs_history_tree", None)
        if tree is None:
            return None
        sel = tree.selection()
        if sel:
            return str(sel[0])
        children = tree.get_children()
        if children:
            return str(children[0])
        return self._active_session_id

    def open_selected_diagnostics(self) -> None:
        sid = self._selected_session_id()
        if not sid or sid not in self._sessions_index:
            messagebox.showinfo("No Session", "No observation session selected or available.")
            return
        sdir = self._sessions_index[sid]
        if "phase48" in str(sdir) or sid.startswith("p48_"):
            meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
            self._append_log(
                f"=== PHASE 48 DIAGNOSTICS ({meta.get('evidence_classification')}): {sid} ===\n"
                + json.dumps(meta.get("per_candidate_diagnostics", {}), indent=2, sort_keys=True)
            )
            return
        diag = compute_phase46_session_diagnostics(session_id=sid, session_dir=sdir)
        self._append_log(
            f"=== WS6 DIAGNOSTICS ({'SYNTHETIC' if diag.is_synthetic else 'GENUINE'}): {sid} ===\n"
            + json.dumps(diag.to_dict(), indent=2, sort_keys=True)
        )

    def verify_selected_session(self) -> None:
        sid = self._selected_session_id()
        if not sid or sid not in self._sessions_index:
            messagebox.showinfo("No Session", "No observation session selected or available.")
            return
        sdir = self._sessions_index[sid]
        if (sdir / "session_manifest.json").is_file() and (
            "phase48" in str(sdir) or sid.startswith("p48_")
        ):
            ok, issues = verify_phase48_session_integrity(sdir)
        else:
            ver = verify_session_integrity(sdir)
            ok = bool(ver.get("is_valid", False))
            issues = list(ver.get("errors", []))
        msg = f"=== SESSION INTEGRITY ({sid}) ===\nValid: {ok}"
        if issues:
            msg += "\nIssues:\n" + "\n".join(f"  - {i}" for i in issues)
        self._append_log(msg)

    def export_selected_evidence(self, export_dir: Path | None = None) -> Path | None:
        sid = self._selected_session_id()
        if not sid or sid not in self._sessions_index:
            messagebox.showinfo("No Session", "No observation session selected or available.")
            return None
        sdir = self._sessions_index[sid]
        if export_dir is None:
            chosen = filedialog.askdirectory(
                title=f"Export Sealed Evidence for {sid}",
                initialdir=str(self.root / PHASE46_EXPORTS_REL),
            )
            target_base = Path(chosen) if chosen else (self.root / PHASE46_EXPORTS_REL)
        else:
            target_base = Path(export_dir)

        out_dir = target_base / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        if "phase48" in str(sdir) or sid.startswith("p48_"):
            for fname in (
                "session.json",
                "session_manifest.json",
                "inputs.jsonl",
                "predictions.jsonl",
                "outcomes.jsonl",
                "diagnostics.jsonl",
            ):
                src_f = sdir / fname
                if src_f.exists():
                    shutil.copy2(src_f, out_dir / fname)
            ok, issues = verify_phase48_session_integrity(sdir)
            meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
            (out_dir / "verification_result.json").write_text(
                json.dumps(
                    {
                        "session_id": sid,
                        "verified_at_utc": datetime.now(UTC).isoformat(),
                        "is_valid": ok,
                        "issues": issues,
                        "is_genuine_mode": bool(meta.get("is_genuine_mode", False)),
                        "evidence_classification": meta.get("evidence_classification"),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self._append_log(f"[Export] Phase 48 sealed evidence exported to {out_dir}")
            messagebox.showinfo(
                "Export Complete",
                f"Phase 48 sealed evidence exported to:\n{out_dir}",
            )
            return out_dir

        diag = compute_phase46_session_diagnostics(session_id=sid, session_dir=sdir)
        json_p, md_p = write_phase46_synthetic_diagnostics_reports(diag, out_dir=out_dir)

        for fname in ("session.json", "manifest.json", "predictions.jsonl", "outcomes.jsonl"):
            src_f = sdir / fname
            if src_f.exists():
                shutil.copy2(src_f, out_dir / fname)

        ver = verify_session_integrity(sdir)
        ok = bool(ver.get("is_valid", False))
        issues = list(ver.get("errors", []))
        ver_path = out_dir / "verification_result.json"
        ver_path.write_text(
            json.dumps(
                {
                    "session_id": sid,
                    "verified_at_utc": datetime.now(UTC).isoformat(),
                    "is_valid": ok,
                    "issues": issues,
                    "is_synthetic": diag.is_synthetic,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self._append_log(
            f"[Phase 46] Exported verified sealed evidence for {sid} to {out_dir} "
            f"(integrity_valid={ok}, json={json_p.name}, md={md_p.name})."
        )
        return out_dir

    def _append_log(self, text: str) -> None:
        widget = getattr(self.panel, "obs_log_text", None)
        if widget is None:
            return
        widget.configure(state="normal")
        existing = widget.get("1.0", "end-1c")
        lines = (existing.splitlines() if existing else []) + text.splitlines()
        # Keep bounded output (last 120 lines)
        if len(lines) > 120:
            lines = lines[-120:]
        widget.delete("1.0", "end")
        widget.insert("end", "\n".join(lines) + "\n")
        widget.yview_moveto(1.0)
        widget.configure(state="disabled")
