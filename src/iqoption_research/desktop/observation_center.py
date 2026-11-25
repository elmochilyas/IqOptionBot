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
        self._fixture_label_to_tf: dict[str, int] = {
            f"{fx.fixture_id} ({fx.timeframe_label} / H{fx.horizon_bars})": fx.timeframe_seconds
            for fx in PHASE46_SYNTHETIC_FIXTURES
        }

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
        if self._poll_after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.panel.dashboard.window.after_cancel(self._poll_after_id)
            self._poll_after_id = None

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def build_ui(self, parent: ttk.Frame) -> None:
        p = self.panel

        # 1) Safety banner
        controls_frame = ttk.LabelFrame(parent, text="Observation Session Configuration", padding=5)
        controls_frame.pack(fill="x", pady=(0, 3))

        p.obs_safety_label = ttk.Label(
            controls_frame,
            text=(
                "Phase 46 Observation Center: EXECUTION_AUTHORITY=NONE | NO_DEMONSTRATED_EDGE. "
                "Offline Synthetic Mode validates lifecycle plumbing only. "
                "Genuine prospective observation is BLOCKED until a candidate qualifies and contract is authorized."
            ),
            font=("Segoe UI", 8, "bold"),
            foreground="#9c3414",
            wraplength=1040,
            justify="left",
        )
        p.obs_safety_label.pack(anchor="w", pady=(0, 3))

        # Row 1: Mode & Eligibility
        r1 = ttk.Frame(controls_frame)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Observation Mode:", width=20).pack(side="left")
        p.var_obs_mode = tk.StringVar(value="synthetic")
        p.rb_obs_synth = ttk.Radiobutton(
            r1,
            text="Offline Synthetic Demonstration (Phase 46)",
            variable=p.var_obs_mode,
            value="synthetic",
            command=self.on_mode_changed,
        )
        p.rb_obs_synth.pack(side="left", padx=4)
        p.rb_obs_live = ttk.Radiobutton(
            r1,
            text="Genuine Observation (Blocked — Unauthorized)",
            variable=p.var_obs_mode,
            value="live",
            command=self.on_mode_changed,
        )
        p.rb_obs_live.pack(side="left", padx=8)

        p.lbl_obs_eligibility = ttk.Label(
            r1,
            text="SYNTHETIC_FIXTURE (Lifecycle Demo Only)",
            font=("Segoe UI", 9, "bold"),
            foreground="#1b4d3e",
        )
        p.lbl_obs_eligibility.pack(side="right", padx=4)

        # Row 2: Model, Timeframe, Prediction Duration
        r2 = ttk.Frame(controls_frame)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Model / Fixture:", width=20).pack(side="left")
        p.combo_obs_model = ttk.Combobox(r2, state="readonly", width=44)
        p.combo_obs_model.pack(side="left", padx=4)
        p.combo_obs_model.bind("<<ComboboxSelected>>", self.on_model_selected)

        ttk.Label(r2, text="Timeframe:", width=11).pack(side="left", padx=(10, 2))
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
        p.lbl_obs_horizon.pack(side="left", padx=(12, 4))

        # Row 3: Account (Genuine), Duration, Pace
        r3 = ttk.Frame(controls_frame)
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Vault Account (Genuine):", width=20).pack(side="left")
        p.combo_obs_account = ttk.Combobox(r3, state="disabled", width=28)
        p.combo_obs_account.pack(side="left", padx=4)

        ttk.Label(r3, text="Duration (min):", width=14).pack(side="left", padx=(10, 2))
        p.combo_obs_duration = ttk.Combobox(
            r3, state="readonly", width=6, values=["15", "30", "60", "120"]
        )
        p.combo_obs_duration.set("15")
        p.combo_obs_duration.pack(side="left", padx=4)

        ttk.Label(r3, text="Sim Pace:", width=9).pack(side="left", padx=(10, 2))
        p.combo_obs_pace = ttk.Combobox(
            r3, state="readonly", width=21, values=list(_PACE_MAP.keys())
        )
        p.combo_obs_pace.set("Interactive (120ms/bar)")
        p.combo_obs_pace.pack(side="left", padx=4)

        # Row 3b: Campaign & Slot (kept for Phase 40B/40C compatibility)
        r3b = ttk.Frame(controls_frame)
        r3b.pack(fill="x", pady=2)
        ttk.Label(r3b, text="Research Campaign:", width=20).pack(side="left")
        p.combo_obs_campaign = ttk.Combobox(r3b, state="readonly", width=24)
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

        # Row 4: Action Buttons
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

        # 2) Live Monitoring & Chart
        live_frame = ttk.LabelFrame(parent, text="Live Monitoring & Performance", padding=5)
        live_frame.pack(fill="x", pady=3)

        p.obs_progress_bar = ttk.Progressbar(live_frame, mode="determinate")
        p.obs_progress_bar.pack(fill="x", pady=(0, 4))

        p.lbl_obs_status = ttk.Label(
            live_frame,
            text="State: IDLE | Source: Disconnected | Elapsed: 0.0s | Latest Candle: N/A | Latest Pred: N/A",
            font=("Segoe UI", 9, "bold"),
            foreground="#20364a",
            wraplength=1040,
        )
        p.lbl_obs_status.pack(anchor="w", pady=2)

        p.lbl_obs_kpis = ttk.Label(
            live_frame,
            text=(
                "Opportunities: 0 | Selected: 0 | Pending: 0 | Resolved: 0 (W:0 / L:0 / T:0) | "
                "Abstentions: 0 | Accuracy: N/A | Coverage: N/A | Baseline: N/A"
            ),
            font=("Consolas", 9),
            foreground="#17324d",
        )
        p.lbl_obs_kpis.pack(anchor="w", pady=2)

        p.lbl_obs_payout = ttk.Label(
            live_frame,
            text="Hypothetical 85% Payout Sensitivity (NOT actual broker settlement): N/A",
            font=("Segoe UI", 8, "italic"),
            foreground="#5a6872",
        )
        p.lbl_obs_payout.pack(anchor="w", pady=(0, 4))

        # Split area: Treeview (left) + Cumulative Accuracy Chart (right)
        mid_frame = ttk.Frame(live_frame)
        mid_frame.pack(fill="both", expand=True)

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
        p.obs_tree = ttk.Treeview(mid_frame, columns=cols, show="headings", height=3)
        headers = {
            "model_id": ("Model ID", 180, "w"),
            "predictions": ("Total", 55, "e"),
            "pending": ("Pending", 60, "e"),
            "trades": ("W/L/T", 75, "center"),
            "win_rate": ("Win%", 70, "e"),
            "matched_maj": ("Baseline%", 75, "e"),
            "ci_95": ("Wilson 95%", 115, "center"),
            "exp_bin": ("Exp Bin", 70, "e"),
            "exp_dig": ("Exp Dig", 70, "e"),
            "be_gap": ("BE Gap", 65, "e"),
        }
        for col, (title, w, anc) in headers.items():
            p.obs_tree.heading(col, text=title)
            p.obs_tree.column(col, width=w, anchor=cast(Literal["w", "e", "center"], anc))
        p.obs_tree.pack(side="left", fill="both", expand=True, padx=(0, 6))

        p.obs_chart_canvas = tk.Canvas(
            mid_frame,
            width=300,
            height=90,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#cfd8e3",
        )
        p.obs_chart_canvas.pack(side="right", fill="y")
        self._draw_accuracy_chart([])

        # 3) Session History & Evidence
        hist_frame = ttk.LabelFrame(
            parent, text="Observation Session History & Sealed Evidence", padding=5
        )
        hist_frame.pack(fill="both", expand=True, pady=3)

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

        hist_cols = ("session_id", "provenance", "tf", "model_id", "state", "resolved", "accuracy")
        p.obs_history_tree = ttk.Treeview(hist_frame, columns=hist_cols, show="headings", height=3)
        for col, title, w, anc in [
            ("session_id", "Session ID", 220, "w"),
            ("provenance", "Provenance", 100, "center"),
            ("tf", "TF", 50, "center"),
            ("model_id", "Candidate / Fixture", 220, "w"),
            ("state", "State", 130, "center"),
            ("resolved", "Resolved", 70, "e"),
            ("accuracy", "Accuracy", 80, "e"),
        ]:
            p.obs_history_tree.heading(col, text=title)
            p.obs_history_tree.column(col, width=w, anchor=cast(Literal["w", "e", "center"], anc))
        p.obs_history_tree.pack(fill="both", expand=True)

        # Bounded diagnostics / event log
        p.obs_log_text = tk.Text(
            parent, height=3, wrap="word", font=("Consolas", 9), state="disabled"
        )
        p.obs_log_text.pack(fill="both", expand=True, pady=(3, 0))

        self.start_polling()
        self.refresh_models()
        self.refresh_session_history()
        self.on_mode_changed()

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
        if mode == "live":
            p.combo_obs_account.config(state="readonly")
            p.lbl_obs_eligibility.config(
                text="BLOCKED: Phase 46 contract UNAUTHORIZED (0 qualified candidates in WS2)",
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

    def cancel_synthetic_observation(self) -> None:
        self._cancel_event.set()
        if self._observer_ref:
            with contextlib.suppress(Exception):
                self._observer_ref[0].cancel()
        self.panel.btn_cancel_obs.config(state="disabled")
        self._append_log("[Phase 46] Cancellation requested; sealing partial evidence...")

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
        p.lbl_obs_kpis.config(
            text=(
                f"Opportunities: {diag.total_predictions} | Selected: {diag.total_predictions - diag.skipped_abstain_count} | "
                f"Pending: {diag.pending_count} | Resolved: {diag.resolved_count} "
                f"(W:{diag.win_count} / L:{diag.loss_count} / T:{diag.tie_count}) | "
                f"Abstentions: {diag.skipped_abstain_count} | "
                f"Accuracy: {diag.win_rate_decisive * 100:.2f}% | Coverage: {diag.coverage * 100:.1f}%"
            )
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

    # ------------------------------------------------------------------
    # Cumulative Accuracy Chart (Tkinter Canvas)
    # ------------------------------------------------------------------
    def _draw_accuracy_chart(self, series: list[float]) -> None:
        canvas = getattr(self.panel, "obs_chart_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        w = int(canvas.cget("width"))
        h = int(canvas.cget("height"))
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
        for rel in (f"data/phase46/{DEFAULT_SHADOW_SESSIONS_DIR}", DEFAULT_SHADOW_SESSIONS_DIR):
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
