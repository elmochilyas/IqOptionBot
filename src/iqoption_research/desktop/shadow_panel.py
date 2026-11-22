"""Desktop Shadow Evaluation panel for offline candidate registration, replay, and diagnostics."""

from __future__ import annotations

import contextlib
import json
import tkinter as tk
from concurrent.futures import Future
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any

from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    ShadowPredictionJournal,
)
from iqoption_research.shadow.registry import (
    FrozenRegistrationRecord,
    ShadowModelRegistry,
)
from iqoption_research.shadow.replay import (
    HistoricalReplayEngine,
    ReplayConfig,
    ReplayResult,
)

if TYPE_CHECKING:
    from iqoption_research.desktop.app import Dashboard


class ShadowEvaluationPanel:
    """Offline Shadow Evaluation tab inside Research Workstation V2."""

    def __init__(self, dashboard: Dashboard, notebook: ttk.Notebook) -> None:
        self.dashboard = dashboard
        self.root = dashboard.root
        self.registry = ShadowModelRegistry(self.root)
        self.dataset_manager = ResearchDatasetManager(self.root)
        self.replay_engine = HistoricalReplayEngine(self.registry, root=self.root)

        self.candidates: list[FrozenRegistrationRecord] = []
        self.active_future: Future[Any] | None = None
        self.is_cancelled = False
        self.current_journal: ShadowPredictionJournal | None = None

        # Main Tab Frame
        self.frame = ttk.Frame(notebook, padding=12)
        notebook.add(self.frame, text="Shadow Evaluation")

        # Top Header & Safety Invariants
        header = ttk.Frame(self.frame)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(
            header,
            text="AI Shadow Evaluation · Offline Prospective Foundation",
            font=("Segoe UI", 13, "bold"),
            foreground="#17324d",
        ).pack(anchor="w")
        self.warning_label = ttk.Label(
            header,
            text=(
                "OFFLINE RESEARCH ONLY    |    ZERO BROKER NETWORK    |    "
                "NO DEMONSTRATED EDGE    |    NO LIVE EXECUTION AUTHORITY"
            ),
            font=("Segoe UI", 9, "bold"),
            foreground="#9c3414",
        )
        self.warning_label.pack(anchor="w", pady=(2, 0))

        # Sub-Notebook with 3 Tabs: Candidates, Replay & Evaluation, Journal Viewer
        self.sub_notebook = ttk.Notebook(self.frame)
        self.sub_notebook.pack(fill="both", expand=True, pady=6)

        # Tab 1: Frozen Candidates
        self.cand_frame = ttk.Frame(self.sub_notebook, padding=10)
        self.sub_notebook.add(self.cand_frame, text="Frozen Candidates")
        self._build_candidate_tab()

        # Tab 2: Offline Replay & Metrics
        self.replay_frame = ttk.Frame(self.sub_notebook, padding=10)
        self.sub_notebook.add(self.replay_frame, text="Historical Replay")
        self._build_replay_tab()

        # Tab 3: Prediction Journal Viewer
        self.journal_frame = ttk.Frame(self.sub_notebook, padding=10)
        self.sub_notebook.add(self.journal_frame, text="Prediction Journal")
        self._build_journal_tab()

        # Tab 4: Prospective Observation
        self.obs_frame = ttk.Frame(self.sub_notebook, padding=10)
        self.sub_notebook.add(self.obs_frame, text="Prospective Observation")
        self._build_observation_tab()

        # Initial refresh
        self.refresh()

    def _build_candidate_tab(self) -> None:
        top_bar = ttk.Frame(self.cand_frame)
        top_bar.pack(fill="x", pady=(0, 8))

        self.btn_refresh_cand = ttk.Button(
            top_bar, text="Refresh Registered Candidates", command=self.refresh
        )
        self.btn_refresh_cand.pack(side="left", padx=(0, 8))

        self.btn_register_lab_model = ttk.Button(
            top_bar,
            text="Register Candidate from Model Laboratory",
            command=self._prompt_register_candidate,
        )
        self.btn_register_lab_model.pack(side="left", padx=(0, 8))

        # Candidate Treeview
        tree_frame = ttk.Frame(self.cand_frame)
        tree_frame.pack(fill="both", expand=True, pady=4)

        cols = ("model_id", "family", "threshold", "cutoff_utc", "verdict", "hash")
        self.cand_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=8)
        self.cand_tree.heading("model_id", text="Frozen Model ID")
        self.cand_tree.heading("family", text="Family")
        self.cand_tree.heading("threshold", text="Threshold")
        self.cand_tree.heading("cutoff_utc", text="Data Cutoff")
        self.cand_tree.heading("verdict", text="Lab Verdict")
        self.cand_tree.heading("hash", text="Bundle SHA-256")

        self.cand_tree.column("model_id", width=280)
        self.cand_tree.column("family", width=120)
        self.cand_tree.column("threshold", width=80)
        self.cand_tree.column("cutoff_utc", width=160)
        self.cand_tree.column("verdict", width=180)
        self.cand_tree.column("hash", width=140)

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.cand_tree.yview)
        self.cand_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.cand_tree.pack(fill="both", expand=True)

        self.cand_tree.bind("<<TreeviewSelect>>", self._on_candidate_selected)

        # Candidate Details Text
        self.cand_details_text = tk.Text(
            self.cand_frame, height=8, wrap="word", font=("Consolas", 10), state="disabled"
        )
        self.cand_details_text.pack(fill="both", expand=True, pady=(6, 0))

    def _build_replay_tab(self) -> None:
        controls_frame = ttk.LabelFrame(self.replay_frame, text="Replay Configuration", padding=10)
        controls_frame.pack(fill="x", pady=(0, 8))

        # Row 1: Candidate Selection
        r1 = ttk.Frame(controls_frame)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Selected Frozen Model:", width=22).pack(side="left")
        self.combo_replay_model = ttk.Combobox(r1, state="readonly", width=45)
        self.combo_replay_model.pack(side="left", padx=4)

        # Row 2: Snapshot Selection
        r2 = ttk.Frame(controls_frame)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Dataset Snapshot:", width=22).pack(side="left")
        self.combo_replay_snapshot = ttk.Combobox(r2, state="readonly", width=45)
        self.combo_replay_snapshot.pack(side="left", padx=4)

        # Row 3: Max Candles & Controls
        r3 = ttk.Frame(controls_frame)
        r3.pack(fill="x", pady=4)
        ttk.Label(r3, text="Max Candles to Replay:", width=22).pack(side="left")
        self.var_max_candles = tk.StringVar(value="500")
        ttk.Entry(r3, textvariable=self.var_max_candles, width=10).pack(side="left", padx=4)

        self.btn_start_replay = ttk.Button(r3, text="Run Offline Replay", command=self.start_replay)
        self.btn_run_replay = self.btn_start_replay
        self.btn_start_replay.pack(side="left", padx=(16, 4))

        self.btn_cancel_replay = ttk.Button(
            r3, text="Cancel Replay", command=self.cancel_replay, state="disabled"
        )
        self.btn_cancel_replay.pack(side="left", padx=4)

        # Progress bar
        self.replay_progress_bar = ttk.Progressbar(self.replay_frame, mode="determinate")
        self.replay_progress_bar.pack(fill="x", pady=4)
        self.lbl_replay_status = ttk.Label(
            self.replay_frame, text="Ready for offline historical replay.", foreground="#20364a"
        )
        self.lbl_replay_status.pack(anchor="w", pady=(0, 6))

        # Replay Results Text Area
        self.replay_results_text = tk.Text(
            self.replay_frame, wrap="word", font=("Consolas", 10), state="disabled"
        )
        self.replay_results_text.pack(fill="both", expand=True)

    def _build_journal_tab(self) -> None:
        top_bar = ttk.Frame(self.journal_frame)
        top_bar.pack(fill="x", pady=(0, 6))

        ttk.Label(top_bar, text="Session ID:").pack(side="left", padx=(0, 4))
        self.combo_sessions = ttk.Combobox(top_bar, state="readonly", width=35)
        self.combo_sessions.pack(side="left", padx=(0, 8))
        self.combo_sessions.bind("<<ComboboxSelected>>", self._on_session_selected)

        self.btn_refresh_journal = ttk.Button(
            top_bar, text="Load Journal", command=self._load_selected_session_journal
        )
        self.btn_refresh_journal.pack(side="left", padx=(0, 8))

        self.lbl_journal_stats = ttk.Label(
            top_bar, text="Total: 0 | Resolved: 0 | Pending: 0", font=("Segoe UI", 9, "bold")
        )
        self.lbl_journal_stats.pack(side="right")

        # Predictions Treeview
        tree_frame = ttk.Frame(self.journal_frame)
        tree_frame.pack(fill="both", expand=True, pady=4)

        cols = ("id", "cutoff_utc", "decision", "p_up", "p_down", "status", "market_dir", "win")
        self.journal_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=10)
        self.journal_tree.heading("id", text="Prediction ID")
        self.journal_tree.heading("cutoff_utc", text="Cutoff UTC")
        self.journal_tree.heading("decision", text="Decision")
        self.journal_tree.heading("p_up", text="P(UP)")
        self.journal_tree.heading("p_down", text="P(DOWN)")
        self.journal_tree.heading("status", text="Outcome Status")
        self.journal_tree.heading("market_dir", text="Actual Dir")
        self.journal_tree.heading("win", text="Trade Win")

        self.journal_tree.column("id", width=200)
        self.journal_tree.column("cutoff_utc", width=150)
        self.journal_tree.column("decision", width=80)
        self.journal_tree.column("p_up", width=70)
        self.journal_tree.column("p_down", width=70)
        self.journal_tree.column("status", width=140)
        self.journal_tree.column("market_dir", width=80)
        self.journal_tree.column("win", width=80)

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.journal_tree.yview)
        self.journal_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.journal_tree.pack(fill="both", expand=True)

    def refresh(self) -> None:
        """Scan and populate registered candidates, snapshots, and replay sessions."""
        self.candidates = self.registry.list_candidates()
        self._update_candidate_tree()

        # Update comboboxes
        cand_ids = [c.model_id for c in self.candidates]
        self.combo_replay_model["values"] = cand_ids
        if cand_ids and not self.combo_replay_model.get():
            self.combo_replay_model.current(0)

        snapshots = self.dataset_manager.list_snapshots()
        snap_ids = [s.manifest.snapshot_id for s in snapshots]
        self.combo_replay_snapshot["values"] = snap_ids
        if snap_ids and not self.combo_replay_snapshot.get():
            self.combo_replay_snapshot.current(0)

        # Update prospective observation comboboxes
        if hasattr(self, "combo_obs_model"):
            self.combo_obs_model["values"] = cand_ids
            if cand_ids and not self.combo_obs_model.get():
                self.combo_obs_model.current(0)

        if hasattr(self, "combo_obs_account"):
            with contextlib.suppress(Exception):
                from iqoption_research.auth.vault import WindowsCredentialVault

                store = WindowsCredentialVault().account_store()
                acc_labels = [f"{a.label} [{a.account_id[:8]}]" for a in store.list_accounts()]
                self.combo_obs_account["values"] = acc_labels
                if acc_labels and not self.combo_obs_account.get():
                    self.combo_obs_account.current(0)

        self._refresh_session_list()

    def _update_candidate_tree(self) -> None:
        for item in self.cand_tree.get_children():
            self.cand_tree.delete(item)
        for c in self.candidates:
            self.cand_tree.insert(
                "",
                "end",
                values=(
                    c.model_id,
                    c.model_family,
                    f"{c.confidence_threshold:.2f}",
                    c.registered_at_utc[:19],
                    c.verdict,
                    c.bundle_sha256[:12] + "...",
                ),
            )

    def _on_candidate_selected(self, _event: Any) -> None:
        selected = self.cand_tree.selection()
        if not selected:
            return
        item = self.cand_tree.item(selected[0])
        model_id = item["values"][0]
        c = self.registry.get_candidate(model_id)
        if c is not None:
            self._write_text(
                self.cand_details_text,
                json.dumps(c.to_dict(), indent=2, sort_keys=True),
            )

    def _refresh_session_list(self) -> None:
        sessions_dir = self.root / DEFAULT_SHADOW_SESSIONS_DIR
        sessions: list[str] = []
        if sessions_dir.is_dir():
            for p in sorted(sessions_dir.glob("*"), reverse=True):
                if p.is_dir() and (p / "predictions.jsonl").is_file():
                    sessions.append(p.name)
        self.combo_sessions["values"] = sessions
        if sessions and not self.combo_sessions.get():
            self.combo_sessions.current(0)

    def _on_session_selected(self, _event: Any) -> None:
        self._load_selected_session_journal()

    def _load_selected_session_journal(self) -> None:
        session_id = self.combo_sessions.get()
        if not session_id:
            return
        s_dir = self.root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
        journal = ShadowPredictionJournal(session_id, session_dir=s_dir)
        self.current_journal = journal

        for item in self.journal_tree.get_children():
            self.journal_tree.delete(item)

        preds = journal.list_predictions()
        for p in preds:
            win_str = "-"
            if p.is_winning_trade is True:
                win_str = "WIN"
            elif p.is_winning_trade is False:
                win_str = "LOSS" if p.outcome_status == "RESOLVED_LOSS" else "TIE"

            self.journal_tree.insert(
                "",
                "end",
                values=(
                    p.prediction_id,
                    p.feature_cutoff_utc[:19],
                    p.decision,
                    f"{p.probabilities.get('UP', 0.0):.4f}",
                    f"{p.probabilities.get('DOWN', 0.0):.4f}",
                    p.outcome_status,
                    p.market_direction or "-",
                    win_str,
                ),
            )

        stats_txt = (
            f"Total: {journal.total_count} | "
            f"Resolved: {journal.resolved_count} | "
            f"Pending: {journal.pending_count}"
        )
        self.lbl_journal_stats.config(text=stats_txt)

    def _prompt_register_candidate(self) -> None:
        lab_models = self.registry.lab.list_candidate_models()
        if not lab_models:
            messagebox.showinfo(
                "No Candidate Models",
                "No candidate models found in data/candidate_models/ to register.",
            )
            return

        model_id = lab_models[0]["model_id"]
        try:
            rec = self.registry.register_candidate(model_id)
            messagebox.showinfo(
                "Candidate Registered",
                f"Successfully registered frozen candidate:\n{rec.model_id}\n\n"
                f"Bundle Hash: {rec.bundle_sha256[:16]}...\n"
                f"Verdict: {rec.verdict}",
            )
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Registration Error", f"Failed to register candidate: {exc}")

    def start_replay(self) -> None:
        model_id = self.combo_replay_model.get()
        snapshot_id = self.combo_replay_snapshot.get()
        if not model_id or not snapshot_id:
            messagebox.showwarning("Selection Required", "Please select both a model and snapshot.")
            return

        try:
            max_c = int(self.var_max_candles.get().strip())
        except ValueError:
            max_c = 500

        self.btn_start_replay.config(state="disabled")
        self.btn_cancel_replay.config(state="normal")
        self.is_cancelled = False
        self.replay_progress_bar["value"] = 0
        self.lbl_replay_status.config(text=f"Loading snapshot '{snapshot_id}'...")

        def task() -> ReplayResult:
            snap = next(
                s
                for s in self.dataset_manager.list_snapshots()
                if s.manifest.snapshot_id == snapshot_id
            )
            candles = snap.load_candles()
            cfg = ReplayConfig(
                model_id=model_id,
                snapshot_id=snapshot_id,
                max_candles=max_c,
            )
            return self.replay_engine.run_replay(
                cfg,
                candles,
                progress_cb=self._on_replay_progress,
                cancel_token=lambda: self.is_cancelled,
            )

        self.active_future = self.dashboard.executor.submit(task)

    def cancel_replay(self) -> None:
        self.is_cancelled = True
        self.lbl_replay_status.config(text="Cancelling replay...")

    def _build_observation_tab(self) -> None:
        controls_frame = ttk.LabelFrame(
            self.obs_frame, text="Supervised Prospective Observation Setup", padding=10
        )
        controls_frame.pack(fill="x", pady=(0, 8))

        self.obs_safety_label = ttk.Label(
            controls_frame,
            text="EXECUTION_AUTHORITY: NONE · NO DEMONSTRATED EDGE · STRICTLY READ-ONLY",
            font=("Segoe UI", 9, "bold"),
            foreground="#9c3414",
        )
        self.obs_safety_label.pack(anchor="w", pady=(0, 6))

        # Row 1: Candidate Model Selection
        r1 = ttk.Frame(controls_frame)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Frozen Candidate Model:", width=24).pack(side="left")
        self.combo_obs_model = ttk.Combobox(r1, state="readonly", width=45)
        self.combo_obs_model.pack(side="left", padx=4)

        # Row 2: Mode Selection (Synthetic Simulator vs Live Supervised)
        r2 = ttk.Frame(controls_frame)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Observation Feed Mode:", width=24).pack(side="left")
        self.var_obs_mode = tk.StringVar(value="synthetic")
        ttk.Radiobutton(
            r2,
            text="Offline Simulator (Synthetic Streaming)",
            variable=self.var_obs_mode,
            value="synthetic",
            command=self._on_obs_mode_changed,
        ).pack(side="left", padx=4)
        ttk.Radiobutton(
            r2,
            text="Supervised Live (Windows Vault)",
            variable=self.var_obs_mode,
            value="live",
            command=self._on_obs_mode_changed,
        ).pack(side="left", padx=8)

        # Row 3: Account & Duration
        r3 = ttk.Frame(controls_frame)
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Vault Account (Live Mode):", width=24).pack(side="left")
        self.combo_obs_account = ttk.Combobox(r3, state="disabled", width=30)
        self.combo_obs_account.pack(side="left", padx=4)

        ttk.Label(r3, text="Duration (Minutes):", width=18).pack(side="left", padx=(16, 4))
        self.combo_obs_duration = ttk.Combobox(
            r3, state="readonly", width=8, values=["15", "30", "60", "120"]
        )
        self.combo_obs_duration.set("15")
        self.combo_obs_duration.pack(side="left", padx=4)

        # Row 4: Action Buttons
        r4 = ttk.Frame(controls_frame)
        r4.pack(fill="x", pady=(6, 2))
        self.btn_start_obs = ttk.Button(
            r4, text="Start Observation Session", command=self.start_observation
        )
        self.btn_start_obs.pack(side="left", padx=(0, 6))

        self.btn_cancel_obs = ttk.Button(
            r4, text="Stop / Cancel", command=self.cancel_observation, state="disabled"
        )
        self.btn_cancel_obs.pack(side="left", padx=4)

        # Session Health & Continuity Status
        health_frame = ttk.LabelFrame(
            self.obs_frame, text="Session Health & Feed Continuity", padding=8
        )
        health_frame.pack(fill="x", pady=4)
        self.obs_progress_bar = ttk.Progressbar(health_frame, mode="determinate")
        self.obs_progress_bar.pack(fill="x", pady=2)
        self.lbl_obs_status = ttk.Label(
            health_frame,
            text=(
                "Status: IDLE    |    Feed: Disconnected    |    "
                "Valid: 0    |    Duplicates: 0    |    Gaps: 0    |    Elapsed: 0s"
            ),
            font=("Segoe UI", 9, "bold"),
            foreground="#20364a",
        )
        self.lbl_obs_status.pack(anchor="w", pady=2)

        # Real-time Metrics Table
        tree_frame = ttk.LabelFrame(
            self.obs_frame, text="Real-Time Streaming Model Evidence", padding=6
        )
        tree_frame.pack(fill="both", expand=True, pady=4)

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
        self.obs_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=5)
        self.obs_tree.heading("model_id", text="Model ID")
        self.obs_tree.heading("predictions", text="Total Preds")
        self.obs_tree.heading("pending", text="Pending")
        self.obs_tree.heading("trades", text="Trades (W/L/T)")
        self.obs_tree.heading("win_rate", text="Decisive Win%")
        self.obs_tree.heading("matched_maj", text="Matched Maj%")
        self.obs_tree.heading("ci_95", text="Wilson 95% CI")
        self.obs_tree.heading("exp_bin", text="Exp Binary")
        self.obs_tree.heading("exp_dig", text="Exp Digital")
        self.obs_tree.heading("be_gap", text="BE Gap")

        self.obs_tree.column("model_id", width=220)
        self.obs_tree.column("predictions", width=90, anchor="e")
        self.obs_tree.column("pending", width=80, anchor="e")
        self.obs_tree.column("trades", width=110, anchor="center")
        self.obs_tree.column("win_rate", width=100, anchor="e")
        self.obs_tree.column("matched_maj", width=100, anchor="e")
        self.obs_tree.column("ci_95", width=130, anchor="center")
        self.obs_tree.column("exp_bin", width=90, anchor="e")
        self.obs_tree.column("exp_dig", width=90, anchor="e")
        self.obs_tree.column("be_gap", width=80, anchor="e")

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.obs_tree.yview)
        self.obs_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.obs_tree.pack(fill="both", expand=True)

        # Log details
        self.obs_log_text = tk.Text(
            self.obs_frame, height=5, wrap="word", font=("Consolas", 9), state="disabled"
        )
        self.obs_log_text.pack(fill="both", expand=True, pady=(4, 0))

    def _on_obs_mode_changed(self) -> None:
        if self.var_obs_mode.get() == "live":
            self.combo_obs_account.config(state="readonly")
        else:
            self.combo_obs_account.config(state="disabled")

    def start_observation(self) -> None:
        model_id = self.combo_obs_model.get()
        if not model_id:
            messagebox.showwarning("Selection Required", "Please select a frozen model to observe.")
            return

        minutes = int(self.combo_obs_duration.get() or "15")
        mode = self.var_obs_mode.get()

        if mode == "live":
            acc_val = self.combo_obs_account.get()
            if not acc_val:
                messagebox.showwarning(
                    "Account Required",
                    "Please select a Windows Vault account for live observation.",
                )
                return
            acc_id = acc_val.split(" [")[-1].rstrip("]") if " [" in acc_val else acc_val

            if not messagebox.askyesno(
                "Authorize Supervised Live Observation?",
                f"You are about to launch a supervised read-only observation session for "
                f"{minutes} minutes.\n\n"
                f"Model: {model_id}\n"
                f"Account: {acc_id}\n\n"
                "Orders remain strictly disabled (Authority NONE).\n"
                "Do you authorize this session?",
                parent=self.dashboard.window,
            ):
                return

            try:
                self.dashboard.process.start_prospective_shadow(
                    model_ids=[model_id],
                    minutes=minutes,
                    account_id=acc_id,
                    synthetic=False,
                )
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror(
                    "Observation Error", f"Failed to start live observation: {exc}"
                )
                return
        else:
            try:
                self.dashboard.process.start_prospective_shadow(
                    model_ids=[model_id],
                    minutes=minutes,
                    synthetic=True,
                )
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Observation Error", f"Failed to start simulator: {exc}")
                return

        self.btn_start_obs.config(state="disabled")
        self.btn_cancel_obs.config(state="normal")

    def cancel_observation(self) -> None:
        self.dashboard.process.stop()
        self.btn_cancel_obs.config(state="disabled")

    def busy(self) -> bool:
        """Return True if an offline replay task is currently active."""
        return self.active_future is not None

    def _on_replay_progress(self, pct: int, msg: str) -> None:
        self.dashboard.window.after(0, lambda: self._update_progress_ui(pct, msg))

    def _update_progress_ui(self, pct: int, msg: str) -> None:
        self.replay_progress_bar["value"] = pct
        self.lbl_replay_status.config(text=msg)

    def tick(self) -> None:
        """Called periodically by dashboard main event loop."""
        if self.active_future is not None and self.active_future.done():
            fut = self.active_future
            self.active_future = None
            self.btn_start_replay.config(state="normal")
            self.btn_cancel_replay.config(state="disabled")
            try:
                result = fut.result()
                status_txt = (
                    f"Replay completed: {result.predictions_recorded} predictions, "
                    f"{result.outcomes_resolved} resolved."
                )
                self.lbl_replay_status.config(text=status_txt)
                self.replay_progress_bar["value"] = 100
                self._write_text(
                    self.replay_results_text,
                    json.dumps(result.to_dict(), indent=2, sort_keys=True),
                )
                self.refresh()
            except Exception as exc:  # noqa: BLE001
                self.lbl_replay_status.config(text=f"Replay failed: {exc}")
                messagebox.showerror("Replay Error", f"Replay failed: {exc}")

        # Update prospective shadow observation progress if active
        if (
            self.dashboard.process.child is not None
            and self.dashboard.process.kind == "prospective-shadow"
        ):
            live = self.dashboard.process.live_state()
            conn = live.get("connection_status", "OBSERVING")
            elapsed = live.get("elapsed_seconds", 0.0)
            valid = live.get("valid_candles_count", 0)
            dups = live.get("duplicate_boundaries_count", 0)
            gaps = live.get("continuity_gaps_count", 0)
            status_txt = (
                f"Status: {conn}    |    Feed: Active    |    "
                f"Valid: {valid}    |    Duplicates: {dups}    |    "
                f"Gaps: {gaps}    |    Elapsed: {elapsed:.0f}s"
            )
            self.lbl_obs_status.config(text=status_txt)
            models_data = live.get("models", {})
            for item in self.obs_tree.get_children():
                self.obs_tree.delete(item)
            for mid, mstate in models_data.items():
                w = mstate.get("wins_count", 0)
                loss = mstate.get("losses_count", 0)
                tie = mstate.get("ties_count", 0)
                wlt = f"{w}/{loss}/{tie}"
                ci = mstate.get("wilson_ci_decisive", (0.0, 1.0))
                ci_str = f"[{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]"
                maj = mstate.get("matched_baseline_accuracy")
                maj_str = f"{maj * 100:.2f}%" if maj is not None else "N/A"
                win_dec = mstate.get("win_rate_decisive", 0.0) * 100.0
                exp_b = mstate.get("expectancy_binary_pct", 0.0)
                exp_d = mstate.get("expectancy_digital_pct", 0.0)
                be_gap = mstate.get("break_even_gap_binary", 0.0) * 100.0
                self.obs_tree.insert(
                    "",
                    "end",
                    values=(
                        mid,
                        mstate.get("total_predictions", 0),
                        mstate.get("pending_predictions", 0),
                        wlt,
                        f"{win_dec:.2f}%",
                        maj_str,
                        ci_str,
                        f"{exp_b:+.2f}%",
                        f"{exp_d:+.2f}%",
                        f"{be_gap:+.2f}%",
                    ),
                )
        elif (
            self.dashboard.process.child is None
            and getattr(self, "btn_start_obs", None) is not None
            and self.btn_start_obs["state"] == "disabled"
            and not self.busy()
        ):
            self.btn_start_obs.config(state="normal")
            self.btn_cancel_obs.config(state="disabled")

    @staticmethod
    def _write_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.yview_moveto(0)
        widget.configure(state="disabled")
