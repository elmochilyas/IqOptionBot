"""Desktop Model Laboratory panel for offline dataset preparation, training, and evaluation."""

from __future__ import annotations

import json
import tkinter as tk
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any

from iqoption_research.ml.laboratory import (
    ExperimentConfig,
    ExperimentResult,
    ModelLaboratory,
)
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
    ResearchDatasetSnapshot,
)

if TYPE_CHECKING:
    from iqoption_research.desktop.app import Dashboard


class ModelLaboratoryPanel:
    """Offline Model Laboratory tab inside Research Workstation V2."""

    def __init__(self, dashboard: Dashboard, notebook: ttk.Notebook) -> None:
        self.dashboard = dashboard
        self.root = dashboard.root
        self.manager = ResearchDatasetManager(self.root)
        self.lab = ModelLaboratory(self.root)

        self.snapshots: list[ResearchDatasetSnapshot] = []
        self.candidate_models: list[dict[str, Any]] = []
        self.active_future: Future[Any] | None = None
        self.is_cancelled = False

        # Main Tab Frame
        self.frame = ttk.Frame(notebook, padding=12)
        notebook.add(self.frame, text="Model Laboratory")

        # Top Header & Safety Invariants
        header = ttk.Frame(self.frame)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(
            header,
            text="Model Laboratory · Offline Research & Evaluation",
            font=("Segoe UI", 13, "bold"),
            foreground="#17324d",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "OFFLINE RESEARCH ONLY    |    ZERO BROKER NETWORK    |    "
                "NO DEMONSTRATED EDGE    |    Phase 38 Model Frozen"
            ),
            font=("Segoe UI", 9, "bold"),
            foreground="#9c3414",
        ).pack(anchor="w", pady=(2, 0))

        # Sub-Notebook with 3 Tabs: Datasets, Training, Candidate Models
        self.sub_notebook = ttk.Notebook(self.frame)
        self.sub_notebook.pack(fill="both", expand=True, pady=6)

        # Tab 1: Datasets
        self.dataset_frame = ttk.Frame(self.sub_notebook, padding=10)
        self.sub_notebook.add(self.dataset_frame, text="Research Datasets")
        self._build_dataset_tab()

        # Tab 2: Training & Experiments
        self.train_frame = ttk.Frame(self.sub_notebook, padding=10)
        self.sub_notebook.add(self.train_frame, text="Model Training")
        self._build_train_tab()

        # Tab 3: Candidate Models & Evaluation
        self.eval_frame = ttk.Frame(self.sub_notebook, padding=10)
        self.sub_notebook.add(self.eval_frame, text="Candidate Models & Evaluation")
        self._build_eval_tab()

        # Initial refresh
        self.refresh()

    def _build_dataset_tab(self) -> None:
        top_bar = ttk.Frame(self.dataset_frame)
        top_bar.pack(fill="x", pady=(0, 8))

        self.btn_refresh_datasets = ttk.Button(
            top_bar, text="Refresh Datasets", command=self.refresh
        )
        self.btn_refresh_datasets.pack(side="left", padx=(0, 8))

        self.btn_prepare_snapshot = ttk.Button(
            top_bar,
            text="Prepare Research Snapshot (Deduplicate & Quarantine)",
            command=self.prepare_snapshot,
        )
        self.btn_prepare_snapshot.pack(side="left", padx=(0, 8))

        self.dataset_status_label = ttk.Label(
            self.dataset_frame, text="Scanning local datasets...", wraplength=1050
        )
        self.dataset_status_label.pack(anchor="w", pady=4)

        select_bar = ttk.Frame(self.dataset_frame)
        select_bar.pack(fill="x", pady=6)
        ttk.Label(select_bar, text="Select Snapshot:").pack(side="left", padx=(0, 8))
        self.snapshot_combo = ttk.Combobox(select_bar, state="readonly", width=55)
        self.snapshot_combo.pack(side="left", padx=(0, 8))
        self.snapshot_combo.bind("<<ComboboxSelected>>", self._on_snapshot_selected)

        # Detail text view
        self.dataset_detail_text = tk.Text(
            self.dataset_frame,
            wrap="word",
            font=("Consolas", 10),
            relief="flat",
            background="#ffffff",
            foreground="#20364a",
            padx=10,
            pady=8,
            height=16,
        )
        scroll = ttk.Scrollbar(
            self.dataset_frame,
            orient="vertical",
            command=self.dataset_detail_text.yview,
        )
        self.dataset_detail_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.dataset_detail_text.pack(fill="both", expand=True)
        self.dataset_detail_text.configure(state="disabled")

    def _build_train_tab(self) -> None:
        opts_frame = ttk.LabelFrame(self.train_frame, text="Experiment Configuration", padding=12)
        opts_frame.pack(fill="x", pady=(0, 10))

        # Row 1: Snapshot and Model Family
        r1 = ttk.Frame(opts_frame)
        r1.pack(fill="x", pady=4)
        ttk.Label(r1, text="Research Dataset:", width=18).pack(side="left")
        self.train_snapshot_combo = ttk.Combobox(r1, state="readonly", width=42)
        self.train_snapshot_combo.pack(side="left", padx=(0, 20))

        ttk.Label(r1, text="Model Family:", width=14).pack(side="left")
        self.train_model_combo = ttk.Combobox(
            r1,
            values=[
                "Random Forest (RandomForestModel)",
                "Logistic Regression (LogisticRegressionModel)",
                "HistGradientBoosting (HistGradientBoostingModel)",
                "Majority Direction Prior (Baseline)",
                "Rule-Based Momentum (Baseline)",
            ],
            state="readonly",
            width=36,
        )
        self.train_model_combo.current(0)
        self.train_model_combo.pack(side="left")

        # Row 2: Horizon and Threshold
        r2 = ttk.Frame(opts_frame)
        r2.pack(fill="x", pady=4)
        ttk.Label(r2, text="Prediction Horizon:", width=18).pack(side="left")
        self.train_horizon_combo = ttk.Combobox(
            r2, values=["1 bar (60s ahead)", "5 bars (300s ahead)"], state="readonly", width=18
        )
        self.train_horizon_combo.current(0)
        self.train_horizon_combo.pack(side="left", padx=(0, 20))

        ttk.Label(r2, text="Confidence Threshold:", width=20).pack(side="left")
        self.train_threshold_combo = ttk.Combobox(
            r2,
            values=["0.50 (no abstention)", "0.55", "0.60 (standard)", "0.65"],
            state="readonly",
            width=18,
        )
        self.train_threshold_combo.current(2)
        self.train_threshold_combo.pack(side="left", padx=(0, 20))

        ttk.Label(r2, text="Calibration Method:", width=18).pack(side="left")
        self.train_cal_combo = ttk.Combobox(
            r2, values=["Isotonic", "Sigmoid (Platt)", "None"], state="readonly", width=16
        )
        self.train_cal_combo.current(0)
        self.train_cal_combo.pack(side="left")

        # Action Buttons & Progress
        action_bar = ttk.Frame(self.train_frame)
        action_bar.pack(fill="x", pady=6)
        self.btn_start_train = ttk.Button(
            action_bar, text="Run Training Experiment", command=self.start_training
        )
        self.btn_start_train.pack(side="left", padx=(0, 8))
        self.btn_cancel_train = ttk.Button(action_bar, text="Cancel", command=self.cancel_training)
        self.btn_cancel_train.pack(side="left", padx=(0, 8))
        self.btn_cancel_train.state(["disabled"])

        self.train_progress = ttk.Progressbar(self.train_frame, maximum=100)
        self.train_progress.pack(fill="x", pady=4)
        self.train_status_label = ttk.Label(
            self.train_frame,
            text="Ready to configure and run offline candidate experiments.",
        )
        self.train_status_label.pack(anchor="w", pady=(0, 4))

        # Log output text view
        self.train_log_text = tk.Text(
            self.train_frame,
            wrap="word",
            font=("Consolas", 10),
            relief="flat",
            background="#ffffff",
            foreground="#20364a",
            padx=10,
            pady=8,
            height=12,
        )
        log_scroll = ttk.Scrollbar(
            self.train_frame, orient="vertical", command=self.train_log_text.yview
        )
        self.train_log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.train_log_text.pack(fill="both", expand=True)
        self.train_log_text.configure(state="disabled")

    def _build_eval_tab(self) -> None:
        eval_bar = ttk.Frame(self.eval_frame)
        eval_bar.pack(fill="x", pady=(0, 6))

        self.btn_refresh_models = ttk.Button(
            eval_bar, text="Refresh Candidate Models", command=self.refresh
        )
        self.btn_refresh_models.pack(side="left", padx=(0, 8))

        columns = ("model_id", "family", "created", "test_acc", "brier", "verdict")
        self.model_tree = ttk.Treeview(self.eval_frame, columns=columns, show="headings", height=6)
        self.model_tree.heading("model_id", text="Candidate Model ID")
        self.model_tree.heading("family", text="Model Family")
        self.model_tree.heading("created", text="Created UTC")
        self.model_tree.heading("test_acc", text="Test Acc")
        self.model_tree.heading("brier", text="Brier Score")
        self.model_tree.heading("verdict", text="Scientific Verdict")

        self.model_tree.column("model_id", width=340)
        self.model_tree.column("family", width=140)
        self.model_tree.column("created", width=150)
        self.model_tree.column("test_acc", width=80)
        self.model_tree.column("brier", width=90)
        self.model_tree.column("verdict", width=220)

        self.model_tree.pack(fill="x", pady=(0, 8))
        self.model_tree.bind("<<TreeviewSelect>>", self._on_model_selected)

        # Detailed evaluation text
        self.eval_detail_text = tk.Text(
            self.eval_frame,
            wrap="word",
            font=("Consolas", 10),
            relief="flat",
            background="#ffffff",
            foreground="#20364a",
            padx=10,
            pady=8,
            height=12,
        )
        eval_scroll = ttk.Scrollbar(
            self.eval_frame, orient="vertical", command=self.eval_detail_text.yview
        )
        self.eval_detail_text.configure(yscrollcommand=eval_scroll.set)
        eval_scroll.pack(side="right", fill="y")
        self.eval_detail_text.pack(fill="both", expand=True)
        self.eval_detail_text.configure(state="disabled")

        # Initial offline refresh
        self.refresh()

    def _write_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.yview_moveto(0)
        widget.configure(state="disabled")

    def _append_log(self, widget: tk.Text, line: str) -> None:
        widget.configure(state="normal")
        widget.insert("end", line + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def busy(self) -> bool:
        return self.active_future is not None and not self.active_future.done()

    def refresh(self) -> None:
        """Scan snapshots and models offline via dashboard executor."""
        if self.busy():
            return
        self.active_future = self.dashboard.executor.submit(self._read_offline_data)

    def _read_offline_data(self) -> dict[str, Any]:
        sources = self.manager.discover_sources()
        eligible = [s for s in sources if s.is_eligible]
        excluded = [s for s in sources if not s.is_eligible]
        snapshots = self.manager.list_snapshots()
        candidate_models = self.lab.list_candidate_models()
        return {
            "eligible_count": len(eligible),
            "excluded_count": len(excluded),
            "excluded_sources": [s.to_dict() for s in excluded],
            "snapshots": snapshots,
            "models": candidate_models,
        }

    def prepare_snapshot(self) -> None:
        """Background dataset snapshot preparation."""
        if self.busy():
            return
        if not messagebox.askyesno(
            "Prepare Research Snapshot",
            "Prepare an immutable, deduplicated, and conflict-quarantined research snapshot "
            "from all eligible authentic datasets?\n\nThis runs entirely offline without "
            "modifying original data.",
        ):
            return

        self.btn_prepare_snapshot.state(["disabled"])
        self.dataset_status_label.configure(
            text="Preparing research dataset snapshot in background..."
        )
        self.active_future = self.dashboard.executor.submit(self.manager.prepare_dataset)

    def start_training(self) -> None:
        """Launch background candidate model training."""
        if self.busy():
            return

        selected_snapshot_id = self.train_snapshot_combo.get()
        if not selected_snapshot_id:
            messagebox.showwarning(
                "No Dataset Selected", "Please select a prepared research dataset snapshot."
            )
            return

        # Find snapshot
        snapshot = next(
            (s for s in self.snapshots if s.manifest.snapshot_id == selected_snapshot_id),
            None,
        )
        if snapshot is None:
            messagebox.showerror("Error", "Selected snapshot is not available.")
            return

        # Parse config
        family_map = {
            "Random Forest (RandomForestModel)": "random_forest",
            "Logistic Regression (LogisticRegressionModel)": "logistic_regression",
            "HistGradientBoosting (HistGradientBoostingModel)": "hist_gradient_boosting",
            "Majority Direction Prior (Baseline)": "majority_class",
            "Rule-Based Momentum (Baseline)": "rule_momentum",
        }
        family = family_map.get(self.train_model_combo.get(), "random_forest")
        horizon = 1 if "1 bar" in self.train_horizon_combo.get() else 5
        threshold_map = {
            "0.50 (no abstention)": 0.50,
            "0.55": 0.55,
            "0.60 (standard)": 0.60,
            "0.65": 0.65,
        }
        threshold = threshold_map.get(self.train_threshold_combo.get(), 0.60)
        cal_map = {
            "Isotonic": "isotonic",
            "Sigmoid (Platt)": "sigmoid",
            "None": "none",
        }
        calibration = cal_map.get(self.train_cal_combo.get(), "isotonic")

        exp_name = f"exp_{family}_h{horizon}_{datetime.now(UTC).strftime('%H%M%S')}"
        config = ExperimentConfig(
            experiment_name=exp_name,
            model_family=family,
            horizon_bars=horizon,
            confidence_threshold=threshold,
            calibration_method=calibration,
            hyperparameters={"n_estimators": 50, "max_depth": 6},
        )

        self._write_text(self.train_log_text, "")
        self._append_log(
            self.train_log_text,
            f"Starting Experiment: {exp_name} ({family}, H={horizon}, Threshold={threshold})",
        )
        self.btn_start_train.state(["disabled"])
        self.btn_cancel_train.state(["!disabled"])
        self.is_cancelled = False

        def progress_cb(pct: int, msg: str) -> None:
            self.dashboard.window.after(0, lambda: self._update_progress(pct, msg))

        def cancel_token() -> bool:
            return self.is_cancelled

        self.active_future = self.dashboard.executor.submit(
            self.lab.run_experiment,
            config,
            snapshot,
            progress_cb=progress_cb,
            cancel_token=cancel_token,
        )

    def _update_progress(self, pct: int, msg: str) -> None:
        self.train_progress["value"] = pct
        self.train_status_label.configure(text=f"[{pct}%] {msg}")
        self._append_log(self.train_log_text, f"[{pct:02d}%] {msg}")

    def cancel_training(self) -> None:
        self.is_cancelled = True
        self.train_status_label.configure(text="Cancelling experiment; please wait...")
        self.btn_cancel_train.state(["disabled"])

    def tick(self) -> None:
        """Polled by dashboard.tick() to handle completed background tasks."""
        if self.active_future is None or not self.active_future.done():
            return

        future = self.active_future
        self.active_future = None

        try:
            result = future.result()
            if isinstance(result, dict) and "snapshots" in result:
                # Result of _read_offline_data
                self.snapshots = result["snapshots"]
                self.candidate_models = result["models"]

                status_msg = (
                    f"Discovered Sources: {result['eligible_count']} eligible, "
                    f"{result['excluded_count']} excluded (empty/OTC/synthetic). "
                    f"Research Snapshots: {len(self.snapshots)} available."
                )
                self.dataset_status_label.configure(text=status_msg)

                # Update Comboboxes
                s_ids = [s.manifest.snapshot_id for s in self.snapshots]
                self.snapshot_combo["values"] = s_ids
                self.train_snapshot_combo["values"] = s_ids
                if s_ids and not self.snapshot_combo.get():
                    self.snapshot_combo.current(0)
                    self._on_snapshot_selected()
                if s_ids and not self.train_snapshot_combo.get():
                    self.train_snapshot_combo.current(0)

                # Update Models Treeview
                self._update_models_tree(self.candidate_models)

            elif isinstance(result, ResearchDatasetSnapshot):
                # Result of prepare_snapshot
                self.btn_prepare_snapshot.state(["!disabled"])
                messagebox.showinfo(
                    "Snapshot Prepared",
                    f"Successfully prepared snapshot: {result.manifest.snapshot_id}\n\n"
                    f"Total candles retained: {result.manifest.total_candles_retained:,}\n"
                    f"Duplicates: {result.manifest.duplicate_boundaries_deduplicated:,}\n"
                    f"Conflicts quarantined: {result.manifest.conflicts_quarantined_count:,}\n"
                    f"Unscheduled gaps: {result.manifest.unscheduled_gaps_count:,}\n"
                    f"Scheduled closures: {result.manifest.scheduled_closures_count:,}",
                )
                self.refresh()

            elif isinstance(result, ExperimentResult):
                # Result of run_experiment
                self.btn_start_train.state(["!disabled"])
                self.btn_cancel_train.state(["disabled"])
                self.train_progress["value"] = 100
                self.train_status_label.configure(
                    text=f"Completed: {result.model_id} | Verdict: {result.verdict}"
                )
                m_test = result.test_metrics
                ci_str = f"[{m_test.ci_lower_95:.4f}, {m_test.ci_upper_95:.4f}]"
                self._append_log(
                    self.train_log_text,
                    f"\n>>> Experiment Finished!\n"
                    f"Model ID: {result.model_id}\n"
                    f"Train Accuracy: {result.train_metrics.directional_accuracy:.4f}\n"
                    f"Val Accuracy:   {result.val_metrics.directional_accuracy:.4f}\n"
                    f"Test Accuracy:  {m_test.directional_accuracy:.4f} (95% CI: {ci_str})\n"
                    f"Break-even Win Rate: {result.break_even_win_rate:.4f}\n"
                    f"Scientific Verdict:  {result.verdict}\n",
                )
                test_ci_pct = f"[{m_test.ci_lower_95:.2%}, {m_test.ci_upper_95:.2%}]"
                messagebox.showinfo(
                    "Experiment Completed",
                    f"Candidate model trained: {result.model_id}\n\n"
                    f"Test Directional Accuracy: {result.test_metrics.directional_accuracy:.2%}\n"
                    f"95% Confidence Interval: {test_ci_pct}\n"
                    f"Break-Even Win Rate: {result.break_even_win_rate:.2%}\n"
                    f"Verdict: {result.verdict}\n\n"
                    f"{result.disclaimer}",
                )
                self.refresh()

        except Exception as exc:  # noqa: BLE001
            self.btn_prepare_snapshot.state(["!disabled"])
            self.btn_start_train.state(["!disabled"])
            self.btn_cancel_train.state(["disabled"])
            self.train_status_label.configure(text=f"Failed: {type(exc).__name__}: {exc}")
            self._append_log(self.train_log_text, f"\n[ERROR] {type(exc).__name__}: {exc}")
            messagebox.showerror(
                "Laboratory Operation Error", f"Operation failed: {type(exc).__name__}: {exc}"
            )

    def _on_snapshot_selected(self, _event: object = None) -> None:
        s_id = self.snapshot_combo.get()
        snapshot = next((s for s in self.snapshots if s.manifest.snapshot_id == s_id), None)
        if snapshot is None:
            return
        m = snapshot.manifest
        summary = (
            f"SNAPSHOT SPECIFICATION & AUDIT\n"
            f"{'=' * 65}\n"
            f"Snapshot ID:         {m.snapshot_id}\n"
            f"Created At:          {m.created_at_utc}\n"
            f"Dataset Version:     {m.dataset_version}\n"
            f"Symbol / Timeframe:  {m.symbol} / {m.candle_size_seconds}s (M1)\n"
            f"Date Range (UTC):    {m.from_utc}  -->  {m.to_utc}\n"
            f"Parquet SHA-256:     {m.sha256_checksum}\n\n"
            f"DATA COUNTS & GOVERNANCE\n"
            f"{'-' * 65}\n"
            f"Total Candles Discovered:    {m.total_candles_discovered:,}\n"
            f"Total Candles Retained:      {m.total_candles_retained:,}\n"
            f"Duplicates Deduplicated:     {m.duplicate_boundaries_deduplicated:,}\n"
            f"Conflicts Quarantined:       {m.conflicts_quarantined_count:,}\n"
            f"Unscheduled Intraday Gaps:   {m.unscheduled_gaps_count:,}\n"
            f"Scheduled Weekend Closures:  {m.scheduled_closures_count:,}\n"
            f"Provenance Breakdown:        {m.provenance_breakdown}\n\n"
            f"CONTRIBUTING SOURCES ({len(m.source_files)})\n"
            f"{'-' * 65}\n"
        )
        for s in m.source_files:
            summary += (
                f"- {s['source_id']} ({s['source_type']}): "
                f"{s['candle_count']} candles | {s['provenance']}\n"
            )

        if m.conflicts_quarantined:
            summary += f"\nQUARANTINED CONFLICTS ({len(m.conflicts_quarantined)})\n{'-' * 65}\n"
            for q in m.conflicts_quarantined[:10]:
                summary += f"Boundary: {q['boundary_utc']} ({q['boundary_ts']}): {q['reason']}\n"
            if len(m.conflicts_quarantined) > 10:
                summary += (
                    f"... and {len(m.conflicts_quarantined) - 10} more quarantined conflicts.\n"
                )

        self._write_text(self.dataset_detail_text, summary)

    def _update_models_tree(self, models: list[dict[str, Any]]) -> None:
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        for m in models:
            self.model_tree.insert(
                "",
                "end",
                values=(
                    m["model_id"],
                    m["model_family"],
                    m["created_at_utc"],
                    f"{m['test_accuracy']:.2%}" if m.get("test_accuracy") is not None else "N/A",
                    f"{m.get('brier', m.get('brier_score', 0.0)):.4f}",
                    m.get("verdict", "UNKNOWN"),
                ),
            )

    def _on_model_selected(self, _event: object = None) -> None:
        sel = self.model_tree.selection()
        if not sel:
            return
        vals = self.model_tree.item(sel[0])["values"]
        model_id = vals[0]
        model_info = next((m for m in self.candidate_models if m["model_id"] == model_id), None)
        if not model_info:
            return

        manifest_path = Path(model_info["path"]) / "manifest.json"
        if not manifest_path.is_file():
            return
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        m_train = data.get("metrics", {}).get("train", {})
        m_val = data.get("metrics", {}).get("val", {})
        m_test = data.get("metrics", {}).get("test", {})

        def f_pct(m: dict[str, Any], key: str) -> str:
            return f"{m.get(key, 0.0):.2%}"

        def f_flt(m: dict[str, Any], key: str) -> str:
            return f"{m.get(key, 0.0):.4f}"

        def f_rat(m: dict[str, Any], k1: str, k2: str) -> str:
            return f"{m.get(k1, 0.0):.1%}/{m.get(k2, 0.0):.1%}"

        def row(label: str, v1: str, v2: str, v3: str) -> str:
            return f"{label:<22} {v1:<14} {v2:<14} {v3}"

        ci_lower = m_test.get("ci_lower_95", 0.0)
        ci_upper = m_test.get("ci_upper_95", 0.0)
        ci_test_str = f"[{ci_lower:.2%}, {ci_upper:.2%}]"
        edge_str = "YES (CONFIRMED)" if data.get("has_statistical_edge") else "NO"

        lines = [
            "CANDIDATE MODEL EVALUATION REPORT",
            "=" * 65,
            f"Model ID:            {data.get('model_id')}",
            f"Family:              {data.get('model_family')}",
            f"Created At (UTC):    {data.get('created_at_utc')}",
            f"Prediction Horizon:  {data.get('horizon_bars')} bar(s)",
            f"Data Cutoff Epoch:   {data.get('data_cutoff_ts')}",
            f"Bundle SHA-256:      {data.get('bundle_sha256')}",
            "",
            "CHRONOLOGICAL PARTITION METRICS",
            "-" * 65,
            f"{'Metric':<22} {'Train':<14} {'Validation':<14} {'Test (Holdout)'}",
            "-" * 65,
            row(
                "Directional Acc:",
                f_pct(m_train, "directional_accuracy"),
                f_pct(m_val, "directional_accuracy"),
                f_pct(m_test, "directional_accuracy"),
            ),
            row(
                "Balanced Acc:",
                f_pct(m_train, "balanced_accuracy"),
                f_pct(m_val, "balanced_accuracy"),
                f_pct(m_test, "balanced_accuracy"),
            ),
            row(
                "Precision (UP/DN):",
                f_rat(m_train, "precision_up", "precision_down"),
                f_rat(m_val, "precision_up", "precision_down"),
                f_rat(m_test, "precision_up", "precision_down"),
            ),
            row(
                "Recall (UP/DN):",
                f_rat(m_train, "recall_up", "recall_down"),
                f_rat(m_val, "recall_up", "recall_down"),
                f_rat(m_test, "recall_up", "recall_down"),
            ),
            row(
                "Brier Score:",
                f_flt(m_train, "brier_score"),
                f_flt(m_val, "brier_score"),
                f_flt(m_test, "brier_score"),
            ),
            row(
                "Trade Count:",
                str(m_train.get("trade_count", 0)),
                str(m_val.get("trade_count", 0)),
                str(m_test.get("trade_count", 0)),
            ),
            row(
                "Coverage/Abstain:",
                f_rat(m_train, "coverage", "abstention_rate"),
                f_rat(m_val, "coverage", "abstention_rate"),
                f_rat(m_test, "coverage", "abstention_rate"),
            ),
            row(
                "Selective Win Rate:",
                f_pct(m_train, "win_rate"),
                f_pct(m_val, "win_rate"),
                f_pct(m_test, "win_rate"),
            ),
            f"Wilson 95% CI (Test): {ci_test_str}",
            f"Break-Even Win Rate:  {data.get('break_even_win_rate', 0.5405):.2%}",
            f"Simulated Expectancy: {m_test.get('expectancy_pct', 0.0):+.2f}% per trade",
            "",
            "SCIENTIFIC ASSESSMENT & VERDICT",
            "-" * 65,
            f"Verdict:              {data.get('verdict')}",
            f"Statistical Edge:     {edge_str}",
            f"Scientific Notice:    {data.get('disclaimer')}",
        ]
        self._write_text(self.eval_detail_text, "\n".join(lines))
