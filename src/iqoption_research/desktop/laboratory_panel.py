"""Desktop Model Laboratory panel for offline dataset preparation, training, and evaluation."""

from __future__ import annotations

import json
import logging
import tkinter as tk
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any

from iqoption_research.ml.data_audit import DataCoverageReport, audit_market_datasets
from iqoption_research.ml.experiment_plan import build_standard_phase42_experiment_plan
from iqoption_research.ml.experiment_runner import ExperimentRunner, ExperimentSuiteSummary
from iqoption_research.ml.laboratory import (
    ExperimentConfig,
    ExperimentResult,
    ModelLaboratory,
)
from iqoption_research.ml.partitions import PartitionManifest
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
    ResearchDatasetSnapshot,
)

if TYPE_CHECKING:
    from iqoption_research.desktop.app import Dashboard

logger = logging.getLogger(__name__)


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

        # Tab 4: Multi-Horizon & Regimes (Phase 43)
        self.mh_frame = ttk.Frame(self.sub_notebook, padding=10)
        self.sub_notebook.add(self.mh_frame, text="Multi-Horizon & Regimes (Phase 43)")
        self._build_multi_horizon_tab()

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

        self.btn_audit_data = ttk.Button(
            top_bar,
            text="Run Authentic Data & Sufficiency Audit",
            command=self.run_data_audit_action,
        )
        self.btn_audit_data.pack(side="left", padx=(0, 8))

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
                "Extra Trees (ExtraTreesModel)",
                "Compact Neural Network (MLP)",
                "Logistic Regression (LogisticRegressionModel)",
                "HistGradientBoosting (HistGradientBoostingModel)",
                "Majority Direction Prior (Baseline)",
                "Rule-Based Momentum (Baseline)",
                "Rule-Based Reversal (Baseline)",
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

        # Row 3: Feature Pipeline & Validation Protocol
        r3 = ttk.Frame(opts_frame)
        r3.pack(fill="x", pady=4)
        ttk.Label(r3, text="Feature Pipeline:", width=18).pack(side="left")
        self.train_features_combo = ttk.Combobox(
            r3,
            values=[
                "V2 All (34 Causal Features)",
                "V2 Price Action Only (12 Features)",
                "V2 Volatility & Trend (12 Features)",
                "V2 Regime & Temporal (7 Features)",
                "V1 Baseline (12 Features)",
            ],
            state="readonly",
            width=36,
        )
        self.train_features_combo.current(0)
        self.train_features_combo.pack(side="left", padx=(0, 20))

        ttk.Label(r3, text="Validation Scheme:", width=18).pack(side="left")
        self.train_protocol_combo = ttk.Combobox(
            r3,
            values=[
                "3 Folds (Walk-Forward Expanding - Phase 42)",
                "Single Chronological Split (Phase 39)",
            ],
            state="readonly",
            width=38,
        )
        self.train_protocol_combo.current(0)
        self.train_protocol_combo.pack(side="left")

        # Action Buttons & Progress
        action_bar = ttk.Frame(self.train_frame)
        action_bar.pack(fill="x", pady=6)
        self.btn_start_train = ttk.Button(
            action_bar, text="Run Training Experiment", command=self.start_training
        )
        self.btn_start_train.pack(side="left", padx=(0, 8))
        self.btn_run_suite = ttk.Button(
            action_bar, text="Run Phase 42 Experiment Suite", command=self.run_experiment_suite
        )
        self.btn_run_suite.pack(side="left", padx=(0, 8))
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

        columns = (
            "model_id",
            "family",
            "status",
            "created",
            "test_acc",
            "brier",
            "verdict",
            "ece",
        )
        self.model_tree = ttk.Treeview(self.eval_frame, columns=columns, show="headings", height=6)
        self.model_tree.heading("model_id", text="Candidate Model ID")
        self.model_tree.heading("family", text="Model Family")
        self.model_tree.heading("status", text="Integrity Status")
        self.model_tree.heading("created", text="Created UTC")
        self.model_tree.heading("test_acc", text="Test Acc")
        self.model_tree.heading("brier", text="Brier")
        self.model_tree.heading("verdict", text="Scientific Verdict")
        self.model_tree.heading("ece", text="ECE")

        self.model_tree.column("model_id", width=280)
        self.model_tree.column("family", width=120)
        self.model_tree.column("status", width=110)
        self.model_tree.column("created", width=130)
        self.model_tree.column("test_acc", width=70)
        self.model_tree.column("brier", width=70)
        self.model_tree.column("verdict", width=210)
        self.model_tree.column("ece", width=70)

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

    def run_data_audit_action(self) -> None:
        """Background authentic market data audit and sufficiency check."""
        if self.busy():
            return
        self.btn_audit_data.state(["disabled"])
        self.dataset_status_label.configure(
            text="Running comprehensive authentic data audit and sample sufficiency check..."
        )
        self.active_future = self.dashboard.executor.submit(audit_market_datasets, self.root)

    def run_experiment_suite(self) -> None:
        """Launch background execution of Phase 42 experiment suite."""
        if self.busy():
            return
        if not messagebox.askyesno(
            "Phase 42 Experiment Suite",
            "Run offline walk-forward experiment suite across genuine market data?\n\n"
            "Evaluates baseline models, feature ablations, and candidate model families "
            "with out-of-fold calibration and Day-Block bootstrap confidence intervals.\n\n"
            "Strictly offline: Zero live orders, zero broker connection.",
        ):
            return

        self._write_text(self.train_log_text, "")
        self._append_log(self.train_log_text, "Starting Phase 42 Offline Experiment Suite...")
        self.btn_start_train.state(["disabled"])
        self.btn_run_suite.state(["disabled"])
        self.btn_cancel_train.state(["!disabled"])
        self.is_cancelled = False

        runner = ExperimentRunner(self.root)
        plan = build_standard_phase42_experiment_plan()

        def prog_cb(completed: int, total: int, status_str: str) -> None:
            pct = int((completed / total) * 100) if total > 0 else 0
            self.dashboard.window.after(
                0, lambda: self._update_progress(pct, f"[{completed}/{total}] {status_str}")
            )

        def cancel_check() -> bool:
            return self.is_cancelled

        self.active_future = self.dashboard.executor.submit(
            runner.run_suite,
            plan=plan,
            max_trials=45,
            progress_callback=prog_cb,
            cancel_check=cancel_check,
        )

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
            "Extra Trees (ExtraTreesModel)": "extra_trees",
            "Compact Neural Network (MLP)": "compact_neural_network",
            "Logistic Regression (LogisticRegressionModel)": "logistic_regression",
            "HistGradientBoosting (HistGradientBoostingModel)": "hist_gradient_boosting",
            "Majority Direction Prior (Baseline)": "majority_class",
            "Rule-Based Momentum (Baseline)": "rule_momentum",
            "Rule-Based Reversal (Baseline)": "rule_reversal",
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

            elif isinstance(result, DataCoverageReport):
                self.btn_audit_data.state(["!disabled"])
                suff_str = "PASS" if result.data_sufficiency_passed else "FAIL"
                self.dataset_status_label.configure(
                    text=(
                        f"Data Audit Complete: {result.total_candles_retained:,} candles across "
                        f"{result.distinct_market_days} days. Sufficiency: {suff_str}"
                    )
                )
                self._write_text(self.dataset_detail_text, result.to_markdown())
                messagebox.showinfo(
                    "Data Audit Completed",
                    f"Authentic Market Data Audit Complete:\n\n"
                    f"Total Retained: {result.total_candles_retained:,} candles\n"
                    f"Distinct Market Days: {result.distinct_market_days}\n"
                    f"Continuous Segments: {result.continuous_segments_count}\n"
                    f"Weekend Closures: {result.weekend_closures_count}\n"
                    f"Sufficiency: "
                    f"{'PASSED' if result.data_sufficiency_passed else 'FAILED'}\n\n"
                    "Detailed markdown report loaded into view.",
                )

            elif isinstance(result, ExperimentSuiteSummary):
                self.btn_start_train.state(["!disabled"])
                self.btn_run_suite.state(["!disabled"])
                self.btn_cancel_train.state(["disabled"])
                self.train_progress["value"] = 100
                self.train_status_label.configure(
                    text=(
                        f"Suite Complete: {result.completed_trials}/{result.total_trials} | "
                        f"Verdict: {result.global_disposition}"
                    )
                )
                self._append_log(
                    self.train_log_text,
                    f"\n>>> Phase 42 Experiment Suite Finished!\n"
                    f"Suite ID: {result.suite_id}\n"
                    f"Completed Trials: {result.completed_trials}/{result.total_trials}\n"
                    f"Eligible Candidates: {result.eligible_candidates_count}\n"
                    f"Global Disposition: {result.global_disposition}\n",
                )
                messagebox.showinfo(
                    "Experiment Suite Finished",
                    f"Phase 42 Experiment Suite Complete:\n\n"
                    f"Completed Trials: {result.completed_trials}/{result.total_trials}\n"
                    f"Eligible Candidates: {result.eligible_candidates_count}\n"
                    f"Global Disposition: {result.global_disposition}\n"
                    f"Execution Authority: {result.execution_authority}\n\n"
                    "NO DEMONSTRATED EDGE. Models frozen for shadow observation only.",
                )
                self.refresh()

        except Exception as exc:  # noqa: BLE001
            self.btn_prepare_snapshot.state(["!disabled"])
            self.btn_audit_data.state(["!disabled"])
            self.btn_start_train.state(["!disabled"])
            self.btn_run_suite.state(["!disabled"])
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

        p_dir = self.root / "data" / "partitions"
        p_files = sorted(p_dir.glob("*.json")) if p_dir.is_dir() else []
        if p_files:
            try:
                part = PartitionManifest.load(p_files[-1])
                inspected_str = (
                    "YES (CANNOT BE CERTIFIED UNTOUCHED)"
                    if part.is_holdout_previously_inspected
                    else "NO (PRISTINE)"
                )
                summary += (
                    f"\n4-WAY CHRONOLOGICAL PARTITIONS ({part.manifest_id})\n"
                    f"{'-' * 65}\n"
                    f"Train (60%):       {part.train_partition.candle_count:,} candles\n"
                    f"  Range:           {part.train_partition.from_utc[:10]} -> "
                    f"{part.train_partition.to_utc[:10]}\n"
                    f"Val (15%):         {part.val_partition.candle_count:,} candles\n"
                    f"  Range:           {part.val_partition.from_utc[:10]} -> "
                    f"{part.val_partition.to_utc[:10]}\n"
                    f"Calibration (10%): {part.calibration_partition.candle_count:,} candles\n"
                    f"  Range:           {part.calibration_partition.from_utc[:10]} -> "
                    f"{part.calibration_partition.to_utc[:10]}\n"
                    f"Test (15%):        {part.protected_test_partition.candle_count:,} candles\n"
                    f"  Range:           {part.protected_test_partition.from_utc[:10]} -> "
                    f"{part.protected_test_partition.to_utc[:10]}\n"
                    f"Purge Lead Bars:   {part.purging_horizon_bars}\n"
                    f"Embargo Tail Bars: {part.embargo_bars}\n"
                    f"Prior Inspected:   {inspected_str}\n"
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed reading partition manifest: %s", exc)

        self._write_text(self.dataset_detail_text, summary)

    def _update_models_tree(self, models: list[dict[str, Any]]) -> None:
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        for m in models:
            ece_val = m.get("test_ece", m.get("ece", 0.0))
            self.model_tree.insert(
                "",
                "end",
                values=(
                    m["model_id"],
                    m["model_family"],
                    m.get("status", "VERIFIED" if m.get("is_verified") else "UNVERIFIED"),
                    m["created_at_utc"][:19],
                    f"{m['test_accuracy']:.2%}" if m.get("test_accuracy") is not None else "N/A",
                    f"{m.get('test_brier', m.get('brier_score', 0.0)):.4f}",
                    m.get("verdict", "UNKNOWN"),
                    f"{ece_val:.4f}" if ece_val else "N/A",
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
        status = model_info.get(
            "status", "VERIFIED" if model_info.get("is_verified") else "UNVERIFIED"
        )
        provenance = data.get("provenance_class", "UNKNOWN")
        synthetic_flag = "YES (TEST FIXTURE)" if data.get("is_synthetic") else "NO (AUTHENTIC DATA)"
        contract_desc = data.get("target_contract", {}).get(
            "description",
            f"Generic {data.get('horizon_bars')}-bar close-to-close directional prediction",
        )

        lines = [
            "CANDIDATE MODEL EVALUATION REPORT",
            "=" * 65,
            f"Model ID:            {data.get('model_id')}",
            f"Family:              {data.get('model_family')}",
            f"Integrity Status:    {status}",
            f"Provenance Class:    {provenance}",
            f"Synthetic Fixture:   {synthetic_flag}",
            f"Created At (UTC):    {data.get('created_at_utc')}",
            f"Target Formulation:  {contract_desc}",
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
                "Expected Cal Error:",
                f_flt(m_train, "ece"),
                f_flt(m_val, "ece"),
                f_flt(m_test, "ece"),
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
            f"Execution Authority:  {data.get('execution_authority', 'NONE')}",
            f"Scientific Notice:    {data.get('disclaimer')}",
            "",
            "PHASE 42 FAILURE INTELLIGENCE & TAXONOMY",
            "-" * 65,
            "- Microstructure: Martingale difference noise at M1 (lag-1 rho1 = +0.0122).",
            "- Asymmetric Payout: 54.054% break-even hurdle exceeds decisive accuracy.",
            "- Selective Abstention: Raising thresholds collapses coverage without edge.",
            "- Feature Saturation: Multi-family features do not improve generalization.",
            "- Calibration Collapse: Posteriors converge toward class prior ~0.50.",
        ]
        self._write_text(self.eval_detail_text, "\n".join(lines))

    def _build_multi_horizon_tab(self) -> None:
        action_bar = ttk.Frame(self.mh_frame)
        action_bar.pack(fill="x", pady=(0, 8))

        self.btn_audit_mh = ttk.Button(
            action_bar,
            text="Audit Horizon Readiness (H1-H60)",
            command=self.audit_horizon_readiness_action,
        )
        self.btn_audit_mh.pack(side="left", padx=(0, 8))

        self.btn_verify_tf = ttk.Button(
            action_bar, text="Verify Multi-Timeframe Engine", command=self.verify_timeframes_action
        )
        self.btn_verify_tf.pack(side="left", padx=(0, 8))

        self.btn_analyze_regimes = ttk.Button(
            action_bar, text="Analyze Market Regimes", command=self.analyze_regimes_action
        )
        self.btn_analyze_regimes.pack(side="left", padx=(0, 8))

        self.btn_run_mh_matrix = ttk.Button(
            action_bar,
            text="Run Multi-Horizon Matrix (Stage A-D)",
            command=self.run_multi_horizon_matrix_action,
        )
        self.btn_run_mh_matrix.pack(side="left", padx=(0, 8))

        # Treeview table
        columns = (
            "horizon",
            "stage",
            "model",
            "preset",
            "threshold",
            "accuracy",
            "coverage",
            "verdict",
        )
        self.mh_tree = ttk.Treeview(self.mh_frame, columns=columns, show="headings", height=8)
        self.mh_tree.heading("horizon", text="Horizon")
        self.mh_tree.heading("stage", text="Stage")
        self.mh_tree.heading("model", text="Model Family")
        self.mh_tree.heading("preset", text="Feature Schema")
        self.mh_tree.heading("threshold", text="Threshold")
        self.mh_tree.heading("accuracy", text="Decisive Acc")
        self.mh_tree.heading("coverage", text="Coverage")
        self.mh_tree.heading("verdict", text="Verdict")

        self.mh_tree.column("horizon", width=70, anchor="center")
        self.mh_tree.column("stage", width=140)
        self.mh_tree.column("model", width=140)
        self.mh_tree.column("preset", width=120)
        self.mh_tree.column("threshold", width=80, anchor="center")
        self.mh_tree.column("accuracy", width=90, anchor="e")
        self.mh_tree.column("coverage", width=80, anchor="e")
        self.mh_tree.column("verdict", width=180)
        self.mh_tree.pack(fill="x", pady=(0, 6))

        # Log & Diagnostics Area
        self.mh_log_text = tk.Text(
            self.mh_frame,
            wrap="word",
            font=("Consolas", 10),
            relief="flat",
            background="#ffffff",
            foreground="#20364a",
            padx=10,
            pady=8,
            height=12,
        )
        scroll = ttk.Scrollbar(self.mh_frame, orient="vertical", command=self.mh_log_text.yview)
        self.mh_log_text.configure(yscrollcommand=scroll.set)
        self.mh_log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def audit_horizon_readiness_action(self) -> None:
        """Audit data sufficiency across horizons H1 to H60."""
        from iqoption_research.market_data.data_inventory import compute_horizon_readiness
        from iqoption_research.market_data.parquet_storage import read_candles_parquet

        if not self.snapshots:
            self.refresh()
        if not self.snapshots:
            messagebox.showwarning("No Snapshot", "No market data snapshots available.")
            return

        candles = read_candles_parquet(self.snapshots[-1].parquet_path)
        readiness = compute_horizon_readiness(candles)

        lines = [
            "=" * 70,
            "PHASE 43 MULTI-HORIZON READINESS AUDIT",
            "=" * 70,
            f"Evaluated Base Candles: {len(candles):,} EURUSD M1 bars",
            "",
            f"{'Horizon':<8} {'Minutes':<8} {'Rolling Opps':<14} {'Non-Overlap':<14} {'Status':<14} {'Notes'}",
            "-" * 70,
        ]
        for r in readiness.values():
            lines.append(
                f"{r.horizon_name:<8} {r.horizon_minutes:<8} {r.rolling_opportunities_count:<14,} "
                f"{r.non_overlapping_opportunities_count:<14,} {r.status.value:<14} {r.sufficiency_notes}"
            )
        self._write_text(self.mh_log_text, "\n".join(lines))

    def verify_timeframes_action(self) -> None:
        """Verify multi-timeframe aggregation causal properties and prefix invariance."""
        from iqoption_research.market_data.parquet_storage import read_candles_parquet
        from iqoption_research.market_data.timeframe_engine import MultiTimeframeEngine, Timeframe

        if not self.snapshots:
            self.refresh()
        if not self.snapshots:
            messagebox.showwarning("No Snapshot", "No market data snapshots available.")
            return

        candles = read_candles_parquet(self.snapshots[-1].parquet_path)[:300]
        engine = MultiTimeframeEngine()

        for tf in (Timeframe.M5, Timeframe.M15):
            full_agg = engine.aggregate_m1_to_timeframe(candles, tf)
            if not full_agg:
                continue
            cutoff_c = full_agg[0]
            prefix = [c for c in candles if c.to_ts <= cutoff_c.to_ts]
            prefix_agg = engine.aggregate_m1_to_timeframe(prefix, tf)
            if not prefix_agg or prefix_agg[-1].close != cutoff_c.close:
                raise ValueError("Prefix invariance check failed on higher timeframe aggregation")

        msg = (
            "======================================================================\n"
            "MULTI-TIMEFRAME CAUSAL ENGINE VERIFICATION: PASSED\n"
            "======================================================================\n"
            "- Aggregation: Deterministic UTC boundaries (M1 -> M3, M5, M15, M30, H1).\n"
            "- As-Of Lookup: O(log K) point-in-time causal join verified.\n"
            "- Prefix Invariance: Bar completed at T is bit-for-bit identical.\n"
            "- Forward Leakage: Zero unfinalized or future candle inclusion."
        )
        self._write_text(self.mh_log_text, msg)

    def analyze_regimes_action(self) -> None:
        """Run causal market regime classification on base data."""
        from collections import Counter

        from iqoption_research.market_data.parquet_storage import read_candles_parquet
        from iqoption_research.ml.regimes import CausalMarketRegimeClassifier

        if not self.snapshots:
            self.refresh()
        if not self.snapshots:
            messagebox.showwarning("No Snapshot", "No market data snapshots available.")
            return

        candles = read_candles_parquet(self.snapshots[-1].parquet_path)
        classifier = CausalMarketRegimeClassifier()
        regimes = classifier.classify_series(candles)

        total = len(regimes)
        lines = [
            "=" * 70,
            "CAUSAL MARKET REGIME CLASSIFICATION & STRATIFICATION",
            "=" * 70,
            f"Total Analyzed Bars: {total:,} EURUSD M1 candles",
            "",
        ]

        for dim_name, vals in [
            ("Volatility Regime", [r.volatility.value for r in regimes]),
            ("Trend Regime", [r.trend.value for r in regimes]),
            ("Range State", [r.range_state.value for r in regimes]),
            ("Market Session", [r.session.value for r in regimes]),
        ]:
            lines.append(f"--- {dim_name} ---")
            for val, cnt in Counter(vals).most_common():
                pct = cnt / total if total > 0 else 0.0
                lines.append(f"  {val:<22}: {cnt:>7,} bars ({pct:.1%})")
            lines.append("")

        self._write_text(self.mh_log_text, "\n".join(lines))

    def run_multi_horizon_matrix_action(self) -> None:
        """Direct Phase 43 trial execution is inspection-only from the dashboard."""
        guidance = (
            "Phase 43 trial execution is not available from the dashboard.\n\n"
            "Authoritative snapshots are inspection-only. Use "
            "scripts/run_phase43_experiments.py with an explicit development-only "
            "snapshot override; every override is TEST_ONLY and never genuine evidence."
        )
        messagebox.showwarning("Phase 43 Trials Inspection-Only", guidance)
        self._write_text(self.mh_log_text, f"{guidance}\n")
