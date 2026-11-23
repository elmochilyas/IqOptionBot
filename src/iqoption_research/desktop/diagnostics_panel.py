"""Desktop Model Diagnostics & Failure Intelligence Panel (Phase 41).

Provides responsive, offline inspection of:
1. Model failure intelligence, confusion matrices, and exact-tie distributions.
2. Market regime analyses (hourly, day-of-week, volatility terciles, candle ranges).
3. Predefined controlled research experiment trials (Areas A-F) with baselines.
4. Market microstructure comparisons (Binary vs Digital vs Forex) and payout sensitivity.
5. Methodological disclaimers, effective sample size warnings, and scientific isolation checks.
"""

from __future__ import annotations

import tkinter as tk
from concurrent.futures import Future
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any

from iqoption_research.diagnostics.experiments import (
    ControlledExperimentTrialResult,
    get_predefined_experiment_matrix,
    run_controlled_experiment_suite,
)
from iqoption_research.diagnostics.failure_intelligence import (
    FailureIntelligenceReport,
    generate_failure_intelligence_report,
)
from iqoption_research.diagnostics.microstructure import (
    calculate_payout_sensitivity_table,
    render_microstructure_comparison_table,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType, TiePolicy
from iqoption_research.research.dataset_manager import ResearchDatasetManager

if TYPE_CHECKING:
    from iqoption_research.desktop.app import Dashboard


class ModelDiagnosticsPanel:
    """Offline Model Diagnostics & Failure Intelligence tab inside Research Workstation V2."""

    def __init__(self, dashboard: Dashboard, notebook: ttk.Notebook) -> None:
        self.dashboard = dashboard
        self.root = dashboard.root
        self.dataset_manager = ResearchDatasetManager(self.root)
        self.active_future: Future[Any] | None = None
        self.last_report: FailureIntelligenceReport | None = None
        self.last_experiment_results: list[ControlledExperimentTrialResult] = []

        # Main Panel Frame
        self.frame = ttk.Frame(notebook, padding=12)
        notebook.add(self.frame, text="Model Diagnostics")

        # Top Header & Invariants Banner
        header = ttk.Frame(self.frame)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(
            header,
            text="Model Failure Intelligence & Controlled Research (Phase 41)",
            font=("Segoe UI", 13, "bold"),
            foreground="#17324d",
        ).pack(anchor="w")

        ttk.Label(
            header,
            text=(
                "OFFLINE DIAGNOSTIC INSPECTION ONLY · ZERO LIVE TRADING AUTHORITY · "
                "HOLDOUT EVALUATIONS ISOLATED · NO DEMONSTRATED EDGE"
            ),
            font=("Segoe UI", 9, "bold"),
            foreground="#9c3414",
        ).pack(anchor="w", pady=(2, 6))

        # Action Button Toolbar
        toolbar = ttk.Frame(self.frame)
        toolbar.pack(fill="x", pady=(0, 8))

        self.btn_refresh = ttk.Button(
            toolbar, text="Generate Failure Report", command=self.on_generate_report
        )
        self.btn_refresh.pack(side="left", padx=(0, 8))

        self.btn_run_experiments = ttk.Button(
            toolbar, text="Run Controlled Experiments (A-F)", command=self.on_run_experiments
        )
        self.btn_run_experiments.pack(side="left", padx=(0, 8))

        self.status_label = ttk.Label(
            toolbar, text="Ready. Click above to execute offline diagnostics."
        )
        self.status_label.pack(side="left", padx=8)

        # Sub-notebook for Organized Inspection
        self.sub_notebook = ttk.Notebook(self.frame)
        self.sub_notebook.pack(fill="both", expand=True)

        # Tab 1: Executive Failure Intelligence & Targets
        self.txt_failure = self._create_text_tab("Failure Intelligence")

        # Tab 2: Temporal & Regime Breakdowns
        self.txt_regimes = self._create_text_tab("Regimes & Temporal")

        # Tab 3: Controlled Research Experiments (Treeview)
        self.frame_experiments = ttk.Frame(self.sub_notebook, padding=6)
        self.sub_notebook.add(self.frame_experiments, text="Controlled Experiments")
        self._setup_experiments_view(self.frame_experiments)

        # Tab 4: Market Microstructure & Economics
        self.txt_microstructure = self._create_text_tab("Microstructure & Economics")

        # Initial populate of static microstructure information
        self._populate_microstructure_view()

    def _create_text_tab(self, title: str) -> tk.Text:
        frame = ttk.Frame(self.sub_notebook, padding=6)
        self.sub_notebook.add(frame, text=title)
        txt = tk.Text(
            frame,
            wrap="word",
            font=("Consolas", 10),
            relief="flat",
            background="#ffffff",
            foreground="#20364a",
            padx=10,
            pady=10,
        )
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.configure(state="disabled")
        return txt

    def _write_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", content)
        widget.yview_moveto(0)
        widget.configure(state="disabled")

    def _setup_experiments_view(self, parent: ttk.Frame) -> None:
        cols = (
            "id",
            "area",
            "hypothesis",
            "val_acc",
            "brier",
            "coverage",
            "d_maj",
            "be_gap",
            "verdict",
        )
        self.tree_experiments = ttk.Treeview(parent, columns=cols, show="headings", height=10)
        self.tree_experiments.heading("id", text="Trial ID")
        self.tree_experiments.heading("area", text="Area")
        self.tree_experiments.heading("hypothesis", text="Hypothesis")
        self.tree_experiments.heading("val_acc", text="Val Acc")
        self.tree_experiments.heading("brier", text="Brier")
        self.tree_experiments.heading("coverage", text="Coverage")
        self.tree_experiments.heading("d_maj", text="vs Maj")
        self.tree_experiments.heading("be_gap", text="BE Gap")
        self.tree_experiments.heading("verdict", text="Scientific Verdict")

        self.tree_experiments.column("id", width=160, anchor="w")
        self.tree_experiments.column("area", width=40, anchor="center")
        self.tree_experiments.column("hypothesis", width=260, anchor="w")
        self.tree_experiments.column("val_acc", width=70, anchor="center")
        self.tree_experiments.column("brier", width=65, anchor="center")
        self.tree_experiments.column("coverage", width=70, anchor="center")
        self.tree_experiments.column("d_maj", width=65, anchor="center")
        self.tree_experiments.column("be_gap", width=65, anchor="center")
        self.tree_experiments.column("verdict", width=220, anchor="w")

        tree_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.tree_experiments.yview)
        self.tree_experiments.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side="right", fill="y")
        self.tree_experiments.pack(fill="both", expand=True)

        # Pre-populate with initial matrix specs
        matrix = get_predefined_experiment_matrix()
        for exp in matrix:
            self.tree_experiments.insert(
                "",
                "end",
                values=(
                    exp.experiment_id,
                    exp.area,
                    exp.hypothesis,
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "PENDING_EXECUTION",
                ),
            )

    def _populate_microstructure_view(self) -> None:
        table_str = render_microstructure_comparison_table()
        points = calculate_payout_sensitivity_table()
        hdr = (
            f"{'Payout':<8} | {'Break-Even':<12} | {'Exp @ 50%':<11} | "
            f"{'Exp @ 52%':<11} | {'Exp @ 55%':<11} | {'Tie Adjustment':<25}"
        )
        sens_lines = [
            "\n=== REALISTIC BROKER PAYOUT SENSITIVITY & BREAK-EVEN HURDLES ===",
            hdr,
            "-" * 85,
        ]
        for p in points:
            exp50 = f"{p.net_expectancy_at_50_pct * 100:+6.2f}%"
            exp52 = f"{p.net_expectancy_at_52_pct * 100:+6.2f}%"
            exp55 = f"{p.net_expectancy_at_55_pct * 100:+6.2f}%"
            tie_adj = p.tie_adjustment_binary[:25]
            sens_lines.append(
                f"{p.payout_rate * 100:5.0f}%  | {p.break_even_win_rate * 100:6.2f}%     | "
                f"{exp50}    | {exp52}    | {exp55}    | {tie_adj:<25}"
            )

        disclaimers = [
            "\n=== MARKET MICROSTRUCTURE EVIDENCE & LIMITATIONS ===",
            "1. IQ Option Binary Option: Expiry round second; ATM tie refunds stake.",
            "2. IQ Option Digital Option: Strike grid pricing; ATM tie results in 100% loss.",
            "3. Generic Prediction: Academic close-to-close ignores latency and quote slippage.",
            "4. Repository Evidence Gap: Continuous tick-level payouts and fills not archived.",
            "   Status: EVIDENCE_GAP_CONFIRMED. Prospective shadow observation required.",
        ]

        full_content = table_str + "\n" + "\n".join(sens_lines) + "\n" + "\n".join(disclaimers)
        self._write_text(self.txt_microstructure, full_content)

    def on_generate_report(self) -> None:
        if self.busy():
            return
        snapshots = self.dataset_manager.list_snapshots()
        if not snapshots:
            messagebox.showwarning("Diagnostics", "No research dataset snapshots found.")
            return

        self.status_label.configure(text="Generating Failure Intelligence report in background...")
        self.btn_refresh.state(["disabled"])
        self.btn_run_experiments.state(["disabled"])

        snapshot = snapshots[-1]
        self.active_future = self.dashboard.executor.submit(
            self._async_generate_report, snapshot.load_candles(), snapshot.manifest.snapshot_id
        )

    def _async_generate_report(
        self, candles: list[Any], snapshot_id: str
    ) -> FailureIntelligenceReport:
        import numpy as np

        from iqoption_research.ml.models.baselines import LogisticRegressionModel
        from iqoption_research.ml.targets.dataset import (
            MLDataset,
            build_ml_dataset,
            fit_scaler_on_dataset,
            transform_dataset,
        )
        from iqoption_research.research.datasets.splitter import ChronologicalSplitter
        from iqoption_research.research.features.pipeline import CausalFeaturePipeline

        spec = TargetSpec(
            target_type=TargetType.DIRECTION_BINARY, horizon_bars=1, tie_policy=TiePolicy.EXCLUDE
        )
        feat_pipe = CausalFeaturePipeline(
            static_schema=True, include_volume=True, enforce_temporal_continuity=True
        )
        feat_ds = feat_pipe.transform(candles)
        ml_ds = build_ml_dataset(candles, feat_ds, spec, enforce_temporal_continuity=True)

        splitter = ChronologicalSplitter(
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, label_horizon_bars=1, embargo_bars=5
        )
        splits = splitter.split(list(range(ml_ds.row_count)))
        train_idx = np.array(splits.train, dtype=np.int64)
        val_idx = np.array(splits.val, dtype=np.int64)

        ds_train = MLDataset(
            symbol=ml_ds.symbol,
            feature_names=ml_ds.feature_names,
            X=ml_ds.X[train_idx],
            y=ml_ds.y[train_idx],
            timestamps=ml_ds.timestamps[train_idx],
            target_spec=ml_ds.target_spec,
        )
        ds_val = MLDataset(
            symbol=ml_ds.symbol,
            feature_names=ml_ds.feature_names,
            X=ml_ds.X[val_idx],
            y=ml_ds.y[val_idx],
            timestamps=ml_ds.timestamps[val_idx],
            target_spec=ml_ds.target_spec,
        )

        scaler = fit_scaler_on_dataset(ds_train)
        scaled_train = transform_dataset(ds_train, scaler)
        scaled_val = transform_dataset(ds_val, scaler)

        model = LogisticRegressionModel(random_state=42)
        model.fit(scaled_train.X, scaled_train.y)
        val_probs = model.predict_proba(scaled_val.X)[:, 1]

        from iqoption_research.diagnostics.alignment import align_predictions_to_market_data

        aligned = align_predictions_to_market_data(
            candles=candles,
            ml_dataset=ds_val,
            predicted_probabilities=val_probs,
            decision_threshold=0.50,
        )
        return generate_failure_intelligence_report(
            candles,
            aligned,
            decision_threshold=0.50,
            dataset_id=snapshot_id,
            spec=spec,
        )

    def on_run_experiments(self) -> None:
        if self.busy():
            return
        snapshots = self.dataset_manager.list_snapshots()
        if not snapshots:
            messagebox.showwarning("Diagnostics", "No research dataset snapshots found.")
            return

        self.status_label.configure(
            text="Running controlled experiment matrix (Areas A-F) in background..."
        )
        self.btn_refresh.state(["disabled"])
        self.btn_run_experiments.state(["disabled"])

        snapshot = snapshots[-1]
        candles = snapshot.load_candles()
        # Evaluate up to 5,000 candles for rapid interactive responsiveness
        sample_candles = candles[-5000:]
        self.active_future = self.dashboard.executor.submit(
            run_controlled_experiment_suite, sample_candles, get_predefined_experiment_matrix()
        )

    def busy(self) -> bool:
        return self.active_future is not None and not self.active_future.done()

    def tick(self) -> None:
        if self.active_future is not None and self.active_future.done():
            future = self.active_future
            self.active_future = None
            self.btn_refresh.state(["!disabled"])
            self.btn_run_experiments.state(["!disabled"])

            try:
                res = future.result()
                if isinstance(res, FailureIntelligenceReport):
                    self.last_report = res
                    self._render_report(res)
                    self.status_label.configure(
                        text=f"Failure Intelligence report updated ({res.dataset_id})."
                    )
                elif isinstance(res, list):
                    self.last_experiment_results = res
                    self._render_experiments(res)
                    self.status_label.configure(
                        text=f"Controlled experiments completed ({len(res)} trials)."
                    )
            except Exception as exc:  # noqa: BLE001
                self.status_label.configure(text=f"Error executing diagnostics: {exc}")
                messagebox.showerror("Diagnostics Error", str(exc))

    def _render_report(self, report: FailureIntelligenceReport) -> None:
        md = report.render_markdown()
        self._write_text(self.txt_failure, md)

        # Render Regimes Tab
        hdr_hour = (
            f"{'UTC Hour':<10} | {'Samples':<8} | {'Accuracy':<10} | {'Brier':<8} | "
            f"{'UP Pred%':<10} | {'Ties':<6} | {'Reliability':<15}"
        )
        regime_lines = [
            "=== TEMPORAL BREAKDOWN: HOURLY PERFORMANCE (UTC) ===",
            hdr_hour,
            "-" * 78,
        ]
        for h in report.hourly_breakdown:
            rel = "RELIABLE" if h.is_statistically_reliable else f"LOW ({h.warning})"
            brier_str = f"{h.brier_score:6.4f}"
            up_str = f"{h.up_pred_ratio * 100:6.2f}%"
            regime_lines.append(
                f"{h.subgroup_name:<10} | {h.sample_count:<8,d} | {h.accuracy * 100:6.2f}%    | "
                f"{brier_str}   | {up_str}    | {h.tie_count:<6,d} | {rel[:15]:<15}"
            )

        hdr_dow = (
            f"{'Day':<12} | {'Samples':<8} | {'Accuracy':<10} | {'Brier':<8} | "
            f"{'UP Pred%':<10} | {'Ties':<6}"
        )
        regime_lines.extend([
            "\n=== DAY OF WEEK BREAKDOWN ===",
            hdr_dow,
            "-" * 65,
        ])
        for d in report.day_of_week_breakdown:
            brier_str = f"{d.brier_score:6.4f}"
            up_str = f"{d.up_pred_ratio * 100:6.2f}%"
            regime_lines.append(
                f"{d.subgroup_name:<12} | {d.sample_count:<8,d} | {d.accuracy * 100:6.2f}%    | "
                f"{brier_str}   | {up_str}    | {d.tie_count:<6,d}"
            )

        regime_lines.extend([
            "\n=== VOLATILITY TERCILES ===",
            f"{'Regime':<18} | {'Samples':<8} | {'Accuracy':<10} | {'Brier':<8}",
            "-" * 50,
        ])
        for v in report.volatility_regimes:
            regime_lines.append(
                f"{v.subgroup_name:<18} | {v.sample_count:<8,d} | "
                f"{v.accuracy * 100:6.2f}%    | {v.brier_score:6.4f}"
            )

        regime_lines.extend([
            "\n=== CANDLE RANGE TERCILES ===",
            f"{'Range Tercile':<18} | {'Samples':<8} | {'Accuracy':<10} | {'Brier':<8}",
            "-" * 50,
        ])
        for r in report.candle_range_regimes:
            regime_lines.append(
                f"{r.subgroup_name:<18} | {r.sample_count:<8,d} | "
                f"{r.accuracy * 100:6.2f}%    | {r.brier_score:6.4f}"
            )

        self._write_text(self.txt_regimes, "\n".join(regime_lines))

    def _render_experiments(self, results: list[ControlledExperimentTrialResult]) -> None:
        for item in self.tree_experiments.get_children():
            self.tree_experiments.delete(item)

        for r in results:
            self.tree_experiments.insert(
                "",
                "end",
                values=(
                    r.experiment_id,
                    r.area,
                    r.hypothesis,
                    f"{r.val_accuracy * 100:.2f}%" if r.val_accuracy is not None else "N/A",
                    f"{r.val_brier_score:.4f}",
                    f"{r.val_coverage * 100:.1f}%",
                    (
                        f"{r.delta_vs_majority * 100:+.2f}%"
                        if r.delta_vs_majority is not None
                        else "N/A"
                    ),
                    (f"{r.break_even_gap * 100:+.2f}%" if r.break_even_gap is not None else "N/A"),
                    r.verdict,
                ),
            )
