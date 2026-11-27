"""Desktop Execution Center panel inside Research Workstation V2 (Phase 50).

Strictly offline PAPER simulation & supervised Practice readiness audit panel.
Never transmits live broker orders from dashboard buttons and never exposes secrets.
"""

from __future__ import annotations

import json
import tkinter as tk
from datetime import UTC, datetime
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

from iqoption_research.execution.models import (
    OrderSide,
    OrderState,
    PaperOrder,
    ProductType,
)
from iqoption_research.execution.offline_demo import OfflineExecutionDemoController

if TYPE_CHECKING:
    from iqoption_research.desktop.app import Dashboard


def _fmt_ts(ts: float | None) -> str:
    if ts is None or ts <= 0:
        return "-"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%m-%d %H:%M:%S")


class ExecutionCenterPanel:
    """Consolidated Execution Center tab inside Research Workstation V2."""

    def __init__(self, dashboard: Dashboard, notebook: ttk.Notebook) -> None:
        self.dashboard = dashboard
        self.root = dashboard.root
        self.controller = OfflineExecutionDemoController(self.root)
        self._account_options = self.controller.list_vault_accounts_safe()
        self._account_label_to_id: dict[str, str | None] = {
            opt.label: opt.account_id for opt in self._account_options
        }

        self.frame = ttk.Frame(notebook, padding=6)
        notebook.add(self.frame, text="Execution Center")

        # Top safety & mode banner
        banner = ttk.Frame(self.frame)
        banner.pack(fill="x", pady=(0, 3))
        ttk.Label(
            banner,
            text="Execution Center · Phase 50 Demo Trading Readiness",
            font=("Segoe UI", 11, "bold"),
            foreground="#17324d",
        ).pack(side="left", anchor="w")
        ttk.Label(
            banner,
            text="REAL DISABLED  |  PRACTICE: BLOCKED  |  PAPER SIMULATION READY",
            font=("Segoe UI", 8, "bold"),
            foreground="#9c3414",
        ).pack(side="right", anchor="e")

        # Row 1: Three uniform status cards (Mode | Account & Safety | Trading Overview)
        cards = ttk.Frame(self.frame)
        cards.pack(fill="x", pady=(0, 3))
        cards.columnconfigure(0, weight=1, uniform="ec_cards")
        cards.columnconfigure(1, weight=1, uniform="ec_cards")
        cards.columnconfigure(2, weight=1, uniform="ec_cards")

        self.lbl_mode_paper = self._build_mode_card(cards)
        self._build_account_safety_card(cards)
        self._build_trading_overview_card(cards)

        # Row 2: Compact two-line PAPER controls toolbar (zero live broker order buttons)
        self._build_controls_bar()

        # Row 3: Full-height tabbed notebook for Order Journal & Diagnostics
        self.diag_notebook = ttk.Notebook(self.frame)
        self.diag_notebook.pack(fill="both", expand=True, pady=(3, 0))

        journal_tab = ttk.Frame(self.diag_notebook, padding=4)
        self.diag_notebook.add(journal_tab, text="Order Journal (PAPER/SIMULATED)")
        self._build_order_journal(journal_tab)
        self._build_diagnostics_tabs()
        self.refresh_view()

    def _build_mode_card(self, parent: ttk.Frame) -> dict[str, ttk.Label]:
        box = ttk.LabelFrame(parent, text="Execution Mode", padding=5)
        box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        box.columnconfigure(1, weight=1)

        labels: dict[str, ttk.Label] = {}
        rows = (
            ("paper", "PAPER Simulation:", "ACTIVE (Air-gapped)"),
            ("practice", "Supervised Practice:", "BLOCKED"),
            ("broker_verif", "Broker Verification:", "0 Orders / 0 Settlements"),
            ("real_mode", "REAL Execution:", "PERMANENTLY DISABLED"),
        )
        for idx, (key, title, init) in enumerate(rows):
            ttk.Label(box, text=title, font=("Segoe UI", 8, "bold")).grid(
                row=idx, column=0, sticky="nw", pady=1
            )
            val = ttk.Label(
                box,
                text=init,
                font=("Segoe UI", 8, "bold" if key in {"real_mode", "practice"} else "normal"),
                foreground="#9c3414" if key in {"real_mode", "practice"} else "#17324d",
                wraplength=240,
            )
            val.grid(row=idx, column=1, sticky="w", padx=(4, 0), pady=1)
            labels[key] = val
        return labels

    def _build_account_safety_card(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Account & Safety", padding=5)
        box.grid(row=0, column=1, sticky="nsew", padx=(0, 4))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Vault Account:", font=("Segoe UI", 8, "bold")).grid(
            row=0, column=0, sticky="w", pady=1
        )
        self.var_vault_account = tk.StringVar(
            value=self._account_options[0].label if self._account_options else ""
        )
        self.cbo_vault = ttk.Combobox(
            box,
            textvariable=self.var_vault_account,
            values=list(self._account_label_to_id.keys()),
            state="readonly",
            width=26,
        )
        self.cbo_vault.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(4, 0), pady=1)
        self.cbo_vault.bind("<<ComboboxSelected>>", lambda _e: self._on_account_changed())

        self.lbl_wallet_authority = self._kv_row(box, 1, "Wallet Authority:", "-")
        self.lbl_broker_conn = self._kv_row(box, 2, "Broker Status:", "OFFLINE_PAPER_AIRGAPPED")
        self.lbl_risk_controls = self._kv_row(box, 3, "Risk Controls:", "14 Active")
        self.lbl_auth_status = self._kv_row(box, 4, "AI Authority:", "NONE (Stage C untouched)")

        btn_row = ttk.Frame(box)
        btn_row.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        self.btn_emergency_stop = ttk.Button(
            btn_row, text="Emergency Stop", command=self.on_emergency_stop
        )
        self.btn_emergency_stop.pack(side="left", padx=(0, 4))
        self.btn_reset_stop = ttk.Button(
            btn_row, text="Reset Stop (Offline)", command=self.on_reset_stop
        )
        self.btn_reset_stop.pack(side="left")

    def _build_trading_overview_card(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Trading Overview (Virtual PAPER)", padding=5)
        box.grid(row=0, column=2, sticky="nsew")
        box.columnconfigure(1, weight=1)

        self.lbl_balance = self._kv_row(box, 0, "Virtual Balance:", "$10,000.00")
        self.lbl_positions = self._kv_row(box, 1, "Open / Closed:", "0 / 0")
        self.lbl_wlt = self._kv_row(box, 2, "Wins / Losses / Ties:", "0 / 0 / 0")
        self.lbl_pnl = self._kv_row(box, 3, "Simulated P&L:", "$+0.00 (Assumed 85% Binary)")
        self.lbl_exposure = self._kv_row(box, 4, "Exposure / Uncertain:", "$0.00 / 0")

    @staticmethod
    def _kv_row(parent: ttk.Frame | ttk.LabelFrame, row: int, key: str, value: str) -> ttk.Label:
        ttk.Label(parent, text=key, font=("Segoe UI", 8, "bold")).grid(
            row=row, column=0, sticky="nw", pady=1
        )
        lbl = ttk.Label(parent, text=value, font=("Segoe UI", 8), wraplength=235)
        lbl.grid(row=row, column=1, columnspan=2, sticky="w", padx=(4, 0), pady=1)
        return lbl

    def _build_controls_bar(self) -> None:
        bar = ttk.LabelFrame(
            self.frame,
            text="Offline PAPER Controls (Zero Live Order Transmission)",
            padding=4,
        )
        bar.pack(fill="x", pady=(0, 2))

        row1 = ttk.Frame(bar)
        row1.pack(fill="x", pady=(0, 2))

        self.var_product = tk.StringVar(value=ProductType.BINARY.value)
        self.var_symbol = tk.StringVar(value="EURUSD")
        self.var_side = tk.StringVar(value=OrderSide.CALL.value)
        self.var_stake = tk.StringVar(value="25.0")

        ttk.Label(row1, text="Product:").pack(side="left")
        ttk.Combobox(
            row1,
            textvariable=self.var_product,
            values=[p.value for p in ProductType],
            state="readonly",
            width=8,
        ).pack(side="left", padx=(2, 6))

        ttk.Label(row1, text="Asset:").pack(side="left")
        ttk.Combobox(
            row1,
            textvariable=self.var_symbol,
            values=["EURUSD", "EURUSD_OTC", "EURUSD_OP", "GBPUSD", "USDJPY"],
            state="readonly",
            width=11,
        ).pack(side="left", padx=(2, 6))

        ttk.Label(row1, text="Side:").pack(side="left")
        ttk.Combobox(
            row1,
            textvariable=self.var_side,
            values=[OrderSide.CALL.value, OrderSide.PUT.value],
            state="readonly",
            width=5,
        ).pack(side="left", padx=(2, 6))

        ttk.Label(row1, text="Stake $:").pack(side="left")
        ttk.Entry(row1, textvariable=self.var_stake, width=6).pack(side="left", padx=(2, 8))

        self.btn_inject_signal = ttk.Button(
            row1, text="Inject Signal", command=self.on_inject_signal
        )
        self.btn_inject_signal.pack(side="left", padx=(0, 4))

        self.btn_advance_candle = ttk.Button(
            row1, text="Advance & Settle", command=self.on_advance_candle
        )
        self.btn_advance_candle.pack(side="left", padx=(0, 4))

        self.lbl_action_status = ttk.Label(row1, text="Ready (PAPER Offline)", foreground="#17324d")
        self.lbl_action_status.pack(side="right", padx=4)

        row2 = ttk.Frame(bar)
        row2.pack(fill="x")

        self.btn_run_ai_replay = ttk.Button(
            row2, text="Run Frozen 3-Model Replay", command=self.on_run_ai_replay
        )
        self.btn_run_ai_replay.pack(side="left", padx=(0, 4))

        self.btn_run_full_demo = ttk.Button(
            row2, text="Run Offline Acceptance Suite", command=self.on_run_full_suite
        )
        self.btn_run_full_demo.pack(side="left", padx=(0, 4))

        self.btn_restart_recovery = ttk.Button(
            row2, text="Simulate Restart Recovery", command=self.on_restart_recovery
        )
        self.btn_restart_recovery.pack(side="left", padx=(0, 4))

        self.btn_export_report = ttk.Button(
            row2, text="Export Report", command=self.on_export_report
        )
        self.btn_export_report.pack(side="left")

    def _build_order_journal(self, parent: ttk.Frame | ttk.LabelFrame) -> None:
        cols = (
            "order_id",
            "source",
            "product",
            "asset",
            "direction",
            "stake",
            "submitted",
            "expiration",
            "status",
            "broker_ticket",
            "settlement_evidence",
            "failure_reason",
        )
        headings = {
            "order_id": ("Order ID", 105),
            "source": ("Source", 130),
            "product": ("Product", 65),
            "asset": ("Asset", 85),
            "direction": ("Dir", 48),
            "stake": ("Stake", 65),
            "submitted": ("Submitted", 100),
            "expiration": ("Expiration", 100),
            "status": ("Lifecycle Status", 120),
            "broker_ticket": ("Broker Ticket", 118),
            "settlement_evidence": ("Settlement Evidence", 165),
            "failure_reason": ("Failure Reason", 155),
        }
        host = ttk.Frame(parent)
        host.pack(fill="both", expand=True)

        self.tree_orders = ttk.Treeview(host, columns=cols, show="headings", height=9)
        ysb = ttk.Scrollbar(host, orient="vertical", command=self.tree_orders.yview)
        xsb = ttk.Scrollbar(host, orient="horizontal", command=self.tree_orders.xview)
        self.tree_orders.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        for col in cols:
            text, width = headings[col]
            self.tree_orders.heading(col, text=text)
            self.tree_orders.column(col, width=width, minwidth=50, stretch=True)

        self.tree_orders.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        host.rowconfigure(0, weight=1)
        host.columnconfigure(0, weight=1)

    def _build_diagnostics_tabs(self) -> None:
        # Tab 1: Execution Readiness & Capability Matrix
        cap_tab = ttk.Frame(self.diag_notebook, padding=4)
        self.diag_notebook.add(cap_tab, text="Readiness & Capability Matrix")
        cap_cols = (
            "operation_id",
            "product",
            "status",
            "evidence_level",
            "live_allowed",
            "blockers",
        )
        self.tree_matrix = ttk.Treeview(cap_tab, columns=cap_cols, show="headings", height=9)
        cap_ysb = ttk.Scrollbar(cap_tab, orient="vertical", command=self.tree_matrix.yview)
        self.tree_matrix.configure(yscrollcommand=cap_ysb.set)
        for col, title, w in (
            ("operation_id", "Operation", 200),
            ("product", "Product", 70),
            ("status", "Status", 165),
            ("evidence_level", "Protocol Evidence", 160),
            ("live_allowed", "Live Allowed", 80),
            ("blockers", "Blockers", 260),
        ):
            self.tree_matrix.heading(col, text=title)
            self.tree_matrix.column(col, width=w, stretch=True)
        self.tree_matrix.pack(side="left", fill="both", expand=True)
        cap_ysb.pack(side="right", fill="y")

        # Tab 2: Protocol Evidence & 6-Layer Authority
        self.txt_protocol = self._diag_text_tab("Protocol & Wallet Authority")
        # Tab 3: Rejected Orders
        self.txt_rejected = self._diag_text_tab("Rejected Orders")
        # Tab 4: Connection Events & Recovery Status
        self.txt_recovery = self._diag_text_tab("Connection & Recovery Status")

    def _diag_text_tab(self, title: str) -> tk.Text:
        frame = ttk.Frame(self.diag_notebook, padding=4)
        self.diag_notebook.add(frame, text=title)
        text = tk.Text(
            frame,
            height=6,
            wrap="word",
            font=("Consolas", 9),
            background="#ffffff",
            foreground="#20364a",
        )
        ysb = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=ysb.set)
        text.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        text.configure(state="disabled")
        return text

    @staticmethod
    def _set_text(widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _on_account_changed(self) -> None:
        self.controller.selected_vault_account_id = self._account_label_to_id.get(
            self.var_vault_account.get()
        )
        self.refresh_view()

    def on_inject_signal(self) -> None:
        try:
            stake = float(self.var_stake.get())
        except ValueError:
            messagebox.showerror("Invalid Stake", "Enter a valid numeric stake amount.")
            return
        res = self.controller.inject_test_signal(
            product_type=ProductType(self.var_product.get()),
            symbol=self.var_symbol.get(),
            side=OrderSide(self.var_side.get()),
            stake=stake,
        )
        self.lbl_action_status.configure(
            text=f"{'Opened' if res['permitted'] else 'Rejected'}: {res['order_id']}"
        )
        self.refresh_view()

    def on_advance_candle(self) -> None:
        res = self.controller.advance_and_settle(price_delta_pips=4.5)
        self.lbl_action_status.configure(
            text=f"Settled {res['settled_count']} order(s); Balance ${res['balance']:,.2f}"
        )
        self.refresh_view()

    def on_run_ai_replay(self) -> None:
        rep = self.controller.run_frozen_three_model_replay()
        cand_a = rep["candidates"].get("p48_exp_cand_a_m1_h15_hgb_v1", {})
        self.lbl_action_status.configure(
            text=(
                f"AI Replay ({rep.get('replay_source', 'OFFLINE')}): "
                f"Cand A orders={cand_a.get('accepted_orders_count', 0)}"
            )
        )
        self.refresh_view()

    def on_run_full_suite(self) -> None:
        self.controller.run_full_offline_acceptance_suite()
        self.lbl_action_status.configure(text="Offline Acceptance Suite Passed")
        self.refresh_view()

    def on_restart_recovery(self) -> None:
        res = self.controller.simulate_restart_recovery()
        self.lbl_action_status.configure(
            text=f"Restart Recovered {res['restored_orders_count']} orders; Bal ${res['restored_balance']:,.2f}"
        )
        self.refresh_view()

    def on_emergency_stop(self) -> None:
        rep = self.controller.trigger_emergency_stop()
        self.lbl_action_status.configure(text=f"EMERGENCY STOP ACTIVE ({rep['circuit_state']})")
        self.refresh_view()

    def on_reset_stop(self) -> None:
        self.controller.reset_emergency_stop()
        self.lbl_action_status.configure(text="Circuit breaker reset (PAPER Offline)")
        self.refresh_view()

    def on_export_report(self) -> None:
        chosen = filedialog.asksaveasfilename(
            title="Export Phase 50 Offline Execution Report",
            defaultextension=".json",
            initialfile="phase50_offline_evidence_report.json",
            filetypes=[("JSON files", "*.json")],
        )
        if not chosen:
            return
        from pathlib import Path

        out = self.controller.export_evidence_bundle(Path(chosen))
        self.lbl_action_status.configure(text=f"Exported: {out['output_path']}")

    def refresh_view(self) -> None:
        """Synchronize all dashboard cards, order journal, and diagnostics from engine state."""
        eng = self.controller.engine
        readiness = self.controller.evaluate_supervised_readiness()
        summary = self.controller.matrix.summary_counts()
        portfolio = eng.portfolio
        all_orders = eng.state_machine.all_orders
        closed_count = sum(
            1
            for o in all_orders
            if o.state in {OrderState.SETTLED, OrderState.CLOSED, OrderState.SIMULATED_SETTLED}
        )
        pending_uncertain = sum(1 for o in all_orders if o.is_uncertain)

        # Execution mode labels
        self.lbl_mode_paper["practice"].configure(
            text=f"{readiness.boundary_status.value} ({', '.join(readiness.blockers[:2])})"
        )
        self.lbl_mode_paper["broker_verif"].configure(
            text=(
                f"{summary.get('broker_order_verified', 0)} Order Verified / "
                f"{summary.get('settlement_verified', 0)} Settlement Verified / "
                f"{summary.get('read_only_broker_verified', 0)} Read-Only"
            )
        )

        # Account & safety labels
        self.lbl_wallet_authority.configure(
            text=f"{self.controller.observed_wallet_classification} | Layer 5/6 UNVERIFIED"
        )
        self.lbl_broker_conn.configure(
            text=(
                f"{'CONNECTED (SIM)' if eng.is_broker_connected else 'DISCONNECTED'} · "
                f"Circuit: {eng.circuit_state.value}"
            )
        )
        rc = eng.risk_manager.config
        self.lbl_risk_controls.configure(
            text=(
                f"Max=${rc.max_stake_per_trade:.0f} | Conc={rc.max_concurrent_positions} | "
                f"Exp=${rc.max_simultaneous_exposure:.0f} | Int={rc.min_order_interval_seconds:.0f}s"
            )
        )
        if eng.risk_manager.is_halted:
            self.lbl_auth_status.configure(
                text=f"HALTED ({eng.risk_manager.halt_reason[:40]})", foreground="#9c3414"
            )
        else:
            self.lbl_auth_status.configure(
                text="PAPER Offline Permitted · Live NONE", foreground="#17324d"
            )

        # Trading overview labels
        self.lbl_balance.configure(
            text=f"${portfolio.current_balance:,.2f} (Cash ${portfolio.available_cash:,.2f})"
        )
        self.lbl_positions.configure(text=f"{len(eng.open_orders)} / {closed_count}")
        self.lbl_wlt.configure(
            text=f"{portfolio.winning_trades} / {portfolio.losing_trades} / {portfolio.flat_trades}"
        )
        self.lbl_pnl.configure(
            text=f"${portfolio.total_pnl:+,.2f} (Fees ${portfolio.total_fees:.2f})"
        )
        self.lbl_exposure.configure(
            text=f"${eng.state_machine.total_open_exposure:,.2f} / {pending_uncertain} uncertain"
        )

        # Populate Order Journal (includes main engine orders + AI PAPER replay sample orders)
        journal_orders = list(all_orders) + list(self.controller.ai_reference_orders)
        for item in self.tree_orders.get_children():
            self.tree_orders.delete(item)
        for o in reversed(journal_orders[-100:]):
            self.tree_orders.insert(
                "",
                "end",
                values=self._order_row_values(o),
            )

        # Populate Capability Matrix
        for item in self.tree_matrix.get_children():
            self.tree_matrix.delete(item)
        for op in self.controller.matrix.filter_operations():
            self.tree_matrix.insert(
                "",
                "end",
                values=(
                    op.operation_id,
                    op.product_type.value if op.product_type else "Global",
                    op.status.value.upper(),
                    op.evidence_level,
                    "Yes" if op.live_broker_allowed else "No",
                    ", ".join(op.blockers) if op.blockers else "-",
                ),
            )

        # Diagnostics Text Tabs
        proto_payload = {
            "supervised_boundary": readiness.boundary_status.value,
            "blockers": list(readiness.blockers),
            "observed_phase48_classification": self.controller.observed_wallet_classification,
            "capability_counts": summary,
            "last_ai_bridge_summary": self.controller.last_ai_bridge_report,
            "last_supervised_rehearsal": self.controller.last_rehearsal_report,
        }
        self._set_text(self.txt_protocol, json.dumps(proto_payload, indent=2))

        rej_lines = [
            f"{o.order_id} | {o.product_type.value}:{o.symbol} | {o.side.value} | "
            f"${o.stake:.2f} | {o.rejection_reason or 'rejected'}"
            for o in eng.rejected_orders
        ]
        self._set_text(
            self.txt_rejected,
            "\n".join(rej_lines)
            if rej_lines
            else "No rejected orders recorded in current journal.",
        )

        rec_payload = {
            "sqlite_journal_path": str(self.controller.db_path),
            "persisted_orders_count": len(all_orders),
            "uncertain_orders": [o.order_id for o in eng.uncertain_orders],
            "risk_state": eng.risk_manager.to_state_dict(),
            "connection_events": eng.connection_events,
        }
        self._set_text(self.txt_recovery, json.dumps(rec_payload, indent=2))

    @staticmethod
    def _order_row_values(o: PaperOrder) -> tuple[str, ...]:
        source = str(o.metadata.get("signal_source") or o.signal_id)
        broker_ticket = (
            str(o.broker_order_id)
            if o.broker_order_id and not str(o.broker_order_id).startswith("sim_")
            else "NONE (PAPER)"
        )
        settlement_ev = o.close_reason or str(o.metadata.get("candle_direction_outcome") or "-")
        if o.simulated_pnl is not None:
            settlement_ev = f"${o.simulated_pnl:+.2f} | {settlement_ev}"
        return (
            o.order_id,
            source,
            o.product_type.value,
            o.symbol,
            o.side.value.upper(),
            f"${o.stake:.2f}",
            _fmt_ts(o.submitted_ts or o.created_ts),
            _fmt_ts(o.expiry_ts),
            o.state.value.upper(),
            broker_ticket,
            settlement_ev,
            o.rejection_reason or o.reconciliation_reason or "-",
        )

    def tick(self) -> None:
        """Periodic idle tick hook invoked by Dashboard."""

    def busy(self) -> bool:
        """Execution Center runs deterministic offline state synchronously."""
        return False

    def shutdown(self) -> None:
        """Release SQLite journal handles cleanly on workstation shutdown."""
        self.controller.engine.close()
