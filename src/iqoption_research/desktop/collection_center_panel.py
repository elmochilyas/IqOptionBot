"""Desktop Collection Center panel for operator-governed market data acquisition (Phase 44).

Enforces strict operator safety, verifiable coverage, and non-blocking background collection:
1. Saved account selector from secure Windows credential vault.
2. Symbol selector with explicit Regular vs OTC instrument indicator.
3. Base timeframe selection (default 60s).
4. Explicit UTC date/time range selection with synchronized Casablanca local time (UTC+1).
5. Quick presets (30, 90, 180, 365 days).
6. Real-time operational metrics: cursor, unique retained, duplicates, conflicts, receipts, connection state.
7. Operator controls: Start, Stop/Pause, Resume, Verify & Snapshot.
8. Background worker thread updating UI via thread-safe window.after().
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import time
import tkinter as tk
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any

from iqoption_research.market_data.collection_engine import (
    CollectionProgress,
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.market_data.sessions import (
    format_casablanca_display,
    get_market_schedule,
    parse_explicit_utc_datetime,
)
from iqoption_research.research.dataset_manager import ResearchDatasetManager

if TYPE_CHECKING:
    from iqoption_research.desktop.app import Dashboard

logger = logging.getLogger(__name__)

UNVERIFIED_OTC_MESSAGE = (
    "OTC schedule unverified; verified collection/snapshot is disabled for this instrument."
)


class CollectionCenterPanel:
    """Dedicated Desktop Collection Center tab inside Research Workstation V2."""

    btn_plan: ttk.Button
    btn_start: ttk.Button
    btn_stop: ttk.Button
    btn_resume: ttk.Button
    btn_verify: ttk.Button
    btn_export_report: ttk.Button
    lbl_job_id: ttk.Label
    lbl_cursor: ttk.Label
    lbl_range_req: ttk.Label
    lbl_candles: ttk.Label
    lbl_duplicates: ttk.Label
    lbl_conflicts: ttk.Label
    lbl_intervals: ttk.Label
    lbl_receipts: ttk.Label

    def __init__(self, dashboard: Dashboard, notebook: ttk.Notebook) -> None:
        self.dashboard = dashboard
        self.root = dashboard.root
        self.coordinator = HistoricalCollectionCoordinator(self.root)
        self.dataset_manager = ResearchDatasetManager(self.root)

        self.current_job_id: str | None = None
        self._cancel_requested = False
        self._worker_thread: threading.Thread | None = None
        self._collection_start_time: float | None = None
        self._active_operation: str | None = None
        self._verify_cancel_requested = False
        self._last_verification_report: dict[str, Any] | None = None
        self._op_start_time: float | None = None
        self._accounts_map: dict[str, str] = {}
        self._jobs_map: dict[str, str] = {}
        self._ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._ui_poll_pending = False
        self._ui_poll_after_id: str | None = None
        # Every background operation carries a generation token. Results from a
        # superseded generation or a job other than the selected one are discarded.
        self._operation_generation = 0
        self._plan_request: tuple[str, str, str, int] | None = None
        self._plan_base_job_id: str | None = None
        self._pending_start_after_plan = False

        # Main frame
        self.frame = ttk.Frame(notebook, padding=12)
        notebook.add(self.frame, text="Collection Center")

        # Top Header & Safety Invariants
        header = ttk.Frame(self.frame)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(
            header,
            text="Operational Market Data Collection Center · Phase 44",
            font=("Segoe UI", 13, "bold"),
            foreground="#17324d",
        ).pack(anchor="w")

        ttk.Label(
            header,
            text=(
                "READ-ONLY MARKET DATA COLLECTION ONLY    |    ZERO TRADING EXECUTION    |    "
                "NO WALLET MUTATIONS    |    FAIL-CLOSED GOVERNANCE"
            ),
            font=("Segoe UI", 9, "bold"),
            foreground="#9c3414",
        ).pack(anchor="w", pady=(2, 4))

        # Main layout: Split into Left (Configuration & Controls) and Right (Operational Status & Log)
        paned = ttk.PanedWindow(self.frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left_pane = ttk.Frame(paned, padding=8)
        right_pane = ttk.Frame(paned, padding=8)
        paned.add(left_pane, weight=2)
        paned.add(right_pane, weight=3)

        # Reserve the controls before allocating remaining height to parameters.
        self._build_controls_ui(left_pane)
        config_host = ttk.Frame(left_pane)
        config_host.pack(fill="both", expand=True)
        canvas = tk.Canvas(config_host, highlightthickness=0, width=1, height=1)
        scrollbar = ttk.Scrollbar(config_host, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        config_content = ttk.Frame(canvas)
        content_id = canvas.create_window((0, 0), window=config_content, anchor="nw")
        config_content.bind(
            "<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", lambda event: canvas.itemconfigure(content_id, width=event.width)
        )
        self._build_config_ui(config_content)
        self._build_status_ui(right_pane)

        # Refresh accounts, jobs, and initialize dates
        self._refresh_accounts()
        self._refresh_jobs()
        self._apply_preset(30)
        self._schedule_ui_poll()

    def _dispatch_to_ui(self, callback: Callable[[], None]) -> None:
        """Thread-safe dispatch of a callback to the main Tkinter thread."""
        self._ui_queue.put(callback)

    def _schedule_ui_poll(self) -> None:
        """Keep exactly one recurring UI poll scheduled on the Tk main thread."""
        if self._ui_poll_pending or self._ui_poll_after_id is not None:
            return
        with contextlib.suppress(Exception):
            if self.dashboard.window.winfo_exists():
                self._ui_poll_pending = True
                self._ui_poll_after_id = self.dashboard.window.after(20, self._scheduled_poll)

    def _scheduled_poll(self) -> None:
        """Tk callback wrapper that owns the recurring poll timer."""
        self._ui_poll_after_id = None
        self._ui_poll_pending = False
        self._poll_ui_queue()
        self._schedule_ui_poll()

    def shutdown(self) -> None:
        """Cancel pending UI callbacks before the Tk window is destroyed."""
        self._ui_poll_pending = True
        if self._ui_poll_after_id is not None:
            with contextlib.suppress(Exception):
                self.dashboard.window.after_cancel(self._ui_poll_after_id)
            self._ui_poll_after_id = None
        while not self._ui_queue.empty():
            with contextlib.suppress(queue.Empty):
                self._ui_queue.get_nowait()

    def _poll_ui_queue(self) -> None:
        """Drain queued UI callbacks strictly on the main thread."""
        # Manual pumps may call this directly. They must not disturb the one
        # recurring Tk timer owned by ``_scheduled_poll``; otherwise every pump
        # would leave another stale ``after`` callback queued for a dead widget.
        while not self._ui_queue.empty():
            try:
                cb = self._ui_queue.get_nowait()
                cb()
            except queue.Empty:
                break
            except Exception as exc:
                logger.exception("Error executing queued UI callback: %s", exc)

    def _next_operation_generation(self) -> int:
        """Invalidate callbacks from every previously started background operation."""
        self._operation_generation += 1
        return self._operation_generation

    def _is_current_worker_result(self, generation: int, job_id: str | None) -> bool:
        """Reject stale worker results before they can mutate the visible job."""
        if generation != self._operation_generation:
            logger.debug("Discarding stale worker result from generation %s", generation)
            return False
        if job_id is not None and job_id != self.current_job_id:
            logger.debug(
                "Discarding worker result for job %s; selected job is %s",
                job_id,
                self.current_job_id,
            )
            return False
        return True

    def _build_config_ui(self, parent: ttk.Frame) -> None:
        cfg_frame = ttk.LabelFrame(parent, text="Collection Parameters", padding=10)
        cfg_frame.pack(fill="x", pady=(0, 8))

        # 1. Saved Account Selector
        ttk.Label(cfg_frame, text="Broker Account (Vault):").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self.combo_account = ttk.Combobox(cfg_frame, state="readonly", width=36)
        self.combo_account.grid(row=0, column=1, sticky="ew", pady=4, padx=4)

        # 2. Symbol & Instrument Type
        ttk.Label(cfg_frame, text="Symbol / Asset:").grid(row=1, column=0, sticky="w", pady=4)
        sym_frame = ttk.Frame(cfg_frame)
        sym_frame.grid(row=1, column=1, sticky="ew", pady=4, padx=4)

        self.combo_symbol = ttk.Combobox(
            sym_frame,
            values=["EURUSD", "GBPUSD", "USDJPY", "EURUSD-OTC", "GBPUSD-OTC"],
            state="readonly",
            width=14,
        )
        self.combo_symbol.set("EURUSD")
        self.combo_symbol.pack(side="left")
        self.combo_symbol.bind("<<ComboboxSelected>>", self._on_symbol_changed)

        self.lbl_instrument_type = ttk.Label(
            sym_frame,
            text="[Regular Forex - Interbank Schedule]",
            foreground="#1e6b20",
            font=("Segoe UI", 9, "bold"),
        )
        self.lbl_instrument_type.pack(side="left", padx=8)

        # 3. Base Timeframe
        ttk.Label(cfg_frame, text="Timeframe:").grid(row=2, column=0, sticky="w", pady=4)
        self.combo_timeframe = ttk.Combobox(
            cfg_frame,
            values=["60 (M1 - 1 Minute)", "300 (M5 - 5 Minutes)", "900 (M15 - 15 Minutes)"],
            state="readonly",
            width=36,
        )
        self.combo_timeframe.current(0)
        self.combo_timeframe.grid(row=2, column=1, sticky="ew", pady=4, padx=4)

        # 4. UTC Date Range Pickers & Local Casablanca Displays
        ttk.Label(cfg_frame, text="Start UTC (ISO):").grid(row=3, column=0, sticky="w", pady=4)
        self.entry_start_utc = ttk.Entry(cfg_frame, width=38)
        self.entry_start_utc.grid(row=3, column=1, sticky="ew", pady=4, padx=4)
        self.entry_start_utc.bind("<KeyRelease>", self._update_casablanca_labels)

        self.lbl_start_casa = ttk.Label(
            cfg_frame,
            text="Local (Casablanca UTC+1): --",
            foreground="#555",
            font=("Segoe UI", 8),
        )
        self.lbl_start_casa.grid(row=4, column=1, sticky="w", padx=4)

        ttk.Label(cfg_frame, text="End UTC (ISO):").grid(row=5, column=0, sticky="w", pady=4)
        self.entry_end_utc = ttk.Entry(cfg_frame, width=38)
        self.entry_end_utc.grid(row=5, column=1, sticky="ew", pady=4, padx=4)
        self.entry_end_utc.bind("<KeyRelease>", self._update_casablanca_labels)

        self.lbl_end_casa = ttk.Label(
            cfg_frame,
            text="Local (Casablanca UTC+1): --",
            foreground="#555",
            font=("Segoe UI", 8),
        )
        self.lbl_end_casa.grid(row=6, column=1, sticky="w", padx=4)

        # 5. Range Presets
        ttk.Label(cfg_frame, text="Presets:").grid(row=7, column=0, sticky="w", pady=6)
        preset_frame = ttk.Frame(cfg_frame)
        preset_frame.grid(row=7, column=1, sticky="w", pady=6, padx=4)

        for days in (30, 90, 180, 365):
            btn = ttk.Button(
                preset_frame,
                text=f"{days} Days",
                command=self._make_preset_handler(days),
                width=8,
            )
            btn.pack(side="left", padx=2)

        # 6. Existing Jobs Selector
        ttk.Label(cfg_frame, text="Existing Jobs:").grid(row=8, column=0, sticky="w", pady=4)
        job_frame = ttk.Frame(cfg_frame)
        job_frame.grid(row=8, column=1, sticky="ew", pady=4, padx=4)

        self.combo_jobs = ttk.Combobox(job_frame, state="readonly", width=26)
        self.combo_jobs.pack(side="left", fill="x", expand=True)
        self.combo_jobs.bind("<<ComboboxSelected>>", self._on_job_selected)

        self.btn_refresh_jobs = ttk.Button(
            job_frame, text="Refresh", command=self._refresh_jobs, width=8
        )
        self.btn_refresh_jobs.pack(side="left", padx=(4, 0))

    def _make_preset_handler(self, days: int) -> Callable[[], None]:
        return lambda: self._apply_preset(days)

    def _build_controls_ui(self, parent: ttk.Frame) -> None:
        ctl_frame = ttk.LabelFrame(parent, text="Operator Controls", padding=10)
        ctl_frame.pack(side="bottom", fill="x", pady=(8, 0))
        ctl_frame.columnconfigure((0, 1), weight=1, uniform="controls")
        controls = (
            ("btn_plan", "Inspect & Plan", self._plan_collection, "normal"),
            ("btn_start", "Start Collection", self._start_collection, "normal"),
            ("btn_stop", "Stop / Pause", self._stop_clicked, "disabled"),
            ("btn_resume", "Resume Interrupted", self._resume_collection, "disabled"),
            ("btn_verify", "Verify Integrity & Snapshot", self._verify_and_snapshot, "normal"),
        )
        for index, (name, text, command, state) in enumerate(controls):
            button = ttk.Button(ctl_frame, text=text, command=command, state=state)
            button.grid(
                row=index // 2,
                column=index % 2,
                columnspan=2 if index == 4 else 1,
                sticky="ew",
                padx=2,
                pady=4,
            )
            setattr(self, name, button)
        # Explicit minima keep spanning-button requests and theme rounding from
        # shrinking either action column below its longest label.
        column_minimum = max(getattr(self, name).winfo_reqwidth() + 4 for name, *_ in controls[:-1])
        ctl_frame.columnconfigure((0, 1), minsize=column_minimum)

    def _build_status_ui(self, parent: ttk.Frame) -> None:
        status_frame = ttk.LabelFrame(parent, text="Live Operational Status", padding=10)
        status_frame.pack(fill="both", expand=True)

        # Connection / Execution badge
        state_row = ttk.Frame(status_frame)
        state_row.pack(fill="x", pady=(0, 6))
        ttk.Label(state_row, text="Status:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.lbl_status_badge = ttk.Label(
            state_row,
            text="IDLE",
            font=("Segoe UI", 10, "bold"),
            foreground="#444",
        )
        self.lbl_status_badge.pack(side="left", padx=8)

        self.lbl_elapsed = ttk.Label(state_row, text="Elapsed: 00:00:00", foreground="#555")
        self.lbl_elapsed.pack(side="right")

        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, maximum=100.0, mode="determinate")
        self.progress_bar.pack(fill="x", pady=(0, 8))

        # Metrics Grid
        grid_frame = ttk.Frame(status_frame)
        grid_frame.pack(fill="x", pady=(0, 8))

        labels = [
            ("Job ID:", "lbl_job_id", 0, 0),
            ("Active Cursor:", "lbl_cursor", 0, 2),
            ("Requested Range:", "lbl_range_req", 1, 0),
            ("Candles Received:", "lbl_candles", 1, 2),
            ("Duplicates Discarded:", "lbl_duplicates", 2, 0),
            ("Conflicts Quarantined:", "lbl_conflicts", 2, 2),
            ("Missing Intervals:", "lbl_intervals", 3, 0),
            ("Receipts Emitted:", "lbl_receipts", 3, 2),
        ]

        for text, attr_name, row, col in labels:
            ttk.Label(grid_frame, text=text, font=("Segoe UI", 9, "bold")).grid(
                row=row, column=col, sticky="w", pady=2, padx=(4, 2)
            )
            lbl = ttk.Label(grid_frame, text="--", foreground="#17324d")
            lbl.grid(row=row, column=col + 1, sticky="w", pady=2, padx=(0, 12))
            setattr(self, attr_name, lbl)

        # Reason for stop/failure & Export Report button
        log_header = ttk.Frame(status_frame)
        log_header.pack(fill="x", pady=(4, 2))
        ttk.Label(log_header, text="Diagnostics & Output Log:").pack(side="left")
        self.btn_export_report = ttk.Button(
            log_header,
            text="Export Report JSON",
            command=self._export_report,
            state="disabled",
            width=18,
        )
        self.btn_export_report.pack(side="right")
        self.txt_log = tk.Text(status_frame, height=12, wrap="word", font=("Consolas", 9))
        self.txt_log.pack(fill="both", expand=True)

    def _log(self, message: str) -> None:
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        self.txt_log.insert("end", f"[{ts} UTC] {message}\n")
        self.txt_log.see("end")

    def _on_symbol_changed(self, _event: Any = None) -> None:
        sym = self.combo_symbol.get().strip().upper()
        if "OTC" in sym:
            self.lbl_instrument_type.config(
                text="[OTC Broker Schedule - Unverified; collection/snapshot disabled]",
                foreground="#b35900",
            )
        else:
            self.lbl_instrument_type.config(
                text="[Regular Forex - Interbank Schedule]", foreground="#1e6b20"
            )

    def _resolve_job_symbol(self, job_id: str) -> str:
        """Resolve a job's instrument from progress/checkpoint, not the visible selector."""
        try:
            return self.coordinator.get_progress(job_id).symbol.strip()
        except Exception as exc:  # noqa: BLE001 - fall back to the durable checkpoint
            logger.debug("Could not resolve job symbol from progress: %s", exc)

        try:
            ckpt_path = self.coordinator.checkpoints_dir / f"{job_id}.json"
            if ckpt_path.is_file():
                symbol = str(json.loads(ckpt_path.read_text(encoding="utf-8")).get("symbol", ""))
                if symbol.strip():
                    return symbol.strip()
        except Exception as exc:  # noqa: BLE001 - fall back to the visible operator input
            logger.debug("Could not resolve job symbol from checkpoint: %s", exc)

        return self.combo_symbol.get().strip().upper()

    @staticmethod
    def _is_otc_symbol(symbol: str) -> bool:
        """Classify OTC instruments, including symbols not in the registry."""
        try:
            return InstrumentRegistry.get(symbol).is_otc
        except Exception:  # noqa: BLE001 - unsupported OTC variants are still OTC by name
            return "OTC" in symbol.strip().upper()

    def _unverified_otc_block_reason(self, job_id: str) -> str | None:
        """Return an operator-facing block reason when OTC verification is unavailable."""
        symbols = [self.combo_symbol.get().strip().upper()]
        if job_id:
            symbols.append(self._resolve_job_symbol(job_id))

        for symbol in dict.fromkeys(s for s in symbols if s):
            if self._is_otc_symbol(symbol) and not get_market_schedule(symbol).is_schedule_verified:
                return f"{symbol}: {UNVERIFIED_OTC_MESSAGE}"
        return None

    def _apply_preset(self, days: int) -> None:
        now_dt = datetime.now(UTC).replace(microsecond=0)
        start_dt = now_dt - timedelta(days=days)
        self.entry_start_utc.delete(0, "end")
        self.entry_start_utc.insert(0, start_dt.strftime("%Y-%m-%d %H:%M:%S"))
        self.entry_end_utc.delete(0, "end")
        self.entry_end_utc.insert(0, now_dt.strftime("%Y-%m-%d %H:%M:%S"))
        self._update_casablanca_labels()

    def _update_casablanca_labels(self, _event: Any = None) -> None:
        s_text = self.entry_start_utc.get().strip()
        e_text = self.entry_end_utc.get().strip()

        try:
            s_dt = parse_explicit_utc_datetime(s_text)
            s_casa = format_casablanca_display(s_dt)
            self.lbl_start_casa.config(text=f"Local (Casablanca): {s_casa}", foreground="#1e6b20")
        except Exception:  # noqa: BLE001
            self.lbl_start_casa.config(
                text="Local (Casablanca): Invalid UTC format", foreground="#a82315"
            )

        try:
            e_dt = parse_explicit_utc_datetime(e_text)
            e_casa = format_casablanca_display(e_dt)
            self.lbl_end_casa.config(text=f"Local (Casablanca): {e_casa}", foreground="#1e6b20")
        except Exception:  # noqa: BLE001
            self.lbl_end_casa.config(
                text="Local (Casablanca): Invalid UTC format", foreground="#a82315"
            )

    def _refresh_accounts(self) -> None:
        with contextlib.suppress(Exception):
            from iqoption_research.auth.vault import WindowsCredentialVault

            store = WindowsCredentialVault().account_store()
            accounts = store.list_accounts()
            self._accounts_map.clear()
            labels = []
            for a in accounts:
                lbl = f"{a.label} ({a.account_id[:6]}...)"
                self._accounts_map[lbl] = a.account_id
                labels.append(lbl)
            self.combo_account["values"] = labels
            if labels:
                self.combo_account.current(0)
            else:
                self.combo_account["values"] = ["No vault accounts (offline mode)"]
                self.combo_account.current(0)

    def _refresh_jobs(self) -> None:
        with contextlib.suppress(Exception):
            jobs = self.coordinator.list_jobs()
            self._jobs_map.clear()
            labels = []
            for j in jobs:
                lbl = f"{j.job_id} [{j.status.value}]"
                self._jobs_map[lbl] = j.job_id
                labels.append(lbl)
            self.combo_jobs["values"] = labels
            if labels and not self.current_job_id:
                self.combo_jobs.current(0)
                self._on_job_selected()

    def _on_job_selected(self, _event: Any = None) -> None:
        selected_lbl = self.combo_jobs.get()
        job_id = self._jobs_map.get(selected_lbl)
        if job_id:
            self.current_job_id = job_id
            try:
                progress = self.coordinator.get_progress(job_id)
                self._update_progress_display(progress)
                self._log(f"Selected existing job: {job_id} ({progress.status.value})")
                if progress.status in (
                    CollectionStatus.CANCELLED,
                    CollectionStatus.INTERRUPTED,
                    CollectionStatus.PARTIALLY_COLLECTED,
                    CollectionStatus.PAUSED,
                    CollectionStatus.FAILED,
                ):
                    self.btn_resume.config(state="normal")
                    self.btn_start.config(state="disabled")
                elif progress.status == CollectionStatus.COMPLETED_VERIFIED:
                    self.btn_resume.config(state="disabled")
                    self.btn_start.config(state="disabled")
                else:
                    self.btn_start.config(state="normal")
                    self.btn_resume.config(state="disabled")
            except Exception as exc:  # noqa: BLE001
                self._log(f"Failed loading job progress: {exc}")

    def _get_timeframe_seconds(self) -> int:
        val = self.combo_timeframe.get()
        if "300" in val:
            return 300
        if "900" in val:
            return 900
        return 60

    def _confirm_live_connection(self, job_id: str, account_id: str, *, resume: bool) -> bool:
        """Require explicit operator confirmation before opening the live read-only broker."""
        symbol = self.combo_symbol.get().strip().upper() or "UNKNOWN"
        timeframe = self._get_timeframe_seconds()
        start_text = self.entry_start_utc.get().strip()
        end_text = self.entry_end_utc.get().strip()
        try:
            progress = self.coordinator.get_progress(job_id)
            symbol = progress.symbol
            timeframe = progress.timeframe_seconds
            start_text = progress.requested_from_utc
            end_text = progress.requested_to_utc
        except Exception as exc:  # noqa: BLE001 - fall back to the visible operator inputs
            logger.debug("Could not load job details for confirmation: %s", exc)

        action = "Resume" if resume else "Start"
        message = (
            f"Confirm {action.lower()} of a LIVE READ-ONLY collection connection.\n\n"
            f"Job ID: {job_id}\n"
            f"Account ID: {account_id}\n"
            f"UTC range: {start_text} .. {end_text}\n"
            f"Symbol: {symbol}\n"
            f"Timeframe: {timeframe} seconds\n\n"
            "No orders, wallet changes, or unattended operation will be performed."
        )
        confirmed = messagebox.askyesno("Confirm Live Read-Only Collection", message)
        if not confirmed:
            self._log("Operator declined live connection; no broker access was attempted.")
        return bool(confirmed)

    def _plan_collection(self, *, start_after: bool = False) -> None:
        """Start planning on a worker thread; never block the Tk event loop."""
        if self._active_operation is not None:
            messagebox.showwarning(
                "Operation in Progress",
                f"An operation ({self._active_operation}) is currently in progress. Please wait.",
            )
            return

        symbol = self.combo_symbol.get().strip().upper()
        s_text = self.entry_start_utc.get().strip()
        e_text = self.entry_end_utc.get().strip()
        tf = self._get_timeframe_seconds()

        try:
            s_dt = parse_explicit_utc_datetime(s_text)
            e_dt = parse_explicit_utc_datetime(e_text)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Planning Error", f"Invalid UTC range:\n{exc}")
            self._log(f"Planning failed: {exc}")
            return

        selected_account_label = self.combo_account.get()
        selected_account_id = self._accounts_map.get(selected_account_label)
        request: tuple[str, str, str, int] = (symbol, s_dt.isoformat(), e_dt.isoformat(), tf)
        self._plan_request = request
        self._plan_base_job_id = self.current_job_id
        self._pending_start_after_plan = bool(start_after)
        generation = self._next_operation_generation()
        self._set_controls_state("planning")
        self.lbl_status_badge.config(text="PLANNING...", foreground="#17324d")
        self._log(f"Analyzing existing coverage for {symbol} in background...")
        self._worker_thread = threading.Thread(
            target=self._run_planning_thread,
            args=(*request, selected_account_id, generation),
            daemon=True,
        )
        self._worker_thread.start()

    def _run_planning_thread(
        self,
        symbol: str,
        start_utc: str,
        end_utc: str,
        timeframe_seconds: int,
        account_id: str | None,
        generation: int,
    ) -> None:
        """Expensive coverage inspection and checkpoint creation run off the UI thread."""
        try:
            job_id, missing, progress = self.coordinator.plan_collection(
                symbol=symbol,
                start_utc=start_utc,
                end_utc=end_utc,
                timeframe_seconds=timeframe_seconds,
                account_id=account_id,
                account_label=account_id,
            )
            self._dispatch_to_ui(
                lambda: self._on_planning_completed(
                    job_id,
                    missing,
                    progress,
                    generation,
                    (symbol, start_utc, end_utc, timeframe_seconds),
                )
            )
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)
            logger.error("Planning worker error: %s", err_msg)
            self._dispatch_to_ui(lambda: self._on_planning_failed(generation, err_msg))

    def _current_plan_inputs(self) -> tuple[str, str, str, int] | None:
        """Snapshot visible inputs so a plan completed after an edit is discarded."""
        try:
            return (
                self.combo_symbol.get().strip().upper(),
                parse_explicit_utc_datetime(self.entry_start_utc.get().strip()).isoformat(),
                parse_explicit_utc_datetime(self.entry_end_utc.get().strip()).isoformat(),
                self._get_timeframe_seconds(),
            )
        except Exception:  # noqa: BLE001
            return None

    def _on_planning_completed(
        self,
        job_id: str,
        missing: list[Any],
        progress: CollectionProgress,
        generation: int,
        request: tuple[str, str, str, int],
    ) -> None:
        if generation != self._operation_generation or self._plan_request != request:
            logger.debug("Discarding stale planning result for job %s", job_id)
            return
        if self.current_job_id != self._plan_base_job_id:
            self._active_operation = None
            self._plan_request = None
            self._plan_base_job_id = None
            self._pending_start_after_plan = False
            self._set_controls_state(None)
            self._log("Selected job changed during planning; planned job was discarded.")
            return
        if self._current_plan_inputs() != request:
            self._active_operation = None
            self._plan_request = None
            self._plan_base_job_id = None
            self._pending_start_after_plan = False
            self._set_controls_state(None)
            self._log("Planning inputs changed during analysis; planned job was discarded.")
            return

        self._plan_request = None
        self._plan_base_job_id = None
        self._active_operation = None
        self.current_job_id = job_id
        self._update_progress_display(progress)
        self._refresh_jobs()
        self._log(f"Planned job {job_id}: {len(missing)} missing intervals identified.")
        self._set_controls_state(None)

        # A background plan alone never opens the broker. Only an explicit Start
        # request that was pending before planning may continue, and that path
        # still requires the live-connection confirmation dialog.
        if self._pending_start_after_plan:
            self._pending_start_after_plan = False
            self._start_collection()

    def _on_planning_failed(self, generation: int, err_msg: str) -> None:
        if generation != self._operation_generation:
            logger.debug("Discarding stale planning failure from generation %s", generation)
            return
        self._active_operation = None
        self._plan_request = None
        self._plan_base_job_id = None
        self._pending_start_after_plan = False
        self._set_controls_state(None)
        self.lbl_status_badge.config(text="PLANNING FAILED", foreground="#a82315")
        self._log(f"Planning failed: {err_msg}")
        messagebox.showerror("Planning Error", f"Failed to plan collection:\n{err_msg}")

    def _update_progress_display(self, p: CollectionProgress) -> None:
        self.lbl_job_id.config(text=p.job_id)
        self.lbl_status_badge.config(text=p.status.value)
        if p.status == CollectionStatus.COMPLETED_VERIFIED:
            self.lbl_status_badge.config(foreground="#1e6b20")
        elif p.status in (
            CollectionStatus.FAILED,
            CollectionStatus.SOURCE_UNAVAILABLE,
        ):
            self.lbl_status_badge.config(foreground="#a82315")
        elif p.status in (
            CollectionStatus.CANCELLED,
            CollectionStatus.PARTIALLY_COLLECTED,
        ):
            self.lbl_status_badge.config(foreground="#b35900")
        else:
            self.lbl_status_badge.config(foreground="#17324d")

        self.lbl_range_req.config(text=f"{p.requested_from_utc[:10]} .. {p.requested_to_utc[:10]}")
        self.lbl_cursor.config(
            text=p.last_collected_utc[:19] if p.last_collected_utc else "Pending"
        )
        self.lbl_candles.config(text=f"{p.candles_collected_this_session:,}")
        self.lbl_duplicates.config(text=f"{p.duplicates_detected:,}")
        self.lbl_conflicts.config(text=f"{p.conflicts_quarantined:,}")
        self.lbl_intervals.config(
            text=f"{p.total_intervals_count - p.current_interval_index} / {p.total_intervals_count}"
        )
        self.lbl_receipts.config(text=f"{p.receipt_count:,}")
        self.progress_bar["value"] = p.percent_complete

        if p.failure_reason:
            self._log(f"Status Note: {p.failure_reason}")

    def _set_controls_state(self, operation: str | None) -> None:
        """Update operator control states according to the active operation."""
        self._active_operation = operation
        if operation == "planning":
            self.btn_plan.config(state="disabled")
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="disabled", text="Stop / Pause")
            self.btn_resume.config(state="disabled")
            self.btn_verify.config(state="disabled")
            self.combo_jobs.config(state="disabled")
            self.btn_refresh_jobs.config(state="disabled")
            self.combo_symbol.config(state="disabled")
            self.combo_timeframe.config(state="disabled")
            if hasattr(self, "btn_export_report"):
                self.btn_export_report.config(state="disabled")
        elif operation == "verification":
            self.btn_plan.config(state="disabled")
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal", text="Stop Verification")
            self.btn_resume.config(state="disabled")
            self.btn_verify.config(state="disabled")
            self.combo_jobs.config(state="disabled")
            self.btn_refresh_jobs.config(state="disabled")
            self.combo_symbol.config(state="disabled")
            self.combo_timeframe.config(state="disabled")
            if hasattr(self, "btn_export_report"):
                self.btn_export_report.config(state="disabled")
        elif operation == "collection":
            self.btn_plan.config(state="disabled")
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal", text="Stop / Pause")
            self.btn_resume.config(state="disabled")
            self.btn_verify.config(state="disabled")
            self.combo_jobs.config(state="disabled")
            self.btn_refresh_jobs.config(state="disabled")
            self.combo_symbol.config(state="disabled")
            self.combo_timeframe.config(state="disabled")
        elif operation == "snapshot":
            self.btn_plan.config(state="disabled")
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="disabled", text="Stop / Pause")
            self.btn_resume.config(state="disabled")
            self.btn_verify.config(state="disabled")
            self.combo_jobs.config(state="disabled")
            self.btn_refresh_jobs.config(state="disabled")
            self.combo_symbol.config(state="disabled")
            self.combo_timeframe.config(state="disabled")
        else:
            self.btn_stop.config(state="disabled", text="Stop / Pause")
            self.btn_plan.config(state="normal")
            self.btn_verify.config(state="normal")
            self.combo_jobs.config(state="readonly")
            self.btn_refresh_jobs.config(state="normal")
            self.combo_symbol.config(state="readonly")
            self.combo_timeframe.config(state="readonly")
            if hasattr(self, "btn_export_report") and self._last_verification_report:
                self.btn_export_report.config(state="normal")

            if self.current_job_id:
                try:
                    progress = self.coordinator.get_progress(self.current_job_id)
                    if progress.status in (
                        CollectionStatus.CANCELLED,
                        CollectionStatus.INTERRUPTED,
                        CollectionStatus.PARTIALLY_COLLECTED,
                        CollectionStatus.PAUSED,
                        CollectionStatus.FAILED,
                    ):
                        self.btn_resume.config(state="normal")
                        self.btn_start.config(state="disabled")
                    elif progress.status == CollectionStatus.COMPLETED_VERIFIED:
                        self.btn_resume.config(state="disabled")
                        self.btn_start.config(state="disabled")
                    else:
                        self.btn_start.config(state="normal")
                        self.btn_resume.config(state="disabled")
                except Exception:  # noqa: BLE001
                    self.btn_start.config(state="normal")
                    self.btn_resume.config(state="disabled")
            else:
                self.btn_start.config(state="normal")
                self.btn_resume.config(state="disabled")

    def _stop_clicked(self) -> None:
        if self._active_operation == "verification":
            self._stop_verification()
        else:
            self._stop_collection()

    def _stop_verification(self) -> None:
        self._verify_cancel_requested = True
        self._log("Cancellation requested by operator. Halting verification worker...")
        self.btn_stop.config(state="disabled")

    def _start_collection(self) -> None:
        if not self.current_job_id:
            # Planning is asynchronous. The completion handler will continue only
            # after the operator has explicitly confirmed the live read-only start.
            self._plan_collection(start_after=True)
            return

        if self._active_operation is not None:
            messagebox.showwarning(
                "Operation in Progress",
                f"An operation ({self._active_operation}) is currently in progress. Please wait.",
            )
            return

        block_reason = self._unverified_otc_block_reason(self.current_job_id)
        if block_reason:
            self._log(f"Start blocked: {block_reason}")
            messagebox.showwarning("Verified Collection Disabled", block_reason)
            return

        selected_account_label = self.combo_account.get()
        selected_account_id = self._accounts_map.get(selected_account_label)
        if not selected_account_id:
            messagebox.showerror(
                "Account Required",
                "Please select an authorized broker account from the vault before starting collection.",
            )
            return

        if not self._confirm_live_connection(
            self.current_job_id, selected_account_id, resume=False
        ):
            return

        job_id = self.current_job_id
        generation = self._next_operation_generation()
        self._cancel_requested = False
        self._collection_start_time = time.monotonic()
        self._set_controls_state("collection")
        self.lbl_status_badge.config(text="CONNECTING...", foreground="#17324d")
        self._log(
            f"Starting authorized collection session for job {job_id} "
            f"using account {selected_account_label} (ID: {selected_account_id})..."
        )

        self._worker_thread = threading.Thread(
            target=self._run_collection_thread,
            args=(job_id, False, selected_account_id, selected_account_id, None, generation),
            daemon=True,
        )
        self._worker_thread.start()

    def _stop_collection(self) -> None:
        self._cancel_requested = True
        if self.current_job_id:
            with contextlib.suppress(Exception):
                self.coordinator.cancel_collection(self.current_job_id)
        self._log("Cancellation requested by operator. Signaling background worker...")
        self.btn_stop.config(state="disabled")

    def _resume_collection(self) -> None:
        if not self.current_job_id:
            messagebox.showwarning(
                "Resume Error", "No collection job selected or loaded to resume."
            )
            return

        if self._active_operation is not None:
            messagebox.showwarning(
                "Operation in Progress",
                f"An operation ({self._active_operation}) is currently in progress. Please wait.",
            )
            return

        # Capture at request time, before confirmation and background broker
        # construction. Any later cancel has a distinct token and survives.
        stale_cancel_token = self.coordinator.capture_stale_cancel_token(self.current_job_id)

        block_reason = self._unverified_otc_block_reason(self.current_job_id)
        if block_reason:
            self._log(f"Resume blocked: {block_reason}")
            messagebox.showwarning("Verified Collection Disabled", block_reason)
            return

        selected_account_label = self.combo_account.get()
        selected_account_id = self._accounts_map.get(selected_account_label)
        if not selected_account_id:
            messagebox.showerror(
                "Account Required",
                "Please select an authorized broker account from the vault before resuming collection.",
            )
            return

        if not self._confirm_live_connection(self.current_job_id, selected_account_id, resume=True):
            return

        job_id = self.current_job_id
        generation = self._next_operation_generation()
        self._cancel_requested = False
        self._collection_start_time = time.monotonic()
        self._set_controls_state("collection")
        self.lbl_status_badge.config(text="RESUMING...", foreground="#b35900")
        self._log(
            f"Resuming collection for job {job_id} "
            f"using account {selected_account_label} (ID: {selected_account_id})..."
        )

        self._worker_thread = threading.Thread(
            target=self._run_collection_thread,
            args=(
                job_id,
                True,
                selected_account_id,
                selected_account_id,
                stale_cancel_token,
                generation,
            ),
            daemon=True,
        )
        self._worker_thread.start()

    def _run_collection_thread(
        self,
        job_id: str,
        is_resume: bool,
        account_id: str | None = None,
        account_label: str | None = None,
        stale_cancel_token: str | None = None,
        generation: int = 0,
    ) -> None:
        """Background worker thread for asynchronous collection."""
        import asyncio

        def progress_cb(prog: CollectionProgress) -> None:
            self._dispatch_to_ui(lambda: self._on_collection_progress(prog, generation, job_id))

        async def _exec() -> CollectionProgress:
            from iqoption_research.cli.commands.backfill_cmd import _connect_live_client

            client, cleanup = await _connect_live_client(
                saved_credentials=True, account_id=account_id
            )
            try:
                if is_resume:
                    return await self.coordinator.resume_collection(
                        job_id=job_id,
                        client=client,
                        operator_authorized=True,
                        cancel_token=lambda: self._cancel_requested,
                        progress_cb=progress_cb,
                        account_id=account_id,
                        account_label=account_label,
                        stale_cancel_token=stale_cancel_token,
                        capture_stale_cancel_token=False,
                    )
                return await self.coordinator.execute_collection(
                    job_id=job_id,
                    client=client,
                    cancel_token=lambda: self._cancel_requested,
                    progress_cb=progress_cb,
                    operator_authorized=True,
                    account_id=account_id,
                    account_label=account_label,
                )
            finally:
                await cleanup()

        try:
            final_progress = asyncio.run(_exec())
            self._dispatch_to_ui(lambda: self._on_collection_completed(final_progress, generation))
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)
            logger.error("Collection thread error: %s", err_msg)
            self._dispatch_to_ui(lambda: self._on_collection_failed(err_msg, generation, job_id))

    def _on_collection_progress(
        self, prog: CollectionProgress, generation: int, job_id: str
    ) -> None:
        if self._is_current_worker_result(generation, job_id):
            self._update_progress_display(prog)

    def _on_collection_completed(self, prog: CollectionProgress, generation: int) -> None:
        if not self._is_current_worker_result(generation, prog.job_id):
            return
        self._active_operation = None
        self._update_progress_display(prog)
        self._set_controls_state(None)

        if prog.status in (
            CollectionStatus.CANCELLED,
            CollectionStatus.PARTIALLY_COLLECTED,
            CollectionStatus.INTERRUPTED,
        ):
            self._log(f"Collection stopped with status: {prog.status.value}. Resume available.")
        elif prog.status == CollectionStatus.COMPLETED_VERIFIED:
            self._log(
                "Collection successfully finished and verified complete! Ready for snapshot registration."
            )
            messagebox.showinfo(
                "Collection Complete",
                f"Job {prog.job_id} successfully completed with full verified coverage!",
            )

    def _on_collection_failed(self, err_msg: str, generation: int, job_id: str) -> None:
        if not self._is_current_worker_result(generation, job_id):
            return
        self._active_operation = None
        self.lbl_status_badge.config(text="ERROR / FAILED", foreground="#a82315")
        self._set_controls_state(None)
        self._log(f"Collection halted due to error: {err_msg}")
        messagebox.showerror("Collection Stopped", f"Collection stopped with error:\n{err_msg}")

    def _verify_and_snapshot(self) -> None:
        if not self.current_job_id:
            messagebox.showwarning("Verify Error", "No collection job selected to verify.")
            return

        if self._active_operation is not None:
            messagebox.showwarning(
                "Operation in Progress",
                f"An operation ({self._active_operation}) is currently in progress. Please wait.",
            )
            return

        block_reason = self._unverified_otc_block_reason(self.current_job_id)
        if block_reason:
            self._log(f"Verification/snapshot blocked: {block_reason}")
            messagebox.showwarning("Verified Snapshot Disabled", block_reason)
            return

        job_id = self.current_job_id
        generation = self._next_operation_generation()
        self._verify_cancel_requested = False
        self._op_start_time = time.monotonic()
        self._set_controls_state("verification")
        self.lbl_status_badge.config(text="VERIFYING...", foreground="#17324d")
        self.progress_bar["value"] = 0.0
        self._log(f"Starting background integrity verification for job {job_id}...")

        self._worker_thread = threading.Thread(
            target=self._run_verification_thread,
            args=(job_id, generation),
            daemon=True,
        )
        self._worker_thread.start()

    def _run_verification_thread(self, job_id: str, generation: int) -> None:
        from iqoption_research.market_data.collection_engine import VerificationProgress

        last_ui_vp_time = 0.0

        def progress_cb(vp: VerificationProgress) -> None:
            nonlocal last_ui_vp_time
            now = time.monotonic()
            if (
                now - last_ui_vp_time >= 0.05
                or abs(vp.percent_complete - 100.0) < 1e-6
                or vp.receipts_checked == 0
            ):
                last_ui_vp_time = now
                self._dispatch_to_ui(lambda: self._on_verification_progress(vp, generation, job_id))

        try:
            report = self.coordinator.verify_job_integrity(
                job_id,
                cancel_token=lambda: self._verify_cancel_requested,
                progress_cb=progress_cb,
            )
            self._dispatch_to_ui(
                lambda: self._on_verification_completed(job_id, report, generation)
            )
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)
            logger.error("Verification worker error: %s", err_msg)
            self._dispatch_to_ui(lambda: self._on_verification_failed(job_id, err_msg, generation))

    def _on_verification_progress(
        self, vp: Any, generation: int = 0, job_id: str | None = None
    ) -> None:
        if not self._is_current_worker_result(generation, job_id):
            return
        self.progress_bar["value"] = vp.percent_complete
        self.lbl_receipts.config(text=f"{vp.receipts_checked:,} / {vp.total_receipts:,}")
        self.lbl_candles.config(text=f"{vp.candles_verified:,}")
        self.lbl_status_badge.config(text=f"VERIFYING ({vp.percent_complete:.0f}%)")
        self.lbl_cursor.config(text=f"Files: {vp.files_checked} | {vp.current_operation}")
        elapsed_sec = int(vp.elapsed_seconds)
        hours = elapsed_sec // 3600
        minutes = (elapsed_sec % 3600) // 60
        seconds = elapsed_sec % 60
        self.lbl_elapsed.config(text=f"Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def _log_verification_summary(self, job_id: str, report: dict[str, Any]) -> None:
        """Display concise summaries instead of dumping thousands of timestamps."""
        verified = report.get("verified", False)
        status_label = "VERIFIED COMPLETE" if verified else f"INCOMPLETE ({report.get('status')})"
        symbol = report.get("symbol", "--")
        candles = report.get("unique_verified_candles", 0)
        receipt_sum = report.get("receipt_candles_sum", 0)
        rcpts = report.get("receipt_count", 0)
        legacy = report.get("legacy_receipt_count", 0)
        completed_ivs = report.get("completed_intervals", 0)
        total_ivs = report.get("total_intervals", 0)
        gaps = report.get("coverage_gaps_count", 0)
        errors = report.get("errors", [])
        prov = ", ".join(report.get("provenance_status", [])) or "None"

        self._log(
            f"--- Verification Summary for {job_id} ---\n"
            f"Result: {status_label} | Symbol: {symbol}\n"
            f"Candles: {candles:,} unique verified (receipt sum: {receipt_sum:,})\n"
            f"Receipts: {rcpts} valid ({legacy} legacy) | Intervals: {completed_ivs}/{total_ivs} completed\n"
            f"Coverage Gaps: {gaps} | Provenance: {prov}\n"
            f"Errors: {len(errors)}" + (f" ({'; '.join(errors[:3])})" if errors else "")
        )

    def _on_verification_completed(
        self, job_id: str, report: dict[str, Any], generation: int = 0
    ) -> None:
        if not self._is_current_worker_result(generation, job_id):
            return
        self._last_verification_report = report
        if hasattr(self, "btn_export_report"):
            self.btn_export_report.config(state="normal")

        self._log_verification_summary(job_id, report)

        status = report.get("status")
        if status == "CANCELLED":
            self._active_operation = None
            self._set_controls_state(None)
            self.lbl_status_badge.config(text="CANCELLED", foreground="#b35900")
            self._log(f"Verification of job {job_id} was cancelled by operator.")
            return

        if not report.get("verified"):
            self._active_operation = None
            self._set_controls_state(None)
            self.lbl_status_badge.config(text=status or "UNVERIFIED", foreground="#a82315")
            msg = (
                f"Job {job_id} could not be verified complete.\n"
                f"Status: {status}\n"
                f"Completed Intervals: {report.get('completed_intervals')} / {report.get('total_intervals')}\n"
                f"Errors: {report.get('errors')}"
            )
            messagebox.showwarning("Verification Incomplete", msg)
            return

        self.lbl_status_badge.config(text="COMPLETED_VERIFIED", foreground="#1e6b20")
        self.progress_bar["value"] = 100.0

        # Verified complete! Offer automatic snapshot registration
        if messagebox.askyesno(
            "Integrity Verified",
            f"Job {job_id} is verified complete!\n\n"
            f"Unique Verified Candles: {report.get('unique_verified_candles', 0):,}\n"
            f"Receipt Candles: {report.get('receipt_candles_sum', 0):,}\n"
            f"Receipts: {report.get('receipt_count', 0):,}\n\n"
            "Would you like to register an immutable research dataset snapshot now?",
        ):
            self._start_snapshot_worker(job_id, report, generation)
        else:
            self._active_operation = None
            self._set_controls_state(None)

    def _on_verification_failed(self, job_id: str, err_msg: str, generation: int = 0) -> None:
        if not self._is_current_worker_result(generation, job_id):
            return
        self._active_operation = None
        self._set_controls_state(None)
        self.lbl_status_badge.config(text="VERIFICATION FAILED", foreground="#a82315")
        self._log(f"Verification failed for job {job_id}: {err_msg}")
        messagebox.showerror("Verification Error", f"Verification failed:\n{err_msg}")

    def _start_snapshot_worker(
        self, job_id: str, report: dict[str, Any], generation: int = 0
    ) -> None:
        self._active_operation = "snapshot"
        self._set_controls_state("snapshot")
        self.lbl_status_badge.config(text="CREATING SNAPSHOT...", foreground="#17324d")
        self._log(f"Registering immutable research snapshot for job {job_id} in background...")

        self._worker_thread = threading.Thread(
            target=self._run_snapshot_thread,
            args=(job_id, report, generation),
            daemon=True,
        )
        self._worker_thread.start()

    def _run_snapshot_thread(
        self, job_id: str, report: dict[str, Any], generation: int = 0
    ) -> None:
        try:
            snap = self.coordinator.create_snapshot_for_job(job_id, integrity=report)
            self._dispatch_to_ui(lambda: self._on_snapshot_completed(job_id, snap, generation))
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)
            logger.error("Snapshot worker error: %s", err_msg)
            self._dispatch_to_ui(lambda: self._on_snapshot_failed(job_id, err_msg, generation))

    def _on_snapshot_completed(self, job_id: str, snap: Any, generation: int = 0) -> None:
        if not self._is_current_worker_result(generation, job_id):
            return
        self._active_operation = None
        self._set_controls_state(None)
        self.lbl_status_badge.config(text="SNAPSHOT REGISTERED", foreground="#1e6b20")
        self._log(f"Created immutable research snapshot: {snap.manifest.snapshot_id}")
        messagebox.showinfo(
            "Snapshot Registered",
            f"Snapshot successfully created:\n{snap.manifest.snapshot_id}\n"
            f"Candles: {snap.manifest.total_candles:,}",
        )

    def _on_snapshot_failed(self, job_id: str, err_msg: str, generation: int = 0) -> None:
        if not self._is_current_worker_result(generation, job_id):
            return
        self._active_operation = None
        self._set_controls_state(None)
        self.lbl_status_badge.config(text="SNAPSHOT FAILED", foreground="#a82315")
        self._log(f"Snapshot registration failed: {err_msg}")
        messagebox.showerror("Snapshot Error", f"Failed registering snapshot:\n{err_msg}")

    def _export_report(self) -> None:
        if not self._last_verification_report:
            messagebox.showinfo("Export Report", "No verification report available to export.")
            return

        job_id = self._last_verification_report.get("job_id", "job")
        default_name = f"integrity_report_{job_id}.json"
        path = filedialog.asksaveasfilename(
            parent=self.dashboard.window,
            title="Export Verification Integrity Report",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if path:
            try:
                Path(path).write_text(
                    json.dumps(self._last_verification_report, indent=2, default=str),
                    encoding="utf-8",
                )
                self._log(f"Exported full verification report to {path}")
                messagebox.showinfo("Export Complete", f"Report saved successfully to:\n{path}")
            except Exception as exc:  # noqa: BLE001
                self._log(f"Export failed: {exc}")
                messagebox.showerror("Export Failed", f"Could not save report:\n{exc}")
