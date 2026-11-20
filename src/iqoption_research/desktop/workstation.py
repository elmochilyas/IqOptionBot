"""Accounts, bounded collector and offline library panels for the existing dashboard."""

from __future__ import annotations

import json
import tkinter as tk
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any

from iqoption_research.auth.vault import WindowsCredentialVault
from iqoption_research.market_data.extended import DURATIONS
from iqoption_research.market_data.extended_store import library

if TYPE_CHECKING:
    from iqoption_research.desktop.app import Dashboard


@dataclass
class AccountSelection:
    accounts: dict[str, str] = field(default_factory=dict)
    selected: str | None = None
    remember: bool = False

    def select(self, account_id: str | None, *, busy: bool) -> None:
        if busy:
            raise ValueError("Account changes are prohibited while a session is active")
        if account_id is not None and account_id not in self.accounts:
            raise ValueError("Selected account is unavailable")
        self.selected = account_id

    def finished(self) -> None:
        if not self.remember:
            self.selected = None


class WorkstationPanels:
    def __init__(self, dashboard: Dashboard, notebook: ttk.Notebook) -> None:
        self.dashboard = dashboard
        self.selection = AccountSelection()
        self.future: Future[dict[str, Any]] | None = None
        self.loaded = False
        self.rows: list[dict[str, Any]] = []
        self.message = "Loading secure account metadata; no broker connection…"
        self.account_frame = ttk.Frame(notebook, padding=16)
        notebook.add(self.account_frame, text="Accounts")
        self.account_status = ttk.Label(self.account_frame, text=self.message, wraplength=1000)
        self.account_status.pack(anchor="w", pady=6)
        self.account_list = ttk.Combobox(self.account_frame, state="readonly", width=68)
        self.account_list.pack(anchor="w", pady=8)
        self.account_list.bind("<<ComboboxSelected>>", self.select)
        buttons = ttk.Frame(self.account_frame)
        buttons.pack(anchor="w", pady=8)
        self.controls: list[ttk.Button] = []
        for label, callback in (
            ("Add account…", lambda: self.entry(False)),
            ("Update credentials…", lambda: self.entry(True)),
            ("Forget selected…", self.forget),
            ("Refresh accounts / library", self.refresh),
        ):
            button = ttk.Button(buttons, text=label, command=callback)
            button.pack(side="left", padx=(0, 8))
            self.controls.append(button)
        self.remember = tk.BooleanVar(value=False)
        self.remember_button = ttk.Checkbutton(
            self.account_frame,
            text="Remember selected account on this Windows installation",
            variable=self.remember,
            command=self.set_remember,
        )
        self.remember_button.pack(anchor="w", pady=12)
        ttk.Label(
            self.account_frame,
            wraplength=1000,
            text=(
                "Account selection never switches the broker wallet.\n"
                "Add / Update opens your own interactive terminal. Only that terminal handles the "
                "password and asks for storage consent. Remember Me retains an identity, never a "
                "broker connection. Without it, select an account for each session."
            ),
        ).pack(anchor="w", pady=12)
        self.account_connection = ttk.Label(
            self.account_frame, text="DISCONNECTED · Wallet UNKNOWN"
        )
        self.account_connection.pack(anchor="w", pady=8)

        collector = ttk.Frame(notebook, padding=16)
        notebook.add(collector, text="Market Collector")
        ttk.Label(
            collector,
            text="Extended Collection · regular EURUSD · M1",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            collector,
            wraplength=1000,
            text=(
                "Collection data supports future research. It creates no model predictions and "
                "never increases the Phase 38 pilot count. No automatic restart or relogin."
            ),
        ).pack(anchor="w", pady=8)
        bar = ttk.Frame(collector)
        bar.pack(anchor="w", pady=8)
        ttk.Label(bar, text="Duration (minutes)").pack(side="left", padx=(0, 8))
        self.duration = ttk.Combobox(
            bar, values=[str(x) for x in DURATIONS], state="readonly", width=8
        )
        self.duration.set("60")
        self.duration.pack(side="left")
        self.start_button = ttk.Button(bar, text="Start Collection", command=self.start_collection)
        self.start_button.pack(side="left", padx=12)
        self.stop_button = ttk.Button(bar, text="Stop Collection", command=dashboard.stop)
        self.stop_button.pack(side="left")
        self.collector_status = ttk.Label(collector, wraplength=1050, justify="left")
        self.collector_status.pack(anchor="w", pady=16)
        self.library_text = dashboard.text_tab(notebook, "Data Library")
        self.refresh()

    def busy(self) -> bool:
        return (
            self.dashboard.process.child is not None
            or self.dashboard.launch_requested
            or self.future is not None
            or self.dashboard.closing
        )

    def _read(self, action: str = "read", selected: str | None = None) -> dict[str, Any]:
        try:
            store = WindowsCredentialVault().account_store()
            if action == "forget":
                if selected is None:
                    raise ValueError("Select an account first")
                store.forget(selected)
            elif action == "remember":
                store.remember(selected)
            accounts = {a.account_id: a.label for a in store.list_accounts()}
            remembered = store.remembered_account()
            message = "Native Windows vault verified · passwords stay in the vault"
        except Exception as exc:  # noqa: BLE001 - no backend payloads in UI or logs
            accounts, remembered = {}, None
            message = f"Vault unavailable or operation failed: {type(exc).__name__}"
        return {
            "accounts": accounts,
            "remembered": remembered,
            "message": message,
            "library": library(self.dashboard.root),
        }

    def refresh(self) -> None:
        if self.future is None:
            self.future = self.dashboard.executor.submit(self._read)

    def select(self, _event: object = None) -> None:
        try:
            ids = list(self.selection.accounts)
            index = self.account_list.current()
            self.selection.select(ids[index] if index >= 0 else None, busy=self.busy())
            if self.selection.remember:
                self.future = self.dashboard.executor.submit(
                    self._read, "remember", self.selection.selected
                )
            if self.dashboard.snapshot is not None:
                self.dashboard.render()
        except ValueError as exc:
            messagebox.showerror("Account selection", str(exc), parent=self.dashboard.window)

    def set_remember(self) -> None:
        if self.busy():
            self.remember.set(self.selection.remember)
            return
        if self.remember.get() and self.selection.selected is None:
            self.remember.set(False)
            return
        self.selection.remember = self.remember.get()
        self.future = self.dashboard.executor.submit(
            self._read, "remember", self.selection.selected if self.selection.remember else None
        )

    def forget(self) -> None:
        if self.busy() or self.selection.selected is None:
            return
        if messagebox.askyesno(
            "Forget account",
            "Delete the selected credentials from Windows Vault?",
            parent=self.dashboard.window,
        ):
            self.future = self.dashboard.executor.submit(
                self._read, "forget", self.selection.selected
            )

    def entry(self, update: bool) -> None:
        if self.busy() or (update and self.selection.selected is None):
            return
        try:
            self.dashboard.process.account_entry(self.selection.selected if update else None)
            self.dashboard.state.configure(text="Enter credentials only in the operator terminal.")
        except Exception as exc:  # noqa: BLE001
            self.message = f"Account terminal failed: {type(exc).__name__}"

    def start_collection(self) -> None:
        if self.busy() or self.selection.selected is None:
            return
        try:
            self.dashboard.process.start_collection(
                self.selection.selected, int(self.duration.get())
            )
            warning = getattr(self.dashboard.process, "_last_duration_warning", "")
            if warning:
                self.dashboard.state.configure(text=f"Extended collection authorized.\n{warning}")
            else:
                self.dashboard.state.configure(
                    text="Extended collection authorized; supervise the console."
                )
        except Exception as exc:  # noqa: BLE001 - no credential payloads shown
            self.message = f"Collection launch refused: {type(exc).__name__}"
            self.dashboard.state.configure(text=self.message)

    def finished(self) -> None:
        self.selection.finished()
        if self.selection.selected is None:
            self.account_list.set("")
        self.refresh()

    def tick(self) -> None:
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                self.selection.accounts = result["accounts"]
                self.message, self.rows = result["message"], result["library"]
                if not self.loaded:
                    self.selection.selected = result["remembered"]
                self.selection.remember = result["remembered"] is not None
                self.remember.set(self.selection.remember)
                if self.selection.selected not in self.selection.accounts:
                    self.selection.selected = None
                self.account_list["values"] = [
                    f"{label}  [{aid[:8]}]" for aid, label in self.selection.accounts.items()
                ]
                if self.selection.selected is not None:
                    self.account_list.current(
                        list(self.selection.accounts).index(self.selection.selected)
                    )
                else:
                    self.account_list.set("")
                self.loaded = True
                self.dashboard.write(
                    self.library_text,
                    "COLLECTION DATA — not predictions\n\n"
                    + (
                        "\n\n".join(self.library_summary(row) for row in self.rows)
                        if self.rows
                        else "No collection sessions yet. No broker connection was opened."
                    ),
                )
                if self.dashboard.snapshot is not None:
                    self.dashboard.render()
            except Exception as exc:  # noqa: BLE001
                self.message = f"Offline library failed: {type(exc).__name__}"
            self.future = None
        busy = self.busy()
        for control in [*self.controls, self.remember_button]:
            control.state(["disabled" if busy else "!disabled"])
        self.account_list.configure(state="disabled" if busy else "readonly")
        self.duration.configure(state="disabled" if busy else "readonly")
        self.start_button.state([
            "!disabled" if not busy and self.selection.selected else "disabled"
        ])
        self.stop_button.state([
            "!disabled"
            if self.dashboard.process.child is not None
            and self.dashboard.process.kind == "collector"
            else "disabled"
        ])
        live = self.dashboard.process.live_state()
        is_active = self.dashboard.process.child is not None

        # Determine what to display: active live state vs. historical last-session state.
        # Never blend elapsed/remaining from a completed session with a "live" label.
        if not is_active and not self.dashboard.process.last_state and self.rows:
            # No process running and no cached last_state: show most-recent library row
            # clearly labelled as historical, not as live metrics.
            historical = {**self.rows[0], "connection": "DISCONNECTED"}
            display_state = historical
            session_mode = "historical (last saved)"
        elif not is_active and self.dashboard.process.last_state:
            display_state = self.dashboard.process.last_state
            session_mode = "post-session (final)"
        else:
            display_state = live
            session_mode = "active" if is_active else "idle"

        connection = display_state.get("connection", "DISCONNECTED")
        wallet = display_state.get("wallet_classification", "UNKNOWN")
        selected = self.selection.accounts.get(self.selection.selected or "", "None selected")
        mode = self.dashboard.process.kind if self.dashboard.process.child else "none"
        default_dest = self.dashboard.root / "data/extended_collection"
        destination = display_state.get("destination", default_dest)
        self.account_status.configure(text=self.message + f"\nSelected: {selected}")
        self.account_connection.configure(text=f"{connection} · Wallet: {wallet}")
        total = sum(row["valid_candles"] for row in self.rows if row.get("integrity") != "FAILED")
        self.dashboard.system_status.configure(
            text=(
                f"Account: {selected} · {connection} · "
                f"Collection library: {total} candles (separate from Phase 38 predictions)"
            )
        )

        # Format the last verified candle as human-readable UTC ISO string.
        last_candle_epoch = display_state.get("last_verified_candle")
        if last_candle_epoch is not None:
            try:
                last_candle_str = datetime.fromtimestamp(last_candle_epoch, UTC).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
            except (OSError, OverflowError, ValueError):
                last_candle_str = str(last_candle_epoch)
        else:
            last_candle_str = "none"

        # Distinguish active elapsed/remaining from historical elapsed (which is final/frozen).
        termination = display_state.get("termination_reason", "IN_PROGRESS")
        if is_active:
            elapsed_label = f"Elapsed: {display_state.get('elapsed_seconds', 0):.0f}s"
            remaining_label = f"Remaining: {display_state.get('remaining_seconds', 0):.0f}s"
            ended_label = ""
        else:
            actual_elapsed = display_state.get(
                "actual_elapsed_seconds",
                display_state.get("elapsed_seconds", 0),
            )
            requested = display_state.get(
                "requested_minutes", display_state.get("authorized_minutes", "?")
            )
            ended_at = display_state.get("ended_at_utc", "")
            elapsed_label = f"Actual elapsed: {actual_elapsed:.0f}s (requested: {requested} min)"
            remaining_label = ""
            ended_label = f"Ended: {ended_at}\n" if ended_at else ""

        coverage_note = display_state.get("coverage_note", "")
        self.collector_status.configure(
            text=(
                f"Selected account: {selected}\n"
                f"Connection: {connection} · Wallet: {wallet}\n"
                f"Active mode: {mode} · Session display: {session_mode}\n"
                f"Termination: {termination}\n"
                + ended_label
                + f"{elapsed_label}"
                + (f"  /  {remaining_label}" if remaining_label else "")
                + f"\nValid candles: {display_state.get('valid_candles', 0)}   "
                f"Rejected: {display_state.get('rejected_candles', 0)}"
                f"   Duplicates: {display_state.get('duplicate_boundaries', 0)}\n"
                f"Missing boundaries: {display_state.get('missing_boundaries', [])}\n"
                f"Coverage: {display_state.get('coverage', 0):.1%}   "
                f"Connection uptime: {display_state.get('connection_uptime_seconds', 0):.0f}s\n"
                f"Last verified candle: {last_candle_str}\n"
                + (f"Coverage note: {coverage_note}\n" if coverage_note else "")
                + f"Storage: {destination}\n"
                "Source: IQ Option broker quotes · receipt verification in Data Library\n"
                f"Last completed operation: {self.dashboard.last_session or 'none'}"
            )
        )

    def overview(self) -> str:
        count = sum(row["valid_candles"] for row in self.rows if row.get("integrity") != "FAILED")
        label = self.selection.accounts.get(self.selection.selected or "", "none")
        return (
            f"SELECTED ACCOUNT       {label}\n"
            f"COLLECTION CANDLES     {count} (separate from prospective predictions)\n"
            f"COLLECTOR STATE        {self.dashboard.process.live_state().get('connection')}\n\n"
        )

    @staticmethod
    def library_summary(row: dict[str, Any]) -> str:
        boundaries = row.get("boundaries", [])
        if boundaries:
            try:
                first_utc = datetime.fromtimestamp(boundaries[0], UTC).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
                last_utc = datetime.fromtimestamp(boundaries[-1], UTC).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
                date_range = f"{first_utc}  to  {last_utc}"
            except (OSError, OverflowError, ValueError):
                date_range = f"epochs {boundaries[0]} to {boundaries[-1]}"
        else:
            date_range = "No committed candles"
        termination = row.get("termination_reason", "UNKNOWN")
        coverage_note = row.get("coverage_note", "")
        actual_secs = row.get("actual_elapsed_seconds", row.get("elapsed_seconds", "unknown"))
        req_mins = row.get("requested_minutes", row.get("authorized_minutes", "?"))
        return (
            f"{row['session_id']}  |  {row['integrity']}\n"
            f"  UTC range: {date_range}\n"
            f"  Valid candles: {row['valid_candles']} · Source: {row.get('source', 'UNVERIFIED')}\n"
            f"  Missing: {row.get('missing_boundaries', [])}\n"
            f"  Rejected: {row.get('rejected_candles', 'unknown')} · "
            f"Duplicates: {row.get('duplicate_boundaries', 'unknown')}\n"
            f"  Receipt/file groups verified: {len(row.get('receipts', {}))}\n"
            f"  Uncommitted files: {json.dumps(row.get('uncommitted_files', []))}\n"
            f"  Termination: {termination}\n"
            + (f"  Coverage note: {coverage_note}\n" if coverage_note else "")
            + f"  Wallet: {row.get('wallet_classification', 'UNKNOWN')}\n"
            f"  Ended: {row.get('ended_at_utc', 'unknown')}\n"
            f"  Actual elapsed: {actual_secs}s  (requested: {req_mins} min)\n"
            f"  Storage: {row['destination']}"
        )
