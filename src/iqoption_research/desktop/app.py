"""Native Tk/ttk research dashboard. All automatic work is local and offline."""

from __future__ import annotations

import json
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any
from zoneinfo import ZoneInfo

from iqoption_research.desktop.core import STUDY, require_slot, slot_used, slots
from iqoption_research.desktop.process import SessionProcess, refresh
from iqoption_research.desktop.windows import DashboardMutex
from iqoption_research.desktop.workstation import WorkstationPanels
from iqoption_research.research.prospective_pilot import (
    PilotStudySpecification,
    ProspectivePilotError,
)


class Dashboard:
    def __init__(self, window: tk.Tk, root: Path) -> None:
        self.window, self.root = window, root
        self.process = SessionProcess(root)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="offline-audit")
        self.pending: Future[dict[str, Any]] | None = None
        self.snapshot: dict[str, Any] | None = None
        self.launch_requested = False
        self.closing = False
        self.last_session = ""
        self.schedule_text = ""
        self.local_zone = ZoneInfo("Africa/Casablanca")
        window.title("IQOPTION · Research Workstation V2")
        window.geometry(f"1180x{min(740, window.winfo_screenheight() - 100)}")
        window.minsize(960, 620)
        style = ttk.Style(window)
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f6")
        style.configure("TLabel", background="#eef2f6", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), foreground="#17324d")
        style.configure("Safety.TLabel", foreground="#9c3414", font=("Segoe UI", 11, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        body = ttk.Frame(window, padding=20)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Research Workstation V2", style="Title.TLabel").pack(anchor="w")
        ttk.Label(body, text="EURUSD / M1 · Frozen Random Forest · Threshold 0.60").pack(anchor="w")
        ttk.Label(
            body,
            text="NO_DEMONSTRATED_EDGE    |    ORDERS DISABLED    |    Authority NONE",
            style="Safety.TLabel",
        ).pack(anchor="w", pady=(10, 12))
        self.clock = ttk.Label(body)
        self.clock.pack(anchor="w")
        notebook = ttk.Notebook(body)
        phase_body = ttk.Frame(notebook, padding=12)
        self.identity = ttk.Label(phase_body, text="Verifying immutable study and both anchors…")
        self.identity.pack(anchor="w", pady=4)
        self.next_slot = ttk.Label(phase_body)
        self.next_slot.pack(anchor="w", pady=4)
        self.market = ttk.Label(
            phase_body,
            text=(
                "Broker connection: NOT PROBED offline. Regular EURUSD only; no OTC substitution. "
                "Calendar eligibility does not verify broker availability."
            ),
            wraplength=1100,
        )
        self.market.pack(anchor="w", pady=4)
        self.wallet = ttk.Label(phase_body, text="Wallet: UNKNOWN (no live verification)")
        self.wallet.pack(anchor="w", pady=4)
        self.system_status = ttk.Label(body, text="Accounts and collector: loading offline state…")
        self.system_status.pack(anchor="w", pady=4)
        controls = ttk.Frame(phase_body)
        controls.pack(fill="x", pady=10)
        self.start_button = ttk.Button(controls, text="Start Observation", command=self.start)
        self.start_button.pack(side="left")
        self.start_button.state(["disabled"])
        self.stop_button = ttk.Button(controls, text="Stop / Cancel", command=self.stop)
        self.stop_button.pack(side="left", padx=8)
        self.stop_button.state(["disabled"])
        self.refresh_button = ttk.Button(controls, text="Refresh offline", command=self.refresh)
        self.refresh_button.pack(side="left")
        self.saved = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Use saved Windows Vault credentials for this session",
            variable=self.saved,
        ).pack(side="left", padx=16)
        self.state = ttk.Label(body, text="Loading offline audit…", wraplength=1100)
        self.state.pack(anchor="w", pady=(0, 8))
        self.progress = ttk.Progressbar(phase_body, maximum=100)
        self.progress.pack(fill="x", pady=(0, 10))
        ttk.Label(
            body,
            text=(
                "Phase 38: ≤5 observations / ≤300 seconds. Collection: ≤120 minutes. "
                "One explicit Start per session. No automatic starts or retries."
            ),
            wraplength=1100,
        ).pack(side="bottom", anchor="w", pady=(12, 0))
        notebook.pack(fill="both", expand=True)
        self.overview = self.text_tab(notebook, "Overview")
        self.workstation = WorkstationPanels(self, notebook)
        evaluation = ttk.Notebook(phase_body)
        evaluation.pack(fill="both", expand=True)
        notebook.add(phase_body, text="Model Evaluation")
        self.audit = self.text_tab(evaluation, "Sessions & wallet audit")
        self.recovery = self.text_tab(evaluation, "Missing labels / recovery preview")
        self.schedule = self.text_tab(evaluation, "Registered slots")
        self.details = self.text_tab(evaluation, "Verified offline report")
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()
        self.tick()

    def text_tab(self, notebook: ttk.Notebook, title: str) -> tk.Text:
        frame = ttk.Frame(notebook, padding=8)
        notebook.add(frame, text=title)
        text = tk.Text(
            frame,
            wrap="word",
            font=("Consolas", 11),
            relief="flat",
            background="#ffffff",
            foreground="#20364a",
            padx=14,
            pady=12,
        )
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        text.configure(state="disabled")
        return text

    @staticmethod
    def write(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", value)
        widget.yview_moveto(0)
        widget.configure(state="disabled")

    def refresh(self) -> None:
        if self.pending is None:
            self.pending = self.executor.submit(refresh, self.root)
            self.refresh_button.state(["disabled"])

    def start(self) -> None:
        if hasattr(self, "workstation") and self.workstation.busy():
            return
        if self.launch_requested or self.process.child is not None or self.closing:
            return
        self.launch_requested = True
        self.start_button.state(["disabled"])
        self.state.configure(text="Rechecking local evidence before dispatch…")
        self.refresh()

    def stop(self) -> None:
        self.launch_requested = False
        self.process.stop()
        self.state.configure(text="Cancelling; waiting for evidence preservation and process exit…")

    def close(self) -> None:
        self.closing = True
        self.stop()
        if self.process.child is None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.window.destroy()

    def render(self) -> None:
        if self.snapshot is None:
            return
        snapshot = self.snapshot
        status, study = snapshot["status"], snapshot["study"]
        classifications = sorted({str(w["classification"]) for w in snapshot["wallets"]})
        self.wallet.configure(
            text="Recorded wallet classification: "
            + (", ".join(classifications) if classifications else "UNKNOWN")
        )
        self.identity.configure(
            text=(
                f"Study {study['study_id']} · BOTH anchors verified\n"
                f"SHA-256 {study['study_specification_sha256']}"
            )
        )
        self.progress["value"] = status["total_unique_committed_predictions"]
        verified = sum(w["journal_verified_predictions"] for w in snapshot["wallets"])
        total = sum(s["total_registry_observations"] for s in status["session_audits"])
        unverified = total - verified
        metrics = status["probabilistic_metrics"]
        self.write(
            self.overview,
            self.workstation.overview()
            + "\n".join([
                "MODEL PREFLIGHT        "
                + ("READY" if snapshot["model_eligible"] else "FAILED / MISSING"),
                "CONNECTION             Not probed; connection occurs in the visible session",
                f"PILOT STATUS           {status['pilot_status']}",
                "",
                f"ELIGIBLE PROGRESS      {status['total_unique_committed_predictions']} / 100",
                f"Historical reference   {status['reference_observations']}",
                f"New compliant          {status['new_unique_committed_predictions']}",
                f"Remaining              {status['remaining_to_target']}",
                "",
                f"Journal verified predictions (all sessions) {verified}",
                f"Other / unverified observations            {unverified}",
                "Journal verification alone does not establish schedule compliance.",
                f"Labeled / unlabeled     {status['verified_labeled_predictions']} / "
                f"{status['unlabeled_predictions']}",
                f"Rejected observations   {status['rejected_observations']}",
                f"Signal coverage         {status['signal_rate']:.1%}",
                f"SKIP rate               {status['abstention_rate']:.1%}",
                f"Brier score             {metrics.get('brier_score', 'N/A — no valid labels')}",
                f"Log loss                {metrics.get('log_loss', 'N/A')}",
                f"Directional accuracy    {metrics.get('directional_accuracy_overall', 'N/A')}",
                "Baseline comparison     "
                + str(metrics.get("brier_skill_score_against_0_50_baseline", "N/A")),
                "Small research sample; directional accuracy does not establish profitability.",
                "",
                "Wallet classifications are per session; practice verification is not inferred.",
                *sorted({str(w["classification"]) for w in snapshot["wallets"]}),
                "",
                *status["data_quality_warnings"],
            ]),
        )
        wallet_by_session = {w["session"]: w for w in snapshot["wallets"]}
        rows = []
        for audit in reversed(status["session_audits"]):
            wallet = wallet_by_session.get(audit["session_id"], {})
            rows.append(
                f"{audit['session_id']}\n"
                f"  {audit['cohort_group']} · {audit['window_conformity']}\n"
                f"  Observed {audit['total_registry_observations']} | "
                f"Eligible {audit['valid_committed_predictions']} | "
                f"Rejected {audit['rejected_observations_count']} | "
                f"Uncommitted {audit['uncommitted_observations_count']}\n"
                f"  Journal verified: {audit['journal_verified']} | "
                f"Wallet: {wallet.get('classification', 'UNKNOWN')}\n"
                f"  Stop: {wallet.get('stop_reason', 'UNKNOWN')}\n"
                f"  Rejections: {json.dumps(audit['rejection_reasons'])}\n"
                f"  Deviation: {audit['deviation_reason'] or 'none'}\n"
            )
        self.write(self.audit, "\n".join(rows) or "No evidence sessions discovered.")
        self.write(
            self.recovery,
            (
                "OFFLINE PREVIEW ONLY — no recovery commands are executed.\n"
                "Historical backfill requires separate operator authorization through the CLI.\n\n"
                + json.dumps(
                    {
                        "proposed_windows": status["missing_label_windows"],
                        "missing_candles": status["missing_candle_details"],
                    },
                    indent=2,
                )
            ),
        )
        self.write(self.details, json.dumps(status, indent=2))

    def tick(self) -> None:
        now = datetime.now(UTC)
        self.clock.configure(
            text=(
                f"UTC  {now:%a %d %b %Y  %H:%M:%S}      |      "
                f"Morocco  {now.astimezone(self.local_zone):%a %d %b %Y  %H:%M:%S %z}"
            )
        )
        if self.pending is not None and self.pending.done():
            try:
                self.snapshot = self.pending.result()
                self.render()
                if self.launch_requested and not self.closing:
                    self.process.start(
                        self.saved.get(),
                        self.snapshot,
                        account_id=self.workstation.selection.selected,
                    )
                    self.state.configure(
                        text="Foreground session active; supervise the visible console."
                    )
                elif self.process.child is None:
                    self.state.configure(
                        text=(self.last_session + " · Offline audit refreshed.")
                        if self.last_session
                        else "Offline audit refreshed. Connection not probed."
                    )
            except Exception as exc:  # noqa: BLE001 - display offline failure, fail closed
                self.snapshot = None
                self.identity.configure(text="Preflight failed — displayed report is stale")
                self.state.configure(text=f"Preflight/refresh failed: {type(exc).__name__}: {exc}")
            finally:
                self.pending = None
                self.launch_requested = False
                self.refresh_button.state(["!disabled"])
        result = self.process.poll()
        if result is not None:
            self.last_session = result[1]
            self.state.configure(text=result[1])
            self.refresh()
            if hasattr(self, "workstation"):
                self.workstation.finished()
        can_start = False
        if self.snapshot is not None:
            study = PilotStudySpecification.from_dict(self.snapshot["study"])
            available = slots(study, now)
            next_available = next(
                (
                    s
                    for s in available
                    if not slot_used(self.root, self.snapshot, s)
                    and (s.end - max(now, s.start)).total_seconds() >= 300
                ),
                None,
            )
            if next_available:
                seconds = max(0, int((next_available.start - now).total_seconds()))
                self.next_slot.configure(
                    text=(
                        f"Next unattempted slot: {next_available.start:%a %H:%M} UTC / "
                        f"{next_available.start.astimezone(self.local_zone):%H:%M} Morocco "
                        f"· in {seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
                    )
                )
            else:
                self.next_slot.configure(text="No unattempted eligible slot in the next eight days")
            schedule_text = "UTC date / window          Morocco window       State\n\n" + "\n".join(
                f"{s.start:%a %Y-%m-%d %H:%M}-{s.end:%H:%M}   "
                f"{s.start.astimezone(self.local_zone):%H:%M}-"
                f"{s.end.astimezone(self.local_zone):%H:%M}           "
                f"{'ATTEMPTED' if slot_used(self.root, self.snapshot, s) else 'registered'}"
                for s in available
            )
            if schedule_text != self.schedule_text:
                self.schedule_text = schedule_text
                self.write(self.schedule, self.schedule_text)
            try:
                current = require_slot(self.root / STUDY, now=now)
                can_start = self.snapshot["model_eligible"] and not slot_used(
                    self.root, self.snapshot, current
                )
            except ProspectivePilotError:
                self.snapshot = None
                self.identity.configure(text="Study integrity check FAILED — report is stale")
                can_start = False
            except (ValueError, OSError):
                can_start = False
        busy = self.process.child is not None
        if hasattr(self, "workstation"):
            self.workstation.tick()
            busy = busy or self.workstation.future is not None
        self.start_button.state([
            "!disabled"
            if can_start and not busy and self.pending is None and not self.closing
            else "disabled"
        ])
        self.stop_button.state(["!disabled" if busy else "disabled"])
        if self.closing and not busy:
            self.close()
            return
        self.window.after(500, self.tick)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    window = tk.Tk()
    mutex = None
    try:
        mutex = DashboardMutex(root)
        Dashboard(window, root)
        window.mainloop()
    except Exception as exc:  # noqa: BLE001 - GUI startup errors must be visible
        messagebox.showerror("Research dashboard", str(exc), parent=window)
    finally:
        if mutex is not None:
            mutex.close()


if __name__ == "__main__":
    main()
