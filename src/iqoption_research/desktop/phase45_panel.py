"""Desktop Phase 45 Research panel for offline historical development research.

The panel is strictly inspection/research oriented: it reads generated Phase 45
development manifests and ledger artifacts, submits the offline runner to the
dashboard's single-worker executor, and marshals every worker update through a
main-thread UI queue. No broker, network, account, or execution capability is
used or exposed here.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import re
import time
import tkinter as tk
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iqoption_research.desktop.app import Dashboard
    from iqoption_research.phase45.runner import Phase45Progress

logger = logging.getLogger(__name__)

SAFETY_LABEL = "NO_DEMONSTRATED_EDGE | execution_authority=NONE | DEVELOPMENT RESEARCH ONLY"

ARTIFACT_DIR = Path("reports") / "phase45"
MANIFEST_DIR = ARTIFACT_DIR / "manifests"
LEDGER_NAME = "experiment_ledger.json"
FINAL_REPORT_NAMES = (
    "scientific_evaluation.json",
    "final_report.json",
    "report.json",
    "summary.json",
)

MAX_TRIALS = 60
BOOTSTRAP_REPLICATES = 1000
MAX_TRIAL_ROWS = 50

_DENIED_KEY_FRAGMENTS = (
    "account",
    "credential",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "private_key",
    "wallet",
    "broker",
    "order",
    "candle",
    "raw",
    "predict",
    "bar",
    "series",
    "array",
    "tick",
    "email",
    "session",
    "user",
    "client_id",
    "device",
    "signature",
    "key_id",
)
_REDACT_PATTERN = re.compile(
    r"(?i)\b(account[_-]?id|wallet[_-]?id|credential|password|secret|token|"
    r"api[_-]?key|private[_-]?key)\b\s*[:=]\s*[^\s,;]+"
)
_TIMEFRAME_EXACT = {
    "60": "M1",
    "60s": "M1",
    "1m": "M1",
    "1min": "M1",
    "m1": "M1",
    "300": "M5",
    "300s": "M5",
    "5m": "M5",
    "5min": "M5",
    "m5": "M5",
    "900": "M15",
    "900s": "M15",
    "15m": "M15",
    "15min": "M15",
    "m15": "M15",
}


def _load_runner_class() -> Any:
    """Import the Phase 45 runner lazily so the desktop shell stays loadable."""
    from iqoption_research.phase45.runner import Phase45Runner

    return Phase45Runner


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _first_string(data: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return ""


def _normalise_timeframe(data: dict[str, Any], manifest_id: str) -> str | None:
    raw = ""
    for key in ("timeframe_seconds", "candle_size_seconds", "base_timeframe_seconds", "timeframe"):
        value = data.get(key)
        if value not in (None, ""):
            raw = str(value).strip().lower()
            break
    if raw in _TIMEFRAME_EXACT:
        return _TIMEFRAME_EXACT[raw]
    for token, label in (("m15", "M15"), ("m5", "M5"), ("m1", "M1")):
        if re.search(rf"(?<![0-9a-z]){token}(?![0-9a-z])", raw):
            return label
    match = re.search(r"(?:^|[_-\s])m(1|5|15)(?:[_-\s]|$)", manifest_id.lower())
    return f"M{match.group(1)}" if match else None


def _is_development_manifest(data: dict[str, Any], manifest_id: str) -> bool:
    """Reject anything that could resolve to an authoritative/protected snapshot."""
    lowered_id = manifest_id.lower()
    if any(
        token in lowered_id
        for token in ("protected", "authoritative", "production", "live", "snapshot")
    ):
        return False
    for key in (
        "is_protected",
        "protected_snapshot",
        "authoritative",
        "is_authoritative",
        "is_production",
        "is_live",
    ):
        if _as_bool(data.get(key)):
            return False
    authority = str(data.get("execution_authority", "NONE")).strip().upper()
    if authority not in {"", "NONE", "NO_EXECUTION"}:
        return False
    for key in ("provenance_class", "dataset_class", "classification", "lineage", "source_type"):
        value = str(data.get(key, "")).strip().lower()
        if value in {"protected", "authoritative", "production", "live", "broker", "account"}:
            return False
    if str(data.get("is_native", True)).strip().lower() in {"false", "0", "no"}:
        return False
    return not (data.get("derived_from") or data.get("aggregated_from"))


def _redact_text(value: str) -> str:
    return _REDACT_PATTERN.sub("[REDACTED]", value)


def _is_denied_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _DENIED_KEY_FRAGMENTS)


def _sanitize_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "[REDACTED_DEPTH]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _redact_text(value)[:_MAX_STRING_LENGTH]
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(item, depth=depth + 1)
            for key, item in value.items()
            if isinstance(key, str) and not _is_denied_key(key) and item is not None
        }
    if isinstance(value, list | tuple):
        if any(isinstance(item, list | tuple) for item in value):
            return "[REDACTED_NESTED_ARRAY]"
        return [_sanitize_value(item, depth=depth + 1) for item in value[:_MAX_LIST_ITEMS]]
    return _redact_text(str(value))[:_MAX_STRING_LENGTH]


def _sanitize_mapping(data: dict[str, Any]) -> dict[str, Any]:
    result = _sanitize_value(data)
    return result if isinstance(result, dict) else {}


_MAX_STRING_LENGTH = 2000
_MAX_LIST_ITEMS = 200


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Phase 45 Research Report", "", f"- Safety: {SAFETY_LABEL}", ""]
    for key in sorted(summary):
        if key == "safety_label":
            continue
        value = summary[key]
        if isinstance(value, dict | list):
            lines.extend([
                f"## {key}",
                "```json",
                json.dumps(value, indent=2, sort_keys=True, default=str),
                "```",
                "",
            ])
        else:
            lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


class Phase45ResearchPanel:
    """Narrow Phase 45 subtab inside the Model Laboratory."""

    btn_refresh: ttk.Button
    btn_start: ttk.Button
    btn_cancel: ttk.Button
    btn_export: ttk.Button
    lbl_safety: ttk.Label
    lbl_artifacts: ttk.Label
    lbl_progress: ttk.Label
    lbl_message: ttk.Label
    progress: ttk.Progressbar
    manifest_tree: ttk.Treeview
    trials_tree: ttk.Treeview
    summary_text: tk.Text

    def __init__(self, dashboard: Dashboard, parent_notebook: ttk.Notebook) -> None:
        self.dashboard = dashboard
        self.root = Path(dashboard.root)
        self.repo_root = Path(dashboard.root)

        self._ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._ui_poll_pending = False
        self._ui_poll_after_id: str | None = None
        # Every background operation carries a generation token. Results from a
        # superseded generation are discarded before they can mutate the view.
        self._operation_generation = 0
        self._active_generation: int | None = None
        self.active_future: Future[Any] | None = None
        self._cancel_requested = False
        self._run_started_at: float | None = None

        self._manifests: list[dict[str, str]] = []
        self._trial_records: tuple[dict[str, Any], ...] = ()
        self._last_result: dict[str, Any] | None = None
        self._last_progress: dict[str, Any] = {
            "completed_trials": 0,
            "total_trials": 0,
            "active_model": "-",
            "elapsed_seconds": 0.0,
        }

        self.frame = ttk.Frame(parent_notebook, padding=10)
        parent_notebook.add(self.frame, text="Phase 45 Research")

        self._build_ui()
        self.refresh_artifacts()
        self._schedule_ui_poll()

    # ---- PR #61 main-thread queue pattern (copied from Collection Center) ----
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

    # ------------------------------------------------------------------------

    def _next_generation(self) -> int:
        self._operation_generation += 1
        return self._operation_generation

    def _is_current_worker_result(self, generation: int) -> bool:
        if generation != self._operation_generation:
            logger.debug("Discarding stale Phase 45 worker result from generation %s", generation)
            return False
        return True

    def tick(self) -> None:
        """Manual pump invoked from ModelLaboratoryPanel.tick() on the main thread."""
        self._poll_ui_queue()

    def busy(self) -> bool:
        return self._active_generation is not None

    def _build_ui(self) -> None:
        header = ttk.Frame(self.frame)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(
            header,
            text="Phase 45 · Historical Research",
            font=("Segoe UI", 13, "bold"),
            foreground="#17324d",
        ).pack(anchor="w")
        self.lbl_safety = ttk.Label(
            header,
            text=SAFETY_LABEL,
            font=("Segoe UI", 9, "bold"),
            foreground="#9c3414",
        )
        self.lbl_safety.pack(anchor="w", pady=(2, 0))

        manifest_frame = ttk.LabelFrame(
            self.frame, text="Development Datasets (Native M1 / M5 / M15)", padding=8
        )
        manifest_frame.pack(fill="x", pady=(0, 8))

        columns = ("manifest_id", "timeframe", "status")
        self.manifest_tree = ttk.Treeview(
            manifest_frame, columns=columns, show="headings", height=4
        )
        self.manifest_tree.heading("manifest_id", text="Development Manifest ID")
        self.manifest_tree.heading("timeframe", text="Timeframe")
        self.manifest_tree.heading("status", text="Status")
        self.manifest_tree.column("manifest_id", width=420)
        self.manifest_tree.column("timeframe", width=90, anchor="center")
        self.manifest_tree.column("status", width=140, anchor="center")
        self.manifest_tree.pack(fill="x", pady=(0, 6))
        self.manifest_tree.bind("<<TreeviewSelect>>", self._on_manifest_selected)

        controls = ttk.Frame(manifest_frame)
        controls.pack(fill="x", pady=(0, 4))
        self.btn_refresh = ttk.Button(
            controls, text="Refresh Artifacts", command=self.refresh_artifacts
        )
        self.btn_refresh.pack(side="left", padx=(0, 8))
        self.btn_start = ttk.Button(controls, text="Start Research", command=self.start_research)
        self.btn_start.pack(side="left", padx=(0, 8))
        self.btn_cancel = ttk.Button(controls, text="Cancel", command=self.cancel_research)
        self.btn_cancel.pack(side="left", padx=(0, 8))
        self.btn_cancel.state(["disabled"])
        self.btn_export = ttk.Button(controls, text="Export Report", command=self.export_report)
        self.btn_export.pack(side="left")

        self.lbl_artifacts = ttk.Label(
            manifest_frame,
            text="Scanning Phase 45 development manifests...",
            wraplength=920,
        )
        self.lbl_artifacts.pack(anchor="w", pady=(2, 0))

        progress_frame = ttk.LabelFrame(self.frame, text="Research Progress", padding=8)
        progress_frame.pack(fill="x", pady=(0, 8))
        self.progress = ttk.Progressbar(progress_frame, maximum=100)
        self.progress.pack(fill="x", pady=(0, 4))
        self.lbl_progress = ttk.Label(
            progress_frame, text="Trials 0/0 | Active: - | Elapsed 00:00 | State: IDLE"
        )
        self.lbl_progress.pack(anchor="w")
        self.lbl_message = ttk.Label(
            progress_frame,
            text="Inspection only until generated Phase 45 development artifacts are present.",
            wraplength=920,
        )
        self.lbl_message.pack(anchor="w", pady=(2, 0))

        trials_frame = ttk.LabelFrame(self.frame, text="Recent Completed Trials", padding=8)
        trials_frame.pack(fill="x", pady=(0, 8))
        trial_columns = ("completed", "model", "manifest", "disposition", "metrics")
        self.trials_tree = ttk.Treeview(
            trials_frame, columns=trial_columns, show="headings", height=6
        )
        self.trials_tree.heading("completed", text="Completed UTC")
        self.trials_tree.heading("model", text="Active Model")
        self.trials_tree.heading("manifest", text="Manifest")
        self.trials_tree.heading("disposition", text="Disposition")
        self.trials_tree.heading("metrics", text="Summary Metrics")
        for column in trial_columns:
            self.trials_tree.column(column, width=140)
        self.trials_tree.column("metrics", width=240)
        self.trials_tree.pack(fill="x")

        ttk.Label(self.frame, text="Final Report Summary", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(0, 2)
        )
        self.summary_text = tk.Text(
            self.frame,
            wrap="word",
            font=("Consolas", 10),
            relief="flat",
            background="#ffffff",
            foreground="#20364a",
            padx=10,
            pady=8,
            height=8,
        )
        summary_scroll = ttk.Scrollbar(
            self.frame, orient="vertical", command=self.summary_text.yview
        )
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        summary_scroll.pack(side="right", fill="y")
        self.summary_text.pack(fill="both", expand=True)
        self.summary_text.configure(state="disabled")

    def _on_manifest_selected(self, _event: object = None) -> None:
        selected = self._selected_manifest_ids()
        if not selected:
            return
        manifest = next(
            (item for item in self._manifests if item["manifest_id"] == selected[0]), None
        )
        if manifest is not None:
            self.lbl_artifacts.configure(
                text=(
                    f"Selected {manifest['manifest_id']} ({manifest['timeframe']}) · "
                    f"status: {manifest['status']}. Protected snapshots are never listed."
                )
            )

    def _selected_manifest_ids(self) -> list[str]:
        ids: list[str] = []
        for item in self.manifest_tree.selection():
            values = self.manifest_tree.item(item, "values")
            if values and str(values[0]).strip():
                ids.append(str(values[0]).strip())
        return ids

    def _scan_development_manifests(self) -> list[dict[str, str]]:
        """Read both per-manifest files and the generated manifest-list artifact."""
        manifest_dir = self.repo_root / MANIFEST_DIR
        if not manifest_dir.is_dir():
            return []
        manifests: dict[str, dict[str, str]] = {}
        for path in sorted(manifest_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.debug("Skipping unreadable Phase 45 manifest %s: %s", path, exc)
                continue
            candidates = data if isinstance(data, list) else [data]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                manifest_id = _first_string(candidate, ("manifest_id", "dataset_id", "id"))
                if not manifest_id or manifest_id in manifests:
                    continue
                timeframe = _normalise_timeframe(candidate, manifest_id)
                if timeframe is None or not _is_development_manifest(candidate, manifest_id):
                    continue
                status = (
                    _first_string(
                        candidate,
                        (
                            "status",
                            "validation_status",
                            "state",
                            "readiness_status",
                            "provenance_status",
                        ),
                    )
                    or "UNKNOWN"
                )
                manifests[manifest_id] = {
                    "manifest_id": manifest_id,
                    "timeframe": timeframe,
                    "status": status,
                    "path": str(path),
                }
        return [manifests[key] for key in sorted(manifests)]

    def _render_manifests(self) -> None:
        for item in self.manifest_tree.get_children():
            self.manifest_tree.delete(item)
        for manifest in self._manifests:
            self.manifest_tree.insert(
                "",
                "end",
                values=(manifest["manifest_id"], manifest["timeframe"], manifest["status"]),
            )
        children = self.manifest_tree.get_children()
        if children and not self.manifest_tree.selection():
            self.manifest_tree.selection_set(children[0])

    def _load_ledger_json(self) -> dict[str, Any] | None:
        path = self.repo_root / ARTIFACT_DIR / LEDGER_NAME
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("Unable to read Phase 45 ledger %s: %s", path, exc)
            return None
        return data if isinstance(data, dict) else None

    def _reload_trial_records_from_ledger(self) -> None:
        """Reload completed trials; never erase preserved rows on cancellation."""
        ledger = self._load_ledger_json()
        if ledger is None:
            return
        raw_records = ledger.get("trial_records") or ledger.get("trials") or ledger.get("records")
        if isinstance(raw_records, list):
            records = tuple(item for item in raw_records if isinstance(item, dict))
            if records:
                self._trial_records = records
        self._render_trials()

    @staticmethod
    def _trial_timestamp(record: dict[str, Any]) -> str:
        value = _first_string(
            record, ("completed_at_utc", "completed_at", "timestamp", "created_at")
        )
        return value[:19].replace("T", " ")

    @staticmethod
    def _trial_metrics(record: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("directional_accuracy", "test_accuracy", "win_rate", "elapsed_seconds"):
            value = record.get(key)
            if isinstance(value, bool | int | float):
                parts.append(f"{key}={value}")
        return ", ".join(parts) or "-"

    def _render_trials(self) -> None:
        for item in self.trials_tree.get_children():
            self.trials_tree.delete(item)
        for record in reversed(self._trial_records[-MAX_TRIAL_ROWS:]):
            self.trials_tree.insert(
                "",
                "end",
                values=(
                    self._trial_timestamp(record),
                    _first_string(record, ("active_model", "model_id", "model_family")) or "-",
                    _first_string(record, ("manifest_id", "dataset_id")) or "-",
                    _first_string(record, ("disposition", "status", "verdict")) or "-",
                    self._trial_metrics(record),
                ),
            )

    def _load_report_summary(self) -> dict[str, Any]:
        if self._last_result is not None:
            return dict(self._last_result)
        for name in FINAL_REPORT_NAMES:
            path = self.repo_root / ARTIFACT_DIR / name
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.debug("Unable to read Phase 45 report %s: %s", path, exc)
                continue
            if isinstance(data, dict):
                return data
        ledger = self._load_ledger_json()
        if ledger is not None:
            return {
                "status": "LEDGER_ONLY",
                "disposition": _first_string(ledger, ("disposition", "global_disposition"))
                or "NO_DEMONSTRATED_EDGE",
                "trial_count": len(self._trial_records),
                "resource_usage": ledger.get("resource_usage", {}),
            }
        return {
            "status": "NO_ARTIFACTS",
            "disposition": "NO_DEMONSTRATED_EDGE",
            "message": "No Phase 45 final report or ledger artifacts are present.",
        }

    def _sanitized_report(self, summary: dict[str, Any]) -> dict[str, Any]:
        clean = _sanitize_mapping(summary)
        clean["safety_label"] = SAFETY_LABEL
        clean["execution_authority"] = "NONE"
        clean.setdefault("disposition", "NO_DEMONSTRATED_EDGE")
        return clean

    @staticmethod
    def _write_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _render_summary(self, summary: dict[str, Any]) -> None:
        clean = self._sanitized_report(summary)
        self._write_text(
            self.summary_text,
            json.dumps(clean, indent=2, sort_keys=True, default=str),
        )

    def _render_progress_label(self, state: str) -> None:
        completed = _safe_int(self._last_progress.get("completed_trials"))
        total = _safe_int(self._last_progress.get("total_trials"))
        active = str(self._last_progress.get("active_model") or "-")
        elapsed = _format_elapsed(_safe_float(self._last_progress.get("elapsed_seconds")))
        self.lbl_progress.configure(
            text=f"Trials {completed}/{total} | Active: {active} | Elapsed {elapsed} | State: {state}"
        )
        pct = (100.0 * completed / total) if total > 0 else 0.0
        self.progress["value"] = min(100.0, pct)

    def _update_controls(self) -> None:
        busy = self.busy()
        self.btn_start.state(["disabled"] if busy or not self._manifests else ["!disabled"])
        self.btn_cancel.state(["disabled"] if not busy or self._cancel_requested else ["!disabled"])

    def refresh_artifacts(self) -> None:
        """Rescan generated development manifests and local report artifacts."""
        self._manifests = self._scan_development_manifests()
        self._render_manifests()
        self._reload_trial_records_from_ledger()
        self._render_summary(self._load_report_summary())
        if self._manifests:
            self.lbl_artifacts.configure(
                text=(
                    f"Loaded {len(self._manifests)} native Phase 45 development manifest(s) "
                    "from reports/phase45/manifests. Protected/authoritative snapshots are excluded."
                )
            )
        else:
            self.lbl_artifacts.configure(
                text=(
                    "No Phase 45 development manifests found - INSPECTION ONLY. "
                    "Start Research is disabled; Refresh Artifacts remains available."
                )
            )
            self.lbl_message.configure(
                text="No generated Phase 45 artifacts are present. Only local inspection is available."
            )
        self._render_progress_label("CANCELLING" if self._cancel_requested else "IDLE")
        self._update_controls()

    def _cancel_token(self) -> bool:
        """Worker-side cooperative cancellation check; touches no Tk objects."""
        return self._cancel_requested

    def start_research(self) -> None:
        """Submit the offline runner to the dashboard's single-worker executor."""
        if self.busy():
            return
        if not self._manifests:
            messagebox.showinfo(
                "Phase 45 Research Inspection Only",
                "No generated Phase 45 development manifests are available. "
                "Start Research is disabled; use Refresh Artifacts after generation.",
            )
            return
        manifest_ids = self._selected_manifest_ids() or [
            manifest["manifest_id"] for manifest in self._manifests
        ]
        generation = self._next_generation()
        self._active_generation = generation
        self._cancel_requested = False
        self._run_started_at = time.monotonic()
        self._last_progress = {
            "completed_trials": 0,
            "total_trials": 0,
            "active_model": "-",
            "elapsed_seconds": 0.0,
        }
        self._render_progress_label("STARTING")
        self.lbl_message.configure(
            text=f"Starting offline research for {len(manifest_ids)} development manifest(s)..."
        )
        self._update_controls()
        self.active_future = self.dashboard.executor.submit(
            self._run_suite_worker, manifest_ids, generation
        )

    def _run_suite_worker(self, manifest_ids: list[str], generation: int) -> None:
        """Worker thread: only queue callbacks, never touch a Tk widget."""
        try:
            runner_class = _load_runner_class()
            runner = runner_class(self.repo_root)

            def progress_callback(progress: Phase45Progress) -> None:
                def apply_progress() -> None:
                    self._on_progress(progress, generation)

                self._dispatch_to_ui(apply_progress)

            result = runner.run_suite(
                manifest_ids,
                max_trials=MAX_TRIALS,
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
                progress_callback=progress_callback,
                cancel_token=self._cancel_token,
            )

            def apply_complete() -> None:
                self._on_complete(result, generation)

            self._dispatch_to_ui(apply_complete)
        except Exception as exc:
            logger.exception("Phase 45 research worker failed: %s", exc)
            error = exc

            def apply_failed() -> None:
                self._on_failed(error, generation)

            self._dispatch_to_ui(apply_failed)

    def _on_progress(self, progress: Phase45Progress, generation: int) -> None:
        if not self._is_current_worker_result(generation):
            return
        self._last_progress = {
            "completed_trials": _safe_int(getattr(progress, "completed_trials", 0)),
            "total_trials": _safe_int(getattr(progress, "total_trials", 0)),
            "active_model": str(getattr(progress, "active_model", "") or "-"),
            "elapsed_seconds": _safe_float(getattr(progress, "elapsed_seconds", 0.0)),
        }
        self._render_progress_label(
            "CANCELLING"
            if self._cancel_requested or getattr(progress, "cancelled", False)
            else "RUNNING"
        )
        message = str(getattr(progress, "message", "") or "")
        if message:
            self.lbl_message.configure(text=message)
        records = getattr(progress, "trial_records", ()) or ()
        new_records = tuple(item for item in records if isinstance(item, dict))
        if new_records:
            self._trial_records = new_records
            self._render_trials()

    @staticmethod
    def _coerce_result(result: Any) -> dict[str, Any]:
        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
            if isinstance(data, dict):
                return data
        if isinstance(result, dict):
            return result
        return {
            "disposition": str(getattr(result, "disposition", "UNKNOWN")),
            "execution_authority": "NONE",
            "trial_count": _safe_int(getattr(result, "trial_count", 0)),
            "resource_usage": getattr(result, "resource_usage", {}),
        }

    def _finish_operation(self) -> None:
        self.active_future = None
        self._active_generation = None
        self._run_started_at = None
        self._update_controls()

    def _on_complete(self, result: Any, generation: int) -> None:
        if not self._is_current_worker_result(generation):
            return
        self._finish_operation()
        summary = self._coerce_result(result)
        self._last_result = summary
        self._reload_trial_records_from_ledger()
        self._render_summary(summary)
        state = "CANCELLED" if self._cancel_requested else "COMPLETE"
        self._render_progress_label(state)
        message = _first_string(summary, ("message", "disposition", "suite_id"))
        self.lbl_message.configure(
            text=message or "Phase 45 research finished; no demonstrated edge."
        )

    def _on_failed(self, exc: Exception, generation: int) -> None:
        if not self._is_current_worker_result(generation):
            return
        self._finish_operation()
        self._reload_trial_records_from_ledger()
        state = "CANCELLED" if self._cancel_requested else "FAILED"
        self._render_progress_label(state)
        self.lbl_message.configure(text=f"{type(exc).__name__}: {exc}")

    def _on_cancel_requested(self, generation: int) -> None:
        if not self._is_current_worker_result(generation):
            return
        self._reload_trial_records_from_ledger()
        self._render_progress_label("CANCELLING")
        self.lbl_message.configure(
            text="Cooperative cancellation requested; completed trial records preserved from ledger."
        )

    def cancel_research(self) -> None:
        """Set the cooperative flag; the queued callback owns all UI changes."""
        if not self.busy():
            return
        self._cancel_requested = True
        self.btn_cancel.state(["disabled"])
        generation = self._active_generation or 0
        self._dispatch_to_ui(lambda: self._on_cancel_requested(generation))

    def export_report(self) -> None:
        """Export only a sanitized JSON/Markdown summary, on the main thread."""
        summary = self._sanitized_report(self._last_result or self._load_report_summary())
        if summary.get("status") == "NO_ARTIFACTS":
            messagebox.showinfo(
                "Export Report", "No Phase 45 report artifacts are available to export."
            )
            return
        target_value = filedialog.asksaveasfilename(
            parent=self.dashboard.window,
            title="Export Phase 45 Research Report",
            defaultextension=".json",
            initialfile="phase45_research_report.json",
            filetypes=[
                ("JSON Files", "*.json"),
                ("Markdown Files", "*.md"),
                ("All Files", "*.*"),
            ],
        )
        if not target_value:
            return
        target = Path(str(target_value))
        try:
            if target.suffix.lower() in {".md", ".markdown"}:
                target.write_text(_render_markdown(summary), encoding="utf-8")
            else:
                target.write_text(
                    json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
                )
            self.lbl_message.configure(text=f"Exported sanitized Phase 45 report to {target}")
        except (OSError, ValueError) as exc:
            logger.exception("Phase 45 report export failed: %s", exc)
            messagebox.showerror("Export Failed", f"Could not export sanitized report: {exc}")
