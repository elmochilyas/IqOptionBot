"""Dispatch exactly one visible, bounded, interactive child per operator click."""

from __future__ import annotations

import json
import os
import subprocess  # noqa: S404 - fixed argument arrays; visible, supervised process only
import time
import uuid
from pathlib import Path
from typing import Any

from iqoption_research.desktop.core import (
    CONTROL,
    STUDY,
    claim_path,
    read_snapshot,
    require_slot,
    slot_used,
)
from iqoption_research.desktop.windows import ChildJob
from iqoption_research.market_data.extended import (
    ScheduledMarketClosure,
    check_duration_vs_market_close,
    require_regular_market,
    validate_duration,
)


def command(root: Path, control: Path, saved: bool, account_id: str | None = None) -> list[str]:
    args = [
        str(root / ".venv/Scripts/python.exe"),
        "-m",
        "iqoption_research.desktop.foreground",
        str(control),
        "timing",
        "shadow-trial",
        "--live",
        "--yes-i-understand",
        "--duration",
        "300",
        "--max-candles",
        "5",
        "--study",
        str(root / STUDY),
        "--cancel-file",
        str(control / "cancel"),
        "--status-file",
        str(control / "status.json"),
    ]
    if saved:
        args.append("--use-saved-credentials")
    if account_id is not None:
        args.extend(["--account-id", account_id])
    return args


class SessionProcess:
    """Called on the UI thread; preflight/reconciliation is supplied by its worker."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.child: subprocess.Popen[bytes] | None = None
        self.job: ChildJob | None = None
        self.control: Path | None = None
        self.stopping_at: float | None = None
        self.kind = "idle"
        self.hard_deadline: float | None = None
        self.last_state: dict[str, Any] = {}
        self._last_duration_warning: str = ""

    def start(self, saved: bool, snapshot: dict[str, Any], account_id: str | None = None) -> None:
        if self.child is not None:
            raise ValueError("A foreground session is already active")
        slot = require_slot(self.root / STUDY)
        if not snapshot["model_eligible"]:
            raise ValueError("Frozen model preflight failed")
        if slot_used(self.root, snapshot, slot):
            raise ValueError("This registered slot has already been attempted")
        control = self.root / CONTROL / "sessions" / uuid.uuid4().hex
        control.mkdir(parents=True)
        claim = claim_path(self.root, snapshot["study"]["study_specification_sha256"], slot)
        claim.parent.mkdir(parents=True, exist_ok=True)
        with claim.open("x", encoding="utf-8") as stream:
            json.dump({"slot": slot.key, "control": control.name}, stream)
        self._launch(command(self.root, control, saved, account_id), control, "shadow")

    def start_collection(self, account_id: str, minutes: int) -> None:
        self.require_idle()
        validate_duration(minutes)
        from datetime import UTC, datetime

        now_utc = datetime.now(UTC)
        try:
            require_regular_market()
        except ScheduledMarketClosure as exc:
            raise ScheduledMarketClosure(
                f"Cannot start collection: {exc}. "
                "The regular EURUSD market is currently closed. "
                "Collection will be available again when the market reopens."
            ) from exc
        # Advisory: warn if duration extends past the next market close.
        duration_warning = check_duration_vs_market_close(now_utc, minutes)
        if duration_warning:
            # Store warning in a way the caller can surface it; not an error.
            self._last_duration_warning = duration_warning
        else:
            self._last_duration_warning = ""
        if uuid.UUID(hex=account_id).hex != account_id:
            raise ValueError("Select an available saved account")
        control = self.root / CONTROL / "sessions" / uuid.uuid4().hex
        control.mkdir(parents=True)
        args = [
            str(self.root / ".venv/Scripts/python.exe"),
            "-m",
            "iqoption_research.desktop.foreground",
            str(control),
            "workstation",
            "collect",
            "--live",
            "--yes-i-understand",
            "--minutes",
            str(minutes),
            "--account-id",
            account_id,
            "--cancel-file",
            str(control / "cancel"),
            "--status-file",
            str(control / "status.json"),
        ]
        self._launch(args, control, "collector")
        self.hard_deadline = time.monotonic() + minutes * 60

    def account_entry(self, account_id: str | None = None) -> None:
        self.require_idle()
        control = self.root / CONTROL / "sessions" / uuid.uuid4().hex
        control.mkdir(parents=True)
        args = [
            str(self.root / ".venv/Scripts/python.exe"),
            "-m",
            "iqoption_research.desktop.foreground",
            str(control),
            "workstation",
            "account-entry",
        ]
        if account_id is not None:
            args.extend(["--account-id", account_id])
        self._launch(args, control, "account-entry")

    def require_idle(self) -> None:
        if self.child is not None:
            raise ValueError("Account changes and overlapping sessions are prohibited while busy")

    def live_state(self) -> dict[str, Any]:
        if self.child is None:
            return {**self.last_state, "connection": "DISCONNECTED"}
        if self.control is not None:
            try:
                return dict(json.loads((self.control / "status.json").read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                pass
        return {"connection": "STARTING" if self.kind != "account-entry" else "DISCONNECTED"}

    def _launch(self, args: list[str], control: Path, kind: str) -> None:
        self.require_idle()
        self.kind = kind
        self.control = control
        self.job = ChildJob()
        try:
            # No PIPEs: the new console owns interactive stdin/stdout. Credentials stay there.
            env = {
                k: v
                for k, v in os.environ.items()
                if k.upper()
                not in {
                    "IQOPTION_EMAIL",
                    "IQOPTION_PASSWORD",
                    "IQOPTION_SSID",
                    "OPENAI_API_KEY",
                }
            }
            self.child = subprocess.Popen(  # noqa: S603
                args,
                cwd=self.root,
                env=env,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                close_fds=True,
            )
            self.job.assign(self.child.pid)
            # The foreground wrapper cannot authenticate until assigned to the lifetime job.
            (control / "ready").touch(exist_ok=False)
        except Exception:
            if self.child is not None:
                self.child.kill()
                self.child.wait(timeout=5)
                self.child = None
            self.job.close()
            self.job = None
            raise

    def stop(self, *, reason: str = "OPERATOR_CANCELLED") -> None:
        if self.child is not None and self.control is not None and self.stopping_at is None:
            (self.control / "cancel").write_text(reason, encoding="utf-8")
            self.stopping_at = time.monotonic()

    def poll(self) -> tuple[int, str] | None:
        if self.child is None:
            return None
        if self.hard_deadline is not None and time.monotonic() >= self.hard_deadline:
            self.stop(reason="DURATION_LIMIT")
        if (
            self.stopping_at is not None
            and time.monotonic() - self.stopping_at > 5
            and self.job is not None
        ):
            self.job.close()
        code = self.child.poll()
        if code is None:
            return None
        message = "Completed" if code == 0 else f"Session failed or cancelled (exit {code})"
        if self.stopping_at is not None:
            message = "Operator cancelled; inspect saved evidence (completion is not assumed)"
        if self.control is not None and (self.control / "result.json").exists():
            try:
                result = json.loads((self.control / "result.json").read_text(encoding="utf-8"))
                if not isinstance(result, dict) or not isinstance(result.get("message"), str):
                    raise ValueError("Invalid foreground result")
                message = result["message"]
            except (OSError, ValueError):
                message = "Foreground result unavailable; inspect saved evidence"
        self.last_state = {**self.live_state(), "connection": "DISCONNECTED"}
        if self.job is not None:
            self.job.close()
        self.child = None
        self.job = None
        self.stopping_at = None
        self.hard_deadline = None
        return code, message


def refresh(root: Path) -> dict[str, Any]:
    return read_snapshot(root)
