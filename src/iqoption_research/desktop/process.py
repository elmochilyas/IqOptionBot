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


def command(root: Path, control: Path, saved: bool) -> list[str]:
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
    ]
    if saved:
        args.append("--use-saved-credentials")
    return args


class SessionProcess:
    """Called on the UI thread; preflight/reconciliation is supplied by its worker."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.child: subprocess.Popen[bytes] | None = None
        self.job: ChildJob | None = None
        self.control: Path | None = None
        self.stopping_at: float | None = None

    def start(self, saved: bool, snapshot: dict[str, Any]) -> None:
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
                command(self.root, control, saved),
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

    def stop(self) -> None:
        if self.child is not None and self.control is not None and self.stopping_at is None:
            (self.control / "cancel").touch(exist_ok=True)
            self.stopping_at = time.monotonic()

    def poll(self) -> tuple[int, str] | None:
        if self.child is None:
            return None
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
            result = json.loads((self.control / "result.json").read_text(encoding="utf-8"))
            message = str(result["message"])
        if self.job is not None:
            self.job.close()
        self.child = None
        self.job = None
        self.stopping_at = None
        return code, message


def refresh(root: Path) -> dict[str, Any]:
    return read_snapshot(root)
