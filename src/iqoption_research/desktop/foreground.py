"""Visible terminal entry point. Only this explicitly dispatched child authenticates."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path


def main() -> None:
    control = Path(sys.argv[1])
    code = 1
    message = "Foreground session failed"
    try:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise ValueError("Interactive terminal required")
        deadline = time.monotonic() + 10
        while not (control / "ready").exists():
            if time.monotonic() >= deadline or (control / "cancel").exists():
                raise TimeoutError("Dashboard lifetime guard unavailable")
            time.sleep(0.05)
        if (control / "cancel").exists():
            raise KeyboardInterrupt
        from iqoption_research.cli.main import main as cli_main

        sys.argv = ["iqresearch", *sys.argv[2:]]
        cli_main()
        code, message = 0, "Completed; results are being reconciled offline"
    except SystemExit as exc:
        code = int(exc.code or 0)
        message = "Completed" if code == 0 else f"CLI rejected session (exit {code})"
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Operator cancelled; saved evidence remains in the audit"
    except Exception as exc:  # noqa: BLE001 - never persist authentication exception contents
        message = f"Session failed: {type(exc).__name__}; inspect the session audit"
    finally:
        # Only fixed text and exception type names cross this boundary, never exception payloads.
        (control / "result.json").write_text(
            json.dumps({"exit_code": code, "message": message}), encoding="utf-8"
        )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
