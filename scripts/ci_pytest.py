"""Run the CI pytest suite and persist its true exit code.

The GitHub-hosted Windows runner has been observed interrupting the PowerShell
process immediately after pytest completes successfully.  CI therefore records
the pytest return code in ``PYTEST_EXIT_FILE`` and a separate step enforces it.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest


def main() -> int:
    """Run pytest, ignoring external console SIGINT, and return its exit code."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    code = int(pytest.main(["-q", "--full-trace", *sys.argv[1:]]))

    exit_file = os.environ.get("PYTEST_EXIT_FILE")
    if exit_file:
        Path(exit_file).write_text(str(code), encoding="utf-8")

    print(f"PYTEST_EXIT_CODE={code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
