"""Phase 46 smoke tests (offline, no broker)."""

from __future__ import annotations

import subprocess  # noqa: S404
import sys
from pathlib import Path


def test_phase46_constants() -> None:
    from iqoption_research.phase46.constants import PHASE46_FOCUSED_TRIAL_IDS

    assert len(PHASE46_FOCUSED_TRIAL_IDS) == 3
    assert "p45_046_M15_M15_H2_random_forest_native_only" in PHASE46_FOCUSED_TRIAL_IDS


def test_run_phase46_script_help() -> None:
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "run_phase46_research.py"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "focused" in proc.stdout.lower()
