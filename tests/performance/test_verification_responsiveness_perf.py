"""Large-scale (100k / 375k candle) Collection Center UI responsiveness benchmarks."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.desktop.collection_center.test_verification_responsiveness import (
    _build_test_job,
    _HeartbeatTracker,
    _mock_modal_dialogs,  # noqa: F401
    _open_dashboard_with_job,
    _pump_until,
)


@pytest.mark.performance
@pytest.mark.desktop
def test_ui_responsiveness_100k_candles(tmp_path: Path) -> None:
    """Desktop UI remains responsive during verification of 100,000 candles."""
    _coord, job_id = _build_test_job(tmp_path, candle_count=100000, batch_size=1000)

    root, dashboard, panel = _open_dashboard_with_job(tmp_path, job_id)
    try:
        tracker = _HeartbeatTracker(root, interval_ms=10)
        tracker.start()

        op_start = time.monotonic()
        panel._verify_and_snapshot()

        _pump_until(root, panel, lambda: panel._active_operation is None, timeout=60.0)
        tracker.stop()
        print(
            "MEASURED 100k verification+snapshot: "
            f"max_ui_delay={tracker.max_delay:.3f}s, "
            f"samples={len(tracker.samples)}, elapsed={time.monotonic() - op_start:.3f}s"
        )

        assert len(tracker.samples) > 20
        assert tracker.max_delay < 0.400, (
            f"UI frozen during 100k verify: max delay was {tracker.max_delay:.3f}s"
        )

        assert panel._last_verification_report is not None
        assert panel._last_verification_report["verified"] is True
        assert panel._last_verification_report["unique_verified_candles"] == 100000
        assert panel.lbl_status_badge.cget("text") in ("COMPLETED_VERIFIED", "SNAPSHOT REGISTERED")
    finally:
        dashboard.close()


@pytest.mark.performance
@pytest.mark.desktop
def test_ui_responsiveness_375k_candles(tmp_path: Path) -> None:
    """Desktop UI remains responsive during verification of 375,000 candles."""
    _coord, job_id = _build_test_job(tmp_path, candle_count=375000, batch_size=5000)

    root, dashboard, panel = _open_dashboard_with_job(tmp_path, job_id)
    try:
        tracker = _HeartbeatTracker(root, interval_ms=10)
        tracker.start()

        op_start = time.monotonic()
        panel._verify_and_snapshot()

        _pump_until(root, panel, lambda: panel._active_operation is None, timeout=120.0)
        tracker.stop()
        print(
            "MEASURED 375k verification+snapshot: "
            f"max_ui_delay={tracker.max_delay:.3f}s, "
            f"samples={len(tracker.samples)}, elapsed={time.monotonic() - op_start:.3f}s"
        )

        assert len(tracker.samples) > 30
        assert tracker.max_delay < 2.5, (
            f"UI frozen during 375k verify: max delay was {tracker.max_delay:.3f}s"
        )

        assert panel._last_verification_report is not None
        assert panel._last_verification_report["verified"] is True
        assert panel._last_verification_report["unique_verified_candles"] == 375000
        assert panel.lbl_status_badge.cget("text") in ("COMPLETED_VERIFIED", "SNAPSHOT REGISTERED")
    finally:
        dashboard.close()
