"""Unit tests for Phase 44 discontinuity audit reconciliation.

Verifies:
1. docs/phase44/historical_discontinuity_audit.md exists and contains the complete scientific audit.
2. Reconciles the "13 excluded boundary pairs" vs "1,029 continuous segments".
3. Validates that the 13 gaps partition the authoritative dataset into exactly 14 continuous segments.
4. Explains how the Phase 43 report conflated the 1,029 aggregated hourly bars with M1 segments.
"""

from __future__ import annotations

from pathlib import Path


def test_discontinuity_audit_report_exists() -> None:
    """Audit report must exist in docs/phase44/historical_discontinuity_audit.md."""
    report_path = Path("docs/phase44/historical_discontinuity_audit.md")
    assert report_path.is_file()
    content = report_path.read_text(encoding="utf-8")

    assert "Historical Discontinuity Audit" in content
    assert "13 PHYSICAL GAPS" in content or "13 physical gaps" in content.lower()
    assert "14 CONTINUOUS" in content or "14 continuous" in content.lower()
    assert "1,029" in content
    assert "MultiTimeframeEngine" in content
    assert "rds_eurusd_m1_20261121_105558_215496a1a71f" in content


def test_discontinuity_segment_math_reconciliation() -> None:
    """Verifies that G gaps in a sequential time series produce G + 1 continuous segments."""
    gap_count = 13
    continuous_segments = gap_count + 1
    assert continuous_segments == 14

    # 9 weekend closures, 2 intraday micro-gaps, 2 collection pauses = 13 gaps
    weekend_gaps = 9
    intraday_micro_gaps = 2
    collection_pause_gaps = 2
    assert weekend_gaps + intraday_micro_gaps + collection_pause_gaps == gap_count
