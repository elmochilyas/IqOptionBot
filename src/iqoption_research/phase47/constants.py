"""Phase 47 constants, governance invariants, and bounded research budgets.

DEVELOPMENT_RESEARCH only. execution_authority=NONE. NO_DEMONSTRATED_EDGE.
"""

from __future__ import annotations

from typing import Final

# Directories
PHASE47_DOCS_DIR: Final[str] = "docs/phase47"
PHASE47_REPORTS_DIR: Final[str] = "reports/phase47"
PHASE47_DATA_DIR: Final[str] = "data/phase47"

# Governance invariants (always preserved)
DISPOSITION: Final[str] = "NO_DEMONSTRATED_EDGE"
EXECUTION_AUTHORITY: Final[str] = "NONE"
RESEARCH_STAGE: Final[str] = "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION"
PROTECTED_BOUNDARY_TS: Final[int] = 1794477600  # 2026-11-12T10:00:00Z
PROTECTED_BOUNDARY_UTC: Final[str] = "2026-11-12T10:00:00Z"

# Confirmed coverage levels for WS6 accuracy-vs-coverage analysis
COVERAGE_TARGETS: Final[tuple[float, ...]] = (0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 1.00)

# Minimum inner selection guards (prevents 2-sample inner selection overfitting)
MIN_INNER_SELECTED_SAMPLES: Final[int] = 50
MIN_INNER_DAYS: Final[int] = 5
MIN_INNER_CLASS_COUNT: Final[int] = 10

# Bounded search budget
MAX_SUBSTANTIVE_EXPERIMENTS: Final[int] = 60
BREAK_EVEN_85_PAYOUT: Final[float] = 1.0 / 1.85  # ~0.54054 (hypothetical sensitivity only)
