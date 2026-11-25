"""Phase 46 constants (confirmed selections, paths, governance labels).

DEVELOPMENT_RESEARCH only. execution_authority=NONE. NO_DEMONSTRATED_EDGE.
"""

from __future__ import annotations

# Confirmed WS1 focus: strongest observed directional-accuracy candidate per timeframe.
# Do not substitute.
PHASE46_FOCUSED_TRIAL_IDS: tuple[str, ...] = (
    "p45_046_M15_M15_H2_random_forest_native_only",  # M15, M15_H2, random_forest
    "p45_029_M5_M5_H3_hist_gradient_boosting_native_only",  # M5, M5_H3, hist_gradient_boosting
    "p45_004_M1_M1_H1_random_forest_native_only",  # M1, M1_H1, random_forest
)

# Paths
PHASE46_DOCS_DIR = "docs/phase46"
PHASE46_REPORTS_DIR = "reports/phase46"
PHASE46_DATA_DIR = "data/phase46"

# Governance
DISPOSITION = "NO_DEMONSTRATED_EDGE"
EXECUTION_AUTHORITY = "NONE"
RESEARCH_STAGE = "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION"
