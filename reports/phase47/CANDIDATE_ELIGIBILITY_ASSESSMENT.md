# Phase 47 — Candidate Eligibility & Prospective Feasibility Assessment

**Disposition:** `NO_DEMONSTRATED_EDGE` | **Execution Authority:** `NONE` | **Stage:** `DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION`

## 1. Candidate Qualification Outcomes
No Phase 47 candidate clears the preserved Stage C / Phase 40C qualification gates:
- All three 10,000-replicate UTC day-block 95% CI lower bounds (`48.42%` M1 H15, `53.29%` M5 H6, `52.96%` M15 H4) remain below the hypothetical 54.05% break-even threshold (85% payout sensitivity).
- Nested chronological validation shows fold-to-fold variance (e.g., M1 H15 Fold 1 `36.03%`; M5 H6 Fold 0 `51.31%` vs Fold 1 `56.10%`).
- **Decision:** **0 candidates qualified for confirmatory freezing.**

## 2. Prospective Observation Feasibility (From WS6 Coverage Curves)
Compared to Phase 45/46 (~1.5%–2.5% coverage, ~0.10–0.98 pred/hr):
- **M15 H4 Extra Trees (`p47_039`) at ~10.55% coverage:** ~`0.422` pred/hr (~`3.38` per 8h supervised day); ~`128` non-overlapping in test span -> ~474–1,185 active hours (~59–148 8h sessions) for 200–500 non-overlapping samples.
- **M5 H6 Extra Trees (`p47_027`) at ~12.20% coverage:** ~`1.464` pred/hr (~`11.71` per 8h day); ~`260` non-overlapping -> ~137–342 active hours (~17–43 8h sessions) for 200–500 non-overlapping samples (with 0.75x operational derating: ~23–57 8h days).
- **M1 H15 HGB (`p47_020`) at ~6.51% coverage:** ~`3.904` pred/hr (~`31.23` per 8h day), but unstable across chronological folds.
