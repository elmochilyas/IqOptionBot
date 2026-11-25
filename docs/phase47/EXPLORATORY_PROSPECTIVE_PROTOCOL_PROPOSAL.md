# Phase 47 WS9 — Exploratory Prospective Research Protocol Proposal (Draft)

**Status:** `PROPOSAL / DRAFT / NOT_ACTIVATED` (requires explicit owner authorization before any genuine observation).  
**Governance Invariants:** `execution_authority = NONE`, `NO_DEMONSTRATED_EDGE`, zero order execution, zero wallet mutation, Stage C and Phase 40C confirmatory gates preserved without weakening.

---

## 1. Circular Dependency Identified in Existing Governance
Inspection of `src/iqoption_research/ml/freezer.py`, Phase 40C campaign contracts, and `docs/phase46/PREREGISTERED_PROSPECTIVE_CONTRACT.md` reveals a structural circular dependency:
1. Confirmatory qualification (`ELIGIBLE_FOR_PROSPECTIVE_RESEARCH` / Stage C) requires **independent out-of-sample evidence**.
2. When historical protected partitions are exhausted or no historical development candidate clears Stage C, **independent prospective observation** is the only way to collect genuinely unseen out-of-sample evidence.
3. Yet Phase 46's draft prospective contract blocks genuine observation until a candidate qualifies.

## 2. Four-Tier Candidate Classification (Proposed)
1. **Development Candidate (`DEVELOPMENT_RESEARCH_CANDIDATE`):** Pre-freeze historical research only. Not eligible for genuine observation.
2. **Immutable Exploratory Candidate (`EXPLORATORY_FROZEN_RESEARCH_CANDIDATE`):** Frozen with immutable SHA-256 digests (`bundle`, `feature_spec`, `calibrator`, `threshold`, `manifest_digest`) for **explicitly owner-authorized read-only exploratory prospective observation** only. Does **not** imply demonstrated edge.
3. **Scientifically Qualified Candidate (`SCIENTIFICALLY_QUALIFIED_FROZEN_RESEARCH_CANDIDATE`):** Cleared preserved Stage C / Phase 40C acceptance gates on independent out-of-sample evidence. Still `execution_authority = NONE`.
4. **Execution-Authorized Candidate:** Out of scope for Phase 47; requires separate operational and risk-governance authorization.

## 3. Mandatory Safeguards for Any Exploratory Campaign
- **Artifact Immutability:** Bound to exact model bytes, ordered feature list, scaler, calibrator, threshold, native timeframe, and horizon; zero mutation or retraining during campaign.
- **Genuine Provenance & Pre-Maturity Commitment:** Every prediction must be committed and hashed before outcome maturity on closed native candles (`EURUSD` regular market only, no OTC).
- **Fixed Windows & Owner Authorization:** Bounded supervised sessions only; no automatic reconnects or starts; explicit owner sign-off per campaign. Realistic session counts must follow non-overlapping operational yield estimates in `reports/phase47/CANDIDATE_ELIGIBILITY_ASSESSMENT.md` (e.g., ~18–48 8h supervised sessions for 200 non-overlapping M1_H15/M5_H6 samples).
- **Separation from Confirmatory Evidence:** Exploratory sessions are tagged `EXPLORATORY_PROSPECTIVE_ONLY` and cannot be retroactively relabeled as confirmatory Stage C passes without a fresh preregistered confirmatory campaign.
- **Calibration & Guardrail Integrity:** Any exploratory frozen candidate must record its calibration status (`CALIBRATED_ISOTONIC` / `CALIBRATED_SIGMOID`), full prediction SHA-256 digests, and must enforce minimum sample/day/class/coverage requirements without silent fallback.
