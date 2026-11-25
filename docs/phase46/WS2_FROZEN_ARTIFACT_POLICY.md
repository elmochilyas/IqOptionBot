# Phase 46 WS2 — Frozen Artifact Policy (immutable)

## Rules
- Freeze **only if** qualification gates are met (reuse/align with `src/iqoption_research/ml/freezer.py` and existing governance tests). **Never weaken** criteria.
- **Never modify** prior frozen candidates/artifacts.
- Artifacts are **immutable** once written (new version = new ID/path).
- Every frozen artifact must carry labels:
  - `candidate_class`: one of `DEVELOPMENT_RESEARCH_CANDIDATE` (not frozen), `EXPLORATORY_FROZEN_RESEARCH_CANDIDATE`, `SCIENTIFICALLY_QUALIFIED_FROZEN_RESEARCH_CANDIDATE`
  - **Neither `EXPLORATORY_FROZEN_RESEARCH_CANDIDATE` nor `SCIENTIFICALLY_QUALIFIED_FROZEN_RESEARCH_CANDIDATE` grants execution authority.**
  - `execution_authority` is always `NONE` for Phase 46 frozen research artifacts.
- If **no candidates qualify**, publish an explicit “none qualified” report and do **not** create weak/soft frozen artifacts.

## Minimum artifact contents (to specify in implementation)
- model payload/reference, feature preset + exact feature list/hash, timeframe/horizon, training/boundary metadata, calibration/decision policy if used, code/config/version pin, created_at, checksums, selection rationale summary (sanitized), and link to WS1 evidence.

## Prospective activation
- Prospective observation using frozen artifacts requires finalized preregistered contract + owner authorization (see `PREREGISTERED_PROSPECTIVE_CONTRACT.md`).