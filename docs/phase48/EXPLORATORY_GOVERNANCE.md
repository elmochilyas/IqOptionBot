# Phase 48 — Exploratory Research Governance Implementation Report

**Status:** `IMPLEMENTED / INACTIVE / FAIL-CLOSED`  
**Disposition:** `NO_DEMONSTRATED_EDGE` | **Execution Authority:** `NONE` | **Evidence Classification:** `EXPLORATORY_PROSPECTIVE_ONLY`

---

## 1. Four-Tier Research Pathway
Implemented in `src/iqoption_research/phase48/governance.py` (`ResearchPathwayTier`), separating exploratory research from confirmatory qualification and execution:

1. `DEVELOPMENT_RESEARCH_CANDIDATE`: Historical development experiments (`Phase 47`). Cannot participate in genuine prospective observation.
2. `EXPLORATORY_FROZEN_RESEARCH_CANDIDATE`: Immutable frozen research artifact with SHA-256 component/bundle/manifest digests (`src/iqoption_research/phase48/artifacts.py`). Eligible **only** for owner-authorized read-only exploratory prospective observation. Carries `NO_DEMONSTRATED_EDGE` and `execution_authority = NONE`.
3. `SCIENTIFICALLY_QUALIFIED_FROZEN_RESEARCH_CANDIDATE`: Preserved Stage C / Phase 40C qualification tier. Currently **0 candidates** hold this status.
4. `EXECUTION_AUTHORIZED_CANDIDATE`: Out of scope; `execution_authority` remains strictly `NONE`.

## 2. Fail-Closed Authorization Mechanism
- Genuine exploratory observation is **blocked by default** (`CampaignAuthorizationStatus.PROPOSED_NOT_AUTHORIZED`).
- `validate_campaign_authorization()` enforces:
  - `execution_authority == "NONE"` and `evidence_classification == "EXPLORATORY_PROSPECTIVE_ONLY"`.
  - `asset == "EURUSD"` and `allow_otc is False`.
  - Valid `contract_sha256` matching the canonical contract specification.
  - `authorization_status == "AUTHORIZED_ACTIVE"`, non-empty `authorized_by`, `authorized_at_utc`, and valid `authorization_signature_sha256`.
  - Exact match between selected candidate artifact digests and `contract.candidate_artifact_digests`.
  - Requested session duration in `(15, 30, 60, 120, 240, 480)` and `<= contract.max_session_duration_minutes`.
  - Selected vault account matching `contract.allowed_account_ref`.
- **Dashboard isolation:** No UI button can authorize or activate a genuine campaign. Owner sign-off requires explicit offline review and `authorize_campaign_contract(...)` written to `data/phase48/campaigns/phase48_exploratory_contract.json`.
- **Preserved Confirmatory Gates:** Stage C (`src/iqoption_research/ml/freezer.py`), Phase 40C (`src/iqoption_research/campaign/`), and Phase 46 (`docs/phase46/PREREGISTERED_PROSPECTIVE_CONTRACT.md`) remain completely unmodified and intact.
