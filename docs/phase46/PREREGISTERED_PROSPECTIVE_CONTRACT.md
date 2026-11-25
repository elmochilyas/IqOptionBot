# Phase 46 — Preregistered Prospective Observation Contract (Draft, Pre-Activation)

**Status:** `DRAFT / UNAUTHORIZED` (must be finalized and explicitly authorized by the project owner before any genuine prospective observations may be recorded or counted under this contract).  
**Governance:** Preserve Phase 40C and Stage C acceptance contracts. Any modification to those contracts, eligibility gates, or activation status requires **explicit written justification and owner authorization** prior to activation.  
**Hard Constraints (Invariants):**
- `execution_authority = NONE`
- `research_disposition = NO_DEMONSTRATED_EDGE`
- Zero broker connections during development or automated testing.
- Zero real orders, zero demo orders, zero wallet mutations.
- No unauthorized prospective activation, no unattended broker sessions, no automatic reconnects, no automatic session starts.
- No automatic model promotion or retraining during active evaluation.
- No fabricated genuine evidence; synthetic fixtures must never be classified as genuine market evidence.

---

## 1. Eligible Immutable Candidate IDs
Only immutable frozen artifacts registered under `src/iqoption_research/ml/freezer.py` and listed below after WS2 qualification may participate in genuine prospective observation:

| Immutable Model ID | Classification Label | Native Timeframe | Horizon | Feature Spec & Digest | Threshold & Calibration | Artifact SHA-256 | Status |
|---|---|---|---|---|---|---|---|
| *(None qualified in Phase 46 WS2)* | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `BLOCKED` |

- **Required Classification Labels:** Every future entry must carry either `EXPLORATORY_FROZEN_RESEARCH_CANDIDATE` or `SCIENTIFICALLY_QUALIFIED_FROZEN_RESEARCH_CANDIDATE`. **Neither label grants execution authority.**
- **Model Immutability:** Frozen models are bound to exact bytes, parameters, ordered feature specifications, training dataset manifest/content digest, preprocessing/calibration, native timeframe/horizon, decision timing, confidence/abstention thresholds, code/config versions, and SHA-256 digests. Models must never be mutated during evaluation; any revision requires a new immutable ID.

## 2. Owner Authorization & Activation Gate
- Genuine prospective observation remains **strictly blocked** until:
  1. At least one eligible immutable candidate is qualified and registered in the frozen registry.
  2. This contract transitions from `DRAFT` to `FINALIZED` with fixed stopping rules and sample targets locked.
  3. Explicit owner authorization is recorded for the specific campaign, account, and timeframe.
- Selecting a model or saved account in the desktop UI must **not** activate genuine prospective observation.

## 3. Universe & Market Scope
- **Asset:** `EURUSD` regular market only.
- **OTC Substitution:** Strictly prohibited. Regular market hours only.

## 4. Permitted Observation Windows & Session-Duration Constraints
- **Supervision:** Supervised sessions only (operator-initiated Start, manual Stop/Cancel, no unattended operation).
- **Connection Policy:** Read-only market data stream only. No automatic reconnect on disconnect; no automatic session restarts.
- **Session Duration:** Bounded by the reviewed authorization contract (e.g., 15–120 minutes per supervised session in the desktop Observation Center; never extended automatically).

## 5. Minimum Evidence Thresholds (Preserved, Not Weakened)
- Minimum observation days and minimum resolved non-overlapping predictions must satisfy preserved Phase 40C and Stage C scientific acceptance contracts.
- **No Gate Weakening:** Thresholds must never be lowered because candidate coverage is low or collection takes many calendar days (see `reports/phase46/duration_estimate.md`).
- **Exposure & Prospective Boundaries:**
  - Protected historical boundary: `2026-11-12T10:00:00Z` (`1794477600`).
  - No historical or previously exposed data may be reused as independent prospective evidence.
  - Only predictions persisted **before** outcome maturity on fully closed native candles after contract activation count as prospective evidence.

## 6. Coverage & Abstention Requirements
- Record total eligible closed-candle opportunities, selected decisive predictions (`UP` / `DOWN`), and abstentions (`ABSTAIN` / `SKIP`) per model and session.
- Coverage = `selected_predictions / total_eligible_closed_candle_opportunities`.
- Abstention reasons must be logged per decision.

## 7. Outcome Definitions & Tie Handling
- **Native Maturity:** Outcomes resolve strictly at `decision_cutoff_ts + (horizon_bars * timeframe_seconds)`:
  - `M1_H1`: 60 seconds
  - `M5_H3`: 900 seconds
  - `M15_H2`: 1,800 seconds
- **Outcome Semantics:** Aligned with `shadow/observer.py` and `ml/targets/specs.py`:
  - `UP` wins if `maturity_close > entry_close`; `DOWN` wins if `maturity_close < entry_close`.
  - **Tie Handling:** Exact equality (`maturity_close == entry_close`) is recorded explicitly as `TIE` (excluded from decisive win-rate denominator `wins / (wins + losses)`, reported separately in total counts).

## 8. Matched Baselines
- Every candidate must be compared against its training-fitted / matched majority baseline on **identical decision timestamps and eligibility rules**, consistent with WS1 methodology.

## 9. Statistical Methodology & Multiple-Comparison Handling
- **Dependence-Aware Inference:** Overlapping predictions are temporally dependent. Primary uncertainty quantification uses UTC day-block bootstrap (10,000 replicates) and 2-day-block sensitivity, alongside non-overlapping sample counts.
- **Multiple-Comparison Handling:** Any multi-candidate or multi-horizon confirmatory family must apply preregistered family-wise error control (e.g., Holm-Bonferroni) across all preregistered hypotheses.
- **Hypothetical Payout Sensitivity:** Any break-even comparison (e.g., 54.05% at 85% payout) must be labeled as hypothetical sensitivity, never actual broker settlement.

## 10. Fixed Stopping Rules & Acceptance Rules
- **Stopping Rules:** Sample targets and maximum campaign horizon must be fixed prior to campaign start. Peeking-driven early stopping or post-hoc threshold tuning invalidates confirmatory status.
- **Acceptance Rules:** Governed by preserved Phase 40C and Stage C contracts (statistically significant positive delta over matched baseline and break-even under dependence-aware CIs, calibration integrity, and zero data-integrity violations).

## 11. Missing-Data Treatment & Interrupted Session Handling
- **Missing Data:** Zero interpolation or backfilling of missing candles. Gaps must be recorded as continuity gaps and excluded from favorable evidence.
- **Interrupted Sessions:** On operator cancellation, disconnect, or crash, all already-committed predictions and resolved outcomes are preserved and sealed as an `INTERRUPTED` / `CANCELLED` session. Unresolved pending predictions remain marked `PENDING` / `UNRESOLVED` and are never fabricated.

## 12. Evidence Verification & Sealing
- Every session must persist `session.json`, `inputs.jsonl`, `predictions.jsonl` (written pre-maturity), `outcomes.jsonl`, `diagnostics.jsonl`, and `session_manifest.json` with SHA-256 digests.
- Independent verification via `verify_session_integrity()` and `verify_session_inputs()` must pass before any session is eligible for diagnostic aggregation.
