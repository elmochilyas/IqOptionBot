# Phase 46 — Preregistered Prospective Observation Contract (Draft, pre-activation)

**Status:** DRAFT (must be finalized and explicitly authorized before any prospective observations count under this contract).  
**Governance:** Preserve Phase 40C and Stage C acceptance contracts. Any change to those contracts or to eligibility/activation requires **explicit justification + owner authorization** before activation.  
**Hard constraints (always):** `execution_authority = NONE`, `NO_DEMONSTRATED_EDGE`. No broker connection during implementation/automated testing. No real/demo orders. No wallet mutations. No unauthorized prospective activation. No fabricated genuine evidence. No auto model promotion/retraining during active eval. No weakening of existing scientific acceptance requirements.

## 1) Allowed frozen model IDs (immutable)
Only models listed here, tied to immutable artifact references in `reports/phase46/` (and/or `data/phase46/` local artifacts) after WS2 completion:
- (to be filled from WS2 frozen artifacts; must include immutable IDs, feature preset/hash, timeframe/horizon, checksums, labels)

Labels (required): each entry must be labeled **Exploratory frozen research candidate** OR **Scientifically qualified frozen research candidate**. **Neither grants execution authority.**

## 2) Universe / market
- **EURUSD regular market only**.

## 3) Observation windows
- Supervised sessions only (no unattended). No automatic reconnects, no automatic session starts.
- Authorized session duration governed by a **reviewed auth contract** (do not loosen existing limits without explicit authorized replacement).
- Only observations recorded **after**: (a) immutable freeze artifacts exist, (b) this contract is finalized, (c) owner authorization is recorded for prospective activation (if required) count as prospective under Phase 46.

## 4) Minimum evidence thresholds (do not weaken)
- Minimum observation days and minimum resolved predictions: **to be set from WS1/WS3 duration estimate and existing acceptance gates**; must not be reduced because coverage is low.
- Outcomes resolved only from genuine evidence at exact maturity; no interpolation of missing data.
- Protected/historically reused periods are **not** reusable as independent evidence.

## 5) Coverage requirements
- Record coverage (selected/predicted vs eligible windows) per model and session.
- Gaps/missing market data: sessions must record gaps; missing data does **not** count as favorable evidence and must not be imputed.

## 6) Matched baselines
- Compare each frozen model against training-fitted/matched baselines on **identical decision timestamps and eligibility rules** (as defined in WS1 methodology).

## 7) Outcomes and ties
- Align with existing shadow/observer/outcome resolution semantics (`shadow/observer.py`, `research/prospective.py`). Tie handling must match preregistered training/eval definitions.

## 8) Statistics
- Dependence-aware inference (e.g. day-block/bootstrap conventions consistent with WS1) and explicit multiple-comparison handling policy.
- Report CIs and effect measures; no independence assumptions presented as valid.

## 9) Stopping / acceptance (high level)
- Stopping rules must be preregistered; no peeking-driven strategy changes that invalidate independence claims.
- Acceptance remains bound to preserved Phase 40C/Stage C contracts.

## 10) Interruptions / missing data
- On interruption: preserve sealed partial evidence; do not backfill. Mark session incomplete.

## 11) Duration estimate (WS3)
See `reports/phase46/duration_estimate.md` (generated from actual Phase 45 coverage/frequencies and candidate horizons). 

**Reminder:** This document is not permission to activate prospective observation until owner authorization and finalization steps are complete.