# Phase 48 — Exploratory Campaign Authorization Proposal

**Status:** `PROPOSED_NOT_AUTHORIZED` (Inactive, fail-closed; requires explicit owner review and sign-off)  
**Contract Artifact:** `reports/phase48/proposed_campaign_contract.json`  
**Evidence Classification:** `EXPLORATORY_PROSPECTIVE_ONLY` | **Execution Authority:** `NONE` | **Disposition:** `NO_DEMONSTRATED_EDGE`

---

## 1. Frozen Experimental Candidates
| Slot | Frozen Candidate ID | Native TF | Horizon | Frozen Threshold (`tau`) | Bundle SHA-256 (prefix) |
|---|---|---:|---:|---:|---|
| A | `p48_exp_cand_a_m1_h15_hgb_v1` | M1 (60s) | H15 (15m) | `0.515474` | `a9d538c80f1a` |
| B | `p48_exp_cand_b_m5_h6_hgb_v1` | M5 (300s) | H6 (30m) | `0.549763` | `fa908c9f8d07` |
| C | `p48_exp_cand_c_m15_h4_hgb_v1` | M15 (900s) | H4 (60m) | `0.564935` | `1efedf7b67e4` |

## 2. Proposed Staged Pilot Progression
- **Stage A (`STAGE_A_15M`):** 15-minute supervised connectivity and lifecycle pilot (no outcome drain).
- **Stage B (`STAGE_B_120M`):** 120-minute supervised multi-model pilot allowing full 15m/30m/60m outcome maturity (optional bounded drain up to 60m).
- **Stage C (`STAGE_C_240M`):** 240-minute supervised operational reliability session (optional bounded drain up to 60m).
- **Stage D (`STAGE_D_480M`):** 480-minute supervised exploratory sessions accumulated toward fixed exploratory sample goals (200 non-overlapping predictions per candidate).

## 3. Planning Yield Estimates & Scientific Limitations
- Phase 47 historical effective non-overlapping yield estimates (0.75x operational factor):
  - **M1 H15:** ~1.415 / hr
  - **M5 H6:** ~0.581 / hr
  - **M15 H4:** ~0.487 / hr
- Actual prospective rates are unknown.
- **No optional stopping:** Peeking-driven early stopping based on favorable interim accuracy is prohibited.
- **No scientific qualification from pilots:** Operational pilots cannot establish statistical significance or promote candidates.
