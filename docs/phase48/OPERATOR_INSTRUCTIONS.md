# Phase 48 — Operator Instructions (Supervised Genuine Exploratory Observation)

**Current State:** Genuine exploratory observation is **BLOCKED** pending explicit owner authorization of `data/phase48/campaigns/phase48_exploratory_contract.json`.

---

## Step-by-Step Procedure

1. **Launch the Dashboard:** Run `Start-ResearchDashboard.cmd` or `python -m iqoption_research.desktop.app`, then open the **Shadow Evaluation / Observation Center** tab. *(Available now)*
2. **Verify the Frozen Models:** Click **Refresh Models & Contract** in the Phase 48 section. Confirm all three bundles show `EXPLORATORY_FROZEN` with matching SHA-256 prefixes (`a9d538c80f`, `fa908c9f8d`, `1efedf7b67`). *(Available now)*
3. **Review the Proposed Campaign:** Inspect `docs/phase48/CAMPAIGN_AUTHORIZATION_PROPOSAL.md` and `reports/phase48/proposed_campaign_contract.json`. *(Available now)*
4. **Explicitly Authorize a Bounded Campaign:** **[BLOCKED PENDING OWNER AUTHORIZATION]** After owner review, run `authorize_campaign_contract(...)` to write an owner-signed `AUTHORIZED_ACTIVE` contract to `data/phase48/campaigns/phase48_exploratory_contract.json` bound to the exact vault account reference and stage (`STAGE_A_15M`..`STAGE_D_480M`).
5. **Authenticate with a Vault Account:** Select the authorized Windows Credential Vault account in **Vault Account**. *(Blocked for genuine mode until Step 4)*
6. **Select Genuine Exploratory Observation:** Select **Genuine Exploratory (Phase 48)**. Verify the status banner transitions from `BLOCKED` to `AUTHORIZED: Phase 48 Supervised Exploratory Campaign`. *(Blocked until Step 4)*
7. **Select Candidate Models & Duration:** Check the desired exploratory candidates (`M1 H15`, `M5 H6`, `M15 H4`) and select an authorized duration (`15`, `30`, `60`, `120`, `240`, or `480` min). *(Available for offline synthetic demo; blocked for genuine mode until Step 4)*
8. **Start an Authorized Session:** Click **Start Observation**. *(Blocked until Step 4)*
9. **Monitor the Models:** Watch per-model Warmup, Opps/Sel/Pending/Resolved KPI cards, the multi-series Cumulative Accuracy chart, and the filterable Live Prediction Stream table.
10. **Stop Safely:** Click **Stop / Cancel** at any time (or let the bounded timer complete). Unresolved predictions are preserved as `UNRESOLVED_SESSION_ENDED`.
11. **Verify Session Integrity:** Select the session in **Session History** and click **Verify Integrity** to run `verify_phase48_session_integrity()`. *(Available now on offline synthetic sessions)*
12. **Export and Interpret Diagnostics:** Click **Export Sealed Evidence** and review per-candidate diagnostics separately without aggregating across horizons or claiming statistical significance from small pilots. *(Available now on offline synthetic sessions)*
