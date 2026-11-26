# Phase 48 — Operator Instructions (Supervised Genuine Exploratory Observation)

**Current State:** Genuine exploratory observation is **BLOCKED** pending explicit owner authorization of `data/phase48/campaigns/phase48_exploratory_contract.json`.

---

## Step-by-Step Procedure

1. **Launch the Dashboard:** Run `Start-ResearchDashboard.cmd` or `python -m iqoption_research.desktop.app`, then open the **Shadow Evaluation / Prospective Observation** tab. *(Available now)*
2. **Verify the Frozen Models:** Click **Refresh Models & Contract** (or run `python -m iqoption_research.cli.main phase48 status`). Confirm all three bundles show `EXPLORATORY_FROZEN` with matching SHA-256 prefixes (`a9d538c80f1a`, `fa908c9f8d07`, `1efedf7b67e4`). *(Available now)*
3. **Review the Proposed Campaign:** Inspect `docs/phase48/CAMPAIGN_AUTHORIZATION_PROPOSAL.md` and `reports/phase48/proposed_campaign_contract.json`. *(Available now)*
4. **Explicitly Authorize a Bounded Campaign:** **[BLOCKED PENDING OWNER AUTHORIZATION]**
   Run the supported CLI authorization command (optionally with `IQOPTION_PHASE48_OWNER_KEY` set for local HMAC-SHA256 signing):
   ```powershell
   python -m iqoption_research.cli.main phase48 authorize `
     --account-id <YOUR_VAULT_ACCOUNT_ID> `
     --stage STAGE_A_15M `
     --owner "<YOUR_NAME>" `
     --hours-valid 24 `
     --note "Stage A 15m connectivity pilot"
   ```
   - `STAGE_A_15M` strictly permits **15-minute** sessions with **no outcome drain** (`target_sessions=1`).
   - `CampaignUsageStore` (`data/phase48/campaigns/phase48_exploratory_eurusd_parallel_v1.usage.json`) atomically tracks consumed sessions to prevent duplicate starts or exceeding stage limits.
   - To revoke at any time: `python -m iqoption_research.cli.main phase48 revoke --owner "<YOUR_NAME>"`.
5. **Select Vault Account:** Select the authorized Windows Credential Vault account in **Vault Account**. *(Blocked for genuine mode until Step 4)*
6. **Select Genuine Exploratory Observation:** Select **Genuine Exploratory (Phase 48)** and click **Refresh Models & Contract**. Verify the status banner transitions from `BLOCKED` to `AUTHORIZED: Phase 48 Supervised Exploratory Campaign`. *(Blocked until Step 4)*
7. **Select Candidate Models & Duration:** Check the desired exploratory candidates (`M1 H15`, `M5 H6`, `M15 H4`) and select the stage-permitted duration (`15` for Stage A).
8. **Start an Authorized Session:** Click **Start Observation** and confirm the supervised read-only authorization dialog. *(Blocked until Step 4)*
9. **Monitor the Models:** Watch per-model Warmup, Opps/Sel/Pending/Resolved KPI cards, the multi-series Cumulative Accuracy chart, and the Live Prediction Stream table (showing `PENDING`, `RESOLVED`, `ABSTAIN`, `actual_dir`, and `outcome`).
10. **Stop Safely:** Click **Stop / Cancel** at any time (or let the bounded timer complete). Unresolved predictions are preserved as `UNRESOLVED_SESSION_ENDED`.
11. **Verify Session Integrity:** Select the session in **Session History** and click **Verify Integrity** (or run `python -m iqoption_research.cli.main phase48 verify-session <SESSION_DIR>`). *(Available now)*
12. **Export and Interpret Diagnostics:** Click **Export Sealed Evidence** and review per-candidate diagnostics separately without aggregating across horizons or claiming statistical significance from small pilots. *(Available now)*

---

## Trust Boundary & Evidence Limitations
- **Local Authorization Trust Boundary:** Without `IQOPTION_PHASE48_OWNER_KEY`, `authorization_signature_sha256` is an unkeyed SHA-256 integrity digest that detects accidental edits or UI bypasses, not a cryptographic proof of owner identity against a host-compromised attacker. With `IQOPTION_PHASE48_OWNER_KEY` set in the owner environment, HMAC-SHA256 binds possession of that local secret.
- **Local Hash-Based Evidence:** `session_manifest.json` and `commitment_digest` detect post-hoc file tampering on disk, but they rely on local host clock/filesystem integrity and cannot substitute for an external third-party timestamping authority.
