# Phase 48.1 — Genuine Observation Readiness Hotfix Report

**Disposition:** `NO_DEMONSTRATED_EDGE` | **Execution Authority:** `NONE` | **Evidence Classification:** `EXPLORATORY_PROSPECTIVE_ONLY`

---

## 1. Fixes Implemented
1. **Dashboard Genuine Dispatch (`shadow_panel.py`, `observation_center.py`):**
   - `ShadowEvaluationPanel.start_observation()` explicitly routes `mode == "exploratory_genuine"` to `ObservationCenterController.start_genuine_exploratory_observation()`.
   - Resolves canonical vault `account_id`, verifies frozen bundles, reloads & validates the campaign contract (stage, duration, drain, expiration, usage), prompts for operator confirmation, loads credentials from `WindowsCredentialVault`, constructs `IQOptionReadOnlyMultiTransport`, and executes `ParallelObservationRuntime` in a background thread.
   - Live prediction table renders resolved outcomes (`RESOLVED`, `actual_direction`, `classification`).
2. **Genuine Candle Provenance (`transport.py`):**
   - Removed unconditional `server_time_fresh=True` and fabricated streaming `request_id`s.
   - Enforces verified `ws.is_server_time_fresh`, `ws.server_time_ns`, `clock_diagnostics.sync_status`, candle alignment (`from_ts % tf == 0`, `to_ts == from_ts + tf`), closed-candle boundary (`to_ts <= server_sec`), data age, practice-account verification, contiguous warmup validation, and per-timeframe heartbeats.
3. **Campaign Authorization & Accounting (`governance.py`, `cli.py`):**
   - Enforces `authorized_stage` exact duration (Stage A = 15m only, drain prohibited), `expires_at_utc`, revocation, and atomic `CampaignUsageStore` session reservation.
   - Added supported CLI commands: `phase48 status`, `phase48 authorize`, `phase48 revoke`, `phase48 verify-session`.
   - Documented local unkeyed SHA-256 vs `IQOPTION_PHASE48_OWNER_KEY` HMAC-SHA256 trust boundary.
4. **Prospective Timing & Commitment Verification (`runtime.py`, `evidence.py`):**
   - Corrected `Candle.timestamp_utc` (`from_ts`) vs `to_ts` semantics to match Phase 47's `Phase47FeaturePipeline` (`timestamps = [c.to_ts]`).
   - Distinguishes `candle_open_ts`, `candle_close_ts`, `feature_cutoff_ts`, `predicted_at_utc`, `committed_at_ts`, and `maturity_timestamp = candle_close_ts + horizon_bars * tf`.
   - Rejects late events (`verified_commit_ts >= maturity_timestamp`), rechecks decision-window deadline after awaiting events, and extends `verify_phase48_session_integrity()` to recompute outcome directions/classifications and verify manifest completeness.

## 2. Preserved Frozen Models
All three frozen experimental bundles under `data/phase48/models/` remain untouched with their exact original SHA-256 digests (`a9d538c80f1a...`, `fa908c9f8d07...`, `1efedf7b67e4...`).

## 3. Remaining Genuine Compatibility Limitations
- Live IQ Option WebSocket multi-timeframe streaming (`M1`, `M5`, `M15` simultaneously on one `AsyncIQOption` session) has not been executed against the live broker; Stage A (15m supervised pilot) is the first live compatibility check once the owner authorizes the contract.
