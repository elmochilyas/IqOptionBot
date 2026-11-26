# Phase 48 — Read-Only Transport, Multi-Model Runtime & Commitment Contract

**Disposition:** `NO_DEMONSTRATED_EDGE` | **Execution Authority:** `NONE`

---

## 1. Shared Read-Only Market Transport (`src/iqoption_research/phase48/transport.py`)
- Reuses the existing verified read-only `AsyncIQOption` connector (`src/iqoption_research/connectors/iqoption/client.py`) with `AsyncIQOption.assert_read_only()` enforced.
- Verified multi-timeframe capability: `IQOptionWebSocketClient.subscribe_candles(active_id, size)` keys subscriptions by `(MSG_CANDLE_GENERATED, active_id, size)`, enabling one supervised WebSocket session to stream native `EURUSD` `M1 (60s)`, `M5 (300s)`, and `M15 (900s)` candles simultaneously.
- **No order, balance-switch, or wallet-mutation messages** are ever sent.
- **No automatic reconnects** (`max_reconnect_attempts=0`). Unexpected disconnects transition the session to `INTERRUPTED_DISCONNECT` and seal all collected evidence.
- **Closed-candle & provenance enforcement:** Only `candle.is_complete is True` candles with validated `server_time_fresh` and non-empty `source_provenance` are accepted; synthetic/resampled substitution is prohibited.

## 2. Parallel Multi-Model Observation Runtime (`src/iqoption_research/phase48/runtime.py`)
- Each selected candidate maintains an independent `_CandidateState`:
  - Bounded native candle `deque(maxlen=256)` and independent `MIN_WARMUP_BARS_P47 = 60` warmup tracking.
  - Independent `Phase47FeaturePipeline(preset="p47_full_native")` computation.
  - Independent `StandardScaler`, `HistGradientBoostingModel`, `ProbabilityCalibrator(ISOTONIC)`, and frozen confidence threshold `tau`.
  - Independent bounded pending outcome queue (`max_pending_per_candidate=512`), separate `accuracy_series`, and separate `CandidateSessionDiagnostics` (`src/iqoption_research/phase48/diagnostics.py`).
- **Failure isolation:** A feature/inference failure in one candidate disables only that candidate (`model_failures += 1`); shared transport or provenance failures halt the entire session fail-closed.
- **No cross-horizon aggregation:** M1 H15 (15m), M5 H6 (30m), and M15 H4 (60m) metrics are never merged.

## 3. Pre-Maturity Prediction Commitment & Evidence Ledger (`src/iqoption_research/phase48/evidence.py`)
For every eligible closed candle boundary:
1. Validate closed candle & provenance; reject duplicates (`ts <= last_candle_ts`) and record gaps (`ts - last_candle_ts > tf_sec`).
2. Compute causal features at `feature_cutoff_ts == decision_timestamp` and hash `feature_digest`.
3. Generate calibrated `prob_up` / `prob_down` and apply frozen `confidence_threshold` (`UP` / `DOWN` / `ABSTAIN`).
4. Seal `PredictionCommitmentRecord` with SHA-256 `commitment_digest` and append to `predictions.jsonl` **before** outcome maturity.
5. On arrival of the genuine closed candle at `maturity_timestamp = decision_timestamp + horizon_bars * timeframe_seconds`, resolve `UP` / `DOWN` / `TIE` and append `OutcomeResolutionRecord` to `outcomes.jsonl`.
6. Unresolved predictions at session end or maturity gap are explicitly recorded as `UNRESOLVED_SESSION_ENDED` or `UNRESOLVED_GAP_AT_MATURITY` — never fabricated.
7. `session_manifest.json` seals SHA-256 hashes of all session files; `verify_phase48_session_integrity()` and `replay_phase48_session()` verify all hashes and commitment digests independently.
