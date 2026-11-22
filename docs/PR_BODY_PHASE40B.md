# PR: Phase 40B — Prospective AI Shadow Observation & Live Readiness Gate

## Overview

Stacked on `codex/model-failure-intelligence` (PR #54), this PR integrates Market Collector V2, the frozen candidate model registry, the immutable two-phase prediction journal, and Phase 41 decision/evaluation rules into a single supervised read-only prospective observation workflow for EURUSD M1 candidate models.

It resolves all prospective-time integrity and session-lifecycle readiness risks prior to supervised live broker acceptance:
- Enforces strict real prospective commitment deadlines ($t_{\text{deadline}} = t_{\text{maturity}} - \text{margin} - \text{clock\_uncertainty}$).
- Provides verified live warmup history acquisition while preventing retroactive predictions.
- Implements market-data gap detection, invalidation of missing maturity evidence, and safe warmup rebuilding.
- Implements cryptographic session sealing (`predictions_sha256`, `outcomes_sha256`, `session_json_sha256`, `manifest_sha256`) and anti-overwrite protection.
- Hardens synthetic vs. prospective mode isolation.
- Enriches dashboard acceptance diagnostics and adds `shadow verify-session` CLI verification.

All functionality has been developed, tested, and verified offline first using deterministic tests and a synthetic streaming harness without broker connection or live order routing.

## Key Changes

### 1. Streaming Infrastructure (`src/iqoption_research/shadow/streaming.py`)
- `StreamingCandleEvent`: Strongly typed container capturing candle payload, UTC receipt timestamps, latency, clock diagnostics, and cryptographic provenance.
- `MarketStreamProtocol`: Clean async protocol for streaming market observations (`connect`, `disconnect`, `get_next_event`, `get_warmup_candles`, `is_connected`, `is_synthetic`).
- `SyntheticMarketStream`: Deterministic offline streaming harness with configurable latency, clock uncertainties, and step arrivals for offline unit testing and dry runs.
- `LiveCollectorStream`: Supervised adapter over `ExtendedCollection` and `supervised_client` for operator-controlled foreground observation sessions with Windows Vault account selection and verified pre-session regular M1 candle warmup.

### 2. Contract Economics & Delayed Evaluator (`src/iqoption_research/shadow/evaluator.py`)
- Distinguishes generic research directional accuracy from financial contract mechanics:
  - **Binary Option**: At-The-Money (ATM) ties refund principal without profit or loss ($\text{PnL} = 0.0$), $W_{\text{be, decisive}} = 1 / (1 + P)$, $W_{\text{be, all}} = (1 - T_{\text{rate}}) / (1 + P)$.
  - **Digital Option**: At-The-Money (ATM) ties lose principal ($\text{PnL} = -1.0$), $W_{\text{be, digital}} = 1 / (1 + P)$.
- Missing settlement evidence due to market gaps resolved as `INVALID_DISCONTINUITY` with `MISSING_SETTLEMENT_EVIDENCE`.
- Added `expectancy_binary_pct`, `expectancy_digital_pct`, and `break_even_win_rate_digital` to `EvaluationSummary`.
- Extended `compute_summary` to support multi-model filtering by `model_id` and zero-prediction safety (`INSUFFICIENT_EVALUATED_OBSERVATIONS_NO_VERDICT`).
- Mandatory scientific disclaimer: *"CANDLE-CLOSE COMPARISONS ARE HYPOTHETICAL RESEARCH BENCHMARKS AND NEVER OBSERVED BROKER ORDER SETTLEMENTS. NO_DEMONSTRATED_EDGE."*

### 3. Prospective Shadow Observer & Sealing (`src/iqoption_research/shadow/observer.py`)
- `ProspectiveShadowObserver`: Core engine coordinating real-time prospective observation.
- Real prospective commitment deadline enforcement:
  $$t_{\text{deadline}} = t_{\text{maturity}} - \text{margin} - (\text{clock\_uncertainty\_ms} / 1000.0)$$
  Late observations and late commitments are rejected from emitting predictions, retaining candles for lookback and delayed outcome evaluation.
- Clock disagreement ($t_{\text{recv}} < t_{\text{cutoff}}$) and excessive clock uncertainty ($>5000\text{ms}$) explicitly rejected.
- Market gap recovery FSM: `OBSERVING` -> `GAP_DETECTED` -> `REBUILDING_WARMUP` -> `OBSERVING`.
- Anti-overwrite protection: Fails closed (`ValueError`) if target session directory already contains `session.json` or `manifest.json`.
- Complete cryptographic sealing: Manifest seals `predictions_sha256`, `outcomes_sha256`, `session_json_sha256`, `models_sha256`, and its own canonical `manifest_sha256`.
- `verify_session_integrity(session_dir)`: Standalone programmatic and CLI verification function.

### 4. Desktop Research Workstation Panel (`src/iqoption_research/desktop/shadow_panel.py`)
- Added 4th subtab "Prospective Observation" to Shadow Evaluation panel.
- Operator controls: candidate model dropdown, mode radio buttons (Offline Simulator vs Supervised Live), Windows Vault account selector, bounded duration selector (15, 30, 60, 120 min), and Start/Stop buttons.
- Real-time streaming metrics table: Total Preds, Pending, Trades (W/L/T), Decisive Win%, Matched Maj%, Wilson 95% CI, Exp Binary, Exp Digital, BE Gap. Shows `N/A` when zero active trades exist.
- Session Health & Continuity status bar: FSM state, warmup status, accepted/late opportunities, genuine/synthetic counts, commitment latency.
- Visible zero-order execution safety banner: `EXECUTION_AUTHORITY: NONE · NO DEMONSTRATED EDGE · STRICTLY READ-ONLY`.

### 5. CLI Command & Session Process (`src/iqoption_research/cli/commands/shadow_cmd.py`, `src/iqoption_research/desktop/process.py`)
- Added `shadow observe` command supporting `--synthetic` (offline), `--live` (supervised broker), bounded `--minutes`, and multi-model `--model-id`.
- Added `shadow verify-session <SESSION_DIR_OR_ID>` command.
- Added `start_prospective_shadow` to desktop session process runner.

### 6. Comprehensive Test Suite (`tests/unit/test_phase40b_prospective_shadow.py`)
- 16 comprehensive unit tests verifying:
  1. Streaming candle event serialization roundtrip.
  2. Synthetic stream sequential playback and connection handling.
  3. Binary vs. Digital contract economics and break-even math.
  4. End-to-end prospective observation with delayed outcome resolution and point-in-time pre-outcome commitment.
  5. Evidence rejection (incomplete, discontinuous, duplicate).
  6. Operator cancellation and sealed manifest publication.
  7. Pre-flight candidate integrity and tamper rejection (`execution_authority = NONE`).
  8. Desktop panel prospective tab initialization and safety labels.
  9. Real prospective commitment deadline enforcement across arrival variations (shortly after close, near deadline, past deadline, at maturity, after maturity, excessive clock uncertainty, clock disagreement).
  10. Pre-session live warmup acquisition from stream (zero retroactive predictions).
  11. Insufficient warmup live accumulation without pre-history.
  12. Market gap detection, pending outcome invalidation (`INVALID_DISCONTINUITY`), and safe post-gap warmup rebuilding.
  13. Duplicate session ID anti-overwrite rejection.
  14. Manifest cryptographic sealing and multi-point tamper detection.
  15. Mode isolation between synthetic and live configurations.
  16. CLI `shadow verify-session` verification command.

---

## Invariant Verification

- [x] `execution_authority = NONE` enforced across all engines and manifests.
- [x] Zero trade execution, order routing, or wallet mutation capability.
- [x] Candle-close comparisons explicitly disclaimed as hypothetical research benchmarks.
- [x] All 16 Phase 40B unit tests pass.
- [x] All 23 Phase 40A shadow evaluation tests pass.
- [x] Full repository offline test suite (2,044+ tests) passes.
- [x] MyPy: 0 errors across 251 source files.
- [x] Ruff check: All checks passed.
- [x] Ruff format: Clean across all 399 repository files.
- [x] CLI Security Check: All 7 checks passed.
