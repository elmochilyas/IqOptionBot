# PR: Phase 40B — Prospective AI Shadow Observation

## Overview

Stacked on `codex/model-failure-intelligence` (PR #54), this PR integrates Market Collector V2, the frozen candidate model registry, the immutable two-phase prediction journal, and Phase 41 decision/evaluation rules into a single supervised read-only prospective observation workflow for EURUSD M1 candidate models.

All functionality has been developed, tested, and verified offline first using deterministic tests and a synthetic streaming harness without broker connection or live order routing.

## Key Changes

### 1. Streaming Infrastructure (`src/iqoption_research/shadow/streaming.py`)
- `StreamingCandleEvent`: Strongly typed container capturing candle payload, UTC receipt timestamps, latency, clock diagnostics, and cryptographic provenance.
- `MarketStreamProtocol`: Clean async protocol for streaming market observations (`connect`, `disconnect`, `get_next_event`, `is_connected`).
- `SyntheticMarketStream`: Deterministic offline streaming harness with configurable latency simulation for offline unit testing and dry runs.
- `LiveCollectorStream`: Supervised adapter over `ExtendedCollection` and `supervised_client` for operator-controlled foreground observation sessions with Windows Vault account selection.

### 2. Contract Economics & Delayed Evaluator (`src/iqoption_research/shadow/evaluator.py`)
- Distinguishes generic research directional accuracy from financial contract mechanics:
  - **Binary Option**: At-The-Money (ATM) ties refund principal without profit or loss ($\text{PnL} = 0.0$), $W_{\text{be, decisive}} = 1 / (1 + P)$, $W_{\text{be, all}} = (1 - T_{\text{rate}}) / (1 + P)$.
  - **Digital Option**: At-The-Money (ATM) ties lose principal ($\text{PnL} = -1.0$), $W_{\text{be, digital}} = 1 / (1 + P)$.
- Added `expectancy_binary_pct`, `expectancy_digital_pct`, and `break_even_win_rate_digital` to `EvaluationSummary`.
- Extended `compute_summary` to support multi-model filtering by `model_id`.
- Mandatory scientific disclaimer: *"CANDLE-CLOSE COMPARISONS ARE HYPOTHETICAL RESEARCH BENCHMARKS AND NEVER OBSERVED BROKER ORDER SETTLEMENTS. NO_DEMONSTRATED_EDGE."*

### 3. Prospective Shadow Observer (`src/iqoption_research/shadow/observer.py`)
- `ProspectiveShadowObserver`: Core engine coordinating real-time prospective observation.
- Strict two-phase point-in-time invariant:
  1. Resolves pending matured predictions upon receipt of completed candle.
  2. Executes frozen models on lookback history $\le t_{\text{cutoff}}$.
  3. Commits prediction record to `predictions.jsonl` with `os.fsync` before any future candle can be processed.
- Evidence rejection: Fails closed on stale data ($> 300\text{s}$), incomplete candles, non-EURUSD / non-60s data, duplicate boundaries, and temporal continuity gaps.
- Multi-model isolation: Concurrently evaluates multiple candidate models with matched baseline tracking.
- Session lifecycle: Writes initial `session.json`, publishes real-time atomic `state.json`, and seals completed/cancelled sessions with immutable `manifest.json`.

### 4. Desktop Research Workstation Panel (`src/iqoption_research/desktop/shadow_panel.py`)
- Added 4th subtab "Prospective Observation" to Shadow Evaluation panel.
- Operator controls: candidate model dropdown, mode radio buttons (Offline Simulator vs Supervised Live), Windows Vault account selector, bounded duration selector (15, 30, 60, 120 min), and Start/Stop buttons.
- Real-time streaming metrics table: Total Preds, Pending, Trades (W/L/T), Decisive Win%, Matched Maj%, Wilson 95% CI, Exp Binary, Exp Digital, BE Gap.
- Visible zero-order execution safety banner: `EXECUTION_AUTHORITY: NONE · NO DEMONSTRATED EDGE · STRICTLY READ-ONLY`.

### 5. CLI Command & Session Process (`src/iqoption_research/cli/commands/shadow_cmd.py`, `src/iqoption_research/desktop/process.py`)
- Added `shadow observe` command supporting `--synthetic` (offline), `--live` (supervised broker), bounded `--minutes`, and multi-model `--model-id`.
- Added `start_prospective_shadow` to desktop session process runner.

### 6. Comprehensive Test Suite (`tests/unit/test_phase40b_prospective_shadow.py`)
- 8 new comprehensive unit tests verifying:
  1. Streaming candle event serialization roundtrip.
  2. Synthetic stream sequential playback and connection handling.
  3. Binary vs. Digital contract economics and break-even math.
  4. End-to-end prospective observation with delayed outcome resolution and point-in-time pre-outcome commitment.
  5. Evidence rejection (incomplete, discontinuous, duplicate).
  6. Operator cancellation and sealed manifest publication.
  7. Pre-flight candidate integrity and tamper rejection (`execution_authority = NONE`).
  8. Desktop panel prospective tab initialization and safety labels.

---

## Invariant Verification

- [x] `execution_authority = NONE` enforced across all engines and manifests.
- [x] Zero trade execution, order routing, or wallet mutation capability.
- [x] Candle-close comparisons explicitly disclaimed as hypothetical research benchmarks.
- [x] All 8 Phase 40B unit tests pass.
- [x] All 23 Phase 40A shadow evaluation tests pass.
- [x] MyPy: 0 errors across 251 source files.
- [x] Ruff check: All checks passed.
- [x] Ruff format: Clean across all 399 repository files.
- [x] CLI Security Check: All 7 checks passed.
