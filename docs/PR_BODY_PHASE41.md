# Phase 41: Model Failure Intelligence & Controlled Research

## Stacked Pull Request Chain
- Stacked Target: `codex/shadow-evaluation-foundation` (PR #53)
- Feature Branch: `codex/model-failure-intelligence`
- Disposition: Strictly offline diagnostics; `NO_DEMONSTRATED_EDGE` preserved; do not merge.

---

## Mission & Executive Summary
Phase 41 establishes an empirical diagnostic and controlled-experiment capability to determine **WHY** current EURUSD directional models fail, which components of their target and feature pipelines are structurally inconsistent, and which hypotheses warrant prospective observation.

### Key Deliverables & Findings
1. **Target Formulation & Downward Bias Root Cause (Confirmed)**:
   - Audited 66,463 authentic EURUSD M1 candles at horizon $H=1$.
   - Established that 3,431 bars (5.163%) are **EXACT TIES** ($C_{t+1} == C_t$).
   - Legacy binary target (`diffs > 0`) mapped all exact ties to `0` (DOWN), creating an artificial +5.818% downward class skew (52.909% DOWN vs 47.091% UP).
   - Models trained on legacy binary targets learned to overpredict DOWN.
   - Introduced `TiePolicy.EXCLUDE` in `TargetSpec` to purge exact ties from directional training without altering legacy frozen targets.
2. **Settlement Contract Mismatch Formalization**:
   - Generic academic close-to-close directional accuracy assumes frictionless execution and credits ties as DOWN wins.
   - In IQ Option **Binary Options**, an exact tie at expiry results in an **At-The-Money (ATM) Refund** (100% stake returned, 0% PnL).
   - In IQ Option **Digital Options**, an exact tie results in an **ATM Loss** (-100% loss of stake).
   - Formalized 4 contract specifications (`GENERIC_CLOSE_TO_CLOSE`, `IQOPTION_BINARY_OPTION`, `IQOPTION_DIGITAL_OPTION`, `FOREX_CFD_MARGIN`) and calculated payout sensitivity tables across broker return tiers.
3. **Train/Inference Feature Parity & Prefix Invariance (Confirmed & Fixed)**:
   - Root cause identified: `has_genuine_volume(candles)` previously evaluated whole datasets passed to `transform()`, leaking future volume presence into past candles and causing `FeatureSchemaError` on real-time slices.
   - Added `static_schema=True` to `CausalFeaturePipeline`, fixing feature presence by configuration rather than future scanning.
   - Implemented deterministic prefix invariance tests verifying that altering unseen future candles produces bitwise zero change to prefix feature values.
4. **Failure Intelligence Report**:
   - Implemented `generate_failure_intelligence_report()` computing confusion matrix with exact ties, UTC hourly breakdown, day-of-week breakdown, rolling volatility terciles, candle range terciles, calibration curves (ECE, Brier), confidence curves, and lag-1 autocorrelation ($\rho_1$) with effective sample size ($N_{\text{eff}}$).
   - Generated report saved to `reports/model_failure_intelligence.md`.
5. **Controlled Research Experiment Matrix (Areas A–F)**:
   - Predefined 14 trials across Target Formulation (A), Feature Parity (B), Feature Redundancy (C), Calibration (D), Market Context (E), and Selective Abstention/SKIP (F).
   - Evaluated strictly using chronological walk-forward partitions with purging ($H=1$) and embargo (5 bars).
   - Tracked all trials in an immutable ledger (`data/experiments/phase41_experiment_ledger.json`).
   - Universal invariant confirmed: all configurations exhibit negative break-even gap ($W - W_{\text{be}} < 0$, where $W_{\text{be}} = 54.05\%$ at 85% payout). `NO_DEMONSTRATED_EDGE` is strictly verified.
6. **Desktop Integration**:
   - Added `ModelDiagnosticsPanel` to the Tkinter desktop workstation notebook with 4 subtabs: Failure Intelligence, Regimes & Temporal, Controlled Experiments, and Microstructure & Economics.
   - Implemented non-blocking background execution with thread-safe UI marshaling.
   - Added CLI commands under `iqresearch diagnostics`.

---

## Verification & Testing Evidence
- Full pytest test suite: **2,487 passed, 3 warnings in 191.77s**.
- Unit tests for Phase 41: `tests/unit/test_phase41_failure_intelligence.py` (17 tests covering ties, prefix invariance, microstructure, failure metrics, experiments, desktop panel, and artifact immutability).
- Code style: `ruff check src tests` passes cleanly (0 errors).
- Type safety: `mypy src` passes cleanly (0 errors across 247 source files).
- Phase 38 artifacts, collection sessions, and Phase 40A candidate models (`data/candidate_models/`) remain 100% frozen and unmodified.
