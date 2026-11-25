# Phase 47 WS2 — Causal Leakage & Boundary Audit

- **Protected Boundary Enforcement:** All research queries verify `max_to_ts <= 1794477600` (`2026-11-12T10:00:00Z`) and pass `max_label_maturity_ts=1794477600` to `MultiHorizonTargetEngine`.
- **Prefix Invariance:** `Phase47FeaturePipeline.compute_features(candles[:k])` produces bit-for-bit identical feature rows at every valid index `i < k` compared to `compute_features(candles)`.
- **Closed-Candle & HTF Alignment:** Higher-timeframe M5/M15 joins use `bisect_right(htf_to_ts, decision_ts) - 1`, guaranteeing only HTF candles with `to_ts <= decision_ts` are visible.
- **Continuity Gap Isolation:** Warmup streak resets to 1 whenever `from_ts[i] - from_ts[i-1] != timeframe_seconds`, preventing cross-gap indicator contamination.
- **Training-Only Fitting:** `StandardScaler`, `ProbabilityCalibrator`, and `CausalRegimeClassifier` are fitted strictly on training/calibration partitions with `purge = horizon_bars + 5` bars separating partitions.
- **Legacy Provenance Isolation:** Unresolved Phase 34/38 `UnverifiedOutcomeProvenanceError` on `data/phase34/operator_supervised_live/` is isolated from Phase 47 research, which loads verified `data/phase45/snapshots/` manifests directly.
