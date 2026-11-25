# Phase 46 WS6 - Synthetic Session Diagnostics (SYNTHETIC)

> **SYNTHETIC / DEVELOPMENT_RESEARCH ONLY.** No broker/orders/wallet. `execution_authority=NONE`, `NO_DEMONSTRATED_EDGE`. Not genuine prospective evidence.

- Session: `p46_synth_m1_1795631660`
- Symbol/timeframe: `EURUSD` / `60s`
- Provenance: `SYNTHETIC`
- Total predictions: 28
- Resolved: 27, Pending: 1
- Wins/Losses/Ties: 27/0/0
- Coverage: 1.0000, Abstention rate: 0.0000
- Directional accuracy: 1.0000, Brier: 0.0026
- Win rate (decisive): 1.0000
- Verdict: `FUNCTIONAL_REPLAY_TEST_ONLY_NO_EDGE`

## Balanced accuracy & confidence

```json
{
  "balanced_accuracy": {
    "status": "AVAILABLE",
    "value": 1.0,
    "up_recall": 1.0,
    "down_recall": 1.0,
    "up_true_count": 23,
    "down_true_count": 4
  },
  "confidence_summary": {
    "status": "AVAILABLE",
    "count": 28,
    "mean": 0.990613,
    "min": 0.73717,
    "max": 1.0
  }
}
```

## Baseline comparisons (matched)

```json
[
  {
    "baseline_name": "majority_class",
    "sample_count": 27,
    "decisive_sample_count": 27,
    "tie_count": 0,
    "directional_accuracy": 0.0,
    "decisive_directional_accuracy": 0.0,
    "brier_score": 0.25,
    "candidate_delta_accuracy": 0.0,
    "candidate_delta_decisive_accuracy": 0.0,
    "candidate_delta_brier": 0.0,
    "is_verified": false,
    "status": "UNAVAILABLE"
  },
  {
    "baseline_name": "random_uniform",
    "sample_count": 27,
    "decisive_sample_count": 27,
    "tie_count": 0,
    "directional_accuracy": 0.5,
    "decisive_directional_accuracy": 0.5,
    "brier_score": 0.25,
    "candidate_delta_accuracy": 0.5,
    "candidate_delta_decisive_accuracy": 0.5,
    "candidate_delta_brier": 0.2474,
    "is_verified": true,
    "status": "VERIFIED"
  }
]
```

## Calibration / reliability

```json
{
  "status": "AVAILABLE",
  "brier_score": 0.0026,
  "ece": 0.009734,
  "bins": [
    {
      "range": "[0.7, 0.8)",
      "count": 1,
      "mean_confidence": 0.73717,
      "empirical_accuracy": 1.0
    },
    {
      "range": "[0.9, 1.0]",
      "count": 26,
      "mean_confidence": 1.0,
      "empirical_accuracy": 1.0
    }
  ],
  "calibration_drift": {
    "status": "UNAVAILABLE",
    "note": "No reference training calibration curve stored alongside session journal."
  }
}
```

## Feature distribution drift

```json
{
  "status": "UNAVAILABLE",
  "note": "No reference training feature distribution snapshot stored alongside this session journal.",
  "observed_feature_fingerprints": 0
}
```

## Regime/session breakdown

```json
{
  "status": "PARTIAL",
  "utc_sessions": {
    "ASIA_00_08Z": {
      "resolved_decisive": 27,
      "wins": 27,
      "losses": 0,
      "accuracy": 1.0
    }
  },
  "causal_market_regimes": {
    "status": "UNAVAILABLE",
    "note": "Volatility/regime proxy values are not persisted per-prediction in the current journal schema."
  }
}
```

## Abstention reasons

```json
{
  "counts_by_reason": {
    "none": 0
  },
  "total_abstentions": 0,
  "status": "AVAILABLE"
}
```

## Data-quality & failure intelligence

```json
{
  "counts_by_note": {
    "none": 0
  },
  "invalid_discontinuity_count": 0,
  "quarantined_late_count": 0,
  "unresolved_pending_count": 1,
  "feature_computation_failures": 0,
  "is_interrupted": false,
  "terminal_status": "STREAM_COMPLETED_EOF",
  "status": "AVAILABLE"
}
```

## Evidence verification & manifest

```json
{
  "evidence_verification": {
    "status": "VERIFIED",
    "is_valid": true,
    "issues": []
  },
  "session_manifest_summary": {
    "status": "AVAILABLE",
    "session_id": "p46_synth_m1_1795631660",
    "terminal_status": "STREAM_COMPLETED_EOF",
    "termination_reason": "STREAM_COMPLETED_EOF",
    "is_interrupted": false,
    "is_synthetic": true,
    "file_digests": {}
  }
}
```

## Candidate-specific

```json
{
  "model_id": "candidate_logistic_regression_20261125_183012_lab_98be193602d4",
  "symbol": "EURUSD",
  "timeframe_seconds": 60,
  "opportunity_count": 28,
  "commitments_verified_count": 27,
  "status": "AVAILABLE"
}
```

## Disclaimer

OFFLINE RESEARCH SHADOW EVALUATION. NO LIVE EXECUTION AUTHORITY. NO DEMONSTRATED EDGE. CANDLE-CLOSE COMPARISONS ARE HYPOTHETICAL RESEARCH BENCHMARKS AND NEVER OBSERVED BROKER ORDER SETTLEMENTS. ASSUMED PAYOUT SIMULATIONS DO NOT CONSTITUTE PROOF OF PROSPECTIVE PROFITABILITY. WILSON SCORE INTERVALS ASSUME I.I.D. BERNOULLI TRIALS AND REPRESENT AN OPTIMAL BOUND UNDER TEMPORAL SERIAL DEPENDENCE. PHASE 38 MODEL REMAINS FROZEN.
