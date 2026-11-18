# Phase 24 research report

Conclusion: **NO_DEMONSTRATED_EDGE**.
Offline historical simulation; no autonomous Demo authorization.

## Data quality

- EURUSD 60s / BROKER_AUTHENTIC_COLLECTION: INVALID_EVALUATION; UNVERIFIED unique candles.
- EURUSD 60s / broker_authentic_collection: INTEGRITY_PASSED; 63,559 unique candles.

## Strategy and ML leaderboard

| Candidate | Trades | Win rate | P&L | Max drawdown | P&L/drawdown | Evidence |
|---|---:|---:|---:|---:|---:|---|
| logistic_regression | 5 | 80.000% | 23.90 | 10.02 | 2.38522954091806 | INSUFFICIENT_AUTHENTIC_DATA |
| no_trade | 0 | N/A | 0.00 | 0.00 | None | INSUFFICIENT_AUTHENTIC_DATA |
| majority_class | 0 | N/A | 0.00 | 0.00 | None | INSUFFICIENT_AUTHENTIC_DATA |
| always_put | 276 | 44.776% | -465.52 | 498.68 | -0.933504451752415 | INSUFFICIENT_AUTHENTIC_DATA |
| random_forest | 521 | 43.574% | -975.92 | 1009.84 | -0.9664105204782754 | NO_DEMONSTRATED_EDGE |
| ma_crossover | 248 | 44.958% | -405.46 | 410.86 | -0.9868568368784871 | INSUFFICIENT_AUTHENTIC_DATA |
| momentum | 649 | 45.614% | -991.98 | 1004.32 | -0.9877130794962993 | NO_DEMONSTRATED_EDGE |
| always_call | 661 | 45.683% | -999.72 | 1005.12 | -0.99462750716304 | NO_DEMONSTRATED_EDGE |
| random | 47 | 34.043% | -174.94 | 174.94 | -0.9999999999998825 | INSUFFICIENT_AUTHENTIC_DATA |

## Walk-forward model results

| Fold | Model | Accuracy | Brier | Log loss | Validation threshold | Trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | majority_class | 0.4996 | 0.2500 | 0.6931 | 0.65 | 0 |
| 0 | logistic_regression | 0.4994 | 0.2500 | 0.6932 | 0.65 | 3 |
| 0 | random_forest | 0.5007 | 0.2501 | 0.6933 | 0.55 | 5 |
| 1 | majority_class | 0.5055 | 0.2500 | 0.6931 | 0.65 | 0 |
| 1 | logistic_regression | 0.5021 | 0.2501 | 0.6934 | 0.65 | 1 |
| 1 | random_forest | 0.4945 | 0.2860 | 0.7705 | 0.65 | 518 |
| 2 | majority_class | 0.4942 | 0.2500 | 0.6932 | 0.65 | 0 |
| 2 | logistic_regression | 0.4956 | 0.2502 | 0.6935 | 0.65 | 1 |
| 2 | random_forest | 0.4942 | 0.3922 | 1.1052 | 0.65 | 92 |

## ML and untouched holdout

{"status": "RESERVED_NOT_EVALUATED", "cutoff_ts": 1794607200, "available_candles": 688, "minimum_calendar_days": 14, "available_calendar_days": 2.4784722222222224, "data_sufficiency": "INSUFFICIENT_AUTHENTIC_DATA", "reservation_fingerprint": "36baa802e16e38e864bb4b7273e60c8295c7f4e39669128ab03e3404ffa1e45a", "reason": "Untouched confirmation data is not consumed by development commands."}

Fold classification, calibration, thresholds and economics are in results.json.

## Risk and uncertainty

The JSON compares sizing, loss brakes, confidence/EV filters and payouts.
Bootstrap intervals are pointwise 95%; Wilson intervals have a family-size adjustment.
Intervals depend on block length, stationarity, costs and the tested family.

## Failure analysis and recommendation

No independent untouched confirmation evaluated.
Development history was used in earlier milestones; selection bias remains.
Only the selected instrument/timeframe/source is benchmarked; no cross-product inference.
Calendar coverage below prespecified minimum.
Collect authentic data; continue paper research. No execution recommendation.

## Reproducibility

Revision: eef365ea4885f387650903baca9d9e2a921965ba
Config fingerprint: 1aa782ab6ef35d2100a92737625c5a5980b5cfa339edca59f5ad4288d2bfaf63
Configuration, fingerprints, seed, assumptions and package versions are in results.json.

## Risk scenarios (descriptive only)

| Candidate | Fixed P&L | Percent P&L | Strict brakes P&L | Confidence/EV P&L |
|---|---:|---:|---:|---:|
| no_trade | 0.00 | 0.00 | 0.00 | 0.00 |
| always_call | -999.72 | -999.21 | -118.70 | 0.00 |
| always_put | -465.52 | -465.54 | -6.94 | 0.00 |
| random | -174.94 | -173.63 | -69.36 | 0.00 |
| momentum | -991.98 | -991.45 | -58.58 | 0.00 |
| ma_crossover | -405.46 | -398.35 | -92.48 | 0.00 |
| majority_class | 0.00 | 0.00 | 0.00 | 0.00 |
| logistic_regression | 23.90 | 23.91 | 23.90 | 23.90 |
| random_forest | -975.92 | -969.49 | -37.78 | -1001.36 |
