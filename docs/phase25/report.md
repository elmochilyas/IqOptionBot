# Phase 25 research report

Conclusion: **NO_DEMONSTRATED_EDGE**.
Development data only; no autonomous execution authorized.

## Authentic coverage

- EURUSD 60s: INVALID_EVALUATION; UNVERIFIED verified unique candles.
- EURUSD 60s: INTEGRITY_PASSED; 63559 verified unique candles.

## Comparative evidence

| Seconds | Strategy | Trades | Unit EV/opportunity | Joint lower bound | Result |
|---:|---|---:|---:|---:|---|
| 60 | no_trade | 0 | 0.000000 | -0.058320 | NO_DEMONSTRATED_EDGE |
| 60 | always_call | 15670 | -0.199594 | -0.257914 | NO_DEMONSTRATED_EDGE |
| 60 | always_put | 15670 | -0.216228 | -0.274548 | NO_DEMONSTRATED_EDGE |
| 60 | random | 15670 | -0.204323 | -0.262643 | NO_DEMONSTRATED_EDGE |
| 60 | momentum | 14909 | -0.209634 | -0.267954 | NO_DEMONSTRATED_EDGE |
| 60 | ma_crossover | 15660 | -0.207037 | -0.265357 | NO_DEMONSTRATED_EDGE |
| 60 | inverse_momentum | 14909 | -0.178211 | -0.236531 | NO_DEMONSTRATED_EDGE |
| 60 | inverse_ma_crossover | 15660 | -0.208396 | -0.266716 | NO_DEMONSTRATED_EDGE |
| 300 | no_trade | 0 | 0.000000 | -0.058320 | NO_DEMONSTRATED_EDGE |
| 300 | always_call | 3094 | -0.132026 | -0.190346 | NO_DEMONSTRATED_EDGE |
| 300 | always_put | 3094 | -0.119954 | -0.178274 | NO_DEMONSTRATED_EDGE |
| 300 | random | 3094 | -0.120148 | -0.178468 | NO_DEMONSTRATED_EDGE |
| 300 | momentum | 3047 | -0.152261 | -0.210581 | NO_DEMONSTRATED_EDGE |
| 300 | ma_crossover | 3094 | -0.135145 | -0.193465 | NO_DEMONSTRATED_EDGE |
| 300 | inverse_momentum | 3047 | -0.094116 | -0.152436 | NO_DEMONSTRATED_EDGE |
| 300 | inverse_ma_crossover | 3094 | -0.116835 | -0.175155 | NO_DEMONSTRATED_EDGE |

## Failure analysis

The JSON compares complete-sample counterfactuals against risk-braked simulations.
It includes all folds, session/regime diagnostics and economic sensitivity.
Reference ML retains the Phase 24 chronological protections.

## Expansion and product limitations

{"status": "READY_FOR_AUTHORIZED_READ_ONLY_COLLECTION", "symbol": "EURUSD", "timeframe_seconds": 60, "current_oldest_utc": "2026-09-15T05:01:00+00:00", "target_oldest_utc": "2026-08-18T09:28:00+00:00", "max_new_candles": 2000, "max_broker_requests": 4, "source_fingerprint": "1447d94f07ea5b025892facd82535afe974d0ab60fcf6d5df3fa7d247ce670b9", "network_requests_made": 0, "blitz_candles": "UNSUPPORTED_VERIFIED_MAPPING_REQUIRED", "current_blitz_payout": "UNVERIFIED_NO_TICKET_TERMS_OBSERVED", "execution": "OFFLINE_PLAN_ONLY_NO_CREDENTIAL_ACCESS"}
Native Blitz candle, payout and settlement evidence remains unverified.

## Uncertainty and sealed confirmation

{"status": "SEALED_NOT_EVALUATED", "available_candles": 688, "reservation_fingerprint": "36baa802e16e38e864bb4b7273e60c8295c7f4e39669128ab03e3404ffa1e45a", "integrity_audit_only": true}
The declared candidate/timeframe/null family shares paired two-day block resampling.
The registry spends alpha prospectively; prior selection bias remains unknown.

No sealed holdout performance was used or evaluated.
Day-block inference is approximate under weak dependence/local stationarity.
Prospective alpha spending does not repair unknown prior research selection bias.
Derived 5-minute bars and assumed payouts do not validate native Blitz execution.
Uncensored results are counterfactual; guarded risk settings remain separate.
Session and volatility summaries are descriptive, not post-hoc selected strategies.

## Recommendation

Collect authentic development data; freeze a protocol before fresh confirmation.
No result here authorizes Demo or REAL execution.

## Reproducibility

Code revision: fc0adf2042a41f32da01992015f6f6e12f2c91ac
Development fingerprint: d28669356b2e0850f4949ca5afaad86cd858db8ec0b2f073e62fae9772851da6
Configuration, registry slot, alpha, fingerprints and results are in results.json.
