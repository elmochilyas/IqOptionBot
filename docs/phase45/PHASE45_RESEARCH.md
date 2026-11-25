# Phase 45 — Historical Dataset Consolidation & Multi-Timeframe AI Research

**Disposition:** `NO_DEMONSTRATED_EDGE`
**Execution authority:** `NONE`
**Research class:** genuine offline development research on historically reused evidence
**Independent confirmation:** unavailable

Phase 45 consolidates verified local EURUSD evidence, publishes immutable pre-boundary
development manifests, runs a bounded genuine offline research matrix on native M1, M5 and
M15 populations, and publishes sanitized aggregate reports. It performs no broker
connection, order submission, wallet mutation or automatic execution.

## Evidence hierarchy

1. Verified research snapshots and canonical candle-content digests.
2. Independently rehashed `COMMITTED` collection receipts and their Parquet payloads.
3. Verified catalog, extended-session and checkpoint metadata.
4. Derived partitions and experiment provenance.
5. Older prose reports only when they do not conflict with cryptographic evidence.

Original artifacts are never modified. Derived development populations are published only
under ignored local storage; sanitized manifests and aggregate results are published under
`reports/phase45/`.

## Protected boundary

The protected boundary is `2026-11-12T10:00:00+00:00` (epoch `1794477600`). All candles at
or after that boundary remain protected and are excluded from development. Labels are also
excluded when their maturity close reaches the boundary, even when the decision timestamp
is earlier. The last eligible development candle `from_ts` values are therefore:

| Native population | Last development candle `from_ts` | Corresponding `to_ts` |
|---|---|---|
| M1 | `2026-11-12T09:59:00+00:00` | `2026-11-12T10:00:00+00:00` |
| M5 | `2026-11-12T09:55:00+00:00` | `2026-11-12T10:00:00+00:00` |
| M15 | `2026-11-12T09:45:00+00:00` | `2026-11-12T10:00:00+00:00` |

A decision at the last `from_ts` remains in the feature population, but its forward label is
invalidated because maturity reaches the protected boundary.

## Horizon semantics

Horizon names encode both the native candle size and the number of native bars:

| Name | Native duration | Prediction duration |
|---|---:|---:|
| `M1_H15` | 1 minute | 15 minutes |
| `M5_H3` | 5 minutes | 15 minutes |
| `M15_H1` | 15 minutes | 15 minutes |

A decision timestamp is the close of a fully closed native candle. Cross-timeframe features
are joined only when the higher-timeframe candle close is at or before that decision
timestamp. Rolling features reset at physical gaps; no interpolation or gap-bridging is used.

## Governance

Every development population has a self-hashed governance manifest binding:

- parent snapshot/source hashes;
- verified receipt/source hashes;
- canonical candle-content digest;
- exact pre-boundary interval and candle count;
- `HISTORICAL_REUSED` provenance and prior exposure history;
- protected interval exclusion;
- `NO_DEMONSTRATED_EDGE` and `execution_authority = NONE`.

Genuine preflight uses `PreflightEnforcement.GENUINE`. `TEST_ONLY` is reserved for synthetic
methodology fixtures and never labels genuine evidence.

## Research design

The screening matrix compares:

- training-fitted majority and rule-based baselines;
- regularized logistic regression;
- random forests;
- histogram gradient-boosted trees;
- compact scikit-learn MLP where sample-size policy permits, otherwise explicitly skipped;
- M1 ablations: M1 only, longer M1 history, M1+M5, M1+M15, M1+M5+M15.

Chronological walk-forward folds use horizon purge, five-bar embargo, train-only scaling,
calibration-segment-only calibrator fitting, and non-overlapping sensitivity populations.
Threshold and hyperparameter choices are preregistered for screening; a bounded nested
chronological check on the best finalist separates selection from assessment where sample
size permits.

## Scientific interpretation

All results are development research. Holm adjustment covers only the listed comparison
family and does **not** eliminate adaptive selection bias. Rolling-horizon predictions are
correlated and are not counted as independent observations. Hypothetical payout sensitivities
are not broker settlements. The Stage C acceptance contract is unchanged; because genuinely
independent evidence is unavailable, no edge claim is made.

## Reproduction

See [`reports/phase45/REPRODUCE.md`](../../reports/phase45/REPRODUCE.md).

## Workstation integration

A narrow `Phase 45 Research` section in the Model Laboratory reuses the PR #61 main-thread
UI queue: worker threads only enqueue callbacks, never call Tk methods directly. It shows
real progress, completed trials, active model, elapsed time and cooperative cancellation
state while preserving completed trial records.
