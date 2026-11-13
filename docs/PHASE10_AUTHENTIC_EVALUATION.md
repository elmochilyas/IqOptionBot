# Phase 10 Authentic Market Data Evaluation

Generated from the local Phase 9 catalog on 2026-11-13. The experiment was fully offline.

## Authentic data audit

- Catalog partitions and sidecars checksum-verified: **44**
- Catalog rows: **20,110**
- Unique EURUSD one-minute candles: **20,000**
- Deduplicated overlap rows: **110**
- Metadata-only duplicate variants: **1** (`at_ts` only)
- Conflicting market-data duplicates: **0**
- Trading coverage: **333.33 hours**
- Calendar span: **17.93 days**
- Hard coverage gaps: **2**
- Aggregate audit fingerprint: `70d8b1e906d5b76f97b93346ad4399e23abe1630dca66e14b1a1828a617e6632`

The gap boundaries were treated as independent segments. No feature window or forward label
crossed either gap. The resulting causal research matrix contained **19,937 rows** and 17
features.

## Evaluation design

- Four expanding walk-forward folds
- 50% initial training window, 10% validation window, four sequential 10% test windows
- One-bar purge between training/validation and validation/test
- Scalers fitted independently on training data only
- Probability calibration fitted on the immediately preceding validation data only
- Fixed confidence threshold of 0.55; no threshold optimization on test data
- Simulated EURUSD notional per trade: 10,000
- Assumed spread: 1.0 pip
- Assumed slippage: 0.1 pip at entry and exit
- Random seed: 42

## Strict out-of-sample results

| Model | Class | Accuracy | Precision | Recall | Trades | Win rate after costs | Execution costs | Net PnL | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| majority_class | baseline | 53.26% | 0.00% | 0.00% | 0 | 0.00% | 0.00 | +0.00 | 0.00% |
| rule_based_momentum | baseline | 50.01% | 46.87% | 52.01% | 7,566 | 7.89% | 8,156.55 | -8,132.64 | 81.33% |
| logistic_regression | machine learning | 53.23% | 25.00% | 0.03% | 988 | 6.78% | 1,072.75 | -1,035.29 | 10.35% |
| random_forest | machine learning | 53.26% | 0.00% | 0.00% | 2,002 | 8.94% | 2,173.52 | -2,119.64 | 21.20% |

The majority baseline abstained because its probability remained inside the fixed confidence
band. The apparent directional accuracy is therefore not a profitable trading result. Every
actionable strategy lost money after the configured spread and slippage assumptions. The best
ML result, logistic regression, did not beat the capital-preserving baseline.

## Assessment

More historical data is required. The current sample covers only 17.93 calendar days and does
not represent enough market regimes for a reliable edge claim. At least 90 calendar days is the
current minimum research gate, and substantially more is preferable. The stored candles also
lack verified bid/ask quotes, so execution costs are assumptions rather than observed spreads.
The best-model label is descriptive across the predeclared candidates; an additional untouched
later dataset is still required for independent confirmation.

> Strictly offline historical simulation. No broker connection or order API was used. Past
> simulated performance is not evidence of future profitability.
