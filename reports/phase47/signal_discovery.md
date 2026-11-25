# Phase 47 WS1 — Predictive Signal Discovery Report

Generated: `2026-11-25T20:35:18.881268+00:00`
Disposition: `NO_DEMONSTRATED_EDGE` | Authority: `NONE` | Stage: `DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION`

## Hypothesis Assessment

| Hypothesis | Status | Detail |
|---|---|---|
| Volatility clustering (|return| autocorrelation) | `SUPPORTED` | Lag-1..10 |return| autocorrelations are strongly positive across M1, M5, and M15, supporting causal volatility/compression features. |
| Unconditional 1-bar directional momentum / mean-reversion | `UNSTABLE_OR_WEAK` | Return autocorrelations and unconditional momentum persistence hover near 50% with day-block 95% CIs straddling 0.50 and H1/H2 drift. |
| Volume field as institutional order flow | `UNSUPPORTED` | Volume is tick/activity count only; treated strictly as activity proxy, never order flow. |
| UTC session structure | `SUPPORTED_AS_CONTEXT` | Volatility and directional activity vary by UTC session; useful as context features but not standalone directional predictors. |

## Timeframe 300s (`gov_eurusd_300s_624de4f68f22`)
- Candles: 59387 (contiguous: 59345)
- Return ACF: `{'lag_1': -0.02281, 'lag_2': -0.00572, 'lag_3': -0.0093, 'lag_5': 0.00461, 'lag_10': -0.00357}`
- Volatility Clustering (|ret| ACF): `{'lag_1': 0.26614, 'lag_2': 0.22879, 'lag_3': 0.22403, 'lag_5': 0.21423, 'lag_10': 0.18705}`
- Volume Audit: `{'zero_volume_fraction': 0.00017, 'volume_range_correlation': 0.77551, 'note': 'IQ Option tick/activity count proxy, not institutional order-book volume.'}`

| Horizon | Valid | UP% | Mom Persist (Day-Block 95% CI) | H1 vs H2 | Strong-Body Cont. |
|---|---:|---:|---|---|---|
| H1_5m | 59303 | 0.49457 | 0.4913 [0.4865, 0.4956] | 0.4907 vs 0.4918 | 0.4835 [0.4761, 0.4910] |
| H3_15m | 59219 | 0.49183 | 0.4896 [0.4855, 0.4937] | 0.4902 vs 0.4891 | 0.4815 [0.4739, 0.4887] |
| H6_30m | 59093 | 0.48747 | 0.4916 [0.4870, 0.4960] | 0.4905 vs 0.4927 | 0.4897 [0.4828, 0.4965] |
| H12_60m | 58841 | 0.48515 | 0.4938 [0.4898, 0.4980] | 0.4954 vs 0.4921 | 0.4950 [0.4888, 0.5014] |

## Timeframe 60s (`gov_eurusd_60s_f8fecd191001`)
- Candles: 60711 (contiguous: 60702)
- Return ACF: `{'lag_1': 0.01407, 'lag_2': 0.00576, 'lag_3': 0.00361, 'lag_5': 0.00117, 'lag_10': 0.00609}`
- Volatility Clustering (|ret| ACF): `{'lag_1': 0.25329, 'lag_2': 0.22737, 'lag_3': 0.21969, 'lag_5': 0.21175, 'lag_10': 0.18603}`
- Volume Audit: `{'zero_volume_fraction': 0.00326, 'volume_range_correlation': 0.76169, 'note': 'IQ Option tick/activity count proxy, not institutional order-book volume.'}`

| Horizon | Valid | UP% | Mom Persist (Day-Block 95% CI) | H1 vs H2 | Strong-Body Cont. |
|---|---:|---:|---|---|---|
| H1_1m | 60693 | 0.49706 | 0.4950 [0.4916, 0.4984] | 0.4943 vs 0.4957 | 0.4941 [0.4886, 0.4996] |
| H5_5m | 60657 | 0.49022 | 0.4975 [0.4935, 0.5013] | 0.4975 vs 0.4976 | 0.4970 [0.4909, 0.5033] |
| H15_15m | 60567 | 0.48721 | 0.4998 [0.4956, 0.5038] | 0.5010 vs 0.4986 | 0.4987 [0.4926, 0.5049] |
| H30_30m | 60432 | 0.48029 | 0.4992 [0.4945, 0.5036] | 0.4991 vs 0.4994 | 0.5009 [0.4946, 0.5069] |
| H60_60m | 60162 | 0.47967 | 0.4999 [0.4956, 0.5042] | 0.5020 vs 0.4979 | 0.5014 [0.4955, 0.5076] |

## Timeframe 900s (`gov_eurusd_900s_399632518d0e`)
- Candles: 30768 (contiguous: 30701)
- Return ACF: `{'lag_1': -0.02008, 'lag_2': -0.01572, 'lag_3': -0.00278, 'lag_5': 0.00893, 'lag_10': 0.00435}`
- Volatility Clustering (|ret| ACF): `{'lag_1': 0.23247, 'lag_2': 0.20357, 'lag_3': 0.19033, 'lag_5': 0.16277, 'lag_10': 0.10731}`
- Volume Audit: `{'zero_volume_fraction': 3e-05, 'volume_range_correlation': 0.0265, 'note': 'IQ Option tick/activity count proxy, not institutional order-book volume.'}`

| Horizon | Valid | UP% | Mom Persist (Day-Block 95% CI) | H1 vs H2 | Strong-Body Cont. |
|---|---:|---:|---|---|---|
| H1_15m | 30634 | 0.49638 | 0.4880 [0.4826, 0.4936] | 0.4897 vs 0.4862 | 0.4778 [0.4684, 0.4876] |
| H2_30m | 30567 | 0.49457 | 0.4864 [0.4810, 0.4920] | 0.4895 vs 0.4832 | 0.4838 [0.4739, 0.4933] |
| H4_60m | 30433 | 0.49212 | 0.4922 [0.4867, 0.4978] | 0.4902 vs 0.4942 | 0.4946 [0.4853, 0.5036] |
