# Phase 47 WS2 — Causal Feature Catalog

All features in `src/iqoption_research/phase47/features.py` are strictly causal, gap-aware (continuity streak resets on gaps), prefix-invariant, and use zero future data, zero raw absolute timestamps, and zero fabricated orderbook/spread/news/settlement data.

## Feature Families
1. **Price-Action (15):** `body_norm_atr`, `upper_wick_ratio`, `lower_wick_ratio`, `wick_asymmetry`, `rejection_up`, `rejection_down`, `consec_bull`, `consec_bear`, `dist_roll_high_20`, `dist_roll_low_20`, `range_pos_20`, `swing_high_confirmed` (3-bar lag-confirmed at `i-2`), `swing_low_confirmed`, `breakout_strength_20`, `failed_breakout_proxy`.
2. **Trend & Momentum (11):** `ret_1`, `ret_3`, `ret_5`, `ret_10`, `ret_20`, `ma_slope_10`, `ma_slope_20`, `mom_accel_3_10`, `trend_persistence_10`, `dir_agreement_3_10_20`, `reversal_zscore_10`.
3. **Volatility & Compression (6):** `realized_vol_20`, `natr_14`, `vol_expansion_ratio`, `vol_percentile_60` (causal rolling rank), `rel_candle_range_20`, `vol_adj_mom_5`.
4. **Market Structure (3):** `sr_dist_upper_20`, `sr_dist_lower_20`, `trend_vs_range_efficiency_20` (Kaufman efficiency ratio).
5. **Market Sessions (8):** `utc_hour_sin`, `utc_hour_cos`, `utc_dow_sin`, `utc_dow_cos`, `sess_asia`, `sess_eu`, `sess_us`, `sess_eu_us_overlap`.
6. **Multi-Timeframe Context (6, closed HTF `to_ts <= decision_ts` via `bisect`):** `htf_m5_trend`, `htf_m15_trend`, `xtf_dir_agree`, `xtf_rel_vol`, `xtf_mom_div`, `htf_m15_regime`.

## Presets
- `p47_price_trend_vol` (32 features)
- `p47_full_native` (43 features)
- `p47_xtf_context` (49 features)
- `p47_ablation_no_session` (41 features)
- `p47_ablation_no_xtf` (43 features)
