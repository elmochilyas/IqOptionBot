"""Phase 47 WS4 — Target and horizon formulation comparison.

Compares standard UP/DOWN classification, volatility-normalized return regression,
and UP/DOWN/NEUTRAL 3-class training (evaluated unconditionally on all actual
market outcomes so difficult outcomes are never hidden).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from iqoption_research.ml.walk_forward import create_model_instance
from iqoption_research.research.features.scalers import StandardScaler


def evaluate_target_formulations(
    X: np.ndarray,
    closes: np.ndarray,
    future_closes: np.ndarray,
    natr: np.ndarray,
    *,
    embargo: int = 20,
) -> dict[str, Any]:
    """Compare target formulations on the exact same held-out unconditional opportunities."""
    n = len(closes)
    split = int(n * 0.65)
    train_end = max(100, split - embargo)
    test_start = split

    raw_ret = (future_closes - closes) / np.maximum(closes, 1e-12)
    y_bin = (raw_ret > 1e-12).astype(np.int64)
    z_ret = raw_ret / np.maximum(natr, 1e-6)

    scaler = StandardScaler().fit(X[:train_end])
    X_tr = np.asarray(scaler.transform(X[:train_end]), dtype=np.float64)
    X_te = np.asarray(scaler.transform(X[test_start:]), dtype=np.float64)
    y_te = y_bin[test_start:]

    # 1) Standard UP/DOWN Logistic Regression
    clf_bin = create_model_instance("logistic_regression", {"C": 0.5, "max_iter": 500})
    clf_bin.fit(X_tr, y_bin[:train_end])
    p_bin = np.asarray(clf_bin.predict_proba(X_te))[:, 1]
    acc_bin = float(np.mean((p_bin >= 0.5).astype(np.int64) == y_te))

    # 2) Volatility-normalized return regression (Ridge -> sigmoid probability)
    reg = Ridge(alpha=10.0)
    reg.fit(X_tr, np.clip(z_ret[:train_end], -5.0, 5.0))
    pred_z = reg.predict(X_te)
    p_reg = 1.0 / (1.0 + np.exp(-np.clip(pred_z, -10.0, 10.0)))
    acc_reg = float(np.mean((p_reg >= 0.5).astype(np.int64) == y_te))

    # 3) UP/DOWN/NEUTRAL training (filter small moves < 0.25*ATR in TRAIN ONLY; eval on ALL test)
    decisive_tr = np.abs(z_ret[:train_end]) >= 0.25
    if int(np.sum(decisive_tr)) >= 100 and len(np.unique(y_bin[:train_end][decisive_tr])) == 2:
        clf_neu = create_model_instance("logistic_regression", {"C": 0.5, "max_iter": 500})
        clf_neu.fit(X_tr[decisive_tr], y_bin[:train_end][decisive_tr])
        p_neu = np.asarray(clf_neu.predict_proba(X_te))[:, 1]
        acc_neu = float(np.mean((p_neu >= 0.5).astype(np.int64) == y_te))
    else:
        acc_neu = acc_bin

    return {
        "unconditional_test_samples": len(y_te),
        "standard_up_down_acc": round(acc_bin, 5),
        "vol_norm_regression_acc": round(acc_reg, 5),
        "neutral_filtered_train_acc": round(acc_neu, 5),
    }
