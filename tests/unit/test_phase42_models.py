"""Unit tests for Phase 42 machine learning model families.

Validates:
1. RuleBasedReversalModel: contrarian logic, step/sigmoid mappings, serialization.
2. ExtraTreesModel: tree ensemble fitting, calibration, single-class handling.
3. CompactNeuralNetModel: MLP fitting, probability calibration, serialization.
4. Extended LogisticRegressionModel: L1, L2, ElasticNet penalties and serialization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    RuleBasedReversalModel,
)
from iqoption_research.ml.models.tabular import (
    CompactNeuralNetModel,
    ExtraTreesModel,
)


def _generate_synthetic_classification_data(n_samples: int = 150, n_features: int = 6):
    """Generate reproducible synthetic classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    # Target linearly dependent on first feature plus noise
    logits = X[:, 0] * 1.5 - X[:, 1] * 0.8 + rng.normal(0, 0.5, size=n_samples)
    y = (logits > 0).astype(np.int64)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


def test_rule_based_reversal_model(tmp_path: Path) -> None:
    """Verify RuleBasedReversalModel contrarian logic and serialization."""
    # When return_1 > threshold, predicts DOWN (p1 < 0.5)
    # When return_1 < -threshold, predicts UP (p1 > 0.5)
    model = RuleBasedReversalModel(
        reversal_feature="return_1",
        threshold=0.0001,
        base_prob=0.58,
        mapping_type="step",
    )

    X = np.array([
        [0.0005],  # Strongly positive return -> expect reversal (DOWN, p1=0.42, p0=0.58)
        [-0.0005],  # Strongly negative return -> expect reversal (UP, p1=0.58, p0=0.42)
        [0.0],  # Neutral return -> expect flat 0.5/0.5
    ])
    y = np.array([0, 1, 0])
    feature_names = ["return_1"]

    model.fit(X, y, feature_names=feature_names)
    assert model.is_fitted

    probs = model.predict_proba(X)
    assert probs.shape == (3, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # First row: return > threshold -> p1 = 1 - base_prob = 0.42
    assert abs(probs[0, 1] - 0.42) < 1e-6
    assert abs(probs[0, 0] - 0.58) < 1e-6

    # Second row: return < -threshold -> p1 = base_prob = 0.58
    assert abs(probs[1, 1] - 0.58) < 1e-6
    assert abs(probs[1, 0] - 0.42) < 1e-6

    # Third row: neutral -> 0.5 / 0.5
    assert abs(probs[2, 1] - 0.50) < 1e-6

    # Test serialization
    save_dir = tmp_path / "reversal_model"
    model.save(save_dir)
    loaded = RuleBasedReversalModel.load(save_dir)
    assert loaded.is_fitted
    loaded_probs = loaded.predict_proba(X)
    assert np.allclose(probs, loaded_probs)


def test_extra_trees_model(tmp_path: Path) -> None:
    """Verify ExtraTreesModel fitting, predictions, single-class handling, and serialization."""
    X, y, feature_names = _generate_synthetic_classification_data()

    model = ExtraTreesModel(
        n_estimators=15,
        max_depth=4,
        random_state=42,
    )
    model.fit(X, y, feature_names=feature_names)
    assert model.is_fitted

    probs = model.predict_proba(X[:10])
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    preds = model.predict(X[:10])
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})

    # Test serialization
    save_dir = tmp_path / "extra_trees_model"
    model.save(save_dir)
    loaded = ExtraTreesModel.load(save_dir)
    assert loaded.is_fitted
    loaded_probs = loaded.predict_proba(X[:10])
    assert np.allclose(probs, loaded_probs)

    # Test single-class handling
    y_single = np.ones(len(X), dtype=np.int64)
    model_single = ExtraTreesModel(n_estimators=10)
    model_single.fit(X, y_single, feature_names=feature_names)
    single_probs = model_single.predict_proba(X[:5])
    assert np.allclose(single_probs[:, 1], 1.0)
    assert np.allclose(single_probs[:, 0], 0.0)


def test_compact_neural_net_model(tmp_path: Path) -> None:
    """Verify CompactNeuralNetModel fitting, predictions, and serialization."""
    X, y, feature_names = _generate_synthetic_classification_data()

    model = CompactNeuralNetModel(
        hidden_layer_sizes=(16, 8),
        max_iter=30,
        random_state=42,
    )
    model.fit(X, y, feature_names=feature_names)
    assert model.is_fitted

    probs = model.predict_proba(X[:10])
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    preds = model.predict(X[:10])
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})

    # Test serialization
    save_dir = tmp_path / "compact_nn_model"
    model.save(save_dir)
    loaded = CompactNeuralNetModel.load(save_dir)
    assert loaded.is_fitted
    loaded_probs = loaded.predict_proba(X[:10])
    assert np.allclose(probs, loaded_probs)


def test_extended_logistic_regression_penalties(tmp_path: Path) -> None:
    """Verify LogisticRegressionModel with L2, L1, and ElasticNet penalties."""
    X, y, feature_names = _generate_synthetic_classification_data()

    # 1. L2 penalty (default lbfgs)
    m_l2 = LogisticRegressionModel(C=1.0, penalty="l2", solver="lbfgs")
    m_l2.fit(X, y, feature_names=feature_names)
    assert m_l2.is_fitted
    p_l2 = m_l2.predict_proba(X[:5])
    assert p_l2.shape == (5, 2)
    assert np.allclose(p_l2.sum(axis=1), 1.0)

    # 2. L1 penalty (should automatically use saga solver)
    m_l1 = LogisticRegressionModel(C=0.5, penalty="l1", solver="lbfgs")
    assert m_l1.solver == "saga"
    m_l1.fit(X, y, feature_names=feature_names)
    assert m_l1.is_fitted
    p_l1 = m_l1.predict_proba(X[:5])
    assert p_l1.shape == (5, 2)
    assert np.allclose(p_l1.sum(axis=1), 1.0)

    # 3. ElasticNet penalty (should use saga and l1_ratio)
    m_en = LogisticRegressionModel(C=0.5, penalty="elasticnet", solver="saga", l1_ratio=0.5)
    m_en.fit(X, y, feature_names=feature_names)
    assert m_en.is_fitted
    p_en = m_en.predict_proba(X[:5])
    assert p_en.shape == (5, 2)
    assert np.allclose(p_en.sum(axis=1), 1.0)

    # Test serialization of ElasticNet model
    save_dir = tmp_path / "elasticnet_lr"
    m_en.save(save_dir)
    loaded = LogisticRegressionModel.load(save_dir)
    assert loaded.is_fitted
    loaded_probs = loaded.predict_proba(X[:5])
    assert np.allclose(p_en, loaded_probs)
