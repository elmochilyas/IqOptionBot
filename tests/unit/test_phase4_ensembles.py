"""Comprehensive Unit Test Suite for Phase 4 Ensembles Subsystem.

Covers:
- Feature 38: Simple Averaging Ensemble (voting.py)
- Feature 39: Weighted Averaging Ensemble (weighted.py)
- Feature 40: Stacking Ensemble with Out-Of-Fold CV (stacking.py)
- Mathematical probability axioms, anti-leakage isolation, and persistence reload parity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqoption_research.ml.ensembles.base import (
    BaseEnsembleModel,
    clone_base_model,
)
from iqoption_research.ml.ensembles.stacking import (
    StackingEnsemble,
    StackingSplitMethod,
)
from iqoption_research.ml.ensembles.voting import AveragingEnsemble
from iqoption_research.ml.ensembles.weighted import (
    WeightedEnsemble,
    WeightingScheme,
)
from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelPersistenceError,
    ModelValidationError,
    NotFittedError,
)
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
    SeededRandomModel,
)
from iqoption_research.ml.models.tabular import RandomForestModel


def _make_data(n: int = 120, d: int = 4, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Helper creating deterministic training data with clear signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float64)
    # Informative linear relationship in first feature
    p = 1.0 / (1.0 + np.exp(-2.0 * X[:, 0]))
    y = rng.binomial(1, p).astype(np.int64)
    y[0] = 0
    y[1] = 1
    return X, y


class TestAveragingEnsembleContract:
    """Verifies BaseMLModel compliance and contract invariants for AveragingEnsemble."""

    def test_averaging_ensemble_inherits_base_ml_model(self) -> None:
        m1 = LogisticRegressionModel(random_state=42)
        m2 = MajorityClassModel()
        ensemble = AveragingEnsemble(models=[m1, m2])
        assert isinstance(ensemble, BaseMLModel)
        assert isinstance(ensemble, BaseEnsembleModel)
        assert ensemble.model_type == "ensemble"
        assert ensemble.n_models == 2
        assert ensemble.submodel_names == ("logistic_regression", "majority_class")

    def test_unfitted_averaging_ensemble_raises_not_fitted_error(self) -> None:
        m1 = LogisticRegressionModel()
        ensemble = AveragingEnsemble(models=[m1])
        X = np.ones((5, 3))
        with pytest.raises(NotFittedError, match="not fitted"):
            ensemble.predict_proba(X)
        with pytest.raises(NotFittedError, match="not fitted"):
            ensemble.predict(X)

    def test_empty_models_list_raises_validation_error(self) -> None:
        with pytest.raises(ModelValidationError, match="at least one base model"):
            AveragingEnsemble(models=[])

    def test_non_base_ml_model_raises_validation_error(self) -> None:
        class NotAModel:
            pass

        with pytest.raises(ModelValidationError, match="must be an instance of BaseMLModel"):
            AveragingEnsemble(models=[NotAModel()])  # type: ignore[list-item]

    def test_fit_returns_self_and_updates_is_fitted(self) -> None:
        X, y = _make_data(n=60, d=3)
        m1 = LogisticRegressionModel(random_state=42)
        m2 = MajorityClassModel()
        ensemble = AveragingEnsemble(models=[m1, m2])

        assert not ensemble.is_fitted
        res = ensemble.fit(X, y, feature_names=["f0", "f1", "f2"])
        assert res is ensemble
        assert ensemble.is_fitted
        assert ensemble.feature_names == ("f0", "f1", "f2")

    def test_feature_names_propagated_and_validated(self) -> None:
        X, y = _make_data(n=50, d=3)
        ensemble = AveragingEnsemble(models=[LogisticRegressionModel()])
        with pytest.raises(ModelValidationError, match="Length of feature_names"):
            ensemble.fit(X, y, feature_names=["only_one"])


class TestAveragingEnsembleMathematicalAxioms:
    """Verifies numerical correctness and probability axioms for AveragingEnsemble."""

    def test_averaging_exact_arithmetic_mean(self) -> None:
        X, y = _make_data(n=80, d=4)
        m1 = LogisticRegressionModel(random_state=42)
        m2 = MajorityClassModel()
        m3 = SeededRandomModel(seed=123)

        ensemble = AveragingEnsemble(models=[m1, m2, m3])
        ensemble.fit(X, y)

        X_test, _ = _make_data(n=20, d=4, seed=999)
        p1 = m1.predict_proba(X_test)
        p2 = m2.predict_proba(X_test)
        p3 = m3.predict_proba(X_test)

        expected_mean = (p1 + p2 + p3) / 3.0
        # Normalize
        expected_norm = expected_mean / expected_mean.sum(axis=1, keepdims=True)

        ens_probs = ensemble.predict_proba(X_test)
        np.testing.assert_allclose(ens_probs, expected_norm, atol=1e-9)

    def test_probabilities_sum_to_one_and_in_range(self) -> None:
        X, y = _make_data(n=60, d=2)
        ensemble = AveragingEnsemble(
            models=[LogisticRegressionModel(), MajorityClassModel()]
        )
        ensemble.fit(X, y)

        X_eval, _ = _make_data(n=30, d=2, seed=1234)
        probs = ensemble.predict_proba(X_eval)

        assert probs.shape == (30, 2)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_matches_argmax_probability(self) -> None:
        X, y = _make_data(n=60, d=2)
        ensemble = AveragingEnsemble(models=[LogisticRegressionModel()])
        ensemble.fit(X, y)

        probs = ensemble.predict_proba(X)
        preds = ensemble.predict(X)
        expected_preds = np.argmax(probs, axis=1)
        np.testing.assert_array_equal(preds, expected_preds)

    def test_single_base_model_identity(self) -> None:
        X, y = _make_data(n=60, d=3)
        m1 = LogisticRegressionModel(random_state=42)
        ensemble = AveragingEnsemble(models=[m1])

        ensemble.fit(X, y)
        probs_ens = ensemble.predict_proba(X)
        probs_single = m1.predict_proba(X)

        np.testing.assert_allclose(probs_ens, probs_single, atol=1e-9)


class TestWeightedEnsembleContractAndWeighting:
    """Verifies weighting algorithms and anti-leakage isolation for WeightedEnsemble."""

    def test_weighted_ensemble_inverse_brier_weights(self) -> None:
        X, y = _make_data(n=100, d=4, seed=42)
        X_train, y_train = X[:70], y[:70]
        X_val, y_val = X[70:], y[70:]

        # Skilled model vs unskilled majority class
        m_skilled = LogisticRegressionModel(random_state=42)
        m_unskilled = MajorityClassModel()

        ensemble = WeightedEnsemble(
            models=[m_skilled, m_unskilled],
            weighting_scheme=WeightingScheme.INVERSE_BRIER,
        )
        ensemble.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        weights = ensemble.weights_
        assert len(weights) == 2
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-6)
        assert all(w >= 0.0 for w in weights)
        # Skilled model should achieve lower validation Brier score and thus higher weight
        assert weights[0] > weights[1]

    def test_weighted_ensemble_softmax_brier_temperature(self) -> None:
        X, y = _make_data(n=100, d=3)
        X_train, y_train = X[:60], y[:60]
        X_val, y_val = X[60:], y[60:]

        m1 = LogisticRegressionModel()
        m2 = MajorityClassModel()

        ens_sharp = WeightedEnsemble(
            models=[m1, m2],
            weighting_scheme=WeightingScheme.SOFTMAX_BRIER,
            temperature=0.01,
        )
        ens_sharp.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        ens_smooth = WeightedEnsemble(
            models=[m1, m2],
            weighting_scheme=WeightingScheme.SOFTMAX_BRIER,
            temperature=5.0,
        )
        ens_smooth.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # Lower temperature should yield more concentrated (sharper) weight distribution
        assert ens_sharp.weights_[0] > ens_smooth.weights_[0]

    def test_custom_user_weights(self) -> None:
        X, y = _make_data(n=60, d=2)
        m1 = LogisticRegressionModel()
        m2 = MajorityClassModel()

        ensemble = WeightedEnsemble(
            models=[m1, m2],
            weighting_scheme=WeightingScheme.CUSTOM,
            custom_weights=[0.8, 0.2],
        )
        ensemble.fit(X, y)

        np.testing.assert_allclose(ensemble.weights_, [0.8, 0.2], atol=1e-6)

        p1 = m1.predict_proba(X)
        p2 = m2.predict_proba(X)
        expected = 0.8 * p1 + 0.2 * p2
        expected_norm = expected / expected.sum(axis=1, keepdims=True)

        np.testing.assert_allclose(ensemble.predict_proba(X), expected_norm, atol=1e-9)

    def test_invalid_weights_and_hyperparameters_raise_error(self) -> None:
        m1 = LogisticRegressionModel()
        m2 = MajorityClassModel()

        with pytest.raises(ModelValidationError, match="custom_weights length"):
            WeightedEnsemble(
                models=[m1, m2],
                weighting_scheme=WeightingScheme.CUSTOM,
                custom_weights=[0.5],
            )

        with pytest.raises(ModelValidationError, match="non-negative"):
            WeightedEnsemble(
                models=[m1, m2],
                weighting_scheme=WeightingScheme.CUSTOM,
                custom_weights=[-0.2, 1.2],
            )

        with pytest.raises(ModelValidationError, match=r"temperature must be > 0.0"):
            WeightedEnsemble(models=[m1, m2], temperature=-1.0)

        with pytest.raises(ModelValidationError, match=r"cv_folds must be >= 2"):
            WeightedEnsemble(models=[m1, m2], cv_folds=1)

    def test_out_of_fold_weight_derivation_without_validation_fold(self) -> None:
        X, y = _make_data(n=80, d=3)
        m1 = LogisticRegressionModel(random_state=42)
        m2 = MajorityClassModel()

        ensemble = WeightedEnsemble(
            models=[m1, m2],
            weighting_scheme=WeightingScheme.INVERSE_BRIER,
            cv_folds=3,
        )
        # Mode B: X_val is None
        ensemble.fit(X, y)

        assert ensemble.is_fitted
        assert len(ensemble.weights_) == 2
        np.testing.assert_allclose(np.sum(ensemble.weights_), 1.0, atol=1e-6)
        assert len(ensemble.model_metrics_) == 2

    def test_feature_importances_weighted_average(self) -> None:
        X, y = _make_data(n=80, d=3)
        m1 = RandomForestModel(n_estimators=10, random_state=42)
        m2 = RandomForestModel(n_estimators=10, random_state=84)

        ensemble = WeightedEnsemble(
            models=[m1, m2],
            weighting_scheme=WeightingScheme.CUSTOM,
            custom_weights=[0.6, 0.4],
        )
        ensemble.fit(X, y)

        imp = ensemble.feature_importances_
        assert imp is not None
        assert imp.shape == (3,)

        imp1 = m1.feature_importances_
        imp2 = m2.feature_importances_
        assert imp1 is not None and imp2 is not None
        expected_imp = 0.6 * imp1 + 0.4 * imp2
        np.testing.assert_allclose(imp, expected_imp, atol=1e-9)


class TestStackingEnsembleAndLeakagePrevention:
    """Verifies two-level stacking meta-learning, OOF generation, and purging/embargo."""

    def test_stacking_ensemble_out_of_fold_training_and_inference(self) -> None:
        X, y = _make_data(n=90, d=4)
        m1 = LogisticRegressionModel(random_state=42)
        m2 = MajorityClassModel()

        ensemble = StackingEnsemble(
            models=[m1, m2],
            cv_folds=3,
            cv_type=StackingSplitMethod.KFOLD,
        )
        ensemble.fit(X, y, feature_names=["f0", "f1", "f2", "f3"])

        assert ensemble.is_fitted
        assert ensemble.meta_model_.is_fitted
        assert len(ensemble.meta_feature_names_) == 2
        assert ensemble.meta_feature_names_ == (
            "logistic_regression_0_prob",
            "majority_class_1_prob",
        )

        # Inference on new data
        X_test, _ = _make_data(n=20, d=4, seed=777)
        probs = ensemble.predict_proba(X_test)
        assert probs.shape == (20, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

        preds = ensemble.predict(X_test)
        assert preds.shape == (20,)
        np.testing.assert_array_equal(preds, np.argmax(probs, axis=1))

    def test_stacking_with_purging_and_embargo(self) -> None:
        X, y = _make_data(n=100, d=3)
        m1 = LogisticRegressionModel()
        m2 = MajorityClassModel()

        ensemble = StackingEnsemble(
            models=[m1, m2],
            cv_folds=3,
            purge_bars=2,
            embargo_bars=2,
        )
        ensemble.fit(X, y)
        assert ensemble.is_fitted

        probs = ensemble.predict_proba(X)
        assert probs.shape == (100, 2)

    def test_stacking_passthrough_mode(self) -> None:
        X, y = _make_data(n=80, d=3)
        m1 = LogisticRegressionModel()
        m2 = MajorityClassModel()

        ensemble = StackingEnsemble(
            models=[m1, m2],
            passthrough=True,
            cv_folds=2,
        )
        ensemble.fit(X, y, feature_names=["f0", "f1", "f2"])

        # Meta-features should include 2 base model probs + 3 original features = 5
        assert len(ensemble.meta_feature_names_) == 5
        assert ensemble.meta_feature_names_ == (
            "logistic_regression_0_prob",
            "majority_class_1_prob",
            "f0",
            "f1",
            "f2",
        )

        probs = ensemble.predict_proba(X)
        assert probs.shape == (80, 2)

    def test_stacking_invalid_parameters_raise_validation_error(self) -> None:
        m1 = LogisticRegressionModel()
        with pytest.raises(ModelValidationError, match="cv_folds must be >= 2"):
            StackingEnsemble(models=[m1], cv_folds=1)
        with pytest.raises(ModelValidationError, match="purge_bars cannot be negative"):
            StackingEnsemble(models=[m1], purge_bars=-1)
        with pytest.raises(ModelValidationError, match="embargo_bars cannot be negative"):
            StackingEnsemble(models=[m1], embargo_bars=-1)
        with pytest.raises(ModelValidationError, match="Unsupported cv_type"):
            StackingEnsemble(models=[m1], cv_type="invalid_type")


class TestEnsembleSerializationAndReloadParity:
    """Verifies hierarchical persistence and bit-for-bit reload parity."""

    def test_averaging_ensemble_save_and_load_parity(self, tmp_path: Path) -> None:
        X, y = _make_data(n=80, d=3)
        ensemble = AveragingEnsemble(
            models=[LogisticRegressionModel(random_state=42), MajorityClassModel()]
        )
        ensemble.fit(X, y)

        save_dir = tmp_path / "averaging_model"
        ensemble.save(save_dir)

        # Reload via AveragingEnsemble.load and BaseMLModel.load
        reloaded_cls = AveragingEnsemble.load(save_dir)
        reloaded_poly = BaseMLModel.load(save_dir)

        assert reloaded_cls.is_fitted
        assert reloaded_poly.is_fitted
        assert isinstance(reloaded_poly, AveragingEnsemble)

        X_test, _ = _make_data(n=25, d=3, seed=888)
        p_orig = ensemble.predict_proba(X_test)
        p_cls = reloaded_cls.predict_proba(X_test)
        p_poly = reloaded_poly.predict_proba(X_test)

        np.testing.assert_array_equal(p_orig, p_cls)
        np.testing.assert_array_equal(p_orig, p_poly)

    def test_weighted_ensemble_save_and_load_parity(self, tmp_path: Path) -> None:
        X, y = _make_data(n=80, d=3)
        ensemble = WeightedEnsemble(
            models=[LogisticRegressionModel(random_state=42), MajorityClassModel()],
            custom_weights=[0.75, 0.25],
        )
        ensemble.fit(X, y)

        save_dir = tmp_path / "weighted_model"
        ensemble.save(save_dir)

        reloaded = BaseMLModel.load(save_dir)
        assert isinstance(reloaded, WeightedEnsemble)
        np.testing.assert_allclose(reloaded.weights_, ensemble.weights_, atol=1e-9)

        X_test, _ = _make_data(n=20, d=3, seed=555)
        np.testing.assert_array_equal(
            ensemble.predict_proba(X_test), reloaded.predict_proba(X_test)
        )

    def test_stacking_ensemble_save_and_load_parity(self, tmp_path: Path) -> None:
        X, y = _make_data(n=80, d=3)
        ensemble = StackingEnsemble(
            models=[LogisticRegressionModel(random_state=42), MajorityClassModel()],
            cv_folds=2,
        )
        ensemble.fit(X, y)

        save_dir = tmp_path / "stacking_model"
        ensemble.save(save_dir)

        reloaded = BaseMLModel.load(save_dir)
        assert isinstance(reloaded, StackingEnsemble)
        assert reloaded.meta_feature_names_ == ensemble.meta_feature_names_

        X_test, _ = _make_data(n=20, d=3, seed=111)
        np.testing.assert_array_equal(
            ensemble.predict_proba(X_test), reloaded.predict_proba(X_test)
        )

    def test_corrupted_metadata_raises_persistence_error(self, tmp_path: Path) -> None:
        bad_dir = tmp_path / "corrupted"
        bad_dir.mkdir()
        (bad_dir / BaseMLModel.METADATA_FILENAME).write_text("NOT_JSON", encoding="utf-8")

        with pytest.raises(ModelPersistenceError, match="Corrupted metadata"):
            BaseMLModel.load(bad_dir)


class TestEnsembleAdversarialAndBoundary:
    """Verifies edge cases, degenerate data, and adversarial inputs."""

    def test_collinear_identical_models_yield_identical_output(self) -> None:
        X, y = _make_data(n=60, d=3)
        m1 = LogisticRegressionModel(random_state=42)
        m2 = LogisticRegressionModel(random_state=42)
        m3 = LogisticRegressionModel(random_state=42)

        ensemble = AveragingEnsemble(models=[m1, m2, m3])
        ensemble.fit(X, y)

        single = LogisticRegressionModel(random_state=42).fit(X, y)

        np.testing.assert_allclose(
            ensemble.predict_proba(X), single.predict_proba(X), atol=1e-12
        )

    def test_nan_or_inf_inference_raises_validation_error(self) -> None:
        X, y = _make_data(n=50, d=2)
        ensemble = AveragingEnsemble(models=[LogisticRegressionModel()])
        ensemble.fit(X, y)

        X_bad = np.array([[np.nan, 1.0], [0.5, 0.5]])
        with pytest.raises(ModelValidationError, match="NaN or infinite"):
            ensemble.predict_proba(X_bad)

    def test_feature_dimension_mismatch_raises_validation_error(self) -> None:
        X, y = _make_data(n=50, d=3)
        ensemble = AveragingEnsemble(models=[LogisticRegressionModel()])
        ensemble.fit(X, y)

        X_wrong_dim = np.ones((10, 5))
        with pytest.raises(ModelValidationError, match="Feature count mismatch"):
            ensemble.predict_proba(X_wrong_dim)

    def test_clone_base_model_copies_unfitted_state(self) -> None:
        m = LogisticRegressionModel(C=2.5, penalty="l2", solver="lbfgs")
        X, y = _make_data(n=50, d=2)
        m.fit(X, y)
        assert m.is_fitted

        cloned = clone_base_model(m)
        assert not cloned.is_fitted
        assert cloned.hyperparameters["C"] == 2.5
