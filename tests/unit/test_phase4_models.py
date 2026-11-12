"""Phase 4.3 Unit Tests: Model Zoo, Common Interface, Baselines, Tabular Models, and Persistence.

Covers:
- Common BaseMLModel interface contract: fit, predict, predict_proba, evaluate, save, load
- Mathematical probability invariants: shape (N, 2), range [0, 1], row sum == 1.0, finite values
- Prediction consistency: predict(X) matches argmax(predict_proba(X))
- Exact baseline behavior:
    * MajorityClassModel: exact empirical class frequencies verified against ground truth priors
    * SeededRandomModel: 100% bit-for-bit reproducibility for same seed, strict seed divergence
    * RuleBasedMomentumModel: monotonic probability response to directional momentum indicators
- Non-linear tabular model learning verification on synthetic separable data:
    * LogisticRegression, RandomForest, HistGradientBoosting (acc >= 0.95, ROC-AUC >= 0.98)
- Capability detection:
    * LightGBMAdapter, XGBoostAdapter, CatBoostAdapter report is_available() == False
    * Raising informative ModelDependencyError with pip installation advice
    * Transparent fallback to HistGradientBoostingModel when fallback_to_hist_gbdt=True
- Safe persistence roundtrip:
    * Saving and reloading reproduces 100% bit-for-bit identical probabilities
    * Complete preservation of hyperparameters, feature names, version, and classes_
- ModelEvaluationMetrics calculation on analytically known distributions:
    * Perfect predictions, inverted predictions, uninformative 50/50 predictions
    * Graceful handling of single-class evaluation data (roc_auc returns None without error)
- Edge case handling & error robustness:
    * Single sample predict and predict_proba
    * Single class training data
    * Accessing unfitted models raises NotFittedError
    * Feature count dimension mismatch raises ModelValidationError
    * Empty training data or length mismatches raise ModelValidationError
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from iqoption_research.ml.models.adapters import (
    CatBoostAdapter,
    LightGBMAdapter,
    ModelDependencyError,
    XGBoostAdapter,
    get_available_gbdt_adapters,
    is_gbdt_available,
)
from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelEvaluationMetrics,
    ModelPersistenceError,
    ModelValidationError,
    NotFittedError,
)
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
    RuleBasedMomentumModel,
    SeededRandomModel,
)
from iqoption_research.ml.models.tabular import (
    HistGradientBoostingModel,
    RandomForestModel,
)

ALL_STANDARD_MODEL_CLASSES: list[type[BaseMLModel]] = [
    MajorityClassModel,
    SeededRandomModel,
    LogisticRegressionModel,
    RuleBasedMomentumModel,
    RandomForestModel,
    HistGradientBoostingModel,
]

TABULAR_LEARNING_CLASSES: list[type[BaseMLModel]] = [
    LogisticRegressionModel,
    RandomForestModel,
    HistGradientBoostingModel,
]

ALL_ADAPTER_CLASSES: list[type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter]] = [
    LightGBMAdapter,
    XGBoostAdapter,
    CatBoostAdapter,
]


@pytest.fixture
def synthetic_binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic classification dataset with 100 samples and 4 features."""
    rng = np.random.default_rng(42)
    X = rng.normal(loc=0.0, scale=1.0, size=(100, 4))
    score = X[:, 0] * 1.5 - X[:, 1] * 0.8 + rng.normal(0, 0.2, size=100)
    y = (score > 0.0).astype(np.int64)
    return X, y


@pytest.fixture
def separable_binary_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Easily separable binary dataset for verifying learning capability (train and test splits)."""
    rng = np.random.default_rng(101)
    n_per_class = 100
    X_0 = rng.normal(loc=-2.0, scale=0.4, size=(n_per_class, 4))
    y_0 = np.zeros(n_per_class, dtype=np.int64)

    X_1 = rng.normal(loc=2.0, scale=0.4, size=(n_per_class, 4))
    y_1 = np.ones(n_per_class, dtype=np.int64)

    X = np.vstack([X_0, X_1])
    y = np.concatenate([y_0, y_1])

    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    split_idx = 140
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


# ==============================================================================
# SECTION 1: COMMON MODEL INTERFACE CONTRACT TESTS
# ==============================================================================


class TestModelCommonContract:
    """Verifies that all models in the zoo strictly conform to the BaseMLModel contract."""

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_model_subclass_hierarchy(self, model_cls: type[BaseMLModel]) -> None:
        """Verify model class inherits from BaseMLModel abstract base class."""
        assert issubclass(model_cls, BaseMLModel)

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_unfitted_initial_state(self, model_cls: type[BaseMLModel]) -> None:
        """Verify initial attribute values on unfitted model instances."""
        model = model_cls()
        assert model.is_fitted is False
        assert isinstance(model.name, str) and len(model.name) > 0
        assert isinstance(model.version, str) and len(model.version) > 0
        assert isinstance(model.model_type, str) and len(model.model_type) > 0
        assert isinstance(model.hyperparameters, dict)
        assert model.feature_names == ()

        with pytest.raises(NotFittedError):
            _ = model.classes_

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_fit_returns_self_and_updates_state(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify fit() returns self for fluent chaining and updates fitted state."""
        X, y = synthetic_binary_data
        model = model_cls()
        f_names = ["feat_0", "feat_1", "feat_2", "feat_3"]
        fitted = model.fit(X, y, feature_names=f_names)

        assert fitted is model
        assert model.is_fitted is True
        assert tuple(model.feature_names) == tuple(f_names)
        assert np.array_equal(model.classes_, np.array([0, 1]))

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_fit_with_anonymous_features_synthesizes_default_names(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify fit() without explicit feature names generates default feature names."""
        X, y = synthetic_binary_data
        model = model_cls()
        model.fit(X, y)

        assert len(model.feature_names) == X.shape[1]
        assert all(isinstance(f, str) for f in model.feature_names)
        assert model.is_fitted is True


# ==============================================================================
# SECTION 2: PROBABILITY PROPERTIES & MATHEMATICAL AXIOMS
# ==============================================================================


class TestProbabilityProperties:
    """Verifies that predicted probability distributions conform to mathematical axioms."""

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_probability_shape_and_dtype(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify predict_proba() returns 2D float array of shape (N, 2)."""
        X, y = synthetic_binary_data
        model = model_cls().fit(X, y)
        proba = model.predict_proba(X[:30])

        assert proba.ndim == 2
        assert proba.shape == (30, 2)
        assert np.issubdtype(proba.dtype, np.floating)

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_probability_bounds_in_zero_to_one(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify all probability values lie strictly within [0.0, 1.0]."""
        X, y = synthetic_binary_data
        model = model_cls().fit(X, y)
        proba = model.predict_proba(X)

        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_probability_rows_sum_to_one(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify each row sums to 1.0 within numerical tolerance of 1e-6."""
        X, y = synthetic_binary_data
        model = model_cls().fit(X, y)
        proba = model.predict_proba(X)

        row_sums = np.sum(proba, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_probability_finiteness_no_nan_no_inf(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify predicted probabilities contain zero NaNs and zero infinite values."""
        X, y = synthetic_binary_data
        model = model_cls().fit(X, y)
        proba = model.predict_proba(X)

        assert np.all(np.isfinite(proba))


# ==============================================================================
# SECTION 3: PREDICTION CONSISTENCY
# ==============================================================================


class TestPredictionConsistency:
    """Verifies that directional label predictions match the argmax of predict_proba."""

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_predict_matches_argmax_predict_proba(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify predict(X) is exactly classes_[argmax(predict_proba(X))]."""
        X, y = synthetic_binary_data
        model = model_cls().fit(X, y)
        preds = model.predict(X)
        proba = model.predict_proba(X)

        assert preds.shape == (len(X),)
        assert np.issubdtype(preds.dtype, np.integer)
        expected = model.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(preds, expected)


# ==============================================================================
# SECTION 4: MAJORITY CLASS MODEL SPECIFIC BEHAVIOR
# ==============================================================================


class TestMajorityClassModel:
    """Verifies exact empirical frequency math and edge cases for MajorityClassModel."""

    def test_exact_empirical_frequencies_70_30(self) -> None:
        """Verify exact prior frequency calculation for 70% ones and 30% zeros."""
        X = np.ones((100, 4))
        y = np.array([1] * 70 + [0] * 30, dtype=np.int64)
        model = MajorityClassModel().fit(X, y)

        X_test = np.zeros((25, 4))
        proba = model.predict_proba(X_test)
        preds = model.predict(X_test)

        np.testing.assert_allclose(proba[:, 0], 0.30, atol=1e-9)
        np.testing.assert_allclose(proba[:, 1], 0.70, atol=1e-9)
        assert np.all(preds == 1)

    def test_exact_empirical_frequencies_minority_positive(self) -> None:
        """Verify exact prior frequency calculation for 15% ones and 85% zeros."""
        X = np.ones((100, 4))
        y = np.array([1] * 15 + [0] * 85, dtype=np.int64)
        model = MajorityClassModel().fit(X, y)

        X_test = np.zeros((20, 4))
        proba = model.predict_proba(X_test)
        preds = model.predict(X_test)

        np.testing.assert_allclose(proba[:, 0], 0.85, atol=1e-9)
        np.testing.assert_allclose(proba[:, 1], 0.15, atol=1e-9)
        assert np.all(preds == 0)

    def test_single_class_training_ones(self) -> None:
        """Verify MajorityClassModel handles 100% positive class without crashing."""
        X = np.ones((50, 4))
        y = np.ones(50, dtype=np.int64)
        model = MajorityClassModel().fit(X, y)

        proba = model.predict_proba(np.zeros((10, 4)))
        preds = model.predict(np.zeros((10, 4)))

        np.testing.assert_allclose(proba[:, 1], 1.0, atol=1e-9)
        np.testing.assert_allclose(proba[:, 0], 0.0, atol=1e-9)
        assert np.all(preds == 1)

    def test_single_class_training_zeros(self) -> None:
        """Verify MajorityClassModel handles 100% negative class without crashing."""
        X = np.ones((50, 4))
        y = np.zeros(50, dtype=np.int64)
        model = MajorityClassModel().fit(X, y)

        proba = model.predict_proba(np.zeros((10, 4)))
        preds = model.predict(np.zeros((10, 4)))

        np.testing.assert_allclose(proba[:, 0], 1.0, atol=1e-9)
        np.testing.assert_allclose(proba[:, 1], 0.0, atol=1e-9)
        assert np.all(preds == 0)

    def test_laplace_smoothing(self) -> None:
        """Verify Laplace smoothing adjusts empirical probabilities correctly."""
        X = np.ones((100, 4))
        y = np.array([1] * 70 + [0] * 30, dtype=np.int64)
        model = MajorityClassModel(smoothing=1.0).fit(X, y)

        proba = model.predict_proba(np.zeros((5, 4)))
        expected_p1 = 71.0 / 102.0
        expected_p0 = 31.0 / 102.0
        np.testing.assert_allclose(proba[:, 1], expected_p1, atol=1e-6)
        np.testing.assert_allclose(proba[:, 0], expected_p0, atol=1e-6)


# ==============================================================================
# SECTION 5: SEEDED RANDOM MODEL REPRODUCIBILITY
# ==============================================================================


class TestSeededRandomModel:
    """Verifies deterministic reproducibility and seed divergence for SeededRandomModel."""

    def test_exact_reproducibility_same_seed(
        self,
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify two instances with same seed produce bit-for-bit identical probabilities."""
        X, y = synthetic_binary_data
        m1 = SeededRandomModel(seed=42).fit(X, y)
        m2 = SeededRandomModel(seed=42).fit(X, y)

        p1 = m1.predict_proba(X)
        p2 = m2.predict_proba(X)

        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))

    def test_divergence_across_different_seeds(
        self,
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify models with different seeds produce distinct probability series."""
        X, y = synthetic_binary_data
        m1 = SeededRandomModel(seed=42).fit(X, y)
        m2 = SeededRandomModel(seed=999).fit(X, y)

        p1 = m1.predict_proba(X)
        p2 = m2.predict_proba(X)

        assert not np.array_equal(p1, p2)
        assert np.max(np.abs(p1 - p2)) > 0.05

    def test_repeated_calls_on_same_instance(
        self,
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify calling predict_proba repeatedly on same instance yields identical output."""
        X, y = synthetic_binary_data
        m = SeededRandomModel(seed=123).fit(X, y)

        run1 = m.predict_proba(X)
        run2 = m.predict_proba(X)

        np.testing.assert_array_equal(run1, run2)


# ==============================================================================
# SECTION 6: RULE-BASED MOMENTUM MODEL SPECIFIC BEHAVIOR
# ==============================================================================


class TestRuleBasedMomentumModel:
    """Verifies indicator mapping and monotonicity for RuleBasedMomentumModel."""

    def test_step_mapping_threshold(self) -> None:
        """Verify piecewise step mapping emits base_prob or 1 - base_prob across threshold."""
        X = np.array([[-1.5], [0.0], [1.5]])
        y = np.array([0, 0, 1], dtype=np.int64)
        model = RuleBasedMomentumModel(
            momentum_feature="f0",
            threshold=0.0,
            base_prob=0.65,
            mapping_type="step",
        ).fit(X, y, feature_names=["f0"])

        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba[0], [0.65, 0.35], atol=1e-6)
        np.testing.assert_allclose(proba[1], [0.50, 0.50], atol=1e-6)
        np.testing.assert_allclose(proba[2], [0.35, 0.65], atol=1e-6)

    def test_sigmoid_mapping_monotonicity(self) -> None:
        """Verify sigmoid mapping produces strictly monotonic directional probabilities."""
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        y = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        model = RuleBasedMomentumModel(
            momentum_feature="ret",
            threshold=0.0,
            mapping_type="sigmoid",
            sigmoid_steepness=2.0,
        ).fit(X, y, feature_names=["ret"])

        proba = model.predict_proba(X)
        p_up = proba[:, 1]

        assert p_up[0] < p_up[1] < p_up[2] < p_up[3] < p_up[4]
        np.testing.assert_allclose(p_up[2], 0.5, atol=1e-5)

    def test_feature_name_lookup_and_missing_error(self) -> None:
        """Verify momentum feature column is resolved by name or raises error if missing."""
        X = np.array([[10.0, -1.0], [20.0, 1.0]])
        y = np.array([0, 1], dtype=np.int64)

        model = RuleBasedMomentumModel(momentum_feature="mom").fit(
            X, y, feature_names=["vol", "mom"]
        )
        proba = model.predict_proba(X)
        assert proba[0, 1] < 0.5 < proba[1, 1]

        with pytest.raises(ModelValidationError):
            RuleBasedMomentumModel(momentum_feature="missing_feat").fit(
                X, y, feature_names=["vol", "mom"]
            )


# ==============================================================================
# SECTION 7: TABULAR MODEL LEARNING VERIFICATION
# ==============================================================================


class TestTabularLearning:
    """Verifies that non-linear and regularized tabular models actually learn patterns."""

    @pytest.mark.parametrize("model_cls", TABULAR_LEARNING_CLASSES)
    def test_learning_verification_separable_dataset(
        self,
        model_cls: type[BaseMLModel],
        separable_binary_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Verify model achieves high accuracy and low loss on easily separable synthetic data."""
        X_train, y_train, X_test, y_test = separable_binary_data
        model = model_cls().fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        assert metrics.accuracy >= 0.95
        assert metrics.balanced_accuracy >= 0.95
        assert metrics.roc_auc is not None and metrics.roc_auc >= 0.98
        assert metrics.brier_score <= 0.05
        assert metrics.log_loss <= 0.25
        assert metrics.f1_macro >= 0.95
        assert metrics.sample_count == len(X_test)

    def test_random_forest_feature_importances(
        self,
        separable_binary_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Verify RandomForestModel provides valid non-negative feature importances."""
        X_train, y_train, _, _ = separable_binary_data
        model = RandomForestModel().fit(X_train, y_train)

        importances = model.feature_importances_
        assert importances is not None
        assert len(importances) == X_train.shape[1]
        assert np.all(importances >= 0.0)
        np.testing.assert_allclose(np.sum(importances), 1.0, atol=1e-5)


# ==============================================================================
# SECTION 8: CAPABILITY DETECTION & GBDT ADAPTERS
# ==============================================================================


class TestCapabilityDetection:
    """Verifies capability detection and safe failure modes for optional GBDT adapters."""

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
    def test_is_available_returns_bool_without_throwing(
        self,
        adapter_cls: type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter],
    ) -> None:
        """Verify is_available() returns a boolean and does not raise unhandled exceptions."""
        result = adapter_cls.is_available()
        assert isinstance(result, bool)

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
    def test_unavailable_adapter_raises_model_dependency_error(
        self,
        adapter_cls: type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter],
    ) -> None:
        """Verify instantiating uninstalled adapter raises informative ModelDependencyError."""
        if not adapter_cls.is_available():
            with pytest.raises(ModelDependencyError) as exc_info:
                adapter_cls(fallback_to_hist_gbdt=False)
            msg = str(exc_info.value).lower()
            assert "pip install" in msg

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
    def test_fallback_to_hist_gbdt_mode(
        self,
        adapter_cls: type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify adapter with fallback_to_hist_gbdt=True succeeds in fitting and predicting."""
        X, y = synthetic_binary_data
        model = adapter_cls(fallback_to_hist_gbdt=True)
        fitted = model.fit(X, y)

        assert fitted is model
        assert model.is_fitted is True
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-6)

    def test_discovery_helper_functions(self) -> None:
        """Verify get_available_gbdt_adapters and is_gbdt_available helpers."""
        available_dict = get_available_gbdt_adapters()
        assert isinstance(available_dict, dict)
        for name in ["lightgbm", "xgboost", "catboost"]:
            avail = is_gbdt_available(name)
            assert isinstance(avail, bool)


# ==============================================================================
# SECTION 9: SAFE PERSISTENCE ROUNDTRIP & RELOAD PARITY
# ==============================================================================


class TestModelPersistence:
    """Verifies that saving and reloading models yields 100% bit-for-bit probability parity."""

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_persistence_roundtrip_bit_for_bit_parity(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify saved model reloads with 100% bit-for-bit identical probabilities and metadata."""
        X, y = synthetic_binary_data
        f_names = ["f0", "f1", "f2", "f3"]
        model = model_cls().fit(X, y, feature_names=f_names)

        X_eval = X[:30]
        proba_before = model.predict_proba(X_eval)
        pred_before = model.predict(X_eval)

        save_dir = tmp_path / f"saved_{model.name}"
        out_path = model.save(save_dir)
        assert out_path.exists()
        assert (out_path / "metadata.json").exists()

        reloaded = model_cls.load(save_dir)
        proba_after = reloaded.predict_proba(X_eval)
        pred_after = reloaded.predict(X_eval)

        np.testing.assert_array_equal(proba_before, proba_after)
        np.testing.assert_array_equal(pred_before, pred_after)

        assert reloaded.name == model.name
        assert reloaded.version == model.version
        assert reloaded.model_type == model.model_type
        assert reloaded.feature_names == model.feature_names
        assert reloaded.is_fitted is True
        np.testing.assert_array_equal(reloaded.classes_, model.classes_)

    def test_top_level_base_model_load_delegation(
        self,
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify BaseMLModel.load(dir) inspects metadata and reconstructs the correct subclass."""
        X, y = synthetic_binary_data
        model = RandomForestModel().fit(X, y)
        save_dir = tmp_path / "rf_factory_saved"
        model.save(save_dir)

        reloaded = BaseMLModel.load(save_dir)
        assert isinstance(reloaded, RandomForestModel)
        np.testing.assert_array_equal(model.predict_proba(X[:10]), reloaded.predict_proba(X[:10]))

    def test_corrupted_directory_handling(self, tmp_path: Path) -> None:
        """Verify attempting to load from an empty directory raises ModelPersistenceError."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        with pytest.raises((ModelPersistenceError, OSError)):
            BaseMLModel.load(empty_dir)


# ==============================================================================
# SECTION 10: METRICS EVALUATION & ANALYTICALLY KNOWN DISTRIBUTIONS
# ==============================================================================


class TestMetricsEvaluation:
    """Verifies ModelEvaluationMetrics calculation against known analytical distributions."""

    def test_perfect_predictions_metrics(self) -> None:
        """Verify metric values when predictions perfectly match ground truth."""
        X_sep = np.array([[-10.0, 0.0], [10.0, 0.0]])
        y_sep = np.array([0, 1], dtype=np.int64)
        model = LogisticRegressionModel().fit(X_sep, y_sep)
        metrics = model.evaluate(X_sep, y_sep)

        assert metrics.accuracy == 1.0
        assert metrics.balanced_accuracy == 1.0
        assert metrics.brier_score < 0.01
        assert metrics.log_loss < 0.05
        assert metrics.roc_auc == 1.0
        assert metrics.f1_macro == 1.0
        assert metrics.confusion_matrix == [[1, 0], [0, 1]]
        assert metrics.sample_count == 2

    def test_uninformative_50_50_predictions(self) -> None:
        """Verify Brier score and log loss for uninformative 50/50 predictions."""
        X = np.ones((100, 2))
        y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        model = MajorityClassModel().fit(X, y)
        metrics = model.evaluate(X, y)

        np.testing.assert_allclose(metrics.brier_score, 0.25, atol=1e-5)
        np.testing.assert_allclose(metrics.log_loss, math.log(2.0), atol=1e-5)
        assert metrics.roc_auc == 0.5

    def test_single_class_evaluation_data_sets_roc_auc_to_none(
        self,
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify evaluating on a single-class slice sets roc_auc=None without error."""
        X, y = synthetic_binary_data
        model = LogisticRegressionModel().fit(X, y)

        X_single = X[:20]
        y_single_ones = np.ones(20, dtype=np.int64)
        metrics_ones = model.evaluate(X_single, y_single_ones)

        assert metrics_ones.roc_auc is None
        assert isinstance(metrics_ones.accuracy, float)
        assert isinstance(metrics_ones.log_loss, float)
        assert isinstance(metrics_ones.brier_score, float)

        y_single_zeros = np.zeros(20, dtype=np.int64)
        metrics_zeros = model.evaluate(X_single, y_single_zeros)
        assert metrics_zeros.roc_auc is None

    def test_metrics_dataclass_immutability_and_serialization(self) -> None:
        """Verify ModelEvaluationMetrics is frozen and roundtrips through to_dict/from_dict."""
        metrics = ModelEvaluationMetrics(
            accuracy=0.85,
            balanced_accuracy=0.84,
            log_loss=0.35,
            brier_score=0.12,
            roc_auc=0.91,
            f1_macro=0.84,
            confusion_matrix=[[40, 10], [5, 45]],
            sample_count=100,
        )

        with pytest.raises(Exception):
            metrics.accuracy = 0.90  # type: ignore[misc]

        data = metrics.to_dict()
        assert isinstance(data, dict)
        restored = ModelEvaluationMetrics.from_dict(data)
        assert restored == metrics


# ==============================================================================
# SECTION 11: ROBUST EDGE CASES & ERROR HANDLING
# ==============================================================================


class TestModelEdgeCases:
    """Verifies defensive validation for abnormal inputs and edge conditions."""

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_unfitted_model_access_raises_not_fitted_error(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify calling inference or persistence methods before fit() raises NotFittedError."""
        X, y = synthetic_binary_data
        model = model_cls()

        with pytest.raises(NotFittedError):
            model.predict(X)

        with pytest.raises(NotFittedError):
            model.predict_proba(X)

        with pytest.raises(NotFittedError):
            model.evaluate(X, y)

        with pytest.raises(NotFittedError):
            model.save(tmp_path / "save")

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_single_sample_predict_and_proba(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify inference works on a single observation of shape (1, D)."""
        X, y = synthetic_binary_data
        model = model_cls().fit(X, y)
        X_single = X[:1]

        proba = model.predict_proba(X_single)
        preds = model.predict(X_single)

        assert proba.shape == (1, 2)
        assert preds.shape == (1,)
        assert preds[0] in [0, 1]
        np.testing.assert_allclose(np.sum(proba[0]), 1.0, atol=1e-6)

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_feature_dimension_mismatch_raises_validation_error(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify predict on input with mismatched feature dimension raises ModelValidationError."""
        X, y = synthetic_binary_data
        model = model_cls().fit(X, y)

        X_mismatch_fewer = np.zeros((10, X.shape[1] - 1))
        with pytest.raises(ModelValidationError):
            model.predict_proba(X_mismatch_fewer)

        X_mismatch_more = np.zeros((10, X.shape[1] + 2))
        with pytest.raises(ModelValidationError):
            model.predict_proba(X_mismatch_more)

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_empty_training_data_raises_validation_error(
        self,
        model_cls: type[BaseMLModel],
    ) -> None:
        """Verify fit() on empty arrays raises ModelValidationError."""
        X_empty = np.empty((0, 4))
        y_empty = np.empty((0,), dtype=np.int64)

        with pytest.raises(ModelValidationError):
            model_cls().fit(X_empty, y_empty)

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_sample_count_mismatch_raises_validation_error(
        self,
        model_cls: type[BaseMLModel],
    ) -> None:
        """Verify fit() when len(X) != len(y) raises ModelValidationError."""
        X = np.zeros((20, 4))
        y = np.zeros((15,), dtype=np.int64)

        with pytest.raises(ModelValidationError):
            model_cls().fit(X, y)

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_1d_feature_array_raises_validation_error(
        self,
        model_cls: type[BaseMLModel],
    ) -> None:
        """Verify passing a 1D array as X raises ModelValidationError."""
        X_1d = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1])

        with pytest.raises(ModelValidationError):
            model_cls().fit(X_1d, y)

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_nan_or_inf_in_features_raises_validation_error(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify passing NaNs or infinite values in X raises ModelValidationError."""
        X, y = synthetic_binary_data
        X_nan = X.copy()
        X_nan[5, 2] = np.nan

        with pytest.raises(ModelValidationError):
            model_cls().fit(X_nan, y)

        X_inf = X.copy()
        X_inf[3, 1] = np.inf
        with pytest.raises(ModelValidationError):
            model_cls().fit(X_inf, y)

    @pytest.mark.parametrize("model_cls", ALL_STANDARD_MODEL_CLASSES)
    def test_single_class_training_fold_graceful_handling(
        self,
        model_cls: type[BaseMLModel],
        synthetic_binary_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify single class training folds are handled gracefully without crashing."""
        X, _ = synthetic_binary_data
        y_all_ones = np.ones(len(X), dtype=np.int64)

        model = model_cls().fit(X, y_all_ones)
        assert model.is_fitted is True

        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-6)
