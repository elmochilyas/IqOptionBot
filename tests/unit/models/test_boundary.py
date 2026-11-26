"""Adversarial boundary and robustness verification tests for Milestone 3.

Stress-tests:
1. Missing dependency detection: LightGBMAdapter, XGBoostAdapter, CatBoostAdapter
   raise ModelDependencyError with actionable message when fallback_to_hist_gbdt=False,
   and do not crash at import time.
2. Fallback mode: fallback_to_hist_gbdt=True executes fit, predict_proba, evaluate,
   and safe serialization/reloading without external library dependencies.
3. Degenerate splits & inputs: single-class targets y in {0}^N or {1}^N in evaluate()
   and fit() return ModelEvaluationMetrics with roc_auc is None without throwing exceptions.
4. Unfitted models: calling predict, predict_proba, evaluate, save, classes_
   prior to fit() cleanly raises NotFittedError.
5. Feature shape mismatch: predicting with mismatched feature dimension raises
   ModelValidationError.
6. Extreme inputs, NaN/Inf guards, and duplicate/invalid feature name contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

ALL_BASE_MODEL_CLASSES: list[type[BaseMLModel]] = [
    MajorityClassModel,
    SeededRandomModel,
    LogisticRegressionModel,
    RuleBasedMomentumModel,
    RandomForestModel,
    HistGradientBoostingModel,
]

ALL_ADAPTER_CLASSES: list[type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter]] = [
    LightGBMAdapter,
    XGBoostAdapter,
    CatBoostAdapter,
]


@pytest.fixture
def clean_synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Provide deterministic binary dataset with 60 samples and 4 features."""
    rng = np.random.default_rng(2026)
    X = rng.normal(0.0, 1.0, size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0.0).astype(np.int64)
    return X, y


# ==============================================================================
# CHALLENGE 1: MISSING DEPENDENCY DETECTION & IMPORT SAFETY
# ==============================================================================


class TestMissingDependenciesAdversarial:
    """Stress-test capability detection and missing dependency handling."""

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
    def test_missing_dependency_is_available_returns_false(
        self,
        adapter_cls: type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter],
    ) -> None:
        """Verify is_available() returns False when package is not installed."""
        assert adapter_cls.is_available() is False

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
    def test_missing_dependency_raises_model_dependency_error(
        self,
        adapter_cls: type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter],
    ) -> None:
        """Verify instantiating with fallback_to_hist_gbdt=False raises ModelDependencyError."""
        with pytest.raises(ModelDependencyError) as exc_info:
            adapter_cls(fallback_to_hist_gbdt=False)

        exc = exc_info.value
        assert isinstance(exc, ImportError)
        assert len(exc.package_name) > 0
        assert len(exc.adapter_name) > 0
        assert "pip install" in exc.pip_install_cmd
        assert exc.package_name in str(exc)
        assert exc.adapter_name in str(exc)
        assert exc.pip_install_cmd in str(exc)

    def test_discovery_helpers_with_missing_dependencies(self) -> None:
        """Verify discovery helpers accurately reflect that no optional GBDT packages exist."""
        assert is_gbdt_available("lightgbm") is False
        assert is_gbdt_available("xgboost") is False
        assert is_gbdt_available("catboost") is False
        assert is_gbdt_available("nonexistent_package") is False
        assert get_available_gbdt_adapters() == {}

    def test_module_import_does_not_crash(self) -> None:
        """Verify importing adapters module is completely safe without optional packages."""
        import sys

        assert "lightgbm" not in sys.modules
        assert "xgboost" not in sys.modules
        assert "catboost" not in sys.modules


# ==============================================================================
# CHALLENGE 2: FALLBACK MODE LIFECYCLE & PERSISTENCE
# ==============================================================================


class TestFallbackModeAdversarial:
    """Stress-test fallback mode with HistGradientBoostingClassifier delegation."""

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
    def test_fallback_instantiation_and_full_lifecycle(
        self,
        adapter_cls: type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify complete train-predict-evaluate-save-reload lifecycle in fallback mode."""
        X, y = clean_synthetic_data
        model = adapter_cls(fallback_to_hist_gbdt=True)

        assert model.fallback_to_hist_gbdt is True
        assert model.is_fitted is False

        # Fit with explicit feature names
        f_names = ["f0", "f1", "f2", "f3"]
        fitted = model.fit(X, y, feature_names=f_names)
        assert fitted is model
        assert model.is_fitted is True
        assert tuple(model.feature_names) == tuple(f_names)
        assert np.array_equal(model.classes_, np.array([0, 1]))

        # Probability predictions
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-6)

        # Label predictions
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        np.testing.assert_array_equal(preds, np.argmax(proba, axis=1))

        # Evaluation metrics
        metrics = model.evaluate(X, y)
        assert isinstance(metrics, ModelEvaluationMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0
        assert metrics.log_loss >= 0.0
        assert metrics.sample_count == len(X)

        # Persistence and reload parity
        save_dir = tmp_path / f"save_{model.name}"
        model.save(save_dir)

        reloaded = adapter_cls.load(save_dir)
        assert reloaded.is_fitted is True
        assert reloaded.feature_names == tuple(f_names)
        assert np.array_equal(reloaded.classes_, np.array([0, 1]))

        reloaded_proba = reloaded.predict_proba(X)
        np.testing.assert_allclose(reloaded_proba, proba, atol=1e-7)


# ==============================================================================
# CHALLENGE 3: DEGENERATE SPLITS & SINGLE-CLASS INPUTS
# ==============================================================================


class TestDegenerateSplitsAdversarial:
    """Stress-test models under degenerate splits (single class labels in y)."""

    @pytest.mark.parametrize("model_cls", ALL_BASE_MODEL_CLASSES)
    def test_evaluate_single_class_target_all_zeros(
        self,
        model_cls: type[BaseMLModel],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify evaluating on y in {0}^N returns roc_auc is None without throwing exceptions."""
        X, y = clean_synthetic_data
        model = model_cls().fit(X, y)

        X_test = X[:25]
        y_zeros = np.zeros(25, dtype=np.int64)

        metrics = model.evaluate(X_test, y_zeros)
        assert metrics.roc_auc is None
        assert metrics.balanced_accuracy == metrics.accuracy
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0
        assert metrics.sample_count == 25
        assert len(metrics.confusion_matrix) == 2

    @pytest.mark.parametrize("model_cls", ALL_BASE_MODEL_CLASSES)
    def test_evaluate_single_class_target_all_ones(
        self,
        model_cls: type[BaseMLModel],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify evaluating on y in {1}^N returns roc_auc is None without throwing exceptions."""
        X, y = clean_synthetic_data
        model = model_cls().fit(X, y)

        X_test = X[:25]
        y_ones = np.ones(25, dtype=np.int64)

        metrics = model.evaluate(X_test, y_ones)
        assert metrics.roc_auc is None
        assert metrics.balanced_accuracy == metrics.accuracy
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0
        assert metrics.sample_count == 25
        assert len(metrics.confusion_matrix) == 2

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
    def test_evaluate_single_class_target_on_fallback_adapters(
        self,
        adapter_cls: type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify fallback adapters handle y in {0}^N and y in {1}^N cleanly."""
        X, y = clean_synthetic_data
        model = adapter_cls(fallback_to_hist_gbdt=True).fit(X, y)

        X_test = X[:20]
        y_zeros = np.zeros(20, dtype=np.int64)
        m0 = model.evaluate(X_test, y_zeros)
        assert m0.roc_auc is None

        y_ones = np.ones(20, dtype=np.int64)
        m1 = model.evaluate(X_test, y_ones)
        assert m1.roc_auc is None

    @pytest.mark.parametrize("model_cls", ALL_BASE_MODEL_CLASSES)
    def test_fit_on_single_class_training_fold(
        self,
        model_cls: type[BaseMLModel],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify models can fit on single-class folds without uncaught crashes."""
        X, _ = clean_synthetic_data
        y_zeros = np.zeros(len(X), dtype=np.int64)

        model = model_cls().fit(X, y_zeros)
        assert model.is_fitted is True

        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-6)

        metrics = model.evaluate(X[:10], y_zeros[:10])
        assert metrics.roc_auc is None


# ==============================================================================
# CHALLENGE 4: UNFITTED MODEL STATE INTEGRITY
# ==============================================================================


class TestUnfittedModelsAdversarial:
    """Stress-test unfitted model method calls and property accessors."""

    @pytest.mark.parametrize("model_cls", ALL_BASE_MODEL_CLASSES)
    def test_unfitted_model_methods_raise_not_fitted_error(
        self,
        model_cls: type[BaseMLModel],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify predict, predict_proba, evaluate, save, classes_ raise NotFittedError."""
        X, y = clean_synthetic_data
        model = model_cls()

        with pytest.raises(NotFittedError):
            model.predict(X)

        with pytest.raises(NotFittedError):
            model.predict_proba(X)

        with pytest.raises(NotFittedError):
            model.evaluate(X, y)

        with pytest.raises(NotFittedError):
            model.save(tmp_path / "fail_save")

        with pytest.raises(NotFittedError):
            _ = model.classes_

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
    def test_unfitted_fallback_adapters_raise_not_fitted_error(
        self,
        adapter_cls: type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify unfitted adapters with fallback raise NotFittedError on inference."""
        X, y = clean_synthetic_data
        model = adapter_cls(fallback_to_hist_gbdt=True)

        with pytest.raises(NotFittedError):
            model.predict(X)

        with pytest.raises(NotFittedError):
            model.predict_proba(X)

        with pytest.raises(NotFittedError):
            model.evaluate(X, y)

        with pytest.raises(NotFittedError):
            model.save(tmp_path / "fail_save")

        with pytest.raises(NotFittedError):
            _ = model.classes_

    def test_random_forest_feature_importances_raises_not_fitted_error(self) -> None:
        """Verify RandomForestModel.feature_importances_ raises NotFittedError before fitting."""
        model = RandomForestModel()
        with pytest.raises(NotFittedError):
            _ = model.feature_importances_


# ==============================================================================
# CHALLENGE 5: FEATURE SHAPE & INPUT VALIDATION GUARDS
# ==============================================================================


class TestFeatureShapeAndValidationAdversarial:
    """Stress-test input validation routines with malformed inputs."""

    @pytest.mark.parametrize("model_cls", ALL_BASE_MODEL_CLASSES)
    def test_feature_dimension_mismatch_raises_model_validation_error(
        self,
        model_cls: type[BaseMLModel],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify predicting with mismatched feature dimension raises ModelValidationError."""
        X, y = clean_synthetic_data  # D = 4
        model = model_cls().fit(X, y)

        rng = np.random.default_rng(999)
        # Test D=1, D=3, D=5, D=10
        for bad_dim in [1, 3, 5, 10]:
            X_bad = rng.normal(size=(20, bad_dim))
            with pytest.raises(ModelValidationError):
                model.predict(X_bad)

            with pytest.raises(ModelValidationError):
                model.predict_proba(X_bad)

            with pytest.raises(ModelValidationError):
                model.evaluate(X_bad, y[:20])

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
    def test_fallback_adapters_feature_dimension_mismatch(
        self,
        adapter_cls: type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify fallback adapters raise ModelValidationError on feature count mismatch."""
        X, y = clean_synthetic_data  # D = 4
        model = adapter_cls(fallback_to_hist_gbdt=True).fit(X, y)

        X_bad = np.ones((10, 2))
        with pytest.raises(ModelValidationError):
            model.predict_proba(X_bad)

    @pytest.mark.parametrize("model_cls", ALL_BASE_MODEL_CLASSES)
    def test_non_2d_feature_array_raises_validation_error(
        self,
        model_cls: type[BaseMLModel],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify 1D or 3D feature arrays are rejected with ModelValidationError."""
        _, y = clean_synthetic_data
        model = model_cls()

        # 1D array
        with pytest.raises(ModelValidationError):
            model.fit(np.ones(40), y[:40])

        # 3D array
        with pytest.raises(ModelValidationError):
            model.fit(np.ones((40, 4, 1)), y[:40])

    @pytest.mark.parametrize("model_cls", ALL_BASE_MODEL_CLASSES)
    def test_nan_or_inf_in_features_raises_validation_error(
        self,
        model_cls: type[BaseMLModel],
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify NaN, Inf, and -Inf in features are rejected."""
        X, y = clean_synthetic_data
        model = model_cls().fit(X, y)

        for bad_val in [np.nan, np.inf, -np.inf]:
            X_bad = X[:10].copy()
            X_bad[0, 0] = bad_val

            with pytest.raises(ModelValidationError):
                model.predict_proba(X_bad)

            with pytest.raises(ModelValidationError):
                model.fit(X_bad, y[:10])

    def test_empty_training_data_raises_validation_error(self) -> None:
        """Verify 0-sample dataset raises ModelValidationError."""
        model = MajorityClassModel()
        with pytest.raises(ModelValidationError):
            model.fit(np.empty((0, 4)), np.empty(0, dtype=np.int64))

    def test_sample_count_mismatch_raises_validation_error(self) -> None:
        """Verify len(X) != len(y) raises ModelValidationError."""
        model = MajorityClassModel()
        with pytest.raises(ModelValidationError):
            model.fit(np.ones((20, 4)), np.ones(15, dtype=np.int64))

    def test_duplicate_feature_names_raises_validation_error(self) -> None:
        """Verify duplicate feature names in fit() raise ModelValidationError."""
        model = MajorityClassModel()
        with pytest.raises(ModelValidationError):
            model.fit(np.ones((10, 3)), np.ones(10, dtype=np.int64), feature_names=["a", "b", "a"])

    def test_feature_name_count_mismatch_raises_validation_error(self) -> None:
        """Verify feature_names length != D raises ModelValidationError."""
        model = MajorityClassModel()
        with pytest.raises(ModelValidationError):
            model.fit(np.ones((10, 3)), np.ones(10, dtype=np.int64), feature_names=["a", "b"])


# ==============================================================================
# CHALLENGE 6: BOUNDARY CONDITIONS & METRIC ROBUSTNESS
# ==============================================================================


class TestMetricBoundaryRobustness:
    """Stress-test metric calculations under extreme probabilities."""

    def test_single_sample_inference_all_models(
        self,
        clean_synthetic_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify inference works reliably on a single observation (1, D)."""
        X, y = clean_synthetic_data
        all_models: list[BaseMLModel] = [
            MajorityClassModel().fit(X, y),
            SeededRandomModel().fit(X, y),
            LogisticRegressionModel().fit(X, y),
            RuleBasedMomentumModel().fit(X, y),
            RandomForestModel().fit(X, y),
            HistGradientBoostingModel().fit(X, y),
            LightGBMAdapter(fallback_to_hist_gbdt=True).fit(X, y),
            XGBoostAdapter(fallback_to_hist_gbdt=True).fit(X, y),
            CatBoostAdapter(fallback_to_hist_gbdt=True).fit(X, y),
        ]

        X_single = X[:1]
        for m in all_models:
            proba = m.predict_proba(X_single)
            pred = m.predict(X_single)
            assert proba.shape == (1, 2)
            assert pred.shape == (1,)
            assert pred[0] in [0, 1]
            np.testing.assert_allclose(np.sum(proba[0]), 1.0, atol=1e-6)

    def test_metrics_dataclass_boundary_validations(self) -> None:
        """Verify ModelEvaluationMetrics rejects invalid out-of-bound values."""
        valid_kwargs: dict[str, Any] = {
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "log_loss": 0.693,
            "brier_score": 0.25,
            "roc_auc": 0.5,
            "f1_macro": 0.5,
            "confusion_matrix": [[10, 10], [10, 10]],
            "sample_count": 40,
        }

        # Negative accuracy
        with pytest.raises(ValueError, match="accuracy must be in"):
            ModelEvaluationMetrics(**{**valid_kwargs, "accuracy": -0.1})

        # Negative log loss
        with pytest.raises(ValueError, match="log_loss must be >= 0"):
            ModelEvaluationMetrics(**{**valid_kwargs, "log_loss": -0.01})

        # Brier score > 1.0
        with pytest.raises(ValueError, match="brier_score must be in"):
            ModelEvaluationMetrics(**{**valid_kwargs, "brier_score": 1.5})

        # Negative sample count
        with pytest.raises(ValueError, match="sample_count must be an integer >= 0"):
            ModelEvaluationMetrics(**{**valid_kwargs, "sample_count": -1})
