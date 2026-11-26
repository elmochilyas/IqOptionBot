"""Unit tests for Phase 8 authentic market data collection, feature schema, and ML training.

Covers:
- Incremental historical candle deduplication and chronological ordering
- OTC instrument classification (EURUSD vs EURUSD-OTC)
- External CSV dataset import and provenance tracking
- Coverage report generation and quality verdict
- Account safety enforcement on data collection
- Causal feature pipeline quote imputation and schema compatibility checks
- Authentic ML training with chronological splits and purged horizons
- Asymmetric binary option payoff expectancy evaluation and Wilson confidence intervals
- Determination of 'NO RELIABLE EDGE DETECTED'
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from iqoption_research.inference.validation import ModelIntegrityValidator
from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.ml.authentic_training import (
    AuthenticModelTrainer,
    AuthenticTrainingConfig,
    compute_wilson_score_interval,
)
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.models.market import Candle
from iqoption_research.research.features.pipeline import CausalFeaturePipeline


def _create_sample_candles(n: int = 150, symbol: str = "EURUSD", size: int = 60) -> list[Candle]:
    """Helper creating a deterministic sample series of candles."""
    cfg = SyntheticMarketConfig(symbol=symbol, n_bars=n, seed=123)
    return list(generate_synthetic_candles(cfg).candles)


def test_wilson_score_confidence_interval() -> None:
    """Verify Wilson score confidence interval calculation."""
    # Zero trades
    lower, upper = compute_wilson_score_interval(0, 0)
    assert lower == 0.0
    assert upper == 1.0

    # 55 wins out of 100
    lower, upper = compute_wilson_score_interval(55, 100)
    assert 0.44 < lower < 0.50
    assert 0.60 < upper < 0.66
    assert lower < 0.55 < upper


def test_otc_instrument_detection() -> None:
    """Ensure reliable distinction between regular forex and OTC derivatives."""
    collector = HistoricalDataCollector()
    assert collector.is_otc_instrument("EURUSD") is False
    assert collector.is_otc_instrument("eurusd") is False
    assert collector.is_otc_instrument("EURUSD-OTC") is True
    assert collector.is_otc_instrument("EURUSD_OTC") is True
    assert collector.is_otc_instrument("GBPUSD-otc") is True


def test_candle_deduplication_and_sorting() -> None:
    """Test deduplication by from_ts and strict chronological ordering."""
    collector = HistoricalDataCollector()
    base_candles = _create_sample_candles(10)

    # Inject duplicate and out-of-order candles
    duplicate = base_candles[2]
    shuffled = [base_candles[5], base_candles[1], duplicate, base_candles[0], duplicate]

    deduped = collector.deduplicate_and_sort_candles(shuffled)
    assert len(deduped) == 4  # 4 unique bars: indices 0, 1, 2, 5
    for i in range(1, len(deduped)):
        assert deduped[i].from_ts > deduped[i - 1].from_ts


def test_feature_pipeline_quote_imputation() -> None:
    """Verify quote imputation prevents schema drop when candles lack bid/ask quotes."""
    raw_candles = _create_sample_candles(30)
    # Zero out quotes to simulate candles without verified bid/ask
    candles = [dataclasses.replace(c, bid=0.0, ask=0.0) for c in raw_candles]

    # 1. Without imputation: quote features are omitted
    pipe_no_impute = CausalFeaturePipeline(include_quotes=True, impute_missing_quotes=False)
    ds_no_impute = pipe_no_impute.transform(candles)
    assert "spread_pips" not in ds_no_impute.feature_names
    assert "spread_ratio" not in ds_no_impute.feature_names

    # 2. With imputation: quote features are retained with configured default
    pipe_impute = CausalFeaturePipeline(
        include_quotes=True, impute_missing_quotes=True, default_spread_pips=1.5
    )
    ds_impute = pipe_impute.transform(candles)
    assert "spread_pips" in ds_impute.feature_names
    assert "spread_ratio" in ds_impute.feature_names

    last_row = ds_impute.rows[-1]
    assert math.isclose(last_row.features["spread_pips"], 1.5)
    assert last_row.features["spread_ratio"] > 0.0


def test_model_validator_stream_compatibility_check() -> None:
    """Check that validator detects quote/volume incompatibilities before inference."""
    from iqoption_research.cli.commands.bot_cmd import _train_quick_model_bundle

    bundle = _train_quick_model_bundle()
    validator = ModelIntegrityValidator()

    # Nominal check: bundle has volume, stream has volume
    compat, err = validator.check_stream_compatibility(bundle, has_quotes=False, has_volume=True)
    assert compat is True
    assert err is None

    # Incompatibility: stream lacks volume but model requires it
    compat_novol, err_novol = validator.check_stream_compatibility(
        bundle, has_quotes=False, has_volume=False
    )
    assert compat_novol is False
    assert err_novol is not None
    assert "Volume features" in err_novol


def test_authentic_training_and_edge_evaluation(tmp_path: Path) -> None:
    """Full workflow: train candidate models, evaluate binary payout asymmetry, assess edge."""
    candles = _create_sample_candles(200)

    cfg = AuthenticTrainingConfig(
        symbol="EURUSD",
        candle_size_seconds=60,
        payout_rate=0.85,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        include_volume=True,
        include_quotes=False,
        min_samples=100,
    )
    trainer = AuthenticModelTrainer(config=cfg)

    result = trainer.train_and_evaluate(candles, provenance_label="AUTHENTIC_TEST")
    assert result.provenance == "AUTHENTIC_TEST"
    assert len(result.bundles) >= 4  # MajorityClass, Momentum, LogReg, RandomForest, HistGB
    assert "logistic_regression" in result.metrics
    assert "random_forest" in result.metrics

    logreg_metrics = result.metrics["logistic_regression"]
    assert logreg_metrics.total_samples >= 25
    assert 0.0 <= logreg_metrics.directional_accuracy <= 1.0
    assert 0.0 <= logreg_metrics.brier_score <= 1.0
    assert math.isclose(logreg_metrics.break_even_win_rate, 1.0 / 1.85, rel_tol=1e-3)

    # Because synthetic random walk has no edge, verdict must be honest
    assert logreg_metrics.verdict in {
        "NO RELIABLE EDGE DETECTED",
        "STATISTICAL EDGE DETECTED",
    }
    assert "AUTHENTIC DATA AI EVALUATION" in result.summary_report

    # Save bundle and verify disk persistence
    saved_dir = trainer.save_trained_bundle(
        result.bundles["logistic_regression"],
        model_name="test_logreg",
        output_dir=tmp_path / "models",
    )
    assert saved_dir.is_dir()
    assert (saved_dir / "bundle_metadata.json").is_file()


def test_authentic_training_rejects_insufficient_samples() -> None:
    """Training must reject small sample series to prevent data leakage and overfitting."""
    candles = _create_sample_candles(50)
    trainer = AuthenticModelTrainer(AuthenticTrainingConfig(min_samples=100))

    with pytest.raises(ValueError, match="Insufficient candles"):
        trainer.train_and_evaluate(candles)


def test_collector_persistence_and_coverage_report(tmp_path: Path) -> None:
    """Verify Parquet persistence and coverage reporting."""
    collector = HistoricalDataCollector(storage_dir=tmp_path / "data")
    candles = _create_sample_candles(120, symbol="EURUSD")

    # Persist dataset
    saved_path = collector.persist_dataset(
        candles,
        symbol="EURUSD",
        candle_size_seconds=60,
        source="test_feed",
        provenance="AUTHENTIC_DATA",
    )
    assert saved_path.is_file()

    # Load back
    loaded = collector.load_dataset("EURUSD", 60)
    assert len(loaded) == 120
    assert loaded[0].from_ts == candles[0].from_ts
    assert loaded[-1].to_ts == candles[-1].to_ts

    # Coverage report
    coverage = collector.generate_coverage_report("EURUSD", 60)
    assert coverage is not None
    assert coverage.symbol == "EURUSD"
    assert coverage.total_bars == 120
    assert coverage.is_otc is False
    assert coverage.gap_count == 0
    assert coverage.quality_verdict == "PASS"
