"""Source-level target audit. No fitting, candle loading or pickle deserialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iqoption_research.ml.tracking.artifacts import verify_manifest
from iqoption_research.research.phase28_audit import file_sha


def target_audit(model: Path) -> dict[str, Any]:
    manifest = verify_manifest(model)
    metadata = json.loads((model / "bundle_metadata.json").read_text(encoding="utf-8"))
    source = Path(__file__).parents[1]
    traces = {
        "phase13_candidate": "ml/phase13.py",
        "phase10_labels": "ml/phase10.py",
        "generic_targets": "ml/targets/generator.py",
        "phase14_development": "ml/phase14.py",
        "phase15_opportunities": "ml/phase15.py",
        "phase24_25_development": "research/phase24.py",
        "observer": "research/prospective.py",
    }
    return {
        "schema_version": 1,
        "model_manifest_sha256": manifest.digest(),
        "declared_lineage": metadata.get("metadata", {}),
        "source_sha256": {name: file_sha(source / path) for name, path in traces.items()},
        "training_lineage_independently_reconstructed": False,
        "source_trace": {
            "candidate": "phase13.train_and_save_candidate_bundle -> phase10.prepare_clean_dataset",
            "label": "int(round(close[t+h] - open[t+1], 9) > 0)",
            "entry": "open[t+1], at feature candle close, zero full-bar latency",
            "expiry": "close[t+h]; h=1 holds the immediately following candle",
            "ties": "rounded differences <= 0 map to class 0, including ties",
            "price_source": "authentic_iqoption_eurusd_60s (declared artifact lineage)",
            "calibration": "Platt validation uses the same training labels",
        },
        "development_targets": [
            {
                "family": "Phase10/13/14 directional models and baselines",
                "models": "majority, logistic, RF and configured tabular comparisons",
                "target": "int(round(close[t+h]-open[t+1],9)>0)",
                "horizons": "configured h; Phase14 defaults 1,3,5",
                "compatibility": "INCOMPATIBLE_DELAY_AND_TIE_POLICY",
            },
            {
                "family": "generic Phase4 target API",
                "target": "close[t+h]-close[t], rounded 9dp; binary/ternary/return + deadband",
                "compatibility": "UNVERIFIED_ARTIFACT_SPEC_AND_DIFFERENT_PRICE_ANCHOR",
            },
            {
                "family": "Phase15 significant movement",
                "target": "abs(clean.exit_prices-clean.entry_prices)*10000 >= threshold_pips",
                "limitation": "h loop reuses supplied clean geometry without rebuilding",
                "compatibility": "INCOMPATIBLE_NON_DIRECTIONAL_TARGET",
            },
            {
                "family": "Phase15 triple barrier RF",
                "target": "tb_arr == +1; first barrier hit or timeout; other events class 0",
                "compatibility": "INCOMPATIBLE_PATH_DEPENDENT_TARGET",
            },
            {
                "family": "Phase24/25 majority, logistic and RF",
                "target": "close[t+h+latency] > open[t+1+latency]",
                "ties": "training/calibration exclude isclose(atol=1e-9,rtol=0) prices",
                "compatibility": "UNVERIFIED_NO_MATCHING_FROZEN_BUNDLE_ATTESTATION",
                "limitation": "feature-relative latency differs from clock entry when data lags",
            },
        ],
        "observer_target": {
            "entry": "(floor(recorded_at_ts/60)+2)*60; future candle OPEN",
            "expiry": "entry+60; same candle CLOSE",
            "latency": "60 to 120 seconds after decision, plus variable source age",
            "ties": "exact equal prices; excluded from directional accuracy and calibration",
            "price_source": "verified authentic EURUSD minute candles; Blitz identity unverified",
        },
        "status": "TARGET_UNVERIFIED",
        "known_mismatches": ["ENTRY_DELAY", "FEATURE_RELATIVE_HORIZON", "TIE_AND_ROUNDING_POLICY"],
        "forecasting_skill_proof_allowed": False,
        "pooling_allowed": False,
        "required_followup": "Separately register any changed model/target/latency/scoring study",
    }
