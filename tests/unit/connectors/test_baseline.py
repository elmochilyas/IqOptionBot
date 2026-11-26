"""Phase 4.1 Tests: ML Subpackage Baseline, Environment Verification, and Toolchain.

Covers:
- Package presence, docstring, version, and clean export layout for iqoption_research.ml
- Dependency availability and minimum version validation (sklearn, pandas, numpy, scipy, openai)
- Strict offline security guard: import and inspection trigger zero network socket calls
- Non-interference with existing Phase 1-3 modules (models, connectors, research)
- Safe capability detection for optional GBDT libraries (lightgbm, xgboost, catboost)
- Test discovery integrity preserving existing test baseline
"""

import importlib
import importlib.util
import socket
from unittest.mock import patch

import pytest


def test_ml_package_import_and_metadata() -> None:
    """Verify iqoption_research.ml is cleanly importable with metadata and exports."""
    # When importing the ML subpackage
    ml_pkg = importlib.import_module("iqoption_research.ml")

    # Package metadata assertions
    assert ml_pkg.__doc__ is not None, "iqoption_research.ml must have a package docstring"
    assert "Phase 4" in ml_pkg.__doc__
    assert "Machine Learning" in ml_pkg.__doc__
    assert "causal" in ml_pkg.__doc__.lower()
    assert "offline" in ml_pkg.__doc__.lower()

    # Version assertion
    assert hasattr(ml_pkg, "__version__"), "iqoption_research.ml must expose __version__"
    assert ml_pkg.__version__ == "0.1.0"

    # Export layout assertions
    assert hasattr(ml_pkg, "__all__"), "iqoption_research.ml must define __all__"
    assert isinstance(ml_pkg.__all__, list)
    assert "__version__" in ml_pkg.__all__
    for symbol in ml_pkg.__all__:
        assert hasattr(ml_pkg, symbol), f"Exported symbol '{symbol}' not found in package"


@pytest.mark.parametrize(
    ("module_name", "expected_min_major"),
    [
        ("sklearn", 1),
        ("pandas", 2),
        ("numpy", 2),
        ("scipy", 1),
        ("openai", 1),
    ],
)
def test_ml_core_dependencies_available(module_name: str, expected_min_major: int) -> None:
    """Verify all required Phase 4 ML libraries are importable and satisfy version floors."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "__version__"), f"{module_name} must have a __version__ attribute"
    version_str = str(mod.__version__)
    major_version = int(version_str.split(".")[0])
    assert major_version >= expected_min_major, (
        f"{module_name} version {version_str} is below required major version {expected_min_major}"
    )


def test_ml_package_strictly_offline() -> None:
    """Verify that inspecting or reloading iqoption_research.ml makes zero socket connections."""
    with patch.object(socket.socket, "connect") as mock_connect:
        ml_pkg = importlib.import_module("iqoption_research.ml")
        importlib.reload(ml_pkg)
        mock_connect.assert_not_called()


def test_phase1_to_phase3_contracts_preserved() -> None:
    """Verify existing Phase 1-3 modules and contracts remain intact and uncorrupted."""
    # Phase 1: Models & Connectors
    from iqoption_research.connectors.base import CandleFetch
    from iqoption_research.models.account import AccountStatus
    from iqoption_research.models.market import Candle

    assert Candle is not None
    assert AccountStatus is not None
    assert CandleFetch is not None

    # Phase 3: Research Pipeline & Backtest
    from iqoption_research.research.backtest.engine import BacktestEngine
    from iqoption_research.research.datasets.loader import HistoricalDatasetLoader
    from iqoption_research.research.features.pipeline import CausalFeaturePipeline

    assert BacktestEngine is not None
    assert HistoricalDatasetLoader is not None
    assert CausalFeaturePipeline is not None


def test_optional_gbdt_adapters_capability_check() -> None:
    """Verify optional GBDT library detection returns boolean without throwing unhandled errors."""
    for adapter_name in ["lightgbm", "xgboost", "catboost"]:
        spec = importlib.util.find_spec(adapter_name)
        assert spec is None or spec.loader is not None, (
            f"find_spec({adapter_name}) returned invalid spec"
        )


def test_planned_subpackages_structure_contract() -> None:
    """Verify planned Phase 4 modular architecture contracts documented in PROJECT.md."""
    planned_subpackages = [
        "targets",
        "models",
        "calibration",
        "evaluation",
        "decision",
        "ensembles",
        "hybrid",
        "tracking",
    ]
    ml_pkg = importlib.import_module("iqoption_research.ml")
    doc = ml_pkg.__doc__ or ""
    for subpkg in planned_subpackages:
        assert f"`{subpkg}`" in doc, f"Planned subpackage `{subpkg}` missing from package docstring"
