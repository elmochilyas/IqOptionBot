"""Phase 4 CLI tests: offline prepare/train/evaluate/walk-forward/compare/backtest/demo.

The commands are exercised through Typer's in-process runner with temporary
workspaces. Assertions focus on exit status, honest evidence labels, artifact
creation, and failure modes -- never on profitability.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.ml.pipeline import PipelineError, load_prepared_dataset
from iqoption_research.ml.tracking.artifacts import MANIFEST_FILENAME
from iqoption_research.ml.tracking.tracker import ExperimentTracker

runner = CliRunner()
N_BARS = "420"


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "ml"


def _invoke(*args: str):
    return runner.invoke(app, ["ml", *args])


def _prepare(workspace: Path) -> Path:
    prepared_dir = workspace / "prepared"
    result = _invoke(
        "prepare",
        "--output",
        str(prepared_dir),
        "--n-bars",
        N_BARS,
        "--momentum-strength",
        "0.45",
        "--seed",
        "7",
        "--overwrite",
    )
    assert result.exit_code == 0, result.output
    return prepared_dir


def _train(workspace: Path, prepared_dir: Path) -> Path:
    bundle_dir = workspace / "bundle"
    result = _invoke(
        "train",
        "--prepared",
        str(prepared_dir),
        "--bundle",
        str(bundle_dir),
        "--report",
        str(workspace / "reports" / "train.md"),
        "--db",
        str(workspace / "experiments.db"),
        "--overwrite",
    )
    assert result.exit_code == 0, result.output
    return bundle_dir


def test_cli_prepare_creates_verified_synthetic_dataset(workspace: Path) -> None:
    prepared_dir = _prepare(workspace)
    result = _invoke(
        "prepare",
        "--output",
        str(workspace / "prepared_second"),
        "--n-bars",
        N_BARS,
        "--seed",
        "7",
        "--overwrite",
    )
    assert result.exit_code == 0, result.output
    assert "Synthetic engineering fixture only" in result.output
    assert "Prepared Dataset" in result.output

    prepared = load_prepared_dataset(prepared_dir)
    assert prepared.is_synthetic is True
    assert prepared.row_count > 0
    assert prepared.dataset.feature_dim == 19
    assert (prepared_dir / MANIFEST_FILENAME).is_file()


def test_cli_prepare_authentic_without_csv_is_rejected(workspace: Path) -> None:
    result = _invoke(
        "prepare",
        "--output",
        str(workspace / "bad"),
        "--authentic",
        "--n-bars",
        "100",
    )
    assert result.exit_code != 0
    assert "requires --csv" in result.output


def test_cli_prepare_rejects_unknown_target(workspace: Path) -> None:
    result = _invoke(
        "prepare",
        "--output",
        str(workspace / "bad"),
        "--target-type",
        "crystal_ball",
        "--n-bars",
        "100",
    )
    assert result.exit_code != 0
    assert "Unknown target type" in result.output


def test_cli_train_evaluate_and_backtest(workspace: Path) -> None:
    prepared_dir = _prepare(workspace)
    bundle_dir = _train(workspace, prepared_dir)

    assert (bundle_dir / MANIFEST_FILENAME).is_file()
    assert (workspace / "reports" / "train.md").is_file()

    evaluate = _invoke(
        "evaluate",
        "--prepared",
        str(prepared_dir),
        "--bundle",
        str(bundle_dir),
        "--report",
        str(workspace / "reports" / "evaluate.md"),
        "--db",
        str(workspace / "experiments.db"),
    )
    assert evaluate.exit_code == 0, evaluate.output
    assert "Bundle verified" in evaluate.output
    assert "SYNTHETIC_ENGINEERING_ONLY" in evaluate.output

    backtest = _invoke(
        "backtest",
        "--prepared",
        str(prepared_dir),
        "--bundle",
        str(bundle_dir),
        "--report",
        str(workspace / "reports" / "backtest.md"),
        "--db",
        str(workspace / "experiments.db"),
    )
    assert backtest.exit_code == 0, backtest.output
    assert "Trades:" in backtest.output
    assert "SYNTHETIC_ENGINEERING_ONLY" in backtest.output

    tracker = ExperimentTracker(workspace / "experiments.db")
    names = {record.experiment_name for record in tracker.list_experiments()}
    assert {"holdout_train", "holdout_evaluate", "offline_backtest"} <= names


def test_cli_train_rejects_unknown_model(workspace: Path) -> None:
    prepared_dir = _prepare(workspace)
    result = _invoke(
        "train",
        "--prepared",
        str(prepared_dir),
        "--bundle",
        str(workspace / "bundle"),
        "--model",
        "mystery_model",
        "--overwrite",
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, PipelineError)
    assert "Unknown model" in str(result.exception)


def test_cli_walk_forward_and_compare(workspace: Path) -> None:
    prepared_dir = _prepare(workspace)

    walk_forward = _invoke(
        "walk-forward",
        "--prepared",
        str(prepared_dir),
        "--folds",
        "3",
        "--report",
        str(workspace / "reports" / "walk_forward.md"),
        "--db",
        str(workspace / "experiments.db"),
    )
    assert walk_forward.exit_code == 0, walk_forward.output
    assert "WalkForward" in walk_forward.output
    assert "SYNTHETIC_ENGINEERING_ONLY" in walk_forward.output
    assert (workspace / "reports" / "walk_forward.md").is_file()

    compare = _invoke(
        "compare",
        "--prepared",
        str(prepared_dir),
        "--models",
        "majority_class,logistic_regression",
        "--report",
        str(workspace / "reports" / "comparison.md"),
        "--db",
        str(workspace / "experiments.db"),
    )
    assert compare.exit_code == 0, compare.output
    assert "Benchmark Comparison Summary" in compare.output
    assert "SYNTHETIC_ENGINEERING_ONLY" in compare.output
    assert (workspace / "reports" / "comparison.md").is_file()


def test_cli_compare_rejects_invalid_split(workspace: Path) -> None:
    prepared_dir = _prepare(workspace)
    result = _invoke(
        "compare",
        "--prepared",
        str(prepared_dir),
        "--split",
        "magic",
    )
    assert result.exit_code != 0
    assert "holdout" in result.output


def test_cli_demo_runs_end_to_end(workspace: Path) -> None:
    result = _invoke(
        "demo",
        "--workspace",
        str(workspace),
        "--n-bars",
        N_BARS,
        "--seed",
        "11",
        "--db",
        str(workspace / "demo.db"),
    )
    assert result.exit_code == 0, result.output
    assert "DEMONSTRATION COMPLETE" in result.output
    assert "SYNTHETIC_ENGINEERING_ONLY" in result.output
    assert (workspace / "reports" / "phase4_demo.md").is_file()
    assert (workspace / "prepared" / MANIFEST_FILENAME).is_file()
    assert (workspace / "bundle" / MANIFEST_FILENAME).is_file()
    assert (workspace / "demo.db").is_file()

    report_text = (workspace / "reports" / "phase4_demo.md").read_text(encoding="utf-8")
    assert "SYNTHETIC ENGINEERING ONLY" in report_text
    assert "not financial advice" in report_text.lower()


def test_cli_list_inspect_and_report(workspace: Path) -> None:
    prepared_dir = _prepare(workspace)
    bundle_dir = _train(workspace, prepared_dir)
    db_path = workspace / "experiments.db"

    tracker = ExperimentTracker(db_path)
    records = tracker.list_experiments(experiment_name="holdout_train")
    assert records
    run_id = records[0].run_id

    listing = _invoke("list", "--db", str(db_path))
    assert listing.exit_code == 0, listing.output
    assert "Experiments (1 total)" in listing.output
    assert "holdout" in listing.output

    inspect_run = _invoke("inspect", "--run-id", run_id, "--db", str(db_path))
    assert inspect_run.exit_code == 0, inspect_run.output
    assert run_id in inspect_run.output

    inspect_bundle = _invoke("inspect", "--bundle", str(bundle_dir))
    assert inspect_bundle.exit_code == 0, inspect_bundle.output
    assert "Verified bundle" in inspect_bundle.output

    report = _invoke(
        "report",
        "--run-id",
        run_id,
        "--db",
        str(db_path),
        "--output",
        str(workspace / "reports" / "experiment.md"),
    )
    assert report.exit_code == 0, report.output
    report_text = (workspace / "reports" / "experiment.md").read_text(encoding="utf-8")
    assert run_id in report_text
    assert "SYNTHETIC_ENGINEERING_ONLY" in report_text


def test_cli_inspect_requires_identifier() -> None:
    result = _invoke("inspect")
    assert result.exit_code != 0
    assert "--run-id" in result.output


def test_cli_phase4_commands_do_not_open_sockets(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _blocked_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Phase 4 must not open network sockets")

    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.setattr(socket, "create_connection", _blocked_socket)

    prepared_dir = _prepare(workspace)
    bundle_dir = _train(workspace, prepared_dir)
    backtest = _invoke(
        "backtest",
        "--prepared",
        str(prepared_dir),
        "--bundle",
        str(bundle_dir),
        "--report",
        str(workspace / "reports" / "backtest.md"),
        "--db",
        str(workspace / "experiments.db"),
    )
    assert backtest.exit_code == 0, backtest.output
