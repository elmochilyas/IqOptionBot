"""Command-line interface for Phase 41 Model Failure Intelligence & Controlled Research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from iqoption_research.diagnostics.experiments import (
    get_predefined_experiment_matrix,
    render_experiment_matrix_table,
    run_controlled_experiment_suite,
)
from iqoption_research.diagnostics.failure_intelligence import (
    generate_failure_intelligence_report,
)
from iqoption_research.diagnostics.microstructure import (
    calculate_payout_sensitivity_table,
    render_microstructure_comparison_table,
)
from iqoption_research.diagnostics.parity import verify_prefix_invariance
from iqoption_research.diagnostics.targets import audit_target_distribution
from iqoption_research.ml.targets.specs import TargetSpec, TargetType, TiePolicy
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.research.features.pipeline import CausalFeaturePipeline

diagnostics_app = typer.Typer(
    name="diagnostics",
    help="Model Failure Intelligence & Controlled Research Diagnostics (Phase 41).",
    no_args_is_help=True,
)


@diagnostics_app.command("audit-target")
def audit_target_cmd(
    snapshot_id: Annotated[
        str | None, typer.Option("--snapshot", help="Snapshot ID to audit")
    ] = None,
    horizon: Annotated[int, typer.Option("--horizon", "-h", help="Prediction horizon in bars")] = 1,
    output_path: Annotated[
        Path | None, typer.Option("--output", "-o", help="Optional output JSON path")
    ] = None,
) -> None:
    """Audit DIRECTION_BINARY target distribution, exact ties, and label bias."""
    dm = ResearchDatasetManager()
    snapshots = dm.list_snapshots()
    if not snapshots:
        typer.secho("Error: No research dataset snapshots available.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    selected = None
    if snapshot_id:
        for s in snapshots:
            if s.manifest.snapshot_id == snapshot_id:
                selected = s
                break
        if selected is None:
            typer.secho(f"Error: Snapshot {snapshot_id} not found.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    else:
        selected = snapshots[-1]

    typer.echo(f"Auditing target on snapshot: {selected.manifest.snapshot_id}...")
    candles = selected.load_candles()
    spec = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=horizon,
        flat_threshold_pips=0.0,
    )
    result = audit_target_distribution(candles, spec, enforce_temporal_continuity=True)
    typer.echo(result.summary_table())

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        typer.secho(f"Saved audit JSON to {output_path}", fg=typer.colors.GREEN)


@diagnostics_app.command("check-parity")
def check_parity_cmd() -> None:
    """Verify feature pipeline prefix invariance and data availability consistency."""
    dm = ResearchDatasetManager()
    snapshots = dm.list_snapshots()
    if not snapshots:
        typer.secho("Error: No research dataset snapshots found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    candles = snapshots[-1].load_candles()
    prefix = candles[:50]
    fut_a = candles[50:80]
    from dataclasses import replace

    fut_b = [replace(c, close=c.open, volume=0) for c in candles[50:80]]

    typer.echo("=== 1. Testing Legacy Pipeline (dynamic dataset volume scan) ===")
    pipe_legacy = CausalFeaturePipeline(static_schema=False, include_volume=True)
    rep_legacy = verify_prefix_invariance(pipe_legacy, prefix, fut_a, fut_b)
    typer.echo(f"Prefix invariant: {rep_legacy.is_strictly_invariant}")
    typer.echo(f"Notes: {rep_legacy.notes}")

    typer.echo("\n=== 2. Testing Static Pipeline (corrected prefix-invariant schema) ===")
    pipe_static = CausalFeaturePipeline(static_schema=True, include_volume=True)
    rep_static = verify_prefix_invariance(pipe_static, prefix, fut_a, fut_b)
    typer.echo(f"Prefix invariant: {rep_static.is_strictly_invariant}")
    typer.echo(f"Notes: {rep_static.notes}")


@diagnostics_app.command("microstructure")
def microstructure_cmd() -> None:
    """Display market microstructure comparison and payout sensitivity economics."""
    typer.echo(render_microstructure_comparison_table())
    typer.echo("\n=== PAYOUT SENSITIVITY TABLE ===")
    points = calculate_payout_sensitivity_table()
    hdr = (
        f"{'Payout':<8} | {'Break-Even':<12} | {'Exp @ 50%':<11} | "
        f"{'Exp @ 52%':<11} | {'Exp @ 55%':<11}"
    )
    typer.echo(hdr)
    typer.echo("-" * 62)
    for p in points:
        exp50 = f"{p.net_expectancy_at_50_pct * 100:+6.2f}%"
        exp52 = f"{p.net_expectancy_at_52_pct * 100:+6.2f}%"
        exp55 = f"{p.net_expectancy_at_55_pct * 100:+6.2f}%"
        row = (
            f"{p.payout_rate * 100:5.0f}%  | {p.break_even_win_rate * 100:6.2f}%     | "
            f"{exp50}    | {exp52}    | {exp55}"
        )
        typer.echo(row)


@diagnostics_app.command("experiment-matrix")
def experiment_matrix_cmd(
    output_path: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output JSON ledger path")
    ] = None,
    sample_limit: Annotated[
        int, typer.Option("--sample-limit", help="Maximum candles to evaluate for speed")
    ] = 10000,
) -> None:
    """Execute the bounded controlled research experiment matrix across Areas A through F."""
    dm = ResearchDatasetManager()
    snapshots = dm.list_snapshots()
    if not snapshots:
        typer.secho("Error: No research dataset snapshots found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    snap = snapshots[-1]
    snap_id = snap.manifest.snapshot_id
    source_hash = snap.manifest.sha256_checksum
    candles = snap.load_candles()
    if sample_limit > 0 and len(candles) > sample_limit:
        candles = candles[-sample_limit:]

    typer.echo(f"Executing controlled research experiment matrix on {len(candles):,} candles...")
    matrix = get_predefined_experiment_matrix()
    results = run_controlled_experiment_suite(
        candles,
        matrix,
        snapshot_id=snap_id,
        source_hash=source_hash,
    )
    typer.echo(render_experiment_matrix_table(results))

    out_file = output_path or Path("data/experiments/phase41_experiment_ledger_v3.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")
    typer.secho(f"Trial ledger recorded to {out_file}", fg=typer.colors.GREEN)


@diagnostics_app.command("report")
def report_cmd(
    output_path: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output Markdown report path")
    ] = None,
) -> None:
    """Generate comprehensive Failure Intelligence report on development EURUSD data."""
    dm = ResearchDatasetManager()
    snapshots = dm.list_snapshots()
    if not snapshots:
        typer.secho("Error: No research dataset snapshots found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    snap = snapshots[-1]
    candles = snap.load_candles()
    snap_id = snap.manifest.snapshot_id
    typer.echo(
        f"Generating Failure Intelligence report on {len(candles):,} candles from {snap_id}..."
    )

    # We evaluate development partition using Logistic Regression model
    spec = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY, horizon_bars=1, tie_policy=TiePolicy.EXCLUDE
    )
    # Fit model to get predictions
    feat_pipe = CausalFeaturePipeline(
        static_schema=True, include_volume=True, enforce_temporal_continuity=True
    )
    feat_ds = feat_pipe.transform(candles)
    from iqoption_research.ml.targets.dataset import (
        build_ml_dataset,
        fit_scaler_on_dataset,
        transform_dataset,
    )

    ml_ds = build_ml_dataset(candles, feat_ds, spec, enforce_temporal_continuity=True)

    from iqoption_research.research.datasets.splitter import ChronologicalSplitter

    splitter = ChronologicalSplitter(
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, label_horizon_bars=1, embargo_bars=5
    )
    splits = splitter.split(list(range(ml_ds.row_count)))
    import numpy as np

    train_idx = np.array(splits.train, dtype=np.int64)
    val_idx = np.array(splits.val, dtype=np.int64)

    from iqoption_research.ml.targets.dataset import MLDataset

    ds_train = MLDataset(
        symbol=ml_ds.symbol,
        feature_names=ml_ds.feature_names,
        X=ml_ds.X[train_idx],
        y=ml_ds.y[train_idx],
        timestamps=ml_ds.timestamps[train_idx],
        target_spec=ml_ds.target_spec,
    )
    ds_val = MLDataset(
        symbol=ml_ds.symbol,
        feature_names=ml_ds.feature_names,
        X=ml_ds.X[val_idx],
        y=ml_ds.y[val_idx],
        timestamps=ml_ds.timestamps[val_idx],
        target_spec=ml_ds.target_spec,
    )

    scaler = fit_scaler_on_dataset(ds_train)
    scaled_train = transform_dataset(ds_train, scaler)
    scaled_val = transform_dataset(ds_val, scaler)

    from iqoption_research.ml.models.baselines import LogisticRegressionModel

    model = LogisticRegressionModel(random_state=42)
    model.fit(scaled_train.X, scaled_train.y)
    val_probs = model.predict_proba(scaled_val.X)[:, 1]

    # Align predictions to market data using verified timestamps
    from iqoption_research.diagnostics.alignment import align_predictions_to_market_data

    aligned = align_predictions_to_market_data(
        candles=candles,
        ml_dataset=ds_val,
        predicted_probabilities=val_probs,
        decision_threshold=0.50,
    )
    report = generate_failure_intelligence_report(
        candles,
        aligned,
        decision_threshold=0.50,
        dataset_id=snap.manifest.snapshot_id,
        spec=spec,
        source_hash=snap.manifest.sha256_checksum,
    )

    md = report.render_markdown()
    out_file = output_path or Path("reports/model_failure_intelligence_v3.md")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md, encoding="utf-8")
    typer.secho(f"Failure Intelligence report written to {out_file}", fg=typer.colors.GREEN)
    typer.echo(md[:1500] + "\n... [truncated, full report written to disk]")
