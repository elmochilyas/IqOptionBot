"""Reproducible authentic-data benchmarks. No execution or credential dependencies."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess  # noqa: S404 - fixed local git metadata commands only
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
)

from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.phase10 import (
    AUTHENTIC_SOURCE,
    CleanResearchDataset,
    Phase10Config,
    audit_authentic_datasets,
    build_walk_forward_folds,
    prepare_clean_dataset,
)
from iqoption_research.ml.pipeline import create_model
from iqoption_research.models.market import Candle
from iqoption_research.research.backtest.models import SignalDirection
from iqoption_research.research.backtest.strategies.baselines import (
    AlwaysCallStrategy,
    AlwaysPutStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    NoTradeStrategy,
    RandomStrategy,
)
from iqoption_research.research.datasets.integrity import build_point_in_time_record
from iqoption_research.research.evaluation_audit import candle_integrity, catalog_coverage
from iqoption_research.research.evaluation_metrics import SimulationPolicy, simulate
from iqoption_research.research.features.pipeline import FeatureRow
from iqoption_research.research.features.scalers import StandardScaler


@dataclass(frozen=True)
class EvaluationConfig:
    symbol: str = "EURUSD"
    timeframe: int = 60
    source: str = AUTHENTIC_SOURCE
    horizon: int = 1
    latency_bars: int = 1
    folds: int = 3
    seed: int = 42
    min_rows: int = 2000
    min_trades: int = 500
    min_calendar_days: int = 90
    models: tuple[str, ...] = ("majority_class", "logistic_regression", "random_forest")
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65)
    payout_sensitivity: tuple[float, ...] = (0.70, 0.85, 0.95)
    policy: SimulationPolicy = field(default_factory=SimulationPolicy)

    def validate(self) -> None:
        self.policy.validate()
        if self.symbol not in {"EURUSD", "EURUSD_OTC", "EURUSD_OP"}:
            raise ValueError("Unsupported evaluation instrument")
        if self.timeframe < 1 or self.horizon < 1 or not 0 <= self.latency_bars <= 10:
            raise ValueError("Invalid candle geometry")
        if not 1 <= self.folds <= 5 or self.min_rows < 100 or self.min_trades < 30:
            raise ValueError("Invalid evaluation sample policy")
        if self.source.lower() != AUTHENTIC_SOURCE or self.min_calendar_days < 1:
            raise ValueError("Unverified evaluation provenance or span")
        if (
            not self.models
            or any(
                name
                not in {
                    "majority_class",
                    "logistic_regression",
                    "random_forest",
                    "hist_gradient_boosting",
                }
                for name in self.models
            )
            or len(set(self.models)) != len(self.models)
        ):
            raise ValueError("Use unique supported bounded CPU models")
        if not self.thresholds or any(not 0.5 <= t < 1 for t in self.thresholds):
            raise ValueError("Invalid validation threshold grid")
        if not self.payout_sensitivity or any(not 0 < p <= 2 for p in self.payout_sensitivity):
            raise ValueError("Invalid payout sensitivity")


def prepared_data(candles: list[Candle], cfg: EvaluationConfig) -> CleanResearchDataset:
    """Reuse causal segmented features; align delayed entry and future labels explicitly."""
    integrity = candle_integrity(candles, cfg.symbol, cfg.timeframe)
    if integrity["status"] != "INTEGRITY_PASSED":
        raise ValueError("Invalid or mixed evaluation candles")
    clean = prepare_clean_dataset(
        candles,
        candle_size_seconds=cfg.timeframe,
        horizon_bars=cfg.horizon + cfg.latency_bars,
        include_volume=False,
    )
    lookup = {c.from_ts: c for c in candles}
    entry_ts = clean.entry_timestamps + cfg.latency_bars * cfg.timeframe
    entry = np.asarray([lookup[int(t)].open for t in entry_ts], dtype=float)
    y = (clean.exit_prices > entry).astype(np.int64)
    digest = hashlib.sha256()
    for values in (
        clean.X,
        y,
        clean.timestamps,
        entry_ts,
        clean.exit_timestamps,
        entry,
        clean.exit_prices,
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    return replace(
        clean, y=y, entry_timestamps=entry_ts, entry_prices=entry, fingerprint=digest.hexdigest()
    )


def assert_split(
    data: CleanResearchDataset, train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> None:
    if not all(len(part) for part in (train, val, test)):
        raise ValueError("Empty chronological partition")
    if not (train[-1] < val[0] <= val[-1] < test[0]):
        raise ValueError("Training/evaluation contamination")
    if int(data.exit_timestamps[train].max()) >= int(data.timestamps[val].min()):
        raise ValueError("Training labels overlap validation decisions")
    if int(data.exit_timestamps[val].max()) >= int(data.timestamps[test].min()):
        raise ValueError("Validation labels overlap test decisions")


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    if len(y) != len(p) or not len(y) or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probability evaluation")
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    predicted = (p >= 0.5).astype(int)
    bins: list[dict[str, Any]] = []
    calibration_error = 0.0
    for lower in np.arange(0, 1, 0.1):
        mask = (p >= lower) & ((p < lower + 0.1) if lower < 0.9 else (p <= 1))
        if mask.any():
            average, frequency = float(p[mask].mean()), float(y[mask].mean())
            calibration_error += float(mask.mean()) * abs(average - frequency)
            bins.append({
                "lower": round(float(lower), 1),
                "samples": int(mask.sum()),
                "predicted_up": average,
                "observed_up": frequency,
            })
    return {
        "samples": len(y),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision_up": float(precision_score(y, predicted, zero_division=0)),
        "recall_up": float(recall_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "expected_calibration_error": calibration_error,
        "calibration_bins": bins,
    }


def signals(p: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(p >= threshold, 1, np.where(p <= 1 - threshold, -1, 0))


def strategy_directions(
    data: CleanResearchDataset, candles: list[Candle], seed: int
) -> dict[str, np.ndarray]:
    """Call the existing strategy contracts with no future FeatureRows exposed."""
    strategies = [
        NoTradeStrategy(),
        AlwaysCallStrategy(),
        AlwaysPutStrategy(),
        RandomStrategy(seed=seed),
        MomentumStrategy(),
        MovingAverageCrossoverStrategy(),
    ]
    output: dict[str, list[int]] = {strategy.name: [] for strategy in strategies}
    lookup = {c.to_ts: c for c in candles}
    for i, timestamp in enumerate(data.timestamps):
        candle = lookup[int(timestamp)]
        feature = FeatureRow(
            index=i,
            symbol=candle.symbol,
            candle_from_ts=candle.from_ts,
            candle_to_ts=candle.to_ts,
            available_at_ts=int(timestamp),
            pit=build_point_in_time_record(candle),
            is_warmup=False,
            features=dict(zip(data.feature_names, map(float, data.X[i]), strict=True)),
        )
        for strategy in strategies:
            direction = strategy.on_bar(feature, ()).direction
            output[strategy.name].append(
                1
                if direction == SignalDirection.CALL
                else -1
                if direction == SignalDirection.PUT
                else 0
            )
    return {name: np.asarray(values, dtype=np.int64) for name, values in output.items()}


def model_predictions(
    data: CleanResearchDataset,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    name: str,
    cfg: EvaluationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    assert_split(data, train, val, test)
    # Flat outcomes are refunded economically; do not teach them as PUT wins.
    train = train[~np.isclose(data.entry_prices[train], data.exit_prices[train], atol=1e-9, rtol=0)]
    if len(train) < 30:
        raise ValueError("Insufficient decisive training samples")
    scaler = StandardScaler().fit(data.X[train], feature_names=data.feature_names)
    xtrain = np.asarray(scaler.transform(data.X[train]))
    xval = np.asarray(scaler.transform(data.X[val]))
    xtest = np.asarray(scaler.transform(data.X[test]))
    model = create_model(name, seed=cfg.seed, quick=True)
    model.fit(xtrain, data.y[train], feature_names=data.feature_names)
    raw_val = model.predict_proba(xval)
    raw_test = model.predict_proba(xtest)
    # Split validation itself: calibration precedes threshold tuning, with purge.
    middle = len(val) // 2
    purge = cfg.horizon + cfg.latency_bars + 1
    calibration_end = middle - purge
    if calibration_end < 10 or len(val) - middle < 10:
        raise ValueError("Insufficient separated validation samples")
    if data.exit_timestamps[val[calibration_end - 1]] >= data.timestamps[val[middle]]:
        raise ValueError("Calibration labels overlap threshold tuning")
    calibration_mask = ~np.isclose(
        data.entry_prices[val[:calibration_end]],
        data.exit_prices[val[:calibration_end]],
        atol=1e-9,
        rtol=0,
    )
    calibrator = ProbabilityCalibrator(method="platt")
    calibrated = len(np.unique(data.y[val[:calibration_end]][calibration_mask])) == 2
    if calibrated:
        calibrator.fit(
            raw_val[:calibration_end][calibration_mask],
            data.y[val[:calibration_end]][calibration_mask],
        )
        pval = calibrator.calibrate(raw_val)[:, 1]
        ptest = calibrator.calibrate(raw_test)[:, 1]
    else:
        pval, ptest = raw_val[:, 1], raw_test[:, 1]
    return (
        pval[middle:],
        ptest,
        {
            "scaler_fit_samples": len(train),
            "calibration": "PLATT" if calibrated else "UNAVAILABLE_SINGLE_CLASS",
            "calibration_last_row": int(val[calibration_end - 1]),
            "threshold_tuning_first_row": int(val[middle]),
            "validation_middle": middle,
        },
    )


def evaluate_development(
    data: CleanResearchDataset, candles: list[Candle], cfg: EvaluationConfig
) -> dict[str, Any]:
    fold_config = Phase10Config(
        n_folds=cfg.folds,
        initial_train_ratio=0.4,
        validation_ratio=0.1,
        test_ratio=0.5 / cfg.folds,
        horizon_bars=cfg.horizon + cfg.latency_bars + 1,
    )
    folds = build_walk_forward_folds(data, fold_config)
    baseline = strategy_directions(data, candles, cfg.seed)
    family = len(baseline) + len(cfg.models) * len(cfg.thresholds)
    reports: list[dict[str, Any]] = []
    pooled: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]] = {}
    for fold in folds:
        train = np.arange(fold.train_start, fold.train_end)
        val = np.arange(fold.validation_start, fold.validation_end)
        test = np.arange(fold.test_start, fold.test_end)
        assert_split(data, train, val, test)
        result: dict[str, Any] = {"split": fold.to_dict(), "models": {}, "strategies": {}}
        for name, directions in baseline.items():
            result["strategies"][name] = simulate(
                data, test, directions[test], cfg.policy, bootstrap=False, family_size=family
            )
            pooled.setdefault(name, []).append((test, directions[test], None))
        for name in cfg.models:
            try:
                tuning_p, test_p, metadata = model_predictions(data, train, val, test, name, cfg)
                tuning_indices = val[metadata["validation_middle"] :]
                scores = []
                for threshold in cfg.thresholds:
                    metrics = simulate(
                        data,
                        tuning_indices,
                        signals(tuning_p, threshold),
                        cfg.policy,
                        probabilities=tuning_p,
                        bootstrap=False,
                    )
                    # Equal opportunity objective; no-trade is preferable to negative EV.
                    scores.append((metrics["total_simulated_pnl"], threshold))
                chosen = max(scores)[1]
                direction = signals(test_p, chosen)
                decisive = ~np.isclose(
                    data.entry_prices[test], data.exit_prices[test], atol=1e-9, rtol=0
                )
                classification = probability_metrics(data.y[test][decisive], test_p[decisive])
                volatility = data.feature_names.index("volatility_20")
                regime_boundary = float(np.median(data.X[train, volatility]))
                regimes = {}
                for label, mask in (
                    ("low_volatility", data.X[test, volatility] <= regime_boundary),
                    ("high_volatility", data.X[test, volatility] > regime_boundary),
                ):
                    selected = mask & decisive
                    regimes[label] = (
                        probability_metrics(data.y[test][selected], test_p[selected])
                        if selected.any()
                        else {"samples": 0}
                    )
                result["models"][name] = {
                    "status": "EVALUATED",
                    "threshold_selected_on_validation": chosen,
                    "validation_threshold_scores": [
                        {"threshold": t, "pnl": score} for score, t in scores
                    ],
                    "preprocessing": metadata,
                    "classification": classification,
                    "regimes_using_train_volatility_boundary": regimes,
                    "economics": simulate(
                        data,
                        test,
                        direction,
                        cfg.policy,
                        probabilities=test_p,
                        bootstrap=False,
                        family_size=family,
                    ),
                    "threshold_sensitivity_descriptive_only": [
                        {
                            "threshold": t,
                            "economics": simulate(
                                data,
                                test,
                                signals(test_p, t),
                                cfg.policy,
                                probabilities=test_p,
                                bootstrap=False,
                            ),
                        }
                        for t in cfg.thresholds
                    ],
                }
                pooled.setdefault(name, []).append((test, direction, test_p))
            except (ValueError, RuntimeError):
                result["models"][name] = {
                    "status": "INVALID_EVALUATION",
                    "reason": "MODEL_FOLD_FAILED",
                }
        reports.append(result)
    leaderboard: list[dict[str, Any]] = []
    risk: dict[str, Any] = {}
    for name, parts in pooled.items():
        indices = np.concatenate([part[0] for part in parts])
        directions = np.concatenate([part[1] for part in parts])
        probabilities = (
            np.concatenate([part[2] for part in parts if part[2] is not None])
            if parts[0][2] is not None
            else None
        )
        metrics = simulate(
            data,
            indices,
            directions,
            cfg.policy,
            probabilities=probabilities,
            seed=cfg.seed,
            family_size=family,
        )
        ci = metrics["block_bootstrap_unit_ev_interval"]
        profitable = metrics["total_simulated_pnl"] > 0 and ci is not None and ci[0] > 0
        status = (
            "INVALID_EVALUATION"
            if len(parts) != len(folds)
            else "INSUFFICIENT_AUTHENTIC_DATA"
            if metrics["accepted_trades"] < cfg.min_trades
            else "PROMISING_BUT_UNVERIFIED"
            if profitable
            else "NO_DEMONSTRATED_EDGE"
        )
        metrics["pnl_to_max_drawdown"] = (
            metrics["total_simulated_pnl"] / metrics["max_drawdown"]
            if metrics["max_drawdown"]
            else None
        )
        leaderboard.append({"name": name, "classification": status, **metrics})
        policies = {
            "fixed": cfg.policy,
            "percent": replace(cfg.policy, sizing="percent"),
            "strict_loss_limits": replace(
                cfg.policy,
                daily_loss_fraction=0.005,
                consecutive_loss_limit=5,
                drawdown_fraction=0.05,
            ),
            "confidence_and_ev": replace(cfg.policy, min_confidence=0.60, min_ev=0.01),
        }
        risk[name] = {
            label: simulate(
                data, indices, directions, policy, probabilities=probabilities, bootstrap=False
            )
            for label, policy in policies.items()
        }
        risk[name]["payout_sensitivity"] = [
            {
                "payout": payout,
                "economics": simulate(
                    data,
                    indices,
                    directions,
                    replace(cfg.policy, payout=payout),
                    probabilities=probabilities,
                    bootstrap=False,
                ),
            }
            for payout in cfg.payout_sensitivity
        ]
    for name in cfg.models:
        if name not in pooled:
            empty = np.asarray([], dtype=np.int64)
            leaderboard.append({
                "name": name,
                "classification": "INVALID_EVALUATION",
                **simulate(data, empty, empty, cfg.policy, bootstrap=False),
                "pnl_to_max_drawdown": None,
            })
    leaderboard.sort(
        key=lambda row: (row["pnl_to_max_drawdown"] or 0, row["total_simulated_pnl"]), reverse=True
    )
    return {
        "folds": reports,
        "leaderboard": leaderboard,
        "risk_research": risk,
        "selection_bias_warning": (
            "Development comparisons cannot establish an untouched-holdout edge."
        ),
        "statistical_policy": {
            "min_trades": cfg.min_trades,
            "family_size": family,
            "bootstrap_trials": 300,
            "block_size_opportunities": 60,
            "alpha": 0.05,
            "stationarity_assumed_locally": True,
            "purge_and_embargo_bars": cfg.horizon + cfg.latency_bars + 1,
            "risk_and_payout_variants": "DESCRIPTIVE_NOT_INDEPENDENT_CONFIRMATIONS",
        },
    }


def run_evaluation(
    storage: Path,
    cfg: EvaluationConfig,
    reservation_path: Path,
    *,
    mode: str = "all",
) -> dict[str, Any]:
    cfg.validate()
    if mode not in {
        "all",
        "coverage",
        "integrity",
        "benchmark",
        "walk-forward",
        "compare",
        "report",
    }:
        raise ValueError("Unknown evaluation mode")
    config = asdict(cfg)
    config["evaluation_mode"] = mode
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    try:
        revision = subprocess.run(  # noqa: S603 - fixed local git metadata only
            [shutil.which("git") or "git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "UNAVAILABLE"
    implementation = hashlib.sha256()
    package = Path(__file__).resolve().parents[1]
    for path in sorted(package.rglob("*.py")):
        implementation.update(path.relative_to(package).as_posix().encode())
        implementation.update(path.read_bytes())
    report: dict[str, Any] = {
        "phase": 24,
        "status": "INSUFFICIENT_AUTHENTIC_DATA",
        "config": config,
        "config_fingerprint": config_digest,
        "code_revision": revision,
        "implementation_fingerprint": implementation.hexdigest(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy")
        },
        "safety": {
            "offline_only": True,
            "autonomous_execution_authorized": False,
            "broker_orders": 0,
            "account_switching": False,
        },
        "coverage": catalog_coverage(storage),
        "assumptions": [
            "Fixed net winning return is an assumption, not recovered Blitz payout.",
            "Next delayed candle open is a simulated fill, not a broker executable quote.",
            "One position; tie refund; per-stake cost and adverse strike slippage.",
            "Legacy bundles are excluded; their prior use prevents fresh holdout claims.",
        ],
    }
    if mode in {"coverage", "integrity"}:
        report["status"] = report["coverage"]["status"]
        return report
    group = next(
        (
            g
            for g in report["coverage"]["datasets"]
            if g["symbol"] == cfg.symbol
            and g["source"] == cfg.source
            and g["timeframe_seconds"] == cfg.timeframe
        ),
        None,
    )
    if group is None or group["status"] != "INTEGRITY_PASSED":
        return report
    try:
        reservation_bytes = reservation_path.read_bytes()
        state = json.loads(reservation_bytes)
        cutoff = state["prior_evaluation_cutoff_ts"]
        if type(cutoff) is not int or state.get("evaluation_status") != "RESERVED":
            raise ValueError("Invalid holdout reservation")
    except (OSError, ValueError, KeyError):
        report["holdout"] = {"status": "INVALID_EVALUATION", "reason": "RESERVATION_REQUIRED"}
        return report
    audit, candles = audit_authentic_datasets(
        storage, symbol=cfg.symbol, candle_size_seconds=cfg.timeframe, source=cfg.source
    )
    development = [c for c in candles if c.to_ts < cutoff]
    holdout_count = sum(c.from_ts >= cutoff for c in candles)
    holdout_span = max(0, candles[-1].to_ts - cutoff) / 86400
    report["dataset_fingerprint"] = audit.aggregate_fingerprint
    report["holdout"] = {
        "status": "RESERVED_NOT_EVALUATED",
        "cutoff_ts": cutoff,
        "available_candles": holdout_count,
        "minimum_calendar_days": state.get("confirmation_days", 14),
        "available_calendar_days": holdout_span,
        "data_sufficiency": "INSUFFICIENT_AUTHENTIC_DATA"
        if holdout_span < state.get("confirmation_days", 14) or holdout_count < cfg.min_rows
        else "AVAILABLE_BUT_RESERVED",
        "reservation_fingerprint": hashlib.sha256(reservation_bytes).hexdigest(),
        "reason": "Untouched confirmation data is not consumed by development commands.",
    }
    if len(development) < cfg.min_rows:
        return report
    clean = prepared_data(development, cfg)
    report["development"] = {
        "candles": len(development),
        "opportunities": clean.row_count,
        "calendar_days": (development[-1].to_ts - development[0].from_ts) / 86400,
        "fingerprint": clean.fingerprint,
        "segments": clean.segment_count,
        "warmup_rows_dropped": clean.dropped_warmup_rows,
    }
    if clean.row_count < cfg.min_rows:
        return report
    effective = replace(cfg, models=()) if mode == "benchmark" else cfg
    report["evaluation"] = evaluate_development(clean, development, effective)
    report["status"] = "NO_DEMONSTRATED_EDGE"
    report["evidence_limits"] = [
        "No independent untouched confirmation evaluated.",
        "Development history was used in earlier milestones; selection bias remains.",
        "Only the selected instrument/timeframe/source is benchmarked; no cross-product inference.",
    ]
    if report["development"]["calendar_days"] < cfg.min_calendar_days:
        report["evidence_limits"].append("Calendar coverage below prespecified minimum.")
    return report


def write_reports(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Phase 24 research report",
        "",
        f"Conclusion: **{report['status']}**.",
        "Offline historical simulation; no autonomous Demo authorization.",
        "",
        "## Data quality",
        "",
    ]
    for group in report["coverage"]["datasets"]:
        count = group.get("unique_candles")
        verified_count = f"{count:,}" if count is not None else "UNVERIFIED"
        lines.append(
            f"- {group['symbol']} {group['timeframe_seconds']}s / {group['source']}: "
            f"{group['status']}; {verified_count} unique candles."
        )
    lines += [
        "",
        "## Strategy and ML leaderboard",
        "",
        "| Candidate | Trades | Win rate | P&L | Max drawdown | P&L/drawdown | Evidence |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.get("evaluation", {}).get("leaderboard", []):
        win = f"{row['win_rate']:.3%}" if row["win_rate"] is not None else "N/A"
        lines.append(
            f"| {row['name']} | {row['accepted_trades']} | {win} | "
            f"{row['total_simulated_pnl']:.2f} | {row['max_drawdown']:.2f} | "
            f"{row.get('pnl_to_max_drawdown')} | "
            f"{row['classification']} |"
        )
    lines += [
        "",
        "## Walk-forward model results",
        "",
        "| Fold | Model | Accuracy | Brier | Log loss | Validation threshold | Trades |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for fold in report.get("evaluation", {}).get("folds", []):
        for name, model in fold["models"].items():
            if model["status"] != "EVALUATED":
                lines.append(f"| {fold['split']['fold']} | {name}: INVALID_EVALUATION | | | | | |")
                continue
            metric = model["classification"]
            lines.append(
                f"| {fold['split']['fold']} | {name} | {metric['accuracy']:.4f} | "
                f"{metric['brier_score']:.4f} | {metric['log_loss']:.4f} | "
                f"{model['threshold_selected_on_validation']} | "
                f"{model['economics']['accepted_trades']} |"
            )
    lines += [
        "",
        "## ML and untouched holdout",
        "",
        json.dumps(report.get("holdout", {})),
        "",
        "Fold classification, calibration, thresholds and economics are in results.json.",
        "",
        "## Risk and uncertainty",
        "",
        "The JSON compares sizing, loss brakes, confidence/EV filters and payouts.",
        "Bootstrap intervals are pointwise 95%; Wilson intervals have a family-size adjustment.",
        "Intervals depend on block length, stationarity, costs and the tested family.",
        "",
        "## Failure analysis and recommendation",
        "",
        *report.get(
            "evidence_limits", ["Authentic data or a protected evaluation split is unavailable."]
        ),
        "Collect authentic data; continue paper research. No execution recommendation.",
        "",
        "## Reproducibility",
        "",
        f"Revision: {report['code_revision']}",
        f"Config fingerprint: {report['config_fingerprint']}",
        "Configuration, fingerprints, seed, assumptions and package versions are in results.json.",
    ]
    lines += [
        "",
        "## Risk scenarios (descriptive only)",
        "",
        "| Candidate | Fixed P&L | Percent P&L | Strict brakes P&L | Confidence/EV P&L |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, scenarios in report.get("evaluation", {}).get("risk_research", {}).items():
        values = [
            scenarios[key]["total_simulated_pnl"]
            for key in (
                "fixed",
                "percent",
                "strict_loss_limits",
                "confidence_and_ev",
            )
        ]
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
