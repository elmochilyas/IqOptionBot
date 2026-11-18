"""Predeclared interpretable research with shared clocks and sealed confirmation data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from iqoption_research.ml.phase10 import (
    CleanResearchDataset,
    Phase10Config,
    audit_authentic_datasets,
    build_walk_forward_folds,
)
from iqoption_research.research.evaluation_metrics import SimulationPolicy, simulate
from iqoption_research.research.phase24 import (
    EvaluationConfig,
    assert_split,
    prepared_data,
    run_evaluation,
    strategy_directions,
)
from iqoption_research.research.phase25_data import (
    aggregate_complete,
    expansion_diagnostics,
    expansion_plan,
)
from iqoption_research.research.phase25_protocol import day_block_inference, register_protocol

FAMILY = (
    "no_trade",
    "always_call",
    "always_put",
    "random",
    "momentum",
    "ma_crossover",
    "inverse_momentum",
    "inverse_ma_crossover",
)


def shared_clock(data: CleanResearchDataset, indices: np.ndarray) -> np.ndarray:
    """Reserve a common decision-to-expiry slot even when a candidate abstains."""
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Shared opportunities must be chronological")
    selected: list[int] = []
    busy = 0
    for index in indices:
        if data.timestamps[index] >= busy:
            selected.append(int(index))
            busy = int(data.exit_timestamps[index])
    return np.asarray(selected, dtype=np.int64)


def unit_payoffs(
    data: CleanResearchDataset, indices: np.ndarray, direction: np.ndarray, policy: SimulationPolicy
) -> np.ndarray:
    """Exact fixed-return reference economics, without risk-stop sample censoring."""
    if len(indices) != len(direction) or np.any(~np.isin(direction, [-1, 0, 1])):
        raise ValueError("Invalid primary research signals")
    difference = (
        direction * (data.exit_prices[indices] - data.entry_prices[indices]) - policy.slippage_price
    )
    gross = np.where(np.abs(difference) < 1e-9, 0.0, np.where(difference > 0, policy.payout, -1.0))
    return np.where(direction != 0, gross - policy.cost_rate, 0.0)


def session_name(timestamp: int) -> str:
    """Reference local working-hour sessions, including DST; not broker hours."""
    utc = datetime.fromtimestamp(timestamp, UTC)
    london, new_york = (
        utc.astimezone(ZoneInfo("Europe/London")),
        utc.astimezone(ZoneInfo("America/New_York")),
    )
    ldn = london.weekday() < 5 and 8 <= london.hour < 17
    ny = new_york.weekday() < 5 and 8 <= new_york.hour < 17
    return "OVERLAP" if ldn and ny else "LONDON_ONLY" if ldn else "NEW_YORK_ONLY" if ny else "OTHER"


def _summary(values: np.ndarray, direction: np.ndarray) -> dict[str, Any]:
    trades = int(np.count_nonzero(direction))
    return {
        "opportunities": len(values),
        "trades": trades,
        "coverage": trades / len(values) if len(values) else 0,
        "unit_ev_per_opportunity": float(values.mean()) if len(values) else None,
        "unit_ev_per_trade": float(values.sum() / trades) if trades else None,
    }


def _development_fingerprint(candles: list[Any]) -> str:
    digest = hashlib.sha256()
    for candle in candles:
        digest.update(
            repr((
                candle.symbol,
                int(candle.size),
                candle.from_ts,
                candle.to_ts,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            )).encode()
        )
    return digest.hexdigest()


def _model_failure_analysis(reference: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    evaluation = reference.get("evaluation", {})
    if not any(fold["models"] for fold in evaluation.get("folds", [])):
        return {"status": "NOT_REQUESTED"}
    for name in ("majority_class", "logistic_regression", "random_forest"):
        row = next((r for r in evaluation.get("leaderboard", []) if r["name"] == name), None)
        folds = []
        for fold in evaluation.get("folds", []):
            model = fold["models"].get(name, {})
            majority = fold["models"].get("majority_class", {})
            if model.get("status") == "EVALUATED":
                score = model["classification"]
                folds.append({
                    "fold": fold["split"]["fold"],
                    "accuracy": score["accuracy"],
                    "brier_score": score["brier_score"],
                    "log_loss": score["log_loss"],
                    "calibration_error": score["expected_calibration_error"],
                    "brier_improvement_over_majority": (
                        majority["classification"]["brier_score"] - score["brier_score"]
                        if majority.get("status") == "EVALUATED"
                        else None
                    ),
                    "selected_threshold": model["threshold_selected_on_validation"],
                    "trade_coverage": model["economics"]["coverage"],
                })
            else:
                folds.append({"fold": fold["split"]["fold"], "status": "INVALID_EVALUATION"})
        output[name] = {
            "folds": folds,
            "pooled_economics": row,
            "interpretation": "DEVELOPMENT_DIAGNOSTIC_NOT_A_NEW_SELECTION_TRIAL",
        }
    return output


def run_phase25(
    storage: Path,
    cfg: EvaluationConfig,
    reservation: Path,
    registry: Path,
    *,
    mode: str = "all",
    bootstrap_draws: int = 4000,
) -> dict[str, Any]:
    cfg.validate()
    if bootstrap_draws < 100 or bootstrap_draws > 10000:
        raise ValueError("Bounded bootstrap budget required")
    if mode not in {
        "all",
        "coverage",
        "integrity",
        "benchmark",
        "walk-forward",
        "compare",
        "report",
    }:
        raise ValueError("Unknown research mode")
    metadata = run_evaluation(storage, cfg, reservation, mode="coverage")
    coverage = metadata["coverage"]
    configuration = {"phase": 25, "mode": mode, "bootstrap_draws": bootstrap_draws, **asdict(cfg)}
    report: dict[str, Any] = {
        "phase": 25,
        "status": "NO_DEMONSTRATED_EDGE",
        "coverage": coverage,
        "configuration": configuration,
        "configuration_fingerprint": hashlib.sha256(
            json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "code_revision": metadata["code_revision"],
        "versions": metadata["versions"],
        "implementation_fingerprint": metadata["implementation_fingerprint"],
        "assumptions": metadata["assumptions"],
        "safety": {
            "offline_only": True,
            "broker_orders": 0,
            "account_switching": False,
            "autonomous_execution_authorized": False,
        },
        "product_evidence": {
            "EURUSD_OP": "NO_VERIFIED_NATIVE_CANDLE_DATA_OR_CONNECTOR_MAPPING",
            "blitz_current_payout": "UNVERIFIED",
            "blitz_settlement_schema": "UNVERIFIED",
            "payout_source": "ASSUMED_NET_RETURN_NOT_TRANSACTION_HISTORY",
            "vendor_reference": "https://blog.iqoption.com/en/iq-option-introduces-blitz-options-up-to-95-profit-from-5-seconds/",
        },
    }
    if cfg.symbol != "EURUSD" or cfg.timeframe != 60:
        report["data_status"] = "INSUFFICIENT_AUTHENTIC_DATA"
        return report
    group = next(
        (
            g
            for g in coverage["datasets"]
            if g["symbol"] == cfg.symbol
            and g["source"] == cfg.source
            and g["timeframe_seconds"] == 60
            and g["status"] == "INTEGRITY_PASSED"
        ),
        None,
    )
    if group is None:
        report["data_status"] = "INSUFFICIENT_AUTHENTIC_DATA"
        return report
    plan = expansion_plan(storage)
    report["expansion"] = expansion_diagnostics(plan)
    report["expansion"]["execution"] = "OFFLINE_PLAN_ONLY_NO_CREDENTIAL_ACCESS"
    if mode in {"coverage", "integrity"}:
        return report
    state_bytes = reservation.read_bytes()
    state = json.loads(state_bytes)
    if not isinstance(state, dict):
        raise ValueError("Invalid confirmation reservation")
    cutoff = state.get("prior_evaluation_cutoff_ts")
    if state.get("evaluation_status") != "RESERVED" or type(cutoff) is not int:
        raise ValueError("A sealed confirmation reservation is required")
    audit, all_candles = audit_authentic_datasets(storage)
    candles = [c for c in all_candles if c.to_ts < cutoff]
    report["holdout"] = {
        "status": "SEALED_NOT_EVALUATED",
        "available_candles": sum(c.from_ts >= cutoff for c in all_candles),
        "reservation_fingerprint": hashlib.sha256(state_bytes).hexdigest(),
        "integrity_audit_only": True,
    }
    report["dataset_fingerprint"] = audit.aggregate_fingerprint
    report["development_fingerprint"] = _development_fingerprint(candles)
    report["development_candles"] = len(candles)
    if len(candles) < cfg.min_rows:
        report["data_status"] = "INSUFFICIENT_AUTHENTIC_DATA"
        return report
    frames: dict[int, tuple[Any, ...]] = {}
    for seconds in (60, 300):
        sample, provenance = (
            (candles, {"provenance": "BROKER_NATIVE_AUTHENTIC", "timeframe_seconds": 60})
            if seconds == 60
            else aggregate_complete(candles, seconds)
        )
        setting = replace(cfg, timeframe=seconds)
        clean = prepared_data(sample, setting)
        if clean.row_count < max(150, cfg.min_rows // (seconds // 60)):
            raise ValueError("Insufficient complete timeframe samples")
        folds = build_walk_forward_folds(
            clean,
            Phase10Config(
                n_folds=cfg.folds,
                initial_train_ratio=0.4,
                validation_ratio=0.1,
                test_ratio=0.5 / cfg.folds,
                horizon_bars=cfg.horizon + cfg.latency_bars + 1,
            ),
        )
        for fold in folds:
            assert_split(
                clean,
                np.arange(fold.train_start, fold.train_end),
                np.arange(fold.validation_start, fold.validation_end),
                np.arange(fold.test_start, fold.test_end),
            )
        clock = shared_clock(
            clean, np.concatenate([np.arange(f.test_start, f.test_end) for f in folds])
        )
        frames[seconds] = (sample, clean, folds, clock, setting, provenance)
    implementation = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        implementation.update(
            path.relative_to(Path(__file__).resolve().parents[1]).as_posix().encode()
        )
        implementation.update(path.read_bytes())
    protocol = {
        "schema": 1,
        "family": list(FAMILY),
        "timeframes": [60, 300],
        "config": asdict(cfg),
        "development_fingerprint": report["development_fingerprint"],
        "implementation_fingerprint": implementation.hexdigest(),
        "bootstrap_draws": bootstrap_draws,
        "primary": "UNCONDITIONAL_UNIT_EV_ON_SHARED_NONOVERLAPPING_CLOCK",
        "nulls": ["ZERO_NET_RETURN", "EXACT_FAIR_RANDOM_DIRECTION_EXPECTATION"],
        "min_test_days": 30,
        "min_trades": cfg.min_trades,
        "random_realizations": 64,
        "confirmation": "NOT_EVALUATED_PRIOR_RESEARCH_UNREGISTERED",
    }
    registered = register_protocol(registry, protocol)
    report["registered_protocol"] = registered
    report["protocol"] = protocol
    candidates: list[dict[str, Any]] = []
    cluster_values: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    report["timeframes"] = []
    for seconds, (sample, data, folds, clock, _setting, provenance) in frames.items():
        directions = strategy_directions(data, sample, cfg.seed)
        directions["inverse_momentum"] = -directions["momentum"]
        directions["inverse_ma_crossover"] = -directions["ma_crossover"]
        call = unit_payoffs(data, clock, np.ones(len(clock)), cfg.policy)
        put = unit_payoffs(data, clock, -np.ones(len(clock)), cfg.policy)
        random_expectation = (call + put) / 2
        rng = np.random.default_rng(cfg.seed)
        null_totals = [
            float(unit_payoffs(data, clock, rng.choice([-1, 1], len(clock)), cfg.policy).sum())
            for _ in range(64)
        ]
        days = data.timestamps[clock] // 86400
        sessions = np.asarray([session_name(int(ts)) for ts in data.timestamps[clock]])
        report["timeframes"].append({
            "seconds": seconds,
            "provenance": provenance,
            "fingerprint": data.fingerprint,
            "full_causal_rows": data.row_count,
            "test_opportunities": len(clock),
            "test_days": len(np.unique(days)),
            "reference_session_counts": {
                str(s): int(np.sum(sessions == s)) for s in np.unique(sessions)
            },
            "random_64_realizations_total_unit_return_interval": np.quantile(
                null_totals, [0.025, 0.975]
            ).tolist(),
            "random_expected_unit_ev": float(random_expectation.mean()),
            "duration_seconds": cfg.horizon * seconds,
            "latency_seconds": cfg.latency_bars * seconds,
        })
        for name in FAMILY:
            direction = directions[name][clock]
            values = unit_payoffs(data, clock, direction, cfg.policy)
            fold_reports = []
            for fold in folds:
                mask = (clock >= fold.test_start) & (clock < fold.test_end)
                training = np.arange(fold.train_start, fold.train_end)
                volatility = data.feature_names.index("volatility_20")
                boundary = float(np.median(data.X[training, volatility]))
                low = data.X[clock, volatility] <= boundary
                fold_reports.append({
                    "fold": fold.fold,
                    **_summary(values[mask], direction[mask]),
                    "regimes_train_boundary": {
                        "low_volatility": _summary(values[mask & low], direction[mask & low]),
                        "high_volatility": _summary(values[mask & ~low], direction[mask & ~low]),
                    },
                })
            uncensored = replace(
                cfg.policy,
                initial_capital=1_000_000,
                fixed_stake=1,
                daily_loss_fraction=1,
                drawdown_fraction=1,
                max_exposure_fraction=1,
                consecutive_loss_limit=len(clock) + 1,
                sizing="fixed",
                min_confidence=0,
                min_ev=-1,
            )
            guarded = simulate(data, clock, direction, cfg.policy, bootstrap=False)
            counterfactual = simulate(data, clock, direction, uncensored, bootstrap=False)
            actionable = int(np.count_nonzero(direction))
            sensitivity = []
            for payout in cfg.payout_sensitivity:
                v = unit_payoffs(data, clock, direction, replace(cfg.policy, payout=payout))
                sensitivity.append({"payout": payout, **_summary(v, direction)})
            for slippage in (0.0, cfg.policy.slippage_price, cfg.policy.slippage_price * 2):
                v = unit_payoffs(
                    data, clock, direction, replace(cfg.policy, slippage_price=slippage)
                )
                sensitivity.append({"slippage_price": slippage, **_summary(v, direction)})
            candidates.append({
                "name": name,
                "timeframe_seconds": seconds,
                **_summary(values, direction),
                "uncensored_counterfactual": counterfactual,
                "risk_brake_excluded_fraction": (
                    1 - guarded["accepted_trades"] / actionable if actionable else None
                ),
                "raw_win_rate_minus_break_even": (
                    counterfactual["win_rate"] - counterfactual["break_even_decisive_win_rate"]
                    if counterfactual["win_rate"] is not None
                    else None
                ),
                "guarded_economics": guarded,
                "folds": fold_reports,
                "sessions_descriptive_not_selected": {
                    str(s): _summary(values[sessions == s], direction[sessions == s])
                    for s in np.unique(sessions)
                },
                "sensitivity_descriptive": sensitivity,
            })
            cluster_values.append((days, values, random_expectation))
    all_days = np.unique(np.concatenate([row[0] for row in cluster_values]))
    matrix = np.zeros((len(all_days), 2 * len(candidates)))
    counts = np.zeros_like(matrix)
    for j, (days, values, random_values) in enumerate(cluster_values):
        for i, day in enumerate(all_days):
            mask = days == day
            matrix[i, 2 * j] = values[mask].sum()
            matrix[i, 2 * j + 1] = (values - random_values)[mask].sum()
            counts[i, 2 * j : 2 * j + 2] = mask.sum()
    inference = day_block_inference(
        matrix, counts, alpha=registered["alpha"], seed=cfg.seed, resamples=bootstrap_draws
    )
    span_days = (candles[-1].to_ts - candles[0].from_ts) / 86400
    for j, candidate in enumerate(candidates):
        absolute, excess = inference[2 * j : 2 * j + 2]
        candidate["absolute_ev_inference"] = absolute
        candidate["advantage_over_random_inference"] = excess
        failures = []
        if candidate["trades"] < cfg.min_trades:
            failures.append("MINIMUM_TRADE_COUNT")
        if absolute["day_clusters"] < 30 or span_days < cfg.min_calendar_days:
            failures.append("INSUFFICIENT_INDEPENDENT_PERIOD_COVERAGE")
        if absolute["simultaneous_lower_bound"] <= 0 or excess["simultaneous_lower_bound"] <= 0:
            failures.append("SIMULTANEOUS_EVIDENCE_NOT_POSITIVE")
        if not absolute["tail_resolution_adequate"]:
            failures.append("MONTE_CARLO_TAIL_RESOLUTION")
        if any(
            f["unit_ev_per_opportunity"] is None or f["unit_ev_per_opportunity"] <= 0
            for f in candidate["folds"]
        ):
            failures.append("NOT_POSITIVE_IN_ALL_FOLDS")
        if any(
            s["unit_ev_per_opportunity"] is None or s["unit_ev_per_opportunity"] <= 0
            for s in candidate["sensitivity_descriptive"]
        ):
            failures.append("COST_PAYOUT_ROBUSTNESS")
        candidate["failed_development_gates"] = failures
        candidate["classification"] = (
            "NO_DEMONSTRATED_EDGE" if failures else "PROMISING_REQUIRES_FROZEN_CONFIRMATION"
        )
    report["candidates"] = candidates
    # Reference models retain all Phase 24 training/calibration/threshold protections.
    reference = run_evaluation(
        storage, cfg, reservation, mode="benchmark" if mode == "benchmark" else "all"
    )
    report["model_failure_analysis"] = _model_failure_analysis(reference)
    report["code_revision"] = reference["code_revision"]
    report["implementation_fingerprint"] = implementation.hexdigest()
    report["versions"] = reference["versions"]
    if reservation.read_bytes() != state_bytes:
        raise ValueError("Confirmation reservation changed during research")
    report["limitations"] = [
        "No sealed holdout performance was used or evaluated.",
        "Day-block inference is approximate under weak dependence/local stationarity.",
        "Prospective alpha spending does not repair unknown prior research selection bias.",
        "Derived 5-minute bars and assumed payouts do not validate native Blitz execution.",
        "Uncensored results are counterfactual; guarded risk settings remain separate.",
        "Session and volatility summaries are descriptive, not post-hoc selected strategies.",
    ]
    return report


def write_phase25_reports(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Phase 25 research report",
        "",
        f"Conclusion: **{report['status']}**.",
        "Development data only; no autonomous execution authorized.",
        "",
        "## Authentic coverage",
        "",
    ]
    for group in report["coverage"]["datasets"]:
        lines.append(
            f"- {group['symbol']} {group['timeframe_seconds']}s: {group['status']}; "
            f"{group.get('unique_candles', 'UNVERIFIED')} verified unique candles."
        )
    lines += [
        "",
        "## Comparative evidence",
        "",
        "| Seconds | Strategy | Trades | Unit EV/opportunity | Joint lower bound | Result |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in report.get("candidates", []):
        lines.append(
            f"| {row['timeframe_seconds']} | {row['name']} | {row['trades']} | "
            f"{row['unit_ev_per_opportunity']:.6f} | "
            f"{row['absolute_ev_inference']['simultaneous_lower_bound']:.6f} | "
            f"{row['classification']} |"
        )
    lines += [
        "",
        "## Failure analysis",
        "",
        "The JSON compares complete-sample counterfactuals against risk-braked simulations.",
        "It includes all folds, session/regime diagnostics and economic sensitivity.",
        "Reference ML retains the Phase 24 chronological protections.",
        "",
        "## Expansion and product limitations",
        "",
        json.dumps(report.get("expansion", {})),
        "Native Blitz candle, payout and settlement evidence remains unverified.",
        "",
        "## Uncertainty and sealed confirmation",
        "",
        json.dumps(report.get("holdout", {})),
        "The declared candidate/timeframe/null family shares paired two-day block resampling.",
        "The registry spends alpha prospectively; prior selection bias remains unknown.",
        "",
        *report.get("limitations", []),
        "",
        "## Recommendation",
        "",
        "Collect authentic development data; freeze a protocol before fresh confirmation.",
        "No result here authorizes Demo or REAL execution.",
        "",
        "## Reproducibility",
        "",
        f"Code revision: {report.get('code_revision', 'UNAVAILABLE')}",
        f"Development fingerprint: {report.get('development_fingerprint', 'UNAVAILABLE')}",
        "Configuration, registry slot, alpha, fingerprints and results are in results.json.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
