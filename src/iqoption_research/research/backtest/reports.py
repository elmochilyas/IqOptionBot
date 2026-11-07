"""Research report generation and reproducibility exports.

Serializes full backtest results, dataset provenance, parameters, metrics,
and trade-by-trade logs into reproducible JSON and CSV artifacts.
Zero sensitive credentials or personal IDs are ever included.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.research.backtest.engine import BacktestRunResult
from iqoption_research.research.backtest.metrics import BacktestMetrics, calculate_metrics
from iqoption_research.research.datasets.loader import LoadedHistoricalDataset
from iqoption_research.research.datasets.splitter import DatasetSplits


@dataclass(frozen=True, slots=True)
class ResearchReport:
    """Structured research report containing full run metadata and metrics."""

    run_id: str
    created_at_utc: str
    config_hash: str
    dataset_id: str
    symbol: str
    timeframe_seconds: int
    is_synthetic: bool
    source_provenance: str
    strategy_name: str
    payoff_model: str
    initial_capital: float
    final_capital: float
    total_net_pnl: float
    metrics: BacktestMetrics
    execution_assumptions: dict[str, Any]
    split_bounds: dict[str, Any] | None
    trade_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "config_hash": self.config_hash,
            "dataset": {
                "dataset_id": self.dataset_id,
                "symbol": self.symbol,
                "timeframe_seconds": self.timeframe_seconds,
                "is_synthetic": self.is_synthetic,
                "source_provenance": self.source_provenance,
            },
            "strategy": {
                "name": self.strategy_name,
                "payoff_model": self.payoff_model,
            },
            "capital": {
                "initial": round(self.initial_capital, 2),
                "final": round(self.final_capital, 2),
                "net_pnl": round(self.total_net_pnl, 2),
            },
            "execution_assumptions": self.execution_assumptions,
            "split_bounds": self.split_bounds,
            "metrics": self.metrics.to_dict(),
            "trade_count": self.trade_count,
        }

    def export_json(self, destination: Path | str) -> Path:
        """Export research report to a JSON file."""
        dest = Path(destination).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return dest

    def export_trades_csv(self, destination: Path | str, run_result: BacktestRunResult) -> Path:
        """Export trade-by-trade log to a CSV file."""
        dest = Path(destination).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trade_id",
                "direction",
                "entry_bar",
                "entry_ts",
                "entry_utc",
                "entry_price",
                "exit_bar",
                "exit_ts",
                "exit_utc",
                "exit_price",
                "stake",
                "outcome",
                "net_pnl",
                "capital_after",
                "notes",
            ])
            for t in run_result.trades:
                writer.writerow([
                    t.trade_id,
                    t.direction.value,
                    t.entry_bar_index,
                    t.entry_ts,
                    t.entry_utc,
                    f"{t.entry_price:.5f}",
                    t.exit_bar_index,
                    t.exit_ts,
                    t.exit_utc,
                    f"{t.exit_price:.5f}",
                    f"{t.stake:.2f}",
                    t.settlement.outcome.value,
                    f"{t.settlement.net_pnl:.2f}",
                    f"{t.capital_after:.2f}",
                    t.settlement.notes,
                ])
        return dest


def generate_research_report(
    run_result: BacktestRunResult,
    dataset: LoadedHistoricalDataset | None = None,
    splits: DatasetSplits[Any] | None = None,
) -> ResearchReport:
    """Generate a structured, reproducible research report from a backtest run."""
    metrics = calculate_metrics(run_result)
    cfg = run_result.config

    # Deterministic configuration fingerprint
    config_dict = {
        "strategy": run_result.strategy_name,
        "payoff": run_result.payoff_model_name,
        "initial_capital": cfg.initial_capital,
        "stake_amount": cfg.stake_amount,
        "stake_mode": cfg.stake_mode,
        "latency_bars": cfg.latency_bars,
        "slippage_pips": cfg.slippage_pips,
        "spread_pips": cfg.spread_pips,
        "default_horizon_bars": cfg.default_horizon_bars,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    cfg_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    run_id = f"run_{run_result.strategy_name}_{cfg_hash}_{int(datetime.now(UTC).timestamp())}"

    # Extract dataset metadata safely
    ds_id = (
        dataset.catalog_entries[0].dataset_id
        if (dataset and dataset.catalog_entries)
        else "unregistered"
    )
    symbol = dataset.symbol if dataset else "EURUSD"
    tf = dataset.candle_size_seconds if dataset else 60
    is_synth = dataset.is_synthetic if dataset else True
    src = dataset.source if dataset else "offline"

    split_dict: dict[str, Any] | None = None
    if splits is not None:
        split_dict = {
            "train": {
                "count": splits.train_boundary.count,
                "start_utc": splits.train_boundary.start_utc,
                "end_utc": splits.train_boundary.end_utc,
                "purged_count": splits.purged_train_count,
            },
            "val": {
                "count": splits.val_boundary.count,
                "start_utc": splits.val_boundary.start_utc,
                "end_utc": splits.val_boundary.end_utc,
                "embargo_count": splits.embargo_val_count,
            },
            "test": {
                "count": splits.test_boundary.count,
                "start_utc": splits.test_boundary.start_utc,
                "end_utc": splits.test_boundary.end_utc,
                "embargo_count": splits.embargo_test_count,
            },
        }

    return ResearchReport(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        config_hash=cfg_hash,
        dataset_id=ds_id,
        symbol=symbol,
        timeframe_seconds=tf,
        is_synthetic=is_synth,
        source_provenance=src,
        strategy_name=run_result.strategy_name,
        payoff_model=run_result.payoff_model_name,
        initial_capital=run_result.initial_capital,
        final_capital=run_result.final_capital,
        total_net_pnl=run_result.total_net_pnl,
        metrics=metrics,
        execution_assumptions=config_dict,
        split_bounds=split_dict,
        trade_count=len(run_result.trades),
    )
