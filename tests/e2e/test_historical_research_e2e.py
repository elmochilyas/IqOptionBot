"""Phase 3 Integration Test: Complete End-to-End Offline Research Workflow.

Tests the full pipeline from raw file to research report:
Raw CSV -> Import & Quality Audit -> Parquet Storage & Catalog Indexing ->
Verification -> Loader -> Causal Feature Pipeline -> Chronological Splitter (Purge/Embargo) ->
Train-Only StandardScaler -> Offline Backtest Engine -> Statistical Metrics ->
Reproducible Report Export.

Strictly verifies that no broker network connections are ever initiated.
"""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.importer import import_and_persist_dataset
from iqoption_research.research.backtest.engine import BacktestConfig, BacktestEngine
from iqoption_research.research.backtest.metrics import calculate_metrics
from iqoption_research.research.backtest.payoffs import BinaryPayoffModel
from iqoption_research.research.backtest.reports import generate_research_report
from iqoption_research.research.backtest.strategies import MomentumStrategy
from iqoption_research.research.datasets.loader import HistoricalDatasetLoader
from iqoption_research.research.datasets.splitter import ChronologicalSplitter
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.features.scalers import StandardScaler

runner = CliRunner()


def test_complete_phase3_end_to_end_research_pipeline(tmp_path: Path) -> None:
    """Execute complete end-to-end research workflow offline."""
    # 1. Create raw CSV fixture
    raw_csv = tmp_path / "raw_eurusd_1m.csv"
    lines = ["timestamp,open,high,low,close,volume"]
    base_ts = 1714560000

    # 50 candles with alternating trend
    price = 1.0850
    for i in range(50):
        ts = base_ts + i * 60
        delta = 0.0003 if i % 2 == 0 else -0.0002
        new_price = round(price + delta, 5)
        op = price
        cl = new_price
        hi = round(max(op, cl) + 0.0002, 5)
        lo = round(min(op, cl) - 0.0002, 5)
        vol = 20 + i
        lines.append(f"{ts},{op:.5f},{hi:.5f},{lo:.5f},{cl:.5f},{vol}")
        price = new_price

    raw_csv.write_text("\n".join(lines), encoding="utf-8")

    # 2. Import into Storage and Catalog
    storage_root = tmp_path / "storage"
    catalog_db = tmp_path / "catalog.db"
    catalog = DatasetCatalog(catalog_db)

    # Mock sockets to guarantee zero network traffic
    with patch("socket.socket") as mock_sock:
        mock_sock.side_effect = AssertionError("Network socket called during offline research!")

        write_res = import_and_persist_dataset(
            raw_csv,
            symbol="EURUSD",
            candle_size_seconds=60,
            storage_root=storage_root,
            catalog=catalog,
            source="test_pipeline",
            is_synthetic=True,
        )
        assert write_res.candle_count == 50
        assert write_res.parquet_path.is_file()
        assert catalog.count_datasets() == 1

        # 3. Load dataset with cryptographic integrity checks
        loader = HistoricalDatasetLoader(catalog, storage_root)
        loaded = loader.load_dataset(symbol="EURUSD", candle_size_seconds=60)
        assert loaded.candle_count == 50
        assert loaded.is_synthetic is True

        # 4. Extract causal features
        pipeline = CausalFeaturePipeline(drop_warmup=True)
        feat_ds = pipeline.transform(loaded.candles)
        assert feat_ds.row_count == 50 - 20  # 30 non-warmup rows

        # 5. Chronological train/validation/test split
        splitter = ChronologicalSplitter(
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            label_horizon_bars=1,
            embargo_bars=1,
        )
        splits = splitter.split(feat_ds.rows)
        assert len(splits.train) > 0
        assert len(splits.val) > 0
        assert len(splits.test) > 0

        # 6. Fit StandardScaler strictly on Train
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(splits.train)
        val_scaled = scaler.transform(splits.val)
        test_scaled = scaler.transform(splits.test)
        assert scaler.is_fitted is True
        assert len(train_scaled) == len(splits.train)
        assert len(val_scaled) == len(splits.val)
        assert len(test_scaled) == len(splits.test)

        # 7. Run backtest with Momentum strategy and Binary payoff
        strategy = MomentumStrategy()
        payoff = BinaryPayoffModel(payout_rate=0.85)
        engine = BacktestEngine(
            strategy=strategy,
            payoff_model=payoff,
            config=BacktestConfig(initial_capital=10000.0, stake_amount=100.0),
        )

        matched_candles = loaded.candles[20:]
        run_res = engine.run(feat_ds.rows, matched_candles)
        assert run_res.trade_count > 0

        # 8. Compute statistical performance metrics
        metrics = calculate_metrics(run_res)
        assert metrics.total_trades == run_res.trade_count
        assert metrics.gross_profit >= 0.0

        # 9. Generate and export research report
        report = generate_research_report(run_res, dataset=loaded, splits=splits)
        json_out = report.export_json(tmp_path / "final_report.json")
        csv_out = report.export_trades_csv(tmp_path / "final_trades.csv", run_res)

        assert json_out.is_file()
        assert csv_out.is_file()


def test_cli_research_demo_command(tmp_path: Path) -> None:
    """CLI 'research demo' command executes with zero exit code and expected output."""
    demo_dir = tmp_path / "demo_space"
    res = runner.invoke(app, ["research", "demo", "--workspace", str(demo_dir)])
    assert res.exit_code == 0
    assert "IQ Option Research Phase 3" in res.stdout
    assert "Demonstration Completed Successfully" in res.stdout
