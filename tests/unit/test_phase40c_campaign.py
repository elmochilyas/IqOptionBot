"""Unit tests for Phase 40C Campaign Management, Scheduling, and Registry."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner

from iqoption_research.campaign.manager import CampaignManager
from iqoption_research.campaign.models import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BREAK_EVEN_HURDLE,
    CampaignProtocolSpec,
    CampaignRecord,
    CampaignState,
    SlotState,
)
from iqoption_research.campaign.registry import CampaignRegistry
from iqoption_research.campaign.scheduler import generate_campaign_slots
from iqoption_research.cli.main import app
from iqoption_research.shadow.registry import ShadowModelRegistry


def _generate_synthetic_parquet(path: Path, start_ts: int, n_candles: int) -> None:
    import math

    import pyarrow as pa
    import pyarrow.parquet as pq

    from iqoption_research.market_data.parquet_storage import CANDLE_ARROW_SCHEMA

    path.parent.mkdir(parents=True, exist_ok=True)
    from_ts = [start_ts + i * 60 for i in range(n_candles)]
    to_ts = [t + 60 for t in from_ts]
    at_ts = [t + 60 for t in from_ts]

    opens = [1.0500 + 0.0010 * math.sin(i / 10.0) for i in range(n_candles)]
    highs = [p + 0.0004 for p in opens]
    lows = [p - 0.0004 for p in opens]
    closes = [opens[i] + 0.0002 * math.cos(i / 8.0) for i in range(n_candles)]

    table = pa.Table.from_arrays(
        [
            pa.array(["EURUSD"] * n_candles, type=pa.string()),
            pa.array([1] * n_candles, type=pa.int32()),
            pa.array([False] * n_candles, type=pa.bool_()),
            pa.array([60] * n_candles, type=pa.int32()),
            pa.array(from_ts, type=pa.int64()),
            pa.array(to_ts, type=pa.int64()),
            pa.array(at_ts, type=pa.int64()),
            pa.array(opens, type=pa.float64()),
            pa.array(highs, type=pa.float64()),
            pa.array(lows, type=pa.float64()),
            pa.array(closes, type=pa.float64()),
            pa.array([150] * n_candles, type=pa.int64()),
            pa.array(closes, type=pa.float64()),
            pa.array([c + 0.00005 for c in closes], type=pa.float64()),
            pa.array([True] * n_candles, type=pa.bool_()),
            pa.array(list(range(n_candles)), type=pa.int32()),
        ],
        schema=CANDLE_ARROW_SCHEMA,
    )
    pq.write_table(table, str(path))


def _setup_candidate_workspace(tmp_path: Path) -> str:
    from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
    from iqoption_research.ml.laboratory import ExperimentConfig, ModelLaboratory
    from iqoption_research.research.dataset_manager import ResearchDatasetManager

    p = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    start_epoch = 1700000000
    n_candles = 600
    _generate_synthetic_parquet(p, start_ts=start_epoch, n_candles=n_candles)

    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        CatalogEntry(
            dataset_id="test_series_600",
            schema_version="1.0.0",
            source="test",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=start_epoch,
            to_ts=start_epoch + (n_candles - 1) * 60,
            from_utc=datetime.fromtimestamp(start_epoch, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(start_epoch + (n_candles - 1) * 60, tz=UTC).isoformat(),
            candle_count=n_candles,
            created_at_utc=datetime.now(UTC).isoformat(),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary={},
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="PHASE24",
            file_path=str(p),
            sha256_checksum=sha,
            metadata_path="",
            metadata_sha256="",
        )
    )

    mgr = ResearchDatasetManager(tmp_path)
    snapshot = mgr.build_snapshot()
    snapshot_id = snapshot.manifest.snapshot_id

    lab = ModelLaboratory(tmp_path)
    cfg1 = ExperimentConfig(
        model_family="logistic_regression",
        snapshot_id=snapshot_id,
        horizon_bars=1,
        min_confidence=0.52,
        calibrate=True,
    )
    res1 = lab.run_experiment(cfg1)

    reg = ShadowModelRegistry(tmp_path)
    cand = reg.register_candidate(
        res1.model_id,
        confidence_threshold=0.52,
        notes="Registered LR candidate for Campaign",
    )
    return cand.model_id


# ---------------------------------------------------------------------------
# 1. Calendar Scheduling & Timezone Tests
# ---------------------------------------------------------------------------


def test_generate_campaign_slots_weekdays_and_count() -> None:
    start_d = date(2026, 11, 23)  # A Monday
    slots = generate_campaign_slots(start_date=start_d, num_market_days=40)

    assert len(slots) == 80
    assert slots[0].day_index == 1
    assert slots[0].window_index == 1
    assert slots[-1].day_index == 40
    assert slots[-1].window_index == 2

    casablanca_tz = ZoneInfo("Africa/Casablanca")
    dates_seen: set[str] = set()

    for slot in slots:
        # Verify 08:00 and 14:00 UTC
        utc_dt = datetime.fromisoformat(slot.scheduled_start_utc)
        assert utc_dt.tzinfo is not None
        assert utc_dt.hour in (8, 14)
        assert utc_dt.minute == 0

        # Verify no weekend slots (0=Monday, 6=Sunday)
        assert utc_dt.weekday() < 5, (
            f"Slot {slot.slot_id} fell on weekend: weekday={utc_dt.weekday()}"
        )
        dates_seen.add(slot.scheduled_date)

        # Verify Casablanca local time translation
        casa_dt = datetime.fromisoformat(slot.scheduled_start_local)
        assert casa_dt.tzinfo is not None
        expected_casa = utc_dt.astimezone(casablanca_tz)
        assert casa_dt.hour == expected_casa.hour
        assert casa_dt.minute == expected_casa.minute

    assert len(dates_seen) == 40


def test_campaign_protocol_canonical_hash_determinism() -> None:
    now_utc = datetime(2026, 11, 23, 12, 0, 0, tzinfo=UTC).isoformat()
    spec1 = CampaignProtocolSpec(
        campaign_id="camp_test_hash",
        protocol_version="40C_CONFIRMATORY_V1",
        symbol="EURUSD",
        timeframe=60,
        candidate_model_id="cand_lr_model_123",
        candidate_threshold=0.55,
        baseline_model_id="baseline_majority_class",
        baseline_threshold=0.55,
        assumed_payout=0.85,
        break_even_hurdle=DEFAULT_BREAK_EVEN_HURDLE,
        min_total_market_days=40,
        min_completed_days=30,
        min_completed_sessions=60,
        min_decisive_outcomes=2400,
        windows_per_day=2,
        window_start_times_utc=("08:00", "14:00"),
        local_timezone_name="Africa/Casablanca",
        bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        alpha=0.05,
        created_at_utc=now_utc,
    )
    spec2 = CampaignProtocolSpec(
        campaign_id="camp_test_hash",
        protocol_version="40C_CONFIRMATORY_V1",
        symbol="EURUSD",
        timeframe=60,
        candidate_model_id="cand_lr_model_123",
        candidate_threshold=0.55,
        baseline_model_id="baseline_majority_class",
        baseline_threshold=0.55,
        assumed_payout=0.85,
        break_even_hurdle=DEFAULT_BREAK_EVEN_HURDLE,
        min_total_market_days=40,
        min_completed_days=30,
        min_completed_sessions=60,
        min_decisive_outcomes=2400,
        windows_per_day=2,
        window_start_times_utc=("08:00", "14:00"),
        local_timezone_name="Africa/Casablanca",
        bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        alpha=0.05,
        created_at_utc=now_utc,
    )
    h1 = spec1.compute_sha256()
    h2 = spec2.compute_sha256()
    assert h1 == h2
    assert len(h1) == 64

    # Any modification changes protocol hash
    spec_diff = CampaignProtocolSpec(
        campaign_id="camp_test_hash",
        protocol_version="40C_CONFIRMATORY_V1",
        symbol="EURUSD",
        timeframe=60,
        candidate_model_id="cand_lr_model_123",
        candidate_threshold=0.56,  # 0.55 -> 0.56
        baseline_model_id="baseline_majority_class",
        baseline_threshold=0.55,
        assumed_payout=0.85,
        break_even_hurdle=DEFAULT_BREAK_EVEN_HURDLE,
        min_total_market_days=40,
        min_completed_days=30,
        min_completed_sessions=60,
        min_decisive_outcomes=2400,
        windows_per_day=2,
        window_start_times_utc=("08:00", "14:00"),
        local_timezone_name="Africa/Casablanca",
        bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        alpha=0.05,
        created_at_utc=now_utc,
    )
    assert spec_diff.compute_sha256() != h1


# ---------------------------------------------------------------------------
# 2. Campaign Registry & State Machine Tests
# ---------------------------------------------------------------------------


def test_campaign_registry_lifecycle(tmp_path: Path) -> None:
    registry = CampaignRegistry(tmp_path)
    now_utc = datetime.now(UTC).isoformat()
    slots = generate_campaign_slots(start_date=date(2026, 12, 1), num_market_days=40)
    spec = CampaignProtocolSpec(
        campaign_id="camp_20261201_test",
        protocol_version="40C_CONFIRMATORY_V1",
        symbol="EURUSD",
        timeframe=60,
        candidate_model_id="cand_model_001",
        candidate_threshold=0.52,
        baseline_model_id="baseline_majority_class",
        baseline_threshold=0.55,
        assumed_payout=0.85,
        break_even_hurdle=DEFAULT_BREAK_EVEN_HURDLE,
        min_total_market_days=40,
        min_completed_days=30,
        min_completed_sessions=60,
        min_decisive_outcomes=2400,
        windows_per_day=2,
        window_start_times_utc=("08:00", "14:00"),
        local_timezone_name="Africa/Casablanca",
        bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        alpha=0.05,
        created_at_utc=now_utc,
    )
    record = CampaignRecord(
        campaign_id="camp_20261201_test",
        state=CampaignState.DRAFT.value,
        protocol=spec,
        slots=slots,
        created_at_utc=now_utc,
    )

    # 1. Save Draft
    registry.save(record)
    assert len(registry.list_all()) == 1

    # 2. Register / Lock Campaign
    registered = registry.register("camp_20261201_test")
    assert registered.state == CampaignState.REGISTERED.value
    assert registered.registered_at_utc is not None
    assert len(registered.protocol.protocol_sha256) == 64

    # 3. Verify on disk files
    camp_dir = tmp_path / "data" / "campaigns" / "camp_20261201_test"
    assert (camp_dir / "campaign.json").is_file()
    assert (camp_dir / "protocol.json").is_file()
    assert (camp_dir / "slots.csv").is_file()

    # 4. Slot status and update
    slot1 = registry.get_slot("camp_20261201_test", slots[0].slot_id)
    assert slot1 is not None
    assert slot1.slot_state == SlotState.PLANNED.value

    updated_slot = registry.update_slot_state(
        "camp_20261201_test",
        slots[0].slot_id,
        new_state=SlotState.COMPLETED_VERIFIED.value,
        session_id="pobs_sess_001",
    )
    assert updated_slot.slot_state == SlotState.COMPLETED_VERIFIED.value
    assert updated_slot.session_id == "pobs_sess_001"


# ---------------------------------------------------------------------------
# 3. Campaign Manager Tests
# ---------------------------------------------------------------------------


def test_campaign_manager_prechecks_and_session_linking(tmp_path: Path) -> None:
    model_id = _setup_candidate_workspace(tmp_path)
    mgr = CampaignManager(root=tmp_path)

    # Creating campaign with invalid/unregistered model fails
    with pytest.raises(ValueError, match="not in frozen registry"):
        mgr.create_campaign(
            campaign_id="camp_invalid_model",
            start_date=date(2026, 12, 1),
            candidate_model_id="cand_does_not_exist",
        )

    # Creating draft with registered model succeeds
    draft = mgr.create_campaign(
        campaign_id="camp_valid_001",
        start_date=date(2026, 12, 1),
        candidate_model_id=model_id,
        candidate_threshold=0.52,
    )
    assert draft.state == CampaignState.DRAFT.value
    assert draft.protocol.candidate_model_id == model_id

    # Register campaign
    registered = mgr.register_campaign("camp_valid_001")
    assert registered.state == CampaignState.REGISTERED.value
    assert len(registered.protocol.protocol_sha256) == 64


# ---------------------------------------------------------------------------
# 4. CLI Subcommand Tests
# ---------------------------------------------------------------------------


def test_cli_campaign_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = _setup_candidate_workspace(tmp_path)

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    # 1. campaign create
    res_create = runner.invoke(
        app,
        [
            "campaign",
            "create",
            "camp_cli_test_01",
            "--candidate-model-id",
            model_id,
            "--start-date",
            "2026-12-01",
        ],
    )
    assert res_create.exit_code == 0
    assert "Research Campaign Created in DRAFT state" in res_create.stdout

    # 2. campaign list
    res_list = runner.invoke(app, ["campaign", "list"])
    assert res_list.exit_code == 0
    assert "camp_cli" in res_list.stdout

    # 3. campaign register
    res_reg = runner.invoke(app, ["campaign", "register", "camp_cli_test_01"])
    assert res_reg.exit_code == 0
    assert "Pre-registration Formalized & Locked" in res_reg.stdout
    assert "Protocol SHA-256" in res_reg.stdout

    # 4. campaign status
    res_stat = runner.invoke(app, ["campaign", "status", "camp_cli_test_01"])
    assert res_stat.exit_code == 0
    assert "REGISTERED" in res_stat.stdout
    assert "camp_cli_test_01" in res_stat.stdout

    # 5. campaign export
    res_exp_json = runner.invoke(
        app, ["campaign", "export", "camp_cli_test_01", "--format", "json"]
    )
    assert res_exp_json.exit_code == 0
    assert '"campaign_id": "camp_cli_test_01"' in res_exp_json.stdout

    res_exp_csv = runner.invoke(app, ["campaign", "export", "camp_cli_test_01", "--format", "csv"])
    assert res_exp_csv.exit_code == 0
    assert "slot_id,day_index" in res_exp_csv.stdout
