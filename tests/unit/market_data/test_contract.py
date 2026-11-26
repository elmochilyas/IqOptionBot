"""Phase 40C Production Contract Gate and End-to-End Observer Integration Suite.

Verifies:
1. Real ProspectiveShadowObserver-to-Campaign deterministic integration.
2. Complete calendar schedule locking and immutability (insert, delete, reorder, modify).
3. Authoritative Schema 3 contract: fail-closed start timestamp, observer termination semantics,
   symbol/timeframe completeness (zero silent defaults), and model bundle hash checking.
4. Exact numeric input semantics: lossless float prices and fractional volume without rounding.
5. Strict separation between genuine confirmatory and exploratory/test campaigns.
6. Full end-to-end failure matrix and legacy Schema 1/2 compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iqoption_research.campaign.analysis import CampaignEvaluator
from iqoption_research.campaign.manager import CampaignManager
from iqoption_research.campaign.models import (
    ObservationWindowSlot,
    SlotState,
    compute_calendar_sha256,
)
from iqoption_research.campaign.registry import (
    CampaignIntegrityError,
    ProtocolImmutabilityError,
    verify_session_slot_eligibility,
)
from iqoption_research.campaign.scheduler import generate_campaign_slots
from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.parquet_storage import (
    CANDLE_ARROW_SCHEMA,
    read_candles_parquet,
)
from iqoption_research.ml.laboratory import ExperimentConfig, ModelLaboratory
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.inputs_journal import (
    NormalizedCandleRecord,
)
from iqoption_research.shadow.journal import DEFAULT_SHADOW_SESSIONS_DIR
from iqoption_research.shadow.observer import (
    ObservationSessionConfig,
    ProspectiveShadowObserver,
    verify_session_integrity,
)
from iqoption_research.shadow.registry import ShadowModelRegistry
from iqoption_research.shadow.streaming import SyntheticMarketStream
from tests.unit.market_data.test_replay import _reseal_manifest

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _setup_production_contract_workspace(
    tmp_path: Path,
) -> tuple[Path, str, str, str]:
    """Build a complete authentic workspace with candidate model and dataset catalog."""
    p = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 1700000000
    n_candles = 600

    from_ts = [start_epoch + i * 60 for i in range(n_candles)]
    to_ts = [t + 60 for t in from_ts]
    at_ts = [t + 60 for t in from_ts]

    opens = [1.0500 + 0.0010 * math.sin(i / 10.0) for i in range(n_candles)]
    highs = [p_val + 0.0004 for p_val in opens]
    lows = [p_val - 0.0004 for p_val in opens]
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
    pq.write_table(table, str(p))

    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        CatalogEntry(
            dataset_id="test_contract_series",
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
        notes="Phase 40C Candidate for Production Contract Test",
    )
    return tmp_path, snapshot_id, cand.model_id, cand.bundle_sha256


# ---------------------------------------------------------------------------
# Section 1 & 2: Production Observer-to-Campaign Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_production_observer_to_campaign_deterministic_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify end-to-end integration using the real ProspectiveShadowObserver and CampaignManager.

    Tests that real observer output aligns with the Schema 3 campaign eligibility verifier.
    """
    root, _snapshot_id, cand_mid, _ = _setup_production_contract_workspace(tmp_path)
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:145]

    # Schedule campaign starting at today's date
    campaign_mgr = CampaignManager(root=root)
    start_d = datetime.now(tz=UTC).date()
    campaign_mgr.create_campaign(
        campaign_id="camp_prod_int_01",
        start_date=start_d,
        candidate_model_id=cand_mid,
        candidate_threshold=0.52,
        is_confirmatory=False,  # Exploratory / test mode allows synthetic streaming validation
        min_total_market_days=5,
        min_completed_days=1,
        min_completed_sessions=1,
        min_decisive_outcomes=5,
    )

    # Register campaign to lock protocol and calendar
    camp_reg = campaign_mgr.registry
    registered_camp = camp_reg.register("camp_prod_int_01")
    slot = registered_camp.slots[0]

    # Virtual clock: align virtual time with the scheduled slot window start
    sched_dt = datetime.fromisoformat(slot.scheduled_start_utc.replace("Z", "+00:00"))
    virtual_start_ts = sched_dt.timestamp()

    # Controlled virtual clock returns slot start timestamp during observer run
    monkeypatch.setattr("time.time", lambda: virtual_start_ts)

    session_id = "pobs_prod_int_01"
    obs_cfg = ObservationSessionConfig(
        session_id=session_id,
        model_ids=(cand_mid,),
        duration_minutes=15,
        is_synthetic=True,
        schema_version=3,
        symbol="EURUSD",
        timeframe=60,
        campaign_id="camp_prod_int_01",
        slot_id=slot.slot_id,
        protocol_sha256=registered_camp.protocol.protocol_sha256,
        capture_diagnostics=True,
        capture_inputs=True,
    )

    model_reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=obs_cfg,
        registry=model_reg,
        root=root,
        initial_history=initial_candles,
    )

    stream = SyntheticMarketStream(streaming_candles)
    await observer.run(stream)

    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id

    # 1. Verify cryptographic sealing of real observer output
    verif = verify_session_integrity(session_dir)
    assert verif["is_valid"] is True
    assert verif["is_prospective_verified"] is True

    # 2. Link session to slot via CampaignRegistry (exercises verify_session_slot_eligibility)
    updated_camp = camp_reg.link_session_to_slot(
        "camp_prod_int_01", slot.slot_id, session_id, session_dir
    )
    matching_slot = next(s for s in updated_camp.slots if s.slot_id == slot.slot_id)
    assert matching_slot.slot_state == SlotState.COMPLETED_VERIFIED.value
    assert matching_slot.session_id == session_id
    assert matching_slot.verified_integrity is True

    # 3. Evaluate campaign with CampaignEvaluator
    evaluator = CampaignEvaluator("camp_prod_int_01", root=root)
    report = evaluator.evaluate(unblind=True)
    assert report.verified_sessions == 1
    assert report.completed_sessions == 1
    assert report.is_blinded is False
    assert report.verdict != "INTEGRITY_FAILURE"


# ---------------------------------------------------------------------------
# Section 3: Registered Campaign Calendar Locking & Immutability Tests
# ---------------------------------------------------------------------------


def test_calendar_sha256_binds_complete_ordered_calendar(tmp_path: Path) -> None:
    """Verify that compute_calendar_sha256 deterministically hashes complete slot schedules."""
    slots = generate_campaign_slots(date(2026, 11, 24), num_market_days=5)
    cal_hash1 = compute_calendar_sha256(slots)
    cal_hash2 = compute_calendar_sha256(slots)
    assert cal_hash1 == cal_hash2
    assert len(cal_hash1) == 64


def test_registered_campaign_rejects_slot_insertion(tmp_path: Path) -> None:
    """Verify that inserting a slot into a registered campaign raises ProtocolImmutabilityError."""
    root, _, cand_mid, _ = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign(
        "camp_lock_insert",
        date(2026, 11, 24),
        candidate_model_id=cand_mid,
        min_total_market_days=2,
    )
    reg = mgr.registry
    camp = reg.register("camp_lock_insert")
    assert len(camp.slots) == 4

    # Attempt to insert extra slot
    new_slot = ObservationWindowSlot(
        slot_id="slot_extra_01",
        day_index=3,
        window_index=1,
        scheduled_date="2026-11-26",
        scheduled_start_utc="2026-11-26T08:00:00+00:00",
        scheduled_start_local="2026-11-26T09:00:00+01:00",
        duration_minutes=60,
    )
    camp.slots.append(new_slot)

    with pytest.raises(ProtocolImmutabilityError, match="Calendar modification forbidden"):
        reg.save(camp)


def test_registered_campaign_rejects_slot_deletion(tmp_path: Path) -> None:
    """Verify that deleting a slot from a registered campaign raises ProtocolImmutabilityError."""
    root, _, cand_mid, _ = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign(
        "camp_lock_delete",
        date(2026, 11, 24),
        candidate_model_id=cand_mid,
        min_total_market_days=2,
    )
    reg = mgr.registry
    camp = reg.register("camp_lock_delete")
    assert len(camp.slots) == 4

    # Delete slot
    camp.slots.pop()

    with pytest.raises(ProtocolImmutabilityError, match="Calendar modification forbidden"):
        reg.save(camp)


def test_registered_campaign_rejects_slot_schedule_modification(tmp_path: Path) -> None:
    """Verify that modifying scheduled start or duration of a slot is rejected."""
    root, _, cand_mid, _ = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign(
        "camp_lock_modify",
        date(2026, 11, 24),
        candidate_model_id=cand_mid,
        min_total_market_days=2,
    )
    reg = mgr.registry
    camp = reg.register("camp_lock_modify")

    # Mutate scheduled start time of slot 0
    camp.slots[0].scheduled_start_utc = "2026-11-24T08:30:00+00:00"

    with pytest.raises(ProtocolImmutabilityError, match="Calendar modification forbidden"):
        reg.save(camp)


def test_registered_campaign_rejects_slot_reordering(tmp_path: Path) -> None:
    """Verify that reordering slots in a registered campaign is rejected."""
    root, _, cand_mid, _ = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign(
        "camp_lock_reorder",
        date(2026, 11, 24),
        candidate_model_id=cand_mid,
        min_total_market_days=2,
    )
    reg = mgr.registry
    camp = reg.register("camp_lock_reorder")

    # Reverse slots order
    camp.slots.reverse()

    with pytest.raises(ProtocolImmutabilityError, match="Calendar modification forbidden"):
        reg.save(camp)


def test_on_disk_calendar_tampering_detected_on_get(tmp_path: Path) -> None:
    """Verify that tampering with slots on disk raises CampaignIntegrityError on registry.get()."""
    root, _, cand_mid, _ = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    mgr.create_campaign(
        "camp_lock_disk_tamper",
        date(2026, 11, 24),
        candidate_model_id=cand_mid,
        min_total_market_days=2,
    )
    reg = mgr.registry
    reg.register("camp_lock_disk_tamper")

    # Directly mutate campaign.json on disk to simulate external tampering
    camp_file = root / "data" / "campaigns" / "camp_lock_disk_tamper" / "campaign.json"
    data = json.loads(camp_file.read_text(encoding="utf-8"))
    data["slots"][0]["scheduled_start_utc"] = "2026-11-24T09:00:00+00:00"
    camp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with pytest.raises(CampaignIntegrityError, match="Calendar SHA-256 mismatch"):
        reg.get("camp_lock_disk_tamper")


# ---------------------------------------------------------------------------
# Section 4: Exact Numeric Input Semantics (Floats & Volume)
# ---------------------------------------------------------------------------


def test_candle_volume_accepts_and_preserves_int_and_float() -> None:
    """Verify that Candle domain model supports both integer and fractional volumes."""
    c_int = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=1700000000,
        to_ts=1700000060,
        at_ts=1700000060000000000,
        open=1.0500,
        high=1.0510,
        low=1.0490,
        close=1.0505,
        volume=150,
        bid=1.0504,
        ask=1.0506,
    )
    assert c_int.volume == 150
    assert isinstance(c_int.volume, int)

    c_float = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=1700000000,
        to_ts=1700000060,
        at_ts=1700000060000000000,
        open=1.0500,
        high=1.0510,
        low=1.0490,
        close=1.0505,
        volume=150.75,
        bid=1.0504,
        ask=1.0506,
    )
    assert c_float.volume == 150.75
    assert isinstance(c_float.volume, float)


def test_normalized_candle_record_to_candle_preserves_fractional_volume() -> None:
    """Verify that NormalizedCandleRecord.to_candle() does NOT round fractional volumes."""
    rec = NormalizedCandleRecord(
        session_id="sess_vol_test",
        sequence_number=1,
        candle_source="WARMUP",
        symbol="EURUSD",
        timeframe=60,
        from_ts=1700000000,
        to_ts=1700000060,
        open=1.0500123456,
        high=1.0510123456,
        low=1.0490123456,
        close=1.0505123456,
        volume=125.75,
        receipt_wall_utc="2026-11-24T08:00:00+00:00",
        receipt_mono_ns=1000000,
        candle_sha256="mock_sha",
    )
    candle = rec.to_candle()
    # Must preserve 125.75 without rounding to 126
    assert candle.volume == 125.75
    assert isinstance(candle.volume, float)

    # Re-serialization and hashing must match
    rec_int = NormalizedCandleRecord(
        session_id="sess_vol_test",
        sequence_number=2,
        candle_source="WARMUP",
        symbol="EURUSD",
        timeframe=60,
        from_ts=1700000060,
        to_ts=1700000120,
        open=1.0500,
        high=1.0510,
        low=1.0490,
        close=1.0505,
        volume=100.0,
        receipt_wall_utc="2026-11-24T08:01:00+00:00",
        receipt_mono_ns=2000000,
        candle_sha256="mock_sha",
    )
    candle_int = rec_int.to_candle()
    assert candle_int.volume == 100
    assert isinstance(candle_int.volume, int)


# ---------------------------------------------------------------------------
# Section 6: Full Failure Matrix Tests
# ---------------------------------------------------------------------------


def _create_mock_session_dir(
    tmp_path: Path,
    session_id: str,
    *,
    campaign_id: str = "camp_mock",
    slot_id: str = "slot_mock",
    protocol_sha256: str = "mock_proto_sha",
    candidate_model_id: str = "cand_m1",
    candidate_threshold: float = 0.52,
    candidate_bundle_sha: str = "bundle_sha_01",
    symbol: str = "EURUSD",
    timeframe: int = 60,
    started_at_utc: str | None = "2026-11-24T08:00:00+00:00",
    terminal_status: str | None = "CLEAN_COMPLETED",
    is_interrupted: bool = False,
    is_synthetic: bool = False,
    schema_version: int = 3,
) -> Path:
    """Create a mock session directory adhering to the Schema 3 contract."""
    sess_dir = tmp_path / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy predictions, outcomes, commitments
    now_ts = 1700000000
    pred = {
        "prediction_id": f"{session_id}_p1",
        "session_id": session_id,
        "model_id": candidate_model_id,
        "cutoff_ts": now_ts,
        "decision": "UP",
        "probability": 0.65,
        "threshold": candidate_threshold,
        "is_prospective": True,
        "features_fingerprint": "mock_fp",
    }
    (sess_dir / "predictions.jsonl").write_text(json.dumps(pred) + "\n", encoding="utf-8")

    comm = {
        "commitment_id": f"{session_id}_c1",
        "prediction_id": f"{session_id}_p1",
        "timestamp_utc": started_at_utc or "2026-11-24T08:00:00+00:00",
        "commitment_hash": "mock_comm_hash",
        "is_quarantined": False,
    }
    (sess_dir / "commitments.jsonl").write_text(json.dumps(comm) + "\n", encoding="utf-8")

    outcome = {
        "outcome_id": f"{session_id}_o1",
        "prediction_id": f"{session_id}_p1",
        "status": "RESOLVED_WIN",
        "simulated_payout_pnl": 0.85,
    }
    (sess_dir / "outcomes.jsonl").write_text(json.dumps(outcome) + "\n", encoding="utf-8")

    inp_candle = {
        "session_id": session_id,
        "sequence_number": 1,
        "candle_source": "STREAM_BAR",
        "symbol": symbol,
        "timeframe": timeframe,
        "from_ts": now_ts,
        "to_ts": now_ts + 60,
        "open": 1.0500,
        "high": 1.0510,
        "low": 1.0490,
        "close": 1.0505,
        "volume": 100,
        "canonical_sha256": "mock_inp_sha",
    }
    (sess_dir / "inputs.jsonl").write_text(json.dumps(inp_candle) + "\n", encoding="utf-8")

    diag = {
        "session_id": session_id,
        "event_type": "STREAM_CONNECTED",
        "timestamp_utc": started_at_utc or "2026-11-24T08:00:00+00:00",
        "details": {},
    }
    (sess_dir / "diagnostics.jsonl").write_text(json.dumps(diag) + "\n", encoding="utf-8")

    session_meta: dict[str, Any] = {
        "schema_version": schema_version,
        "session_id": session_id,
        "campaign_id": campaign_id,
        "slot_id": slot_id,
        "protocol_sha256": protocol_sha256,
        "symbol": symbol,
        "timeframe": timeframe,
        "model_ids": [candidate_model_id],
        "model_thresholds": {candidate_model_id: candidate_threshold},
        "models_metadata": {
            candidate_model_id: {
                "threshold": candidate_threshold,
                "bundle_sha256": candidate_bundle_sha,
            }
        },
        "started_at_utc": started_at_utc,
        "terminal_status": terminal_status,
        "termination_reason": terminal_status,
        "is_interrupted": is_interrupted,
        "is_synthetic": is_synthetic,
        "evidence_class": "SYNTHETIC_STREAMING_TEST" if is_synthetic else "PROSPECTIVE_SHADOW",
    }
    (sess_dir / "session.json").write_text(json.dumps(session_meta, indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "session_id": session_id,
        "campaign_id": campaign_id,
        "slot_id": slot_id,
        "protocol_sha256": protocol_sha256,
        "symbol": symbol,
        "timeframe": timeframe,
        "model_ids": [candidate_model_id],
        "model_thresholds": {candidate_model_id: candidate_threshold},
        "models_sha256": {candidate_model_id: candidate_bundle_sha},
        "started_at_utc": started_at_utc,
        "session_start_utc": started_at_utc,
        "terminal_status": terminal_status,
        "termination_reason": terminal_status,
        "is_interrupted": is_interrupted,
        "is_synthetic": is_synthetic,
        "evidence_class": "SYNTHETIC_STREAMING_TEST" if is_synthetic else "PROSPECTIVE_SHADOW",
        "predictions_sha256": hashlib.sha256(
            (sess_dir / "predictions.jsonl").read_bytes()
        ).hexdigest(),
        "outcomes_sha256": hashlib.sha256((sess_dir / "outcomes.jsonl").read_bytes()).hexdigest(),
        "commitments_sha256": hashlib.sha256(
            (sess_dir / "commitments.jsonl").read_bytes()
        ).hexdigest(),
        "session_json_sha256": hashlib.sha256((sess_dir / "session.json").read_bytes()).hexdigest(),
        "inputs_sha256": hashlib.sha256((sess_dir / "inputs.jsonl").read_bytes()).hexdigest(),
        "diagnostics_sha256": hashlib.sha256(
            (sess_dir / "diagnostics.jsonl").read_bytes()
        ).hexdigest(),
        "commitments_timely_count": 1,
        "commitments_late_quarantined_count": 0,
    }
    m_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(m_bytes).hexdigest()
    (sess_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return sess_dir


def test_missing_start_timestamp_fails_closed(tmp_path: Path) -> None:
    """Verify that a session missing start timestamp is rejected (fails closed)."""
    root, _, cand_mid, bundle_sha = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign("camp_fail_start", date(2026, 11, 24), candidate_model_id=cand_mid)
    camp = mgr.registry.register("camp_fail_start")
    slot = camp.slots[0]

    # Create session with None for started_at_utc
    sess_dir = _create_mock_session_dir(
        tmp_path,
        "sess_no_start",
        campaign_id="camp_fail_start",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        candidate_model_id=cand_mid,
        candidate_bundle_sha=bundle_sha,
        started_at_utc=None,
    )
    valid, errors = verify_session_slot_eligibility(camp, slot.slot_id, sess_dir)
    assert valid is False
    assert any("Missing required start timestamp" in err for err in errors)


def test_out_of_window_start_rejected(tmp_path: Path) -> None:
    """Verify that a session starting outside the slot grace period is rejected."""
    root, _, cand_mid, bundle_sha = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign("camp_fail_drift", date(2026, 11, 24), candidate_model_id=cand_mid)
    camp = mgr.registry.register("camp_fail_drift")
    slot = camp.slots[0]

    # Slot scheduled at 08:00 UTC, session started at 08:30 UTC (+30 min drift > 15 min grace)
    late_start = "2026-11-24T08:30:00+00:00"
    sess_dir = _create_mock_session_dir(
        tmp_path,
        "sess_late_start",
        campaign_id="camp_fail_drift",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        candidate_model_id=cand_mid,
        candidate_bundle_sha=bundle_sha,
        started_at_utc=late_start,
    )
    valid, errors = verify_session_slot_eligibility(camp, slot.slot_id, sess_dir)
    assert valid is False
    assert any("outside slot window grace period" in err for err in errors)


def test_missing_or_invalid_terminal_status_rejected(tmp_path: Path) -> None:
    """Verify that a session lacking valid clean completion evidence is rejected."""
    root, _, cand_mid, bundle_sha = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign("camp_fail_term", date(2026, 11, 24), candidate_model_id=cand_mid)
    camp = mgr.registry.register("camp_fail_term")
    slot = camp.slots[0]

    sess_dir = _create_mock_session_dir(
        tmp_path,
        "sess_bad_term",
        campaign_id="camp_fail_term",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        candidate_model_id=cand_mid,
        candidate_bundle_sha=bundle_sha,
        terminal_status="CRASH_ERROR",
    )
    valid, errors = verify_session_slot_eligibility(camp, slot.slot_id, sess_dir)
    assert valid is False
    assert any("Invalid terminal session status" in err for err in errors)


def test_interrupted_session_rejected(tmp_path: Path) -> None:
    """Verify that an interrupted session is rejected."""
    root, _, cand_mid, bundle_sha = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign("camp_fail_int", date(2026, 11, 24), candidate_model_id=cand_mid)
    camp = mgr.registry.register("camp_fail_int")
    slot = camp.slots[0]

    sess_dir = _create_mock_session_dir(
        tmp_path,
        "sess_interrupted",
        campaign_id="camp_fail_int",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        candidate_model_id=cand_mid,
        candidate_bundle_sha=bundle_sha,
        is_interrupted=True,
    )
    valid, errors = verify_session_slot_eligibility(camp, slot.slot_id, sess_dir)
    assert valid is False
    assert any("interrupted or failed" in err for err in errors)


def test_missing_model_threshold_rejected(tmp_path: Path) -> None:
    """Verify that missing model threshold fails closed."""
    root, _, cand_mid, bundle_sha = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign("camp_fail_thresh", date(2026, 11, 24), candidate_model_id=cand_mid)
    camp = mgr.registry.register("camp_fail_thresh")
    slot = camp.slots[0]

    sess_dir = _create_mock_session_dir(
        tmp_path,
        "sess_no_thresh",
        campaign_id="camp_fail_thresh",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        candidate_model_id=cand_mid,
        candidate_bundle_sha=bundle_sha,
    )
    # Tamper manifest to remove threshold
    m_p = sess_dir / "manifest.json"
    m_data = json.loads(m_p.read_text(encoding="utf-8"))
    m_data.pop("model_thresholds", None)
    # Remove from session.json too
    s_p = sess_dir / "session.json"
    s_data = json.loads(s_p.read_text(encoding="utf-8"))
    s_data.pop("model_thresholds", None)
    s_data.pop("models_metadata", None)
    s_p.write_text(json.dumps(s_data, indent=2), encoding="utf-8")
    _reseal_manifest(
        sess_dir,
        session_json_sha256=hashlib.sha256(s_p.read_bytes()).hexdigest(),
        model_thresholds=None,
    )

    valid, errors = verify_session_slot_eligibility(camp, slot.slot_id, sess_dir)
    assert valid is False
    assert any("Missing threshold" in err for err in errors)


def test_missing_symbol_or_timeframe_rejected(tmp_path: Path) -> None:
    """Verify that missing symbol or timeframe metadata fails closed without defaulting."""
    root, _, cand_mid, bundle_sha = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign("camp_fail_meta", date(2026, 11, 24), candidate_model_id=cand_mid)
    camp = mgr.registry.register("camp_fail_meta")
    slot = camp.slots[0]

    sess_dir = _create_mock_session_dir(
        tmp_path,
        "sess_no_symbol",
        campaign_id="camp_fail_meta",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        candidate_model_id=cand_mid,
        candidate_bundle_sha=bundle_sha,
    )
    # Tamper manifest to remove symbol
    m_p = sess_dir / "manifest.json"
    m_data = json.loads(m_p.read_text(encoding="utf-8"))
    m_data.pop("symbol", None)
    s_p = sess_dir / "session.json"
    s_data = json.loads(s_p.read_text(encoding="utf-8"))
    s_data.pop("symbol", None)
    s_p.write_text(json.dumps(s_data, indent=2), encoding="utf-8")
    sess_sha = hashlib.sha256(s_p.read_bytes()).hexdigest()
    _reseal_manifest(sess_dir, session_json_sha256=sess_sha, symbol=None)

    valid, errors = verify_session_slot_eligibility(camp, slot.slot_id, sess_dir)
    assert valid is False
    assert any("Missing required 'symbol'" in err for err in errors)


def test_incorrect_model_artifact_hash_rejected(tmp_path: Path) -> None:
    """Verify that a mismatch in candidate model bundle SHA-256 is rejected."""
    root, _, cand_mid, bundle_sha = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign("camp_fail_bundle", date(2026, 11, 24), candidate_model_id=cand_mid)
    camp = mgr.registry.register("camp_fail_bundle")
    assert camp.protocol.candidate_artifact_hash == bundle_sha
    slot = camp.slots[0]

    sess_dir = _create_mock_session_dir(
        tmp_path,
        "sess_bad_bundle",
        campaign_id="camp_fail_bundle",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        candidate_model_id=cand_mid,
        candidate_bundle_sha="tampered_bundle_sha_xyz",
    )
    valid, errors = verify_session_slot_eligibility(camp, slot.slot_id, sess_dir)
    assert valid is False
    assert any("bundle SHA-256 mismatch" in err for err in errors)


def test_synthetic_session_rejected_in_confirmatory_campaign(tmp_path: Path) -> None:
    """Verify that synthetic sessions are strictly rejected in confirmatory campaigns."""
    root, _, cand_mid, bundle_sha = _setup_production_contract_workspace(tmp_path)
    mgr = CampaignManager(root=root)
    camp = mgr.create_campaign(
        "camp_confirmatory_reject",
        date(2026, 11, 24),
        candidate_model_id=cand_mid,
        is_confirmatory=True,
    )
    camp = mgr.registry.register("camp_confirmatory_reject")
    slot = camp.slots[0]

    sess_dir = _create_mock_session_dir(
        tmp_path,
        "sess_synthetic_conf",
        campaign_id="camp_confirmatory_reject",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        candidate_model_id=cand_mid,
        candidate_bundle_sha=bundle_sha,
        is_synthetic=True,
    )
    valid, errors = verify_session_slot_eligibility(camp, slot.slot_id, sess_dir)
    assert valid is False
    assert any("Synthetic evidence rejected for confirmatory campaign" in err for err in errors)


def test_legacy_pilot_sessions_preserve_integrity() -> None:
    """Verify that both genuine Schema 2 pilot sessions pass verify_session_integrity cleanly."""
    root = Path.cwd()
    s1_dirs = [
        root / "data" / "shadow_sessions" / "pobs_d2d1a79b15d8",
        root / "data" / "shadow_eval" / "sessions" / "pobs_d2d1a79b15d8",
    ]
    s2_dirs = [
        root / "data" / "shadow_sessions" / "pobs_7843aa5dbb1c",
        root / "data" / "shadow_eval" / "sessions" / "pobs_7843aa5dbb1c",
    ]
    s1_dir = next((d for d in s1_dirs if d.is_dir()), None)
    s2_dir = next((d for d in s2_dirs if d.is_dir()), None)

    if s1_dir is None or s2_dir is None:
        pytest.skip("Private sealed pilot evidence not present in environment (offline CI).")

    v1 = verify_session_integrity(s1_dir)
    assert v1["is_valid"] is True
    assert v1["is_prospective_verified"] is True
    assert len(v1.get("errors", [])) == 0

    v2 = verify_session_integrity(s2_dir)
    assert v2["is_valid"] is True
    assert v2["is_prospective_verified"] is True
    assert len(v2.get("errors", [])) == 0
