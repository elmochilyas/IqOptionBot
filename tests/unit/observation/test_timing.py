"""Unit tests for Phase 31 timing diagnostics, latency decomposition, and feasibility."""

from __future__ import annotations

from pathlib import Path

from iqoption_research.timing.blitz_audit import audit_blitz_market_evidence
from iqoption_research.timing.feasibility import (
    cold_cli_batch_budget,
    evaluate_horizon_scaling,
    run_feasibility_assessment,
    streaming_websocket_budget,
    warm_polling_budget,
)
from iqoption_research.timing.protocol import get_supervised_protocol_spec
from iqoption_research.timing.reconstruction import audit_checkpoint_timing
from iqoption_research.timing.taxonomy import (
    LatencyInterval,
    MeasurementQuality,
    TimestampAuditRecord,
    TimestampRole,
)


def test_timestamp_taxonomy_roles_and_quality() -> None:
    rec = TimestampAuditRecord(
        name="test_record",
        role=TimestampRole.CLIENT_INVOCATION_CLOCK,
        quality=MeasurementQuality.GENUINELY_MEASURED,
        timestamp_epoch=1700000000.0,
        timestamp_iso="2023-11-14T22:13:20+00:00",
        description="Test client startup",
        source_context="unit_test",
    )
    d = rec.to_dict()
    assert d["role"] == "CLIENT_INVOCATION_CLOCK"
    assert d["quality"] == "GENUINELY_MEASURED"
    assert d["timestamp_epoch"] == 1700000000.0

    interval = LatencyInterval(
        name="test_interval",
        start_role=TimestampRole.NOMINAL_BUCKET_BOUNDARY,
        end_role=TimestampRole.CLIENT_INVOCATION_CLOCK,
        duration_seconds=25.5,
        quality=MeasurementQuality.GENUINELY_MEASURED,
        description="Invocation latency",
    )
    idict = interval.to_dict()
    assert idict["duration_seconds"] == 25.5
    assert idict["start_role"] == "NOMINAL_BUCKET_BOUNDARY"
    assert idict["end_role"] == "CLIENT_INVOCATION_CLOCK"


def test_audit_checkpoint_timing_synthetic() -> None:
    cp = {
        "job_id": "synthetic-test",
        "target_completed_end_ts": 1700000000,
        "created_at_utc": "2023-11-14T22:13:45+00:00",  # +25s
        "updated_at_utc": "2023-11-14T22:13:47.500000+00:00",  # +2.5s duration -> +27.5s total
        "requests_made": 1,
        "new_records_saved": 50,
    }
    result = audit_checkpoint_timing(cp, "synthetic_file.json")
    assert result["job_id"] == "synthetic-test"
    assert result["target_completed_end_ts"] == 1700000000
    assert result["invocation_delay_seconds"] == 25.0
    assert result["collection_duration_seconds"] == 2.5
    assert result["total_offset_seconds"] == 27.5
    assert result["single_request_batch"] is True
    assert result["satisfies_phase30_15s_gate"] is False


def test_feasibility_cold_batch_budget() -> None:
    budget = cold_cli_batch_budget(operator_jitter=2.0, broker_delay=1.0)
    assert budget.architecture_name == "COLD_CLI_BATCH_COLLECTION"
    assert budget.read_only_safety_compliant is True
    assert budget.verified_in_codebase is True

    # Total stages alone: 2.4+3.5+2.0+2.07+1.5+0.005+0.010 = 11.485s
    # Plus broker delay (1.0s) = 12.485s best case
    assert 12.0 < budget.total_best_case < 13.0
    # Typical: ~18s (exceeds 15s)
    assert budget.total_typical > 15.0
    # Worst case: ~23s
    assert budget.total_worst_case > 20.0
    assert budget.probability_of_meeting_15s() < 0.35


def test_feasibility_warm_and_streaming() -> None:
    warm = warm_polling_budget(broker_delay=1.0)
    assert warm.architecture_name == "WARM_BACKGROUND_POLLING"
    assert warm.total_best_case < 5.0
    assert warm.probability_of_meeting_15s() == 1.0
    assert warm.read_only_safety_compliant is False  # Requires scope expansion

    streaming = streaming_websocket_budget(broker_delay=1.0)
    assert streaming.architecture_name == "STREAMING_WEBSOCKET_CANDLE_GENERATED"
    assert streaming.total_best_case < 2.0
    assert streaming.probability_of_meeting_15s() == 1.0
    assert streaming.read_only_safety_compliant is False


def test_horizon_scaling_evaluations() -> None:
    horizons = evaluate_horizon_scaling()
    assert len(horizons) == 4

    h60 = next(h for h in horizons if h.timeframe_seconds == 60)
    assert h60.cold_latency_fraction == 0.25  # 15s / 60s
    assert "NO-GO" in h60.feasibility_verdict

    h300 = next(h for h in horizons if h.timeframe_seconds == 300)
    assert h300.cold_latency_fraction == 0.05  # 15s / 300s = 5%
    assert h300.cold_pre_entry_margin_seconds == 285.0
    assert "FEASIBLE" in h300.feasibility_verdict

    h900 = next(h for h in horizons if h.timeframe_seconds == 900)
    assert h900.cold_latency_fraction < 0.02  # 1.67%
    assert h900.cold_pre_entry_margin_seconds == 885.0

    h3600 = next(h for h in horizons if h.timeframe_seconds == 3600)
    assert h3600.cold_latency_fraction < 0.005  # 0.42%


def test_feasibility_comprehensive_assessment() -> None:
    assessment = run_feasibility_assessment()
    assert "architectures" in assessment
    verdict = assessment["phase30_gate_verdict"]
    assert verdict["current_verified_workflow_status"] == "DEFINITIVE_NO_GO"
    assert "do not relax" in verdict["gate_relaxation_policy"].lower()


def test_blitz_audit_dimensions_and_synthesis(tmp_path: Path) -> None:
    audit = audit_blitz_market_evidence(tmp_path)
    assert audit["evidence_summary"]["total_dimensions_audited"] >= 8
    assert audit["evidence_summary"]["verified_facts_count"] >= 4
    assert audit["evidence_summary"]["unverified_unknowns_count"] >= 4
    assert audit["synthesis"]["research_conclusion"] == "NO_DEMONSTRATED_EDGE"

    # Spot active_id is 1, Blitz active_id is 1861
    items = {item["dimension"]: item for item in audit["items"]}
    assert "1" in items["spot_market_data_active_id"]["details"]
    assert "1861" in items["blitz_option_instrument_identity"]["details"]
    assert items["strike_price_determination"]["empirical_status"] == "UNVERIFIED_UNKNOWN"
    assert items["dynamic_payout_rate"]["empirical_status"] == "UNVERIFIED_UNKNOWN"


def test_supervised_protocol_specification() -> None:
    spec = get_supervised_protocol_spec()
    assert spec["protocol_name"] == "OPERATOR_SUPERVISED_ACQUISITION_TIMING_PROTOCOL_V1"
    assert spec["status"] == "SPECIFICATION_ONLY_UNEXECUTED"
    assert spec["execution_rules"]["max_duration_seconds"] == 300
    assert spec["execution_rules"]["max_candle_cycles"] == 5
    assert "--live" in spec["execution_rules"]["cli_opt_in_flags"]
    assert "--yes-i-understand" in spec["execution_rules"]["cli_opt_in_flags"]
    assert "FAIL_CLOSED" in spec["execution_rules"]["reconnection_policy"]
    order_policy = spec["execution_rules"]["broker_order_policy"]
    assert "FORBIDDEN" in order_policy
