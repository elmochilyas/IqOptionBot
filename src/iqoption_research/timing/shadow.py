"""Foreground EURUSD spot shadow observations. All decisions have zero execution authority."""

from __future__ import annotations

import asyncio
import itertools
import math
import sqlite3
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import IQOptionError
from iqoption_research.inference.loader import SafeModelArtifactLoader
from iqoption_research.inference.pipeline import ModelInferencePipeline
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.ml.calibration.abstention import BinaryPayoffGate
from iqoption_research.ml.tracking.artifacts import ArtifactError, verify_manifest
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.market_risk import market_risk_rejection
from iqoption_research.research.prospective import (
    ProspectiveJournal,
    ProspectivePolicy,
    digest,
    initialize_event_journal,
)
from iqoption_research.timing.evidence import TimingEvidenceRegistry, compute_file_sha256
from iqoption_research.timing.instrumentation import CandleTimingObservation, TimingRecorder
from iqoption_research.timing.supervised_session import SupervisedSessionConfig, supervised_client

MODEL_PATH = Path("data/phase13/candidate_models/candidate_h1_rf")
MODEL_DIGEST = "8dc27f329148af54a5c44234268c4b4e31233b6b3b4e7ed79bd13e5cd98de872"
POLICY = ProspectivePolicy()


def eligible_model(path: Path) -> ModelInferencePipeline | None:
    """Pin the reviewed manifest before opening any serialized model weights."""
    try:
        if verify_manifest(path).digest() != MODEL_DIGEST:
            return None
        bundle, report = SafeModelArtifactLoader().load(path)
        lineage = bundle.metadata["metadata"]
        if (
            report.warnings
            or report.provenance != "authentic_trained"
            or lineage["source_dataset"] != "authentic_iqoption_eurusd_60s"
            or lineage["horizon_bars"] != 1
            or bundle.calibrator is None
            or tuple(bundle.metadata["feature_names"]) != tuple(bundle.feature_names)
            or datetime.fromisoformat(bundle.metadata["created_at_utc"]) > datetime.now(UTC)
        ):
            return None
        return ModelInferencePipeline(bundle)
    except (ArtifactError, OSError, ValueError, KeyError, TypeError):
        return None


class ShadowTrial:
    def __init__(self, config: SupervisedSessionConfig, root: Path, model_path: Path) -> None:
        config.validate()
        if config.symbol != "EURUSD" or config.candle_size_seconds != 60:
            raise ValueError("Shadow trial requires EURUSD 60-second candles")
        self.config = config
        self.mode = "OPERATOR_SUPERVISED_LIVE" if config.live_mode else "OFFLINE_SIMULATION"
        self.session = f"shadow_{uuid.uuid4().hex}"
        self.root = root / self.mode.lower()
        self.output = self.root / f"{self.session}.json"
        self.recorder = TimingRecorder(
            self.session, provenance=self.mode, is_synthetic=not config.live_mode
        )
        self.model = eligible_model(model_path)
        self.gate = BinaryPayoffGate(
            POLICY.assumed_payout, margin=POLICY.threshold - 1 / (1 + POLICY.assumed_payout)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        initialize_event_journal(
            self.root,
            {
                "kind": "CONFIG",
                "protocol": "PHASE34_V1",
                "mode": self.mode,
                "instrument": "EURUSD_SPOT_60S",
            },
        )
        self.journal = ProspectiveJournal(self.root)
        self.metadata: dict[str, Any] = {
            "source": "IQOPTION_EURUSD_SPOT_60S" if config.live_mode else "SYNTHETIC_FIXTURE",
            "model_eligible": self.model is not None,
            "model_manifest_sha256": MODEL_DIGEST if self.model else None,
            "feature_schema": list(self.model.feature_names) if self.model else [],
            "model_name": self.model.model_name if self.model else None,
            "model_provenance": "authentic_trained" if self.model else None,
            "feature_pipeline": "CausalFeaturePipeline/v1",
            "policy": "ProspectivePolicy/0.60/assumed-payout-0.85/v1",
            "trading_eligibility": "UNPROVEN",
            "execution_authority": "NONE",
            "orders_sent": 0,
            "max_candle_cycles": config.max_candle_cycles,
            "max_duration_seconds": config.max_duration_seconds,
            "stop_reason": "SESSION_INCOMPLETE",
            "allowlist": "validate_timing_frame",
            "disposition": "NO_DEMONSTRATED_EDGE",
        }

    def decide(
        self, result: CandleFetchResult, obs: CandleTimingObservation, deadline: float
    ) -> None:
        candles = result.candles
        if (
            not candles
            or candles[-1].to_ts != obs.feature_close_F
            or result.source != "iqoption"
            or result.errors
            or result.raw_count != len(candles)
            or not result.quality.is_research_grade
            or any(
                not c.is_complete or c.symbol != "EURUSD" or c.to_ts - c.from_ts != 60
                for c in candles
            )
            or any(a.to_ts != b.from_ts for a, b in itertools.pairwise(candles))
        ):
            obs.rejection_reason = "INVALID_OR_INCOMPLETE_WINDOW"
            return
        self.recorder.record_completed_recognized(time.perf_counter_ns(), candles[-1].to_ts)
        rows = CausalFeaturePipeline(drop_warmup=True).transform(candles).rows
        action = "SKIP"
        reason: str | None = "MODEL_UNAVAILABLE" if self.model is None else "MISSING_FEATURES"
        if rows and rows[-1].available_at_ts == obs.feature_close_F:
            row = rows[-1]
            self.recorder.record_features_computed(time.perf_counter_ns(), len(row.features))
            if self.model and all(
                math.isfinite(row.features.get(n, math.nan)) for n in self.model.feature_names
            ):
                prediction = self.model.infer(row)
                raw, cal = prediction.raw_probability, prediction.calibrated_prob_up
                if (prediction.abstention_reason or "").startswith((
                    "SCHEMA_MISMATCH",
                    "INFERENCE_EXCEPTION",
                )) or not all(math.isfinite(p) and 0 <= p <= 1 for p in (raw, cal)):
                    reason = "INVALID_INFERENCE"
                else:
                    self.recorder.record_inference_completed(
                        time.perf_counter_ns(), raw, cal, "NOT_MEASURED"
                    )
                    gate = self.gate.evaluate(cal)
                    action = gate.recommended_direction.value.upper() if gate.passed else "SKIP"
                    reason = None if gate.passed else "BELOW_FROZEN_THRESHOLD"

        offset = self.recorder.clock.mono_to_utc_ns(time.perf_counter_ns()) - obs.feature_close_F_ns
        if (
            obs.clock_step_detected
            or self.recorder.clock.total_uncertainty_ms() > 5000
            or not 0 <= offset <= 15_000_000_000
        ):
            reason, action = "TIMING_RISK", "SKIP"
        calibrated = obs.calibrated_probability
        confidence = max(calibrated, 1 - calibrated) if calibrated is not None else None
        # The spot feed supplies no tradable contract payout.
        risk = market_risk_rejection(None, confidence, minimum_payout=0.70, minimum_confidence=0.55)
        obs.decision = action
        decision = self.recorder.record_stage(
            "STAGE_POLICY_DECISION",
            details={
                "decision": action,
                "skip_reason": reason,
                "side_probability": confidence,
                "eligibility_block": risk[0] if risk else None,
            },
        )
        if time.perf_counter() >= deadline:
            obs.rejection_reason = "SESSION_DEADLINE"
            return
        key = f"shadow:EURUSD:60:{obs.feature_close_F}"
        payload = {
            "kind": "SHADOW_DECISION",
            "session": self.session,
            "boundary": obs.feature_close_F,
            "decision": action,
            "skip_reason": reason,
            "raw_probability": obs.raw_probability,
            "calibrated_probability": obs.calibrated_probability,
            "decision_time": decision.to_dict(),
            "window_sha256": digest([
                (c.from_ts, c.to_ts, c.open, c.high, c.low, c.close, c.volume) for c in candles
            ]),
            "precommit_stages": {name: stage.to_dict() for name, stage in obs.stages.items()},
            "clock_uncertainty_ms": self.recorder.clock.total_uncertainty_ms(),
            **self.metadata,
        }
        payload_hash = digest(payload)
        _, inserted = self.journal.append(key, payload)
        if not inserted:
            obs.rejection_reason = "DUPLICATE_DECISION"
            return
        self.recorder.record_stage(
            "STAGE_9_JOURNAL_COMMIT", details={"event_key": key, "payload_sha256": payload_hash}
        )
        if time.perf_counter() > deadline:
            obs.rejection_reason = "SESSION_DEADLINE"

    async def cycle(self, client: AsyncIQOption, boundary: int, deadline: float) -> bool:
        obs = self.recorder.start_candle_observation(boundary)
        obs.measurement_status = "SHADOW_DECISION"
        try:
            async with asyncio.timeout(max(0, deadline - time.perf_counter())):
                result = await client.get_candles("EURUSD", 60, 22, endtime=boundary)
            self.decide(result, obs, deadline)
        except (IQOptionError, TimeoutError, sqlite3.Error, ValueError, OSError) as exc:
            obs.rejection_reason = type(exc).__name__
            return False
        except asyncio.CancelledError:
            obs.rejection_reason = "OPERATOR_CANCELLED"
            raise
        finally:
            self.recorder.finalize_current_observation(obs.rejection_reason)
        return not obs.clock_step_detected

    async def run_live(self) -> TimingEvidenceRegistry:
        if not self.config.live_mode or self.config.simulation_mode:
            raise ValueError("Live mode and operator acknowledgement required")
        try:
            async with await supervised_client(self.recorder) as client:
                await self.observe(client)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.metadata["stop_reason"] = "OPERATOR_CANCELLED"
            raise
        finally:
            registry = self.save()
        return registry

    async def observe(self, client: AsyncIQOption) -> None:
        client.assert_read_only()
        self.metadata["wallet_classification_at_start"] = client.measurement_wallet_status().name
        self.metadata["observation_started_utc_ns"] = time.time_ns()
        deadline = time.perf_counter() + self.config.max_duration_seconds
        previous = 0
        while (
            len(self.recorder.observations) < self.config.max_candle_cycles
            and time.perf_counter() < deadline
        ):
            now = time.time()
            boundary = int(now) // 60 * 60
            if boundary <= previous or now - boundary > 15:
                await asyncio.sleep(
                    min(60 - (now - boundary), max(0, deadline - time.perf_counter()))
                )
                continue
            previous = boundary
            if not await self.cycle(client, boundary, deadline):
                self.metadata["stop_reason"] = "CYCLE_FAILED"
                break
        else:
            self.metadata["stop_reason"] = (
                "CYCLE_LIMIT"
                if len(self.recorder.observations) >= self.config.max_candle_cycles
                else "DURATION_LIMIT"
            )

    def save(self) -> TimingEvidenceRegistry:
        registry = TimingEvidenceRegistry.create(
            self.session,
            self.mode,
            self.config.project_root,
            self.recorder.clock,
            self.recorder.observations,
        )
        registry.instrumentation_version = "PHASE_34_SHADOW_V1"
        registry.code_fingerprints["shadow.py"] = compute_file_sha256(Path(__file__))
        registry.summary.update(self.metadata)
        registry.integrity_hash = registry.compute_hash()
        registry.save_to_file(self.output)
        return registry


def verify_shadow_receipts(registry: TimingEvidenceRegistry, directory: Path) -> None:
    """Read the hash chain and bind every claimed commit to its original session and payload."""
    events = {event["key"]: event for event in ProspectiveJournal(directory).events()}
    for obs in registry.observations:
        commit = obs["stages"].get("STAGE_9_JOURNAL_COMMIT")
        if commit is None:
            continue
        receipt = commit["details"]
        event = events.get(receipt["event_key"], {})
        if (
            event.get("session") != registry.session_id
            or event.get("boundary") != obs["feature_close_F"]
            or event.get("decision") != obs["decision"]
            or digest({k: v for k, v in event.items() if k != "key"}) != receipt["payload_sha256"]
        ):
            raise ValueError("Shadow journal commit receipt mismatch")
