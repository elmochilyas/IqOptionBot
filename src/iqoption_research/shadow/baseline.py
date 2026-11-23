"""Pre-observation majority-label benchmarks; never changes a trading policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from iqoption_research.shadow.registry import FrozenRegistrationRecord, ShadowModelRegistry


@dataclass(frozen=True, slots=True)
class FrozenMajorityBenchmark:
    """Immutable in-session linkage to a verified training-frequency classifier."""

    status: str = "UNAVAILABLE"
    reason: str = "NO_UNAMBIGUOUS_LINKAGE"
    source_id: str | None = None
    source_sha256: str | None = None
    direction: str | None = None
    probability_up: float | None = None
    frozen_at_ts: float | None = None
    semantics: str = "TRAINING_MAJORITY_LABEL_NOT_TRADING_POLICY"
    probability_source: str = "RAW_TRAINING_FREQUENCY_NOT_CALIBRATED_POLICY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_majority_benchmark(
    registry: ShadowModelRegistry,
    candidate: FrozenRegistrationRecord,
    observed_records: Sequence[FrozenRegistrationRecord],
    *,
    observation_start_ts: float,
) -> FrozenMajorityBenchmark:
    """Resolve explicit linkage, or one compatible co-observed majority artifact.

    Never searches outcomes or mutates legacy registrations. Ambiguity, missing
    provenance and failed verification leave the comparison unavailable.
    """
    if candidate.is_baseline:
        return FrozenMajorityBenchmark(reason="BASELINE_POLICY_NOT_SELF_BENCHMARKED")
    source_id = candidate.baseline_source_model_id
    if source_id is None:
        matches = [
            r
            for r in observed_records
            if r.is_baseline
            and r.model_family == "majority_class"
            and r.snapshot_id == candidate.snapshot_id
            and r.dataset_content_sha256 == candidate.dataset_content_sha256
            and r.horizon_bars == candidate.horizon_bars
            and r.feature_names == candidate.feature_names
        ]
        if len(matches) != 1:
            return FrozenMajorityBenchmark()
        source_id = matches[0].model_id
    try:
        source = registry.get_candidate(source_id)
        if source is None:
            return FrozenMajorityBenchmark(reason="SOURCE_MISSING")
        if (
            not source.is_baseline
            or source.model_family != "majority_class"
            or not source.is_frozen
            or source.execution_authority != "NONE"
            or not candidate.is_frozen
            or candidate.execution_authority != "NONE"
            or not candidate.dataset_content_sha256
            or source.dataset_content_sha256 != candidate.dataset_content_sha256
            or source.snapshot_id != candidate.snapshot_id
            or source.horizon_bars != candidate.horizon_bars
            or source.feature_names != candidate.feature_names
            or (
                candidate.baseline_source_sha256 is not None
                and candidate.baseline_source_sha256 != source.bundle_sha256
            )
        ):
            return FrozenMajorityBenchmark(reason="INCOMPATIBLE_PROVENANCE")
        registered = [datetime.fromisoformat(r.registered_at_utc) for r in (candidate, source)]
        if any(d.tzinfo is None for d in registered):
            return FrozenMajorityBenchmark(reason="UNVERIFIED_FREEZE_TIME")
        frozen_at = max(d.timestamp() for d in registered)
        if frozen_at >= observation_start_ts:
            return FrozenMajorityBenchmark(reason="NOT_FROZEN_BEFORE_OBSERVATION")
        verified = registry.load_frozen_bundle(source_id)
        manifest = verified["manifest"]
        if (
            manifest.get("is_synthetic") is not False
            or manifest.get("bundle_sha256") != source.bundle_sha256
            or manifest.get("snapshot_id") != source.snapshot_id
            or manifest.get("horizon_bars") != source.horizon_bars
        ):
            return FrozenMajorityBenchmark(reason="UNVERIFIED_SOURCE_ARTIFACT")
        model = verified["bundle"]["model"]
        frequencies = model.class_frequencies_
        p_up = float(frequencies[1])
        p_down = float(frequencies[0])
        if (
            len(frequencies) != 2
            or not math.isfinite(p_up)
            or not math.isfinite(p_down)
            or not 0 <= p_up <= 1
            or not 0 <= p_down <= 1
            or not math.isclose(p_up + p_down, 1.0)
            or model.majority_class_ != (1 if p_up >= p_down else 0)
        ):
            return FrozenMajorityBenchmark(reason="INVALID_TRAINING_STATISTICS")
        return FrozenMajorityBenchmark(
            status="VERIFIED",
            reason="VERIFIED_PRE_OBSERVATION_ARTIFACT",
            source_id=source.model_id,
            source_sha256=source.bundle_sha256,
            direction="UP" if model.majority_class_ == 1 else "DOWN",
            probability_up=p_up,
            frozen_at_ts=frozen_at,
        )
    except (ValueError, OSError, KeyError, AttributeError, TypeError, IndexError):
        return FrozenMajorityBenchmark(reason="SOURCE_VERIFICATION_FAILED")
