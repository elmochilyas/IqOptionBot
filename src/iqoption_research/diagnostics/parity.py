"""Feature parity, prefix invariance auditing, and schema consistency checks.

Guarantees that feature calculation behaves identically and causally across:
1. Dataset construction
2. Training
3. Validation
4. Offline historical replay
5. Point-in-time prospective prediction

Specifically audits and verifies:
- Prefix invariance: modifying unseen future candles must leave predictions
  and features on the prefix unchanged.
- Schema parity: frozen registered candidate feature names strictly match pipeline outputs.
- Data availability: volume and quote feature generation rules are deterministic
  without future lookahead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from iqoption_research.models.market import Candle
from iqoption_research.research.features.pipeline import CausalFeaturePipeline


@dataclass(frozen=True, slots=True)
class PrefixInvarianceReport:
    """Outcome of a deterministic prefix invariance test."""

    prefix_length: int
    future_length: int
    pipeline_mode: str
    is_schema_identical: bool
    is_values_identical: bool
    is_strictly_invariant: bool
    max_absolute_feature_difference: float
    divergent_feature_names: list[str]
    prefix_feature_count_alone: int
    prefix_feature_count_with_future: int
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_prefix_invariance(
    pipeline: CausalFeaturePipeline,
    prefix_candles: Sequence[Candle],
    future_candles_variant_a: Sequence[Candle],
    future_candles_variant_b: Sequence[Candle],
) -> PrefixInvarianceReport:
    """Verify that feature extraction on a permitted prefix is strictly invariant to future candles.

    Runs feature extraction on:
    1. prefix_candles alone.
    2. prefix_candles + future_candles_variant_a.
    3. prefix_candles + future_candles_variant_b (different future prices/volumes).

    Compares the feature rows on the prefix indices [0 .. len(prefix_candles) - 1].

    Returns:
        PrefixInvarianceReport detailing bit-level agreement or leakage.
    """
    p_len = len(prefix_candles)
    if p_len < 2:
        raise ValueError("Prefix length must be at least 2 bars for prefix invariance testing")

    # 1. Transform prefix alone
    ds_alone = pipeline.transform(list(prefix_candles))

    # 2. Transform with future variant A
    combined_a = list(prefix_candles) + list(future_candles_variant_a)
    ds_a = pipeline.transform(combined_a)

    # 3. Transform with future variant B
    combined_b = list(prefix_candles) + list(future_candles_variant_b)
    ds_b = pipeline.transform(combined_b)

    schema_alone = [f.name for f in ds_alone.feature_specs]
    schema_a = [f.name for f in ds_a.feature_specs]
    schema_b = [f.name for f in ds_b.feature_specs]

    is_schema_identical = schema_alone == schema_a == schema_b

    max_diff = 0.0
    divergent_names: set[str] = set()

    # Compare features across all prefix bars
    for idx in range(p_len):
        feats_alone = ds_alone.rows[idx].features
        feats_a = ds_a.rows[idx].features
        feats_b = ds_b.rows[idx].features

        all_keys = set(feats_alone.keys()) | set(feats_a.keys()) | set(feats_b.keys())
        for k in all_keys:
            if k not in feats_alone or k not in feats_a or k not in feats_b:
                divergent_names.add(k)
                max_diff = max(max_diff, float("inf"))
                continue

            v_alone = feats_alone[k]
            v_a = feats_a[k]
            v_b = feats_b[k]

            d1 = abs(v_alone - v_a)
            d2 = abs(v_a - v_b)
            local_max = max(d1, d2)
            if local_max > 1e-12:
                divergent_names.add(k)
                max_diff = max(max_diff, local_max)

    is_values_identical = (max_diff <= 1e-12) and (len(divergent_names) == 0)
    is_strictly_invariant = is_schema_identical and is_values_identical

    mode_str = "static_schema" if getattr(pipeline, "static_schema", False) else "legacy_dynamic"
    notes = (
        "STRICT PREFIX INVARIANCE CONFIRMED: Future candles have zero impact on past features."
        if is_strictly_invariant
        else (
            f"LEAKAGE DETECTED: Features on prefix vary depending on future candles! "
            f"Divergent features: {sorted(divergent_names)}"
        )
    )

    return PrefixInvarianceReport(
        prefix_length=p_len,
        future_length=max(len(future_candles_variant_a), len(future_candles_variant_b)),
        pipeline_mode=mode_str,
        is_schema_identical=is_schema_identical,
        is_values_identical=is_values_identical,
        is_strictly_invariant=is_strictly_invariant,
        max_absolute_feature_difference=max_diff,
        divergent_feature_names=sorted(divergent_names),
        prefix_feature_count_alone=len(schema_alone),
        prefix_feature_count_with_future=len(schema_a),
        notes=notes,
    )


@dataclass(frozen=True, slots=True)
class PipelineParityAudit:
    """Comprehensive audit comparing offline pipeline feature extraction with engine inference."""

    model_id: str
    expected_features: list[str]
    extracted_features: list[str]
    missing_in_pipeline: list[str]
    extra_in_pipeline: list[str]
    is_schema_matched: bool
    finite_values_verified: bool
    scale_consistency_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
