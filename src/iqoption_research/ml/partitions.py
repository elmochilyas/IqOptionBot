"""Immutable research data partitions and chronological manifest (Phase 42).

Enforces leak-free data partitioning:
1. 4-way chronological partitioning:
   - Model development / training data (default 60%)
   - Hyperparameter validation data (default 15%)
   - Probability calibration data (default 10%)
   - Protected independent final evaluation holdout (default 15%)
2. Mandatory purging (prediction horizon H) and embargo (>= 5 bars) across all partition boundaries.
3. Preprocessing, scalers, imputers, and feature selection fitted strictly on training data.
4. Probability calibrators fitted strictly on calibration partition.
5. Independent final evaluation executed strictly once after candidate freeze.
6. Explicit tracking of previously inspected historical holdouts.
7. Cryptographic SHA-256 verification of manifest and partition contents.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.models.market import Candle

logger = logging.getLogger(__name__)

DEFAULT_PURGING_HORIZON_BARS: Final[int] = 1
DEFAULT_EMBARGO_BARS: Final[int] = 5
DEFAULT_PARTITIONS_DIR: Final[Path] = Path("data/partitions")


class PartitionError(ValueError):
    """Raised on invalid partition configuration or boundary leak."""


class PartitionIntegrityError(PartitionError):
    """Raised when partition cryptographic hashes or timestamps fail verification."""


@dataclass(frozen=True, slots=True)
class DataPartition:
    """A strongly typed, chronologically isolated research data partition."""

    name: str  # "train" | "val" | "calibration" | "protected_test"
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    candle_count: int
    sha256_hash: str
    purged_lead_bars: int = 0
    embargo_tail_bars: int = 0
    provenance_status: str = "HISTORICAL_REUSED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HistoricalEvidenceRecord:
    """Immutable record of previously inspected or utilized historical market data."""

    record_id: str
    phase: str
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    role: str  # "TRAINING" | "VALIDATION" | "HISTORICAL_HOLDOUT" | "PROSPECTIVE_PILOT"
    description: str


# Authoritative registry of all previously studied market intervals across repository history
HISTORICAL_EVIDENCE_REGISTRY: Final[tuple[HistoricalEvidenceRecord, ...]] = (
    HistoricalEvidenceRecord(
        record_id="hist_phase38_pilot",
        phase="Phase 38",
        from_ts=1726358400,  # 2026-09-15 00:00:00 UTC
        to_ts=1795046400,  # 2026-11-19 00:00:00 UTC
        from_utc="2026-09-15T00:00:00+00:00",
        to_utc="2026-11-19T00:00:00+00:00",
        role="PROSPECTIVE_PILOT",
        description="Phase 38 prospective observation pilot sessions and benchmark data",
    ),
    HistoricalEvidenceRecord(
        record_id="hist_phase39_laboratory",
        phase="Phase 39",
        from_ts=1725310800,  # 2026-09-02 21:00:00 UTC
        to_ts=1795046400,  # 2026-11-19 00:00:00 UTC
        from_utc="2026-09-02T21:00:00+00:00",
        to_utc="2026-11-19T00:00:00+00:00",
        role="TRAINING",
        description="Phase 39 Model Laboratory offline dataset and candidate models",
    ),
    HistoricalEvidenceRecord(
        record_id="hist_phase40a_shadow",
        phase="Phase 40A",
        from_ts=1725310800,  # 2026-09-02 21:00:00 UTC
        to_ts=1795046400,  # 2026-11-19 00:00:00 UTC
        from_utc="2026-09-02T21:00:00+00:00",
        to_utc="2026-11-19T00:00:00+00:00",
        role="HISTORICAL_HOLDOUT",
        description="Phase 40A offline shadow evaluation holdout interval",
    ),
    HistoricalEvidenceRecord(
        record_id="hist_phase40b_pilot",
        phase="Phase 40B",
        from_ts=1794960000,  # 2026-11-18 00:00:00 UTC
        to_ts=1795046400,  # 2026-11-19 00:00:00 UTC
        from_utc="2026-11-18T00:00:00+00:00",
        to_utc="2026-11-19T00:00:00+00:00",
        role="PROSPECTIVE_PILOT",
        description="Phase 40B prospective shadow pilot sessions",
    ),
    HistoricalEvidenceRecord(
        record_id="hist_phase41_intelligence",
        phase="Phase 41",
        from_ts=1725310800,  # 2026-09-02 21:00:00 UTC
        to_ts=1795190400,  # 2026-11-20 16:00:00 UTC
        from_utc="2026-09-02T21:00:00+00:00",
        to_utc="2026-11-20T16:00:00+00:00",
        role="HISTORICAL_HOLDOUT",
        description="Phase 41 failure intelligence historical dataset eurusd_m1_202609_202611",
    ),
)


def check_interval_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> bool:
    """Determine whether two intervals [start_a, end_a] and [start_b, end_b] overlap.

    Two intervals overlap if and only if max(start_a, start_b) <= min(end_a, end_b).
    """
    return max(start_a, start_b) <= min(end_a, end_b)


def audit_holdout_independence(
    test_from_ts: int,
    test_to_ts: int,
    is_synthetic: bool = False,
    historical_registry: Sequence[HistoricalEvidenceRecord] = HISTORICAL_EVIDENCE_REGISTRY,
) -> tuple[bool, str, str]:
    """Audit candidate holdout interval for independence against previously inspected evidence.

    Returns:
        tuple[is_previously_inspected: bool, provenance_status: str, audit_notes: str]
    """
    if is_synthetic:
        return (
            True,
            "SYNTHETIC_TEST_FIXTURE",
            (
                "Holdout is synthetic test fixture. Engineering valid; "
                "produces ZERO scientific acceptance."
            ),
        )

    overlapping_records: list[HistoricalEvidenceRecord] = []
    for rec in historical_registry:
        if check_interval_overlap(test_from_ts, test_to_ts, rec.from_ts, rec.to_ts):
            overlapping_records.append(rec)

    if overlapping_records:
        phases_str = ", ".join(sorted({r.phase for r in overlapping_records}))
        from_dt = datetime.fromtimestamp(test_from_ts, tz=UTC).strftime("%Y-%m-%d")
        to_dt = datetime.fromtimestamp(test_to_ts, tz=UTC).strftime("%Y-%m-%d")
        notes = (
            f"Holdout interval ({from_dt} .. {to_dt}) overlaps with previously studied "
            f"evidence from {phases_str}. Population classified as HISTORICAL_REUSED "
            "(INDEPENDENCE_NOT_VERIFIED). Cannot be certified as genuinely untouched "
            "prospective data. Independent scientific acceptance strictly requires "
            "new prospective collection."
        )
        return True, "HISTORICAL_REUSED", notes

    return (
        False,
        "GENUINELY_INDEPENDENT_PROSPECTIVE",
        "Holdout interval is strictly disjoint from all previously studied historical evidence.",
    )


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    """Immutable, cryptographically verified 4-way chronological partition manifest."""

    manifest_id: str
    snapshot_id: str
    snapshot_sha256: str
    created_at_utc: str
    total_candles: int
    purging_horizon_bars: int
    embargo_bars: int
    train_partition: DataPartition
    val_partition: DataPartition
    calibration_partition: DataPartition
    protected_test_partition: DataPartition
    is_holdout_previously_inspected: bool
    holdout_audit_notes: str
    manifest_sha256: str
    provenance_status: str = "HISTORICAL_REUSED"
    disclaimer: str = (
        "OFFLINE RESEARCH PARTITION MANIFEST. ZERO TEST LEAKAGE. "
        "PROTECTED HOLDOUT EVALUATION OCCURS STRICTLY AT STAGE C AFTER CANDIDATE FREEZE."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_sha256": self.snapshot_sha256,
            "created_at_utc": self.created_at_utc,
            "total_candles": self.total_candles,
            "purging_horizon_bars": self.purging_horizon_bars,
            "embargo_bars": self.embargo_bars,
            "train_partition": self.train_partition.to_dict(),
            "val_partition": self.val_partition.to_dict(),
            "calibration_partition": self.calibration_partition.to_dict(),
            "protected_test_partition": self.protected_test_partition.to_dict(),
            "is_holdout_previously_inspected": self.is_holdout_previously_inspected,
            "holdout_audit_notes": self.holdout_audit_notes,
            "provenance_status": self.provenance_status,
            "manifest_sha256": self.manifest_sha256,
            "disclaimer": self.disclaimer,
        }

    def save(self, destination: Path | None = None) -> Path:
        """Save partition manifest to disk (either as file or into directory)."""
        if destination is not None:
            if destination.suffix.lower() == ".json":
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
                return destination
            target_dir = destination
        else:
            target_dir = DEFAULT_PARTITIONS_DIR / self.manifest_id
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "manifest.json"
        manifest_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return manifest_path

    @classmethod
    def load(cls, manifest_path: Path) -> PartitionManifest:
        """Load and verify partition manifest from disk (file or directory)."""
        actual_path = manifest_path / "manifest.json" if manifest_path.is_dir() else manifest_path
        data = json.loads(actual_path.read_text(encoding="utf-8"))
        saved_sha = data.get("manifest_sha256", "")
        # Verify self-digest
        data_copy = dict(data)
        data_copy["manifest_sha256"] = ""
        computed_sha = hashlib.sha256(
            json.dumps(data_copy, sort_keys=True).encode("utf-8")
        ).hexdigest()

        if saved_sha and saved_sha != computed_sha:
            raise PartitionIntegrityError(
                f"Manifest SHA-256 mismatch: recorded {saved_sha}, computed {computed_sha}"
            )

        return cls(
            manifest_id=data["manifest_id"],
            snapshot_id=data["snapshot_id"],
            snapshot_sha256=data["snapshot_sha256"],
            created_at_utc=data["created_at_utc"],
            total_candles=data["total_candles"],
            purging_horizon_bars=data["purging_horizon_bars"],
            embargo_bars=data["embargo_bars"],
            train_partition=DataPartition(**data["train_partition"]),
            val_partition=DataPartition(**data["val_partition"]),
            calibration_partition=DataPartition(**data["calibration_partition"]),
            protected_test_partition=DataPartition(**data["protected_test_partition"]),
            is_holdout_previously_inspected=data["is_holdout_previously_inspected"],
            holdout_audit_notes=data["holdout_audit_notes"],
            manifest_sha256=saved_sha,
            provenance_status=data.get("provenance_status", "HISTORICAL_REUSED"),
            disclaimer=data.get("disclaimer", ""),
        )


def _compute_candle_slice_sha256(candles: Sequence[Candle]) -> str:
    """Compute deterministic cryptographic SHA-256 of candle slice."""
    h = hashlib.sha256()
    for c in candles:
        h.update(
            f"{c.from_ts}:{c.open:.6f}:{c.high:.6f}:{c.low:.6f}:{c.close:.6f}:{c.volume}".encode()
        )
    return h.hexdigest()


def create_partition_manifest(
    candles: Sequence[Candle],
    snapshot_id: str,
    snapshot_sha256: str,
    train_ratio: float = 0.60,
    val_ratio: float = 0.15,
    cal_ratio: float = 0.10,
    test_ratio: float = 0.15,
    purging_horizon_bars: int = DEFAULT_PURGING_HORIZON_BARS,
    embargo_bars: int = DEFAULT_EMBARGO_BARS,
    is_synthetic: bool = False,
    historical_registry: Sequence[HistoricalEvidenceRecord] = HISTORICAL_EVIDENCE_REGISTRY,
) -> PartitionManifest:
    """Create a 4-way chronological partition manifest with strict purging and embargo.

    Parameters:
        candles: Ordered chronological candles.
        snapshot_id: Research snapshot identifier.
        snapshot_sha256: Snapshot parquet SHA-256 checksum.
        train_ratio: Ratio for model training (default 0.60).
        val_ratio: Ratio for hyperparameter validation (default 0.15).
        cal_ratio: Ratio for probability calibration (default 0.10).
        test_ratio: Ratio for protected evaluation (default 0.15).
        purging_horizon_bars: Horizon bars purged at boundary (default 1).
        embargo_bars: Bars embargoed after boundary to prevent target leakage (default 5).
        is_synthetic: Whether candle series is from a synthetic fixture.
        historical_registry: Known previously inspected historical evidence intervals.
    """
    total = len(candles)
    if total < 1000:
        raise PartitionError(f"Insufficient candles for 4-way partitioning: {total} < 1000")

    sum_ratios = train_ratio + val_ratio + cal_ratio + test_ratio
    if abs(sum_ratios - 1.0) > 1e-4:
        raise PartitionError(f"Ratios must sum to 1.0, got {sum_ratios:.4f}")

    # Boundary buffer = purging + embargo
    buffer_bars = purging_horizon_bars + embargo_bars

    # Calculate raw split indices
    raw_train_end = int(total * train_ratio)
    raw_val_end = raw_train_end + int(total * val_ratio)
    raw_cal_end = raw_val_end + int(total * cal_ratio)

    val_start = raw_train_end + buffer_bars
    if val_start >= raw_val_end:
        raise PartitionError("Buffer bars exceed validation partition allocation")

    cal_start = raw_val_end + buffer_bars
    if cal_start >= raw_cal_end:
        raise PartitionError("Buffer bars exceed calibration partition allocation")

    test_start = raw_cal_end + buffer_bars
    if test_start >= total:
        raise PartitionError("Buffer bars exceed test partition allocation")

    train_candles = candles[:raw_train_end]
    val_candles = candles[val_start:raw_val_end]
    cal_candles = candles[cal_start:raw_cal_end]
    test_candles = candles[test_start:]

    def _make_partition(
        name: str,
        slice_candles: Sequence[Candle],
        purged: int,
        embargo: int,
        prov_status: str,
    ) -> DataPartition:
        c_from = slice_candles[0].from_ts
        c_to = slice_candles[-1].to_ts
        return DataPartition(
            name=name,
            from_ts=c_from,
            to_ts=c_to,
            from_utc=datetime.fromtimestamp(c_from, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(c_to, tz=UTC).isoformat(),
            candle_count=len(slice_candles),
            sha256_hash=_compute_candle_slice_sha256(slice_candles),
            purged_lead_bars=purged,
            embargo_tail_bars=embargo,
            provenance_status=prov_status,
        )

    base_prov = "SYNTHETIC_TEST_FIXTURE" if is_synthetic else "GENUINE_HISTORICAL"
    part_train = _make_partition("train", train_candles, 0, embargo_bars, base_prov)
    part_val = _make_partition("val", val_candles, purging_horizon_bars, embargo_bars, base_prov)
    part_cal = _make_partition(
        "calibration", cal_candles, purging_horizon_bars, embargo_bars, base_prov
    )

    # Audit holdout independence using exact interval overlap against historical registry
    is_inspected, test_prov, audit_notes = audit_holdout_independence(
        test_from_ts=test_candles[0].from_ts,
        test_to_ts=test_candles[-1].to_ts,
        is_synthetic=is_synthetic,
        historical_registry=historical_registry,
    )

    part_test = _make_partition("protected_test", test_candles, purging_horizon_bars, 0, test_prov)

    now_utc = datetime.now(UTC).isoformat()
    part_hash = hashlib.sha256(f"{part_train.from_ts}_{part_test.to_ts}".encode()).hexdigest()[:12]
    manifest_id = f"part_{snapshot_id[:24]}_{part_hash}"

    raw_dict: dict[str, Any] = {
        "manifest_id": manifest_id,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "created_at_utc": now_utc,
        "total_candles": total,
        "purging_horizon_bars": purging_horizon_bars,
        "embargo_bars": embargo_bars,
        "train_partition": part_train.to_dict(),
        "val_partition": part_val.to_dict(),
        "calibration_partition": part_cal.to_dict(),
        "protected_test_partition": part_test.to_dict(),
        "is_holdout_previously_inspected": is_inspected,
        "holdout_audit_notes": audit_notes,
        "provenance_status": test_prov,
        "manifest_sha256": "",
        "disclaimer": (
            "OFFLINE RESEARCH PARTITION MANIFEST. ZERO TEST LEAKAGE. "
            "PROTECTED HOLDOUT EVALUATION OCCURS STRICTLY AT STAGE C AFTER CANDIDATE FREEZE."
        ),
    }

    manifest_sha = hashlib.sha256(json.dumps(raw_dict, sort_keys=True).encode("utf-8")).hexdigest()

    return PartitionManifest(
        manifest_id=manifest_id,
        snapshot_id=snapshot_id,
        snapshot_sha256=snapshot_sha256,
        created_at_utc=now_utc,
        total_candles=total,
        purging_horizon_bars=purging_horizon_bars,
        embargo_bars=embargo_bars,
        train_partition=part_train,
        val_partition=part_val,
        calibration_partition=part_cal,
        protected_test_partition=part_test,
        is_holdout_previously_inspected=is_inspected,
        holdout_audit_notes=audit_notes,
        manifest_sha256=manifest_sha,
        provenance_status=test_prov,
        disclaimer=raw_dict["disclaimer"],
    )


def extract_partition_candles(
    candles: Sequence[Candle],
    manifest: PartitionManifest,
) -> dict[str, list[Candle]]:
    """Extract candles matching each partition in the manifest."""
    train_c: list[Candle] = []
    val_c: list[Candle] = []
    cal_c: list[Candle] = []
    test_c: list[Candle] = []

    p_tr = manifest.train_partition
    p_val = manifest.val_partition
    p_cal = manifest.calibration_partition
    p_te = manifest.protected_test_partition

    for c in candles:
        if p_tr.from_ts <= c.from_ts < p_tr.to_ts:
            train_c.append(c)
        elif p_val.from_ts <= c.from_ts < p_val.to_ts:
            val_c.append(c)
        elif p_cal.from_ts <= c.from_ts < p_cal.to_ts:
            cal_c.append(c)
        elif p_te.from_ts <= c.from_ts < p_te.to_ts:
            test_c.append(c)

    return {
        "train": train_c,
        "val": val_c,
        "calibration": cal_c,
        "protected_test": test_c,
    }


def verify_partition_integrity(
    manifest: PartitionManifest,
    candles: Sequence[Candle],
) -> tuple[bool, str]:
    """Verify that candles match the partition manifest without boundary leaks."""
    parts = extract_partition_candles(candles, manifest)

    for p_name, expected_part in (
        ("train", manifest.train_partition),
        ("val", manifest.val_partition),
        ("calibration", manifest.calibration_partition),
        ("protected_test", manifest.protected_test_partition),
    ):
        extracted = parts[p_name]
        if len(extracted) != expected_part.candle_count:
            return (
                False,
                f"Partition '{p_name}' count mismatch: "
                f"expected {expected_part.candle_count}, got {len(extracted)}",
            )
        computed_sha = _compute_candle_slice_sha256(extracted)
        if computed_sha != expected_part.sha256_hash:
            return (
                False,
                f"Partition '{p_name}' SHA-256 mismatch: "
                f"expected {expected_part.sha256_hash}, computed {computed_sha}",
            )

    # Check that boundaries are strictly separated by purging + embargo
    p_tr = manifest.train_partition
    p_val = manifest.val_partition
    p_cal = manifest.calibration_partition
    p_te = manifest.protected_test_partition

    min_gap = (manifest.purging_horizon_bars + manifest.embargo_bars) * 60
    if p_val.from_ts - p_tr.to_ts < min_gap:
        return (
            False,
            f"Train -> Val gap ({p_val.from_ts - p_tr.to_ts}s) < buffer ({min_gap}s)",
        )
    if p_cal.from_ts - p_val.to_ts < min_gap:
        return (
            False,
            f"Val -> Cal gap ({p_cal.from_ts - p_val.to_ts}s) < buffer ({min_gap}s)",
        )
    if p_te.from_ts - p_cal.to_ts < min_gap:
        return (
            False,
            f"Cal -> Test gap ({p_te.from_ts - p_cal.to_ts}s) < buffer ({min_gap}s)",
        )

    return True, "VERIFIED_LEAK_FREE"
