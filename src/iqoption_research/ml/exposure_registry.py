"""Persistent Evidence Exposure Registry (Phase 43).

Tracks cryptographic provenance, timestamp intervals, and access history of all
market evidence ever used across:
- Training
- Calibration
- Validation
- Feature selection
- Model selection
- Final evaluation
- Outcome inspection

Enforces Section 17 holdout policy:
- Prevents silent designation of previously inspected data as untouched holdout.
- Binds dataset identity, content hashes, timestamp intervals, and access purpose.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

DEFAULT_EXPOSURE_DB: Final[Path] = Path("data/evidence_exposure.db")


class ExposurePurpose(StrEnum):
    TRAINING = "TRAINING"
    CALIBRATION = "CALIBRATION"
    VALIDATION = "VALIDATION"
    FEATURE_SELECTION = "FEATURE_SELECTION"
    FEATURE_DISCOVERY = "FEATURE_DISCOVERY"
    MODEL_SELECTION = "MODEL_SELECTION"
    FINAL_EVALUATION = "FINAL_EVALUATION"
    OUTCOME_INSPECTION = "OUTCOME_INSPECTION"


class ExposureClassification(StrEnum):
    GENUINELY_UNTOUCHED = "GENUINELY_UNTOUCHED"
    HISTORICAL_REUSED = "HISTORICAL_REUSED"
    PREVIOUSLY_INSPECTED = "PREVIOUSLY_INSPECTED"


@dataclass(frozen=True, slots=True)
class EvidenceExposureRecord:
    """An immutable record of evidence access within a research workflow."""

    record_id: str
    dataset_id: str
    dataset_sha256: str
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    sample_count: int
    purpose: str
    actor_stage: str
    model_id: str | None
    plan_id: str | None
    recorded_at_utc: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceExposureRegistry:
    """SQLite-backed persistent evidence exposure registry."""

    def __init__(self, repo_root: Path, db_path: Path | None = None) -> None:
        self.repo_root = repo_root
        self.db_path = db_path or (repo_root / DEFAULT_EXPOSURE_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_exposure (
                    record_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    from_ts INTEGER NOT NULL,
                    to_ts INTEGER NOT NULL,
                    from_utc TEXT NOT NULL,
                    to_utc TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    purpose TEXT NOT NULL,
                    actor_stage TEXT NOT NULL,
                    model_id TEXT,
                    plan_id TEXT,
                    recorded_at_utc TEXT NOT NULL,
                    notes TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exposure_ts ON evidence_exposure(from_ts, to_ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exposure_dataset ON evidence_exposure(dataset_id)"
            )
            conn.commit()
        finally:
            conn.close()

    def record_exposure(
        self,
        dataset_id: str,
        dataset_sha256: str,
        from_ts: int,
        to_ts: int,
        purpose: ExposurePurpose,
        actor_stage: str,
        sample_count: int,
        model_id: str | None = None,
        plan_id: str | None = None,
        notes: str = "",
    ) -> EvidenceExposureRecord:
        """Register an access event for a specific market interval."""
        now_utc = datetime.now(UTC).isoformat()
        rec_id = f"exp_{dataset_id[:12]}_{from_ts}_{to_ts}_{purpose.value.lower()[:4]}"

        record = EvidenceExposureRecord(
            record_id=rec_id,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
            from_ts=from_ts,
            to_ts=to_ts,
            from_utc=datetime.fromtimestamp(from_ts, UTC).isoformat(),
            to_utc=datetime.fromtimestamp(to_ts, UTC).isoformat(),
            sample_count=sample_count,
            purpose=purpose.value,
            actor_stage=actor_stage,
            model_id=model_id,
            plan_id=plan_id,
            recorded_at_utc=now_utc,
            notes=notes,
        )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO evidence_exposure (
                    record_id, dataset_id, dataset_sha256, from_ts, to_ts,
                    from_utc, to_utc, sample_count, purpose, actor_stage,
                    model_id, plan_id, recorded_at_utc, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.dataset_id,
                    record.dataset_sha256,
                    record.from_ts,
                    record.to_ts,
                    record.from_utc,
                    record.to_utc,
                    record.sample_count,
                    record.purpose,
                    record.actor_stage,
                    record.model_id,
                    record.plan_id,
                    record.recorded_at_utc,
                    record.notes,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return record

    def check_interval_exposure(
        self,
        from_ts: int,
        to_ts: int,
        _dataset_sha256: str | None = None,
    ) -> tuple[ExposureClassification, list[EvidenceExposureRecord]]:
        """Check whether an interval overlaps previously inspected evidence."""
        # Historical Phase 42 known interval: Nov 12 - Nov 20 (both 2026 and 2024 epochs)
        # 2026 epoch: 1794441600 to 1795219200 (2026-11-12T00:00:00 to 2026-11-20T23:59:59 UTC)
        # 2024 epoch: 1731369600 to 1732147200 (2024-11-12T00:00:00 to 2024-11-20T23:59:59 UTC)
        phase42_holdout_start_2026 = 1794441600
        phase42_holdout_end_2026 = 1795219200
        phase42_holdout_start_2024 = 1731369600
        phase42_holdout_end_2024 = 1732147200

        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM evidence_exposure
                WHERE NOT (to_ts < ? OR from_ts > ?)
                ORDER BY from_ts ASC
                """,
                (from_ts, to_ts),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        records = [
            EvidenceExposureRecord(
                record_id=r["record_id"],
                dataset_id=r["dataset_id"],
                dataset_sha256=r["dataset_sha256"],
                from_ts=r["from_ts"],
                to_ts=r["to_ts"],
                from_utc=r["from_utc"],
                to_utc=r["to_utc"],
                sample_count=r["sample_count"],
                purpose=r["purpose"],
                actor_stage=r["actor_stage"],
                model_id=r["model_id"],
                plan_id=r["plan_id"],
                recorded_at_utc=r["recorded_at_utc"],
                notes=r["notes"] or "",
            )
            for r in rows
        ]

        if records:
            # If inspected or evaluated
            if any(
                r.purpose
                in (
                    ExposurePurpose.FINAL_EVALUATION.value,
                    ExposurePurpose.OUTCOME_INSPECTION.value,
                )
                for r in records
            ):
                return ExposureClassification.PREVIOUSLY_INSPECTED, records
            return ExposureClassification.HISTORICAL_REUSED, records

        # Check hardcoded historical baseline intervals
        if not (to_ts < phase42_holdout_start_2026 or from_ts > phase42_holdout_end_2026):
            return ExposureClassification.HISTORICAL_REUSED, []
        if not (to_ts < phase42_holdout_start_2024 or from_ts > phase42_holdout_end_2024):
            return ExposureClassification.HISTORICAL_REUSED, []

        return ExposureClassification.GENUINELY_UNTOUCHED, []
