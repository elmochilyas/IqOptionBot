"""Local SQLite experiment tracker for Phase 4 research runs.

The tracker is deliberately local-only. It stores experiment configuration,
metrics, verdicts, and artifact references. It never stores credentials and
never contacts a broker or external service.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TrackerError(Exception):
    """Base exception for experiment tracking errors."""


_ALLOWED_STATUS = {"running", "completed", "failed", "aborted"}
_ALLOWED_VERDICTS = {
    "",
    "NO_RELIABLE_EDGE_DETECTED",
    "INSUFFICIENT_OUT_OF_SAMPLE_DATA",
    "SYNTHETIC_ENGINEERING_ONLY",
    "CANDIDATE_EDGE_REQUIRES_REPLICATION",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    status TEXT NOT NULL,
    dataset_symbol TEXT NOT NULL DEFAULT '',
    dataset_rows INTEGER NOT NULL DEFAULT 0,
    dataset_fingerprint TEXT NOT NULL DEFAULT '',
    is_synthetic INTEGER NOT NULL DEFAULT 1,
    target_type TEXT NOT NULL DEFAULT '',
    horizon_bars INTEGER NOT NULL DEFAULT 1,
    model_name TEXT NOT NULL DEFAULT '',
    split_type TEXT NOT NULL DEFAULT '',
    calibration_method TEXT NOT NULL DEFAULT '',
    evidence_class TEXT NOT NULL DEFAULT '',
    verdict TEXT NOT NULL DEFAULT '',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    config_json TEXT NOT NULL DEFAULT '{}',
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_manifest_sha256 TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(experiment_name);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at_utc);
"""

_SELECT_BASE = (
    "SELECT run_id, created_at_utc, experiment_name, status, dataset_symbol, dataset_rows, "
    "dataset_fingerprint, is_synthetic, target_type, horizon_bars, model_name, split_type, "
    "calibration_method, evidence_class, verdict, metrics_json, config_json, artifact_path, "
    "artifact_manifest_sha256, notes, tags_json FROM experiments"
)

_INSERT_SQL = (
    "INSERT INTO experiments (run_id, created_at_utc, experiment_name, status, "
    "dataset_symbol, dataset_rows, dataset_fingerprint, is_synthetic, target_type, "
    "horizon_bars, model_name, split_type, calibration_method, evidence_class, verdict, "
    "metrics_json, config_json, artifact_path, artifact_manifest_sha256, notes, tags_json) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _json_loads_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _json_loads_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Immutable experiment record loaded from or written to SQLite."""

    run_id: str
    created_at_utc: str
    experiment_name: str
    status: str = "completed"
    dataset_symbol: str = ""
    dataset_rows: int = 0
    dataset_fingerprint: str = ""
    is_synthetic: bool = True
    target_type: str = ""
    horizon_bars: int = 1
    model_name: str = ""
    split_type: str = ""
    calibration_method: str = ""
    evidence_class: str = ""
    verdict: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    artifact_path: str = ""
    artifact_manifest_sha256: str = ""
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "dataset_symbol": self.dataset_symbol,
            "dataset_rows": int(self.dataset_rows),
            "dataset_fingerprint": self.dataset_fingerprint,
            "is_synthetic": bool(self.is_synthetic),
            "target_type": self.target_type,
            "horizon_bars": int(self.horizon_bars),
            "model_name": self.model_name,
            "split_type": self.split_type,
            "calibration_method": self.calibration_method,
            "evidence_class": self.evidence_class,
            "verdict": self.verdict,
            "metrics": dict(self.metrics),
            "config": dict(self.config),
            "artifact_path": self.artifact_path,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "notes": self.notes,
            "tags": list(self.tags),
        }

    def summary(self) -> str:
        verdict = self.verdict or "N/A"
        evidence = self.evidence_class or "N/A"
        return (
            f"{self.run_id} | {self.experiment_name} | {self.model_name or 'n/a'} | "
            f"{self.split_type or 'n/a'} | {evidence} | {verdict}"
        )


class ExperimentTracker:
    """Small, transactional SQLite experiment store."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise TrackerError(f"Failed to initialize experiment database: {exc}") from exc

    def log_experiment(
        self,
        experiment_name: str,
        *,
        model_name: str = "",
        dataset_symbol: str = "",
        dataset_rows: int = 0,
        dataset_fingerprint: str = "",
        is_synthetic: bool = True,
        target_type: str = "",
        horizon_bars: int = 1,
        split_type: str = "",
        calibration_method: str = "",
        evidence_class: str = "",
        verdict: str = "",
        metrics: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        artifact_path: Path | str | None = None,
        artifact_manifest_sha256: str = "",
        notes: str = "",
        tags: list[str] | tuple[str, ...] | None = None,
        status: str = "completed",
        run_id: str | None = None,
    ) -> ExperimentRecord:
        """Insert a new experiment record and return the persisted value."""
        if not experiment_name.strip():
            raise TrackerError("experiment_name must be a non-empty string.")
        if status not in _ALLOWED_STATUS:
            raise TrackerError(
                f"Invalid status {status!r}; expected one of {sorted(_ALLOWED_STATUS)}"
            )
        normalized_verdict = str(verdict).strip().upper()
        if normalized_verdict not in _ALLOWED_VERDICTS:
            raise TrackerError(
                f"Invalid verdict {verdict!r}; expected one of "
                f"{sorted(v for v in _ALLOWED_VERDICTS if v)}"
            )
        if dataset_rows < 0:
            raise TrackerError("dataset_rows must be >= 0.")
        if horizon_bars < 1:
            raise TrackerError("horizon_bars must be >= 1.")

        resolved_run_id = (run_id or uuid.uuid4().hex).strip()
        if not resolved_run_id:
            raise TrackerError("run_id must be a non-empty string.")
        created_at = datetime.now(UTC).isoformat()
        record = ExperimentRecord(
            run_id=resolved_run_id,
            created_at_utc=created_at,
            experiment_name=experiment_name.strip(),
            status=status,
            dataset_symbol=dataset_symbol,
            dataset_rows=int(dataset_rows),
            dataset_fingerprint=dataset_fingerprint,
            is_synthetic=bool(is_synthetic),
            target_type=target_type,
            horizon_bars=int(horizon_bars),
            model_name=model_name,
            split_type=split_type,
            calibration_method=calibration_method,
            evidence_class=evidence_class,
            verdict=normalized_verdict,
            metrics=dict(metrics or {}),
            config=dict(config or {}),
            artifact_path=str(Path(artifact_path).resolve()) if artifact_path else "",
            artifact_manifest_sha256=artifact_manifest_sha256,
            notes=notes,
            tags=list(tags or []),
        )

        try:
            with self._connect() as connection:
                connection.execute(
                    _INSERT_SQL,
                    (
                        record.run_id,
                        record.created_at_utc,
                        record.experiment_name,
                        record.status,
                        record.dataset_symbol,
                        record.dataset_rows,
                        record.dataset_fingerprint,
                        int(record.is_synthetic),
                        record.target_type,
                        record.horizon_bars,
                        record.model_name,
                        record.split_type,
                        record.calibration_method,
                        record.evidence_class,
                        record.verdict,
                        _json_dumps(record.metrics),
                        _json_dumps(record.config),
                        record.artifact_path,
                        record.artifact_manifest_sha256,
                        record.notes,
                        _json_dumps(record.tags),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise TrackerError(f"Experiment run_id already exists: {record.run_id}") from exc
        except sqlite3.Error as exc:
            raise TrackerError(f"Failed to log experiment: {exc}") from exc
        return record

    def get_experiment(self, run_id: str) -> ExperimentRecord:
        """Fetch one experiment by run ID."""
        if not run_id:
            raise TrackerError("run_id must be non-empty.")
        try:
            with self._connect() as connection:
                row = connection.execute(f"{_SELECT_BASE} WHERE run_id = ?", (run_id,)).fetchone()
        except sqlite3.Error as exc:
            raise TrackerError(f"Failed to read experiment: {exc}") from exc
        if row is None:
            raise TrackerError(f"Experiment not found: {run_id}")
        return self._row_to_record(row)

    def list_experiments(
        self,
        *,
        limit: int = 50,
        experiment_name: str | None = None,
        verdict: str | None = None,
    ) -> list[ExperimentRecord]:
        """List recent experiments, optionally filtered by name or verdict."""
        if limit < 1:
            raise TrackerError("limit must be >= 1.")
        clauses: list[str] = []
        params: list[Any] = []
        if experiment_name:
            clauses.append("experiment_name = ?")
            params.append(experiment_name)
        if verdict:
            clauses.append("verdict = ?")
            params.append(verdict.upper())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    f"{_SELECT_BASE}{where} ORDER BY created_at_utc DESC, rowid DESC LIMIT ?",
                    tuple(params),
                ).fetchall()
        except sqlite3.Error as exc:
            raise TrackerError(f"Failed to list experiments: {exc}") from exc
        return [self._row_to_record(row) for row in rows]

    def latest_experiment(
        self,
        *,
        experiment_name: str | None = None,
    ) -> ExperimentRecord | None:
        """Return the most recent experiment, optionally filtered by name."""
        records = self.list_experiments(limit=1, experiment_name=experiment_name)
        return records[0] if records else None

    def delete_experiment(self, run_id: str) -> bool:
        """Delete one experiment record. Returns True when a row was removed."""
        try:
            with self._connect() as connection:
                cursor = connection.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))
        except sqlite3.Error as exc:
            raise TrackerError(f"Failed to delete experiment: {exc}") from exc
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return total experiment count."""
        try:
            with self._connect() as connection:
                row = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()
        except sqlite3.Error as exc:
            raise TrackerError(f"Failed to count experiments: {exc}") from exc
        return int(row[0]) if row is not None else 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            run_id=str(row["run_id"]),
            created_at_utc=str(row["created_at_utc"]),
            experiment_name=str(row["experiment_name"]),
            status=str(row["status"]),
            dataset_symbol=str(row["dataset_symbol"]),
            dataset_rows=int(row["dataset_rows"]),
            dataset_fingerprint=str(row["dataset_fingerprint"]),
            is_synthetic=bool(row["is_synthetic"]),
            target_type=str(row["target_type"]),
            horizon_bars=int(row["horizon_bars"]),
            model_name=str(row["model_name"]),
            split_type=str(row["split_type"]),
            calibration_method=str(row["calibration_method"]),
            evidence_class=str(row["evidence_class"]),
            verdict=str(row["verdict"]),
            metrics=_json_loads_object(str(row["metrics_json"])),
            config=_json_loads_object(str(row["config_json"])),
            artifact_path=str(row["artifact_path"]),
            artifact_manifest_sha256=str(row["artifact_manifest_sha256"]),
            notes=str(row["notes"]),
            tags=_json_loads_list(str(row["tags_json"])),
        )
