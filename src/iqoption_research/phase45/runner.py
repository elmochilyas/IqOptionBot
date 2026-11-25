"""Phase 45 bounded, fail-closed offline research runner.

The runner executes the approved native-timeframe and cross-timeframe screening
matrix on verified, pre-boundary development populations only. It never touches
broker connectivity, wallet state, frozen candidates, or protected evidence.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import binomtest

from iqoption_research.market_data.parquet_storage import read_candles_parquet
from iqoption_research.ml.calibration.calibrator import (
    CalibrationMethod,
    ProbabilityCalibrator,
)
from iqoption_research.ml.features_v3 import FeatureMatrixV3
from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
    PreflightEnforcement,
    compute_candle_content_digest,
    validate_development_population_preflight,
)
from iqoption_research.ml.targets.generator import TargetGenerationError
from iqoption_research.ml.targets.specs import TiePolicy
from iqoption_research.ml.targets_v3 import MultiHorizonTargetEngine
from iqoption_research.ml.walk_forward import BREAK_EVEN_PAYOUT_85, create_model_instance
from iqoption_research.ml.walk_forward_v3 import run_walk_forward_v3_evaluation
from iqoption_research.models.market import Candle
from iqoption_research.phase45.evaluation import (
    balanced_accuracy,
    build_comparison_summary,
    class_distribution,
    holm_adjust,
    recompute_prediction_bootstrap,
    summarize_trial_report,
)
from iqoption_research.phase45.features import Phase45FeaturePipeline
from iqoption_research.phase45.governance import (
    ResearchDatasetGovernanceManifest,
    build_preflight_contract,
    load_governance_manifest,
    verify_governance_manifest,
)
from iqoption_research.research.features.scalers import StandardScaler

REPORTS_DIR = Path("reports/phase45")
DATA_DIR = Path("data/phase45")
SNAPSHOT_ROOT = DATA_DIR / "snapshots"
LEDGER_PATH = REPORTS_DIR / "experiment_ledger.json"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PLAN_ID = "phase45_multi_timeframe_research"

TIMEFRAME_HORIZONS: dict[int, tuple[int, ...]] = {
    60: (1, 5, 15),
    300: (1, 3, 6),
    900: (1, 2, 4),
}

BASELINES: tuple[tuple[str, dict[str, Any], str, float], ...] = (
    ("majority_class", {}, "none", 0.50),
    ("rule_momentum", {"threshold": 0.0, "base_prob": 0.55}, "none", 0.55),
)

MODEL_FAMILIES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("logistic_regression", {"C": 0.5, "penalty": "l2", "max_iter": 1000}),
    (
        "random_forest",
        {"n_estimators": 60, "max_depth": 8, "min_samples_leaf": 20, "n_jobs": 1},
    ),
    (
        "hist_gradient_boosting",
        {"max_iter": 80, "learning_rate": 0.05, "max_leaf_nodes": 15, "early_stopping": False},
    ),
    (
        "compact_neural_network",
        {"hidden_layer_sizes": [16, 8], "alpha": 0.01, "max_iter": 80, "batch_size": 128},
    ),
)

MTF_ABLATIONS: tuple[str, ...] = (
    "m1_only",
    "m1_long_history",
    "m1_m5",
    "m1_m15",
    "m1_m5_m15",
)
CROSS_TIMEFRAME_PRESETS: frozenset[str] = frozenset({"m1_m5", "m1_m15", "m1_m5_m15"})

DISPOSITION = "NO_DEMONSTRATED_EDGE"
EXECUTION_AUTHORITY = "NONE"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timeframe_name(timeframe_seconds: int) -> str:
    return f"M{timeframe_seconds // 60}"


def _horizon_name(timeframe_seconds: int, horizon_bars: int) -> str:
    return f"{_timeframe_name(timeframe_seconds)}_H{horizon_bars}"


def _duration_minutes(timeframe_seconds: int, horizon_bars: int) -> float:
    return round(timeframe_seconds * horizon_bars / 60.0, 6)


@dataclass(frozen=True, slots=True)
class Phase45Progress:
    completed_trials: int
    total_trials: int
    active_model: str
    elapsed_seconds: float
    cancelled: bool
    message: str
    trial_records: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "trial_records": list(self.trial_records),
            "disposition": DISPOSITION,
            "execution_authority": EXECUTION_AUTHORITY,
        }


@dataclass(frozen=True, slots=True)
class Phase45TrialSpec:
    trial_id: str
    stage: str
    timeframe_seconds: int
    timeframe_name: str
    horizon_bars: int
    horizon_name: str
    prediction_duration_minutes: float
    model_family: str
    feature_preset: str
    calibration_method: str
    confidence_threshold: float
    hyperparameters: dict[str, Any]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CachedFeaturePipeline:
    """Delegate to a Phase45 pipeline while reusing one matrix per candle population."""

    def __init__(self, delegate: Phase45FeaturePipeline) -> None:
        self._delegate = delegate
        self._cache: dict[tuple[int, int], FeatureMatrixV3] = {}

    @property
    def preset_name(self) -> str:
        return self._delegate.preset_name

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._delegate.feature_names

    @property
    def feature_count(self) -> int:
        return self._delegate.feature_count

    @property
    def min_warmup_bars(self) -> int:
        return self._delegate.min_warmup_bars

    @property
    def base_timeframe_seconds(self) -> int:
        return self._delegate.base_timeframe_seconds

    def compute_features(self, candles: Sequence[Candle]) -> FeatureMatrixV3:
        key = (id(candles), len(candles))
        cached = self._cache.get(key)
        if cached is None:
            cached = self._delegate.compute_features(candles)
            self._cache[key] = cached
        return cached


@dataclass(frozen=True, slots=True)
class Phase45SuiteResult:
    suite_id: str
    started_at_utc: str
    completed_at_utc: str
    total_planned_trials: int
    completed_trials: int
    failed_trials: int
    skipped_trials: int
    trial_records: tuple[dict[str, Any], ...]
    nested_evaluation: dict[str, Any]
    comparison: dict[str, Any]
    resource_usage: dict[str, Any]
    cancelled: bool
    disposition: str = DISPOSITION
    execution_authority: str = EXECUTION_AUTHORITY

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "trial_records": list(self.trial_records),
        }


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    started_at_utc: str
    started_perf_counter: float = field(default_factory=time.perf_counter)
    peak_working_set_bytes: int = 0
    python_peak_allocated_bytes: int = 0
    elapsed_seconds: float = 0.0

    def snapshot(self) -> ResourceUsage:
        elapsed = time.perf_counter() - self.started_perf_counter
        return ResourceUsage(
            started_at_utc=self.started_at_utc,
            started_perf_counter=self.started_perf_counter,
            peak_working_set_bytes=max(self.peak_working_set_bytes, _peak_working_set_bytes()),
            python_peak_allocated_bytes=max(self.python_peak_allocated_bytes, _tracemalloc_peak()),
            elapsed_seconds=elapsed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at_utc": self.started_at_utc,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "peak_working_set_bytes": self.peak_working_set_bytes,
            "python_peak_allocated_bytes": self.python_peak_allocated_bytes,
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "measurement": (
                "Windows GetProcessMemoryInfo PeakWorkingSetSize and tracemalloc peak"
                if sys.platform == "win32"
                else "tracemalloc peak"
            ),
        }


def _tracemalloc_peak() -> int:
    import tracemalloc

    if tracemalloc.is_tracing():
        return int(tracemalloc.get_traced_memory()[1])
    return 0


def _peak_working_set_bytes() -> int:
    if sys.platform != "win32":
        return 0

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = ctypes.c_void_p
        process = current_process()
        get_memory_info = (
            getattr(ctypes.windll.psapi, "GetProcessMemoryInfo", None)
            or ctypes.windll.kernel32.GetProcessMemoryInfo
        )
        get_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_memory_info.restype = ctypes.c_int
        ok = get_memory_info(process, ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else 0
    except (AttributeError, OSError, TypeError, ValueError):  # pragma: no cover
        return 0


def build_phase45_plan(*, max_trials: int = 60) -> list[Phase45TrialSpec]:
    """Preregister the bounded Phase 45 screening matrix."""
    trials: list[Phase45TrialSpec] = []
    serial = 0

    def add(
        *,
        stage: str,
        timeframe_seconds: int,
        horizon_bars: int,
        model_family: str,
        feature_preset: str,
        calibration_method: str,
        confidence_threshold: float,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        nonlocal serial
        serial += 1
        tf_name = _timeframe_name(timeframe_seconds)
        horizon = _horizon_name(timeframe_seconds, horizon_bars)
        trials.append(
            Phase45TrialSpec(
                trial_id=f"p45_{serial:03d}_{tf_name}_{horizon}_{model_family}_{feature_preset}",
                stage=stage,
                timeframe_seconds=timeframe_seconds,
                timeframe_name=tf_name,
                horizon_bars=horizon_bars,
                horizon_name=horizon,
                prediction_duration_minutes=_duration_minutes(timeframe_seconds, horizon_bars),
                model_family=model_family,
                feature_preset=feature_preset,
                calibration_method=calibration_method,
                confidence_threshold=confidence_threshold,
                hyperparameters=dict(hyperparameters or {}),
                seed=42,
            )
        )

    for timeframe_seconds, horizons in TIMEFRAME_HORIZONS.items():
        for horizon_bars in horizons:
            for family, hp, cal, threshold in BASELINES:
                add(
                    stage="STAGE_A_BASELINES",
                    timeframe_seconds=timeframe_seconds,
                    horizon_bars=horizon_bars,
                    model_family=family,
                    feature_preset="native_only",
                    calibration_method=cal,
                    confidence_threshold=threshold,
                    hyperparameters=hp,
                )
            for family, hp in MODEL_FAMILIES:
                add(
                    stage="STAGE_B_NATIVE_MODELS",
                    timeframe_seconds=timeframe_seconds,
                    horizon_bars=horizon_bars,
                    model_family=family,
                    feature_preset="native_only",
                    calibration_method="isotonic",
                    confidence_threshold=0.55,
                    hyperparameters=hp,
                )

    for preset in MTF_ABLATIONS:
        add(
            stage="STAGE_C_TIMEFRAME_ABLATION",
            timeframe_seconds=60,
            horizon_bars=15,
            model_family="logistic_regression",
            feature_preset=preset,
            calibration_method="isotonic",
            confidence_threshold=0.55,
            hyperparameters={"C": 0.5, "penalty": "l2", "max_iter": 1000},
        )

    deduped: list[Phase45TrialSpec] = []
    seen: set[str] = set()
    for trial in trials:
        if trial.trial_id not in seen:
            seen.add(trial.trial_id)
            deduped.append(trial)
    return deduped[:max_trials]


class Phase45Runner:
    """Execute and persist the approved Phase 45 development research suite."""

    def __init__(self, repo_root: Path, *, boundary_ts: int = DEFAULT_PROTECTED_HOLDOUT_START_TS):
        self.repo_root = Path(repo_root).resolve()
        self.boundary_ts = int(boundary_ts)
        self.reports_dir = self.repo_root / REPORTS_DIR
        self.data_dir = self.repo_root / DATA_DIR
        self.ledger_path = self.repo_root / LEDGER_PATH
        self.predictions_dir = self.repo_root / PREDICTIONS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self._pipelines: dict[tuple[int, str], CachedFeaturePipeline] = {}

    def discover_manifests(self, manifest_ids: Sequence[str] | None = None) -> list[Path]:
        root = self.repo_root / SNAPSHOT_ROOT
        paths: list[Path] = []
        if root.exists():
            paths = sorted(root.glob("*/manifest.json"))
        if manifest_ids:
            wanted = set(manifest_ids)

            def matches(path: Path) -> bool:
                if path.parent.name in wanted or path.stem in wanted:
                    return True
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return False
                return str(payload.get("manifest_id", "")) in wanted

            paths = [p for p in paths if matches(p)]
        return paths

    def _load_population(
        self, manifest_path: Path
    ) -> tuple[ResearchDatasetGovernanceManifest, list[Candle]]:
        manifest = load_governance_manifest(manifest_path)
        if not verify_governance_manifest(manifest, self.repo_root):
            raise ValueError(f"Phase45 manifest verification failed: {manifest.manifest_id}")
        if int(manifest.boundary_ts) != self.boundary_ts:
            raise ValueError(f"Boundary mismatch for {manifest.manifest_id}")
        declared = Path(manifest.parquet_path)
        parquet_path = declared if declared.is_absolute() else self.repo_root / declared
        parquet_path = parquet_path.resolve()
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Phase45 population parquet missing: {parquet_path}")
        candles = read_candles_parquet(parquet_path, verify_checksum=False)
        if _sha256_file(parquet_path) != manifest.parquet_sha256:
            raise ValueError(f"Parquet hash mismatch for {manifest.manifest_id}")
        if compute_candle_content_digest(candles) != manifest.content_digest:
            raise ValueError(f"Candle content digest mismatch for {manifest.manifest_id}")
        if len(candles) != manifest.candle_count:
            raise ValueError(f"Candle count mismatch for {manifest.manifest_id}")
        if any(c.from_ts >= self.boundary_ts for c in candles):
            raise ValueError(f"Protected decision timestamp in {manifest.manifest_id}")
        candles.sort(key=lambda candle: candle.from_ts)
        contract = build_preflight_contract(manifest)
        preflight = validate_development_population_preflight(
            candles,
            contract=contract,
            verified_snapshot_id=manifest.snapshot_id,
            enforcement=PreflightEnforcement.GENUINE,
        )
        if not preflight.is_valid or preflight.status != "AUTHORIZED_DEVELOPMENT_POPULATION":
            raise ValueError(
                f"Preflight failed for {manifest.manifest_id}: {preflight.rejection_reason}"
            )
        return manifest, candles

    def _pipeline(
        self,
        timeframe_seconds: int,
        preset: str,
        cross_tf: Mapping[int, Sequence[Candle]],
        digest: str,
    ) -> CachedFeaturePipeline:
        key = (timeframe_seconds, preset)
        if key not in self._pipelines:
            delegate = Phase45FeaturePipeline(
                base_timeframe_seconds=timeframe_seconds,
                preset=preset,
                source_dataset_digest=digest,
                cross_timeframe_candles=dict(cross_tf) if timeframe_seconds == 60 else None,
                boundary_ts=self.boundary_ts,
            )
            self._pipelines[key] = CachedFeaturePipeline(delegate)
        return self._pipelines[key]

    def _write_ledger(self, records: Sequence[dict[str, Any]]) -> None:
        payload = {
            "schema_version": "phase45.1",
            "suite_id": "phase45_screening",
            "records": list(records),
            "record_count": len(records),
            "disposition": DISPOSITION,
            "execution_authority": EXECUTION_AUTHORITY,
            "note": (
                "Sanitized aggregate ledger. Prediction-level records are stored only in "
                "ignored local storage; this file preserves their SHA-256 digest and counts."
            ),
        }
        self.ledger_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _persist_predictions(self, trial_id: str, report: Any) -> dict[str, Any]:
        records = [
            {
                k: v
                for k, v in rec.items()
                if k
                in {
                    "timestamp",
                    "decision",
                    "true_dir",
                    "win",
                    "is_non_overlapping",
                    "probability_up",
                }
            }
            for rec in getattr(report, "prediction_records", ())
        ]
        raw = _canonical_json(records)
        path = self.predictions_dir / f"{trial_id}.json"
        path.write_text(raw, encoding="utf-8")
        return {
            "prediction_records_count": len(records),
            "prediction_records_sha256": _sha256_text(raw),
            "prediction_records_local_path": str(path.relative_to(self.repo_root)),
        }

    @staticmethod
    def _skip_reason(
        trial: Phase45TrialSpec, pipeline: CachedFeaturePipeline, candle_count: int
    ) -> str | None:
        if trial.model_family != "compact_neural_network":
            return None
        d = pipeline.feature_count
        required_train = max(2500, d * 50)
        if candle_count < required_train * 2 + 1000:
            return (
                f"SKIPPED_INSUFFICIENT_SAMPLE: {candle_count} candles; "
                f"required approximately {required_train * 2 + 1000}"
            )
        return None

    def run_suite(
        self,
        manifest_ids: Sequence[str] | None = None,
        *,
        max_trials: int = 60,
        bootstrap_replicates: int = 1000,
        progress_callback: Callable[[Phase45Progress], None] | None = None,
        cancel_token: Callable[[], bool] | None = None,
        run_nested: bool = True,
    ) -> Phase45SuiteResult:
        import tracemalloc

        started_at = _now()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        resource = ResourceUsage(started_at_utc=started_at)
        plan = build_phase45_plan(max_trials=max_trials)
        manifests = self.discover_manifests(manifest_ids)
        if not manifests:
            raise FileNotFoundError(
                "No Phase45 development manifests found. Run `phase45 inventory` first."
            )

        populations: dict[int, tuple[ResearchDatasetGovernanceManifest, list[Candle]]] = {}
        for path in manifests:
            manifest, candles = self._load_population(path)
            if manifest.timeframe_seconds in populations:
                raise ValueError(
                    f"Duplicate Phase45 development manifest for {manifest.timeframe_seconds}s"
                )
            populations[manifest.timeframe_seconds] = (manifest, candles)

        missing = sorted(set(TIMEFRAME_HORIZONS) - set(populations))
        if missing:
            raise FileNotFoundError(f"Missing Phase45 development populations: {missing}")

        cross_tf = {tf: candles for tf, (_, candles) in populations.items() if tf in (300, 900)}
        cross_digest = "|".join(
            f"{tf}:{populations[tf][0].content_digest}" for tf in sorted(cross_tf)
        )

        records: list[dict[str, Any]] = []
        cancelled = False
        failed = 0
        skipped = 0
        self._write_ledger(records)

        def emit(message: str, active_model: str = "") -> None:
            if progress_callback is None:
                return
            snapshot = resource.snapshot()
            progress_callback(
                Phase45Progress(
                    completed_trials=len(records),
                    total_trials=len(plan),
                    active_model=active_model or (records[-1]["trial_id"] if records else ""),
                    elapsed_seconds=snapshot.elapsed_seconds,
                    cancelled=cancelled or bool(cancel_token and cancel_token()),
                    message=message,
                    trial_records=tuple(records),
                )
            )

        emit("Phase45 suite started", "INIT")
        for trial in plan:
            if cancel_token and cancel_token():
                cancelled = True
                emit("Cancelled; completed trial records preserved", trial.trial_id)
                break

            manifest, candles = populations[trial.timeframe_seconds]
            source_digest = manifest.content_digest
            if trial.timeframe_seconds == 60 and trial.feature_preset in CROSS_TIMEFRAME_PRESETS:
                source_digest = f"{source_digest}|{cross_digest}"
            pipeline = self._pipeline(
                trial.timeframe_seconds, trial.feature_preset, cross_tf, source_digest
            )
            skip = self._skip_reason(trial, pipeline, len(candles))
            if skip:
                skipped += 1
                record = {
                    **trial.to_dict(),
                    "status": "SKIPPED",
                    "reason": skip,
                    "recorded_at_utc": _now(),
                }
                records.append(record)
                self._write_ledger(records)
                emit(skip, trial.trial_id)
                continue

            emit(f"Running {trial.horizon_name} {trial.model_family}", trial.trial_id)
            trial_started = time.perf_counter()
            try:
                report = run_walk_forward_v3_evaluation(
                    candles,
                    cast(Any, pipeline),
                    horizon_bars=trial.horizon_bars,
                    model_family=trial.model_family,
                    hyperparameters=trial.hyperparameters,
                    calibration_method=trial.calibration_method,
                    confidence_threshold=trial.confidence_threshold,
                    trial_id=trial.trial_id,
                    k_folds=3,
                    embargo_bars=5,
                    min_selected_samples=30,
                    min_coverage=0.01,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=trial.seed,
                    enforce_family_constraint=True,
                    timeframe_seconds=trial.timeframe_seconds,
                    horizon_name=trial.horizon_name,
                    max_label_maturity_ts=self.boundary_ts,
                )
                prediction_meta = self._persist_predictions(trial.trial_id, report)
                summary = summarize_trial_report(
                    report,
                    bootstrap_replicates=bootstrap_replicates,
                    metadata={
                        **trial.to_dict(),
                        "dataset_manifest_id": manifest.manifest_id,
                        "dataset_content_digest": manifest.content_digest,
                        "trial_duration_seconds": time.perf_counter() - trial_started,
                        **prediction_meta,
                    },
                )
                summary["status"] = "COMPLETED"
                records.append(summary)
            except TargetGenerationError as exc:
                failed += 1
                records.append({
                    **trial.to_dict(),
                    "status": "FAILED",
                    "reason": f"TARGET_GENERATION: {exc}",
                    "trial_duration_seconds": time.perf_counter() - trial_started,
                    "recorded_at_utc": _now(),
                })
            except Exception as exc:  # noqa: BLE001 - record negatives without aborting the suite
                failed += 1
                records.append({
                    **trial.to_dict(),
                    "status": "FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "trial_duration_seconds": time.perf_counter() - trial_started,
                    "recorded_at_utc": _now(),
                })
            self._write_ledger(records)
            emit(
                f"Completed {trial.trial_id} in {time.perf_counter() - trial_started:.1f}s",
                trial.trial_id,
            )

        nested: dict[str, Any] = {
            "status": "NOT_RUN",
            "reason": "Disabled for this invocation",
            "adaptive_selection_bias_note": (
                "Any nested selection result remains development research and does not remove "
                "adaptive selection bias across the screening matrix."
            ),
        }
        if run_nested and not cancelled:
            try:
                nested = self._nested_evaluation(records, populations, cross_tf)
            except Exception as exc:  # noqa: BLE001 - nested check must not abort the suite
                nested = {
                    "status": "FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "adaptive_selection_bias_note": (
                        "Nested selection does not eliminate adaptive selection bias."
                    ),
                }

        completed_records = [record for record in records if record.get("status") == "COMPLETED"]
        p_values: list[float] = []
        p_value_trial_ids: list[str] = []
        for record in completed_records:
            wins = int(record.get("wins", 0) or 0)
            losses = int(record.get("losses", 0) or 0)
            decisive = wins + losses
            p_value = (
                float(
                    binomtest(
                        wins,
                        decisive,
                        p=float(BREAK_EVEN_PAYOUT_85),
                        alternative="greater",
                    ).pvalue
                )
                if decisive > 0
                else 1.0
            )
            record["break_even_binomial_p"] = p_value
            p_values.append(p_value)
            p_value_trial_ids.append(str(record.get("trial_id", "unknown")))

        holm_rows = holm_adjust(p_values, alpha=0.05) if p_values else []
        for trial_id, row in zip(p_value_trial_ids, holm_rows, strict=True):
            row["trial_id"] = trial_id
            for record in completed_records:
                if record.get("trial_id") == trial_id:
                    record["holm_adjusted_p"] = float(row["p_adjusted"])
                    break

        comparison = build_comparison_summary(completed_records)
        comparison["holm_adjustment"] = {
            "method": "Holm step-down",
            "family": "One-sided exact binomial tests against break-even payout 0.85",
            "alpha": 0.05,
            "comparisons": holm_rows,
            "note": (
                "Holm controls family-wise error only for this declared test family. Tests are "
                "not independent because rolling horizons share overlapping outcomes, and the "
                "adjustment does not eliminate adaptive selection bias from the screening matrix."
            ),
        }
        comparison["holm_adjustment_limitations"] = (
            "Holm adjustment controls the listed family-wise comparisons only. Rolling-horizon "
            "outcomes are correlated, and the adjustment does not eliminate adaptive selection "
            "bias from the screening matrix."
        )
        comparison["independent_evidence"] = "UNAVAILABLE"
        comparison["research_stage"] = "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION"
        comparison["disposition"] = DISPOSITION
        comparison["execution_authority"] = EXECUTION_AUTHORITY

        resource = resource.snapshot()
        result = Phase45SuiteResult(
            suite_id=f"p45_suite_{int(time.time())}",
            started_at_utc=started_at,
            completed_at_utc=_now(),
            total_planned_trials=len(plan),
            completed_trials=sum(1 for r in records if r.get("status") == "COMPLETED"),
            failed_trials=failed,
            skipped_trials=skipped,
            trial_records=tuple(records),
            nested_evaluation=nested,
            comparison=comparison,
            resource_usage=resource.to_dict(),
            cancelled=cancelled,
        )
        self._write_ledger(records)
        emit("Phase45 suite complete", "COMPLETE")
        return result

    def _nested_evaluation(
        self,
        records: Sequence[dict[str, Any]],
        populations: dict[int, tuple[ResearchDatasetGovernanceManifest, list[Candle]]],
        cross_tf: Mapping[int, Sequence[Candle]],
    ) -> dict[str, Any]:
        """Run a small nested chronological selection/assessment check on the best finalist."""

        def _meta(record: dict[str, Any], key: str) -> Any:
            metadata = record.get("metadata_summary")
            return record.get(key) or (metadata.get(key) if isinstance(metadata, dict) else None)

        candidates = [
            r
            for r in records
            if r.get("status") == "COMPLETED"
            and _meta(r, "stage") == "STAGE_B_NATIVE_MODELS"
            and _meta(r, "model_family") == "logistic_regression"
        ]
        if not candidates:
            return {"status": "NOT_RUN", "reason": "No eligible logistic-regression finalist"}

        def score(record: dict[str, Any]) -> tuple[float, float]:
            acc = float(
                record.get("balanced_accuracy") or record.get("directional_accuracy") or 0.0
            )
            brier = float(record.get("brier_score") or 1.0)
            return (acc, -brier)

        best = max(candidates, key=score)
        timeframe_seconds = int(_meta(best, "timeframe_seconds"))
        horizon_bars = int(_meta(best, "horizon_bars"))
        manifest, candles = populations[timeframe_seconds]
        pipeline = self._pipeline(
            timeframe_seconds,
            "native_only",
            cross_tf,
            manifest.content_digest,
        )
        features = pipeline.compute_features(candles)
        target_bundle = MultiHorizonTargetEngine.generate_horizon_targets(
            candles,
            horizon_bars=horizon_bars,
            tie_policy=TiePolicy.EXCLUDE,
            warmup_bars=pipeline.min_warmup_bars,
            enforce_temporal_continuity=True,
            timeframe_seconds=timeframe_seconds,
            horizon_name=str(_meta(best, "horizon_name")),
            max_label_maturity_ts=self.boundary_ts,
        )
        valid = target_bundle.valid_mask & (~features.warmup_mask)
        X_all = np.asarray(features.features[valid], dtype=np.float64)
        y_all = np.asarray(target_bundle.targets[valid], dtype=np.int64)
        timestamps_all = np.asarray(features.timestamps[valid], dtype=np.int64)
        if len(y_all) < 3000:
            return {"status": "NOT_RUN", "reason": "Insufficient mature labels for nested check"}

        outer_folds: list[dict[str, Any]] = []
        n = len(y_all)
        fold_bounds = (
            (0.00, 0.40, 0.45, 0.52),
            (0.52, 0.68, 0.73, 0.82),
        )
        for fold_index, (
            dev_start_fraction,
            dev_end_fraction,
            assess_start_fraction,
            assess_end_fraction,
        ) in enumerate(fold_bounds):
            dev_start = int(n * dev_start_fraction)
            dev_end = int(n * dev_end_fraction)
            assess_start = int(n * assess_start_fraction)
            assess_end = int(n * assess_end_fraction)
            cal_start = min(dev_end + horizon_bars + 5, assess_start - 1)
            cal_end = min(cal_start + max(300, (assess_start - cal_start) // 3), assess_start)
            if not (
                0 <= dev_start < dev_end < cal_start < cal_end < assess_start < assess_end <= n
            ):
                continue

            candidates_specs = [
                {"C": 0.25, "threshold": 0.52},
                {"C": 0.5, "threshold": 0.55},
                {"C": 1.0, "threshold": 0.58},
            ]
            inner_scores: list[dict[str, Any]] = []
            for spec_index, spec in enumerate(candidates_specs):
                inner_train_end = dev_start + int((dev_end - dev_start) * 0.75)
                inner_cal_start = min(inner_train_end + horizon_bars + 5, dev_end - 1)
                inner_cal_end = min(
                    inner_cal_start + max(200, (dev_end - inner_cal_start) // 3), dev_end
                )
                if not (dev_start < inner_train_end < inner_cal_start < inner_cal_end <= dev_end):
                    continue
                scaler = StandardScaler().fit(X_all[dev_start:inner_train_end])
                model = create_model_instance(
                    "logistic_regression", {"C": spec["C"], "penalty": "l2", "max_iter": 1000}
                )
                model.fit(
                    np.asarray(
                        scaler.transform(X_all[dev_start:inner_train_end]), dtype=np.float64
                    ),
                    y_all[dev_start:inner_train_end],
                    feature_names=pipeline.feature_names,
                )
                calibrator = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
                X_inner_cal = np.asarray(scaler.transform(X_all[inner_cal_start:inner_cal_end]))
                calibrator.fit(
                    np.asarray(model.predict_proba(X_inner_cal)),
                    y_all[inner_cal_start:inner_cal_end],
                )
                X_inner_dev = np.asarray(scaler.transform(X_all[inner_cal_end:dev_end]))
                probs = calibrator.calibrate(np.asarray(model.predict_proba(X_inner_dev)))
                chosen = _decision_records(
                    probs,
                    y_all[inner_cal_end:dev_end],
                    timestamps_all[inner_cal_end:dev_end],
                    float(spec["threshold"]),
                    horizon_bars,
                )
                acc = balanced_accuracy(chosen) or 0.0
                inner_scores.append({
                    "candidate_index": spec_index,
                    "C": spec["C"],
                    "threshold": spec["threshold"],
                    "inner_balanced_accuracy": acc,
                    "inner_samples": len(chosen),
                })
            if not inner_scores:
                continue
            selected = max(inner_scores, key=lambda item: item["inner_balanced_accuracy"])

            scaler = StandardScaler().fit(X_all[dev_start:dev_end])
            model = create_model_instance(
                "logistic_regression",
                {"C": selected["C"], "penalty": "l2", "max_iter": 1000},
            )
            model.fit(
                np.asarray(scaler.transform(X_all[dev_start:dev_end])),
                y_all[dev_start:dev_end],
                feature_names=pipeline.feature_names,
            )
            calibrator = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
            X_cal = np.asarray(scaler.transform(X_all[cal_start:cal_end]))
            calibrator.fit(np.asarray(model.predict_proba(X_cal)), y_all[cal_start:cal_end])
            X_assess = np.asarray(scaler.transform(X_all[assess_start:assess_end]))
            probs = calibrator.calibrate(np.asarray(model.predict_proba(X_assess)))
            assessment = _decision_records(
                probs,
                y_all[assess_start:assess_end],
                timestamps_all[assess_start:assess_end],
                float(selected["threshold"]),
                horizon_bars,
            )
            outer_folds.append({
                "fold_index": fold_index,
                "training_window": {
                    "start_ts": int(timestamps_all[dev_start]),
                    "end_ts": int(timestamps_all[dev_end - 1]),
                },
                "selected": selected,
                "inner_candidates": inner_scores,
                "assessment_window": {
                    "start_ts": int(timestamps_all[assess_start]),
                    "end_ts": int(timestamps_all[assess_end - 1]),
                },
                "assessment_samples": len(assessment),
                "assessment_balanced_accuracy": balanced_accuracy(assessment),
                "assessment_non_overlapping_balanced_accuracy": balanced_accuracy([
                    record for record in assessment if record["is_non_overlapping"]
                ]),
                "assessment_class_distribution": class_distribution(assessment),
                "assessment_day_block_bootstrap_ci_95": recompute_prediction_bootstrap(
                    assessment, replicates=1000, seed=42 + fold_index
                ),
                "assessment_status": "ASSESSMENT_COMPLETED",
                "prediction_correlation": "ROLLING_PREDICTIONS_ARE_CORRELATED",
            })

        return {
            "status": "COMPLETED" if outer_folds else "NOT_RUN",
            "reason": "Nested chronological selection on the best eligible finalist",
            "finalist_trial_id": best["trial_id"],
            "timeframe_seconds": timeframe_seconds,
            "horizon_name": best["horizon_name"],
            "prediction_duration_minutes": best["prediction_duration_minutes"],
            "manifest_id": manifest.manifest_id,
            "outer_folds": outer_folds,
            "fold_overlap_status": (
                "SEQUENTIAL_TRAINING_AND_DISJOINT_ASSESSMENT_WINDOWS_WITHIN_ROLLING_SAMPLES"
            ),
            "adaptive_selection_bias_note": (
                "Nested selection separates threshold/hyperparameter choice from each assessment "
                "fold, but it does not eliminate adaptive selection bias across the full matrix."
            ),
            "disposition": DISPOSITION,
            "execution_authority": EXECUTION_AUTHORITY,
        }


def _decision_records(
    probabilities: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    horizon_bars: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    probs = np.asarray(probabilities)
    p_up = probs[:, 1]
    p_down = probs[:, 0]
    ts = np.asarray(timestamps)
    for index, probability_up in enumerate(p_up):
        if probability_up >= threshold:
            decision = "UP"
        elif p_down[index] >= threshold:
            decision = "DOWN"
        else:
            continue
        true_dir = "UP" if int(labels[index]) == 1 else "DOWN"
        records.append({
            "timestamp": int(ts[index]),
            "decision": decision,
            "true_dir": true_dir,
            "win": decision == true_dir,
            "probability_up": float(probability_up),
            "probability_down": float(p_down[index]),
            "is_non_overlapping": bool(index % max(1, horizon_bars) == 0),
        })
    return records


__all__ = [
    "DISPOSITION",
    "EXECUTION_AUTHORITY",
    "Phase45Progress",
    "Phase45Runner",
    "Phase45SuiteResult",
    "Phase45TrialSpec",
    "build_phase45_plan",
]
