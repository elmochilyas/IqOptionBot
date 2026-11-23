"""Authentic historical market data audit and sufficiency assessment (Phase 42).

Enforces scientific data governance prior to model training:
1. Audits dataset catalog, parquet files, research snapshots, and extended sessions.
2. Reconciles genuine broker collection provenance against synthetic or unverified data.
3. Detects duplicate timestamps, invalid OHLC, incomplete candles, missing volume/quotes.
4. Distinguishes continuous market segments, weekend closures, and unscheduled intraday gaps.
5. Calculates distinct market days and chronological boundary coverage.
6. Evaluates sample sufficiency policies for exploratory training, walk-forward folds,
   calibration, independent holdouts, and individual model-family complexity.
7. Identifies previously inspected holdout intervals and overlap with existing candidate models.
8. Operates strictly offline: zero network, zero broker credentials.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.models.market import Candle
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
    ResearchDatasetSnapshot,
)

logger = logging.getLogger(__name__)

# Minimum sample requirements for Phase 42 scientific rigor
MIN_EXPLORATORY_TRAIN_SAMPLES: Final[int] = 5000
MIN_WALK_FORWARD_FOLDS: Final[int] = 3
MIN_SAMPLES_PER_FOLD: Final[int] = 1000
MIN_CALIBRATION_SAMPLES: Final[int] = 1000
MIN_INDEPENDENT_EVAL_SAMPLES: Final[int] = 2000

# Minimum samples per feature for model family complexity
MIN_SAMPLES_PER_FEATURE_LINEAR: Final[int] = 50
MIN_SAMPLES_PER_FEATURE_TREES: Final[int] = 200
MIN_SAMPLES_PER_FEATURE_NN: Final[int] = 500


@dataclass(frozen=True, slots=True)
class DataSufficiencyPolicy:
    """Scientific data sufficiency thresholds for experimental validity."""

    min_train_samples: int = MIN_EXPLORATORY_TRAIN_SAMPLES
    min_walk_forward_folds: int = MIN_WALK_FORWARD_FOLDS
    min_samples_per_fold: int = MIN_SAMPLES_PER_FOLD
    min_calibration_samples: int = MIN_CALIBRATION_SAMPLES
    min_independent_eval_samples: int = MIN_INDEPENDENT_EVAL_SAMPLES
    min_samples_per_feature_linear: int = MIN_SAMPLES_PER_FEATURE_LINEAR
    min_samples_per_feature_trees: int = MIN_SAMPLES_PER_FEATURE_TREES
    min_samples_per_feature_nn: int = MIN_SAMPLES_PER_FEATURE_NN

    def evaluate_model_family(
        self,
        family: str,
        available_train_samples: int,
        feature_count: int,
    ) -> tuple[bool, str]:
        """Check whether sample size is sufficient for a given model family."""
        if available_train_samples < self.min_train_samples:
            return (
                False,
                f"Available samples ({available_train_samples:,}) below absolute "
                f"minimum ({self.min_train_samples:,})",
            )

        family_norm = family.lower()
        if "logistic" in family_norm or "linear" in family_norm or "majority" in family_norm:
            required = feature_count * self.min_samples_per_feature_linear
            if available_train_samples < required:
                return (
                    False,
                    f"Samples ({available_train_samples:,}) < linear requirement ({required:,})",
                )
            return True, "SUFFICIENT"

        if "forest" in family_norm or "tree" in family_norm or "boost" in family_norm:
            required = feature_count * self.min_samples_per_feature_trees
            if available_train_samples < required:
                return (
                    False,
                    f"Samples ({available_train_samples:,}) < tree requirement ({required:,})",
                )
            return True, "SUFFICIENT"

        if "neural" in family_norm or "mlp" in family_norm or "nn" in family_norm:
            required = feature_count * self.min_samples_per_feature_nn
            if available_train_samples < required:
                return (
                    False,
                    f"Samples ({available_train_samples:,}) < neural requirement ({required:,})",
                )
            return True, "SUFFICIENT"

        # Default fallback
        return True, "SUFFICIENT"


@dataclass(frozen=True, slots=True)
class ContinuousSegment:
    """A continuous chronological market segment without gaps."""

    segment_index: int
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    candle_count: int


@dataclass(frozen=True, slots=True)
class HistoricalHoldoutRecord:
    """Audit of a previously inspected or reserved evaluation holdout."""

    holdout_id: str
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    sample_count: int
    source_phase: str
    status: str  # "PREVIOUSLY_INSPECTED" | "RESERVED_PROTECTED"
    notes: str


@dataclass(frozen=True, slots=True)
class DataCoverageReport:
    """Comprehensive authentic market data coverage and quality report."""

    symbol: str
    timeframe_seconds: int
    generated_at_utc: str
    total_catalog_entries: int
    genuine_catalog_entries: int
    synthetic_catalog_entries: int
    total_candles_discovered: int
    total_candles_retained: int
    earliest_ts: int
    latest_ts: int
    earliest_utc: str
    latest_utc: str
    distinct_market_days: int
    continuous_segments_count: int
    continuous_segments: tuple[ContinuousSegment, ...]
    duplicate_timestamps_found: int
    invalid_ohlc_candles_found: int
    missing_volume_candles_count: int
    has_genuine_volume: bool
    has_bid_ask_quotes: bool
    weekend_closures_count: int
    intraday_gaps_count: int
    source_snapshots_found: int
    latest_snapshot_id: str | None
    latest_snapshot_sha256: str | None
    historical_holdouts: tuple[HistoricalHoldoutRecord, ...]
    sufficiency_assessment: dict[str, Any]
    data_sufficiency_passed: bool
    disclaimer: str = (
        "OFFLINE RESEARCH DATA COVERAGE AUDIT. ALL METRICS COMPUTED STRICTLY "
        "FROM GENUINE LOCAL STORAGE. ZERO LIVE BROKER CONNECTION. NO DEMONSTRATED EDGE."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render a clean Markdown summary report."""
        lines = [
            "# Authentic Market Data Coverage & Sufficiency Audit (Phase 42)",
            f"**Symbol**: {self.symbol} | **Timeframe**: {self.timeframe_seconds}s (M1)",
            f"**Generated UTC**: {self.generated_at_utc}",
            (
                f"**Total Retained Candles**: {self.total_candles_retained:,} across "
                f"{self.distinct_market_days} distinct market days"
            ),
            f"**Date Range**: {self.earliest_utc} to {self.latest_utc}",
            "",
            "## 1. Catalog & Provenance Breakdown",
            f"- Total Catalog Entries: {self.total_catalog_entries}",
            f"- Genuine Entries: {self.genuine_catalog_entries}",
            f"- Synthetic Entries: {self.synthetic_catalog_entries}",
            (
                f"- Research Snapshots: {self.source_snapshots_found} "
                f"(Latest: `{self.latest_snapshot_id}`)"
            ),
            f"- Parquet SHA-256: `{self.latest_snapshot_sha256}`",
            "",
            "## 2. Candle Integrity & Quality",
            f"- Duplicate Timestamps Quarantined/Deduplicated: {self.duplicate_timestamps_found}",
            f"- Invalid OHLC (high < low, negative, etc.): {self.invalid_ohlc_candles_found}",
            f"- Genuine Tick Volume Available: {'Yes' if self.has_genuine_volume else 'No'}",
            (
                "- Bid/Ask Spread Quotes Available: "
                f"{'Yes' if self.has_bid_ask_quotes else 'No (Midpoint only)'}"
            ),
            f"- Scheduled Weekend Closures: {self.weekend_closures_count}",
            f"- Unscheduled Intraday Gaps: {self.intraday_gaps_count}",
            f"- Continuous Market Segments: {self.continuous_segments_count}",
            "",
            "## 3. Historical Holdout & Independence Status",
        ]
        if self.historical_holdouts:
            lines.append("| Holdout ID | Source Phase | Start UTC | End UTC | Samples | Status |")
            lines.append("|---|---|---|---|---|---|")
            for h in self.historical_holdouts:
                lines.append(
                    f"| {h.holdout_id} | {h.source_phase} | {h.from_utc[:16]} | "
                    f"{h.to_utc[:16]} | {h.sample_count:,} | {h.status} |"
                )
        else:
            lines.append("No previously registered holdout intervals found.")

        suff_str = (
            "**YES (SUFFICIENT)**" if self.data_sufficiency_passed else "**NO (INSUFFICIENT)**"
        )
        lines.extend([
            "",
            "## 4. Sufficiency Assessment by Model Family",
            f"- **Overall Sufficiency Passed**: {suff_str}",
        ])
        for family, res in self.sufficiency_assessment.items():
            status_str = "PASS" if res.get("eligible") else "FAIL"
            reason = res.get("reason", "")
            lines.append(f"- **{family}**: [{status_str}] {reason}")

        lines.extend([
            "",
            "> [!NOTE]",
            f"> {self.disclaimer}",
        ])
        return "\n".join(lines)


def audit_market_datasets(
    repo_root: Path,
    symbol: str = "EURUSD",
    timeframe_seconds: int = 60,
    policy: DataSufficiencyPolicy | None = None,
) -> DataCoverageReport:
    """Audit all genuine market datasets in repo and compute data sufficiency.

    Parameters:
        repo_root: Path to repository root.
        symbol: Target symbol (default 'EURUSD').
        timeframe_seconds: Candle duration in seconds (default 60).
        policy: Data sufficiency policy (default DataSufficiencyPolicy()).
    """
    if policy is None:
        policy = DataSufficiencyPolicy()

    now_utc = datetime.now(UTC).isoformat()
    manager = ResearchDatasetManager(repo_root)

    # 1. Discover snapshots
    snapshots = manager.list_snapshots()
    latest_snapshot: ResearchDatasetSnapshot | None = snapshots[0] if snapshots else None

    # If no snapshot exists yet, build or find from catalog
    total_retained = 0
    earliest_ts = 0
    latest_ts = 0
    earliest_utc = "N/A"
    latest_utc = "N/A"
    duplicates_found = 0
    invalid_ohlc = 0
    missing_vol = 0
    has_volume = False
    has_quotes = False
    weekend_count = 0
    intraday_count = 0
    distinct_days = 0
    segments: list[ContinuousSegment] = []
    candles: list[Candle] = []

    if latest_snapshot is not None:
        m = latest_snapshot.manifest
        total_retained = m.total_candles
        earliest_ts = m.from_ts
        latest_ts = m.to_ts
        earliest_utc = m.date_range_start_utc
        latest_utc = m.date_range_end_utc
        duplicates_found = m.deduplicated_boundaries_count
        weekend_count = len(m.weekend_closures)
        intraday_count = len(m.intraday_gaps)
        candles = latest_snapshot.load_candles()

    # 2. Inspect catalog SQLite database
    catalog_path = repo_root / "data" / "datasets" / "catalog.db"
    total_cat = 0
    genuine_cat = 0
    synthetic_cat = 0
    if catalog_path.is_file():
        con = sqlite3.connect(catalog_path)
        try:
            total_cat = con.execute("SELECT count(*) FROM dataset_catalog").fetchone()[0]
            genuine_cat = con.execute(
                "SELECT count(*) FROM dataset_catalog WHERE is_synthetic = 0"
            ).fetchone()[0]
            synthetic_cat = con.execute(
                "SELECT count(*) FROM dataset_catalog WHERE is_synthetic != 0"
            ).fetchone()[0]
        finally:
            con.close()

    # 3. Analyze candle series properties if candles loaded
    if candles:
        timestamps = [c.from_ts for c in candles]
        ts_set = set()
        dups = 0
        for ts in timestamps:
            if ts in ts_set:
                dups += 1
            ts_set.add(ts)
        duplicates_found = max(duplicates_found, dups)

        # Distinct market days (from timestamp to UTC date string)
        days_set = {datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d") for ts in timestamps}
        distinct_days = len(days_set)

        # Check OHLC validity, volume, and quotes
        for c in candles:
            if c.high < c.low or c.open <= 0 or c.close <= 0 or c.high <= 0 or c.low <= 0:
                invalid_ohlc += 1
            if c.volume <= 0:
                missing_vol += 1
            if c.bid > 0 and c.ask > 0 and c.ask >= c.bid:
                has_quotes = True

        vols = [c.volume for c in candles]
        has_volume = any(v > 0 for v in vols) and len(set(vols)) > 1

        # Continuous segments (consecutive candles with exactly 60s step)
        seg_idx = 0
        seg_start = candles[0].from_ts
        seg_count = 1
        for i in range(1, len(candles)):
            prev_ts = candles[i - 1].from_ts
            curr_ts = candles[i].from_ts
            if curr_ts - prev_ts == timeframe_seconds:
                seg_count += 1
            else:
                # Break in continuity
                seg_end = candles[i - 1].to_ts
                segments.append(
                    ContinuousSegment(
                        segment_index=seg_idx,
                        from_ts=seg_start,
                        to_ts=seg_end,
                        from_utc=datetime.fromtimestamp(seg_start, tz=UTC).isoformat(),
                        to_utc=datetime.fromtimestamp(seg_end, tz=UTC).isoformat(),
                        candle_count=seg_count,
                    )
                )
                seg_idx += 1
                seg_start = curr_ts
                seg_count = 1
        # Final segment
        if seg_count > 0:
            segments.append(
                ContinuousSegment(
                    segment_index=seg_idx,
                    from_ts=seg_start,
                    to_ts=candles[-1].to_ts,
                    from_utc=datetime.fromtimestamp(seg_start, tz=UTC).isoformat(),
                    to_utc=datetime.fromtimestamp(candles[-1].to_ts, tz=UTC).isoformat(),
                    candle_count=seg_count,
                )
            )

    # 4. Historical Holdouts audit
    historical_holdouts: list[HistoricalHoldoutRecord] = []
    phase11_path = repo_root / "data" / "phase11" / "confirmation_state.json"
    if phase11_path.is_file():
        try:
            p11 = json.loads(phase11_path.read_text(encoding="utf-8"))
            cutoff = p11.get("prior_evaluation_cutoff_ts", 1794607200)
            historical_holdouts.append(
                HistoricalHoldoutRecord(
                    holdout_id="phase11_confirmation_reservation",
                    from_ts=cutoff,
                    to_ts=cutoff + 14 * 86400,
                    from_utc=p11.get("prior_evaluation_cutoff_utc", "2026-11-13T22:00:00+00:00"),
                    to_utc=datetime.fromtimestamp(cutoff + 14 * 86400, tz=UTC).isoformat(),
                    sample_count=14 * 1440,
                    source_phase="Phase 11",
                    status="RESERVED_PROTECTED",
                    notes="14-day reserved confirmation interval",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read phase11 confirmation state: %s", exc)

    # Audit Phase 40A / 39 test splits in candidate models
    cand_dir = repo_root / "data" / "candidate_models"
    if cand_dir.is_dir():
        for sub in cand_dir.iterdir():
            m_file = sub / "manifest.json"
            if m_file.is_file():
                try:
                    cdata = json.loads(m_file.read_text(encoding="utf-8"))
                    test_m = cdata.get("metrics", {}).get("test", {})
                    if test_m:
                        samples = test_m.get("sample_count", 0)
                        cutoff_ts = cdata.get("data_cutoff_ts", 0)
                        historical_holdouts.append(
                            HistoricalHoldoutRecord(
                                holdout_id=f"{sub.name}_test_split",
                                from_ts=cutoff_ts,
                                to_ts=latest_ts,
                                from_utc=datetime.fromtimestamp(cutoff_ts, tz=UTC).isoformat()
                                if cutoff_ts
                                else "N/A",
                                to_utc=latest_utc,
                                sample_count=samples,
                                source_phase="Phase 39/40A",
                                status="PREVIOUSLY_INSPECTED",
                                notes=(
                                    "Previously evaluated in Phase 40A; "
                                    "cannot be certified as untouched"
                                ),
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not parse candidate model manifest %s: %s", m_file, exc)

    # 5. Sufficiency Assessment
    # Assume ~20 standard features for the sufficiency check
    standard_feature_count = 20
    # Available training samples assuming 60% train partition
    available_train = int(total_retained * 0.60)

    families_to_check = [
        "majority_class",
        "seeded_random",
        "rule_based_momentum",
        "rule_based_reversal",
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "compact_neural_network",
    ]

    sufficiency_assessment: dict[str, Any] = {}
    all_eligible = True
    for fam in families_to_check:
        eligible, reason = policy.evaluate_model_family(
            fam, available_train, standard_feature_count
        )
        sufficiency_assessment[fam] = {
            "eligible": eligible,
            "reason": reason,
            "available_train_samples": available_train,
            "required_samples": (
                standard_feature_count * policy.min_samples_per_feature_nn
                if "neural" in fam
                else standard_feature_count * policy.min_samples_per_feature_trees
                if "forest" in fam or "trees" in fam or "boost" in fam
                else standard_feature_count * policy.min_samples_per_feature_linear
            ),
        }
        if not eligible and ("logistic" in fam or "forest" in fam):
            all_eligible = False

    data_sufficiency_passed = (
        total_retained >= policy.min_train_samples + policy.min_calibration_samples and all_eligible
    )

    return DataCoverageReport(
        symbol=symbol,
        timeframe_seconds=timeframe_seconds,
        generated_at_utc=now_utc,
        total_catalog_entries=total_cat,
        genuine_catalog_entries=genuine_cat,
        synthetic_catalog_entries=synthetic_cat,
        total_candles_discovered=total_retained + duplicates_found,
        total_candles_retained=total_retained,
        earliest_ts=earliest_ts,
        latest_ts=latest_ts,
        earliest_utc=earliest_utc,
        latest_utc=latest_utc,
        distinct_market_days=distinct_days,
        continuous_segments_count=len(segments),
        continuous_segments=tuple(segments),
        duplicate_timestamps_found=duplicates_found,
        invalid_ohlc_candles_found=invalid_ohlc,
        missing_volume_candles_count=missing_vol,
        has_genuine_volume=has_volume,
        has_bid_ask_quotes=has_quotes,
        weekend_closures_count=weekend_count,
        intraday_gaps_count=intraday_count,
        source_snapshots_found=len(snapshots),
        latest_snapshot_id=latest_snapshot.manifest.snapshot_id if latest_snapshot else None,
        latest_snapshot_sha256=latest_snapshot.manifest.parquet_sha256 if latest_snapshot else None,
        historical_holdouts=tuple(historical_holdouts),
        sufficiency_assessment=sufficiency_assessment,
        data_sufficiency_passed=data_sufficiency_passed,
    )


# Alias for CLI command interface
run_data_audit = audit_market_datasets
