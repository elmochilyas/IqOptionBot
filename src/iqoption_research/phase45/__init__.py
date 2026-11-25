"""Phase 45 historical evidence consolidation and dataset governance.

This package is strictly offline and read-only with respect to all original
evidence artifacts. Derived development populations are published only into
``data/phase45/snapshots`` and remain immutable once written.
"""

from __future__ import annotations

from iqoption_research.phase45.evidence import (
    AUTHORITATIVE_M1_CONTENT_DIGEST,
    AUTHORITATIVE_M1_PARQUET_SHA256,
    AUTHORITATIVE_M1_SNAPSHOT_ID,
    SUPPORTED_EVIDENCE_TIMEFRAMES,
    EvidenceInventory,
    PopulationInventory,
    build_evidence_inventory,
    load_population_candles,
    verify_all_receipts,
)
from iqoption_research.phase45.governance import (
    ResearchDatasetGovernanceManifest,
    build_preflight_contract,
    load_governance_manifest,
    publish_development_population,
    verify_governance_manifest,
)

__all__ = [
    "AUTHORITATIVE_M1_CONTENT_DIGEST",
    "AUTHORITATIVE_M1_PARQUET_SHA256",
    "AUTHORITATIVE_M1_SNAPSHOT_ID",
    "SUPPORTED_EVIDENCE_TIMEFRAMES",
    "EvidenceInventory",
    "PopulationInventory",
    "ResearchDatasetGovernanceManifest",
    "build_evidence_inventory",
    "build_preflight_contract",
    "load_governance_manifest",
    "load_population_candles",
    "publish_development_population",
    "verify_all_receipts",
    "verify_governance_manifest",
]
