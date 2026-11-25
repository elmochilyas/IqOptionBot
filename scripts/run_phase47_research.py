"""CLI entrypoint for Phase 47 Predictive Signal Discovery & Bounded Research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqoption_research.phase47.experiments import run_phase47_research
from iqoption_research.phase47.reporting import publish_phase47_reports
from iqoption_research.phase47.signal_discovery import (
    run_signal_discovery,
    write_signal_discovery_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 47 offline predictive research.")
    parser.add_argument("--repo-root", type=Path, default=Path())
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    sig = run_signal_discovery(root)
    sig_paths = write_signal_discovery_reports(root, sig)
    res = run_phase47_research(root, bootstrap_replicates=args.bootstrap_replicates)
    rep_paths = publish_phase47_reports(root, res)
    print(
        json.dumps(
            {
                "signal_discovery": [str(p) for p in sig_paths],
                "reports": rep_paths,
                "completed_count": res["completed_count"],
                "total_planned": res["total_planned"],
                "elapsed_seconds": res["resource_usage"]["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
