"""Real second-interpreter protection without any broker transport."""

import subprocess  # noqa: S404 - controlled offline child interpreter
import sys

from iqoption_research.market_data.collection_engine import HistoricalCollectionCoordinator


def test_foreign_worker_cannot_acquire_owned_job(tmp_path):
    coordinator = HistoricalCollectionCoordinator(tmp_path)
    coordinator.acquire_job_lock("offline-concurrent-job")
    try:
        code = """
import sys
from pathlib import Path
from iqoption_research.market_data.collection_engine import HistoricalCollectionCoordinator
try:
    HistoricalCollectionCoordinator(Path(sys.argv[1])).acquire_job_lock("offline-concurrent-job")
except RuntimeError:
    sys.exit(0)
sys.exit(9)
"""
        result = subprocess.run(  # noqa: S603 - controlled offline child
            [sys.executable, "-c", code, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert coordinator._job_lock_owned_by_me("offline-concurrent-job")
    finally:
        coordinator.release_job_lock("offline-concurrent-job")
