"""Phase 46 research CLI - focused validation + report publishing.

Usage examples (from repo root):
  python scripts/run_phase46_research.py focused --repo-root . --bootstrap-replicates 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def cmd_focused(args: argparse.Namespace) -> int:
    from iqoption_research.phase46.focused_validation import run_phase46_focused_validation
    from iqoption_research.phase46.reporting import publish_focused_validation

    repo_root = Path(args.repo_root).resolve()
    result = run_phase46_focused_validation(
        repo_root,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_replicates_sensitivity=args.bootstrap_replicates_sensitivity,
        run_nested=not args.no_nested,
    )
    paths = publish_focused_validation(repo_root, result)
    print("Phase 46 focused validation complete.")
    print("Artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 46 research tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    p_focused = sub.add_parser("focused", help="Run WS1 focused validation for confirmed 3 candidates")
    p_focused.add_argument("--repo-root", default=".", help="Repository root path")
    p_focused.add_argument("--bootstrap-replicates", type=int, default=10000)
    p_focused.add_argument("--bootstrap-replicates-sensitivity", type=int, default=10000)
    p_focused.add_argument("--no-nested", action="store_true", help="Skip focused nested evaluation emphasis")
    p_focused.set_defaults(func=cmd_focused)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
