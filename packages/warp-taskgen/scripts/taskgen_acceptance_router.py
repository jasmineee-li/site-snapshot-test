"""Decide whether a change needs the Taskgen acceptance boundary."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

CANONICAL_PACKAGE_PREFIX = "packages/warp-taskgen/"
ROOT_ACCEPTANCE_COMMAND = "scripts/accept_taskgen.sh"
ACCEPTANCE_WORKFLOW = ".github/workflows/taskgen-acceptance.yml"


def requires_acceptance(paths: Iterable[str]) -> bool:
    """Return whether any changed path can affect the Taskgen distribution."""

    for path in paths:
        normalized = path.strip()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized.startswith(CANONICAL_PACKAGE_PREFIX):
            return True
        if normalized in {ROOT_ACCEPTANCE_COMMAND, ACCEPTANCE_WORKFLOW}:
            return True
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Changed repository path; repeat once per path.",
    )
    parser.add_argument(
        "--paths-from-stdin",
        action="store_true",
        help="Read one changed repository path per line from stdin.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = list(args.path)
    if args.paths_from_stdin:
        paths.extend(line.rstrip("\n") for line in sys.stdin)
    print("run" if requires_acceptance(paths) else "skip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
