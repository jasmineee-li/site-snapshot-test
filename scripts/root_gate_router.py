"""Decide whether the required root gates need to run for changed paths."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

TASKGEN_ROOT = "packages/warp-taskgen"
TASKGEN_ENTRYPOINTS = frozenset(
    {
        ".github/workflows/taskgen-acceptance.yml",
        "scripts/accept_taskgen.sh",
    }
)


def _normalize_path(path: str) -> str:
    """Normalize the checkout-relative spelling without resolving filesystem paths."""

    normalized = path.strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("/") or ".." in normalized.split("/"):
        return ""
    return normalized


def _is_taskgen_owned(path: str) -> bool:
    normalized = _normalize_path(path)
    return (
        normalized == TASKGEN_ROOT
        or normalized.startswith(f"{TASKGEN_ROOT}/")
        or normalized in TASKGEN_ENTRYPOINTS
    )


def requires_root_gate(paths: Iterable[str]) -> bool:
    """Return whether changed paths require the complete root gate.

    An empty or whitespace-only path set is intentionally treated as requiring
    the gate. The caller may not have resolved the base revision or may have
    observed an empty diff, and both cases must fail safe.
    """

    saw_path = False
    for path in paths:
        raw_path = path.strip()
        if not raw_path:
            continue
        normalized = _normalize_path(raw_path)
        if not normalized:
            return True
        saw_path = True
        if not _is_taskgen_owned(normalized):
            return True
    return not saw_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Changed checkout-relative path; repeat once per path.",
    )
    parser.add_argument(
        "--paths-from-stdin",
        action="store_true",
        help="Read one changed checkout-relative path per line from stdin.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = list(args.path)
    if args.paths_from_stdin:
        paths.extend(line.rstrip("\n") for line in sys.stdin)
    print("run" if requires_root_gate(paths) else "skip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
