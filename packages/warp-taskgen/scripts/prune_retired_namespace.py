#!/usr/bin/env python3
"""Remove only cache residue from the retired top-level ``worldsim`` tree.

The host sync excludes Python caches from rsync's delete set.  A long-lived
checkout can therefore retain an empty namespace-package directory after the
source files are removed.  This helper is intentionally fail-closed: it first
validates every entry and only mutates the tree when every file is a bytecode
cache directly below a ``__pycache__`` directory.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path


class CleanupRefused(RuntimeError):
    """Raised when the retired namespace contains a substantive entry."""


def _has_path(path: Path) -> bool:
    """Return whether ``path`` exists, including as a broken symlink."""

    return path.exists() or path.is_symlink()


def _substantive_entries(retired: Path) -> list[Path]:
    """List entries that are not permitted cache residue."""

    if retired.is_symlink() or not retired.is_dir():
        return [retired]

    substantive: list[Path] = []
    for path in retired.rglob("*"):
        # ``Path.is_*`` follows links.  Reject links before inspecting their
        # targets so cleanup can never escape the retired tree.
        if path.is_symlink():
            substantive.append(path)
            continue
        if path.is_dir():
            continue
        if path.is_file() and path.suffix == ".pyc" and path.parent.name == "__pycache__":
            continue
        substantive.append(path)
    return substantive


def prune_retired_namespace(repo_root: Path) -> None:
    """Prune an allowed cache-only retired namespace and assert its absence."""

    retired = repo_root / "worldsim"
    if not _has_path(retired):
        return

    substantive = _substantive_entries(retired)
    if substantive:
        shown = ", ".join(str(path.relative_to(repo_root)) for path in substantive[:10])
        suffix = " ..." if len(substantive) > 10 else ""
        raise CleanupRefused(
            f"retired worldsim path contains substantive entries; refusing cleanup: {shown}{suffix}"
        )

    # Repeat the substantive scan immediately before mutation.  The sync guard
    # prevents known active jobs, but another process can still change the
    # retired tree between the first inspection and cleanup.
    substantive = _substantive_entries(retired)
    if substantive:
        shown = ", ".join(str(path.relative_to(repo_root)) for path in substantive[:10])
        suffix = " ..." if len(substantive) > 10 else ""
        raise CleanupRefused(
            f"retired worldsim path changed during validation; refusing cleanup: {shown}{suffix}"
        )

    # Remove only the explicitly accepted bytecode files. Empty directories are
    # then removed bottom-up.
    cache_files: list[tuple[Path, tuple[int, int, int]]] = []
    for path in retired.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.suffix != ".pyc":
            continue
        if path.parent.name != "__pycache__":
            continue
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise CleanupRefused(
                f"retired cache entry disappeared during validation: {path}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or path.suffix != ".pyc":
            raise CleanupRefused(f"retired cache entry changed during validation: {path}")
        cache_files.append((path, (metadata.st_dev, metadata.st_ino, metadata.st_mode)))

    for path, identity in cache_files:
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise CleanupRefused(f"retired cache entry disappeared during cleanup: {path}") from exc
        if (metadata.st_dev, metadata.st_ino, metadata.st_mode) != identity:
            raise CleanupRefused(f"retired cache entry changed during cleanup: {path}")
        path.unlink()
    for path in sorted(
        (path for path in retired.rglob("*") if path.is_dir() and not path.is_symlink()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.rmdir()
    retired.rmdir()

    if _has_path(retired):
        raise CleanupRefused("retired worldsim path remains after cache cleanup")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <repo-root>", file=sys.stderr)
        return 2
    try:
        prune_retired_namespace(Path(argv[1]).resolve())
    except CleanupRefused as exc:
        print(f"sync guard blocked: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"sync cleanup failed: {exc}", file=sys.stderr)
        return 2
    print("retired namespace absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
