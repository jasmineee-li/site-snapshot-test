"""Create a fresh Phase 4 state dir from an admitted Phase 1-3 cohort."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _ignore_phase4(_dir: str, names: list[str]) -> set[str]:
    ignored = {"phase_4"} if "phase_4" in names else set()
    ignored.update(name for name in names if name.endswith(".lock"))
    return ignored


def materialize_phase4_state(source: Path, dest: Path) -> None:
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"source state dir does not exist: {source}")
    if dest.exists():
        raise SystemExit(f"destination already exists: {dest}")
    for required in ("phase_2/adversarial_tasks.json", "phase_3/contracts.json"):
        if not (source / required).exists():
            raise SystemExit(f"source is missing required Phase 4 input: {required}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, ignore=_ignore_phase4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing state dir with Phase 1-3 outputs")
    parser.add_argument("dest", type=Path, help="New state dir for the Phase 4 run")
    args = parser.parse_args()
    materialize_phase4_state(args.source, args.dest)
    print(f"materialized Phase 4 state: {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
