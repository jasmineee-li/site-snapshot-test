"""One-shot: drop ``image_alt_text`` concealment from the dataset.

Bug H (2026-04-23): PVPO's TreeWalker in ``worldsim/phase_4/pvpo_query.js``
substring-matches text nodes only and never DOM attributes. Alt text
lives in ``<img alt="...">`` attribute space, so Phase 4's encounter
gate ``max_coverage == 0`` is deterministic for every image_alt_text
task. Phase 2c's text-content probe is correctly predicting that Phase 4
failure, but each such task still costs probe wall-clock + infeasible-
bucket noise that masks other fixes. Drop them at the dataset layer.

Scope: filter entries with ``concealment == "image_alt_text"`` out of
- ``logs/phase_2/adversarial_plans.json`` (the generation snapshot)
- ``logs/phase_2/adversarial_tasks.json`` (verified admitted tasks)
- ``logs/phase_2/adversarial_tasks.infeasible.json`` (infeasibles)
- ``logs/phase_2/shards/*.json`` (so a future orphan-recovery re-run
  cannot re-inject them)

Audit sidecar (``--preserve-history`` default ON): write all dropped
records to ``logs/phase_2/adversarial_tasks.dropped_image_alt_text.json``
with a ``dropped_from`` field marking which input file they came from.
Backups for each modified file: ``<path>.pre-image-alt-drop.bak`` —
does not overwrite existing backups.

Usage::

    # dry-run
    uv run python scripts/drop_image_alt_text_tasks.py

    # apply (backs up + overwrites)
    uv run python scripts/drop_image_alt_text_tasks.py --apply

    # apply without writing the audit sidecar
    uv run python scripts/drop_image_alt_text_tasks.py --apply --no-preserve-history

Idempotent: a re-run after ``--apply`` finds zero image_alt_text
entries and is a no-op.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

_CONCEALMENT = "image_alt_text"


def _filter_records(
    records: list[dict[str, Any]], *, source: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict) and record.get("concealment") == _CONCEALMENT:
            audit = dict(record)
            audit["dropped_from"] = source
            dropped.append(audit)
        else:
            kept.append(record)
    return kept, dropped


def _process_json_array_file(
    path: pathlib.Path, *, apply: bool
) -> tuple[int, int, list[dict[str, Any]]]:
    """Return (total_before, dropped_count, dropped_records)."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return 0, 0, []
    if not isinstance(data, list):
        return 0, 0, []
    kept, dropped = _filter_records(data, source=path.name)
    if dropped and apply:
        backup = path.with_suffix(path.suffix + ".pre-image-alt-drop.bak")
        if not backup.exists():
            backup.write_text(path.read_text())
        path.write_text(json.dumps(kept, indent=2))
    return len(data), len(dropped), dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        default="logs",
        help="Base logs dir (default: ./logs).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Backup + overwrite input files. Without this, dry-run only.",
    )
    parser.add_argument(
        "--no-preserve-history",
        dest="preserve_history",
        action="store_false",
        default=True,
        help=(
            "Skip writing adversarial_tasks.dropped_image_alt_text.json audit "
            "sidecar. Default is to write it."
        ),
    )
    args = parser.parse_args()

    base = pathlib.Path(args.state_dir) / "phase_2"
    targets: list[pathlib.Path] = [
        base / "adversarial_plans.json",
        base / "adversarial_tasks.json",
        base / "adversarial_tasks.infeasible.json",
        *sorted((base / "shards").glob("*.json")),
    ]

    all_dropped: list[dict[str, Any]] = []
    totals = {"total_before": 0, "total_dropped": 0, "files_touched": 0}

    for path in targets:
        if not path.exists():
            print(f"SKIP (missing): {path}")
            continue
        before, dropped_count, dropped_records = _process_json_array_file(path, apply=args.apply)
        totals["total_before"] += before
        totals["total_dropped"] += dropped_count
        if dropped_count:
            totals["files_touched"] += 1
        suffix = " (applied)" if args.apply and dropped_count else ""
        print(f"{path.name}: total_before={before} dropped={dropped_count}{suffix}")
        all_dropped.extend(dropped_records)

    if args.preserve_history and all_dropped:
        sidecar = base / "adversarial_tasks.dropped_image_alt_text.json"
        if args.apply:
            existing: list[dict[str, Any]] = []
            if sidecar.exists():
                try:
                    prior = json.loads(sidecar.read_text())
                    if isinstance(prior, list):
                        existing = prior
                except Exception:
                    pass
            seen_ids = {r.get("id") for r in existing if isinstance(r, dict)}
            for r in all_dropped:
                if r.get("id") not in seen_ids:
                    existing.append(r)
                    seen_ids.add(r.get("id"))
            sidecar.write_text(json.dumps(existing, indent=2))
            print(f"AUDIT sidecar written: {sidecar} ({len(existing)} total records)")
        else:
            print(f"DRY-RUN: would write {len(all_dropped)} records to {sidecar.name}")

    print()
    print(
        f"SUMMARY: {totals['total_dropped']} image_alt_text records across "
        f"{totals['files_touched']} files "
        f"(of {totals['total_before']} total records scanned). "
        f"{'APPLIED' if args.apply else 'DRY-RUN — pass --apply to commit'}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
