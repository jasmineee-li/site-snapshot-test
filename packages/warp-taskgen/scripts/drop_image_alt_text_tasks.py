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
        raw = path.read_text()
    except FileNotFoundError:
        return 0, 0, []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return 0, 0, []
    if not isinstance(data, list):
        return 0, 0, []
    kept, dropped = _filter_records(data, source=path.name)
    if dropped and apply:
        backup = path.with_suffix(path.suffix + ".pre-image-alt-drop.bak")
        if not backup.exists():
            # Reuse the bytes we just read — one read, no TOCTOU window
            # between the parse and the backup. Also survives the case
            # where the source mutates between the two read_text calls.
            backup.write_text(raw)
        path.write_text(json.dumps(kept, indent=2))
    return len(data), len(dropped), dropped


def _load_json_array(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _site_of(record: dict[str, Any]) -> str:
    site = record.get("site")
    return str(site).strip().lower() if isinstance(site, str) and site.strip() else "unknown"


def _per_site_report_counts(
    verified: list[dict[str, Any]],
    infeasible: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in verified:
        bucket = counts.setdefault(_site_of(record), {"verified": 0, "infeasible": 0, "skipped": 0})
        feasibility = record.get("feasibility")
        if isinstance(feasibility, dict) and "last_reverify_skipped_at" in feasibility:
            bucket["skipped"] += 1
        else:
            bucket["verified"] += 1
    for record in infeasible:
        bucket = counts.setdefault(_site_of(record), {"verified": 0, "infeasible": 0, "skipped": 0})
        bucket["infeasible"] += 1
    return counts


def _sync_feasibility_report_counts(base: pathlib.Path) -> bool:
    report_path = base / "feasibility_report.json"
    try:
        report = json.loads(report_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if not isinstance(report, dict):
        return False

    verified = _load_json_array(base / "adversarial_tasks.json")
    infeasible = _load_json_array(base / "adversarial_tasks.infeasible.json")
    dropped_source_data = _load_json_array(base / "adversarial_tasks.dropped_source_data.json")
    source_data_dropped_by_kind: dict[str, int] = {}
    for record in dropped_source_data:
        issue = record.get("source_data_issue")
        kind = str(issue.get("kind") or "unknown") if isinstance(issue, dict) else "unknown"
        source_data_dropped_by_kind[kind] = source_data_dropped_by_kind.get(kind, 0) + 1

    report["verified_count"] = len(verified)
    report["infeasible_count"] = len(infeasible)
    report["skipped_already_verified_count"] = sum(
        1
        for record in verified
        if isinstance(record.get("feasibility"), dict)
        and "last_reverify_skipped_at" in record["feasibility"]
    )
    report["per_site"] = _per_site_report_counts(verified, infeasible)
    report["source_data_dropped_count"] = len(dropped_source_data)
    report["source_data_dropped_by_kind"] = source_data_dropped_by_kind
    report_path.write_text(json.dumps(report, indent=2))
    return True


def _sync_pipeline_state_counts(state_dir: pathlib.Path) -> bool:
    state_path = state_dir / "pipeline_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if not isinstance(state, dict) or state.get("step") != "phase_2":
        return False

    base = state_dir / "phase_2"
    verified = _load_json_array(base / "adversarial_tasks.json")
    infeasible = _load_json_array(base / "adversarial_tasks.infeasible.json")
    dropped_source_data = _load_json_array(base / "adversarial_tasks.dropped_source_data.json")
    skipped_count = sum(
        1
        for record in verified
        if isinstance(record.get("feasibility"), dict)
        and "last_reverify_skipped_at" in record["feasibility"]
    )

    state["adversarial_tasks_path"] = str(base / "adversarial_tasks.json")
    state["feasibility_report_path"] = str(base / "feasibility_report.json")
    state["feasibility_infeasible_path"] = str(base / "adversarial_tasks.infeasible.json")
    state["feasibility_dropped_source_data_path"] = str(
        base / "adversarial_tasks.dropped_source_data.json"
    )
    state["feasibility_verified_count"] = len(verified)
    state["feasibility_infeasible_count"] = len(infeasible)
    state["feasibility_skipped_count"] = skipped_count
    state["feasibility_dropped_source_data_count"] = len(dropped_source_data)
    state_path.write_text(json.dumps(state, indent=2))
    return True


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)

    base = pathlib.Path(args.state_dir) / "phase_2"
    targets: list[pathlib.Path] = [
        base / "adversarial_plans.json",
        base / "adversarial_tasks.json",
        base / "adversarial_tasks.infeasible.json",
        base / "adversarial_tasks.dropped_source_data.json",
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
            seen_keys = {
                (r.get("id"), r.get("dropped_from")) for r in existing if isinstance(r, dict)
            }
            for r in all_dropped:
                key = (r.get("id"), r.get("dropped_from"))
                if key not in seen_keys:
                    existing.append(r)
                    seen_keys.add(key)
            sidecar.write_text(json.dumps(existing, indent=2))
            print(f"AUDIT sidecar written: {sidecar} ({len(existing)} total records)")
        else:
            print(f"DRY-RUN: would write {len(all_dropped)} records to {sidecar.name}")

    if args.apply:
        if _sync_feasibility_report_counts(base):
            print("feasibility_report.json counts synchronized with current artifacts")
        else:
            print("SKIP (missing/invalid): feasibility_report.json")
        if _sync_pipeline_state_counts(pathlib.Path(args.state_dir)):
            print("pipeline_state.json phase_2 counts synchronized with current artifacts")
        else:
            print("SKIP (missing/non-phase2/invalid): pipeline_state.json")

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
