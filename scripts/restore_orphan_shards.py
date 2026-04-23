"""One-shot recovery: merge persisted Phase 2a shards into
``adversarial_plans.json`` so orphaned shard output cannot be silently
dropped by a partial Phase 2a re-run.

Background: Phase 2a writes each validated shard to
``logs/<run>/phase_2/shards/<site>-shard-<n>.json`` before returning
(see ``worldsim/phases/phase_2_injections.py:1725`` — the explicit
comment reads "persist this shard's validated output to disk immediately
so a later orchestrator failure (or another shard's failure) cannot
discard it"). However, the orchestrator aggregates in-memory
``SiteInjectionResult`` objects from the current run only. If one shard
is re-run later in isolation, the on-disk sidecars from the prior runs
survive while the aggregator emits only the latest in-memory set —
producing an ``adversarial_plans.json`` that drops dozens of validated
tasks.

This script reconstructs a merged set from disk. Dedup rule: entries
already present in ``adversarial_plans.json`` are authoritative (they
may carry post-aggregation patches). Orphans — ids present in shards
but not in ``adversarial_plans.json`` — are folded in; on collision
between shards, newest-mtime wins.

Idempotent: re-running after ``--apply`` is a no-op because the orphans
are now in ``adversarial_plans.json`` and the recovery pass finds none.

Usage::

    # dry-run: write side-by-side sidecar, report counts, don't modify plans
    uv run python scripts/restore_orphan_shards.py

    # apply: backup and overwrite logs/phase_2/adversarial_plans.json
    uv run python scripts/restore_orphan_shards.py --apply

    # point at a different run dir
    uv run python scripts/restore_orphan_shards.py --state-dir /path/to/logs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from worldsim.phases.phase_2_injections import (
    _normalize_l4_benign_task_ids_in_place,
)


def _site_of(task: dict[str, Any]) -> str:
    sites = task.get("sites") or []
    if isinstance(sites, list) and sites:
        return str(sites[0]).strip().lower() or "unknown"
    primary = task.get("site")
    if isinstance(primary, str) and primary.strip():
        return primary.strip().lower()
    return "unknown"


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  skip {path.name}: {exc}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        print(f"  skip {path.name}: not a JSON array", file=sys.stderr)
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def _collect_shard_orphans(
    shards_dir: Path,
    existing_ids: set[str],
    allowed_sites: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return orphan tasks + per-site orphan counts.

    Only tasks whose ``id`` is absent from ``existing_ids`` are returned.
    When ``allowed_sites`` is not None, tasks whose site is outside the
    set are also skipped. On collision between shards for the same id,
    the entry from the newest-mtime shard wins — this matches the logic
    Phase 2a would use if it re-read the sidecars on aggregation.
    """
    best_by_id: dict[str, tuple[float, dict[str, Any]]] = {}
    shard_files = sorted(shards_dir.glob("*-shard-*.json"))
    for shard_path in shard_files:
        mtime = shard_path.stat().st_mtime
        for task in _load_json_array(shard_path):
            task_id = str(task.get("id") or "")
            if not task_id or task_id in existing_ids:
                continue
            if allowed_sites is not None and _site_of(task) not in allowed_sites:
                continue
            prior = best_by_id.get(task_id)
            if prior is None or mtime > prior[0]:
                best_by_id[task_id] = (mtime, task)

    orphans = [task for _, task in best_by_id.values()]
    counts: dict[str, int] = defaultdict(int)
    for task in orphans:
        counts[_site_of(task)] += 1
    return orphans, dict(counts)


def _per_site_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for task in tasks:
        counts[_site_of(task)] += 1
    return dict(counts)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("logs"),
        help="Base logs dir (default: ./logs).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Overwrite adversarial_plans.json in place. Without this flag, "
        "the script only writes a sidecar and reports what would change.",
    )
    parser.add_argument(
        "--sites",
        type=str,
        default="gitlab,reddit",
        help="Comma-separated allowed site names (default: 'gitlab,reddit' — "
        "strict WASP alignment per CLAUDE.md). Use 'all' to disable the filter.",
    )
    args = parser.parse_args(argv[1:])
    if args.sites.strip().lower() == "all":
        allowed_sites: set[str] | None = None
    else:
        allowed_sites = {s.strip().lower() for s in args.sites.split(",") if s.strip()}

    phase_2_dir = args.state_dir / "phase_2"
    plans_path = phase_2_dir / "adversarial_plans.json"
    shards_dir = phase_2_dir / "shards"
    sidecar_path = phase_2_dir / "adversarial_plans.recovered.json"

    if not plans_path.exists():
        print(f"error: {plans_path} not found", file=sys.stderr)
        return 2
    if not shards_dir.is_dir():
        print(f"error: {shards_dir} not found", file=sys.stderr)
        return 2

    existing_plans = _load_json_array(plans_path)
    existing_ids = {str(task.get("id") or "") for task in existing_plans if task.get("id")}
    existing_by_site = _per_site_counts(existing_plans)

    orphans, orphan_by_site = _collect_shard_orphans(shards_dir, existing_ids, allowed_sites)

    merged = list(existing_plans) + orphans
    _normalize_l4_benign_task_ids_in_place(merged)
    merged_by_site = _per_site_counts(merged)

    sidecar_path.write_text(json.dumps(merged, indent=2, sort_keys=True))

    all_sites = sorted(set(existing_by_site) | set(orphan_by_site) | set(merged_by_site))
    print("Phase 2 orphan-shard recovery")
    print(f"  shards scanned: {len(sorted(shards_dir.glob('*-shard-*.json')))}")
    print(f"  plans before  : {len(existing_plans)}")
    print(f"  orphans found : {len(orphans)}")
    print(f"  plans after   : {len(merged)}")
    print("  per-site:")
    print(f"    {'site':<18} {'before':>8} {'orphan':>8} {'after':>8}")
    for site in all_sites:
        print(
            f"    {site:<18} {existing_by_site.get(site, 0):>8d} "
            f"{orphan_by_site.get(site, 0):>8d} {merged_by_site.get(site, 0):>8d}"
        )
    if orphans:
        print("  new ids:")
        for task in sorted(orphans, key=lambda t: (_site_of(t), str(t.get("id") or ""))):
            concealment = task.get("concealment") or "?"
            framing = task.get("framing") or "?"
            site = _site_of(task)
            print(f"    [{site}] {task.get('id')}  framing={framing} concealment={concealment}")
    print(f"  sidecar written: {sidecar_path}")

    if not args.apply:
        print("  dry-run complete; rerun with --apply to overwrite adversarial_plans.json")
        return 0

    if not orphans:
        print("  nothing to apply (no orphans).")
        return 0

    backup_path = plans_path.with_suffix(plans_path.suffix + ".pre-orphan-recovery.bak")
    if not backup_path.exists():
        backup_path.write_text(plans_path.read_text())
        print(f"  backup: {backup_path}")
    plans_path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    print(f"  applied: {plans_path} now has {len(merged)} plans")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
