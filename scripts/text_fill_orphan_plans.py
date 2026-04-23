"""Targeted Phase 2b for orphan shard tasks only.

After `scripts/restore_orphan_shards.py` folds shard orphans into
``adversarial_plans.json``, those new plans still need text fill
(their ``seed_template`` carries the literal ``{{PAYLOAD_TEXT}}``
placeholder). Running the full ``phase 2`` command would trigger text
fill across every plan — including the already-filled ones whose
``payload_texts`` live in ``adversarial_tasks.json``, not in the plan
dict — so the whole dataset would be re-sampled non-deterministically.

This script fills only the orphans. Existing ``adversarial_tasks.json``
and ``adversarial_tasks.infeasible.json`` entries are preserved
verbatim, merged with the freshly-filled orphans, and written back to
``adversarial_tasks.json`` in a single pass. The feasibility sidecar
is rebuilt downstream by ``phase 2c --force-reverify``.

Usage::

    uv run python scripts/text_fill_orphan_plans.py

    # or target a specific state dir:
    uv run python scripts/text_fill_orphan_plans.py --state-dir /path/to/logs

    # concurrency / model overrides mirror the main phase 2 CLI:
    uv run python scripts/text_fill_orphan_plans.py \\
        --text-model claude-sonnet-4-6 --concurrency 8 --texts-per-plan 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from worldsim.phases.phase_2_injections import (
    DEFAULT_TEXT_FILL_CONCURRENCY,
    DEFAULT_TEXT_FILL_MODEL,
    DEFAULT_TEXTS_PER_PLAN,
)
from worldsim.phases.phase_2_text_fill import fill_texts_for_tasks


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: failed to read {path} ({exc})", file=sys.stderr)
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict)]


async def _run(args: argparse.Namespace) -> int:
    phase_2_dir = args.state_dir / "phase_2"
    plans_path = phase_2_dir / "adversarial_plans.json"
    tasks_path = phase_2_dir / "adversarial_tasks.json"
    infeasible_path = phase_2_dir / "adversarial_tasks.infeasible.json"
    diag_path = phase_2_dir / "text_fill_diagnostics.json"

    plans = _load_json_array(plans_path)
    verified_tasks = _load_json_array(tasks_path)
    infeasible_tasks = _load_json_array(infeasible_path)
    existing_tasks = verified_tasks + infeasible_tasks
    existing_by_id = {str(task.get("id") or ""): task for task in existing_tasks if task.get("id")}

    orphan_plans = [plan for plan in plans if str(plan.get("id") or "") not in existing_by_id]
    if not orphan_plans:
        print(f"no orphan plans to fill; {tasks_path} already covers all {len(plans)} plans")
        return 0

    print(
        f"filling {len(orphan_plans)} orphan plan(s); "
        f"{len(existing_by_id)} existing task(s) preserved as-is"
    )
    print(
        f"  text_model={args.text_model} concurrency={args.concurrency} "
        f"texts_per_plan={args.texts_per_plan}"
    )

    filled_orphans, diagnostics = await fill_texts_for_tasks(
        orphan_plans,
        texts_per_plan=args.texts_per_plan,
        concurrency=args.concurrency,
        model=args.text_model,
    )

    by_status: dict[str, int] = {}
    for diag in diagnostics:
        status = str(diag.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    print("  diagnostics:")
    for status, count in sorted(by_status.items()):
        print(f"    {status:<20} {count:>4d}")

    if not filled_orphans:
        print("error: text fill produced zero filled tasks; writing nothing", file=sys.stderr)
        return 1

    # Preserve existing task order, then append filled orphans in the plan
    # order to keep the merge deterministic and human-scannable.
    plan_order = {str(plan.get("id") or ""): idx for idx, plan in enumerate(plans)}
    filled_orphans.sort(key=lambda t: plan_order.get(str(t.get("id") or ""), 10**9))

    backup_path = tasks_path.with_suffix(tasks_path.suffix + ".pre-orphan-fill.bak")
    if not backup_path.exists() and tasks_path.exists():
        backup_path.write_text(tasks_path.read_text())
        print(f"  backup: {backup_path}")

    merged = existing_tasks + filled_orphans
    tasks_path.write_text(json.dumps(merged, indent=2))
    print(
        f"  wrote {tasks_path} with {len(merged)} task(s) "
        f"({len(existing_tasks)} preserved + {len(filled_orphans)} newly filled)"
    )

    # Persist text-fill diagnostics for the orphan subset; do not trample
    # the existing diagnostics file — append with a sub-section marker.
    existing_diag = _load_json_array(diag_path)
    diag_path.write_text(json.dumps(existing_diag + diagnostics, indent=2))
    print(
        f"  diagnostics appended to {diag_path} "
        f"(existing {len(existing_diag)} → {len(existing_diag) + len(diagnostics)})"
    )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("logs"))
    parser.add_argument("--text-model", type=str, default=DEFAULT_TEXT_FILL_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_TEXT_FILL_CONCURRENCY)
    parser.add_argument("--texts-per-plan", type=int, default=DEFAULT_TEXTS_PER_PLAN)
    args = parser.parse_args(argv[1:])
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
