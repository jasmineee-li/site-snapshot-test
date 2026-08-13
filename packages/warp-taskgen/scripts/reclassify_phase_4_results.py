"""Offline reclassifier for Phase 4 ``processed_result.json`` files.

Reads trajectory artifacts already on disk and enriches each
``processed_result.json`` with the new outcome-taxonomy fields
(``outcome_fine``, ``flags``, ``diagnosable_confidence``, ``signals``,
``classifier_version``, ``classifier_rationale``).

Idempotent: skips files whose ``classifier_version`` already matches
unless ``--force`` is passed.

Usage::

    uv run python scripts/reclassify_phase_4_results.py <run_dir> [--force] \\
        [--tasks PATH [PATH ...]]

``<run_dir>`` is a directory like ``logs/phase_4/20260417_190423/``.
Each immediate subdirectory is treated as one task's trajectory dir.

Task metadata is resolved in priority order:
  1. Embedded under ``placement_fix.final_task`` or
     ``ecological_validity_fix.final_task`` in the processed_result itself.
  2. One of the ``--tasks`` JSON files (list of task dicts with ``id``).

Canonical derivation: ``docs/handoffs/codex-handoff-outcome-taxonomy.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from warp_taskgen import outcome_taxonomy

logger = logging.getLogger("reclassify_phase_4")


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _load_task_lookup(paths: list[Path]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for p in paths:
        try:
            data = _load_json(p)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read tasks file %s: %s", p, exc)
            continue
        items: list[Any]
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            maybe = data.get("tasks")
            items = maybe if isinstance(maybe, list) else [data]
        else:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            tid = item.get("id")
            if isinstance(tid, str) and tid:
                lookup[tid] = item
    return lookup


def _resolve_task(
    processed: dict[str, Any], task_lookup: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    placement = processed.get("placement_fix")
    if isinstance(placement, dict):
        ft = placement.get("final_task")
        if isinstance(ft, dict):
            return ft
    ecoval = processed.get("ecological_validity_fix")
    if isinstance(ecoval, dict):
        ft = ecoval.get("final_task")
        if isinstance(ft, dict):
            return ft
    tid = str(processed.get("task_id", ""))
    if tid and tid in task_lookup:
        return task_lookup[tid]
    return None


def _is_trajectory_outcome(processed: dict[str, Any]) -> bool:
    outcome = processed.get("outcome")
    non_trajectory = {
        "seed_preflight_mismatch",
        "seed_error",
        "configuration_error",
        "unknown_task",
    }
    if outcome in non_trajectory:
        return False
    # ``error`` is ambiguous — some ``error`` outcomes do have a trajectory
    # (agent ran but the reward function raised). Let the on-disk history
    # file be the authority.
    return True


def reclassify_one(
    task_dir: Path,
    *,
    task_lookup: dict[str, dict[str, Any]],
    force: bool,
) -> dict[str, str]:
    """Reclassify one task dir in place. Returns a short status dict."""
    processed_path = task_dir / "processed_result.json"
    if not processed_path.exists():
        return {"task_id": task_dir.name, "status": "skipped_no_processed"}

    try:
        processed = _load_json(processed_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"task_id": task_dir.name, "status": f"error_read: {exc}"}
    if not isinstance(processed, dict):
        return {"task_id": task_dir.name, "status": "error_not_object"}

    if not force and processed.get("classifier_version") == outcome_taxonomy.CLASSIFIER_VERSION:
        return {
            "task_id": str(processed.get("task_id", task_dir.name)),
            "status": "skipped_up_to_date",
            "outcome_fine": str(processed.get("outcome_fine", "")),
        }

    if not _is_trajectory_outcome(processed):
        return {
            "task_id": str(processed.get("task_id", task_dir.name)),
            "status": f"skipped_non_trajectory_{processed.get('outcome')}",
        }

    trajectory_dir = _resolve_trajectory_dir(processed, task_dir)
    if not (trajectory_dir / "history.json").exists():
        return {
            "task_id": str(processed.get("task_id", task_dir.name)),
            "status": "skipped_no_history",
        }

    task = _resolve_task(processed, task_lookup)
    if task is None:
        return {
            "task_id": str(processed.get("task_id", task_dir.name)),
            "status": "skipped_no_task_metadata",
        }

    # ecologically_valid is now derived from encounter.max_coverage inside
    # extract_signals when left None. We still pass legacy values from older
    # result.json artefacts so pre-migration runs reclassify consistently
    # with what they recorded at the time.
    legacy_eco = processed.get("ecologically_valid")
    try:
        classified = outcome_taxonomy.classify_from_dir(
            trajectory_dir,
            task,
            benign_passed=processed.get("benign_passed"),
            adversarial_passed=processed.get("adversarial_passed"),
            ecologically_valid=legacy_eco if isinstance(legacy_eco, bool) else None,
            seed_ok=True,
        )
    except Exception as exc:  # defensive
        return {
            "task_id": str(processed.get("task_id", task_dir.name)),
            "status": f"error_classify: {exc}",
        }

    merged = dict(processed)
    merged.update(outcome_taxonomy.as_result_fields(classified))
    tmp = processed_path.with_suffix(processed_path.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(processed_path)
    return {
        "task_id": str(processed.get("task_id", task_dir.name)),
        "status": "classified",
        "outcome_fine": classified.outcome_fine,
        "confidence": classified.diagnosable_confidence,
    }


def _resolve_trajectory_dir(processed: dict[str, Any], task_dir: Path) -> Path:
    """Resolve ``processed_result.trajectory_dir`` against disk.

    Absolute paths are used as-is when they exist. Relative paths are
    resolved against, in order: the run dir (``task_dir.parent``), the
    current working directory, and the task dir itself. Fix-loop
    trajectories (``__placement_1``, ``__ecoval_1``) live as siblings of
    the task dir, so the run dir is the most useful root.
    """
    tdir = processed.get("trajectory_dir")
    if not isinstance(tdir, str) or not tdir:
        return task_dir
    candidate = Path(tdir)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        logger.warning(
            "trajectory_dir %s is absolute but does not exist; falling back to task_dir",
            tdir,
        )
        return task_dir
    for root in (task_dir.parent, Path.cwd(), task_dir):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    logger.warning(
        "trajectory_dir %s is relative and unresolvable; falling back to task_dir",
        tdir,
    )
    return task_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reclassify Phase 4 processed_result.json files with the outcome-taxonomy classifier."
    )
    parser.add_argument(
        "run_dir", type=Path, help="Phase 4 run directory (e.g. logs/phase_4/<run_id>/)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-classify even if classifier_version matches"
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        nargs="*",
        default=[],
        help="Optional JSON file(s) holding task dicts (list or {'tasks': [...]}). Used as fallback when task metadata is not embedded in processed_result.json.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    if not args.run_dir.exists() or not args.run_dir.is_dir():
        logger.error("run_dir does not exist or is not a directory: %s", args.run_dir)
        return 2

    task_lookup = _load_task_lookup(list(args.tasks))
    task_dirs = sorted(p for p in args.run_dir.iterdir() if p.is_dir())

    classified = 0
    skipped = 0
    errors = 0
    for d in task_dirs:
        report = reclassify_one(d, task_lookup=task_lookup, force=args.force)
        status = report["status"]
        if status == "classified":
            classified += 1
            logger.info(
                "%s: %s (confidence=%s)",
                report["task_id"],
                report.get("outcome_fine", ""),
                report.get("confidence", ""),
            )
        elif status.startswith("error"):
            errors += 1
            logger.warning("%s: %s", report["task_id"], status)
        else:
            skipped += 1
            logger.info("%s: %s", report["task_id"], status)

    logger.info(
        "Reclassified %d tasks; skipped %d; errors %d (run_dir=%s)",
        classified,
        skipped,
        errors,
        args.run_dir,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
