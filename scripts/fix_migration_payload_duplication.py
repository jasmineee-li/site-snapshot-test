#!/usr/bin/env python3
"""Remove duplicate {{PAYLOAD_TEXT}} placeholders left by migrate_phase_2_seeds_to_targets.

During migration, calls that already had a target.create spec still retained their
legacy top-level body / body_form field verbatim. When that field contained
{{PAYLOAD_TEXT}}, the same placeholder appeared in both the legacy body and the
resolver-consumed target.create spec, triggering:

    seed_template must contain exactly one {{PAYLOAD_TEXT}} placeholder

The authoritative location per resource type is:
  - shopping/product_review, shopping_admin/product_review:
      target.create.product_review  (shopping.py resolver reads this via create_spec)
      => redundant: top-level body
  - reddit/submission:
      target.create.submission.title_template  (reddit resolver reads create_spec)
      => redundant: top-level body_form
  - gitlab/issue:
      target.create.issue.body_template  (gitlab.py _ensure_issue reads create_spec)
      => redundant: top-level body
  - gitlab/mr:
      target.create.mr.body_template  (_prepare_mr_payload prefers mr_spec over body)
      => redundant: top-level body

This script removes the placeholder from the redundant location only. It is
idempotent: re-running when all counts are already 1 is a no-op.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("logs/phase_2/adversarial_tasks.json")
PLACEHOLDER = "{{PAYLOAD_TEXT}}"
_PLACEHOLDER_RE = re.compile(r"\{\{PAYLOAD_TEXT\}\}")


def _placeholder_count(value: Any) -> int:
    return len(_PLACEHOLDER_RE.findall(json.dumps(value)))


def _strip_placeholder_from_body(body: Any) -> Any:
    """Return a copy of body with all {{PAYLOAD_TEXT}} occurrences removed."""
    if isinstance(body, dict):
        return {k: _strip_placeholder_from_body(v) for k, v in body.items()}
    if isinstance(body, list):
        return [_strip_placeholder_from_body(item) for item in body]
    if isinstance(body, str):
        return body.replace(PLACEHOLDER, "")
    return body


def _fix_call(call: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Fix a single api_call dict.

    Returns (fixed_call, was_changed).
    """
    target = call.get("target")
    if not isinstance(target, dict):
        # Legacy call without a target block — nothing to deduplicate.
        return call, False

    create_spec = target.get("create")
    if not isinstance(create_spec, dict):
        return call, False

    resource_type = str(target.get("resource_type", "")).strip().lower()

    # Determine which top-level body key is redundant for this resource type.
    if resource_type in ("product_review",):
        redundant_key = "body"
    elif resource_type in ("issue", "mr"):
        redundant_key = "body"
    elif resource_type == "submission":
        redundant_key = "body_form"
    else:
        return call, False

    redundant_value = call.get(redundant_key)
    if redundant_value is None:
        return call, False

    # Only act if the redundant field actually contains the placeholder.
    if _placeholder_count(redundant_value) == 0:
        return call, False

    # Confirm the authoritative location (target.create) also has the placeholder
    # so we're not removing the only copy.
    if _placeholder_count(create_spec) == 0:
        return call, False

    # Strip placeholder from the redundant body / body_form.
    fixed_call = dict(call)
    fixed_call[redundant_key] = _strip_placeholder_from_body(redundant_value)
    return fixed_call, True


def _fix_seed(seed: Any) -> tuple[Any, int]:
    """Fix a seed dict (seed_template or adversarial_data_seed).

    Returns (fixed_seed, calls_changed).
    """
    if not isinstance(seed, dict):
        return seed, 0
    api_calls = seed.get("api_calls")
    if not isinstance(api_calls, list):
        return seed, 0

    fixed_calls = []
    calls_changed = 0
    for call in api_calls:
        if isinstance(call, dict):
            fixed_call, changed = _fix_call(call)
            fixed_calls.append(fixed_call)
            if changed:
                calls_changed += 1
        else:
            fixed_calls.append(call)

    if calls_changed == 0:
        return seed, 0

    fixed_seed = dict(seed)
    fixed_seed["api_calls"] = fixed_calls
    return fixed_seed, calls_changed


def _fix_task(task: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Fix a single task. Returns (fixed_task, was_changed)."""
    changed = False
    fixed = dict(task)
    for seed_key in ("seed_template", "adversarial_data_seed"):
        seed = task.get(seed_key)
        if not isinstance(seed, dict):
            continue
        fixed_seed, n = _fix_seed(seed)
        if n > 0:
            fixed[seed_key] = fixed_seed
            changed = True
    return fixed, changed


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="Path to the Phase 2 adversarial task dataset.",
    )
    args = parser.parse_args()

    path: Path = args.path
    tasks: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit(f"{path} must contain a JSON list of tasks")

    normalized = 0
    skipped = 0

    fixed_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            fixed_tasks.append(task)
            skipped += 1
            continue

        task_id = task.get("id", "?")
        before = _placeholder_count(
            {k: task[k] for k in ("seed_template", "adversarial_data_seed") if k in task}
        )

        fixed_task, changed = _fix_task(task)
        fixed_tasks.append(fixed_task)

        if changed:
            after = _placeholder_count(
                {
                    k: fixed_task[k]
                    for k in ("seed_template", "adversarial_data_seed")
                    if k in fixed_task
                }
            )
            print(f"  fixed {task_id}: {{{{PAYLOAD_TEXT}}}} count {before} -> {after}")
            normalized += 1
        else:
            skipped += 1

    if normalized == 0:
        print("fix_migration_payload_duplication: already clean, nothing to do")
        print(f"Normalized 0 tasks; skipped {skipped} already-clean tasks.")
        return 0

    # Write back in place (git history is the backup).
    path.write_text(json.dumps(fixed_tasks, indent=2) + "\n", encoding="utf-8")
    print(f"Normalized {normalized} tasks; skipped {skipped} already-clean tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
