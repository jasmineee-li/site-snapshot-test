"""One-shot patch: rewrite ``start_url_resolved`` in materialised
Phase 2 tasks from anchors, so Phase 2c probes navigate to the concrete
entity where the seed was planted (not the benign task's stale
``start_urls[0]`` template).

Mirrors the logic in
``worldsim.phase_2.target_resolution.reconstruction._reconstruct_start_url_from_anchors``
but runs against already-emitted
``adversarial_tasks.json`` records so we do not have to re-run the
expensive Phase 2a sandbox pass. Idempotent: rerunning leaves already-
reconstructed URLs unchanged.

Usage::

    uv run python scripts/patch_benign_target_resource_urls.py [path]

Defaults to ``logs/phase_2/adversarial_tasks.json``. Writes back in
place and prints a one-line summary with counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from worldsim.phase_2.target_resolution.constants import PHASE_2A_SYNTHETIC_PLACEHOLDERS
from worldsim.phase_2.target_resolution.reconstruction import _reconstruct_start_url_from_anchors


def _site_kind(task: dict) -> str | None:
    sites = task.get("sites") or []
    for site in sites:
        lower = str(site).strip().lower()
        if lower == "gitlab":
            return "gitlab"
        if lower == "reddit":
            return "reddit"
    primary = str(task.get("site") or "").strip().lower()
    return primary if primary in {"gitlab", "reddit"} else None


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "logs/phase_2/adversarial_tasks.json")
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    tasks = json.loads(path.read_text())
    if not isinstance(tasks, list):
        print(f"expected list, got {type(tasks).__name__}", file=sys.stderr)
        return 2

    updated = 0
    skipped = 0
    for task in tasks:
        resource = task.get("benign_target_resource") or {}
        kind = str(resource.get("kind") or "")
        anchors = resource.get("anchors") or {}
        site_kind = _site_kind(task)
        if not kind or not site_kind:
            skipped += 1
            continue
        reconstructed = _reconstruct_start_url_from_anchors(
            site_kind, kind, anchors, PHASE_2A_SYNTHETIC_PLACEHOLDERS
        )
        if reconstructed and reconstructed != resource.get("start_url_resolved"):
            resource["start_url_resolved"] = reconstructed
            task["benign_target_resource"] = resource
            updated += 1
        else:
            skipped += 1

    path.write_text(json.dumps(tasks, indent=2, sort_keys=True))
    print(
        f"patched {updated} / {len(tasks)} records in {path}; "
        f"{skipped} unchanged (no reconstruction available or already reconstructed)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
