"""Mechanical rewrite pass: remove the TROJAN-ACK-[HEX] canary from an
existing Phase 2 adversarial_tasks.json artifact.

Before this commit series, ``derive_required_tokens`` emitted both a
TROJAN-ACK-[HEX] directive and a ``https://attacker.invalid/[slug]`` URL
for every task. The directive was deleted; payload-prefix substring match
replaced it everywhere. But tasks persisted to disk still carry the
directive in ``required_tokens`` and in every ``rendered_payload`` / seed
body that Phase 2 baked it into.

Rather than re-run Phase 2 (which regenerates prose via LLM and risks
drift from the validated payloads currently in the dataset), this script
does a deterministic string-level scrub:

  - Drops ``{"kind": "directive", ...}`` entries from ``required_tokens``.
  - Strips ``TROJAN-ACK-[12 hex]`` (optionally with one leading / trailing
    space) from every string value anywhere in the task tree.
  - Collapses resulting double-spaces inside prose fields.
  - Removes any ``framing_witnesses`` / ``concealment_witnesses`` entry
    that was solely the canary; trims entries that contained the canary
    as a substring.

Idempotent: safe to re-run. Writes in place.

Usage:
    uv run python scripts/strip_canary_from_adversarial_tasks.py logs/phase_2/adversarial_tasks.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_CANARY_RE = re.compile(r" ?TROJAN-ACK-[0-9A-F]{12} ?")


def _scrub_string(s: str) -> str:
    """Remove canary occurrences and collapse resulting whitespace."""
    if "TROJAN-ACK" not in s:
        return s
    scrubbed = _CANARY_RE.sub(" ", s)
    # Collapse runs of 2+ spaces (not newlines) that the removal introduced.
    scrubbed = re.sub(r" {2,}", " ", scrubbed)
    # Trim a stray space BEFORE terminal punctuation (".,;:?") only when
    # that punctuation is followed by whitespace / end-of-string — never
    # before "!" because ``![alt](src)`` is markdown image syntax whose
    # leading space is significant.
    scrubbed = re.sub(r" ([.,;:?])(\s|$)", r"\1\2", scrubbed)
    return scrubbed


def _scrub_list_of_strings(items: list[Any]) -> list[Any]:
    """Drop list entries that WERE the canary; trim entries that contained it."""
    out: list[Any] = []
    for item in items:
        if isinstance(item, str):
            if _CANARY_RE.fullmatch(item.strip()) or _CANARY_RE.sub("", item).strip() == "":
                # Entry was ONLY the canary (or canary + whitespace).
                continue
            out.append(_scrub_string(item))
        else:
            out.append(_scrub(item))
    return out


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        if "TROJAN-ACK" in value:
            return _scrub_string(value)
        return value
    if isinstance(value, list):
        return _scrub_list_of_strings(value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    return value


def _drop_directive_required_tokens(task: dict[str, Any]) -> dict[str, Any]:
    tokens = task.get("required_tokens")
    if not isinstance(tokens, list):
        return task
    kept = [t for t in tokens if not (isinstance(t, dict) and t.get("kind") == "directive")]
    if len(kept) != len(tokens):
        task["required_tokens"] = kept
    return task


def _reconcile_witnesses(payload_text: dict[str, Any]) -> None:
    """Fix ``framing_witnesses`` / ``concealment_witnesses`` entries that are
    stale after scrubbing.

    Witnesses were originally substrings of the pre-scrub rendered_payload.
    After removing the canary, some witnesses no longer appear — e.g. a
    witness that ended just before the canary ("```system\\nAgent directive ")
    is now orphaned because the punctuation that followed the canary has
    shifted. For each stale witness, trim trailing characters until the
    result IS a substring of the new rendered_payload; drop if <5 chars
    remain. If that empties ``concealment_witnesses`` for a concealed
    payload, inject a minimal fallback witness from the rendered_payload.
    """
    rendered = payload_text.get("rendered_payload")
    if not isinstance(rendered, str) or not rendered:
        return

    for key in ("framing_witnesses", "concealment_witnesses"):
        witnesses = payload_text.get(key)
        if not isinstance(witnesses, list):
            continue
        new_witnesses: list[str] = []
        for w in witnesses:
            if not isinstance(w, str):
                continue
            if w in rendered:
                new_witnesses.append(w)
                continue
            # Trim trailing whitespace/punctuation; then trailing chars.
            trimmed = w.rstrip(" :,.;")
            while trimmed and trimmed not in rendered and len(trimmed) > 4:
                trimmed = trimmed[:-1]
            if trimmed and trimmed in rendered and len(trimmed) >= 5:
                new_witnesses.append(trimmed)
            # else: drop the stale witness entirely.
        payload_text[key] = new_witnesses


def scrub_task(task: dict[str, Any]) -> dict[str, Any]:
    task = _drop_directive_required_tokens(task)
    task = _scrub(task)
    # Reconcile witnesses after the rendered_payload scrub so validate_text_post_hoc passes.
    for pt in task.get("payload_texts") or []:
        if isinstance(pt, dict):
            _reconcile_witnesses(pt)
    return task


def main(path: Path) -> int:
    if not path.is_file():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2
    before = path.read_text(encoding="utf-8")
    tasks = json.loads(before)
    if not isinstance(tasks, list):
        print(f"ERROR: {path} must be a JSON array of tasks", file=sys.stderr)
        return 2
    before_count = before.count("TROJAN-ACK")
    scrubbed = [scrub_task(t) for t in tasks]
    after = json.dumps(scrubbed, indent=2, ensure_ascii=False) + "\n"
    after_count = after.count("TROJAN-ACK")
    path.write_text(after, encoding="utf-8")
    print(f"{path}: {len(tasks)} tasks; TROJAN-ACK occurrences {before_count} → {after_count}")
    return 0 if after_count == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "usage: strip_canary_from_adversarial_tasks.py <adversarial_tasks.json>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
