"""Translate cached gitlab Mode B novel benigns from `mechanism: "api"` to
`mechanism: "none"` with `editor_calls`.

Phase 1's Mode B generator emitted gitlab benigns with raw `api_calls`. Phase 2
and Phase 4 expect `editor_calls` matching methods registered on
`worldsim/editors/gitlab.py`. This script maps each cached `(http_method,
path_template)` bucket to the corresponding editor method and rewrites the
cache in place. Idempotent: tasks already on `mechanism != "api"` are skipped.

Usage::

    uv run python scripts/migrate_phase_1_api_to_editor.py
    uv run python scripts/migrate_phase_1_api_to_editor.py --cache logs/phase_1/novel_tasks_gitlab.json --dry-run

The cache `.metadata.json` fingerprint is intentionally left stale; the next
Phase 1 ``--generate-novel`` run will detect drift and regenerate.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_CACHE = Path("logs/phase_1/novel_tasks_gitlab.json")
GITLAB_PLACEHOLDER = "__GITLAB__"

_USER_PATH = re.compile(r"^/api/v4/users/(\d+)$")
_USER_STATUS_PATH = re.compile(r"^/api/v4/users/(\d+)/status$")
_PROJECT_MILESTONES_PATH = re.compile(r"^/api/v4/projects/(\d+)/milestones$")
_PROJECT_LABELS_PATH = re.compile(r"^/api/v4/projects/(\d+)/labels$")
_SNIPPETS_PATH = re.compile(r"^/api/v4/snippets$")
_GROUP_PATH = re.compile(r"^/api/v4/groups/([^/]+)$")


class TranslationError(RuntimeError):
    pass


def _editor_call(method: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "site": "gitlab",
        "method": method,
        "args": args,
    }


def _strip_gitlab_prefix(start_url: str) -> str:
    if not start_url.startswith(f"{GITLAB_PLACEHOLDER}/"):
        raise TranslationError(
            f"start_url does not begin with {GITLAB_PLACEHOLDER}/: {start_url!r}"
        )
    return start_url[len(GITLAB_PLACEHOLDER) + 1 :]


def _project_path_from_start_url(start_url: str) -> str:
    """Extract ``namespace/project`` from a ``__GITLAB__/...-/<resource>/...`` URL."""
    tail = _strip_gitlab_prefix(start_url)
    head, sep, _ = tail.partition("/-/")
    if not sep:
        raise TranslationError(f"start_url has no '/-/' separator: {start_url!r}")
    if "/" not in head:
        raise TranslationError(f"derived project path lacks namespace: {head!r}")
    return head


def _handle_from_start_url(start_url: str) -> str:
    """Extract a single trailing handle (user or group) from ``__GITLAB__/<handle>``."""
    tail = _strip_gitlab_prefix(start_url)
    if "/" in tail:
        raise TranslationError(f"expected single-segment handle, got: {tail!r}")
    return tail


def _translate_user_bio(api_call: dict[str, Any], start_url: str) -> dict[str, Any]:
    body = api_call.get("body") or {}
    bio = body.get("bio")
    if not isinstance(bio, str):
        raise TranslationError(f"PUT users body missing string 'bio': {body!r}")
    username = _handle_from_start_url(start_url)
    return _editor_call("update_user_profile_by_id", {"username": username, "bio": bio})


def _translate_user_status(api_call: dict[str, Any], start_url: str) -> dict[str, Any]:
    body = api_call.get("body") or {}
    message = body.get("message")
    if not isinstance(message, str):
        raise TranslationError(f"PUT users/status body missing string 'message': {body!r}")
    username = _handle_from_start_url(start_url)
    args: dict[str, Any] = {"username": username, "message": message}
    emoji = body.get("emoji")
    if isinstance(emoji, str) and emoji:
        args["emoji"] = emoji
    return _editor_call("update_user_status_by_id", args)


def _translate_milestone_create_to_update(
    api_call: dict[str, Any], start_url: str
) -> dict[str, Any]:
    """No `create_milestone` editor exists. The cached `start_urls` already point
    at an existing `/-/milestones/<iid>` (typically iid=1 on r5), so we translate
    the create-style POST into an update against that iid. The cached `title` is
    dropped because `update_milestone` has no title binding.
    """
    body = api_call.get("body") or {}
    description = body.get("description")
    if not isinstance(description, str):
        raise TranslationError(f"POST milestones body missing string 'description': {body!r}")
    project_path = _project_path_from_start_url(start_url)
    # Derive milestone_iid from the start_url tail: ".../-/milestones/<iid>"
    suffix_match = re.search(r"/-/milestones/(\d+)(?:/|$)", start_url)
    if not suffix_match:
        raise TranslationError(
            f"start_url has no '/-/milestones/<iid>' suffix to anchor update: {start_url!r}"
        )
    milestone_iid = suffix_match.group(1)
    return _editor_call(
        "update_milestone",
        {
            "project_path_template": project_path,
            "milestone_iid": milestone_iid,
            "description": description,
        },
    )


def _translate_label_create(api_call: dict[str, Any], start_url: str) -> dict[str, Any]:
    body = api_call.get("body") or {}
    name = body.get("name")
    if not isinstance(name, str):
        raise TranslationError(f"POST labels body missing string 'name': {body!r}")
    project_path = _project_path_from_start_url(start_url)
    args: dict[str, Any] = {
        "project_path_template": project_path,
        "name": name,
    }
    description = body.get("description")
    if isinstance(description, str):
        args["description"] = description
    color = body.get("color")
    if isinstance(color, str) and color:
        args["color"] = color
    return _editor_call("create_label", args)


def _translate_snippet_create(api_call: dict[str, Any], start_url: str) -> dict[str, Any]:
    body = api_call.get("body") or {}
    title = body.get("title")
    content = body.get("content")
    if not isinstance(title, str):
        raise TranslationError(f"POST snippets body missing string 'title': {body!r}")
    if not isinstance(content, str):
        raise TranslationError(f"POST snippets body missing string 'content': {body!r}")
    args: dict[str, Any] = {"title": title, "content": content}
    file_name = body.get("file_name")
    if isinstance(file_name, str) and file_name:
        args["file_name"] = file_name
    # `visibility` is dropped: create_snippet hard-codes visibility="public".
    return _editor_call("create_snippet", args)


def _translate_group_update(api_call: dict[str, Any], start_url: str) -> dict[str, Any]:
    body = api_call.get("body") or {}
    description = body.get("description")
    if not isinstance(description, str):
        raise TranslationError(f"PUT groups body missing string 'description': {body!r}")
    group_path = _handle_from_start_url(start_url)
    return _editor_call(
        "update_group",
        {"group_path": group_path, "description": description},
    )


def _translate_api_call(api_call: dict[str, Any], start_url: str) -> dict[str, Any]:
    method = api_call.get("method")
    path = api_call.get("path")
    if not isinstance(method, str) or not isinstance(path, str):
        raise TranslationError(f"api_call missing string method/path: {api_call!r}")

    if method == "PUT" and _USER_PATH.match(path):
        return _translate_user_bio(api_call, start_url)
    if method == "PUT" and _USER_STATUS_PATH.match(path):
        return _translate_user_status(api_call, start_url)
    if method == "POST" and _PROJECT_MILESTONES_PATH.match(path):
        return _translate_milestone_create_to_update(api_call, start_url)
    if method == "POST" and _PROJECT_LABELS_PATH.match(path):
        return _translate_label_create(api_call, start_url)
    if method == "POST" and _SNIPPETS_PATH.match(path):
        return _translate_snippet_create(api_call, start_url)
    if method == "PUT" and _GROUP_PATH.match(path):
        return _translate_group_update(api_call, start_url)

    raise TranslationError(f"no translation rule for {method} {path}")


def _translate_task(task: dict[str, Any]) -> tuple[bool, str]:
    """Mutate ``task`` in place. Returns ``(changed, bucket_label)``."""
    seed = task.get("data_seed")
    if not isinstance(seed, dict):
        return False, "no-data-seed"
    if seed.get("mechanism") != "api":
        return False, "skip-non-api"

    api_calls = seed.get("api_calls")
    if not isinstance(api_calls, list) or not api_calls:
        raise TranslationError(f"task {task.get('id')!r} has empty api_calls")
    if len(api_calls) > 1:
        raise TranslationError(
            f"task {task.get('id')!r} has multi-call api seed; out of scope for this translator"
        )

    start_urls = task.get("start_urls") or []
    if not start_urls or not isinstance(start_urls[0], str):
        raise TranslationError(f"task {task.get('id')!r} missing start_urls[0]")
    start_url = start_urls[0]

    api_call = api_calls[0]
    editor_call = _translate_api_call(api_call, start_url)

    task["data_seed"] = {
        "mechanism": "none",
        "editor_calls": [editor_call],
    }
    bucket = f"{api_call.get('method')} {api_call.get('path')} -> {editor_call['method']}"
    return True, bucket


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Path to novel_tasks_gitlab.json (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Translate in memory and report counts without writing the cache.",
    )
    args = parser.parse_args()

    cache_path: Path = args.cache
    if not cache_path.exists():
        raise SystemExit(f"cache file not found: {cache_path}")

    raw = json.loads(cache_path.read_text())
    if not isinstance(raw, list):
        raise SystemExit(f"cache root must be a list, got {type(raw).__name__}")

    bucket_counts: Counter[str] = Counter()
    translated = 0
    skipped = 0
    failures: list[str] = []

    for task in raw:
        if not isinstance(task, dict):
            failures.append(f"non-dict task entry: {type(task).__name__}")
            continue
        try:
            changed, bucket = _translate_task(task)
        except TranslationError as exc:
            failures.append(f"{task.get('id')!r}: {exc}")
            continue
        bucket_counts[bucket] += 1
        if changed:
            translated += 1
        else:
            skipped += 1

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} translation failure(s); cache not written")

    mechanisms_after = sorted({task.get("data_seed", {}).get("mechanism") for task in raw})
    print(f"translated: {translated}; skipped: {skipped}; total: {len(raw)}")
    print(f"mechanisms after: {mechanisms_after}")
    print("buckets:")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {count:>3}  {bucket}")

    if args.dry_run:
        print("--dry-run set; cache not written")
        return 0

    cache_path.write_text(json.dumps(raw, indent=2) + "\n")
    print(f"wrote: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
