#!/usr/bin/env bash
# Sweeps residue left by WorldSim seeding runs against dev instances.
#
# The cleanup script is a belt-and-suspenders safety net: Phase 2c and Phase 4
# both run per-task immediate cleanup through ``SeedCleanupHandle`` (see
# ``worldsim/seeding.py``). Anything this script finds means a task aborted
# after a successful editor call but before its cleanup stack ran — e.g.
# SIGINT, OOM, network blip.
#
# Scope:
# - GitLab: delete projects matching ``webagent-task-*`` OR ``webagent-verify-*``
#   under the current API user's namespace.
# - Reddit (PostMill): enumerate forums matching ``webagent-task-*`` /
#   ``webagent-verify-*`` and warn. PostMill does not expose forum deletion to
#   the regular authenticated user that our threat model permits, so the
#   sweep is informational.
# - Magento (shopping / shopping_admin): enumerate reviews whose nickname
#   starts with ``WorldSim`` and attempt DELETE via the REST admin API.
set -euo pipefail

INSTANCES_PATH="${1:-instances.json}"

uv run python - "$INSTANCES_PATH" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

from warp_taskgen.auth_tokens import acquire_tokens_for_instances
from warp_taskgen.editors.gitlab import GitlabEditor
from warp_taskgen.editors.reddit import RedditEditor

_PREFIXES = ("webagent-task-", "webagent-verify-")


def _load_instances(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("instances"), list):
        return [item for item in payload["instances"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise SystemExit(f"{path} must contain either a JSON array of instances or an object with an 'instances' list")


def _sweep_gitlab(instances: list[dict]) -> int:
    gitlab_instances = [i for i in instances if str(i.get("site_name", "")).lower() == "gitlab"]
    if not gitlab_instances:
        return 0
    total_deleted = 0
    for instance in gitlab_instances:
        with requests.Session() as session:
            editor = GitlabEditor(instance, session)
            current_user = editor._current_user()
            user_id = current_user.get("id")
            username = str(current_user.get("username") or "").strip()
            if user_id in (None, "") or not username:
                raise SystemExit(
                    f"GitLab instance {instance.get('site_url')} did not return current user identity"
                )

            for prefix in _PREFIXES:
                page = 1
                deleted_here = 0
                while True:
                    projects = editor._gitlab_request_json(
                        "GET",
                        f"/api/v4/users/{editor._quote(user_id)}/projects",
                        params={
                            "search": prefix,
                            "per_page": 100,
                            "page": page,
                            "simple": True,
                        },
                    )
                    if not isinstance(projects, list) or not projects:
                        break
                    for project in projects:
                        if not isinstance(project, dict):
                            continue
                        path_with_namespace = editor._project_path_with_namespace(project)
                        leaf = str(project.get("path") or "").strip().lower()
                        if not path_with_namespace.lower().startswith(f"{username.lower()}/"):
                            continue
                        if not leaf.startswith(prefix):
                            continue
                        project_id = project.get("id")
                        if project_id in (None, ""):
                            continue
                        editor.delete_project(project_id)
                        deleted_here += 1
                        total_deleted += 1
                        print(f"deleted gitlab project {path_with_namespace}")
                    page += 1
                if deleted_here:
                    print(
                        f"gitlab ({instance.get('site_url')}): deleted {deleted_here} project(s) matching {prefix!r}"
                    )
    return total_deleted


def _sweep_reddit(instances: list[dict]) -> int:
    """Enumerate residual PostMill forums and warn.

    PostMill does not expose forum deletion to the threat-model-permitted
    regular authenticated user. This sweep reports residue so operators can
    escalate; it does NOT attempt destructive action.
    """
    reddit_instances = [i for i in instances if str(i.get("site_name", "")).lower() == "reddit"]
    if not reddit_instances:
        return 0
    total_residue = 0
    for instance in reddit_instances:
        with requests.Session() as session:
            editor = RedditEditor(instance, session)
            try:
                response = editor._form_get("/forums")
            except Exception as exc:
                print(f"reddit ({instance.get('site_url')}): unable to enumerate forums ({exc})")
                continue
            text = response.text if response is not None else ""
            residual = []
            for prefix in _PREFIXES:
                residual.extend(
                    token
                    for token in _find_forum_slugs(text)
                    if token.startswith(prefix)
                )
            if residual:
                total_residue += len(residual)
                print(
                    f"reddit ({instance.get('site_url')}): found {len(residual)} residual forum(s): "
                    + ", ".join(sorted(set(residual)))
                )
                print(
                    "  NOTE: PostMill does not expose regular-user forum deletion. "
                    "Restore DB snapshot if cleanup is required."
                )
    return total_residue


def _find_forum_slugs(text: str) -> list[str]:
    import re

    pattern = re.compile(r"/f/([A-Za-z0-9_\-]+)")
    return [match.group(1) for match in pattern.finditer(text)]


# _sweep_magento removed 2026-04-21 with the WASP-aligned scoping decision.


def main() -> int:
    instances_path = Path(sys.argv[1]).expanduser()
    instances = _load_instances(instances_path)
    if not instances:
        print("No instances found; nothing to clean.")
        return 0

    # Acquire fresh tokens for every instance that declares one; the editors
    # resolve bearer tokens through ``_build_headers``, which relies on the
    # per-run cache populated here.
    auth_errors = acquire_tokens_for_instances(instances)
    if auth_errors:
        raise SystemExit("Could not acquire instance auth:\n" + "\n".join(auth_errors))

    gitlab_deleted = _sweep_gitlab(instances)
    reddit_residue = _sweep_reddit(instances)

    print(
        f"summary: gitlab deleted={gitlab_deleted}, "
        f"reddit_residue={reddit_residue}"
    )
    return 0


raise SystemExit(main())
PY
