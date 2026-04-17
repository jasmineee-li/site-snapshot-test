#!/usr/bin/env bash
set -euo pipefail

INSTANCES_PATH="${1:-instances.json}"

uv run python - "$INSTANCES_PATH" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.editors.gitlab import GitlabEditor


def main() -> int:
    instances_path = Path(sys.argv[1]).expanduser()
    payload = json.loads(instances_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"{instances_path} must contain a JSON array of instances")

    gitlab_instances = [
        instance
        for instance in payload
        if isinstance(instance, dict) and str(instance.get("site_name") or "").strip().lower() == "gitlab"
    ]
    if not gitlab_instances:
        print("No gitlab instances found; nothing to clean.")
        return 0

    auth_errors = acquire_tokens_for_instances(gitlab_instances)
    if auth_errors:
        raise SystemExit("Could not acquire instance auth:\n" + "\n".join(auth_errors))

    total_deleted = 0
    for instance in gitlab_instances:
        with requests.Session() as session:
            editor = GitlabEditor(instance, session)
            current_user = editor._current_user()
            user_id = current_user.get("id")
            username = str(current_user.get("username") or "").strip()
            if user_id in (None, "") or not username:
                raise SystemExit(f"GitLab instance {instance.get('site_url')} did not return current user identity")

            page = 1
            deleted_here = 0
            while True:
                projects = editor._gitlab_request_json(
                    "GET",
                    f"/api/v4/users/{editor._quote(user_id)}/projects",
                    params={
                        "search": "webagent-task-",
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
                    if not leaf.startswith("webagent-task-"):
                        continue
                    project_id = project.get("id")
                    if project_id in (None, ""):
                        continue
                    editor.delete_project(project_id)
                    deleted_here += 1
                    total_deleted += 1
                    print(f"deleted gitlab project {path_with_namespace}")
                page += 1
            print(
                f"gitlab cleanup complete for {instance.get('site_url')}: deleted {deleted_here} project(s)"
            )

    print(f"total deleted: {total_deleted}")
    return 0


raise SystemExit(main())
PY
