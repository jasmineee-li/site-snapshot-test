#!/usr/bin/env python3
"""Provision and declare disposable GitLab Tier 3 fixture repositories.

This is a host-ops helper for r5-style execution-local instance files. It does
not touch Phase 1/2 artifacts. It makes the host-local ``instances.scale.json``
declare a fixture pool only after every GitLab replica can prove a canary
write/read/delete round trip against the configured repository path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import requests

from worldsim.adversarial_actions.tier3_fixtures import verify_gitlab_repository_fixture_pool
from worldsim.editors.gitlab import GitlabEditor

DEFAULT_PROJECT = "byteblaze/worldsim-tier3-fixture-01"
DEFAULT_PREFIX = "worldsim-fixtures"
DEFAULT_SCOPE = "worldsim_disposable"


class ProvisionError(RuntimeError):
    pass


def build_fixture_config(
    *,
    project_paths: list[str],
    file_path_prefix: str = DEFAULT_PREFIX,
    scope: str = DEFAULT_SCOPE,
) -> dict[str, Any]:
    projects = [{"project_path": path} for path in project_paths]
    return {
        "gitlab": {
            "repository_content": {
                "scope": scope,
                "file_path_prefix": file_path_prefix,
                "projects": projects,
            }
        }
    }


def apply_fixture_config(document: dict[str, Any], fixture_config: Mapping[str, Any]) -> None:
    existing = document.get("tier3_fixtures")
    if isinstance(existing, Mapping):
        merged = deepcopy(dict(existing))
    else:
        merged = {}
    merged["gitlab"] = deepcopy(dict(fixture_config["gitlab"]))
    document["tier3_fixtures"] = merged


def provision_document(
    document: dict[str, Any],
    *,
    project_paths: list[str],
    file_path_prefix: str = DEFAULT_PREFIX,
    scope: str = DEFAULT_SCOPE,
    create_missing: bool = True,
) -> dict[str, Any]:
    instances = _instances(document)
    gitlab_instances = [
        inst for inst in instances if str(inst.get("site_name") or "").strip().lower() == "gitlab"
    ]
    if not gitlab_instances:
        raise ProvisionError("instance file contains no gitlab instances")

    fixture_config = build_fixture_config(
        project_paths=project_paths,
        file_path_prefix=file_path_prefix,
        scope=scope,
    )
    reports: list[dict[str, Any]] = []
    for index, instance in enumerate(gitlab_instances):
        _ensure_projects_on_instance(
            instance,
            project_paths=project_paths,
            create_missing=create_missing,
        )
        probe_instance = dict(instance)
        probe_instance["tier3_fixtures"] = fixture_config
        fixtures, report = verify_gitlab_repository_fixture_pool(probe_instance)
        report = dict(report)
        report["replica_index"] = instance.get("replica_index", index)
        report["site_url"] = instance.get("site_url")
        reports.append(report)
        if not fixtures:
            raise ProvisionError(
                "fixture canary verification failed for "
                f"{instance.get('site_url')}: {json.dumps(report, sort_keys=True)}"
            )

    apply_fixture_config(document, fixture_config)
    return {
        "status": "ready",
        "gitlab_replicas": len(gitlab_instances),
        "project_paths": project_paths,
        "file_path_prefix": file_path_prefix,
        "scope": scope,
        "replica_reports": reports,
    }


def _instances(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    instances = document.get("instances")
    if not isinstance(instances, list):
        return []
    return [dict(item) for item in instances if isinstance(item, Mapping)]


def _ensure_projects_on_instance(
    instance: Mapping[str, Any],
    *,
    project_paths: list[str],
    create_missing: bool,
) -> None:
    with requests.Session() as session:
        editor = GitlabEditor(dict(instance), session)
        current_user = editor._current_user()
        username = str(current_user.get("username") or "").strip()
        if not username:
            raise ProvisionError(f"could not determine current GitLab user for {instance}")
        for project_path in project_paths:
            _ensure_one_project(
                editor,
                project_path=project_path,
                username=username,
                create_missing=create_missing,
            )


def _ensure_one_project(
    editor: GitlabEditor,
    *,
    project_path: str,
    username: str,
    create_missing: bool,
) -> None:
    project_path = project_path.strip().strip("/")
    if "/" not in project_path:
        raise ProvisionError(f"fixture project path must be namespace-qualified: {project_path!r}")
    if not project_path.lower().startswith(f"{username.lower()}/"):
        raise ProvisionError(
            "fixture project path must live under the authenticated disposable namespace "
            f"{username!r}: {project_path!r}"
        )
    project = editor._gitlab_get_json(
        f"/api/v4/projects/{editor._quote(project_path)}",
        allow_missing=True,
    )
    if isinstance(project, Mapping):
        return
    if not create_missing:
        raise ProvisionError(f"fixture project does not exist: {project_path}")
    leaf = project_path.rsplit("/", 1)[-1]
    created = editor._gitlab_request_json(
        "POST",
        "/api/v4/projects",
        json_body={
            "name": leaf,
            "path": leaf,
            "description": "WorldSim disposable Tier 3 repository-content fixture.",
            "initialize_with_readme": True,
            "visibility": "private",
        },
    )
    resolved = str((created or {}).get("path_with_namespace") or "").strip("/")
    if resolved.lower() != project_path.lower():
        raise ProvisionError(
            f"created fixture project path mismatch: expected {project_path!r}, got {resolved!r}"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", default="instances.scale.json", help="instance JSON path")
    parser.add_argument(
        "--project-path",
        action="append",
        default=[],
        help=f"namespace-qualified fixture repo path (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument("--file-path-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--no-create", action="store_true", help="fail if a fixture repo is missing")
    parser.add_argument("--dry-run", action="store_true", help="verify and print without writing")
    parser.add_argument("--no-backup", action="store_true", help="do not write <instances>.bak")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    instances_path = Path(args.instances)
    try:
        document = json.loads(instances_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ProvisionError("instances file must be a JSON object")
        report = provision_document(
            document,
            project_paths=args.project_path or [DEFAULT_PROJECT],
            file_path_prefix=args.file_path_prefix,
            scope=args.scope,
            create_missing=not args.no_create,
        )
        if args.dry_run:
            print(json.dumps({"report": report, "document": document}, indent=2))
            return 0
        if not args.no_backup:
            backup_path = instances_path.with_suffix(instances_path.suffix + ".bak")
            backup_path.write_text(instances_path.read_text(encoding="utf-8"), encoding="utf-8")
        _write_json(instances_path, document)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"provision_tier3_gitlab_fixtures failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
