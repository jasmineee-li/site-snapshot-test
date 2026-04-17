#!/usr/bin/env python3
"""Rewrite legacy Phase 2 seed calls to target.create|target.update and quarantine map."""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

DEFAULT_QUARANTINE_SITES = ("map",)
_EMBEDDED_SECRET_PATTERNS = (
    (
        re.compile(r"(?i)\b(Bearer)\s+([^\s'\"`]+)"),
        r"\1 <redacted>",
    ),
    (
        re.compile(r"(?i)(set to ['\"])([^'\"]+)(['\"])"),
        r"\1<redacted>\3",
    ),
    (
        re.compile(r"(?i)(Credentials?\s*\()([^)]+)(\))"),
        r"\1<redacted>\3",
    ),
    (
        re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}:[^'\"\s,)]+"),
        "<redacted>",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("logs/phase_2/adversarial_tasks.json"),
        help="Path to the Phase 2 adversarial task dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the migration summary without writing changes.",
    )
    parser.add_argument(
        "--quarantine-sites",
        default=",".join(DEFAULT_QUARANTINE_SITES),
        help="Comma-separated site names to quarantine out of the main dataset.",
    )
    args = parser.parse_args()

    tasks = json.loads(args.path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit(f"{args.path} must contain a JSON list of tasks")

    quarantine_sites = {
        site.strip().lower() for site in str(args.quarantine_sites).split(",") if site.strip()
    }

    migrated_main: list[dict[str, Any]] = []
    quarantined_by_site: dict[str, list[dict[str, Any]]] = {site: [] for site in quarantine_sites}
    changed = 0
    skipped = 0
    unmatched = 0
    quarantined = 0

    for task in tasks:
        site_name = _effective_site(task)
        if site_name in quarantine_sites:
            quarantined_by_site.setdefault(site_name, []).append(_sanitize_task_for_output(task))
            quarantined += 1
            continue
        updated_task, task_changed, task_skipped, task_unmatched = _migrate_task(task)
        migrated_main.append(updated_task)
        changed += task_changed
        skipped += task_skipped
        unmatched += task_unmatched

    if changed == 0 and unmatched == 0 and quarantined == 0:
        print("migrate_phase_2_seeds_to_targets: already migrated, nothing to do")
        return 0

    print(
        "migrate_phase_2_seeds_to_targets: "
        f"changed={changed} skipped={skipped} unmatched_calls={unmatched} quarantined={quarantined}"
    )
    for site_name, items in sorted(quarantined_by_site.items()):
        if items:
            print(
                f"quarantine: site={site_name} count={len(items)} "
                f"path={_quarantine_path(args.path, site_name)}"
            )

    if unmatched:
        print("migrate_phase_2_seeds_to_targets: refusing to write partially migrated main dataset")
        return 1

    if args.dry_run:
        return 0

    pending_writes: list[tuple[Path, list[dict[str, Any]]]] = []
    for site_name, items in sorted(quarantined_by_site.items()):
        if items:
            pending_writes.append((_quarantine_path(args.path, site_name), items))
    pending_writes.append((args.path, migrated_main))

    temp_paths: list[tuple[Path, Path]] = []
    for final_path, payload in pending_writes:
        temp_path = final_path.with_name(f"{final_path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp_paths.append((final_path, temp_path))

    for final_path, temp_path in temp_paths[:-1]:
        temp_path.replace(final_path)
        print(f"wrote {final_path}")
    final_path, temp_path = temp_paths[-1]
    temp_path.replace(final_path)
    print(f"updated {final_path}")
    return 0


def _already_migrated(tasks: list[Any]) -> bool:
    saw_target = False
    for task in tasks:
        if not isinstance(task, dict):
            continue
        site_name = str(task.get("site", "")).strip().lower()
        if site_name in DEFAULT_QUARANTINE_SITES:
            return False
        api_calls = task.get("adversarial_data_seed", {}).get("api_calls")
        if not isinstance(api_calls, list):
            continue
        for call in api_calls:
            if not isinstance(call, dict):
                continue
            if isinstance(call.get("target"), dict):
                saw_target = True
                continue
            path = call.get("path") or call.get("url")
            if isinstance(path, str) and path.strip():
                return False
    return saw_target


def _quarantine_path(path: Path, site_name: str) -> Path:
    suffix = path.suffix or ".json"
    stem = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.name
    return path.with_name(f"{stem}.{site_name}_quarantine{suffix}")


def _migrate_task(task: dict[str, Any]) -> tuple[dict[str, Any], int, int, int]:
    updated = json.loads(json.dumps(task))
    changed = 0
    unmatched = 0
    skipped = 0
    saw_unmatched = False

    for seed_key in ("seed_template", "adversarial_data_seed"):
        seed = updated.get(seed_key)
        if not isinstance(seed, dict):
            if seed_key == "adversarial_data_seed":
                skipped += 1
            continue
        api_calls = seed.get("api_calls")
        if not isinstance(api_calls, list):
            if seed_key == "adversarial_data_seed":
                skipped += 1
            continue

        rewritten_calls: list[dict[str, Any]] = []
        seed_changed = 0
        seed_unmatched = 0
        for call in api_calls:
            migrated_call, call_changed, call_unmatched = _migrate_call(updated, call)
            rewritten_calls.append(migrated_call)
            seed_changed += int(call_changed)
            seed_unmatched += int(call_unmatched)
        seed["api_calls"] = rewritten_calls
        changed += seed_changed
        unmatched += seed_unmatched
        if seed_unmatched:
            saw_unmatched = True

    if saw_unmatched:
        updated = json.loads(json.dumps(task))

    updated = _sanitize_task_for_output(updated)
    return updated, int(changed > 0 or updated != task), skipped, unmatched


def _migrate_call(
    task: dict[str, Any], call: dict[str, Any]
) -> tuple[dict[str, Any], bool, bool]:
    if not isinstance(call, dict):
        return call, False, True
    if isinstance(call.get("target"), dict):
        normalized_call = _normalize_target_call(task, call)
        return normalized_call, normalized_call != call, False

    path = call.get("path") or call.get("url")
    method = str(call.get("method") or "").strip().upper()
    if not isinstance(path, str) or not path.strip() or not method:
        return call, False, True

    normalized_path = _normalize_call_path(task, path.strip())
    if not isinstance(normalized_path, str):
        print(
            f"unmatched: task={task.get('id', '?')} method={method} "
            f"path={path.strip()!r} reason=off_origin_or_untrusted_absolute_url"
        )
        return call, False, True

    target = _target_for_call(task, normalized_path, method, call)
    if target is None:
        print(f"unmatched: task={task.get('id', '?')} method={method} path={normalized_path!r}")
        return call, False, True

    migrated = {key: value for key, value in call.items() if key not in {"path", "url", "method"}}
    migrated["target"] = target
    return migrated, True, False


def _target_for_call(
    task: dict[str, Any], path: str, method: str, call: dict[str, Any]
) -> dict[str, Any] | None:
    delivery_site = _delivery_site(task)
    site_name = delivery_site or str(task.get("site", "")).strip()

    update_target = _update_target_for_call(task, path, method, call, site_name=site_name)
    if update_target is not None:
        return update_target

    create_target = _create_target_for_call(task, path, method, call, site_name=site_name)
    if create_target is not None:
        return create_target

    return None


def _create_target_for_call(
    task: dict[str, Any], path: str, method: str, call: dict[str, Any], *, site_name: str
) -> dict[str, Any] | None:
    if method != "POST":
        return None

    body = call.get("body")
    body_form = call.get("body_form")
    task_id = "{task_id}"

    if path == "/rest/V1/reviews":
        review_spec = _shopping_review_spec(task, call)
        return _target(site_name or "shopping", "product_review", create={"product_review": review_spec})

    if path == "/create_forum":
        return _target(
            "reddit",
            "forum",
            create={
                "forum": {
                    "name_template": _reddit_forum_name(task, path=path),
                    "description_template": f"Forum seeded for task {task_id}",
                }
            },
        )

    if re.fullmatch(r"/submit/[^/]+|/submit/\{forum_name\}", path):
        return _target(
            "reddit",
            "submission",
            create={
                "forum": {
                    "name_template": _reddit_forum_name(task, path=path),
                    "description_template": f"Forum seeded for task {task_id}",
                },
                "submission": {
                    "title_template": _reddit_submission_title(call),
                    "body_template": _reddit_submission_body(call),
                },
            },
        )

    if re.fullmatch(r"/f/[^/]+/\d+/-/comment|/f/\{forum_name\}/\{submission_id\}/-/comment", path):
        return _target(
            "reddit",
            "comment",
            create={
                "forum": {
                    "name_template": _reddit_forum_name(task, path=path),
                    "description_template": f"Forum seeded for task {task_id}",
                },
                "submission": {
                    "title_template": _expected_reddit_post_title(task) or f"Submission {task_id}",
                    "body_template": _expected_reddit_post_body(task),
                },
            },
        )

    if path == "/api/v4/projects":
        return _target(
            "gitlab",
            "project",
            create={
                "project": {
                    "name_template": _gitlab_project_name(task, body=body, body_form=body_form),
                    "path_template": _gitlab_project_path(task, body=body, body_form=body_form),
                }
            },
        )

    if path == "/api/v4/groups":
        return _target(
            "gitlab",
            "group",
            create={
                "group": {
                    "name_template": _gitlab_group_name(body),
                    "path_template": _gitlab_group_path(body),
                }
            },
        )

    if path == "/projects":
        return _target(
            "gitlab",
            "project",
            create={
                "project": {
                    "name_template": _gitlab_project_name(task, body=body, body_form=body_form),
                    "path_template": _gitlab_project_path(task, body=body, body_form=body_form),
                }
            },
        )

    if path == "/groups":
        return _target(
            "gitlab",
            "group",
            create={
                "group": {
                    "name_template": _gitlab_group_name(body, body_form=body_form),
                    "path_template": _gitlab_group_path(body, body_form=body_form),
                }
            },
        )

    if re.fullmatch(r"/api/v4/projects/(?:\d+|\{project_id\})/issues", path):
        return _target(
            "gitlab",
            "issue",
            create={
                "project": _gitlab_project_spec(task),
                "issue": {
                    "title_template": _gitlab_issue_title(body),
                    "body_template": _gitlab_issue_body(body),
                },
            },
        )

    if re.fullmatch(r"/api/v4/projects/(?:\d+|\{project_id\})/issues/(?:\d+|\{issue_iid\})/notes", path):
        return _target(
            "gitlab",
            "issue_note",
            create={
                "project": _gitlab_project_spec(task),
                "issue": {
                    "title_template": f"Seed issue {task_id}",
                    "body_template": f"Seed issue context for task {task_id}.",
                },
            },
        )

    if re.fullmatch(r"/api/v4/projects/(?:\d+|\{project_id\})/merge_requests", path):
        return _target(
            "gitlab",
            "mr",
            create={
                "project": _gitlab_project_spec(task),
                "mr": {
                    "title_template": _gitlab_mr_title(body),
                    "body_template": _gitlab_mr_body(body),
                    "source_branch_template": _gitlab_mr_source_branch(body),
                    "target_branch_template": _gitlab_mr_target_branch(body),
                },
            },
        )

    if re.fullmatch(r"/api/v4/projects/(?:\d+|\{project_id\})/merge_requests/(?:\d+|\{mr_iid\})/notes", path):
        return _target(
            "gitlab",
            "mr_note",
            create={
                "project": _gitlab_project_spec(task),
                "mr": {
                    "title_template": f"Seed MR {task_id}",
                    "body_template": f"Seed MR context for task {task_id}.",
                    "source_branch_template": f"webagent-{task_id}",
                    "target_branch_template": "main",
                },
            },
        )

    if re.fullmatch(r"/api/v4/projects/(?:\d+|\{project_id\})/repository/commits", path):
        return _target(
            "gitlab",
            "repo_file",
            create={"project": _gitlab_project_spec(task)},
        )

    return None


def _update_target_for_call(
    task: dict[str, Any], path: str, method: str, call: dict[str, Any], *, site_name: str
) -> dict[str, Any] | None:
    if (method == "PUT" and path == "/api/v4/user") or (method == "POST" and path == "/-/profile"):
        return _target("gitlab", "user_profile", update={})
    if method == "PUT" and path == "/api/v4/user/status":
        return _target("gitlab", "user_status", update={})
    if method == "PUT" and path == "/rest/V1/customers/me":
        return _target("shopping", "customer_profile", update={})
    if method == "PUT" and path == "/rest/V1/users/me":
        return _target("shopping_admin", "admin_profile", update={})
    if method == "POST" and re.fullmatch(r"/user/[^/]+/edit_biography|/user/\{username\}/edit_biography", path):
        return _target("reddit", "user_bio", update={})
    return None


def _target(
    site_name: str,
    resource_type: str,
    *,
    create: dict[str, Any] | None = None,
    update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target: dict[str, Any] = {
        "benchmark": "webarena_verified",
        "site": site_name,
        "resource_type": resource_type,
    }
    if create is not None:
        target["create"] = create
    if update is not None:
        target["update"] = update
    return target


def _delivery_site(task: dict[str, Any]) -> str:
    delivery_channel = task.get("delivery_channel")
    if not isinstance(delivery_channel, dict):
        return ""
    delivery_site = delivery_channel.get("delivery_site")
    if isinstance(delivery_site, str):
        normalized = delivery_site.strip()
        if normalized.lower() == "none":
            return ""
        return normalized
    return ""


def _effective_site(task: dict[str, Any]) -> str:
    delivery_site = _delivery_site(task).strip().lower()
    if delivery_site:
        return delivery_site
    return str(task.get("site", "")).strip().lower()


def _gitlab_project_spec(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "name_template": f"webagent-task-{{task_id}}",
        "path_template": f"webagent-task-{{task_id}}",
        "description_template": f"Seeded project for task {{task_id}}",
    }


def _gitlab_project_name(
    task: dict[str, Any], *, body: Any = None, body_form: Any = None
) -> str:
    if isinstance(body, dict):
        for key in ("name", "project_name"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(body_form, dict):
        for key in ("project[name]",):
            value = body_form.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"webagent-task-{{task_id}}"


def _gitlab_project_path(
    task: dict[str, Any], *, body: Any = None, body_form: Any = None
) -> str:
    if isinstance(body, dict):
        for key in ("path", "project_path"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(body_form, dict):
        for key in ("project[path]",):
            value = body_form.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"webagent-task-{{task_id}}"


def _gitlab_group_name(body: Any = None, body_form: Any = None) -> str:
    if isinstance(body, dict):
        value = body.get("name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(body_form, dict):
        value = body_form.get("group[name]")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"webagent-group-{{task_id}}"


def _gitlab_group_path(body: Any = None, body_form: Any = None) -> str:
    if isinstance(body, dict):
        value = body.get("path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(body_form, dict):
        value = body_form.get("group[path]")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"webagent-group-{{task_id}}"


def _gitlab_issue_title(body: Any) -> str:
    if isinstance(body, dict):
        value = body.get("title")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Seed issue {task_id}"


def _gitlab_issue_body(body: Any) -> str:
    if isinstance(body, dict):
        value = body.get("description")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Seed issue context for task {task_id}."


def _gitlab_mr_title(body: Any) -> str:
    if isinstance(body, dict):
        value = body.get("title")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Seed MR {task_id}"


def _gitlab_mr_body(body: Any) -> str:
    if isinstance(body, dict):
        value = body.get("description")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Seed merge request context for task {task_id}."


def _gitlab_mr_source_branch(body: Any) -> str:
    if isinstance(body, dict):
        value = body.get("source_branch")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "webagent-{task_id}"


def _gitlab_mr_target_branch(body: Any) -> str:
    if isinstance(body, dict):
        value = body.get("target_branch")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "main"


def _reddit_forum_name(task: dict[str, Any], path: str | None = None) -> str:
    if isinstance(path, str) and path.strip():
        match = re.fullmatch(r"/submit/([^/]+)", path) or re.fullmatch(r"/f/([^/]+)/.+", path)
        if match:
            forum_name = match.group(1).strip()
            if forum_name and "{" not in forum_name:
                return forum_name
    instantiation = task.get("instantiation_dict")
    if isinstance(instantiation, dict):
        forum = instantiation.get("forum")
        if isinstance(forum, str) and forum.strip():
            return forum.strip()
    return "webagent-task-{task_id}"


def _normalize_call_path(task: dict[str, Any], path_or_url: str) -> str | None:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        parsed = urllib.parse.urlparse(path_or_url)
        hostname = str(parsed.hostname or "").strip().lower()
        site_token = _effective_site(task).replace("_", "-")
        if not hostname or not site_token or not hostname.startswith(site_token):
            return None
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        return path
    return path_or_url


def _sanitize_task_for_output(task: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(task))
    for key in ("agent_context",):
        if key in sanitized:
            sanitized[key] = _sanitize_agent_context(sanitized[key])
    return sanitized


def _sanitize_agent_context(value: Any) -> Any:
    secrets = _collect_agent_context_secrets(value)
    return _sanitize_agent_context_node(value, secrets)


def _sanitize_agent_context_node(value: Any, secrets: set[str]) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            lowered = key_str.lower()
            if lowered in {"credentials", "headers"} and isinstance(item, dict):
                sanitized[key_str] = {inner_key: "<redacted>" for inner_key in item}
                continue
            if any(token in lowered for token in ("password", "token", "secret", "api_key")):
                sanitized[key_str] = "<redacted>"
                continue
            sanitized[key_str] = _sanitize_agent_context_node(item, secrets)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_agent_context_node(item, secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in sorted(secrets, key=len, reverse=True):
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
        for pattern, replacement in _EMBEDDED_SECRET_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted
    return value


def _collect_agent_context_secrets(value: Any) -> set[str]:
    secrets: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                lowered = str(key).lower()
                if lowered in {"credentials", "headers"} and isinstance(item, dict):
                    for inner in item.values():
                        if isinstance(inner, str) and inner:
                            secrets.add(inner)
                elif any(token in lowered for token in ("password", "token", "secret", "api_key")):
                    if isinstance(item, str) and item:
                        secrets.add(item)
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return secrets


def _reddit_submission_title(call: dict[str, Any]) -> str:
    body_form = call.get("body_form")
    if isinstance(body_form, dict):
        value = body_form.get("submission[title]")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _expected_reddit_post_title_from_call(call) or "Submission {task_id}"


def _reddit_submission_body(call: dict[str, Any]) -> str:
    body_form = call.get("body_form")
    if isinstance(body_form, dict):
        value = body_form.get("submission[body]")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Submission body for task {task_id}."


def _expected_reddit_post_title(task: dict[str, Any]) -> str | None:
    instantiation = task.get("instantiation_dict")
    if isinstance(instantiation, dict):
        title = instantiation.get("post_title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return None


def _expected_reddit_post_title_from_call(call: dict[str, Any]) -> str | None:
    body_form = call.get("body_form")
    if isinstance(body_form, dict):
        title = body_form.get("submission[title]")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return None


def _expected_reddit_post_body(task: dict[str, Any]) -> str:
    instruction = str(task.get("instruction", "")).strip()
    if instruction:
        return instruction
    return "Submission body for task {task_id}."


def _normalize_target_call(task: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(call))
    normalized.pop("method", None)
    normalized.pop("path", None)
    normalized.pop("url", None)
    target = normalized.get("target")
    if not isinstance(target, dict):
        return normalized

    target.setdefault("benchmark", "webarena_verified")
    resource_type = str(target.get("resource_type", "")).strip().lower()
    create_spec = target.get("create")
    if resource_type == "product_review" and isinstance(create_spec, dict):
        review_spec = create_spec.get("product_review")
        if not isinstance(review_spec, dict):
            review_spec = {}
        merged = _shopping_review_spec(task, normalized)
        if merged:
            create_spec["product_review"] = merged
    return normalized


def _shopping_review_spec(task: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    review_spec: dict[str, Any] = {}
    required_body = task.get("delivery_channel", {}).get("required_body")
    if isinstance(required_body, dict):
        required_review = required_body.get("review")
        if isinstance(required_review, dict):
            review_spec.update(required_review)

    body = call.get("body")
    if isinstance(body, dict):
        nested_review = body.get("review")
        if isinstance(nested_review, dict):
            review_spec.update(nested_review)
        else:
            for key in (
                "title",
                "nickname",
                "entity_pk_value",
                "review_entity",
                "stores",
                "ratings",
                "review_status",
            ):
                value = body.get(key)
                if value not in (None, ""):
                    review_spec[key] = value

    review_spec.setdefault("review_entity", "product")
    review_spec.setdefault("stores", [1])
    review_spec.setdefault("ratings", [{"rating_name": "Quality", "value": 4}])
    return review_spec


if __name__ == "__main__":
    raise SystemExit(main())
