from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("logs/phase_2/adversarial_tasks.json")
DEFAULT_QUARANTINE = Path("logs/phase_2/adversarial_tasks.map_quarantine.json")
_REDDIT_COMMENT_BODY_FIELD_PATTERN = re.compile(
    r"^reply_to_submission_(?:\{[^}\]]+\}|[^[]+)\[comment\]$"
)
_EDITOR_BODY_FIELD_ALIASES: dict[tuple[str, str], dict[str, str]] = {
    ("gitlab", "create_project"): {
        "name": "name_template",
        "path": "path_template",
        "description": "description_template",
    },
    ("gitlab", "create_group"): {
        "name": "name_template",
        "path": "path_template",
        "description": "description_template",
    },
    ("gitlab", "create_issue"): {"body": "body_template", "description": "body_template"},
    ("gitlab", "create_issue_note"): {"body": "note_body"},
    ("gitlab", "create_mr"): {"body": "body_template", "description": "body_template"},
    ("gitlab", "create_mr_note"): {"body": "note_body"},
    ("gitlab", "create_repo_file"): {
        "path": "path",
        "content": "content",
        "commit_message": "commit_message",
    },
    ("gitlab", "update_user_status"): {"message": "message"},
    ("gitlab", "update_user_profile"): {
        "bio": "bio",
        "name": "name",
        "user[bio]": "bio",
        "user[name]": "name",
    },
    ("reddit", "create_forum"): {
        "name": "name_template",
        "description": "description_template",
        "forum[name]": "name_template",
        "forum[description]": "description_template",
    },
    ("reddit", "create_submission"): {"title": "title_template", "body": "body_template"},
    ("reddit", "create_comment"): {"body": "body"},
    ("reddit", "update_user_bio"): {"bio": "bio_text"},
    ("shopping", "create_product_review"): {"detail": "detail", "title": "title"},
    ("shopping", "update_customer_profile"): {"value": "value"},
    ("shopping_admin", "update_admin_profile"): {"value": "value"},
}
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


def _prefer_placeholder(*values: Any) -> Any:
    """Pick the first value containing ``{{PAYLOAD_TEXT}}``; fall back to
    the first non-empty string; else ``None``. Used when a seed carries
    both a rendered body and a templated version — the templated one is
    canonical for seed_template, the rendered one is canonical for
    adversarial_data_seed. Whichever holds the placeholder is the
    right source to preserve across migration."""
    for v in values:
        if isinstance(v, str) and "{{PAYLOAD_TEXT}}" in v:
            return v
    for v in values:
        if isinstance(v, str) and v:
            return v
    for v in values:
        if v is not None:
            return v
    return None


def _editor_call_from_target_call(call: dict[str, Any]) -> dict[str, Any]:
    target = call.get("target")
    if not isinstance(target, dict):
        raise ValueError("call is missing target")
    benchmark = str(target.get("benchmark") or "webarena_verified").strip() or "webarena_verified"
    site = str(target.get("site", "")).strip().lower()
    resource_type = str(target.get("resource_type", "")).strip().lower()
    create_spec = target.get("create") if isinstance(target.get("create"), dict) else {}
    update_spec = target.get("update") if isinstance(target.get("update"), dict) else {}
    body = call.get("body") if isinstance(call.get("body"), dict) else {}
    body_form = call.get("body_form") if isinstance(call.get("body_form"), dict) else {}

    if site == "shopping" and resource_type == "product_review":
        review = (
            create_spec.get("product_review")
            if isinstance(create_spec.get("product_review"), dict)
            else {}
        )
        rating_value = 4
        ratings = review.get("ratings")
        if isinstance(ratings, list) and ratings and isinstance(ratings[0], dict):
            rating_value = int(ratings[0].get("value") or 4)
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_product_review",
            "args": {
                "title": _prefer_placeholder(body.get("title"), review.get("title")),
                "detail": _prefer_placeholder(body.get("detail"), review.get("detail")),
                "nickname": body.get("nickname", review.get("nickname")),
                "entity_pk_value": body.get("entity_pk_value", review.get("entity_pk_value")),
                "rating": rating_value,
            },
        }

    if site == "shopping" and resource_type == "customer_profile":
        field, value = next(iter(body.items()), (None, None))
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "update_customer_profile",
            "args": {
                "field": field,
                "value": value,
            },
        }

    if site == "reddit" and resource_type == "forum":
        forum = create_spec.get("forum") if isinstance(create_spec.get("forum"), dict) else {}
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_forum",
            "args": {
                "name_template": _prefer_placeholder(
                    body_form.get("forum[name]"), forum.get("name_template")
                ),
                "description_template": _prefer_placeholder(
                    body_form.get("forum[description]"),
                    forum.get("description_template"),
                ),
            },
        }

    if site == "reddit" and resource_type == "submission":
        forum = create_spec.get("forum") if isinstance(create_spec.get("forum"), dict) else {}
        submission = (
            create_spec.get("submission") if isinstance(create_spec.get("submission"), dict) else {}
        )
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_submission",
            "args": {
                "forum_name": forum.get("name_template", "{forum_name}"),
                "title_template": _prefer_placeholder(
                    body_form.get("submission[title]"), submission.get("title_template")
                ),
                "body_template": _prefer_placeholder(
                    body_form.get("submission[body]"), submission.get("body_template")
                ),
            },
        }

    if site == "reddit" and resource_type == "comment":
        forum = create_spec.get("forum") if isinstance(create_spec.get("forum"), dict) else {}
        comment_create = (
            create_spec.get("comment") if isinstance(create_spec.get("comment"), dict) else {}
        )
        comment_body_form = next(
            (value for key, value in body_form.items() if key.startswith("reply_to_submission_")),
            None,
        )
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_comment",
            "args": {
                "forum_name": forum.get("name_template", "{forum_name}"),
                "submission_id": "{submission_id}",
                "body": _prefer_placeholder(
                    comment_body_form,
                    comment_create.get("body_template"),
                    comment_create.get("body"),
                ),
            },
        }

    if site == "gitlab" and resource_type == "project":
        project = create_spec.get("project") if isinstance(create_spec.get("project"), dict) else {}
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_project",
            "args": {
                "name_template": _prefer_placeholder(
                    body.get("name"),
                    body_form.get("project[name]"),
                    project.get("name_template"),
                ),
                "path_template": _prefer_placeholder(
                    body.get("path"),
                    body_form.get("project[path]"),
                    project.get("path_template"),
                ),
                "description_template": _prefer_placeholder(
                    body.get("description"),
                    body_form.get("project[description]"),
                    project.get("description_template"),
                ),
            },
        }

    if site == "gitlab" and resource_type == "group":
        group = create_spec.get("group") if isinstance(create_spec.get("group"), dict) else {}
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_group",
            "args": {
                "name_template": _prefer_placeholder(
                    body.get("name"),
                    body_form.get("group[name]"),
                    group.get("name_template"),
                ),
                "path_template": _prefer_placeholder(
                    body.get("path"),
                    body_form.get("group[path]"),
                    group.get("path_template"),
                ),
                "description_template": _prefer_placeholder(
                    body.get("description"),
                    body_form.get("group[description]"),
                    group.get("description_template"),
                ),
            },
        }

    if site == "gitlab" and resource_type == "issue":
        project = create_spec.get("project") if isinstance(create_spec.get("project"), dict) else {}
        issue = create_spec.get("issue") if isinstance(create_spec.get("issue"), dict) else {}
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_issue",
            "args": {
                "project_name_template": project.get("name_template"),
                "project_path_template": project.get("path_template"),
                "project_description_template": project.get("description_template"),
                "title_template": _prefer_placeholder(
                    body.get("title"), issue.get("title_template")
                ),
                "body_template": _prefer_placeholder(
                    body.get("description", body.get("body")),
                    issue.get("body_template"),
                ),
            },
        }

    if site == "gitlab" and resource_type == "issue_note":
        project = create_spec.get("project") if isinstance(create_spec.get("project"), dict) else {}
        issue = create_spec.get("issue") if isinstance(create_spec.get("issue"), dict) else {}
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_issue_note",
            "args": {
                "project_name_template": project.get("name_template"),
                "project_path_template": project.get("path_template"),
                "project_description_template": project.get("description_template"),
                "issue_title_template": issue.get("title_template"),
                "issue_body_template": issue.get("body_template"),
                "note_body": body.get("body"),
            },
        }

    if site == "gitlab" and resource_type == "mr":
        project = create_spec.get("project") if isinstance(create_spec.get("project"), dict) else {}
        merge_request = create_spec.get("mr") if isinstance(create_spec.get("mr"), dict) else {}
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_mr",
            "args": {
                "project_name_template": project.get("name_template"),
                "project_path_template": project.get("path_template"),
                "project_description_template": project.get("description_template"),
                "title_template": _prefer_placeholder(
                    body.get("title"), merge_request.get("title_template")
                ),
                "body_template": _prefer_placeholder(
                    body.get("description"), merge_request.get("body_template")
                ),
                "source_branch": body.get(
                    "source_branch", merge_request.get("source_branch_template")
                ),
                "target_branch": body.get(
                    "target_branch", merge_request.get("target_branch_template")
                ),
            },
        }

    if site == "gitlab" and resource_type == "mr_note":
        project = create_spec.get("project") if isinstance(create_spec.get("project"), dict) else {}
        merge_request = create_spec.get("mr") if isinstance(create_spec.get("mr"), dict) else {}
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_mr_note",
            "args": {
                "project_name_template": project.get("name_template"),
                "project_path_template": project.get("path_template"),
                "project_description_template": project.get("description_template"),
                "mr_title_template": merge_request.get("title_template"),
                "mr_body_template": merge_request.get("body_template"),
                "source_branch": merge_request.get("source_branch_template"),
                "target_branch": merge_request.get("target_branch_template"),
                "note_body": body.get("body"),
            },
        }

    if site == "gitlab" and resource_type == "repo_file":
        project = create_spec.get("project") if isinstance(create_spec.get("project"), dict) else {}
        action = {}
        actions = body.get("actions")
        if isinstance(actions, list) and actions and isinstance(actions[0], dict):
            action = actions[0]
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "create_repo_file",
            "args": {
                "project_name_template": project.get("name_template"),
                "project_path_template": project.get("path_template"),
                "project_description_template": project.get("description_template"),
                "branch": body.get("branch"),
                "path": action.get("file_path"),
                "content": action.get("content"),
                "commit_message": body.get("commit_message"),
            },
        }

    if site == "gitlab" and resource_type == "user_status":
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "update_user_status",
            "args": {
                "message": body.get("message"),
                "emoji": body.get("emoji"),
            },
        }

    if site == "gitlab" and resource_type == "user_profile":
        args: dict[str, Any] = {}
        if "user[bio]" in body_form or "bio" in body:
            args["bio"] = body_form.get("user[bio]", body.get("bio"))
        if "user[name]" in body_form or "name" in body:
            args["name"] = body_form.get("user[name]", body.get("name"))
        if args:
            return {
                "benchmark": benchmark,
                "site": site,
                "method": "update_user_profile",
                "args": args,
            }
        raise ValueError("unsupported gitlab user_profile payload for migration")

    if site == "shopping_admin" and resource_type == "admin_profile":
        field, value = next(iter(body.items()), (None, None))
        return {
            "benchmark": benchmark,
            "site": site,
            "method": "update_admin_profile",
            "args": {
                "field": field,
                "value": value,
            },
        }

    raise ValueError(f"unsupported target resource for migration: {(site, resource_type)!r}")


def _effective_delivery_site(task: dict[str, Any]) -> str:
    delivery_channel = task.get("delivery_channel")
    if isinstance(delivery_channel, dict):
        delivery_site = str(delivery_channel.get("delivery_site") or "").strip().lower()
        if delivery_site and delivery_site != "none":
            return delivery_site
    return str(task.get("site") or "").strip().lower()


def migrate_seed(seed: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(seed, dict):
        return seed
    api_calls = seed.get("api_calls")
    existing_editor_calls = (
        list(seed.get("editor_calls", [])) if isinstance(seed.get("editor_calls"), list) else []
    )
    if not isinstance(api_calls, list):
        return seed
    editor_calls: list[dict[str, Any]] = list(existing_editor_calls)
    preserved_api_calls: list[dict[str, Any]] = []
    converted = 0
    for call in api_calls:
        if not isinstance(call, dict):
            continue
        target = call.get("target")
        if not isinstance(target, dict):
            preserved_api_calls.append(call)
            continue
        editor_calls.append(_editor_call_from_target_call(call))
        converted += 1
    if converted and preserved_api_calls:
        raise ValueError("mixed legacy api_calls and target-based calls cannot be migrated safely")
    if seed.get("mechanism") == "editor" and preserved_api_calls:
        raise ValueError("editor seeds must not retain legacy api_calls after migration")
    if converted:
        seed["editor_calls"] = editor_calls
        seed["api_calls"] = preserved_api_calls
        if not preserved_api_calls:
            seed.pop("api_calls")
        if seed.get("mechanism") in (None, "api", "form") and "api_calls" not in seed:
            seed["mechanism"] = "editor"
    elif existing_editor_calls and not preserved_api_calls:
        seed.pop("api_calls", None)
    return seed


def migrate_task(task: dict[str, Any]) -> dict[str, Any]:
    migrated = json.loads(json.dumps(task))
    for field_name in ("seed_template", "adversarial_data_seed"):
        seed = migrated.get(field_name)
        if isinstance(seed, dict):
            migrated[field_name] = migrate_seed(seed)
    _rewrite_delivery_channel_body_fields(migrated)
    return _sanitize_task_for_output(migrated)


def _rewrite_delivery_channel_body_fields(task: dict[str, Any]) -> None:
    delivery_channel = task.get("delivery_channel")
    if not isinstance(delivery_channel, dict):
        return

    candidate_calls: list[dict[str, Any]] = []
    for field_name in ("seed_template", "adversarial_data_seed"):
        seed = task.get(field_name)
        if not isinstance(seed, dict):
            continue
        editor_calls = seed.get("editor_calls")
        if not isinstance(editor_calls, list):
            continue
        for call in editor_calls:
            if isinstance(call, dict):
                candidate_calls.append(call)

    if not candidate_calls:
        return

    body_field = delivery_channel.get("body_field")
    rewritten = _canonical_editor_body_field(candidate_calls, body_field)
    if isinstance(rewritten, str):
        delivery_channel["body_field"] = rewritten

    postcondition = delivery_channel.get("postcondition")
    where = postcondition.get("where") if isinstance(postcondition, dict) else None
    if not isinstance(where, dict):
        return
    for source in where.values():
        if not isinstance(source, dict) or len(source) != 1 or "body_field" not in source:
            continue
        source["body_field"] = _canonical_editor_body_field(candidate_calls, source["body_field"])


def _canonical_editor_body_field(candidate_calls: list[dict[str, Any]], raw_field: Any) -> Any:
    if not isinstance(raw_field, str) or not raw_field.strip():
        return raw_field
    for call in candidate_calls:
        args = call.get("args")
        if not isinstance(args, dict):
            continue
        if raw_field in args:
            return raw_field
        alias = _editor_arg_name(call, raw_field)
        if alias and alias in args:
            return alias
    return raw_field


def _editor_arg_name(call: dict[str, Any], canonical_name: str) -> str | None:
    site_name = str(call.get("site") or "").strip().lower()
    method_name = str(call.get("method") or "").strip()
    if (site_name, method_name) == (
        "reddit",
        "create_comment",
    ) and _REDDIT_COMMENT_BODY_FIELD_PATTERN.match(canonical_name):
        return "body"
    aliases = _EDITOR_BODY_FIELD_ALIASES.get((site_name, method_name), {})
    arg_name = aliases.get(canonical_name)
    return str(arg_name) if isinstance(arg_name, str) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--quarantine-path", type=Path, default=DEFAULT_QUARANTINE)
    args = parser.parse_args()

    data = json.loads(args.path.read_text())
    migrated: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for task in data:
        if not isinstance(task, dict):
            continue
        if _effective_delivery_site(task) == "map":
            quarantined.append(_sanitize_task_for_output(task))
            continue
        migrated.append(migrate_task(task))

    _atomic_write_json(args.path, migrated)
    if quarantined:
        _atomic_write_json(args.quarantine_path, quarantined)
    print(
        f"migrated {len(migrated)} tasks to editor_calls"
        + (f", quarantined {len(quarantined)} map tasks" if quarantined else "")
    )
    return 0


def _sanitize_task_for_output(task: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(task))
    if "agent_context" in sanitized:
        sanitized["agent_context"] = _sanitize_agent_context_for_output(sanitized["agent_context"])
    return _sanitize_value(sanitized)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, str):
        sanitized = value
        for pattern, replacement in _EMBEDDED_SECRET_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized
    return value


def _sanitize_agent_context_for_output(value: Any) -> Any:
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


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
