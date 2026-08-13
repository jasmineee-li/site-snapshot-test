"""Seed delivery contract helpers.

This module owns deterministic parsing and matching for seed editor/API calls.
It is intentionally domain-specific: these helpers describe how a seed writes
payload text to an admitted delivery surface, not generic validation utilities.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any

from warp_taskgen.benchmark_capabilities import normalize_benchmark_name
from warp_taskgen.editors._registry import method_spec

PAYLOAD_PLACEHOLDER = "{{PAYLOAD_TEXT}}"

_TARGET_DELIVERY_PATHS = {
    ("gitlab", "project"): "/api/v4/projects",
    ("gitlab", "group"): "/api/v4/groups",
    ("gitlab", "issue"): "/api/v4/projects/{project_id}/issues",
    ("gitlab", "issue_note"): "/api/v4/projects/{project_id}/issues/{issue_iid}/notes",
    ("gitlab", "mr"): "/api/v4/projects/{project_id}/merge_requests",
    ("gitlab", "mr_note"): "/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes",
    ("gitlab", "repo_file"): "/api/v4/projects/{project_id}/repository/commits",
    ("gitlab", "user_profile"): "/api/v4/user",
    ("gitlab", "user_status"): "/api/v4/user/status",
    ("map", "node"): "/api/0.6/node/{node_id}",
    ("map", "way"): "/api/0.6/way/{way_id}",
    ("map", "relation"): "/api/0.6/relation/{relation_id}",
    ("reddit", "forum"): "/create_forum",
    ("reddit", "submission"): "/submit/{forum_name}",
    ("reddit", "comment"): "/f/{forum_name}/{submission_id}/-/comment",
    ("reddit", "user_bio"): "/user/{username}/edit_biography",
    ("shopping", "customer_profile"): "/customer/account/editpost/",
    ("shopping", "product_review"): "/rest/V1/reviews",
    ("shopping_admin", "admin_profile"): "/admin/admin/user/save/",
    ("shopping_admin", "product_review"): "/rest/V1/reviews",
}
_EDITOR_DELIVERY_PATHS = {
    ("gitlab", "create_project"): ("POST", "/api/v4/projects"),
    ("gitlab", "create_group"): ("POST", "/api/v4/groups"),
    ("gitlab", "create_issue"): ("POST", "/api/v4/projects/{project_id}/issues"),
    ("gitlab", "create_issue_description"): ("POST", "/api/v4/projects/{project_id}/issues"),
    ("gitlab", "create_issue_title"): ("POST", "/api/v4/projects/{project_id}/issues"),
    ("gitlab", "create_issue_note"): (
        "POST",
        "/api/v4/projects/{project_id}/issues/{issue_iid}/notes",
    ),
    ("gitlab", "create_mr"): ("POST", "/api/v4/projects/{project_id}/merge_requests"),
    ("gitlab", "create_mr_note"): (
        "POST",
        "/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes",
    ),
    ("gitlab", "create_repo_file"): ("POST", "/api/v4/projects/{project_id}/repository/commits"),
    ("gitlab", "update_user_status"): ("PUT", "/api/v4/user/status"),
    ("gitlab", "update_user_profile"): ("PUT", "/api/v4/user"),
    ("reddit", "create_forum"): ("POST", "/create_forum"),
    ("reddit", "create_submission"): ("POST", "/submit/{forum_name}"),
    ("reddit", "create_submission_title"): ("POST", "/submit/{forum_name}"),
    ("reddit", "create_comment"): ("POST", "/f/{forum_name}/{submission_id}/-/comment"),
    ("reddit", "update_user_bio"): ("POST", "/user/{username}/edit_biography"),
    ("shopping", "create_product_review"): ("POST", "/rest/V1/reviews"),
    ("shopping", "update_customer_profile"): ("POST", "/customer/account/editpost/"),
    ("shopping_admin", "create_product_review"): ("POST", "/rest/V1/reviews"),
    ("shopping_admin", "update_admin_profile"): ("POST", "/admin/admin/user/save/"),
}
_EDITOR_BODY_FIELD_ALIASES = {
    ("gitlab", "create_project"): {
        "name": "name_template",
        "path": "path_template",
        "description": "description_template",
        "project[name]": "name_template",
        "project[path]": "path_template",
        "project[description]": "description_template",
    },
    ("gitlab", "create_group"): {
        "name": "name_template",
        "path": "path_template",
        "description": "description_template",
        "group[name]": "name_template",
        "group[path]": "path_template",
        "group[description]": "description_template",
    },
    ("gitlab", "create_issue"): {
        "title": "title_template",
        "body": "body_template",
        "description": "body_template",
    },
    ("gitlab", "create_issue_description"): {"body": "body", "description": "body"},
    ("gitlab", "create_issue_title"): {"title": "title"},
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
    ("reddit", "create_submission"): {
        "title": "title_template",
        "body": "body_template",
        "submission[title]": "title_template",
        "submission[body]": "body_template",
    },
    ("reddit", "create_submission_title"): {
        "title": "title",
        "body": "body",
        "submission[title]": "title",
        "submission[body]": "body",
    },
    ("reddit", "create_comment"): {"body": "body"},
    ("reddit", "update_user_bio"): {"bio": "bio_text"},
    ("shopping", "create_product_review"): {"detail": "detail", "title": "title"},
    ("shopping", "update_customer_profile"): {"value": "value"},
    ("shopping_admin", "update_admin_profile"): {"value": "value"},
}
_EDITOR_BODY_FIELD_ALIASES_BY_BENCHMARK = {
    ("webarena_verified", site, method): aliases
    for (site, method), aliases in _EDITOR_BODY_FIELD_ALIASES.items()
}


_REDDIT_COMMENT_BODY_FIELD_PATTERN = re.compile(
    r"^reply_to_submission_(?:\{[^}\]]+\}|[^[]+)\[comment\]$"
)


def _call_body_field_value(call: dict[str, Any], body_key: str, field_name: str) -> Any:
    editor_args = call.get("args")
    if isinstance(editor_args, dict):
        for candidate in _editor_body_field_candidates(call, field_name):
            if candidate in editor_args:
                return editor_args[candidate]
        return _find_nested_field(editor_args, field_name)
    body = call.get(body_key)
    if isinstance(body, dict):
        if field_name in body:
            return body[field_name]
        nested_review = body.get("review")
        if isinstance(nested_review, dict) and field_name in nested_review:
            return nested_review[field_name]
    return None


def _editor_body_field_candidates(call: dict[str, Any], field_name: str) -> list[str]:
    """Return equivalent editor arg names for a delivery body field.

    Reddit/Postmill is the motivating case: profile delivery channels use
    concrete Symfony form names (``submission[title]``), the editor contract
    exposes LLM-facing names (``title``), and the Python editor method accepts
    implementation names (``title_template``). Postcondition validation should
    treat those as one field without hard-coding a special case in each caller.
    """

    out: list[str] = []

    def add(value: str | None) -> None:
        if isinstance(value, str) and value and value not in out:
            out.append(value)

    add(field_name)
    primary_alias = _editor_arg_name(call, field_name)
    add(primary_alias)
    if primary_alias is not None:
        for canonical_name, arg_name in _editor_arg_alias_pairs(call):
            if arg_name == primary_alias:
                add(canonical_name)
    return out


def _has_conflicting_nested_review_body(call: dict[str, Any], body_key: str) -> bool:
    body = call.get(body_key)
    if not isinstance(body, dict):
        return False
    nested_review = body.get("review")
    if not isinstance(nested_review, dict):
        return False
    return any(key != "review" for key in body)


def _find_nested_field(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        if field_name in value:
            return value[field_name]
        for item in value.values():
            resolved = _find_nested_field(item, field_name)
            if resolved is not None:
                return resolved
    elif isinstance(value, list):
        for item in value:
            resolved = _find_nested_field(item, field_name)
            if resolved is not None:
                return resolved
    return None


def _call_body_fields(call: dict[str, Any], body_key: str) -> dict[str, Any]:
    return {
        field_name: value
        for field_name, value, _source_name in _call_body_field_entries(call, body_key)
    }


def _call_body_field_entries(call: dict[str, Any], body_key: str) -> list[tuple[str, Any, str]]:
    editor_args = call.get("args")
    if isinstance(editor_args, dict):
        editor_key = _editor_delivery_key(call)
        if editor_key in {
            ("shopping", "update_customer_profile"),
            ("shopping_admin", "update_admin_profile"),
        }:
            field_name = str(editor_args.get("field") or "").strip()
            if field_name:
                return [(field_name, editor_args.get("value"), "value")]
        fields = [(str(key), value, str(key)) for key, value in editor_args.items()]
        field_names = {field_name for field_name, _value, _source_name in fields}
        for canonical_name, arg_name in _editor_arg_alias_pairs(call):
            if arg_name in editor_args and canonical_name not in field_names:
                fields.append((canonical_name, editor_args[arg_name], arg_name))
                field_names.add(canonical_name)
        for field_name, value, source_name in list(fields):
            for equivalent in _editor_body_field_candidates(call, field_name):
                if equivalent not in field_names:
                    fields.append((equivalent, value, source_name))
                    field_names.add(equivalent)
        dynamic_field = editor_args.get("field")
        if isinstance(dynamic_field, str) and dynamic_field.strip() and "value" in editor_args:
            fields.append((dynamic_field.strip(), editor_args["value"], "value"))
        return fields
    body = call.get(body_key)
    if not isinstance(body, dict):
        return []
    nested_review = body.get("review")
    if isinstance(nested_review, dict) and all(str(key) == "review" for key in body):
        return [(str(key), value, str(key)) for key, value in nested_review.items()]
    return [(str(key), value, str(key)) for key, value in body.items() if str(key) != "review"]


def _seed_calls(seed: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    api_calls = seed.get("api_calls")
    if isinstance(api_calls, list):
        calls.extend(call for call in api_calls if isinstance(call, dict))
    editor_calls = seed.get("editor_calls")
    if isinstance(editor_calls, list):
        calls.extend(call for call in editor_calls if isinstance(call, dict))
    return calls


def _call_body_key(seed: dict[str, Any], call: dict[str, Any]) -> str:
    if isinstance(call.get("args"), dict):
        return "args"
    return "body_form" if seed.get("mechanism") == "form" else "body"


def _call_matches_delivery_entry(
    call: dict[str, Any],
    *,
    mechanism: str,
    entry: Any,
) -> bool:
    if not isinstance(entry, dict) or entry.get("mechanism") != mechanism:
        return False
    path = _call_delivery_path(call)
    method = _call_method(call)
    path_template = entry.get("path_template")
    entry_method = entry.get("method")
    if (
        not isinstance(path, str)
        or not isinstance(method, str)
        or not isinstance(path_template, str)
        or not isinstance(entry_method, str)
    ):
        return False
    return entry_method.strip().upper() == method.strip().upper() and _normalize_delivery_path(
        path_template
    ) == _normalize_delivery_path(path)


def _call_satisfies_path_param(call: dict[str, Any], path_param: str) -> bool:
    path = _call_delivery_path(call)
    if not isinstance(path, str):
        return False
    if isinstance(call.get("args"), dict):
        return f"{{{path_param}}}" in path
    if "target" in call:
        return f"{{{path_param}}}" in path
    return f"{{{path_param}}}" not in path


def _call_delivery_path(call: dict[str, Any]) -> str | None:
    path = call.get("path")
    if isinstance(path, str) and path:
        return path
    url = call.get("url")
    if isinstance(url, str) and url:
        return _url_to_path(url)
    editor_key = _editor_delivery_key(call)
    if editor_key is not None:
        binding = _editor_delivery_binding(call)
        if binding is None:
            return None
        return binding[1]
    target = call.get("target")
    if not isinstance(target, dict):
        return None
    return _target_delivery_path(target, call)


def _call_method(call: dict[str, Any]) -> str | None:
    editor_key = _editor_delivery_key(call)
    if editor_key is not None:
        binding = _editor_delivery_binding(call)
        if binding is None:
            return None
        return binding[0]
    method = call.get("method")
    if isinstance(method, str) and method.strip():
        return method
    target = call.get("target")
    if not isinstance(target, dict):
        return None
    if "update" in target:
        return "PUT"
    if "create" in target:
        return "POST"
    return None


def _target_delivery_path(target: dict[str, Any], call: dict[str, Any] | None = None) -> str | None:
    site_name = str(target.get("site", "")).strip().lower()
    resource_type = str(target.get("resource_type", "")).strip().lower()
    if isinstance(call, dict):
        if resource_type == "project" and isinstance(call.get("body_form"), dict):
            return "/projects"
        if resource_type == "group" and isinstance(call.get("body_form"), dict):
            return "/groups"
    return _TARGET_DELIVERY_PATHS.get((site_name, resource_type))


def _editor_delivery_key(call: dict[str, Any]) -> tuple[str, str] | None:
    site_name = str(call.get("site", "")).strip().lower()
    method_name = str(call.get("method", "")).strip()
    if site_name and method_name and isinstance(call.get("args"), dict):
        return (site_name, method_name)
    return None


def _call_has_benchmark_metadata(call: dict[str, Any]) -> bool:
    return any(
        isinstance(call.get(key), str) and str(call.get(key)).strip()
        for key in ("benchmark", "benchmark_name", "benchmark_adapter")
    )


def _editor_delivery_contract_key(call: dict[str, Any]) -> tuple[str, str, str] | None:
    benchmark = normalize_benchmark_name(
        call.get("benchmark") or call.get("benchmark_name") or call.get("benchmark_adapter")
    )
    if not benchmark:
        if _call_has_benchmark_metadata(call):
            return None
        benchmark = "webarena_verified"
    site_name = str(call.get("site", "")).strip().lower()
    method_name = str(call.get("method", "")).strip()
    if site_name and method_name and isinstance(call.get("args"), dict):
        return (benchmark, site_name, method_name)
    return None


def _editor_delivery_binding(call: dict[str, Any]) -> tuple[str, str] | None:
    contract_key = _editor_delivery_contract_key(call)
    if contract_key is not None:
        benchmark, site, method = contract_key
        try:
            return method_spec(site, method, benchmark=benchmark).http
        except KeyError:
            if _call_has_benchmark_metadata(call) and benchmark != "webarena_verified":
                return None
    legacy_key = _editor_delivery_key(call)
    if legacy_key is not None:
        return _EDITOR_DELIVERY_PATHS.get(legacy_key)
    return None


def _editor_arg_alias_pairs(call: dict[str, Any]) -> list[tuple[str, str]]:
    aliases = None
    contract_key = _editor_delivery_contract_key(call)
    if contract_key is not None:
        aliases = _EDITOR_BODY_FIELD_ALIASES_BY_BENCHMARK.get(contract_key)
        if (
            aliases is None
            and _call_has_benchmark_metadata(call)
            and contract_key[0] != "webarena_verified"
        ):
            return []
    if aliases is None:
        editor_key = _editor_delivery_key(call)
        if editor_key is None:
            return []
        aliases = _EDITOR_BODY_FIELD_ALIASES.get(editor_key)
    if not isinstance(aliases, dict):
        return []
    return [(str(canonical), str(arg_name)) for canonical, arg_name in aliases.items()]


def _editor_arg_name(call: dict[str, Any], canonical_name: str) -> str | None:
    editor_key = _editor_delivery_key(call)
    if (
        (not _call_has_benchmark_metadata(call) or _editor_delivery_contract_key(call) is not None)
        and editor_key == ("reddit", "create_comment")
        and _REDDIT_COMMENT_BODY_FIELD_PATTERN.match(canonical_name)
    ):
        return "body"
    for canonical, arg_name in _editor_arg_alias_pairs(call):
        if canonical == canonical_name:
            return arg_name
    return None


def _call_delivery_mechanism(seed: dict[str, Any], call: dict[str, Any]) -> str | None:
    if _editor_delivery_binding(call) is not None:
        editor_key = _editor_delivery_key(call)
        if editor_key is not None and (
            editor_key[0] == "reddit"
            or editor_key
            in {
                ("shopping", "update_customer_profile"),
                ("shopping_admin", "update_admin_profile"),
            }
        ):
            return "form"
        return "api"
    contract_key = _editor_delivery_contract_key(call)
    if (
        _editor_delivery_key(call) is not None
        and _call_has_benchmark_metadata(call)
        and (contract_key is None or contract_key[0] != "webarena_verified")
    ):
        return None
    mechanism = seed.get("mechanism")
    if isinstance(mechanism, str) and mechanism in {"api", "form"}:
        return mechanism
    return None


def _url_to_path(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    return path


def _contains_deferred_map_target(seed: dict[str, Any]) -> bool:
    for call in _seed_calls(seed):
        if not isinstance(call, dict):
            continue
        if str(call.get("site", "")).strip().lower() == "map":
            return True
        target = call.get("target")
        if not isinstance(target, dict):
            continue
        if str(target.get("site", "")).strip().lower() == "map":
            return True
    return False


def _call_site(call: dict[str, Any]) -> str | None:
    site_name = call.get("site")
    if isinstance(site_name, str) and site_name.strip():
        return site_name.strip()
    return None


def _normalize_payload_value(value: Any) -> str:
    if isinstance(value, str):
        return "".join(value.split()).lower()
    return json.dumps(value, sort_keys=True).lower()


def _normalize_delivery_path(path: str) -> str:
    return re.sub(r"/\{[^}/]+\}(?=/|$)", "/{id}", re.sub(r"/\d+(?=/|$)", "/{id}", path))
