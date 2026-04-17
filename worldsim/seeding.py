"""Data seeding dispatchers.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 3 / Evaluation
Infrastructure" section.

Supported task-setup mechanisms:

- ``api``: make a list of HTTP requests against ``instance["site_url"]``
- ``form``: submit a list of form-style HTTP requests against ``instance["site_url"]``
- ``state_push``: PUT a JSON blob to the instance's ``/api/state`` endpoint

SQL seeding was evaluated and excluded from the methodology because it violates
the threat model (a regular authenticated user cannot write to the database
directly). Database read access is retained for postcondition verification and
reward evaluation.

Each benchmark's Phase 0a manifest declares which mechanism its sites use.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import urllib.parse
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from worldsim.db_urls import parse_supported_db_connection
from worldsim.placeholders import apply_placeholders, merge_placeholder_maps
from worldsim.seed_resolvers import get_resolver

logger = logging.getLogger(__name__)

# Host-side backstop: the real allowlist is in ``worldsim.fix_validation``.
# This mirrors ``fix_validation.ALLOWED_METHODS`` so destructive verbs
# (DELETE) and probing verbs (HEAD, OPTIONS) are blocked at the lowest
# layer too.
_ALLOWED_API_METHODS = frozenset({"GET", "POST", "PUT", "PATCH"})
_FORM_METHODS = frozenset({"POST", "PUT", "PATCH"})
_CSRF_TOKEN_CACHE: weakref.WeakKeyDictionary[
    requests.Session,
    dict[tuple[str, str], tuple[str | None, str | None]],
] = weakref.WeakKeyDictionary()
_REDDIT_TABLE_NAME_CACHE: dict[tuple[str, str], str] = {}
_BLOCKED_CALL_HEADER_NAMES = frozenset(
    {
        "authorization",
        "cookie",
        "origin",
        "referer",
        "x-csrf-token",
        "x-csrftoken",
        "x-xsrf-token",
        "x-xsrftoken",
        "host",
        "forwarded",
        "proxy",
        "proxy-authorization",
        "proxy-authenticate",
        "proxy-connection",
        "transfer-encoding",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
    }
)
_CSRF_INPUT_PATTERNS = (
    re.compile(
        r'name=["\'](form_key|authenticity_token|csrf_token|token)["\'][^>]*value=["\']([^"\']+)'
    ),
    re.compile(
        r'<meta[^>]+name=["\']csrf-token["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
)
_CSRF_PARAM_META = re.compile(
    r'<meta[^>]+name=["\']csrf-param["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_PATH_PARAM_PATTERN = re.compile(r"\{([^}/]+)\}")
_UNRESOLVED_TEMPLATE_TOKEN = re.compile(r"\{[^}/]+\}")
_FORMAT_TOKEN_PATTERN = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_\.]*)\}(?!\})")


@dataclass(frozen=True)
class PreparedSeedCall:
    method: str
    raw_path: str
    url: str
    headers: dict[str, str]
    json_body: Any
    body_form: Any
    rendered_call: dict[str, Any]
    context_additions: dict[str, Any]


def self_contained_adversarial_seed_error(benign_seed: Any, adversarial_seed: Any) -> str | None:
    """Return an error when an adversarial seed drops benign setup state.

    Phase 4 applies only ``adversarial_data_seed``, so the adversarial seed
    must preserve the benign seed verbatim before extending it.
    """
    if not isinstance(benign_seed, dict) or not isinstance(adversarial_seed, dict):
        return None

    mechanism = benign_seed.get("mechanism")
    if mechanism in (None, "none"):
        return None

    if not _seed_preserves_prefix(benign_seed, adversarial_seed):
        return (
            "adversarial_data_seed must preserve the benign data_seed verbatim before extending it"
        )
    return None


def validate_data_seed(seed: dict[str, Any], *, allow_none: bool = False) -> None:
    """Validate a seed payload before it is persisted or executed."""
    if not isinstance(seed, dict):
        raise ValueError("data seed must be an object")

    mechanism = seed.get("mechanism")
    if mechanism in (None, "none"):
        if allow_none:
            return
        raise ValueError("data seed must declare a non-empty mechanism")

    if mechanism in {"api", "form"}:
        api_calls = seed.get("api_calls")
        if not isinstance(api_calls, list) or not api_calls:
            raise ValueError(f"{mechanism} data seed must include a non-empty api_calls list")
        saw_target = False
        saw_legacy = False
        # Belt-and-suspenders: reject destructive/probing HTTP methods here
        # even when the upstream Phase 3 validator was bypassed. The
        # authoritative allowlist (path, origin) lives in
        # ``worldsim.fix_validation.validate_fix_patch``; this check keeps
        # any caller of ``apply_data_seed`` on the safe subset of verbs.
        for call in api_calls:
            if not isinstance(call, dict):
                raise ValueError("api data seed calls must be objects")
            method = call.get("method")
            target = call.get("target")
            if isinstance(target, dict):
                saw_target = True
                site_name = target.get("site")
                resource_type = target.get("resource_type")
                has_create = "create" in target
                has_update = "update" in target
                if not isinstance(site_name, str) or not site_name.strip():
                    raise ValueError("targeted data seed calls must include target.site")
                if not isinstance(resource_type, str) or not resource_type.strip():
                    raise ValueError("targeted data seed calls must include target.resource_type")
                if has_create == has_update:
                    raise ValueError(
                        "targeted data seed calls must include exactly one of target.create or target.update"
                    )
                verb_payload = target.get("create") if has_create else target.get("update")
                verb_name = "create" if has_create else "update"
                if not isinstance(verb_payload, dict):
                    raise ValueError(
                        f"targeted data seed calls must include target.{verb_name} as an object"
                    )
                if method is not None:
                    if not isinstance(method, str) or not method.strip():
                        raise ValueError(
                            "targeted data seed call method must be a non-empty string"
                        )
                    if method.strip().upper() not in _ALLOWED_API_METHODS:
                        raise ValueError(
                            f"{mechanism} data seed method {method!r} not allowed "
                            f"(allowed: {sorted(_ALLOWED_API_METHODS)})"
                        )
                    if mechanism == "form" and method.strip().upper() not in _FORM_METHODS:
                        raise ValueError(
                            f"form data seed method {method!r} not allowed "
                            f"(allowed: {sorted(_FORM_METHODS)})"
                        )
            else:
                saw_legacy = True
                if not isinstance(method, str) or not method.strip():
                    raise ValueError("api data seed calls must include a method")
                if method.strip().upper() not in _ALLOWED_API_METHODS:
                    raise ValueError(
                        f"{mechanism} data seed method {method!r} not allowed "
                        f"(allowed: {sorted(_ALLOWED_API_METHODS)})"
                    )
                if mechanism == "form" and method.strip().upper() not in _FORM_METHODS:
                    raise ValueError(
                        f"form data seed method {method!r} not allowed "
                        f"(allowed: {sorted(_FORM_METHODS)})"
                    )
                raw_ref = _call_reference(call)
                if not isinstance(raw_ref, str) or not raw_ref.strip():
                    raise ValueError(
                        f"{mechanism} data seed calls must include a path starting with '/' or a url"
                    )
                if not (
                    raw_ref.startswith("/")
                    or raw_ref.startswith("http://")
                    or raw_ref.startswith("https://")
                ):
                    raise ValueError(
                        f"{mechanism} data seed calls must include a path starting with '/' or a url"
                    )
            json_body = call.get("body")
            body_form = call.get("body_form")
            if mechanism == "form":
                if not isinstance(body_form, dict) or not body_form:
                    raise ValueError(
                        "form data seed calls must include a non-empty body_form object"
                    )
                if json_body is not None:
                    raise ValueError("form data seed calls must not include JSON body")
            elif body_form is not None:
                raise ValueError("api data seed calls must use body, not body_form")
        if saw_target and saw_legacy:
            raise ValueError(
                "mixed legacy and target-based api_calls are not supported; use one shape per seed"
            )
        return

    if mechanism == "state_push":
        if "state" not in seed:
            raise ValueError("state_push data seed must include a state payload")
        return

    raise ValueError(f"unknown data seed mechanism: {mechanism!r}")


def apply_data_seed(seed: dict[str, Any], instance: dict[str, Any]) -> None:
    """Apply a data seed to a running benchmark instance.

    Args:
        seed: Seed spec with a ``mechanism`` field and mechanism-specific
            extras. See the v5 spec for the field schemas.
        instance: Benchmark instance dict with ``site_url`` and any
            mechanism-specific auth configuration.

    Raises:
        ValueError: If ``seed["mechanism"]`` is unknown.
    """
    validate_data_seed(seed)
    mechanism = seed["mechanism"]
    if mechanism in {"api", "form"}:
        seed_context = _build_seed_context(seed, instance)
        with requests.Session() as session:
            _perform_web_login_if_needed(session, instance, mechanism)
            for call in seed["api_calls"]:
                prepared_call = _prepare_http_seed_call(
                    call,
                    instance,
                    mechanism=mechanism,
                    seed_context=seed_context,
                )
                _merge_seed_context(seed_context, prepared_call.context_additions)
                response = _apply_http_seed_call(session, mechanism, prepared_call, instance)
                _merge_seed_context(seed_context, _extract_response_seed_context(response))
    elif mechanism == "state_push":
        resp = requests.put(
            f"{instance['site_url']}/api/state",
            json=seed["state"],
            timeout=30,
        )
        resp.raise_for_status()
    else:
        raise ValueError(f"unknown data seed mechanism: {mechanism!r}")


async def apply_data_seed_async(seed: dict[str, Any], instance: dict[str, Any]) -> None:
    """Apply a data seed without blocking the event loop."""
    await asyncio.to_thread(apply_data_seed, seed, instance)


def _build_seed_context(seed: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    explicit_context = instance.get("seed_context")
    if isinstance(explicit_context, dict):
        _merge_seed_context(context, explicit_context)

    task = instance.get("seed_task")
    if isinstance(task, dict):
        task_id = str(task.get("id") or "").strip()
        if task_id:
            context["task_id"] = task_id
        instruction = str(task.get("instruction", "")).strip()
        if instruction:
            context["instruction"] = instruction
        topic = str(task.get("topic") or instruction).strip()
        if topic:
            context["topic"] = topic
        intent = str(task.get("intent") or instruction).strip()
        if intent:
            context["intent"] = intent
        benign_task_id = task.get("benign_task_id")
        if benign_task_id not in (None, ""):
            context["benign_task_id"] = str(benign_task_id)
        _merge_seed_context(context, _derive_task_seed_context(task, seed, instance))
    context.setdefault("task_id", "task")
    context.setdefault("instruction", "")
    context.setdefault("topic", context.get("task_id", "task"))
    context.setdefault("intent", context.get("instruction") or context.get("topic") or "task")
    return context


def _derive_task_seed_context(
    task: dict[str, Any],
    seed: dict[str, Any],
    instance: dict[str, Any],
) -> dict[str, Any]:
    placeholders = _seed_placeholder_names(seed)
    if not placeholders:
        return {}

    site_name = str(instance.get("site_name", task.get("site", ""))).strip().lower()
    if site_name == "reddit":
        return _derive_reddit_seed_context(task, instance, placeholders)
    if site_name == "map":
        return _derive_map_seed_context(task, instance, placeholders)
    return {}


def _seed_placeholder_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, str):
        names.update(match.group(1) for match in _FORMAT_TOKEN_PATTERN.finditer(value))
        return names
    if isinstance(value, dict):
        for key, item in value.items():
            names.update(_seed_placeholder_names(key))
            names.update(_seed_placeholder_names(item))
        return names
    if isinstance(value, list):
        for item in value:
            names.update(_seed_placeholder_names(item))
    return names


def _merge_seed_context(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if value is None:
            continue
        target[key] = value


def _render_http_seed_call(
    call: dict[str, Any],
    *,
    seed_context: dict[str, Any],
) -> dict[str, Any]:
    rendered = _render_seed_value(call, seed_context)
    if not isinstance(rendered, dict):
        raise RuntimeError("rendered seed call must be an object")
    unresolved = sorted(_seed_placeholder_names(rendered))
    if unresolved:
        raise RuntimeError(
            f"HTTP seed call has unresolved template placeholders: {', '.join(unresolved)}"
        )
    return rendered


def preflight_http_seed_calls(seed: dict[str, Any], instance: dict[str, Any]) -> list[str]:
    """Resolve HTTP calls without firing mutations.

    Legacy concrete-path seeds may depend on response-derived placeholders from
    earlier calls. Those chains remain validated at real execution time only.
    """
    validate_data_seed(seed, allow_none=False)
    mechanism = str(seed.get("mechanism", "")).strip().lower()
    if mechanism not in {"api", "form"}:
        return []

    seed_context = _build_seed_context(seed, instance)
    preflight_instance = dict(instance)
    preflight_instance["seed_resolver_dry_run"] = True
    errors: list[str] = []
    for index, call in enumerate(seed.get("api_calls", [])):
        if not isinstance(call, dict):
            continue
        if not isinstance(call.get("target"), dict) and _seed_placeholder_names(call):
            continue
        try:
            prepared_call = _prepare_http_seed_call(
                call,
                preflight_instance,
                mechanism=mechanism,
                seed_context=seed_context,
            )
        except Exception as exc:
            errors.append(f"api_calls[{index}]: {exc}")
            continue
        _merge_seed_context(seed_context, prepared_call.context_additions)
    return errors


def _render_seed_value(value: Any, seed_context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        whole_match = _FORMAT_TOKEN_PATTERN.fullmatch(value)
        if whole_match is not None:
            resolved = _lookup_seed_context_value(seed_context, whole_match.group(1))
            if resolved is not None:
                return resolved

        def _replace(match: re.Match[str]) -> str:
            resolved = _lookup_seed_context_value(seed_context, match.group(1))
            if resolved is None:
                return match.group(0)
            return str(resolved)

        return _FORMAT_TOKEN_PATTERN.sub(_replace, value)

    if isinstance(value, dict):
        rendered: dict[Any, Any] = {}
        for key, item in value.items():
            rendered_key = _render_seed_value(key, seed_context) if isinstance(key, str) else key
            rendered[rendered_key] = _render_seed_value(item, seed_context)
        return rendered

    if isinstance(value, list):
        return [_render_seed_value(item, seed_context) for item in value]

    return value


def _lookup_seed_context_value(seed_context: dict[str, Any], key: str) -> Any:
    if key in seed_context:
        return seed_context[key]

    current: Any = seed_context
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _prepare_http_seed_call(
    call: dict[str, Any],
    instance: dict[str, Any],
    *,
    mechanism: str,
    seed_context: dict[str, Any],
) -> PreparedSeedCall:
    target = call.get("target")
    if isinstance(target, dict):
        rendered_target = _render_seed_value(target, seed_context)
        if not isinstance(rendered_target, dict):
            raise RuntimeError("rendered seed target must be an object")
        unresolved_target = sorted(_seed_placeholder_names(rendered_target))
        if unresolved_target:
            raise RuntimeError(
                "seed target has unresolved template placeholders: " + ", ".join(unresolved_target)
            )
        target_site = str(rendered_target.get("site", "")).strip().lower()
        target_benchmark = (
            str(rendered_target.get("benchmark") or "webarena_verified").strip().lower()
        )
        instance_site = str(instance.get("site_name", "")).strip().lower()
        if target_site and instance_site and target_site != instance_site:
            raise RuntimeError(
                f"target.site {target_site!r} does not match bound seed instance site {instance_site!r}"
            )
        has_create = "create" in rendered_target
        has_update = "update" in rendered_target
        if has_create == has_update:
            raise RuntimeError("target must have exactly one of {create, update}")
        rendered_method = _render_seed_value(
            call.get("method") or ("PUT" if has_update else "POST"),
            seed_context,
        )
        method = str(rendered_method).strip().upper()
        resolver = get_resolver(target_benchmark, target_site or instance_site)
        resolver_target = dict(rendered_target)
        resolver_target["method"] = method
        resolver_target["mechanism"] = mechanism
        resolver_target["body"] = call.get("body")
        resolver_target["body_form"] = call.get("body_form")
        if has_create:
            resolved = resolver.create(resolver_target, instance, seed_context)
        else:
            resolved = resolver.update(resolver_target, instance, seed_context)
        method = str(resolved.method).strip().upper()
        next_context = dict(seed_context)
        _merge_seed_context(next_context, resolved.context_additions)
        rendered_call = _render_http_seed_call(
            {**call, "method": method},
            seed_context=next_context,
        )
        effective_call = dict(rendered_call)
        if resolved.body is not None:
            body_key = "body_form" if mechanism == "form" else "body"
            effective_call[body_key] = _render_seed_value(resolved.body, next_context)
        if resolved.headers:
            merged_headers = dict(effective_call.get("headers") or {})
            merged_headers.update(resolved.headers)
            effective_call["headers"] = merged_headers
        url = resolved.url
        concrete_path = urllib.parse.urlparse(url).path
        if urllib.parse.urlparse(url).query:
            concrete_path += f"?{urllib.parse.urlparse(url).query}"
        effective_call["path"] = concrete_path
        raw_path = _format_target_descriptor(rendered_target)
        context_additions = dict(resolved.context_additions)
        instance_origin = _origin_for_url(str(instance["site_url"]))
        if _origin_for_url(url) != instance_origin:
            raise RuntimeError(
                f"HTTP seed target must stay on origin {instance_origin!r}, got {url!r}"
            )
    else:
        rendered_call = _render_http_seed_call(call, seed_context=seed_context)
        method = str(rendered_call["method"]).strip().upper()
        raw_path = _call_reference(rendered_call)
        if raw_path is None:
            raise RuntimeError("rendered legacy seed call must include path or url")
        context_additions = {}
        effective_call = dict(rendered_call)
        url = _resolve_call_url(raw_path, instance)
        effective_call.setdefault("path", _concrete_call_path(url))

    headers = _build_request_headers(instance, effective_call, mechanism=mechanism)
    return PreparedSeedCall(
        method=method,
        raw_path=raw_path,
        url=url,
        headers=headers,
        json_body=effective_call.get("body"),
        body_form=effective_call.get("body_form"),
        rendered_call=effective_call,
        context_additions=context_additions,
    )


def _format_target_descriptor(target: dict[str, Any]) -> str:
    site_name = str(target.get("site", "")).strip() or "unknown"
    resource_type = str(target.get("resource_type", "")).strip() or "unknown"
    if "create" in target:
        return f"target:{site_name}:{resource_type}:create"
    if "update" in target:
        return f"target:{site_name}:{resource_type}:update"
    return f"target:{site_name}:{resource_type}"


def _call_reference(call: dict[str, Any]) -> str | None:
    raw_path = call.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        return raw_path
    raw_url = call.get("url")
    if isinstance(raw_url, str) and raw_url.strip():
        return raw_url
    return None


def _concrete_call_path(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    return path


def _extract_response_seed_context(response: requests.Response) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    _extract_location_seed_context(flattened, response)
    try:
        payload = response.json()
    except ValueError:
        return flattened
    if not isinstance(payload, dict):
        return flattened
    _flatten_response_payload(flattened, payload)
    return flattened


def _extract_location_seed_context(flattened: dict[str, Any], response: requests.Response) -> None:
    location = response.headers.get("Location")
    if not isinstance(location, str) or not location.strip():
        return
    parsed = urllib.parse.urlparse(location)
    path = parsed.path or location
    match = re.search(r"/f/([^/]+)/(\d+)(?:/|$)", path)
    if match:
        flattened.setdefault("forum_name", match.group(1))
        flattened.setdefault("submission_id", match.group(2))


def _flatten_response_payload(
    flattened: dict[str, Any],
    payload: dict[str, Any],
    *,
    prefix: str = "",
) -> None:
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_response_payload(flattened, value, prefix=full_key)
            continue
        if isinstance(value, list):
            continue
        flattened[full_key] = value
        flattened.setdefault(key, value)


def _derive_reddit_seed_context(
    task: dict[str, Any],
    instance: dict[str, Any],
    placeholders: set[str],
) -> dict[str, Any]:
    if "forum_name" not in placeholders and "submission_id" not in placeholders:
        return {}

    forum = _resolve_reddit_forum(task, instance)
    context: dict[str, Any] = {}
    if forum is not None:
        forum_name = forum.get("name")
        forum_id = forum.get("id")
        if isinstance(forum_name, str) and forum_name.strip():
            context["forum_name"] = forum_name.strip()
        if forum_id is not None:
            context["forum_id"] = forum_id

    if "submission_id" in placeholders and "forum_name" in context:
        submission_id = _resolve_reddit_submission_id(
            task, instance, forum_name=context["forum_name"]
        )
        if submission_id is not None:
            context["submission_id"] = submission_id
    return context


def _resolve_reddit_forum(task: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any] | None:
    instantiation = task.get("instantiation_dict")
    forum_hint = None
    if isinstance(instantiation, dict):
        raw_forum = instantiation.get("forum")
        if isinstance(raw_forum, str) and raw_forum.strip():
            forum_hint = raw_forum.strip()
    if forum_hint is None:
        return None

    db_connection = instance.get("db_connection")
    if not db_connection:
        return {"name": forum_hint}

    parsed = _parse_runtime_db_connection(
        db_connection,
        purpose="Reddit seed placeholder resolution requires instance['db_connection']",
    )
    conn = _connect_db(parsed)
    try:
        scheme = parsed.scheme.lower()
        _configure_read_only_connection(conn, scheme)
        forum_table = _quote_identifier(
            _resolve_reddit_table_name(
                conn,
                scheme,
                db_connection,
                logical_name="forum",
                candidates=("forums", "forum"),
            ),
            scheme,
        )
        name_col = _quote_identifier("name", scheme)
        title_col = _quote_identifier("title", scheme)
        id_col = _quote_identifier("id", scheme)
        query = (
            f"SELECT {id_col}, {name_col}, {title_col} "
            f"FROM {forum_table} "
            f"WHERE LOWER({name_col}) = LOWER(%s) OR LOWER({title_col}) = LOWER(%s) "
            f"ORDER BY CASE WHEN LOWER({name_col}) = LOWER(%s) THEN 0 ELSE 1 END "
            "LIMIT 1"
        )
        with conn.cursor() as cursor:
            cursor.execute(query, [forum_hint, forum_hint, forum_hint])
            row = cursor.fetchone()
    except Exception as exc:
        raise RuntimeError(
            f"failed to resolve reddit forum_name for {forum_hint!r}: {exc}"
        ) from exc
    finally:
        try:
            conn.rollback()
        except Exception:
            logger.debug("Failed to rollback reddit forum lookup", exc_info=True)
        conn.close()

    if not row:
        return {"name": forum_hint}
    if isinstance(row, dict):
        return {"id": row.get("id"), "name": row.get("name"), "title": row.get("title")}
    row_values = list(row)
    return {
        "id": row_values[0] if len(row_values) > 0 else None,
        "name": row_values[1] if len(row_values) > 1 else forum_hint,
        "title": row_values[2] if len(row_values) > 2 else None,
    }


def _resolve_reddit_submission_id(
    task: dict[str, Any],
    instance: dict[str, Any],
    *,
    forum_name: str,
) -> Any:
    benign_reward = task.get("reward_function", {})
    if isinstance(benign_reward, dict) and isinstance(benign_reward.get("benign_reward"), dict):
        reward = benign_reward["benign_reward"]
    else:
        reward = benign_reward

    expected_title = _expected_reddit_post_title(reward)
    if not expected_title:
        return None

    db_connection = instance.get("db_connection")
    if not db_connection:
        raise RuntimeError(
            "reddit seed placeholder resolution requires instance['db_connection'] "
            "to map the target post title to a submission_id"
        )

    parsed = _parse_runtime_db_connection(
        db_connection,
        purpose="Reddit seed placeholder resolution requires instance['db_connection']",
    )
    conn = _connect_db(parsed)
    try:
        scheme = parsed.scheme.lower()
        _configure_read_only_connection(conn, scheme)
        submission_table = _quote_identifier(
            _resolve_reddit_table_name(
                conn,
                scheme,
                db_connection,
                logical_name="submission",
                candidates=("submissions", "submission"),
            ),
            scheme,
        )
        forum_table = _quote_identifier(
            _resolve_reddit_table_name(
                conn,
                scheme,
                db_connection,
                logical_name="forum",
                candidates=("forums", "forum"),
            ),
            scheme,
        )
        submission_id_col = _quote_identifier("id", scheme)
        forum_id_col = _quote_identifier("forum_id", scheme)
        timestamp_col = _quote_identifier("timestamp", scheme)
        title_col = _quote_identifier("title", scheme)
        forum_name_col = _quote_identifier("name", scheme)
        forum_title_col = _quote_identifier("title", scheme)
        query = (
            f"SELECT s.{submission_id_col} "
            f"FROM {submission_table} s "
            f"JOIN {forum_table} f ON s.{forum_id_col} = f.{_quote_identifier('id', scheme)} "
            f"WHERE s.{title_col} = %s "
            f"AND (LOWER(f.{forum_name_col}) = LOWER(%s) OR LOWER(f.{forum_title_col}) = LOWER(%s)) "
            f"ORDER BY s.{timestamp_col} DESC "
            "LIMIT 1"
        )
        with conn.cursor() as cursor:
            cursor.execute(query, [expected_title, forum_name, forum_name])
            row = cursor.fetchone()
    except Exception as exc:
        raise RuntimeError(
            f"failed to resolve reddit submission_id for {expected_title!r} in forum {forum_name!r}: {exc}"
        ) from exc
    finally:
        try:
            conn.rollback()
        except Exception:
            logger.debug("Failed to rollback reddit submission lookup", exc_info=True)
        conn.close()

    if row is None:
        raise RuntimeError(
            f"could not resolve reddit submission_id for title {expected_title!r} in forum "
            f"{forum_name!r}"
        )
    if isinstance(row, dict):
        return row.get("id")
    if isinstance(row, (list, tuple)):
        return row[0] if row else None
    return row


def _resolve_reddit_table_name(
    conn: Any,
    scheme: str,
    db_connection: str,
    *,
    logical_name: str,
    candidates: tuple[str, ...],
) -> str:
    cache_key = (db_connection, logical_name)
    cached = _REDDIT_TABLE_NAME_CACHE.get(cache_key)
    if cached:
        return cached
    if scheme not in ("postgresql", "postgres"):
        resolved = candidates[0]
        _REDDIT_TABLE_NAME_CACHE[cache_key] = resolved
        return resolved

    with conn.cursor() as cursor:
        for candidate in candidates:
            cursor.execute("SELECT to_regclass(%s)", [candidate])
            row = cursor.fetchone()
            resolved = None
            if isinstance(row, dict):
                resolved = row.get("to_regclass")
            elif isinstance(row, (list, tuple)):
                resolved = row[0] if row else None
            else:
                resolved = row
            if resolved not in (None, ""):
                _REDDIT_TABLE_NAME_CACHE[cache_key] = candidate
                return candidate

    tried = ", ".join(candidates)
    raise RuntimeError(
        f"reddit schema table resolution failed for logical table {logical_name!r} (tried: {tried})"
    )


def _expected_reddit_post_title(reward_function: dict[str, Any]) -> str | None:
    eval_entries = reward_function.get("eval")
    if not isinstance(eval_entries, list):
        return None
    for entry in eval_entries:
        if not isinstance(entry, dict):
            continue
        expected = entry.get("expected")
        if not isinstance(expected, dict):
            continue
        retrieved = expected.get("retrieved_data")
        if not isinstance(retrieved, list):
            continue
        for item in retrieved:
            if not isinstance(item, dict):
                continue
            post_title = item.get("post_title")
            if isinstance(post_title, str) and post_title.strip():
                return post_title.strip()
    return None


def _derive_map_seed_context(
    task: dict[str, Any],
    instance: dict[str, Any],
    placeholders: set[str],
) -> dict[str, Any]:
    needs_way = "way_id" in placeholders
    needs_relation = "relation_id" in placeholders
    if not needs_way and not needs_relation:
        return {}

    instantiation = task.get("instantiation_dict")
    place = instantiation.get("place") if isinstance(instantiation, dict) else None
    if not isinstance(place, str) or not place.strip():
        return {}

    site_url = str(instance.get("site_url", "")).rstrip("/")
    if not site_url:
        return {}
    search_url = f"{site_url}/nominatim/search"
    try:
        response = requests.get(
            search_url,
            params={"q": place.strip(), "format": "jsonv2", "limit": 10},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"failed to resolve map seed placeholders for {place!r}: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError(f"map placeholder lookup for {place!r} returned invalid JSON payload")

    context: dict[str, Any] = {}
    if needs_way:
        way = _pick_map_search_result(payload, osm_type="way", place=place)
        if way is not None:
            context["way_id"] = way
    if needs_relation:
        relation = _pick_map_search_result(payload, osm_type="relation", place=place)
        if relation is not None:
            context["relation_id"] = relation
    return context


def _pick_map_search_result(payload: list[Any], *, osm_type: str, place: str) -> Any:
    place_lower = place.lower()
    fallback = None
    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("osm_type", "")).strip().lower() != osm_type:
            continue
        osm_id = item.get("osm_id")
        if fallback is None:
            fallback = osm_id
        haystack = " ".join(
            str(item.get(key, "")).lower()
            for key in ("display_name", "name")
            if item.get(key) is not None
        )
        if place_lower and place_lower in haystack:
            return osm_id
    return fallback


def _perform_web_login_if_needed(
    session: requests.Session, instance: dict[str, Any], mechanism: str
) -> None:
    """Log in via web form if the effective auth type is ``web_login``.

    Performs a two-step flow: GET the login page to extract a CSRF token,
    then POST credentials with the token. The resulting session cookies are
    stored on *session* for subsequent seeding requests.
    """
    auth = _effective_auth(instance, mechanism)
    if not isinstance(auth, dict) or auth.get("type") != "web_login":
        return
    site_url = str(instance.get("site_url", "")).rstrip("/")
    login_path = str(auth.get("login_url", "/login"))
    login_url = f"{site_url}{login_path}"
    credentials = auth.get("credentials", {})
    if not isinstance(credentials, dict) or not credentials:
        raise RuntimeError(
            f"web_login auth for {instance.get('site_name', '?')} requires credentials"
        )

    # GET login page — extract CSRF token from HTML.
    resp = session.get(login_url, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    token_name, token_value = _extract_csrf_token(resp.text)

    login_data: dict[str, str] = {}
    login_data.update(credentials)
    if token_name and token_value:
        login_data[token_name] = token_value

    post_resp = session.post(login_url, data=login_data, timeout=30, allow_redirects=False)
    if post_resp.status_code not in (200, 302):
        raise RuntimeError(
            f"Web login failed for {instance.get('site_name', '?')}: HTTP {post_resp.status_code}"
        )


def collect_seed_runtime_errors(
    tasks: list[dict[str, Any]],
    instances: list[Any],
    *,
    seed_field: str,
) -> list[str]:
    """Return deduplicated runtime-configuration errors for selected seeds."""
    errors: list[str] = []
    seen: set[str] = set()

    for task in tasks:
        if not isinstance(task, dict):
            continue
        seed = task.get(seed_field)
        if not isinstance(seed, dict):
            continue
        mechanism = seed.get("mechanism")
        if mechanism in (None, "none"):
            continue
        try:
            validate_data_seed(seed, allow_none=True)
        except ValueError as exc:
            _append_runtime_error(
                errors,
                seen,
                f"task {task.get('id', '?')!r} has invalid {seed_field}: {exc}",
            )
            continue

        seed_site = _task_seed_site(task)
        site_instances = [
            instance
            for instance in instances
            if _instance_value(instance, "site_name") == seed_site
        ]
        if not site_instances:
            _append_runtime_error(
                errors,
                seen,
                f"site {seed_site!r} has seeded task(s) but no configured instances",
            )
            continue

        requires_http_auth = mechanism in {"api", "form"}
        for instance in site_instances:
            site_url = _instance_value(instance, "site_url") or "<unknown>"
            if requires_http_auth:
                auth_error = _instance_http_seed_auth_runtime_error(instance, mechanism=mechanism)
                if auth_error is not None:
                    _append_runtime_error(
                        errors,
                        seen,
                        f"site {seed_site!r} has HTTP-seeded task(s) but instance {site_url!r} "
                        f"has invalid auth config: {auth_error}",
                    )

    return errors


def _seed_preserves_prefix(benign_value: Any, adversarial_value: Any) -> bool:
    """Return True when ``adversarial_value`` structurally contains ``benign_value``."""
    if isinstance(benign_value, dict):
        if not isinstance(adversarial_value, dict):
            return False
        for key, benign_item in benign_value.items():
            if key not in adversarial_value:
                return False
            if not _seed_preserves_prefix(benign_item, adversarial_value[key]):
                return False
        return True

    if isinstance(benign_value, list):
        if not isinstance(adversarial_value, list) or len(adversarial_value) < len(benign_value):
            return False
        return all(
            _seed_preserves_prefix(benign_item, adversarial_value[index])
            for index, benign_item in enumerate(benign_value)
        )

    return benign_value == adversarial_value


def _instance_value(instance: Any, field: str) -> Any:
    if isinstance(instance, dict):
        return instance.get(field)
    return getattr(instance, field, None)


def _append_runtime_error(errors: list[str], seen: set[str], message: str) -> None:
    if message in seen:
        return
    seen.add(message)
    errors.append(message)


def _task_seed_site(task: dict[str, Any]) -> str:
    delivery_channel = task.get("delivery_channel")
    if isinstance(delivery_channel, dict):
        delivery_site = delivery_channel.get("delivery_site")
        if isinstance(delivery_site, str) and delivery_site.strip():
            normalized = delivery_site.strip()
            if normalized.lower() != "none":
                return normalized
    site = str(task.get("site", "")).strip()
    return site or "<unknown>"


def _task_http_seed_requires_db(task: dict[str, Any], seed: dict[str, Any]) -> bool:
    if seed.get("mechanism") not in {"api", "form"}:
        return False
    delivery_channel = task.get("delivery_channel")
    if not isinstance(delivery_channel, dict):
        return False
    postcondition = delivery_channel.get("postcondition")
    return isinstance(postcondition, dict) and postcondition.get("type") == "db_row_value"


def _instance_http_seed_auth_runtime_error(instance: Any, *, mechanism: str = "form") -> str | None:
    auth = _effective_auth(
        instance if isinstance(instance, dict) else {},
        mechanism,
    )
    if not isinstance(auth, dict):
        return None

    auth_type = str(auth.get("type", "")).strip()
    if auth_type == "http_headers":
        headers = auth.get("headers")
        if not isinstance(headers, dict) or not headers:
            return "http_headers auth requires a non-empty headers dict"
        for value in headers.values():
            try:
                _resolve_header_value(value)
            except RuntimeError as exc:
                return str(exc)
        return None

    if auth_type == "bearer_token":
        token = auth.get("token")
        if isinstance(token, str) and token.strip():
            return None
        token_generator = auth.get("token_generator")
        if isinstance(token_generator, str) and token_generator.strip():
            # Runtime generation from credentials; no file needed.
            credentials = auth.get("credentials")
            if not isinstance(credentials, dict) or not credentials:
                return "token_generator auth requires a non-empty credentials dict"
            return None
        token_source = auth.get("token_source")
        if isinstance(token_source, str) and token_source.strip():
            try:
                path = _resolve_token_source_path(token_source)
            except RuntimeError as exc:
                return str(exc)
            if not path.exists():
                return f"token_source {path} does not exist"
            try:
                token_text = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                return f"token_source {path} could not be read: {exc}"
            if not token_text:
                return f"token_source {path} is empty"
            return None
        token_endpoint = auth.get("token_endpoint")
        if isinstance(token_endpoint, str) and token_endpoint.strip():
            return None
        return "bearer_token auth requires token, token_source, token_generator, or token_endpoint"

    if auth_type == "web_login":
        credentials = auth.get("credentials")
        if not isinstance(credentials, dict) or not credentials:
            return "web_login auth requires a non-empty credentials dict"
        return None

    return None


def _parse_runtime_db_connection(
    db_connection: Any,
    *,
    purpose: str,
) -> urllib.parse.ParseResult:
    try:
        return parse_supported_db_connection(db_connection, purpose=purpose)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _apply_http_seed_call(
    session: requests.Session,
    mechanism: str,
    call: PreparedSeedCall,
    instance: dict[str, Any],
) -> requests.Response:
    method = call.method
    raw_path = call.raw_path
    url = call.url
    headers = dict(call.headers)
    json_body = call.json_body
    form_body = _prepare_form_body(method, url, headers, call.body_form, instance, session)

    response = _request_with_context(
        session,
        method=method,
        url=url,
        headers=headers,
        json_body=json_body if form_body is None else None,
        form_body=form_body,
        instance=instance,
        raw_path=raw_path,
    )
    if form_body is not None and response.status_code in {403, 419, 422}:
        _clear_cached_csrf_token(session, instance, url)
        retried_form_body = _prepare_form_body(
            method,
            url,
            headers,
            call.body_form,
            instance,
            session,
            force_refresh=True,
        )
        response = _request_with_context(
            session,
            method=method,
            url=url,
            headers=headers,
            json_body=None,
            form_body=retried_form_body,
            instance=instance,
            raw_path=raw_path,
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        site_name = instance.get("site_name", "<unknown>")
        raise RuntimeError(
            f"HTTP seed failed for site {site_name!r} {method} {raw_path}: "
            f"status={response.status_code}"
        ) from exc

    try:
        _verify_http_seed_postcondition(
            mechanism=mechanism,
            call=call.rendered_call,
            instance=instance,
            raw_path=raw_path,
        )
    except RuntimeError as exc:
        # DB-based postcondition verification requires the site's internal
        # DB to be reachable from the orchestrator host. For some sites
        # (notably gitlab's embedded postgres, which binds only to 127.0.0.1
        # inside the container) the DB port mapping doesn't actually expose
        # postgres externally. In that case the postcondition check fails
        # with a wrapped psycopg2.OperationalError even though the seed POST
        # itself succeeded. Setting WORLDSIM_SKIP_DB_POSTCONDITION=1 lets us
        # soft-skip verification and proceed with the optimistic assumption
        # that the POST succeeded (the HTTP status was already 2xx above).
        # Long-term fix: expose the site's DB on TCP and drop this flag.
        cause_text = str(exc.__cause__) if exc.__cause__ is not None else str(exc)
        is_db_connection_failure = "OperationalError" in cause_text or (
            "connection" in cause_text.lower() and "refused" in cause_text.lower()
        )
        if is_db_connection_failure and os.environ.get(
            "WORLDSIM_SKIP_DB_POSTCONDITION", ""
        ).strip() in ("1", "true", "yes"):
            site_name = instance.get("site_name", "<unknown>")
            logger.warning(
                "seed postcondition DB unreachable for site %r (%s); WORLDSIM_SKIP_DB_POSTCONDITION=1 set, accepting POST status as proxy",
                site_name,
                cause_text.splitlines()[0][:200],
            )
        else:
            raise
    return response


def _resolve_call_url(raw_path: str, instance: dict[str, Any]) -> str:
    placeholders = merge_placeholder_maps(instance.get("url_placeholders"))
    resolved_path = apply_placeholders(raw_path, placeholders, strict=True)
    instance_origin = _origin_for_url(str(instance["site_url"]))
    if resolved_path.startswith("http://") or resolved_path.startswith("https://"):
        resolved_url = resolved_path
    else:
        resolved_url = f"{str(instance['site_url']).rstrip('/')}{resolved_path}"
    if _origin_for_url(resolved_url) != instance_origin:
        raise RuntimeError(
            f"HTTP seed target must stay on origin {instance_origin!r}, got {resolved_url!r}"
        )
    return resolved_url


def _effective_auth(instance: dict[str, Any], mechanism: str) -> dict[str, Any] | None:
    """Return the auth config for the given seeding mechanism.

    API-mechanism tasks prefer ``api_auth`` (e.g. admin bearer token) over the
    default ``auth`` (e.g. customer auto-login headers). Form-mechanism tasks
    always use ``auth``.
    """
    if mechanism == "api":
        api_auth = instance.get("api_auth")
        if isinstance(api_auth, dict):
            return api_auth
    return instance.get("auth")


def _build_request_headers(
    instance: dict[str, Any], call: dict[str, Any], *, mechanism: str = "form"
) -> dict[str, str]:
    headers: dict[str, str] = {}
    auth = _effective_auth(instance, mechanism)
    site_url = str(instance.get("site_url", ""))
    auth_header_names: set[str] = set()
    if isinstance(auth, dict):
        auth_type = str(auth.get("type", "")).strip()
        if auth_type == "http_headers":
            declared_headers = auth.get("headers")
            if isinstance(declared_headers, dict):
                for key, value in declared_headers.items():
                    resolved = _resolve_header_value(value)
                    headers[str(key)] = resolved
                    auth_header_names.add(str(key).lower())
        elif auth_type == "bearer_token":
            token = _resolve_bearer_token(auth, site_url=site_url)
            header_name = str(auth.get("header_name") or "Authorization")
            if header_name.lower() == "authorization" and not token.lower().startswith("bearer "):
                token = f"Bearer {token}"
            headers[header_name] = token
            auth_header_names.add(header_name.lower())

    call_headers = call.get("headers")
    if isinstance(call_headers, dict):
        sanitized = _sanitize_call_headers(call_headers, protected_headers=auth_header_names)
        merged = dict(sanitized)
        merged.update(headers)
        headers = merged
    return headers


_BEARER_API_TOKEN_CACHE: dict[str, str] = {}


def _resolve_bearer_token(auth: dict[str, Any], *, site_url: str = "") -> str:
    inline = auth.get("token")
    if isinstance(inline, str) and inline.strip():
        return inline.strip()

    # Runtime token generation via token_generator or token_endpoint.
    # Delegates to auth_tokens.acquire_token which handles both named
    # generators (e.g. gitlab_pat) and endpoint-based acquisition.
    token_generator = auth.get("token_generator")
    token_endpoint = auth.get("token_endpoint")
    if (isinstance(token_generator, str) and token_generator.strip()) or (
        isinstance(token_endpoint, str) and token_endpoint.strip()
    ):
        from worldsim.auth_tokens import acquire_token

        cache_key_parts = [site_url]
        if isinstance(token_generator, str) and token_generator.strip():
            cache_key_parts.append(f"gen:{token_generator}")
        if isinstance(token_endpoint, str) and token_endpoint.strip():
            cache_key_parts.append(token_endpoint)
        credentials = auth.get("credentials", {})
        if isinstance(credentials, dict):
            cache_key_parts.append(credentials.get("username", ""))
        cache_key = "|".join(cache_key_parts)
        cached = _BEARER_API_TOKEN_CACHE.get(cache_key)
        if cached:
            return cached
        token_text = acquire_token(auth, site_url)
        _BEARER_API_TOKEN_CACHE[cache_key] = token_text
        return token_text

    token_source = auth.get("token_source")
    if isinstance(token_source, str) and token_source.strip():
        path = _resolve_token_source_path(token_source)
        token = path.read_text(encoding="utf-8").strip()
        if not token:
            raise RuntimeError(f"token_source {path} is empty")
        return token

    raise RuntimeError(
        "bearer_token auth requires token, token_source, token_generator, or token_endpoint"
    )


def _resolve_header_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        env_name = value.get("from_env")
        if isinstance(env_name, str) and env_name:
            resolved = os.environ.get(env_name)
            if not resolved:
                raise RuntimeError(f"required auth header env var {env_name!r} is not set")
            return resolved
    raise RuntimeError('auth header values must be strings or {"from_env": "VAR_NAME"}')


def _sanitize_call_headers(
    call_headers: dict[str, Any],
    *,
    protected_headers: set[str],
) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in call_headers.items():
        key_str = str(key)
        lowered = key_str.lower()
        if lowered in _BLOCKED_CALL_HEADER_NAMES or lowered in protected_headers:
            continue
        sanitized[key_str] = str(value)
    return sanitized


def _resolve_token_source_path(token_source: str) -> Path:
    path = Path(token_source).expanduser().resolve(strict=False)
    allowed_roots = _allowed_token_source_roots()
    if not any(path.is_relative_to(root) for root in allowed_roots):
        raise RuntimeError(
            "token_source must be under one of: "
            + ", ".join(str(root) for root in sorted(allowed_roots))
        )
    return path


def _allowed_token_source_roots() -> set[Path]:
    roots = {(Path.cwd() / "logs" / "phase_0d").resolve(strict=False)}
    state_dir = os.environ.get("WORLDSIM_STATE_DIR")
    if state_dir:
        roots.add((Path(state_dir).expanduser() / "phase_0d").resolve(strict=False))
    return roots


def _origin_for_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _normalize_delivery_path(path: str) -> str:
    return re.sub(r"/\{[^}/]+\}(?=/|$)", "/{id}", re.sub(r"/\d+(?=/|$)", "/{id}", path))


def _prepare_form_body(
    method: str,
    url: str,
    headers: dict[str, str],
    body_form: object,
    instance: dict[str, Any],
    session: requests.Session,
    *,
    force_refresh: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(body_form, dict):
        return None
    form_body = dict(body_form)
    if method not in _FORM_METHODS:
        return form_body

    token_name, token_value = _get_csrf_token(
        session,
        url,
        headers,
        instance,
        force_refresh=force_refresh,
    )
    if token_name and token_value:
        form_body[token_name] = token_value
    return form_body


def _get_csrf_token(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    instance: dict[str, Any],
    *,
    force_refresh: bool = False,
) -> tuple[str | None, str | None]:
    origin = _origin_for_url(url)
    cache_key = _csrf_cache_key(session, instance, url)
    session_cache = _CSRF_TOKEN_CACHE.setdefault(session, {})
    if not force_refresh and cache_key in session_cache:
        return session_cache[cache_key]

    for candidate_url in (url, origin):
        try:
            response = session.get(
                candidate_url,
                headers=headers,
                timeout=30,
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                continue
            response.raise_for_status()
        except requests.RequestException:
            continue
        token = _extract_csrf_token(response.text)
        if token != (None, None):
            session_cache[cache_key] = token
            return token

    return (None, None)


def _clear_cached_csrf_token(
    session: requests.Session,
    instance: dict[str, Any],
    url: str,
) -> None:
    cache_key = _csrf_cache_key(session, instance, url)
    session_cache = _CSRF_TOKEN_CACHE.get(session)
    if session_cache is not None:
        session_cache.pop(cache_key, None)


def _csrf_cache_key(
    session: requests.Session,
    instance: dict[str, Any],
    url: str,
) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url)
    normalized_path = _normalize_delivery_path(parsed.path or "/")
    query_suffix = f"?{parsed.query}" if parsed.query else ""
    return (
        str(instance.get("site_name", "")),
        f"{parsed.scheme}://{parsed.netloc}{normalized_path}{query_suffix}",
    )


def _extract_csrf_token(html: str) -> tuple[str | None, str | None]:
    for pattern in _CSRF_INPUT_PATTERNS[:1]:
        match = pattern.search(html)
        if match:
            return match.group(1), match.group(2)
    meta_match = _CSRF_INPUT_PATTERNS[1].search(html)
    if meta_match:
        # Read csrf-param meta tag for the correct POST parameter name.
        # Rails sets <meta name="csrf-param" content="authenticity_token">.
        param_match = _CSRF_PARAM_META.search(html)
        param_name = param_match.group(1) if param_match else "csrf_token"
        return param_name, meta_match.group(1)
    return (None, None)


def _verify_http_seed_postcondition(
    *,
    mechanism: str,
    call: dict[str, Any],
    instance: dict[str, Any],
    raw_path: str,
) -> None:
    site_profile = instance.get("site_profile")
    if not isinstance(site_profile, dict):
        raise RuntimeError(
            f"HTTP seed for {raw_path} requires instance['site_profile'] for postcondition verification"
        )

    surface_id = instance.get("seed_target_surface_id")
    matches = _matching_http_delivery_channels(
        site_profile,
        mechanism=mechanism,
        call=call,
        surface_id=surface_id if isinstance(surface_id, str) and surface_id else None,
    )
    if not matches:
        hint = f" on surface {surface_id!r}" if isinstance(surface_id, str) and surface_id else ""
        raise RuntimeError(
            f"HTTP seed for {raw_path} does not match any registered delivery channel{hint}"
        )
    if len(matches) > 1:
        surface_ids = sorted({str(surface.get("id", "?")) for surface, _ in matches})
        raise RuntimeError(
            f"HTTP seed for {raw_path} matches multiple delivery channels: {surface_ids}"
        )

    surface, entry = matches[0]
    postcondition = entry.get("postcondition")
    if not isinstance(postcondition, dict):
        raise RuntimeError(
            f"HTTP seed for {raw_path} is missing delivery_channels postcondition metadata"
        )

    _verify_db_row_value_postcondition(
        postcondition=postcondition,
        entry=entry,
        call=call,
        instance=instance,
        raw_path=raw_path,
        surface_id=str(surface.get("id", "?")),
    )


def _matching_http_delivery_channels(
    site_profile: dict[str, Any],
    *,
    mechanism: str,
    call: dict[str, Any],
    surface_id: str | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    surfaces = site_profile.get("injection_surface")
    if not isinstance(surfaces, list):
        return matches
    for surface in surfaces:
        if not isinstance(surface, dict):
            continue
        if surface_id is not None and surface.get("id") != surface_id:
            continue
        for entry in surface.get("delivery_channels", []):
            if not isinstance(entry, dict):
                continue
            if _entry_matches_http_call(entry, mechanism=mechanism, call=call):
                matches.append((surface, entry))
    return matches


def _entry_matches_http_call(
    entry: dict[str, Any],
    *,
    mechanism: str,
    call: dict[str, Any],
) -> bool:
    if entry.get("mechanism") != mechanism:
        return False
    entry_method = entry.get("method")
    entry_path_template = entry.get("path_template")
    body_field = entry.get("body_field")
    call_method = call.get("method")
    call_path = call.get("path")
    if (
        not isinstance(entry_method, str)
        or not isinstance(entry_path_template, str)
        or not isinstance(body_field, str)
        or not isinstance(call_method, str)
        or not isinstance(call_path, str)
    ):
        return False
    if entry_method.strip().upper() != call_method.strip().upper():
        return False
    parsed_path = urllib.parse.urlparse(call_path).path
    if _normalize_delivery_path(entry_path_template) != _normalize_delivery_path(parsed_path):
        return False
    return body_field in _extract_http_body(call)


def _extract_http_body(call: dict[str, Any]) -> dict[str, Any]:
    for body_key in ("body_form", "body"):
        body = call.get(body_key)
        if isinstance(body, dict):
            nested_review = body.get("review")
            if isinstance(nested_review, dict):
                return nested_review
            return body
    return {}


def _verify_db_row_value_postcondition(
    *,
    postcondition: dict[str, Any],
    entry: dict[str, Any],
    call: dict[str, Any],
    instance: dict[str, Any],
    raw_path: str,
    surface_id: str,
) -> None:
    if postcondition.get("type") != "db_row_value":
        raise RuntimeError(
            f"HTTP seed for {raw_path} on surface {surface_id!r} uses unsupported "
            f"postcondition type {postcondition.get('type')!r}"
        )

    # Graceful skip when the instance has no db_connection. The HTTP 2xx
    # response already confirmed the seed was accepted by the server. DB
    # verification is a bonus integrity check, not a hard requirement. This
    # covers sites where the database port is internal to the container
    # (e.g. GitLab PostgreSQL on a Unix socket).
    db_connection = instance.get("db_connection")
    if not db_connection:
        logger.warning(
            "Skipping db_row_value postcondition for %s on surface %r: "
            "no db_connection configured on instance",
            raw_path,
            surface_id,
        )
        return

    table = postcondition.get("table")
    value_column = postcondition.get("value_column")
    where = postcondition.get("where")
    path_template = entry.get("path_template")
    body_field = entry.get("body_field")
    body = _extract_http_body(call)
    if not isinstance(body_field, str) or body_field not in body:
        raise RuntimeError(
            f"HTTP seed for {raw_path} on surface {surface_id!r} is missing body field "
            f"{body_field!r} required for postcondition verification"
        )
    if (
        not isinstance(table, str)
        or not _IDENTIFIER_PATTERN.match(table)
        or not isinstance(value_column, str)
        or not _IDENTIFIER_PATTERN.match(value_column)
        or not isinstance(where, dict)
        or not where
    ):
        raise RuntimeError(
            f"HTTP seed for {raw_path} on surface {surface_id!r} has invalid postcondition metadata"
        )

    path_params = _extract_path_params(path_template, call.get("path"))
    predicates: list[tuple[str, Any]] = []
    for where_column, source in where.items():
        if not isinstance(where_column, str) or not _IDENTIFIER_PATTERN.match(where_column):
            raise RuntimeError(
                f"HTTP seed for {raw_path} on surface {surface_id!r} has invalid "
                f"postcondition where column {where_column!r}"
            )
        predicates.append(
            (
                where_column,
                _resolve_postcondition_source(
                    source,
                    body=body,
                    path_params=path_params,
                    raw_path=raw_path,
                    surface_id=surface_id,
                ),
            )
        )

    expected_value = body[body_field]
    actual_values = _select_db_values(
        db_connection=instance.get("db_connection"),
        table=table,
        value_column=value_column,
        predicates=predicates,
    )
    if not any(_values_equal(actual, expected_value) for actual in actual_values):
        raise RuntimeError(
            f"HTTP seed for {raw_path} on surface {surface_id!r} did not satisfy "
            f"postcondition: expected {table}.{value_column}={expected_value!r} "
            f"for selectors {dict(predicates)!r}, got {actual_values[:5]!r}"
        )


def _extract_path_params(path_template: object, actual_path: object) -> dict[str, str]:
    if not isinstance(path_template, str) or not isinstance(actual_path, str):
        return {}
    template_path = urllib.parse.urlparse(path_template).path
    parsed_path = urllib.parse.urlparse(actual_path).path
    pattern_parts: list[str] = []
    last_index = 0
    param_names: list[str] = []
    for match in _PATH_PARAM_PATTERN.finditer(template_path):
        pattern_parts.append(re.escape(template_path[last_index : match.start()]))
        pattern_parts.append(r"([^/]+)")
        param_names.append(match.group(1))
        last_index = match.end()
    pattern_parts.append(re.escape(template_path[last_index:]))
    match = re.match("^" + "".join(pattern_parts) + "$", parsed_path)
    if match is None:
        return {}
    return {name: value for name, value in zip(param_names, match.groups(), strict=False)}


def _resolve_postcondition_source(
    source: object,
    *,
    body: dict[str, Any],
    path_params: dict[str, str],
    raw_path: str,
    surface_id: str,
) -> Any:
    if not isinstance(source, dict) or len(source) != 1:
        raise RuntimeError(
            f"HTTP seed for {raw_path} on surface {surface_id!r} has malformed postcondition source"
        )
    source_key, source_value = next(iter(source.items()))
    if source_key == "path_param":
        if not isinstance(source_value, str) or source_value not in path_params:
            raise RuntimeError(
                f"HTTP seed for {raw_path} on surface {surface_id!r} references missing "
                f"path_param {source_value!r} in postcondition"
            )
        return path_params[source_value]
    if source_key == "body_field":
        if not isinstance(source_value, str) or source_value not in body:
            raise RuntimeError(
                f"HTTP seed for {raw_path} on surface {surface_id!r} references missing "
                f"body_field {source_value!r} in postcondition"
            )
        return body[source_value]
    if source_key == "literal":
        return source_value
    raise RuntimeError(
        f"HTTP seed for {raw_path} on surface {surface_id!r} uses unsupported "
        f"postcondition source {source_key!r}"
    )


def _select_db_values(
    *,
    db_connection: Any,
    table: str,
    value_column: str,
    predicates: list[tuple[str, Any]],
) -> list[Any]:
    parsed = _parse_runtime_db_connection(
        db_connection,
        purpose="HTTP seed postcondition requires instance['db_connection']",
    )
    conn = _connect_db(parsed)
    try:
        scheme = parsed.scheme.lower()
        _configure_read_only_connection(conn, scheme)
        quoted_table = _quote_identifier(table, scheme)
        quoted_value_column = _quote_identifier(value_column, scheme)
        where_clause = " AND ".join(
            f"{_quote_identifier(column, scheme)} = %s" for column, _ in predicates
        )
        query = f"SELECT {quoted_value_column} FROM {quoted_table} WHERE {where_clause} LIMIT 5"
        with conn.cursor() as cursor:
            cursor.execute(query, [value for _, value in predicates])
            rows = cursor.fetchall()
    except Exception as exc:
        raise RuntimeError(f"HTTP seed postcondition query failed: {exc}") from exc
    finally:
        try:
            conn.rollback()
        except Exception:
            logger.debug("Failed to rollback postcondition verification connection", exc_info=True)
        conn.close()

    values: list[Any] = []
    for row in rows:
        if isinstance(row, (list, tuple)) and row:
            values.append(row[0])
        else:
            values.append(row)
    return values


def _connect_db(parsed: urllib.parse.ParseResult) -> Any:
    scheme = parsed.scheme.lower()
    if scheme == "mysql":
        import pymysql

        return pymysql.connect(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password,
            database=(parsed.path or "").lstrip("/"),
        )
    if scheme in ("postgresql", "postgres"):
        import psycopg2

        return psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            dbname=(parsed.path or "").lstrip("/"),
        )
    raise RuntimeError(f"unsupported DB dialect for HTTP seed verification: {scheme}")


def _configure_read_only_connection(conn: Any, scheme: str) -> None:
    try:
        if hasattr(conn, "autocommit"):
            conn.autocommit = False
        with conn.cursor() as cursor:
            if scheme == "mysql":
                cursor.execute("SET SESSION TRANSACTION READ ONLY")
                cursor.execute("START TRANSACTION READ ONLY")
            elif scheme in ("postgresql", "postgres"):
                cursor.execute("BEGIN")
                cursor.execute("SET TRANSACTION READ ONLY")
            else:
                raise RuntimeError(f"unsupported DB dialect: {scheme}")
    except Exception as exc:
        raise RuntimeError("could not enable read-only transaction guard") from exc


def _quote_identifier(identifier: str, scheme: str) -> str:
    if not _IDENTIFIER_PATTERN.match(identifier):
        raise RuntimeError(f"invalid SQL identifier {identifier!r}")
    quote = "`" if scheme == "mysql" else '"'
    return ".".join(f"{quote}{part}{quote}" for part in identifier.split("."))


def _values_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(actual, bytes) and isinstance(expected, str):
        return actual.decode("utf-8", errors="replace") == expected
    return False


def _request_with_context(
    session: requests.Session,
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    json_body: Any,
    form_body: dict[str, Any] | None,
    instance: dict[str, Any],
    raw_path: str,
) -> requests.Response:
    try:
        response = session.request(
            method,
            url,
            headers=headers,
            json=json_body,
            data=form_body,
            timeout=30,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            location = "<present>" if response.headers.get("Location") else "<missing>"
            raise RuntimeError(
                f"HTTP seed request for {method} {raw_path} returned redirect "
                f"status={response.status_code} location={location!r}"
            )
        return response
    except requests.RequestException as exc:
        site_name = instance.get("site_name", "<unknown>")
        raise RuntimeError(
            f"HTTP seed request failed for site {site_name!r} {method} {raw_path}: {exc}"
        ) from exc
