"""Seed runtime configuration validation."""

from __future__ import annotations

from typing import Any

from warp_taskgen.seeding.db import _resolve_token_source_path
from warp_taskgen.seeding.editor_args import _editor_delivery_key
from warp_taskgen.seeding.validation import validate_data_seed


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
        if mechanism in (None, "none") and not seed.get("editor_calls"):
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

        required_http_mechanisms = _seed_required_http_mechanisms(seed)
        for instance in site_instances:
            site_url = _instance_value(instance, "site_url") or "<unknown>"
            for effective_mechanism in required_http_mechanisms:
                auth_error = _instance_http_seed_auth_runtime_error(
                    instance,
                    mechanism=effective_mechanism,
                )
                if auth_error is not None:
                    _append_runtime_error(
                        errors,
                        seen,
                        f"site {seed_site!r} has {effective_mechanism} HTTP-seeded task(s) but instance {site_url!r} "
                        f"has invalid auth config: {auth_error}",
                    )

    return errors


def _seed_required_http_mechanisms(seed: dict[str, Any]) -> list[str]:
    required: set[str] = set()
    for call in seed.get("editor_calls", []):
        editor_mechanism = _editor_call_http_mechanism(call)
        if editor_mechanism is not None:
            required.add(editor_mechanism)
    return sorted(required)


def _editor_call_http_mechanism(call: Any) -> str | None:
    if not isinstance(call, dict):
        return None
    site_name = str(call.get("site") or "").strip().lower()
    method_name = str(call.get("method") or "").strip()
    if not site_name or not method_name:
        return None
    if site_name == "reddit" or (site_name, method_name) in {
        ("shopping", "update_customer_profile"),
        ("shopping_admin", "update_admin_profile"),
    }:
        return "form"
    return "api"


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


def _instance_http_seed_auth_runtime_error(instance: Any, *, mechanism: str = "form") -> str | None:
    from warp_taskgen.auth_tokens import _resolve_header_value, pick_auth_lane

    auth = pick_auth_lane(
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
        from warp_taskgen.auth_tokens import _token_strategy, bearer_token_config_error

        strategy = _token_strategy(auth)
        token_source = auth.get("token_source")
        if strategy == "token_source" and isinstance(token_source, str) and token_source.strip():
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

        config_error = bearer_token_config_error(auth)
        if config_error is not None:
            return config_error
        return None

    if auth_type == "web_login":
        credentials = auth.get("credentials")
        if not isinstance(credentials, dict) or not credentials:
            return "web_login auth requires a non-empty credentials dict"
        return None

    return None


__all__ = [
    "_append_runtime_error",
    "_editor_call_http_mechanism",
    "_editor_delivery_key",
    "_instance_http_seed_auth_runtime_error",
    "_instance_value",
    "_seed_required_http_mechanisms",
    "_task_seed_site",
    "collect_seed_runtime_errors",
]
