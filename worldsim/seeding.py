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
from pathlib import Path
from typing import Any

import requests

from worldsim.benchmark_capabilities import infer_benchmark_name, normalize_benchmark_name
from worldsim.db_urls import parse_supported_db_connection
from worldsim.editors import EDITOR_REGISTRY, EditorError
from worldsim.placeholders import apply_placeholders, merge_placeholder_maps

logger = logging.getLogger(__name__)

# Destructive (DELETE) and probing (HEAD, OPTIONS) HTTP verbs are blocked:
# data seeding must stay within verbs a regular authenticated user would emit
# via the site's forms or API, per the threat model.
_ALLOWED_API_METHODS = frozenset({"GET", "POST", "PUT", "PATCH"})
_FORM_METHODS = frozenset({"POST", "PUT", "PATCH"})
_CSRF_TOKEN_CACHE: weakref.WeakKeyDictionary[
    requests.Session,
    dict[tuple[str, str], tuple[str | None, str | None]],
] = weakref.WeakKeyDictionary()
_REDDIT_TABLE_NAME_CACHE: dict[tuple[str, str], str] = {}
_REDDIT_COMMENT_BODY_FIELD_PATTERN = re.compile(
    r"^reply_to_submission_(?:\{[^}\]]+\}|[^[]+)\[comment\]$"
)
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


def _response_redirect_path(response: requests.Response) -> str:
    location = response.headers.get("Location")
    if not isinstance(location, str) or not location.strip():
        return ""
    return (urllib.parse.urlparse(location).path or "").strip().lower()


def _response_redirects_to_login(response: requests.Response) -> bool:
    path = _response_redirect_path(response)
    if not path:
        return False
    return any(
        token in path
        for token in (
            "/login",
            "/sign_in",
            "/users/sign_in",
            "/session",
        )
    )


class SeedCleanupHandle:
    def __init__(
        self,
        *,
        session: requests.Session,
        editor_instances: dict[tuple[str, str], Any],
    ) -> None:
        self._session = session
        self._editor_instances = editor_instances
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        failures: list[str] = []
        try:
            for editor in reversed(list(self._editor_instances.values())):
                try:
                    editor.cleanup()
                except Exception as exc:
                    logger.exception("seed editor cleanup failed")
                    failures.append(str(exc) or exc.__class__.__name__)
        finally:
            self._session.close()
            self._cleaned = True
        if failures:
            raise RuntimeError("seed cleanup failed: " + "; ".join(failures))


def seed_has_actions(seed: Any) -> bool:
    if not isinstance(seed, dict):
        return False
    mechanism = seed.get("mechanism")
    if mechanism == "state_push":
        return "state" in seed
    editor_calls = seed.get("editor_calls")
    if isinstance(editor_calls, list) and editor_calls:
        return True
    api_calls = seed.get("api_calls")
    return bool(isinstance(api_calls, list) and api_calls)


def seed_requires_reset(seed: Any) -> bool:
    if not isinstance(seed, dict):
        return False
    if seed.get("mechanism") == "state_push":
        return "state" in seed
    editor_calls = seed.get("editor_calls")
    if isinstance(editor_calls, list) and editor_calls:
        return True
    api_calls = seed.get("api_calls")
    return bool(isinstance(api_calls, list) and api_calls)


def self_contained_adversarial_seed_error(benign_seed: Any, adversarial_seed: Any) -> str | None:
    """Return an error when an adversarial seed drops benign setup state.

    Phase 4 applies only ``adversarial_data_seed``, so the adversarial seed
    must preserve the benign seed verbatim before extending it.
    """
    if not isinstance(benign_seed, dict) or not isinstance(adversarial_seed, dict):
        return None

    if not seed_has_actions(benign_seed):
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
    editor_calls = seed.get("editor_calls")
    has_editor_calls = isinstance(editor_calls, list) and bool(editor_calls)
    if mechanism in (None, "none"):
        if has_editor_calls:
            _validate_editor_calls(editor_calls)
            return
        if allow_none:
            return
        raise ValueError("data seed must declare a non-empty mechanism")

    if mechanism == "editor":
        if not has_editor_calls:
            raise ValueError("editor data seed must include a non-empty editor_calls list")
        if seed.get("api_calls") is not None:
            raise ValueError("editor data seed must not include api_calls")
        _validate_editor_calls(editor_calls)
        return

    if mechanism in {"api", "form"}:
        api_calls = seed.get("api_calls")
        has_api_calls = isinstance(api_calls, list) and bool(api_calls)
        if not has_api_calls and not has_editor_calls:
            raise ValueError(
                f"{mechanism} data seed must include a non-empty api_calls or editor_calls list"
            )
        for call in api_calls or []:
            if not isinstance(call, dict):
                raise ValueError("api data seed calls must be objects")
            if "target" in call:
                raise ValueError(
                    "target-based api_calls are no longer supported; migrate to editor_calls"
                )
            method = call.get("method")
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
        if has_editor_calls:
            _validate_editor_calls(editor_calls)
        return

    if mechanism == "state_push":
        if "state" not in seed:
            raise ValueError("state_push data seed must include a state payload")
        return

    raise ValueError(f"unknown data seed mechanism: {mechanism!r}")


def _validate_editor_calls(editor_calls: Any) -> None:
    if not isinstance(editor_calls, list) or not editor_calls:
        raise ValueError("editor data seed must include a non-empty editor_calls list")
    for call in editor_calls:
        if not isinstance(call, dict):
            raise ValueError("editor_calls entries must be objects")
        benchmark = call.get("benchmark")
        if benchmark is not None and (not isinstance(benchmark, str) or not benchmark.strip()):
            raise ValueError("editor_calls benchmark must be a non-empty string when provided")
        site = call.get("site")
        method = call.get("method")
        args = call.get("args")
        if not isinstance(site, str) or not site.strip():
            raise ValueError("editor_calls entries must include site")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("editor_calls entries must include method")
        if not isinstance(args, dict):
            raise ValueError("editor_calls entries must include args as an object")
        method_name = method.strip()
        if method_name.startswith("_"):
            raise ValueError("editor_calls method must not be private")
        benchmark_key = normalize_benchmark_name(benchmark or "webarena_verified")
        editor_cls = EDITOR_REGISTRY.get((benchmark_key, site.strip().lower()))
        if editor_cls is not None and method_name not in editor_cls.supported_methods:
            raise ValueError(
                f"editor_calls method {method_name!r} is not supported for {(benchmark_key, site.strip().lower())!r}"
            )
        _validate_untrusted_selector_args(site.strip().lower(), args)


def apply_data_seed(
    seed: dict[str, Any], instance: dict[str, Any]
) -> tuple[SeedCleanupHandle | None, dict[str, Any]]:
    """Apply a data seed to a running benchmark instance.

    Returns a ``(cleanup_handle, metadata)`` tuple where ``metadata`` carries
    the editor-emitted read-surface URLs (C1b signal, see
    ``docs/handoffs/codex-handoff-c1-read-surface.md`` §5.4)::

        {
          "read_surface_urls": [...],  # deduped, first-occurrence order
          "read_surface_provenance": {"source": ..., "editor_method": ...},
        }

    Args:
        seed: Seed spec with a ``mechanism`` field and mechanism-specific
            extras. See the v5 spec for the field schemas.
        instance: Benchmark instance dict with ``site_url`` and any
            mechanism-specific auth configuration.

    Raises:
        ValueError: If ``seed["mechanism"]`` is unknown.
    """
    from worldsim.editors._read_surface import normalize_surface_urls

    validate_data_seed(seed)
    mechanism = seed.get("mechanism")
    if mechanism == "state_push":
        resp = requests.put(
            f"{instance['site_url']}/api/state",
            json=seed["state"],
            timeout=30,
        )
        resp.raise_for_status()
        return None, {}

    seed_context = _build_seed_context(seed, instance)
    editor_instances: dict[tuple[str, str], Any] = {}
    session = requests.Session()
    cleanup_handle: SeedCleanupHandle | None = None
    read_surface_accumulator: list[str] = []
    read_surface_provenance: dict[str, Any] = {}
    try:
        if mechanism in {"api", "form"}:
            _perform_web_login_if_needed(session, instance, mechanism)
            for call in seed.get("api_calls", []):
                rendered_call = _render_http_seed_call(call, seed_context=seed_context)
                response = _apply_legacy_http_seed_call(
                    session,
                    mechanism,
                    rendered_call,
                    instance,
                )
                _merge_seed_context(seed_context, _extract_response_seed_context(response))
        for call in seed.get("editor_calls", []):
            _apply_editor_seed_call(
                session,
                call,
                instance,
                seed_context=seed_context,
                editor_instances=editor_instances,
                read_surface_accumulator=read_surface_accumulator,
                read_surface_provenance=read_surface_provenance,
            )
        metadata: dict[str, Any] = {}
        # Handoff §5.5: task-author explicit override unions with editor
        # contributions. Explicit entries come first so their order is
        # preserved in the deduped result; the provenance source reflects
        # whether explicit, editor, or both contributed.
        explicit_override: list[str] = []
        seed_task = instance.get("seed_task")
        if isinstance(seed_task, dict):
            raw_override = seed_task.get("read_surface_urls")
            if isinstance(raw_override, list):
                explicit_override = [
                    str(u).strip() for u in raw_override if isinstance(u, str) and str(u).strip()
                ]
        editor_contribution = list(read_surface_accumulator)
        deduped = normalize_surface_urls(explicit_override + editor_contribution)
        if deduped:
            metadata["read_surface_urls"] = deduped
            if explicit_override and editor_contribution:
                source = "explicit_override+editor"
            elif explicit_override:
                source = "explicit_override"
            else:
                source = None
            if source is not None:
                # Build / overlay provenance. If editors also stamped, keep
                # their editor_method attribution; only replace source.
                provenance = dict(read_surface_provenance) if read_surface_provenance else {}
                provenance["source"] = source
                if "captured_at" not in provenance:
                    from datetime import UTC, datetime

                    provenance["captured_at"] = datetime.now(UTC).isoformat()
                metadata["read_surface_provenance"] = provenance
            elif read_surface_provenance:
                metadata["read_surface_provenance"] = read_surface_provenance
        # Hoist authoritative write-identifier tokens from the merged
        # seed_context into metadata so downstream verifiers (render-check
        # read-your-write fastpath) can match server-reported IDs instead
        # of racing the DOM hydration cascade.
        for token_key in ("note_id", "comment_id", "submission_id", "review_id"):
            token_value = seed_context.get(token_key)
            if token_value not in (None, ""):
                metadata[token_key] = token_value
        if editor_instances:
            cleanup_handle = SeedCleanupHandle(
                session=session,
                editor_instances=editor_instances,
            )
            return cleanup_handle, metadata
        session.close()
        return None, metadata
    except Exception:
        if cleanup_handle is not None:
            cleanup_handle.cleanup()
        else:
            for editor in reversed(list(editor_instances.values())):
                editor.cleanup()
            session.close()
        raise


async def apply_data_seed_async(
    seed: dict[str, Any], instance: dict[str, Any]
) -> tuple[SeedCleanupHandle | None, dict[str, Any]]:
    """Apply a data seed without blocking the event loop."""
    return await asyncio.to_thread(apply_data_seed, seed, instance)


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

        # Option A placement anchors (docs/handoffs/phase-2-placement-systemic-gap.md).
        # The 2a prompt emits {benign_*} tokens in editor_calls[].args so the
        # seed attaches to the existing benign-task resource. Surface the
        # concrete values here so _FORMAT_TOKEN_PATTERN substitution resolves
        # them at seed-apply time.
        anchors = (task.get("benign_target_resource") or {}).get("anchors") or {}
        if isinstance(anchors, dict):
            if anchors.get("project_id") is not None:
                context["benign_project_id"] = str(anchors["project_id"])
            if anchors.get("project_path"):
                context["benign_project_path"] = str(anchors["project_path"])
            if anchors.get("issue_iid") is not None:
                context["benign_issue_iid"] = str(anchors["issue_iid"])
            if anchors.get("mr_iid") is not None:
                context["benign_mr_iid"] = str(anchors["mr_iid"])
            if anchors.get("submission_id") is not None:
                context["benign_submission_id"] = str(anchors["submission_id"])
            if anchors.get("forum_name"):
                context["benign_forum_name"] = str(anchors["forum_name"])
        # benign user handle — needed for dashboard-list @mention routing
        # when encounter_requirements.requires_at_mention is set.
        agent_ctx = task.get("agent_context") or {}
        auth = agent_ctx.get("authentication") or {}
        creds = auth.get("credentials") if isinstance(auth, dict) else None
        handle = creds.get("username") if isinstance(creds, dict) else None
        if isinstance(handle, str) and handle.strip():
            context["benign_user_handle"] = handle.strip()

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
    """Resolve legacy direct HTTP calls without firing mutations."""
    validate_data_seed(seed, allow_none=False)
    mechanism = str(seed.get("mechanism", "")).strip().lower()
    if mechanism not in {"api", "form"}:
        return []

    seed_context = _build_seed_context(seed, instance)
    errors: list[str] = []
    for index, call in enumerate(seed.get("api_calls", [])):
        if not isinstance(call, dict):
            continue
        if _seed_placeholder_names(call):
            continue
        try:
            rendered_call = _render_http_seed_call(call, seed_context=seed_context)
            raw_ref = _call_reference(rendered_call)
            if raw_ref is None:
                raise RuntimeError("rendered legacy seed call must include path or url")
            _resolve_call_url(raw_ref, instance)
        except Exception as exc:
            errors.append(f"api_calls[{index}]: {exc}")
    return errors


def preflight_editor_seed_calls(
    seed: dict[str, Any],
    instance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Render and validate editor calls without firing mutations."""
    validate_data_seed(seed, allow_none=False)
    seed_context = _build_seed_context(seed, instance)
    errors: list[dict[str, Any]] = []
    with requests.Session() as session:
        editor_instances: dict[tuple[str, str], Any] = {}
        try:
            for index, call in enumerate(seed.get("editor_calls", [])):
                if not isinstance(call, dict):
                    continue
                try:
                    rendered = _render_editor_seed_call(call, seed_context)
                    editor = _get_editor_for_seed_call(
                        rendered,
                        instance,
                        session=session,
                        editor_instances=editor_instances,
                    )
                    method_name = rendered["method"]
                    args = rendered["args"]
                    editor.validate_args(method_name, args)
                    preview = editor.preview_context(method_name, args)
                    if isinstance(preview, dict):
                        _merge_seed_context(seed_context, preview)
                except EditorError as exc:
                    errors.append(
                        {
                            "call_index": index,
                            "site": str(call.get("site", "")).strip() or "unknown",
                            "kind": exc.kind,
                            "detail": exc.detail,
                            "method": str(call.get("method", "")).strip() or "unknown",
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "call_index": index,
                            "site": str(call.get("site", "")).strip() or "unknown",
                            "kind": "editor_error",
                            "detail": str(exc),
                            "method": str(call.get("method", "")).strip() or "unknown",
                        }
                    )
        finally:
            for editor in reversed(list(editor_instances.values())):
                editor.cleanup()
    return errors


class UnboundTokenError(ValueError):
    """Raised at seed-apply time when a ``{benign_<anchor_key>}`` token
    is referenced but not reachable via the task's
    ``benign_target_resource.anchors`` — the "silently renders empty"
    failure mode commit 4's registry validator now catches at Phase 2a,
    and which this check guarantees at Phase 2c.

    Phase 2c categorizes this as ``error.kind = "contract_violation"``,
    distinct from ``schema_mismatch`` (shape/type violations).
    """

    def __init__(
        self,
        *,
        task_id: str,
        kind: str,
        token: str,
        available_tokens: frozenset[str],
        anchors: dict[str, Any],
    ) -> None:
        msg = (
            f"Phantom token {token!r} in seed for task {task_id!r} "
            f"(kind={kind!r}): not reachable via anchors. "
            f"Available: {sorted(available_tokens)}; anchors: {dict(anchors)}"
        )
        super().__init__(msg)
        self.task_id = task_id
        self.kind = kind
        self.token = token
        self.available_tokens = frozenset(available_tokens)
        self.anchors = dict(anchors)


def _collect_benign_tokens(value: Any) -> set[str]:
    """Walk ``value`` and collect every ``{benign_<x>}`` token it
    references. Used by :func:`_assert_benign_tokens_bound` to pre-check
    seed calls before substitution."""
    tokens: set[str] = set()
    if isinstance(value, str):
        for match in _FORMAT_TOKEN_PATTERN.finditer(value):
            key = match.group(1)
            if key.startswith("benign_"):
                tokens.add(f"{{{key}}}")
    elif isinstance(value, dict):
        for k, v in value.items():
            tokens |= _collect_benign_tokens(k)
            tokens |= _collect_benign_tokens(v)
    elif isinstance(value, list):
        for item in value:
            tokens |= _collect_benign_tokens(item)
    return tokens


def _assert_benign_tokens_bound(value: Any, task: Any) -> None:
    """Raise :class:`UnboundTokenError` if ``value`` references any
    ``{benign_<x>}`` token not in the contract's
    :func:`available_tokens_for_kind` for the task's kind + anchors.

    No-op when the task lacks a ``benign_target_resource`` with a
    non-null kind (legacy tasks, pending-L3 records). The Option A
    validator rejects null-kind tasks at Phase 2a; feasibility still
    benefits from the check because it catches cases where Phase 2a
    regeneration hasn't happened yet.
    """
    if not isinstance(task, dict):
        return
    resource = task.get("benign_target_resource")
    if not isinstance(resource, dict):
        return
    kind = resource.get("kind")
    if not isinstance(kind, str) or not kind:
        return
    anchors_raw = resource.get("anchors")
    anchors = anchors_raw if isinstance(anchors_raw, dict) else {}

    tokens_referenced = _collect_benign_tokens(value)
    if not tokens_referenced:
        return

    # Lazy import — worldsim.editors package pulls in requests etc;
    # defer until a seed call actually needs the contract check.
    from worldsim.editors._registry import available_tokens_for_kind

    site = str(task.get("site") or "").strip().lower() or None
    try:
        benchmark = _infer_task_benchmark(task)
    except ValueError as exc:
        raise ValueError(f"seed token contract benchmark metadata is invalid: {exc}") from exc
    available = available_tokens_for_kind(kind, anchors, benchmark=benchmark, site=site)
    for token in sorted(tokens_referenced):
        if token not in available:
            task_id = task.get("id") or task.get("benign_task_id") or "unknown"
            raise UnboundTokenError(
                task_id=str(task_id),
                kind=kind,
                token=token,
                available_tokens=available,
                anchors=anchors,
            )


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


def _render_editor_seed_call(call: dict[str, Any], seed_context: dict[str, Any]) -> dict[str, Any]:
    rendered = _render_seed_value(call, seed_context)
    if not isinstance(rendered, dict):
        raise RuntimeError("rendered editor call must be an object")
    # Rename LLM-facing arg names to Python-facing names before the
    # editor receives them. The contract registry's bindings use
    # LLM-facing names (e.g., ``body`` for gitlab.create_issue_note)
    # because that's what the Phase 2a prompt documents and the Option A
    # validator checks, but editor method signatures use Python-facing
    # names (e.g., ``note_body``). The alias table lives in
    # :func:`_editor_arg_name`; applying it here keeps the rest of the
    # editor-call pipeline free of the LLM/Python naming split.
    args = rendered.get("args")
    if isinstance(args, dict) and args:
        renamed: dict[str, Any] = {}
        for key, value in args.items():
            canonical = str(key)
            alias = _editor_arg_name(rendered, canonical)
            target = alias if alias else canonical
            renamed[target] = value
        rendered["args"] = renamed
    unresolved = sorted(_seed_placeholder_names(rendered.get("args", {})))
    if unresolved:
        raise RuntimeError(
            "editor call has unresolved template placeholders: " + ", ".join(unresolved)
        )
    return rendered


def _get_editor_for_seed_call(
    call: dict[str, Any],
    instance: dict[str, Any],
    *,
    session: requests.Session,
    editor_instances: dict[tuple[str, str], Any],
) -> Any:
    benchmark = _infer_editor_call_benchmark(call, instance)
    site = str(call.get("site") or instance.get("site_name") or "").strip().lower()
    instance_site = str(instance.get("site_name") or "").strip().lower()
    if site and instance_site and site != instance_site:
        raise EditorError(
            "site_mismatch",
            f"editor call site {site!r} does not match bound seed instance site {instance_site!r}",
        )
    key = (benchmark, site)
    editor = editor_instances.get(key)
    if editor is not None:
        return editor
    editor_cls = EDITOR_REGISTRY.get(key)
    if editor_cls is None:
        raise EditorError(
            "unsupported_site",
            f"no editor registered for benchmark={benchmark!r} site={site!r}",
        )
    editor = editor_cls(instance, session)
    editor_instances[key] = editor
    return editor


def _infer_editor_call_benchmark(call: dict[str, Any], instance: dict[str, Any]) -> str:
    try:
        benchmark = infer_benchmark_name(
            (
                call.get("benchmark"),
                call.get("benchmark_name"),
                call.get("benchmark_adapter"),
                instance.get("benchmark"),
                instance.get("benchmark_name"),
                instance.get("benchmark_adapter"),
            )
        )
    except ValueError as exc:
        raise EditorError("benchmark_mismatch", str(exc)) from exc
    if benchmark is not None:
        return benchmark
    return normalize_benchmark_name("webarena_verified")


def _infer_task_benchmark(task: dict[str, Any]) -> str:
    values: list[Any] = [
        task.get("benchmark"),
        task.get("benchmark_name"),
        task.get("benchmark_adapter"),
    ]
    seed = task.get("adversarial_data_seed")
    calls = seed.get("editor_calls") if isinstance(seed, dict) else None
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                values.extend(
                    (
                        call.get("benchmark"),
                        call.get("benchmark_name"),
                        call.get("benchmark_adapter"),
                    )
                )
    benchmark = infer_benchmark_name(values)
    return benchmark or normalize_benchmark_name("webarena_verified")


def _apply_editor_seed_call(
    session: requests.Session,
    call: dict[str, Any],
    instance: dict[str, Any],
    *,
    seed_context: dict[str, Any],
    editor_instances: dict[tuple[str, str], Any],
    read_surface_accumulator: list[str] | None = None,
    read_surface_provenance: dict[str, Any] | None = None,
) -> None:
    from datetime import UTC, datetime

    # Fail-loud: reject the call if it references a {benign_*} token that
    # the resolver's anchors don't support. Catches plans that pass the
    # legacy Option A validator (which only checks the innermost anchor)
    # but would render an empty string at substitution time.
    _assert_benign_tokens_bound(call, instance.get("seed_task"))

    rendered = _render_editor_seed_call(call, seed_context)
    editor = _get_editor_for_seed_call(
        rendered,
        instance,
        session=session,
        editor_instances=editor_instances,
    )
    method_name = str(rendered["method"]).strip()
    args = rendered["args"]
    if method_name.startswith("_") or method_name not in editor.supported_methods:
        raise EditorError(
            "unsupported_method",
            f"{editor.site_name} editor does not support method {method_name!r}",
        )
    editor.validate_args(method_name, args)
    editor_method = getattr(editor, method_name, None)
    if not callable(editor_method):
        raise EditorError(
            "unsupported_method",
            f"{editor.site_name} editor does not support method {method_name!r}",
        )
    result = editor_method(**args)
    if isinstance(result, dict):
        # C1b read-surface URLs must NOT round-trip through seed_context
        # (namespace-flat; multi-call seeds would clobber each other — §12.9).
        surface_urls = result.get("read_surface_urls")
        if read_surface_accumulator is not None and isinstance(surface_urls, list):
            for url in surface_urls:
                if isinstance(url, str) and url.strip():
                    read_surface_accumulator.append(url.strip())
        if read_surface_provenance is not None and isinstance(surface_urls, list) and surface_urls:
            # Handoff §12.9: multi-call seeds (e.g. gitlab.create_project +
            # gitlab.create_issue) each contribute a method. Accumulate the
            # methods as a list (first-occurrence order, deduped); keep the
            # most-specific source seen so far (api_response beats
            # constructed); stamp captured_at only once on first contribution.
            provenance_source = str(
                result.get("read_surface_provenance_source") or "editor_api_response"
            )
            editor_method_str = f"{editor.site_name}.{method_name}"
            if not read_surface_provenance:
                read_surface_provenance.update(
                    {
                        "source": provenance_source,
                        "editor_method": [editor_method_str],
                        "captured_at": datetime.now(UTC).isoformat(),
                    }
                )
            else:
                methods = read_surface_provenance.get("editor_method")
                if not isinstance(methods, list):
                    methods = [str(methods)] if methods else []
                    read_surface_provenance["editor_method"] = methods
                if editor_method_str not in methods:
                    methods.append(editor_method_str)
                # api_response is the stronger claim — prefer it over constructed.
                current_source = read_surface_provenance.get("source")
                if (
                    current_source == "editor_constructed"
                    and provenance_source == "editor_api_response"
                ):
                    read_surface_provenance["source"] = provenance_source
        # Strip C1b-only keys before merging into seed_context so they do not
        # surface as placeholder values to later calls.
        if surface_urls is not None or "read_surface_provenance_source" in result:
            sanitized = {
                k: v
                for k, v in result.items()
                if k not in {"read_surface_urls", "read_surface_provenance_source"}
            }
            if sanitized:
                _merge_seed_context(seed_context, sanitized)
        else:
            _merge_seed_context(seed_context, result)


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
    if _response_redirects_to_login(post_resp):
        raise RuntimeError(
            f"Web login failed for {instance.get('site_name', '?')}: redirected back to login"
        )
    if post_resp.status_code == 200 and _looks_like_login_page(post_resp.text):
        raise RuntimeError(
            f"Web login failed for {instance.get('site_name', '?')}: login form was re-rendered"
        )

    validation_endpoint = auth.get("validation_endpoint")
    if isinstance(validation_endpoint, str) and validation_endpoint.strip():
        validation_url = f"{site_url}{validation_endpoint.strip()}"
        validation_resp = session.get(validation_url, timeout=30, allow_redirects=False)
        if validation_resp.status_code in {401, 403}:
            raise RuntimeError(
                f"Web login failed for {instance.get('site_name', '?')}: "
                f"validation endpoint returned HTTP {validation_resp.status_code}"
            )
        if _response_redirects_to_login(validation_resp):
            raise RuntimeError(
                f"Web login failed for {instance.get('site_name', '?')}: "
                "validation endpoint redirected to login"
            )
        if validation_resp.status_code == 200 and _looks_like_login_page(validation_resp.text):
            raise RuntimeError(
                f"Web login failed for {instance.get('site_name', '?')}: "
                "validation endpoint still served the login page"
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
    mechanism = seed.get("mechanism")
    if mechanism in {"api", "form"}:
        required.add(str(mechanism))
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


def _editor_delivery_key(call: Any) -> tuple[str, str] | None:
    if not isinstance(call, dict):
        return None
    site_name = str(call.get("site") or "").strip().lower()
    method_name = str(call.get("method") or "").strip()
    if not site_name or not method_name or not isinstance(call.get("args"), dict):
        return None
    return (site_name, method_name)


def _editor_arg_name(call: Any, canonical_name: str) -> str | None:
    editor_key = _editor_delivery_key(call)
    if editor_key == ("reddit", "create_comment") and _REDDIT_COMMENT_BODY_FIELD_PATTERN.match(
        canonical_name
    ):
        return "body"
    aliases = {
        ("gitlab", "create_issue"): {"body": "body_template", "description": "body_template"},
        ("gitlab", "create_issue_note"): {"body": "note_body"},
        ("gitlab", "create_mr"): {"body": "body_template", "description": "body_template"},
        ("gitlab", "create_mr_note"): {"body": "note_body"},
        ("gitlab", "update_user_status"): {"message": "message"},
        ("gitlab", "update_user_profile"): {
            "bio": "bio",
            "name": "name",
            "user[bio]": "bio",
            "user[name]": "name",
        },
        ("reddit", "create_submission"): {"title": "title_template", "body": "body_template"},
        ("reddit", "create_comment"): {"body": "body"},
        ("reddit", "update_user_bio"): {"bio": "bio_text"},
        ("shopping", "create_product_review"): {"detail": "detail", "title": "title"},
        ("shopping", "update_customer_profile"): {"value": "value"},
        ("shopping_admin", "update_admin_profile"): {"value": "value"},
    }.get(editor_key, {})
    arg_name = aliases.get(canonical_name)
    return str(arg_name) if isinstance(arg_name, str) else None


def _validate_untrusted_selector_args(site_name: str, args: dict[str, Any]) -> None:
    if site_name != "gitlab":
        return
    for selector in ("project_id", "issue_iid", "mr_iid"):
        value = args.get(selector)
        if value in (None, ""):
            continue
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            continue
        raise ValueError(
            f"editor_calls gitlab selector {selector!r} must come from prior seed context, not a literal value"
        )


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
        from worldsim.auth_tokens import _token_strategy, bearer_token_config_error

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


def _parse_runtime_db_connection(
    db_connection: Any,
    *,
    purpose: str,
) -> urllib.parse.ParseResult:
    try:
        return parse_supported_db_connection(db_connection, purpose=purpose)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _apply_legacy_http_seed_call(
    session: requests.Session,
    mechanism: str,
    call: dict[str, Any],
    instance: dict[str, Any],
) -> requests.Response:
    method = str(call["method"]).strip().upper()
    raw_path = _call_reference(call)
    if raw_path is None:
        raise RuntimeError("rendered legacy seed call must include path or url")
    url = _resolve_call_url(raw_path, instance)
    headers = _build_request_headers(instance, call, mechanism=mechanism)
    json_body = call.get("body")
    form_body = _prepare_form_body(method, url, headers, call.get("body_form"), instance, session)

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
            call.get("body_form"),
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


def _resolve_bearer_token(auth: dict[str, Any], *, site_url: str = "") -> str:
    from worldsim.auth_tokens import resolve_bearer_token

    return resolve_bearer_token(auth, site_url=site_url)


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


def _looks_like_login_page(html: str) -> bool:
    lowered = (html or "").lower()
    if not lowered:
        return False
    indicators = (
        "user[password]",
        'type="password"',
        'name="_password"',
        "input#login-password",
        "sign in",
        "log in",
    )
    return any(indicator in lowered for indicator in indicators)


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
    editor_args = call.get("args")
    if isinstance(editor_args, dict):
        return editor_args
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
                    call=call,
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
    call: dict[str, Any],
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
        resolved_field = source_value
        if isinstance(source_value, str) and source_value not in body:
            alias = _editor_arg_name(call, source_value)
            if alias and alias in body:
                resolved_field = alias
        if not isinstance(resolved_field, str) or resolved_field not in body:
            raise RuntimeError(
                f"HTTP seed for {raw_path} on surface {surface_id!r} references missing "
                f"body_field {source_value!r} in postcondition"
            )
        return body[resolved_field]
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
