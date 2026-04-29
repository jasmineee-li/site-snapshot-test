"""Data seeding dispatchers.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 3 / Evaluation
Infrastructure" section.

The only supported seed mechanism is ``editor``: each ``editor_calls`` entry
dispatches to a per-site editor method in ``worldsim/editors/`` which performs
the underlying HTTP write against ``instance["site_url"]``. ``mechanism: "none"``
is allowed for navigate-only tasks. ``api``, ``form``, and ``state_push`` were
deprecated in the editor migration and are rejected at the validator boundary;
see ``docs/handoffs/researcher-handoff-project-status.md``.

SQL seeding was evaluated and excluded from the methodology because it violates
the threat model (a regular authenticated user cannot write to the database
directly). Database read access is retained for postcondition verification and
reward evaluation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

import requests

from worldsim.benchmark_capabilities import infer_benchmark_name, normalize_benchmark_name
from worldsim.db_urls import parse_supported_db_connection
from worldsim.editors import EDITOR_REGISTRY, EditorError

logger = logging.getLogger(__name__)

# Destructive (DELETE) and probing (HEAD, OPTIONS) HTTP verbs are blocked:
# data seeding must stay within verbs a regular authenticated user would emit
# via the site's forms or API, per the threat model.
_ALLOWED_API_METHODS = frozenset({"GET", "POST", "PUT", "PATCH"})
_REDDIT_TABLE_NAME_CACHE: dict[tuple[str, str], str] = {}
_REDDIT_COMMENT_BODY_FIELD_PATTERN = re.compile(
    r"^reply_to_submission_(?:\{[^}\]]+\}|[^[]+)\[comment\]$"
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_PATH_PARAM_PATTERN = re.compile(r"\{([^}/]+)\}")
_UNRESOLVED_TEMPLATE_TOKEN = re.compile(r"\{[^}/]+\}")
_FORMAT_TOKEN_PATTERN = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_\.]*)\}(?!\})")


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
    editor_calls = seed.get("editor_calls")
    return bool(isinstance(editor_calls, list) and editor_calls)


def seed_requires_reset(seed: Any) -> bool:
    if not isinstance(seed, dict):
        return False
    editor_calls = seed.get("editor_calls")
    return bool(isinstance(editor_calls, list) and editor_calls)


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

    if mechanism in {"api", "form", "state_push"}:
        raise ValueError(
            f"data_seed.mechanism={mechanism!r} is deprecated; use mechanism='editor' "
            "with editor_calls referencing site editor methods. The api/form/state_push "
            "paths were removed in the editor migration; see "
            "docs/handoffs/researcher-handoff-project-status.md."
        )

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

    validate_data_seed(seed, allow_none=True)

    seed_context = _build_seed_context(seed, instance)
    editor_instances: dict[tuple[str, str], Any] = {}
    session = requests.Session()
    cleanup_handle: SeedCleanupHandle | None = None
    read_surface_accumulator: list[str] = []
    read_surface_provenance: dict[str, Any] = {}
    created_resource_accumulator: list[dict[str, Any]] = []
    editor_call_result_accumulator: list[dict[str, Any]] = []
    try:
        for call_index, call in enumerate(seed.get("editor_calls", [])):
            _apply_editor_seed_call(
                session,
                call,
                instance,
                call_index=call_index,
                seed_context=seed_context,
                editor_instances=editor_instances,
                read_surface_accumulator=read_surface_accumulator,
                read_surface_provenance=read_surface_provenance,
                created_resource_accumulator=created_resource_accumulator,
                editor_call_result_accumulator=editor_call_result_accumulator,
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
        created_resources = _dedupe_created_resources(created_resource_accumulator)
        if created_resources:
            metadata["created_resources"] = created_resources
            metadata["created_resource"] = _primary_created_resource(created_resources)
        if editor_call_result_accumulator:
            metadata["editor_call_results"] = editor_call_result_accumulator
        # Hoist authoritative write-identifier tokens from the merged
        # seed_context into metadata so downstream verifiers (render-check
        # read-your-write fastpath) can match server-reported IDs instead
        # of racing the DOM hydration cascade.
        for token_key in (
            "note_id",
            "issue_iid",
            "project_id",
            "comment_id",
            "submission_id",
            "review_id",
        ):
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
            if anchors.get("label_names"):
                context["benign_label_names"] = str(anchors["label_names"])
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
                    editor_method = getattr(editor, method_name, None)
                    if callable(editor_method):
                        args = _filter_editor_method_args(
                            editor_method,
                            args,
                            editor_site_name=str(call.get("site", "")).strip() or "unknown",
                            method_name=str(method_name),
                        )
                        unresolved = sorted(_seed_placeholder_names(args))
                        if unresolved:
                            raise RuntimeError(
                                "editor call has unresolved template placeholders: "
                                + ", ".join(unresolved)
                            )
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
    call_index: int | None = None,
    seed_context: dict[str, Any],
    editor_instances: dict[tuple[str, str], Any],
    read_surface_accumulator: list[str] | None = None,
    read_surface_provenance: dict[str, Any] | None = None,
    created_resource_accumulator: list[dict[str, Any]] | None = None,
    editor_call_result_accumulator: list[dict[str, Any]] | None = None,
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
    editor_site_name = str(getattr(editor, "site_name", rendered.get("site") or "")).strip()
    if method_name.startswith("_") or method_name not in editor.supported_methods:
        raise EditorError(
            "unsupported_method",
            f"{editor_site_name} editor does not support method {method_name!r}",
        )
    editor_method = getattr(editor, method_name, None)
    if not callable(editor_method):
        raise EditorError(
            "unsupported_method",
            f"{editor_site_name} editor does not support method {method_name!r}",
        )
    args = _filter_editor_method_args(
        editor_method, args, editor_site_name=editor_site_name, method_name=method_name
    )
    unresolved = sorted(_seed_placeholder_names(args))
    if unresolved:
        raise RuntimeError(
            "editor call has unresolved template placeholders: " + ", ".join(unresolved)
        )
    editor.validate_args(method_name, args)
    result = editor_method(**args)
    if isinstance(result, dict):
        if editor_call_result_accumulator is not None and call_index is not None:
            editor_call_result_accumulator.append(
                _editor_call_result_record(
                    result,
                    call_index=call_index,
                    editor_site_name=editor_site_name,
                    method_name=method_name,
                )
            )
        if created_resource_accumulator is not None:
            created_resource_accumulator.extend(
                _created_resources_from_editor_result(
                    result,
                    editor_method=f"{editor_site_name}.{method_name}",
                )
            )
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
            editor_method_str = f"{editor_site_name}.{method_name}"
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


def _filter_editor_method_args(
    editor_method: Any,
    args: dict[str, Any],
    *,
    editor_site_name: str,
    method_name: str,
) -> dict[str, Any]:
    """Drop kwargs not in the editor method's signature.

    Phase 4 placement_fix and variant_api can hallucinate extra editor args
    (e.g. ``position``, ``score``, ``author``) that the editor method does
    not declare. Calling ``editor_method(**args)`` with such args raises
    ``TypeError`` and aborts the whole post-processing pass. Filter unknown
    kwargs here so a single API hallucination does not cascade. Required
    args were already checked by ``editor.validate_args`` upstream.
    """
    try:
        sig = inspect.signature(editor_method)
    except (TypeError, ValueError):
        return args
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return args
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    unknown = sorted(set(args) - accepted)
    if not unknown:
        return args
    logger.warning(
        "editor %s.%s received %d unknown arg(s) %s; dropping before invocation",
        editor_site_name,
        method_name,
        len(unknown),
        unknown,
    )
    return {k: v for k, v in args.items() if k in accepted}


def _created_resources_from_editor_result(
    result: dict[str, Any],
    *,
    editor_method: str,
) -> list[dict[str, Any]]:
    """Extract generic created-resource descriptors from an editor result.

    Editors own site-specific write semantics. Callers should not need to
    know that a Postmill-created resource is called a submission or that a
    GitLab-created resource is called an issue. This helper preserves the
    editor-declared, generic transition targets that Phase 2c can later use
    for exposure verification.
    """
    raw_items: list[Any] = []
    raw_single = result.get("created_resource")
    if isinstance(raw_single, dict):
        raw_items.append(raw_single)
    raw_many = result.get("created_resources")
    if isinstance(raw_many, list):
        raw_items.extend(raw_many)

    resources: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        resource: dict[str, Any] = {}
        for key in ("role", "kind", "id", "url", "parent_url"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                resource[key] = value.strip()
            elif key == "id" and value not in (None, ""):
                resource[key] = str(value)
        if not isinstance(resource.get("url"), str):
            continue
        resource.setdefault("role", "created_resource")
        resource["editor_method"] = editor_method
        resources.append(resource)
    return resources


def _editor_call_result_record(
    result: dict[str, Any],
    *,
    call_index: int,
    editor_site_name: str,
    method_name: str,
) -> dict[str, Any]:
    """Return per-call write/read metadata for call-aware verification.

    The aggregate ``read_surface_urls`` / ``issue_iid`` metadata intentionally
    preserves older callers' simple shape. Phase 2c also needs to prove that a
    rendered signature came from the same editor call that produced the read
    surface being checked, especially for self-contained seeds that preserve a
    benign setup call before appending the adversarial write.
    """
    record: dict[str, Any] = {
        "call_index": call_index,
        "site": editor_site_name,
        "method": method_name,
        "editor_method": f"{editor_site_name}.{method_name}",
    }
    surface_urls = result.get("read_surface_urls")
    if isinstance(surface_urls, list):
        urls = [url.strip() for url in surface_urls if isinstance(url, str) and url.strip()]
        if urls:
            record["read_surface_urls"] = urls
    provenance_source = result.get("read_surface_provenance_source")
    if isinstance(provenance_source, str) and provenance_source.strip():
        record["read_surface_provenance_source"] = provenance_source.strip()
    created_resources = _created_resources_from_editor_result(
        result,
        editor_method=f"{editor_site_name}.{method_name}",
    )
    if created_resources:
        record["created_resources"] = created_resources
        record["created_resource"] = _primary_created_resource(created_resources)
    write_tokens: dict[str, Any] = {}
    for token_key in (
        "note_id",
        "issue_iid",
        "project_id",
        "comment_id",
        "submission_id",
        "review_id",
    ):
        token_value = result.get(token_key)
        if token_value not in (None, ""):
            write_tokens[token_key] = token_value
    if write_tokens:
        record["write_tokens"] = write_tokens
    return record


def _dedupe_created_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for resource in resources:
        key = (
            str(resource.get("role") or ""),
            str(resource.get("kind") or ""),
            str(resource.get("url") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(resource))
    return deduped


def _primary_created_resource(resources: list[dict[str, Any]]) -> dict[str, Any]:
    for resource in reversed(resources):
        if resource.get("role") == "seed_render_surface":
            return dict(resource)
    return dict(resources[-1])


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


def _instance_http_seed_auth_runtime_error(instance: Any, *, mechanism: str = "form") -> str | None:
    from worldsim.auth_tokens import _resolve_header_value, pick_auth_lane

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
