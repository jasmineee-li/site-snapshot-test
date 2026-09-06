"""Seed context assembly and editor-call template rendering."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from warp_taskgen.seeding.editor_args import _editor_arg_name
from warp_taskgen.seeding.map_context import _derive_map_seed_context
from warp_taskgen.seeding.reddit_context import _derive_reddit_seed_context

_FORMAT_TOKEN_PATTERN = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_\.]*)\}(?!\})")


_BENIGN_ANCHOR_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


_FREE_TEXT_EDITOR_ARG_NAMES = frozenset(
    {
        "body",
        "body_template",
        "description",
        "description_template",
        "text",
        "text_template",
        "title",
        "title_template",
    }
)


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
            _merge_seed_context(context, _project_benign_anchor_context(anchors))
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


def _project_benign_anchor_context(anchors: Mapping[str, Any]) -> dict[str, str]:
    """Project safe scalar task anchors into generic ``benign_*`` tokens.

    Site-specific anchor aliases remain above for compatibility with the
    existing GitLab and Reddit seed shapes.  New Site methods can declare a
    semantic ``Token("{benign_<anchor>}")`` without adding a resolver branch;
    only bounded, non-secret scalar anchors are exposed.
    """

    projected: dict[str, str] = {}
    for raw_key, raw_value in anchors.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip().lower()
        if _BENIGN_ANCHOR_KEY_PATTERN.fullmatch(key) is None:
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, (str, int)):
            continue
        value = str(raw_value).strip()
        if not value or len(value) > 200 or "\n" in value or "\r" in value or "://" in value:
            continue
        projected[f"benign_{key}"] = value
    return projected


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


def _unresolved_structural_seed_placeholder_names(
    value: Any,
    *,
    free_text_arg_names: frozenset[str] = _FREE_TEXT_EDITOR_ARG_NAMES,
) -> set[str]:
    names: set[str] = set()
    if isinstance(value, str):
        names.update(match.group(1) for match in _FORMAT_TOKEN_PATTERN.finditer(value))
        return names
    if isinstance(value, dict):
        for key, item in value.items():
            names.update(
                _unresolved_structural_seed_placeholder_names(
                    key,
                    free_text_arg_names=free_text_arg_names,
                )
            )
            if isinstance(key, str) and key in free_text_arg_names:
                continue
            names.update(
                _unresolved_structural_seed_placeholder_names(
                    item,
                    free_text_arg_names=free_text_arg_names,
                )
            )
        return names
    if isinstance(value, list):
        for item in value:
            names.update(
                _unresolved_structural_seed_placeholder_names(
                    item,
                    free_text_arg_names=free_text_arg_names,
                )
            )
    return names


def _editor_free_text_arg_names(editor_method: Any) -> frozenset[str]:
    spec = getattr(editor_method, "_editor_method_spec", None)
    if not isinstance(spec, dict):
        return _FREE_TEXT_EDITOR_ARG_NAMES
    bindings = spec.get("bindings")
    if not isinstance(bindings, dict):
        return _FREE_TEXT_EDITOR_ARG_NAMES
    names = {
        str(name)
        for name, binding in bindings.items()
        if getattr(binding, "kind", None) == "free_text"
    }
    return frozenset(names) if names else _FREE_TEXT_EDITOR_ARG_NAMES


def _merge_seed_context(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if value is None:
            continue
        target[key] = value


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


__all__ = [
    "_build_seed_context",
    "_derive_task_seed_context",
    "_lookup_seed_context_value",
    "_merge_seed_context",
    "_render_editor_seed_call",
    "_render_seed_value",
    "_seed_placeholder_names",
]
