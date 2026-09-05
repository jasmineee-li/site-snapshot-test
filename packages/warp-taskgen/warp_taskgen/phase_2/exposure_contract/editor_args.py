"""Exposure editor argument templates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.editors._method_spec import BindingSpec
from warp_taskgen.editors._registry import EditorMethodSpec, iter_specs
from warp_taskgen.phase_2.exposure_contract.constants import (
    PREFERRED_PAYLOAD_ARGS,
    PREFERRED_TOKEN_ORDER,
)
from warp_taskgen.phase_2.text_fill.constants import PAYLOAD_PLACEHOLDER
from warp_taskgen.phases.phase_2_core_surfaces import canonical_core_surface


def _viable_specs(
    kind: str,
    *,
    site: str,
    benchmark: str,
    available: frozenset[str],
    allowed_editor_methods: frozenset[str] | None = None,
    surface_visibility_by_id: Mapping[str, str] | None = None,
) -> list[EditorMethodSpec]:
    specs = sorted(
        (spec for spec in iter_specs(site=site, benchmark=benchmark) if kind in spec.kinds),
        key=lambda item: (
            _surface_visibility_rank(item, kind, surface_visibility_by_id),
            item.method,
        ),
    )
    if allowed_editor_methods is not None:
        specs = [spec for spec in specs if spec.method in allowed_editor_methods]
    return [spec for spec in specs if _method_viable_under_anchors(spec, available)]


def _allowed_editor_methods(resource: Mapping[str, Any]) -> frozenset[str] | None:
    raw = resource.get("allowed_editor_methods")
    if not isinstance(raw, list):
        return None
    methods = frozenset(str(item).strip() for item in raw if str(item).strip())
    return methods or None


def _surface_visibility_rank(
    spec: EditorMethodSpec,
    kind: str,
    surface_visibility_by_id: Mapping[str, str] | None,
) -> int:
    if spec.site == "reddit" and kind == "reddit_forum":
        surface_id = spec.surface_id_per_kind.get(kind, spec.method)
        if canonical_core_surface("reddit", surface_id) == "submission.title":
            return -1
    if spec.site == "gitlab" and kind == "gitlab_search_result":
        surface_id = spec.surface_id_per_kind.get(kind, spec.method)
        if canonical_core_surface("gitlab", surface_id) == "issue.title":
            return -1
    if not isinstance(surface_visibility_by_id, Mapping):
        return 1
    surface_id = spec.surface_id_per_kind.get(kind, spec.method)
    visibility = surface_visibility_by_id.get(surface_id)
    return 0 if visibility == "always_shown" else 1


def _method_viable_under_anchors(spec: EditorMethodSpec, available: frozenset[str]) -> bool:
    groups: dict[str, list[BindingSpec]] = {}
    for binding in spec.bindings.values():
        if binding.kind == "selector":
            groups.setdefault(binding.selector_group or "", []).append(binding)

    for members in groups.values():
        if not any(member.required for member in members):
            continue
        if not any((not member.tokens) or bool(member.tokens & available) for member in members):
            return False

    for binding in spec.bindings.values():
        if binding.kind != "token" or binding.selector_group is not None:
            continue
        if binding.required and not (binding.tokens & available):
            return False
    return True


def _editor_args_template(
    spec: EditorMethodSpec,
    *,
    available: frozenset[str],
    benign_task_id: str,
) -> dict[str, Any] | None:
    payload_arg = _payload_arg(spec)
    if payload_arg is None:
        return None

    args: dict[str, Any] = {}
    grouped: dict[str, list[tuple[str, BindingSpec]]] = {}
    for arg, binding in spec.bindings.items():
        if binding.kind == "selector":
            grouped.setdefault(binding.selector_group or "", []).append((arg, binding))

    for _group_name, members in grouped.items():
        if not any(binding.required for _, binding in members):
            continue
        selected = _select_group_member(members, available)
        if selected is None:
            return None
        arg, value = selected
        args[arg] = value

    for arg, binding in spec.bindings.items():
        if binding.kind == "selector":
            continue
        if arg == payload_arg:
            args[arg] = PAYLOAD_PLACEHOLDER
        elif binding.kind == "token":
            token = _choose_token(binding.tokens & available)
            if binding.required and token is None:
                return None
            if token is not None:
                args[arg] = token
        elif binding.kind == "free_text" and binding.required:
            args[arg] = _default_free_text(arg, benign_task_id)

    for required_arg in spec.required_editor_args:
        if required_arg == payload_arg:
            args[required_arg] = PAYLOAD_PLACEHOLDER
        elif required_arg not in args:
            binding = spec.bindings.get(required_arg)
            if binding is None:
                return None
            if binding.kind == "free_text":
                args[required_arg] = _default_free_text(required_arg, benign_task_id)
            elif binding.kind == "token":
                token = _choose_token(binding.tokens & available)
                if token is None:
                    return None
                args[required_arg] = token

    args["__payload_arg__"] = payload_arg
    return args


def _payload_arg(spec: EditorMethodSpec) -> str | None:
    free_text_args = [arg for arg, binding in spec.bindings.items() if binding.kind == "free_text"]
    if not free_text_args:
        return None
    for preferred in PREFERRED_PAYLOAD_ARGS:
        if preferred in free_text_args:
            return preferred
    required_free_text = [
        arg
        for arg in spec.required_editor_args
        if arg in spec.bindings and spec.bindings[arg].kind == "free_text"
    ]
    if required_free_text:
        return required_free_text[0]
    return free_text_args[0]


def _select_group_member(
    members: list[tuple[str, BindingSpec]], available: frozenset[str]
) -> tuple[str, str] | None:
    candidates: list[tuple[int, str, str]] = []
    for arg, binding in members:
        if not binding.tokens:
            candidates.append(
                (len(PREFERRED_TOKEN_ORDER) + 1, arg, _default_free_text(arg, "group"))
            )
            continue
        token = _choose_token(binding.tokens & available)
        if token is None:
            continue
        try:
            rank = PREFERRED_TOKEN_ORDER.index(token)
        except ValueError:
            rank = len(PREFERRED_TOKEN_ORDER)
        candidates.append((rank, arg, token))
    if not candidates:
        return None
    _, arg, value = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    return arg, value


def _choose_token(tokens: frozenset[str] | set[str]) -> str | None:
    if not tokens:
        return None
    for preferred in PREFERRED_TOKEN_ORDER:
        if preferred in tokens:
            return preferred
    return sorted(tokens)[0]


def _default_free_text(arg: str, benign_task_id: str) -> str:
    normalized = arg.replace("_", " ").strip() or "context"
    return f"WorldSim {normalized} for {benign_task_id}"


def _selector_hint(kind: str, mode: str) -> str | None:
    if kind in {"gitlab_issue", "gitlab_mr"}:
        return ".note, .timeline-entry, .issuable"
    if kind == "reddit_submission":
        return ".comment, .submission"
    if kind == "gitlab_user_profile":
        return ".user-profile, .profile-bio, .user-status"
    if kind == "gitlab_snippet":
        return ".blob, .snippet, .file-content"
    if kind == "gitlab_snippets_index":
        return ".snippets-list, .snippet-row, body"
    if kind == "gitlab_project_milestone":
        return ".milestone-detail, .description, .wiki"
    if kind == "gitlab_project_labels":
        return ".manage-labels-list, .label-row, body"
    if kind == "gitlab_group":
        return ".group-home-panel, .group-description"
    if mode == "inline_listing":
        return ".issuable-list, .submission, body"
    if mode.startswith("bounded_transitive"):
        return ".issuable-list, .submission, body"
    return None


__all__ = [
    "_allowed_editor_methods",
    "_choose_token",
    "_default_free_text",
    "_editor_args_template",
    "_method_viable_under_anchors",
    "_payload_arg",
    "_select_group_member",
    "_selector_hint",
    "_surface_visibility_rank",
    "_viable_specs",
]
