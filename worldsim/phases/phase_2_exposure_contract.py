"""Deterministic Phase 2 exposure contracts.

An exposure contract is a composed view over the resolver output and the
editor-method registry. It is not a new placement registry: editor
decorators remain the source of truth for methods, bindings, surface IDs,
and reachable tokens.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from worldsim.editors._method_spec import BindingSpec
from worldsim.editors._registry import (
    EditorMethodSpec,
    available_tokens_for_kind,
    iter_specs,
)
from worldsim.phases.phase_2_text_fill import PAYLOAD_PLACEHOLDER

PREFERRED_PAYLOAD_ARGS: tuple[str, ...] = ("body", "description", "message", "text", "content")
PREFERRED_TOKEN_ORDER: tuple[str, ...] = (
    "{benign_project_id}",
    "{benign_project_path}",
    "{benign_issue_iid}",
    "{benign_mr_iid}",
    "{benign_submission_id}",
    "{benign_forum_name}",
    "{benign_user_handle}",
)

DIRECT_KINDS = frozenset({"gitlab_issue", "gitlab_mr", "reddit_submission"})
LISTING_SOURCE_KINDS = frozenset(
    {"gitlab_search_result", "gitlab_dashboard_list", "reddit_forum", "reddit_dashboard_list"}
)
TRANSITIVE_EXISTING_SOURCE_KINDS = frozenset(
    {"gitlab_search_result", "gitlab_dashboard_list", "reddit_dashboard_list"}
)
CREATE_CHILD_LISTING_KINDS = frozenset({"reddit_forum"})


def exposure_contract_signature() -> dict[str, Any]:
    """Fingerprint knobs whose change invalidates persisted Phase 2 plans."""
    return {
        "version": 2,
        "modes": [
            "direct_detail",
            "inline_listing",
            "bounded_transitive_existing",
            "bounded_transitive_created_child",
            "ineligible",
        ],
        "payload_arg_preference": list(PREFERRED_PAYLOAD_ARGS),
        "token_preference": list(PREFERRED_TOKEN_ORDER),
        "created_child_target_source": "seed_metadata.created_resource.url",
    }


def signature_hash() -> str:
    payload = json.dumps(exposure_contract_signature(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_exposure_contract(
    *,
    benign_task_id: str,
    site: str,
    benchmark: str,
    benign_target_resource: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the exposure contract for one resolved benign target."""
    resource = dict(benign_target_resource or {})
    kind = resource.get("kind")
    anchors_raw = resource.get("anchors")
    anchors = dict(anchors_raw) if isinstance(anchors_raw, Mapping) else {}
    site = site.strip().lower()

    base: dict[str, Any] = {
        "contract_id": _contract_id(site, benign_task_id, kind, anchors),
        "benign_task_id": benign_task_id,
        "site": site,
        "kind": kind,
        "anchors": anchors,
        "benign_read_url": _benign_read_url(resource),
        "eligibility": {"status": "ineligible", "reason": "unresolved_target_resource"},
    }

    if not isinstance(kind, str) or not kind:
        return base

    mode, ineligible_reason = _mode_for_resource(resource, kind)
    if mode == "ineligible":
        base["mode"] = "ineligible"
        base["eligibility"] = {
            "status": "ineligible",
            "reason": ineligible_reason or f"kind_not_supported_for_exposure:{kind}",
        }
        return base
    if not base["benign_read_url"]:
        base["mode"] = mode
        base["eligibility"] = {
            "status": "ineligible",
            "reason": "missing_benign_read_url",
        }
        return base

    available = available_tokens_for_kind(kind, anchors, benchmark=benchmark, site=site)
    for spec in _viable_specs(kind, site=site, benchmark=benchmark, available=available):
        template = _editor_args_template(
            spec,
            available=available,
            benign_task_id=benign_task_id,
        )
        if template is None:
            continue
        payload_arg = template.pop("__payload_arg__")
        target_surface_id = spec.surface_id_per_kind.get(kind, spec.method)
        verification = _verification_contract(resource, mode, base["benign_read_url"], kind)
        base.update(
            {
                "mode": mode,
                "editor_method": spec.method,
                "target_surface_id": target_surface_id,
                "payload_arg": payload_arg,
                "editor_args_template": template,
                "required_tokens": sorted(available),
                "verification": verification,
                "eligibility": {"status": "eligible"},
            }
        )
        return base

    base["mode"] = mode
    base["required_tokens"] = sorted(available)
    base["eligibility"] = {
        "status": "ineligible",
        "reason": "no_viable_editor_method_under_anchors",
    }
    return base


def materialize_seed_template_from_contract(
    contract: Mapping[str, Any],
    *,
    benchmark: str = "webarena_verified",
) -> dict[str, Any]:
    """Return the editor seed template encoded by an eligible contract."""
    eligibility = contract.get("eligibility")
    if not isinstance(eligibility, Mapping) or eligibility.get("status") != "eligible":
        raise ValueError("cannot materialize seed_template from ineligible exposure_contract")
    method = str(contract.get("editor_method") or "")
    site = str(contract.get("site") or "")
    args = contract.get("editor_args_template")
    if not method or not site or not isinstance(args, Mapping):
        raise ValueError("exposure_contract missing editor method/site/args template")
    return {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": benchmark,
                "site": site,
                "method": method,
                "args": dict(args),
            }
        ],
    }


def _contract_id(site: str, benign_task_id: str, kind: Any, anchors: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"site": site, "benign_task_id": benign_task_id, "kind": kind, "anchors": anchors},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:10]
    return f"{site}:{benign_task_id}:{digest}"


def _benign_read_url(resource: Mapping[str, Any]) -> str | None:
    value = resource.get("benign_read_url") or resource.get("start_url_resolved")
    return str(value) if isinstance(value, str) and value.strip() else None


def _mode_for_resource(resource: Mapping[str, Any], kind: str) -> tuple[str, str | None]:
    source_kind = resource.get("source_listing_kind")
    if isinstance(source_kind, str) and source_kind in TRANSITIVE_EXISTING_SOURCE_KINDS:
        if _transitive_entry_supported(source_kind, resource):
            return "bounded_transitive_existing", None
        return (
            "ineligible",
            f"unsupported_transitive_entry:{source_kind}",
        )
    if kind in DIRECT_KINDS:
        return "direct_detail", None
    if kind in CREATE_CHILD_LISTING_KINDS:
        return "bounded_transitive_created_child", None
    if kind in LISTING_SOURCE_KINDS:
        return "inline_listing", None
    return "ineligible", f"kind_not_supported_for_exposure:{kind}"


def _verification_contract(
    resource: Mapping[str, Any],
    mode: str,
    benign_read_url: str,
    kind: str,
) -> dict[str, Any]:
    verification: dict[str, Any] = {
        "url": benign_read_url,
        "entry": {"url": benign_read_url, "role": "benign_entry"},
        "mode": mode,
        "selector_hint": _selector_hint(kind, mode),
        "witness_policy": "signature_plus_second_witness",
    }
    if mode == "bounded_transitive_existing":
        target_url = resource.get("seeded_detail_url") or resource.get("start_url_resolved")
        if isinstance(target_url, str) and target_url.strip():
            verification["target"] = {
                "url": target_url,
                "role": "seed_render_surface",
            }
        verification["transition"] = {
            "type": "visible_link",
            "max_depth": 1,
            "same_origin": True,
            "edge_role": _edge_role_for_kind(kind),
        }
    elif mode == "bounded_transitive_created_child":
        verification["target"] = {
            "url_source": "seed_metadata.created_resource.url",
            "role": "seed_render_surface",
        }
        verification["transition"] = {
            "type": "visible_link",
            "max_depth": 1,
            "same_origin": True,
            "edge_role": "created_child_detail",
        }
    return verification


def _edge_role_for_kind(kind: str) -> str:
    if kind == "gitlab_issue":
        return "issue_detail"
    if kind == "gitlab_mr":
        return "merge_request_detail"
    if kind == "reddit_submission":
        return "submission_detail"
    return "detail"


def _transitive_entry_supported(source_kind: str, resource: Mapping[str, Any]) -> bool:
    url = resource.get("benign_read_url") or resource.get("start_url_resolved")
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    path = parsed.path or "/"
    if source_kind == "gitlab_search_result":
        # A project root is not a search/list surface. Admitting arbitrary
        # project-root -> issue links reintroduces the commit-count bug:
        # the agent can complete the benign task without reading issues.
        return path == "/search" or path.endswith("/-/issues") or path.endswith("/-/merge_requests")
    if source_kind == "gitlab_dashboard_list":
        return path.startswith("/dashboard/")
    if source_kind == "reddit_dashboard_list":
        return path.startswith("/user/")
    return False


def _viable_specs(
    kind: str,
    *,
    site: str,
    benchmark: str,
    available: frozenset[str],
) -> list[EditorMethodSpec]:
    specs = sorted(
        (spec for spec in iter_specs(site=site, benchmark=benchmark) if kind in spec.kinds),
        key=lambda item: item.method,
    )
    return [spec for spec in specs if _method_viable_under_anchors(spec, available)]


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
    free_text_args = [
        arg for arg, binding in spec.bindings.items() if binding.kind == "free_text"
    ]
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
            candidates.append((len(PREFERRED_TOKEN_ORDER) + 1, arg, _default_free_text(arg, "group")))
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
    if mode == "inline_listing":
        return ".issuable-list, .submission, body"
    if mode.startswith("bounded_transitive"):
        return ".issuable-list, .submission, body"
    return None
