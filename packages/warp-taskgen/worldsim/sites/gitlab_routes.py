"""GitLab-owned profile route facts for Phase 1.

Only deterministic profile/inventory interpretation belongs here.  Phase 1
still owns editor/core/active-carrier, exposure, instruction, and answer
stability policy after consuming these facts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from worldsim.placeholders import placeholder_for_site
from worldsim.sites.contracts import SiteRouteContractFacts


def route_contract_facts(
    *,
    benchmark: str,
    profile: Mapping[str, Any],
    kind: str,
    route_patterns: tuple[str, ...],
) -> SiteRouteContractFacts:
    """Return GitLab's profile/inventory-backed facts for ``kind``."""

    if benchmark != "webarena_verified":
        return SiteRouteContractFacts()
    placeholder = placeholder_for_site("gitlab")
    local_kind = _to_local_kind(kind)
    if placeholder is None or not route_patterns:
        return SiteRouteContractFacts()
    allowed_patterns = tuple(f"{placeholder}{pattern}" for pattern in route_patterns)
    anchors = _anchor_examples(local_kind, placeholder, profile)
    requires_inventory = local_kind in {"issue", "merge_request", "search_result"}
    variant = _route_variant(anchors)
    return SiteRouteContractFacts(
        allowed_start_url_patterns=allowed_patterns,
        anchor_examples=tuple(anchors),
        requires_inventory_backed_start_url=requires_inventory or bool(anchors),
        route_variant=variant,
    )


def _anchor_examples(
    kind: str,
    placeholder: str,
    profile: Mapping[str, Any],
) -> list[dict[str, str]]:
    if kind == "search_result":
        return _project_issue_list_examples(placeholder, profile)
    entity_names = {
        "issue": ("issue", "issues"),
        "merge_request": ("merge_request", "merge_requests"),
    }.get(kind)
    if entity_names is None:
        return []
    examples: list[dict[str, str]] = []
    for sample in _data_model_sample_values(profile, entity_names):
        project_path = _project_path_from_sample(sample, profile)
        iid = sample.get("iid") or sample.get("issue_iid") or sample.get("mr_iid")
        if not project_path or iid is None:
            continue
        iid_text = str(iid).strip()
        if not iid_text:
            continue
        if kind == "issue":
            examples.append(
                {
                    "project_path": project_path,
                    "issue_iid": iid_text,
                    "start_url": f"{placeholder}/{project_path}/-/issues/{iid_text}",
                }
            )
        else:
            examples.append(
                {
                    "project_path": project_path,
                    "mr_iid": iid_text,
                    "start_url": (f"{placeholder}/{project_path}/-/merge_requests/{iid_text}"),
                }
            )
    return examples


def _project_issue_list_examples(
    placeholder: str,
    profile: Mapping[str, Any],
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    seen: set[str] = set()
    samples = [
        *((sample, False) for sample in _data_model_sample_values(profile, ("issue", "issues"))),
        *((sample, True) for sample in _available_entity_records(profile, ("projects",))),
        *((sample, True) for sample in _data_model_sample_values(profile, ("project", "projects"))),
    ]
    for sample, sample_is_project in samples:
        project_path = _project_path_from_sample(sample, profile)
        if not project_path or project_path in seen:
            continue
        seen.add(project_path)
        example = {
            "route_variant": "project_issue_list",
            "project_path": project_path,
            "scope": "issues",
            "start_url": (f"{placeholder}/{project_path}/-/issues?sort=created_date&state=opened"),
        }
        project_id = _project_id_from_sample(sample, sample_is_project=sample_is_project)
        if project_id:
            example["project_id"] = project_id
        examples.append(example)
    return examples


def _available_entity_records(
    profile: Mapping[str, Any], keys: tuple[str, ...]
) -> list[Mapping[str, Any]]:
    available = profile.get("available_entities")
    if not isinstance(available, Mapping):
        return []
    records: list[Mapping[str, Any]] = []
    for key in keys:
        values = available.get(key)
        if isinstance(values, list):
            records.extend(item for item in values if isinstance(item, Mapping))
    return records


def _project_path_from_sample(sample: Mapping[str, Any], profile: Mapping[str, Any]) -> str:
    for key in ("project", "project_path", "path_with_namespace", "full_path"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            path = _normalize_project_path(value)
            if _is_resolvable_project_path(path):
                return path
    name = sample.get("name")
    if isinstance(name, str) and "/" in name:
        path = _normalize_project_path(name)
        if _is_resolvable_project_path(path):
            return path
    namespace = str(sample.get("namespace") or "").strip().strip("/")
    path = str(sample.get("path") or "").strip().strip("/")
    if namespace and path:
        joined = _normalize_project_path(f"{namespace}/{path}")
        if _is_resolvable_project_path(joined):
            return joined
    for key in ("project_id", "target_project_id", "source_project_id"):
        project_id = sample.get(key)
        if project_id not in (None, ""):
            path = _project_path_by_id(profile, project_id)
            if _is_resolvable_project_path(path):
                return _normalize_project_path(path)
    return ""


def _project_id_from_sample(sample: Mapping[str, Any], *, sample_is_project: bool = False) -> str:
    keys = ["project_id", "target_project_id", "source_project_id"]
    if sample_is_project:
        keys.append("id")
    for key in keys:
        value = sample.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_project_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" in raw:
        raw = urlsplit(raw).path
    raw = raw.strip().strip("/")
    if raw.startswith("__GITLAB__/"):
        raw = raw[len("__GITLAB__/") :]
    if "/-/" in raw:
        raw = raw.split("/-/", 1)[0]
    parts = [part for part in raw.split("/") if part]
    if len(parts) >= 3 and (":" in parts[0] or parts[0] in {"localhost", "gitlab.local"}):
        parts = parts[1:]
    return "/".join(parts)


def _is_resolvable_project_path(value: Any) -> bool:
    return len([part for part in _normalize_project_path(value).split("/") if part]) >= 2


def _project_path_by_id(profile: Mapping[str, Any], project_id: Any) -> str:
    wanted = str(project_id).strip()
    if not wanted:
        return ""
    samples = [
        *_available_entity_records(profile, ("projects",)),
        *_data_model_sample_values(profile, ("project", "projects")),
    ]
    for sample in samples:
        if str(sample.get("id") or "").strip() != wanted:
            continue
        for key in ("project", "project_path", "path_with_namespace", "full_path"):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                path = _normalize_project_path(value)
                if _is_resolvable_project_path(path):
                    return path
        name = sample.get("name")
        if isinstance(name, str) and "/" in name:
            path = _normalize_project_path(name)
            if _is_resolvable_project_path(path):
                return path
        namespace = str(sample.get("namespace") or "").strip().strip("/")
        path = str(sample.get("path") or sample.get("name") or "").strip().strip("/")
        if namespace and path:
            return f"{namespace}/{path}"
    return ""


def _data_model_sample_values(
    profile: Mapping[str, Any], entity_names: str | tuple[str, ...]
) -> list[Mapping[str, Any]]:
    names = {entity_names} if isinstance(entity_names, str) else set(entity_names)
    normalized_names = {name.casefold() for name in names}
    data_model = profile.get("data_model")
    if not isinstance(data_model, list):
        return []
    for entity in data_model:
        if not isinstance(entity, Mapping):
            continue
        if str(entity.get("entity") or "").strip().casefold() not in normalized_names:
            continue
        samples = entity.get("sample_values")
        if not isinstance(samples, list):
            return []
        return [sample for sample in samples if isinstance(sample, Mapping)]
    return []


def _route_variant(examples: list[Mapping[str, Any]]) -> str | None:
    variants = {
        str(example.get("route_variant") or "").strip()
        for example in examples
        if isinstance(example, Mapping)
    }
    variants.discard("")
    return next(iter(variants)) if len(variants) == 1 else None


def _to_local_kind(kind: str) -> str:
    return {
        "gitlab_issue": "issue",
        "gitlab_mr": "merge_request",
        "gitlab_search_result": "search_result",
        "gitlab_dashboard_list": "dashboard_list",
        "gitlab_user_profile": "user_profile",
        "gitlab_snippet": "snippet",
        "gitlab_snippets_index": "snippets_index",
        "gitlab_project_milestone": "project_milestone",
        "gitlab_project_labels": "project_labels",
        "gitlab_group": "group",
    }.get(kind, kind)


__all__ = ["route_contract_facts"]
