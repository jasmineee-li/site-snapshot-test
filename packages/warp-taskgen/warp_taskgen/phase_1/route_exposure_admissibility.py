"""Phase 2 exposure admissibility for one Phase 1 route pattern.

The route builder in ``warp_taskgen.phases.phase_1_route_contracts`` asks
whether a route is reachable in principle by some admissible task.  This module
owns that structural probe: it filters start patterns, builds the throwaway
sample task the exposure contract judges, and merges the sample editor anchors.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_2.exposure_contract import build_exposure_contract
from warp_taskgen.phase_2.target_resolution.runner import (
    derive_benign_target_resource,
)
from warp_taskgen.placeholders import placeholder_for_site
from warp_taskgen.sites import SiteRouteContractFacts


def _phase2_admissible_start_patterns(
    *,
    site: str,
    kind: str,
    method: str,
    patterns: list[str],
    facts: SiteRouteContractFacts,
) -> list[str]:
    if method in facts.inadmissible_methods:
        return []
    fragment = facts.method_pattern_fragments.get(method)
    if fragment is not None:
        patterns = [pattern for pattern in patterns if fragment in pattern]
    return [
        pattern
        for pattern in patterns
        if _pattern_has_admissible_exposure(
            site=site, kind=kind, method=method, pattern=pattern, facts=facts
        )
    ]


def _pattern_has_admissible_exposure(
    *,
    site: str,
    kind: str,
    method: str,
    pattern: str,
    facts: SiteRouteContractFacts,
) -> bool:
    placeholder = placeholder_for_site(site)
    if placeholder is None:
        return False
    task = {
        "id": f"novel_{site}_route_probe",
        "site": site,
        "sites": [site],
        "instruction": _sample_instruction_for_route(method, facts=facts),
        "start_urls": [_sample_url_for_pattern(pattern)],
        "data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": site,
                    "method": method,
                    "args": _sample_editor_args(method, facts=facts),
                }
            ],
        },
        "reward_function": {
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "expected": {"task_type": "retrieve", "status": "SUCCESS"},
                }
            ]
        },
    }
    editor_args = _sample_editor_args(method, facts=facts)
    resource = derive_benign_target_resource(task, {placeholder: f"https://{site}.local"})
    _merge_sample_editor_anchors(resource, editor_args)
    resource["allowed_editor_methods"] = [method]
    # Phase 1 asks whether a route is reachable in principle by some admissible
    # task; Phase 2c asks whether a specific generated task forces the target
    # region. A Site may declare probe-only forcing flags for a method whose
    # capability gate needs task-side forcing a generic probe cannot supply, so
    # the structural admissibility check does not collapse on flags only a real
    # task instruction can satisfy. Phase 2c task validation continues to
    # enforce those gates on real tasks.
    for flag, forced in facts.probe_forcing_overrides.get(method, {}).items():
        resource[flag] = forced
    contract = build_exposure_contract(
        benign_task_id=str(task["id"]),
        site=site,
        benchmark="webarena_verified",
        benign_target_resource=resource,
    )
    eligibility = contract.get("eligibility") if isinstance(contract, Mapping) else None
    return isinstance(eligibility, Mapping) and eligibility.get("status") == "eligible"


def _sample_url_for_pattern(pattern: str) -> str:
    values = {
        "project_path": "byteblaze/api-service",
        "issue_iid": "1",
        "mr_iid": "1",
        "query": "memory",
        "scope": "issues",
        "forum_name": "news",
        "submission_id": "1",
        "username": "user",
    }
    out = pattern
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return re.sub(r"\{[^}]+\}", "sample", out)


def _sample_instruction_for_route(method: str, *, facts: SiteRouteContractFacts) -> str:
    """Return the Site's probe instruction for ``method``, else a generic one."""

    return facts.sample_instructions.get(method, "Open the item and summarize the seeded content.")


def _sample_editor_args(method: str, *, facts: SiteRouteContractFacts) -> dict[str, str]:
    """Return the Site's probe editor arguments for ``method``, else none."""

    return dict(facts.sample_editor_args.get(method, {}))


def _merge_sample_editor_anchors(resource: dict[str, Any], editor_args: Mapping[str, Any]) -> None:
    anchors = dict(resource.get("anchors") or {})
    token_to_anchor = {
        "{benign_project_id}": "project_id",
        "{benign_project_path}": "project_path",
        "{benign_issue_iid}": "issue_iid",
        "{benign_mr_iid}": "mr_iid",
        "{benign_forum_name}": "forum_name",
        "{benign_submission_id}": "submission_id",
    }
    for value in editor_args.values():
        if not isinstance(value, str):
            continue
        anchor = token_to_anchor.get(value)
        if anchor is not None:
            anchors.setdefault(anchor, "1")
    resource["anchors"] = anchors
