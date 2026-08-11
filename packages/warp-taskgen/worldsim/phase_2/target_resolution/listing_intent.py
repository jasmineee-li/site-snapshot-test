"""Phase 2 target resolution deterministic listing-intent heuristics."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from worldsim.phase_2.target_resolution.encounter import (
    _assert_anchor_contract_conformance,
    _attach_surfaces_for,
    _encounter_requirements,
    _route_evidence_flags,
)
from worldsim.phase_2.target_resolution.reconstruction import _reconstruct_start_url_from_anchors
from worldsim.sites.gitlab import GitLabSite

_GITLAB_SITE = GitLabSite()


def _gitlab_issue_listing_intent(
    task: Mapping[str, Any],
    *,
    resolved_start: str | None,
    placeholders: Mapping[str, str],
    benchmark: str,
) -> dict[str, Any] | None:
    instruction = str(task.get("instruction") or "")
    lowered = instruction.casefold()
    if "issue" not in lowered or not any(
        phrase in lowered
        for phrase in (
            "issues page",
            "list of open issues",
            "list of closed issues",
            "list of all issues",
            "most recent open issues",
            "opened issues",
            "latest issue",
            "most recent issue",
            "newest issue",
            "first issue",
            "first visible issue",
            "first shown issue",
            "seeded issue",
        )
    ):
        return None

    project_path = _project_path_from_gitlab_listing_task(
        instruction,
        resolved_start=resolved_start,
        placeholders=placeholders,
    )
    if not project_path:
        return None

    anchors: dict[str, str] = {"project_path": project_path}
    label_names = _label_names_from_gitlab_issue_listing_instruction(instruction)
    if label_names:
        anchors["label_names"] = label_names
    reconstructed = _reconstruct_start_url_from_anchors(
        "gitlab", "gitlab_search_result", anchors, placeholders
    )
    if reconstructed is None:
        base = (placeholders.get("__GITLAB__") or "").rstrip("/")
        reconstructed = f"{base}/{project_path}/-/issues" if base else resolved_start
    record = {
        "kind": "gitlab_search_result",
        "anchors": anchors,
        "start_url_resolved": reconstructed,
        "attach_surfaces": _attach_surfaces_for(
            "gitlab_search_result", benchmark=benchmark, site="gitlab"
        ),
        "encounter_requirements": _encounter_requirements("gitlab_search_result", task, anchors),
        "layer": "L2",
        "resolver_note": "deterministic_gitlab_issue_listing_intent",
        "skip_l4_expansion": True,
    }
    record.update(_route_evidence_flags("gitlab_search_result", task))
    _assert_anchor_contract_conformance(record, benchmark=benchmark, site="gitlab")
    return record


def _project_path_from_gitlab_listing_task(
    instruction: str,
    *,
    resolved_start: str | None,
    placeholders: Mapping[str, str],
) -> str | None:
    return _GITLAB_SITE.project_path_from_listing_task(
        instruction,
        resolved_start=resolved_start,
    )


def _label_names_from_gitlab_issue_listing_instruction(instruction: str) -> str | None:
    lowered = instruction.casefold()
    if "don't have any labels" in lowered or "do not have any labels" in lowered:
        return None
    label_match = re.search(
        r"labels? related to (?P<label>.+?)(?:$|[.])",
        instruction,
        re.IGNORECASE,
    )
    if label_match:
        return label_match.group("label").strip().strip("\"'")
    if "report bugs" in lowered or "opened issues that report bugs" in lowered:
        return "bug"
    if "requesting new features" in lowered:
        return "feature"
    if "opt model related questions" in lowered:
        return "question"
    return None
