"""WebArena Verified GitLab final-state reward composition."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlparse

from warp_taskgen.rewards import final_state_webarena_verified_gitlab as gitlab_state
from warp_taskgen.rewards.final_state_catalog import (
    FinalStateEvaluationRequest,
    thaw_final_state_value,
)
from warp_taskgen.rewards.network_trace import _network_event_url_candidates
from warp_taskgen.rewards.shared import _state_probe_kind


class GitLabFinalStateEvaluator:
    benchmark = "webarena_verified"
    site = "gitlab"

    def evaluate(self, request: FinalStateEvaluationRequest) -> tuple[bool, str]:
        network_trace = thaw_final_state_value(request.network_trace)
        instance = thaw_final_state_value(request.instance)
        state_probe = thaw_final_state_value(request.state_probe)
        network_expected = thaw_final_state_value(request.network_expected)
        events = thaw_final_state_value(request.initial_events)
        message = request.initial_message

        if request.action_kind in {"create_issue", "create_issue_note"}:
            events, source_message = gitlab_state._gitlab_filter_valid_mutation_source_events(
                action_kind=request.action_kind,
                witness=request.witness,
                expected=network_expected,
                events=events,
                instance=instance,
            )
            if source_message:
                message = source_message
        if (
            not events
            and request.action_kind == "create_issue"
            and _state_probe_kind(state_probe) in {"", "issue_contains"}
        ):
            events, message = gitlab_state._matching_gitlab_issue_source_events(
                network_expected,
                network_trace,
                instance,
                witness=request.witness,
            )
        if (
            not events
            and request.action_kind == "create_issue_note"
            and _state_probe_kind(state_probe) in {"", "issue_note_contains"}
        ):
            events, message = gitlab_state._matching_gitlab_issue_note_source_events(
                network_expected,
                network_trace,
                instance,
                witness=request.witness,
            )
        if (
            not events
            and request.action_kind == "create_issue"
            and request.evidence_policy.requires_state_readback
            and request.evidence_policy.allows_ui_state_transition
            and _state_probe_kind(state_probe) in {"", "issue_contains"}
        ):
            events = _gitlab_ui_issue_creation_events(network_trace, instance, state_probe)
            if not events:
                return False, (
                    f"final-state source event missing: {message}; "
                    "gitlab creation proof unavailable: no action-linked UI issue transition evidence"
                )
            passed, readback_message = gitlab_state._eval_gitlab_final_state(
                request.action_kind,
                request.witness,
                events,
                network_trace,
                instance,
                state_probe,
            )
            if passed:
                return True, (
                    f"{readback_message} "
                    "(proof_channel=state_readback; support=ui_state_transition)"
                )
            return False, readback_message
        if not events:
            return False, f"final-state source event missing: {message}; creation proof unavailable"
        return gitlab_state._eval_gitlab_final_state(
            request.action_kind,
            request.witness,
            events,
            network_trace,
            instance,
            state_probe,
        )


def _gitlab_ui_issue_creation_events(
    network_trace: list[dict],
    instance: dict,
    state_probe: Mapping | None,
) -> list[dict]:
    """A creation response redirect can support UI evidence; a detail GET cannot."""
    events = []
    for event in network_trace:
        if gitlab_state._network_event_method(event) != "POST":
            continue
        if gitlab_state._network_event_status(event) not in {200, 201, 202, 204, 302, 303}:
            continue
        for event_url in _network_event_url_candidates(event, instance):
            if urlparse(event_url).netloc != urlparse(str(instance.get("site_url") or "")).netloc:
                continue
            if not gitlab_state._gitlab_project_path_from_issue_create_ui_path(
                urlparse(event_url).path
            ):
                continue
            if gitlab_state._gitlab_response_locations(event):
                events.append(event)
            break
    return events


__all__ = ["GitLabFinalStateEvaluator"]
