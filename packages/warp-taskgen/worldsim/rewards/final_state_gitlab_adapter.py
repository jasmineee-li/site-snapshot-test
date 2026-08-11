"""WebArena Verified GitLab final-state reward composition."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import unquote, urlparse

from worldsim.rewards import final_state_webarena_verified_gitlab as gitlab_state
from worldsim.rewards.final_state_catalog import (
    FinalStateEvaluationRequest,
    thaw_final_state_value,
)
from worldsim.rewards.network_trace import _network_event_url_candidates
from worldsim.rewards.shared import _state_probe_kind


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
            if not _gitlab_ui_issue_transition_seen(network_trace, instance, state_probe):
                return False, (
                    f"final-state source event missing: {message}; "
                    "no gitlab UI issue transition evidence"
                )
            passed, readback_message = gitlab_state._eval_gitlab_final_state(
                request.action_kind,
                request.witness,
                [{}],
                network_trace,
                instance,
                state_probe,
            )
            if passed:
                return True, (
                    f"{readback_message} "
                    "(proof_channel=state_readback; support=ui_state_transition)"
                )
            message = f"{message}; {readback_message}" if message else readback_message
        if not events:
            return False, f"final-state source event missing: {message}"
        return gitlab_state._eval_gitlab_final_state(
            request.action_kind,
            request.witness,
            events,
            network_trace,
            instance,
            state_probe,
        )


def _gitlab_ui_issue_transition_seen(
    network_trace: list[dict],
    instance: dict,
    state_probe: Mapping | None,
) -> bool:
    probe = state_probe if isinstance(state_probe, Mapping) else {}
    allowed_projects = {
        str(value).strip().strip("/")
        for value in (probe.get("project_path"), probe.get("project_id"))
        if value not in (None, "")
    }
    trace_projects = _gitlab_new_issue_form_projects(network_trace, instance)
    observed_anchor = gitlab_state._gitlab_issue_anchor_from_network_trace(network_trace)
    excluded_iids = gitlab_state._gitlab_excluded_issue_iids(probe)
    if observed_anchor is not None and observed_anchor[1] not in excluded_iids:
        trace_projects.add(observed_anchor[0])
    if not trace_projects:
        return False
    return not allowed_projects or bool(trace_projects & allowed_projects)


def _gitlab_new_issue_form_projects(
    network_trace: list[dict],
    instance: dict,
) -> set[str]:
    projects: set[str] = set()
    for event in network_trace:
        for event_url in _network_event_url_candidates(event, instance):
            path = urlparse(event_url).path
            marker = "/-/issues/new"
            if marker not in path:
                continue
            project_path = unquote(path.split(marker, 1)[0]).strip("/")
            if project_path:
                projects.add(project_path)
    return projects


__all__ = ["GitLabFinalStateEvaluator"]
