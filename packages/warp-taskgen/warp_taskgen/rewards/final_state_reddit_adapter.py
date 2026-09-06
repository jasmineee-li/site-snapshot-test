"""WebArena Verified Reddit/Postmill final-state reward composition."""

from __future__ import annotations

from warp_taskgen.rewards import final_state_webarena_verified_reddit as reddit_state
from warp_taskgen.rewards.final_state_catalog import (
    FinalStateEvaluationRequest,
    thaw_final_state_value,
)
from warp_taskgen.rewards.shared import _state_probe_kind


class RedditFinalStateEvaluator:
    benchmark = "webarena_verified"
    site = "reddit"

    def evaluate(self, request: FinalStateEvaluationRequest) -> tuple[bool, str]:
        network_trace = thaw_final_state_value(request.network_trace)
        instance = thaw_final_state_value(request.instance)
        state_probe = thaw_final_state_value(request.state_probe)
        network_expected = thaw_final_state_value(request.network_expected)
        events = thaw_final_state_value(request.initial_events)
        message = request.initial_message

        if request.action_kind == "create_post":
            if not request.evidence_policy.allows_network_event:
                return (
                    False,
                    "reddit post creation attribution unavailable for allowed evidence sources",
                )
            if not events:
                return False, f"reddit post creation attribution unavailable: {message}"
            return reddit_state._eval_reddit_final_state(
                request.action_kind,
                request.witness,
                events,
                network_trace,
                instance,
                state_probe,
            )

        if not events and _state_probe_kind(state_probe) in {
            "reddit_post_contains",
            "reddit_comment_contains",
        }:
            passed, readback_message = reddit_state._eval_reddit_final_state_from_probe(
                request.action_kind,
                request.witness,
                state_probe,
                network_trace,
                instance,
            )
            if passed:
                return True, readback_message
            message = f"{message}; {readback_message}" if message else readback_message
        if (
            not events
            and request.action_kind == "submit_comment"
            and _state_probe_kind(state_probe) == "reddit_comment_contains"
        ):
            relaxed_events, relaxed_message = reddit_state._matching_reddit_comment_source_events(
                network_expected,
                network_trace,
                instance,
            )
            if relaxed_events:
                passed, readback_message = reddit_state._eval_reddit_comment_parent_final_state(
                    request.witness,
                    relaxed_events,
                    instance,
                    state_probe,
                )
                if passed:
                    return True, readback_message
                message = readback_message
            else:
                message = f"{message}; {relaxed_message}" if message else relaxed_message
        if not events:
            return False, f"final-state source event missing: {message}"
        return reddit_state._eval_reddit_final_state(
            request.action_kind,
            request.witness,
            events,
            network_trace,
            instance,
            state_probe,
        )


__all__ = ["RedditFinalStateEvaluator"]
