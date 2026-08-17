"""Fake local final-state evaluator for the test-only Site."""

from __future__ import annotations

import hashlib

from tests.sites.synthetic_discussion_forum.site import ORIGIN, SITE
from warp_taskgen.rewards.final_state_catalog import (
    FinalStateEvaluationRequest,
    FinalStateEvaluatorCatalog,
)


class SyntheticDiscussionForumEvaluator:
    benchmark = "webarena_verified"
    site = SITE

    def evaluate(self, request: FinalStateEvaluationRequest) -> tuple[bool, str]:
        state = request.state_probe or {}
        expected = request.network_expected
        source_event = any(
            event.get("method") == expected.get("method")
            and str(event.get("url") or "") == f"{ORIGIN}{expected.get('path')}"
            for event in request.network_trace
        )
        exact = (
            request.action_kind == "submit_comment"
            and source_event
            and state.get("site") == SITE
            and state.get("origin") == ORIGIN
            and state.get("parent_kind") == "thread"
            and state.get("resource_kind") == "comment"
            and state.get("thread_id") == "17"
            and state.get("comment_id") == "42"
            and state.get("actor") == "participant"
            and hashlib.sha256(str(state.get("body") or "").encode("utf-8")).hexdigest()
            == request.witness
        )
        return (
            exact,
            "exact synthetic discussion forum final state"
            if exact
            else "synthetic final state mismatch",
        )


evaluator_catalog = FinalStateEvaluatorCatalog.from_evaluators(
    (SyntheticDiscussionForumEvaluator(),)
)
