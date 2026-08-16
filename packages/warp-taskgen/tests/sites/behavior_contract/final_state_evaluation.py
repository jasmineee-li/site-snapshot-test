"""Assertions for the local Final State Evaluation owner seam."""

from __future__ import annotations

from dataclasses import replace

from tests.sites.behavior_contract.exact_resource_evidence import ExactResourceEvidenceCase
from warp_taskgen.rewards.evidence import EvidencePolicy
from warp_taskgen.rewards.final_state_catalog import (
    FinalStateEvaluationRequest,
    FinalStateEvaluatorCatalog,
)


def assert_final_state_evaluation_behavior(
    evaluator_catalog: FinalStateEvaluatorCatalog,
    case: ExactResourceEvidenceCase,
    *,
    expected_message: str,
) -> None:
    """Check source-event plus exact-state matching and fail-closed negatives."""

    request = _request(case)
    result = evaluator_catalog.evaluate(request)
    assert result == (True, expected_message)

    missing_source = replace(request, network_trace=())
    assert evaluator_catalog.evaluate(missing_source)[0] is False

    wrong_site = replace(request, site="different-site")
    assert evaluator_catalog.evaluate(wrong_site)[0] is False

    stale_state = dict(request.state_probe or {})
    stale_state["comment_id"] = "stale-foreign-id"
    assert evaluator_catalog.evaluate(replace(request, state_probe=stale_state))[0] is False


def _request(case: ExactResourceEvidenceCase) -> FinalStateEvaluationRequest:
    path = case.action_path
    state_probe = {
        "site": case.site,
        "origin": case.origin,
        "thread_id": case.parent_id,
        "comment_id": case.resource_id,
        "actor": case.actor,
        "body": case.body,
        "parent_kind": case.parent_kind,
        "resource_kind": case.resource_kind,
    }
    return FinalStateEvaluationRequest(
        benchmark=case.benchmark,
        site=case.site,
        action_kind=case.action_kind,
        witness=case.signature,
        network_expected={"method": "POST", "path": path},
        state_probe=state_probe,
        evidence_policy=EvidencePolicy(
            required=frozenset({"source_event", "state_readback"}),
            allowed_source=frozenset({"network_event"}),
        ),
        network_trace=({"method": "POST", "url": f"{case.origin}{path}"},),
        instance={"site": case.site, "site_url": case.origin},
        initial_events=(),
        initial_message="",
    )
