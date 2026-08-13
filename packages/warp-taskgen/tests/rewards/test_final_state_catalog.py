from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from warp_taskgen.rewards import (
    FinalStateEvaluationRequest,
    FinalStateEvaluatorCatalog,
    default_final_state_evaluator_catalog,
    run_reward_function,
)
from warp_taskgen.rewards.evidence import EvidencePolicy


@dataclass(frozen=True)
class _FakeFinalStateEvaluator:
    benchmark: str = "webarena_verified"
    site: str = "fake"

    def evaluate(self, request: FinalStateEvaluationRequest) -> tuple[bool, str]:
        if not request.initial_events:
            return False, "fake source event missing"
        probe = request.state_probe or {}
        if probe.get("created_id") != "17":
            return False, "fake created resource identity mismatch"
        if request.witness != "exact witness":
            return False, "fake persisted witness mismatch"
        return True, "fake exact final state contains witness"


def _fake_reward() -> dict[str, Any]:
    return {
        "type": "FinalStateEvaluator",
        "expected": {
            "action_kind": "create_message",
            "contains": "exact witness",
            "network_event": {
                "url": "https://fake.local/messages",
                "http_method": "POST",
            },
            "state_probe": {"site": "fake", "created_id": "17"},
        },
    }


def _fake_instance() -> dict[str, Any]:
    return {
        "benchmark": "WebArena Verified",
        "site_name": "fake",
        "site_url": "https://fake.local",
    }


def _fake_trace() -> list[dict[str, Any]]:
    return [{"url": "https://fake.local/messages", "method": "POST"}]


def test_injected_final_state_evaluator_is_scoped_to_one_catalog() -> None:
    catalog = FinalStateEvaluatorCatalog.from_evaluators([_FakeFinalStateEvaluator()])
    passed, message = run_reward_function(
        _fake_reward(),
        _fake_instance(),
        network_trace=_fake_trace(),
        final_state_catalog=catalog,
    )
    assert (passed, message) == (True, "fake exact final state contains witness")
    assert default_final_state_evaluator_catalog().bindings == (
        ("webarena_verified", "gitlab"),
        ("webarena_verified", "reddit"),
    )

    removed, removed_message = run_reward_function(
        _fake_reward(),
        _fake_instance(),
        network_trace=_fake_trace(),
    )
    assert removed is False
    assert "unsupported site 'fake'" in removed_message


def test_fake_final_state_evaluator_fails_closed_on_missing_or_stale_evidence() -> None:
    catalog = FinalStateEvaluatorCatalog.from_evaluators([_FakeFinalStateEvaluator()])
    missing, missing_message = run_reward_function(
        _fake_reward(),
        _fake_instance(),
        network_trace=[],
        final_state_catalog=catalog,
    )
    assert (missing, missing_message) == (False, "fake source event missing")

    stale_reward = _fake_reward()
    stale_reward["expected"]["state_probe"]["created_id"] = "seeded-3"
    stale, stale_message = run_reward_function(
        stale_reward,
        _fake_instance(),
        network_trace=_fake_trace(),
        final_state_catalog=catalog,
    )
    assert (stale, stale_message) == (False, "fake created resource identity mismatch")


def test_catalog_rejects_duplicates_and_adapter_errors_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate final-state evaluator"):
        FinalStateEvaluatorCatalog.from_evaluators(
            [_FakeFinalStateEvaluator(), _FakeFinalStateEvaluator(benchmark="webarena-verified")]
        )
    with pytest.raises(ValueError, match="keys must contain text"):
        FinalStateEvaluatorCatalog(
            {("webarena_verified", 7): _FakeFinalStateEvaluator()}  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="only WebArena Verified"):
        FinalStateEvaluatorCatalog.from_evaluators(
            [_FakeFinalStateEvaluator(benchmark="stwebagentbench")]
        )

    class BrokenEvaluator(_FakeFinalStateEvaluator):
        def evaluate(self, request: FinalStateEvaluationRequest) -> tuple[bool, str]:
            raise RuntimeError("broken fake")

    catalog = FinalStateEvaluatorCatalog.from_evaluators([BrokenEvaluator()])
    passed, message = run_reward_function(
        _fake_reward(),
        _fake_instance(),
        network_trace=_fake_trace(),
        final_state_catalog=catalog,
    )
    assert passed is False
    assert "adapter failed" in message
    assert "broken fake" in message

    class InvalidEvaluator(_FakeFinalStateEvaluator):
        def evaluate(self, request: FinalStateEvaluationRequest):
            return {"passed": True}

    catalog = FinalStateEvaluatorCatalog.from_evaluators([InvalidEvaluator()])
    passed, message = run_reward_function(
        _fake_reward(),
        _fake_instance(),
        network_trace=_fake_trace(),
        final_state_catalog=catalog,
    )
    assert passed is False
    assert "invalid result" in message


def test_catalog_rejects_malformed_request_without_raising() -> None:
    catalog = FinalStateEvaluatorCatalog.from_evaluators([_FakeFinalStateEvaluator()])

    assert catalog.evaluate(None) == (  # type: ignore[arg-type]
        False,
        "FinalStateEvaluator received an invalid request",
    )


def test_injected_catalog_requires_explicit_benchmark_identity() -> None:
    catalog = FinalStateEvaluatorCatalog.from_evaluators([_FakeFinalStateEvaluator()])
    instance = _fake_instance()
    instance.pop("benchmark")

    passed, message = run_reward_function(
        _fake_reward(),
        instance,
        network_trace=_fake_trace(),
        final_state_catalog=catalog,
    )

    assert passed is False
    assert message == "FinalStateEvaluator requires explicit benchmark metadata"


def test_homebrew_composite_threads_injected_final_state_catalog() -> None:
    catalog = FinalStateEvaluatorCatalog.from_evaluators([_FakeFinalStateEvaluator()])
    reward = {"eval": [{"evaluator": "FinalStateEvaluator", **_fake_reward()}]}
    passed, message = run_reward_function(
        reward,
        _fake_instance(),
        network_trace=_fake_trace(),
        final_state_catalog=catalog,
    )
    assert passed is True
    assert "[FinalStateEvaluator] PASS" in message


def test_canonical_task_id_keeps_vendor_evaluator_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warp_taskgen.rewards import dispatcher

    calls: list[dict[str, Any]] = []

    def fake_vendor(reward, instance, agent_result, network_trace):
        calls.append(reward)
        return True, "vendor evaluator"

    monkeypatch.setattr(dispatcher, "_run_webarena_verified_eval", fake_vendor)
    catalog = FinalStateEvaluatorCatalog.from_evaluators([_FakeFinalStateEvaluator()])
    passed, message = run_reward_function(
        {"task_id": 17, "eval": [{"evaluator": "FinalStateEvaluator"}]},
        _fake_instance(),
        final_state_catalog=catalog,
    )
    assert (passed, message) == (True, "vendor evaluator")
    assert len(calls) == 1


def test_final_state_request_snapshots_nested_evidence() -> None:
    trace = [{"url": "https://fake.local/messages", "post_data": {"id": "17"}}]
    probe = {"site": "fake", "created": {"id": "17"}}
    request = FinalStateEvaluationRequest(
        benchmark="webarena_verified",
        site="fake",
        action_kind="create_message",
        witness="exact witness",
        network_expected={"url": "https://fake.local/messages"},
        state_probe=probe,
        evidence_policy=EvidencePolicy(frozenset({"source_event"}), frozenset({"network_event"})),
        network_trace=tuple(trace),
        instance=_fake_instance(),
        initial_events=tuple(trace),
        initial_message="matched",
    )
    trace[0]["post_data"]["id"] = "changed"
    probe["created"]["id"] = "changed"

    assert request.network_trace[0]["post_data"]["id"] == "17"
    assert request.state_probe["created"]["id"] == "17"
