"""Provider, accounting, budget, and diagnostics tests for the matched rewrite study."""

from __future__ import annotations

import asyncio
import json
import re
from copy import deepcopy
from types import SimpleNamespace
from typing import cast

import pytest

from phase_4.test_matched_rewrite_study import (
    RecordingProvider,
    _baseline,
    _candidate,
    _pass_qa,
)
from warp_taskgen.phase_4 import matched_rewrite_study as study
from warp_taskgen.phase_4.matched_rewrite_contracts import (
    BrowserOutcome,
    DiagnosisOutcome,
    MatchedCallPolicy,
    MatchedStudyBudget,
    OrdinaryGuidance,
    Phase4Runtime,
    ProposalOutcome,
    Usage,
)
from warp_taskgen.phase_4.matched_rewrite_ordinary_api import usage_from_diagnostics
from warp_taskgen.phase_4.matched_rewrite_provider import ExistingPhase4AttemptAdapter


def test_existing_adapter_ordinary_critique_is_typed_and_neutral(tmp_path, monkeypatch):
    baseline = _baseline()
    config = study.MatchedRewriteStudyConfig(
        call_policy=MatchedCallPolicy(
            model="sandbox-model",
            provider="anthropic",
            runner="messages",
        ),
        budget=MatchedStudyBudget(
            per_arm_max_tokens=10_000,
            total_max_tokens=20_000,
            per_arm_max_cost_usd=10.0,
            total_max_cost_usd=20.0,
        )
    )

    calls: list[dict[str, object]] = []
    cost_phases: list[str] = []

    from warp_taskgen.phase_4 import matched_rewrite_ordinary_api as ordinary_api

    monkeypatch.setattr(
        ordinary_api.cost_tracker,
        "record",
        lambda phase, summary, **kwargs: cost_phases.append(phase),
    )

    async def create(**kwargs: object):
        calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=json.dumps(
                        {
                            "critique": "Review ordinary wording.",
                            "guidance": "Preserve anchors.",
                            "rewrite_guidance": "Use concise site-native wording.",
                            "focus": "clarity",
                            "confidence": "medium",
                        }
                    ),
                )
            ],
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=8,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    adapter = ExistingPhase4AttemptAdapter(
        Phase4Runtime(
            primary_instance=object(),
            all_instances=(),
            agent_factory=lambda: object(),
            task_dir_root=tmp_path,
            host_client=client,
            host_provider="anthropic",
        )
    )
    adapter.bind(baseline.binding)
    request = study._request(
        baseline,
        config,
        arm="ordinary",
        stage="ordinary_critique",
    )
    outcome = asyncio.run(adapter.run(request))
    assert isinstance(outcome, DiagnosisOutcome)
    assert isinstance(outcome.guidance, OrdinaryGuidance)
    assert "tp" not in str(outcome.guidance.to_dict()).lower()
    assert "reward" not in str(outcome.guidance.to_dict()).lower()
    assert outcome.usage.available
    assert outcome.diagnostics is not None
    assert outcome.diagnostics["label"] == "matched-rewrite-ordinary-critique"
    assert outcome.diagnostics["provider"] == "anthropic"
    assert outcome.diagnostics["runner"] == "messages"
    assert outcome.diagnostics["completion_responses"]
    assert outcome.diagnostics["completion_kwargs"][0]["extra_body"]
    assert calls[0]["model"] == "sandbox-model"
    assert calls[0]["max_tokens"] == 8192
    ordinary_prompt = str(calls[0]["messages"])
    assert "transcript_purpose" not in ordinary_prompt
    assert "verbalized_eval_awareness" not in ordinary_prompt
    assert "reward_trace" not in ordinary_prompt
    assert "<witness>" not in ordinary_prompt
    sections = set(re.findall(r"<([a-z_]+)>", ordinary_prompt))
    assert {"payload", "trajectory_summary", "rewrite_constraints", "task_context"} <= sections
    assert cost_phases == ["phase_4:matched_rewrite_study:ordinary_critique"]


def test_existing_adapter_passes_matched_policy_to_tp_and_rewrite(monkeypatch, tmp_path):
    baseline = _baseline()
    config = study.MatchedRewriteStudyConfig(
        call_policy=MatchedCallPolicy(
            model="sandbox-model",
            provider="anthropic",
            runner="messages",
            max_tokens=2048,
            semantic_retries=1,
            transport_retries=2,
            temperature=0.4,
        ),
        budget=MatchedStudyBudget(
            per_arm_max_tokens=10_000,
            total_max_tokens=20_000,
            per_arm_max_cost_usd=10.0,
            total_max_cost_usd=20.0,
        ),
    )
    diagnostics = {
        "attempts": 1,
        "transport_attempts": 1,
        "completion_responses": [
            {"usage": {"input_tokens": 1, "output_tokens": 1}}
        ],
    }
    captured: list[dict[str, object]] = []

    from warp_taskgen.phase_4 import eval_awareness_cue_api, eval_awareness_rewrite_api

    async def fake_cue(*args: object, **kwargs: object):
        captured.append(kwargs)
        return {
            "status": "ok",
            "diagnosis": {"trigger_source": "non-real TP"},
            "api_diagnostics": diagnostics,
        }

    async def fake_rewrite(*args: object, **kwargs: object):
        captured.append(kwargs)
        return {
            **_candidate(baseline.task_copy()),
            "matched_rewrite_api_diagnostics": diagnostics,
        }

    monkeypatch.setattr(eval_awareness_cue_api, "run_eval_awareness_cue_api", fake_cue)
    monkeypatch.setattr(
        eval_awareness_rewrite_api,
        "generate_eval_awareness_rewrite_api",
        fake_rewrite,
    )
    adapter = ExistingPhase4AttemptAdapter(
        Phase4Runtime(
            primary_instance=object(),
            all_instances=(),
            agent_factory=lambda: object(),
            task_dir_root=tmp_path,
            host_client=object(),
            host_provider="anthropic",
        )
    )
    adapter.bind(baseline.binding)
    tp_request = study._request(baseline, config, arm="tp_guided", stage="tp_diagnosis")
    tp_outcome = asyncio.run(adapter.run(tp_request))
    assert isinstance(tp_outcome, DiagnosisOutcome) and tp_outcome.status == "ok"
    rewrite_request = study._request(
        baseline,
        config,
        arm="tp_guided",
        stage="proposal",
        guidance=tp_outcome.guidance,
    )
    rewrite_outcome = asyncio.run(adapter.run(rewrite_request))
    assert isinstance(rewrite_outcome, ProposalOutcome) and rewrite_outcome.status == "ok"
    assert captured[0]["sandbox_model"] == captured[1]["sandbox_model"] == "sandbox-model"
    assert captured[0]["max_tokens"] == captured[1]["max_tokens"] == 2048
    assert captured[0]["semantic_retries"] == captured[1]["semantic_retries"] == 1
    assert captured[0]["transport_retries"] == captured[1]["transport_retries"] == 2
    assert captured[0]["temperature"] == captured[1]["temperature"] == 0.4
    assert captured[0]["cost_phase"] == "phase_4:matched_rewrite_study:tp_diagnosis"
    assert captured[1]["cost_phase"] == "phase_4:matched_rewrite_study:proposal"


def test_pair_accounting_retains_model_usage_when_browser_artifact_is_unavailable():
    accounting = study.PairAccounting()
    accounting.record(
        "ordinary_critique",
        DiagnosisOutcome(
            status="ok",
            guidance=OrdinaryGuidance(critique="Keep the note clear."),
            usage=Usage(12, 8, 0.25),
        ),
    )
    accounting.record(
        "browser",
        BrowserOutcome(
            status="ok",
            result={"task_id": "task-1", "outcome": "complied"},
            usage=Usage.unavailable("browser_usage_recorded_by_phase4_artifact"),
        ),
    )
    projection = accounting.to_dict()
    assert projection["input_tokens"] == 12
    assert projection["output_tokens"] == 8
    assert projection["usage_status"] == "available"
    assert projection["usage_unavailable_reasons"] == [
        "browser_usage_recorded_by_phase4_artifact"
    ]


def test_usage_from_diagnostics_counts_semantic_and_transport_attempts():
    response = {"usage": {"input_tokens": 12, "output_tokens": 8}}
    semantic = usage_from_diagnostics(
        {
            "attempts": 3,
            "transport_attempts": 1,
            "completion_responses": [response, response],
        },
        model="sandbox-model",
        fallback_reason="usage_missing",
    )
    assert semantic.available
    assert semantic.attempts == 3
    assert (semantic.input_tokens, semantic.output_tokens) == (24, 16)

    transport = usage_from_diagnostics(
        {
            "attempts": 1,
            "transport_attempts": 4,
            "completion_responses": [response],
        },
        model="sandbox-model",
        fallback_reason="usage_missing",
    )
    assert transport.available
    assert transport.attempts == 4
    assert (transport.input_tokens, transport.output_tokens) == (12, 8)


def test_existing_adapter_requires_budget_before_call(tmp_path):
    baseline = _baseline()
    calls = 0

    async def create(**kwargs: object):
        nonlocal calls
        calls += 1
        raise AssertionError("provider call must be blocked without matched budget")

    adapter = ExistingPhase4AttemptAdapter(
        Phase4Runtime(
            primary_instance=object(),
            all_instances=(),
            agent_factory=lambda: object(),
            task_dir_root=tmp_path,
            host_client=SimpleNamespace(messages=SimpleNamespace(create=create)),
            host_provider="anthropic",
        )
    )
    adapter.bind(baseline.binding)
    request = study._request(
        baseline,
        study.MatchedRewriteStudyConfig(),
        arm="ordinary",
        stage="ordinary_critique",
    )
    outcome = asyncio.run(adapter.run(request))
    assert isinstance(outcome, DiagnosisOutcome)
    assert outcome.status == "failed"
    assert outcome.failure == "matched_budget_missing"
    assert outcome.usage.unavailable_reason == "matched_budget_missing"
    assert calls == 0


def test_existing_adapter_rejects_unconfigured_provider_identity(tmp_path):
    baseline = _baseline()
    config = study.MatchedRewriteStudyConfig(
        budget=MatchedStudyBudget(
            per_arm_max_tokens=10_000,
            total_max_tokens=20_000,
            per_arm_max_cost_usd=10.0,
            total_max_cost_usd=20.0,
        )
    )
    calls = 0

    async def create(**kwargs: object):
        nonlocal calls
        calls += 1
        raise AssertionError("provider call must be blocked without identity")

    adapter = ExistingPhase4AttemptAdapter(
        Phase4Runtime(
            primary_instance=object(),
            all_instances=(),
            agent_factory=lambda: object(),
            task_dir_root=tmp_path,
            host_client=SimpleNamespace(messages=SimpleNamespace(create=create)),
            host_provider="anthropic",
        )
    )
    adapter.bind(baseline.binding)
    request = study._request(baseline, config, arm="ordinary", stage="ordinary_critique")
    outcome = asyncio.run(adapter.run(request))
    assert isinstance(outcome, DiagnosisOutcome)
    assert outcome.failure == "matched_provider_identity_unconfigured"
    assert outcome.usage.unavailable_reason == "matched_provider_identity_unconfigured"
    assert calls == 0


def test_existing_adapter_rejects_injected_client_provider_mismatch(tmp_path):
    baseline = _baseline()
    config = study.MatchedRewriteStudyConfig(
        call_policy=MatchedCallPolicy(
            model="sandbox-model",
            provider="openrouter",
            runner="messages",
        ),
        budget=MatchedStudyBudget(
            per_arm_max_tokens=10_000,
            total_max_tokens=20_000,
            per_arm_max_cost_usd=10.0,
            total_max_cost_usd=20.0,
        ),
    )
    adapter = ExistingPhase4AttemptAdapter(
        Phase4Runtime(
            primary_instance=object(),
            all_instances=(),
            agent_factory=lambda: object(),
            task_dir_root=tmp_path,
            host_client=object(),
            host_provider="anthropic",
        )
    )
    adapter.bind(baseline.binding)

    outcome = asyncio.run(
        adapter.run(
            study._request(
                baseline,
                config,
                arm="ordinary",
                stage="ordinary_critique",
            )
        )
    )

    assert outcome.failure == "matched_provider_identity_mismatch"


def test_existing_adapter_rejects_default_client_transport_mismatch(
    tmp_path, monkeypatch
):
    baseline = _baseline()
    config = study.MatchedRewriteStudyConfig(
        call_policy=MatchedCallPolicy(
            model="sandbox-model",
            provider="openrouter",
            runner="messages",
        ),
        budget=MatchedStudyBudget(
            per_arm_max_tokens=10_000,
            total_max_tokens=20_000,
            per_arm_max_cost_usd=10.0,
            total_max_cost_usd=20.0,
        ),
    )
    from warp_taskgen.phase_4 import matched_rewrite_provider as provider_module

    monkeypatch.setattr(provider_module, "resolved_messages_provider", lambda: "anthropic")
    monkeypatch.setattr(
        provider_module,
        "get_client",
        lambda: pytest.fail("transport mismatch must fail before client construction"),
    )
    adapter = ExistingPhase4AttemptAdapter(
        Phase4Runtime(
            primary_instance=object(),
            all_instances=(),
            agent_factory=lambda: object(),
            task_dir_root=tmp_path,
            host_provider="openrouter",
        )
    )
    adapter.bind(baseline.binding)

    outcome = asyncio.run(
        adapter.run(
            study._request(
                baseline,
                config,
                arm="ordinary",
                stage="ordinary_critique",
            )
        )
    )

    assert outcome.failure == "matched_provider_transport_mismatch"


def test_existing_adapter_blocks_after_browser_usage_is_unavailable(tmp_path):
    baseline = _baseline()
    config = study.MatchedRewriteStudyConfig(
        call_policy=MatchedCallPolicy(
            model="sandbox-model",
            provider="anthropic",
            runner="messages",
        ),
        budget=MatchedStudyBudget(
            per_arm_max_tokens=10_000,
            total_max_tokens=20_000,
            per_arm_max_cost_usd=10.0,
            total_max_cost_usd=20.0,
        ),
    )
    calls = 0

    async def create(**kwargs: object):
        nonlocal calls
        calls += 1
        raise AssertionError("provider call must be blocked after unknown browser usage")

    adapter = ExistingPhase4AttemptAdapter(
        Phase4Runtime(
            primary_instance=object(),
            all_instances=(),
            agent_factory=lambda: object(),
            task_dir_root=tmp_path,
            host_client=SimpleNamespace(messages=SimpleNamespace(create=create)),
            host_provider="anthropic",
        )
    )
    adapter.bind(baseline.binding)
    browser = study._request(baseline, config, arm="ordinary", stage="browser")
    browser_outcome = asyncio.run(adapter.run(browser))
    assert isinstance(browser_outcome, BrowserOutcome)
    assert browser_outcome.usage.unavailable_reason == "browser_usage_unavailable"
    ordinary = study._request(baseline, config, arm="ordinary", stage="ordinary_critique")
    ordinary_outcome = asyncio.run(adapter.run(ordinary))
    assert isinstance(ordinary_outcome, DiagnosisOutcome)
    assert ordinary_outcome.failure == "matched_budget_usage_unknown"
    assert calls == 0


def test_existing_adapter_blocks_next_call_at_known_token_ceiling(tmp_path):
    baseline = _baseline()
    config = study.MatchedRewriteStudyConfig(
        call_policy=MatchedCallPolicy(
            model="sandbox-model",
            provider="anthropic",
            runner="messages",
        ),
        budget=MatchedStudyBudget(
            per_arm_max_tokens=20,
            total_max_tokens=100,
            per_arm_max_cost_usd=10.0,
            total_max_cost_usd=20.0,
        )
    )
    calls: list[dict[str, object]] = []

    async def create(**kwargs: object):
        calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=json.dumps(
                        {
                            "critique": "Review ordinary wording.",
                            "guidance": "Preserve anchors.",
                            "rewrite_guidance": "Use concise site-native wording.",
                            "focus": "clarity",
                            "confidence": "medium",
                        }
                    ),
                )
            ],
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=8,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )

    adapter = ExistingPhase4AttemptAdapter(
        Phase4Runtime(
            primary_instance=object(),
            all_instances=(),
            agent_factory=lambda: object(),
            task_dir_root=tmp_path,
            host_client=SimpleNamespace(messages=SimpleNamespace(create=create)),
            host_provider="anthropic",
        )
    )
    adapter.bind(baseline.binding)
    request = study._request(baseline, config, arm="ordinary", stage="ordinary_critique")
    first = asyncio.run(adapter.run(request))
    second = asyncio.run(adapter.run(request))
    assert isinstance(first, DiagnosisOutcome) and first.status == "ok"
    assert isinstance(second, DiagnosisOutcome) and second.status == "failed"
    assert second.failure == "matched_budget_per_arm_token_ceiling"
    assert len(calls) == 1


def test_checkpoint_rejects_pre_budget_schema_version(monkeypatch):
    baseline = _baseline()
    monkeypatch.setattr(study, "_select", lambda baseline, rows: {"arms": rows})
    monkeypatch.setattr(study, "_qa", _pass_qa)
    result = asyncio.run(
        study.run_matched_rewrite_study(baseline, attempt_provider=RecordingProvider())
    )
    old_checkpoint = deepcopy(result["checkpoint"])
    old_checkpoint["schema_version"] = study.STUDY_SCHEMA_VERSION - 1
    with pytest.raises(study.IncompatibleMatchedRewriteResume):
        asyncio.run(study.run_matched_rewrite_study(baseline, checkpoint=old_checkpoint))


def test_aggregate_keeps_browser_unavailable_reason_with_model_totals():
    rows = {
        "tp_guided": {
            "accounting": {
                "diagnosis_calls": 1,
                "proposal_calls": 1,
                "repair_calls": 0,
                "browser_attempts": 1,
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20,
                "cost_usd": 0.25,
                "retry_attempts": 0,
                "usage_status": "available",
                "usage_unavailable_reasons": ["browser_usage_recorded_by_phase4_artifact"],
            }
        },
        "ordinary": {
            "accounting": {
                "diagnosis_calls": 1,
                "proposal_calls": 1,
                "repair_calls": 0,
                "browser_attempts": 1,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cost_usd": 0.10,
                "retry_attempts": 0,
                "usage_status": "available",
                "usage_unavailable_reasons": [],
            }
        },
    }
    aggregate = study._aggregate(cast(dict, rows))
    assert aggregate["input_tokens"] == 22
    assert aggregate["output_tokens"] == 13
    assert aggregate["usage_status"] == "available"
    assert aggregate["usage_unavailable_reasons"] == [
        "browser_usage_recorded_by_phase4_artifact"
    ]


def test_matched_call_policy_is_bound_to_baseline_model_and_retry_ceiling():
    policy = MatchedCallPolicy(
        model="sandbox-model",
        max_tokens=4096,
        semantic_retries=1,
        transport_retries=2,
    )
    config = study.MatchedRewriteStudyConfig(call_policy=policy)
    assert config.resolve_call_policy("sandbox-model").to_dict() == {
        "model": "sandbox-model",
        "provider": "unconfigured",
        "runner": "unconfigured",
        "max_tokens": 4096,
        "semantic_retries": 1,
        "transport_retries": 2,
        "temperature": 0.2,
    }
    with pytest.raises(ValueError, match="must match"):
        config.resolve_call_policy("other-model")


def test_matched_call_policy_identity_is_bound_to_run_definition():
    baseline = _baseline()
    config = study.MatchedRewriteStudyConfig(
        call_policy=MatchedCallPolicy(
            model="sandbox-model",
            provider="openrouter",
            runner="messages",
        )
    )
    with pytest.raises(ValueError, match="call_policy"):
        study.validate_baseline_binding(baseline, config)
