from __future__ import annotations

import json

import pytest

from worldsim.phase_3_triage import (
    TRIAGE_CACHE_VERSION,
    _build_triage_context,
    _build_triage_prompt,
    _finalize_model_decision,
    triage_failure_rules,
    triage_failures,
)


def _task(task_id: str = "task-1") -> dict:
    return {
        "id": task_id,
        "site": "shopping",
        "instruction": "Find the order",
        "start_urls": ["http://shopping.test/orders"],
        "agent_context": {},
    }


def _result(tmp_path, task_id: str = "task-1", *, outcome: str = "failed", message: str = "reward mismatch") -> dict:
    trajectory_dir = tmp_path / task_id
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    return {
        "task_id": task_id,
        "passed": False,
        "outcome": outcome,
        "message": message,
        "trajectory_dir": str(trajectory_dir),
    }


def test_triage_failure_rules_classifies_auth_failures(tmp_path):
    result = _result(tmp_path, message="reward mismatch")
    trajectory_dir = tmp_path / "task-1"
    (trajectory_dir / "history.json").write_text(
        json.dumps(
            [
                {"current_url": "http://shopping.test/login"},
                {"status": "session expired"},
            ]
        )
    )

    decision = triage_failure_rules(context=_build_triage_context(result=result, task=_task()))

    assert decision is not None
    assert decision.decision == "agent_limitation"
    assert decision.source == "rules"


def test_triage_failure_rules_classifies_infra_failures(tmp_path):
    result = _result(tmp_path, outcome="error", message="browser crashed")

    decision = triage_failure_rules(context=_build_triage_context(result=result, task=_task()))

    assert decision is not None
    assert decision.decision == "infra_error"
    assert decision.likely_root_cause == "infra_error"


def test_triage_failure_rules_classifies_offsite_drift(tmp_path):
    result = _result(tmp_path, message="reward mismatch")
    trajectory_dir = tmp_path / "task-1"
    (trajectory_dir / "history.json").write_text(
        json.dumps(
            [
                {"current_url": "https://duckduckgo.com/?q=shopping+order"},
                {"current_url": "https://duckduckgo.com/?q=shopping+tracking"},
            ]
        )
    )

    decision = triage_failure_rules(context=_build_triage_context(result=result, task=_task()))

    assert decision is not None
    assert decision.decision == "agent_limitation"


@pytest.mark.asyncio
async def test_triage_failures_escalates_when_no_model_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = _result(tmp_path, message="reward rejected final answer")

    decisions = await triage_failures(
        failed_results=[result],
        prepared_by_id={"task-1": _task()},
    )

    assert decisions == [
        {
            "task_id": "task-1",
            "decision": "needs_deep_diagnosis",
            "likely_root_cause": None,
            "confidence": 1.0,
            "reason": "No host-side triage model credentials configured; escalating conservatively.",
            "source": "rules",
            "escalate": True,
        }
    ]


@pytest.mark.asyncio
async def test_triage_failures_reuses_cached_decisions(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    def fail_if_called(prompt: str, model: str) -> str:
        raise AssertionError("cached triage should skip host-side model calls")

    monkeypatch.setattr("worldsim.phase_3_triage._call_openrouter", fail_if_called)
    result = _result(tmp_path, message="reward rejected final answer")
    result.update(
        {
            "triage_decision": "agent_limitation",
            "triage_likely_root_cause": "agent_limitation",
            "triage_confidence": 0.99,
            "triage_reason": "Clear login wall.",
            "triage_source": "rules",
            "triage_escalate": False,
            "triage_cache_version": TRIAGE_CACHE_VERSION,
        }
    )

    decisions = await triage_failures(
        failed_results=[result],
        prepared_by_id={"task-1": _task()},
    )

    assert decisions[0]["decision"] == "agent_limitation"
    assert decisions[0]["escalate"] is False


@pytest.mark.asyncio
async def test_triage_failures_ignores_stale_model_short_circuit_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    prompt_calls: list[str] = []

    def fake_call_openrouter(prompt: str, model: str) -> str:
        prompt_calls.append(prompt)
        return json.dumps(
            {
                "decision": "agent_limitation",
                "likely_root_cause": "agent_limitation",
                "confidence": 0.99,
                "reason": "Looks like a login wall.",
            }
        )

    monkeypatch.setattr("worldsim.phase_3_triage._call_openrouter", fake_call_openrouter)
    result = _result(tmp_path, message="reward rejected final answer")
    result.update(
        {
            "triage_decision": "agent_limitation",
            "triage_likely_root_cause": "agent_limitation",
            "triage_confidence": 0.99,
            "triage_reason": "Old model short circuit.",
            "triage_source": "model",
            "triage_escalate": False,
            "triage_cache_version": TRIAGE_CACHE_VERSION - 1,
        }
    )

    decisions = await triage_failures(
        failed_results=[result],
        prepared_by_id={"task-1": _task()},
    )

    assert len(prompt_calls) == 1
    assert decisions[0]["decision"] == "needs_deep_diagnosis"
    assert decisions[0]["source"] == "model"


def test_finalize_model_decision_always_escalates():
    low_confidence = _finalize_model_decision(
        {
            "task_id": "task-1",
            "decision": "agent_limitation",
            "likely_root_cause": "agent_limitation",
            "confidence": 0.89,
            "reason": "Maybe auth.",
            "source": "model",
        }
    )
    high_confidence = _finalize_model_decision(
        {
            "task_id": "task-1",
            "decision": "agent_limitation",
            "likely_root_cause": "agent_limitation",
            "confidence": 0.95,
            "reason": "Clear login wall.",
            "source": "model",
        }
    )

    assert low_confidence["decision"] == "needs_deep_diagnosis"
    assert low_confidence["escalate"] is True
    assert high_confidence["decision"] == "needs_deep_diagnosis"
    assert high_confidence["escalate"] is True


@pytest.mark.asyncio
async def test_triage_failures_model_path_escalates_and_redacts_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    prompt_holder: dict[str, str] = {}

    def fake_call_openrouter(prompt: str, model: str) -> str:
        prompt_holder["prompt"] = prompt
        return json.dumps(
            {
                "decision": "agent_limitation",
                "likely_root_cause": "agent_limitation",
                "confidence": 0.99,
                "reason": "Looks like a login wall.",
            }
        )

    monkeypatch.setattr("worldsim.phase_3_triage._call_openrouter", fake_call_openrouter)
    result = _result(tmp_path, message="reward rejected final answer")
    trajectory_dir = tmp_path / "task-1"
    (trajectory_dir / "history.json").write_text(
        json.dumps(
            [
                {
                    "text": "Agent saw password: super-secret and bearer token abc123 while browsing."
                }
            ]
        )
    )
    task = _task()
    task["agent_context"] = {
        "authentication": {
            "pre_authenticated": False,
            "credentials": {"username": "demo", "password": "super-secret"},
            "description": "Use the seeded shopping account.",
        },
        "auth_mechanism": {
            "type": "http_headers",
            "http_headers": {"headers": {"Authorization": "Bearer abc123"}},
        },
    }

    decisions = await triage_failures(
        failed_results=[result],
        prepared_by_id={"task-1": task},
    )

    assert decisions[0]["decision"] == "needs_deep_diagnosis"
    assert decisions[0]["source"] == "model"
    assert "super-secret" not in prompt_holder["prompt"]
    assert "abc123" not in prompt_holder["prompt"]
    assert '"credential_keys": [' in prompt_holder["prompt"]


@pytest.mark.asyncio
async def test_triage_failures_model_refusal_escalates_conservatively(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fake_call_openrouter(prompt: str, model: str) -> str:
        return "I'm not going to help with that."

    monkeypatch.setattr("worldsim.phase_3_triage._call_openrouter", fake_call_openrouter)
    result = _result(tmp_path, message="reward rejected final answer")

    decisions = await triage_failures(
        failed_results=[result],
        prepared_by_id={"task-1": _task()},
    )

    assert decisions[0]["decision"] == "needs_deep_diagnosis"
    assert decisions[0]["source"] == "model"
    assert "openrouter_invalid_or_refused" in decisions[0]["reason"]


def test_build_triage_prompt_sanitizes_auth_context(tmp_path):
    result = _result(tmp_path, message="reward mismatch")
    trajectory_dir = tmp_path / "task-1"
    (trajectory_dir / "history.json").write_text(json.dumps([{"text": "password=super-secret"}]))
    task = _task()
    task["agent_context"] = {
        "authentication": {
            "pre_authenticated": False,
            "credentials": {"username": "demo", "password": "super-secret"},
            "description": "Use seeded creds.",
        }
    }

    prompt = _build_triage_prompt(context=_build_triage_context(result=result, task=task))

    assert "super-secret" not in prompt
    assert '"has_credentials": true' in prompt
