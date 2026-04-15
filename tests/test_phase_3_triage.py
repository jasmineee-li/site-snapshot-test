from __future__ import annotations

import json

import pytest

from worldsim.phase_3_triage import _finalize_model_decision, triage_failure_rules, triage_failures


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
    result = _result(tmp_path, message="Please sign in to continue")
    trajectory_dir = tmp_path / "task-1"
    (trajectory_dir / "history.json").write_text(json.dumps([{"text": "Please sign in"}]))

    decision = triage_failure_rules(result=result, task=_task())

    assert decision is not None
    assert decision.decision == "agent_limitation"
    assert decision.source == "rules"


def test_triage_failure_rules_classifies_infra_failures(tmp_path):
    result = _result(tmp_path, outcome="error", message="browser crashed")

    decision = triage_failure_rules(result=result, task=_task())

    assert decision is not None
    assert decision.decision == "infra_error"
    assert decision.likely_root_cause == "infra_error"


def test_triage_failure_rules_classifies_offsite_drift(tmp_path):
    result = _result(tmp_path, message="searched for answer on DuckDuckGo")
    trajectory_dir = tmp_path / "task-1"
    (trajectory_dir / "history.json").write_text(
        json.dumps([{"text": "Opened DuckDuckGo and searched for answer"}])
    )

    decision = triage_failure_rules(result=result, task=_task())

    assert decision is not None
    assert decision.decision == "agent_limitation"


@pytest.mark.asyncio
async def test_triage_failures_escalates_when_no_model_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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


def test_finalize_model_decision_requires_high_confidence_to_short_circuit():
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
    assert high_confidence["decision"] == "agent_limitation"
    assert high_confidence["escalate"] is False
