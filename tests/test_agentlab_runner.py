from __future__ import annotations

import json

import pytest

from worldsim.runners.agentlab import (
    AgentLabAgentWrapper,
    _parse_exp_result,
    _persist_result_sentinel,
)


@pytest.mark.asyncio
async def test_agentlab_wrapper_satisfies_worker_lifecycle_contract():
    agent = AgentLabAgentWrapper(agent_args=object(), model="demo-model")

    await agent.setup("http://example.test")
    await agent.teardown()


def test_parse_exp_result_marks_missing_summary_as_error(tmp_path):
    parsed = _parse_exp_result("task-1", tmp_path, elapsed=1.5)

    assert parsed["task_id"] == "task-1"
    assert parsed["status"] == "error"
    assert parsed["outcome"] == "error"


def test_persist_result_sentinel_writes_canonical_result_json(tmp_path):
    _persist_result_sentinel(
        {"id": "task-2"},
        tmp_path,
        {
            "task_id": "task-2",
            "passed": True,
            "message": "passed",
            "status": "success",
            "elapsed": 2.0,
            "steps": 4,
            "is_done": True,
            "errors": [],
        },
    )

    data = json.loads((tmp_path / "result.json").read_text())
    assert data["task_id"] == "task-2"
    assert data["passed"] is True
    assert data["status"] == "success"
