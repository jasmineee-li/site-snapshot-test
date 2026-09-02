from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from warp_taskgen.agent_config import bind_task_to_instance
from warp_taskgen.agent_runtime import AgentResult
from warp_taskgen.phase_4 import execution as phase_4_execution
from warp_taskgen.phase_4 import preflight as phase_4_preflight
from warp_taskgen.runtime_composition import classifieds_listing_reply_poc

from ._fixtures import _prepared_adv_task


@pytest.mark.asyncio
async def test_loader_attaches_attempt_local_evidence() -> None:
    captured: dict[str, object] = {}

    action_started_at = datetime.now(UTC)

    def load(task, instance, seed_metadata, started_at):
        captured.update(
            {
                "task": task,
                "instance": instance,
                "seed_metadata": seed_metadata,
                "action_started_at": started_at,
            }
        )
        return "persisted-evidence"

    composition = replace(
        classifieds_listing_reply_poc(),
        reward_evidence_loader=load,
    )
    result = SimpleNamespace()
    task = {"id": "notification-1"}
    instance = {"site_url": "http://rocketchat.test"}
    seed_metadata = {"editor_call_results": [{"write_tokens": {"thread_id": "root-1"}}]}

    await phase_4_execution._attach_runtime_reward_evidence(
        composition,
        task=task,
        instance=instance,
        seed_metadata=seed_metadata,
        result=result,
        task_id="notification-1",
        action_started_at=action_started_at,
    )

    assert result.runtime_reward_evidence == "persisted-evidence"
    assert captured == {
        "task": task,
        "instance": instance,
        "seed_metadata": seed_metadata,
        "action_started_at": action_started_at,
    }


@pytest.mark.asyncio
async def test_loader_timeout_fails_closed(monkeypatch) -> None:
    def load(*_args):
        time.sleep(0.2)
        return "late-evidence"

    composition = replace(
        classifieds_listing_reply_poc(),
        reward_evidence_loader=load,
    )
    result = SimpleNamespace()
    monkeypatch.setattr(phase_4_execution, "_REWARD_EVALUATION_TIMEOUT_S", 0.01)

    await phase_4_execution._attach_runtime_reward_evidence(
        composition,
        task={},
        instance={},
        seed_metadata={},
        result=result,
        task_id="notification-timeout",
        action_started_at=datetime.now(UTC),
    )

    assert result.runtime_reward_evidence is None
    assert "timed out" in result.runtime_reward_evidence_error


@pytest.mark.asyncio
async def test_execution_loads_evidence_after_agent_before_grading(monkeypatch, tmp_path) -> None:
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    events: list[str] = []
    first_attempt_finished_at: datetime | None = None
    seed_metadata = {"editor_call_results": [{"write_tokens": {"thread_id": "root-1"}}]}

    async def fake_reset(task):
        return None

    async def fake_preflight(*args, **kwargs):
        return phase_4_preflight.PreflightReport(ok=True, mismatches=())

    class _FakeCleanup:
        def cleanup(self) -> None:
            events.append("cleanup")

    async def fake_apply_seed(seed, instance, **kwargs):
        events.append("seed")
        return _FakeCleanup(), seed_metadata

    def load_evidence(loaded_task, loaded_instance, loaded_metadata, action_started_at):
        events.append("evidence")
        assert loaded_task is task
        assert loaded_metadata is seed_metadata
        assert loaded_instance["site_url"] == instances[0].site_url
        assert action_started_at.tzinfo is UTC
        assert first_attempt_finished_at is not None
        assert action_started_at >= first_attempt_finished_at
        return "exact-readback"

    def fake_run_reward_function(reward, instance, agent_result=None, network_trace=None):
        events.append("reward")
        assert agent_result.runtime_reward_evidence == "exact-readback"
        return reward is task["reward_function"]["benign_reward"], "ok"

    class AgentLabAgentWrapper:
        calls = 0

        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            nonlocal first_attempt_finished_at
            events.append("agent")
            self.calls += 1
            if self.calls == 1:
                first_attempt_finished_at = datetime.now(UTC)
                return AgentResult(
                    elapsed=0.1,
                    steps=1,
                    is_done=False,
                    final_result=None,
                    status="error",
                    errors=["AgentLab browser step exceeded step timeout"],
                    network_trace=[],
                )
            return AgentResult(
                elapsed=0.1,
                steps=1,
                is_done=True,
                final_result="done",
                status="success",
                errors=[],
                network_trace=[],
            )

    composition = replace(
        classifieds_listing_reply_poc(),
        reward_evidence_loader=load_evidence,
    )
    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "preflight_adversarial_seed", fake_preflight)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_run_reward_function)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=AgentLabAgentWrapper(),
        instance=instances[0],
        task_dir=tmp_path,
        runtime_composition=composition,
    )

    assert result["outcome"] == "refused_or_ignored"
    assert events == ["seed", "agent", "agent", "evidence", "reward", "reward", "cleanup"]
