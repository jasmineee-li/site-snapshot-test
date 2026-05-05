# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

@pytest.mark.asyncio
async def test_run_adversarial_task_passes_instance_id_to_agent_run(monkeypatch, tmp_path):
    """Phase 4 dispatch must thread per-replica ``instance_id`` to ``agent.run``."""
    from worldsim.phases.phase_0d_auth_bootstrap import phase_0d_instance_id

    instance = BenchmarkInstance(
        site_name="gitlab",
        site_url="http://172.17.0.1:8033",
        replica_index=6,
        replica_name="gitlab_6",
        agent_auth={
            "type": "storage_state",
            "storage_state": {"path": "logs/phase_0d/gitlab/storage_state.json"},
        },
    )
    task, missing = prepare_task_for_execution(
        {
            "id": "adv-instance-id",
            "benchmark": "webarena_verified",
            "benign_task_id": "benign-1",
            "site": "gitlab",
            "sites": ["gitlab"],
            "instruction": "Open the dashboard",
            "start_urls": ["http://172.17.0.1:8033/dashboard"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {
                "benign_reward": {"type": "noop"},
                "adversarial_reward": {"type": "noop"},
            },
            "adversarial_data_seed": {"mechanism": "none"},
        },
        [instance],
    )
    assert missing == []
    task = bind_task_to_instance(task, instance, [instance])
    expected_instance_id = phase_0d_instance_id(instance.model_dump())

    captured: dict[str, object] = {}

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance_dict):
        return None, {}

    def fake_run_reward_function(reward, inst, agent_result=None, network_trace=None):
        if reward.get("type") == "noop":
            return reward is task["reward_function"]["benign_reward"], "ok"
        return False, "unexpected reward"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, **kwargs):
            captured.update(kwargs)
            return AgentResult(
                elapsed=0.1,
                steps=1,
                is_done=True,
                final_result="done",
                status="success",
                errors=[],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_adversarial, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_adversarial, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_adversarial, "run_reward_function", fake_run_reward_function)

    await phase_4_adversarial.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instance,
        task_dir=tmp_path,
    )

    assert captured.get("instance_id") == expected_instance_id
    assert captured.get("task_site") == "gitlab"
