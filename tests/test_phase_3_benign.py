from __future__ import annotations

from pathlib import Path

import pytest

from worldsim.agent_config import RUNTIME_METADATA_KEY, prepare_task_for_execution
from worldsim.config import BenchmarkInstance
from worldsim.phases import phase_3_benign


def _prepared_task() -> tuple[dict, list[BenchmarkInstance]]:
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping.test",
            reset_endpoint="http://shopping.test/init",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab.test",
            reset_endpoint="http://gitlab.test/init",
        ),
    ]
    task, missing = prepare_task_for_execution(
        {
            "id": "task-1",
            "site": "shopping",
            "sites": ["shopping", "gitlab"],
            "instruction": "Find the order details",
            "start_urls": ["__SHOPPING__/orders"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {
                "eval": [
                    {
                        "expected": {
                            "retrieved_data": ["old value"],
                        }
                    }
                ]
            },
        },
        instances,
    )
    assert missing == []
    return task, instances


@pytest.mark.asyncio
async def test_fix_loop_applies_reward_patch_and_preserves_runtime(monkeypatch, tmp_path):
    task, instances = _prepared_task()

    async def fake_diagnose_failure(*args, **kwargs):
        return {
            "root_cause": "reward_bug",
            "explanation": "reward mismatch",
            "suggested_fix": {
                "target": "reward_function",
                "patch": {
                    "eval": [
                        {
                            "expected": {
                                "retrieved_data": ["new value"],
                            }
                        }
                    ]
                },
            },
        }

    async def fake_rerun_live_task(task, instance, instances, agent_factory, task_dir):
        assert task["reward_function"]["eval"][0]["expected"]["retrieved_data"] == ["new value"]
        assert task[RUNTIME_METADATA_KEY]["sites"] == ["shopping", "gitlab"]
        return {"passed": True, "trajectory_dir": str(task_dir)}

    monkeypatch.setattr(phase_3_benign, "diagnose_failure", fake_diagnose_failure)
    monkeypatch.setattr(phase_3_benign, "_rerun_live_task", fake_rerun_live_task)

    result = await phase_3_benign.fix_loop(
        task=task,
        trajectory_dir=tmp_path / "initial",
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        instances=instances,
        agent_factory=lambda: None,
        max_iterations=1,
    )

    assert result["action"] == "fixed"
    assert result["fixed_task"]["reward_function"]["eval"][0]["expected"]["retrieved_data"] == [
        "new value"
    ]
    assert result["fixed_task"][RUNTIME_METADATA_KEY]["sites"] == ["shopping", "gitlab"]


@pytest.mark.asyncio
async def test_fix_loop_applies_seed_patch_and_preserves_runtime(monkeypatch, tmp_path):
    task, instances = _prepared_task()
    task["data_seed"] = {
        "mechanism": "api",
        "api_calls": [{"method": "POST", "path": "/api/orders", "body": {"id": 1}}],
    }

    async def fake_diagnose_failure(*args, **kwargs):
        return {
            "root_cause": "seed_bug",
            "explanation": "seed mismatch",
            "suggested_fix": {
                "target": "data_seed",
                "patch": {
                    "api_calls": [
                        {"method": "POST", "path": "/api/orders", "body": {"id": 2}}
                    ]
                },
            },
        }

    async def fake_rerun_live_task(task, instance, instances, agent_factory, task_dir):
        assert task["data_seed"]["api_calls"][0]["body"]["id"] == 2
        assert task[RUNTIME_METADATA_KEY]["sites"] == ["shopping", "gitlab"]
        return {"passed": True, "trajectory_dir": str(task_dir)}

    monkeypatch.setattr(phase_3_benign, "diagnose_failure", fake_diagnose_failure)
    monkeypatch.setattr(phase_3_benign, "_rerun_live_task", fake_rerun_live_task)

    result = await phase_3_benign.fix_loop(
        task=task,
        trajectory_dir=tmp_path / "initial",
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        instances=instances,
        agent_factory=lambda: None,
        max_iterations=1,
    )

    assert result["action"] == "fixed"
    assert result["fixed_task"]["data_seed"]["api_calls"][0]["body"]["id"] == 2
    assert result["fixed_task"][RUNTIME_METADATA_KEY]["sites"] == ["shopping", "gitlab"]


@pytest.mark.asyncio
async def test_rerun_live_task_rebinds_runtime_metadata(monkeypatch, tmp_path):
    task, instances = _prepared_task()

    async def fake_run_task(task, agent, instance, task_dir):
        assert task[RUNTIME_METADATA_KEY]["reset_endpoints"] == [
            "http://shopping.test/init",
            "http://gitlab.test/init",
        ]
        return {"passed": True, "trajectory_dir": str(task_dir)}

    class FakeAgent:
        async def setup(self, server_url):
            return None

        async def teardown(self):
            return None

    monkeypatch.setattr(phase_3_benign, "run_task", fake_run_task)

    result = await phase_3_benign._rerun_live_task(
        task=task,
        instance=instances[0],
        instances=instances,
        agent_factory=FakeAgent,
        task_dir=tmp_path / "rerun",
    )

    assert result["passed"] is True


@pytest.mark.asyncio
async def test_diagnose_failure_uses_unknown_sanity_when_unavailable(monkeypatch, tmp_path):
    task, _ = _prepared_task()
    captured = {}

    async def fake_run_claude_in_sandbox(*, site_files, prompt, output_paths, **kwargs):
        captured["prompt"] = prompt
        return {
            "/workspace/output/diagnosis.json": '{"root_cause":"agent_limitation","suggested_fix":{"target":"none","patch":null}}',
            "_summary": None,
        }

    monkeypatch.setattr(phase_3_benign, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    await phase_3_benign.diagnose_failure(
        task=task,
        trajectory_dir=tmp_path,
        profile_path=tmp_path / "profile.json",
    )

    assert "`unknown`" in captured["prompt"]


def test_render_diagnosis_prompt_uses_explicit_sanity_result():
    prompt = phase_3_benign._render_diagnosis_prompt({"sanity_check": {"result": "pass"}})

    assert "`pass`" in prompt
