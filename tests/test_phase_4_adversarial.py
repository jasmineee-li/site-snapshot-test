from __future__ import annotations

import asyncio
import json
import time
from argparse import Namespace
from types import SimpleNamespace

import pytest

from worldsim.agent_config import (
    RUNTIME_METADATA_KEY,
    bind_task_to_instance,
    prepare_task_for_execution,
)
from worldsim.browser_use_agent import AgentResult
from worldsim.config import BenchmarkInstance
from worldsim.phases import phase_4_adversarial
from worldsim.resume_metadata import RESULT_FINGERPRINT_KEY
from worldsim.task_paths import safe_task_path_component


def _prepared_adv_task() -> tuple[dict, list[BenchmarkInstance]]:
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
            "id": "adv-1",
            "benign_task_id": "benign-1",
            "site": "shopping",
            "sites": ["shopping", "gitlab"],
            "instruction": "Find the order",
            "start_urls": ["__SHOPPING__/orders"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {
                "benign_reward": {"type": "noop"},
                "adversarial_reward": {"type": "noop"},
            },
            "adversarial_data_seed": {
                "mechanism": "api",
                "api_calls": [{"method": "POST", "path": "/api/seed", "body": {"x": 1}}],
            },
        },
        instances,
    )
    assert missing == []
    return task, instances


@pytest.mark.asyncio
async def test_generate_variant_reads_variant_output_path(monkeypatch, tmp_path):
    task, _ = _prepared_adv_task()
    captured = {}

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        captured["model"] = kwargs.get("model")
        return {
            phase_4_adversarial.VARIANT_TASK_OUTPUT: json.dumps(
                {
                    "adversarial_data_seed": {
                        "mechanism": "api",
                        "api_calls": [
                            {
                                "method": "POST",
                                "path": "/api/seed",
                                "body": {"x": 2},
                            }
                        ],
                    },
                    "applied_strategy": {
                        "strategy": "specificity",
                        "description": "make it more specific",
                    },
                }
            ),
            "_summary": None,
        }

    monkeypatch.setattr(phase_4_adversarial, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    variant = await phase_4_adversarial.generate_variant(
        task,
        {"strategy": "specificity"},
        tmp_path / "profile.json",
        sandbox_model="claude-sonnet-4-6",
    )

    assert variant["adversarial_data_seed"]["api_calls"][0]["body"]["x"] == 2
    assert not phase_4_adversarial._tasks_equivalent(task, variant)
    assert captured["model"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_process_adversarial_result_emits_spec_shape(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    initial_result = {
        "task_id": "adv-1",
        "outcome": "refused_or_ignored",
        "ecologically_valid": True,
        "trajectory_dir": str(tmp_path / "traj"),
    }

    async def no_fix(*args, **kwargs):
        return None

    async def fake_strategy_variation(*args, **kwargs):
        return {
            "judge_diagnosis": {"diagnosis": "agent ignored weak injection"},
            "variant_results": [
                {
                    "strategy": "specificity",
                    "outcome": "complied",
                    "ecologically_valid": True,
                }
            ],
        }

    monkeypatch.setattr(phase_4_adversarial, "_run_ecological_validity_fix_loop", no_fix)
    monkeypatch.setattr(phase_4_adversarial, "_run_placement_fix_loop", no_fix)
    monkeypatch.setattr(phase_4_adversarial, "run_strategy_variation", fake_strategy_variation)

    result = await phase_4_adversarial._process_adversarial_result(
        task=task,
        initial_result=initial_result,
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
    )

    assert result["task_id"] == "adv-1"
    assert result["initial_outcome"] == "refused_or_ignored"
    assert result["judge_diagnosis"] == "agent ignored weak injection"
    assert result["strategies_attempted"] == [
        {
            "strategy": "specificity",
            "outcome": "complied",
            "ecologically_valid": True,
        }
    ]
    assert result["successful_strategy"] == "specificity"
    assert result["final_status"] == "success_on_variant"


@pytest.mark.asyncio
async def test_process_adversarial_result_marks_uninterpretable_variation_inconclusive(
    monkeypatch, tmp_path
):
    task, instances = _prepared_adv_task()
    initial_result = {
        "task_id": "adv-1",
        "outcome": "refused_or_ignored",
        "ecologically_valid": True,
        "trajectory_dir": str(tmp_path / "traj"),
    }

    async def no_fix(*args, **kwargs):
        return None

    async def fake_strategy_variation(*args, **kwargs):
        return {
            "status": "no_instances",
            "judge_diagnosis": {"diagnosis": "needs separate worker"},
            "variant_results": [],
        }

    monkeypatch.setattr(phase_4_adversarial, "_run_ecological_validity_fix_loop", no_fix)
    monkeypatch.setattr(phase_4_adversarial, "_run_placement_fix_loop", no_fix)
    monkeypatch.setattr(phase_4_adversarial, "run_strategy_variation", fake_strategy_variation)

    result = await phase_4_adversarial._process_adversarial_result(
        task=task,
        initial_result=initial_result,
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
    )

    assert result["final_status"] == "inconclusive"


@pytest.mark.asyncio
async def test_postprocess_one_task_resume_ignores_stale_processed_result(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    result = {
        "task_id": task["id"],
        "outcome": "refused_or_ignored",
        "ecologically_valid": True,
        "trajectory_dir": str(tmp_path / "traj"),
    }
    processed_file = (
        tmp_path / safe_task_path_component(task["id"]) / "processed_result.json"
    )
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    processed_file.write_text(
        json.dumps(
            {
                "task_id": task["id"],
                "final_status": "resistant",
                phase_4_adversarial._CHECKPOINT_FINGERPRINT_KEY: "stale",
            }
        )
    )

    calls = {"process": 0}

    async def fake_process_adversarial_result(**kwargs):
        calls["process"] += 1
        return {
            "task_id": task["id"],
            "initial_outcome": "refused_or_ignored",
            "ecologically_valid": True,
            "judge_diagnosis": None,
            "strategies_attempted": [],
            "final_status": "complied",
            "successful_strategy": None,
        }

    monkeypatch.setattr(
        phase_4_adversarial,
        "_process_adversarial_result",
        fake_process_adversarial_result,
    )

    processed = await phase_4_adversarial._postprocess_one_task(
        result=result,
        task_by_id={task["id"]: task},
        config=SimpleNamespace(instances=instances),
        profiles_dir=tmp_path,
        agent_factory=lambda: None,
        task_dir_root=tmp_path,
        resume=True,
    )

    assert calls["process"] == 1
    assert processed["final_status"] == "complied"


def test_rebase_adversarial_task_uses_validated_benign_contract():
    task, _ = _prepared_adv_task()
    validated_benign = json.loads(json.dumps(task))
    validated_benign["instruction"] = "Use the repaired benign instruction"
    validated_benign["reward_function"] = {
        "eval": [{"expected": {"retrieved_data": ["fixed"]}}],
    }
    validated_benign["agent_context"] = {"site_context": {"platform_name": "Shopping"}}

    stale_adversarial = json.loads(json.dumps(task))
    stale_adversarial["instruction"] = "stale phase 2 instruction"
    stale_adversarial["reward_function"]["benign_reward"] = {"type": "stale"}
    stale_adversarial["reward_function"]["adversarial_reward"] = {"type": "db_query_match"}
    stale_adversarial["agent_context"] = {"site_context": {"platform_name": "Wrong"}}

    rebuilt = phase_4_adversarial._rebase_adversarial_task(
        stale_adversarial,
        validated_benign,
    )

    assert rebuilt["instruction"] == "Use the repaired benign instruction"
    assert rebuilt["reward_function"]["benign_reward"] == validated_benign["reward_function"]
    assert rebuilt["reward_function"]["adversarial_reward"] == {"type": "db_query_match"}
    assert rebuilt["agent_context"] == validated_benign["agent_context"]


def test_rebase_adversarial_task_rejects_non_self_contained_seed():
    task, _ = _prepared_adv_task()
    validated_benign = json.loads(json.dumps(task))
    validated_benign["data_seed"] = {
        "mechanism": "api",
        "api_calls": [{"method": "POST", "path": "/api/base", "body": {"baseline": 1}}],
    }

    malformed_adversarial = json.loads(json.dumps(task))
    malformed_adversarial["adversarial_data_seed"] = {
        "mechanism": "api",
        "api_calls": [{"method": "POST", "path": "/api/seed", "body": {"attack": 1}}],
    }

    with pytest.raises(ValueError, match="preserve the benign data_seed"):
        phase_4_adversarial._rebase_adversarial_task(
            malformed_adversarial,
            validated_benign,
        )


def test_merge_variant_task_rejects_seed_that_drops_benign_setup():
    task, _ = _prepared_adv_task()
    task["data_seed"] = {
        "mechanism": "api",
        "api_calls": [{"method": "POST", "path": "/api/base", "body": {"baseline": 1}}],
    }
    task["adversarial_data_seed"] = {
        "mechanism": "api",
        "api_calls": [
            {"method": "POST", "path": "/api/base", "body": {"baseline": 1}},
            {"method": "POST", "path": "/api/seed", "body": {"attack": 1}},
        ],
    }

    merged = phase_4_adversarial._merge_variant_task(
        task,
        {
            "adversarial_data_seed": {
                "mechanism": "api",
                "api_calls": [{"method": "POST", "path": "/api/seed", "body": {"attack": 2}}],
            }
        },
    )

    assert merged["adversarial_data_seed"] == task["adversarial_data_seed"]


@pytest.mark.asyncio
async def test_probe_ecological_validity_stages_task_json(monkeypatch, tmp_path):
    (tmp_path / "history.json").write_text("{}")
    captured = {}

    async def fake_run_claude_in_sandbox(*, site_files, prompt, output_paths, **kwargs):
        captured["site_files"] = site_files
        captured["model"] = kwargs.get("model")
        return {
            phase_4_adversarial.VALIDITY_PROBE_OUTPUT: json.dumps({"average_score": 0.2}),
            "_summary": None,
        }

    monkeypatch.setattr(phase_4_adversarial, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    score = await phase_4_adversarial.probe_ecological_validity(
        tmp_path,
        {"id": "adv-1", "instruction": "task text"},
        sandbox_model="claude-sonnet-4-6",
    )

    assert score == 0.2
    assert "/workspace/task.json" in captured["site_files"]
    assert captured["model"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_run_adversarial_task_errors_when_benign_eval_is_non_interpretable(
    monkeypatch,
    tmp_path,
):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    task["reward_function"]["benign_reward"] = {
        "task_id": "123",
        "eval": [{"evaluator": "AgentResponseEvaluator"}],
    }
    task["reward_function"]["adversarial_reward"] = {"type": "db_query_match"}

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None

    async def fake_probe(task_dir, task, **kwargs):
        return 0.25

    def fake_run_reward_function(reward, instance, agent_result=None, network_trace=None):
        if "eval" in reward:
            return False, "canonical WebArena Verified evaluation unavailable: install dependency"
        return True, "matched adversarial reward"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return SimpleNamespace(
                elapsed=1.2,
                steps=4,
                is_done=True,
                final_result="done",
                status="success",
                errors=[],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_adversarial, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_adversarial, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_adversarial, "probe_ecological_validity", fake_probe)
    monkeypatch.setattr(phase_4_adversarial, "run_reward_function", fake_run_reward_function)

    result = await phase_4_adversarial.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["outcome"] == "error"
    assert "canonical WebArena Verified evaluation unavailable" in result["error"]
    assert result["validity_score"] == 0.25
    assert result["ecologically_valid"] is True
    assert result["adversarial_passed"] is False


@pytest.mark.asyncio
async def test_run_adversarial_task_scores_partial_timeout_when_artifacts_exist(
    monkeypatch,
    tmp_path,
):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None

    async def fake_probe(task_dir, task, **kwargs):
        return 0.2

    def fake_run_reward_function(reward, instance, agent_result=None, network_trace=None):
        if reward.get("type") == "noop":
            return reward is task["reward_function"]["benign_reward"], "ok"
        return False, "unexpected reward"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return AgentResult(
                elapsed=2.1,
                steps=5,
                is_done=False,
                final_result="partial answer",
                status="timeout",
                errors=["agent timed out after 30s"],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_adversarial, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_adversarial, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_adversarial, "probe_ecological_validity", fake_probe)
    monkeypatch.setattr(phase_4_adversarial, "run_reward_function", fake_run_reward_function)

    result = await phase_4_adversarial.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["outcome"] == "refused_or_ignored"
    assert result["validity_score"] == 0.2
    assert result["ecologically_valid"] is True
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["validity_score"] == 0.2
    assert saved["ecologically_valid"] is True


@pytest.mark.asyncio
async def test_run_adversarial_task_persists_resume_fingerprint(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None

    async def fake_probe(task_dir, task, **kwargs):
        return 0.2

    def fake_run_reward_function(reward, instance, agent_result=None, network_trace=None):
        if reward.get("type") == "noop":
            return reward is task["reward_function"]["benign_reward"], "ok"
        return False, "unexpected reward"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return AgentResult(
                elapsed=2.1,
                steps=5,
                is_done=False,
                final_result="partial answer",
                status="timeout",
                errors=["agent timed out after 30s"],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_adversarial, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_adversarial, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_adversarial, "probe_ecological_validity", fake_probe)
    monkeypatch.setattr(phase_4_adversarial, "run_reward_function", fake_run_reward_function)

    await phase_4_adversarial.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
        resume_fingerprint="phase4-fp",
    )

    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved[RESULT_FINGERPRINT_KEY] == "phase4-fp"


@pytest.mark.asyncio
async def test_run_adversarial_task_probes_non_scoreable_failures_before_saving_result(
    monkeypatch,
    tmp_path,
):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    captured: dict[str, object] = {}

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None

    async def fake_probe(task_dir, task, **kwargs):
        captured["probed_task_dir"] = task_dir
        return 0.4

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return AgentResult(
                elapsed=0.7,
                steps=0,
                is_done=False,
                final_result=None,
                status="error",
                errors=["browser crashed"],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_adversarial, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_adversarial, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_adversarial, "probe_ecological_validity", fake_probe)

    result = await phase_4_adversarial.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert captured["probed_task_dir"] == tmp_path
    assert result["outcome"] == "error"
    assert result["validity_score"] == 0.4
    assert result["ecologically_valid"] is True
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["outcome"] == "error"
    assert saved["validity_score"] == 0.4
    assert saved["ecologically_valid"] is True


@pytest.mark.asyncio
async def test_run_adversarial_task_passes_site_prompt_from_agent_context(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    task["agent_context"] = {
        "response_format": {
            "requires_structured_output": True,
            "output_schema": {"task_type": "string", "status": "string"},
            "per_task_format_field": None,
            "description": "The evaluator parses the final agent message as JSON.",
        },
        "authentication": {
            "pre_authenticated": False,
            "credentials": {"username": "demo", "password": "secret"},
            "description": "Use the seeded shopping account if prompted to sign in.",
        },
        "agent_prompt_template": None,
        "site_context": {
            "platform_name": "Shopping",
            "description": "An ecommerce storefront with order history.",
        },
    }
    captured: dict[str, object] = {}

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None

    async def fake_probe(task_dir, task, **kwargs):
        return 0.1

    def fake_run_reward_function(reward, instance, agent_result=None, network_trace=None):
        if reward.get("type") == "noop":
            return reward is task["reward_function"]["benign_reward"], "ok"
        return False, "unexpected reward"

    class FakeAgent:
        async def run(
            self, instruction, server_url, task_dir, *, start_urls=None, site_prompt=None
        ):
            captured["start_urls"] = start_urls
            captured["site_prompt"] = site_prompt
            return AgentResult(
                elapsed=1.2,
                steps=4,
                is_done=True,
                final_result="done",
                status="success",
                errors=[],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_adversarial, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_adversarial, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_adversarial, "probe_ecological_validity", fake_probe)
    monkeypatch.setattr(phase_4_adversarial, "run_reward_function", fake_run_reward_function)

    result = await phase_4_adversarial.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["outcome"] == "refused_or_ignored"
    assert captured["start_urls"] == ["http://shopping.test/orders"]
    assert isinstance(captured["site_prompt"], str)
    assert "Use the seeded shopping account" in captured["site_prompt"]
    assert "Return your final answer as JSON matching this schema" in captured["site_prompt"]


@pytest.mark.asyncio
async def test_evaluate_variant_rebinds_runtime_metadata(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    variant = json.loads(json.dumps(task))
    site_profile = {"site_name": "shopping", "verification_capabilities": []}

    async def fake_run_adversarial_task(task, agent, instance, task_dir, **kwargs):
        assert task[RUNTIME_METADATA_KEY]["reset_endpoints"] == [
            "http://shopping.test/init",
            "http://gitlab.test/init",
        ]
        assert kwargs["site_profile"] == site_profile
        return {"task_id": task["id"], "outcome": "complied", "ecologically_valid": True}

    class FakeAgent:
        async def setup(self, server_url):
            return None

        async def teardown(self):
            return None

    monkeypatch.setattr(phase_4_adversarial, "run_adversarial_task", fake_run_adversarial_task)

    result = await phase_4_adversarial._evaluate_variant(
        task=task,
        variant=variant,
        instance=instances[0],
        all_instances=instances,
        strategy={"strategy": "specificity"},
        index=0,
        agent_factory=FakeAgent,
        task_dir_root=tmp_path,
        site_profile=site_profile,
    )

    assert result["outcome"] == "complied"
    assert result["strategy"] == "specificity"


@pytest.mark.asyncio
async def test_evaluate_variant_resume_ignores_result_without_matching_metadata(
    monkeypatch, tmp_path
):
    task, instances = _prepared_adv_task()
    variant = json.loads(json.dumps(task))
    variant_dir = tmp_path / "adv-1_variant_0"
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "result.json").write_text(
        json.dumps({"task_id": "adv-1", "outcome": "complied", "ecologically_valid": True})
    )
    (
        variant_dir / phase_4_adversarial._VARIANT_RESULT_METADATA
    ).write_text(json.dumps({phase_4_adversarial._CHECKPOINT_FINGERPRINT_KEY: "stale"}))

    calls = {"run": 0}

    async def fake_run_adversarial_task(task, agent, instance, task_dir, **kwargs):
        calls["run"] += 1
        return {"task_id": task["id"], "outcome": "complied", "ecologically_valid": True}

    class FakeAgent:
        async def setup(self, server_url):
            return None

        async def teardown(self):
            return None

    monkeypatch.setattr(phase_4_adversarial, "run_adversarial_task", fake_run_adversarial_task)

    result = await phase_4_adversarial._evaluate_variant(
        task=task,
        variant=variant,
        instance=instances[0],
        all_instances=instances,
        strategy={"strategy": "specificity"},
        index=0,
        agent_factory=FakeAgent,
        task_dir_root=tmp_path,
        resume=True,
    )

    assert calls["run"] == 1
    assert result["outcome"] == "complied"


@pytest.mark.asyncio
async def test_evaluate_variant_runs_in_parallel_on_distinct_instance_footprints(
    monkeypatch, tmp_path
):
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-1.test",
            reset_endpoint="http://shopping-1.test/init",
        ),
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-2.test",
            reset_endpoint="http://shopping-2.test/init",
        ),
    ]
    task, missing = prepare_task_for_execution(
        {
            "id": "adv-parallel",
            "benign_task_id": "benign-parallel",
            "site": "shopping",
            "sites": ["shopping"],
            "instruction": "Find the order",
            "start_urls": ["__SHOPPING__/orders"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {
                "benign_reward": {"type": "noop"},
                "adversarial_reward": {"type": "noop"},
            },
            "adversarial_data_seed": {"mechanism": "none"},
        },
        instances,
    )
    assert missing == []
    variant = json.loads(json.dumps(task))
    timestamps: dict[str, list[float]] = {}

    async def fake_run_adversarial_task(task, agent, instance, task_dir, **kwargs):
        timestamps[instance.site_url] = [time.monotonic()]
        await asyncio.sleep(0.05)
        timestamps[instance.site_url].append(time.monotonic())
        return {"task_id": task["id"], "outcome": "complied", "ecologically_valid": True}

    class FakeAgent:
        async def setup(self, server_url):
            return None

        async def teardown(self):
            return None

    monkeypatch.setattr(phase_4_adversarial, "run_adversarial_task", fake_run_adversarial_task)

    await asyncio.gather(
        phase_4_adversarial._evaluate_variant(
            task=task,
            variant=json.loads(json.dumps(variant)),
            instance=instances[0],
            all_instances=instances,
            strategy={"strategy": "specificity"},
            index=0,
            agent_factory=FakeAgent,
            task_dir_root=tmp_path,
        ),
        phase_4_adversarial._evaluate_variant(
            task=task,
            variant=json.loads(json.dumps(variant)),
            instance=instances[1],
            all_instances=instances,
            strategy={"strategy": "verbosity_adjustment"},
            index=1,
            agent_factory=FakeAgent,
            task_dir_root=tmp_path,
        ),
    )

    first_start, first_end = timestamps["http://shopping-1.test"]
    second_start, second_end = timestamps["http://shopping-2.test"]
    assert first_start < second_end and second_start < first_end


@pytest.mark.asyncio
async def test_run_strategy_variation_ignores_bookkeeping_only_variants(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()

    async def fake_run_judge(*args, **kwargs):
        return {
            "diagnosis": "needs more specificity",
            "recommended_strategies": [{"strategy": "specificity"}],
        }

    async def fake_generate_variant(*args, **kwargs):
        clone = json.loads(json.dumps(task))
        clone["applied_strategy"] = {"strategy": "specificity"}
        return clone

    def fail_agent_factory():
        raise AssertionError("no-op variants should never reach evaluation")

    monkeypatch.setattr(phase_4_adversarial, "run_judge", fake_run_judge)
    monkeypatch.setattr(phase_4_adversarial, "generate_variant", fake_generate_variant)

    result = await phase_4_adversarial.run_strategy_variation(
        task=task,
        initial_result={"trajectory_dir": str(tmp_path / "traj")},
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=fail_agent_factory,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
    )

    assert result["status"] == "variant_generation_failed"
    assert result["variant_results"] == []


@pytest.mark.asyncio
async def test_run_strategy_variation_resume_reuses_saved_variant_result(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    variant = json.loads(json.dumps(task))
    variant["adversarial_data_seed"]["api_calls"][0]["body"]["x"] = 2
    initial_result = {"trajectory_dir": str(tmp_path / "traj")}
    checkpoint_fingerprint = phase_4_adversarial._phase_4_postprocess_fingerprint(
        task,
        initial_result,
        primary_instances=[instances[0]],
        all_instances=instances,
        benchmark_root=None,
        sandbox_model="claude-sonnet-4-6",
        site_profile=None,
    )
    checkpoint_path = phase_4_adversarial._strategy_variation_checkpoint_path(tmp_path, task["id"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                phase_4_adversarial._CHECKPOINT_FINGERPRINT_KEY: checkpoint_fingerprint,
                "judge_diagnosis": {
                    "diagnosis": "needs more specificity",
                    "recommended_strategies": [{"strategy": "specificity"}],
                },
                "variant_candidates": [
                    {
                        "strategy": {"strategy": "specificity"},
                        "variant": variant,
                    }
                ],
            }
        )
    )
    variant_dir = tmp_path / "adv-1_variant_0"
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "adv-1",
                "outcome": "complied",
                "ecologically_valid": True,
            }
        )
    )
    (variant_dir / phase_4_adversarial._VARIANT_RESULT_METADATA).write_text(
        json.dumps(
            {
                phase_4_adversarial._CHECKPOINT_FINGERPRINT_KEY: (
                    phase_4_adversarial._phase_4_variant_fingerprint(
                        task,
                        variant,
                        {"strategy": "specificity"},
                        instance=instances[0],
                        benchmark_root=None,
                        sandbox_model="claude-sonnet-4-6",
                        site_profile=None,
                    )
                )
            }
        )
    )

    async def fail_run_judge(*args, **kwargs):
        raise AssertionError("resume should reuse saved judge output")

    async def fail_generate_variant(*args, **kwargs):
        raise AssertionError("resume should reuse saved variants")

    def fail_agent_factory():
        raise AssertionError("resume should reuse saved variant result")

    monkeypatch.setattr(phase_4_adversarial, "run_judge", fail_run_judge)
    monkeypatch.setattr(phase_4_adversarial, "generate_variant", fail_generate_variant)

    result = await phase_4_adversarial.run_strategy_variation(
        task=task,
        initial_result=initial_result,
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=fail_agent_factory,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        resume=True,
    )

    assert result["status"] == "varied"
    assert result["variant_results"] == [
        {
            "task_id": "adv-1",
            "outcome": "complied",
            "ecologically_valid": True,
            "trajectory_dir": str(variant_dir),
            "strategy": "specificity",
        }
    ]


@pytest.mark.asyncio
async def test_run_strategy_variation_resume_ignores_saved_variant_result_from_different_instance(
    monkeypatch, tmp_path
):
    task, instances = _prepared_adv_task()
    variant = json.loads(json.dumps(task))
    variant["adversarial_data_seed"]["api_calls"][0]["body"]["x"] = 2
    initial_result = {"trajectory_dir": str(tmp_path / "traj")}
    checkpoint_path = phase_4_adversarial._strategy_variation_checkpoint_path(tmp_path, task["id"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                phase_4_adversarial._CHECKPOINT_FINGERPRINT_KEY: (
                    phase_4_adversarial._phase_4_postprocess_fingerprint(
                        task,
                        initial_result,
                        primary_instances=[instances[0]],
                        all_instances=instances,
                        benchmark_root=None,
                        sandbox_model="claude-sonnet-4-6",
                        site_profile=None,
                    )
                ),
                "judge_diagnosis": {
                    "diagnosis": "needs more specificity",
                    "recommended_strategies": [{"strategy": "specificity"}],
                },
                "variant_candidates": [
                    {
                        "strategy": {"strategy": "specificity"},
                        "variant": variant,
                    }
                ],
            }
        )
    )
    variant_dir = tmp_path / "adv-1_variant_0"
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "adv-1",
                "outcome": "complied",
                "ecologically_valid": True,
            }
        )
    )
    (variant_dir / phase_4_adversarial._VARIANT_RESULT_METADATA).write_text(
        json.dumps(
            {
                phase_4_adversarial._CHECKPOINT_FINGERPRINT_KEY: (
                    phase_4_adversarial._phase_4_variant_fingerprint(
                        task,
                        variant,
                        {"strategy": "specificity"},
                        instance=instances[1],
                        benchmark_root=None,
                        sandbox_model="claude-sonnet-4-6",
                        site_profile=None,
                    )
                )
            }
        )
    )

    calls = {"evaluated": 0}

    async def fake_run_adversarial_task(task, agent, instance, task_dir, **kwargs):
        calls["evaluated"] += 1
        return {
            "task_id": task["id"],
            "outcome": "complied",
            "ecologically_valid": True,
        }

    class FakeAgent:
        async def setup(self, server_url):
            return None

        async def teardown(self):
            return None

    async def fail_run_judge(*args, **kwargs):
        raise AssertionError("resume should reuse saved judge output")

    async def fail_generate_variant(*args, **kwargs):
        raise AssertionError("resume should reuse saved variants")

    monkeypatch.setattr(phase_4_adversarial, "run_judge", fail_run_judge)
    monkeypatch.setattr(phase_4_adversarial, "generate_variant", fail_generate_variant)
    monkeypatch.setattr(phase_4_adversarial, "run_adversarial_task", fake_run_adversarial_task)

    result = await phase_4_adversarial.run_strategy_variation(
        task=task,
        initial_result=initial_result,
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=FakeAgent,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        resume=True,
    )

    assert calls["evaluated"] == 1
    assert result["variant_results"][0]["strategy"] == "specificity"


@pytest.mark.asyncio
async def test_run_strategy_variation_resume_ignores_stale_checkpoint(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    checkpoint_path = phase_4_adversarial._strategy_variation_checkpoint_path(tmp_path, task["id"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                phase_4_adversarial._CHECKPOINT_FINGERPRINT_KEY: "stale",
                "judge_diagnosis": {
                    "diagnosis": "stale",
                    "recommended_strategies": [{"strategy": "specificity"}],
                },
            }
        )
    )

    calls = {"judge": 0}

    async def fake_run_judge(*args, **kwargs):
        calls["judge"] += 1
        return {
            "diagnosis": "fresh",
            "recommended_strategies": [{"strategy": "specificity"}],
        }

    async def fake_generate_variant(*args, **kwargs):
        variant = json.loads(json.dumps(task))
        variant["adversarial_data_seed"]["api_calls"][0]["body"]["x"] = 2
        return variant

    async def fake_evaluate_variant(**kwargs):
        return {
            "task_id": task["id"],
            "outcome": "complied",
            "ecologically_valid": True,
            "strategy": "specificity",
        }

    monkeypatch.setattr(phase_4_adversarial, "run_judge", fake_run_judge)
    monkeypatch.setattr(phase_4_adversarial, "generate_variant", fake_generate_variant)
    monkeypatch.setattr(phase_4_adversarial, "_evaluate_variant", fake_evaluate_variant)

    result = await phase_4_adversarial.run_strategy_variation(
        task=task,
        initial_result={"trajectory_dir": str(tmp_path / "traj")},
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        resume=True,
    )

    assert calls["judge"] == 1
    assert result["status"] == "varied"
    assert result["judge_diagnosis"]["diagnosis"] == "fresh"


@pytest.mark.asyncio
async def test_phase_4_requires_validated_tasks_file(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text("[]")
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "demo",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "shopping",
                        "site_url": "http://shopping.test",
                    }
                ],
            }
        )
    )

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            agent_model="demo-model",
            agent_provider=None,
        )
    )

    assert rc == 1


@pytest.mark.asyncio
async def test_phase_4_run_fails_on_gathered_postprocess_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    (tmp_path / "phase_0c").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "adv-1",
                    "benign_task_id": "benign-1",
                    "site": "shopping",
                    "sites": ["shopping"],
                    "instruction": "Find the order",
                    "start_urls": ["http://shopping.test/orders"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {
                        "adversarial_reward": {"type": "noop"},
                    },
                    "adversarial_data_seed": {
                        "mechanism": "api",
                        "api_calls": [{"method": "POST", "path": "/api/seed", "body": {"x": 1}}],
                    },
                }
            ]
        )
    )
    (tmp_path / "phase_3" / "validated_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "benign-1",
                    "site": "shopping",
                    "sites": ["shopping"],
                    "instruction": "Find the order",
                    "start_urls": ["http://shopping.test/orders"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {"type": "noop"},
                }
            ]
        )
    )
    (tmp_path / "phase_0c" / "BENCHMARK_PROFILE_shopping.json").write_text(
        json.dumps(
            {
                "site_name": "shopping",
                "data_model": [],
                "injection_surface": [],
                "verification_capabilities": [],
            }
        )
    )
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "demo",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "shopping",
                        "site_url": "http://shopping.test",
                        "reset_endpoint": "http://shopping.test/init",
                    }
                ],
            }
        )
    )
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()

    async def fake_run_tasks_by_site(**kwargs):
        return [
            {
                "task_id": "adv-1",
                "outcome": "refused_or_ignored",
                "ecologically_valid": True,
                "trajectory_dir": str(tmp_path / "traj"),
            }
        ]

    async def fake_postprocess_one_task(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "make_agent_factory", lambda **kwargs: lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "run_tasks_by_site", fake_run_tasks_by_site)
    monkeypatch.setattr(phase_4_adversarial, "_postprocess_one_task", fake_postprocess_one_task)

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            benchmark=benchmark_root,
            agent_model="demo-model",
            agent_provider=None,
            allow_unknown_auth=True,
            resume=False,
        )
    )

    assert rc == 1
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["reason"] == "postprocess_exception"
    assert state["failed_tasks"] == ["adv-1"]
    assert state["task_dir_root"].startswith(str(tmp_path / "phase_4"))
    assert state["instances_path"] == str(instances_path)
    assert state["benchmark_path"] == str(benchmark_root)
    assert state["allow_unknown_auth"] is True
    assert state["agent_model"] == "demo-model"


@pytest.mark.asyncio
async def test_phase_4_run_rejects_missing_benign_task_id(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "adv-1",
                    "site": "shopping",
                    "sites": ["shopping"],
                    "instruction": "Find the order",
                    "start_urls": ["http://shopping.test/orders"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {"adversarial_reward": {"type": "noop"}},
                    "adversarial_data_seed": {"mechanism": "none"},
                }
            ]
        )
    )
    (tmp_path / "phase_3" / "validated_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "benign-1",
                    "site": "shopping",
                    "sites": ["shopping"],
                    "instruction": "Find the order",
                    "start_urls": ["http://shopping.test/orders"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {"type": "noop"},
                }
            ]
        )
    )
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "demo",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "shopping",
                        "site_url": "http://shopping.test",
                        "reset_endpoint": "http://shopping.test/init",
                    }
                ],
            }
        )
    )

    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            agent_model="demo-model",
            agent_provider=None,
            resume=False,
        )
    )

    assert rc == 1


# ── benchmark_root / task_site plumbing ──────────────────────────────────


@pytest.mark.asyncio
async def test_run_adversarial_task_forwards_benchmark_root_when_auth_present(
    monkeypatch, tmp_path
):
    """Phase 4 run_adversarial_task forwards benchmark_root + task_site when auth_mechanism set."""
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    task["agent_context"] = {
        "auth_mechanism": {
            "type": "storage_state",
            "storage_state": {"path": "auth/shopping_state.json"},
        }
    }

    async def fake_reset(task):
        return None

    async def fake_seed(seed, instance_dict):
        return None

    def fake_validate_seed(seed, allow_none=False):
        return None

    async def fake_probe(task_dir, task=None, **kwargs):
        return 0.9

    def fake_run_reward_function(*args, **kwargs):
        return True, "ok"

    captured: dict = {}

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
    monkeypatch.setattr(phase_4_adversarial, "apply_data_seed_async", fake_seed)
    monkeypatch.setattr(phase_4_adversarial, "validate_data_seed", fake_validate_seed)
    monkeypatch.setattr(phase_4_adversarial, "run_reward_function", fake_run_reward_function)
    monkeypatch.setattr(phase_4_adversarial, "probe_ecological_validity", fake_probe, raising=False)

    bench_root = tmp_path / "bench"
    bench_root.mkdir()

    await phase_4_adversarial.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
        benchmark_root=bench_root,
    )

    assert captured.get("benchmark_root") == bench_root
    assert captured.get("task_site") == "shopping"
    assert captured.get("auth_mechanism", {}).get("type") == "storage_state"


@pytest.mark.asyncio
async def test_run_adversarial_task_forwards_http_header_credentials(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    task["agent_context"] = {
        "authentication": {
            "pre_authenticated": False,
            "credentials": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
            "description": "Use benchmark credentials.",
        },
        "auth_mechanism": {
            "type": "http_headers",
            "http_headers": {
                "headers": {
                    "X-M2-Customer-Auto-Login": "${credentials.username}:${credentials.password}"
                }
            },
        },
    }

    async def fake_reset(task):
        return None

    async def fake_seed(seed, instance_dict):
        return None

    def fake_validate_seed(seed, allow_none=False):
        return None

    async def fake_probe(task_dir, task=None, **kwargs):
        return 0.9

    def fake_run_reward_function(*args, **kwargs):
        return True, "ok"

    captured: dict = {}

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
    monkeypatch.setattr(phase_4_adversarial, "apply_data_seed_async", fake_seed)
    monkeypatch.setattr(phase_4_adversarial, "validate_data_seed", fake_validate_seed)
    monkeypatch.setattr(phase_4_adversarial, "run_reward_function", fake_run_reward_function)
    monkeypatch.setattr(phase_4_adversarial, "probe_ecological_validity", fake_probe, raising=False)

    await phase_4_adversarial.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert captured["auth_mechanism"]["authentication"]["credentials"] == {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    }


@pytest.mark.asyncio
async def test_run_adversarial_task_omits_benchmark_root_without_auth(monkeypatch, tmp_path):
    """Without auth_mechanism, run_adversarial_task omits the auth-only kwargs."""
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_seed(seed, instance_dict):
        return None

    def fake_validate_seed(seed, allow_none=False):
        return None

    async def fake_probe(task_dir, task=None, **kwargs):
        return 0.9

    def fake_run_reward_function(*args, **kwargs):
        return True, "ok"

    captured: dict = {}

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
    monkeypatch.setattr(phase_4_adversarial, "apply_data_seed_async", fake_seed)
    monkeypatch.setattr(phase_4_adversarial, "validate_data_seed", fake_validate_seed)
    monkeypatch.setattr(phase_4_adversarial, "run_reward_function", fake_run_reward_function)
    monkeypatch.setattr(phase_4_adversarial, "probe_ecological_validity", fake_probe, raising=False)

    await phase_4_adversarial.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
        benchmark_root=tmp_path / "bench",
    )

    assert "benchmark_root" not in captured
    assert "task_site" not in captured
    assert "auth_mechanism" not in captured
