# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from typing import Any

from ._fixtures import *  # noqa: F403,F401


def test_phase_4_variant_fingerprint_changes_when_instance_auth_or_placeholders_change():
    task, instances = _prepared_adv_task()
    variant = json.loads(json.dumps(task))
    strategy = {"strategy": "specificity"}
    with_auth = BenchmarkInstance(
        site_name="shopping",
        site_url="http://shopping.test",
        reset_endpoint="http://shopping.test/init",
        url_placeholders={"__SHOPPING__": "http://shopping.test"},
        auth={"type": "bearer_token", "token": "one", "validation_endpoint": "/health"},
    )
    changed_auth = BenchmarkInstance(
        site_name="shopping",
        site_url="http://shopping.test",
        reset_endpoint="http://shopping.test/init",
        url_placeholders={"__SHOPPING__": "http://shopping-alt.test"},
        auth={"type": "bearer_token", "token": "two", "validation_endpoint": "/health"},
    )

    assert phase_4_adversarial._phase_4_variant_fingerprint(
        task,
        variant,
        strategy,
        instance=with_auth,
        all_instances=instances,
        config_url_placeholders={"__GITLAB__": "http://gitlab.test"},
        benchmark_root=None,
        sandbox_model="claude-sonnet-4-6",
        site_profile=None,
    ) != phase_4_adversarial._phase_4_variant_fingerprint(
        task,
        variant,
        strategy,
        instance=changed_auth,
        all_instances=instances,
        config_url_placeholders={"__GITLAB__": "http://gitlab.test"},
        benchmark_root=None,
        sandbox_model="claude-sonnet-4-6",
        site_profile=None,
    )


def test_phase_4_variant_fingerprint_changes_when_secondary_instances_change():
    task, instances = _prepared_adv_task()
    variant = json.loads(json.dumps(task))
    strategy = {"strategy": "specificity"}
    changed_secondary = [
        instances[0],
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-alt.test",
            reset_endpoint="http://gitlab-alt.test/init",
        ),
    ]

    assert phase_4_adversarial._phase_4_variant_fingerprint(
        task,
        variant,
        strategy,
        instance=instances[0],
        all_instances=instances,
        config_url_placeholders=None,
        benchmark_root=None,
        sandbox_model="claude-sonnet-4-6",
        site_profile=None,
    ) != phase_4_adversarial._phase_4_variant_fingerprint(
        task,
        variant,
        strategy,
        instance=instances[0],
        all_instances=changed_secondary,
        config_url_placeholders=None,
        benchmark_root=None,
        sandbox_model="claude-sonnet-4-6",
        site_profile=None,
    )


def test_phase_4_result_fingerprint_ignores_unrelated_site_instances_and_placeholders():
    task, instances = _prepared_adv_task()
    unrelated = BenchmarkInstance(
        site_name="reddit",
        site_url="http://reddit.test",
        reset_endpoint="http://reddit.test/init",
        url_placeholders={"__REDDIT__": "http://reddit.test"},
        api_auth={"type": "bearer_token", "token": "one", "validation_endpoint": "/health"},
    )
    changed_unrelated = BenchmarkInstance(
        site_name="reddit",
        site_url="http://reddit-alt.test",
        reset_endpoint="http://reddit-alt.test/init",
        url_placeholders={"__REDDIT__": "http://reddit-alt.test"},
        api_auth={"type": "bearer_token", "token": "two", "validation_endpoint": "/health"},
    )

    base = phase_4_adversarial._phase_4_result_fingerprint(
        task,
        eval_context=phase_4_adversarial._phase_4_eval_context_for_task(
            task,
            instances=[*instances, unrelated],
            config_url_placeholders={
                "__SHOPPING__": "http://shopping.test",
                "__GITLAB__": "http://gitlab.test",
                "__REDDIT__": "http://reddit.test",
            },
            agent_model="claude-sonnet-4-6",
            agent_provider="anthropic",
            sandbox_model="claude-sonnet-4-6",
            benchmark_root=None,
        ),
        site_profile=None,
    )
    changed = phase_4_adversarial._phase_4_result_fingerprint(
        task,
        eval_context=phase_4_adversarial._phase_4_eval_context_for_task(
            task,
            instances=[*instances, changed_unrelated],
            config_url_placeholders={
                "__SHOPPING__": "http://shopping.test",
                "__GITLAB__": "http://gitlab.test",
                "__REDDIT__": "http://reddit-alt.test",
            },
            agent_model="claude-sonnet-4-6",
            agent_provider="anthropic",
            sandbox_model="claude-sonnet-4-6",
            benchmark_root=None,
        ),
        site_profile=None,
    )

    assert base == changed


def test_phase_4_result_fingerprint_changes_when_resume_version_changes(monkeypatch):
    task, instances = _prepared_adv_task()
    base = phase_4_adversarial._phase_4_result_fingerprint(
        task,
        eval_context=phase_4_adversarial._phase_4_eval_context_for_task(
            task,
            instances=instances,
            config_url_placeholders=None,
            agent_model="claude-sonnet-4-6",
            agent_provider="anthropic",
            sandbox_model="claude-sonnet-4-6",
            benchmark_root=None,
        ),
        site_profile=None,
    )

    monkeypatch.setattr(phase_4_adversarial, "_PHASE_4_RESUME_VERSION", "test-version")

    changed = phase_4_adversarial._phase_4_result_fingerprint(
        task,
        eval_context=phase_4_adversarial._phase_4_eval_context_for_task(
            task,
            instances=instances,
            config_url_placeholders=None,
            agent_model="claude-sonnet-4-6",
            agent_provider="anthropic",
            sandbox_model="claude-sonnet-4-6",
            benchmark_root=None,
        ),
        site_profile=None,
    )

    assert base != changed


def test_phase_4_result_fingerprint_changes_when_agent_timeout_changes():
    task, instances = _prepared_adv_task()
    base = phase_4_adversarial._phase_4_result_fingerprint(
        task,
        eval_context=phase_4_adversarial._phase_4_eval_context_for_task(
            task,
            instances=instances,
            config_url_placeholders=None,
            agent_model="claude-sonnet-4-6",
            agent_provider="anthropic",
            sandbox_model="claude-sonnet-4-6",
            benchmark_root=None,
            agent_llm_timeout=30,
            agent_step_timeout=120,
            agent_task_timeout=900,
        ),
        site_profile=None,
    )
    changed = phase_4_adversarial._phase_4_result_fingerprint(
        task,
        eval_context=phase_4_adversarial._phase_4_eval_context_for_task(
            task,
            instances=instances,
            config_url_placeholders=None,
            agent_model="claude-sonnet-4-6",
            agent_provider="anthropic",
            sandbox_model="claude-sonnet-4-6",
            benchmark_root=None,
            agent_llm_timeout=45,
            agent_step_timeout=120,
            agent_task_timeout=900,
        ),
        site_profile=None,
    )

    assert base != changed


def test_phase_4_variant_fingerprint_changes_when_agent_or_api_auth_changes():
    task, instances = _prepared_adv_task()
    variant = json.loads(json.dumps(task))
    strategy = {"strategy": "specificity"}
    with_runtime_auth = BenchmarkInstance(
        site_name="shopping",
        site_url="http://shopping.test",
        reset_endpoint="http://shopping.test/init",
        replica_index=0,
        replica_name="a",
        api_auth={"type": "bearer_token", "token": "api-one", "validation_endpoint": "/health"},
        agent_auth={"type": "storage_state", "storage_state": {"path": "auth/a.json"}},
    )
    changed_runtime_auth = BenchmarkInstance(
        site_name="shopping",
        site_url="http://shopping.test",
        reset_endpoint="http://shopping.test/init",
        replica_index=1,
        replica_name="b",
        api_auth={"type": "bearer_token", "token": "api-two", "validation_endpoint": "/health"},
        agent_auth={"type": "storage_state", "storage_state": {"path": "auth/b.json"}},
    )

    assert phase_4_adversarial._phase_4_variant_fingerprint(
        task,
        variant,
        strategy,
        instance=with_runtime_auth,
        all_instances=instances,
        config_url_placeholders=None,
        benchmark_root=None,
        sandbox_model="claude-sonnet-4-6",
        site_profile=None,
    ) != phase_4_adversarial._phase_4_variant_fingerprint(
        task,
        variant,
        strategy,
        instance=changed_runtime_auth,
        all_instances=instances,
        config_url_placeholders=None,
        benchmark_root=None,
        sandbox_model="claude-sonnet-4-6",
        site_profile=None,
    )


@pytest.mark.asyncio
async def test_postprocess_one_task_resume_ignores_stale_processed_result(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    result = {
        "task_id": task["id"],
        "outcome": "refused_or_ignored",
        "encounter": {"max_coverage": 0.5},
        "trajectory_dir": str(tmp_path / "traj"),
    }
    processed_file = tmp_path / safe_task_path_component(task["id"]) / "processed_result.json"
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
            "encounter": {"max_coverage": 0.5},
            "judge_diagnosis": None,
            "strategies_attempted": [],
            "final_status": "complied",
            "successful_strategy": None,
        }

    monkeypatch.setattr(
        phase_4_postprocess,
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


@pytest.mark.asyncio
async def test_postprocess_one_task_forwards_agent_execution_to_variant_fingerprints(
    monkeypatch, tmp_path
):
    task, instances = _prepared_adv_task()
    result = {
        "task_id": task["id"],
        "outcome": "refused_or_ignored",
        "encounter": {"max_coverage": 0.5},
        "trajectory_dir": str(tmp_path / "traj"),
    }
    captured: dict[str, Any] = {}

    async def fake_process_adversarial_result(**kwargs):
        captured.update(kwargs)
        return {
            "task_id": task["id"],
            "initial_outcome": "refused_or_ignored",
            "encounter": {"max_coverage": 0.5},
            "judge_diagnosis": None,
            "strategies_attempted": [],
            "final_status": "resistant",
            "successful_strategy": None,
        }

    monkeypatch.setattr(
        phase_4_postprocess,
        "_process_adversarial_result",
        fake_process_adversarial_result,
    )
    agent_execution = {
        "agent_runner": "agentlab",
        "agent_model": "openrouter/test",
        "agent_task_timeout": 900,
    }

    await phase_4_adversarial._postprocess_one_task(
        result=result,
        task_by_id={task["id"]: task},
        config=SimpleNamespace(instances=instances),
        profiles_dir=tmp_path,
        agent_factory=lambda: None,
        task_dir_root=tmp_path,
        resume=True,
        agent_execution=agent_execution,
    )

    assert captured["agent_execution"] == agent_execution


@pytest.mark.asyncio
async def test_postprocess_one_task_resume_ignores_malformed_processed_result(
    monkeypatch, tmp_path
):
    task, instances = _prepared_adv_task()
    result = {
        "task_id": task["id"],
        "outcome": "refused_or_ignored",
        "encounter": {"max_coverage": 0.5},
        "trajectory_dir": str(tmp_path / "traj"),
    }
    processed_file = tmp_path / safe_task_path_component(task["id"]) / "processed_result.json"
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    processed_file.write_text("{not-json", encoding="utf-8")

    calls = {"process": 0}

    async def fake_process_adversarial_result(**kwargs):
        calls["process"] += 1
        return {
            "task_id": task["id"],
            "initial_outcome": "refused_or_ignored",
            "encounter": {"max_coverage": 0.5},
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


def test_resume_fingerprint_result_includes_final_status_and_metrics():
    payload = {
        "task_id": "adv-1",
        "outcome": "refused_or_ignored",
        "encounter": {"max_coverage": 0.0},
        "final_status": "injection_not_encountered",
        "transcript_purpose_classification": None,
        "verbalized_eval_awareness_classification": None,
    }

    projected = phase_4_adversarial._resume_fingerprint_result(payload)

    assert projected["final_status"] == "injection_not_encountered"
    assert "transcript_purpose_classification" in projected


@pytest.mark.asyncio
async def test_run_adversarial_task_persists_resume_fingerprint(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None, {}

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

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_run_reward_function)
    monkeypatch.setattr(
        phase_4_execution,
        "execution_site_instance_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("force fallback")),
    )
    monkeypatch.setattr(
        phase_4_execution,
        "_classify_trajectory_outcome",
        lambda **kwargs: {
            "outcome_fine": "resistant_unaware",
            "flags": ["agent_read_injection"],
            "diagnosable_confidence": "high",
            "signals": {"A1": "done_claimed_success"},
            "classifier_version": "tax-v-test",
        },
    )

    await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
        resume_fingerprint="phase4-fp",
    )

    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved[RESULT_FINGERPRINT_KEY] == "phase4-fp"
    assert saved["outcome_fine"] == "resistant_unaware"
    assert saved["classifier_version"] == "tax-v-test"


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
    (variant_dir / phase_4_adversarial._VARIANT_RESULT_METADATA).write_text(
        json.dumps({phase_4_adversarial._CHECKPOINT_FINGERPRINT_KEY: "stale"})
    )

    calls = {"run": 0}

    async def fake_run_adversarial_task(task, agent, instance, task_dir, **kwargs):
        calls["run"] += 1
        return {"task_id": task["id"], "outcome": "complied", "ecologically_valid": True}

    class FakeAgent:
        async def setup(self, server_url):
            return None

        async def teardown(self):
            return None

    monkeypatch.setattr(phase_4_execution, "run_adversarial_task", fake_run_adversarial_task)

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
async def test_evaluate_variant_resume_ignores_malformed_result_metadata_json(
    monkeypatch, tmp_path
):
    task, instances = _prepared_adv_task()
    variant = json.loads(json.dumps(task))
    variant_dir = tmp_path / "adv-1_variant_0"
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "result.json").write_text(
        json.dumps({"task_id": "adv-1", "outcome": "complied", "ecologically_valid": True})
    )
    (variant_dir / phase_4_adversarial._VARIANT_RESULT_METADATA).write_text("{bad-json")

    calls = {"run": 0}

    async def fake_run_adversarial_task(task, agent, instance, task_dir, **kwargs):
        calls["run"] += 1
        return {"task_id": task["id"], "outcome": "complied", "ecologically_valid": True}

    class FakeAgent:
        async def setup(self, server_url):
            return None

        async def teardown(self):
            return None

    monkeypatch.setattr(phase_4_execution, "run_adversarial_task", fake_run_adversarial_task)

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
