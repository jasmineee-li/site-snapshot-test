# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401


@pytest.mark.asyncio
async def test_run_adversarial_task_does_not_mark_public_site_pre_authenticated(
    monkeypatch, tmp_path
):
    task, instances = _prepared_adv_task()
    instances[0].agent_auth = {"type": "none"}
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_seed(seed, instance_dict):
        return None, {}

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

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_seed)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_run_reward_function)

    await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert captured["auth_mechanism"]["type"] == "none"
    assert task.get("agent_context") in (None, {})


@pytest.mark.asyncio
async def test_run_adversarial_task_omits_benchmark_root_without_auth(monkeypatch, tmp_path):
    """Without auth_mechanism, run_adversarial_task omits the auth-only kwargs."""
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_seed(seed, instance_dict):
        return None, {}

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

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_seed)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_run_reward_function)

    await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
        benchmark_root=tmp_path / "bench",
    )

    assert "benchmark_root" not in captured
    assert "task_site" not in captured
    assert "auth_mechanism" not in captured


@pytest.mark.asyncio
async def test_run_adversarial_task_seeds_against_delivery_site_instance(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    task["delivery_channel"] = {
        "mechanism": "form",
        "body_field": "detail",
        "delivery_site": "gitlab",
        "postcondition": {"type": "db_row_value"},
    }
    task = bind_task_to_instance(task, instances[0], instances)

    captured: dict[str, object] = {}

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance_dict):
        captured["seed_instance"] = instance_dict
        return None, {}

    def fake_run_reward_function(reward, instance, agent_result=None, network_trace=None):
        return reward is task["reward_function"]["benign_reward"], "ok"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None, **kwargs):
            captured["server_url"] = server_url
            return AgentResult(
                elapsed=0.1,
                steps=1,
                is_done=True,
                final_result="done",
                status="success",
                errors=[],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_run_reward_function)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["outcome"] == "refused_or_ignored"
    assert captured["server_url"] == "http://shopping.test"
    assert isinstance(captured["seed_instance"], dict)
    assert captured["seed_instance"]["site_name"] == "gitlab"
    assert captured["seed_instance"]["site_url"] == "http://gitlab.test"


@pytest.mark.asyncio
async def test_run_adversarial_task_does_not_fall_back_to_payload_zero(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    task["payload_texts"] = [
        {"rendered_payload": "payload zero"},
        {"rendered_payload": "payload one"},
    ]
    task["selected_payload_index"] = 9

    captured: dict[str, object] = {}

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance_dict):
        return None, {}

    def fake_run_reward_function(reward, instance, agent_result=None, network_trace=None):
        return reward is task["reward_function"]["benign_reward"], "ok"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None, **kwargs):
            captured["payload_text"] = kwargs.get("payload_text")
            captured["payload_witnesses"] = kwargs.get("payload_witnesses")
            return AgentResult(
                elapsed=0.1,
                steps=1,
                is_done=True,
                final_result="done",
                status="success",
                errors=[],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_run_reward_function)

    await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert captured["payload_text"] is None
    assert captured["payload_witnesses"] is None


@pytest.mark.asyncio
async def test_run_adversarial_task_fallback_delivery_site_uses_deterministic_replica(
    monkeypatch, tmp_path
):
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping.test",
            reset_endpoint="http://shopping.test/init",
            replica_index=0,
            replica_name="shopping_0",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-0.test",
            reset_endpoint="http://gitlab-0.test/init",
            replica_index=0,
            replica_name="gitlab_0",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-1.test",
            reset_endpoint="http://gitlab-1.test/init",
            replica_index=1,
            replica_name="gitlab_1",
        ),
    ]

    task_payload = None
    for index in range(100):
        candidate = {
            "id": f"adv-fallback-{index}",
            "benchmark": "webarena_verified",
            "benign_task_id": "benign-1",
            "site": "shopping",
            "instruction": "Find the order",
            "start_urls": ["__SHOPPING__/orders"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {
                "benign_reward": {"type": "noop"},
                "adversarial_reward": {"type": "noop"},
            },
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "method": "create_product_review",
                        "args": {"entity_pk_value": 1, "detail": "payload"},
                    }
                ],
            },
            "delivery_channel": {
                "mechanism": "form",
                "body_field": "detail",
                "delivery_site": "gitlab",
                "postcondition": {"type": "db_row_value"},
            },
        }
        if select_task_site_instance(candidate, "gitlab", instances).replica_index == 1:
            task_payload = candidate
            break
    assert task_payload is not None

    task, missing = prepare_task_for_execution(task_payload, instances)
    assert missing == []
    task = bind_task_to_instance(task, instances[0], instances)

    captured: dict[str, object] = {}

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance_dict):
        captured["seed_instance"] = instance_dict
        return None, {}

    def fake_run_reward_function(reward, instance, agent_result=None, network_trace=None):
        return reward is task["reward_function"]["benign_reward"], "ok"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None, **kwargs):
            return AgentResult(
                elapsed=0.1,
                steps=1,
                is_done=True,
                final_result="done",
                status="success",
                errors=[],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_run_reward_function)

    await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
        all_instances=instances,
    )

    expected = select_task_site_instance(task, "gitlab", instances)
    assert isinstance(captured["seed_instance"], dict)
    assert captured["seed_instance"]["site_name"] == "gitlab"
    assert captured["seed_instance"]["site_url"] == expected.site_url


@pytest.mark.asyncio
async def test_run_adversarial_task_marks_cross_site_delivery_replica_dirty_in_reset_cache(
    monkeypatch, tmp_path
):
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping.test",
            reset_endpoint="http://shopping.test/init",
            replica_index=0,
            replica_name="shopping_0",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-0.test",
            reset_endpoint="http://gitlab-0.test/init",
            replica_index=0,
            replica_name="gitlab_0",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-1.test",
            reset_endpoint="http://gitlab-1.test/init",
            replica_index=1,
            replica_name="gitlab_1",
        ),
    ]

    shopping_task_payload = None
    for index in range(100):
        candidate = {
            "id": f"adv-cross-site-{index}",
            "benchmark": "webarena_verified",
            "benign_task_id": "benign-1",
            "site": "shopping",
            "instruction": "Find the order",
            "start_urls": ["__SHOPPING__/orders"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {
                "benign_reward": {"type": "noop"},
                "adversarial_reward": {"type": "noop"},
            },
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "method": "create_product_review",
                        "args": {"entity_pk_value": 1, "detail": "payload"},
                    }
                ],
            },
            "delivery_channel": {
                "mechanism": "form",
                "body_field": "detail",
                "delivery_site": "gitlab",
                "postcondition": {"type": "db_row_value"},
            },
        }
        if select_task_site_instance(candidate, "gitlab", instances).replica_index == 1:
            shopping_task_payload = candidate
            break
    assert shopping_task_payload is not None

    shopping_task, missing = prepare_task_for_execution(shopping_task_payload, instances)
    assert missing == []
    shopping_task = bind_task_to_instance(shopping_task, instances[0], instances)

    gitlab_task, missing = prepare_task_for_execution(
        {
            "id": "adv-gitlab-1",
            "benchmark": "webarena_verified",
            "benign_task_id": "benign-1",
            "site": "gitlab",
            "instruction": "Check the issue",
            "start_urls": ["__GITLAB__/issues"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {
                "benign_reward": {"type": "noop"},
                "adversarial_reward": {"type": "noop"},
            },
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "method": "create_product_review",
                        "args": {"entity_pk_value": 1, "detail": "payload"},
                    }
                ],
            },
        },
        instances,
    )
    assert missing == []
    gitlab_task = bind_task_to_instance(gitlab_task, instances[2], instances)

    reset_calls: list[str] = []

    async def fake_reset(task):
        reset_calls.append(task["id"])

    async def fake_apply_seed(seed, instance_dict):
        return None, {}

    def fake_run_reward_function(reward, instance, agent_result=None, network_trace=None):
        return reward is shopping_task["reward_function"]["benign_reward"], "ok"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None, **kwargs):
            return AgentResult(
                elapsed=0.1,
                steps=1,
                is_done=True,
                final_result="done",
                status="success",
                errors=[],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_run_reward_function)

    reset_cache = phase_4_execution.TaskResetCache()
    await phase_4_execution.run_adversarial_task(
        task=shopping_task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path / "shopping",
        all_instances=instances,
        reset_cache=reset_cache,
    )
    await phase_4_execution.run_adversarial_task(
        task=gitlab_task,
        agent=FakeAgent(),
        instance=instances[2],
        task_dir=tmp_path / "gitlab",
        all_instances=instances,
        reset_cache=reset_cache,
    )

    assert reset_calls == [shopping_task["id"], gitlab_task["id"]]
