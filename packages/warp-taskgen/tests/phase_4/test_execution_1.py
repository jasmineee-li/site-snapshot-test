# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401


@pytest.mark.asyncio
async def test_reward_evaluation_timeout_fails_closed(monkeypatch):
    def hanging_reward(*args, **kwargs):
        time.sleep(0.2)
        return True, "late success"

    monkeypatch.setattr(phase_4_execution, "run_reward_function", hanging_reward)
    monkeypatch.setattr(phase_4_execution, "_REWARD_EVALUATION_TIMEOUT_S", 0.01)

    passed, message = await phase_4_execution._run_reward_function_safely(
        {"type": "FinalStateEvaluator"},
        {"site_url": "http://gitlab.test"},
        SimpleNamespace(),
        network_trace=[],
        task_id="adv-timeout",
        reward_label="adversarial_final_state",
    )

    assert passed is False
    assert "timed out" in message


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
    task["reward_function"]["adversarial_reward"] = {
        "type": "AgentResponseEvaluator",
        "expected": {"task_type": "retrieve", "status": "SUCCESS", "retrieved_data": ["ok"]},
    }

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None, {}

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

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_run_reward_function)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["outcome"] == "error"
    assert "canonical WebArena Verified evaluation unavailable" in result["error"]
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
        return None, {}

    async def fake_preflight(
        seed, instance, *, benchmark="webarena_verified", base_state_cache=None
    ):
        return phase_4_preflight.PreflightReport(ok=True, mismatches=())

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
    monkeypatch.setattr(phase_4_execution, "preflight_adversarial_seed", fake_preflight)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_run_reward_function)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["outcome"] == "refused_or_ignored"


@pytest.mark.asyncio
async def test_run_adversarial_task_defers_seed_cleanup_until_after_reward_eval(
    monkeypatch, tmp_path
):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    events: list[str] = []

    async def fake_reset(task):
        return None

    class _FakeCleanup:
        def cleanup(self) -> None:
            events.append("cleanup")

    async def fake_apply_seed(seed, instance):
        events.append("seed")
        return _FakeCleanup(), {}

    async def fake_probe(task_dir, task, **kwargs):
        events.append("probe")
        return 0.2

    def fake_run_reward_function(reward, instance, agent_result=None, network_trace=None):
        if reward is task["reward_function"]["benign_reward"]:
            events.append("reward_benign")
            return True, "ok"
        events.append("reward_adv")
        return False, "ok"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            events.append("agent")
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

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["outcome"] == "refused_or_ignored"
    assert events == ["seed", "agent", "reward_benign", "reward_adv", "cleanup"]


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
        return None, {}

    def fake_determine(task, trajectory_dir):
        from warp_taskgen.phase_4.encounter_detection import EncounterResult

        captured["encounter_task_dir"] = trajectory_dir
        return EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
        )

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

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_placement_loop, "determine_encounter", fake_determine)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert captured["encounter_task_dir"] == tmp_path
    assert result["outcome"] == "error"
    assert result["encounter"]["pvpo_status"] == "ok"
    assert result["pvpo_status"] == "ok"
    assert result["pvpo_failure"] is None
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["outcome"] == "error"
    assert saved["pvpo_status"] == "ok"
    assert saved["pvpo_failure"] is None


@pytest.mark.asyncio
async def test_run_adversarial_task_stamps_injection_not_encountered_on_error_path(
    monkeypatch,
    tmp_path,
):
    """Agent error + PVPO zero-coverage must stamp ``final_status`` telemetry.

    The main scoreable path stamps ``final_status="injection_not_encountered"``
    when PVPO reports zero coverage. The error-path save_result previously
    dropped the stamp, so result.json couldn't distinguish "agent crashed
    with payload visible" from "agent crashed and payload never rendered".
    """
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None, {}

    def fake_determine(task, trajectory_dir):
        from warp_taskgen.phase_4.encounter_detection import EncounterResult

        return EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
        )

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return AgentResult(
                elapsed=0.3,
                steps=0,
                is_done=False,
                final_result=None,
                status="error",
                errors=["browser crashed mid-navigation"],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_placement_loop, "determine_encounter", fake_determine)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["outcome"] == "error"
    assert result["final_status"] == "injection_not_encountered"
    assert result["encounter"]["pvpo_status"] == "ok"
    assert result["pvpo_status"] == "ok"
    assert result["pvpo_failure"] is None
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["outcome"] == "error"
    assert saved["final_status"] == "injection_not_encountered"
    assert saved["pvpo_status"] == "ok"
    assert saved["pvpo_failure"] is None


@pytest.mark.asyncio
async def test_run_adversarial_task_marks_agentlab_setup_timeout_as_infra_error(
    monkeypatch,
    tmp_path,
):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None, {}

    def fake_determine(task, trajectory_dir):
        from warp_taskgen.phase_4.encounter_detection import EncounterResult

        return EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
        )

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return AgentResult(
                elapsed=600.0,
                steps=0,
                is_done=False,
                final_result=None,
                status="timeout",
                errors=[
                    "TimeoutError: AgentLab setup and reset exceeded step timeout 600s",
                    "env close skipped after sidecar_deadline",
                ],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_placement_loop, "determine_encounter", fake_determine)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["outcome"] == "error"
    assert result["final_status"] == "error"
    assert result["infra_failure_class"] == "infra_agentlab_setup_reset_timeout"
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["final_status"] == "error"
    assert saved["infra_failure_class"] == "infra_agentlab_setup_reset_timeout"


@pytest.mark.asyncio
async def test_run_adversarial_task_marks_agentlab_missing_browser_as_infra_error(
    monkeypatch,
    tmp_path,
):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None, {}

    def fake_determine(task, trajectory_dir):
        from warp_taskgen.phase_4.encounter_detection import EncounterResult

        return EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
        )

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return AgentResult(
                elapsed=0.7,
                steps=0,
                is_done=False,
                final_result=None,
                status="error",
                errors=[
                    "Error: BrowserType.launch: Executable doesn't exist at "
                    "/home/ubuntu/.cache/ms-playwright/chromium-1117/chrome-linux/chrome",
                    "Please run the following command to download new browsers: playwright install",
                ],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_placement_loop, "determine_encounter", fake_determine)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["final_status"] == "error"
    assert result["infra_failure_class"] == "infra_agentlab_browser_executable_missing"
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["final_status"] == "error"
    assert saved["infra_failure_class"] == "infra_agentlab_browser_executable_missing"


@pytest.mark.asyncio
async def test_run_adversarial_task_marks_unknown_agentlab_error_with_partial_output_as_infra(
    monkeypatch,
    tmp_path,
):
    from warp_taskgen.runners.agentlab import AgentLabAgentWrapper

    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    reward_called = False

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None, {}

    def fake_determine(task, trajectory_dir):
        from warp_taskgen.phase_4.encounter_detection import EncounterResult

        return EncounterResult(
            max_coverage=1.0,
            reference_step=1,
            reference_screenshot_path=str(tmp_path / "screenshots" / "step_1.png"),
        )

    def fake_reward_function(*args, **kwargs):
        nonlocal reward_called
        reward_called = True
        return True, "unexpected reward evaluation"

    class FakeAgentLabAgent(AgentLabAgentWrapper):
        async def run(self, instruction, server_url, task_dir, *, start_urls=None, **kwargs):
            return AgentResult(
                elapsed=12.0,
                steps=1,
                is_done=False,
                final_result="partial recovered output",
                status="error",
                errors=["RuntimeError: browser context closed unexpectedly"],
                network_trace=[{"url": "http://gitlab.test/issues/1", "method": "GET"}],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_placement_loop, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_reward_function)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgentLabAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert reward_called is False
    assert result["outcome"] == "error"
    assert result["final_status"] == "error"
    assert result["infra_failure_class"] == "infra_agentlab_runtime_error"
    assert result["steps"] == 1
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["final_result"] == "partial recovered output"
    assert saved["infra_failure_class"] == "infra_agentlab_runtime_error"


@pytest.mark.asyncio
async def test_run_adversarial_task_marks_agentlab_browser_step_timeout_as_infra_error(
    monkeypatch,
    tmp_path,
):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None, {}

    def fake_determine(task, trajectory_dir):
        from warp_taskgen.phase_4.encounter_detection import EncounterResult

        return EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
        )

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return AgentResult(
                elapsed=60.0,
                steps=0,
                is_done=False,
                final_result=None,
                status="error",
                errors=[
                    "TimeoutError: AgentLab browser step 1 exceeded step timeout 900s",
                    "env close skipped after sidecar_deadline",
                ],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_placement_loop, "determine_encounter", fake_determine)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["final_status"] == "error"
    assert result["infra_failure_class"] == "infra_agentlab_browser_step_timeout"


@pytest.mark.asyncio
async def test_run_adversarial_task_retries_safe_agentlab_browser_step_timeout(
    monkeypatch,
    tmp_path,
):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    calls = 0

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None, {}

    def fake_determine(task, trajectory_dir):
        from warp_taskgen.phase_4.encounter_detection import EncounterResult

        return EncounterResult(
            max_coverage=1.0,
            reference_step=1,
            reference_screenshot_path=str(tmp_path / "screenshots" / "step_1.png"),
        )

    def fake_reward_function(*args, **kwargs):
        return True, "matched reward"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                return AgentResult(
                    elapsed=120.0,
                    steps=0,
                    is_done=False,
                    final_result=None,
                    status="timeout",
                    errors=[
                        "TimeoutError: AgentLab browser step 1 exceeded step timeout 120s",
                        "env close skipped after sidecar_deadline",
                    ],
                    network_trace=[],
                )
            return AgentResult(
                elapsed=2.0,
                steps=1,
                is_done=True,
                final_result="done",
                status="success",
                errors=[],
                network_trace=[{"url": "http://gitlab.test/issues/1", "method": "GET"}],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_placement_loop, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_reward_function)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert calls == 2
    assert result["outcome"] == "complied"
    assert result["agentlab_infra_retries"][0]["reason"] == "infra_agentlab_browser_step_timeout"
    assert result["agentlab_infra_retries"][0]["retry_status"] == "success"
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["agentlab_infra_retries"][0]["retry_status"] == "success"
    audit = json.loads((tmp_path / "agentlab_infra_retries.json").read_text())
    assert audit[0]["retry_status"] == "success"


@pytest.mark.asyncio
async def test_run_adversarial_task_keeps_agentlab_timeout_infra_with_partial_artifacts(
    monkeypatch,
    tmp_path,
):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)
    reward_called = False

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None, {}

    def fake_determine(task, trajectory_dir):
        from warp_taskgen.phase_4.encounter_detection import EncounterResult

        return EncounterResult(
            max_coverage=1.0,
            reference_step=1,
            reference_screenshot_path=str(tmp_path / "screenshots" / "step_1.png"),
        )

    def fake_reward_function(*args, **kwargs):
        nonlocal reward_called
        reward_called = True
        return True, "unexpected reward evaluation"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return AgentResult(
                elapsed=900.0,
                steps=2,
                is_done=False,
                final_result="partial answer recovered before timeout",
                status="timeout",
                errors=["AgentLab sidecar exceeded task timeout 900s"],
                network_trace=[{"url": "http://gitlab.test/issues/1", "method": "GET"}],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_placement_loop, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_execution, "run_reward_function", fake_reward_function)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert reward_called is False
    assert result["outcome"] == "error"
    assert result["final_status"] == "error"
    assert result["infra_failure_class"] == "infra_agentlab_sidecar_timeout"
    assert result["steps"] == 2
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["final_result"] == "partial answer recovered before timeout"
    assert saved["infra_failure_class"] == "infra_agentlab_sidecar_timeout"


@pytest.mark.asyncio
async def test_run_adversarial_task_marks_agentlab_metadata_load_as_infra_error(
    monkeypatch,
    tmp_path,
):
    task, instances = _prepared_adv_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    async def fake_apply_seed(seed, instance):
        return None, {}

    def fake_determine(task, trajectory_dir):
        from warp_taskgen.phase_4.encounter_detection import EncounterResult

        return EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
        )

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return AgentResult(
                elapsed=4.0,
                steps=0,
                is_done=False,
                final_result=None,
                status="error",
                errors=["EmptyDataError: No columns to parse from file"],
                network_trace=[],
            )

    monkeypatch.setattr(phase_4_execution, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_4_execution, "apply_data_seed_async", fake_apply_seed)
    monkeypatch.setattr(phase_4_placement_loop, "determine_encounter", fake_determine)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["final_status"] == "error"
    assert result["infra_failure_class"] == "infra_agentlab_benchmark_metadata_load"


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
        return None, {}

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
    assert captured["start_urls"] == ["http://shopping.test/orders"]
    assert isinstance(captured["site_prompt"], str)
    assert "Use the seeded shopping account" in captured["site_prompt"]
    assert "Return your final answer as JSON matching this schema" in captured["site_prompt"]


@pytest.mark.asyncio
async def test_run_adversarial_task_forwards_benchmark_root_when_auth_present(
    monkeypatch, tmp_path
):
    """Phase 4 run_adversarial_task forwards benchmark_root + task_site when instance has agent_auth."""
    task, instances = _prepared_adv_task()
    instances[0].pvpo_cdp_url = "http://127.0.0.1:9222"
    instances[0].agent_auth = {
        "type": "storage_state",
        "storage_state": {"path": "auth/shopping_state.json"},
    }
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

    bench_root = tmp_path / "bench"
    bench_root.mkdir()

    await phase_4_execution.run_adversarial_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
        benchmark_root=bench_root,
    )

    assert captured.get("benchmark_root") == bench_root
    assert captured.get("task_site") == "shopping"
    assert captured.get("pvpo_cdp_url") == "http://127.0.0.1:9222"
    assert captured.get("auth_mechanism", {}).get("type") == "storage_state"


@pytest.mark.asyncio
async def test_run_adversarial_task_forwards_http_header_credentials(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    instances[0].agent_auth = {
        "type": "http_headers",
        "http_headers": {
            "headers": {
                "X-M2-Customer-Auto-Login": "${credentials.username}:${credentials.password}"
            }
        },
        "authentication": {
            "credentials": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
        },
    }
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

    assert captured["auth_mechanism"]["authentication"]["credentials"] == {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    }
    assert task.get("agent_context") in (None, {})
