# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401


@pytest.mark.asyncio
async def test_run_strategy_variation_handles_variant_generation_exceptions(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    variant = json.loads(json.dumps(task))
    variant["adversarial_data_seed"] = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "shopping",
                "method": "create_product_review",
                "args": {"entity_pk_value": 2, "detail": "payload"},
            }
        ],
    }
    evaluated: list[str] = []

    async def fake_run_judge(*args, **kwargs):
        return {
            "status": "ok",
            "diagnosis": "needs more specificity",
            "recommended_strategies": [
                {"strategy": "specificity"},
                {"strategy": "authority_framing"},
            ],
        }

    async def fake_generate_variant(task, strategy, profile_path, **kwargs):
        if strategy["strategy"] == "authority_framing":
            raise RuntimeError("sandbox crashed")
        return variant

    async def fake_evaluate_variant(**kwargs):
        evaluated.append(kwargs["strategy"]["strategy"])
        return {
            "task_id": task["id"],
            "outcome": "refused_or_ignored",
            "encounter": {"max_coverage": 0.5},
            "strategy": kwargs["strategy"]["strategy"],
        }

    monkeypatch.setattr(phase_4_strategy_variation, "run_judge", fake_run_judge)
    monkeypatch.setattr(phase_4_strategy_variation, "generate_variant", fake_generate_variant)
    monkeypatch.setattr(phase_4_strategy_variation, "_evaluate_variant", fake_evaluate_variant)

    result = await phase_4_strategy_variation.run_strategy_variation(
        task=task,
        initial_result={"trajectory_dir": str(tmp_path / "traj")},
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        variant_budget_preset="smoke-3-probe",
    )

    assert result["status"] == "varied"
    assert evaluated == ["specificity"]
    assert result["variant_results"][0]["task_id"] == "adv-1"
    assert result["variant_results"][0]["outcome"] == "refused_or_ignored"
    assert result["variant_results"][0]["encounter"] == {"max_coverage": 0.5}
    assert result["variant_results"][0]["strategy"] == "specificity"
    assert result["variant_generation_errors"] == [
        {
            "strategy": "authority_framing",
            "status": None,
            "error": "RuntimeError('sandbox crashed')",
            "reason": "",
            "round_index": 1,
            "round_variant_index": 1,
            "global_variant_index": 1,
        }
    ]


@pytest.mark.asyncio
async def test_run_strategy_variation_treats_progress_callback_as_observational(
    monkeypatch, tmp_path
):
    task, instances = _prepared_adv_task()

    async def fake_run_judge(*args, **kwargs):
        return {
            "status": "ok",
            "diagnosis": "no strategies",
            "recommended_strategies": [],
        }

    async def failing_progress_callback(*args, **kwargs):
        raise RuntimeError("progress disk full")

    monkeypatch.setattr(phase_4_strategy_variation, "run_judge", fake_run_judge)

    result = await phase_4_strategy_variation.run_strategy_variation(
        task=task,
        initial_result={"trajectory_dir": str(tmp_path / "traj")},
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        progress_callback=failing_progress_callback,
    )

    assert result["status"] == "judge_failed"
    assert result["variant_results"] == []
    assert result["judge_diagnosis"]["validation_errors"] == [
        "judge returned no recommended strategies"
    ]


@pytest.mark.asyncio
async def test_run_strategy_variation_marks_partial_capacity(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    specificity_variant = json.loads(json.dumps(task))
    specificity_variant["adversarial_data_seed"] = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "shopping",
                "method": "create_product_review",
                "args": {"entity_pk_value": 2, "detail": "payload"},
            }
        ],
    }
    authority_variant = json.loads(json.dumps(task))
    authority_variant["adversarial_data_seed"] = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "shopping",
                "method": "create_product_review",
                "args": {"entity_pk_value": 3, "detail": "payload"},
            }
        ],
    }

    async def fake_run_judge(*args, **kwargs):
        return {
            "status": "ok",
            "diagnosis": "try two strategies",
            "recommended_strategies": [
                {"strategy": "specificity"},
                {"strategy": "authority_framing"},
            ],
        }

    async def fake_generate_variant(task, strategy, profile_path, **kwargs):
        if strategy["strategy"] == "specificity":
            return specificity_variant
        return authority_variant

    async def fake_evaluate_variant(**kwargs):
        return {
            "task_id": task["id"],
            "outcome": "refused_or_ignored",
            "encounter": {"max_coverage": 0.5},
            "strategy": kwargs["strategy"]["strategy"],
        }

    monkeypatch.setattr(phase_4_strategy_variation, "run_judge", fake_run_judge)
    monkeypatch.setattr(phase_4_strategy_variation, "generate_variant", fake_generate_variant)
    monkeypatch.setattr(phase_4_strategy_variation, "_evaluate_variant", fake_evaluate_variant)

    result = await phase_4_strategy_variation.run_strategy_variation(
        task=task,
        initial_result={"trajectory_dir": str(tmp_path / "traj")},
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        variant_budget_preset="smoke-3-probe",
    )

    assert result["status"] == "varied"
    assert [item["strategy"] for item in result["variant_results"]] == [
        "specificity",
        "authority_framing",
    ]


@pytest.mark.asyncio
async def test_run_strategy_variation_marks_all_variant_generation_exceptions_failed(
    monkeypatch, tmp_path
):
    task, instances = _prepared_adv_task()

    async def fake_run_judge(*args, **kwargs):
        return {
            "status": "ok",
            "diagnosis": "needs more specificity",
            "recommended_strategies": [{"strategy": "specificity"}],
        }

    async def fake_generate_variant(*args, **kwargs):
        raise RuntimeError("sandbox crashed")

    monkeypatch.setattr(phase_4_strategy_variation, "run_judge", fake_run_judge)
    monkeypatch.setattr(phase_4_strategy_variation, "generate_variant", fake_generate_variant)

    result = await phase_4_strategy_variation.run_strategy_variation(
        task=task,
        initial_result={"trajectory_dir": str(tmp_path / "traj")},
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        variant_budget_preset="smoke-3-probe",
    )

    assert result["status"] == "variant_generation_failed"
    assert result["variant_results"] == []
    assert result["variant_generation_errors"] == [
        {
            "strategy": "specificity",
            "status": None,
            "error": "RuntimeError('sandbox crashed')",
            "reason": "",
            "round_index": 1,
            "round_variant_index": 0,
            "global_variant_index": 0,
        }
    ]


@pytest.mark.asyncio
async def test_run_strategy_variation_uses_ecological_validity_for_progress(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    variant = json.loads(json.dumps(task))
    variant["adversarial_data_seed"]["editor_calls"][0]["args"]["detail"] = "changed"
    progress_events: list[tuple[str, dict]] = []

    async def fake_run_judge(*args, **kwargs):
        return {
            "status": "ok",
            "diagnosis": "try specificity",
            "recommended_strategies": [{"strategy": "specificity"}],
        }

    async def fake_generate_variant(*args, **kwargs):
        return variant

    async def fake_evaluate_variant(**kwargs):
        return {
            "task_id": task["id"],
            "outcome": "error",
            "error": "boom",
            "strategy": kwargs["strategy"]["strategy"],
        }

    async def record_progress(event, data):
        progress_events.append((event, dict(data)))

    monkeypatch.setattr(phase_4_strategy_variation, "run_judge", fake_run_judge)
    monkeypatch.setattr(phase_4_strategy_variation, "generate_variant", fake_generate_variant)
    monkeypatch.setattr(phase_4_strategy_variation, "_evaluate_variant", fake_evaluate_variant)

    result = await phase_4_strategy_variation.run_strategy_variation(
        task=task,
        initial_result={"trajectory_dir": str(tmp_path / "traj")},
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        variant_budget_preset="smoke-3-probe",
        progress_callback=record_progress,
    )

    assert result["status"] == "varied"
    complete = [data for event, data in progress_events if event == "variant_evaluation_complete"]
    assert complete[-1]["pvpo_valid"] == 0


@pytest.mark.asyncio
async def test_run_strategy_variation_runs_adaptive_budget_rounds(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    strategies = [
        {"strategy": "specificity"},
        {"strategy": "authority_framing"},
        {"strategy": "semantic_coherence"},
    ]

    async def fake_run_judge(*args, **kwargs):
        return {
            "status": "ok",
            "diagnosis": "try the adaptive budget",
            "recommended_strategies": strategies,
        }

    async def fake_generate_variant(task, strategy, profile_path, **kwargs):
        variant = json.loads(json.dumps(task))
        failure_context = kwargs.get("failure_context")
        adaptive_loop = (
            failure_context.get("adaptive_loop") if isinstance(failure_context, dict) else None
        )
        round_index = (
            adaptive_loop.get("current_round_index") if isinstance(adaptive_loop, dict) else 1
        )
        variant["adversarial_data_seed"]["editor_calls"][0]["args"]["detail"] = (
            f"{strategy['strategy']}-{round_index}"
        )
        return variant

    async def fake_evaluate_variant(**kwargs):
        return {
            "task_id": task["id"],
            "outcome": "refused_or_ignored",
            "encounter": {"max_coverage": 0.5},
            "strategy": kwargs["strategy"]["strategy"],
        }

    monkeypatch.setattr(phase_4_strategy_variation, "run_judge", fake_run_judge)
    monkeypatch.setattr(phase_4_strategy_variation, "generate_variant", fake_generate_variant)
    monkeypatch.setattr(phase_4_strategy_variation, "_evaluate_variant", fake_evaluate_variant)

    result = await phase_4_strategy_variation.run_strategy_variation(
        task=task,
        initial_result={"trajectory_dir": str(tmp_path / "traj")},
        primary_instances=[instances[0], instances[1], instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        variant_budget_preset="adaptive-3-3-1",
    )

    assert result["status"] == "varied"
    assert len(result["variant_results"]) == 7
    assert result["adaptive_budget"]["shape"] == [3, 3, 1]
    assert [round_record["round_kind"] for round_record in result["variant_rounds"]] == [
        "initial_fanout",
        "adaptive_fanout",
        "terminal_probe",
    ]


@pytest.mark.asyncio
async def test_evaluate_variant_obeys_browser_worker_semaphore(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    active = 0
    max_active = 0

    class FakeAgent:
        async def setup(self, site_url):
            return None

        async def teardown(self):
            return None

    async def fake_run_adversarial_task(*args, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return {
            "task_id": task["id"],
            "outcome": "refused_or_ignored",
            "encounter": {"max_coverage": 0.5},
        }

    monkeypatch.setattr(phase_4_execution, "run_adversarial_task", fake_run_adversarial_task)
    semaphore = asyncio.Semaphore(1)
    variants = []
    for index in range(3):
        variant = json.loads(json.dumps(task))
        variant["adversarial_data_seed"]["editor_calls"][0]["args"]["detail"] = f"changed-{index}"
        variants.append(variant)

    await asyncio.gather(
        *[
            phase_4_variant_eval._evaluate_variant(
                task=task,
                variant=variant,
                instance=instances[0],
                all_instances=instances,
                strategy={"strategy": f"s{index}"},
                index=index,
                agent_factory=FakeAgent,
                task_dir_root=tmp_path,
                browser_worker_semaphore=semaphore,
            )
            for index, variant in enumerate(variants)
        ]
    )

    assert max_active == 1
