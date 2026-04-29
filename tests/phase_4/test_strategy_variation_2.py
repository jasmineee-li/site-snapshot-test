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
    )

    assert result["status"] == "varied"
    assert evaluated == ["specificity"]
    assert result["variant_results"] == [
        {
            "task_id": "adv-1",
            "outcome": "refused_or_ignored",
            "encounter": {"max_coverage": 0.5},
            "strategy": "specificity",
        }
    ]
    assert result["variant_generation_errors"] == [
        {
            "strategy": "authority_framing",
            "error": "RuntimeError('sandbox crashed')",
        }
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
    )

    assert result["status"] == "partial_capacity"
    assert [item["strategy"] for item in result["variant_results"]] == ["specificity"]
    assert result["skipped_strategies"] == ["authority_framing"]

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

    monkeypatch.setattr(phase_4_adversarial, "run_judge", fake_run_judge)
    monkeypatch.setattr(phase_4_adversarial, "generate_variant", fake_generate_variant)

    result = await phase_4_adversarial.run_strategy_variation(
        task=task,
        initial_result={"trajectory_dir": str(tmp_path / "traj")},
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
    )

    assert result["status"] == "variant_generation_failed"
    assert result["variant_results"] == []
    assert result["variant_generation_errors"] == [
        {
            "strategy": "specificity",
            "error": "RuntimeError('sandbox crashed')",
        }
    ]
