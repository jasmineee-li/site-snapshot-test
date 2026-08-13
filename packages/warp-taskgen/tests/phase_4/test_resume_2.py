# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401


@pytest.mark.asyncio
async def test_run_strategy_variation_resume_reuses_saved_variant_result(monkeypatch, tmp_path):
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
    initial_result = {"trajectory_dir": str(tmp_path / "traj")}
    checkpoint_fingerprint = phase_4_adversarial._phase_4_postprocess_fingerprint(
        task,
        initial_result,
        primary_instances=[instances[0]],
        all_instances=instances,
        config_url_placeholders=None,
        benchmark_root=None,
        sandbox_model="claude-sonnet-4-6",
        site_profile=None,
        variant_system="strategy-variation",
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
                phase_4_adversarial._VARIANT_GENERATION_RECORDS_KEY: [
                    {
                        "index": 0,
                        "strategy": {"strategy": "specificity"},
                        "variant": variant,
                    }
                ],
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
                "encounter": {"max_coverage": 0.5},
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
                        all_instances=instances,
                        config_url_placeholders=None,
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
    assert result["variant_results"][0]["task_id"] == "adv-1"
    assert result["variant_results"][0]["outcome"] == "complied"
    assert result["variant_results"][0]["encounter"] == {"max_coverage": 0.5}
    assert result["variant_results"][0]["trajectory_dir"] == str(variant_dir)
    assert result["variant_results"][0]["strategy"] == "specificity"


@pytest.mark.asyncio
async def test_run_strategy_variation_resume_ignores_saved_variant_result_from_different_instance(
    monkeypatch, tmp_path
):
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
                        config_url_placeholders=None,
                        benchmark_root=None,
                        sandbox_model="claude-sonnet-4-6",
                        site_profile=None,
                        variant_system="strategy-variation",
                    )
                ),
                "judge_diagnosis": {
                    "diagnosis": "needs more specificity",
                    "recommended_strategies": [{"strategy": "specificity"}],
                },
                phase_4_adversarial._VARIANT_GENERATION_RECORDS_KEY: [
                    {
                        "index": 0,
                        "strategy": {"strategy": "specificity"},
                        "variant": variant,
                    }
                ],
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
                "encounter": {"max_coverage": 0.5},
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
                        all_instances=instances,
                        config_url_placeholders=None,
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
            "encounter": {"max_coverage": 0.5},
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
    monkeypatch.setattr(phase_4_execution, "run_adversarial_task", fake_run_adversarial_task)

    await phase_4_adversarial.run_strategy_variation(
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


@pytest.mark.asyncio
async def test_run_strategy_variation_resume_reuses_variant_result_fingerprint_without_sidecar(
    monkeypatch, tmp_path
):
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
    initial_result = {"trajectory_dir": str(tmp_path / "traj")}
    source_fingerprint = phase_4_adversarial._phase_4_variant_fingerprint(
        task,
        variant,
        {"strategy": "specificity"},
        instance=instances[0],
        all_instances=instances,
        config_url_placeholders=None,
        benchmark_root=None,
        sandbox_model="claude-sonnet-4-6",
        site_profile=None,
    )
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
                        config_url_placeholders=None,
                        benchmark_root=None,
                        sandbox_model="claude-sonnet-4-6",
                        site_profile=None,
                        variant_system="strategy-variation",
                    )
                ),
                "judge_diagnosis": {
                    "diagnosis": "needs more specificity",
                    "recommended_strategies": [{"strategy": "specificity"}],
                },
                phase_4_adversarial._VARIANT_GENERATION_RECORDS_KEY: [
                    {
                        "index": 0,
                        "strategy": {"strategy": "specificity"},
                        "variant": variant,
                    }
                ],
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
                "encounter": {"max_coverage": 0.5},
                RESULT_FINGERPRINT_KEY: source_fingerprint,
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
    assert result["variant_results"][0]["outcome"] == "complied"
    assert result["variant_results"][0]["strategy"] == "specificity"


@pytest.mark.asyncio
async def test_run_strategy_variation_resume_reruns_refusal_variant_without_history(
    monkeypatch, tmp_path
):
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
                        config_url_placeholders=None,
                        benchmark_root=None,
                        sandbox_model="claude-sonnet-4-6",
                        site_profile=None,
                        variant_system="strategy-variation",
                    )
                ),
                "judge_diagnosis": {
                    "diagnosis": "needs more specificity",
                    "recommended_strategies": [{"strategy": "specificity"}],
                },
                phase_4_adversarial._VARIANT_GENERATION_RECORDS_KEY: [
                    {
                        "index": 0,
                        "strategy": {"strategy": "specificity"},
                        "variant": variant,
                    }
                ],
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
                "outcome": "refused_or_ignored",
                "encounter": {"max_coverage": 0.5},
                RESULT_FINGERPRINT_KEY: phase_4_adversarial._phase_4_variant_fingerprint(
                    task,
                    variant,
                    {"strategy": "specificity"},
                    instance=instances[0],
                    all_instances=instances,
                    config_url_placeholders=None,
                    benchmark_root=None,
                    sandbox_model="claude-sonnet-4-6",
                    site_profile=None,
                ),
            }
        )
    )

    calls = {"evaluated": 0}

    async def fail_run_judge(*args, **kwargs):
        raise AssertionError("resume should reuse saved judge output")

    async def fail_generate_variant(*args, **kwargs):
        raise AssertionError("resume should reuse saved variant generation")

    async def fake_run_adversarial_task(task, agent, instance, task_dir, **kwargs):
        calls["evaluated"] += 1
        return {
            "task_id": task["id"],
            "outcome": "refused_or_ignored",
            "encounter": {"max_coverage": 0.5},
        }

    class FakeAgent:
        async def setup(self, server_url):
            return None

        async def teardown(self):
            return None

    monkeypatch.setattr(phase_4_adversarial, "run_judge", fail_run_judge)
    monkeypatch.setattr(phase_4_adversarial, "generate_variant", fail_generate_variant)
    monkeypatch.setattr(phase_4_execution, "run_adversarial_task", fake_run_adversarial_task)

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
    assert result["variant_results"][0]["outcome"] == "refused_or_ignored"
