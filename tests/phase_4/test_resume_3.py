# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

@pytest.mark.asyncio
async def test_run_strategy_variation_resume_continues_partial_generation_checkpoint(
    monkeypatch, tmp_path
):
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
                    )
                ),
                "judge_diagnosis": {
                    "diagnosis": "try two strategies",
                    "recommended_strategies": [
                        {"strategy": "specificity"},
                        {"strategy": "authority_framing"},
                    ],
                },
                phase_4_adversarial._VARIANT_GENERATION_RECORDS_KEY: [
                    {
                        "index": 0,
                        "strategy": {"strategy": "specificity"},
                        "variant": specificity_variant,
                    }
                ],
            }
        )
    )

    generate_calls: list[str] = []

    async def fail_run_judge(*args, **kwargs):
        raise AssertionError("resume should reuse saved judge output")

    async def fake_generate_variant(task, strategy, profile_path, **kwargs):
        generate_calls.append(strategy["strategy"])
        assert strategy["strategy"] == "authority_framing"
        return authority_variant

    async def fake_evaluate_variant(**kwargs):
        return {
            "task_id": task["id"],
            "outcome": "refused_or_ignored",
            "encounter": {"max_coverage": 0.5},
            "strategy": kwargs["strategy"]["strategy"],
        }

    monkeypatch.setattr(phase_4_adversarial, "run_judge", fail_run_judge)
    monkeypatch.setattr(phase_4_adversarial, "generate_variant", fake_generate_variant)
    monkeypatch.setattr(phase_4_adversarial, "_evaluate_variant", fake_evaluate_variant)

    result = await phase_4_adversarial.run_strategy_variation(
        task=task,
        initial_result=initial_result,
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        resume=True,
    )

    assert generate_calls == ["authority_framing"]
    assert result["status"] == "partial_capacity"
    assert result["skipped_strategies"] == ["authority_framing"]
    saved_checkpoint = json.loads(checkpoint_path.read_text())
    assert len(saved_checkpoint[phase_4_adversarial._VARIANT_GENERATION_RECORDS_KEY]) == 2

@pytest.mark.asyncio
async def test_run_strategy_variation_resume_reruns_when_generation_records_missing(
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

    calls = {"generated": 0}

    async def fail_run_judge(*args, **kwargs):
        raise AssertionError("resume should reuse saved judge output")

    async def fake_generate_variant(*args, **kwargs):
        calls["generated"] += 1
        return variant

    async def fake_evaluate_variant(**kwargs):
        return {
            "task_id": task["id"],
            "outcome": "complied",
            "encounter": {"max_coverage": 0.5},
            "strategy": kwargs["strategy"]["strategy"],
        }

    monkeypatch.setattr(phase_4_adversarial, "run_judge", fail_run_judge)
    monkeypatch.setattr(phase_4_adversarial, "generate_variant", fake_generate_variant)
    monkeypatch.setattr(phase_4_adversarial, "_evaluate_variant", fake_evaluate_variant)

    result = await phase_4_adversarial.run_strategy_variation(
        task=task,
        initial_result=initial_result,
        primary_instances=[instances[0]],
        all_instances=instances,
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        resume=True,
    )

    assert calls["generated"] == 1
    assert result["status"] == "varied"

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
        return variant

    async def fake_evaluate_variant(**kwargs):
        return {
            "task_id": task["id"],
            "outcome": "complied",
            "encounter": {"max_coverage": 0.5},
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
async def test_run_strategy_variation_resume_ignores_malformed_checkpoint(monkeypatch, tmp_path):
    task, instances = _prepared_adv_task()
    checkpoint_path = phase_4_adversarial._strategy_variation_checkpoint_path(tmp_path, task["id"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("{bad-json", encoding="utf-8")

    calls = {"judge": 0}

    async def fake_run_judge(*args, **kwargs):
        calls["judge"] += 1
        return {
            "diagnosis": "fresh",
            "recommended_strategies": [{"strategy": "specificity"}],
        }

    async def fake_generate_variant(*args, **kwargs):
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
        return variant

    async def fake_evaluate_variant(**kwargs):
        return {
            "task_id": task["id"],
            "outcome": "complied",
            "encounter": {"max_coverage": 0.5},
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

def test_sweep_orphan_inflight_sentinels_removes_legacy(tmp_path):
    """The Phase 4 entry sweep must unlink the legacy ``.aer_inflight``
    sentinel so re-runs of older trajectories don't leave empty marker
    files on disk. The current-cutover sentinel was removed entirely as
    a write-only resume hint with no consumer; only the legacy name is
    swept now.
    """
    task_a = tmp_path / "task_a"
    task_a.mkdir()
    legacy = task_a / phase_4_adversarial._LEGACY_AER_INFLIGHT_SENTINEL
    legacy.touch()

    removed = phase_4_adversarial._sweep_orphan_inflight_sentinels(tmp_path)

    assert removed == 1
    assert not legacy.exists()

def test_sweep_orphan_inflight_sentinels_on_missing_dir(tmp_path):
    """No-op when the task-dir root hasn't been created yet."""
    missing = tmp_path / "nope"
    assert phase_4_adversarial._sweep_orphan_inflight_sentinels(missing) == 0

@pytest.mark.asyncio
async def test_placement_fix_resume_reuses_pending_iteration_result(monkeypatch, tmp_path):
    task = {
        "id": "placement-fix-resume",
        "site": "shopping",
        "adversarial_data_seed": {"mechanism": "api", "actions": []},
    }
    initial_result = {"outcome": "task_broke", "trajectory_dir": str(tmp_path / "orig")}
    source_fingerprint = "placement-source"
    revised_task = {
        **task,
        "adversarial_data_seed": {"mechanism": "api", "actions": [{"moved": True}]},
    }
    checkpoint_path = phase_4_adversarial._placement_fix_checkpoint_path(tmp_path, task["id"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                phase_4_adversarial._CHECKPOINT_FINGERPRINT_KEY: source_fingerprint,
                "attempts": [initial_result],
                "current_task": revised_task,
                "current_result": initial_result,
                "next_iteration": 0,
                "pending_iteration": 0,
            }
        )
    )
    iteration_dir = tmp_path / "placement-fix-resume__placement_1"
    iteration_dir.mkdir(parents=True)
    iteration_fingerprint = phase_4_adversarial._placement_iteration_result_fingerprint(
        revised_task,
        base_source_fingerprint=source_fingerprint,
        iteration=0,
    )
    (iteration_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "placement-fix-resume",
                "outcome": "complied",
                "encounter": {"max_coverage": 0.5},
                RESULT_FINGERPRINT_KEY: iteration_fingerprint,
            }
        )
    )

    class _FakeInstance:
        site_url = "http://example.com"
        site_name = "shopping"
        reset_endpoint = None

    class _FakeLock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

    monkeypatch.setattr(
        phase_4_adversarial,
        "run_placement_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("placement API should not rerun for pending checkpoint")
        ),
    )
    monkeypatch.setattr(phase_4_adversarial, "task_lock", lambda _task: _FakeLock())
    monkeypatch.setattr(phase_4_adversarial, "bind_task_to_instance", lambda task, *_: task)

    result = await phase_4_adversarial._run_placement_fix_loop(
        task=task,
        initial_result=initial_result,
        instance=_FakeInstance(),
        all_instances=[_FakeInstance()],
        agent_factory=lambda: (_ for _ in ()).throw(AssertionError("agent should not start")),
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        resume=True,
        source_fingerprint=source_fingerprint,
    )

    assert result is not None
    assert result["status"] == "fixed"
    assert result["final_result"]["outcome"] == "complied"

@pytest.mark.asyncio
async def test_placement_fix_resume_reuses_completed_checkpoint(monkeypatch, tmp_path):
    task = {
        "id": "placement-fix-complete",
        "site": "shopping",
        "adversarial_data_seed": {"mechanism": "api", "actions": []},
    }
    initial_result = {"outcome": "task_broke", "trajectory_dir": str(tmp_path / "orig")}
    source_fingerprint = "placement-source"
    completed = {
        "status": "fixed",
        "attempts": [initial_result, {"outcome": "complied"}],
        "final_result": {"outcome": "complied"},
        "final_task": task,
    }
    checkpoint_path = phase_4_adversarial._placement_fix_checkpoint_path(tmp_path, task["id"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                phase_4_adversarial._CHECKPOINT_FINGERPRINT_KEY: source_fingerprint,
                "attempts": completed["attempts"],
                "current_task": task,
                "current_result": completed["final_result"],
                "next_iteration": 1,
                "pending_iteration": None,
                "completed_result": completed,
            }
        )
    )

    class _FakeInstance:
        site_url = "http://example.com"
        site_name = "shopping"
        reset_endpoint = None

    monkeypatch.setattr(
        phase_4_adversarial,
        "run_placement_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed checkpoint should short-circuit placement loop")
        ),
    )

    result = await phase_4_adversarial._run_placement_fix_loop(
        task=task,
        initial_result=initial_result,
        instance=_FakeInstance(),
        all_instances=[_FakeInstance()],
        agent_factory=lambda: None,
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        resume=True,
        source_fingerprint=source_fingerprint,
    )

    assert result == completed
