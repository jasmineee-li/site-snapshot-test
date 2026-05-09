# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

@pytest.mark.asyncio
async def test_phase_4_run_rejects_non_worldsim_v5_benchmark(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    adv_task = {
        "id": "adv-wasp",
        "benchmark": "wasp",
        "benign_task_id": "benign-1",
        "site": "gitlab",
        "sites": ["gitlab"],
        "instruction": "Open the issue",
        "start_urls": ["http://gitlab.test/issues"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"adversarial_reward": {"type": "noop"}},
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
        **_phase4_exposure_fields(),
    }
    benign_task = dict(adv_task)
    benign_task["id"] = "benign-1"
    benign_task["reward_function"] = {"type": "noop"}
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(json.dumps([adv_task]))
    (tmp_path / "phase_3" / "contracts.json").write_text(json.dumps(_as_contracts([benign_task])))
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "wasp",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                        "pvpo_cdp_url": "http://127.0.0.1:9222",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            benchmark=None,
            agent_model="demo-model",
            agent_provider=None,
            resume=False,
        )
    )

    assert rc == 1
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["reason"] == "unsupported_benchmark"
    assert "does not support WARP Taskgen Phase 4" in state["error"]

@pytest.mark.asyncio
async def test_phase_4_run_clears_storage_error_after_retry_success(monkeypatch, tmp_path):
    from worldsim import storage_state_preflight

    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    adv_task = {
        "id": "adv-auth",
        "benign_task_id": "benign-1",
        "site": "gitlab",
        "sites": ["gitlab"],
        "instruction": "Open the issue",
        "start_urls": ["http://gitlab.test/issues"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"adversarial_reward": {"type": "noop"}},
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
    }
    adv_task.update(_phase4_exposure_fields())
    benign_task = dict(adv_task)
    benign_task["id"] = "benign-1"
    benign_task["reward_function"] = {"type": "noop"}
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(json.dumps([adv_task]))
    (tmp_path / "phase_3" / "contracts.json").write_text(json.dumps(_as_contracts([benign_task])))
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                        "pvpo_cdp_url": "http://127.0.0.1:9222",
                        "agent_auth": {
                            "type": "storage_state",
                            "storage_state": {"path": "auth/gitlab-state.json"},
                        },
                    }
                ],
            }
        )
    )
    attempts = 0

    async def fake_ensure_storage_state(instance, *, benchmark_root, benchmark_name):
        nonlocal attempts
        _ = instance, benchmark_root, benchmark_name
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return tmp_path / "auth" / "gitlab-state.json"

    async def fake_run_tasks_by_site(**kwargs):
        _ = kwargs
        return []

    monkeypatch.setattr(storage_state_preflight, "ensure_storage_state", fake_ensure_storage_state)
    monkeypatch.setattr(
        phase_4_adversarial,
        "inspect_storage_state_preflight",
        lambda *args, **kwargs: SimpleNamespace(errors=(), mismatches=()),
    )
    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "make_agent_factory", lambda **kwargs: lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "_load_site_profiles", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        phase_4_adversarial,
        "_collect_agent_auth_runtime_errors",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        phase_4_adversarial,
        "_probe_seed_base_state_for_task_targets",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(phase_4_adversarial, "run_tasks_by_site", fake_run_tasks_by_site)

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            benchmark=None,
            agent_model="demo-model",
            agent_provider=None,
            resume=False,
            skip_host_bound_storage_state_auth=False,
        )
    )

    assert rc == 0
    assert attempts == 2

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
    (tmp_path / "phase_3" / "contracts.json").write_text(
        json.dumps(
            _as_contracts(
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
    )
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
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

@pytest.mark.asyncio
async def test_phase_4_run_filters_tasks_by_sites_before_rebase(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    (tmp_path / "phase_0c").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "adv-gitlab",
                    "benign_task_id": "benign-gitlab",
                    "site": "gitlab",
                    "sites": ["gitlab"],
                    "adversarial_action": {"kind": "create_issue"},
                    "instruction": "Open issue",
                    "start_urls": ["http://gitlab.test/issues"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {"adversarial_reward": {"type": "noop"}},
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
                    **_phase4_exposure_fields(benign_task_id="benign-gitlab"),
                },
                {
                    "id": "adv-shopping",
                    "benign_task_id": "benign-shopping",
                    "site": "shopping",
                    "sites": ["shopping"],
                    "adversarial_action": {"kind": "create_post"},
                    "instruction": "Open order",
                    "start_urls": ["http://shopping.test/orders"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {"adversarial_reward": {"type": "noop"}},
                    "adversarial_data_seed": {
                        "mechanism": "api",
                        "api_calls": [{"method": "POST", "path": "/api/seed", "body": {"x": 2}}],
                    },
                    **_phase4_exposure_fields(
                        benign_task_id="benign-shopping",
                        site="shopping",
                        url="http://shopping.test/orders",
                    ),
                },
            ]
        )
    )
    (tmp_path / "phase_3" / "contracts.json").write_text(
        json.dumps(
            _as_contracts(
                [
                    {
                        "id": "benign-gitlab",
                        "site": "gitlab",
                        "sites": ["gitlab"],
                        "instruction": "Open issue",
                        "start_urls": ["http://gitlab.test/issues"],
                        "data_seed": {"mechanism": "none"},
                        "reward_function": {"type": "noop"},
                    }
                ]
            )
        )
    )
    (tmp_path / "phase_0c" / "BENCHMARK_PROFILE_gitlab.json").write_text(
        json.dumps(
            {
                "site_name": "gitlab",
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
                "benchmark_name": "WebArena Verified",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                        "reset_endpoint": "http://gitlab.test/init",
                        "pvpo_cdp_url": "http://127.0.0.1:9222",
                    }
                ],
            }
        )
    )
    captured: dict[str, object] = {}

    async def fake_run_tasks_by_site(**kwargs):
        captured["tasks"] = kwargs["tasks"]
        captured["max_workers"] = kwargs.get("max_workers")
        return []

    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "make_agent_factory", lambda **kwargs: lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "run_tasks_by_site", fake_run_tasks_by_site)

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            benchmark=tmp_path,
            agent_model="demo-model",
            agent_provider=None,
            max_tasks_per_site=1,
            sites="gitlab",
            allow_unknown_auth=True,
            phase_4_max_workers=2,
            phase_4_variant_budget="smoke-3-probe",
            adversarial_action_kind="create_issue",
            resume=False,
        )
    )

    assert rc == 0
    tasks = captured["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) == 1
    assert tasks[0]["site"] == "gitlab"
    assert tasks[0]["adversarial_action"]["kind"] == "create_issue"
    assert captured["max_workers"] == 2


@pytest.mark.asyncio
async def test_phase_4_run_writes_failed_progress_on_postprocess_exception(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    (tmp_path / "phase_0c").mkdir(parents=True)
    adv_task = {
        "id": "adv-gitlab",
        "benign_task_id": "benign-gitlab",
        "site": "gitlab",
        "sites": ["gitlab"],
        "instruction": "Open issue",
        "start_urls": ["http://gitlab.test/issues"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"adversarial_reward": {"type": "noop"}},
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
        **_phase4_exposure_fields(benign_task_id="benign-gitlab"),
    }
    benign_task = {
        "id": "benign-gitlab",
        "site": "gitlab",
        "sites": ["gitlab"],
        "instruction": "Open issue",
        "start_urls": ["http://gitlab.test/issues"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"type": "noop"},
    }
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(json.dumps([adv_task]))
    (tmp_path / "phase_3" / "contracts.json").write_text(json.dumps(_as_contracts([benign_task])))
    (tmp_path / "phase_0c" / "BENCHMARK_PROFILE_gitlab.json").write_text(
        json.dumps(
            {
                "site_name": "gitlab",
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
                "benchmark_name": "WebArena Verified",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                        "reset_endpoint": "http://gitlab.test/init",
                        "pvpo_cdp_url": "http://127.0.0.1:9222",
                    }
                ],
            }
        )
    )

    async def fake_run_tasks_by_site(**kwargs):
        callback = kwargs.get("result_callback")
        result = {"task_id": "adv-gitlab", "final_status": "resistant"}
        if callback is not None:
            await callback(result)
        return [result]

    async def fail_postprocess(**kwargs):
        _ = kwargs
        raise RuntimeError("postprocess exploded")

    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "make_agent_factory", lambda **kwargs: lambda: None)
    monkeypatch.setattr(
        phase_4_adversarial,
        "_collect_agent_auth_runtime_errors",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        phase_4_adversarial,
        "_probe_seed_base_state_for_task_targets",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(phase_4_adversarial, "run_tasks_by_site", fake_run_tasks_by_site)
    monkeypatch.setattr(phase_4_adversarial, "_postprocess_one_task", fail_postprocess)

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            benchmark=tmp_path,
            agent_model="demo-model",
            agent_provider=None,
            max_tasks_per_site=None,
            sites="gitlab",
            allow_unknown_auth=True,
            phase_4_max_workers=2,
            resume=False,
        )
    )

    assert rc == 1
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["reason"] == "postprocess_exception"
    assert state["failed_tasks"] == ["adv-gitlab"]
    progress = json.loads((tmp_path / "phase_4" / "progress.json").read_text())
    assert progress["schema_version"] == 1
    assert progress["status"] == "failed"
    assert progress["stage"] == "postprocess_exception"
    assert progress["completed_initial_tasks"] == 1
    assert progress["postprocessed_tasks"] == 0
    assert progress["postprocess_attempted_tasks"] == 1
    assert progress["postprocess_failed_tasks"] == 1
    assert progress["phase_4_max_workers"] == 2


@pytest.mark.asyncio
async def test_phase_4_run_rejects_unknown_sites_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "adv-shopping",
                    "benign_task_id": "benign-shopping",
                    "site": "shopping",
                    "sites": ["shopping"],
                    "instruction": "Open order",
                    "start_urls": ["http://shopping.test/orders"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {"adversarial_reward": {"type": "noop"}},
                    "adversarial_data_seed": {"mechanism": "none"},
                }
            ]
        )
    )

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=tmp_path / "missing.json",
            benchmark=tmp_path,
            agent_model="demo-model",
            agent_provider=None,
            max_tasks_per_site=None,
            sites="gitlab",
            allow_unknown_auth=True,
            resume=False,
        )
    )

    assert rc == 1
