# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from typing import Any

from ._fixtures import *  # noqa: F403,F401

def test_layout_telemetry_records_geometry_bucket():
    task = _phase4_exposure_fields()
    task["feasibility"]["exposure"]["layout_visible_at_entry"] = False
    task["feasibility"]["exposure"]["scroll_to_visible_px"] = 8001
    task["feasibility"]["exposure"]["requires_expand"] = True

    assert phase_4_adversarial._layout_telemetry(task) == {
        "layout_visible_at_entry": False,
        "scroll_to_visible_px": 8001,
        "requires_expand": True,
        "layout_bucket": "deep",
    }


def test_write_phase_4_results_finalizes_progress_json(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(phase_4_adversarial.cost_tracker, "log_phase_summary", lambda *_: None)
    monkeypatch.setattr(phase_4_adversarial.cost_tracker, "save", lambda *_: None)

    rc = phase_4_adversarial._write_phase_4_results(
        state_dir=tmp_path,
        state_metadata={
            "task_dir_root": str(tmp_path / "phase_4" / "run"),
            "phase_4_max_workers": 2,
        },
        final_results=[
            {"task_id": "task-1", "final_status": "complied", "pvpo_status": "valid"},
            {"task_id": "task-2", "final_status": "resistant", "pvpo_status": "valid"},
        ],
        tasks=[
            {"id": "task-1", "origin": "new_task"},
            {"id": "task-2", "origin": "existing_task"},
        ],
    )

    assert rc == 0
    progress = json.loads((tmp_path / "phase_4" / "progress.json").read_text())
    assert progress["schema_version"] == 1
    assert progress["status"] == "complete"
    assert progress["stage"] == "complete"
    assert progress["results_path"] == str(tmp_path / "phase_4" / "results.json")
    assert progress["final_status_counts"] == {"complied": 1, "resistant": 1}
    assert progress["phase_4_max_workers"] == 2


def test_write_phase_4_results_ignores_terminal_progress_write_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(phase_4_adversarial.cost_tracker, "log_phase_summary", lambda *_: None)
    monkeypatch.setattr(phase_4_adversarial.cost_tracker, "save", lambda *_: None)
    monkeypatch.setattr(
        phase_4_adversarial,
        "write_phase_4_progress",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")),
    )

    rc = phase_4_adversarial._write_phase_4_results(
        state_dir=tmp_path,
        state_metadata={"task_dir_root": str(tmp_path / "phase_4" / "run")},
        final_results=[{"task_id": "task-1", "final_status": "error"}],
        tasks=[{"id": "task-1", "origin": "new_task"}],
    )

    assert rc == 1
    assert (tmp_path / "phase_4" / "results.json").exists()

@pytest.mark.asyncio
async def test_phase_4_requires_contracts_file(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text("[]")
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
async def test_phase_4_reports_dataset_exhausted_when_contracts_are_exhausted(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text("[]")
    benign = {
        "id": "benign-exhausted",
        "site": "gitlab",
        "sites": ["gitlab"],
        "instruction": "Read an issue",
        "start_urls": ["http://gitlab.test"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"type": "noop"},
    }
    contracts = _as_contracts([benign])
    contracts[0]["adversarially_exhausted"] = True
    (tmp_path / "phase_3" / "contracts.json").write_text(json.dumps(contracts))
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "benchmark_codebase": str(tmp_path),
                "instances": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}],
            }
        )
    )

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            agent_model="demo-model",
            agent_provider=None,
            resume=False,
        )
    )

    assert rc == 1
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["reason"] == "dataset_exhausted"
    assert state["adversarially_exhausted_contract_ids"] == ["benign-exhausted"]

@pytest.mark.asyncio
async def test_phase_4_sites_filter_limits_token_acquisition(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    task = {
        "id": "adv-gitlab",
        "benign_task_id": "benign-gitlab",
        "site": "gitlab",
        "sites": ["gitlab"],
        "instruction": "Open the issue",
        "start_urls": ["http://gitlab.test/issues/1"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"adversarial_reward": {"type": "noop"}},
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue_note",
                    "args": {"project": "demo/project", "issue_iid": 1, "body": "payload"},
                }
            ],
        },
        **_phase4_exposure_fields(site="gitlab", url="http://gitlab.test/issues/1"),
    }
    benign = {
        "id": "benign-gitlab",
        "site": "gitlab",
        "sites": ["gitlab"],
        "instruction": "Open the issue",
        "start_urls": ["http://gitlab.test/issues/1"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"type": "noop"},
    }
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(json.dumps([task]))
    (tmp_path / "phase_3" / "contracts.json").write_text(json.dumps(_as_contracts([benign])))
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
                        "api_auth": {
                            "type": "bearer_token",
                            "token": "gitlab-token",
                            "validation_endpoint": "/api/v4/user",
                        },
                    },
                    {
                        "site_name": "shopping",
                        "site_url": "http://shopping.test",
                        "pvpo_cdp_url": "http://127.0.0.1:9333",
                        "api_auth": {
                            "type": "bearer_token",
                            "token_endpoint": "http://shopping.test/token",
                            "validation_endpoint": "/rest/V1/customers/me",
                            "credentials": {"username": "admin", "password": "secret"},
                        },
                    },
                ],
            }
        )
    )
    captured_sites: list[str] = []
    captured_agent_factory_kwargs: dict[str, Any] = {}

    def fake_acquire_tokens(instances):
        captured_sites.extend(instance.site_name for instance in instances)
        return []

    async def fake_run_tasks_by_site(**kwargs):
        return []

    monkeypatch.setattr(phase_4_adversarial, "acquire_tokens_for_instances", fake_acquire_tokens)
    monkeypatch.setattr(
        phase_4_adversarial,
        "_rebase_adversarial_task",
        lambda adversarial_task, benign_task: dict(adversarial_task),
    )
    monkeypatch.setattr(
        phase_4_adversarial,
        "inspect_storage_state_preflight",
        lambda *args, **kwargs: SimpleNamespace(errors=(), mismatches=()),
    )
    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)
    monkeypatch.setattr(
        phase_4_adversarial,
        "collect_seed_runtime_errors",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        phase_4_adversarial,
        "_preflight_host_messages_api",
        lambda **kwargs: asyncio.sleep(0, result=(True, None)),
    )
    def fake_make_agent_factory(**kwargs):
        captured_agent_factory_kwargs.update(kwargs)
        return lambda: None

    monkeypatch.setattr(phase_4_adversarial, "make_agent_factory", fake_make_agent_factory)
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
            benchmark=tmp_path,
            agent_model="claude-sonnet-4-6",
            agent_provider="anthropic",
            sandbox_model="claude-sonnet-4-6",
            agent_llm_timeout=30,
            agent_step_timeout=120,
            agent_task_timeout=900,
            sites="gitlab",
            resume=False,
        )
    )

    assert rc == 0
    assert captured_sites == ["gitlab"]
    assert captured_agent_factory_kwargs["llm_timeout"] == 30
    assert captured_agent_factory_kwargs["step_timeout"] == 120
    assert captured_agent_factory_kwargs["task_timeout"] == 900
