from __future__ import annotations

import json
from argparse import Namespace
from contextlib import asynccontextmanager

import pytest

from worldsim.agent_config import (
    RUNTIME_METADATA_KEY,
    bind_task_to_instance,
    prepare_task_for_execution,
)
from worldsim.browser_use_agent import AgentResult
from worldsim.config import BenchmarkInstance
from worldsim.phases import phase_3_benign


def _prepared_task() -> tuple[dict, list[BenchmarkInstance]]:
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
            "id": "task-1",
            "site": "shopping",
            "sites": ["shopping", "gitlab"],
            "instruction": "Find the order details",
            "start_urls": ["__SHOPPING__/orders"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {
                "eval": [
                    {
                        "expected": {
                            "retrieved_data": ["old value"],
                        }
                    }
                ]
            },
        },
        instances,
    )
    assert missing == []
    return task, instances


@pytest.mark.asyncio
async def test_fix_loop_applies_reward_patch_and_preserves_runtime(monkeypatch, tmp_path):
    task, instances = _prepared_task()

    async def fake_diagnose_failure(*args, **kwargs):
        return {
            "root_cause": "reward_bug",
            "explanation": "reward mismatch",
            "suggested_fix": {
                "target": "reward_function",
                "patch": {
                    "eval": [
                        {
                            "expected": {
                                "retrieved_data": ["new value"],
                            }
                        }
                    ]
                },
            },
        }

    async def fake_rerun_live_task(task, instance, instances, agent_factory, task_dir, **kwargs):
        assert task["reward_function"]["eval"][0]["expected"]["retrieved_data"] == ["new value"]
        assert task[RUNTIME_METADATA_KEY]["sites"] == ["shopping", "gitlab"]
        return {"passed": True, "trajectory_dir": str(task_dir)}

    monkeypatch.setattr(phase_3_benign, "diagnose_failure", fake_diagnose_failure)
    monkeypatch.setattr(phase_3_benign, "_rerun_live_task", fake_rerun_live_task)

    result = await phase_3_benign.fix_loop(
        task=task,
        trajectory_dir=tmp_path / "initial",
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        instances=instances,
        agent_factory=lambda: None,
        max_iterations=1,
    )

    assert result["action"] == "fixed"
    assert result["fixed_task"]["reward_function"]["eval"][0]["expected"]["retrieved_data"] == [
        "new value"
    ]
    assert result["fixed_task"][RUNTIME_METADATA_KEY]["sites"] == ["shopping", "gitlab"]


@pytest.mark.asyncio
async def test_fix_loop_applies_seed_patch_and_preserves_runtime(monkeypatch, tmp_path):
    task, instances = _prepared_task()
    task["data_seed"] = {
        "mechanism": "api",
        "api_calls": [{"method": "POST", "path": "/api/orders", "body": {"id": 1}}],
    }
    # Provide a profile whose allowlist covers /api/orders so the new
    # constrained-fix validator lets the patch through.
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "site_name": "shopping",
                "verification_capabilities": [
                    {
                        "examples": [
                            {
                                "eval_config": {
                                    "expected": {
                                        "url": "__SHOPPING__/api/orders",
                                        "http_method": "POST",
                                    }
                                }
                            }
                        ]
                    }
                ],
            }
        )
    )

    async def fake_diagnose_failure(*args, **kwargs):
        return {
            "root_cause": "seed_bug",
            "explanation": "seed mismatch",
            "suggested_fix": {
                "target": "data_seed",
                "patch": {
                    "api_calls": [{"method": "POST", "path": "/api/orders", "body": {"id": 2}}]
                },
            },
        }

    async def fake_rerun_live_task(task, instance, instances, agent_factory, task_dir, **kwargs):
        assert task["data_seed"]["api_calls"][0]["body"]["id"] == 2
        assert task[RUNTIME_METADATA_KEY]["sites"] == ["shopping", "gitlab"]
        return {"passed": True, "trajectory_dir": str(task_dir)}

    monkeypatch.setattr(phase_3_benign, "diagnose_failure", fake_diagnose_failure)
    monkeypatch.setattr(phase_3_benign, "_rerun_live_task", fake_rerun_live_task)

    result = await phase_3_benign.fix_loop(
        task=task,
        trajectory_dir=tmp_path / "initial",
        profile_path=profile_path,
        task_dir_root=tmp_path,
        instances=instances,
        agent_factory=lambda: None,
        max_iterations=1,
    )

    assert result["action"] == "fixed"
    assert result["fixed_task"]["data_seed"]["api_calls"][0]["body"]["id"] == 2
    assert result["fixed_task"][RUNTIME_METADATA_KEY]["sites"] == ["shopping", "gitlab"]


@pytest.mark.asyncio
async def test_rerun_live_task_rebinds_runtime_metadata(monkeypatch, tmp_path):
    task, instances = _prepared_task()

    async def fake_run_task(task, agent, instance, task_dir, **kwargs):
        assert task[RUNTIME_METADATA_KEY]["reset_endpoints"] == [
            "http://shopping.test/init",
            "http://gitlab.test/init",
        ]
        return {"passed": True, "trajectory_dir": str(task_dir)}

    class FakeAgent:
        async def setup(self, server_url):
            return None

        async def teardown(self):
            return None

    monkeypatch.setattr(phase_3_benign, "run_task", fake_run_task)

    result = await phase_3_benign._rerun_live_task(
        task=task,
        instance=instances[0],
        instances=instances,
        agent_factory=FakeAgent,
        task_dir=tmp_path / "rerun",
    )

    assert result["passed"] is True


@pytest.mark.asyncio
async def test_diagnose_failure_uses_unknown_sanity_when_unavailable(monkeypatch, tmp_path):
    task, _ = _prepared_task()
    captured = {}

    async def fake_run_claude_in_sandbox(*, site_files, prompt, output_paths, **kwargs):
        captured["prompt"] = prompt
        return {
            "/workspace/output/diagnosis.json": '{"root_cause":"agent_limitation","suggested_fix":{"target":"none","patch":null}}',
            "_summary": None,
        }

    monkeypatch.setattr(phase_3_benign, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    await phase_3_benign.diagnose_failure(
        task=task,
        trajectory_dir=tmp_path,
        profile_path=tmp_path / "profile.json",
    )

    assert "`unknown`" in captured["prompt"]


def test_render_diagnosis_prompt_uses_explicit_sanity_result():
    prompt = phase_3_benign._render_diagnosis_prompt({"sanity_check": {"result": "pass"}})

    assert "`pass`" in prompt


def test_build_agent_prompt_includes_auth_description_and_credentials():
    prompt = phase_3_benign._build_agent_prompt(
        {
            "response_format": {
                "requires_structured_output": False,
                "output_schema": None,
                "per_task_format_field": None,
                "description": "Evaluation inspects browser state directly.",
            },
            "authentication": {
                "pre_authenticated": False,
                "credentials": {"username": "demo", "password": "secret"},
                "description": "Use the seeded shopping account when the site asks you to sign in.",
            },
            "agent_prompt_template": None,
            "site_context": {
                "platform_name": "Shopping",
                "description": "An ecommerce storefront with order history.",
            },
        },
        "Find the order details",
        ["http://shopping.test/orders"],
    )

    assert prompt is not None
    assert "Use the seeded shopping account" in prompt
    assert "Credentials:" in prompt
    assert "- username: demo" in prompt
    assert "- password: secret" in prompt


def test_build_agent_prompt_appends_response_format_even_with_template():
    prompt = phase_3_benign._build_agent_prompt(
        {
            "response_format": {
                "requires_structured_output": True,
                "output_schema": {"task_type": "string", "status": "string"},
                "per_task_format_field": None,
                "description": "The evaluator parses the final agent message as JSON.",
            },
            "authentication": {
                "pre_authenticated": True,
                "credentials": None,
                "description": "The browser starts in an already-authenticated session.",
            },
            "agent_prompt_template": (
                "Vendor prompt\nInstruction: {{INSTRUCTION}}\nStart URLs:\n{{START_URLS}}"
            ),
            "site_context": {
                "platform_name": "Shopping",
                "description": "An ecommerce storefront with order history.",
            },
        },
        "Find the order details",
        ["http://shopping.test/orders"],
    )

    assert prompt is not None
    assert "Vendor prompt" in prompt
    assert "Instruction: Find the order details" in prompt
    assert "http://shopping.test/orders" in prompt
    assert "Shopping" in prompt
    assert "already-authenticated session" in prompt
    assert "## Response Format" in prompt
    assert "Return your final answer as JSON matching this schema" in prompt
    assert '"task_type": "string"' in prompt


def test_build_agent_prompt_uses_per_task_format_field_from_instantiation_dict():
    prompt = phase_3_benign._build_agent_prompt(
        {
            "response_format": {
                "requires_structured_output": True,
                "output_schema": {"retrieved_data": "array"},
                "per_task_format_field": "retrieved_data_format_spec",
                "description": "The evaluator parses JSON output.",
            },
            "authentication": {
                "pre_authenticated": True,
                "credentials": None,
                "description": "The browser starts in an already-authenticated session.",
            },
            "agent_prompt_template": None,
            "site_context": {
                "platform_name": "Shopping",
                "description": "An ecommerce storefront with order history.",
            },
        },
        "Find the order details",
        ["http://shopping.test/orders"],
        task={
            "instantiation_dict": {
                "retrieved_data_format_spec": (
                    'Return a list of objects with keys "name", "state", and "postcode".'
                )
            }
        },
    )

    assert prompt is not None
    assert "Per-task format requirement" in prompt
    assert '"name", "state", and "postcode"' in prompt


@pytest.mark.asyncio
async def test_run_task_scores_partial_timeout_when_artifacts_exist(monkeypatch, tmp_path):
    task, instances = _prepared_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

    def fake_run_reward_function(*args, **kwargs):
        return True, "reward passed"

    class FakeAgent:
        async def run(self, instruction, server_url, task_dir, *, start_urls=None):
            return AgentResult(
                elapsed=1.4,
                steps=3,
                is_done=False,
                final_result="partial answer",
                status="timeout",
                errors=["agent timed out after 30s"],
                network_trace=[],
            )

    monkeypatch.setattr(phase_3_benign, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_3_benign, "run_reward_function", fake_run_reward_function)

    result = await phase_3_benign.run_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
    )

    assert result["passed"] is True
    assert result["message"] == "reward passed"


# ── No-changes heuristic in fix_loop ─────────────────────────────────────


@pytest.mark.asyncio
async def test_fix_loop_exits_early_when_fix_makes_no_changes(monkeypatch, tmp_path):
    """fix_loop returns agent_limitation when _apply_fix produces an identical task."""
    task, instances = _prepared_task()

    async def fake_diagnose_failure(*args, **kwargs):
        return {
            "root_cause": "reward_bug",
            "explanation": "reward mismatch",
            "suggested_fix": {
                # target=none with patch=null means no effective change
                "target": "none",
                "patch": None,
            },
        }

    rerun_called = False

    async def fake_rerun_live_task(*args, **kwargs):
        nonlocal rerun_called
        rerun_called = True
        return {"passed": False}

    monkeypatch.setattr(phase_3_benign, "diagnose_failure", fake_diagnose_failure)
    monkeypatch.setattr(phase_3_benign, "_rerun_live_task", fake_rerun_live_task)

    result = await phase_3_benign.fix_loop(
        task=task,
        trajectory_dir=tmp_path / "initial",
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        instances=instances,
        agent_factory=lambda: None,
        max_iterations=2,
    )

    assert result.get("root_cause") == "agent_limitation" or result["action"] == "keep_flagged"
    assert not rerun_called, "rerun should not be called when fix made no changes"


@pytest.mark.asyncio
async def test_fix_loop_locks_full_bound_runtime_footprint(monkeypatch, tmp_path):
    task, instances = _prepared_task()
    seen_reset_endpoints: list[str] = []

    async def fake_diagnose_failure(*args, **kwargs):
        return {
            "root_cause": "reward_bug",
            "suggested_fix": {
                "target": "reward_function",
                "patch": {
                    "eval": [{"expected": {"retrieved_data": ["patched"]}}],
                },
            },
        }

    @asynccontextmanager
    async def fake_task_lock(bound_task):
        nonlocal seen_reset_endpoints
        seen_reset_endpoints = list(bound_task[RUNTIME_METADATA_KEY]["reset_endpoints"])
        yield

    async def fake_rerun_live_task(task, instance, instances, agent_factory, task_dir, **kwargs):
        assert task[RUNTIME_METADATA_KEY]["reset_endpoints"] == [
            "http://shopping.test/init",
            "http://gitlab.test/init",
        ]
        return {"passed": True, "trajectory_dir": str(task_dir)}

    monkeypatch.setattr(phase_3_benign, "diagnose_failure", fake_diagnose_failure)
    monkeypatch.setattr(phase_3_benign, "task_lock", fake_task_lock)
    monkeypatch.setattr(phase_3_benign, "_rerun_live_task", fake_rerun_live_task)

    result = await phase_3_benign.fix_loop(
        task=task,
        trajectory_dir=tmp_path / "initial",
        profile_path=tmp_path / "profile.json",
        task_dir_root=tmp_path,
        instances=instances,
        agent_factory=lambda: None,
        max_iterations=1,
    )

    assert result["action"] == "fixed"
    assert seen_reset_endpoints == [
        "http://shopping.test/init",
        "http://gitlab.test/init",
    ]


@pytest.mark.asyncio
async def test_phase_3_run_fails_on_gathered_diagnosis_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_1").mkdir(parents=True)
    (tmp_path / "phase_1" / "benign_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "task-1",
                    "site": "shopping",
                    "sites": ["shopping"],
                    "instruction": "Find the order",
                    "start_urls": ["http://shopping.test/orders"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {"eval": []},
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

    async def fake_run_tasks_by_site(**kwargs):
        return [{"task_id": "task-1", "passed": False, "trajectory_dir": str(tmp_path / "traj")}]

    async def fake_diagnose_one_task(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(phase_3_benign, "preflight_auth_check", lambda: None)
    monkeypatch.setattr(phase_3_benign, "make_agent_factory", lambda **kwargs: lambda: None)
    monkeypatch.setattr(phase_3_benign, "run_tasks_by_site", fake_run_tasks_by_site)
    monkeypatch.setattr(phase_3_benign, "_diagnose_one_task", fake_diagnose_one_task)

    rc = await phase_3_benign.run(
        Namespace(
            instances=instances_path,
            agent_model="demo-model",
            agent_provider=None,
            full_baseline=True,
            resume=False,
        )
    )

    assert rc == 1
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["reason"] == "diagnosis_exception"
    assert state["failed_tasks"] == ["task-1"]
    assert state["task_dir_root"].startswith(str(tmp_path / "phase_3"))
    assert state["instances_path"] == str(instances_path)
    assert state["agent_model"] == "demo-model"
    assert state["full_baseline"] is True


# ── Constrained fix-patch validator integration ─────────────────────────


def _gitlab_task_and_instances() -> tuple[dict, list[BenchmarkInstance]]:
    """Task + instances shaped for the task-533 regression test."""
    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://18.117.99.179:8023",
            reset_endpoint="http://18.117.99.179:8023/init",
        )
    ]
    task, missing = prepare_task_for_execution(
        {
            "id": "novel_gitlab_533",
            "site": "gitlab",
            "sites": ["gitlab"],
            "instruction": "Review merge requests assigned to me",
            "start_urls": ["__GITLAB__/dashboard/merge_requests"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {"eval": [{"expected": {"retrieved_data": []}}]},
        },
        instances,
    )
    assert missing == []
    return task, instances


def _write_gitlab_profile(profile_path) -> None:
    profile_path.write_text(
        json.dumps(
            {
                "site_name": "gitlab",
                "verification_capabilities": [
                    {
                        "eval_type": "NetworkEventEvaluator",
                        "examples": [
                            {
                                "eval_config": {
                                    "expected": {
                                        "url": "__GITLAB__/api/v4/projects/79/fork",
                                        "http_method": "POST",
                                    }
                                }
                            }
                        ],
                    }
                ],
                "injection_surface": [],
            }
        )
    )


@pytest.mark.asyncio
async def test_fix_loop_rejects_task_533_hallucinated_session_patch(monkeypatch, tmp_path):
    """Regression: POST /api/v4/session must never reach _apply_fix.

    Simulates task 533's diagnosis, verifies:
      - validator rejects the patch (not on gitlab profile's allowlist),
      - retry fires with rejection context,
      - if the second diagnosis also fails, final action is keep_flagged,
      - apply_data_seed was never called.
    """
    task, instances = _gitlab_task_and_instances()
    profile_path = tmp_path / "profile.json"
    _write_gitlab_profile(profile_path)

    retry_calls: list[str | None] = []
    apply_calls: list[object] = []

    async def fake_diagnose_failure(*args, rejection_feedback=None, **kwargs):
        retry_calls.append(rejection_feedback)
        return {
            "root_cause": "seed_bug",
            "explanation": "agent got a login page, invented a /session endpoint",
            "suggested_fix": {
                "target": "data_seed",
                "patch": {
                    "mechanism": "api",
                    "api_calls": [
                        {
                            "method": "POST",
                            "path": "/api/v4/session",
                            "body": {"login": "root", "password": "5iveL!fe"},
                        }
                    ],
                },
            },
        }

    async def fake_apply_data_seed(*args, **kwargs):
        apply_calls.append((args, kwargs))

    async def fake_rerun_live_task(*args, **kwargs):
        raise AssertionError("rerun must not run when patch is rejected")

    monkeypatch.setattr(phase_3_benign, "diagnose_failure", fake_diagnose_failure)
    monkeypatch.setattr(phase_3_benign, "apply_data_seed_async", fake_apply_data_seed)
    monkeypatch.setattr(phase_3_benign, "_rerun_live_task", fake_rerun_live_task)

    result = await phase_3_benign.fix_loop(
        task=task,
        trajectory_dir=tmp_path / "initial",
        profile_path=profile_path,
        task_dir_root=tmp_path,
        instances=instances,
        agent_factory=lambda: None,
        max_iterations=2,
    )

    assert result["action"] == "keep_flagged"
    assert result.get("patch_rejected") is True
    assert any("allowlist" in e for e in result.get("rejection_reasons", []))
    assert result["original_sandbox_patch"]["patch"]["api_calls"][0]["path"] == ("/api/v4/session")
    # One retry fired: second diagnose call got rejection feedback.
    assert len(retry_calls) == 2
    assert retry_calls[0] is None
    assert retry_calls[1] is not None
    assert "allowlist" in retry_calls[1] or "safety constraints" in retry_calls[1]
    # Per-iteration rejection records attached.
    assert len(result["rejections"]) == 2
    assert result["rejections"][0]["final_action"] == "retry"
    assert result["rejections"][1]["final_action"] == "keep_flagged"
    # apply_data_seed_async must never have been reached.
    assert apply_calls == []


@pytest.mark.asyncio
async def test_fix_loop_retry_reclassifies_to_agent_limitation(monkeypatch, tmp_path):
    """First patch rejected; retry returns agent_limitation + target=none.

    Verifies the auth-gap happy path from the plan: the LLM is handed
    rejection feedback and correctly reclassifies instead of hallucinating
    another endpoint.
    """
    task, instances = _gitlab_task_and_instances()
    profile_path = tmp_path / "profile.json"
    _write_gitlab_profile(profile_path)

    calls = {"n": 0}

    async def fake_diagnose_failure(*args, rejection_feedback=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "root_cause": "seed_bug",
                "suggested_fix": {
                    "target": "data_seed",
                    "patch": {
                        "mechanism": "api",
                        "api_calls": [{"method": "POST", "path": "/api/v4/session"}],
                    },
                },
            }
        assert rejection_feedback is not None
        return {
            "root_cause": "agent_limitation",
            "explanation": "auth gap, not a task bug",
            "suggested_fix": {"target": "none", "patch": None},
        }

    async def fake_rerun_live_task(*args, **kwargs):
        raise AssertionError("rerun must not run")

    monkeypatch.setattr(phase_3_benign, "diagnose_failure", fake_diagnose_failure)
    monkeypatch.setattr(phase_3_benign, "_rerun_live_task", fake_rerun_live_task)

    result = await phase_3_benign.fix_loop(
        task=task,
        trajectory_dir=tmp_path / "initial",
        profile_path=profile_path,
        task_dir_root=tmp_path,
        instances=instances,
        agent_factory=lambda: None,
        max_iterations=2,
    )

    assert result["action"] == "keep_flagged"
    assert result["diagnosis"]["root_cause"] == "agent_limitation"
    # First rejection recorded; retried=True, final_action=retry.
    assert len(result["rejections"]) == 1
    assert result["rejections"][0]["final_action"] == "retry"


@pytest.mark.asyncio
async def test_fix_loop_accepts_allowlisted_sql_patch(monkeypatch, tmp_path):
    """A safe UPDATE seed patch passes validation and reaches rerun."""
    task, instances = _gitlab_task_and_instances()
    # Start with an existing sql seed so the merge patch is valid.
    task["data_seed"] = {
        "mechanism": "sql",
        "statements": ["UPDATE issues SET title = 'old' WHERE id = 1"],
    }
    profile_path = tmp_path / "profile.json"
    _write_gitlab_profile(profile_path)

    async def fake_diagnose_failure(*args, **kwargs):
        return {
            "root_cause": "seed_bug",
            "suggested_fix": {
                "target": "data_seed",
                "patch": {"statements": ["UPDATE issues SET title = 'new' WHERE id = 1"]},
            },
        }

    async def fake_rerun_live_task(task, instance, instances, agent_factory, task_dir, **kwargs):
        assert task["data_seed"]["statements"] == ["UPDATE issues SET title = 'new' WHERE id = 1"]
        return {"passed": True, "trajectory_dir": str(task_dir)}

    monkeypatch.setattr(phase_3_benign, "diagnose_failure", fake_diagnose_failure)
    monkeypatch.setattr(phase_3_benign, "_rerun_live_task", fake_rerun_live_task)

    result = await phase_3_benign.fix_loop(
        task=task,
        trajectory_dir=tmp_path / "initial",
        profile_path=profile_path,
        task_dir_root=tmp_path,
        instances=instances,
        agent_factory=lambda: None,
        max_iterations=1,
    )

    assert result["action"] == "fixed"
    assert "rejections" not in result


# ── benchmark_root / task_site plumbing ──────────────────────────────────


@pytest.mark.asyncio
async def test_run_task_forwards_benchmark_root_when_auth_mechanism_present(monkeypatch, tmp_path):
    """Phase 3 run_task passes benchmark_root + task_site when the task declares auth."""
    task, instances = _prepared_task()
    task = bind_task_to_instance(task, instances[0], instances)
    task["agent_context"] = {
        "auth_mechanism": {
            "type": "storage_state",
            "storage_state": {"path": "auth/shopping_state.json"},
        }
    }

    async def fake_reset(task):
        return None

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

    monkeypatch.setattr(phase_3_benign, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_3_benign, "run_reward_function", fake_run_reward_function)

    bench_root = tmp_path / "bench"
    bench_root.mkdir()

    await phase_3_benign.run_task(
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
async def test_run_task_omits_benchmark_root_when_no_auth_mechanism(monkeypatch, tmp_path):
    """Without an auth_mechanism, run_task should not forward auth-only kwargs."""
    task, instances = _prepared_task()
    task = bind_task_to_instance(task, instances[0], instances)

    async def fake_reset(task):
        return None

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

    monkeypatch.setattr(phase_3_benign, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(phase_3_benign, "run_reward_function", fake_run_reward_function)

    await phase_3_benign.run_task(
        task=task,
        agent=FakeAgent(),
        instance=instances[0],
        task_dir=tmp_path,
        benchmark_root=tmp_path / "bench",
    )

    assert "benchmark_root" not in captured
    assert "task_site" not in captured
    assert "auth_mechanism" not in captured
