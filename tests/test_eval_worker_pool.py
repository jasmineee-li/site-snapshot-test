from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from worldsim.config import BenchmarkInstance
from worldsim.eval_worker_pool import _normalize_completed_result, load_completed_results, run_eval
from worldsim.resume_metadata import RESULT_FINGERPRINT_KEY
from worldsim.task_paths import safe_task_path_component


class _NoopAgent:
    def __init__(self, fail_setup: bool = False) -> None:
        self._fail_setup = fail_setup

    async def setup(self, server_url: str) -> None:
        if self._fail_setup:
            raise RuntimeError("setup failed")

    async def teardown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_run_eval_marks_all_tasks_failed_when_setup_fails(monkeypatch, tmp_path):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    results = await run_eval(
        tasks=[{"id": "task-a"}, {"id": "task-b"}],
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        agent_factory=lambda: _NoopAgent(fail_setup=True),
        task_runner=lambda *args, **kwargs: None,
        task_dir_root=tmp_path,
    )

    assert {result["task_id"] for result in results} == {"task-a", "task-b"}
    assert all(result["passed"] is False for result in results)


@pytest.mark.asyncio
async def test_run_eval_sanitizes_task_directory_names(monkeypatch, tmp_path):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)
    seen_task_dirs: list[Path] = []

    async def task_runner(task, agent, instance, task_dir):
        seen_task_dirs.append(task_dir)
        return {"task_id": task["id"], "passed": True}

    await run_eval(
        tasks=[{"id": "../escape/me"}],
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        agent_factory=lambda: _NoopAgent(),
        task_runner=task_runner,
        task_dir_root=tmp_path,
    )

    assert len(seen_task_dirs) == 1
    assert seen_task_dirs[0].resolve().parent == tmp_path.resolve()
    assert ".." not in seen_task_dirs[0].name


@pytest.mark.asyncio
async def test_run_eval_calls_result_callback_for_completed_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)
    callback_results: list[str] = []

    async def task_runner(task, agent, instance, task_dir):
        return {"task_id": task["id"], "passed": True}

    async def result_callback(result: dict[str, object]) -> None:
        callback_results.append(str(result["task_id"]))

    results = await run_eval(
        tasks=[{"id": "task-a"}, {"id": "task-b"}],
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        agent_factory=lambda: _NoopAgent(),
        task_runner=task_runner,
        task_dir_root=tmp_path,
        result_callback=result_callback,
    )

    assert {result["task_id"] for result in results} == {"task-a", "task-b"}
    assert sorted(callback_results) == ["task-a", "task-b"]


@pytest.mark.asyncio
async def test_run_eval_deterministically_routes_tasks_despite_setup_race(monkeypatch, tmp_path):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-0.test",
            replica_index=0,
            replica_name="shopping_0",
        ),
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-1.test",
            replica_index=1,
            replica_name="shopping_1",
        ),
    ]
    tasks = [{"id": f"task-{index}"} for index in range(8)]

    async def run_with_setup_delays(delays: dict[str, float]) -> dict[str, str]:
        seen: dict[str, str] = {}

        class _DelayedAgent:
            async def setup(self, server_url: str) -> None:
                await asyncio.sleep(delays.get(server_url, 0))

            async def teardown(self) -> None:
                return None

        async def task_runner(task, agent, instance, task_dir):
            seen[task["id"]] = instance.site_url
            return {"task_id": task["id"], "passed": True}

        results = await run_eval(
            tasks=tasks,
            instances=instances,
            agent_factory=lambda: _DelayedAgent(),
            task_runner=task_runner,
            task_dir_root=tmp_path,
        )

        assert len(results) == len(tasks)
        return seen

    first = await run_with_setup_delays(
        {
            "http://shopping-0.test": 0.05,
            "http://shopping-1.test": 0,
        }
    )
    second = await run_with_setup_delays(
        {
            "http://shopping-0.test": 0,
            "http://shopping-1.test": 0.05,
        }
    )

    assert first == second
    assert {site_url for site_url in first.values()} == {
        "http://shopping-0.test",
        "http://shopping-1.test",
    }


@pytest.mark.asyncio
async def test_run_eval_global_max_workers_caps_active_agents(monkeypatch, tmp_path):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)
    monkeypatch.setattr(
        "worldsim.eval_worker_pool._worker_index_for_task",
        lambda task, *, replica_count, site_name: int(task["worker"]),
    )

    active_agents = 0
    max_active_agents = 0
    lock = asyncio.Lock()

    class _CountingAgent:
        async def setup(self, server_url: str) -> None:
            nonlocal active_agents, max_active_agents
            async with lock:
                active_agents += 1
                max_active_agents = max(max_active_agents, active_agents)

        async def teardown(self) -> None:
            nonlocal active_agents
            async with lock:
                active_agents -= 1

    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url=f"http://shopping-{index}.test",
            replica_index=index,
            replica_name=f"shopping_{index}",
        )
        for index in range(3)
    ]
    tasks = [{"id": f"task-{index}", "worker": index} for index in range(3)]
    seen_instances: set[str] = set()

    async def task_runner(task, agent, instance, task_dir):
        seen_instances.add(instance.site_url)
        await asyncio.sleep(0.01)
        return {"task_id": task["id"], "passed": True}

    results = await run_eval(
        tasks=tasks,
        instances=instances,
        agent_factory=lambda: _CountingAgent(),
        task_runner=task_runner,
        task_dir_root=tmp_path,
        max_workers=1,
    )

    assert len(results) == 3
    assert max_active_agents == 1
    assert seen_instances == {instance.site_url for instance in instances}


@pytest.mark.asyncio
async def test_run_eval_resume_keeps_task_on_same_replica(monkeypatch, tmp_path):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-0.test",
            replica_index=0,
            replica_name="shopping_0",
        ),
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-1.test",
            replica_index=1,
            replica_name="shopping_1",
        ),
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-2.test",
            replica_index=2,
            replica_name="shopping_2",
        ),
    ]
    tasks = [{"id": f"task-{index}"} for index in range(5)]

    initial_seen: dict[str, str] = {}

    async def initial_runner(task, agent, instance, task_dir):
        initial_seen[task["id"]] = instance.site_url
        return {"task_id": task["id"], "passed": True}

    await run_eval(
        tasks=tasks,
        instances=instances,
        agent_factory=lambda: _NoopAgent(),
        task_runner=initial_runner,
        task_dir_root=tmp_path / "initial",
    )

    remaining_task_id = "task-4"
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    for task in tasks:
        if task["id"] == remaining_task_id:
            continue
        task_dir = resume_dir / safe_task_path_component(task["id"])
        task_dir.mkdir()
        (task_dir / "result.json").write_text(
            json.dumps({"task_id": task["id"], "passed": True, "message": "prior"})
        )

    resumed_seen: dict[str, str] = {}

    async def resumed_runner(task, agent, instance, task_dir):
        resumed_seen[task["id"]] = instance.site_url
        return {"task_id": task["id"], "passed": True}

    await run_eval(
        tasks=tasks,
        instances=instances,
        agent_factory=lambda: _NoopAgent(),
        task_runner=resumed_runner,
        task_dir_root=resume_dir,
        resume=True,
    )

    assert resumed_seen == {remaining_task_id: initial_seen[remaining_task_id]}


@pytest.mark.asyncio
async def test_run_eval_without_instances_returns_soft_failures(monkeypatch, tmp_path):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    results = await run_eval(
        tasks=[{"id": "task-a"}, {"id": "task-b"}],
        instances=[],
        agent_factory=lambda: _NoopAgent(),
        task_runner=lambda *args, **kwargs: None,
        task_dir_root=tmp_path,
    )

    assert {result["task_id"] for result in results} == {"task-a", "task-b"}
    assert all(result["outcome"] == "error" for result in results)
    assert all("no benchmark instances configured" in result["message"] for result in results)


@pytest.mark.asyncio
async def test_run_eval_with_mixed_site_instances_returns_soft_failures(monkeypatch, tmp_path):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    results = await run_eval(
        tasks=[{"id": "task-a"}],
        instances=[
            BenchmarkInstance(site_name="shopping", site_url="http://shopping.test"),
            BenchmarkInstance(site_name="gitlab", site_url="http://gitlab.test"),
        ],
        agent_factory=lambda: _NoopAgent(),
        task_runner=lambda *args, **kwargs: None,
        task_dir_root=tmp_path,
    )

    assert len(results) == 1
    assert results[0]["task_id"] == "task-a"
    assert results[0]["outcome"] == "error"
    assert "same-site instances" in results[0]["message"]


# ── Per-task resume tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_eval_resume_skips_completed_and_merges(monkeypatch, tmp_path):
    """resume=True skips tasks with existing result.json and merges prior + new results."""
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    # Pre-populate a completed result for task-a
    task_a_dir = tmp_path / safe_task_path_component("task-a")
    task_a_dir.mkdir()
    (task_a_dir / "result.json").write_text(
        json.dumps({"task_id": "task-a", "passed": True, "message": "prior run"})
    )

    executed_ids: list[str] = []

    async def task_runner(task, agent, instance, task_dir):
        executed_ids.append(task["id"])
        return {"task_id": task["id"], "passed": True, "message": "new run"}

    results = await run_eval(
        tasks=[{"id": "task-a"}, {"id": "task-b"}],
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        agent_factory=lambda: _NoopAgent(),
        task_runner=task_runner,
        task_dir_root=tmp_path,
        resume=True,
    )

    # task-a was skipped (not executed); task-b was executed
    assert executed_ids == ["task-b"]
    result_ids = {r["task_id"] for r in results}
    assert result_ids == {"task-a", "task-b"}
    prior = next(r for r in results if r["task_id"] == "task-a")
    assert prior["message"] == "prior run"


def test_normalize_completed_result_preserves_phase_4_passed_and_error_fields(tmp_path):
    normalized = _normalize_completed_result(
        {
            "task_id": "adv-1",
            "outcome": "seed_preflight_mismatch",
            "passed": False,
            "error": "missing required args",
            "message": "seed preflight mismatch",
        },
        trajectory_dir=tmp_path / "adv-1",
    )

    assert normalized["benign_passed"] is False
    assert normalized["adversarial_passed"] is False
    assert normalized["error"] == "missing required args"


def test_load_completed_results_reuses_terminal_phase_4_result_without_sidecars(tmp_path):
    task_dir = tmp_path / "adv-1"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "adv-1",
                "outcome": "complied",
                "passed": False,
                "encounter": {"max_coverage": 0.5},
            }
        )
    )

    completed = load_completed_results(tmp_path)

    assert completed["adv-1"]["outcome"] == "complied"
    assert completed["adv-1"]["benign_passed"] is False


def test_load_completed_results_requires_only_history_for_reprocessable_phase_4_result(tmp_path):
    task_dir = tmp_path / "adv-1"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "adv-1",
                "outcome": "refused_or_ignored",
                "passed": True,
                "encounter": {"max_coverage": 0.5},
            }
        )
    )

    assert load_completed_results(tmp_path) == {}

    (task_dir / "history.json").write_text(json.dumps([{"step": 1}]))
    completed = load_completed_results(tmp_path)

    assert completed["adv-1"]["outcome"] == "refused_or_ignored"


@pytest.mark.asyncio
async def test_run_eval_resume_all_completed_returns_prior_only(monkeypatch, tmp_path):
    """resume=True with all tasks completed returns prior results without starting workers."""
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    for tid in ("task-a", "task-b"):
        d = tmp_path / safe_task_path_component(tid)
        d.mkdir()
        (d / "result.json").write_text(
            json.dumps(
                {
                    "task_id": tid,
                    "passed": True,
                    "message": "done",
                    RESULT_FINGERPRINT_KEY: f"fp-{tid}",
                }
            )
        )

    factory_called = False

    def agent_factory():
        nonlocal factory_called
        factory_called = True
        return _NoopAgent()

    results = await run_eval(
        tasks=[{"id": "task-a"}, {"id": "task-b"}],
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        agent_factory=agent_factory,
        task_runner=lambda *a, **kw: None,
        task_dir_root=tmp_path,
        resume=True,
        expected_result_fingerprints={"task-a": "fp-task-a", "task-b": "fp-task-b"},
    )

    assert not factory_called
    assert len(results) == 2
    assert {r["task_id"] for r in results} == {"task-a", "task-b"}


@pytest.mark.asyncio
async def test_run_eval_resume_empty_dir_behaves_like_fresh_run(monkeypatch, tmp_path):
    """resume=True with empty task_dir_root runs all tasks normally."""
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    executed_ids: list[str] = []

    async def task_runner(task, agent, instance, task_dir):
        executed_ids.append(task["id"])
        return {"task_id": task["id"], "passed": True, "message": "ran"}

    results = await run_eval(
        tasks=[{"id": "task-x"}, {"id": "task-y"}],
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        agent_factory=lambda: _NoopAgent(),
        task_runner=task_runner,
        task_dir_root=tmp_path,
        resume=True,
    )

    assert set(executed_ids) == {"task-x", "task-y"}
    assert len(results) == 2


@pytest.mark.asyncio
async def test_run_eval_resume_nonexistent_dir_behaves_like_fresh_run(monkeypatch, tmp_path):
    """resume=True with nonexistent task_dir_root runs all tasks normally."""
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    executed_ids: list[str] = []

    async def task_runner(task, agent, instance, task_dir):
        executed_ids.append(task["id"])
        return {"task_id": task["id"], "passed": True, "message": "ran"}

    results = await run_eval(
        tasks=[{"id": "task-z"}],
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        agent_factory=lambda: _NoopAgent(),
        task_runner=task_runner,
        task_dir_root=tmp_path / "does_not_exist",
        resume=True,
    )

    assert executed_ids == ["task-z"]
    assert len(results) == 1


@pytest.mark.asyncio
async def test_run_eval_resume_reruns_task_when_result_fingerprint_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    task_dir = tmp_path / safe_task_path_component("task-a")
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps({"task_id": "task-a", "passed": True, "message": "stale"})
    )

    executed_ids: list[str] = []

    async def task_runner(task, agent, instance, task_dir):
        executed_ids.append(task["id"])
        return {"task_id": task["id"], "passed": True, "message": "reran"}

    results = await run_eval(
        tasks=[{"id": "task-a"}],
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        agent_factory=lambda: _NoopAgent(),
        task_runner=task_runner,
        task_dir_root=tmp_path,
        resume=True,
        expected_result_fingerprints={"task-a": "expected-fp"},
    )

    assert executed_ids == ["task-a"]
    assert results == [{"task_id": "task-a", "passed": True, "message": "reran"}]


@pytest.mark.asyncio
async def test_run_eval_resume_reruns_task_when_result_fingerprint_mismatches(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    task_dir = tmp_path / safe_task_path_component("task-a")
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "passed": True,
                "message": "stale",
                RESULT_FINGERPRINT_KEY: "stale-fp",
            }
        )
    )

    executed_ids: list[str] = []

    async def task_runner(task, agent, instance, task_dir):
        executed_ids.append(task["id"])
        return {"task_id": task["id"], "passed": True, "message": "reran"}

    results = await run_eval(
        tasks=[{"id": "task-a"}],
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        agent_factory=lambda: _NoopAgent(),
        task_runner=task_runner,
        task_dir_root=tmp_path,
        resume=True,
        expected_result_fingerprints={"task-a": "expected-fp"},
    )

    assert executed_ids == ["task-a"]
    assert results == [{"task_id": "task-a", "passed": True, "message": "reran"}]


@pytest.mark.asyncio
async def test_run_eval_resume_ignores_malformed_result_json(monkeypatch, tmp_path):
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    task_dir = tmp_path / safe_task_path_component("task-a")
    task_dir.mkdir()
    (task_dir / "result.json").write_text("{bad-json", encoding="utf-8")

    executed_ids: list[str] = []

    async def task_runner(task, agent, instance, task_dir):
        executed_ids.append(task["id"])
        return {"task_id": task["id"], "passed": True, "message": "reran"}

    results = await run_eval(
        tasks=[{"id": "task-a"}],
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        agent_factory=lambda: _NoopAgent(),
        task_runner=task_runner,
        task_dir_root=tmp_path,
        resume=True,
    )

    assert executed_ids == ["task-a"]
    assert results == [{"task_id": "task-a", "passed": True, "message": "reran"}]
