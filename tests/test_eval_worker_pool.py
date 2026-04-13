from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldsim.config import BenchmarkInstance
from worldsim.eval_worker_pool import run_eval
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


@pytest.mark.asyncio
async def test_run_eval_resume_all_completed_returns_prior_only(monkeypatch, tmp_path):
    """resume=True with all tasks completed returns prior results without starting workers."""
    monkeypatch.setattr("worldsim.eval_worker_pool.STAGGER_DELAY", 0)

    for tid in ("task-a", "task-b"):
        d = tmp_path / safe_task_path_component(tid)
        d.mkdir()
        (d / "result.json").write_text(
            json.dumps({"task_id": tid, "passed": True, "message": "done"})
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
