from __future__ import annotations

from pathlib import Path

import pytest

from worldsim.config import BenchmarkInstance
from worldsim.eval_worker_pool import run_eval


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
