"""Managed Phase 4 restoration scope boundaries."""

from __future__ import annotations

import json
from typing import Any

import pytest

from warp_taskgen.agent_config import bind_task_to_instance
from warp_taskgen.config import BenchmarkInstance
from warp_taskgen.host_restoration import HostRestorationError
from warp_taskgen.phase_4 import execution as phase_4_execution

from ._fixtures import _prepared_adv_task


class _Scope:
    def __init__(self, *, restore_failure_at: int | None = None) -> None:
        self.restore_failure_at = restore_failure_at
        self.operation_ids: list[str] = []
        self.restore_calls: list[str] = []
        self.release_calls: list[str | None] = []

    def matches_instance(self, instance: object) -> bool:
        return True

    def operation_id(self) -> str:
        operation_id = f"phase4-op-{len(self.operation_ids) + 1}"
        self.operation_ids.append(operation_id)
        return operation_id

    async def restore(self, operation_id: str) -> dict[str, str]:
        self.restore_calls.append(operation_id)
        if self.restore_failure_at == len(self.restore_calls):
            raise HostRestorationError("final_restore_failed")
        return {"status": "restored", "operation_id": operation_id}

    async def release(self, *, operation_id: str | None = None) -> dict[str, str | None]:
        self.release_calls.append(operation_id)
        return {"status": "released", "operation_id": operation_id}


def _managed_task(tmp_path) -> tuple[dict[str, Any], BenchmarkInstance]:
    source_task, _ = _prepared_adv_task()
    # Keep this wrapper case single-site so the managed instance has one
    # concrete restoration scope.  The execution contract rejects a task that
    # mixes one managed bound instance with another site scope.
    source_task.pop("sites", None)
    source_task.pop("_worldsim_runtime", None)
    instance = BenchmarkInstance(
        site_name="shopping",
        site_url="http://shopping.test",
        restoration={
            "socket_path": str(tmp_path / "owner.sock"),
            "instance_id": "shopping-r1",
        },
    )
    return bind_task_to_instance(source_task, instance, [instance]), instance


def _scoreable_sentinel(task: dict[str, Any], task_dir) -> dict[str, Any]:
    return {
        "task_id": task["id"],
        "status": "complete",
        "passed": True,
        "is_done": True,
        "errors": [],
        "score": 0.75,
        "trajectory_dir": str(task_dir),
        "outcome": "complied",
    }


@pytest.mark.asyncio
async def test_final_restore_failure_keeps_completed_score_and_trajectory(
    monkeypatch, tmp_path
) -> None:
    task, instance = _managed_task(tmp_path)
    scope = _Scope(restore_failure_at=2)

    async def acquire_scope(*args: Any, **kwargs: Any) -> _Scope:
        return scope

    async def refresh_auth(*args: Any, **kwargs: Any) -> None:
        return None

    async def capture_baseline(*args: Any, **kwargs: Any) -> object:
        return object()

    async def verify_baseline(*args: Any, **kwargs: Any) -> dict[str, str]:
        return {"status": "verified"}

    async def fake_core(task, agent, instance, task_dir, **kwargs):
        sentinel = _scoreable_sentinel(task, task_dir)
        phase_4_execution.save_result_payload(task_dir, sentinel)
        return {"task_id": task["id"], "outcome": "complied", "score": 0.75}

    monkeypatch.setattr(phase_4_execution, "acquire_restoration_scope", acquire_scope)
    monkeypatch.setattr(phase_4_execution, "_run_adversarial_task_core", fake_core)
    monkeypatch.setattr(
        "warp_taskgen.storage_state_preflight.refresh_instance_auth", refresh_auth
    )
    monkeypatch.setattr(phase_4_execution, "capture_restoration_baseline_async", capture_baseline)
    monkeypatch.setattr(phase_4_execution, "verify_restoration_baseline_async", verify_baseline)

    with pytest.raises(HostRestorationError, match="final_restore_failed"):
        await phase_4_execution.run_adversarial_task(
            task=task,
            agent=object(),
            instance=instance,
            task_dir=tmp_path / "trajectory",
        )

    saved = json.loads((tmp_path / "trajectory" / "result.json").read_text())
    assert saved["status"] == "error"
    assert saved["restoration_status"] == "unknown"
    assert saved["restoration_failure"] == "final_restore_failed"
    assert saved["score"] == 0.75
    assert saved["trajectory_dir"] == str(tmp_path / "trajectory")
    assert scope.restore_calls == ["phase4-op-1", "phase4-op-2"]
    assert scope.release_calls == ["phase4-op-2"]


@pytest.mark.asyncio
async def test_managed_auth_failure_does_not_run_core_or_create_scoreable_sentinel(
    monkeypatch, tmp_path
) -> None:
    task, instance = _managed_task(tmp_path)
    scope = _Scope()
    core_called = False

    async def acquire_scope(*args: Any, **kwargs: Any) -> _Scope:
        return scope

    async def fail_refresh(*args: Any, **kwargs: Any) -> None:
        raise HostRestorationError("auth_refresh_failed")

    async def fake_core(*args: Any, **kwargs: Any):
        nonlocal core_called
        core_called = True
        raise AssertionError("core must not run after auth refresh failure")

    monkeypatch.setattr(phase_4_execution, "acquire_restoration_scope", acquire_scope)
    monkeypatch.setattr(phase_4_execution, "_run_adversarial_task_core", fake_core)
    monkeypatch.setattr(
        "warp_taskgen.storage_state_preflight.refresh_instance_auth", fail_refresh
    )

    with pytest.raises(HostRestorationError, match="auth_refresh_failed"):
        await phase_4_execution.run_adversarial_task(
            task=task,
            agent=object(),
            instance=instance,
            task_dir=tmp_path / "trajectory",
        )

    saved = json.loads((tmp_path / "trajectory" / "result.json").read_text())
    assert core_called is False
    assert saved["status"] == "error"
    assert saved["restoration_status"] == "unknown"
    assert "score" not in saved
    assert scope.restore_calls == ["phase4-op-1", "phase4-op-2"]
    assert scope.release_calls == ["phase4-op-2"]


@pytest.mark.asyncio
async def test_managed_success_preserves_existing_sentinel_fields_and_operation_order(
    monkeypatch, tmp_path
) -> None:
    task, instance = _managed_task(tmp_path)
    scope = _Scope()

    async def acquire_scope(*args: Any, **kwargs: Any) -> _Scope:
        return scope

    async def refresh_auth(*args: Any, **kwargs: Any) -> None:
        return None

    async def capture_baseline(*args: Any, **kwargs: Any) -> object:
        return object()

    async def verify_baseline(*args: Any, **kwargs: Any) -> dict[str, str]:
        return {"status": "verified"}

    async def fake_core(task, agent, instance, task_dir, **kwargs):
        phase_4_execution.save_result_payload(task_dir, _scoreable_sentinel(task, task_dir))
        return {"task_id": task["id"], "outcome": "complied"}

    monkeypatch.setattr(phase_4_execution, "acquire_restoration_scope", acquire_scope)
    monkeypatch.setattr(phase_4_execution, "_run_adversarial_task_core", fake_core)
    monkeypatch.setattr(
        "warp_taskgen.storage_state_preflight.refresh_instance_auth", refresh_auth
    )
    monkeypatch.setattr(phase_4_execution, "capture_restoration_baseline_async", capture_baseline)
    monkeypatch.setattr(phase_4_execution, "verify_restoration_baseline_async", verify_baseline)

    result = await phase_4_execution.run_adversarial_task(
        task=task,
        agent=object(),
        instance=instance,
        task_dir=tmp_path / "trajectory",
    )

    saved = json.loads((tmp_path / "trajectory" / "result.json").read_text())
    assert result["restoration_readback"] == {"status": "verified"}
    assert saved["status"] == "complete"
    assert saved["passed"] is True
    assert saved["is_done"] is True
    assert saved["score"] == 0.75
    assert saved["trajectory_dir"] == str(tmp_path / "trajectory")
    assert scope.restore_calls == ["phase4-op-1", "phase4-op-2"]
    assert scope.restore_calls[0] != scope.restore_calls[1]
    assert scope.release_calls == ["phase4-op-2"]
