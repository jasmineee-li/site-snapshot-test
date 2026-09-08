"""Managed Phase 2c restoration scope boundaries."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from warp_taskgen.host_restoration import HostRestorationError
from warp_taskgen.phase_2.phase_2c import runner as feasibility_runner
from warp_taskgen.phase_2.phase_2c import verifier
from warp_taskgen.runtime_composition import RuntimeComposition

from ._fixtures import (
    _STUB_SEED_REGISTRY,
    _bundle,
    _bypass_preflight,  # noqa: F401
    _gitlab_instance,
    _stable_git_fingerprint,  # noqa: F401
    _task,
    _write_tasks,
)


class _Scope:
    def __init__(self, *, restore_failure: str | None = None) -> None:
        self.restore_failure = restore_failure
        self.operation_ids: list[str] = []
        self.restore_calls: list[str] = []
        self.release_calls: list[str | None] = []

    def matches_instance(self, instance: object) -> bool:
        return True

    def operation_id(self) -> str:
        operation_id = f"phase2c-op-{len(self.operation_ids) + 1}"
        self.operation_ids.append(operation_id)
        return operation_id

    async def restore(self, operation_id: str) -> dict[str, str]:
        self.restore_calls.append(operation_id)
        if self.restore_failure is not None and len(self.restore_calls) == 1:
            raise HostRestorationError(self.restore_failure)
        return {"status": "restored", "operation_id": operation_id}

    async def release(self, *, operation_id: str | None = None) -> dict[str, str | None]:
        self.release_calls.append(operation_id)
        return {"status": "released", "operation_id": operation_id}


@pytest.mark.parametrize(
    "failure_kind",
    ("restore_failed", "auth_refresh_failed", "readback_failed"),
)
def test_managed_failure_is_not_a_reusable_feasibility_checkpoint(
    tmp_path, monkeypatch, failure_kind: str
) -> None:
    """A managed failure reruns the owner boundary instead of reusing a checkpoint."""

    scopes: list[_Scope] = []

    async def acquire_scope(*args: Any, **kwargs: Any) -> _Scope:
        scope = _Scope(restore_failure=failure_kind if failure_kind == "restore_failed" else None)
        scopes.append(scope)
        return scope

    async def fail_auth_refresh(*args: Any, **kwargs: Any) -> None:
        raise HostRestorationError("auth_refresh_failed")

    async def fail_readback(*args: Any, **kwargs: Any) -> None:
        raise HostRestorationError("readback_failed")

    monkeypatch.setattr(verifier, "acquire_restoration_scope", acquire_scope)
    if failure_kind == "auth_refresh_failed":
        monkeypatch.setattr(
            "warp_taskgen.storage_state_preflight.refresh_instance_auth", fail_auth_refresh
        )
    elif failure_kind == "readback_failed":
        async def refresh_auth(*args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(
            "warp_taskgen.storage_state_preflight.refresh_instance_auth", refresh_auth
        )
        monkeypatch.setattr(verifier, "capture_restoration_baseline_async", fail_readback)

    instance = _gitlab_instance(
        restoration={
            "socket_path": str(tmp_path / "owner.sock"),
            "instance_id": "gitlab-r1",
        }
    )
    tasks_path = _write_tasks(tmp_path, [_task()])
    checkpoint_dir = tmp_path / "checkpoints"

    def run_once():
        return asyncio.run(
            feasibility_runner.verify_feasibility(
                tasks_path,
                probes=_bundle(),
                seed_registry=_STUB_SEED_REGISTRY,
                instances=[instance],
                concurrency=1,
                retry_count=0,
                checkpoint_dir=checkpoint_dir,
                run_id="managed-phase2c",
                definition_digest="d" * 64,
            )
        )

    first = run_once()
    second = run_once()

    assert len(first.infeasible) == 1
    assert len(second.infeasible) == 1
    assert second.reused_checkpoints == 0
    assert first.infeasible[0]["restoration_status"] == "unknown"
    assert first.infeasible[0]["restoration_failure"] == failure_kind
    assert len(scopes) == 2
    assert all(scope.restore_calls for scope in scopes)
    assert all(scope.release_calls for scope in scopes)


def test_managed_phase2c_success_restores_with_fresh_operations(monkeypatch, tmp_path) -> None:
    """The full managed unit uses distinct pre/final operation identities."""

    scope = _Scope()

    async def acquire_scope(*args: Any, **kwargs: Any) -> _Scope:
        return scope

    async def fake_core(task: dict[str, Any], instance: dict[str, Any], **kwargs: Any):
        return {
            **task,
            "feasibility": {
                "status": "verified",
                "host_fingerprint": {"instances_digest": "digest"},
            },
        }

    async def refresh_auth(*args: Any, **kwargs: Any) -> None:
        return None

    async def capture_baseline(*args: Any, **kwargs: Any) -> object:
        return object()

    async def verify_baseline(*args: Any, **kwargs: Any) -> dict[str, str]:
        return {"status": "verified"}

    monkeypatch.setattr(verifier, "acquire_restoration_scope", acquire_scope)
    monkeypatch.setattr(verifier, "_verify_one_core", fake_core)
    monkeypatch.setattr(
        "warp_taskgen.storage_state_preflight.refresh_instance_auth", refresh_auth
    )
    monkeypatch.setattr(verifier, "capture_restoration_baseline_async", capture_baseline)
    monkeypatch.setattr(verifier, "verify_restoration_baseline_async", verify_baseline)

    task = _task()
    instance = _gitlab_instance(
        restoration={
            "socket_path": str(tmp_path / "owner.sock"),
            "instance_id": "gitlab-r1",
        }
    )
    result = asyncio.run(
        verifier._verify_one(
            task,
            instance,
            retry_count=0,
            fingerprint_base={"instances_digest": "digest"},
            ttl_hours=None,
            force_reverify=False,
            cleanup_warnings=[],
            runtime_composition=RuntimeComposition.default(),
            probes=_bundle(),
        )
    )

    assert result["feasibility"]["status"] == "verified"
    assert result["restoration_readback"] == {"status": "verified"}
    assert len(scope.restore_calls) == 2
    assert scope.restore_calls[0] != scope.restore_calls[1]
    assert scope.release_calls == [scope.restore_calls[1]]
